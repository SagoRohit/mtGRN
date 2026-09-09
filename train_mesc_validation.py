"""
Phase 1 validation: train MTGRN on the mESC BEELINE dataset and compare
against the paper's own Table 1 mESC row (AUROC=0.713, AUPRC=0.748,
F1=0.694) and GENIE3's reported mESC baseline (AUROC=0.531, AUPRC=0.168)
for calibration. See mtgrn/PROGRESS.md for the full spec, judgment calls,
and validation bar (Section "PHASE 1 VALIDATION BAR" in
mtgrn_working_prompt.md, logged into PROGRESS.md before this script was
written).

Data: mtgrn/data_mesc.py (run first, or run with --prepare_data here).
Prior: Marlene's FROZEN load_trrust()/load_regnetwork(), vendored verbatim
into mtgrn/vendor_marlene_datasets.py (+ mtgrn/data/trrust_rawdata.mouse.tsv,
mtgrn/data/RegNetwork-mouse.source) so this whole folder is self-contained
and can be pushed/run standalone (e.g. on Kaggle) without also needing the
sibling marlene/ project -- see vendor_marlene_datasets.py's docstring.

USAGE
-----
python train_mesc_validation.py --data_dir data_mesc --seed 0
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from model import MTGRN, assert_spatial_attention_memory_safe  # noqa: E402

# --- FROZEN import: verbatim vendored copy of Marlene's load_trrust/
# load_regnetwork (see vendor_marlene_datasets.py's module docstring for
# why this is vendored inside mtgrn/ rather than imported from the
# sibling marlene/ project -- mtgrn/ needs to be a single, self-contained
# folder that can be pushed/uploaded to Kaggle on its own). ------------
from vendor_marlene_datasets import load_trrust, load_regnetwork  # noqa: E402
# ---------------------------------------------------------------------------


def make_windows(E_t: np.ndarray, W: int, M: int):
    """E_t: (T, G) pseudotime-ordered expression. Returns X: (N, W, G),
    Y: (N, M, G) per Section 3.1's sliding-window construction (Eq
    surrounding text -- X_i=[e_i,...,e_{i+W-1}], Y_i=[e_{i+W},...,e_{i+W+M-1}]).
    """
    T = E_t.shape[0]
    N = T - W - M + 1
    if N <= 0:
        raise ValueError(f"T={T} too short for W={W}+M={M} (need T >= W+M)")
    X = np.stack([E_t[i:i + W] for i in range(N)])
    Y = np.stack([E_t[i + W:i + W + M] for i in range(N)])
    return X.astype(np.float32), Y.astype(np.float32)


def chronological_split(N: int, train_frac: float = 0.8):
    """PROGRESS.md judgment call #4: chronological (not random) split on
    window start index, since windows overlap along the pseudotime axis --
    a random shuffle would leak temporally-adjacent windows across
    train/test."""
    n_train = int(N * train_frac)
    return np.arange(0, n_train), np.arange(n_train, N)


def warmup_cosine_lr(optimizer, step, total_steps, warmup_steps, base_lr):
    if step < warmup_steps:
        lr = base_lr * (step + 1) / max(1, warmup_steps)
    else:
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        lr = base_lr * 0.5 * (1 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


def compute_continuous_auprc_auroc(H: np.ndarray, gene_names: list[str], gt_edges: set):
    """Primary metric (task requirement): score the RAW continuous
    attention matrix against ground truth directly, ranked, no
    thresholding -- confirmed this matches the paper's own AUPRC/AUROC
    computation (Section 4 Metrics: "ranked edges... as predictions"),
    not just the top-K discrete version (which is separate, F1-only).
    H: (G, G) attention matrix, H[query=target, key=regulator] (see
    model.py's build_spatial_prior_mask ORIENTATION note) -- so an edge
    (reg, target) scores at H[target_idx, reg_idx].
    """
    idx = {g: i for i, g in enumerate(gene_names)}
    y_true, y_score = [], []
    for i, target in enumerate(gene_names):
        for j, reg in enumerate(gene_names):
            if i == j:
                continue
            y_true.append(1 if (reg, target) in gt_edges else 0)
            y_score.append(H[i, j])  # H[target_row, reg_col]
    y_true, y_score = np.array(y_true), np.array(y_score)
    if y_true.sum() == 0:
        return float("nan"), float("nan")
    return average_precision_score(y_true, y_score), roc_auc_score(y_true, y_score)


def compute_degree_weighted_topk_f1(H: np.ndarray, gene_names: list[str], gt_edges: set,
                                     prior_out_degree: np.ndarray):
    """Secondary metric (task requirement, sanity check against paper's
    reported F1 only): "multiply the regulatory scores by the degree of
    TFG in the network, rank descending, select top K edges (K = |true
    edges|)" (Section 3.4). prior_out_degree: (G,) -- TFG's out-degree in
    the PRIOR network, indexed by regulator position.
    """
    idx = {g: i for i, g in enumerate(gene_names)}
    K = len(gt_edges)
    candidates = []
    for i, target in enumerate(gene_names):
        for j, reg in enumerate(gene_names):
            if i == j:
                continue
            score = H[i, j] * prior_out_degree[j]
            candidates.append((score, reg, target))
    candidates.sort(key=lambda t: -t[0])
    top_k = candidates[:K]
    predicted_edges = {(reg, target) for _, reg, target in top_k}
    tp = len(predicted_edges & gt_edges)
    precision = tp / max(1, len(predicted_edges))
    recall = tp / max(1, len(gt_edges))
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return f1, precision, recall


def evaluate_in_batches(model, X, Y, prior_t, batch_size, device, criterion,
                         collect_attn: bool = False):
    """Chunked eval, same batch_size as training. Needed because a single
    unbatched forward over the whole test set OOMs for large G (confirmed:
    the positive_control sanity-check prior run, G=769 vs the earlier
    TRRUST run's G=146, killed (exit 137) on this machine's 7GB RAM with
    the original single-shot `model(X_test, ...)` call -- the spatial
    attention tensor is (B*W, n_heads, G, G), which scales with G^2 and
    with the FULL test-set size at once when unbatched). Returns
    (val_loss, H) where H is (G, G), averaged over heads/W/all test
    windows -- identical math to the original one-shot version, just
    accumulated across chunks instead of materializing all of them
    simultaneously.
    """
    total_loss, total_n = 0.0, 0
    attn_sum, attn_count = None, 0
    with torch.no_grad():
        for b in range(0, len(X), batch_size):
            xb = X[b:b + batch_size].to(device)
            yb = Y[b:b + batch_size].to(device)
            y_hat, spatial_attn = model(xb, prior_t)
            loss = criterion(y_hat, yb)
            total_loss += loss.item() * len(xb)
            total_n += len(xb)
            if collect_attn:
                # spatial_attn: (b, n_heads, G, G) -- sum over batch+heads,
                # matching the original .mean(dim=(0,1)) reduction.
                chunk_sum = spatial_attn.sum(dim=(0, 1))
                attn_sum = chunk_sum if attn_sum is None else attn_sum + chunk_sum
                attn_count += len(xb) * spatial_attn.shape[1]
    val_loss = total_loss / total_n
    H = (attn_sum / attn_count).cpu().numpy() if collect_attn else None
    return val_loss, H


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_dir", type=str, default="data_mesc")
    ap.add_argument("--species", type=str, default="mouse", choices=["mouse", "human"])
    ap.add_argument("--prior_source", type=str, default="trrust",
                     choices=["trrust", "regnetwork", "positive_control"],
                     help="'positive_control': union of TRRUST + RegNetwork + "
                          "the mESC ChIP-seq ground truth itself, on the FULL "
                          "(unfiltered) data_mesc gene set -- guarantees every "
                          "true edge is reachable through the spatial mask "
                          "(100% recall ceiling by construction). Used only as "
                          "a sanity check for whether TRRUST/RegNetwork's near-"
                          "chance Phase 1 result was a prior-coverage artifact "
                          "vs a real architecture bug -- see PROGRESS.md.")
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--n_temporal_blocks", type=int, default=2)
    ap.add_argument("--n_spatial_blocks", type=int, default=2)
    ap.add_argument("--W", type=int, default=10)
    ap.add_argument("--M", type=int, default=5)
    ap.add_argument("--n_epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--peak_lr", type=float, default=1e-4)
    ap.add_argument("--warmup_frac", type=float, default=0.1)
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--out_dir", type=str, default="mesc_validation_results")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    data_dir = Path(args.data_dir)
    expr = pd.read_csv(data_dir / "ExpressionData.csv", index_col=0)  # genes x cells
    pseudotime = pd.read_csv(data_dir / "PseudoTime.csv", index_col=0)
    # data_mesc.py's own output: the CHIP-seq ground-truth network (NOT the
    # prior -- see this script's and data_mesc.py's module docstrings for
    # why these are two separate resources, same as the paper's own design
    # NicheNet-prior vs Pratapa-ground-truth split).
    gt_df = pd.read_csv(data_dir / "gt_edges.csv")
    gt_edges_raw = set(zip(gt_df["regulator"], gt_df["target"]))

    # order cells by pseudotime (first non-null column, matching BEELINE's
    # single-lineage PseudoTime.csv convention for mESC)
    pt_col = pseudotime.columns[0]
    pt = pseudotime[pt_col].dropna()
    cell_order = pt.sort_values().index.tolist()
    expr = expr[cell_order]  # genes x cells, pseudotime-ordered columns

    gene_names = expr.index.tolist()
    E_t = expr.T.to_numpy(dtype=np.float32)  # (T=n_cells, G)
    print(f"E_t shape (T, G) = {E_t.shape}")

    # --- PRIOR (spatial-mask input P, fed INTO the model): Marlene's
    # FROZEN load_trrust()/load_regnetwork(), per task instructions, in
    # place of the paper's NicheNet (R-only, avoided per task instructions).
    # This further subsets genes to TRRUST/RegNetwork's own mouse gene
    # coverage -- expected to shrink G below data_mesc.py's 769.
    #
    # positive_control (added after Phase 1's TRRUST/RegNetwork runs both
    # came back chance-level, see PROGRESS.md): a sanity-check prior = union
    # of TRRUST edges + RegNetwork edges + the mESC ChIP-seq ground truth
    # itself, on the FULL (unfiltered) data_mesc gene set -- deliberately
    # guarantees every true edge is reachable through the spatial mask
    # (100% recall ceiling by construction), to test whether the earlier
    # chance-level result was a prior-coverage artifact vs an architecture
    # bug. NOT a real prior (leaks ground truth into the mask) -- only ever
    # used for this one diagnostic run, never for a reported Phase 1
    # pass/fail or Phase 2. ---
    import anndata, os
    adata = anndata.AnnData(X=E_t, var=pd.DataFrame(index=gene_names))
    # FIX: load_trrust/load_regnetwork use a RELATIVE path ('data/...'),
    # resolved against the process's cwd, not the file's own location --
    # confirmed via direct read of the vendored source (path =
    # f'data/trrust_rawdata.{species}.tsv'). Since mtgrn/ now vendors its
    # own data/trrust_rawdata.mouse.tsv + data/RegNetwork-mouse.source
    # (see vendor_marlene_datasets.py's module docstring), chdir into
    # THIS_DIR (mtgrn/ itself) for just this call rather than editing the
    # frozen function bodies. Same class of fix as every other frozen-code
    # compatibility shim in this project (PseudoGRN's pandas patch,
    # RiTINI's train_epoch/attention fixes) -- adapt the CALLER, never the
    # frozen source.
    _cwd = os.getcwd()
    os.chdir(THIS_DIR)
    try:
        if args.prior_source == "positive_control":
            # index_adata=False: get raw (regulator, target) gene-symbol
            # edge lists WITHOUT subsetting adata's gene set (we want the
            # full 769-gene data_mesc set here, not TRRUST/RegNetwork's
            # own narrower coverage).
            trrust_edges = load_trrust(species=args.species, adata=adata.copy(), index_adata=False)
            regnet_edges = load_regnetwork(species=args.species, adata=adata.copy(), index_adata=False)
        elif args.prior_source == "trrust":
            prior_edges_raw = load_trrust(species=args.species, adata=adata, index_adata=True)
        else:
            prior_edges_raw = load_regnetwork(species=args.species, adata=adata, index_adata=True)
    finally:
        os.chdir(_cwd)

    if args.prior_source == "positive_control":
        # keep the FULL data_mesc gene set (no TRRUST/RegNetwork filtering).
        gene_names_final = gene_names
    else:
        gene_names_final = adata.var_names.tolist()  # post-prior-filter gene set
    gene_pos_in_original = {g: i for i, g in enumerate(gene_names)}
    keep_positions = [gene_pos_in_original[g] for g in gene_names_final]
    E_t = E_t[:, keep_positions]  # realign to the final, prior-filtered gene set
    G = len(gene_names_final)

    # z-score per gene (standard preprocessing for this kind of regression
    # target; documented since the paper doesn't specify normalization
    # for MTGRN's own input beyond what's implicit in "gene expression
    # matrix")
    mean = E_t.mean(axis=0, keepdims=True)
    std = E_t.std(axis=0, keepdims=True) + 1e-8
    E_t_norm = (E_t - mean) / std

    idx = {g: i for i, g in enumerate(gene_names_final)}

    # ground truth restricted to the final gene set -- needed BEFORE the
    # positive_control prior is built (it unions this in), and this is
    # what AUPRC/AUROC/F1 are scored against either way, never prior_edges_raw.
    gene_names = gene_names_final
    gt_edges = set(e for e in gt_edges_raw if e[0] in idx and e[1] in idx)

    prior_adjacency = np.zeros((G, G), dtype=np.float32)
    if args.prior_source == "positive_control":
        union_edges = set(trrust_edges) | set(regnet_edges) | gt_edges
        for reg, target in union_edges:
            if reg in idx and target in idx:
                prior_adjacency[idx[reg], idx[target]] = 1.0
    else:
        for reg, target in prior_edges_raw:
            if reg in idx and target in idx:
                prior_adjacency[idx[reg], idx[target]] = 1.0
    np.fill_diagonal(prior_adjacency, 0.0)
    prior_out_degree = prior_adjacency.sum(axis=1)  # per-regulator out-degree
    print(f"Prior ({args.prior_source}) adjacency: G={G}, {int(prior_adjacency.sum())} edges")
    print(f"Ground truth (ChIP-seq, from data_mesc.py): {len(gt_edges)} edges "
          f"survive prior-filtered gene set (of {len(gt_edges_raw)} total)")
    baseline_auprc_here = len(gt_edges) / (G * (G - 1)) if G > 1 else float("nan")
    print(f"This dataset's own chance-level AUPRC baseline: {baseline_auprc_here:.4f} "
          f"(paper's reported mESC proportion: 0.014 -- NOT directly comparable, "
          f"see PROGRESS.md)")

    # Memory guard (cheap insurance, see model.py's docstring): this is
    # exactly the failure mode that OOM-killed this project's dev machine
    # before (G=769 unbatched eval, an earlier G=12,291 exploration, and a
    # CUDA OOM on a Kaggle T4 -- see PROGRESS.md). Checks the ACTUAL
    # batch_size/W/G/n_spatial_blocks about to be used, against currently
    # available memory ON THE ACTUAL DEVICE (GPU VRAM if --device cuda,
    # else system RAM), before any tensor is allocated.
    assert_spatial_attention_memory_safe(n_genes=G, n_heads=args.n_heads,
                                          batch_size=args.batch_size, W=args.W,
                                          n_spatial_blocks=args.n_spatial_blocks,
                                          device=device)

    X, Y = make_windows(E_t_norm, args.W, args.M)
    train_idx, test_idx = chronological_split(len(X), args.train_frac)
    print(f"Windows: {len(X)} total, {len(train_idx)} train, {len(test_idx)} test (chronological split)")

    X_train, Y_train = torch.tensor(X[train_idx]), torch.tensor(Y[train_idx])
    X_test, Y_test = torch.tensor(X[test_idx]), torch.tensor(Y[test_idx])
    prior_t = torch.tensor(prior_adjacency, dtype=torch.float32).to(device)

    model = MTGRN(n_genes=G, d_model=args.d_model, n_heads=args.n_heads,
                   n_temporal_blocks=args.n_temporal_blocks, n_spatial_blocks=args.n_spatial_blocks,
                   W=args.W, M=args.M).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.peak_lr)
    criterion = nn.MSELoss()

    n_train = len(X_train)
    steps_per_epoch = max(1, n_train // args.batch_size)
    total_steps = steps_per_epoch * args.n_epochs
    warmup_steps = int(total_steps * args.warmup_frac)

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    global_step = 0

    for epoch in range(args.n_epochs):
        model.train()
        perm = torch.randperm(n_train)
        total_loss = 0.0
        n_batches = 0
        for b in range(0, n_train, args.batch_size):
            batch_idx = perm[b:b + args.batch_size]
            xb = X_train[batch_idx].to(device)
            yb = Y_train[batch_idx].to(device)
            lr = warmup_cosine_lr(optimizer, global_step, total_steps, warmup_steps, args.peak_lr)
            optimizer.zero_grad()
            y_hat, _ = model(xb, prior_t)
            loss = criterion(y_hat, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
            global_step += 1
        train_loss = total_loss / n_batches

        model.eval()
        val_loss, _ = evaluate_in_batches(model, X_test, Y_test, prior_t,
                                           args.batch_size, device, criterion)
        print(f"epoch {epoch + 1}/{args.n_epochs}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} lr={lr:.2e}")

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch + 1} (patience={args.patience})")
                break

    model.load_state_dict(best_state)
    model.eval()

    _, H = evaluate_in_batches(model, X_test, Y_test, prior_t, args.batch_size,
                                device, criterion, collect_attn=True)  # (G, G)

    auprc, auroc = compute_continuous_auprc_auroc(H, gene_names, gt_edges)
    f1, precision, recall = compute_degree_weighted_topk_f1(H, gene_names, gt_edges, prior_out_degree)

    print(f"\n=== mESC Phase 1 validation results ===")
    print(f"Continuous AUPRC={auprc:.4f} AUROC={auroc:.4f}")
    print(f"Degree-weighted top-K F1={f1:.4f} (precision={precision:.4f} recall={recall:.4f})")
    print(f"\nTHIS dataset's own chance-level baseline: AUPRC={baseline_auprc_here:.4f} AUROC=0.5000")
    print(f"  (PRIMARY comparison -- see PROGRESS.md for why the paper's own "
          f"absolute Table 1 numbers below are not directly comparable here)")
    print(f"\nPaper's Table 1 mESC row (reference only, NOT apples-to-apples -- "
          f"different edge density, see PROGRESS.md): AUROC=0.713 AUPRC=0.748 F1=0.694")
    print(f"GENIE3 mESC baseline (paper's Table 1, reference only): AUROC=0.531 AUPRC=0.168")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "auprc": float(auprc), "auroc": float(auroc), "f1": float(f1),
        "precision": float(precision), "recall": float(recall),
        "n_genes": G, "n_gt_edges": len(gt_edges),
        "this_dataset_baseline_auprc": float(baseline_auprc_here),
        "this_dataset_baseline_auroc": 0.5,
        "n_train_windows": len(train_idx), "n_test_windows": len(test_idx),
        "paper_auroc_reference_only": 0.713, "paper_auprc_reference_only": 0.748,
        "paper_f1_reference_only": 0.694,
        "genie3_baseline_auroc_reference_only": 0.531,
        "genie3_baseline_auprc_reference_only": 0.168,
        "seed": args.seed,
    }
    with open(out_dir / f"metrics_seed{args.seed}.json", "w") as f:
        json.dump(results, f, indent=2)
    torch.save(best_state, out_dir / f"best_model_seed{args.seed}.ckpt")
    print(f"\nSaved results to {out_dir}")


if __name__ == "__main__":
    main()
