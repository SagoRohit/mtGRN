"""
Phase 2: train MTGRN on ONE (tier, seed) SERGIO combo prepared by
sergio_prepare_data_mtgrn.py, and score against our own ground truth.
metrics.json schema matches this project's "Group B" (pooled/continuous,
no per-discrete-timepoint loop) convention -- see pseudogrn/train_sergio.py's
own metrics dict, which this mirrors (auprc_per_t/auroc_per_t are
single-element lists: MTGRN produces one prediction over the whole pooled
pseudotime-ordered test split, not one per discrete timepoint).

Reuses make_windows/chronological_split/warmup_cosine_lr/
compute_continuous_auprc_auroc/compute_degree_weighted_topk_f1/
evaluate_in_batches from train_mesc_validation.py directly (same folder,
same generic logic, no need to duplicate -- importing it has no side
effects, its own training code is guarded by __main__).

--results_root ACTUALLY determines where metrics.json is written (an
explicit CLI-plumbed value, not a hardcoded relative path the caller has
to independently know to match -- avoiding the exact dead-flag bug found
in ritini/run_density_experiment_ritini.py, where --results_root was
accepted but never forwarded to the training subprocess, which silently
wrote to a different, hardcoded path instead).

USAGE
-----
python train_sergio_mtgrn.py --data_dir data_tier1_seed0 --runid tier1_seed0 --seed 0
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from model import MTGRN, assert_spatial_attention_memory_safe  # noqa: E402
from train_mesc_validation import (  # noqa: E402
    make_windows, chronological_split, warmup_cosine_lr,
    compute_continuous_auprc_auroc, compute_degree_weighted_topk_f1,
    evaluate_in_batches,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_dir", type=str, required=True,
                     help="Output dir from sergio_prepare_data_mtgrn.py.")
    ap.add_argument("--runid", type=str, required=True)
    ap.add_argument("--W", type=int, default=None,
                     help="History window length. Defaults to whatever "
                          "sergio_prepare_data_mtgrn.py used (read from "
                          "prep_meta.json) -- override only if you know "
                          "what you're doing, since the feasibility check "
                          "at prep time was computed against THAT value.")
    ap.add_argument("--M", type=int, default=None, help="Forecast window length -- see --W.")
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--n_temporal_blocks", type=int, default=2)
    ap.add_argument("--n_spatial_blocks", type=int, default=2)
    ap.add_argument("--n_epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--peak_lr", type=float, default=1e-4)
    ap.add_argument("--warmup_frac", type=float, default=0.1)
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument(
        "--results_root", type=str, default="MTGRN_results/SERGIO",
        help="Where <runid>/metrics.json actually gets written -- unlike "
             "ritini's run_density_experiment_ritini.py, this value is "
             "used directly (see module docstring), not just checked for "
             "resumability while training silently writes elsewhere.",
    )
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    data_dir = Path(args.data_dir)
    with open(data_dir / "prep_meta.json") as f:
        prep_meta = json.load(f)

    W = args.W if args.W is not None else prep_meta["W"]
    M = args.M if args.M is not None else prep_meta["M"]
    if not prep_meta.get("feasible", True):
        raise SystemExit(
            f"prep_meta.json for {data_dir} marked feasible=False "
            f"(n_windows_estimate={prep_meta.get('n_windows_estimate')}) -- "
            f"re-run sergio_prepare_data_mtgrn.py with a smaller --W/--M or "
            f"a larger --max_cells before training."
        )

    expr = pd.read_csv(data_dir / "ExpressionData.csv", index_col=0)  # genes x cells
    pseudotime = pd.read_csv(data_dir / "PseudoTime.csv", index_col=0)
    gt_df = pd.read_csv(data_dir / "gt_edges.csv")
    gt_edges = set(zip(gt_df["regulator"], gt_df["target"]))
    prior_adjacency = np.load(data_dir / "prior_adjacency.npy")

    pt_col = pseudotime.columns[0]
    cell_order = pseudotime[pt_col].sort_values().index.tolist()
    expr = expr[cell_order]  # genes x cells, pseudotime-ordered columns

    gene_names = expr.index.tolist()
    G = len(gene_names)
    E_t = expr.T.to_numpy(dtype=np.float32)  # (T=n_cells, G)
    print(f"E_t shape (T, G) = {E_t.shape}")

    prior_out_degree = prior_adjacency.sum(axis=1)
    print(f"Prior: G={G}, {int(prior_adjacency.sum())} edges "
          f"(dense TF-bipartite, from sergio_prepare_data_mtgrn.py)")
    baseline_auprc_here = len(gt_edges) / (G * (G - 1)) if G > 1 else float("nan")
    print(f"This dataset's own chance-level AUPRC baseline: {baseline_auprc_here:.4f}")

    # Memory guard (see model.py's docstring) -- checks GPU VRAM if
    # --device cuda, else system RAM, BEFORE any tensor is allocated.
    assert_spatial_attention_memory_safe(n_genes=G, n_heads=args.n_heads,
                                          batch_size=args.batch_size, W=W,
                                          n_spatial_blocks=args.n_spatial_blocks,
                                          device=device)

    mean = E_t.mean(axis=0, keepdims=True)
    std = E_t.std(axis=0, keepdims=True) + 1e-8
    E_t_norm = (E_t - mean) / std

    X, Y = make_windows(E_t_norm, W, M)
    train_idx, test_idx = chronological_split(len(X), args.train_frac)
    print(f"Windows: {len(X)} total, {len(train_idx)} train, {len(test_idx)} test (chronological split)")

    X_train, Y_train = torch.tensor(X[train_idx]), torch.tensor(Y[train_idx])
    X_test, Y_test = torch.tensor(X[test_idx]), torch.tensor(Y[test_idx])
    prior_t = torch.tensor(prior_adjacency, dtype=torch.float32).to(device)

    model = MTGRN(n_genes=G, d_model=args.d_model, n_heads=args.n_heads,
                   n_temporal_blocks=args.n_temporal_blocks, n_spatial_blocks=args.n_spatial_blocks,
                   W=W, M=M).to(device)
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

    print(f"\n=== Phase 2 SERGIO results ({args.runid}) ===")
    print(f"Continuous AUPRC={auprc:.4f} AUROC={auroc:.4f}")
    print(f"Degree-weighted top-K F1={f1:.4f} (precision={precision:.4f} recall={recall:.4f})")
    print(f"THIS dataset's own chance-level baseline: AUPRC={baseline_auprc_here:.4f} AUROC=0.5000")

    out_dir = Path(args.results_root) / args.runid
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "mean_auprc": float(auprc),
        "mean_auroc": float(auroc),
        # Group B: one prediction over the whole pooled pseudotime test
        # split, not one per discrete timepoint -- single-element lists
        # kept for structural parity with every other model's metrics.json
        # (aggregation scripts only read mean_auprc/mean_auroc/n_timepoints).
        "auprc_per_t": [float(auprc)],
        "auroc_per_t": [float(auroc)],
        "n_timepoints": prep_meta["n_timepoints_kept"],  # tier value (15/5/3)
        "n_cells_pooled": prep_meta["n_cells_written"],
        "n_genes": G,
        "n_tfs": prep_meta["n_tfs"],
        "n_prior_edges": prep_meta["n_prior_edges"],
        "f1": float(f1), "precision": float(precision), "recall": float(recall),
        "this_dataset_baseline_auprc": float(baseline_auprc_here),
        "this_dataset_baseline_auroc": 0.5,
        "n_train_windows": len(train_idx), "n_test_windows": len(test_idx),
        "W": W, "M": M,
        "seed": args.seed,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    torch.save(best_state, out_dir / "best_model.ckpt")
    print(f"\nSaved metrics.json + best_model.ckpt to {out_dir}")


if __name__ == "__main__":
    main()
