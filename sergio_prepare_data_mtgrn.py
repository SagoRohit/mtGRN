"""
Converts a SERGIO-simulated dataset into MTGRN's raw input format, for one
(tier, seed) combination. Phase 2 of mtgrn/PROGRESS.md.

WHY THIS POOLS ACROSS ALL BINS (same design as pseudogrn/, NOT ritini/marlene)
-------------------------------------------------------------------------------
MTGRN's Trajectory Inference Module (Section 3.1) wants ONE pseudotime-ordered
sequence of individual cells, each cell its own timestep ("T is equal to C" --
see mtgrn/PROGRESS.md's Phase 0 spec). This is the same native input shape as
PseudoGRN (Group B: continuous/pseudotime), NOT Marlene/RiTINI's per-bin
discrete-snapshot treatment (Group A) -- confirmed by mtgrn/PROGRESS.md's own
Phase 0 notes ("pseudogrn/'s replicate-pooling + DPT + --max_cells pattern
already well understood... will reuse the same pooling/DPT/max_cells
structure directly"). So this script is adapted from
pseudogrn/sergio_prepare_data.py's pattern, not ritini's or marlene's:
  - Same load_sergio_dataset (same file format).
  - Same --n_timepoints_keep/--seed density-tier subsampling MECHANISM
    (identical np.random.RandomState(seed).choice(...) call), so a given
    (tier, seed) selects the SAME replicate indices every other model in
    this project selected for that (tier, seed).
  - Kept replicates are POOLED across ALL 9 bins into one unlabeled cell
    set (bin id used ONLY to pick a deterministic DPT root cell, then
    discarded -- MTGRN never sees a bin/replicate label).
  - Same scanpy DPT pipeline (sc.pp.pca -> sc.pp.neighbors -> sc.tl.diffmap
    -> sc.tl.dpt) and the same --max_cells cap (pseudotime computed over
    the FULL pool first for manifold quality; --max_cells only subsamples
    what's WRITTEN, for downstream compute tractability -- MTGRN's spatial
    attention cost doesn't scale with cell count directly, but the NUMBER
    OF TRAINING WINDOWS does, and Tier 1 pools ~40,500 cells -> ~40,486
    windows/epoch, which would make even a few epochs slow. Default here
    is lower than pseudogrn's 3000 -- see --max_cells help for why.

TIER-3 FEASIBILITY (checked before writing this script, not assumed)
-----------------------------------------------------------------------
Unlike RiTINI (which hit a hard architectural floor at n_timepoints=3
because it needs >=4 DISCRETE kept-replicate timepoints for its Neural
ODE), MTGRN's "timesteps" are individual pooled CELLS, not discrete
replicate labels. Confirmed via real prep_meta.json from this project's
own RiTINI runs: n_cells_in_bin_per_replicate=300 (9 bins x 300 = 2700
cells/replicate). At Tier 3 (3 kept replicates, pooled across all 9
bins): 3 x 2700 = 8,100 cells -> 8,086 valid (W=10, M=5) sliding windows.
Nowhere near a floor -- Tier 3 is comfortably feasible. (Tier 1: 40,500
cells/40,486 windows; Tier 2: 13,500/13,486.) This script still writes a
"feasible"/"n_windows_estimate" field to prep_meta.json as a defensive
check (in case --W/--M/--max_cells are changed to something that DOES
make a tier infeasible), matching every other model's safety-net
convention in this project -- not because Tier 3 is expected to trip it.

PREPROCESSING
--------------
sc.pp.normalize_total + sc.pp.log1p, same parity fix every other model in
this project applies (SERGIO bins differ hugely in raw expression scale
by design).

PRIOR (spatial-mask input, per mtgrn/PROGRESS.md's Phase 0 judgment call)
---------------------------------------------------------------------------
Dense TF-bipartite: every TF a candidate regulator for every non-self
target (prior[tf_row, :] = 1, zero diagonal) -- the SAME simple
construction RiTINI's harness originally used before RiTINI's own
graph_reg BCE loss made it mathematically unsatisfiable against per-target
softmax attention (see ritini/DEBUG_ATTENTION_COLLAPSE.md Checkpoints 0-3).
MTGRN's spatial mask is a HARD additive mask (-1e9 outside the prior), not
a soft BCE regularization target, so there's no analogous mismatch --
per mtgrn/PROGRESS.md's Phase 0 decision. Flagged there to watch for a
similar collapse symptom during Phase 2 validation in case the lesson
transfers for a reason not yet apparent; the Phase 1 positive-control
result (10.8x baseline, near-perfect AUROC) is independent evidence
MTGRN's architecture does NOT share RiTINI's loss-design problem, but
Phase 2's real SERGIO data has not been checked yet.

OUTPUT
------
<out_dir>/
    ExpressionData.csv   # genes (rows) x cells (columns), pooled+normalized
    PseudoTime.csv       # cells (rows) x 1 pseudotime column
    gt_edges.csv         # regulator,target -- OUR OWN evaluation ground
                          # truth (never fed into the prior)
    prior_adjacency.npy  # (n_genes, n_genes) dense TF-bipartite prior
    prep_meta.json        # n_timepoints_kept (=n_replicates_kept, the tier
                          # value), seed, n_cells_pooled_full,
                          # n_cells_written, max_cells_cap, n_genes, n_tfs,
                          # n_prior_edges, n_windows_estimate, feasible

USAGE
-----
python sergio_prepare_data_mtgrn.py \
    --dataset_dir /kaggle/working/SERGIO/data_sets/De-noised_400G_9T_300cPerT_5_DS2 \
    --n_timepoints_keep 15 --seed 0 --out_dir data_tier1_seed0
"""
import argparse
import json
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import scanpy as sc


def load_sergio_dataset(dataset_dir: Path, n_bins: int, replicate_ids: list[int]):
    """Identical loading logic to every other model's sergio_prepare_data*.py
    in this project (same file format). Returns X (n_cells, n_genes),
    cell_type (=bin id), replicate (=SERGIO replicate id), gene_ids, gt_edges.
    """
    gt_path = dataset_dir / "gt_GRN.csv"
    gt_df = pd.read_csv(gt_path, header=None, names=["reg", "target"])
    gt_edges = list(zip(gt_df["reg"].astype(int), gt_df["target"].astype(int)))

    X_list, CT_list, REP_list = [], [], []
    gene_ids = None

    for rep in replicate_ids:
        csv_path = dataset_dir / f"simulated_noNoise_{rep}.csv"
        if not csv_path.exists():
            print(f"  [skip] {csv_path.name} not found")
            continue

        raw = pd.read_csv(csv_path, header=None)
        gene_ids_this = raw.iloc[1:, 0].to_numpy().astype(int)
        expr = raw.iloc[1:, 1:].to_numpy().astype(np.float32)  # (n_genes, n_cells)

        if gene_ids is None:
            gene_ids = gene_ids_this
        else:
            assert np.array_equal(gene_ids, gene_ids_this), \
                "gene order mismatch between replicates"

        n_genes, n_cells_total = expr.shape
        assert n_cells_total % n_bins == 0, \
            f"{n_cells_total} cells not divisible by {n_bins} bins"
        cells_per_bin = n_cells_total // n_bins

        cell_type = np.repeat(np.arange(n_bins), cells_per_bin)
        replicate = np.full(n_cells_total, rep, dtype=int)

        X_list.append(expr.T)  # -> (n_cells, n_genes)
        CT_list.append(cell_type)
        REP_list.append(replicate)
        print(f"  loaded {csv_path.name} (replicate={rep}): "
              f"{expr.shape[1]} cells x {n_genes} genes, {n_bins} cell types")

    X = np.concatenate(X_list, axis=0)
    cell_type = np.concatenate(CT_list, axis=0)
    replicate = np.concatenate(REP_list, axis=0)
    return X, cell_type, replicate, gene_ids, gt_edges


def build_dense_tf_bipartite_prior(is_tf: np.ndarray) -> np.ndarray:
    """See module docstring PRIOR section. prior[tf_row, :] = 1 for every
    TF, zero diagonal (a gene is never its own regulator here)."""
    n_genes = len(is_tf)
    prior = np.zeros((n_genes, n_genes), dtype=np.float32)
    prior[is_tf, :] = 1.0
    np.fill_diagonal(prior, 0.0)
    return prior


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--dataset_dir", type=str,
        default="/kaggle/working/SERGIO/data_sets/De-noised_400G_9T_300cPerT_5_DS2",
        help="SERGIO bundled dataset folder (same 400-gene/37-TF dataset "
             "every other model in this project uses).",
    )
    ap.add_argument("--n_bins", type=int, default=9)
    ap.add_argument("--n_replicates", type=int, default=15,
                     help="All 15 pre-simulated replicates are loaded; tiers "
                          "then subsample down to n_timepoints_keep of them "
                          "(matches every other model's default so seeds align).")
    ap.add_argument("--n_timepoints_keep", type=int, required=True,
                     help="Density-tier value (15/5/3) -- how many of "
                          "--n_replicates to keep and pool. SAME MECHANISM "
                          "as every other model's sergio_prepare_data*.py.")
    ap.add_argument("--seed", type=int, default=0,
                     help="Must match the same --seed every other model "
                          "uses for a given (tier, seed) to select "
                          "identical replicates.")
    ap.add_argument("--root_bin", type=int, default=0,
                     help="SERGIO bin id whose cells seed the DPT root. "
                          "Bin id is otherwise discarded after this choice "
                          "-- MTGRN never sees it. Arbitrary but "
                          "deterministic, same convention as pseudogrn/.")
    ap.add_argument(
        "--max_cells", type=int, default=22500,
        help="Cap on pooled cells actually WRITTEN (DPT is still computed "
             "over the full pool first, for manifold quality) -- only "
             "applied via min(natural_pool_size, max_cells): a tier whose "
             "natural pool is already below the cap is left untouched, "
             "NOT force-set to the cap value. CORRECTED DEFAULT (was 1500 "
             "-- see mtgrn/PROGRESS.md's Phase 2 first-sweep postmortem): "
             "1500 is smaller than even Tier 3's full 8,100-cell pool, so "
             "EVERY tier got capped down to the identical 1500 cells, "
             "silently destroying the density-tier comparison this sweep "
             "exists to measure (all three tiers trained on the same-sized "
             "data). 22500 sits strictly ABOVE Tier 1's compressed target "
             "and Tier 2/3's full natural pools (13,500/8,100) -- so with "
             "this default, Tier 1 (40,500 natural) is the ONLY tier "
             "actually subsampled, down to 22,500; Tier 2 and Tier 3 are "
             "left at their full natural sizes, preserving the tier "
             "ordering (22500 > 13500 > 8100) the experiment needs. Timing "
             "calibrated from a real 2-epoch Kaggle GPU smoke test on "
             "uncapped Tier 1 (40,500 cells, 24m34s/2 epochs) -- see "
             "PROGRESS.md for the full 9-run sweep projection this default "
             "is based on (~13.4h worst-case, no early stopping). Set to 0 "
             "to disable capping entirely (fully-uncapped scenario, "
             "~18.8h worst-case per PROGRESS.md).",
    )
    ap.add_argument("--W", type=int, default=10,
                     help="History window length, forwarded to "
                          "train_sergio_mtgrn.py -- only used HERE to "
                          "compute the feasibility check written to "
                          "prep_meta.json.")
    ap.add_argument("--M", type=int, default=5,
                     help="Forecast window length -- see --W.")
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading SERGIO dataset from {dataset_dir}")
    X, cell_type, replicate, gene_ids, gt_edges = load_sergio_dataset(
        dataset_dir, n_bins=args.n_bins, replicate_ids=list(range(args.n_replicates)),
    )
    gene_names = np.array([f"G{gid}" for gid in gene_ids])
    id_to_name = {gid: f"G{gid}" for gid in gene_ids}
    n_genes = len(gene_ids)

    # --- density-tier subsampling: IDENTICAL mechanism to every other model ---
    k = min(args.n_timepoints_keep, args.n_replicates)
    rng = np.random.RandomState(args.seed)
    keep_rep = np.sort(rng.choice(args.n_replicates, size=k, replace=False))
    mask = np.isin(replicate, keep_rep)
    X, cell_type, replicate = X[mask], cell_type[mask], replicate[mask]
    print(f"Tier subsampling: kept replicates {list(keep_rep)} (seed={args.seed})")
    print(f"Pooled cell set (all {args.n_bins} bins, replicate id discarded "
          f"after this point): {X.shape[0]} cells x {X.shape[1]} genes")

    regulators = set(r for r, _ in gt_edges)
    is_tf = np.array([gid in regulators for gid in gene_ids])
    print(f"{is_tf.sum()} / {n_genes} genes are regulators (TFs)")
    if is_tf.sum() < 2:
        print("WARNING: fewer than 2 TFs -- prior will be degenerate.")

    adata = anndata.AnnData(
        X=X,
        obs=pd.DataFrame({"cell_type": np.array([f"CT{c}" for c in cell_type])}),
        var=pd.DataFrame({"is_TF": is_tf}, index=gene_names),
    )
    adata.obs_names = [f"cell{i}" for i in range(adata.n_obs)]

    # Parity preprocessing, same as every other model.
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)

    # --- Diffusion pseudotime over the pooled, unlabeled cell set ---------
    n_pcs = min(30, adata.n_vars - 1, adata.n_obs - 1)
    n_neighbors = min(15, adata.n_obs - 1)
    sc.pp.pca(adata, n_comps=max(2, n_pcs))
    sc.pp.neighbors(adata, n_neighbors=max(2, n_neighbors))
    sc.tl.diffmap(adata)

    root_mask = cell_type == args.root_bin
    if not root_mask.any():
        print(f"WARNING: --root_bin={args.root_bin} not present in pooled "
              f"cells; falling back to bin {int(cell_type.min())}")
        root_mask = cell_type == cell_type.min()
    root_idx = int(np.flatnonzero(root_mask)[0])
    adata.uns["iroot"] = root_idx
    sc.tl.dpt(adata)
    print(f"Computed DPT pseudotime, root cell index={root_idx} (bin {args.root_bin})")

    # --- Cap cell count for compute tractability (see --max_cells help) ---
    n_full_pool = adata.n_obs
    if args.max_cells and adata.n_obs > args.max_cells:
        sub_rng = np.random.RandomState(args.seed)
        keep_idx = np.sort(sub_rng.choice(adata.n_obs, size=args.max_cells, replace=False))
        adata = adata[keep_idx].copy()
        print(f"Capped pooled cells for compute tractability: "
              f"{n_full_pool} -> {adata.n_obs} (--max_cells={args.max_cells}, "
              f"seed={args.seed}); pseudotime was computed before this cap")

    n_cells_written = adata.n_obs

    # --- Prior (see module docstring PRIOR section) ------------------------
    prior = build_dense_tf_bipartite_prior(is_tf)
    n_prior_edges = int(prior.sum())
    print(f"Dense TF-bipartite prior: {n_prior_edges} edges "
          f"({is_tf.sum()} TFs x {n_genes - 1} possible targets each)")

    # --- Feasibility check (defensive -- see module docstring TIER-3 FEASIBILITY) ---
    n_windows_estimate = n_cells_written - args.W - args.M + 1
    feasible = n_windows_estimate >= 10  # arbitrary small floor: need enough
    # windows for a non-degenerate chronological 80/20 train/test split.
    if not feasible:
        print(f"WARNING: only {n_windows_estimate} estimated windows at "
              f"W={args.W}, M={args.M} -- too few for a meaningful "
              f"train/test split. Reduce --W/--M or raise --max_cells.")

    # --- Write MTGRN-native input files (same read convention as
    # train_mesc_validation.py: ExpressionData.csv genes x cells) ----------
    expr_df = pd.DataFrame(
        adata.X.T, index=adata.var_names, columns=adata.obs_names,
    )
    expr_path = out_dir / "ExpressionData.csv"
    expr_df.to_csv(expr_path)
    print(f"Wrote {expr_path} ({expr_df.shape[0]} genes x {expr_df.shape[1]} cells)")

    pt_df = pd.DataFrame(
        {"PseudoTime1": adata.obs["dpt_pseudotime"].to_numpy()},
        index=adata.obs_names,
    )
    pt_path = out_dir / "PseudoTime.csv"
    pt_df.to_csv(pt_path)
    print(f"Wrote {pt_path}")

    gt_edges_named = [(id_to_name[r], id_to_name[t]) for r, t in gt_edges
                       if r in id_to_name and t in id_to_name]
    gt_path = out_dir / "gt_edges.csv"
    pd.DataFrame(gt_edges_named, columns=["regulator", "target"]).to_csv(gt_path, index=False)
    print(f"Wrote {gt_path} ({len(gt_edges_named)} edges)")

    prior_path = out_dir / "prior_adjacency.npy"
    np.save(prior_path, prior)
    print(f"Wrote {prior_path}")

    meta = {
        "n_timepoints_kept": int(len(keep_rep)),  # the tier value (15/5/3)
        "replicate_ids_kept": [int(r) for r in keep_rep],
        "seed": args.seed,
        "n_cells_pooled_full": int(n_full_pool),
        "n_cells_written": int(n_cells_written),
        "max_cells_cap": args.max_cells,
        "n_genes": n_genes,
        "n_tfs": int(is_tf.sum()),
        "n_prior_edges": n_prior_edges,
        "prior_method": "dense_tf_bipartite",
        "W": args.W,
        "M": args.M,
        "n_windows_estimate": int(n_windows_estimate),
        "feasible": bool(feasible),
    }
    meta_path = out_dir / "prep_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {meta_path} (feasible={feasible})")


if __name__ == "__main__":
    main()
