"""
Phase 1: acquire + prepare the mESC BEELINE dataset for MTGRN validation.
See mtgrn/PROGRESS.md's "Phase 1 -- mESC data acquisition" section for the
full story of how the gene/edge set here was chosen (a documented,
gene-count-calibrated reconstruction, NOT a bit-exact match to the
paper's Table 2 774-genes/8085-edges -- edge count could not be
reproduced from the public BEELINE files; see PROGRESS.md for what was
tried).

DATA SOURCE (exact commands, also logged in PROGRESS.md):
    curl -sL -o BEELINE-data.zip "https://zenodo.org/records/3701939/files/BEELINE-data.zip"
    curl -sL -o BEELINE-Networks.zip "https://zenodo.org/records/3701939/files/BEELINE-Networks.zip"
Zenodo record 10.5281/zenodo.3701939 (Pratapa et al. 2020) -- the exact
source the MTGRN paper itself cites for ground truth.

GENE SELECTION: top-N (default 580, see PROGRESS.md for calibration)
genes by BEELINE's own GeneOrdering.csv variability ranking, UNION the
mESC-ChIP-seq-network.csv's own regulator (TF) set, INTERSECTED with
genes present in both ExpressionData.csv and the ground-truth network.
This lands at 769 genes (target: 774, Table 2) -- see PROGRESS.md.

GROUND TRUTH vs PRIOR -- these are TWO DIFFERENT resources, not one:
  - GROUND TRUTH (for scoring AUPRC/AUROC/F1): mESC-ChIP-seq-network.csv,
    matching the paper's own "we used the ground truth network provided
    in Pratapa et al. (2020)" (Section 4). Written here as gt_edges.csv.
  - PRIOR (spatial-mask input P, fed INTO the model): Marlene's frozen
    load_trrust()/load_regnetwork() loaders, built in
    train_mesc_validation.py (not here) since building it requires the
    FINAL gene set this script produces, and Marlene's loaders take an
    AnnData to subset/align against. Kept as two separate scripts/steps
    deliberately so the ground-truth-vs-prior distinction stays explicit
    and neither accidentally leaks into the other.

OUTPUT
------
data_mesc/
    ExpressionData.csv   # genes (rows) x cells (cols), pseudotime-ordered columns
    PseudoTime.csv       # cell_id, PseudoTime
    gt_edges.csv         # regulator,target -- ChIP-seq ground truth (see above)
    prep_meta.json        # gene/edge counts, source files, top_n used
"""
import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw_dir", type=str, default="data_raw/extracted",
                     help="Directory containing the extracted BEELINE-data/ "
                          "and Networks/ subdirectories.")
    ap.add_argument("--top_n_variable", type=int, default=580,
                     help="Top-N genes by GeneOrdering.csv variability, "
                          "unioned with the ground-truth network's own "
                          "regulator set. Calibrated to land near the "
                          "paper's Table 2 774 genes for mESC -- see "
                          "PROGRESS.md for how this value was chosen and "
                          "why the resulting edge count still differs "
                          "from Table 2's 8085.")
    ap.add_argument("--out_dir", type=str, default="data_mesc")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    expr_path = raw_dir / "BEELINE-data/inputs/scRNA-Seq/mESC/ExpressionData.csv"
    pt_path = raw_dir / "BEELINE-data/inputs/scRNA-Seq/mESC/PseudoTime.csv"
    go_path = raw_dir / "BEELINE-data/inputs/scRNA-Seq/mESC/GeneOrdering.csv"
    net_path = raw_dir / "Networks/mouse/mESC-ChIP-seq-network.csv"

    print(f"Loading {expr_path}")
    expr = pd.read_csv(expr_path, index_col=0)  # genes x cells, raw gene case
    print(f"  raw shape (genes, cells) = {expr.shape}")

    pt = pd.read_csv(pt_path, index_col=0)
    go = pd.read_csv(go_path, index_col=0)
    net = pd.read_csv(net_path)

    # uppercase gene symbols throughout, matching Marlene's frozen
    # load_trrust/load_regnetwork convention (adata.var_names = upper(...))
    # so the gene sets align later in train_mesc_validation.py.
    gene_map = {g: g.upper() for g in expr.index}
    expr.index = [gene_map[g] for g in expr.index]
    expr = expr[~expr.index.duplicated(keep="first")]  # dedupe after uppercasing

    expr_genes = set(expr.index)
    net_genes = set(net["Gene1"].unique()) | set(net["Gene2"].unique())
    net_regulators = set(net["Gene1"].unique()) & expr_genes

    go_sorted = go.sort_values("VGAMpValue")
    top_var_genes = set(g.upper() for g in go_sorted.index[:args.top_n_variable])

    selected = (top_var_genes | net_regulators) & expr_genes & net_genes
    selected = sorted(selected)
    print(f"Selected {len(selected)} genes (top_n_variable={args.top_n_variable} "
          f"union network-regulators, intersected with expr+network genes)")

    expr_sel = expr.loc[selected]

    net_sel = net[net["Gene1"].isin(selected) & net["Gene2"].isin(selected)]
    net_sel = net_sel[net_sel["Gene1"] != net_sel["Gene2"]].drop_duplicates()
    print(f"Ground-truth edges after gene selection: {len(net_sel)}")
    print(f"  (paper's Table 2 mESC: 774 genes, 8085 edges -- see "
          f"PROGRESS.md for why edge count doesn't match here)")

    # order cells by pseudotime
    pt_col = pt.columns[0]
    pt_valid = pt[pt_col].dropna()
    cell_order = pt_valid.sort_values().index.tolist()
    cell_order = [c for c in cell_order if c in expr_sel.columns]
    expr_sel = expr_sel[cell_order]
    pt_out = pt_valid.loc[cell_order]

    expr_path_out = out_dir / "ExpressionData.csv"
    expr_sel.to_csv(expr_path_out)
    print(f"Wrote {expr_path_out} ({expr_sel.shape[0]} genes x {expr_sel.shape[1]} cells)")

    pt_path_out = out_dir / "PseudoTime.csv"
    pt_out.to_frame(name="PseudoTime").to_csv(pt_path_out)
    print(f"Wrote {pt_path_out}")

    gt_path_out = out_dir / "gt_edges.csv"
    net_sel.rename(columns={"Gene1": "regulator", "Gene2": "target"}).to_csv(gt_path_out, index=False)
    print(f"Wrote {gt_path_out}")

    n_genes = len(selected)
    n_edges = len(net_sel)
    meta = {
        "n_genes": n_genes,
        "n_gt_edges": n_edges,
        "n_cells": len(cell_order),
        "proportion": n_edges / (n_genes * (n_genes - 1)),
        "top_n_variable": args.top_n_variable,
        "source_expr": str(expr_path),
        "source_ground_truth_network": str(net_path),
        "paper_table2_n_genes": 774,
        "paper_table2_n_positive_edges": 8085,
        "paper_table2_proportion": 0.014,
    }
    with open(out_dir / "prep_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nWrote prep_meta.json: {meta}")


if __name__ == "__main__":
    main()
