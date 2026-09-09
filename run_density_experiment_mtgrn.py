"""
Orchestrates the temporal-density sparsity experiment (proposal Section 5.5)
for MTGRN across 3 density tiers x N_SEEDS seeds, by calling
sergio_prepare_data_mtgrn.py + train_sergio_mtgrn.py as subprocesses for
each (tier, seed) combination. Phase 2 of mtgrn/PROGRESS.md.

WHY NO BIN DIMENSION (unlike ritini's/marlene's own orchestrators)
-----------------------------------------------------------------------
MTGRN pools ALL SERGIO bins into one continuous pseudotime-ordered cell
set per (tier, seed) -- same design as pseudogrn/run_density_experiment.py,
which this file is directly adapted from (see sergio_prepare_data_mtgrn.py's
module docstring for why). So the sweep is (tier, seed) only: 3 tiers x
N_SEEDS seeds = 9 runs total, not 9 bins x 3 tiers x 3 seeds = 81 like
ritini's.

KAGGLE-ONLY -- NOT RUN LOCALLY
--------------------------------
Per explicit user instruction, all MTGRN training happens on Kaggle, same
as every other model in this project. This script is written to be
Kaggle-runnable; it has not been executed in this local dev environment
(--dry_run only, if at all, to sanity-check the command list without
touching real data).

RESUMABILITY
------------
Before running a (tier, seed) combo, this script checks whether
MTGRN_results/SERGIO/tier<N>_seed<S>/metrics.json already exists and skips
it if so -- an interrupted sweep (e.g. a Kaggle session timeout) can
simply be re-run from the top. --results_root is forwarded all the way to
train_sergio_mtgrn.py and actually determines where it writes (see that
script's own docstring for why this is NOT the dead-flag pattern found in
ritini/run_density_experiment_ritini.py).

USAGE
-----
    python run_density_experiment_mtgrn.py

Sanity-check the sweep plan first (prints every command, touches nothing):

    python run_density_experiment_mtgrn.py --dry_run
"""
import argparse
import subprocess
import sys
import time
import traceback
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PREPARE_SCRIPT = SCRIPT_DIR / "sergio_prepare_data_mtgrn.py"
TRAIN_SCRIPT = SCRIPT_DIR / "train_sergio_mtgrn.py"

# Proposal Section 5.5 temporal-density tiers -- IDENTICAL to every other
# model's TIERS in this project.
TIERS = {
    1: 15,  # Tier 1 -- dense (all available replicates)
    2: 5,   # Tier 2 -- sparse
    3: 3,   # Tier 3 -- ultra-sparse (confirmed FEASIBLE for MTGRN, unlike
            # RiTINI -- see sergio_prepare_data_mtgrn.py's TIER-3
            # FEASIBILITY note: 8,100 pooled cells -> 8,086 valid windows
            # at W=10/M=5, nowhere near a floor).
}

N_SEEDS = 3


def log(msg: str, log_file: Path) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(log_file, "a") as f:
        f.write(line + "\n")


def run_prepare(*, seed, n_timepoints_keep, out_dir, dataset_dir, n_bins,
                 n_replicates, root_bin, max_cells, W, M, dry_run, log_file):
    cmd = [
        sys.executable, str(PREPARE_SCRIPT),
        "--dataset_dir", dataset_dir,
        "--n_bins", str(n_bins),
        "--n_replicates", str(n_replicates),
        "--n_timepoints_keep", str(n_timepoints_keep),
        "--seed", str(seed),
        "--root_bin", str(root_bin),
        "--max_cells", str(max_cells),
        "--W", str(W),
        "--M", str(M),
        "--out_dir", str(out_dir),
    ]
    log(f"  [prepare] {' '.join(cmd)}", log_file)
    if not dry_run:
        subprocess.run(cmd, check=True)


def run_train(*, data_dir, runid, seed, device, d_model, n_heads,
              n_temporal_blocks, n_spatial_blocks, n_epochs, patience,
              batch_size, peak_lr, warmup_frac, train_frac, results_root,
              dry_run, log_file):
    cmd = [
        sys.executable, str(TRAIN_SCRIPT),
        "--data_dir", str(data_dir),
        "--runid", runid,
        "--seed", str(seed),
        "--device", device,
        "--d_model", str(d_model),
        "--n_heads", str(n_heads),
        "--n_temporal_blocks", str(n_temporal_blocks),
        "--n_spatial_blocks", str(n_spatial_blocks),
        "--n_epochs", str(n_epochs),
        "--patience", str(patience),
        "--batch_size", str(batch_size),
        "--peak_lr", str(peak_lr),
        "--warmup_frac", str(warmup_frac),
        "--train_frac", str(train_frac),
        "--results_root", str(results_root),
    ]
    log(f"  [train]   {' '.join(cmd)}", log_file)
    if not dry_run:
        subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--dataset_dir", type=str,
        default="/kaggle/working/SERGIO/data_sets/De-noised_400G_9T_300cPerT_5_DS2",
        help="SERGIO bundled dataset folder, forwarded to "
             "sergio_prepare_data_mtgrn.py (same 400-gene/37-TF dataset "
             "every other model in this project uses).",
    )
    ap.add_argument("--n_bins", type=int, default=9)
    ap.add_argument("--n_replicates", type=int, default=15,
                     help="All 15 pre-simulated replicates are loaded; "
                          "tiers then subsample down to n_timepoints_keep "
                          "of them (matches every other model's default so "
                          "seeds align).")
    ap.add_argument("--root_bin", type=int, default=0,
                     help="Forwarded to sergio_prepare_data_mtgrn.py's DPT "
                          "root cell selection.")
    ap.add_argument("--max_cells", type=int, default=1500,
                     help="Forwarded to sergio_prepare_data_mtgrn.py's "
                          "--max_cells -- see that script's help for why "
                          "this default is lower than pseudogrn's 3000 "
                          "(NOT independently timed on real data here).")
    ap.add_argument("--W", type=int, default=10)
    ap.add_argument("--M", type=int, default=5)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--n_temporal_blocks", type=int, default=2)
    ap.add_argument("--n_spatial_blocks", type=int, default=2)
    ap.add_argument("--n_epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=32,
                     help="Forwarded to train_sergio_mtgrn.py. Its own "
                          "memory guard will raise a clear error with a "
                          "suggested safe value if this doesn't fit the "
                          "actual device's memory -- see model.py.")
    ap.add_argument("--peak_lr", type=float, default=1e-4)
    ap.add_argument("--warmup_frac", type=float, default=0.1)
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--device", type=str, default="cpu",
                     help="Pass 'cuda' on a Kaggle GPU session. See "
                          "PROGRESS.md for the CPU-vs-GPU torch install "
                          "gotcha (requirements.txt deliberately doesn't "
                          "pin torch, to avoid overwriting Kaggle's own "
                          "CUDA build).")
    ap.add_argument("--n_seeds", type=int, default=N_SEEDS)
    ap.add_argument("--data_root", type=str, default=".",
                     help="Where per-(tier,seed) data_tier<N>_seed<S>/ dirs "
                          "are created.")
    ap.add_argument(
        "--results_root", type=str, default="MTGRN_results/SERGIO",
        help="Where train_sergio_mtgrn.py writes <runid>/metrics.json -- "
             "actually forwarded and used (see train_sergio_mtgrn.py's own "
             "docstring for why this differs from ritini's dead-flag bug).",
    )
    ap.add_argument(
        "--dry_run", action="store_true",
        help="Print every command that would run for every (tier, seed) "
             "combo, without executing them or touching the filesystem.",
    )
    ap.add_argument("--log_file", type=str, default="density_experiment_log.txt")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    results_root = Path(args.results_root)
    log_file = Path(args.log_file)

    combos = [
        (tier, n_tp, seed)
        for tier, n_tp in TIERS.items()
        for seed in range(args.n_seeds)
    ]

    log(f"Starting MTGRN density sweep: {len(TIERS)} tiers x "
        f"{args.n_seeds} seeds = {len(combos)} runs (no bin dimension -- "
        f"MTGRN pools all bins per (tier, seed), see module docstring). "
        f"dry_run={args.dry_run}", log_file)

    n_done = n_skipped = n_failed = 0
    failed_combos = []
    t_sweep_start = time.time()

    for tier, n_tp, seed in combos:
        runid = f"tier{tier}_seed{seed}"
        data_dir = data_root / f"data_tier{tier}_seed{seed}"
        metrics_path = results_root / runid / "metrics.json"

        log(f"=== {runid} (n_timepoints_keep={n_tp}) ===", log_file)

        if metrics_path.exists() and not args.dry_run:
            log(f"  SKIP -- {metrics_path} already exists (resuming)", log_file)
            n_skipped += 1
            continue

        t0 = time.time()
        try:
            run_prepare(
                seed=seed, n_timepoints_keep=n_tp, out_dir=data_dir,
                dataset_dir=args.dataset_dir, n_bins=args.n_bins,
                n_replicates=args.n_replicates, root_bin=args.root_bin,
                max_cells=args.max_cells, W=args.W, M=args.M,
                dry_run=args.dry_run, log_file=log_file,
            )
            run_train(
                data_dir=data_dir, runid=runid, seed=seed, device=args.device,
                d_model=args.d_model, n_heads=args.n_heads,
                n_temporal_blocks=args.n_temporal_blocks,
                n_spatial_blocks=args.n_spatial_blocks,
                n_epochs=args.n_epochs, patience=args.patience,
                batch_size=args.batch_size, peak_lr=args.peak_lr,
                warmup_frac=args.warmup_frac, train_frac=args.train_frac,
                results_root=results_root,
                dry_run=args.dry_run, log_file=log_file,
            )
        except Exception:
            elapsed = time.time() - t0
            log(f"  FAILED after {elapsed / 60:.1f} min:\n{traceback.format_exc()}",
                log_file)
            n_failed += 1
            failed_combos.append(runid)
            continue

        elapsed = time.time() - t0
        log(f"  done in {elapsed / 60:.1f} min", log_file)
        n_done += 1

    total_elapsed = time.time() - t_sweep_start
    log(f"\nSweep finished in {total_elapsed / 60:.1f} min. "
        f"done={n_done} skipped={n_skipped} failed={n_failed} "
        f"(of {len(combos)} total)", log_file)
    if failed_combos:
        log(f"WARNING: failed combo(s): {failed_combos} -- see {log_file} "
            f"for tracebacks. Re-running this script will retry them.",
            log_file)


if __name__ == "__main__":
    main()
