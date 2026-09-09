# MTGRN implementation — progress / resume log

Read this file FIRST in any session working on this task, before doing
anything else, to see what's already done.

## Phase 0 — Setup & spec finalization

- [DONE] 2026-09-09: Read marlene/Marlene/marlene/datasets.py's
  `load_trrust()` / `load_regnetwork()` (frozen, will be imported as-is,
  never modified). Both take `species='human'|'mouse'` and an optional
  `adata` to filter/index; both read from
  `marlene/Marlene/data/{trrust_rawdata.<species>.tsv,
  RegNetwork-<species>.source}` (mouse files confirmed present locally,
  needed for mESC in Phase 1). Both set `adata.var['is_TF']` and return a
  list of (regulator, target) gene-symbol tuples.
- [DONE] 2026-09-09: Confirmed `run_density_experiment.py` / `train_sergio.py`
  "house convention" files actually live at `pseudogrn/run_density_experiment.py`
  / `pseudogrn/train_sergio.py` (the prompt says "project root" but no such
  files exist there — this is the closest/correct match, and I wrote both
  of those myself in this same project, so I already know their CLI/
  metrics.json conventions well). metrics.json schema to match: keys
  `mean_auprc, mean_auroc, auprc_per_t, auroc_per_t, n_timepoints` (see
  pseudogrn/train_sergio.py's own `metrics` dict for the exact
  reference).
- [DONE] 2026-09-09: pseudogrn/'s replicate-pooling + DPT + --max_cells
  pattern already well understood (I wrote pseudogrn/sergio_prepare_data.py
  in this same project) -- no need to re-read, will reuse the same
  pooling/DPT/max_cells structure directly.
- [DONE] 2026-09-09: ritini/'s prior construction has CHANGED since this
  prompt was written -- the original dense TF-bipartite construction
  (`prior[is_tf_rows, :] = 1`) was REPLACED during RiTINI debugging
  (ritini/DEBUG_ATTENTION_COLLAPSE.md Checkpoint 5) with a pooled-per-cell
  Spearman top-k correlation prior, because RiTINI's graph_reg BCE loss
  made the dense prior mathematically unsatisfiable against per-target
  softmax attention. **This does NOT necessarily apply to MTGRN**: MTGRN's
  spatial mask is a HARD attention mask (-inf where P_ij=0), not a soft
  BCE-regularization target like RiTINI's -- there's no analogous
  "attention must match a dense binary target" mismatch. Decision: reuse
  the SIMPLE dense TF-bipartite construction logic (a few lines, will
  reimplement fresh since the original file no longer has it) as MTGRN's
  Phase 2 prior, per the prompt's explicit instruction ("reuse the
  harness's existing synthetic TF-bipartite prior ... do not build a new
  one") -- but flag this as a judgment call and watch for a similar
  collapse symptom during Phase 1/2 validation using the same diagnostic
  playbook (per-epoch loss logging, attn(true)-attn(false) gap tracking)
  already established for RiTINI, in case the lesson DOES transfer for a
  reason not yet apparent.

## BLOCKED: Cannot access the MTGRN paper PDF

**What's blocked:** `openreview.net/pdf?id=Ecb6HBoo1r` (and
`openreview.net/forum?id=...`, and the `api2.openreview.net/notes` API)
all return a bot-verification / CAPTCHA challenge page, not the paper
content. Tried: direct curl, WebFetch tool, r.jina.ai text-extraction
proxy, the OpenReview API endpoint directly. All blocked identically
("Verifying your browser" / `ChallengeRequiredError`, HTTP 403).

**What I confirmed instead:** a WebSearch for the exact paper title
returned only the OpenReview listing (title: "Deciphering Cell Lineage
Gene Regulatory Network via MTGRN", dated 2024-10-04) plus a search-engine
summary snippet: "transformer-based model... temporal blocks and spatial
blocks... forecasting gene expression... input passes sequentially through
temporal block and spatial block... attention matrix from spatial block
extracted to derive inferred GRN... highest AUROC on 4 of 6 datasets."
This is CONSISTENT with the user's own summary in
mtgrn_working_prompt.md but is not a substitute for the actual Section 3
methodology text, exact hyperparameters, or Table 1/2 numbers -- I have
NOT independently verified the architecture spec against the primary
source, which the task explicitly required before implementing
("Re-fetch and re-read that PDF yourself before implementing — my summary
below is a guide, not a substitute for the source").

**Why I'm not proceeding past this on the user's summary alone:** the
task itself frames faithfulness to the paper as the top priority given
there's no reference code to check against -- implementing from a
second-hand summary without verifying exact details (attention formula
constants, block count N, whether LayerNorm is pre- or post-attention,
exact loss formulation, Table 1/2 numbers to validate against) risks
building something that looks plausible but doesn't match what Phase 1's
validation bar is actually supposed to be checked against.

**What would unblock this:** the user providing the PDF directly (e.g.
downloading it themselves while logged into OpenReview and sharing the
file, or pasting the relevant Section 3 + Table 1/2 text/screenshots).

## UNBLOCKED 2026-09-09: user supplied the PDF directly
(`mtgrn/5401_Deciphering_Cell_Lineage_.pdf`, 12 pages, ICLR 2025
under-review submission, matches openreview.net/pdf?id=Ecb6HBoo1r).
Read in full (all 12 pages, including references -- no appendix exists).
Confirms the user's own prompt summary was accurate on every checkable
point (exact match: mESC AUROC=0.713/AUPRC=0.748/F1=0.694, mESC genes=774/
positive edges=8085/proportion=0.014; d_model=128, 4 heads, Adam+warmup+
CosineAnnealingLR, 20 epochs, patience=3, A100 80GB).

## Phase 0 — FINALIZED ARCHITECTURE SPEC (from primary source, Sections 3-4)

**Three modules, in order (Fig 2a, Sections 3.1-3.3):**

1. **Trajectory Inference Module**: raw E in R^(C x G) -> assign pseudotime
   per cell -> sort chronologically -> E^(t) in R^(T x G), T=C (every cell
   its own timestep, NOT a discrete/macro timepoint -- confirmed, Section
   3.1: "T is the number of time steps... T is equal to C"). Split into
   sliding windows: X_i = [e_i,...,e_{i+W-1}] (W-step history), Y_i =
   [e_{i+W},...,e_{i+W+M-1}] (M-step forecast target), dataset D =
   {(X_i,Y_i)}, X_i in R^(WxG), Y_i in R^(MxG). Paper uses Slingshot;
   PROJECT SUBSTITUTION (per task instructions): scanpy DPT pipeline
   (sc.tl.pca -> sc.tl.diffmap -> sc.tl.dpt), reusing pseudogrn/'s exact
   pattern, for consistency with the rest of this harness. Documented
   deviation, not silent.

2. **Temporal Attention Module** (Section 3.2, Eq 2-3): embed each scalar
   value X in R^(WxG) -> R^(WxGxd_model) (a learned embedding of the
   1-dim expression value into d_model), transpose to X_input in
   R^(GxWxd_model) -- attention runs WITHIN each gene, ACROSS the W
   timesteps (genes are the "batch" dim here). Causal mask M^t_ij = -inf
   if j>i else 1 (cell at time i cannot attend to future cell j).
   TemporalAttention(Q,K,V) = softmax((QK^T/sqrt(d_k)) had-mult M^t) V,
   standard multi-head. Block layout (Fig 2a, confirmed): LayerNorm ->
   TemporalAttention -> residual add -> LayerNorm -> FeedForward ->
   residual add (pre-LN transformer encoder block), stacked N times
   ("Nx" in Fig 2a, no numeric value given anywhere in the paper --
   confirmed, no appendix).

3. **Spatial Attention Module** (Section 3.3, Eq 4-5): temporal block's
   output X_output in R^(GxWxd_model) transposed to X_hat_output in
   R^(WxGxd_model) -- attention runs across G genes, per timestep (W is
   the "batch" dim). Requires prior adjacency P in R^(GxG) (paper uses
   NicheNet, R-only; PROJECT SUBSTITUTION per task instructions: reuse
   this harness's TF-bipartite-style synthetic prior for Phase 2 SERGIO,
   and Marlene's frozen load_trrust/load_regnetwork for Phase 1 mESC --
   both documented deviations). Spatial mask M^s_ij = -inf if P_ij=0 else
   1. SpatialAttention same formula as temporal, masked by M^s instead of
   M^t. Same pre-LN block layout, stacked N times (same N as temporal
   block, per Fig 2a showing "Nx" on both halves symmetrically -- reading
   this as ONE shared N, not two independently-tuned values, since the
   paper never distinguishes them).
   Output -> single Linear layer -> Y_hat in R^(MxG). Loss = MSE(Y_hat,
   Y_i) (Eq 6) -- no other loss term, confirmed (no classification
   pretext, no graph regularization loss unlike Marlene/RiTINI).

**GRN extraction** (Section 3.4, Fig 2b): AFTER training, FREEZE the
model, run on TEST split, extract attention matrix H in R^(GxG) from the
spatial block. Values outside the prior mask are ~0 post-softmax
(already masked). Discrete-edge extraction (for F1, paper's own metric):
multiply H by each TFG's prior-network out-degree, rank descending, take
top-K (K=|true edges|). For OUR primary metrics (AUPRC/AUROC, matching
pseudogrn/train_sergio.py's compute_auprc_auroc convention): score the
raw continuous H values against ground truth directly, ranked, no
thresholding -- confirmed this is exactly what the paper's own AUPRC/AUROC
computation does too (Section 4, "Metrics": "areas under precision-recall
and ROC curves, using edges in the true GRN as ground truth and the
ranked edges... as predictions" -- i.e. the paper's own AUPRC/AUROC is
ALREADY continuous/unthresholded scoring of H, not the top-K discrete
version; top-K is only for their separately-reported F1). Will implement
BOTH scoring paths as the task requires (continuous = primary, degree-
weighted top-K = secondary sanity check against paper's F1 only).

**Confirmed hyperparameters** (Section 4, "Reproducibility"): d_model=128,
n_heads=4, Adam optimizer, warmup 0->1e-4 then CosineAnnealingLR, 20
epochs, early stopping patience=3, trained on 80GB A100 (we're on Kaggle
T4 -- will adjust batch size/epoch budget only if forced, and document if
so, per task instructions). No batch size, no train/test split fraction,
no optimizer warmup LENGTH (just "warmup...to 1e-4") given anywhere in the
12 pages -- also judgment calls, see below.

## Phase 0 — JUDGMENT CALLS (unspecified in paper, decided here, not guessed silently)

1. **N (block repeat count) = 2.** Rationale: paper shows "Nx" with no
   value; 2 is a modest, defensible default balancing capacity (d_model=
   128/4-heads already substantial per block) against overfitting risk on
   SERGIO's comparatively small windowed training sets (unlike mESC's
   likely-thousands of cells). Will note in Phase 1 if 2 clearly
   underfits/overfits mESC and reconsider before Phase 2.
2. **Final Y_hat projection (W-length internal repr -> M-length forecast)
   is unspecified in the paper's text.** Fig 2a shows spatial block's
   output going through "a linear layer" directly to Y_hat in R^(MxG),
   but doesn't say how the W-indexed spatial-block output (still shape
   (W,G,d_model) at that point -- spatial attention operates PER
   timestep, doesn't change W) becomes M-indexed. Decision: per-gene
   flatten the (W, d_model) representation and apply a single
   Linear(W*d_model, M) shared across genes (G is the batch dim at this
   point, same as it was for spatial attention) -> output (G,M) ->
   transpose to (M,G) = Y_hat. This is the simplest, most standard
   forecasting-head design consistent with "a linear layer" (singular)
   in the text and requires no autoregression (matches the paper's
   single-shot MSE loss over the whole Y_i at once).
3. **W (history window) = 10, M (prediction window) = 5.** No guidance in
   the paper. Chosen to (a) give the causal temporal mask enough real
   history to be meaningful (W=1 or 2 would make temporal attention
   nearly vacuous), (b) keep M modest so single-shot forecasting stays a
   tractable, well-posed regression target, (c) keep W+M small enough
   that even SERGIO's smallest density tier (pooled cell count, not
   n_timepoints_kept -- see Phase 2 notes) yields many training windows.
   Revisit if Phase 1 mESC results look poor -- may need tuning.
4. **Train/test split = 80/20, chronological (NOT random shuffle) on the
   pseudotime-ordered sequence.** I.e. the first 80% of pseudotime-ordered
   cells (by window start index) are train, the last 20% are test/GRN-
   extraction, matching the paper's own Fig 2(b) framing of running
   "Test Data" through the "Fixed MTGRN" for GRN extraction -- a
   temporally-held-out split is the only choice consistent with
   "extract GRN from cells at the LATER end of the trajectory" style
   evaluation implied by the paper's in-silico-perturbation section
   (Section 5.3) discussing directional differentiation. A random
   shuffle split would leak temporally-adjacent windows across train/test
   given the sliding-window construction, which would be a validity bug,
   not a neutral choice.
5. **Batch size**: will pick empirically based on what fits in Kaggle T4
   memory during Phase 1 (paper doesn't report one); documenting whatever
   value is actually used once Phase 1 runs.
6. **Optimizer warmup length**: paper says "warmup...0 to 1e-4" with no
   step/epoch count. Will use a standard linear warmup over the first 10%
   of total training steps (common Transformer convention), documented as
   a guess if Phase 1 results look sensitive to it.

## Phase 0 — remaining reused-file reading (deferred, not blocking)
Have NOT yet re-read pseudogrn/'s exact DPT/max_cells code line-by-line
in this session (already know the pattern from having authored it) --
will do a final confirmation pass before writing sergio_prepare_data_mtgrn.py
in Phase 2, not needed for Phase 1 model implementation.

## STATUS: Phase 0 complete. Starting Phase 1: model.py implementation +
## tiny synthetic unit test, THEN mESC data acquisition.

## Phase 1 — [DONE] 2026-09-09: model.py implemented + unit-tested

`mtgrn/model.py` -- MultiHeadMaskedAttention, TransformerBlock (pre-LN),
BlockStack (N-stacked), MTGRN top-level module. `mtgrn/test_model.py` --
shape/gradient/NaN/mask-correctness/causality checks, all passing (run
with `mtgrn/.venv` -- CPU torch 2.14.0, isolated env per proposal
Section 5.6, `pip install -r requirements.txt` once written).

**Two real bugs found and fixed during unit testing (not guessed --
caught by the test suite itself):**

1. **Spatial mask orientation was backwards.** Built `mask = P` directly
   (query=source/TF rows), which left every non-TF gene's query row
   entirely masked (all NEG_INF) since only TF rows had 1s in P -- softmax
   over an all-masked row is NaN (confirmed: first test run produced NaN
   in Y_hat immediately). Root cause: attention must let a TARGET's query
   aggregate FROM its regulating TFs' key/value vectors, so the mask needs
   query=target, key=regulator, which is `P.T`, not `P`. Fixed in
   `build_spatial_prior_mask` (now documented in its own docstring,
   ORIENTATION section) -- this is a genuine judgment call the paper's Eq
   4-5 doesn't spell out (P_ij is defined as "gene i regulates gene j"
   but the paper never states which axis plays query vs key in Eq 5).
2. **Literal `-inf` in the mask is numerically unsafe.** Even with
   orientation fixed, any gene with genuinely ZERO regulators in a
   sparser prior (a real, valid case, not just this bug) would still
   produce an all-masked query row -> NaN. Switched to a large finite
   constant (`NEG_INF = -1e9`) everywhere instead of literal `-inf` --
   makes an all-masked row softmax to a harmless uniform distribution
   instead of crashing, with no effect on genuine masking behavior for
   rows that do have real entries (the numeric gap swamps ordinary
   attention-score magnitudes).

**Also resolved a genuine notational ambiguity in the paper** (not a
bug, a documented interpretation): Eq 3/5 write the mask as a Hadamard
(elementwise) PRODUCT with a matrix of {-inf, 1}, which is numerically
broken if read completely literally (negative raw score x -inf = +inf,
exactly backwards; 0 x -inf = NaN). Implemented as the standard,
numerically-correct Transformer convention instead: ADDITIVE masking
(score + 0 or score + NEG_INF), which achieves the paper's stated intent
(masked positions get ~0 attention post-softmax) without the sign/NaN
hazard. Documented in model.py's own module docstring (MASK FORMULA
section) since this is exactly the kind of paper-fidelity judgment call
this task's instructions asked to be logged, not silently resolved.

**Rigorous causality verification** (beyond just checking the mask
tensor looks right): confirmed end-to-end that perturbing the LAST
(most-future) input timestep produces IDENTICAL temporal-block output at
the EARLIEST position (diff=0.0 exactly), while correctly changing the
latest position's own output -- proves the causal mask is actually being
applied through the real attention computation, not just present as a
correctly-shaped tensor that might not be wired in correctly.

## STATUS: Phase 1 architecture DONE + verified. Next: mESC data
## acquisition (data_mesc.py), then train_mesc_validation.py.

## Phase 1 — [DONE] 2026-09-09: data_mesc.py written + run
Produced `mtgrn/data_mesc/{ExpressionData.csv, PseudoTime.csv,
gt_edges.csv, prep_meta.json}`: 769 genes x 421 cells, 52,874 ground-truth
(ChIP-seq) edges. See the acquisition section above for the full
calibration story.

## Phase 1 — [DONE] 2026-09-09: train_mesc_validation.py written, fixed, smoke-tested

**Bug found and fixed while wiring this up**: `marlene.datasets.load_trrust`/
`load_regnetwork` use a RELATIVE path (`'data/trrust_rawdata.<species>.tsv'`),
resolved against the process's CWD, not the frozen file's own location.
Since this script runs from `mtgrn/`, not `marlene/Marlene/`, it would
raise FileNotFoundError as-is. Fixed by temporarily `os.chdir`-ing into
`marlene/Marlene/` around just the loader call (restored in a `finally`
block) -- same class of fix as every other frozen-code compatibility
adaptation in this project (adapt the CALLER, never edit the frozen
source).

**Also fixed a real conflation bug in my own first draft**: had been using
the PRIOR network's own edges (TRRUST/RegNetwork) AS the ground truth for
scoring, instead of the separate Pratapa/BEELINE ChIP-seq ground truth
(data_mesc.py's gt_edges.csv). These are two different resources with two
different roles (see data_mesc.py and train_mesc_validation.py module
docstrings) -- caught before running any real training, not after.

**2-epoch smoke test** (--n_epochs 2 --patience 10, to check wiring only,
not a real result): TRRUST prior filtering shrinks the gene set further,
769 -> **G=146** (TRRUST's mouse coverage is much narrower than the full
selected set -- expected, TRRUST only covers well-studied regulatory
genes). Ground truth after this filtering: 4,695 edges among 146 genes --
a much DENSER subgraph than the full 769-gene set (TRRUST-covered genes
tend to be well-connected regulatory hubs), giving this run's own chance
baseline AUPRC=0.222 (yet another reminder the absolute numbers here
won't resemble the paper's 0.014-baseline mESC setup -- see acquisition
section above). Pipeline ran end-to-end without errors: 407 total
windows (325 train/82 test chronological split), training loss dropped
epoch 1->2, no NaNs, no crashes.

**Full 20-epoch run (paper's exact epoch/patience settings) launched,
seed=0** -- in progress as of this checkpoint (background task
brboirvbi, output written to mesc_validation_results/). If this session
is resumed before that finishes: check `mtgrn/mesc_validation_results/
metrics_seed0.json` for whether it already completed.

## STATUS: awaiting Phase 1 seed-0 training run to finish. Next: read
## the result, decide if it's worth running seeds 1-2 for the "seed-
## robust margin" bar, then write the PASS/FAIL verdict into this file.

## Phase 1 — mESC data acquisition [IN PROGRESS] 2026-09-09

**Source confirmed and used**: the original BEELINE benchmark data itself
(Pratapa et al. 2020, Nature Methods -- exactly the paper's own cited
ground-truth source, "Pratapa et al. (2020)"), via its Zenodo record:
DOI 10.5281/zenodo.3701939, "Benchmarking algorithms for gene regulatory
network inference from single-cell transcriptomic data". Two files:
`BEELINE-data.zip` (262MB, expression matrices for all cell
types/lineages) and `BEELINE-Networks.zip` (16.8MB, ground-truth
reference networks). Exact download commands used:

```bash
curl -sL -o BEELINE-data.zip "https://zenodo.org/records/3701939/files/BEELINE-data.zip"
curl -sL -o BEELINE-Networks.zip "https://zenodo.org/records/3701939/files/BEELINE-Networks.zip"
```

This is the PRIMARY source (not a GitHub mirror like CEFCON/GENELink) --
preferred since it's exactly what the MTGRN paper itself cites, removing
any doubt about which processed variant (full gene set vs TFs+500/+1000
subset) is inside. Download complete: `BEELINE-data/inputs/scRNA-Seq/mESC/`
(ExpressionData.csv: 18,385 genes x 421 cells, PseudoTime.csv, and
GeneOrdering.csv -- BEELINE's own per-gene variability ranking, VGAM
p-value + variance) and `Networks/mouse/mESC-ChIP-seq-network.csv`
(977,841 raw candidate edges) from BEELINE-Networks.zip.

**Table 2 sanity check: PARTIALLY RESOLVED, documented limitation.**
Spent real effort trying to reproduce the paper's exact 774 genes / 8085
edges from the raw BEELINE files before giving up on an exact match --
tried: (1) full expr-genes ∩ ChIP-seq-network-genes = 12,291 genes /
806,741 edges (way too many, and G=12,291 is computationally infeasible
for GxG attention regardless -- confirms SOME gene reduction is required
for practicality, not just fidelity); (2) the four standard, PUBLICLY
DOCUMENTED BEELINE variants used by follow-up papers (GENELink, CEFCON,
etc, confirmed via web search): top-500 var genes, top-1000, TF+500,
TF+1000 -- none land near 774 genes when reconstructed from GeneOrdering.csv
+ the network's own regulator set (top500-based: 694 genes/48,312 edges;
top1000-based: 1,147/78,261); (3) tried the alternate mESC-lofgof-network.csv
ground-truth variant instead of ChIP-seq -- 8,351 genes/39,720 edges,
also no match; (4) calibrated top-N directly against gene COUNT: top-580
variable genes (by GeneOrdering.csv's VGAM p-value) unioned with the
ChIP-seq network's own regulator set, intersected with genes present in
both ExpressionData and the network, lands at **769 genes** -- within
0.6% of the paper's 774, i.e. essentially an exact gene-count match.
**But edges at that same gene set = 52,969, vs the paper's 8085 (~6.5x
too many)**, and this ratio holds at every gene-count I tried (edges
never come down to 8085-scale no matter which N is chosen) -- meaning
whatever ground-truth EDGE refinement the paper's authors actually
applied (likely a specific ChIP-seq peak-confidence/significance
threshold, or a curated subset BEELINE's own undocumented processing
scripts apply) is not reproducible from the raw Zenodo files alone, and
no public description of that exact step was found (checked BEELINE's
own README, searched for GENELink/CEFCON preprocessing descriptions).

**Decision**: proceed with the topn=580 reconstruction (769 genes, 52,969
edges from mESC-ChIP-seq-network.csv) as the closest achievable
reasonable variant, GENE-count validated against Table 2, EDGE-count
NOT validated -- this is a real, acknowledged limitation, not a silent
gap. Consequence: this dataset's own chance-level/prevalence baseline is
~0.09 (52969/(769*768)), ~6.5x higher than the paper's reported mESC
proportion of 0.014 -- so Phase 1's AUPRC number will NOT be directly
comparable in absolute terms to the paper's Table 1 0.748, and the
GENIE3 comparison baseline (0.168) is similarly not apples-to-apples on
a much denser graph. Per the validation bar's own wording ("clearly
demonstrate the model is learning real structure, not chance-level" --
not "must hit 0.748 exactly"), Phase 1's PASS/FAIL judgment will instead
compare MTGRN's AUPRC/AUROC against THIS reconstructed dataset's OWN
chance baseline (~0.09 AUPRC / 0.5 AUROC) -- consistent with how every
other model in this project (Marlene/PseudoGRN/RiTINI) already reports
a baseline-relative verdict, not an absolute cross-paper number.

## Phase 1 — [DONE] 2026-09-09: seed-0, 20-epoch training run complete
(background task from the prior session; found already finished, with no
verdict yet recorded, when this session resumed -- reading `mtgrn/
mesc_validation_results/metrics_seed0.json` directly rather than
re-running). TRRUST prior, G=146 (see acquisition section above),
n_train_windows=325, n_test_windows=82, ran the full paper hyperparameters
(20 epochs, patience=3) to completion (no NaN/crash).

**Result: AUROC=0.5046 (baseline 0.5000), AUPRC=0.2248 (this dataset's
own baseline 0.2218), degree-weighted top-K F1=0.2884.** Both metrics are
essentially AT chance level -- AUROC is 0.0046 above 0.5 (noise-scale),
AUPRC is 0.0030 above its own baseline (a ~1.4% relative lift, not a
"clearly and substantially exceeds baseline" result). This does NOT
clear the validation bar in PHASE 1 step 5 ("results near baseline ...
-> STOP, do NOT proceed to Phase 2 ... debug").

## Phase 1 — VERDICT: FAIL. Root cause diagnosed (not just re-confirmed
## near-baseline), root cause is a DATA/PRIOR structural mismatch, not an
## architecture bug. See diagnostic below before deciding how to proceed.

**Diagnostic performed** (same playbook as RiTINI's DEBUG_ATTENTION_COLLAPSE.md
Checkpoints 1.5-4: load the trained checkpoint, split attention scores by
true-edge / false-edge / out-of-prior-mask, rather than only looking at
the aggregate AUPRC/AUROC number). Script was throwaway
(`/tmp/.../scratchpad/diagnose_mesc.py`), not added to the repo; results
below are what matters.

1. **The TRRUST prior for this 146-gene set is extremely sparse: only 218
   edges total (density 0.0103, avg in-degree 1.49 per target out of 145
   possible regulators).** MTGRN's spatial attention is a HARD mask
   (additive -1e9 outside the prior, confirmed correctly implemented and
   verified again in this diagnostic: out-of-prior attention mean =
   0.002, effectively zero) -- so attention can architecturally never be
   nonzero for any gene pair NOT in the prior, no matter how training
   goes.
2. **Consequence: only 89 of the 4,695 ChIP-seq ground-truth edges (1.9%)
   are even reachable through the prior mask.** This is a hard RECALL
   CEILING baked into the experiment design itself, not a model failure
   -- 98.1% of true edges are structurally unscoreable regardless of
   training quality. Exactly the same category of problem as RiTINI's
   Checkpoint 4 finding (Granger-causality prior had a 22% recall
   ceiling; this is worse, at 1.9%).
3. **Worse: even among the 89 reachable true edges, mean attention
   (0.380) is LOWER than the 129 reachable-but-false edges (0.536) --
   attention is, if anything, weakly anti-correlated with truth on the
   small reachable subset**, not just failing to differentiate. Sample
   sizes are small (89 vs 129) so this specific direction could be noisy,
   but it is certainly not showing a positive learned signal.
4. **Checked whether a different/denser frozen prior would raise the
   recall ceiling meaningfully, before concluding this is structural**:
   - RegNetwork instead of TRRUST: G=609 (larger gene set, RegNetwork's
     mouse coverage is broader), 2,821 prior edges, density 0.0076, GT
     edges reachable = 1,328/39,012 = **3.4%** -- better than TRRUST's
     1.9% but still means 96.6% of true edges are unreachable.
   - Union of TRRUST ∪ RegNetwork: 3,004 edges, density 0.0078, reachable
     = 1,383/39,743 = **3.5%** -- essentially no improvement over
     RegNetwork alone (RegNetwork already dominates the union).
   All three frozen-prior options tested land in the same 2-4% recall-
   ceiling regime. This is NOT a per-prior-choice fluke.

**Root cause (structural, not a code bug)**: this validation setup pairs
a very SPARSE curated prior (TRRUST/RegNetwork, ~0.8-1% edge density --
these are curated, high-confidence literature resources by design) with
a very DENSE reconstructed ChIP-seq ground truth (~9-22% edge density,
already flagged above in the acquisition section as ~6.5x denser than
the paper's own reported mESC edge count/proportion of 0.014, because
the exact BEELINE edge-refinement step the paper's authors used could
not be reproduced from the raw Zenodo files). A ~1% density prior simply
cannot cover more than a few percent of a ~10-20% density ground truth
by construction, independent of whether MTGRN's architecture is
faithful or buggy. Confirmed the architecture mechanics themselves are
NOT the problem: masking direction, causal-mask orientation, and
additive-mask numerics were all independently unit-tested in Phase 1's
model.py work (see above) and re-confirmed here (out-of-prior attention
≈0, row-sums over in-prior entries ≈1.0 as expected for a masked
softmax).

**This is a genuine, unresolved ambiguity requiring a user decision**,
per this task's own BLOCKED-entry instructions -- not something to
silently work around by picking a fix myself, matching the RiTINI
precedent (DEBUG_ATTENTION_COLLAPSE.md Checkpoint 3's "reported to user
for direction rather than guessing at threshold tuning"). Options on the
table (not yet decided, see chat with user):
(a) Use RegNetwork instead of TRRUST as the Phase 1 prior (recall
    ceiling 3.4% vs 1.9% -- still low, but the best of the tested
    single-source options, and a larger gene set to boot).
(b) Try to tighten the ChIP-seq ground-truth reconstruction further
    (accept it will very likely still land far denser than the paper's
    0.014 proportion, per the acquisition section's already-exhausted
    attempts -- diminishing-returns risk).
(c) Treat Phase 1 mESC validation as inconclusive due to a resource
    mismatch (not a code-fidelity question) and note that Phase 2's
    SERGIO sweep uses a COMPLETELY DIFFERENT, denser, purpose-built
    synthetic TF-bipartite prior (not TRRUST/RegNetwork) -- so this
    specific recall-ceiling problem may not carry over to Phase 2 at
    all, even though Phase 1 (as specified) can't produce a clean
    pass/fail verdict with the currently-available real-data priors.
(d) Some combination/other option the user prefers.

## STATUS: BLOCKED on user decision for how to proceed past Phase 1's
## FAIL verdict (structural prior/ground-truth density mismatch, not an
## architecture bug -- see diagnostic above). NOT proceeding to Phase 2
## or attempting further ad hoc fixes until the user decides.

---

## Phase 1 -- positive-control test (undecided-among-options question:
## is the recall-ceiling problem the WHOLE story, or is there also a real
## bug underneath it?). User's own framing: don't pick among options
## (a)/(b)/(c)/(d) above until this is actually tested, not assumed.

**What this tests**: a prior = TRRUST union RegNetwork union the mESC
ChIP-seq ground truth ITSELF, on the FULL (unfiltered) 769-gene data_mesc
set -- guarantees every true edge is reachable through the spatial mask
(100% recall ceiling by construction, not an estimate). NOT a real prior
(leaks ground truth into the mask) -- only ever a diagnostic: if AUPRC/
AUROC are STILL chance-level under this prior, the recall-ceiling
diagnosis above is wrong or incomplete and there's a real bug to find. If
they clear baseline decisively, the recall-ceiling diagnosis is confirmed
as the whole story.

**Already implemented** (`--prior_source positive_control` in
train_mesc_validation.py) -- this was built and partially run in an
earlier pass through this session that was never written back to this
file (a documentation gap, not intentional). What actually happened,
reconstructed from the code's own comments:
1. First attempt: unbatched single-shot eval forward over the full
   82-window test set at G=769 (`model(X_test, ...)` in one call).
   **OOM-killed (exit 137) on this machine's 7GB RAM.** Root cause: the
   spatial attention module treats W (=10) as an extra batch dimension
   (see model.py's MTGRN.forward), so an unbatched eval forward
   materializes a (82*10, 4, 769, 769) attention-score tensor = ~7.8GB
   for ONE tensor alone, already exceeding this machine's total RAM
   before even considering the softmax output / autograd overhead.
2. Fixed by adding `evaluate_in_batches()` -- chunks eval over
   `batch_size` (default 32) instead of the full test set at once, under
   `torch.no_grad()`. This shrinks the same estimate to ~3GB/tensor for
   eval, which survives (no autograd graph retained under no_grad).
   **This fix was applied but the positive-control run was NEVER
   completed successfully after it** -- no `metrics_*positive_control*.json`
   exists anywhere in this repo. Training (not just eval) at the same
   batch_size=32 does NOT get the no_grad benefit and was very likely
   the second crash the user reported -- never confirmed by a log,
   because there wasn't one to check.

**NOT re-attempting this run locally.** Per explicit user instruction,
this machine is for writing/reviewing code only (same policy as every
other model in this project) -- training happens on Kaggle. Two things
done here instead:

1. **Added a memory-safety guard to `model.py`** (shared harness code,
   not just this one script): `estimate_spatial_attention_bytes()` +
   `assert_spatial_attention_memory_safe()`. Estimates the spatial
   attention module's peak tensor footprint
   (batch_size * W * n_heads * n_genes^2 * 4 bytes * a conservative
   3x-live-tensors multiplier) BEFORE any tensor is allocated, compares
   against a safety fraction (default 50%) of CURRENTLY AVAILABLE memory
   (read live from `/proc/meminfo`, no new dependency), and raises a
   clear `MemoryError` with a suggested safe `--batch_size` instead of
   letting the OS OOM-kill the process. Verified (not just written):
   - G=769, batch_size=32, W=10 (the exact positive_control default that
     crashed before): correctly raises on this machine (9.08GB estimated
     vs 2.28GB budget), suggests `--batch_size 8`.
   - G=146 (the TRRUST run that actually succeeded): correctly passes
     (0.33GB estimated).
   - G=12,291 (the earlier full-gene-set mistake from the acquisition
     section above): correctly raises by 3 orders of magnitude (2.32TB
     estimated) -- confirms the guard would have caught that one too.
   Wired into `train_mesc_validation.py`'s `main()`, called right after
   the final gene count G is known, before any window/model tensors are
   built. Same cheap-insurance spirit as sergio_prepare_data.py's <30-TF
   warning, but a hard stop (not a soft print) since an OOM kill is
   destructive (can take the whole session down) where a bad TF count
   just gives a weak result.
2. **The positive-control script itself is otherwise unchanged and ready
   to run on Kaggle**: `python train_mesc_validation.py --data_dir
   data_mesc --prior_source positive_control --seed 0`. On Kaggle's
   larger RAM budget, the guard's live /proc/meminfo read means the SAME
   command may pass at batch_size=32, or may still ask for a smaller
   batch_size depending on the actual instance -- the error message (if
   any) will say exactly what to use, no guessing needed.

**mtgrn/ is now fully self-contained** (per explicit user instruction --
"every needed code should be inside one folder", so the whole folder can
be pushed/uploaded to Kaggle on its own). Previously this script reached
across to the sibling `marlene/Marlene/` project via a relative sys.path
insert for `load_trrust`/`load_regnetwork` -- fixed by vendoring:
- `mtgrn/vendor_marlene_datasets.py`: verbatim copy of just
  `load_trrust`/`load_regnetwork` (not the rest of marlene's datasets.py,
  which mtgrn/ doesn't need) -- frozen, byte-identical, never modified,
  same convention as ritini/'s vendored `ritini_repo/` clone.
- `mtgrn/data/trrust_rawdata.mouse.tsv` + `mtgrn/data/RegNetwork-mouse.source`:
  copied unmodified from `marlene/Marlene/data/` (the two loader
  functions' relative-path convention, `data/trrust_rawdata.<species>.tsv`
  etc., needed these to live under mtgrn/'s own `data/` for the existing
  chdir-based fix to keep working without editing the frozen function
  bodies -- `train_mesc_validation.py` now chdirs into `THIS_DIR`, mtgrn/
  itself, instead of the old `MARLENE_DIR`).
Verified (not just assumed): re-imported `vendor_marlene_datasets` fresh
and called both loaders standalone -- 7,057 TRRUST rows / 323,636
RegNetwork rows loaded correctly from the vendored files.
`mtgrn/data_mesc/` (5.7MB, already-prepared mESC data) was already
self-contained -- no change needed there. Nothing in `mtgrn/*.py` now
references any path outside `mtgrn/` (checked by grep).

## STATUS: mtgrn/ is self-contained -- push/upload this one folder to
## Kaggle, nothing else needed. Positive-control script ready to run
## there. Memory guard added and verified. NOT deciding among Phase 1
## options (a)/(b)/(c)/(d) yet -- waiting on the user to run this on
## Kaggle and report back AUPRC/AUROC, per their explicit instruction not
## to guess at this before the test is actually run.

