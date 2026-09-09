"""
MTGRN architecture, implemented from scratch from the paper's Section 3
("Deciphering Cell Lineage Gene Regulatory Network via MTGRN", ICLR 2025
submission -- see mtgrn/5401_Deciphering_Cell_Lineage_.pdf). No public
reference code exists (confirmed: checked OpenReview, GitHub, arXiv) --
see mtgrn/PROGRESS.md for the full spec derivation, judgment calls for
everything the paper leaves unspecified (N, W, M, projection head), and
what's been validated so far. Read that file before changing anything here.

MASK FORMULA -- a deliberate resolution of the paper's ambiguous notation
---------------------------------------------------------------------------
The paper writes (Eq 3, Eq 5):
    Attention(Q,K,V) = softmax( (QK^T / sqrt(d_k)) (hadamard) M ) V
with M_ij in {-inf, 1}. Read completely literally (elementwise
MULTIPLICATION of the raw score by -inf or 1), this is numerically broken:
a negative raw score times -inf gives +inf (exactly backwards -- would
make a masked-out position the MOST attended, not the least), and a raw
score of exactly 0 times -inf gives NaN. This is essentially certain to be
loose paper notation for the standard, numerically-correct Transformer
masking convention (additive: score + 0 where allowed, score + (-inf)
where disallowed) which achieves the identical intended effect (masked
positions get -inf pre-softmax, unmasked positions pass through
unchanged) without the sign/NaN hazard. Implemented here as additive
masking. This is a deliberate, documented engineering resolution of
notational ambiguity, not a deviation from the paper's intent.

BLOCK LAYOUT (Fig 2a, both temporal and spatial blocks, confirmed
identical structure): pre-LN transformer encoder block --
    x = x + Attention(LayerNorm(x))
    x = x + FeedForward(LayerNorm(x))
stacked N times (N unspecified in the paper; PROGRESS.md judgment call:
N=2).
"""
import math

import torch
import torch.nn as nn

# Large finite negative constant used for all additive masking, instead of
# literal -inf -- see build_spatial_prior_mask's NUMERICAL SAFETY docstring
# note for why (a fully-masked row's softmax should be a harmless uniform
# distribution, not NaN).
NEG_INF = -1e9


class MultiHeadMaskedAttention(nn.Module):
    """Generic multi-head self-attention with an additive mask (see module
    docstring for why additive, not the paper's literal Hadamard-with-inf
    notation). Operates on input of shape (batch, seq_len, d_model), where
    "batch" is genes G for temporal attention or timesteps W for spatial
    attention (see MTGRN.forward for the reshaping that makes this true).
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, additive_mask: torch.Tensor):
        """
        x: (batch, seq_len, d_model)
        additive_mask: (seq_len, seq_len) with 0.0 where attention is
            allowed and -inf where it is not. Broadcast across batch/heads.

        Returns:
            out: (batch, seq_len, d_model)
            attn_weights: (batch, n_heads, seq_len, seq_len) -- post-softmax,
                returned for GRN extraction from the spatial block (Section 3.4).
        """
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        # q,k,v: (B, n_heads, T, d_k)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)  # (B, n_heads, T, T)
        scores = scores + additive_mask  # broadcast (T,T) -> (B, n_heads, T, T)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = attn_weights @ v  # (B, n_heads, T, d_k)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.out_proj(out)
        return out, attn_weights


class TransformerBlock(nn.Module):
    """Pre-LN block: x + Attn(LN(x)) -> x + FF(LN(x)). See module docstring."""

    def __init__(self, d_model: int, n_heads: int, ff_dim: int | None = None, dropout: float = 0.0):
        super().__init__()
        ff_dim = ff_dim or 4 * d_model
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadMaskedAttention(d_model, n_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, additive_mask: torch.Tensor):
        attn_out, attn_weights = self.attn(self.norm1(x), additive_mask)
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x, attn_weights


class BlockStack(nn.Module):
    """N stacked TransformerBlocks sharing the same mask. Returns the
    LAST block's attention weights (the ones GRN extraction reads from,
    per Section 3.4 -- the paper extracts "the" attention matrix from the
    spatial block, singular, which we take to mean the final block's
    attention in the stack).
    """

    def __init__(self, d_model: int, n_heads: int, n_blocks: int, dropout: float = 0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout=dropout) for _ in range(n_blocks)
        ])

    def forward(self, x: torch.Tensor, additive_mask: torch.Tensor):
        attn_weights = None
        for block in self.blocks:
            x, attn_weights = block(x, additive_mask)
        return x, attn_weights


def _read_available_memory_bytes() -> int | None:
    """Live available-memory read from /proc/meminfo (Linux only -- both
    this project's dev machine and Kaggle run Linux, so no extra
    dependency like psutil is needed). Returns None instead of raising if
    unreadable, so the memory guard below degrades to a skipped-check
    warning on an unsupported platform rather than blocking a run.
    """
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024  # kB -> bytes
    except (FileNotFoundError, ValueError, IndexError):
        return None
    return None


def _read_available_gpu_memory_bytes(device) -> int | None:
    """Live free-VRAM read via torch.cuda.mem_get_info() for the given
    device. Returns None if CUDA isn't actually available/selected, so
    the caller falls back to the system-RAM check."""
    if not (torch.cuda.is_available() and str(device).startswith("cuda")):
        return None
    free_bytes, _total_bytes = torch.cuda.mem_get_info(device)
    return free_bytes


def estimate_spatial_attention_bytes(n_genes: int, n_heads: int, batch_size: int, W: int,
                                      n_spatial_blocks: int = 1, dtype_bytes: int = 4,
                                      n_live_tensors_per_block: int = 3) -> int:
    """Worst-case memory estimate for the spatial attention module's
    largest intermediate tensors, shape (batch_size * W, n_heads, n_genes,
    n_genes) -- see MTGRN.forward's spatial-block reshape, where W (the
    history window) becomes the attention "batch" dimension. This is what
    actually crashed the dev machine before (mtgrn/PROGRESS.md's
    acquisition section documents an early G=12,291 attempt, an exit-137
    OOM kill at G=769 with an unbatched eval forward, and a CUDA OOM at
    G=769/batch_size=32 on a 14.56GB T4 during Kaggle Phase 1 -- see
    below). Scales as O(batch_size * W * n_heads * n_genes^2) -- quadratic
    in gene count is the dominant risk, not batch size or W.

    n_live_tensors_per_block=3 is a deliberately rough (not precisely
    profiled) multiplier for how many tensors of this shape are
    realistically alive at once PER BLOCK during a training step (raw
    scores, softmax output, and autograd's saved-for-backward copy).
    n_spatial_blocks multiplies this again -- EACH stacked spatial block
    (BlockStack's n_blocks, MTGRN's own n_spatial_blocks) runs its own
    full G x G attention and needs its own saved activations for
    backprop through the whole stack, not just the last block's (the one
    actually returned). This factor was MISSING from the first version of
    this guard and is why it under-estimated a real GPU OOM: at G=769,
    batch_size=32, n_spatial_blocks=2 (this harness's default), the
    single-block estimate was ~8.46GB, but the actual CUDA OOM traceback
    showed ~13.9GB already in use before failing to allocate one more
    2.82GB tensor (~16.7GB actual peak) -- almost exactly the 2x a
    2-block multiplier predicts (~16.9GB). Always pass the SAME
    n_spatial_blocks the model is actually constructed with.
    """
    return (batch_size * W * n_heads * n_genes * n_genes * dtype_bytes
            * n_live_tensors_per_block * n_spatial_blocks)


def assert_spatial_attention_memory_safe(n_genes: int, n_heads: int, batch_size: int, W: int,
                                          n_spatial_blocks: int = 1, device="cpu",
                                          safety_fraction: float = 0.5) -> None:
    """Hard guard: estimate the spatial attention module's peak memory
    footprint BEFORE any tensor is allocated, and raise a clear
    MemoryError with an actionable suggestion instead of letting the OS
    or CUDA driver OOM-kill the process (the failure modes that crashed
    this project's dev machine, and separately OOM'd on a Kaggle GPU --
    see estimate_spatial_attention_bytes's docstring). Same spirit/cheap-
    insurance intent as sergio_prepare_data.py's <30-TF warning, but a
    hard stop rather than a soft print, since an OOM kill/crash is
    destructive where a bad TF count merely gives a weak result.

    IMPORTANT: pass the actual `device` the model will run on. GPU VRAM
    (checked via torch.cuda.mem_get_info when device is "cuda*") is
    typically far SMALLER than system RAM (checked via /proc/meminfo
    otherwise) -- a config that easily passes the system-RAM check can
    still CUDA-OOM on the actual GPU (confirmed: 9.08GB estimated vs
    31.61GB system RAM available passed cleanly on a Kaggle GPU
    instance, then CUDA OOM'd against the T4's real 14.56GB VRAM, because
    the original version of this guard never checked GPU memory at all
    when a GPU was requested).

    Compares against `safety_fraction` (default 50%) of CURRENTLY
    AVAILABLE memory on whichever device applies -- so the exact same
    n_genes/batch_size/W/n_spatial_blocks choice is checked against
    whatever's actually going to hold the tensors, not a hardcoded
    threshold. If neither /proc/meminfo nor CUDA's memory API is
    readable, prints a warning and skips the check rather than blocking
    the run.
    """
    needed = estimate_spatial_attention_bytes(n_genes, n_heads, batch_size, W,
                                               n_spatial_blocks=n_spatial_blocks)
    available = _read_available_gpu_memory_bytes(device)
    source = "GPU VRAM (torch.cuda.mem_get_info)"
    if available is None:
        available = _read_available_memory_bytes()
        source = "system RAM (/proc/meminfo)"
    if available is None:
        print("WARNING: could not read GPU or system memory -- skipping spatial-attention "
              "memory safety check. Proceed with caution on large gene counts.")
        return
    budget = available * safety_fraction
    if needed > budget:
        per_batch_unit = W * n_heads * n_genes * n_genes * 4 * 3 * n_spatial_blocks
        max_safe_batch = max(1, int(budget // per_batch_unit))
        raise MemoryError(
            f"Spatial attention memory guard: estimated peak footprint "
            f"{needed / 1e9:.2f}GB (n_genes={n_genes}, n_heads={n_heads}, "
            f"batch_size={batch_size}, W={W}, n_spatial_blocks={n_spatial_blocks}) "
            f"exceeds {safety_fraction:.0%} of currently available {source} "
            f"({available / 1e9:.2f}GB available, {budget / 1e9:.2f}GB budget). "
            f"This is the failure mode that crashed this project before (see "
            f"mtgrn/PROGRESS.md) -- reduce n_genes, batch_size, or W (e.g. try "
            f"--batch_size {max_safe_batch}), or run on a device with more memory, "
            f"rather than proceeding and risking an OOM kill/crash."
        )
    print(f"Spatial attention memory guard: estimated peak footprint {needed / 1e9:.2f}GB, "
          f"within {safety_fraction:.0%} of {available / 1e9:.2f}GB available {source} -- OK.")


def build_temporal_causal_mask(W: int, device=None) -> torch.Tensor:
    """M^t_ij = NEG_INF if j > i else 0 (additive form -- see module
    docstring MASK FORMULA and build_spatial_prior_mask's NUMERICAL SAFETY
    notes for why NEG_INF, not literal -inf). Cell at time i cannot attend
    to future cell j. Every row has at least the diagonal unmasked (i can
    always attend to itself), so this one can never produce an all-masked
    row -- literal -inf would have been fine here too, but NEG_INF is used
    for consistency with the spatial mask.
    """
    mask = torch.zeros(W, W, device=device)
    mask.masked_fill_(torch.triu(torch.ones(W, W, dtype=torch.bool, device=device), diagonal=1), NEG_INF)
    return mask


def build_spatial_prior_mask(prior_adjacency: torch.Tensor) -> torch.Tensor:
    """Additive spatial mask, shape (G, G), indexed [query, key].

    ORIENTATION (a judgment call the paper doesn't spell out): P_ij == 1
    means "gene i regulates gene j" (source=TF row, target=column, same
    convention as every other harness file in this project, e.g.
    pseudogrn's gt_edges / ritini's prior). Attention must let a TARGET
    gene's query aggregate information FROM its regulating TFs' key/value
    vectors -- i.e. query=target, key=regulator. So the additive mask at
    [target_row, tf_col] must be 0 (allowed) wherever P[tf, target] == 1,
    which is P transposed. Built as mask = P.T here, NOT P directly --
    using P directly would put TF rows in the query role and leave every
    non-regulating-TF gene's query row entirely masked (all -inf), which
    produced NaN after softmax in this file's own unit test before this
    was caught and fixed (see test_model.py).

    NUMERICAL SAFETY: uses a large finite negative constant (NEG_INF =
    -1e9) instead of literal -inf. A gene with ZERO regulators in the
    prior (a real, valid case -- e.g. a target no TF happens to point to
    under a sparse top-k-style prior) would otherwise get an entirely
    -inf query row, and softmax(all -inf) = NaN (0/0). With a large
    finite value instead, that row softmaxes to a harmless uniform
    distribution rather than crashing training -- functionally
    indistinguishable from true -inf for any row that DOES have real
    unmasked entries (the numeric gap swamps ordinary attention-score
    magnitudes), so genuine masking behavior is unaffected.
    """
    prior_transposed = prior_adjacency.t()
    mask = torch.zeros_like(prior_transposed, dtype=torch.float32)
    mask.masked_fill_(prior_transposed == 0, NEG_INF)
    return mask


class MTGRN(nn.Module):
    """
    Args:
        n_genes: G
        d_model: embedding dimension (paper: 128)
        n_heads: attention heads (paper: 4)
        n_temporal_blocks: N for the temporal stack (PROGRESS.md: 2)
        n_spatial_blocks: N for the spatial stack (PROGRESS.md: 2, same as temporal)
        W: history window length (PROGRESS.md judgment call: 10)
        M: forecast window length (PROGRESS.md judgment call: 5)
    """

    def __init__(self, n_genes: int, d_model: int = 128, n_heads: int = 4,
                 n_temporal_blocks: int = 2, n_spatial_blocks: int = 2,
                 W: int = 10, M: int = 5, dropout: float = 0.0):
        super().__init__()
        self.n_genes = n_genes
        self.d_model = d_model
        self.W = W
        self.M = M

        self.embed = nn.Linear(1, d_model)  # per-value embedding, Section 3.2
        self.temporal_stack = BlockStack(d_model, n_heads, n_temporal_blocks, dropout=dropout)
        self.spatial_stack = BlockStack(d_model, n_heads, n_spatial_blocks, dropout=dropout)
        # Final projection: (W * d_model) -> M, applied per gene (PROGRESS.md
        # judgment call #2 -- the paper only says "a linear layer", doesn't
        # specify how the W-indexed spatial-block output becomes M-indexed).
        self.output_proj = nn.Linear(W * d_model, M)

        self.register_buffer("temporal_mask", build_temporal_causal_mask(W), persistent=False)

    def forward(self, X: torch.Tensor, prior_adjacency: torch.Tensor):
        """
        X: (batch, W, G) -- a batch of history windows.
        prior_adjacency: (G, G) binary.

        Returns:
            Y_hat: (batch, M, G)
            spatial_attn: (batch, n_heads, G, G) -- last spatial block's
                post-softmax attention, averaged over W internally (see
                below) -- this is what GRN extraction reads (Section 3.4).
        """
        B, W, G = X.shape
        assert W == self.W and G == self.n_genes

        x = self.embed(X.unsqueeze(-1))  # (B, W, G, d_model)

        # --- Temporal block: attention WITHIN each gene, ACROSS W timesteps.
        # Genes become the "batch" dim: (B, W, G, d) -> (B*G, W, d)
        x_t = x.permute(0, 2, 1, 3).reshape(B * G, W, self.d_model)
        x_t, _ = self.temporal_stack(x_t, self.temporal_mask)
        # back to (B, G, W, d) -> (B, W, G, d) for the spatial block
        x_t = x_t.view(B, G, W, self.d_model).permute(0, 2, 1, 3)

        # --- Spatial block: attention ACROSS G genes, per timestep W.
        # Timesteps become the "batch" dim: (B, W, G, d) -> (B*W, G, d)
        spatial_mask = build_spatial_prior_mask(prior_adjacency)
        x_s = x_t.reshape(B * W, G, self.d_model)
        x_s, spatial_attn = self.spatial_stack(x_s, spatial_mask)
        # spatial_attn: (B*W, n_heads, G, G) -- one G x G matrix per
        # (batch sample, timestep). GRN extraction (Section 3.4) wants ONE
        # G x G matrix; average over the W timesteps within each sample
        # (documented choice: the paper doesn't say how a W-length window
        # collapses to one attention matrix at inference time either).
        n_heads = spatial_attn.shape[1]
        spatial_attn = spatial_attn.view(B, W, n_heads, G, G).mean(dim=1)  # (B, n_heads, G, G)

        x_s = x_s.view(B, W, G, self.d_model)

        # --- Final projection: per-gene flatten (W, d_model) -> M.
        x_out = x_s.permute(0, 2, 1, 3).reshape(B, G, W * self.d_model)  # (B, G, W*d)
        Y_hat = self.output_proj(x_out)  # (B, G, M)
        Y_hat = Y_hat.permute(0, 2, 1)  # (B, M, G)

        return Y_hat, spatial_attn
