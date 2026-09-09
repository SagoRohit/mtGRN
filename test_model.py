"""Tiny synthetic unit test for model.py -- checks shapes, gradient flow,
NaN-safety, and (critically) that the causal and spatial masks actually
block what they're supposed to block. Not part of the Phase 1 validation
(that needs real mESC data) -- this is a pure mechanics/correctness check
on the architecture code itself, run before spending any time on data
acquisition.
"""
import torch
from model import MTGRN, build_temporal_causal_mask, build_spatial_prior_mask, NEG_INF

torch.manual_seed(0)

G, W, M, d_model, n_heads = 12, 6, 3, 16, 2
B = 4

model = MTGRN(n_genes=G, d_model=d_model, n_heads=n_heads,
              n_temporal_blocks=2, n_spatial_blocks=2, W=W, M=M)

X = torch.randn(B, W, G)
prior = torch.zeros(G, G)
# make a sparse TF-bipartite-style prior: genes 0-2 are "TFs"
prior[:3, :] = 1.0
prior.fill_diagonal_(0)

Y_hat, spatial_attn = model(X, prior)

print(f"Y_hat shape: {Y_hat.shape} (expect ({B},{M},{G}))")
assert Y_hat.shape == (B, M, G)
print(f"spatial_attn shape: {spatial_attn.shape} (expect ({B},{n_heads},{G},{G}))")
assert spatial_attn.shape == (B, n_heads, G, G)
assert not torch.isnan(Y_hat).any(), "NaN in Y_hat!"
assert not torch.isnan(spatial_attn).any(), "NaN in spatial_attn!"
print("No NaNs. Shapes correct.")

# --- gradient flow check ---
loss = ((Y_hat - torch.randn_like(Y_hat)) ** 2).mean()
loss.backward()
n_params_with_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
n_params_total = sum(1 for p in model.parameters())
print(f"Params with nonzero gradient: {n_params_with_grad}/{n_params_total}")
assert n_params_with_grad == n_params_total, "some params got no gradient!"

# --- spatial mask correctness: attention must be exactly 0 outside the prior ---
# (paper: "no regulatory edges exist between those gene pairs... values in
# the masked regions are zero"). spatial_attn is indexed [query=target,
# key=regulator] (see build_spatial_prior_mask's ORIENTATION note), so
# compare against prior.T, not prior directly.
masked_out = prior.t() == 0  # [target, tf] indexing
attn_avg_heads = spatial_attn.mean(dim=1)  # (B, G, G)
for b in range(B):
    off_prior_attn = attn_avg_heads[b][masked_out]
    assert torch.allclose(off_prior_attn, torch.zeros_like(off_prior_attn), atol=1e-6), \
        f"batch {b}: nonzero attention outside prior mask! max={off_prior_attn.abs().max()}"
print("Spatial mask correctness OK: attention is exactly 0 outside the prior everywhere.")

# --- causal mask correctness: verify via the raw temporal mask tensor ---
causal_mask = build_temporal_causal_mask(W)
# upper triangle (j > i) must be NEG_INF, lower+diagonal must be 0
upper = torch.triu(torch.ones(W, W, dtype=torch.bool), diagonal=1)
assert (causal_mask[upper] == NEG_INF).all()
assert (causal_mask[~upper] == 0).all()
print("Causal mask correctness OK: strictly upper-triangular is NEG_INF, rest is 0.")

# --- spatial mask helper correctness (note: mask is built from prior.T --
# see build_spatial_prior_mask's ORIENTATION docstring) ---
sm = build_spatial_prior_mask(prior)
masked_out_t = prior.t() == 0
assert (sm[masked_out_t] == NEG_INF).all()
assert (sm[~masked_out_t] == 0).all()
print("build_spatial_prior_mask correctness OK.")

# --- rigorous causality check: perturbing the LAST (most future) input
# timestep must NOT change the temporal stack's internal representation
# at EARLIER positions. Checking the raw mask tensor shape (above) proves
# the tensor is right; this proves it's actually being applied correctly
# end-to-end through the attention computation. ---
model.eval()
X1 = torch.randn(1, W, G)
X2 = X1.clone()
X2[:, -1, :] = torch.randn(1, G)  # perturb only the final (most recent) timestep

with torch.no_grad():
    x1 = model.embed(X1.unsqueeze(-1)).permute(0, 2, 1, 3).reshape(G, W, d_model)
    out1, _ = model.temporal_stack(x1, model.temporal_mask)
    x2 = model.embed(X2.unsqueeze(-1)).permute(0, 2, 1, 3).reshape(G, W, d_model)
    out2, _ = model.temporal_stack(x2, model.temporal_mask)

pos0_diff = (out1[:, 0, :] - out2[:, 0, :]).abs().max().item()
pos_last_diff = (out1[:, -1, :] - out2[:, -1, :]).abs().max().item()
assert pos0_diff < 1e-5, f"CAUSALITY VIOLATION: earliest position leaked future info (diff={pos0_diff})"
assert pos_last_diff > 1e-5, "sanity check failed: perturbation had no effect anywhere"
print(f"Causality check OK: earliest-position diff={pos0_diff:.2e} (no leakage), "
      f"latest-position diff={pos_last_diff:.2e} (correctly perturbed).")

print("\nALL CHECKS PASSED.")
