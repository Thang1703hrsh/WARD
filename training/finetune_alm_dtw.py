"""
ALM + Soft-DTW hidden-state distillation.

How it works
------------
ALM uses JAX/Flax — not PyTorch — so the distillm PyTorch DTW cannot be reused.
This file implements Soft-DTW in JAX and registers it as loss name "dtw"
inside the ALM training pipeline by monkey-patching two things:

  1. tokenkit.training.losses  — adds compute_dtw_hidden_loss()
  2. cross_tokenizer_distill   — extends the loss dispatch loop to accept "dtw"

The ALM config-YAML + --overrides CLI is preserved unchanged.

Usage
-----
  python finetune_alm_dtw.py \\
      --config=methods/alm/configs/cross_tokenizer_distill.yaml \\
      --overrides \\
      losses=[sft,alm_unconstrained,dtw] \\
      dtw_window=32 \\
      dtw_gamma=0.1 \\
      dtw_normalize=true \\
      dtw_use_divergence=false \\
      student.pretrained_model_name_or_path=... \\
      ...
"""

import os
import sys
from dataclasses import dataclass

# ── Make ALM importable ───────────────────────────────────────────────────────
REPO_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # WARD/
ALM_DIR     = os.path.join(REPO_DIR, "methods", "alm")
for _p in (REPO_DIR, ALM_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── JAX ───────────────────────────────────────────────────────────────────────
import jax
import jax.numpy as jnp

# ── ALM modules ───────────────────────────────────────────────────────────────
import cross_tokenizer_distill as _ctd   # methods/alm/cross_tokenizer_distill.py
from tokenkit import parse_args as _pa   # methods/alm/tokenkit/parse_args.py
from tokenkit.training import losses as _losses  # methods/alm/tokenkit/training/losses.py


# ═════════════════════════════════════════════════════════════════════════════
# Soft-DTW in JAX
# ═════════════════════════════════════════════════════════════════════════════

def _softmin3(a, b, c, gamma):
    """Differentiable softmin of three values: -γ · logsumexp(-[a,b,c]/γ)."""
    vals = jnp.stack([a, b, c], axis=-1) / gamma
    return -gamma * jax.nn.logsumexp(-vals, axis=-1)


def _soft_dtw_jax(cost, gamma, window):
    """
    Soft-DTW with Sakoe-Chiba band on a (T_s, T_t) cost matrix.

    Parameters
    ----------
    cost   : (T_s, T_t) float32
    gamma  : softmin temperature (smaller = closer to hard DTW)
    window : band half-width — only |i-j| <= window cells are computed

    Returns scalar alignment cost.
    """
    T_s, T_t = cost.shape
    INF = jnp.array(1e9, dtype=cost.dtype)

    # R is (T_s+1, T_t+1); index 0 is padding
    R_init = jnp.full((T_s + 1, T_t + 1), INF)
    R_init = R_init.at[0, 0].set(0.0)

    def body_i(i, R):
        def body_j(j, R):
            c   = cost[i - 1, j - 1]
            r   = _softmin3(R[i - 1, j], R[i, j - 1], R[i - 1, j - 1], gamma)
            val = jnp.where(jnp.abs(i - j) <= window, c + r, INF)
            return R.at[i, j].set(val)
        return jax.lax.fori_loop(1, T_t + 1, body_j, R)

    R = jax.lax.fori_loop(1, T_s + 1, body_i, R_init)
    return R[T_s, T_t]


def _cosine_cost_matrix(s, t):
    """
    Cosine-distance matrix between row-vectors of s and t.

    s : (T_s, D)  t : (T_t, D)  → (T_s, T_t) in [0, 2]
    """
    s = s / (jnp.linalg.norm(s, axis=-1, keepdims=True) + 1e-8)
    t = t / (jnp.linalg.norm(t, axis=-1, keepdims=True) + 1e-8)
    return 1.0 - jnp.matmul(s, t.T)


def compute_dtw_hidden_loss(args, loss_args):
    """
    Soft-DTW loss on the last hidden states of teacher and student.

    Expects loss_args.student_out.hidden_states and
    loss_args.teacher_out.hidden_states to be populated
    (requires need_hidden_states=True in the training loop).

    If args.dtw_use_divergence is True, computes:
        DTW(s, t) - 0.5 * (DTW(s, s) + DTW(t, t))

    Returns a scalar JAX array.
    """
    if (loss_args.student_out is None
            or loss_args.teacher_out is None
            or loss_args.student_out.hidden_states is None
            or loss_args.teacher_out.hidden_states is None):
        return jnp.array(0.0)

    gamma       = float(getattr(args, "dtw_gamma",          0.1))
    window      = int(getattr(args,   "dtw_window",         32))
    normalize   = bool(getattr(args,  "dtw_normalize",      True))
    use_div     = bool(getattr(args,  "dtw_use_divergence", False))

    s_h = loss_args.student_out.hidden_states[-1].astype(jnp.float32)  # (B, T_s, D_s)
    t_h = loss_args.teacher_out.hidden_states[-1].astype(jnp.float32)  # (B, T_t, D_t)

    T_s = s_h.shape[1]
    T_t = t_h.shape[1]

    def dtw_pair(s_seq, t_seq):
        """DTW between one student sequence and one teacher sequence."""
        cost = _cosine_cost_matrix(s_seq, t_seq)
        d_st = _soft_dtw_jax(cost, gamma, window)
        if use_div:
            d_ss = _soft_dtw_jax(_cosine_cost_matrix(s_seq, s_seq), gamma, window)
            d_tt = _soft_dtw_jax(_cosine_cost_matrix(t_seq, t_seq), gamma, window)
            return d_st - 0.5 * (d_ss + d_tt)
        return d_st

    batch_dtw = jax.vmap(dtw_pair)(s_h, t_h)   # (B,)

    if normalize:
        batch_dtw = batch_dtw / (T_s + T_t)

    return batch_dtw.mean()


# ═════════════════════════════════════════════════════════════════════════════
# Patch ALM's loss dispatch to recognise "dtw"
# ═════════════════════════════════════════════════════════════════════════════

def _patch_alm():
    """
    Insert "dtw" into the loss dispatch of cross_tokenizer_distill.train_step.

    Strategy
    --------
    The dispatch loop in train_step reads:
        elif loss.startswith("alm"): ...
    We rename user-visible "dtw" → "alm_dtwhidden" so it enters the
    existing `elif loss.startswith("alm"):` branch. Then we patch
    losses.compute_alm_loss to detect kind == "dtwhidden" and
    route to compute_dtw_hidden_loss instead.

    We also ensure need_hidden_states is True by ensuring "alm_latents"
    is present (with weight 0) when "alm_dtwhidden" is in the losses list.
    """

    # 1. Attach DTW function to the losses module so it's reachable
    _losses.compute_dtw_hidden_loss = compute_dtw_hidden_loss  # type: ignore[attr-defined]

    # 2. Wrap compute_alm_loss to intercept kind == "dtwhidden"
    _original_compute_alm = _losses.compute_alm_loss

    def _dtw_aware_compute_alm_loss(chunk_kind, args, loss_args, **kwargs):
        if chunk_kind == "dtwhidden":
            return compute_dtw_hidden_loss(args, loss_args)
        return _original_compute_alm(chunk_kind=chunk_kind, args=args,
                                     loss_args=loss_args, **kwargs)

    _losses.compute_alm_loss = _dtw_aware_compute_alm_loss  # type: ignore[assignment]


def _prepare_args(args):
    """
    Rename "dtw" → "alm_dtwhidden" in args.losses and inject
    a zero-weight "alm_latents" entry to ensure need_hidden_states=True.
    """
    has_dtw = "dtw" in args.losses

    # Rename dtw → alm_dtwhidden
    args.losses = [
        "alm_dtwhidden" if l == "dtw" else l
        for l in args.losses
    ]

    if not has_dtw:
        return args

    # Ensure hidden states are computed:
    # need_hidden_states = "alm_latents" in losses or "baseline_dskd" in losses
    needs_injection = (
        "alm_latents"    not in args.losses
        and "baseline_dskd" not in args.losses
    )
    if needs_injection:
        args.losses = ["alm_latents"] + args.losses
        if args.loss_weights is not None:
            args.loss_weights = [0.0] + list(args.loss_weights)
        else:
            # Default: all losses weight 1.0 except the dummy alm_latents = 0
            n = len(args.losses) - 1   # excludes the newly inserted alm_latents
            args.loss_weights = [0.0] + [1.0] * n

    return args


# ═════════════════════════════════════════════════════════════════════════════
# Extended args dataclass (adds dtw_* fields to CrossTokenizerDistillArgs)
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class CrossTokenizerDistillArgsDTW(_ctd.CrossTokenizerDistillArgs):
    """CrossTokenizerDistillArgs extended with Soft-DTW hyperparameters."""
    # Sakoe-Chiba band half-width (|i-j| <= dtw_window)
    dtw_window: int = 32
    # Softmin temperature γ — smaller = harder alignment
    dtw_gamma: float = 0.1
    # Divide DTW cost by (T_s + T_t) for scale invariance
    dtw_normalize: bool = True
    # Use DTW divergence: DTW(s,t) - 0.5*(DTW(s,s)+DTW(t,t))
    dtw_use_divergence: bool = False


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════

def main():
    # Apply patches before anything else
    _patch_alm()

    # Register extended args class so ALM's HfArgumentParser picks up dtw_* fields
    _ctd.CrossTokenizerDistillArgs = CrossTokenizerDistillArgsDTW  # type: ignore[assignment]

    # Parse YAML config + --overrides (ALM's own parse_args mechanism)
    args = _pa.parse_args(CrossTokenizerDistillArgsDTW)

    # Rename "dtw" and inject hidden-state trigger
    args = _prepare_args(args)

    # Run ALM training
    _ctd.main(args)


if __name__ == "__main__":
    main()
