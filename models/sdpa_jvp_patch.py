"""Drop-in replacement for `F.scaled_dot_product_attention` that detects
forward-AD dual tensors (the mechanism torch.func.jvp uses internally) and
routes those calls through the NVlabs/rcm Triton flash-attention+JVP kernel.

Non-JVP calls fall through to the regular SDPA (flash kernel via PyTorch).
"""

import torch
import torch.nn.functional as F
import torch.autograd.forward_ad as fwAD
import triton

from models.flash_attention_jvp_triton import _attn_fwd


_orig_sdpa = F.scaled_dot_product_attention


def _has_tangent(t):
    if t is None or not torch.is_tensor(t):
        return False
    return fwAD.unpack_dual(t).tangent is not None


def _triton_jvp_forward(q, k, v, tq, tk, tv, sm_scale):
    """Call the rcm Triton kernel directly (no autograd.Function, so no functorch issue)."""
    assert q.shape[:-2] == k.shape[:-2] == v.shape[:-2]
    assert k.shape[-2] == v.shape[-2] and q.shape[-1] == k.shape[-1]
    B, H = q.shape[:-2]
    SEQ_Q, SEQ_K = q.shape[-2], k.shape[-2]
    HEAD_Q, HEAD_V = q.shape[-1], v.shape[-1]
    if sm_scale is None:
        sm_scale = HEAD_Q ** (-0.5)
    o = torch.empty((B, H, SEQ_Q, HEAD_V), device=q.device, dtype=q.dtype)
    to = torch.empty_like(o)
    M = torch.empty((B, H, SEQ_Q), device=q.device, dtype=torch.float32)

    def grid(args):
        return (triton.cdiv(SEQ_Q, args["BLOCK_M"]), B * H, 1)

    _attn_fwd[grid](
        q, k, v, tq, tk, tv, sm_scale, M, o, to,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        B, H, SEQ_Q, SEQ_K, HEAD_Q, HEAD_V,
    )
    return o, to


def sdpa_jvp_aware(query, key, value, attn_mask=None, dropout_p=0.0,
                   is_causal=False, scale=None, enable_gqa=False):
    """Patched SDPA: same API as F.scaled_dot_product_attention.

    When q/k/v carry a forward-AD tangent (inside torch.func.jvp or
    fwAD.dual_level), call the Triton kernel directly; otherwise fall through.
    """
    if attn_mask is None and dropout_p == 0.0 and not is_causal and not enable_gqa and (
        _has_tangent(query) or _has_tangent(key) or _has_tangent(value)
    ):
        qp, qt = fwAD.unpack_dual(query)
        kp, kt = fwAD.unpack_dual(key)
        vp, vt = fwAD.unpack_dual(value)
        qt = qt if qt is not None else torch.zeros_like(qp)
        kt = kt if kt is not None else torch.zeros_like(kp)
        vt = vt if vt is not None else torch.zeros_like(vp)
        # Ensure contiguous (Triton kernel requires matching strides).
        qp = qp.contiguous(); kp = kp.contiguous(); vp = vp.contiguous()
        qt = qt.contiguous(); kt = kt.contiguous(); vt = vt.contiguous()
        out_p, out_t = _triton_jvp_forward(qp, kp, vp, qt, kt, vt, scale)
        return fwAD.make_dual(out_p, out_t)

    return _orig_sdpa(query, key, value,
                      attn_mask=attn_mask, dropout_p=dropout_p,
                      is_causal=is_causal, scale=scale, enable_gqa=enable_gqa)


def install():
    """Replace F.scaled_dot_product_attention globally."""
    F.scaled_dot_product_attention = sdpa_jvp_aware
