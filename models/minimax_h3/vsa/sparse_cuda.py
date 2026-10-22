# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""VSA fine stage on WGPKernels' CUDA block-sparse INT8-QK attention (SageAttention 2 class): FP8 PV with per-channel V scales on
sm_89 and later, FP16 PV on sm_80/86. The tiles, scales and index lists come from the fused gather kernels; K is smoothed by its
per-head mean before quantization (softmax-invariant, as SageAttention does)."""
import functools
import os

import torch

V_FP8_RANGE = 2.25  # SageAttention 2's FP8 V range on sm_89: V / scale fits [-2.25, 2.25], the scale is applied in the epilogue
_KERNEL = os.environ.get("WGP_VSA_KERNEL", "auto").strip().lower()  # auto (WGPKernels CUDA when available) or triton: debug comparisons


@functools.cache
def _library():
    try:
        from wgpkernels import attn
        attn.load_library()
        return attn
    except (ImportError, OSError, AttributeError):
        return None


@functools.cache
def kernel_kind(device):
    """"fp8", "fp16" or None: the WGPKernels VSA kernel this device runs."""
    if _KERNEL == "triton":
        return None
    library = _library()
    return library.kernel_kind(device) if library is not None else None


PV_FP16_ACCUM = os.environ.get("WGP_VSA_PV_ACCUM", "fp32") == "fp16"  # FP16 in-block PV accumulation (SageAttention 2's variant): same speed here, less precise


def attend(kind, q_int8, q_scale, k_int8, k_scale, v, v_scale, out, lut, count, valid_count, dim):
    """out [N, H', D] = attention of the tiled INT8 queries over the listed key tiles; v is fp8 transposed [H', D, N] with v_scale
    [H', D] for "fp8", fp16 [N, H', D] for "fp16"."""
    library = _library()
    library.vsa_attention(q_int8, k_int8, v, out, lut, count, valid_count, q_scale, k_scale, v_scale, dim ** -0.5, pv_fp16_accum=PV_FP16_ACCUM)
    return out
