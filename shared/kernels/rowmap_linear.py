# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Projections that MMGP row views compute in place from a fused weight interleaving its projections head by head (mmgp.row_views):
the Triton GEMM reads each projection's rows where they are (as precise as cuBLAS or better: FP32 accumulation, one rounding); the
PyTorch path runs one cuBLAS product per head. Registered with MMGP at import, which then keeps such weights fused instead of copying
their projections apart at load."""
import torch
from mmgp import row_views
from .dispatch import choose, cuda_inference, triton_dtype_supported

try:
    from . import rowmap_triton as _triton
except (ImportError, OSError, RuntimeError):
    _triton = None

_DTYPES = (torch.bfloat16, torch.float16)


def _triton_usable(x, weight, head_dim):
    return (_triton is not None and cuda_inference(x, dtypes=_DTYPES, hip=True)  # the GEMM runs on HIP too (Triton's own conversions)
            and weight.dtype == x.dtype and weight.is_cuda and weight.stride(1) == 1 and x.stride(1) == 1 and head_dim % 64 == 0)


def rows(x, weight, bias, slot, slots, head_dim, start, stop):
    """Columns [start, stop) (multiples of head_dim) of projection `slot` of a weight whose rows are (heads, slots, head_dim), for x of
    shape (tokens, in_features)."""
    if choose("Interleaved Projections", owned_triton=_triton_usable(x, weight, head_dim)):
        return _triton.rows(x, weight, bias, slot, slots, head_dim, start, stop)
    out = x.new_empty((x.shape[0], stop - start))
    for column in range(start, stop, head_dim):  # the rows of one head are contiguous
        row = column // head_dim * slots * head_dim + slot * head_dim
        target = out[:, column - start:column - start + head_dim]
        if bias is None:
            torch.mm(x, weight[row:row + head_dim].T, out=target)
        else:
            torch.addmm(bias[row:row + head_dim], x, weight[row:row + head_dim].T, out=target)
    return out


def available(dtype, head_dim):
    """Whether MMGP should keep a fused interleaved weight of this dtype: the Triton GEMM makes its projections as fast as split ones."""
    return (_triton is not None and dtype in _DTYPES and head_dim % 64 == 0 and torch.cuda.is_available()
            and triton_dtype_supported(torch.device("cuda", torch.cuda.current_device()), dtype))  # BF16 needs sm80+


row_views.register_kernel(rows, available)
