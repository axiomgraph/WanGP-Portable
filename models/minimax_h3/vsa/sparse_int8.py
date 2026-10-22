# SPDX-License-Identifier: BSD-3-Clause
# Reuses the Veda / SageAttention-derived block-sparse INT8 kernel (models/minimax_h3/veda/sparse_int8.py, see its NOTICE and
# SAGE_LICENSE) with 64-row query tiles: VSA's tiles are the kernel's key blocks.
"""Fast VSA fine stage: INT8 QK (one scale per 64-token block and head) and FP16 PV, as SageAttention 2 does for the dense attention
WanGP selects by default; partial tiles are masked by their valid count. The exact BF16 kernel (kernel.py) serves Preserve Precision."""
import torch

from shared.kernels.triton_tuner import LaunchTuner
from ..veda.sparse_int8 import LOG2E, _attention_kernel, quantize
from .tiling import BLOCK

_tuner = LaunchTuner("h3_vsa_attention_int8", 1, "H3 VSA INT8")
_launches = ((4, 3), (4, 2), (8, 3), (8, 2))


def query_pre_scale(dim):
    """The softmax scale folded into q before its quantization, in log2 units as the kernel expects."""
    return dim ** -0.5 * LOG2E


class Quantized:
    """INT8 q/k (one scale per tile and head) and FP16 v of one head chunk, shared by the video-row and prefix-row launches."""

    def __init__(self, q_int8, q_scale, k_int8, k_scale, v16):
        self.q_int8, self.q_scale, self.k_int8, self.k_scale, self.v16 = q_int8, q_scale, k_int8, k_scale, v16

    @classmethod
    def from_tiles(cls, q, k, v):
        q_int8, q_scale = quantize(q, BLOCK, pre_scale=query_pre_scale(q.shape[-1]))
        k_int8, k_scale = quantize(k, BLOCK)
        return cls(q_int8, q_scale, k_int8, k_scale, v if v.dtype == torch.float16 else v.to(torch.float16))


def attend(quantized, out, index, count, valid_count, row_offset):
    """out[rows of the query tiles row_offset..] = INT8-QK attention of the tiled queries over their index lists (see kernel.attend)."""
    q, k, v16 = quantized.q_int8, quantized.k_int8, quantized.v16
    heads, dim = q.shape[1], q.shape[2]
    rows = index.shape[1]
    q_rows = q[row_offset * BLOCK:]
    out_rows = out[row_offset * BLOCK:]
    q_scale = quantized.q_scale[:, row_offset:]

    def launch(destination, config):
        _attention_kernel[(rows, heads)](q_rows, k, v16, q_scale, quantized.k_scale, index, count, valid_count, destination,
                                         q.stride(0), q.stride(1), v16.stride(0), v16.stride(1), quantized.q_scale.stride(0), quantized.k_scale.stride(0),
                                         index.stride(0), index.stride(1), n_q_tiles=rows, D=dim, BLK=BLOCK, BLOCK_M=BLOCK,
                                         num_warps=config[0], num_stages=config[1])

    def prepare_scratch():
        scratch = torch.empty_like(out)
        return lambda config: launch(scratch[row_offset * BLOCK:], config)

    config = _tuner.config("attention", (dim, heads), q.device, _launches, prepare_scratch, tunable=rows >= 8)
    launch(out_rows, config)
    return out
