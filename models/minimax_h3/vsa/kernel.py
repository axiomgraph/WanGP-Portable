# SPDX-License-Identifier: Apache-2.0
# Derived from FastVideo's fastvideo-kernel Triton block-sparse attention (hao-ai-lab/FastVideo, Apache-2.0), itself from the
# OpenAI Triton fused attention tutorial; adapted to WanGP's seq-major tiled layout, per-row index lists and launch tuning.
"""Exact BF16 block-sparse attention over 64-token tiles: every query tile attends the key tiles of its index list.

Flash-attention forward with FP32 online softmax and FP32 accumulation, probabilities in BF16 for the PV product (as the dense
flash kernels); padding slots of a partial key tile are masked by its valid count. The index lists have one row per query tile of
the launch (`row_offset` locates the first query tile), the kernel reads only `count[row]` entries of a row.
"""
import math

import torch
import triton
import triton.language as tl

from shared.kernels.triton_tuner import LaunchTuner
from shared.kernels.triton_compilation_log import install_triton_compilation_logger

from .tiling import BLOCK

_tuner = LaunchTuner("h3_vsa_attention", 1, "H3 VSA")
_launches = ((4, 2), (4, 3), (8, 2), (8, 3))


@triton.jit(do_not_specialize=["n_rows", "row_offset", "qk_scale"])
def _attention_kernel(Q, K, V, Out, Index, Count, Valid, stride_n, stride_h, stride_ih, stride_iq, n_rows, row_offset, qk_scale,
                      D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    head = tl.program_id(1)
    q_tile = row + row_offset
    offs_m = (q_tile * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)  # row offsets in int64: long videos exceed 2^31 elements
    offs_n = tl.arange(0, BLOCK)
    offs_d = tl.arange(0, D)
    q = tl.load(Q + head * stride_h + offs_m[:, None] * stride_n + offs_d[None, :])
    m_i = tl.full([BLOCK], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK], dtype=tl.float32)
    acc = tl.zeros([BLOCK, D], dtype=tl.float32)
    count = tl.load(Count + head * n_rows + row)
    index_row = Index + head * stride_ih + row * stride_iq
    for i in range(0, count):
        tile = tl.load(index_row + i)
        start = tile.to(tl.int64) * BLOCK
        k = tl.load(K + head * stride_h + (start + offs_n)[None, :] * stride_n + offs_d[:, None])
        qk = tl.dot(q, k) * qk_scale
        valid = tl.load(Valid + tile)
        qk = tl.where(offs_n[None, :] < valid, qk, -float("inf"))
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_ij[:, None])
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = tl.load(V + head * stride_h + (start + offs_n)[:, None] * stride_n + offs_d[None, :])
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        m_i = m_ij
    acc = acc / l_i[:, None]
    tl.store(Out + head * stride_h + offs_m[:, None] * stride_n + offs_d[None, :], acc.to(Out.type.element_ty))


def attend(q, k, v, out, index, count, valid_count, row_offset):
    """out[rows of the query tiles row_offset..] = attention of tiled q [N, H, D] over the key tiles listed per (head, row).

    index: [H, rows, width] int32 tile ids; count: [H, rows] int32 entries used per row; valid_count: [n_tiles] int32.
    q, k, v and out share the [N, H, D] layout (any strides on N and H); the rows of out outside the launch are untouched."""
    heads, dim = q.shape[1], q.shape[2]
    rows = index.shape[1]
    qk_scale = dim ** -0.5 * 1.4426950408889634

    def launch(destination, config):
        _attention_kernel[(rows, heads)](q, k, v, destination, index, count, valid_count, q.stride(0), q.stride(1), index.stride(0), index.stride(1),
                                         rows, row_offset, qk_scale, D=dim, BLOCK=BLOCK, num_warps=config[0], num_stages=config[1])

    def prepare_scratch():
        scratch = torch.empty_like(out)
        return lambda config: launch(scratch, config)

    config = _tuner.config("attention", (dim, heads), q.device, _launches, prepare_scratch, tunable=rows >= 8)
    launch(out, config)
    return out


install_triton_compilation_logger()
