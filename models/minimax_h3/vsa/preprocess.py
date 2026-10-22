# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Fused tile gathering for VSA: one pass over the packed rows produces the tiled INT8 (or FP16 / FP8) copy of a head group, its
per-block scales and the fp32 sums of every kernel block, from which the per-tile means of the selection are read. The INT8 rounding
is that of veda.sparse_int8.quantize. The lookup-table kernel writes the key-block lists of the selected tiles, and the scatter kernel
returns the fine attention output to the packed rows with the gated coarse branch added."""
import torch
import triton
import triton.language as tl

from .tiling import BLOCK


@triton.jit
def _gather_block(X, Perm, Heads, block, head, sxn, sxh, D: tl.constexpr, B: tl.constexpr):
    row = block * B + tl.arange(0, B)
    source = tl.load(Perm + row)
    source_head = tl.load(Heads + head)
    channel = tl.arange(0, D)
    return tl.load(X + source[:, None] * sxn + source_head * sxh + channel[None, :], source[:, None] >= 0, 0)


@triton.jit
def _gather_int8_sum(X, Perm, Heads, Mean, Xq, Scale, Sum, sxn, sxh, pre_scale, heads, n_blocks, D: tl.constexpr, B: tl.constexpr,
                     SMOOTH: tl.constexpr):
    block, head = tl.program_id(0), tl.program_id(1)
    x = _gather_block(X, Perm, Heads, block, head, sxn, sxh, D, B).to(tl.float32)
    channel = tl.arange(0, D)
    tl.store(Sum + (head * n_blocks + block) * D + channel, tl.sum(x, 0))
    if SMOOTH:  # keys minus their per-head mean (a per-query constant of the scores: softmax-invariant), padding rows stay zero
        row = block * B + tl.arange(0, B)
        real = tl.load(Perm + row) >= 0
        x = tl.where(real[:, None], x - tl.load(Mean + head * D + channel)[None, :], 0.0)
    x = x * pre_scale
    scale = tl.max(tl.abs(x)) / 127.0
    scale = tl.where(scale > 0.0, scale, 1.0)  # an all-padding block keeps its zeros
    quantized = x / scale
    quantized += 0.5 * tl.where(quantized >= 0, 1, -1)
    row = block * B + tl.arange(0, B)
    tl.store(Xq + (row[:, None] * heads + head) * D + channel[None, :], quantized.to(tl.int8))
    tl.store(Scale + head * n_blocks + block, scale)


@triton.jit
def _gather_fp16_sum(X, Perm, Heads, X16, Sum, sxn, sxh, heads, n_blocks, D: tl.constexpr, B: tl.constexpr):
    block, head = tl.program_id(0), tl.program_id(1)
    x = _gather_block(X, Perm, Heads, block, head, sxn, sxh, D, B)
    channel = tl.arange(0, D)
    tl.store(Sum + (head * n_blocks + block) * D + channel, tl.sum(x.to(tl.float32), 0))
    row = block * B + tl.arange(0, B)
    tl.store(X16 + (row[:, None] * heads + head) * D + channel[None, :], x.to(tl.float16))


@triton.jit
def _gather_fp8_sum_t(X, Perm, Heads, Low, High, X8, Sum, VScale, sxn, sxh, heads, n_blocks, pad_len, fp8_range, D: tl.constexpr, B: tl.constexpr):
    """fp8 e4m3 blocks stored transposed [H', D, pad_len] (SageAttention 2's V layout) scaled per head channel, plus the block sums.
    The channel scale is max(|low|, |high|) / fp8_range (1 where the channel is zero), written to VScale by the first block's program."""
    block, head = tl.program_id(0), tl.program_id(1)
    x = _gather_block(X, Perm, Heads, block, head, sxn, sxh, D, B).to(tl.float32)
    channel = tl.arange(0, D)
    tl.store(Sum + (head * n_blocks + block) * D + channel, tl.sum(x, 0))
    amax = tl.maximum(-tl.load(Low + head * D + channel).to(tl.float32), tl.load(High + head * D + channel).to(tl.float32))
    scale = tl.where(amax > 0.0, tl.div_rn(amax, fp8_range), 1.0)
    if block == 0:
        tl.store(VScale + head * D + channel, scale)
    scaled = x * tl.div_rn(1.0, scale)[None, :]
    # SageAttention 2's FP8 V order: inside every 16 tokens, token t sits at (t // 8) * 2 + ((t // 2) % 4) * 4 + t % 2 (its fp8 MMA fragments)
    slot = tl.arange(0, B)
    t = slot % 16
    col = block * B + (slot // 16) * 16 + (t // 8) * 2 + ((t // 2) % 4) * 4 + t % 2
    tl.store(X8 + (head * D + channel)[None, :] * pad_len + col[:, None], scaled.to(tl.float8e4nv))


def _pool(sums, layout):
    """Per-tile means [H', n_tiles, D] from the per-block sums [H', n_blocks, D] (fp32)."""
    heads, _, dim = sums.shape
    return sums.view(heads, layout.n_tiles, layout.blocks_per_tile, dim).sum(2).div_(layout.valid_count.clamp_min(1).to(torch.float32)[None, :, None])


def gather_int8_pool(x, layout, heads, pre_scale=1.0, mean=None):
    """[S, H, D] -> (int8 blocks [N, H', D], block scales [H', n_blocks] fp32, tile means [H', n_tiles, D] fp32) of the heads `heads`;
    mean [H', D] fp32 is subtracted from the real rows before quantization (the tile means are of the raw rows)."""
    count, dim = heads.numel(), x.shape[-1]
    n_blocks = layout.n_blocks
    quantized = torch.empty((layout.num_slots, count, dim), device=x.device, dtype=torch.int8)
    scale = torch.empty((count, n_blocks), device=x.device, dtype=torch.float32)
    sums = torch.empty((count, n_blocks, dim), device=x.device, dtype=torch.float32)
    _gather_int8_sum[(n_blocks, count)](x, layout.perm, heads, mean if mean is not None else scale, quantized, scale, sums, x.stride(0), x.stride(1),
                                        pre_scale, count, n_blocks, D=dim, B=BLOCK, SMOOTH=mean is not None, num_warps=4, enable_fp_fusion=False)
    return quantized, scale, _pool(sums, layout)


def gather_fp16_pool(x, layout, heads):
    """[S, H, D] -> (fp16 blocks [N, H', D], tile means [H', n_tiles, D] fp32) of the heads `heads`."""
    count, dim = heads.numel(), x.shape[-1]
    n_blocks = layout.n_blocks
    tiles = torch.empty((layout.num_slots, count, dim), device=x.device, dtype=torch.float16)
    sums = torch.empty((count, n_blocks, dim), device=x.device, dtype=torch.float32)
    _gather_fp16_sum[(n_blocks, count)](x, layout.perm, heads, tiles, sums, x.stride(0), x.stride(1), count, n_blocks, D=dim, B=BLOCK, num_warps=4,
                                        enable_fp_fusion=False)
    return tiles, _pool(sums, layout)


def gather_fp8_pool_t(x, layout, heads, low, high, fp8_range):
    """[S, H, D] -> (fp8 blocks transposed [H', D, N], tile means [H', n_tiles, D] fp32, channel scales [H', D] fp32) of the heads
    `heads`; low / high [H', D] are the per-channel min / max of x over its rows, the scales max(|low|, |high|) / fp8_range."""
    count, dim = heads.numel(), x.shape[-1]
    n_blocks, pad_len = layout.n_blocks, layout.num_slots
    tiles = torch.empty((count, dim, pad_len), device=x.device, dtype=torch.float8_e4m3fn)  # every column is written (padding rows as 0)
    sums = torch.empty((count, n_blocks, dim), device=x.device, dtype=torch.float32)
    scale = torch.empty((count, dim), device=x.device, dtype=torch.float32)
    _gather_fp8_sum_t[(n_blocks, count)](x, layout.perm, heads, low, high, tiles, sums, scale, x.stride(0), x.stride(1), count, n_blocks, pad_len,
                                         fp8_range, D=dim, B=BLOCK, num_warps=4, enable_fp_fusion=False)
    return tiles, _pool(sums, layout), scale


@triton.jit
def _index_lut(Top, Lut, n_prefix_blocks, keep, n_video_tiles, n_blocks, BPT: tl.constexpr, W: tl.constexpr):
    """The rows of the query blocks of video tile `tile` (BPT consecutive rows): every prefix block, then the blocks of its kept
    tiles, ascending, as increments (the first absolute)."""
    head, tile = tl.program_id(0), tl.program_id(1)
    j = tl.arange(0, W)
    width = n_prefix_blocks + keep * BPT
    k = (j - n_prefix_blocks) // BPT
    kept = tl.load(Top + (head * n_video_tiles + tile) * keep + k, (j >= n_prefix_blocks) & (j < width), 0).to(tl.int32)
    ids = tl.where(j < n_prefix_blocks, j, kept * BPT + (j - n_prefix_blocks) % BPT)
    ids = tl.sort(tl.where(j < width, ids, 2147483647))  # padding sorts last
    previous = tl.gather(ids, tl.maximum(j - 1, 0), 0)
    increments = tl.where(j == 0, ids, ids - previous)
    first = (n_prefix_blocks // BPT + tile) * BPT  # the tile's first query block
    for sub in tl.static_range(BPT):
        tl.store(Lut + (head * n_blocks + first + sub) * n_blocks + j, increments, j < width)


def index_lut(top, lut, layout):
    """Writes the video query blocks' rows of lut [H', n_blocks, n_blocks] from top [H', n_video_tiles, keep_total] (kept tile ids, any order)."""
    heads, n_video_tiles, keep = top.shape
    n_prefix_blocks = layout.n_prefix_tiles * layout.blocks_per_tile
    _index_lut[(heads, n_video_tiles)](top, lut, n_prefix_blocks, keep, n_video_tiles, layout.n_blocks, BPT=layout.blocks_per_tile,
                                       W=triton.next_power_of_2(n_prefix_blocks + keep * layout.blocks_per_tile), num_warps=4)


@triton.jit
def _scatter_gated(X, Coarse, Gate, Perm, Heads, Out, sgn, sgh, out_heads, heads, n_tiles, elements, D: tl.constexpr, B: tl.constexpr, TILE: tl.constexpr):
    """Out[row of slot] = X[slot] + Coarse[tile of slot] * Gate[row of slot]: the tiled fine attention output scattered back to the
    packed rows with the gated coarse branch added, in one pass (fp32 math, one rounding)."""
    offset = tl.program_id(0) * B + tl.arange(0, B)
    slot, head, channel = offset // (heads * D), offset // D % heads, offset % D
    inside = offset < elements
    destination = tl.load(Perm + slot, inside, -1)
    destination_head = tl.load(Heads + head)
    real = inside & (destination >= 0)
    x = tl.load(X + offset, real, 0).to(tl.float32)
    coarse = tl.load(Coarse + (head * n_tiles + slot // TILE) * D + channel, real, 0)
    gate = tl.load(Gate + destination * sgn + destination_head * sgh + channel, real, 0).to(tl.float32)
    tl.store(Out + (destination * out_heads + destination_head) * D + channel, tl.fma(coarse, gate, x).to(Out.dtype.element_ty), real)


def scatter_gated(out, tiled, coarse, gate, layout, heads):
    """out [S, H, D] rows of the heads `heads` = tiled [N, H', D] fine output + coarse [H', n_tiles, D] fp32 (per tile) * gate [S, H, D]."""
    count = heads.numel()
    _scatter_gated[(triton.cdiv(tiled.numel(), 1024),)](tiled, coarse, gate, layout.perm, heads, out, gate.stride(0), gate.stride(1), out.shape[1], count,
                                                       layout.n_tiles, tiled.numel(), D=out.shape[-1], B=1024, TILE=layout.tile)
    torch.autograd.graph.increment_version(out)


from shared.kernels.triton_compilation_log import install_triton_compilation_logger
install_triton_compilation_logger()
