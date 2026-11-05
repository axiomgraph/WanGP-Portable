# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Fused tile packing, TripPool and compact attention-index construction.

Counts, head-group sizes and tile-derived strides are runtime arguments without value or alignment specialization, so a new resolution,
video length or Attention Head Split level compiles nothing."""
import struct

import torch
import triton
import triton.language as tl

from . import selection


@triton.jit(do_not_specialize=("elements",))
def _residual(Projected, Pool, Out, elements, D: tl.constexpr, B: tl.constexpr):
    offset = tl.program_id(0) * B + tl.arange(0, B)
    projected = tl.load(Projected + offset, offset < elements, 0).to(tl.float32)
    mean = tl.load(Pool + offset // D * (3 * D) + offset % D, offset < elements, 0)
    tl.store(Out + offset, projected + mean, offset < elements)


def residual(projected, pooled):
    out = torch.empty_like(projected, dtype=torch.float32)
    _residual[(triton.cdiv(out.numel(), 512),)](projected, pooled, out, out.numel(), D=projected.shape[-1], B=512)
    return out


@triton.jit(do_not_specialize=("heads", "video_tiles"))
def _gather_pool(X, Perm, Heads, Valid, Out, Pool, sxn, sxh, heads, video_tiles, D: tl.constexpr, C: tl.constexpr):
    tile, head, channel = tl.program_id(0), tl.program_id(1), tl.program_id(2) * C + tl.arange(0, C)
    row = tile * 128 + tl.arange(0, 128)
    source = tl.load(Perm + row)
    source_head = tl.load(Heads + head)
    x = tl.load(X + source[:, None] * sxn + source_head * sxh + channel[None, :], source[:, None] >= 0, 0)
    tl.store(Out + (row[:, None] * heads + head) * D + channel[None, :], x)
    if tile < video_tiles:
        valid = tl.load(Valid + tile)
        xf = x.to(tl.float32)
        mean = tl.sum(xf, 0) / tl.maximum(valid, 1)
        high = tl.max(tl.where(source[:, None] >= 0, xf, -float('inf')), 0)
        low = tl.min(tl.where(source[:, None] >= 0, xf, float('inf')), 0)
        base = Pool + (head * video_tiles + tile) * (3 * D) + channel
        tl.store(base, mean)
        tl.store(base + D, tl.where(valid > 0, high, 0.))
        tl.store(base + 2 * D, tl.where(valid > 0, low, 0.))


def gather_pool(x, layout, heads):
    count, dim = heads.numel(), x.shape[-1]
    out = x.new_empty((layout.num_slots, count, dim))
    pooled = torch.empty((count, layout.n_video_tiles, dim * 3), device=x.device, dtype=torch.float32)
    _gather_pool[(layout.n_tiles, count, dim // 32)](x, layout.perm, heads, layout.valid_count, out, pooled, x.stride(0), x.stride(1), count, layout.n_video_tiles, D=dim, C=32, num_warps=4, enable_fp_fusion=False)
    return out, pooled


@triton.jit(do_not_specialize=("heads", "elements"))
def _gather(X, Perm, Heads, Out, sxn, sxh, heads, elements, D: tl.constexpr, B: tl.constexpr):
    offset = tl.program_id(0) * B + tl.arange(0, B)
    row, head, channel = offset // (heads * D), offset // D % heads, offset % D
    source = tl.load(Perm + row, offset < elements, -1)
    source_head = tl.load(Heads + head)
    x = tl.load(X + source * sxn + source_head * sxh + channel, (offset < elements) & (source >= 0), 0)
    tl.store(Out + offset, x, offset < elements)


def gather(x, layout, heads):
    out = x.new_empty((layout.num_slots, heads.numel(), x.shape[-1]))
    _gather[(triton.cdiv(out.numel(), 1024),)](x, layout.perm, heads, out, x.stride(0), x.stride(1), heads.numel(), out.numel(), D=x.shape[-1], B=1024)
    return out


@triton.jit(do_not_specialize=("out_heads", "heads", "elements"))
def _scatter(X, Perm, Heads, Out, out_heads, heads, elements, D: tl.constexpr, B: tl.constexpr):
    offset = tl.program_id(0) * B + tl.arange(0, B)
    row, head, channel = offset // (heads * D), offset // D % heads, offset % D
    destination = tl.load(Perm + row, offset < elements, -1)
    destination_head = tl.load(Heads + head)
    x = tl.load(X + offset, offset < elements, 0)
    tl.store(Out + (destination * out_heads + destination_head) * D + channel, x, (offset < elements) & (destination >= 0))


def scatter(out, tiled, layout, heads):
    _scatter[(triton.cdiv(tiled.numel(), 1024),)](tiled, layout.perm, heads, out, out.shape[1], heads.numel(), tiled.numel(), D=out.shape[-1], B=1024)
    torch.autograd.graph.increment_version(out)


@triton.jit(do_not_specialize=("elements", "tiles"))
def _mask_scores(Scores, Valid, elements, tiles, B: tl.constexpr):
    offset = tl.program_id(0) * B + tl.arange(0, B)
    row, col = offset // tiles % tiles, offset % tiles
    valid = tl.load(Valid + col)
    score = tl.load(Scores + offset, offset < elements, 0)
    score = tl.where(row == col, float('inf'), tl.where(valid > 0, score, -float('inf')))
    tl.store(Scores + offset, score, offset < elements)


@triton.jit
def _select_ranks(selected, I, V, ranks, k, keep_below, start, c0, C: tl.constexpr, R: tl.constexpr):
    """Marks the kept ranks (by chunks of R) whose column falls in the chunk starting at c0."""
    for r0 in range(0, k, R):
        rank = r0 + tl.arange(0, R)
        index = tl.load(I + ranks + rank, rank < k, 0).to(tl.int32) + start - c0
        value = tl.load(V + ranks + rank, rank < k, -float('inf'))
        keep = (rank < keep_below) & (rank < k) & (value > -float('inf')) & (index >= 0) & (index < C - 1)
        # Ranks not kept here use the spare last lane; it never holds a column.
        selected = selected | (tl.histogram(tl.where(keep, index, C - 1), C) > 0).to(tl.int32)
    return selected


@triton.jit(do_not_specialize=("tiles", "video_tiles", "split", "k0", "k1", "lo0", "lo1", "frac0", "frac1", "start0"))
def _indices(I0, V0, I1, V1, Valid, Index, Count, tiles, video_tiles, split, k0, k1, lo0, lo1, frac0: tl.int64, frac1: tl.int64, start0,
             ALL0: tl.constexpr, ALL1: tl.constexpr, SECOND: tl.constexpr, C: tl.constexpr, R: tl.constexpr):
    # Columns by chunks of C lanes holding C - 1 columns, ranks by chunks of R: chunk sizes do not depend on the geometry. frac0 / frac1
    # are float64 bit patterns: a float argument would be FP32.
    row, head = tl.program_id(0), tl.program_id(1)
    lane = tl.arange(0, C)
    fraction = frac0.to(tl.float64, bitcast=True)
    keep0 = lo0 + (tl.floor((row + 1).to(tl.float64) * fraction) > tl.floor(row.to(tl.float64) * fraction))
    fraction = frac1.to(tl.float64, bitcast=True)
    keep1 = lo1 + (tl.floor((row + 1).to(tl.float64) * fraction) > tl.floor(row.to(tl.float64) * fraction))
    total = tl.full((), 0, tl.int32)
    for c0 in range(0, tiles, C - 1):
        col = c0 + lane
        real = (lane < C - 1) & (col < tiles)
        valid = tl.load(Valid + col, real, 0) > 0
        if row < video_tiles:
            selected = tl.full((C,), 0, tl.int32)
            if ALL0:
                selected = tl.where((col >= start0) & (col < split) & valid, 1, selected)
            else:
                selected = _select_ranks(selected, I0, V0, (head * video_tiles + row) * k0, k0, keep0, start0, c0, C, R)
            if SECOND:
                if ALL1:
                    selected = tl.where((col >= split) & (col < video_tiles) & valid, 1, selected)
                else:
                    selected = _select_ranks(selected, I1, V1, (head * video_tiles + row) * k1, k1, keep1, split, c0, C, R)
            selected = tl.where(real, tl.where(col >= video_tiles, valid.to(tl.int32), selected), 0)
        else:
            selected = valid.to(tl.int32)
        positions = total + tl.cumsum(selected) - 1
        tl.store(Index + (head * tiles + row) * tiles + positions, col, selected > 0)
        total += tl.sum(selected)
    tl.store(Count + head * tiles + row, total)


def _float64_bits(value):
    return struct.unpack("<q", struct.pack("<d", value))[0]


def indices(scores, layout, blocks):
    """Same torch.topk selection; compact sorted tile IDs without a dense mask or argsort."""
    _mask_scores[(triton.cdiv(scores.numel(), 512),)](scores, layout.valid_count, scores.numel(), layout.n_video_tiles, B=512)
    torch.autograd.graph.increment_version(scores)
    selected = []
    for block in blocks:
        lo, hi, fraction = selection.split_budget(block.budget.per_row(block.real_tokens, block.stop - block.start), block.stop - block.start)
        if block.budget.keeps_all:
            values = index = scores
        else:
            values, index = torch.topk(scores[:, :, block.start:block.stop], hi, dim=-1, sorted=True)
        selected.append((index, values, hi, lo, fraction, block.budget.keeps_all))
    first = selected[0]
    second = selected[-1]
    index = torch.empty((scores.shape[0], layout.n_tiles, layout.n_tiles), device=scores.device, dtype=torch.int32)
    count = torch.empty((scores.shape[0], layout.n_tiles), device=scores.device, dtype=torch.int32)
    _indices[(layout.n_tiles, scores.shape[0])](first[0], first[1], second[0], second[1], layout.valid_count, index, count,
        layout.n_tiles, layout.n_video_tiles, blocks[0].stop, first[2], second[2], first[3], second[3], _float64_bits(first[4]), _float64_bits(second[4]),
        blocks[0].start, ALL0=first[5], ALL1=second[5], SECOND=len(blocks) == 2, C=256, R=256, enable_fp_fusion=False)
    return index, count
