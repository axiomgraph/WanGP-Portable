# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Means over the token axis of attention keys (the smoothing mean SageAttention subtracts before quantizing them), as
x.mean(dim, keepdim=True) gives them but without the partial-sum buffer of PyTorch's split reduction: its FP32 partial sums follow the
shape (twice the keys for an image's 8k tokens, a quarter of them for a 720p video's 75k), allocated before the INT8 keys.

Triton (compute capability 7.5 and up; BF16 read as bits on GPUs without BF16 conversions), for keys of MIN_BYTES and more:
- each program sums a span of tokens for HB adjacent heads (one contiguous run of channels per token), BLOCK_T rows per load, in
  one FP32 accumulator per element of the tile (elementwise operations only: Triton 3.2 miscompiled a reshape fold of vectorized tiles),
  compensated (two-sum: each accumulator carries the rounding error of its additions), so that the sums stay exact to about one FP32
  rounding whatever the span (the precision does not depend on the GPU), then reduced over the rows in FP64 into its partial sums;
- a second pass adds the partial sums in a fixed order in FP64, divides once and rounds once, correctly, to the output dtype (BF16
  written as bits on every GPU, so Turing and the others give the same values). The mean is as close to the exact one as its dtype
  allows: never further than PyTorch's FP32 sums. Deterministic for a given GPU and configuration.
The launch configuration (rows per load, warps, pipelining stages, programs per SM) is tuned once per GPU, head width, heads per program
and size class (keys held by the L2 cache or not), on a slice of a real call (the kernel only reads it), and cached. The span is a
runtime value from the programs per SM: a new resolution or video length compiles nothing (the token, head and batch counts and the
strides are not compiled in either). The partial sums take (batch, heads, spans, head_dim) FP64, a few MB. Smaller keys, where
PyTorch's single launch is faster and its buffer small, and other devices or dtypes, take the PyTorch expression."""
import functools

import torch

from . import dispatch

try:
    import triton
    import triton.language as tl
    from .bf16_bits import bf16_bits_to_fp32, fp64_to_bf16_bits
    from .triton_tuner import LaunchTuner
except ImportError:
    triton = None

MIN_BYTES = 32 << 20  # keys below this size: PyTorch (two Triton launches cost more than its one, and its buffer is small there)
# (rows per load, warps, stages, programs per SM); the first is the untuned default (RTX 40: one wave, pipelined)
_CANDIDATES = ((8, 4, 3, 1), (8, 4, 3, 2), (16, 8, 3, 1), (16, 8, 3, 2), (8, 4, 1, 8), (16, 8, 1, 8), (16, 8, 3, 16), (8, 4, 3, 32))
_TUNE_L2_MULTIPLE, _TUNE_MIN_SLICE = 4, 256 << 20  # the slice timed for keys beyond the L2 cache: the larger of 4x its size and 256 MB
# (a 5.5 MB L2 gives slices of a few thousand tokens, where launch latency, not streaming, ranks the candidates), or the whole keys
_TUNE_MIN_BYTES = 32 << 20  # smaller keys never tune (forced through Triton in tests): their timings would not represent real calls

if triton is not None:
    _TUNER = LaunchTuner("wan2gp_token_mean", 5, "Attention")

    # the strides keep Triton's divisibility specialization: always multiples of 16 elements (head widths of 16 and more times head and
    # token counts), so it never changes with the shape and gives 16-byte vector loads
    @triton.jit(do_not_specialize=["TOKENS", "SPAN", "SPANS", "HEADS"])
    def _partial_sums_kernel(X, PARTIAL, TOKENS, SPAN, SPANS, HEADS, SA0, SA1, ST, D: tl.constexpr, HB: tl.constexpr, BLOCK_T: tl.constexpr,
                             STAGES: tl.constexpr, BITS: tl.constexpr):
        span, group, a0 = tl.program_id(0), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
        c = tl.arange(0, HB * D)
        head = group * HB + c // D  # every head exists (HB divides the heads): masks by rows only, uniform across a vector
        # one contiguous run of channels per token: a head, or HB adjacent heads (SA1 == D, chosen so by the launcher)
        column = group * SA1 + c if HB == 1 else group * (HB * D) + c
        start = span.to(tl.int64) * SPAN
        end = tl.minimum(start + SPAN, TOKENS)
        base = X + a0 * SA0
        # one accumulator per element of the loaded tile (elementwise operations only, the same on every Triton version), compensated
        hi = tl.zeros((BLOCK_T, HB * D), dtype=tl.float32)
        lo = tl.zeros((BLOCK_T, HB * D), dtype=tl.float32)
        for offset in tl.range(0, SPAN, BLOCK_T, num_stages=STAGES):
            rows = start + offset + tl.arange(0, BLOCK_T)
            x = tl.load(base + rows[:, None] * ST + column[None, :], (rows < end)[:, None], 0)
            x = bf16_bits_to_fp32(x) if BITS else x.to(tl.float32)
            total = hi + x  # two-sum: lo collects the rounding error of each addition
            back = total - hi
            lo += (hi - (total - back)) + (x - back)
            hi = total
        tl.store(PARTIAL + ((a0 * HEADS + head) * SPANS + span) * D + c % D, tl.sum(hi.to(tl.float64) + lo.to(tl.float64), axis=0))

    @triton.jit
    def _round_to_odd_fp32(x):
        # FP32 rounded to odd (truncated, lowest bit set when inexact): its rounding to a shorter dtype is that of x
        f = x.to(tl.float32)
        back = f.to(tl.float64)
        bits = f.to(tl.uint32, bitcast=True)
        bits = tl.where(back != x, tl.where(tl.abs(back) > tl.abs(x), bits - 1, bits) | 1, bits)
        return bits.to(tl.float32, bitcast=True)

    @triton.jit(do_not_specialize=["TOKENS", "SPANS"])
    def _finish_kernel(PARTIAL, OUT, TOKENS, SPANS, D: tl.constexpr, KIND: tl.constexpr):
        head, a0 = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
        d = tl.arange(0, D)
        row = a0 * tl.num_programs(0) + head
        partial = PARTIAL + row * SPANS * D + d
        total = tl.zeros((D,), dtype=tl.float64)
        for span in range(0, SPANS):  # a fixed order: deterministic
            total += tl.load(partial + span * D)
        mean = total / TOKENS.to(tl.float64)  # IEEE division in FP64 (only FP32's "/" is approximate)
        if KIND == 0:  # FP32
            value = mean.to(tl.float32)
        elif KIND == 1:  # FP16
            value = _round_to_odd_fp32(mean).to(tl.float16)
        else:  # BF16 bits
            value = fp64_to_bf16_bits(mean)
        tl.store(OUT + row * D + d, value)


@functools.cache
def _device(index):
    properties = torch.cuda.get_device_properties(index)
    return properties.multi_processor_count, properties.L2_cache_size


@functools.lru_cache(maxsize=256)
def _plan(shape, strides, dim):
    """The (a0, heads, tokens) view of a mean over dim of a tensor of this shape and these strides: its sizes and strides, and the heads
    per program (4 when the heads are adjacent in memory and a multiple of 4: one contiguous run of channels per token, no head mask)."""
    others = [d for d in range(len(shape) - 1) if d != dim]
    if len(others) == 1:
        others.insert(0, None)
    a0, sa0 = (1, 0) if others[0] is None else (shape[others[0]], strides[others[0]])
    heads, sa1, width = shape[others[1]], strides[others[1]], shape[-1]
    return a0, heads, shape[dim], width, sa0, sa1, strides[dim], 4 if sa1 == width and heads % 4 == 0 else 1


def _launch(source, out, a0, heads, tokens, width, sa0, sa1, st, hb, bits, kind, config, sms):
    block_t, warps, stages, per_sm = config
    groups = triton.cdiv(heads, hb)
    spans = max(1, min(triton.cdiv(tokens, block_t), per_sm * sms // (groups * a0)))
    span = triton.cdiv(tokens, spans)
    partial = torch.empty((a0, heads, spans, width), dtype=torch.float64, device=source.device)
    _partial_sums_kernel[(spans, groups, a0)](source, partial, tokens, span, spans, heads, sa0, sa1, st, width, hb, block_t, stages, bits,
                                              num_warps=warps)
    _finish_kernel[(heads, a0)](partial, out, tokens, spans, width, kind, num_warps=1)


def _eligible(x):
    width = x.shape[-1]
    return (triton is not None and x.dim() in (3, 4) and x.numel() * x.element_size() >= MIN_BYTES and x.stride(-1) == 1 and 16 <= width <= 256
            and width & (width - 1) == 0 and dispatch.cuda_inference(x, dtypes=(torch.bfloat16, torch.float16), capability_floor=(7, 5)))


def token_mean(x, dim, fp32=False):
    """x.mean(dim, keepdim=True) of 16-bit keys or values (3 or 4 dimensions, the head dimension last and contiguous) rounded to x's
    dtype, or unrounded in FP32 with fp32=True (torch.mean(x, dim, keepdim=True, dtype=torch.float32))."""
    if not _eligible(x):
        return torch.mean(x, dim=dim, keepdim=True, dtype=torch.float32) if fp32 else x.mean(dim=dim, keepdim=True)
    a0, heads, tokens, width, sa0, sa1, st, hb = _plan(x.shape, x.stride(), dim)
    shape = list(x.shape)
    shape[dim] = 1
    out = torch.empty(shape, dtype=torch.float32 if fp32 else x.dtype, device=x.device)  # (a0, heads, width) in memory
    bits = x.dtype == torch.bfloat16 and not dispatch.triton_dtype_supported(x.device, torch.bfloat16)
    source = x.view(torch.int16) if bits else x
    kind = 0 if fp32 else 1 if x.dtype == torch.float16 else 2
    target = out.view(torch.int16) if kind == 2 else out
    sms, l2 = _device(x.device.index)
    resident = x.numel() * x.element_size() <= l2

    def scratch():  # a slice of this call's keys (read only) beyond the L2 cache unless they fit in it, and outputs of its own
        rows = tokens if resident else min(tokens, triton.cdiv(max(_TUNE_L2_MULTIPLE * l2, _TUNE_MIN_SLICE), x.numel() * x.element_size() // tokens))
        part = source.narrow(dim, 0, rows)
        sample = torch.empty_like(target)
        return lambda config: _launch(part, sample, a0, heads, rows, width, sa0, sa1, st, hb, bits, kind, config, sms)

    config = _TUNER.config("partial_sums", f"{width}|{hb}|{'l2' if resident else 'dram'}", x.device, _CANDIDATES, scratch,
                           x.numel() * x.element_size() >= _TUNE_MIN_BYTES)
    _launch(source, target, a0, heads, tokens, width, sa0, sa1, st, hb, bits, kind, tuple(config), sms)
    return out


if triton is not None:
    from shared.kernels.triton_compilation_log import install_triton_compilation_logger  # noqa: E402

    install_triton_compilation_logger()
