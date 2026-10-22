# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Decoding attention over a fixed-capacity BF16/FP16 cache (the MTP draft cache) whose length is read on the GPU, for CUDA graphs on
GPUs without FlashAttention (Turing): each query row attends the keys up to its own position only, so the slots past the written ones
are never read and no copy of the cache is made. The 16-bit values are decoded from their bits and every product, sum and exponential
is FP32 (IEEE), without BF16 instructions. Split over the keys (flash decoding); the partial results are merged with PyTorch."""
import triton
import triton.language as tl
from triton.language.extra import libdevice

_RUNTIME = ["COUNT", "SPLITS"]  # query rows of the call and key splits: one binary whatever their values


@triton.jit
def _load16(ptrs, mask, BF16: tl.constexpr):
    if BF16:
        return (tl.load(ptrs, mask=mask, other=0).to(tl.uint32) << 16).to(tl.float32, bitcast=True)
    return tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)


@triton.jit(do_not_specialize=_RUNTIME, do_not_specialize_on_alignment=_RUNTIME)
def _static_decode(Q, K, V, SEQLENS, PM, PL, PACC, COUNT, SPLITS, scale, STRIDE_QC, STRIDE_KN, GROUPS: tl.constexpr, KV_HEADS: tl.constexpr,
                   HEAD_DIM: tl.constexpr, ROWS_PAD: tl.constexpr, BF16: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_N: tl.constexpr):
    kv, split, row_block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    seqlen = tl.load(SEQLENS)
    total = seqlen + COUNT  # the keys written so far, this call's included
    span = tl.cdiv(tl.cdiv(total, SPLITS), BLOCK_N) * BLOCK_N
    start = split * span
    stop = tl.minimum(start + span, total)
    rows = row_block * BLOCK_R + tl.arange(0, BLOCK_R)  # row r: query i = r % COUNT of head kv * GROUPS + r // COUNT
    row_ok = rows < GROUPS * COUNT
    position = seqlen + rows % COUNT
    head = kv * GROUPS + rows // COUNT
    d = tl.arange(0, HEAD_DIM)
    q = _load16(Q + (rows % COUNT)[:, None] * STRIDE_QC + head[:, None] * HEAD_DIM + d[None, :], row_ok[:, None], BF16)
    m = tl.full((BLOCK_R,), float("-inf"), tl.float32)
    l = tl.zeros((BLOCK_R,), tl.float32)
    acc = tl.zeros((BLOCK_R, HEAD_DIM), tl.float32)
    for first in range(start, stop, BLOCK_N):
        n = first + tl.arange(0, BLOCK_N)
        key_ok = n < stop
        offsets = n[:, None] * STRIDE_KN + kv * HEAD_DIM + d[None, :]
        k = _load16(K + offsets, key_ok[:, None], BF16)
        s = tl.dot(q, tl.trans(k), input_precision="ieee") * scale
        s = tl.where(key_ok[None, :] & (n[None, :] <= position[:, None]), s, float("-inf"))
        m_new = tl.maximum(m, tl.max(s, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)  # a row with no visible key yet keeps zero weights
        alpha = libdevice.exp(m - m_safe)
        p = libdevice.exp(s - m_safe[:, None])
        v = _load16(V + offsets, key_ok[:, None], BF16)
        l = l * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p, v, input_precision="ieee")
        m = m_new
    out = (split * KV_HEADS + kv) * ROWS_PAD + rows
    tl.store(PM + out, m)
    tl.store(PL + out, l)
    tl.store(PACC + out[:, None] * HEAD_DIM + d[None, :], acc)


# Below the kernel, whose line numbers are part of Triton's cache keys.
import torch
from shared.kernels.triton_tuner import LaunchTuner

_TUNER = LaunchTuner("wan2gp_static_cache_attention", 1, "Static Cache Attention")
# (BLOCK_N, warps, stages); the first is the static configuration. FP32 tiles: q (16 rows), and one key and one value tile per stage
_CONFIGS = [(16, 4, 1), (32, 4, 1), (16, 8, 1), (32, 8, 1), (16, 4, 2), (64, 8, 1)]
_BLOCK_R = 16
_SPLITS_PER_SM = 2


def _fits(config, head_dim, device):
    # the FP32 query and key tiles, plus a little per key (Triton 3.2 and 3.6 for sm_75, head_dim 256: 33.9 / 51.3 / 86.1 KB for 16 / 32 /
    # 64 keys, whatever the stages): Turing's 64 KB take up to 32 keys
    shared = (_BLOCK_R + config[0]) * head_dim * 4 + 128 * config[0]
    return shared <= torch.cuda.get_device_properties(device).shared_memory_per_block_optin


def usable(query, cache):
    return query.is_cuda and query.dtype in (torch.bfloat16, torch.float16) and cache.dtype == query.dtype and cache.is_contiguous() \
        and query.is_contiguous() and query.shape[-1] in (64, 128, 256)


def attention(query, key_cache, value_cache, seqlens, scale):
    """query (1, count, heads, head_dim) at positions seqlens + 0..count-1, whose keys and values are already written; key_cache and
    value_cache (1, capacity, kv_heads, head_dim), contiguous; seqlens: int32 (1,) on the GPU. Returns (1, count, heads, head_dim)."""
    _, count, heads, head_dim = query.shape
    kv_heads = key_cache.shape[2]
    groups = heads // kv_heads
    rows = groups * count
    row_blocks = triton.cdiv(rows, _BLOCK_R)
    rows_pad = row_blocks * _BLOCK_R
    splits = max(1, torch.cuda.get_device_properties(query.device).multi_processor_count * _SPLITS_PER_SM // (kv_heads * row_blocks))
    bf16 = query.dtype == torch.bfloat16
    bits = (lambda t: t.view(torch.uint16)) if bf16 else (lambda t: t)
    q, k, v = bits(query), bits(key_cache), bits(value_cache)
    candidates = [config for config in _CONFIGS if _fits(config, head_dim, query.device)]

    def launch(pm, pl, pacc, config):
        block_n, warps, stages = config
        _static_decode[(kv_heads, splits, row_blocks)](q, k, v, seqlens, pm, pl, pacc, count, splits, scale, query.stride(1), key_cache.stride(1),
                                                       GROUPS=groups, KV_HEADS=kv_heads, HEAD_DIM=head_dim, ROWS_PAD=rows_pad, BF16=bf16,
                                                       BLOCK_R=_BLOCK_R, BLOCK_N=block_n, num_warps=warps, num_stages=stages)

    def partials():
        return (torch.empty((splits, kv_heads, rows_pad), dtype=torch.float32, device=query.device),
                torch.empty((splits, kv_heads, rows_pad), dtype=torch.float32, device=query.device),
                torch.empty((splits, kv_heads, rows_pad, head_dim), dtype=torch.float32, device=query.device))

    def prepare_scratch():  # the cache and the query are only read: the candidates write scratch partial results
        scratch = partials()
        return lambda config: launch(*scratch, config)

    config = _TUNER.config("decode", f"{query.dtype}|D{head_dim}|G{groups}|KV{kv_heads}", query.device, candidates, prepare_scratch, True)
    pm, pl, pacc = partials()
    launch(pm, pl, pacc, config)
    pm, pl, pacc = pm[:, :, :rows], pl[:, :, :rows], pacc[:, :, :rows]
    weights = torch.exp(pm - pm.amax(0))  # a split without keys: exp(-inf) = 0
    output = (weights.unsqueeze(-1) * pacc).sum(0).div_((weights * pl).sum(0).unsqueeze(-1))  # (kv_heads, groups * count, head_dim)
    return output.view(kv_heads, groups, count, head_dim).permute(2, 0, 1, 3).reshape(1, count, heads, head_dim).to(query.dtype)


from shared.kernels.triton_compilation_log import install_triton_compilation_logger
install_triton_compilation_logger()
