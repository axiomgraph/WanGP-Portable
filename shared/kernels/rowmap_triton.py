# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""GEMM for MMGP row views: output columns [start, start + N) of one projection of a fused weight that interleaves its projections
head by head, rows (heads, slots, head_dim): out[:, n] = x @ W[row(n)].T (+ b[row(n)]), row(n) = (n // HEAD) * HEAD_STRIDE + slot offset
+ n % HEAD. A tile of output columns lies within one head (BLOCK_N divides HEAD), so its weight rows are contiguous; tiles sharing rows
of x run together for L2 reuse. FP32 accumulation, one rounding to the output dtype."""
import triton
import triton.language as tl

_RUNTIME = ["M", "N"]  # token and column counts; strides and offsets stay specialized (multiples of 16 for a model): vectorized loads


@triton.jit(do_not_specialize=_RUNTIME, do_not_specialize_on_alignment=_RUNTIME)
def _rowmap_gemm(X, W, B, OUT, M, N, K, START, SLOT_OFFSET, STRIDE_XM, STRIDE_WN, STRIDE_OM, HEAD: tl.constexpr, HEAD_STRIDE: tl.constexpr,
                 HAS_BIAS: tl.constexpr, EVEN_K: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                 GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    tiles_m, tiles_n = tl.cdiv(M, BLOCK_M), N // BLOCK_N
    group_size = GROUP_M * tiles_n
    first_m = pid // group_size * GROUP_M
    rows_in_group = tl.minimum(tiles_m - first_m, GROUP_M)
    pid_m = first_m + (pid % group_size) % rows_in_group
    pid_n = (pid % group_size) // rows_in_group
    column = START + pid_n * BLOCK_N
    weight_row = (column // HEAD) * HEAD_STRIDE + SLOT_OFFSET + column % HEAD
    rm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    rn = tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    x_ptrs = X + rm[:, None] * STRIDE_XM + rk[None, :]
    w_ptrs = W + (weight_row + rn).to(tl.int64)[None, :] * STRIDE_WN + rk[:, None]
    row_valid = rm[:, None] < M
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        if EVEN_K:
            a = tl.load(x_ptrs, mask=row_valid, other=0.0)
            b = tl.load(w_ptrs)
        else:
            a = tl.load(x_ptrs, mask=row_valid & (k + rk[None, :] < K), other=0.0)
            b = tl.load(w_ptrs, mask=k + rk[:, None] < K, other=0.0)
        acc = tl.dot(a, b, acc)
        x_ptrs += BLOCK_K
        w_ptrs += BLOCK_K
    if HAS_BIAS:
        acc += tl.load(B + weight_row + rn).to(tl.float32)[None, :]
    out_ptrs = OUT + rm[:, None] * STRIDE_OM + (pid_n * BLOCK_N + rn)[None, :]
    tl.store(out_ptrs, acc.to(OUT.dtype.element_ty), mask=row_valid)


# Below the kernel, whose line numbers are part of Triton's cache keys.
import torch
from shared.kernels.triton_tuner import LaunchTuner

_TUNER = LaunchTuner("wan2gp_rowmap_linear", 2, "Row Views")
# (BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, warps, stages); the first is the static configuration of calls too short to tune
_CONFIGS = [(128, 128, 32, 8, 4, 3), (128, 128, 32, 8, 4, 4), (128, 128, 64, 8, 4, 3), (128, 128, 64, 8, 8, 3), (256, 128, 32, 8, 8, 3),
            (128, 64, 64, 8, 4, 4), (64, 128, 64, 8, 4, 4), (128, 64, 32, 8, 4, 4)]
_TUNE_ROWS = 4096  # calls with fewer rows use the static configuration until a longer one tunes it
_SCRATCH_ROWS = 16384


def rows(x, weight, bias, slot, slots, head_dim, start, stop):
    """Columns [start, stop) (multiples of head_dim) of projection `slot` of `weight` (heads * slots * head_dim, in_features), for x
    (tokens, in_features) with contiguous rows; weight and bias in x's dtype."""
    tokens, width = x.shape
    columns = stop - start
    out = torch.empty((tokens, columns), dtype=x.dtype, device=x.device)
    candidates = [config for config in _CONFIGS if head_dim % config[1] == 0]
    if tokens == 0:
        return out

    def launch(source, target, config):
        block_m, block_n, block_k, group_m, warps, stages = config
        grid = (triton.cdiv(source.shape[0], block_m) * (columns // block_n),)
        _rowmap_gemm[grid](source, weight, weight if bias is None else bias, target, source.shape[0], columns, width, start, slot * head_dim,
                           source.stride(0), weight.stride(0), target.stride(0), HEAD=head_dim, HEAD_STRIDE=slots * head_dim,
                           HAS_BIAS=bias is not None, EVEN_K=width % block_k == 0, BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
                           GROUP_M=group_m, num_warps=warps, num_stages=stages)

    def prepare_scratch():
        sample = x[:_SCRATCH_ROWS]  # read only: the candidates write a scratch output
        scratch = torch.empty((sample.shape[0], columns), dtype=x.dtype, device=x.device)
        return lambda config: launch(sample, scratch, config)

    config = _TUNER.config("rowmap", f"{x.dtype}|K{width}|{head_dim}x{slots}|bias{bias is not None}", x.device, candidates, prepare_scratch,
                           tokens >= _TUNE_ROWS)
    launch(x, out, config)
    return out


from shared.kernels.triton_compilation_log import install_triton_compilation_logger
install_triton_compilation_logger()
