# Adapted from nano-vllm (https://github.com/GeeeekExplorer/nano-vllm), Copyright (c) 2025 Xingkai Yu,
# MIT License (see docs/third_party_licenses/MIT.txt). Modified for WanGP.
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import torch
from torch import nn
import torch.nn.functional as F
from shared.kernels import bf16_bits
from shared.kernels.dispatch import triton_dtype_supported

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except ImportError:
    triton = None


if triton is not None:
    @triton.jit(do_not_specialize=("TOTAL",), do_not_specialize_on_alignment=("TOTAL",))
    def _silu_mul_kernel(x, out, COLUMNS: tl.constexpr, TOTAL, BLOCK: tl.constexpr):
        index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        source = index // COLUMNS * (2 * COLUMNS) + index % COLUMNS
        gate = tl.load(x + source, index < TOTAL, other=0).to(tl.float32)
        value = tl.load(x + source + COLUMNS, index < TOTAL, other=0).to(tl.float32)
        # Preserve the rounding between the existing SiLU and multiply ops.
        activated = tl.div_rn(gate, 1.0 + libdevice.exp(-gate)).to(out.dtype.element_ty).to(tl.float32)
        tl.store(out + index, activated * value, index < TOTAL)


class SiluAndMul(nn.Module):

    def __init__(self, use_triton: bool = True):
        super().__init__()
        self.use_triton = use_triton
        self.use_bf16_bits = False  # set by models for BF16 on GPUs that run Triton without BF16 conversions (shared.kernels.bf16_bits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y

    def forward_list(self, x_list: list[torch.Tensor]) -> torch.Tensor:
        x = x_list[0]
        x_list.clear()
        if self.use_bf16_bits and x.dtype == torch.bfloat16:
            return bf16_bits.silu_mul(x)
        return silu_mul(x, self.use_triton)


def silu_mul(x, use_triton=True):
    if use_triton and x.is_cuda and triton is not None and triton_dtype_supported(x.device, x.dtype):
        x = x.contiguous()
        output = torch.empty((*x.shape[:-1], x.shape[-1] // 2), device=x.device, dtype=x.dtype)
        _silu_mul_kernel[(triton.cdiv(output.numel(), 256),)](x, output, output.shape[-1], output.numel(), 256)
        return output
    gate, value = x.chunk(2, -1)
    F.silu(gate, inplace=True).mul_(value)
    return gate.contiguous()


# Register after definitions to preserve Triton's line-number-sensitive cache keys.
if triton is not None:
    from shared.kernels.triton_compilation_log import install_triton_compilation_logger
    install_triton_compilation_logger()
