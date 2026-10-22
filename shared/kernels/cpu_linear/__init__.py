# Copyright (c) 2026 DeepBeepMeep - WanGP Community License 2.0 (see LICENSE.txt)
"""CPU products of quantized linear weights (linear.cpp): the linears of the blocks that MMGP leaves on the CPU (cpuCompute) compute in the
weights' own storage (mmap or pinned), with GGUF (Q4_K, Q5_K, Q6_K, Q8_0, IQ2_S, IQ3_S, Prism PTQ1_0), INT8 (Quanto qint8, INT8 ConvRot),
scaled FP8 (e4m3, e5m2) and NVFP4 weights. The modules apply their own input transforms first (ConvRot and Prism rotations, NVFP4
pre-quantization scale). Activations cross pinned buffers that GPU kernels read and write in place. Importing or using this module never
invokes a compiler: python -m shared.kernels.cpu_linear.build rebuilds the libraries."""
import ctypes
import os
import platform
from pathlib import Path

import torch

from mmgp.expert_compute import cpu_threads

FORMAT_GGUF, FORMAT_INT8_ROWS, FORMAT_FP8_ROWS, FORMAT_NVFP4 = 0, 1, 2, 3  # linear.cpp formats
FP8_E4M3, FP8_E5M2 = 0, 1
NVFP4_LEGACY, NVFP4_TENSORCORE = 0, 1

_library = None
_attempted = False
_buffers = {}


class Weight(ctypes.Structure):
    """One weight of a call (linear.cpp Weight)."""
    _fields_ = [("data", ctypes.c_void_p), ("scales", ctypes.c_void_p), ("out", ctypes.c_void_p), ("n", ctypes.c_int64), ("format", ctypes.c_int32),
                ("type", ctypes.c_int32), ("scale", ctypes.c_float), ("reserved", ctypes.c_int32)]


def library_path():
    return Path(__file__).with_name("cpu_linear_" + ("win_amd64.dll" if os.name == "nt" else "linux_x86_64.so"))


def library():
    """The native library, or None where it is not built for this platform."""
    global _library, _attempted
    if not _attempted:
        _attempted = True
        path = library_path()
        if platform.machine().lower() in ("amd64", "x86_64") and path.is_file():
            lib = ctypes.CDLL(str(path))
            lib.cpu_linear_supported.argtypes = [ctypes.c_int, ctypes.c_int]
            lib.cpu_linear_supported.restype = ctypes.c_int
            lib.cpu_linear.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int, ctypes.POINTER(Weight), ctypes.c_int]
            lib.cpu_linear.restype = ctypes.c_int
            _library = lib
    return _library


def _float_rows(scale, n):
    """A per-row scale as contiguous float32 [n] (kept with the weight), or a per-tensor scale as a float; (False, None) otherwise."""
    if scale.numel() == 1:
        return None, float(scale.float().item())
    if scale.numel() != n:
        return False, None
    return scale.reshape(n).to(torch.float32).contiguous(), 1.0


def _describe(weight):
    """(format, type, data, scales, scale, out_features) of a CPU weight, or None without a CPU kernel."""
    from shared.qtypes.gguf import GGUFWeightTensor
    data = getattr(weight, "_data", None)
    if not torch.is_tensor(data) or data.device.type != "cpu" or not data.is_contiguous():
        return None
    if isinstance(weight, GGUFWeightTensor):
        if len(weight._tensor_shape) != 2:
            return None
        return FORMAT_GGUF, int(weight._tensor_type), data, None, 1.0, int(weight._tensor_shape[0])
    n, k = map(int, weight.shape)
    if hasattr(weight, "_layout") and hasattr(weight, "_alpha"):  # NVFP4
        from shared.qtypes.nvfp4 import _NVFP4_LAYOUT_TENSORCORE
        scales = weight._scale
        if (k % 64 or tuple(data.shape) != (n, k // 2) or scales.dim() != 2 or tuple(scales.shape) != ((n + 127) // 128 * 128, k // 16)
                or not scales.is_contiguous() or scales.device.type != "cpu"):
            return None
        tensorcore = weight._layout == _NVFP4_LAYOUT_TENSORCORE
        factor = weight._alpha.float() if tensorcore else weight._alpha.float() * weight._input_global_scale.float()
        return FORMAT_NVFP4, NVFP4_TENSORCORE if tensorcore else NVFP4_LEGACY, data, scales, float(factor.item()), n
    scale = getattr(weight, "_scale", None)
    if not torch.is_tensor(scale) or tuple(data.shape) != (n, k):
        return None
    scales, value = _float_rows(scale, n)
    if scales is False:
        return None
    if data.dtype == torch.int8:
        return FORMAT_INT8_ROWS, 0, data, scales, value, n
    if data.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return FORMAT_FP8_ROWS, FP8_E4M3 if data.dtype == torch.float8_e4m3fn else FP8_E5M2, data, scales, value, n
    return None


def _spec(weight):
    spec = getattr(weight, "_cpu_linear_spec", None)
    if spec is None:
        spec = weight._cpu_linear_spec = _describe(weight) or ()
    return spec


def supported(weight):
    """Whether this CPU weight computes here (its format and this CPU)."""
    spec = _spec(weight)
    lib = library()
    return bool(spec) and lib is not None and bool(lib.cpu_linear_supported(spec[0], spec[1]))


class _Mapped:
    """A CUDA view of pinned host memory: GPU kernels read and write it in place, without the copy engines, whose queues the transfers of
    streamed blocks fill."""
    def __init__(self, host):
        self.__cuda_array_interface__ = {"shape": tuple(host.shape), "typestr": "<f4", "data": (host.data_ptr(), False), "version": 2, "strides": None}


def _buffer(name, numel, device):
    """(pinned host buffer, its CUDA view) of at least numel floats."""
    current = _buffers.get(name)
    if current is None or current[0].numel() < numel:
        if current is not None:
            torch.cuda.synchronize(device)  # GPU kernels may still read the old buffer
        host = torch.empty(max(numel, 2 * (0 if current is None else current[0].numel())), dtype=torch.float32, device="cpu", pin_memory=True)
        current = _buffers[name] = (host, torch.as_tensor(_Mapped(host), device=device))
    return current


def linear(weight, x, bias, dtype):
    """x [..., k] on the GPU (already transformed by the module), weight on the CPU (supported). Returns x . W^T (+ bias) on x's device in
    dtype. The host waits for x: the CPU computes between a GPU kernel writing x to pinned memory and one reading the result from it."""
    format, type, data, scales, scale, n = _spec(weight)
    shape, k, device = x.shape, x.shape[-1], x.device
    rows = x.numel() // k
    (host_in, mapped_in), (host_out, mapped_out) = _buffer("x", rows * k, device), _buffer("y", rows * n, device)
    mapped_in[:rows * k].view(rows, k).copy_(x.reshape(rows, k))
    x = None
    torch.cuda.current_stream(device).synchronize()  # also ends the last reads of host_out before it is written again
    call = Weight(data.data_ptr(), None if scales is None else scales.data_ptr(), host_out.data_ptr(), n, format, type, scale, 0)
    if library().cpu_linear(host_in.data_ptr(), rows, k, 1, ctypes.byref(call), cpu_threads()):
        raise RuntimeError(f"CPU linear: unsupported weight (format {format}, type {type}, {n}x{k})")
    y = mapped_out[:rows * n].view(*shape[:-1], n).to(dtype, copy=True)
    return y if bias is None else y.add_(bias.to(device=device, dtype=dtype))
