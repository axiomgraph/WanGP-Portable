# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Optional ahead-of-time WGPKernels CUDA operations, separate from INT8 linears."""
import functools
import torch

from . import dispatch

try:
    from wgpkernels import ops as implementation
    implementation.load_library()
except (ImportError, OSError):
    implementation = None


@functools.cache
def available(device):
    """The library is built for compute capability 8.0+, but its operations only beat the PyTorch / Triton paths on 12.x GPUs where they
    were measured (its conv3d is 0.7-1.3x slower than cuDNN on 8.9): a performance choice. Eligibility before launch; never hide kernel
    execution/ABI errors."""
    return implementation is not None and device.type == 'cuda' and dispatch.capability(device.index) == (12, 0)
