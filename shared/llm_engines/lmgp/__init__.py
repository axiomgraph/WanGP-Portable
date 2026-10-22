# Adapted from nano-vllm (https://github.com/GeeeekExplorer/nano-vllm), Copyright (c) 2025 Xingkai Yu,
# MIT License (see docs/third_party_licenses/MIT.txt). Modified for WanGP.
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
__all__ = ["LLM", "SamplingParams"]


def __getattr__(name):
    if name == "LLM":
        from .llm import LLM
        return LLM
    if name == "SamplingParams":
        from .sampling_params import SamplingParams
        return SamplingParams
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
