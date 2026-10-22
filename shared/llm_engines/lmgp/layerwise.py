# Copyright (c) 2026 DeepBeepMeep - WanGP Community License 2.0 (see LICENSE.txt)
"""GPU segments of hybrid CPU/GPU autoregressive decoders.

The sequence loop stays in Python for CPU experts. MMGP graphs each GPU segment,
and owns all weight placement. Reuse the same attention/DeltaNet kernels as the
ordinary LM engine runner, each where the LM features allow it (Triton, FlashAttention 2).
"""
from functools import cache

import torch
from shared.attention import pay_attention


@cache
def _fla_supported(device):
    if device.type != "cuda":
        return False
    # FLA's DeltaNet kernels compute BF16 in Triton: Ampere or newer on NVIDIA. Keep HIP's existing path.
    if torch.version.hip is None and torch.cuda.get_device_capability(device) < (8, 0):
        print("[LM] PyTorch DeltaNet: the FLA kernels need an NVIDIA GPU of compute capability 8.0 or newer.")
        return False
    return True


def cached_attention(qkv, group, cache, positions, kv_len, features, *, triton_cache=False):
    query, key, value = qkv
    qkv.clear()
    key_cache, value_cache = cache
    start, stop = group.kv.start, group.kv.stop
    heads = stop - start
    groups = query.shape[2] // heads
    key, value = key[:, :, ::groups], value[:, :, ::groups]
    if features.triton and triton_cache and start == 0 and stop == key_cache.shape[2]:
        from .layers.attention import store_kvcache
        slots = positions[None, :] + torch.arange(query.shape[0], device=positions.device)[:, None] * key_cache.shape[1]
        store_kvcache(key.contiguous().flatten(0, 1), value.contiguous().flatten(0, 1), key_cache, value_cache,
                      slots.flatten(), use_triton_kv_cache=True)
    else:
        key_cache[:, :, start:stop].index_copy_(1, positions, key)
        value_cache[:, :, start:stop].index_copy_(1, positions, value)
    if features.flash:
        from .layers.attention import _DEFAULT_FLASH_ATTN_WITH_KVCACHE
        lengths = (positions[-1:] + 1).to(dtype=torch.int32).expand(query.shape[0]).contiguous()
        # Several query rows (prompt processing) fill the GPU without splitting the keys, whose float32 partial outputs would grow with
        # the rows; a decoding step keeps FlashAttention's choice.
        return _DEFAULT_FLASH_ATTN_WITH_KVCACHE(query, key_cache[:, :, start:stop], value_cache[:, :, start:stop],
                                              cache_seqlens=lengths, causal=True, num_splits=1 if query.shape[1] > 1 else 0)
    key, value = key_cache[:, :kv_len, start:stop], value_cache[:, :kv_len, start:stop]
    if groups > 1:
        key, value = key.repeat_interleave(groups, dim=2), value.repeat_interleave(groups, dim=2)
    mask = torch.arange(kv_len, device=query.device)[None, :] <= positions[:, None]
    return pay_attention([query, key, value], attention_mask=mask[None, :, None, :], force_attention="sdpa")


def delta_rule(query, key, value, decay, beta, initial_state, *, features, recurrent, torch_impl):
    if features.triton and _fla_supported(query.device):
        from .models.qwen3_5 import (_DEFAULT_FAST_RECURRENT_GATED_DELTA_RULE, _DEFAULT_FAST_CHUNK_GATED_DELTA_RULE,
                                    configure_qwen35_fla_prefill_autotune)
        configure_qwen35_fla_prefill_autotune(query.device)
        fn = _DEFAULT_FAST_RECURRENT_GATED_DELTA_RULE if recurrent else _DEFAULT_FAST_CHUNK_GATED_DELTA_RULE
        if fn is None:
            raise RuntimeError("The layerwise decoder's Triton DeltaNet requires the shared FLA kernels")
        return fn(query.contiguous(), key.contiguous(), value.contiguous(), g=decay.contiguous(), beta=beta.contiguous(),
                  initial_state=initial_state, output_final_state=True, use_qk_l2norm_in_kernel=True)
    return torch_impl(query, key, value, decay, beta, initial_state)
