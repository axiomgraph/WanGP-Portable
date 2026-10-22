# Adapted from OpenMOSS/MOSS-TTS, revision 934d6826b084c46a0d033402174d5f8ac4ed2519.
# Copyright 2026 OpenMOSS and HuggingFace contributors. Apache-2.0; see LICENSE.
"""Qwen3 delay-pattern audio decoder, using WanGP's shared attention/cache runtime."""
import math
import torch
from torch import nn
from mmgp import runtime
from shared import attention_kit
from shared.kernels import ops
from shared.llm_engines.lmgp.layerwise import cached_attention
from shared.utils.lora_mapping import convert_lora_keys


class GenerationCancelled(Exception):
    pass


class RMSNorm(nn.Module):
    def __init__(self, width, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        return self.weight * ops.rms_norm_rounded([x], self.eps, manual=True)


class Attention(nn.Module):
    def __init__(self, cfg, features):
        super().__init__()
        self.heads, self.kv_heads, self.head_dim = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        self.features = features
        self.q_proj = nn.Linear(cfg.hidden_size, self.heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.heads * self.head_dim, cfg.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)

    def forward(self, x_list, positions, cos, sin, cache, kv_len):
        def normalize(query, key, group):
            def rotate(part, first, last, norm):
                part = norm(part)
                a, b = part.chunk(2, dim=-1)
                # Qwen3 rounds both products before addition, including in BF16.
                return part * cos[:, first:last] + torch.cat((-b, a), dim=-1) * sin[:, first:last]
            if query is not None:
                attention_kit.rows_(query, lambda part, first, last: rotate(part, first, last, self.q_norm))
            if key is not None:
                attention_kit.rows_(key, lambda part, first, last: rotate(part, first, last, self.k_norm))
        output = attention_kit.qkv_attention(x_list, self.q_proj, self.k_proj, self.v_proj, self.heads, self.head_dim, normalize,
            kv_heads=self.kv_heads, attention_fn=lambda qkv, group: cached_attention(qkv, group, cache, positions, kv_len, self.features, triton_cache=True))
        return self.o_proj(output.flatten(2))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x_list):
        return ops.gated_mlp(x_list, self.gate_proj, self.up_proj, self.down_proj, torch.nn.functional.silu)


class DecoderLayer(nn.Module):
    def __init__(self, cfg, features):
        super().__init__()
        self.self_attn = Attention(cfg, features)
        self.mlp = MLP(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, x, positions, cos, sin, cache, kv_len):
        x = x + self.self_attn([self.input_layernorm(x)], positions, cos, sin, cache, kv_len)
        return x + self.mlp([self.post_attention_layernorm(x)])


class Backbone(nn.Module):
    def __init__(self, cfg, features):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([DecoderLayer(cfg, features) for _ in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)


class MossTTS(nn.Module):
    _no_split_modules = ["DecoderLayer"]

    def __init__(self, config, features):
        super().__init__()
        self.config, self.features = config, features
        cfg = config.language_config
        self.language_model = Backbone(cfg, features)
        self.emb_ext = nn.ModuleList([nn.Embedding(config.audio_vocab_size + 1, cfg.hidden_size) for _ in range(config.n_vq)])
        self.lm_heads = nn.ModuleList([nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)] +
                                    [nn.Linear(cfg.hidden_size, config.audio_vocab_size + 1, bias=False) for _ in range(config.n_vq)])
        self._convertWeightsFloatTo = None
        self._model_dtype = torch.bfloat16
        self.kv = None
        self.abort_requested = lambda: False
        self.layer_completed = None

    def preprocess_loras(self, model_type, sd):
        return convert_lora_keys(sd, dict(self.named_modules()))

    def setup_cache(self, capacity):
        cfg = self.config.language_config
        self.kv = torch.zeros(2, cfg.num_hidden_layers, 1, capacity, cfg.num_key_value_heads, cfg.head_dim,
                              device=runtime.execution_device(self), dtype=self._model_dtype)

    def release_cache(self):
        self.kv = None

    def check_abort(self):
        if self.abort_requested():
            raise GenerationCancelled()

    def embeddings(self, tokens):
        hidden = self.language_model.embed_tokens(tokens[..., 0])
        for index, embedding in enumerate(self.emb_ext, 1):
            hidden = hidden + embedding(tokens[..., index])
        return hidden

    def forward(self, tokens, positions, kv_len):
        hidden = self.embeddings(tokens)
        cfg = self.config.language_config
        inverse = 1.0 / (cfg.rope_theta ** (torch.arange(0, cfg.head_dim, 2, device=hidden.device, dtype=torch.float32) / cfg.head_dim))
        angles = positions.float()[:, None] * inverse[None, :]
        angles = torch.cat((angles, angles), dim=-1)
        cos, sin = angles.cos()[None, :, None].to(hidden.dtype), angles.sin()[None, :, None].to(hidden.dtype)
        for index, layer in enumerate(self.language_model.layers):
            self.check_abort()
            hidden = layer(hidden, positions, cos, sin, (self.kv[0, index], self.kv[1, index]), kv_len)
            self.check_abort()
            if self.layer_completed is not None:
                self.layer_completed()
        hidden = self.language_model.norm(hidden[:, -1:])
        return [head(hidden)[:, 0] for head in self.lm_heads]

    def step(self, tokens, positions, end):
        # Decode buckets keep graph shapes stable; the device positions mask unwritten cache rows.
        kv_len = min(self.kv.shape[3], math.ceil(end / 256) * 256) if tokens.shape[1] == 1 else end
        return self(tokens, positions, kv_len)
