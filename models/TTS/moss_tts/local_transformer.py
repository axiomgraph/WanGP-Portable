# Adapted from OpenMOSS/MOSS-TTS, revision 934d6826b084c46a0d033402174d5f8ac4ed2519.
# Copyright 2026 OpenMOSS contributors. Apache-2.0; see LICENSE and NOTICE.
"""Qwen3 temporal decoder and the released one-block codebook decoder."""
import math
import torch
from torch import nn
from mmgp import runtime
from shared import attention_kit
from shared.llm_engines.lmgp.layerwise import cached_attention
from shared.utils.lora_mapping import convert_lora_keys
from .transformer import Backbone, GenerationCancelled


class DepthAttention(nn.Module):
    def __init__(self, cfg, features):
        super().__init__()
        self.heads, self.width = cfg.n_head, cfg.n_embd // cfg.n_head
        self.features = features
        self.q_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.k_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.v_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)

    def forward(self, x_list, positions, cos, sin, cache):
        def rotate(q, k, group):
            def rows(part, first, last):
                even, odd = part[..., ::2], part[..., 1::2]
                rotated = torch.stack((-odd, even), dim=-1).flatten(-2)
                return part * cos[:, first:last] + rotated * sin[:, first:last]
            for value in (q, k):
                if value is not None:
                    attention_kit.rows_(value, rows)
        value = attention_kit.qkv_attention(x_list, self.q_proj, self.k_proj, self.v_proj, self.heads, self.width, rotate,
            attention_fn=lambda qkv, group: cached_attention(qkv, group, cache, positions, 13, self.features, triton_cache=True))
        return self.c_proj(value.flatten(2))


class DepthMLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc_in = nn.Linear(cfg.n_embd, cfg.n_inner)
        self.fc_out = nn.Linear(cfg.n_inner, cfg.n_embd)

    def forward(self, x):
        return self.fc_out(torch.nn.functional.silu(self.fc_in(x), inplace=True))


class DepthBlock(nn.Module):
    def __init__(self, cfg, features):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, eps=cfg.layer_norm_epsilon)
        self.attn = DepthAttention(cfg, features)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, eps=cfg.layer_norm_epsilon)
        self.mlp = DepthMLP(cfg)

    def forward(self, x, positions, cos, sin, cache):
        x = x + self.attn([self.ln_1(x)], positions, cos, sin, cache)
        return x + self.mlp(self.ln_2(x))


class DepthTransformer(nn.Module):
    def __init__(self, cfg, features):
        super().__init__()
        self.h = nn.ModuleList([DepthBlock(cfg, features) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd, eps=cfg.layer_norm_epsilon)


class MossLocalTTS(nn.Module):
    _no_split_modules = ["DecoderLayer", "DepthBlock"]
    _offload_hooks = ["forward", "depth_step", "audio_embedding", "audio_head", "text_head"]

    def __init__(self, config, features):
        super().__init__()
        self.config, self.features = config, features
        cfg = config.qwen3_config
        self.transformer = Backbone(cfg, features)
        self.local_transformer = DepthTransformer(config.gpt2_config, features)
        self.audio_embeddings = nn.ModuleList([nn.Embedding(size, cfg.hidden_size) for size in config.audio_codebook_sizes])
        self.audio_lm_heads = nn.ModuleList([nn.Linear(cfg.hidden_size, size, bias=False) for size in config.audio_codebook_sizes])
        self.local_text_lm_head = nn.Linear(cfg.hidden_size, 2, bias=False)
        self.split_linear_modules_map = {"c_attn": {"mapped_modules": ["q_proj", "k_proj", "v_proj"], "split_sizes": [config.gpt2_config.n_embd] * 3}}
        self._convertWeightsFloatTo = None
        self._model_dtype = torch.bfloat16
        self.kv = self.depth_kv = None
        self.abort_requested = lambda: False
        self.layer_completed = None

    def split_map(self):
        return {f"local_transformer.h.{i}.attn.c_attn": {
            "mapped_modules": [f"local_transformer.h.{i}.attn.{name}_proj" for name in ("q", "k", "v")],
            "split_sizes": [self.config.gpt2_config.n_embd] * 3}
            for i in range(self.config.gpt2_config.n_layer)}

    def preprocess_loras(self, model_type, sd):
        return convert_lora_keys(sd, dict(self.named_modules()), split_linear_modules_map=self.split_linear_modules_map)

    def check_abort(self):
        if self.abort_requested():
            raise GenerationCancelled()

    def setup_cache(self, capacity):
        cfg = self.config.qwen3_config
        device = runtime.execution_device(self)
        self.kv = torch.zeros(2, cfg.num_hidden_layers, 1, capacity, cfg.num_key_value_heads, cfg.head_dim, device=device, dtype=self._model_dtype)
        cfg = self.config.gpt2_config
        self.depth_kv = torch.zeros(2, cfg.n_layer, 1, 13, cfg.n_head, cfg.n_embd // cfg.n_head, device=device, dtype=self._model_dtype)

    def release_cache(self):
        self.kv = self.depth_kv = None

    def forward(self, tokens, positions, kv_len):
        cfg = self.config.qwen3_config
        hidden = self.transformer.embed_tokens(tokens[..., 0])
        for index, embedding in enumerate(self.audio_embeddings, 1):
            valid = tokens[..., index] != self.config.audio_pad_code
            hidden = hidden + embedding(tokens[..., index].masked_fill(~valid, 0)) * valid.unsqueeze(-1)
        inverse = 1.0 / (cfg.rope_theta ** (torch.arange(0, cfg.head_dim, 2, device=hidden.device, dtype=torch.float32) / cfg.head_dim))
        angles = positions.float()[:, None] * inverse[None]
        angles = torch.cat((angles, angles), dim=-1)
        cos, sin = angles.cos()[None, :, None].to(hidden.dtype), angles.sin()[None, :, None].to(hidden.dtype)
        for index, layer in enumerate(self.transformer.layers):
            self.check_abort()
            hidden = layer(hidden, positions, cos, sin, (self.kv[0, index], self.kv[1, index]), kv_len)
            self.check_abort()
            if self.layer_completed is not None:
                self.layer_completed()
        return self.transformer.norm(hidden[:, -1:])

    def step(self, tokens, positions, end):
        kv_len = min(self.kv.shape[3], math.ceil(end / 256) * 256) if tokens.shape[1] == 1 else end
        return self(tokens, positions, kv_len)

    def depth_step(self, hidden, position):
        cfg = self.config.gpt2_config
        width = cfg.n_embd // cfg.n_head
        inverse = 1.0 / (cfg.rope_base ** (torch.arange(0, width, 2, device=hidden.device, dtype=torch.float32) / width))
        angles = position.float()[:, None] * inverse[None]
        cos = angles.cos().repeat_interleave(2, -1)[None, :, None].to(hidden.dtype)
        sin = angles.sin().repeat_interleave(2, -1)[None, :, None].to(hidden.dtype)
        for index, layer in enumerate(self.local_transformer.h):
            self.check_abort()
            hidden = layer(hidden, position, cos, sin, (self.depth_kv[0, index], self.depth_kv[1, index]))
            self.check_abort()
        return self.local_transformer.ln_f(hidden)

    def audio_embedding(self, token, index):
        return self.audio_embeddings[index](token)

    def audio_head(self, hidden, index):
        return self.audio_lm_heads[index](hidden)[:, 0]

    def text_head(self, hidden):
        return self.local_text_lm_head(hidden)[:, 0]
