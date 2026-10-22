# Copyright 2026 The MiniMax Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""WanGP-local port of Diffusers' MiniMax Music 3 flow transformer."""

import math
from functools import lru_cache

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.embeddings import TimestepEmbedding
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from mmgp import cuda_graphs

from shared import attention_kit
from shared.kernels import ops


class MiniMaxMusic3FourierEmbedding(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(embedding_dim // 2, 1))

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        angles = 2.0 * math.pi * timestep.unsqueeze(-1) @ self.weight.T
        return torch.cat((angles.cos(), angles.sin()), dim=-1)


class MiniMaxMusic3RotaryEmbedding(nn.Module):
    def __init__(self, rotary_dim: int, theta: float = 10000.0):
        super().__init__()
        self.rotary_dim = rotary_dim
        self.theta = theta

    @lru_cache(maxsize=32)
    def forward(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.rotary_dim, 2, device=device).float() / self.rotary_dim))
        freqs = torch.outer(torch.arange(seq_len, device=device, dtype=torch.float32), inv_freq)
        freqs = torch.cat((freqs, freqs), dim=-1)
        return freqs.cos().contiguous(), freqs.sin().contiguous()


class MiniMaxMusic3Attention(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.inner_dim = heads * head_dim
        self.to_q = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_k = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_v = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(self.inner_dim, dim, bias=False), nn.Dropout(0.0)])

    def forward(self, x_list: list[torch.Tensor], rotary_emb: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Attention of x (handed off in x_list), (batch, tokens, heads, head_dim), before the output projection. RoPE rotates the
        first rotary_dim channels of each head (split-half pairs); the other channels pass through."""
        cos, sin = rotary_emb
        rotary_dim = 2 * cos.shape[-1]
        table = torch.stack((cos, sin), dim=-1).unsqueeze(0)  # (1, tokens, 1, pairs, 2) for the fused quantization

        def rope_(query, key, group):
            for x in (query, key):
                if x is not None:
                    ops.paired_rope_(x[..., :rotary_dim], cos, sin)

        return attention_kit.qkv_attention(x_list, self.to_q, self.to_k, self.to_v, self.heads, self.head_dim, rope_,
                                           fused_quant=lambda query, key, group, core, settings: ops.split_half_norm_rope_quant_(query, key, table, None, None, core, settings))


class MiniMaxMusic3TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int, ff_inner_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MiniMaxMusic3Attention(dim, heads, head_dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff_in = nn.Linear(dim, ff_inner_dim * 2)
        self.ff_out = nn.Linear(ff_inner_dim, dim)

    def forward(self, hidden_states: torch.Tensor, rotary_emb: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Updates hidden_states in place: the residual adds are the output projections' epilogues where the weight format has them."""
        attention = [self.attn([self.norm1(hidden_states)], rotary_emb).flatten(2)]
        ops.linear_residual_(self.attn.to_out[0], attention, hidden_states)
        # SwiGLU of the rows value | gate in one fc1 GEMM (ops.mlp takes gate | value rows, or a separate up projection that costs more
        # here); the activation in a new contiguous tensor: a 16-bit ff_out reading a strided view picks a less precise GEMM
        value, gate = self.ff_in(self.norm2(hidden_states)).chunk(2, dim=-1)
        activation = [ops.swiglu([gate, value], packed=False, in_place=False)]
        del value, gate
        return ops.linear_residual_(self.ff_out, activation, hidden_states)


class MiniMaxMusic3Transformer1DModel(ModelMixin, ConfigMixin):
    """Flow-matching transformer that turns Music 3 frame conditioning into audio latents."""

    _supports_gradient_checkpointing = True
    _no_split_modules = ["MiniMaxMusic3TransformerBlock"]
    _repeated_blocks = ["MiniMaxMusic3TransformerBlock"]
    _skip_layerwise_casting_patterns = ["time_proj", "norm"]

    @register_to_config
    def __init__(
        self,
        in_channels: int = 128,
        condition_dim: int = 2048,
        num_layers: int = 36,
        num_attention_heads: int = 32,
        attention_head_dim: int = 64,
        ff_inner_dim: int = 8192,
        rotary_dim: int = 32,
        fourier_embedding_dim: int = 256,
    ):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim
        concat_channels = 2 * in_channels + condition_dim
        self.time_proj = MiniMaxMusic3FourierEmbedding(fourier_embedding_dim)
        self.time_embed = TimestepEmbedding(fourier_embedding_dim, inner_dim)
        self.preprocess_conv = nn.Conv1d(concat_channels, concat_channels, 1, bias=False)
        self.proj_in = nn.Linear(concat_channels, inner_dim, bias=False)
        self.rotary_emb = MiniMaxMusic3RotaryEmbedding(rotary_dim)
        self.transformer_blocks = nn.ModuleList([
            MiniMaxMusic3TransformerBlock(inner_dim, num_attention_heads, attention_head_dim, ff_inner_dim)
            for _ in range(num_layers)
        ])
        self.proj_out = nn.Linear(inner_dim, in_channels, bias=False)
        self.postprocess_conv = nn.Conv1d(in_channels, in_channels, 1, bias=False)
        self.gradient_checkpointing = False
        self._interrupt_check = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        return_dict: bool = True,
    ) -> tuple[torch.Tensor] | Transformer2DModelOutput:
        residual = torch.cat((hidden_states, torch.zeros_like(hidden_states), encoder_hidden_states.transpose(1, 2)), dim=1)
        hidden_states = self.preprocess_conv(residual) + residual
        del residual
        hidden_states = hidden_states.transpose(1, 2)
        temb = self.time_embed(self.time_proj(timestep))
        hidden_states = self.proj_in(hidden_states)
        hidden_states = torch.cat((temb.unsqueeze(1), hidden_states), dim=1)
        # half of each split-half table (its two halves are equal), in the activation dtype, once per call
        rotary_emb = tuple(table[:, None, :table.shape[-1] // 2].to(hidden_states.dtype) for table in self.rotary_emb(hidden_states.shape[1], hidden_states.device))
        with cuda_graphs.scope():  # the blocks replay CUDA graphs in the CUDA Graph mode (model def cuda_graphs)
            for block in self.transformer_blocks:
                hidden_states = block(hidden_states, rotary_emb)
                if self._interrupt_check is not None and self._interrupt_check():
                    raise InterruptedError("MiniMax Music 3 generation interrupted")
        hidden_states = self.proj_out(hidden_states[:, 1:]).transpose(1, 2)
        hidden_states = self.postprocess_conv(hidden_states) + hidden_states
        if not return_dict:
            return (hidden_states,)
        return Transformer2DModelOutput(sample=hidden_states)
