# Adapted from OpenMOSS/MOSS-Audio-Tokenizer (Apache-2.0); see LICENSE and NOTICE.
import math
from contextlib import ExitStack
import torch
from torch import nn
from mmgp import runtime
from shared import attention_kit
from shared.attention import pay_attention
from .modeling_moss_audio_tokenizer import MossAudioTokenizerModel, MossAudioTokenizerMultiheadAttention, StreamingModule


def adapt_attention(codec):
    """Split the v1 codec's fused projections through MMGP's checkpoint map."""
    splits = {}
    for name, module in codec.named_modules():
        if isinstance(module, MossAudioTokenizerMultiheadAttention):
            module.__class__ = CodecAttention
            width = module.embed_dim
            if module.mult != 1:
                raise ValueError("The MOSS v1 codec requires one set of attention weights")
            del module.in_projs
            module.q_proj = nn.Linear(width, width, bias=False)
            module.k_proj = nn.Linear(width, width, bias=False)
            module.v_proj = nn.Linear(width, width, bias=False)
            splits[name + ".in_projs.0"] = {"mapped_modules": [name + "." + key for key in ("q_proj", "k_proj", "v_proj")], "split_sizes": [width] * 3}
    return splits


class CodecAttention(MossAudioTokenizerMultiheadAttention):
    def _init_streaming_state(self, batch_size):
        from .modeling_moss_audio_tokenizer import MHAState, RingKVCache
        device = runtime.execution_device(self)
        # Retain the previous window while writing the new tile. A context-sized
        # ring would overwrite keys still needed by the tile's earliest queries.
        cache = RingKVCache(batch_size, self.num_heads, self.embed_dim // self.num_heads, getattr(self, "cache_capacity", 2 * self.context), device=device, dtype=self.q_proj.weight.dtype)
        return MHAState(batch_size, device, cache, offset=torch.zeros(batch_size, device=device, dtype=torch.long), offset_cpu=0)

    def forward(self, query, key, value):
        state = self._streaming_state
        batch, length = query.shape[:2]
        offset = torch.zeros(batch, device=query.device, dtype=torch.long) if state is None else state.offset
        width = self.embed_dim // self.num_heads
        frequencies = torch.exp(torch.arange(width // 2, device=query.device, dtype=torch.float32) * (-math.log(self.rope.max_period) * 2 / width))
        times = offset.float()[:, None] + torch.arange(length, device=query.device, dtype=torch.float32)
        angles = times[:, :, None, None] * frequencies
        cos, sin = angles.cos(), angles.sin()
        def normalize(q, k, group):
            def rotate(part, first, last):
                even, odd = part[..., 0::2].float(), part[..., 1::2].float()
                c, s = cos[:, first:last], sin[:, first:last]
                return torch.stack((even * c - odd * s, even * s + odd * c), dim=-1).flatten(-2).to(part.dtype)
            for tensor in (q, k):
                if tensor is not None:
                    attention_kit.rows_(tensor, rotate)
        def attend(qkv, group):
            q, k, v = qkv
            qkv.clear()
            k, v, pos_k = self._complete_kv(k.transpose(1, 2), v.transpose(1, 2))
            delta = offset[:, None, None] + torch.arange(length, device=q.device)[None, :, None] - pos_k[:, None]
            mask = (pos_k[:, None] >= 0) & (delta >= 0) & (delta < self.context)
            return pay_attention([q, k.transpose(1, 2), v.transpose(1, 2)], attention_mask=mask[:, :, None], force_attention="sdpa")
        x = attention_kit.qkv_attention([query], self.q_proj, self.k_proj, self.v_proj, self.num_heads, width, normalize,
                                         split_heads=False, attention_fn=attend)
        x = self.out_projs[0](x.flatten(2))
        if state is not None:
            state.offset[:] = torch.where(state.exec_mask, state.offset + length, state.offset)
            state.offset_cpu += length
        return x


class Codec(MossAudioTokenizerModel):
    _offload_hooks = ["encode_tile", "decode_tile"]
    _offload_separate_blocks = ("encoder", "decoder")

    def encode_tile(self, audio):
        with torch.autocast(device_type=audio.device.type, dtype=torch.bfloat16):
            return self._encode_frame(audio).audio_codes

    def decode_tile(self, codes):
        with torch.autocast(device_type=codes.device.type, dtype=torch.bfloat16):
            return self._decode_frame(codes).audio

    def stream(self, encode=False):
        stack = ExitStack()
        for module in self.encoder if encode else self.decoder:
            if isinstance(module, StreamingModule):
                stack.enter_context(module.streaming(batch_size=1))
        return stack
