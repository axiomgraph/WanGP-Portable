# Adapted from OpenMOSS/MOSS-Audio-Tokenizer-v2 (Apache-2.0); see LICENSE and NOTICE.
"""Stereo v2 codec using the family's shared causal transformer and LFQ blocks."""
import torch
from torch import nn
from .codec import Codec
from .modeling_moss_audio_tokenizer import (
    MossAudioTokenizerPatchedPretransform, MossAudioTokenizerProjectedTransformer,
    MossAudioTokenizerResidualLFQ, MossAudioTokenizerMultiheadAttention,
)


class QuantizerConv(nn.Module):
    """BF16 stored parameters with the released quantizer's FP32 computation."""
    def __init__(self, convolution):
        super().__init__()
        self.parametrizations = convolution.parametrizations
        self.bias = convolution.bias

    def forward(self, x):
        weights = self.parametrizations.weight
        weight = torch._weight_norm(weights.original1.float(), weights.original0.float(), 0)
        return torch.nn.functional.conv1d(x.float(), weight, self.bias.float())


class CodecV2(Codec):
    def __init__(self, config):
        # Reuse inference blocks; v2 interleaves stereo and has a context per scale.
        nn.Module.__init__(self)
        self.config = config
        self.sampling_rate = config.sampling_rate
        self.downsample_rate = config.downsample_rate
        self.number_channels = config.number_channels
        frame_rate = float(config.sampling_rate * config.number_channels)
        for stage, down in (("encoder", True), ("decoder", False)):
            blocks = nn.ModuleList()
            for raw in getattr(config, stage + "_kwargs"):
                spec = dict(raw)
                if spec["module_type"] == "PatchedPretransform":
                    block = MossAudioTokenizerPatchedPretransform(**spec, is_downsample=down)
                else:
                    context = round(frame_rate * spec.pop("context_duration", 10.0))
                    block = MossAudioTokenizerProjectedTransformer(**spec, context=context)
                    block.input_proj = nn.Linear(spec["input_dimension"], spec["d_model"], bias=False)
                    block.output_proj = nn.Linear(spec["d_model"], spec["output_dimension"], bias=False)
                    for module in block.modules():
                        if isinstance(module, MossAudioTokenizerMultiheadAttention):
                            module.cache_capacity = context + round(frame_rate * 8)
                blocks.append(block)
                frame_rate = frame_rate / block.downsample_ratio if down else frame_rate * block.downsample_ratio
            setattr(self, stage, blocks)
        self.quantizer = MossAudioTokenizerResidualLFQ(**config.quantizer_kwargs)
        for parent in list(self.quantizer.modules()):
            for name, child in list(parent.named_children()):
                if isinstance(child, nn.Conv1d):
                    setattr(parent, name, QuantizerConv(child))

    @staticmethod
    def checkpoint_keys(state_dict):
        return {key.replace(".self_attn.in_proj.", ".self_attn.in_projs.0.")
                   .replace(".self_attn.out_proj.", ".self_attn.out_projs.0.")
                   .replace(".ffn.0.", ".linear1.").replace(".ffn.2.", ".linear2."): value
                for key, value in state_dict.items()}

    def encode_tile(self, audio):
        length = audio.shape[-1]
        audio = torch.nn.functional.pad(audio, (0, -length % self.downsample_rate))
        hidden = audio.transpose(1, 2).contiguous().view(audio.shape[0], 1, -1)
        lengths = torch.full((audio.shape[0],), length * self.number_channels, device=audio.device, dtype=torch.long)
        with torch.autocast(device_type=audio.device.type, dtype=torch.bfloat16):
            for block in self.encoder:
                hidden, lengths = block(hidden, lengths)
        with torch.autocast(device_type=audio.device.type, enabled=False):
            _, codes, _ = self.quantizer(hidden.float(), lengths, 12)
        return codes

    def decode_tile(self, codes):
        lengths = torch.full((codes.shape[1],), codes.shape[-1], device=codes.device, dtype=torch.long)
        with torch.autocast(device_type=codes.device.type, enabled=False):
            audio = self.quantizer.decode_codes(codes).float()
        with torch.autocast(device_type=codes.device.type, dtype=torch.bfloat16):
            for block in self.decoder:
                audio, lengths = block(audio, lengths)
        return audio.squeeze(1).contiguous().view(audio.shape[0], -1, self.number_channels).transpose(1, 2).contiguous().float()
