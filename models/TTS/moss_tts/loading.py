import json
from pathlib import Path
from types import SimpleNamespace
import torch
from accelerate import init_empty_weights
from mmgp import offload
from .transformer import MossTTS
from .codec import Codec, adapt_attention
from .configuration_moss_audio_tokenizer import MossAudioTokenizerConfig


def load_transformer(path, features, local=False):
    config = json.loads(Path(__file__).with_name("moss_tts_local_v1_5.json" if local else "moss_tts_v1_5.json").read_text())
    config["language_config"] = SimpleNamespace(**config["language_config"])
    if local:
        config["qwen3_config"] = SimpleNamespace(**config["qwen3_config"])
        config["gpt2_config"] = SimpleNamespace(**config["gpt2_config"])
    config = SimpleNamespace(**config)
    with init_empty_weights(include_buffers=True):
        if local:
            from .local_transformer import MossLocalTTS
            model = MossLocalTTS(config, features)
        else:
            model = MossTTS(config, features)
    offload.load_model_data(model, str(path), writable_tensors=False, default_dtype=torch.bfloat16,
                           **({"fused_split_map": model.split_map()} if local else {}))
    return model.eval()


def load_codec(path, local=False):
    config = MossAudioTokenizerConfig(**json.loads(Path(__file__).with_name("codec_v2.json" if local else "codec.json").read_text()))
    with init_empty_weights(include_buffers=True):
        if local:
            from .codec_v2 import CodecV2
            codec = CodecV2(config)
        else:
            codec = Codec(config)
        splits = adapt_attention(codec)
    offload.load_model_data(codec, str(path), writable_tensors=False, default_dtype=None, fp32_dtype=None, fused_split_map=splits,
                           preprocess_sd=codec.checkpoint_keys if local else None)
    codec._convertWeightsFloatTo = None
    codec._model_dtype = torch.bfloat16
    return codec.eval()
