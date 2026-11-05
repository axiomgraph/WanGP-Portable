# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import torch

from shared.utils.hf import build_hf_url
from .texts import HY_NOMAS_DEEPY_INFOS, HY_NOMAS_DEEPY_PROMPT_INFOS, HY_NOMAS_ENHANCER_INSTRUCTIONS, HY_NOMAS_INFOS, HY_NOMAS_PROMPT_INFOS

ARCHITECTURE = "hy_nomas"
TEXT_ENCODER_FOLDER = "Qwen2.5-VL-7B-Instruct"


class family_handler:
    @staticmethod
    def query_supported_types():
        return [ARCHITECTURE]

    @staticmethod
    def query_family_maps():
        return {}, {}

    @staticmethod
    def query_model_family():
        return "hy_nomas"

    @staticmethod
    def query_family_infos():
        return {"hy_nomas": (52, "Hy-Nomas")}

    @staticmethod
    def get_lora_dir(base_model_type):
        return base_model_type

    @staticmethod
    def query_model_def(base_model_type, model_def):
        return {
            "device_explicit": True,
            "world_outputs": True,  # explorable worlds (.wgpworld in the Worlds gallery) rendered as videos
            "dtype": "bf16",
            "accelerated": "native",
            "fps": 24,
            "vae_block_size": 32,  # DCAE-V: 32 pixels per latent
            "frames_minimum": 5,
            "frames_steps": 4,
            "frames_selection_maximum": 1441,
            "sliding_window": False,
            "image_prompt_types_allowed": "",
            "guidance_max_phases": 0,
            "no_negative_prompt": True,
            "skip_prompt_template": True,  # world descriptions are JSON: no {variable} macros
            "inference_steps": True,
            # a world of the Worlds gallery: generation continues it with the prompt's segments (its exploration also runs on it)
            "custom_guide": {"id": "custom_guide", "name": "World", "label": "Continue a World (.wgpworld of the Worlds gallery)", "type": "file", "default": None,
                             "required": False, "file_types": [".wgpworld"]},
            "profiles_dir": [base_model_type],
            "text_encoder_folder": TEXT_ENCODER_FOLDER,
            "text_encoder_URLs": [
                build_hf_url("DeepBeepMeep/Qwen_image", TEXT_ENCODER_FOLDER, "Qwen2.5-VL-7B-Instruct_bf16.safetensors"),
                build_hf_url("DeepBeepMeep/Qwen_image", TEXT_ENCODER_FOLDER, "Qwen2.5-VL-7B-Instruct_quanto_bf16_int8.safetensors"),
            ],
            "infos": HY_NOMAS_INFOS,
            "prompt_infos": HY_NOMAS_PROMPT_INFOS,
            "deepy_infos": HY_NOMAS_DEEPY_INFOS,
            "deepy_prompt_infos": HY_NOMAS_DEEPY_PROMPT_INFOS,
            "prompt_enhancer_button_label": "Write",
            "prompt_enhancer_def": {"selection": ["T"], "labels": {"T": "A Hy-Nomas World from Text"}, "default": ""},
            "text_prompt_enhancer_instructions": HY_NOMAS_ENHANCER_INSTRUCTIONS,
            "text_prompt_enhancer_max_tokens": 4096,
        }

    @staticmethod
    def query_world_explorer(base_model_type, manifest, memory):
        from .exploration import explorer_menus
        return explorer_menus(manifest, memory)

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        return [
            {
                "repoId": "DeepBeepMeep/Qwen_image",
                "sourceFolderList": [TEXT_ENCODER_FOLDER],
                "fileList": [["merges.txt", "tokenizer_config.json", "config.json", "vocab.json", "chat_template.json"]],
            },
            {
                "repoId": "DeepBeepMeep/Worlds",
                "sourceFolderList": ["hy_nomas"],
                "fileList": [["pixelumm_decoder_fp8.safetensors"]],
            },
        ]

    @staticmethod
    def load_model(model_filename, model_type, base_model_type, model_def, quantizeTransformer=False, text_encoder_quantization=None, dtype=torch.bfloat16,
                   VAE_dtype=torch.float32, mixed_precision_transformer=False, save_quantized=False, submodel_no_list=None, text_encoder_filename=None, profile=-1, **kwargs):
        from mmgp import quant_router
        from .hy_nomas_main import model_factory

        quant_router.register_handler("shared.qtypes.block_scaled")  # the checkpoints' own block-scaled INT8 (DiT) and FP8 (decoder) weights
        model = model_factory(model_filename, model_type, base_model_type, text_encoder_filename, save_quantized=save_quantized)
        pipe = {"transformer": model.transformer, "text_encoder": model.text_encoder.model, "decoder": model.decoder}
        # exploration alternates the DiT and the decoder at every move: they share the VRAM. The decoder, a 36-layer transformer run at every
        # move, is handled as a second transformer (as wgp.py does for transformer2): the transformer's budget, and pinned with it where the
        # profile pins the transformer alone
        offload_kwargs = {"pipe": pipe, "coTenantsMap": {"transformer": ["decoder"], "decoder": ["transformer"]}, "budgets": {"decoder": 100}}
        if profile in (3, 4):
            offload_kwargs["pinnedMemory"] = ["transformer", "decoder"]
        return model, offload_kwargs

    @staticmethod
    def get_rgb_factors(base_model_type):
        from shared.RGB_factors import get_rgb_factors
        return get_rgb_factors("hy_nomas")  # DCAE-V latents

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        ui_defaults.update({
            "num_inference_steps": 8,
            "guidance_scale": 1.0,
            "video_length": 97,
            "resolution": "832x480",
            "image_prompt_type": "",
            "multi_prompts_gen_type": "FG",  # the whole prompt is one world description, whatever its lines
        })
