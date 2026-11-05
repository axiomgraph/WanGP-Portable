# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import torch

from shared.utils.hf import build_hf_url
from .texts import EVOKE_DEEPY_INFOS, EVOKE_DEEPY_PROMPT_INFOS, EVOKE_ENHANCER_INSTRUCTIONS, EVOKE_INFOS, EVOKE_PROMPT_INFOS

ARCHITECTURE = "evoke_turbo"
TEXT_ENCODER_FOLDER = "umt5-xxl"
CHUNK_FRAMES = 36
# False (default): the resolution is locked to 640x384, the only size EVOKE was trained at, and WanGP center-crops the start image to it.
# True: the start image keeps its shape (WanGP's canvas fit, e.g. 576x448 for a 4:3 image) and other resolutions can be chosen, to see
# how EVOKE does at sizes it was not trained at.
ANY_IMAGE_RATIO = False


class family_handler:
    @staticmethod
    def query_supported_types():
        return [ARCHITECTURE]

    @staticmethod
    def query_family_maps():
        return {}, {}

    @staticmethod
    def query_model_family():
        return "evoke"

    @staticmethod
    def query_family_infos():
        return {"evoke": (53, "EVOKE")}

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
            "vae_block_size": 64,  # the long history tier and the first pyramid stage need multiples of 64 pixels
            "resolutions": [["640x384 (5:3)", "640x384"]],
            "lock_resolutions": not ANY_IMAGE_RATIO,  # the resolution it was trained at (start images cropped to it)
            "frames_minimum": CHUNK_FRAMES,
            "frames_steps": CHUNK_FRAMES,
            "frames_selection_maximum": CHUNK_FRAMES * 40,
            "sliding_window": False,
            "image_prompt_types_allowed": "TS",  # a new world starts from an image; a continued world (custom guide) needs none
            "guidance_max_phases": 0,
            "no_negative_prompt": True,
            "inference_steps": False,  # three fixed pyramid forwards
            "tiny_vae_architecture": "t2v",  # Wan2.1 latents
            "custom_guide": {"id": "custom_guide", "name": "World", "label": "Continue a World (.wgpworld of the Worlds gallery)", "type": "file", "default": None,
                             "required": False, "file_types": [".wgpworld"]},
            "profiles_dir": [base_model_type],
            "text_encoder_folder": TEXT_ENCODER_FOLDER,
            "text_encoder_URLs": [
                build_hf_url("DeepBeepMeep/Wan2.1", TEXT_ENCODER_FOLDER, "models_t5_umt5-xxl-enc-bf16.safetensors"),
                build_hf_url("DeepBeepMeep/Wan2.1", TEXT_ENCODER_FOLDER, "models_t5_umt5-xxl-enc-quanto_int8.safetensors"),
            ],
            "infos": EVOKE_INFOS,
            "prompt_infos": EVOKE_PROMPT_INFOS,
            "deepy_infos": EVOKE_DEEPY_INFOS,
            "deepy_prompt_infos": EVOKE_DEEPY_PROMPT_INFOS,
            "prompt_enhancer_def": {"selection": ["TI"], "labels": {"TI": "An EVOKE World from the Start Image"}, "default": ""},
            "video_prompt_enhancer_instructions": EVOKE_ENHANCER_INSTRUCTIONS,
        }

    @staticmethod
    def query_world_explorer(base_model_type, manifest, memory):  # the memory of an EVOKE world is RAM only, whatever the profile
        from .exploration import explorer_menus
        return explorer_menus(manifest)

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        return [
            {"repoId": "DeepBeepMeep/Wan2.1", "sourceFolderList": [TEXT_ENCODER_FOLDER, ""],
             "fileList": [["special_tokens_map.json", "spiece.model", "tokenizer.json", "tokenizer_config.json"], ["Wan2.1_VAE.safetensors"]]},
            {"repoId": "DeepBeepMeep/Worlds", "sourceFolderList": ["evoke"], "fileList": [["vigeo1_1.safetensors"]]},
        ]

    @staticmethod
    def load_model(model_filename, model_type, base_model_type, model_def, quantizeTransformer=False, text_encoder_quantization=None, dtype=torch.bfloat16,
                   VAE_dtype=torch.float32, mixed_precision_transformer=False, save_quantized=False, submodel_no_list=None, text_encoder_filename=None, profile=-1, **kwargs):
        from .evoke_main import model_factory

        model = model_factory(model_filename, model_type, base_model_type, text_encoder_filename, save_quantized=save_quantized, VAE_dtype=VAE_dtype)
        pipe = {"transformer": model.transformer, "text_encoder": model.text_encoder.model, "vae": model.vae.model, "vigeo": model.vigeo}
        # every chunk runs the DiT, the VAE twice (memory view encode, decode) and ViGeo (memory write): the small VAE stays beside the
        # DiT; ViGeo (2.3 GB) is swapped in from pinned RAM rather than kept resident (VRAM for 8 GB GPUs)
        offload_kwargs = {"pipe": pipe, "coTenantsMap": {"transformer": ["vae"], "vae": ["transformer", "text_encoder"], "text_encoder": ["vae"]}}
        if profile in (3, 4):
            offload_kwargs["pinnedMemory"] = ["transformer", "vigeo"]
        return model, offload_kwargs

    @staticmethod
    def get_rgb_factors(base_model_type):
        from shared.RGB_factors import get_rgb_factors
        return get_rgb_factors("wan", "t2v")  # Wan2.1 latents

    @staticmethod
    def validate_generative_settings(base_model_type, model_def, inputs):
        if inputs["custom_guide"] is not None:  # a world to continue (or explore): its memory replaces the start image
            return None
        if "S" not in inputs["image_prompt_type"]:
            return "A new EVOKE world starts from a Start Image (Start with Image); without one, choose a world of the Worlds gallery to continue"
        from shared.utils.world_instructions import split_instructions
        if not split_instructions(inputs["prompt"])[0]:
            return "A new EVOKE world needs a description of the place before its exploration instructions"
        return None

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        ui_defaults.update({
            "num_inference_steps": 3,
            "guidance_scale": 1.0,
            "video_length": CHUNK_FRAMES * 4,
            "resolution": "640x384",
            "image_prompt_type": "S",
            "multi_prompts_gen_type": "FG",  # the whole prompt is one world description, whatever its lines
        })
