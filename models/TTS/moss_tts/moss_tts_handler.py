from pathlib import Path
import torch
from shared.utils import files_locator as fl
from .help import INFOS, PROMPT_INFOS, DEEPY_INFOS, DEEPY_PROMPT_INFOS, ENHANCER, DIALOGUE_ENHANCER, DEFAULT_PROMPT

ARCHITECTURE = "moss_tts_v1_5"
LOCAL_ARCHITECTURE = "moss_tts_local_v1_5"
CODEC = "MOSS_audio_tokenizer_bf16.safetensors"
LOCAL_CODEC = "MOSS_audio_tokenizer_v2_bf16.safetensors"
TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json", "merges.txt", "vocab.json", "chat_template.jinja"]


class family_handler:
    @staticmethod
    def query_supported_types():
        return [ARCHITECTURE, LOCAL_ARCHITECTURE]

    @staticmethod
    def query_family_maps():
        return {}, {}

    @staticmethod
    def query_model_family():
        return "tts"

    @staticmethod
    def query_family_infos():
        return {"tts": (2200, "TTS")}

    @staticmethod
    def get_lora_dir(base_model_type):
        return base_model_type

    @staticmethod
    def query_model_def(base_model_type, model_def):
        local = base_model_type == LOCAL_ARCHITECTURE
        infos = INFOS.replace("MOSS-TTS v1.5", "MOSS-TTS Local Transformer v1.5") if local else INFOS
        deepy_infos = DEEPY_INFOS.replace("v1.5 8B", "Local Transformer v1.5 4B").replace("24000 Hz mono", "48000 Hz stereo") if local else DEEPY_INFOS
        return {"group": "tts", "audio_only": True, "image_outputs": False, "sliding_window": False,
            "device_explicit": True, "dtype": "bf16", "auto_quantize": False,
            "no_negative_prompt": True, "inference_steps": False, "guidance_max_phases": 0,
            "temperature": True, "temperature_slider": {"end": 2.0}, "top_k_slider": True, "top_p_slider": True,
            "compile": False, "cuda_graphs": True, "supports_early_stop": True,
            "profiles_dir": [base_model_type], "lm_decoder": True,
            "image_prompt_types_allowed": "", "any_audio_prompt": True, "audio_prompt_choices": True,
            "audio_prompt_type_sources": {"selection": ["", "A", "AB", "ABD"], "default": "", "letters_filter": "ABD", "labels": {"": "New Voice", "A": "Voice Reference", "AB": "Voice References for Speakers 1 and 2", "ABD": "Voice References for Speakers 1, 2 and 3"}},
            "audio_guide_label": "Speaker 1 Voice Reference", "audio_guide2_label": "Speaker 2 Voice Reference", "audio_guide3_label": "Speaker 3 Voice Reference",
            "alt_prompt": {"label": "Spoken Language", "name": "Language", "lines": 1, "placeholder": "English, French, Chinese... Leave empty for unspecified language."},
            "model_modes": {"label": "Speech Timing", "choices": [("Natural Completion", 0), ("Target Duration", 1)], "default": 0},
            "duration_slider": {"label": "Maximum Speech Duration (seconds)", "name": "Maximum Duration", "min": 1, "max": 600, "increment": 1, "default": 30},
            "prompt_class": "Speech", "preserve_empty_prompt_lines": True, "pause_between_sentences": True,
            "prompt_enhancer_def": {"selection": ["T", "T1"], "labels": {"T": "Write a Speech", "T1": "Write a Dialogue"}, "default": ""},
            "prompt_enhancer_button_label": "Write", "text_prompt_enhancer_instructions": ENHANCER, "text_prompt_enhancer_max_tokens": 2048,
            "text_prompt_enhancer_instructions1": DIALOGUE_ENHANCER, "text_prompt_enhancer_max_tokens1": 2048,
            "infos": infos, "prompt_infos": PROMPT_INFOS, "deepy_infos": deepy_infos, "deepy_prompt_infos": DEEPY_PROMPT_INFOS}

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        local = base_model_type == LOCAL_ARCHITECTURE
        return [{"repoId": "DeepBeepMeep/TTS", "sourceFolderList": ["moss_tts_local" if local else "moss_tts"], "fileList": [[LOCAL_CODEC if local else CODEC, *TOKENIZER_FILES]]}]

    @staticmethod
    def load_model(model_filename, model_type, base_model_type, model_def, save_quantized=False, lm_features=None, **kwargs):
        from .pipeline import MossTTSPipeline
        model_filename, = model_filename
        local = base_model_type == LOCAL_ARCHITECTURE
        codec, folder = (LOCAL_CODEC, "moss_tts_local") if local else (CODEC, "moss_tts")
        files = {name: fl.locate_file(str(Path(folder) / name)) for name in (codec, *TOKENIZER_FILES)}
        pipeline = MossTTSPipeline(model_filename, files[codec], str(Path(files["tokenizer.json"]).parent), lm_features, local=local)
        if save_quantized:
            from wgp import save_quantized_model
            save_quantized_model(pipeline.transformer, model_type, model_filename, torch.bfloat16,
                                 str(Path(__file__).with_name(base_model_type + ".json")), convrot_prefixes=("",))
        return pipeline, {"pipe": {"transformer": pipeline.transformer, "vae": pipeline.vae}, "cudaGraphs": lm_features.graphs}

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        ui_defaults.update(prompt=DEFAULT_PROMPT, alt_prompt="English", audio_prompt_type="", model_mode=0, duration_seconds=30,
            temperature=1.7, top_k=25, top_p=0.8, prompt_enhancer="", pause_seconds=0.2, guidance_scale=1.0, num_inference_steps=1,
            video_length=0, repeat_generation=1, multi_prompts_gen_type="FG")

    @staticmethod
    def fix_settings(base_model_type, settings_version, model_def, ui_defaults):
        pass
