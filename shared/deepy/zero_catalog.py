# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Deepy Zero's tool catalog: the only built-in tools and parameters that Deepy Zero's own model sees.

Every built-in assistant tool also serves the MCP toolbox of Deepy Prime and external agents, which see all of them. Deepy Zero is
the lightweight assistant: its whole tool list is part of every prefill and of its context window. A tool or parameter added for
the toolbox therefore stays out of Zero until it is listed here on purpose, and wording written for agents goes to
agent_description, which Zero never reads. Plugin tools registered for Deepy Zero are explicit Zero tools and are not listed.
"""
from __future__ import annotations

from typing import Any

ZERO_TOOL_PARAMETERS: dict[str, tuple[str, ...]] = {
    "add_to_gallery": ("path", "paths"),
    "append_text": ("file_path", "text"),
    "create_color_frame": ("width", "height", "color", "boxes", "box_color"),
    "edit": ("file_path", "old_string", "new_string", "replace_all"),
    "edit_image": ("media_id", "prompt", "width", "height", "num_inference_steps", "extra_settings"),
    "extract_audio": ("media_id", "start_s", "end_s", "duration_s", "start_frame", "end_frame", "num_frames", "audio_track_no"),
    "extract_image": ("media_id", "frame_no", "time_s"),
    "extract_video": ("media_id", "start_s", "end_s", "duration_s", "start_frame", "end_frame", "num_frames"),
    "gen_image": ("prompt", "width", "height", "num_inference_steps", "extra_settings", "custom_file"),
    "gen_song": ("lyrics", "music_caption", "duration_seconds", "extra_settings", "loras", "custom_file"),
    "gen_speech_from_description": ("prompt", "voice_description", "duration_seconds", "extra_settings"),
    "gen_speech_from_sample": ("prompt", "media_id", "duration_seconds", "extra_settings"),
    "gen_video": ("prompt", "image_start", "image_end", "width", "height", "num_frames", "duration_seconds", "fps", "num_inference_steps", "extra_settings", "loras", "custom_file"),
    "gen_video_with_refs": ("prompt", "image_start", "image_end", "width", "height", "num_frames", "duration_seconds", "fps", "num_inference_steps", "extra_settings", "loras", "custom_file", "image_refs", "video_refs"),
    "gen_video_with_speech": ("prompt", "image_start", "audio_media_id", "width", "height", "num_frames", "duration_seconds", "fps", "num_inference_steps", "extra_settings", "loras"),
    "get_default_settings": ("tool_id",),
    "get_loras": ("tool_id", "name"),
    "get_media_details": ("media_id",),
    "get_selected_media": ("media_type",),
    "inspect_media": ("media_id", "question", "frame_no"),
    "inspect_video": ("media_id", "start_s", "end_s", "question", "mid_res_sampling", "min_gap_frames", "min_gap_s"),
    "list_files": ("path", "pattern", "media_type", "limit", "cursor", "summary_only"),
    "load_doc_section": ("doc_id", "section"),
    "merge_videos": ("media_ids", "transitions", "transition_s"),
    "mute_video": ("media_id",),
    "notify": ("message", "title"),
    "postprocessing": ("media_id", "process", "parameters"),
    "query_file": ("path",),
    "replace_audio": ("video_id", "audio_id"),
    "resize_crop": ("media_id", "width", "height", "crop_left", "crop_top", "crop_right", "crop_bottom", "crop_unit", "crop_anchor", "stretch_to_fit"),
    "resolve_media_reference": ("reference", "media_type"),
    "rg": ("arguments",),
    "search_doc": ("query", "doc_id"),
    "side_by_side": ("media_ids", "layout", "legends"),
    "transcribe_media": ("media_id", "timestamp_type", "audio_track_no"),
}


def check_zero_catalog(tools: dict[str, dict[str, Any]]) -> None:
    """Fail at import when an entry names a missing tool or parameter, or leaves out a parameter its tool requires."""
    for name, parameters in ZERO_TOOL_PARAMETERS.items():
        if name not in tools:
            raise RuntimeError(f"Deepy Zero catalog lists unknown tool '{name}'.")
        declared = tools[name]["parameters"]
        unknown = [parameter for parameter in parameters if parameter not in declared]
        missing = [parameter for parameter, meta in declared.items() if meta.get("required", True) and parameter not in parameters]
        if unknown or missing:
            raise RuntimeError(f"Deepy Zero catalog entry '{name}' does not match the tool: unknown {unknown}, required but missing {missing}.")
