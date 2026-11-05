# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Deepy Zero's tool catalog: the only built-in tools and parameters that Deepy Zero's own model sees.

Every built-in assistant tool also serves the MCP toolbox of Deepy Prime and external agents, which see all of them. Deepy Zero is
the lightweight assistant: its whole tool list is part of every prefill and of its context window. A tool or parameter added for
the toolbox therefore stays out of Zero until it is listed here on purpose, and wording written for agents goes to
agent_description, which Zero never reads. Plugin tools registered for Deepy Zero are explicit Zero tools and are not listed.

Zero discovers most of its tools progressively, as Prime does: only ZERO_DIRECT tools are in its prompt, with the short texts of
ZERO_DIRECT_TEXT; the other catalog tools sit behind one `toolbox` tool, which lists them (ZERO_TOOLBOX_SUMMARIES), describes one
on demand (its full schema plus ZERO_TOOLBOX_NOTES) and runs them. Small models (Qwen3.5 4B and 9B) keep the common path direct.
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


# The tools Zero's model sees directly; every other catalog tool is reached through `toolbox`.
ZERO_DIRECT = ("get_selected_media", "resolve_media_reference", "inspect_media", "gen_image", "edit_image", "gen_video", "search_doc")

# Short texts of the direct tools: {tool: (description, {parameter: description})}; parameters not named keep their own text.
ZERO_DIRECT_TEXT: dict[str, tuple[str, dict[str, str]]] = {
    "get_selected_media": ("The media selected in the Gallery: only when the user says this, selected or current. For last, latest or previous media use resolve_media_reference.", {
        "media_type": "image, video, audio, 3d or all."}),
    "resolve_media_reference": ("The media_id of the last, latest or previous media, or of one the user describes (not for selected media).", {
        "reference": "What the user said, e.g. last, previous, the cat picture.",
        "media_type": "image, video, audio, 3d or all."}),
    "inspect_media": ("Look at an image (or one frame of a video) and answer a question about it.", {
        "media_id": "The image or video media_id.",
        "question": "What to look for.",
        "frame_no": "Frame of a video (default the first)."}),
    "gen_image": ("Generate an image from a prompt with the configured image template.", {
        "prompt": "Detailed description of the image.",
        "width": "Width in pixels, only when the user asks for a size.",
        "height": "Height in pixels, only when the user asks for a size.",
        "num_inference_steps": "Steps, only when the user asks.",
        "extra_settings": "Other template settings by their exact names from toolbox get_default_settings.",
        "custom_file": "Extra input file, only when get_default_settings reports custom_file."}),
    "edit_image": ("Change a still image and get a still image back (add, remove, restyle). Never for a video: to animate a picture use gen_video with image_start.", {
        "media_id": "The image to edit.",
        "prompt": "The edit to make, described precisely.",
        "width": "Width in pixels, only when the user asks for a size.",
        "height": "Height in pixels, only when the user asks for a size.",
        "num_inference_steps": "Steps, only when the user asks.",
        "extra_settings": "Other template settings by their exact names from toolbox get_default_settings."}),
    "gen_video": ("Make a video: from a prompt alone, or animating a picture (image_start = its media_id), optionally ending on another picture (image_end).", {
        "prompt": "Detailed description of the shot: subject, action, camera, sound or speech when the template makes sound.",
        "image_start": "media_id of the first frame image.",
        "image_end": "media_id of the last frame image.",
        "width": "Width in pixels, only when the user asks for a size.",
        "height": "Height in pixels, only when the user asks for a size.",
        "num_frames": "Frame count; prefer duration_seconds.",
        "duration_seconds": "Length in seconds when the user gives one.",
        "fps": "Frames per second (15 to 60), only when asked.",
        "num_inference_steps": "Steps, only when the user asks.",
        "extra_settings": "Other template settings by their exact names from toolbox get_default_settings.",
        "loras": "Only when the user asks for LoRAs: [{\"name\": a file name returned by toolbox get_loras, \"multiplier\": 1.0}].",
        "custom_file": "Extra input file, only when get_default_settings reports custom_file."}),
    "search_doc": ("Search WanGP's documentation, only to answer a question about how WanGP works (never to carry out a request). The LoRAs installed for a tool are listed by toolbox get_loras, not by the docs.", {
        "query": "Words to search for.",
        "doc_id": "Optional: getting_started, overview, processing, prompts, loras or finetunes."}),
}

# One line per toolbox action, shown when Zero lists the toolbox.
ZERO_TOOLBOX_SUMMARIES = {
    "gen_video_with_refs": "Video keeping the look of reference images (a character, a product) or following reference videos.",
    "gen_video_with_speech": "Talking or singing video driven by an existing speech or song audio clip.",
    "gen_speech_from_description": "Speech audio from text with a described voice.",
    "gen_speech_from_sample": "Speech audio from text in the voice of a sample clip.",
    "gen_song": "Song or music from lyrics and a music caption.",
    "get_default_settings": "Settings, inputs and extra settings of the template behind a generation tool.",
    "get_loras": "LoRAs available for a generation tool.",
    "get_media_details": "Size, duration, fps, frame count and audio tracks of a media item.",
    "inspect_video": "Answer a question about a time range of a video (compares sampled frames).",
    "extract_image": "Save one frame of a video as an image.",
    "extract_video": "Cut a clip from a video.",
    "extract_audio": "Take or cut the audio of a video or audio item.",
    "transcribe_media": "Transcript of the speech in an audio or video item.",
    "merge_videos": "Join videos one after another, with optional transitions.",
    "mute_video": "Remove the sound of a video.",
    "replace_audio": "Replace the soundtrack of a video.",
    "resize_crop": "Resize or crop an image or video.",
    "side_by_side": "Put images or videos next to each other in one picture or video.",
    "create_color_frame": "Solid colour image of an exact size (black frame, colour card).",
    "postprocessing": "Upscale, refine, interpolate or clean media (call with media_id alone to list the operations).",
    "notify": "Send the user a notification.",
    "add_to_gallery": "Add existing files to the Gallery.",
    "list_files": "List authorized folders and files.",
    "query_file": "Metadata or text of one file.",
    "rg": "Search file names or text in authorized folders.",
    "edit": "Replace text in a text file.",
    "append_text": "Append text to a text file (created if missing).",
    "load_doc_section": "Load one section of WanGP's documentation.",
}

# Guidance shown with one toolbox action's description (moved out of Zero's system prompt).
ZERO_TOOLBOX_NOTES = {
    "gen_video_with_refs": "For a subject, character or product that must keep its look, or a motion to follow: image_refs=[media_id] (video_refs when supported). Read get_default_settings once for the template's input guidance. Use gen_video when there are only start or end images.",
    "gen_video_with_speech": "Use when a speech or song clip already exists; for a talking video from text only, gen_video can speak the words when its template makes sound.",
    "gen_speech_from_description": "For standalone speech or when a separate speech clip is needed first.",
    "gen_speech_from_sample": "For standalone speech in the voice of a sample; if the voice is in a video, extract_audio first.",
    "gen_song": "lyrics in sections ([verse], [chorus]) and a music_caption with genre, instruments, vocals, mood and tempo.",
    "get_default_settings": "Inspection only: never copy unchanged defaults back into a generation call. A setting the user asks for that is not a direct parameter must appear in extra_settings here, under that exact name; otherwise it is not available: tell the user and do not generate.",
    "get_loras": "Copy the returned identifiers exactly (with subfolders) and keep the user's multipliers, including ',' and ';' separators.",
    "extract_image": "For the selected frame of a selected video, use get_selected_media's current_time_seconds exactly.",
    "extract_video": "Use time arguments (start_s, end_s, duration_s) or frame arguments (start_frame, end_frame, num_frames), not both.",
    "extract_audio": "Time arguments for audio or video, frame arguments for video; audio_track_no when get_media_details shows several tracks.",
    "transcribe_media": "timestamp_type: segment (default), word or none.",
    "resize_crop": "Read get_media_details first. Crop values in pixels or percent; crop_anchor keeps a side or corner; stretch_to_fit only when the user asks to stretch.",
    "postprocessing": "Call with media_id only to list the operations and their parameters, then with one returned process and parameters. Never guess a process.",
    "side_by_side": "layout and legends only when requested.",
}

ZERO_TOOLBOX_DESCRIPTION = "More tools: run one directly with toolbox(action=..., arguments={...}). Actions with their arguments (* required): {actions}. With an action and no arguments it returns that action's description and rules; with no arguments, one line per action."

# Words small models use for toolbox actions.
ZERO_TOOLBOX_SYNONYMS = {
    "cut": "extract_video", "cut_video": "extract_video", "trim": "extract_video", "trim_video": "extract_video", "clip": "extract_video", "clip_video": "extract_video",
    "screenshot": "extract_image", "frame": "extract_image", "get_frame": "extract_image", "trim_audio": "extract_audio", "cut_audio": "extract_audio",
    "join": "merge_videos", "concat": "merge_videos", "concatenate": "merge_videos", "merge": "merge_videos", "combine_videos": "merge_videos",
    "crop": "resize_crop", "resize": "resize_crop", "song": "gen_song", "music": "gen_song", "generate_song": "gen_song", "gen_music": "gen_song",
    "speech": "gen_speech_from_description", "tts": "gen_speech_from_description", "voice": "gen_speech_from_description", "gen_speech": "gen_speech_from_description",
    "clone_voice": "gen_speech_from_sample", "transcribe": "transcribe_media", "transcript": "transcribe_media", "upscale": "postprocessing", "enhance": "postprocessing",
    "mute": "mute_video", "remove_audio": "mute_video", "add_audio": "replace_audio", "set_audio": "replace_audio", "details": "get_media_details", "media_details": "get_media_details",
    "settings": "get_default_settings", "loras": "get_loras", "list_loras": "get_loras",
}
