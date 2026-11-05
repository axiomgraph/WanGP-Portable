# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""The Worlds explorer's menus for an EVOKE world (shared.utils.world_exploration): moves of whole 1.5 s chunks, no quality choice (three
fixed pyramid forwards), and the memory as the video the world keeps in its memory, counted in latent frames (the explorer's unit)."""
from shared.utils.world_instructions import CAMERA_CONTROLS
from .camera import CHUNK_FRAMES, CHUNK_LATENTS, FPS
from .vigeo import fit_patch_res
from .world_bank import INGEST_FRAMES

MEMORY_SECONDS = (30, 60, 90, 180, 300)  # the Memory menu: video kept in the memory
DEFAULT_MEMORY_SECONDS = 90  # the paper's long-session setting
LATENTS_PER_SECOND = CHUNK_LATENTS * FPS / CHUNK_FRAMES


def memory_frames(seconds):
    return round(seconds * LATENTS_PER_SECOND)


def explorer_menus(manifest):
    """Controls and menus of the explorer for an EVOKE world (handler query_world_explorer): a walking camera with text events. The memory's RAM: the depth and image of the 12 frames a chunk
    stores, at ViGeo's processing size (its co-visibility samples on the GPU are a few KB per frame)."""
    height, width = fit_patch_res(manifest["height"], manifest["width"])
    frame_bytes = height * width * (4 + 3)
    def needs(seconds):
        return f"{seconds * FPS / CHUNK_FRAMES * INGEST_FRAMES * frame_bytes / 2**30:.1f} GB RAM"
    return {"controls": CAMERA_CONTROLS,
            "seconds": [(value, f"{value:g} s", round(value * LATENTS_PER_SECOND)) for value in (1.5, 3, 4.5)],
            "steps": [(3, "Turbo")],
            "memory": [(memory_frames(seconds), str(seconds), needs(seconds)) for seconds in MEMORY_SECONDS],
            "memory_title": "How much of the explored video the world remembers: places seen within it keep their layout when you come back. More needs the RAM shown "
                            "when the menu is open, not more time",
            "defaults": {"seconds": 1.5, "steps": 3, "memory": memory_frames(DEFAULT_MEMORY_SECONDS)},
            # the world's memory replaces the start image; the world keeps its own size (worlds made before start images were cropped
            # can have another one): the task gives the model's locked resolution only to pass WanGP's check
            "task": {"video_length": CHUNK_FRAMES, "image_prompt_type": "", "resolution": "640x384"}}
