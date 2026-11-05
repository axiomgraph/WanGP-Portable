# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Interactive exploration of a Hy-Nomas world (shared.utils.world_exploration): the length of its moves and the Worlds explorer's menus
(Move, Quality, Memory) for a Hy-Nomas world."""
from shared.utils.world_instructions import CAMERA_CONTROLS
from .prompts import FPS, TEMPORAL_RATIO

MOVE_SECONDS = 1.0  # length of a move or an exploration instruction that gives none (the explorer's default)
EVENT_SECONDS = 2.5  # least length of a move with a typed event: upstream's turns give actions and scene changes the time to happen
MOVE_STRIDE = 5  # latent frames of a decoder stride (pixelumm.OVERLAP): moves last whole strides


def move_frames(seconds, event=False):
    """Latent frames of a move of `seconds` (EVENT_SECONDS at least with a text event), in whole decoder strides (5 latent frames,
    0.83 s): each one costs exactly one more decoder window."""
    return max(MOVE_STRIDE, round(max(seconds, EVENT_SECONDS if event else 0.0) * FPS / TEMPORAL_RATIO / MOVE_STRIDE) * MOVE_STRIDE)


def explorer_menus(manifest, memory):
    """The explorer's controls and menus for a Hy-Nomas world (handler query_world_explorer): a walking camera with text events, moves of
    1 to 3 s, denoising steps, history frames seen, whose needs follow the memory profile (memory["preload"]: one more block of the history
    on the GPU)."""
    from .history import memory_choices
    from .transformer import DEPTH, HEAD_DIM, HEADS
    return {"controls": CAMERA_CONTROLS,
            "seconds": [(value, f"{value} s", move_frames(value)) for value in (1, 2, 3)],
            "steps": [(8, "Best"), (6, "Balanced"), (4, "Fast")],
            "memory": memory_choices((manifest["width"] // 32) * (manifest["height"] // 32), DEPTH, HEADS, HEAD_DIM, TEMPORAL_RATIO / manifest["fps"], memory["preload"]),
            "memory_title": "How many moves of the chosen length each move sees: a sample spread over the whole exploration, denser for recent parts. "
                            "More remembers places better but makes moves slower (it stays in RAM and is sent to the graphics card at each step) "
                            "and needs the memory shown when the menu is open",
            "defaults": {"seconds": 1, "steps": 4, "memory": 18},
            "task": {"num_inference_steps": manifest["steps"], "video_length": 5}}
