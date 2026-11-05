# Adapted from Alaya-EVOKE (https://github.com/AlayaLab/Evoke, InteractiveUI/ui/controls.py, ui/live_runtime.py plan_poses,
# ui/trajectory.py, evoke/pipelines/pipeline_evoke.py _geo_da3_setup), Copyright 2026 The Evoke Team, Apache License 2.0. Modified for WanGP.
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Camera of EVOKE worlds: held keys (the explorer's names: w / s walk, a / d step sideways, the arrows look) move a camera in the ground
plane and turn it (yaw, pitch within +-80 degrees), one camera-to-world pose (OpenCV axes) per video frame at 24 fps, absolute for the
whole world (the memory stores its frames under them). Units: the depth scale makes the start view's median depth 5."""
import math
from dataclasses import dataclass, replace

import torch

FPS, CHUNK_FRAMES, TARGET_FRAMES, CHUNK_LATENTS = 24, 36, 33, 9  # a chunk: 36 frames (1.5 s), its memory view 33 frames, 9 latents
MOVE_SPEED, LOOK_SPEED = 0.5, 15.0  # pose units per second, degrees per second: the source's live defaults
HORIZONTAL_FOV, SOURCE_SIZE = 70.0, (720, 1280)  # the live sessions' camera: 70 degrees at 1280x720, scaled per axis to the output
EXPLORER_KEYS = {"w": "w", "s": "s", "a": "a", "d": "d", "left": "j", "right": "l", "up": "i", "down": "k"}


def intrinsics(height, width):
    """K (3, 3) of the output frames: the 70 degree 1280x720 camera with each axis scaled to the output (non-square at 640x384)."""
    focal = (SOURCE_SIZE[1] / 2.0) / math.tan(math.radians(HORIZONTAL_FOV) / 2.0)
    sy, sx = height / SOURCE_SIZE[0], width / SOURCE_SIZE[1]
    return torch.tensor([[focal * sx, 0.0, SOURCE_SIZE[1] / 2 * sx], [0.0, focal * sy, SOURCE_SIZE[0] / 2 * sy], [0.0, 0.0, 1.0]], dtype=torch.float32)


@dataclass
class Camera:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0

    def step(self, keys, dt, speed=MOVE_SPEED, look_speed=LOOK_SPEED):
        """Moves by dt seconds with the source keys held (w / a / s / d, i / j / k / l)."""
        look = math.radians(look_speed)
        turn = (("l" in keys) - ("j" in keys)) * look * dt
        midpoint = self.yaw + turn / 2
        self.yaw += turn
        self.pitch = max(-math.radians(80), min(math.radians(80), self.pitch + (("i" in keys) - ("k" in keys)) * look * dt))
        right, forward = ("d" in keys) - ("a" in keys), ("w" in keys) - ("s" in keys)
        arc = math.sin(turn / 2) / (turn / 2) if abs(turn) > 1e-8 else 1.0
        distance = speed * dt * arc / max(1.0, math.hypot(right, forward))
        self.x += (right * math.cos(midpoint) + forward * math.sin(midpoint)) * distance
        self.z += (forward * math.cos(midpoint) - right * math.sin(midpoint)) * distance

    def pose(self):
        cy, sy, cp, sp = math.cos(self.yaw), math.sin(self.yaw), math.cos(self.pitch), math.sin(self.pitch)
        return torch.tensor([[cy, sy * sp, sy * cp, self.x], [0.0, cp, -sp, self.y], [-sy, cy * sp, cy * cp, self.z], [0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)

    def as_dict(self):
        return {"x": self.x, "y": self.y, "z": self.z, "yaw": self.yaw, "pitch": self.pitch}


def source_keys(keys):
    """The explorer's key names (w, a, s, d, left, right, up, down; "+" joined or a list) as the source's keys."""
    if isinstance(keys, str):
        keys = [key for key in keys.split("+") if key]
    return {EXPLORER_KEYS[key] for key in keys or []}


def plan(camera, keys, frames=CHUNK_FRAMES, first=False):
    """(poses (frames, 4, 4), camera after them) of a chunk with keys held: one pose per frame, each 1/24 s after the previous one (the
    world's first pose is the starting camera itself)."""
    camera, poses = replace(camera), []
    for index in range(frames):
        if index or not first:
            camera.step(keys, 1.0 / FPS)
        poses.append(camera.pose())
    return torch.stack(poses), camera
