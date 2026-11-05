# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""The .wgpworld file of a Hy-Nomas world: an uncompressed zip with
- manifest.json: format version, model type, resolution, fps, seed, the world description, and per segment its latent frame range,
  video frames, the number of its text tokens later segments see and whether its camera was driven;
- tensors.safetensors: the clip's latents (1, 32, L, h, w) BF16, each segment's text embeddings (text_<i>, BF16) and the camera poses
  of the clip (4L, 4, 4) FP32 when a camera was driven;
- thumbnail.jpg: the last frame, small (galleries embed it); poster.jpg: the last frame at the world's resolution (the viewer's still).
Exploring a world rebuilds its history from these (one forward per segment, no text encoder, no decoding); the KV cache itself is never
stored."""
import io
import json
import zipfile

import torch
from safetensors.torch import load, save

FORMAT, VERSION, EXTENSION = "wgpworld", 1, ".wgpworld"


def build(manifest, latents, text_states, camera, thumbnail_jpeg, poster_jpeg):
    tensors = {"latents": latents.to("cpu", torch.bfloat16).contiguous()}
    tensors.update({f"text_{index}": states.to("cpu", torch.bfloat16).contiguous() for index, states in enumerate(text_states)})
    if camera is not None:
        tensors["camera"] = camera.to("cpu", torch.float32).contiguous()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
        archive.writestr("manifest.json", json.dumps({"format": FORMAT, "version": VERSION, **manifest}, ensure_ascii=False, indent=1))
        archive.writestr("tensors.safetensors", save(tensors))
        archive.writestr("thumbnail.jpg", thumbnail_jpeg)
        archive.writestr("poster.jpg", poster_jpeg)
    return buffer.getvalue()


def read(path):
    """(manifest, tensors) of a world file."""
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        if manifest.get("format") != FORMAT or manifest.get("version") != VERSION:
            raise ValueError(f"{path} is not a version {VERSION} world file")
        return manifest, load(archive.read("tensors.safetensors"))


def read_thumbnail(path):
    with zipfile.ZipFile(path) as archive:
        return archive.read("thumbnail.jpg")
