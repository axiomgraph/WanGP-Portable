# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""The .wgpworld file of an EVOKE world: an uncompressed zip with
- manifest.json: format version, model type, resolution, fps, seed, the world description, its segments (latent range, video frames, keys,
  event, instruction), the camera after the last frame, the locked depth scale and the memory retention;
- tensors.safetensors: the start image latent and its fake history latent (1, 16, 1, h, w), the chunks' latents (1, 16, 9 n, h, w) BF16,
  the pose of every frame (36 n, 4, 4) and the stored memory (bank_*: depth FP16, RGB uint8, intrinsics, poses, co-visibility samples);
- thumbnail.jpg (small, for galleries) and poster.jpg (the last frame at the world's resolution).
The memory is stored because it is the world's memory: rebuilding it would need every chunk decoded again and ViGeo over every window."""
import io
import json
import zipfile

import torch
from safetensors.torch import load, save

FORMAT, VERSION, EXTENSION = "wgpworld", 1, ".wgpworld"


def build(manifest, session, thumbnail_jpeg, poster_jpeg):
    tensors = {"image_latent": session.image_latent.to("cpu", torch.bfloat16).contiguous(), "fake_latent": session.fake_latent.to("cpu", torch.bfloat16).contiguous(),
               "latents": session.latents.to("cpu", torch.bfloat16).contiguous(), "poses": session.poses.to("cpu", torch.float32).contiguous()}
    tensors.update({key: value.contiguous() for key, value in session.bank.state().items()})
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
