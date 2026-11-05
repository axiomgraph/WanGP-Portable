# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Facts of the Worlds gallery files (.wgpworld) for the gallery, Deepy and MCP: a world file is a zip whose manifest.json describes
the world (model, resolution, fps, its segments and the world description they were generated from) and settings.json holds its
generation settings; the video of the world, when it was generated too, is the .mp4 of the same name, in the image / video gallery."""
from __future__ import annotations

import itertools
import json
import os
import tempfile
import zipfile
from pathlib import Path


def read_manifest(path):
    with zipfile.ZipFile(path) as archive:
        return json.loads(archive.read("manifest.json"))


def video_path(path):
    return os.path.splitext(path)[0] + ".mp4"


def poster_image(path):
    """A JPEG of the world's last frame, where an exploration starts (its poster, or its thumbnail for worlds saved before posters),
    extracted once to the temporary folder: what visual inspection looks at."""
    with zipfile.ZipFile(path) as archive:
        name = "poster.jpg" if "poster.jpg" in archive.namelist() else "thumbnail.jpg"
        target = Path(tempfile.gettempdir()) / "wangp_world_posters" / f"{json.loads(archive.read('manifest.json'))['world_id']}.jpg"
        if not target.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(name))
    return str(target)


def save_world_metadata(path, settings):
    """Stores the generation settings in the world file (once: a world keeps the settings it was generated with)."""
    with zipfile.ZipFile(path) as archive:
        if "settings.json" in archive.namelist():
            return
    with zipfile.ZipFile(path, "a") as archive:
        archive.writestr("settings.json", json.dumps(settings, ensure_ascii=False))


def read_world_metadata(path):
    """Generation settings of a world: stored in the world file, else those of its video (worlds saved before settings.json)."""
    with zipfile.ZipFile(path) as archive:
        if "settings.json" in archive.namelist():
            return json.loads(archive.read("settings.json"))
    video = video_path(path)
    if not os.path.isfile(video):
        return None
    from shared.utils.video_metadata import read_metadata_from_video
    return read_metadata_from_video(video)


def world_summary(path, segments=True):
    """Compact facts of a world: model, resolution, fps, frame_count / duration_seconds of its video, perspective, style, subjects,
    its video file when present and, with segments, each segment's time range (seconds), exploration instruction, description, actions
    and camera keys."""
    manifest = read_manifest(path)
    world = manifest["world"]
    frames = sum(segment["video_frames"] for segment in manifest["segments"])
    summary = {"model_type": manifest["model_type"], "resolution": f"{manifest['width']}x{manifest['height']}", "fps": manifest["fps"], "frame_count": frames,
               "duration_seconds": round(frames / manifest["fps"], 3), "segment_count": len(manifest["segments"]), "explorable": True}
    summary.update({key: world[source] for key, source in (("perspective", "perspective_view"), ("style", "Style"), ("subjects", "subject_list")) if world.get(source)})
    video = video_path(path)
    if os.path.isfile(video):
        summary["video_filename"] = os.path.basename(video)
    if segments:
        starts = list(itertools.accumulate((segment["video_frames"] for segment in manifest["segments"]), initial=0))
        summary["segments"] = [{"start_seconds": round(starts[index] / manifest["fps"], 2), "end_seconds": round(starts[index + 1] / manifest["fps"], 2),
                                **({"instruction": segment["instruction"]} if segment.get("instruction") else {}),
                                "description": segment.get("visual_description_long") or segment.get("description_change") or "",
                                **({"actions": segment["contained_actions"]} if segment.get("contained_actions") else {}),
                                **({"camera_keys": segment["camera"]["keys"]} if (segment.get("camera") or {}).get("keys") else {})} for index, segment in enumerate(world["segments"])]
    else:
        summary["description"] = next((segment.get("visual_description_long") for segment in world["segments"] if segment.get("visual_description_long")), "")[:240]
    return summary
