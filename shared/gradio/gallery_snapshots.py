# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Add a snapshot of the 3D viewer or of a world being explored to the image gallery, so that the view can feed image and video models,
and the video of a world exploration to the video gallery."""
import base64
import binascii
import io
import json
from pathlib import Path

import gradio as gr
from PIL import Image

from shared.utils.audio_video import save_image_metadata
from shared.utils.gallery_sources import MESH, WORLD

_PNG_PREFIX = "data:image/png;base64,"
_MAX_SNAPSHOT_BYTES = 64 * 1024 * 1024
_SNAPSHOT_SOURCES = {MESH.id: (MESH, "3D Viewer Snapshot", "3D asset"), WORLD.id: (WORLD, "World Snapshot", "world")}  # metadata type, item name


def add_snapshot(service, payload):
    data, source_id = payload['image'], payload.get('gallery', MESH.id)  # the 3D viewer's script sends no gallery
    if source_id not in _SNAPSHOT_SOURCES:
        raise ValueError('Invalid snapshot source.')
    gallery, label, item = _SNAPSHOT_SOURCES[source_id]
    if not isinstance(data, str) or not data.startswith(_PNG_PREFIX) or len(data) > _MAX_SNAPSHOT_BYTES:
        raise ValueError('Invalid snapshot image.')
    try:
        image = Image.open(io.BytesIO(base64.b64decode(data[len(_PNG_PREFIX):], validate=True)))
        image.load()
    except (binascii.Error, OSError) as error:
        raise ValueError('Invalid snapshot image.') from error
    with service._mutation_lock:
        if payload['workspace'] != service.workspace_id:
            raise ValueError('The workspace changed. Please try again.')
        gen = service._state['gen']
        source = payload['path']
        if source not in gen[gallery.files_key]:
            raise ValueError(f'The {item} is no longer in this gallery.')
        # A rendered view, not a generation: the settings of the source do not apply to it, only a comment names its source
        # (the WanGP type keeps this metadata when the gallery reads it back from the file)
        metadata = {'type': f'WanGP DeepBeepMeep - {label}', 'comments': f'Snapshot of "{Path(source).name}"'}
        deps = service._deps.controller._deps
        name = f'{Path(source).stem}_snapshot.png'
        # Exclusive creation keeps WanGP's naming policy safe against another writer.
        while True:
            output = Path(deps.get_output_filepath(name, True, False))
            output.parent.mkdir(parents=True, exist_ok=True)
            try:
                writer = output.open('xb')
                break
            except FileExistsError:
                continue
        try:
            with writer:
                image.save(writer, format='PNG')
            if not save_image_metadata(str(output), metadata):
                raise OSError('Could not save snapshot metadata.')
        except Exception:
            output.unlink(missing_ok=True)
            raise
        finally:
            image.close()
        deps.record_file_metadata(str(output), metadata, True, False, gen, notify_generation=False, write_metadata=False, record_notification=False)
        gen['last_selected'] = False
        service.save_workspace()
        service.host_changed()
        return output


def add_exploration_video(service, world_path, frames, fps, moves):
    """Saves the video of a world exploration (frames: uint8 (frames, height, width, 3): the world's own video, then the moves) as a
    new video of the Video gallery; a comment names the world (the world's generation settings do not describe the moves)."""
    import imageio.v3 as iio
    with service._mutation_lock:
        gen = service._state['gen']
        deps = service._deps.controller._deps
        metadata = {'type': 'WanGP DeepBeepMeep - World Exploration', 'comments': f'Exploration of "{Path(world_path).name}": its video and {moves} moves'}
        name = f'{Path(world_path).stem}_exploration.mp4'
        while True:  # exclusive creation keeps WanGP's naming policy safe against another writer
            output = Path(deps.get_output_filepath(name, False, False))
            output.parent.mkdir(parents=True, exist_ok=True)
            try:
                output.open('xb').close()
                break
            except FileExistsError:
                continue
        try:
            iio.imwrite(output, frames, fps=fps, codec='libx264', quality=8)
        except Exception:
            output.unlink(missing_ok=True)
            raise
        deps.record_file_metadata(str(output), metadata, False, False, gen, notify_generation=False, record_notification=False)
        gen['last_selected'] = False  # as for snapshots: the gallery keeps its selection
        service.save_workspace()
        service.host_changed()
        return output


def add_exploration_world(service, world_path, data, moves):
    """Saves a world as explored so far (data: its world file's bytes, its segments then the moves) as a new world of the Worlds gallery,
    with the generation settings of the world it comes from and a comment. Not shown at once: refreshing the gallery would replace the
    viewer exploring it (the explorer refreshes it when the viewer leaves)."""
    from shared.utils.world_outputs import read_world_metadata
    with service._mutation_lock:
        gen = service._state['gen']
        deps = service._deps.controller._deps
        stem = Path(world_path).stem.removesuffix('_explored')
        metadata = {**(read_world_metadata(world_path) or {}), 'comments': f'"{Path(world_path).name}" explored: its world and {moves} moves'}
        while True:  # exclusive creation keeps WanGP's naming policy safe against another writer
            output = Path(deps.get_output_filepath(f'{stem}_explored.wgpworld', False, False))
            output.parent.mkdir(parents=True, exist_ok=True)
            try:
                with output.open('xb') as writer:
                    writer.write(data)
                break
            except FileExistsError:
                continue
        deps.record_file_metadata(str(output), metadata, False, False, gen, notify_generation=False, record_notification=False)
        gen['last_selected'] = False  # as for snapshots: the gallery keeps its selection
        service.save_workspace()
        return output


def bind_gallery_snapshots(service):
    request = gr.Textbox(visible=False, elem_id='wangp-gallery-add-snapshot')

    def add(payload):
        try:
            output = add_snapshot(service, json.loads(payload))
        except (ValueError, OSError) as error:
            raise gr.Error(str(error), print_exception=False) from error
        gr.Info(f'Added {output.name} to the Image Gallery.')

    request.input(add, inputs=[request], outputs=None, queue=True, show_progress='hidden', trigger_mode='multiple', api_name=False)
