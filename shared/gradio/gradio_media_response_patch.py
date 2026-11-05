# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Stop abandoned Gradio media transfers, batch small range reads and serve the media the galleries list."""
import os

from gradio import ranged_response, routes, utils

from shared.utils.http_disconnect import DisconnectAwareResponse, install_http_disconnect_patch
from shared.gradio.gallery_previews import GalleryFileResponse


class _RangedFileResponse(DisconnectAwareResponse, ranged_response.RangedFileResponse):
    chunk_size = 256 * 1024


_listed = set()
_gradio_is_allowed_file = utils.is_allowed_file


def allow_listed_media(paths):
    """Serve these gallery files wherever they were saved (an earlier output folder, a file outside the workspaces).
    Gradio refuses files outside its launch folders, and one refused file fails the whole image/video gallery update."""
    _listed.update(os.path.normcase(os.path.abspath(path)) for path in paths)


def _is_allowed_file(path, blocked_paths, allowed_paths, created_paths):
    allowed, reason = _gradio_is_allowed_file(path, blocked_paths, allowed_paths, created_paths)
    if reason == "not_created_or_allowed" and os.path.normcase(str(path)) in _listed:
        return True, "allowed"
    return allowed, reason


def install():
    if ranged_response.RangedFileResponse is _RangedFileResponse:
        return
    install_http_disconnect_patch()
    routes.FileResponse = GalleryFileResponse
    ranged_response.RangedFileResponse = _RangedFileResponse
    utils.is_allowed_file = _is_allowed_file
