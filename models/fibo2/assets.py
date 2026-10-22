# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from shared.utils.download import process_files_def_if_needed
from shared.utils import files_locator as fl
from .text_images import load_font


def query_font_files():
    return {"repoId": "DeepBeepMeep/Fibo2", "sourceFolderList": ["fibo2"],
            "fileList": [["Aileron-Regular.otf", "DejaVuSans.ttf", "LICENSE-Aileron.txt", "LICENSE-DejaVu.txt"]]}


def load_fonts():
    # The shared downloader inherits the active generation's cancellation context.
    process_files_def_if_needed(query_font_files(), status_text="Downloading Text Rendering Fonts")
    fonts = [load_font(fl.locate_file(f"fibo2/{name}")) for name in ("Aileron-Regular.otf", "DejaVuSans.ttf")]
    for name in ("LICENSE-Aileron.txt", "LICENSE-DejaVu.txt"):
        fl.locate_file(f"fibo2/{name}")
    return fonts
