# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""The prompt enhancer called from inside a running WanGP process (a generation or an exploration task, a plugin job): the caller already
owns the GPU, so no GPU lock is asked for (asking would wait for the caller itself). WanGP registers its runner at startup; enhance()
runs the configured enhancer (local or remote engine) on texts with the caller's instructions. A local engine is loaded for the call and
unloaded after it, the caller's models leaving the VRAM first (MMGP brings them back at their next forward)."""
_runner = {"enhance": None, "available": None}


def register(enhance, available):
    """enhance(texts, instructions, max_tokens, seed) -> one text per text; available() -> whether an enhancer is configured."""
    _runner.update(enhance=enhance, available=available)


def available():
    """Whether a prompt enhancer is configured (outside WanGP, none is)."""
    return _runner["available"] is not None and _runner["available"]()


def enhance(texts, instructions, max_tokens=512, seed=0):
    """The texts rewritten by the prompt enhancer following instructions (its system prompt), one result per text."""
    return _runner["enhance"](texts, instructions, max_tokens, seed)
