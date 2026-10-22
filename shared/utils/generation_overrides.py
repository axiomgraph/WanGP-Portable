# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""The `overrides` setting: configuration choices replaced for one generation, written as `name=value` pairs
separated by commas, for example "attention_mode='sage2', memory_profile=4". Only non-default choices are written."""

import ast

OVERRIDE_DEFAULTS = {"memory_profile": -1, "compilation": "", "attention_mode": "", "transformer_quantization": "", "text_encoder_quantization": ""}


def parse_overrides(text):
    """The overrides written in text as a dictionary; ValueError for a malformed text or an unknown name."""
    if not text.strip():
        return {}
    try:
        call = ast.parse(f"f({text})", mode="eval").body
        if call.args or not isinstance(call.func, ast.Name) or any(keyword.arg is None for keyword in call.keywords):
            raise SyntaxError
        values = {keyword.arg: ast.literal_eval(keyword.value) for keyword in call.keywords}
        if len(values) < len(call.keywords):
            raise SyntaxError
    except (SyntaxError, ValueError):
        raise ValueError(f"overrides must be name=value pairs separated by commas, such as \"attention_mode='sdpa', memory_profile=4\", not {text!r}") from None
    unknown = [name for name in values if name not in OVERRIDE_DEFAULTS]
    if unknown:
        raise ValueError(f"Unknown override{'s' if len(unknown) > 1 else ''} {', '.join(unknown)}: the overrides are {', '.join(OVERRIDE_DEFAULTS)}")
    return values


def format_overrides(values):
    return ", ".join(f"{name}={values[name]!r}" for name, default in OVERRIDE_DEFAULTS.items() if name in values and values[name] != default)


def get_override(text, name):
    return parse_overrides(text).get(name, OVERRIDE_DEFAULTS[name])


def set_override(text, name, value):
    return format_overrides({**parse_overrides(text), name: value})
