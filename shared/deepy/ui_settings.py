# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from __future__ import annotations

from typing import Any

from shared.deepy.config import (
    DEEPY_MODEL_SPEED_KEY,
    DEEPY_MODEL_SIZE_KEY,
    DEEPY_TYPE_KEY,
    DEEPY_AUTO_CANCEL_QUEUE_TASKS_DEFAULT,
    DEEPY_AUTO_CANCEL_QUEUE_TASKS_KEY,
    DEEPY_SEPARATE_REQUESTS_WITH_EMPTY_LINE_DEFAULT,
    DEEPY_SEPARATE_REQUESTS_WITH_EMPTY_LINE_KEY,
    DEEPY_SESSION_GALLERY_MEDIA_MODE_DEFAULT,
    DEEPY_SESSION_GALLERY_MEDIA_MODE_KEY,
    DEEPY_SESSION_RESET_MODE_DEFAULT,
    DEEPY_SESSION_RESET_MODE_KEY,
    DEEPY_MULTI_SESSION_DEFAULT,
    DEEPY_MULTI_SESSION_KEY,
    get_deepy_config_value,
    normalize_deepy_auto_cancel_queue_tasks,
    normalize_deepy_separate_requests_with_empty_line,
    normalize_deepy_session_gallery_media_mode,
    normalize_deepy_session_reset_mode,
    normalize_deepy_session_mode,
)
from shared.deepy import tool_settings as deepy_tool_settings


ASSISTANT_OVERRIDE_DIMENSION_MIN = 256
ASSISTANT_OVERRIDE_DIMENSION_MAX = 3840
ASSISTANT_OVERRIDE_DIMENSION_STEP = 16
ASSISTANT_OVERRIDE_WIDTH_DEFAULT = 1280
ASSISTANT_OVERRIDE_HEIGHT_DEFAULT = 720
ASSISTANT_OVERRIDE_FRAMES_MIN = 5
ASSISTANT_OVERRIDE_FRAMES_MAX = 3000
ASSISTANT_OVERRIDE_FRAMES_DEFAULT = 81
ASSISTANT_OVERRIDE_AUDIO_DURATION_MIN = 1
ASSISTANT_OVERRIDE_AUDIO_DURATION_MAX = 600
ASSISTANT_OVERRIDE_AUDIO_DURATION_DEFAULT = 10
ASSISTANT_OVERRIDE_SEED_DEFAULT = -1
ASSISTANT_USE_TEMPLATE_PROPERTIES_KEY = "deepy_use_template_properties"
ASSISTANT_OVERRIDE_WIDTH_KEY = "deepy_width"
ASSISTANT_OVERRIDE_HEIGHT_KEY = "deepy_height"
ASSISTANT_OVERRIDE_NUM_FRAMES_KEY = "deepy_num_frames"
ASSISTANT_OVERRIDE_AUDIO_DURATION_KEY = "deepy_audio_duration"
ASSISTANT_OVERRIDE_SEED_KEY = "deepy_seed"


def _pairs(tools):
    return tuple(tuple(tools[index:index + 2]) for index in range(0, len(tools), 2))


# the template pickers, two per row in the registry order: Deepy Zero's tasks, then the tasks of agents only (shown for Deepy Prime)
TEMPLATE_TOOL_LAYOUT = (_pairs([tool for tool in deepy_tool_settings.GENERATION_TOOL_IDS if tool in deepy_tool_settings.ZERO_GENERATION_TOOL_IDS])
                        + _pairs(deepy_tool_settings.AGENT_TOOL_IDS))
TEMPLATE_TOOL_UI_KEY = deepy_tool_settings.TOOL_UI_KEYS

MODEL_SELECTION_FIELDS = [
    {"key": "model_speed", "label": "Speed", "choices": [("Fast", "fast"), ("Standard", "standard"), ("No preference", "any")]},
    {"key": "model_size", "label": "Model Size", "choices": [("Smaller", "smaller"), ("Larger", "larger"), ("No preference", "any")]},
]


def normalize_model_preference(value, key):
    field = next(field for field in MODEL_SELECTION_FIELDS if field["key"] == key)
    if value not in [choice[1] for choice in field["choices"]]:
        raise ValueError(f"Invalid {field['label']} preference: {value}")
    return value


PROPERTY_MODE_CHOICES = [
    ("Use template defaults first", True),
    ("Use the values below", False),
]
GENERATION_PROPERTY_FIELDS = [
    {"key": "width", "label": "Default Width", "minimum": ASSISTANT_OVERRIDE_DIMENSION_MIN, "maximum": ASSISTANT_OVERRIDE_DIMENSION_MAX, "step": ASSISTANT_OVERRIDE_DIMENSION_STEP},
    {"key": "height", "label": "Default Height", "minimum": ASSISTANT_OVERRIDE_DIMENSION_MIN, "maximum": ASSISTANT_OVERRIDE_DIMENSION_MAX, "step": ASSISTANT_OVERRIDE_DIMENSION_STEP},
    {"key": "num_frames", "label": "Default Frames", "minimum": ASSISTANT_OVERRIDE_FRAMES_MIN, "maximum": ASSISTANT_OVERRIDE_FRAMES_MAX, "step": 1},
    {"key": "audio_duration", "label": "Default Audio (s)", "minimum": ASSISTANT_OVERRIDE_AUDIO_DURATION_MIN, "maximum": ASSISTANT_OVERRIDE_AUDIO_DURATION_MAX, "step": 1},
    {"key": "seed", "label": "Seed (-1 for random)", "minimum": -1, "maximum": 999999999, "step": 1},
]


def get_simplified_settings_form(settings, *, prime=None):
    if prime is None:
        prime = get_deepy_config_value(DEEPY_TYPE_KEY, "zero") == "prime"
    preferences = MODEL_SELECTION_FIELDS if prime else []
    templates = [{"key": TEMPLATE_TOOL_UI_KEY[tool], "label": deepy_tool_settings.TOOL_DISPLAY_NAMES[tool], "choices": deepy_tool_settings.list_tool_variant_choices(tool, current_variant=settings[TEMPLATE_TOOL_UI_KEY[tool]])}
                 for row in TEMPLATE_TOOL_LAYOUT for tool in row if prime or tool not in deepy_tool_settings.AGENT_TOOL_IDS]
    keys = [*(field["key"] for field in preferences), "use_template_properties", *(field["key"] for field in GENERATION_PROPERTY_FIELDS), *(field["key"] for field in templates)]
    return {"values": {key: settings[key] for key in keys}, "preferences": preferences, "property_modes": PROPERTY_MODE_CHOICES, "properties": GENERATION_PROPERTY_FIELDS, "templates": templates}


def validate_simplified_settings(values, form):
    if not isinstance(values, dict) or values.keys() - form["values"].keys():
        raise ValueError("Only the displayed Deepy settings can be changed here.")
    fields = {field["key"]: field for field in form["properties"]}
    choices = {field["key"]: [value for label, value in field["choices"]] for field in [*form["templates"], *form["preferences"]]}
    for key, value in values.items():
        if key == "use_template_properties":
            if type(value) is not bool:
                raise ValueError("The property mode must be a boolean.")
        elif key in fields:
            field = fields[key]
            if type(value) is not int or not field["minimum"] <= value <= field["maximum"] or (value - field["minimum"]) % field["step"]:
                raise ValueError(f'{field["label"]}: expected {field["minimum"]}–{field["maximum"]}, step {field["step"]}.')
        elif not isinstance(value, str) or value not in choices[key]:
            raise ValueError("Select one of the displayed choices.")


def _clamp_int(value: Any, default: int, minimum: int, maximum: int, step: int = 1) -> int:
    try:
        number = int(round(float(value)))
    except Exception:
        number = int(default)
    number = max(minimum, min(maximum, number))
    if step > 1:
        number = minimum + int(round((number - minimum) / step)) * step
        number = max(minimum, min(maximum, number))
    return int(number)


def _normalize_bool(value: Any) -> bool:
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "0", "false", "off", "no"}:
            return False
        if text in {"1", "true", "on", "yes"}:
            return True
    return bool(value)


def normalize_assistant_use_template_properties(value: Any) -> bool:
    return _normalize_bool(value)


def normalize_assistant_override_width(value: Any) -> int:
    return _clamp_int(value, ASSISTANT_OVERRIDE_WIDTH_DEFAULT, ASSISTANT_OVERRIDE_DIMENSION_MIN, ASSISTANT_OVERRIDE_DIMENSION_MAX, ASSISTANT_OVERRIDE_DIMENSION_STEP)


def normalize_assistant_override_height(value: Any) -> int:
    return _clamp_int(value, ASSISTANT_OVERRIDE_HEIGHT_DEFAULT, ASSISTANT_OVERRIDE_DIMENSION_MIN, ASSISTANT_OVERRIDE_DIMENSION_MAX, ASSISTANT_OVERRIDE_DIMENSION_STEP)


def normalize_assistant_override_num_frames(value: Any) -> int:
    return _clamp_int(value, ASSISTANT_OVERRIDE_FRAMES_DEFAULT, ASSISTANT_OVERRIDE_FRAMES_MIN, ASSISTANT_OVERRIDE_FRAMES_MAX, 1)


def normalize_assistant_override_audio_duration(value: Any) -> int:
    return _clamp_int(value, ASSISTANT_OVERRIDE_AUDIO_DURATION_DEFAULT, ASSISTANT_OVERRIDE_AUDIO_DURATION_MIN, ASSISTANT_OVERRIDE_AUDIO_DURATION_MAX, 1)


def normalize_assistant_override_seed(value: Any) -> int:
    return _clamp_int(value, ASSISTANT_OVERRIDE_SEED_DEFAULT, -1, 999999999, 1)


def normalize_assistant_separate_requests_with_empty_line(value: Any) -> bool:
    return normalize_deepy_separate_requests_with_empty_line(value)


def get_persisted_assistant_tool_ui_settings(server_config: dict[str, Any] | None = None) -> dict[str, Any]:
    source = server_config if isinstance(server_config, dict) else {}
    return normalize_assistant_tool_ui_settings(
        auto_cancel_queue_tasks=source.get(DEEPY_AUTO_CANCEL_QUEUE_TASKS_KEY, get_deepy_config_value(DEEPY_AUTO_CANCEL_QUEUE_TASKS_KEY, DEEPY_AUTO_CANCEL_QUEUE_TASKS_DEFAULT)),
        separate_requests_with_empty_line=source.get(DEEPY_SEPARATE_REQUESTS_WITH_EMPTY_LINE_KEY, get_deepy_config_value(DEEPY_SEPARATE_REQUESTS_WITH_EMPTY_LINE_KEY, DEEPY_SEPARATE_REQUESTS_WITH_EMPTY_LINE_DEFAULT)),
        model_speed=source.get(DEEPY_MODEL_SPEED_KEY, get_deepy_config_value(DEEPY_MODEL_SPEED_KEY, "fast")),
        model_size=source.get(DEEPY_MODEL_SIZE_KEY, get_deepy_config_value(DEEPY_MODEL_SIZE_KEY, "smaller")),
        use_template_properties=source.get(ASSISTANT_USE_TEMPLATE_PROPERTIES_KEY, get_deepy_config_value(ASSISTANT_USE_TEMPLATE_PROPERTIES_KEY, True)),
        width=source.get(ASSISTANT_OVERRIDE_WIDTH_KEY, get_deepy_config_value(ASSISTANT_OVERRIDE_WIDTH_KEY, ASSISTANT_OVERRIDE_WIDTH_DEFAULT)),
        height=source.get(ASSISTANT_OVERRIDE_HEIGHT_KEY, get_deepy_config_value(ASSISTANT_OVERRIDE_HEIGHT_KEY, ASSISTANT_OVERRIDE_HEIGHT_DEFAULT)),
        num_frames=source.get(ASSISTANT_OVERRIDE_NUM_FRAMES_KEY, get_deepy_config_value(ASSISTANT_OVERRIDE_NUM_FRAMES_KEY, ASSISTANT_OVERRIDE_FRAMES_DEFAULT)),
        audio_duration=source.get(ASSISTANT_OVERRIDE_AUDIO_DURATION_KEY, get_deepy_config_value(ASSISTANT_OVERRIDE_AUDIO_DURATION_KEY, ASSISTANT_OVERRIDE_AUDIO_DURATION_DEFAULT)),
        seed=source.get(ASSISTANT_OVERRIDE_SEED_KEY, get_deepy_config_value(ASSISTANT_OVERRIDE_SEED_KEY, ASSISTANT_OVERRIDE_SEED_DEFAULT)),
        **{key: source.get(deepy_tool_settings.TOOL_CONFIG_KEYS[tool], deepy_tool_settings.get_default_tool_variant(tool)) for tool, key in TEMPLATE_TOOL_UI_KEY.items()},
    )


def store_assistant_tool_ui_settings(server_config: dict[str, Any] | None, settings: dict[str, Any] | None) -> bool:
    if not isinstance(server_config, dict):
        return False
    normalized = normalize_assistant_tool_ui_settings(**dict(settings or {}))
    updates = {
        DEEPY_AUTO_CANCEL_QUEUE_TASKS_KEY: normalized["auto_cancel_queue_tasks"],
        DEEPY_SEPARATE_REQUESTS_WITH_EMPTY_LINE_KEY: normalized["separate_requests_with_empty_line"],
        DEEPY_MODEL_SPEED_KEY: normalized["model_speed"],
        DEEPY_MODEL_SIZE_KEY: normalized["model_size"],
        ASSISTANT_USE_TEMPLATE_PROPERTIES_KEY: normalized["use_template_properties"],
        ASSISTANT_OVERRIDE_WIDTH_KEY: normalized["width"],
        ASSISTANT_OVERRIDE_HEIGHT_KEY: normalized["height"],
        ASSISTANT_OVERRIDE_NUM_FRAMES_KEY: normalized["num_frames"],
        ASSISTANT_OVERRIDE_AUDIO_DURATION_KEY: normalized["audio_duration"],
        ASSISTANT_OVERRIDE_SEED_KEY: normalized["seed"],
        **{deepy_tool_settings.TOOL_CONFIG_KEYS[tool]: normalized[key] for tool, key in TEMPLATE_TOOL_UI_KEY.items()},
    }
    server_config.update(updates)
    return True


def get_persisted_assistant_session_ui_settings(server_config: dict[str, Any] | None = None) -> dict[str, Any]:
    source = server_config if isinstance(server_config, dict) else {}
    return {
        "multi_session": normalize_deepy_session_mode(source.get(DEEPY_MULTI_SESSION_KEY, get_deepy_config_value(DEEPY_MULTI_SESSION_KEY, DEEPY_MULTI_SESSION_DEFAULT))),
        "reset_mode": normalize_deepy_session_reset_mode(source.get(DEEPY_SESSION_RESET_MODE_KEY, get_deepy_config_value(DEEPY_SESSION_RESET_MODE_KEY, DEEPY_SESSION_RESET_MODE_DEFAULT))),
        "gallery_media_mode": normalize_deepy_session_gallery_media_mode(source.get(DEEPY_SESSION_GALLERY_MEDIA_MODE_KEY, get_deepy_config_value(DEEPY_SESSION_GALLERY_MEDIA_MODE_KEY, DEEPY_SESSION_GALLERY_MEDIA_MODE_DEFAULT))),
    }


def store_assistant_session_ui_settings(server_config: dict[str, Any] | None, *, multi_session: Any, reset_mode: Any, gallery_media_mode: Any) -> bool:
    if not isinstance(server_config, dict):
        return False
    server_config.update({
        DEEPY_MULTI_SESSION_KEY: normalize_deepy_session_mode(multi_session),
        DEEPY_SESSION_RESET_MODE_KEY: normalize_deepy_session_reset_mode(reset_mode),
        DEEPY_SESSION_GALLERY_MEDIA_MODE_KEY: normalize_deepy_session_gallery_media_mode(gallery_media_mode),
    })
    return True


def get_template_selector_state() -> dict[str, dict[str, Any]]:
    """{task: {"choices", "selected"}} of the template pickers, from the persisted settings."""
    persisted = get_persisted_assistant_tool_ui_settings()
    return {tool: {"choices": deepy_tool_settings.list_tool_variant_choices(tool, current_variant=persisted[key]), "selected": persisted[key]} for tool, key in TEMPLATE_TOOL_UI_KEY.items()}


def refresh_template_selector_state(tool_values: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """{task: {"choices", "selected"}} of the template pickers after the presets were reloaded, for the {task: template} shown."""
    deepy_tool_settings.refresh_tool_presets()
    return {tool: {"choices": deepy_tool_settings.list_tool_variant_choices(tool, current_variant=tool_values.get(tool)), "selected": deepy_tool_settings.find_tool_variant(tool, tool_values.get(tool))}
            for tool in TEMPLATE_TOOL_UI_KEY}


def normalize_assistant_tool_ui_settings(
    *,
    auto_cancel_queue_tasks: Any = None,
    separate_requests_with_empty_line: Any = None,
    use_template_properties: Any = None,
    priority: Any = None,
    model_speed: Any = None,
    model_size: Any = None,
    width: Any = None,
    height: Any = None,
    num_frames: Any = None,
    audio_duration: Any = None,
    seed: Any = None,
    **variants: Any,
) -> dict[str, Any]:
    """The Deepy tool settings, each template given as its task's ui_key (TEMPLATE_TOOL_UI_KEY); omitted values come from the config."""
    unknown = variants.keys() - set(TEMPLATE_TOOL_UI_KEY.values())
    if unknown:
        raise TypeError(f"Unknown Deepy tool settings: {', '.join(sorted(unknown))}")
    return {
        "auto_cancel_queue_tasks": normalize_deepy_auto_cancel_queue_tasks(get_deepy_config_value(DEEPY_AUTO_CANCEL_QUEUE_TASKS_KEY, DEEPY_AUTO_CANCEL_QUEUE_TASKS_DEFAULT) if auto_cancel_queue_tasks is None else auto_cancel_queue_tasks),
        "separate_requests_with_empty_line": normalize_assistant_separate_requests_with_empty_line(get_deepy_config_value(DEEPY_SEPARATE_REQUESTS_WITH_EMPTY_LINE_KEY, DEEPY_SEPARATE_REQUESTS_WITH_EMPTY_LINE_DEFAULT) if separate_requests_with_empty_line is None else separate_requests_with_empty_line),
        "model_speed": normalize_model_preference(get_deepy_config_value(DEEPY_MODEL_SPEED_KEY, "fast") if model_speed is None else model_speed, "model_speed"),
        "model_size": normalize_model_preference(get_deepy_config_value(DEEPY_MODEL_SIZE_KEY, "smaller") if model_size is None else model_size, "model_size"),
        "use_template_properties": normalize_assistant_use_template_properties(get_deepy_config_value(ASSISTANT_USE_TEMPLATE_PROPERTIES_KEY, True) if use_template_properties is None else use_template_properties),
        "width": normalize_assistant_override_width(get_deepy_config_value(ASSISTANT_OVERRIDE_WIDTH_KEY, ASSISTANT_OVERRIDE_WIDTH_DEFAULT) if width is None else width),
        "height": normalize_assistant_override_height(get_deepy_config_value(ASSISTANT_OVERRIDE_HEIGHT_KEY, ASSISTANT_OVERRIDE_HEIGHT_DEFAULT) if height is None else height),
        "num_frames": normalize_assistant_override_num_frames(get_deepy_config_value(ASSISTANT_OVERRIDE_NUM_FRAMES_KEY, ASSISTANT_OVERRIDE_FRAMES_DEFAULT) if num_frames is None else num_frames),
        "audio_duration": normalize_assistant_override_audio_duration(get_deepy_config_value(ASSISTANT_OVERRIDE_AUDIO_DURATION_KEY, ASSISTANT_OVERRIDE_AUDIO_DURATION_DEFAULT) if audio_duration is None else audio_duration),
        "seed": normalize_assistant_override_seed(get_deepy_config_value(ASSISTANT_OVERRIDE_SEED_KEY, ASSISTANT_OVERRIDE_SEED_DEFAULT) if seed is None else seed),
        **{key: deepy_tool_settings.get_default_tool_variant(tool) if variants.get(key) is None
           else deepy_tool_settings.resolve_tool_variant(tool, variants[key], default_variant=deepy_tool_settings.get_default_tool_variant(tool))
           for tool, key in TEMPLATE_TOOL_UI_KEY.items()},
    }


__all__ = [
    "GENERATION_PROPERTY_FIELDS",
    "PROPERTY_MODE_CHOICES",
    "TEMPLATE_TOOL_LAYOUT",
    "TEMPLATE_TOOL_UI_KEY",
    "get_simplified_settings_form",
    "validate_simplified_settings",
    "ASSISTANT_OVERRIDE_DIMENSION_MAX",
    "ASSISTANT_OVERRIDE_DIMENSION_MIN",
    "ASSISTANT_OVERRIDE_DIMENSION_STEP",
    "ASSISTANT_OVERRIDE_FRAMES_DEFAULT",
    "ASSISTANT_OVERRIDE_FRAMES_MAX",
    "ASSISTANT_OVERRIDE_FRAMES_MIN",
    "ASSISTANT_OVERRIDE_HEIGHT_DEFAULT",
    "ASSISTANT_OVERRIDE_HEIGHT_KEY",
    "ASSISTANT_OVERRIDE_NUM_FRAMES_KEY",
    "ASSISTANT_OVERRIDE_AUDIO_DURATION_DEFAULT",
    "ASSISTANT_OVERRIDE_AUDIO_DURATION_KEY",
    "ASSISTANT_OVERRIDE_AUDIO_DURATION_MAX",
    "ASSISTANT_OVERRIDE_AUDIO_DURATION_MIN",
    "ASSISTANT_OVERRIDE_SEED_DEFAULT",
    "ASSISTANT_OVERRIDE_SEED_KEY",
    "ASSISTANT_OVERRIDE_WIDTH_DEFAULT",
    "ASSISTANT_OVERRIDE_WIDTH_KEY",
    "ASSISTANT_USE_TEMPLATE_PROPERTIES_KEY",
    "get_persisted_assistant_tool_ui_settings",
    "get_persisted_assistant_session_ui_settings",
    "store_assistant_tool_ui_settings",
    "store_assistant_session_ui_settings",
    "get_template_selector_state",
    "normalize_assistant_override_height",
    "normalize_assistant_override_num_frames",
    "normalize_assistant_override_audio_duration",
    "normalize_assistant_override_seed",
    "normalize_assistant_override_width",
    "normalize_assistant_separate_requests_with_empty_line",
    "normalize_assistant_tool_ui_settings",
    "normalize_assistant_use_template_properties",
    "refresh_template_selector_state",
]
