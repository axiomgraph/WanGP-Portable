# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Shared availability and menu visibility contract for post-processing handlers."""

from __future__ import annotations

from typing import Any


PROCESSOR_STATUS_ENABLED = "enabled"
PROCESSOR_STATUS_DISABLED = "disabled"
PROCESSOR_STATUS_UNKNOWN = "unknown"
PROCESSOR_STATUSES = (PROCESSOR_STATUS_ENABLED, PROCESSOR_STATUS_DISABLED, PROCESSOR_STATUS_UNKNOWN)


def handler_method_available(handler: Any, method: str) -> bool:
    return not hasattr(handler, "method_available") or handler.method_available(method)


def handler_method_visible(handler: Any, method: str) -> bool:
    """Hidden methods remain available for metadata, restoration and execution."""
    return not hasattr(handler, "method_visible") or handler.method_visible(method)


def handler_method_in_choices(handler: Any, method: str, *, selected_method: str = "", include_hidden: bool = False) -> bool:
    return handler_method_available(handler, method) and (include_hidden or method == selected_method or handler_method_visible(handler, method))


def handler_status(handler: Any) -> str:
    if hasattr(handler, "enabled"):
        return PROCESSOR_STATUS_ENABLED if handler.enabled() else PROCESSOR_STATUS_DISABLED
    value = str(getattr(handler, "status", PROCESSOR_STATUS_UNKNOWN) or "").strip().lower()
    return value if value in PROCESSOR_STATUSES else PROCESSOR_STATUS_UNKNOWN


def handler_reason_disabled(handler: Any) -> str:
    if handler_status(handler) != PROCESSOR_STATUS_DISABLED:
        return ""
    return str(getattr(handler, "reason_disabled", "") or "").strip()
