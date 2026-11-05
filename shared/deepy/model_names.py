# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Light model lookup: WanGP model types found from the words a user names a model with."""
from __future__ import annotations

import re
from typing import Any

_TO = re.compile(r"(?<=\w)[\s_-]*\bto\b[\s_-]*(?=\w)")  # "text to video", "text-to-video" -> "text2video", as WanGP names write it


def _normalize(text: str) -> str:
    return re.sub(r"[^0-9a-z]+", "", _TO.sub("2", text.casefold()))


def match_models(query: str, records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
    """Records whose type, name or family hold every word of query (case, spaces, dots and dashes ignored), in catalog order; when none
    holds them all, the records holding the most words. Returns (records, complete)."""
    words = [word for word in (_normalize(part) for part in _TO.sub("2", query.casefold()).split()) if word]
    if not words:
        return list(records), True
    scored = []
    for record in records:
        haystack = _normalize(" ".join(str(record.get(key) or "") for key in ("model_type", "name", "family_label", "family")))
        scored.append((sum(word in haystack for word in words), record))
    best = max((score for score, _record in scored), default=0)
    return [record for score, record in scored if score == best and best > 0], best == len(words)


def lookup_row(record: dict[str, Any]) -> dict[str, Any]:
    """model_type, name and main output media of a model: what a request needs to name a model."""
    return {"model_type": record["model_type"], "name": record.get("name", record["model_type"]), "output": "/".join(record.get("main_output") or [])}
