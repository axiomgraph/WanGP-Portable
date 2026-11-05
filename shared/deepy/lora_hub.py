# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""LoRA hub of Deepy's web toolbox: find LoRAs on Civitai and Hugging Face that fit a WanGP model, and install one into that model's LoRA folder."""
from __future__ import annotations

import json
import re
import struct
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, quote, urlencode, urlsplit

CIVITAI_API = "https://civitai.com/api/v1"
HF_API = "https://huggingface.co/api"
PAGE_SIZE = 6
CACHE_TTL_S = 1800.0
TRIGGER_WORDS_SHOWN = 3
DESCRIPTION_SHOWN = 160
INSTALL_MB_DEFAULT = 2048
INSTALL_MB_MAX = 8192
DESCRIPTION_CHARS = 300
CIVITAI_SORT = {"downloads": "Most Downloaded", "rating": "Highest Rated", "newest": "Newest"}
# Civitai base-model names -> (WanGP family, LoRA folder name when only part of the family shares the LoRA format, else None)
CIVITAI_BASE_MODELS: dict[str, tuple[str, str | None]] = {
    "Flux.1 D": ("flux", None), "Flux.1 S": ("flux", None), "Flux.1 Kontext": ("flux", None), "Flux.1 Krea": ("flux", None), "Chroma": ("flux", None),
    "Flux.2 D": ("flux2", "flux2"), "Flux.2 Klein 9B": ("flux2", "flux2_klein_9b"), "Flux.2 Klein 9B-base": ("flux2", "flux2_klein_9b"), "Flux.2 Klein 4B": ("flux2", "flux2_klein_4b"), "Flux.2 Klein 4B-base": ("flux2", "flux2_klein_4b"),
    "Flux.3": ("flux3", None), "Flux.3 Klein": ("flux3", "flux3_klein"), "Flux.3 Klein-base": ("flux3", "flux3_klein_base"),
    "Krea 2": ("krea2", None), "Krea 2 Turbo": ("krea2", None),
    "Qwen": ("qwen", None), "Qwen 2.1": ("qwen_image_2", None),
    "ZImageTurbo": ("z_image", None), "ZImageBase": ("z_image", None),
    "MOSS Image": ("moss_image", None), "MiniMax H3": ("minimax_h3", None), "Ideogram 4.0": ("ideogram4", None), "HiDream": ("hidream", None),
    "Hunyuan Video": ("hunyuan", None), "Hunyuan Video 1.5": ("hunyuan_1_5", None),
    "LTXV": ("ltxv", None), "LTXV 2": ("ltx2", None), "LTX-2": ("ltx2", None),
    "Wan Video": ("wan", None), "Wan Video 1.3B t2v": ("wan", "wan_1.3B"), "Wan Video 14B t2v": ("wan", "wan"), "Wan Video 14B i2v 480p": ("wan", "wan_i2v"), "Wan Video 14B i2v 720p": ("wan", "wan_i2v"),
    "Wan Video 2.2 TI2V-5B": ("wan2_2", "wan_5B"), "Wan Video 2.2 T2V-A14B": ("wan2_2", "wan"), "Wan Video 2.2 I2V-A14B": ("wan2_2", "wan"),
}
# Hugging Face base_model ids (lower-case substrings, specific first) -> WanGP family
HF_BASE_MODELS: list[tuple[str, str]] = [
    ("flux.1-kontext", "flux"), ("flux.1", "flux"), ("chroma", "flux"), ("flux.2", "flux2"), ("flux-2", "flux2"), ("flux.3", "flux3"), ("flux-3", "flux3"), ("krea-2", "krea2"), ("krea2", "krea2"),
    ("qwen-image-2.1", "qwen_image_2"), ("qwen-image-21", "qwen_image_2"), ("qwen-image", "qwen"), ("z-image", "z_image"), ("moss", "moss_image"), ("minimax-h3", "minimax_h3"), ("h3-", "minimax_h3"), ("ideogram", "ideogram4"), ("hidream", "hidream"),
    ("hunyuanvideo-1.5", "hunyuan_1_5"), ("hunyuanvideo", "hunyuan"), ("ltx-2", "ltx2"), ("ltx2", "ltx2"), ("ltx-video", "ltxv"), ("wan2.2", "wan2_2"), ("wan-2.2", "wan2_2"), ("wan2.1", "wan"), ("wan-ai", "wan"), ("kandinsky-5", "kandinsky5"),
]
# Hugging Face base models name a size inside a family; the LoRA folder follows it (Wan 2.2 5B and 14B LoRAs are not interchangeable)
HF_FOLDER_HINTS = {"wan2_2": (("5b", "wan_5B"), ("14b", "wan")), "wan": (("1.3b", "wan_1.3B"), ("i2v", "wan_i2v"), ("14b", "wan"))}
HF_SEARCH_HINT = {"flux": "flux", "flux2": "flux2", "flux3": "flux3", "krea2": "krea", "qwen": "qwen-image", "qwen_image_2": "qwen-image", "z_image": "z-image", "wan": "wan2.1", "wan2_2": "wan2.2", "ltx2": "ltx-2", "ltxv": "ltx", "minimax_h3": "h3", "hunyuan": "hunyuanvideo", "hunyuan_1_5": "hunyuanvideo"}
_ID = {"type": "string", "minLength": 1, "description": "A result id from lora_search (civitai:<version id>, huggingface:<repo>[#file]) or a Civitai / Hugging Face url."}
ACTIONS = {
    "lora_search": {
        "description": "Find LoRAs on Civitai and Hugging Face for a WanGP model: theme or style query, compatibility with the model's LoRA format, trigger words, size and popularity.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "minLength": 1, "maxLength": 200, "description": "Theme, style, subject or effect, e.g. watercolor, film noir lighting, slow dolly zoom."},
            "model_type": {"type": "string", "description": "The WanGP model the LoRA will be used with (its model_type, or its name when that names one model family): restricts the search to its base model and marks each result compatible or not. Without it, results of every base model are returned with the WanGP families they fit."},
            "source": {"type": "string", "enum": ["both", "civitai", "huggingface"], "default": "both"},
            "max_results": {"type": "integer", "minimum": 1, "maximum": 20, "default": PAGE_SIZE, "description": "Results per page."},
            "page": {"type": "integer", "minimum": 1, "default": 1, "description": "Page of the same search (next_page gives the call); later pages come from the session's cache, without new requests."},
            "sort": {"type": "string", "enum": list(CIVITAI_SORT), "default": "downloads", "description": "Order among equally relevant results."},
            "nsfw": {"type": "boolean", "default": False, "description": "Include adult-rated LoRAs and previews (Civitai)."},
            "include_incompatible": {"type": "boolean", "default": False, "description": "With model_type, also list LoRAs made for other base models (by default they are only counted in other_base_models)."},
        }, "required": ["query"], "additionalProperties": False},
        "example": {"action": "lora_search", "arguments": {"query": "watercolor", "model_type": "krea2_turbo"}},
    },
    "lora_install": {
        "description": "Download a LoRA found with lora_search into the LoRA folder of a WanGP model (a zip of several files, such as a HIGH / LOW noise pair, is unpacked there), check it is safetensors and return the file names for activated_loras with its trigger words. WanGP finds installed LoRAs by file name: nothing to copy.",
        "parameters": {"type": "object", "properties": {
            "id": _ID,
            "model_type": {"type": "string", "minLength": 1, "description": "The WanGP model whose LoRA folder receives the file; the LoRA must be made for its base model unless force is set."},
            "name": {"type": "string", "pattern": "^[^\\\\/:*?\"<>|]{1,120}$", "description": "Optional file name (no folder); defaults to the hub's file name."},
            "max_mb": {"type": "number", "minimum": 1, "maximum": INSTALL_MB_MAX, "default": INSTALL_MB_DEFAULT},
            "force": {"type": "boolean", "default": False, "description": "Install despite a base-model mismatch, or replace a file of the same name."},
        }, "required": ["id", "model_type"], "additionalProperties": False},
        "example": {"action": "lora_install", "arguments": {"id": "civitai:1782702", "model_type": "flux"}},
    },
}


def strip_html(text: Any, limit: int = DESCRIPTION_CHARS) -> str:
    text = re.sub(r"<[^>]+>", " ", str(text or ""))
    text = re.sub(r"\s+", " ", text.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"').replace("&#39;", "'")).strip()
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"


def relevance(record: dict[str, Any], query: str) -> int:
    """How well a record matches the query: the whole phrase, then each word, in its name, tags, trigger words and description."""
    text = record.get("_search", "")
    name = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", str(record.get("name", ""))).casefold()  # PixelArt -> pixel art
    phrase = query.casefold().strip()
    words = [word for word in re.split(r"[^\w]+", phrase) if len(word) > 1]
    score = (6 if phrase and phrase in name else 3 if phrase and phrase in text else 0)
    for word in words:  # whole words and their inflections: light matches lights / lighting, not Lightning
        pattern = re.compile(rf"\b{re.escape(word)}(?:s|es|ed|ing|er|ers)?\b")
        score += 3 if pattern.search(name) else 1 if pattern.search(text) else 0
    return score


def compact_record(record: dict[str, Any]) -> dict[str, Any]:
    """A result as shown to the assistant: short description, the first trigger words, no internal search fields."""
    shown = {key: value for key, value in record.items() if not key.startswith("_")}
    words = [str(word)[:60] for word in shown.get("trigger_words") or []]
    if words:
        shown["trigger_words"] = words[:TRIGGER_WORDS_SHOWN]
        if len(words) > TRIGGER_WORDS_SHOWN:
            shown["trigger_words_more"] = len(words) - TRIGGER_WORDS_SHOWN
    if shown.get("description"):
        shown["description"] = strip_html(shown["description"], DESCRIPTION_SHOWN)
    return shown


def hf_family(base_model: str) -> str | None:
    base_model = base_model.lower()
    return next((family for needle, family in HF_BASE_MODELS if needle in base_model), None)


def lora_usage(names: list[str]) -> dict[str, Any]:
    """activated_loras / loras_multipliers for installed files: a HIGH and a LOW noise file make a two-phase pair (high-noise phase, then low-noise phase)."""
    high = [name for name in names if re.search("high", name, re.I) and not re.search("low", name, re.I)]
    low = [name for name in names if re.search("low", name, re.I) and not re.search("high", name, re.I)]
    if len(names) == 2 and len(high) == 1 and len(low) == 1:
        return {"activated_loras": [high[0], low[0]], "loras_multipliers": "1;0 0;1"}
    return {"activated_loras": names[:1], "loras_multipliers": "1.0"}


def is_safetensors(path: Path) -> bool:
    try:
        with open(path, "rb") as handle:
            header = handle.read(9)
        length = struct.unpack("<Q", header[:8])[0]
        return len(header) == 9 and header[8:9] == b"{" and 0 < length < path.stat().st_size
    except (OSError, struct.error):
        return False


class LoraHub:
    """http: the WebSearch of the session (paced, cookie-aware, public-host checked requests); catalog(): WanGP model metadata records;
    lora_dir(model_type): the model's LoRA folder; civitai_key(): the configured Civitai API key or ''."""

    def __init__(self, http, catalog: Callable[[], list[dict]], lora_dir: Callable[[str], str], civitai_key: Callable[[], str], hf_token: Callable[[], str] = lambda: ""):
        self._http, self._catalog, self._lora_dir, self._civitai_key, self._hf_token = http, catalog, lora_dir, civitai_key, hf_token
        self._records: list[dict] | None = None
        self._cache: dict[tuple, tuple[float, list[dict], list[str]]] = {}

    # ---- WanGP side
    def _models(self) -> list[dict]:
        if self._records is None:
            self._records = list(self._catalog())
        return self._records

    def _profile(self, model_type: str) -> dict[str, Any]:
        record = next((record for record in self._models() if record["model_type"] == model_type), None)
        if record is None:  # a model name ("Wan 2.2 text to video"): fine when every model it names shares one LoRA folder
            from shared.deepy.model_names import match_models

            found, complete = match_models(model_type, self._models())
            folders = {}
            for item in found if complete else []:
                try:
                    folders.setdefault(str(self._lora_dir(item["model_type"])), item)
                except Exception:  # a model without LoRAs
                    continue
            if len(folders) == 1:
                record = next(iter(folders.values()))
            else:
                import difflib

                types = {item["model_type"]: item.get("name", "") for item in self._models()}
                close = [item["model_type"] for item in found][:8] or difflib.get_close_matches(model_type, types, n=4, cutoff=0.5)
                hint = f" Candidates: {', '.join(f'{key} ({types[key]})' for key in close)}." if close else ""
                if len(folders) > 1:
                    raise ValueError(f"{model_type} names several WanGP models whose LoRAs are not interchangeable.{hint} Ask the user which one they use, unless the conversation already says (the model of a generation they mention), then pass its model_type.")
                raise ValueError(f"{model_type} is not one WanGP model.{hint} Pass the model_type of the model (wangp_models lookup).")
            model_type = record["model_type"]
        try:
            folder = Path(self._lora_dir(model_type))
        except Exception as error:
            raise ValueError(f"{model_type} ({record.get('name', model_type)}) takes no LoRAs: {error}") from None
        return {"model_type": model_type, "name": record.get("name", model_type), "family": record.get("family", ""), "family_label": record.get("family_label", record.get("family", "")), "folder": folder}

    def _family_labels(self) -> dict[str, str]:
        labels: dict[str, str] = {}
        for record in self._models():
            labels.setdefault(record.get("family", ""), record.get("family_label", record.get("family", "")))
        return labels

    def _target_of(self, base_model: str, source: str) -> tuple[str | None, str | None]:
        """(WanGP family, LoRA folder name or None) a hub base-model name maps to."""
        if source == "civitai":
            match = CIVITAI_BASE_MODELS.get(base_model) or next((value for name, value in CIVITAI_BASE_MODELS.items() if name.lower() == base_model.lower()), None)
            return match if match else (None, None)
        family = hf_family(base_model)
        return family, next((folder for needle, folder in HF_FOLDER_HINTS.get(family, ()) if needle in base_model.lower()), None)

    def civitai_names_for(self, profile: dict[str, Any]) -> list[str]:
        """Civitai base models whose LoRAs fit the profile, its own kind first (I2V names first for an image-to-video model)."""
        names = [name for name, (family, folder) in CIVITAI_BASE_MODELS.items() if family == profile["family"] and folder in (None, profile["folder"].name)]
        i2v = "i2v" in profile["model_type"].casefold() or "image2video" in profile["name"].casefold().replace(" ", "")
        return sorted(names, key=lambda name: ("i2v" in name.casefold()) != i2v)

    def _fits(self, base_model: str, source: str, profile: dict[str, Any] | None) -> tuple[bool | None, str]:
        """(compatible with the profile, or None without a profile; the WanGP families the base model fits as text)."""
        family, folder = self._target_of(base_model, source)
        if family is None:
            return (None if profile is None else False), "no WanGP model uses this base model"
        fits = f"{self._family_labels().get(family, family)} models" + (f" ({folder} LoRAs)" if folder else "")
        if profile is None:
            return None, fits
        return (family == profile["family"] and folder in (None, profile["folder"].name)), fits

    # ---- hub requests
    def _json(self, url: str, host: str, headers: dict[str, str] | None = None) -> Any:
        _final, response = self._http._request(url, headers=headers or None, headers_host=host, accept="application/json")
        return json.loads(response.text)

    def _civitai_headers(self) -> dict[str, str]:
        key = (self._civitai_key() or "").strip()
        return {"Authorization": f"Bearer {key}"} if key else {}

    def _hf_headers(self) -> dict[str, str]:
        token = (self._hf_token() or "").strip()
        return {"Authorization": f"Bearer {token}"} if token else {}

    def _civitai_item(self, item: dict, names: list[str], profile: dict[str, Any] | None, nsfw: bool) -> dict[str, Any] | None:
        versions = [version for version in item.get("modelVersions", []) if version.get("files")]
        version = next((version for name in names for version in versions if version.get("baseModel") == name), versions[0] if versions else None)
        if version is None:
            return None
        record = self._civitai_record(item, version, profile, nsfw)
        if names and version.get("baseModel") != names[0]:
            record["_secondary"] = True  # fits the model, but made for a sibling (an I2V LoRA for a T2V model)
        siblings = [other for other in versions if other is not version and other.get("baseModel") == version.get("baseModel")]
        if siblings:  # Wan 2.2 LoRAs often come as a HIGH and a LOW noise version; other releases of one LoRA too
            record["other_versions"] = [{"id": f"civitai:{other['id']}", "name": str(other.get("name", ""))[:60]} for other in siblings[:4]]
        return record

    def _civitai_record(self, item: dict, version: dict, profile: dict[str, Any] | None, nsfw: bool) -> dict[str, Any]:
        files = [file for file in version.get("files", []) if file.get("type", "Model") == "Model"] or version.get("files", [])
        file = next((file for file in files if file.get("primary")), files[0] if files else {})
        compatible, fits = self._fits(str(version.get("baseModel", "")), "civitai", profile)
        preview = next((image["url"] for image in version.get("images", []) if image.get("type", "image") == "image" and (nsfw or int(image.get("nsfwLevel") or 1) <= 1)), None)
        stats = item.get("stats") or version.get("stats") or {}
        record = {"id": f"civitai:{version['id']}", "name": item.get("name") or (item.get("model") or {}).get("name", ""), "version": str(version.get("name", "")), "base_model": version.get("baseModel", ""), "fits": fits,
                  "trigger_words": list(version.get("trainedWords") or []), "downloads": stats.get("downloadCount"), "likes": stats.get("thumbsUpCount"), "creator": (item.get("creator") or {}).get("username"),
                  "file": file.get("name"), "size_mb": round(float(file.get("sizeKB") or 0) / 1024, 1) or None, "page_url": f"https://civitai.com/models/{item.get('id') or version.get('modelId')}?modelVersionId={version['id']}", "preview_url": preview,
                  "description": strip_html(item.get("description")) or None, "nsfw": True if item.get("nsfw") or (item.get("model") or {}).get("nsfw") else None,
                  "_search": " ".join([str(item.get("name") or ""), " ".join(map(str, item.get("tags") or [])), " ".join(map(str, version.get("trainedWords") or [])), strip_html(item.get("description"), 2000)]).casefold()}
        if compatible is not None:
            record["compatible"] = compatible
        return {key: value for key, value in record.items() if value not in (None, "", [])}

    def _civitai_search(self, query: str, names: list[str], sort: str, nsfw: bool, profile: dict[str, Any] | None) -> list[dict]:
        """Civitai's search index misses most queries filtered by base model, so three requests: the filtered search, the base models' 100 most
        popular LoRAs matched against the query here, and the unfiltered search (other base models, counted or listed on request)."""
        base = [("types", "LORA"), ("types", "DoRA"), ("types", "LoCon"), ("sort", CIVITAI_SORT[sort]), ("nsfw", "true" if nsfw else "false")]
        requests_made = [base + [("limit", "50"), ("query", query)] + [("baseModels", name) for name in names]]
        if names:
            requests_made += [base + [("limit", "100")] + [("baseModels", name) for name in names], base + [("limit", "50"), ("query", query)]]
        records, seen = [], set()
        for index, params in enumerate(requests_made):
            for item in self._json(f"{CIVITAI_API}/models?{urlencode(params, doseq=True)}", "civitai.com", self._civitai_headers()).get("items", []):
                record = self._civitai_item(item, names, profile, nsfw)
                if record is None or record["id"] in seen or (index == 1 and relevance(record, query) == 0):
                    continue
                seen.add(record["id"])
                records.append(record)
        return records

    def _hf_record(self, item: dict, profile: dict[str, Any] | None) -> dict[str, Any] | None:
        files = [sibling["rfilename"] for sibling in item.get("siblings", []) if sibling.get("rfilename", "").endswith((".safetensors", ".sft"))]
        if not files:
            return None
        card = item.get("cardData") or {}
        bases = card.get("base_model") or [tag.split(":", 1)[1] for tag in item.get("tags", []) if tag.startswith("base_model:") and ":adapter:" not in tag]
        base_model = (bases if isinstance(bases, list) else [bases])[0] if bases else ""
        compatible, fits = self._fits(base_model, "huggingface", profile)
        trigger = card.get("instance_prompt")
        repo = item.get("id") or item.get("modelId")
        record = {"id": f"huggingface:{repo}", "name": repo, "base_model": base_model or None, "fits": fits, "trigger_words": [trigger] if trigger else [], "downloads": item.get("downloads"), "likes": item.get("likes"), "creator": item.get("author") or repo.split("/")[0],
                  "file": files[0] if len(files) == 1 else None, "files": files if len(files) > 1 else None, "page_url": f"https://huggingface.co/{repo}", "license": card.get("license_name") or card.get("license"),
                  "_search": " ".join([repo.replace("-", " ").replace("_", " "), " ".join(map(str, card.get("tags") or [])), str(trigger or "")]).casefold()}
        if compatible is not None:
            record["compatible"] = compatible
        return {key: value for key, value in record.items() if value not in (None, "", [])}

    def _hf_search(self, query: str, profile: dict[str, Any] | None) -> list[dict]:
        hint = HF_SEARCH_HINT.get(profile["family"], profile["family_label"].split()[0].lower()) if profile else ""
        params = [("search", f"{hint} {query}".strip()), ("filter", "lora"), ("sort", "downloads"), ("direction", "-1"), ("limit", "30"), ("expand[]", "cardData"), ("expand[]", "siblings"), ("expand[]", "downloads"), ("expand[]", "likes")]
        items = self._json(f"{HF_API}/models?{urlencode(params, doseq=True)}", "huggingface.co", self._hf_headers())
        return [record for item in items if (record := self._hf_record(item, profile)) is not None]

    # ---- actions
    def _ranked(self, query: str, source: str, sort: str, nsfw: bool, profile: dict[str, Any] | None) -> tuple[list[dict], list[str]]:
        """Every candidate of a search, best first (compatible, then relevance to the query, then popularity), cached for paging."""
        key = (query.casefold(), source, sort, nsfw, profile["model_type"] if profile else "")
        cached = self._cache.get(key)
        if cached is not None and time.monotonic() - cached[0] <= CACHE_TTL_S:
            return cached[1], cached[2]
        names = self.civitai_names_for(profile) if profile else []
        results, notes = [], []
        if source in ("both", "civitai"):
            if profile and not names:
                notes.append(f"Civitai has no base-model category for {profile['family_label']}: Civitai was not searched.")
            else:
                results += self._civitai_search(query, names, sort, nsfw, profile)
        if source in ("both", "huggingface"):
            results += self._hf_search(query, profile)
        metric = {"downloads": "downloads", "rating": "likes", "newest": ""}[sort]
        for record in results:
            record["_relevance"] = relevance(record, query)
        results.sort(key=lambda record: (not record.get("compatible", profile is None), -record["_relevance"], bool(record.get("_secondary")), -(record.get(metric) or 0) if metric else 0))
        self._cache = {k: v for k, v in self._cache.items() if time.monotonic() - v[0] <= CACHE_TTL_S}
        self._cache[key] = (time.monotonic(), results, notes)
        return results, notes

    def search(self, arguments: dict[str, Any]) -> dict[str, Any]:
        from shared.deepy.web_search import paged

        query, source = str(arguments["query"]).strip(), arguments.get("source") or "both"
        page_size, page = int(arguments.get("max_results") or PAGE_SIZE), max(1, int(arguments.get("page") or 1))
        sort, nsfw = arguments.get("sort") or "downloads", bool(arguments.get("nsfw"))
        profile = self._profile(str(arguments["model_type"])) if arguments.get("model_type") else None
        results, notes = self._ranked(query, source, sort, nsfw, profile)
        shown = results if profile is None or arguments.get("include_incompatible") else [record for record in results if record.get("compatible")]
        hidden = [] if shown is results else [record for record in results if not record.get("compatible")]
        response: dict[str, Any] = {"status": "done", "query": query, **paged([compact_record(record) for record in shown], page, page_size, {"action": "lora_search", "arguments": {**{k: v for k, v in arguments.items() if k != "page"}, "page": page + 1}})}
        rest = shown[(page - 1) * page_size:]
        if response.get("next_page") and rest and max(record["_relevance"] for record in rest) <= 1:  # results are best first
            response.pop("next_page")
            response["paging"] = "The remaining results only mention the query in passing (not in their name, tags or trigger words): further pages will not hold a better match."
        if profile:
            response["for_model"] = {"model_type": profile["model_type"], "name": profile["name"], "civitai_base_models": self.civitai_names_for(profile)}
        if hidden:
            counts: dict[str, int] = {}
            for record in hidden:
                counts[record.get("base_model") or "unknown"] = counts.get(record.get("base_model") or "unknown", 0) + 1
            response["other_base_models"] = dict(sorted(counts.items(), key=lambda item: -item[1])[:8])
            response["note"] = f"{len(hidden)} more LoRAs match the query but are made for other base models (not usable with {profile['name']}); include_incompatible=true lists them."
        if notes:
            response["note"] = " ".join(notes + ([response["note"]] if "note" in response else []))
        if not shown:
            response["hint"] = ("No compatible LoRA for this query: try one other wording (a shorter theme word, a synonym). If that finds nothing either, none is published for this model: "
                                "tell the user, suggest prompt words for the look or a model whose LoRAs have it; do not query Civitai or Hugging Face yourself (this search already covers both).") if page == 1 else "No more results for this search."
        else:
            response["next"] = "lora_install with the chosen id and the model_type; then put a trigger word in the prompt and the file name in activated_loras."
        return response

    def _resolve(self, reference: str, profile: dict[str, Any]) -> dict[str, Any]:
        reference = reference.strip()
        parts = urlsplit(reference) if "://" in reference else None
        if reference.startswith("civitai:") or reference.isdigit() or (parts and parts.hostname and parts.hostname.endswith("civitai.com")):
            version_id = model_id = None
            if parts is not None:
                query = parse_qs(parts.query)
                if match := re.search(r"/(?:api/download/models|model-versions)/(\d+)", parts.path):
                    version_id = int(match.group(1))
                elif match := re.search(r"/models/(\d+)", parts.path):
                    model_id, version_id = int(match.group(1)), int(query["modelVersionId"][0]) if query.get("modelVersionId") else None
            else:
                version_id = int(reference.removeprefix("civitai:"))
            if version_id is None:
                if model_id is None:
                    raise ValueError(f"Cannot read a Civitai model or version id from {reference}.")
                model = self._json(f"{CIVITAI_API}/models/{model_id}", "civitai.com", self._civitai_headers())
                names = self.civitai_names_for(profile)
                version = next((version for version in model["modelVersions"] if version.get("baseModel") in names), model["modelVersions"][0])
                record = self._civitai_record(model, version, profile, True)
                version_id = version["id"]
            else:
                version = self._json(f"{CIVITAI_API}/model-versions/{version_id}", "civitai.com", self._civitai_headers())
                record = self._civitai_record({"id": version.get("modelId"), "name": (version.get("model") or {}).get("name"), "nsfw": (version.get("model") or {}).get("nsfw"), "stats": version.get("stats")}, version, profile, True)
            files = [file for file in version.get("files", []) if file.get("type", "Model") == "Model"] or version.get("files", [])
            file = next((file for file in files if file.get("primary")), files[0] if files else None)
            if file is None:
                raise ValueError(f"Civitai version {version_id} has no downloadable model file.")
            return {**record, "source": "civitai", "host": "civitai.com", "download_url": file.get("downloadUrl") or f"{CIVITAI_API.rsplit('/v1', 1)[0]}/download/models/{version_id}", "file": file["name"], "headers": self._civitai_headers()}
        repo, file = reference.removeprefix("huggingface:").removeprefix("hf:"), ""
        if parts is not None:
            if not (parts.hostname or "").endswith("huggingface.co"):
                raise ValueError(f"Unsupported LoRA source: {reference}. Use a Civitai or Hugging Face id or url.")
            segments = parts.path.strip("/").split("/")
            repo = "/".join(segments[:2])
            if len(segments) > 4 and segments[2] in ("resolve", "blob"):
                file = "/".join(segments[4:])
        elif "#" in repo:
            repo, file = repo.split("#", 1)
        if repo.count("/") != 1:
            raise ValueError(f"A Hugging Face LoRA is named owner/repo[#file]: {reference}")
        item = self._json(f"{HF_API}/models/{repo}", "huggingface.co", self._hf_headers())
        record = self._hf_record(item, profile) or {}
        files = record.get("files") or ([record["file"]] if record.get("file") else [])
        if not files:
            raise ValueError(f"{repo} has no safetensors file.")
        if not file:
            plain = [name for name in files if not re.search(r"[-_]comfy", Path(name).stem, re.I)] or files
            if len(plain) > 1:
                raise ValueError(f"{repo} holds several LoRA files: {', '.join(files)}. Install one with id='huggingface:{repo}#<file>'.")
            file = plain[0]
        elif file not in files:
            raise ValueError(f"{repo} has no file {file}; its LoRA files: {', '.join(files)}.")
        return {**record, "source": "huggingface", "host": "huggingface.co", "download_url": f"https://huggingface.co/{repo}/resolve/main/{quote(file)}", "file": Path(file).name, "headers": self._hf_headers()}

    def install(self, arguments: dict[str, Any]) -> dict[str, Any]:
        profile = self._profile(str(arguments["model_type"]))
        force, limit = bool(arguments.get("force")), int(float(arguments.get("max_mb") or INSTALL_MB_DEFAULT) * (1 << 20))
        info = self._resolve(str(arguments["id"]), profile)
        if info.get("compatible") is False and not force:
            raise ValueError(f"{info['name']} is made for {info.get('base_model') or 'another base model'} ({info['fits']}), not for {profile['name']} ({profile['family_label']}). Pick a LoRA for that model, use a model it fits, or pass force=true to install it anyway.")
        name = str(arguments.get("name") or info["file"])
        archive = name.lower().endswith(".zip")
        if not archive and not name.lower().endswith((".safetensors", ".sft")):
            name += ".safetensors"
        target = profile["folder"] / name
        common = {"model_type": profile["model_type"], "folder": str(profile["folder"]), "base_model": info.get("base_model"), "trigger_words": info.get("trigger_words", []), "source": info.get("page_url")}
        if not archive and target.exists() and not force:
            return {"status": "exists", "file": name, **common, "usage": lora_usage([name]), "bytes": target.stat().st_size, "next": "The file is already installed: use it as shown in usage, or pass force=true to replace it."}
        hub_dir = profile["folder"] / ".hub"
        hub_dir.mkdir(parents=True, exist_ok=True)
        sign_in = "Civitai asks for a sign-in to download: create an API key at civitai.com/user/account (API Keys) and paste it in Configuration > Deepy > Civitai API Key, or import the browser's civitai.com cookies with action cookies. Nothing was installed."
        try:
            path, written, content_type, final_url = self._http._download_stream(info["download_url"], hub_dir, name, limit, headers=info["headers"], headers_host=info["host"], exact_name=True)
        except ValueError as error:
            if info["source"] == "civitai" and ("civitai.com/login" in str(error) or "HTTP 401" in str(error)):
                raise ValueError(sign_in) from None
            raise
        try:
            if zipfile.is_zipfile(path):
                names = self._unpack(path, profile["folder"], force, limit)
            elif is_safetensors(path):
                if target.exists():
                    target.unlink()
                path.replace(target)
                names = [name]
            else:
                host = (urlsplit(final_url).hostname or "").lower()
                if info["source"] == "civitai" and (host.endswith("civitai.com") and "/login" in final_url or "html" in content_type):
                    raise ValueError(sign_in)
                raise ValueError(f"The download is not a safetensors file or a zip of them ({content_type.split(';')[0] or 'unknown type'}, {written} bytes): nothing was installed.")
        finally:
            path.unlink(missing_ok=True)
        for file in names:
            (hub_dir / f"{Path(file).stem}.json").write_text(json.dumps({key: info.get(key) for key in ("id", "name", "version", "base_model", "trigger_words", "page_url", "creator", "license")} | {"model_type": profile["model_type"], "bytes": (profile["folder"] / file).stat().st_size, "installed_at": time.strftime("%Y-%m-%dT%H:%M:%S")}, ensure_ascii=False, indent=1), encoding="utf-8")
        usage = lora_usage(names)
        response = {"status": "done", "file": names[0] if len(names) == 1 else None, "files": names if len(names) > 1 else None, **common, "usage": usage, "bytes": written}
        response = {key: value for key, value in response.items() if value is not None}
        if usage["loras_multipliers"] == "1;0 0;1":
            response["next"] = f"Two-phase pair: activated_loras {usage['activated_loras']} with loras_multipliers '1;0 0;1' (HIGH noise file on the high-noise phase, LOW noise file on the low-noise phase)"
        elif len(names) > 1:
            response["next"] = f"The zip held {len(names)} LoRAs (all installed); usage shows the first: activate the one the user needs, loras_multipliers '1.0'"
        else:
            response["next"] = f"Use it with activated_loras {usage['activated_loras']} and loras_multipliers {usage['loras_multipliers']!r} in the generation settings"
        response["next"] += (f", with one of its trigger words in the prompt: {', '.join(info['trigger_words'])}." if info.get("trigger_words") else "; it has no declared trigger word.")
        return response

    @staticmethod
    def _unpack(archive: Path, folder: Path, force: bool, limit: int) -> list[str]:
        """Install the safetensors files of a downloaded zip into folder (flat: no folder of the archive is created); returns their names."""
        with zipfile.ZipFile(archive) as bundle:
            members = [member for member in bundle.infolist() if not member.is_dir() and member.filename.lower().endswith((".safetensors", ".sft"))]
            if not members:
                raise ValueError(f"The zip holds no safetensors file ({', '.join(member.filename for member in bundle.infolist()[:8])}): nothing was installed.")
            if sum(member.file_size for member in members) > 4 * limit:
                raise ValueError(f"The zip unpacks to more than {4 * limit >> 20} MB: nothing was installed. Raise max_mb if the user wants it.")
            names = []
            for member in members:
                name = member.filename.replace(chr(92), "/").rsplit("/", 1)[-1]
                target = folder / name
                if not target.exists() or force:
                    partial = target.with_name(name + ".part")
                    try:
                        with bundle.open(member) as source, open(partial, "wb") as handle:
                            shutil.copyfileobj(source, handle, 4 << 20)
                        if not is_safetensors(partial):
                            raise ValueError(f"{member.filename} in the zip is not a safetensors file: it was not installed.")
                        partial.replace(target)
                    finally:
                        partial.unlink(missing_ok=True)
                names.append(name)
        return names
