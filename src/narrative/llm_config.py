"""LLM provider configuration for the narrative.

The key is read from the environment first, then from a gitignored `secrets.local.json`
at the repo root. It is never hard-coded in committed source (see `.gitignore`).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]

# OpenRouter is OpenAI-compatible. A free instruction-following model is the default;
# override with the OPENROUTER_MODEL env var if a stronger one is available on the key.
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# gpt-oss-20b returns clean four-sentence prose; the Nemotron models leak their
# chain-of-thought ("We need to craft...") so they are not used. Gemma is a rate-limited
# fallback. Verified against the live key on 2026-07-21.
DEFAULT_MODEL = "openai/gpt-oss-20b:free"
_FALLBACK_MODELS = (
    "openai/gpt-oss-20b:free",
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
)


def openrouter_key() -> str | None:
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key.strip()
    f = _ROOT / "secrets.local.json"
    if f.exists():
        try:
            return (json.loads(f.read_text(encoding="utf-8")).get("openrouter_api_key") or "").strip() or None
        except Exception:
            return None
    return None


def models() -> tuple[str, ...]:
    override = os.environ.get("OPENROUTER_MODEL")
    if override:
        return (override, *(_m for _m in _FALLBACK_MODELS if _m != override))
    return _FALLBACK_MODELS


def is_configured() -> bool:
    return openrouter_key() is not None
