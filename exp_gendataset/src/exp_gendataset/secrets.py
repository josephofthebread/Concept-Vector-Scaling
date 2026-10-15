from __future__ import annotations

import os
from pathlib import Path


def _read_file(path: Path) -> str | None:
    if not path.exists():
        return None
    value = path.read_text(encoding="utf-8").strip()
    return value or None


def openrouter_api_key() -> str | None:
    env = os.environ.get("OPENROUTER_API_KEY")
    if env and env.strip():
        return env.strip()
    root = Path(__file__).resolve().parents[2]
    return _read_file(root / "secrets" / "openrouter_api_key.txt")


def resolve_openrouter_proxy(explicit: str | None = None) -> str | None:
    if explicit is not None:
        value = explicit.strip()
        return value or None
    for key in (
        "OPENROUTER_PROXY_URL",
        "HTTPS_PROXY",
        "https_proxy",
        "HTTP_PROXY",
        "http_proxy",
        "ALL_PROXY",
        "all_proxy",
    ):
        value = os.environ.get(key)
        if value and value.strip():
            return value.strip()
    root = Path(__file__).resolve().parents[2]
    return _read_file(root / "secrets" / "openrouter_proxy_url.txt")
