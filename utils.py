import math
import os
import tempfile
from typing import Any
from urllib.parse import urlparse

import requests

from config import MAX_DOWNLOAD_BYTES


def is_url(value: str) -> bool:
    if not value:
        return False
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def directus_auth_headers(url: str | None = None) -> dict[str, str]:
    headers: dict[str, str] = {}

    token = os.getenv("DIRECTUS_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    return headers


def download_temp_file(url: str, suffix: str, timeout: tuple[int, int] = (10, 30)):
    headers = directus_auth_headers(url)

    response = requests.get(
        url,
        headers=headers,
        timeout=timeout,
        stream=True,
        allow_redirects=True,
    )

    response.raise_for_status()

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    total = 0

    for chunk in response.iter_content(chunk_size=8192):
        if not chunk:
            continue

        total += len(chunk)
        if total > MAX_DOWNLOAD_BYTES:
            tmp.close()
            os.remove(tmp.name)
            raise ValueError("Downloaded file exceeds size limit")

        tmp.write(chunk)

    tmp.close()
    return tmp.name


def remove_temp_file(path: str) -> None:
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def clamp01(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(numeric) or math.isinf(numeric):
        return default
    return max(0.0, min(numeric, 1.0))


def safe_number(value: Any, digits: int = 4) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return round(numeric, digits)


def clean_warning_codes(values: list[Any] | None) -> list[str]:
    seen: set[str] = set()
    cleaned: list[str] = []
    for value in values or []:
        if not value:
            continue
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        cleaned.append(text)
    return cleaned


def sanitize_text(value: Any, fallback: str | None = None, max_len: int | None = None) -> str | None:
    if value is None:
        return fallback
    text = str(value).strip()
    if not text or text.lower() == "undefined":
        return fallback
    if max_len is not None and len(text) > max_len:
        return text[:max_len]
    return text


def sanitize_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): sanitized
            for key, raw in value.items()
            if (sanitized := sanitize_payload(raw)) is not None
        }
    if isinstance(value, list):
        return [sanitized for raw in value if (sanitized := sanitize_payload(raw)) is not None]
    if isinstance(value, tuple):
        return [sanitized for raw in value if (sanitized := sanitize_payload(raw)) is not None]
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return safe_number(value, digits=6)
    if isinstance(value, str):
        return sanitize_text(value)
    return value
