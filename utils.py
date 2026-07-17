import math
import os
import tempfile
from typing import Any
from urllib.parse import urljoin, urlsplit

import requests

from config import MAX_DOWNLOAD_BYTES


_DEFAULT_PORTS = {"http": 80, "https": 443}
_MAX_REDIRECTS = 3
_MAX_SAFE_DIGITS = 12


def _contains_control_or_whitespace(value: str) -> bool:
    return any(ch.isspace() or ord(ch) < 32 or ord(ch) == 127 for ch in value)


def _parse_http_origin(value: Any, *, label: str) -> tuple[str, str, int]:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a valid http(s) URL")
    text = value.strip()
    if not text or _contains_control_or_whitespace(text):
        raise ValueError(f"{label} must be a valid http(s) URL")
    parsed = urlsplit(text)
    scheme = parsed.scheme.lower()
    if scheme not in _DEFAULT_PORTS:
        raise ValueError(f"{label} must be a valid http(s) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(f"{label} must be a valid http(s) URL")
    host = parsed.hostname
    if not host:
        raise ValueError(f"{label} must be a valid http(s) URL")
    try:
        port = parsed.port
    except ValueError:
        raise ValueError(f"{label} must be a valid http(s) URL") from None
    if port is None:
        port = _DEFAULT_PORTS[scheme]
    return scheme, host.lower(), port


def _directus_origin() -> tuple[str, str, int]:
    directus_url = os.getenv("DIRECTUS_URL", "")
    return _parse_http_origin(directus_url, label="DIRECTUS_URL")


def _same_origin(left: tuple[str, str, int], right: tuple[str, str, int]) -> bool:
    return left == right


def _validate_timeout(timeout: tuple[int, int]) -> tuple[float, float]:
    if not isinstance(timeout, tuple) or len(timeout) != 2:
        raise ValueError("timeout must be a 2-item tuple of positive finite numbers")
    values = []
    for value in timeout:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("timeout must be a 2-item tuple of positive finite numbers")
        numeric = float(value)
        if not math.isfinite(numeric) or numeric <= 0:
            raise ValueError("timeout must be a 2-item tuple of positive finite numbers")
        values.append(numeric)
    return values[0], values[1]


def _validate_suffix(suffix: str) -> str:
    if not isinstance(suffix, str):
        raise ValueError("suffix must be a string")
    if "\x00" in suffix or ".." in suffix:
        raise ValueError("suffix contains invalid path components")
    for separator in {os.sep, os.altsep}:
        if separator and separator in suffix:
            raise ValueError("suffix contains invalid path components")
    return suffix


def _validate_digits(digits: int) -> int:
    if isinstance(digits, bool) or not isinstance(digits, int):
        raise ValueError("digits must be an integer between 0 and 12")
    if digits < 0 or digits > _MAX_SAFE_DIGITS:
        raise ValueError("digits must be an integer between 0 and 12")
    return digits


def is_url(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        _parse_http_origin(value, label="URL")
    except ValueError:
        return False
    return True


def directus_auth_headers(url: str | None = None) -> dict[str, str]:
    token = os.getenv("DIRECTUS_TOKEN", "").strip()
    if not token or url is None:
        return {}
    try:
        destination_origin = _parse_http_origin(url, label="url")
    except ValueError:
        return {}
    directus_origin = _directus_origin()
    if not _same_origin(destination_origin, directus_origin):
        return {}
    return {"Authorization": f"Bearer {token}"}


def download_temp_file(url: str, suffix: str, timeout: tuple[int, int] = (10, 30)):
    validated_url = _parse_http_origin(url, label="url")
    directus_origin = _directus_origin()
    if not _same_origin(validated_url, directus_origin):
        raise ValueError("download URL must match DIRECTUS_URL origin")

    validated_suffix = _validate_suffix(suffix)
    validated_timeout = _validate_timeout(timeout)
    current_url = url.strip()
    redirects = 0

    while True:
        response = None
        temp_path = None
        raw_fd = None
        try:
            response = requests.get(
                current_url,
                headers=directus_auth_headers(current_url),
                timeout=validated_timeout,
                stream=True,
                allow_redirects=False,
            )

            if 300 <= getattr(response, "status_code", 0) < 400:
                location = response.headers.get("Location")
                if not location:
                    raise ValueError("redirect response missing Location header")
                next_url = urljoin(current_url, location)
                next_origin = _parse_http_origin(next_url, label="redirect URL")
                if not _same_origin(next_origin, directus_origin):
                    raise ValueError("redirect target must match DIRECTUS_URL origin")
                redirects += 1
                if redirects > _MAX_REDIRECTS:
                    raise ValueError("too many redirects")
                current_url = next_url
                continue

            response.raise_for_status()
            content_length = response.headers.get("Content-Length")
            if content_length is not None:
                try:
                    parsed_content_length = int(content_length)
                except (TypeError, ValueError):
                    parsed_content_length = None
                if parsed_content_length is not None and parsed_content_length > MAX_DOWNLOAD_BYTES:
                    raise ValueError("Downloaded file exceeds size limit")

            fd, temp_path = tempfile.mkstemp(suffix=validated_suffix)
            raw_fd = fd
            total = 0
            try:
                try:
                    handle = os.fdopen(fd, "wb")
                except Exception:
                    raise
                else:
                    raw_fd = None
                with handle:
                    for chunk in response.iter_content(chunk_size=8192):
                        if not chunk:
                            continue
                        total += len(chunk)
                        if total > MAX_DOWNLOAD_BYTES:
                            raise ValueError("Downloaded file exceeds size limit")
                        handle.write(chunk)
            finally:
                if raw_fd is not None:
                    try:
                        os.close(raw_fd)
                    except OSError:
                        pass

            if total <= 0:
                raise ValueError("Downloaded file is empty")
            return temp_path
        except Exception:
            if temp_path:
                remove_temp_file(temp_path)
            raise
        finally:
            if response is not None:
                response.close()


def remove_temp_file(path: str) -> None:
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except (FileNotFoundError, IsADirectoryError, PermissionError, OSError):
        pass


def clamp01(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    if isinstance(value, bool):
        return default
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(numeric) or math.isinf(numeric):
        return default
    return max(0.0, min(numeric, 1.0))


def safe_number(value: Any, digits: int = 4) -> float | None:
    _validate_digits(digits)
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return round(numeric, digits)


def clean_warning_codes(values: list[Any] | None) -> list[str]:
    if values is None:
        items: list[Any] = []
    elif isinstance(values, str):
        items = [values]
    else:
        try:
            items = list(values)
        except TypeError:
            items = [values]
    seen: set[str] = set()
    cleaned: list[str] = []
    for value in items:
        if not value:
            continue
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        cleaned.append(text)
    return cleaned


def sanitize_text(value: Any, fallback: str | None = None, max_len: int | None = None) -> str | None:
    if max_len is not None:
        if isinstance(max_len, bool) or not isinstance(max_len, int) or max_len < 0:
            raise ValueError("max_len must be None or a non-negative integer")
    if value is None:
        return fallback[:max_len] if isinstance(fallback, str) and max_len is not None else fallback
    text = str(value).strip()
    if not text or text.lower() == "undefined":
        if fallback is None:
            return None
        return fallback[:max_len] if isinstance(fallback, str) and max_len is not None else fallback
    if max_len is not None and len(text) > max_len:
        return text[:max_len]
    return text


def sanitize_payload(value: Any) -> Any:
    def _sanitize(current: Any, seen: set[int]) -> Any:
        if current is None:
            return None
        if isinstance(current, bool):
            return current
        if isinstance(current, int):
            return current
        if isinstance(current, float):
            return safe_number(current, digits=6)
        if isinstance(current, str):
            return sanitize_text(current)
        if isinstance(current, dict):
            object_id = id(current)
            if object_id in seen:
                return None
            seen.add(object_id)
            cleaned: dict[str, Any] = {}
            for key, raw in current.items():
                sanitized = _sanitize(raw, seen)
                if sanitized is None:
                    continue
                cleaned[key if isinstance(key, str) else str(key)] = sanitized
            seen.remove(object_id)
            return cleaned
        if isinstance(current, list):
            object_id = id(current)
            if object_id in seen:
                return None
            seen.add(object_id)
            cleaned_list = []
            for raw in current:
                sanitized = _sanitize(raw, seen)
                if sanitized is not None:
                    cleaned_list.append(sanitized)
            seen.remove(object_id)
            return cleaned_list
        if isinstance(current, tuple):
            object_id = id(current)
            if object_id in seen:
                return None
            seen.add(object_id)
            cleaned_tuple = []
            for raw in current:
                sanitized = _sanitize(raw, seen)
                if sanitized is not None:
                    cleaned_tuple.append(sanitized)
            seen.remove(object_id)
            return cleaned_tuple
        return None

    return _sanitize(value, set())
