from __future__ import annotations

from datetime import datetime, timezone
import math
import os
import threading
from typing import Any
from urllib.parse import quote, urlsplit

import requests
from requests import HTTPError

from logger import get_logger

try:
    import httpx
except Exception:  # pragma: no cover
    httpx = None


logger = get_logger()

_ALLOWED_SCHEMES = {"http", "https"}
_ALLOWED_METHODS = {"GET", "POST", "PATCH", "PUT", "DELETE"}
_MAX_SCHEMA_CACHE_COLLECTIONS = 128
_MAX_LOG_ERROR_ITEMS = 3
_MAX_LOG_TEXT_LENGTH = 240


def _relation_id(value: Any) -> Any:
    if isinstance(value, dict):
        return value.get("id")
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _normalize_base_url(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise TypeError("DIRECTUS_URL must be a string")

    normalized = value.strip().rstrip("/")
    if not normalized:
        return ""

    parsed = urlsplit(normalized)
    if parsed.scheme.lower() not in _ALLOWED_SCHEMES or not parsed.netloc:
        raise ValueError("DIRECTUS_URL must be an absolute http(s) URL")
    if parsed.username or parsed.password:
        raise ValueError("DIRECTUS_URL must not contain embedded credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("DIRECTUS_URL must not contain a query or fragment")
    return normalized


def _normalize_token(value: Any, *, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")

    token = value.strip()
    if not token:
        return None
    if any(character in token for character in ("\r", "\n", "\x00")):
        raise ValueError(f"{field_name} contains invalid control characters")
    return token


def _parse_timeout(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("DIRECTUS_TIMEOUT must be a positive number")

    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("DIRECTUS_TIMEOUT cannot be blank")
        try:
            timeout = float(text)
        except ValueError as exc:
            raise ValueError("DIRECTUS_TIMEOUT must be a positive number") from exc
    elif type(value) in {int, float}:
        timeout = float(value)
    else:
        raise TypeError("DIRECTUS_TIMEOUT must be a positive number")

    if not math.isfinite(timeout) or timeout <= 0.0:
        raise ValueError("DIRECTUS_TIMEOUT must be a positive finite number")
    return timeout


def _path_segment(value: Any, *, field_name: str) -> str:
    if value is None or isinstance(value, bool):
        raise ValueError(f"{field_name} is required")

    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} cannot be blank")
    if any(character in text for character in ("\r", "\n", "\x00", "/", "\\")):
        raise ValueError(f"{field_name} contains invalid path characters")
    return quote(text, safe="-._~")


def _fields_param(fields: list[str] | None) -> str | None:
    if fields is None:
        return None
    if not isinstance(fields, list):
        raise TypeError("fields must be a list of strings")

    normalized: list[str] = []
    for field in fields:
        if not isinstance(field, str):
            raise TypeError("fields must contain strings only")
        value = field.strip()
        if not value:
            raise ValueError("fields cannot contain blank values")
        if any(character in value for character in ("\r", "\n", "\x00")):
            raise ValueError("field contains invalid control characters")
        normalized.append(value)
    return ",".join(normalized)


def _json_safe_copy(
    value: Any,
    *,
    path: str = "payload",
    seen: set[int] | None = None,
    depth: int = 0,
) -> Any:
    if depth > 32:
        raise ValueError(f"{path} exceeds the maximum nesting depth")

    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains NaN or Infinity")
        return value

    if seen is None:
        seen = set()

    if isinstance(value, dict):
        identity = id(value)
        if identity in seen:
            raise ValueError(f"{path} contains a cyclic dictionary")
        seen.add(identity)
        try:
            result: dict[str, Any] = {}
            for key, item in value.items():
                if not isinstance(key, str) or not key:
                    raise TypeError(f"{path} keys must be non-empty strings")
                result[key] = _json_safe_copy(
                    item,
                    path=f"{path}.{key}",
                    seen=seen,
                    depth=depth + 1,
                )
            return result
        finally:
            seen.remove(identity)

    if isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in seen:
            raise ValueError(f"{path} contains a cyclic sequence")
        seen.add(identity)
        try:
            return [
                _json_safe_copy(
                    item,
                    path=f"{path}[{index}]",
                    seen=seen,
                    depth=depth + 1,
                )
                for index, item in enumerate(value)
            ]
        finally:
            seen.remove(identity)

    raise TypeError(f"{path} contains unsupported type {type(value).__name__}")


def _safe_log_text(value: Any) -> str:
    text = str(value or "").replace("\r", " ").replace("\n", " ").strip()
    if len(text) > _MAX_LOG_TEXT_LENGTH:
        return text[:_MAX_LOG_TEXT_LENGTH] + "..."
    return text


def _response_summary(response: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "status_code": getattr(response, "status_code", None),
    }
    try:
        body = response.json()
    except Exception:
        text = ""
        try:
            text = response.text or ""
        except Exception:
            pass
        summary["body_type"] = "text"
        summary["body_length"] = len(text)
        return summary

    if isinstance(body, dict):
        errors = body.get("errors")
        if isinstance(errors, list):
            parsed_errors: list[dict[str, Any]] = []
            for raw in errors[:_MAX_LOG_ERROR_ITEMS]:
                if not isinstance(raw, dict):
                    continue
                item: dict[str, Any] = {}
                extensions = raw.get("extensions")
                if isinstance(extensions, dict):
                    code = extensions.get("code")
                    if code is not None:
                        item["code"] = _safe_log_text(code)
                if item:
                    parsed_errors.append(item)
            summary["error_count"] = len(errors)
            summary["errors"] = parsed_errors
        else:
            summary["body_keys"] = sorted(str(key) for key in body.keys())[:20]
        return summary

    if isinstance(body, list):
        summary["body_type"] = "list"
        summary["body_size"] = len(body)
    else:
        summary["body_type"] = type(body).__name__
    return summary


def _is_unique_conflict(exc: Exception, *, field_name: str) -> bool:
    if not isinstance(exc, HTTPError):
        return False
    response = exc.response
    if response is None or response.status_code not in {400, 409}:
        return False

    body = ""
    try:
        parsed = response.json()
        if isinstance(parsed, dict):
            body = str(parsed)
        else:
            body = response.text or ""
    except Exception:
        try:
            body = response.text or ""
        except Exception:
            body = ""

    normalized = body.lower()
    if "duplicate key" in normalized or "unique constraint" in normalized:
        return True
    return "duplicate" in normalized and field_name.lower() in normalized


def _payload_string_lengths(payload: dict[str, Any] | None) -> dict[str, int]:
    if not isinstance(payload, dict):
        return {}
    return {
        key: len(value)
        for key, value in payload.items()
        if isinstance(value, str)
    }


def _payload_field_types(payload: dict[str, Any] | None) -> dict[str, str]:
    if not isinstance(payload, dict):
        return {}
    return {key: type(value).__name__ for key, value in payload.items()}


def _payload_shape(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}

    summary: dict[str, Any] = {}
    for key, value in payload.items():
        item: dict[str, Any] = {"type": type(value).__name__}
        if isinstance(value, str):
            item["length"] = len(value)
        elif isinstance(value, (list, dict)):
            item["size"] = len(value)
        summary[key] = item
    return summary


def _log_write_payload(
    collection: str,
    method: str,
    payload: dict[str, Any],
) -> None:
    if collection not in {"scan_results", "employee_baselines"}:
        return

    ai_model_version = payload.get("ai_model_version")
    logger.info(
        "directus_write_payload_ready collection=%s method=%s payload_keys=%s "
        "ai_model_version=%r ai_model_version_type=%s ai_model_version_len=%s "
        "payload_types=%s payload_string_lengths=%s payload_shape=%s",
        collection,
        method,
        sorted(payload.keys()),
        ai_model_version,
        type(ai_model_version).__name__
        if ai_model_version is not None
        else None,
        len(ai_model_version) if isinstance(ai_model_version, str) else None,
        _payload_field_types(payload),
        _payload_string_lengths(payload),
        _payload_shape(payload),
    )


class DirectusClient:
    def __init__(
        self,
        base_url: str | None = None,
        token: str | None = None,
        timeout: int = 30,
    ):
        raw_base_url = base_url or os.getenv("DIRECTUS_URL", "")
        raw_token = token or os.getenv("DIRECTUS_TOKEN")
        raw_timeout = os.getenv("DIRECTUS_TIMEOUT")
        if raw_timeout is None:
            raw_timeout = timeout

        self.base_url = _normalize_base_url(raw_base_url)
        self.token = _normalize_token(raw_token, field_name="DIRECTUS_TOKEN")
        self.timeout = _parse_timeout(raw_timeout)
        self._field_cache: dict[str, set[str] | None] = {}
        self._field_meta_cache: dict[
            str,
            dict[str, dict[str, Any]] | None,
        ] = {}
        self._schema_lock = threading.RLock()

    def clear_schema_cache(self, collection: str | None = None) -> None:
        with self._schema_lock:
            if collection is None:
                self._field_cache.clear()
                self._field_meta_cache.clear()
                return

            normalized = str(collection).strip()
            self._field_cache.pop(normalized, None)
            self._field_meta_cache.pop(normalized, None)

    def is_configured(self) -> bool:
        return bool(self.base_url and self.token)

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _user_token_headers(self, access_token: str) -> dict[str, str]:
        normalized = _normalize_token(
            access_token,
            field_name="access_token",
        )
        if not normalized:
            raise ValueError("access_token is required")
        return {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {normalized}",
        }

    def _url(self, path: str) -> str:
        if not self.base_url:
            raise RuntimeError("DIRECTUS_URL is not configured")
        if not isinstance(path, str) or not path.startswith("/"):
            raise ValueError("Directus path must be an absolute API path")
        if "://" in path or any(
            character in path for character in ("\r", "\n", "\x00")
        ):
            raise ValueError("Directus path is invalid")
        return f"{self.base_url}{path}"

    def _decode_response(self, response: Any) -> Any:
        if getattr(response, "status_code", None) == 204:
            return None

        try:
            body = response.json()
        except Exception as exc:
            raise RuntimeError(
                "Directus returned a non-JSON response"
            ) from exc

        if isinstance(body, dict) and "data" in body:
            return body.get("data")
        return body

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
    ) -> Any:
        if not self.is_configured():
            raise RuntimeError("Directus credentials are not configured")

        normalized_method = str(method).strip().upper()
        if normalized_method not in _ALLOWED_METHODS:
            raise ValueError(f"unsupported Directus method: {method!r}")
        if params is not None and not isinstance(params, dict):
            raise TypeError("params must be a dictionary")

        payload = (
            _json_safe_copy(json)
            if json is not None
            else None
        )
        response = requests.request(
            normalized_method,
            self._url(path),
            headers=self._headers(),
            params=dict(params) if params is not None else None,
            json=payload,
            timeout=self.timeout,
            allow_redirects=False,
        )

        if 300 <= response.status_code < 400:
            self._log_http_error(
                method=normalized_method,
                path=path,
                response=response,
                payload=payload,
            )
            raise HTTPError(
                "Directus redirects are not allowed",
                response=response,
            )

        try:
            response.raise_for_status()
        except HTTPError:
            self._log_http_error(
                method=normalized_method,
                path=path,
                response=response,
                payload=payload,
            )
            raise

        return self._decode_response(response)

    def get_current_user(self, access_token: str) -> dict | None:
        if not self.base_url:
            raise RuntimeError("DIRECTUS_URL is not configured")

        response = requests.get(
            self._url("/users/me"),
            headers=self._user_token_headers(access_token),
            params={"fields": "id,status"},
            timeout=self.timeout,
            allow_redirects=False,
        )

        if 300 <= response.status_code < 400:
            self._log_http_error(
                method="GET",
                path="/users/me",
                response=response,
                payload=None,
            )
            raise HTTPError(
                "Directus redirects are not allowed",
                response=response,
            )

        try:
            response.raise_for_status()
        except HTTPError:
            self._log_http_error(
                method="GET",
                path="/users/me",
                response=response,
                payload=None,
            )
            raise

        user = self._decode_response(response)
        if user is None:
            return None
        if not isinstance(user, dict):
            raise RuntimeError("Directus /users/me returned an invalid payload")
        return user

    async def _arequest(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
    ) -> Any:
        if httpx is None:
            raise RuntimeError("httpx is not installed")
        if not self.is_configured():
            raise RuntimeError("Directus credentials are not configured")

        normalized_method = str(method).strip().upper()
        if normalized_method not in _ALLOWED_METHODS:
            raise ValueError(f"unsupported Directus method: {method!r}")
        if params is not None and not isinstance(params, dict):
            raise TypeError("params must be a dictionary")

        payload = (
            _json_safe_copy(json)
            if json is not None
            else None
        )

        async with httpx.AsyncClient(
            timeout=self.timeout,
            follow_redirects=False,
        ) as client:
            response = await client.request(
                normalized_method,
                self._url(path),
                headers=self._headers(),
                params=dict(params) if params is not None else None,
                json=payload,
            )

        if 300 <= response.status_code < 400:
            self._log_http_error(
                method=normalized_method,
                path=path,
                response=response,
                payload=payload,
            )
            raise RuntimeError("Directus redirects are not allowed")

        try:
            response.raise_for_status()
        except Exception:
            self._log_http_error(
                method=normalized_method,
                path=path,
                response=response,
                payload=payload,
            )
            raise

        return self._decode_response(response)

    def _log_http_error(
        self,
        *,
        method: str,
        path: str,
        response: Any,
        payload: dict[str, Any] | None,
    ) -> None:
        ai_model_version = (payload or {}).get("ai_model_version")
        logger.error(
            "directus_request_failed method=%s path=%s response_summary=%s "
            "payload_keys=%s ai_model_version=%r ai_model_version_type=%s "
            "ai_model_version_len=%s payload_types=%s "
            "payload_string_lengths=%s payload_shape=%s",
            method,
            path,
            _response_summary(response),
            sorted((payload or {}).keys()),
            ai_model_version,
            type(ai_model_version).__name__
            if ai_model_version is not None
            else None,
            len(ai_model_version)
            if isinstance(ai_model_version, str)
            else None,
            _payload_field_types(payload),
            _payload_string_lengths(payload),
            _payload_shape(payload),
        )

    def get_item(
        self,
        collection: str,
        item_id: Any,
        fields: list[str] | None = None,
    ) -> dict | None:
        collection_segment = _path_segment(
            collection,
            field_name="collection",
        )
        item_segment = _path_segment(item_id, field_name="item_id")
        fields_value = _fields_param(fields)
        params = {"fields": fields_value} if fields_value is not None else None
        result = self._request(
            "GET",
            f"/items/{collection_segment}/{item_segment}",
            params=params,
        )
        if result is None:
            return None
        if not isinstance(result, dict):
            raise RuntimeError("Directus item response must be a dictionary")
        return result

    async def aget_item(
        self,
        collection: str,
        item_id: Any,
        fields: list[str] | None = None,
    ) -> dict | None:
        collection_segment = _path_segment(
            collection,
            field_name="collection",
        )
        item_segment = _path_segment(item_id, field_name="item_id")
        fields_value = _fields_param(fields)
        params = {"fields": fields_value} if fields_value is not None else None
        result = await self._arequest(
            "GET",
            f"/items/{collection_segment}/{item_segment}",
            params=params,
        )
        if result is None:
            return None
        if not isinstance(result, dict):
            raise RuntimeError("Directus item response must be a dictionary")
        return result

    def create_item(self, collection: str, payload: dict) -> dict:
        collection_text = str(collection).strip()
        collection_segment = _path_segment(
            collection_text,
            field_name="collection",
        )
        safe_payload = _json_safe_copy(payload)
        if not isinstance(safe_payload, dict):
            raise TypeError("payload must be a dictionary")

        _log_write_payload(collection_text, "POST", safe_payload)
        result = self._request(
            "POST",
            f"/items/{collection_segment}",
            json=safe_payload,
        )
        if not isinstance(result, dict):
            raise RuntimeError("Directus create response must be a dictionary")
        return result

    def update_item(
        self,
        collection: str,
        item_id: Any,
        payload: dict,
    ) -> dict:
        collection_text = str(collection).strip()
        collection_segment = _path_segment(
            collection_text,
            field_name="collection",
        )
        item_segment = _path_segment(item_id, field_name="item_id")
        safe_payload = _json_safe_copy(payload)
        if not isinstance(safe_payload, dict):
            raise TypeError("payload must be a dictionary")

        _log_write_payload(collection_text, "PATCH", safe_payload)
        result = self._request(
            "PATCH",
            f"/items/{collection_segment}/{item_segment}",
            json=safe_payload,
        )
        if not isinstance(result, dict):
            raise RuntimeError("Directus update response must be a dictionary")
        return result

    def list_items(
        self,
        collection: str,
        *,
        filters: dict[str, Any] | None = None,
        fields: list[str] | None = None,
        limit: int | None = None,
        sort: str | None = None,
    ) -> list[dict]:
        collection_segment = _path_segment(
            collection,
            field_name="collection",
        )
        params: dict[str, Any] = {}

        fields_value = _fields_param(fields)
        if fields_value is not None:
            params["fields"] = fields_value

        if limit is not None:
            if isinstance(limit, bool) or type(limit) is not int or limit <= 0:
                raise ValueError("limit must be a positive integer")
            params["limit"] = limit

        if sort is not None:
            if not isinstance(sort, str) or not sort.strip():
                raise ValueError("sort must be a non-empty string")
            params["sort"] = sort.strip()

        if filters is not None:
            if not isinstance(filters, dict):
                raise TypeError("filters must be a dictionary")
            for key, value in filters.items():
                if not isinstance(key, str) or not key:
                    raise TypeError("filter keys must be non-empty strings")
                params[key] = value

        result = self._request(
            "GET",
            f"/items/{collection_segment}",
            params=params or None,
        )
        if result is None:
            return []
        if not isinstance(result, list) or not all(
            isinstance(row, dict) for row in result
        ):
            raise RuntimeError("Directus list response must be a list of dictionaries")
        return result

    def check_processing_readiness(self) -> None:
        """Verify the service token can perform a minimal processing read.

        This intentionally reads only one non-sensitive field and never mutates
        Directus data. An empty collection is a successful dependency check.
        """
        self.list_items(
            "wellness_scans",
            fields=["id"],
            limit=1,
        )


    def get_collection_fields(self, collection: str) -> set[str] | None:
        collection_text = str(collection).strip()
        with self._schema_lock:
            if collection_text in self._field_cache:
                cached = self._field_cache[collection_text]
                return set(cached) if cached is not None else None

        meta = self.get_collection_field_meta(collection_text)
        if not meta:
            return None

        fields = set(meta.keys())
        with self._schema_lock:
            if len(self._field_cache) >= _MAX_SCHEMA_CACHE_COLLECTIONS:
                self._field_cache.clear()
            self._field_cache[collection_text] = fields
        return set(fields)

    def get_collection_field_meta(
        self,
        collection: str,
    ) -> dict[str, dict[str, Any]] | None:
        collection_text = str(collection).strip()
        with self._schema_lock:
            if collection_text in self._field_meta_cache:
                cached = self._field_meta_cache[collection_text]
                return dict(cached) if cached is not None else None

        collection_segment = _path_segment(
            collection_text,
            field_name="collection",
        )
        try:
            rows = self._request("GET", f"/fields/{collection_segment}")
        except Exception as exc:
            logger.warning(
                "directus_schema_read_failed collection=%s error_type=%s",
                collection_text,
                type(exc).__name__,
            )
            return None

        if rows is None:
            return None
        if not isinstance(rows, list):
            raise RuntimeError("Directus field metadata must be a list")

        meta = {
            row.get("field"): row
            for row in rows
            if isinstance(row, dict)
            and isinstance(row.get("field"), str)
            and row.get("field")
        }
        normalized = meta or None

        with self._schema_lock:
            if len(self._field_meta_cache) >= _MAX_SCHEMA_CACHE_COLLECTIONS:
                self._field_meta_cache.clear()
            self._field_meta_cache[collection_text] = normalized
            if normalized is not None:
                self._field_cache[collection_text] = set(normalized.keys())
        return dict(normalized) if normalized is not None else None

    def get_field_definition(
        self,
        collection: str,
        field_name: str,
    ) -> dict[str, Any] | None:
        if not isinstance(field_name, str) or not field_name.strip():
            raise ValueError("field_name must be a non-empty string")
        meta = self.get_collection_field_meta(collection)
        if not meta:
            return None
        definition = meta.get(field_name.strip())
        return dict(definition) if isinstance(definition, dict) else None

    def get_field_choices(
        self,
        collection: str,
        field_name: str,
    ) -> list[dict[str, Any]]:
        definition = self.get_field_definition(collection, field_name) or {}
        meta = definition.get("meta")
        if not isinstance(meta, dict):
            return []
        options = meta.get("options")
        if not isinstance(options, dict):
            return []
        raw_choices = options.get("choices")
        if not isinstance(raw_choices, list):
            return []

        parsed: list[dict[str, Any]] = []
        for raw in raw_choices:
            if isinstance(raw, dict):
                value = raw.get("value")
                label = raw.get(
                    "text",
                    raw.get("label", raw.get("name", value)),
                )
                if value is not None:
                    parsed.append({"value": value, "label": label})
            elif raw is not None:
                parsed.append({"value": raw, "label": raw})
        return parsed

    def get_field_max_length(
        self,
        collection: str,
        field_name: str,
    ) -> int | None:
        definition = self.get_field_definition(collection, field_name)
        if not definition:
            return None

        for container_name in ("schema", "meta"):
            container = definition.get(container_name)
            if not isinstance(container, dict):
                continue
            for key in ("max_length", "length"):
                value = container.get(key)
                if isinstance(value, bool):
                    continue
                try:
                    length = int(value)
                except (TypeError, ValueError, OverflowError):
                    continue
                if length > 0:
                    return length
        return None

    def get_field_schema_type(
        self,
        collection: str,
        field_name: str,
    ) -> str | None:
        definition = self.get_field_definition(collection, field_name)
        if not definition:
            return None

        schema = definition.get("schema")
        if isinstance(schema, dict):
            for key in ("data_type", "type"):
                value = schema.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip().lower()

        meta = definition.get("meta")
        if not isinstance(meta, dict):
            return None
        special = meta.get("special")
        if isinstance(special, list):
            joined = ",".join(
                str(item).strip().lower()
                for item in special
                if item
            )
            return joined or None
        if isinstance(special, str) and special.strip():
            return special.strip().lower()
        return None

    def is_field_required(
        self,
        collection: str,
        field_name: str,
    ) -> bool | None:
        definition = self.get_field_definition(collection, field_name)
        if not definition:
            return None

        schema = definition.get("schema")
        if isinstance(schema, dict):
            is_nullable = schema.get("is_nullable")
            if isinstance(is_nullable, bool):
                return not is_nullable

        meta = definition.get("meta")
        if isinstance(meta, dict):
            required = meta.get("required")
            if isinstance(required, bool):
                return required
        return None

    def supports_fields(
        self,
        collection: str,
        field_names: list[str],
    ) -> set[str]:
        if not isinstance(field_names, list) or not all(
            isinstance(field, str) and field
            for field in field_names
        ):
            raise TypeError("field_names must be a list of non-empty strings")

        available = self.get_collection_fields(collection)
        if not available:
            return set()
        return {field for field in field_names if field in available}

    def filter_payload_fields(
        self,
        collection: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        safe_payload = _json_safe_copy(payload)
        if not isinstance(safe_payload, dict):
            raise TypeError("payload must be a dictionary")

        available = self.get_collection_fields(collection)
        if not available:
            return dict(safe_payload)
        return {
            key: value
            for key, value in safe_payload.items()
            if key in available
        }

    def first_supported_field(
        self,
        collection: str,
        candidates: list[str],
    ) -> str | None:
        if not isinstance(candidates, list) or not all(
            isinstance(candidate, str) and candidate
            for candidate in candidates
        ):
            raise TypeError("candidates must be a list of non-empty strings")

        available = self.get_collection_fields(collection)
        if not available:
            return None
        for name in candidates:
            if name in available:
                return name
        return None

    def get_scan_context(self, scan_id: Any) -> dict:
        self.clear_schema_cache("wellness_scans")
        optional_scan_fields = sorted(
            self.supports_fields(
                "wellness_scans",
                [
                    "ai_model_version",
                    "failure_message",
                    "failure_reason",
                    "expected_phrase",
                    "processing_attempts",
                    "processing_started_at",
                    "user_message",
                    "validation_warnings",
                    "spoken_transcript",
                ],
            )
        )
        scan = self.get_item(
            "wellness_scans",
            scan_id,
            fields=[
                "id",
                "status",
                "user",
                "started_at",
                "completed_at",
                "task_metrics",
                "business_profile",
                "member",
                "department",
                "request_source",
                "date_created",
                "date_updated",
            ]
            + optional_scan_fields,
        )
        if scan is None:
            raise LookupError(f"wellness scan not found: {scan_id}")

        media_rows = self.list_items(
            "scan_media",
            filters={
                "filter[scan_id][_eq]": scan_id,
                "filter[is_deleted][_neq]": "true",
            },
            fields=[
                "id",
                "scan_id",
                "video_file",
                "audio_file",
                "thumbnail",
                "duration_seconds",
                "business_profile",
                "is_deleted",
            ],
            limit=10,
            sort="-date_created",
        )
        media_row = media_rows[0] if media_rows else None
        scan["scan_media"] = media_row
        scan["resolved_media"] = {
            "image": _relation_id(media_row.get("thumbnail")) if media_row else None,
            "audio": _relation_id(media_row.get("audio_file")) if media_row else None,
            "video": _relation_id(media_row.get("video_file")) if media_row else None,
        }
        return scan

    def get_scan_auth_context(self, scan_id: Any) -> dict | None:
        self.clear_schema_cache("wellness_scans")
        optional_scan_fields = sorted(
            self.supports_fields(
                "wellness_scans",
                ["processing_attempts"],
            )
        )
        return self.get_item(
            "wellness_scans",
            scan_id,
            fields=[
                "id",
                "status",
                "user",
                "business_profile",
                "member",
                "department",
            ]
            + optional_scan_fields,
        )

    def list_business_profile_members(
        self,
        user_id: Any,
        business_profile_id: Any,
        *,
        require_schema: bool = False,
    ) -> list[dict]:
        available = self.get_collection_fields("business_profile_members")
        if not available or not {"id", "user", "business_profile"}.issubset(available):
            if require_schema:
                raise RuntimeError("Directus membership schema is unavailable")
            return []

        fields = [
            field
            for field in [
                "id",
                "user",
                "business_profile",
                "member",
                "department",
                "status",
                "is_active",
                "active",
            ]
            if field in available
        ]
        filters: dict[str, Any] = {
            "filter[user][_eq]": user_id,
            "filter[business_profile][_eq]": business_profile_id,
        }

        return self.list_items(
            "business_profile_members",
            filters=filters,
            fields=fields,
            limit=25,
        )

    def get_scan_media(self, scan_id: Any) -> dict | None:
        media_rows = self.list_items(
            "scan_media",
            filters={
                "filter[scan_id][_eq]": scan_id,
                "filter[is_deleted][_neq]": "true",
            },
            fields=[
                "id",
                "scan_id",
                "video_file",
                "audio_file",
                "thumbnail",
                "duration_seconds",
                "business_profile",
                "is_deleted",
                "date_created",
            ],
            limit=1,
            sort="-date_created",
        )
        return media_rows[0] if media_rows else None

    def get_employee_baseline(self, member_id: Any, business_profile_id: Any) -> dict | None:
        rows = self.get_employee_baselines(member_id, business_profile_id)
        if len(rows) != 1:
            return None
        return rows[0]

    def get_employee_baselines(self, member_id: Any, business_profile_id: Any) -> list[dict]:
        rows = self.list_items(
            "employee_baselines",
            filters={
                "filter[business_profile][_eq]": business_profile_id,
                "filter[member][_eq]": member_id,
            },
            fields=["*"],
            sort="-scan_count,-date_updated",
        )
        return rows or []

    def upsert_employee_baseline(self, baseline_id: Any | None, payload: dict) -> dict:
        if baseline_id:
            return self.update_item("employee_baselines", baseline_id, payload)
        return self.create_item("employee_baselines", payload)

    def create_scan_result(self, payload: dict) -> dict:
        return self.create_item("scan_results", payload)

    def get_scan_result_by_scan_id(self, scan_id: Any) -> dict | None:
        rows = self.list_items(
            "scan_results",
            filters={"filter[scan_id][_eq]": scan_id},
            fields=["*"],
            limit=1,
            sort="-date_created",
        )
        return rows[0] if rows else None

    def upsert_scan_result(self, scan_id: Any, payload: dict) -> tuple[str, dict]:
        existing = self.get_scan_result_by_scan_id(scan_id)
        if existing:
            item_id = _relation_id(existing.get("id"))
            return "updated", self.update_item("scan_results", item_id, payload)
        create_error: Exception | None = None
        try:
            return "created", self.create_item("scan_results", payload)
        except Exception as exc:
            if not _is_unique_conflict(exc, field_name="scan_id"):
                raise
            create_error = exc
        existing = self.get_scan_result_by_scan_id(scan_id)
        if existing:
            item_id = _relation_id(existing.get("id"))
            return "updated_after_conflict", self.update_item("scan_results", item_id, payload)
        raise create_error or RuntimeError("scan_results upsert failed after duplicate conflict")

    def update_wellness_scan(self, scan_id: Any, payload: dict) -> dict:
        return self.update_item("wellness_scans", scan_id, payload)

    def update_member_last_result(self, member_id: Any, payload: dict) -> dict:
        return self.update_item("business_profile_members", member_id, payload)

    def get_latest_scan_result_for_member(self, member_id: Any) -> dict | None:
        if not member_id:
            return None
        rows = self.list_items(
            "scan_results",
            filters={"filter[member][_eq]": member_id},
            fields=["id", "scan_id", "confidence", "risk_level", "date_created"],
            limit=1,
            sort="-date_created",
        )
        return rows[0] if rows else None

    def find_matching_scan_request(self, scan_context: dict) -> dict | None:
        request_source = scan_context.get("request_source")
        if request_source not in {"manager_request", "bulk_request"}:
            return None

        business_profile = _relation_id(scan_context.get("business_profile"))
        member = _relation_id(scan_context.get("member"))
        if not business_profile or not member:
            return None

        now = _utc_now()
        rows = self.list_items(
            "scan_requests",
            filters={
                "filter[business_profile][_eq]": business_profile,
                "filter[target_member][_eq]": member,
                "filter[status][_in]": "pending,sent,opened",
                "filter[_or][0][due_at][_null]": "true",
                "filter[_or][1][due_at][_gte]": now,
            },
            fields=[
                "id",
                "status",
                "due_at",
                "business_profile",
                "target_member",
                "requested_at",
            ],
            limit=1,
            sort="-requested_at",
        )
        return rows[0] if rows else None

    def update_scan_request_if_needed(
        self,
        *,
        request_id: Any | None,
        scan_context: dict,
        scan_id: Any,
    ) -> dict | None:
        target_request_id = request_id
        if not target_request_id:
            match = self.find_matching_scan_request(scan_context)
            target_request_id = _relation_id(match.get("id")) if match else None
        if not target_request_id:
            return None
        return self.update_item(
            "scan_requests",
            target_request_id,
            {
                "status": "completed",
                "completed_scan": scan_id,
                "completed_at": _utc_now(),
            },
        )

    def create_alert_if_needed(
        self,
        *,
        risk_level: str,
        confidence: float,
        scan_id: Any,
        member_id: Any,
        business_profile_id: Any,
        department_id: Any,
        user_id: Any,
    ) -> dict | None:
        if isinstance(confidence, bool) or type(confidence) not in {int, float}:
            raise TypeError("confidence must be a finite number")
        confidence = float(confidence)
        if not math.isfinite(confidence) or confidence < 0.0 or confidence > 1.0:
            raise ValueError("confidence must be in the range 0..1")

        if risk_level == "high_risk":
            severity = "critical" if confidence >= 0.8 else "high"
            title = "High readiness risk detected"
            message = "Do not continue safety-sensitive work. Supervisor review is required."
        elif risk_level == "elevated_fatigue":
            if confidence < 0.8:
                return None
            severity = "high" if confidence >= 0.9 else "medium"
            title = "Elevated fatigue detected"
            message = "A scan suggests elevated fatigue and should be reviewed."
        else:
            return None

        return self.create_item(
            "alerts",
            {
                "business_profile": business_profile_id,
                "department": department_id,
                "target_member": member_id,
                "target_user": user_id,
                "scan": scan_id,
                "severity": severity,
                "title": title,
                "message": message,
                "status": "new",
                "action_type": "none",
            },
        )

    def list_readiness_alert_recipients(
        self,
        *,
        business_profile_id: Any,
        target_user_id: Any,
    ) -> list[Any]:
        if not business_profile_id:
            return []
        rows = self.list_items(
            "business_profile_members",
            filters={
                "filter[business_profile][_eq]": business_profile_id,
                "filter[status][_in]": "active,accepted",
                "filter[member_role][_in]": "owner,hr,manager,manger",
            },
            fields=[
                "id",
                "user",
                "user.id",
                "member_role",
                "status",
                "business_profile",
            ],
            limit=50,
        )
        recipients: list[Any] = []
        excluded = str(_relation_id(target_user_id) or "").strip()
        seen: set[str] = set()
        for row in rows or []:
            user_id = _relation_id((row or {}).get("user"))
            text = str(user_id or "").strip()
            if not text or text == excluded or text in seen:
                continue
            seen.add(text)
            recipients.append(user_id)
        return recipients

    def create_notification(
        self,
        *,
        user_id: Any,
        business_profile_id: Any,
        alert_id: Any,
        scan_id: Any,
        member_id: Any,
        risk_level: str,
    ) -> dict | None:
        if not user_id:
            return None
        return self.create_item(
            "notifications",
            {
                "user": user_id,
                "title": "Readiness alert",
                "body": "Do not continue safety-sensitive work. Supervisor review is required.",
                "type": "alert",
                "status": "unread",
                "link_id": alert_id,
                "link_type": "alert",
                "business_profile": business_profile_id,
                "meta": {
                    "scan_id": scan_id,
                    "member_id": member_id,
                    "risk_level": risk_level,
                },
            },
        )
