from __future__ import annotations

from datetime import datetime, timezone
import os
from typing import Any

import requests
from requests import HTTPError

from logger import get_logger

try:
    import httpx
except Exception:  # pragma: no cover
    httpx = None


logger = get_logger()


def _relation_id(value: Any) -> Any:
    if isinstance(value, dict):
        return value.get("id")
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _is_unique_conflict(exc: Exception, *, field_name: str) -> bool:
    if not isinstance(exc, HTTPError):
        return False
    response = exc.response
    if response is None or response.status_code not in {400, 409}:
        return False
    body = ""
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
    return {key: len(value) for key, value in payload.items() if isinstance(value, str)}


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


def _log_write_payload(collection: str, method: str, payload: dict[str, Any]) -> None:
    if collection not in {"scan_results", "employee_baselines"}:
        return
    ai_model_version = payload.get("ai_model_version")
    logger.info(
        "directus_write_payload_ready collection=%s method=%s payload_keys=%s ai_model_version=%r ai_model_version_type=%s ai_model_version_len=%s payload_types=%s payload_string_lengths=%s payload_shape=%s",
        collection,
        method,
        sorted(payload.keys()),
        ai_model_version,
        type(ai_model_version).__name__ if ai_model_version is not None else None,
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
        self.base_url = (base_url or os.getenv("DIRECTUS_URL", "")).rstrip("/")
        self.token = token or os.getenv("DIRECTUS_TOKEN")
        self.timeout = int(os.getenv("DIRECTUS_TIMEOUT", timeout))
        self._field_cache: dict[str, set[str] | None] = {}
        self._field_meta_cache: dict[str, dict[str, dict[str, Any]] | None] = {}

    def clear_schema_cache(self, collection: str | None = None) -> None:
        if collection is None:
            self._field_cache.clear()
            self._field_meta_cache.clear()
            return
        self._field_cache.pop(collection, None)
        self._field_meta_cache.pop(collection, None)

    def is_configured(self) -> bool:
        return bool(self.base_url and self.token)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _user_token_headers(self, access_token: str) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {access_token}",
        }

    def _url(self, path: str) -> str:
        if not self.base_url:
            raise RuntimeError("DIRECTUS_URL is not configured")
        return f"{self.base_url}{path}"

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
        response = requests.request(
            method,
            self._url(path),
            headers=self._headers(),
            params=params,
            json=json,
            timeout=self.timeout,
        )
        try:
            response.raise_for_status()
        except HTTPError:
            self._log_http_error(
                method=method,
                path=path,
                response=response,
                payload=json,
            )
            raise
        data = response.json()
        return data.get("data", data)

    def get_current_user(self, access_token: str) -> dict | None:
        if not self.base_url:
            raise RuntimeError("DIRECTUS_URL is not configured")
        response = requests.get(
            self._url("/users/me"),
            headers=self._user_token_headers(access_token),
            params={"fields": "id,status"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        user = data.get("data", data)
        return user if isinstance(user, dict) else None

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
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.request(
                method,
                self._url(path),
                headers=self._headers(),
                params=params,
                json=json,
            )
            try:
                response.raise_for_status()
            except Exception:
                self._log_http_error(
                    method=method,
                    path=path,
                    response=response,
                    payload=json,
                )
                raise
            data = response.json()
            return data.get("data", data)

    def _response_body(self, response: Any) -> Any:
        try:
            body = response.json()
            if body is not None:
                return body
        except Exception:
            pass
        try:
            return response.text
        except Exception:
            return None

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
            "directus_request_failed method=%s status_code=%s path=%s url=%s payload_keys=%s ai_model_version=%r ai_model_version_type=%s ai_model_version_len=%s payload_types=%s payload_string_lengths=%s payload_shape=%s response_body=%s",
            method,
            getattr(response, "status_code", None),
            path,
            getattr(response, "url", self._url(path)),
            sorted((payload or {}).keys()),
            ai_model_version,
            type(ai_model_version).__name__ if ai_model_version is not None else None,
            len(ai_model_version) if isinstance(ai_model_version, str) else None,
            _payload_field_types(payload),
            _payload_string_lengths(payload),
            _payload_shape(payload),
            self._response_body(response),
        )

    def get_item(self, collection: str, item_id: Any, fields: list[str] | None = None) -> dict | None:
        params = {"fields": ",".join(fields)} if fields else None
        return self._request("GET", f"/items/{collection}/{item_id}", params=params)

    async def aget_item(self, collection: str, item_id: Any, fields: list[str] | None = None) -> dict | None:
        params = {"fields": ",".join(fields)} if fields else None
        return await self._arequest("GET", f"/items/{collection}/{item_id}", params=params)

    def create_item(self, collection: str, payload: dict) -> dict:
        _log_write_payload(collection, "POST", payload)
        return self._request("POST", f"/items/{collection}", json=payload)

    def update_item(self, collection: str, item_id: Any, payload: dict) -> dict:
        _log_write_payload(collection, "PATCH", payload)
        return self._request("PATCH", f"/items/{collection}/{item_id}", json=payload)

    def list_items(
        self,
        collection: str,
        *,
        filters: dict[str, Any] | None = None,
        fields: list[str] | None = None,
        limit: int | None = None,
        sort: str | None = None,
    ) -> list[dict]:
        params: dict[str, Any] = {}
        if fields:
            params["fields"] = ",".join(fields)
        if limit is not None:
            params["limit"] = limit
        if sort:
            params["sort"] = sort
        for key, value in (filters or {}).items():
            params[key] = value
        return self._request("GET", f"/items/{collection}", params=params)

    def get_collection_fields(self, collection: str) -> set[str] | None:
        if collection in self._field_cache:
            return self._field_cache[collection]
        meta = self.get_collection_field_meta(collection)
        if not meta:
            self._field_cache[collection] = None
            return None
        self._field_cache[collection] = set(meta.keys()) or None
        return self._field_cache[collection]

    def get_collection_field_meta(self, collection: str) -> dict[str, dict[str, Any]] | None:
        if collection in self._field_meta_cache:
            return self._field_meta_cache[collection]
        try:
            rows = self._request("GET", f"/fields/{collection}")
        except Exception:
            self._field_meta_cache[collection] = None
            self._field_cache[collection] = None
            return None
        meta = {
            row.get("field"): row
            for row in (rows or [])
            if isinstance(row, dict) and row.get("field")
        }
        self._field_meta_cache[collection] = meta or None
        self._field_cache[collection] = set(meta.keys()) or None
        return self._field_meta_cache[collection]

    def get_field_definition(self, collection: str, field_name: str) -> dict[str, Any] | None:
        meta = self.get_collection_field_meta(collection)
        if not meta:
            return None
        return meta.get(field_name)

    def get_field_choices(self, collection: str, field_name: str) -> list[dict[str, Any]]:
        definition = self.get_field_definition(collection, field_name) or {}
        meta = definition.get("meta") or {}
        options = meta.get("options") or {}
        raw_choices = options.get("choices") or []
        parsed: list[dict[str, Any]] = []
        for raw in raw_choices:
            if isinstance(raw, dict):
                value = raw.get("value")
                label = raw.get("text", raw.get("label", raw.get("name", value)))
                if value is not None:
                    parsed.append({"value": value, "label": label})
            elif raw is not None:
                parsed.append({"value": raw, "label": raw})
        return parsed

    def get_field_max_length(self, collection: str, field_name: str) -> int | None:
        definition = self.get_field_definition(collection, field_name)
        if not definition:
            return None
        schema = definition.get("schema") or {}
        for key in ["max_length", "length"]:
            value = schema.get(key)
            try:
                if value is not None:
                    return int(value)
            except (TypeError, ValueError):
                continue
        meta = definition.get("meta") or {}
        for key in ["max_length", "length"]:
            value = meta.get(key)
            try:
                if value is not None:
                    return int(value)
            except (TypeError, ValueError):
                continue
        return None

    def get_field_schema_type(self, collection: str, field_name: str) -> str | None:
        definition = self.get_field_definition(collection, field_name)
        if not definition:
            return None
        schema = definition.get("schema") or {}
        for key in ["data_type", "type"]:
            value = schema.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip().lower()
        meta = definition.get("meta") or {}
        special = meta.get("special")
        if isinstance(special, list):
            joined = ",".join(str(item).strip().lower() for item in special if item)
            if joined:
                return joined
        if isinstance(special, str) and special.strip():
            return special.strip().lower()
        return None

    def is_field_required(self, collection: str, field_name: str) -> bool | None:
        definition = self.get_field_definition(collection, field_name)
        if not definition:
            return None
        schema = definition.get("schema") or {}
        if "is_nullable" in schema:
            return not bool(schema.get("is_nullable"))
        meta = definition.get("meta") or {}
        if "required" in meta:
            return bool(meta.get("required"))
        return None

    def supports_fields(self, collection: str, field_names: list[str]) -> set[str]:
        available = self.get_collection_fields(collection)
        if not available:
            return set()
        return {field for field in field_names if field in available}

    def filter_payload_fields(self, collection: str, payload: dict[str, Any]) -> dict[str, Any]:
        available = self.get_collection_fields(collection)
        if not available:
            return payload
        return {key: value for key, value in payload.items() if key in available}

    def first_supported_field(self, collection: str, candidates: list[str]) -> str | None:
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

    def list_business_profile_members(self, user_id: Any, business_profile_id: Any) -> list[dict]:
        available = self.get_collection_fields("business_profile_members")
        if not available or not {"id", "user", "business_profile"}.issubset(available):
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
