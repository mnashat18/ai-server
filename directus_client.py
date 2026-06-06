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
        logger.error(
            "directus_request_failed method=%s status_code=%s path=%s url=%s payload_keys=%s response_body=%s",
            method,
            getattr(response, "status_code", None),
            path,
            getattr(response, "url", self._url(path)),
            sorted((payload or {}).keys()),
            self._response_body(response),
        )

    def get_item(self, collection: str, item_id: Any, fields: list[str] | None = None) -> dict | None:
        params = {"fields": ",".join(fields)} if fields else None
        return self._request("GET", f"/items/{collection}/{item_id}", params=params)

    async def aget_item(self, collection: str, item_id: Any, fields: list[str] | None = None) -> dict | None:
        params = {"fields": ",".join(fields)} if fields else None
        return await self._arequest("GET", f"/items/{collection}/{item_id}", params=params)

    def create_item(self, collection: str, payload: dict) -> dict:
        return self._request("POST", f"/items/{collection}", json=payload)

    def update_item(self, collection: str, item_id: Any, payload: dict) -> dict:
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
        rows = self.list_items(
            "employee_baselines",
            filters={
                "filter[business_profile][_eq]": business_profile_id,
                "filter[member][_eq]": member_id,
            },
            fields=["*"],
            limit=1,
        )
        return rows[0] if rows else None

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
            message = "A scan requires manager review before safety-critical work."
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
                "body": "A team member has a scan result that needs review.",
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
