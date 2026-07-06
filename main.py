from __future__ import annotations

import concurrent.futures
from datetime import datetime, timezone
import math
import os
import shutil
import subprocess
import tempfile
import time
import traceback
import sys
from typing import Any

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import requests
from requests import HTTPError

from baseline import (
    baseline_ready_for_personalized_scoring,
    baseline_signal_payload,
    baseline_status_payload,
    evaluate_baseline_eligibility,
)
from config import MAX_DOWNLOAD_BYTES, MODEL_VERSION
from directus_client import DirectusClient
from logger import get_logger
from ml.features import features_from_signals, vector_from_features
from ml.runtime import MLRuntime
from quality import assess_quality
from scoring import compute_result
from utils import directus_auth_headers, download_temp_file, is_url, remove_temp_file, safe_number, sanitize_payload, sanitize_text
from validation import ValidationPolicy, fail_validation, failure_message, validate_scan_inputs


app = FastAPI()
logger = get_logger()
ml_runtime = MLRuntime()
ml_runtime.load()
directus = DirectusClient()

VALIDATION_POLICY = ValidationPolicy.from_env()
AI_SERVER_ENV = os.getenv("AI_SERVER_ENV", "production").strip().lower()
DEBUG_SCAN_ENDPOINT_ENABLED = AI_SERVER_ENV in {"dev", "development", "local", "test"} and os.getenv(
    "DEBUG_SCAN_ENDPOINT_ENABLED",
    "",
).strip().lower() in {"1", "true", "yes", "on"}
OPTIONAL_PHRASE_TIMEOUT_SECONDS = 1.5
TRANSCRIPTION_TIMEOUT_SECONDS = float(os.getenv("AUDIO_TRANSCRIPTION_TIMEOUT_SECONDS", "4.0"))
FAST_SCAN_MODE = os.getenv("FAST_SCAN_MODE", "true").strip().lower() in {"1", "true", "yes", "on"}
FAST_SCAN_DOWNLOAD_TIMEOUT_SECONDS = float(os.getenv("FAST_SCAN_DOWNLOAD_TIMEOUT_SECONDS", "2.5"))
MEDIA_VALIDATION_WALL_TIMEOUT_SECONDS = float(os.getenv("MEDIA_VALIDATION_WALL_TIMEOUT_SECONDS", "8.0"))

SCAN_STATUS_PENDING = "pending"
SCAN_STATUS_MEDIA_READY = "media_ready"
SCAN_STATUS_PROCESSING = "processing"
SCAN_STATUS_COMPLETED = "completed"
SCAN_STATUS_FAILED = "failed"

FAILURE_REASON_MISSING_MEDIA = "missing_media"
FAILURE_REASON_VIDEO_MISSING = "video_missing"
FAILURE_REASON_VIDEO_TOO_DARK = "video_too_dark"
FAILURE_REASON_VIDEO_BLURRY = "video_blurry"
FAILURE_REASON_VIDEO_TOO_SHORT = "video_too_short"
FAILURE_REASON_UNSTABLE_VIDEO = "unstable_video"
FAILURE_REASON_IMAGE_MISSING = "image_missing"
FAILURE_REASON_IMAGE_TOO_DARK = "image_too_dark"
FAILURE_REASON_IMAGE_BLURRY = "image_blurry"
FAILURE_REASON_AUDIO_MISSING = "audio_missing"
FAILURE_REASON_AUDIO_TOO_QUIET = "audio_too_quiet"
FAILURE_REASON_AUDIO_TOO_NOISY = "audio_too_noisy"
FAILURE_REASON_SPEECH_NOT_DETECTED = "speech_not_detected"
FAILURE_REASON_AUDIO_TOO_SHORT = "audio_too_short"
FAILURE_REASON_FACE_NOT_VISIBLE = "face_not_visible"
FAILURE_REASON_PHRASE_MISMATCH = "phrase_mismatch"
FAILURE_REASON_TRANSCRIPTION_FAILED = "transcription_failed"
FAILURE_REASON_EXPECTED_PHRASE_MISSING = "expected_phrase_missing"
FAILURE_REASON_LOW_QUALITY_MEDIA = "low_quality_media"
FAILURE_REASON_AUDIO_VALIDATION_TIMEOUT = "audio_validation_timeout"
FAILURE_REASON_MODEL_NOT_LOADED = "model_not_loaded"
FAILURE_REASON_DIRECTUS_DOWNLOAD_FAILED = "directus_download_failed"
FAILURE_REASON_ANALYSIS_EXCEPTION = "analysis_exception"
FAILURE_REASON_WRITEBACK_FAILED = "writeback_failed"

OPTIONAL_SCAN_RESULT_FIELDS = [
    "analysis_metadata",
    "media_quality",
    "warnings",
    "modality_scores",
    "fusion_details",
    "internal_analysis",
    "result_status",
    "capture_quality_score",
    "measurement_reliability_score",
    "observed_fatigue_score",
    "fatigue_evidence_score",
    "personal_deviation_score",
    "task_completion_status",
    "baseline_status_at_inference",
    "baseline_confidence",
    "baseline_eligible",
    "hard_gates_triggered",
    "explainable_reasons",
]

SCAN_RESULT_NUMERIC_FIELDS: dict[str, bool] = {
    "readiness_score": True,
    "observed_fatigue_score": True,
    "confidence": False,
    "camera_confidence": False,
    "voice_confidence": False,
    "task_performance_score": True,
    "confidence_drift": False,
    "phrase_match_score": False,
    "audio_quality_score": False,
    "video_quality_score": False,
    "image_quality_score": False,
    "capture_quality_score": False,
    "measurement_reliability_score": False,
    "personal_deviation_score": False,
    "baseline_confidence": False,
    "fatigue_evidence_score": False,
}

SCAN_RESULT_CHOICE_ALIASES: dict[str, dict[str, list[str]]] = {
    "risk_level": {
        "stable": ["Stable"],
        "low_focus": ["Low Focus"],
        "elevated_fatigue": ["Elevated Fatigue"],
        "high_risk": ["High Risk"],
    },
    "baseline_status": {
        "collecting": ["Collecting"],
        "provisional": ["Provisional"],
        "active": ["Active"],
        "needs_review": ["Needs Review"],
        "disabled": ["Disabled"],
    },
    "result_status": {
        "scored": ["Scored"],
        "retake_required": ["Retake Required"],
        "incomplete": ["Incomplete"],
        "low_confidence": ["Low Confidence"],
        "failed": ["Failed"],
    },
    "suggested_action": {
        "continue_normal_activity": ["Continue Normal Activity"],
        "review_required": ["Review Required"],
        "rescan_recommended": ["Rescan Recommended"],
        "rest_advised": ["Rest Advised"],
        "manager_review": ["Manager Review"],
    },
    "task_completion_status": {
        "completed": ["Completed"],
        "incomplete_required_speech": ["Incomplete Required Speech"],
        "incomplete_required_task": ["Incomplete Required Task"],
        "not_required": ["Not Required"],
    },
}


class Media(BaseModel):
    image: str | None = Field(None, description="Local path, URL, or Directus asset ID")
    audio: str | None = Field(None, description="Local path, URL, or Directus asset ID")
    video: str | None = Field(None, description="Local path, URL, or Directus asset ID")


class Task(BaseModel):
    reaction_time: float | None = None
    errors: int | None = None
    attempts: int | None = None


class ScanRequest(BaseModel):
    scan_id: str = Field(..., min_length=1)

    class Config:
        extra = "ignore"


class BaselineRequest(BaseModel):
    scan_id: str | None = None
    media: Media | None = None
    manually_unreliable: bool = False


class ProcessResponse(BaseModel):
    ok: bool = True
    status: str
    scan_id: str


class ScanResultResponse(BaseModel):
    status: str
    retake_required: bool = False
    failure_reason: str | None = None
    readiness_score: int | None = None
    risk_level: str | None = None
    confidence: float | None = None
    camera_confidence: float | None = None
    voice_confidence: float | None = None
    task_performance_score: int | None = None
    baseline_used: bool = False
    confidence_drift: float | None = None
    face_metrics: dict | None = None
    voice_metrics: dict | None = None
    reaction_metrics: dict | None = None
    explanation: str
    suggested_action: str
    ai_model_version: str
    diagnostics: dict | None = None
    writeback_status: dict | None = None


class BaselineStatusResponse(BaseModel):
    is_active: bool
    scan_count: int
    scans_remaining: int
    is_provisional: bool
    needs_morning_scan: bool
    needs_evening_scan: bool
    message: str


class ProcessingError(RuntimeError):
    def __init__(self, reason: str, message: str | None = None):
        super().__init__(message or reason)
        self.reason = reason
        self.message = message or reason


class SchemaValidationError(ProcessingError):
    def __init__(self, collection: str, field_name: str, actual_length: int, max_length: int):
        super().__init__(
            FAILURE_REASON_WRITEBACK_FAILED,
            (
                f"{collection}.{field_name} exceeds Directus max length "
                f"({actual_length}>{max_length})"
            ),
        )
        self.collection = collection
        self.field_name = field_name
        self.actual_length = actual_length
        self.max_length = max_length


@app.get("/")
def root():
    return {"status": "ok"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _relation_id(value: Any) -> Any:
    if isinstance(value, dict):
        return value.get("id", value.get("uuid"))
    return value


def _model_to_dict(model: BaseModel | None) -> dict:
    if not model:
        return {}
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _scan_timestamp(scan_context: dict) -> str:
    return (
        scan_context.get("completed_at")
        or scan_context.get("started_at")
        or scan_context.get("date_created")
        or _utc_now()
    )


def _safe_string(value: Any, max_len: int = 255, fallback: str | None = None) -> str | None:
    return sanitize_text(value, fallback=fallback, max_len=max_len)


def _field_length_checked_string(
    collection: str,
    field_name: str,
    value: Any,
    *,
    fallback: str | None = None,
    default_max_len: int = 65535,
    required: bool = False,
) -> str | None:
    text = sanitize_text(value, fallback=fallback, max_len=default_max_len)
    if text is None:
        return None
    max_length = directus.get_field_max_length(collection, field_name)
    if max_length is None or len(text) <= max_length:
        return text
    logger.warning(
        "directus_field_too_long collection=%s field=%s max_length=%s actual_length=%s required=%s",
        collection,
        field_name,
        max_length,
        len(text),
        required,
    )
    if required:
        raise SchemaValidationError(collection, field_name, len(text), max_length)
    return None


def _normalize_choice_token(value: Any) -> str:
    text = _safe_string(value, max_len=255, fallback="") or ""
    normalized = []
    for char in text.lower():
        normalized.append(char if char.isalnum() else "_")
    compact = "".join(normalized).strip("_")
    while "__" in compact:
        compact = compact.replace("__", "_")
    return compact


def _safe_numeric(value: Any, *, integer: bool) -> int | float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    if integer:
        return int(round(numeric))
    return round(numeric, 6)


def _is_json_field(collection: str, field_name: str) -> bool:
    schema_type = directus.get_field_schema_type(collection, field_name) or ""
    return schema_type in {"json", "jsonb"} or "json" in schema_type


def _truncate_string_to_schema(
    collection: str,
    field_name: str,
    value: Any,
    *,
    fallback: str | None = None,
    default_max_len: int = 65535,
) -> str | None:
    text = sanitize_text(value, fallback=fallback, max_len=default_max_len)
    if text is None:
        return None
    max_length = directus.get_field_max_length(collection, field_name)
    if max_length is None or len(text) <= max_length:
        return text
    logger.warning(
        "directus_field_truncated collection=%s field=%s max_length=%s actual_length=%s",
        collection,
        field_name,
        max_length,
        len(text),
    )
    return text[:max_length]


def _coerce_json_field(collection: str, field_name: str, value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        schema_type = directus.get_field_schema_type(collection, field_name)
        if not schema_type or _is_json_field(collection, field_name):
            return value
        logger.warning(
            "directus_json_field_skipped collection=%s field=%s schema_type=%s value_type=%s",
            collection,
            field_name,
            schema_type,
            type(value).__name__,
        )
        return None
    return value


def _scan_result_payload_diagnostics(payload: dict[str, Any]) -> tuple[dict[str, int], dict[str, str]]:
    string_lengths = {key: len(value) for key, value in payload.items() if isinstance(value, str)}
    field_types = {key: type(value).__name__ for key, value in payload.items()}
    return string_lengths, field_types


def _log_scan_result_payload_ready(scan_id: str, payload: dict[str, Any]) -> None:
    string_lengths, field_types = _scan_result_payload_diagnostics(payload)
    ai_model_version = payload.get("ai_model_version")
    logger.info(
        "scan_id=%s step=scan_result_payload_ready ai_model_version=%r ai_model_version_type=%s ai_model_version_len=%s payload_keys=%s payload_types=%s payload_string_lengths=%s",
        scan_id,
        ai_model_version,
        type(ai_model_version).__name__ if ai_model_version is not None else None,
        len(str(ai_model_version)) if ai_model_version is not None else 0,
        sorted(payload.keys()),
        field_types,
        string_lengths,
    )


def _coerce_scan_result_choice(field_name: str, value: Any) -> Any:
    text = _safe_string(value, max_len=255)
    if text is None:
        return None
    choices = directus.get_field_choices("scan_results", field_name)
    if not choices:
        return text

    normalized_allowed: dict[str, Any] = {}
    allowed_values: list[Any] = []
    for choice in choices:
        actual = choice.get("value")
        label = choice.get("label")
        if actual is None:
            continue
        allowed_values.append(actual)
        for token in [actual, label]:
            normalized = _normalize_choice_token(token)
            if normalized:
                normalized_allowed[normalized] = actual

    normalized_text = _normalize_choice_token(text)
    if text in allowed_values:
        return text
    if normalized_text in normalized_allowed:
        return normalized_allowed[normalized_text]

    for alias in SCAN_RESULT_CHOICE_ALIASES.get(field_name, {}).get(text, []):
        normalized_alias = _normalize_choice_token(alias)
        if normalized_alias in normalized_allowed:
            mapped = normalized_allowed[normalized_alias]
            logger.info(
                "scan_result_choice_mapped field=%s source=%s mapped=%s allowed=%s",
                field_name,
                text,
                mapped,
                allowed_values,
            )
            return mapped

    logger.warning(
        "scan_result_choice_unmapped field=%s value=%s allowed=%s",
        field_name,
        text,
        allowed_values,
    )
    return None


def _scan_result_scan_id_value(scan_id: Any) -> str | int | None:
    value = _relation_id(scan_id)
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    text = _safe_string(value, max_len=255)
    return text


def _schema_aware_scan_result_payload(payload: dict[str, Any]) -> dict[str, Any]:
    candidate = dict(payload)

    candidate["scan_id"] = _scan_result_scan_id_value(candidate.get("scan_id"))

    for field_name, integer in SCAN_RESULT_NUMERIC_FIELDS.items():
        if field_name in candidate:
            candidate[field_name] = _safe_numeric(candidate.get(field_name), integer=integer)

    for field_name in [
        "risk_level",
        "suggested_action",
        "result_status",
        "task_completion_status",
        "baseline_status_at_inference",
    ]:
        if field_name in candidate:
            choice_field = "baseline_status" if field_name == "baseline_status_at_inference" else field_name
            candidate[field_name] = _coerce_scan_result_choice(choice_field, candidate.get(field_name))

    candidate["explanation"] = _truncate_string_to_schema(
        "scan_results",
        "explanation",
        candidate.get("explanation"),
        fallback="Analysis completed.",
        default_max_len=65535,
    )
    candidate["suggested_action"] = _field_length_checked_string(
        "scan_results",
        "suggested_action",
        candidate.get("suggested_action"),
        default_max_len=255,
        required=True,
    )
    candidate["ai_model_version"] = _field_length_checked_string(
        "scan_results",
        "ai_model_version",
        MODEL_VERSION,
        fallback=MODEL_VERSION,
        default_max_len=255,
        required=True,
    )

    for field_name in ["spoken_transcript", "expected_phrase"]:
        if field_name in candidate:
            candidate[field_name] = _field_length_checked_string(
                "scan_results",
                field_name,
                candidate.get(field_name),
                default_max_len=65535,
                required=False,
            )

    for field_name in [
        "face_metrics",
        "voice_metrics",
        "reaction_metrics",
        "analysis_metadata",
        "media_quality",
        "warnings",
        "modality_scores",
        "fusion_details",
        "validation_warnings",
        "hard_gates_triggered",
        "explainable_reasons",
    ]:
        if field_name in candidate:
            candidate[field_name] = _coerce_json_field("scan_results", field_name, candidate.get(field_name))

    filtered = _scan_result_payload(candidate)

    required_missing: list[str] = []
    for field_name in ["scan_id", "risk_level", "confidence", "readiness_score", "explanation", "suggested_action", "ai_model_version"]:
        if field_name not in filtered:
            required = directus.is_field_required("scan_results", field_name)
            if required:
                required_missing.append(field_name)
    if required_missing:
        raise ProcessingError(
            FAILURE_REASON_WRITEBACK_FAILED,
            f"scan_results required fields missing after schema validation: {', '.join(required_missing)}",
        )
    return filtered


def _directus_payload_preserve_nulls(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _directus_payload_preserve_nulls(raw) for key, raw in value.items()}
    if isinstance(value, list):
        return [_directus_payload_preserve_nulls(raw) for raw in value]
    if isinstance(value, tuple):
        return [_directus_payload_preserve_nulls(raw) for raw in value]
    if isinstance(value, float):
        return safe_number(value, digits=6)
    if isinstance(value, str):
        return sanitize_text(value)
    return value


def _log_step(scan_id: str, step: str, **details: Any) -> None:
    if details:
        logger.info("scan_id=%s step=%s details=%s", scan_id, step, sanitize_payload(details))
        return
    logger.info("scan_id=%s step=%s", scan_id, step)


def _elapsed_ms(started_at: float) -> int:
    return int(round((time.perf_counter() - started_at) * 1000))


def _log_perf(scan_id: str, metric: str, elapsed_ms: int | float | None) -> None:
    value = int(round(float(elapsed_ms or 0)))
    logger.info("[PERF] %s scan_id=%s value=%s", metric, scan_id, value)


def _log_validation_decision(
    scan_id: str,
    *,
    valid_modalities: list[str],
    timed_out_modalities: list[str],
    terminal_reason: str,
) -> None:
    logger.info("[VALIDATION_DECISION] valid_modalities scan_id=%s value=%s", scan_id, valid_modalities)
    logger.info("[VALIDATION_DECISION] timed_out_modalities scan_id=%s value=%s", scan_id, timed_out_modalities)
    logger.info("[VALIDATION_DECISION] terminal_reason scan_id=%s value=%s", scan_id, terminal_reason)


def _log_validation_lifecycle(scan_id: str, *, all_workers_terminal: bool, running_modalities: list[str]) -> None:
    logger.info(
        "[VALIDATION_LIFECYCLE] all_workers_terminal=%s running_modalities=%s scan_id=%s",
        str(bool(all_workers_terminal)).lower(),
        running_modalities,
        scan_id,
    )


def _shutdown_executor(executor: concurrent.futures.ThreadPoolExecutor | None) -> None:
    if executor is None:
        return
    executor.shutdown(wait=False, cancel_futures=True)


def _has_meaningful_evidence(value: Any) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return not math.isclose(float(value), 0.0, abs_tol=0.0)
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, dict):
        return any(_has_meaningful_evidence(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(_has_meaningful_evidence(item) for item in value)
    return bool(value)


def _result_has_valid_evidence(result: dict[str, Any]) -> bool:
    for field_name in ["face_metrics", "voice_metrics", "reaction_metrics", "modality_scores"]:
        if _has_meaningful_evidence(result.get(field_name)):
            return True
    return False


def _build_scan_result_response(
    *,
    ok: bool,
    scan_id: str,
    status: str | None = None,
    error: str | None = None,
    current_status: str | None = None,
    status_code: int = 200,
) -> JSONResponse:
    payload: dict[str, Any] = {"ok": ok, "scan_id": scan_id}
    if status is not None:
        payload["status"] = status
    if error is not None:
        payload["error"] = error
    if current_status is not None:
        payload["current_status"] = current_status
    return JSONResponse(status_code=status_code, content=payload)


def _bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    parts = authorization.strip().split()
    if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1].strip():
        return None
    return parts[1].strip()


def _ids_equal(left: Any, right: Any) -> bool:
    left_id = _relation_id(left)
    right_id = _relation_id(right)
    if left_id is None or right_id is None:
        return False
    return str(left_id) == str(right_id)


def _truthy_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "active", "enabled", "approved", "published"}
    return bool(value)


def _membership_row_is_active(row: dict[str, Any]) -> bool:
    status = row.get("status")
    if status not in (None, ""):
        status_value = str(_relation_id(status)).strip().lower()
        if status_value not in {"active", "enabled", "approved", "published"}:
            return False
    for field_name in ["is_active", "active"]:
        if field_name in row and row.get(field_name) is not None and not _truthy_flag(row.get(field_name)):
            return False
    return True


def _membership_row_matches_scan_scope(row: dict[str, Any], scan_context: dict[str, Any]) -> bool:
    scan_member_id = _relation_id(scan_context.get("member"))
    scan_department_id = _relation_id(scan_context.get("department"))

    if scan_member_id:
        member_candidates = [_relation_id(row.get("id")), _relation_id(row.get("member"))]
        if not any(_ids_equal(candidate, scan_member_id) for candidate in member_candidates if candidate is not None):
            return False

    row_department_id = _relation_id(row.get("department"))
    if scan_department_id and row_department_id and not _ids_equal(row_department_id, scan_department_id):
        return False

    return True


def _authenticate_process_user(authorization: str | None, scan_id: str) -> Any:
    token = _bearer_token(authorization)
    if not token:
        logger.info("process_auth_failed scan_id=%s reason=missing_or_malformed_authorization", scan_id)
        raise HTTPException(status_code=401, detail="invalid_authorization")

    try:
        user = directus.get_current_user(token)
    except HTTPError as exc:
        status_code = getattr(exc.response, "status_code", None)
        if status_code in {401, 403}:
            logger.info("process_auth_failed scan_id=%s reason=invalid_or_expired_token status_code=%s", scan_id, status_code)
            raise HTTPException(status_code=401, detail="invalid_authorization") from exc
        logger.warning("process_identity_request_failed scan_id=%s status_code=%s", scan_id, status_code)
        raise HTTPException(status_code=502, detail="directus_identity_request_failed") from exc
    except Exception as exc:
        logger.warning("process_identity_request_failed scan_id=%s error_type=%s", scan_id, type(exc).__name__)
        raise HTTPException(status_code=502, detail="directus_identity_request_failed") from exc

    user_id = _relation_id((user or {}).get("id"))
    if not user_id:
        logger.info("process_auth_failed scan_id=%s reason=missing_user_id", scan_id)
        raise HTTPException(status_code=401, detail="invalid_authorization")
    user_status = str(_relation_id((user or {}).get("status")) or "").strip().lower()
    if not user_status or user_status != "active":
        logger.info("process_auth_failed scan_id=%s reason=inactive_user", scan_id)
        raise HTTPException(status_code=401, detail="invalid_authorization")
    return user_id


def _resolve_scan_auth_context(scan_id: str) -> dict[str, Any]:
    if not directus.is_configured():
        raise HTTPException(status_code=500, detail="Directus credentials are not configured")
    try:
        scan_context = directus.get_scan_auth_context(scan_id)
    except HTTPError as exc:
        status_code = getattr(exc.response, "status_code", None)
        if status_code == 404:
            raise HTTPException(status_code=404, detail="wellness_scans record not found") from exc
        logger.warning("scan_auth_context_failed scan_id=%s status_code=%s", scan_id, status_code)
        raise HTTPException(status_code=502, detail="scan_auth_context_failed") from exc
    except Exception as exc:
        logger.exception("scan_auth_context_failed scan_id=%s error=%s", scan_id, exc)
        raise HTTPException(status_code=502, detail="scan_auth_context_failed") from exc

    if not scan_context or not _relation_id(scan_context.get("id")):
        raise HTTPException(status_code=404, detail="wellness_scans record not found")
    return scan_context


def _authorize_scan_access(scan_context: dict[str, Any], authenticated_user_id: Any) -> None:
    scan_id = _relation_id(scan_context.get("id"))
    scan_user_id = _relation_id(scan_context.get("user"))
    if not scan_user_id or not _ids_equal(scan_user_id, authenticated_user_id):
        logger.info("process_scan_not_found_or_not_owned scan_id=%s authenticated_user_id=%s", scan_id, authenticated_user_id)
        raise HTTPException(status_code=404, detail="wellness_scans record not found")

    business_profile_id = _relation_id(scan_context.get("business_profile"))
    if not business_profile_id:
        logger.info("process_membership_denied scan_id=%s user_id=%s reason=missing_business_profile", scan_id, authenticated_user_id)
        raise HTTPException(status_code=403, detail="active_membership_required")

    try:
        membership_rows = directus.list_business_profile_members(authenticated_user_id, business_profile_id)
    except Exception as exc:
        logger.warning(
            "process_membership_lookup_failed scan_id=%s user_id=%s business_profile_id=%s error=%s",
            scan_id,
            authenticated_user_id,
            business_profile_id,
            exc,
        )
        raise HTTPException(status_code=403, detail="active_membership_required") from exc

    for row in membership_rows:
        if not _ids_equal(row.get("user"), authenticated_user_id):
            continue
        if not _ids_equal(row.get("business_profile"), business_profile_id):
            continue
        if not _membership_row_is_active(row):
            continue
        if _membership_row_matches_scan_scope(row, scan_context):
            return

    logger.info(
        "process_membership_denied scan_id=%s user_id=%s business_profile_id=%s member_id=%s department_id=%s",
        scan_id,
        authenticated_user_id,
        business_profile_id,
        _relation_id(scan_context.get("member")),
        _relation_id(scan_context.get("department")),
    )
    raise HTTPException(status_code=403, detail="active_membership_required")


def _ensure_scan_media_ready(scan_id: str) -> None:
    try:
        media_row = directus.get_scan_media(scan_id)
    except Exception as exc:
        logger.warning("process_scan_media_lookup_failed scan_id=%s error_type=%s", scan_id, type(exc).__name__)
        raise HTTPException(status_code=409, detail="scan_media_not_ready") from exc

    if not media_row:
        logger.info("process_scan_media_not_ready scan_id=%s reason=missing_scan_media", scan_id)
        raise HTTPException(status_code=409, detail="scan_media_not_ready")

    required_fields = []
    if VALIDATION_POLICY.require_video:
        required_fields.append("video_file")
    if VALIDATION_POLICY.require_audio:
        required_fields.append("audio_file")
    if getattr(VALIDATION_POLICY, "require_image", False):
        required_fields.append("thumbnail")

    missing_fields = [field for field in required_fields if not _relation_id(media_row.get(field))]
    if missing_fields:
        logger.info("process_scan_media_not_ready scan_id=%s missing_fields=%s", scan_id, missing_fields)
        raise HTTPException(status_code=409, detail="scan_media_not_ready")


def _model_health() -> dict[str, Any]:
    model_path = ml_runtime.model_path
    return {
        "status": "ok",
        "model_version": MODEL_VERSION,
        "ml_loaded": ml_runtime.is_loaded(),
        "ml_error": ml_runtime.error,
        "configured_model_path": model_path,
        "model_file_exists": bool(model_path and os.path.exists(model_path)),
        "local_model_required": ml_runtime.local_model_required(),
        "directus_configured": directus.is_configured(),
        "validation": {
            "require_video": VALIDATION_POLICY.require_video,
            "require_audio": VALIDATION_POLICY.require_audio,
            "require_face": VALIDATION_POLICY.require_face,
            "require_image": VALIDATION_POLICY.require_image,
            "require_phrase_match": VALIDATION_POLICY.require_phrase_match,
            "phrase_match_threshold": VALIDATION_POLICY.phrase_match_threshold,
        },
    }


def _should_convert_audio(path: str) -> bool:
    _, ext = os.path.splitext(path)
    return ext.lower() not in [".wav", ".wave"]


def _convert_audio_to_wav(path: str) -> str:
    if shutil.which("ffmpeg") is None:
        raise ProcessingError(FAILURE_REASON_ANALYSIS_EXCEPTION, "ffmpeg not installed")
    fd, out = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        path,
        "-t",
        "3.0",
        "-ac",
        "1",
        "-ar",
        "16000",
        out,
    ]
    try:
        subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=3.5,
        )
    except subprocess.TimeoutExpired as exc:
        remove_temp_file(out)
        raise ProcessingError(FAILURE_REASON_AUDIO_VALIDATION_TIMEOUT, "audio conversion timed out") from exc
    except Exception as exc:
        remove_temp_file(out)
        raise ProcessingError(FAILURE_REASON_ANALYSIS_EXCEPTION, f"audio conversion failed: {exc}") from exc
    return out


def _download_directus_asset(asset_id: str, suffix: str) -> str:
    if not directus.is_configured():
        raise ProcessingError(FAILURE_REASON_DIRECTUS_DOWNLOAD_FAILED, "Directus credentials are not configured")
    url = f"{directus.base_url}/assets/{asset_id}"
    download_timeout = (3, 5) if FAST_SCAN_MODE else (10, 30)
    response = requests.get(url, headers=directus_auth_headers(url), timeout=download_timeout, stream=True)
    response.raise_for_status()
    fd, path = tempfile.mkstemp(suffix=suffix)
    total = 0
    with os.fdopen(fd, "wb") as handle:
        for chunk in response.iter_content(chunk_size=8192):
            if not chunk:
                continue
            total += len(chunk)
            if total > MAX_DOWNLOAD_BYTES:
                remove_temp_file(path)
                raise ProcessingError(FAILURE_REASON_DIRECTUS_DOWNLOAD_FAILED, f"Downloaded file too large: {asset_id}")
            handle.write(chunk)
    return path


def _resolve_media_input(
    value: str | None,
    suffix: str,
    media_kind: str,
    *,
    allow_url: bool = True,
    allow_local_path: bool = True,
) -> tuple[str | None, bool]:
    if not value:
        return None, False
    if allow_local_path and os.path.exists(value):
        return value, False
    try:
        if allow_url and is_url(value):
            path = download_temp_file(value, suffix, timeout=(3, 5) if FAST_SCAN_MODE else (10, 30))
            return path, True
        if is_url(value) or os.path.isabs(value) or any(sep in value for sep in ["/", "\\"]):
            raise ProcessingError(
                FAILURE_REASON_DIRECTUS_DOWNLOAD_FAILED,
                f"{media_kind} must reference a Directus file id",
            )
        path = _download_directus_asset(value, suffix)
        return path, True
    except Exception as exc:
        raise ProcessingError(
            FAILURE_REASON_DIRECTUS_DOWNLOAD_FAILED,
            f"{media_kind} download failed: {exc}",
        ) from exc


def _resolve_scan_context(scan_id: str) -> dict:
    if not directus.is_configured():
        raise HTTPException(status_code=500, detail="Directus credentials are not configured")
    try:
        return directus.get_scan_context(scan_id)
    except Exception as exc:
        logger.exception("scan_context_failed scan_id=%s error=%s", scan_id, exc)
        raise HTTPException(status_code=404, detail="wellness_scans record not found") from exc


def _merge_media(scan_context: dict) -> Media:
    directus_values = scan_context.get("resolved_media", {})
    return Media(
        image=directus_values.get("image"),
        audio=directus_values.get("audio"),
        video=directus_values.get("video"),
    )


def _merge_task(scan_context: dict) -> Task | None:
    task_metrics = scan_context.get("task_metrics")
    if not isinstance(task_metrics, dict):
        return None
    return Task(
        reaction_time=task_metrics.get("reaction_time"),
        errors=task_metrics.get("errors"),
        attempts=task_metrics.get("attempts"),
    )


def _baseline_for_member(member_id: str | None, business_profile_id: str | None) -> dict | None:
    rows = _baseline_rows_for_member(member_id, business_profile_id)
    if len(rows) != 1:
        if len(rows) > 1:
            logger.warning(
                "baseline_duplicate_rows member_id=%s business_profile_id=%s count=%s",
                member_id,
                business_profile_id,
                len(rows),
            )
        return None
    return rows[0]


def _baseline_rows_for_member(member_id: str | None, business_profile_id: str | None) -> list[dict]:
    if not directus.is_configured() or not member_id or not business_profile_id:
        return []
    try:
        return directus.get_employee_baselines(member_id, business_profile_id)
    except Exception as exc:
        logger.warning("baseline_fetch_failed member_id=%s error=%s", member_id, exc)
        return []


def _identifier_payload(scan_context: dict) -> dict:
    return {
        "user_id": _relation_id(scan_context.get("user")),
        "member_id": _relation_id(scan_context.get("member")),
        "business_profile_id": _relation_id(scan_context.get("business_profile")),
        "department_id": _relation_id(scan_context.get("department")),
    }


def _high_risk_evidence_summary(result: dict, internal_analysis: dict) -> str:
    quality = (internal_analysis or {}).get("quality") or {}
    warnings = set(quality.get("warnings") or result.get("validation_warnings") or [])
    parts: list[str] = []
    if "sustained_eye_closure" in warnings:
        parts.append("sustained_eye_closure")
    if "speech_not_detected" in warnings:
        parts.append("speech_not_detected")
    if "too_much_silence" in warnings:
        parts.append("too_much_silence")
    if not parts:
        parts.append("score_confidence_policy")
    return ",".join(parts[:4])


def _dispatch_high_risk_notifications(
    *,
    scan_id: str,
    alert: dict | None,
    identifiers: dict,
    risk_level: str,
) -> str:
    if not alert:
        return "not_available"
    alert_id = _relation_id(alert.get("id"))
    if not alert_id:
        return "not_available"
    recipients = directus.list_readiness_alert_recipients(
        business_profile_id=identifiers.get("business_profile_id"),
        target_user_id=identifiers.get("user_id"),
    )
    if not recipients:
        return "not_available"
    created = 0
    for user_id in recipients:
        directus.create_notification(
            user_id=user_id,
            business_profile_id=identifiers.get("business_profile_id"),
            alert_id=alert_id,
            scan_id=scan_id,
            member_id=identifiers.get("member_id"),
            risk_level=risk_level,
        )
        created += 1
    return "attempted" if created else "not_available"


def _expected_phrase(scan_context: dict) -> str | None:
    return _safe_string(scan_context.get("expected_phrase"), max_len=500)


def _wellness_scan_update_payload(payload: dict[str, Any]) -> dict[str, Any]:
    directus.clear_schema_cache("wellness_scans")
    original = dict(payload)
    candidate = dict(payload)
    if "failure_reason" in candidate:
        candidate["failure_reason"] = _field_length_checked_string(
            "wellness_scans",
            "failure_reason",
            candidate.get("failure_reason"),
            default_max_len=255,
            required=False,
        )
    if "failure_message" in candidate:
        candidate["failure_message"] = _field_length_checked_string(
            "wellness_scans",
            "failure_message",
            candidate.get("failure_message"),
            default_max_len=65535,
            required=False,
        )
    if "ai_model_version" in candidate:
        candidate["ai_model_version"] = _field_length_checked_string(
            "wellness_scans",
            "ai_model_version",
            MODEL_VERSION,
            fallback=MODEL_VERSION,
            default_max_len=255,
            required=False,
        )

    filtered = directus.filter_payload_fields("wellness_scans", candidate)
    fallback_field = directus.first_supported_field("wellness_scans", ["failure_message", "user_message"])
    if "failure_message" in candidate and "failure_message" not in filtered and fallback_field:
        filtered[fallback_field] = candidate["failure_message"]
    cleaned = sanitize_payload(filtered)
    for field_name in ["failure_reason", "failure_message", "completed_at", "ai_model_version", "status"]:
        if field_name in filtered and filtered[field_name] is None and original.get(field_name) is None:
            cleaned[field_name] = None
    if fallback_field and fallback_field in filtered and filtered[fallback_field] is None and original.get("failure_message") is None:
        cleaned[fallback_field] = None
    return cleaned


def _scan_result_payload(payload: dict[str, Any]) -> dict[str, Any]:
    filtered = directus.filter_payload_fields("scan_results", payload)
    metadata_field = directus.first_supported_field("scan_results", ["analysis_metadata"])
    if metadata_field and metadata_field not in filtered:
        extras = {}
        for key in [
            "spoken_transcript",
            "expected_phrase",
            "phrase_match_score",
            "audio_quality_score",
            "video_quality_score",
            "image_quality_score",
            "validation_warnings",
        ]:
            if key in payload:
                extras[key] = payload[key]
        if extras:
            filtered[metadata_field] = extras
    return _directus_payload_preserve_nulls(filtered)


def _safe_analyze(fn, path: str | None, missing_warning: str) -> dict:
    if not path:
        return {"score": None, "details": {"status": "missing", "warnings": [missing_warning]}}
    try:
        result = fn(path)
        return result if isinstance(result, dict) else {"score": None, "details": {"status": "invalid"}}
    except Exception as exc:
        logger.exception("analyzer_error path=%s error=%s", path, exc)
    return {"score": None, "details": {"status": "error", "warnings": [missing_warning]}}


def _timed_safe_analyze(
    scan_id: str,
    metric_name: str,
    fn,
    path: str | None,
    missing_warning: str,
) -> dict:
    started = time.perf_counter()
    result = _safe_analyze(fn, path, missing_warning)
    _log_perf(scan_id, metric_name, _elapsed_ms(started))
    return result


def _analysis_timeout_placeholder(media_kind: str) -> dict:
    warning_key = {
        "video": "visual_warnings",
        "audio": "audio_warnings",
        "image": "image_warnings",
    }.get(media_kind, "warnings")
    status = "load_failed" if media_kind in {"video", "audio"} else "invalid_image"
    return {
        "score": None,
        "details": {
            "status": status,
            warning_key: [f"{media_kind}_timeout"],
        },
    }


def _analyze_audio_file(path: str | None) -> dict:
    from audio import analyze_audio

    return analyze_audio(path)


def _transcribe_audio_file(path: str, timeout_seconds: float | None = None) -> str:
    if not path:
        raise RuntimeError("audio_missing")
    effective_timeout = TRANSCRIPTION_TIMEOUT_SECONDS if timeout_seconds is None else float(timeout_seconds)
    cmd = [
        sys.executable,
        "-c",
        (
            "import sys; "
            "from audio import transcribe_audio; "
            "print(transcribe_audio(sys.argv[1]))"
        ),
        path,
    ]
    try:
        completed = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
            timeout=effective_timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError("transcription_timeout") from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        raise RuntimeError(stderr or "transcription_failed") from exc
    transcript = (completed.stdout or "").strip()
    if not transcript:
        raise RuntimeError("transcription_failed")
    return transcript


def _transcribe_audio_file_optional(path: str | None, timeout_seconds: float = OPTIONAL_PHRASE_TIMEOUT_SECONDS) -> tuple[str | None, str]:
    if not path:
        return None, "audio_missing"
    try:
        return _transcribe_audio_file(path, timeout_seconds=timeout_seconds), "completed"
    except TimeoutError:
        logger.warning("phrase_transcription_timeout timeout_seconds=%s", timeout_seconds)
        return None, "timeout"
    except Exception as exc:
        logger.warning("phrase_transcription_error error=%s", exc)
        return None, "error"


def _analyze_video_file(path: str | None) -> dict:
    from video import analyze_video

    return analyze_video(path)


def _analyze_face_image(path: str | None) -> dict:
    from vision import analyze_face

    return analyze_face(path)


def _analyze_media(scan_id: str, media: Media) -> tuple[dict, list[str]]:
    if not any([media.image, media.audio, media.video]):
        raise ProcessingError(FAILURE_REASON_MISSING_MEDIA, "No media linked to scan")

    temp_files: list[str] = []
    image_path, image_temp = _resolve_media_input(media.image, ".jpg", "image")
    audio_path, audio_temp = _resolve_media_input(media.audio, ".bin", "audio")
    video_path, video_temp = _resolve_media_input(media.video, ".mp4", "video")
    for path, is_temp in [(image_path, image_temp), (audio_path, audio_temp), (video_path, video_temp)]:
        if path and is_temp:
            temp_files.append(path)

    if audio_path and _should_convert_audio(audio_path):
        converted = _convert_audio_to_wav(audio_path)
        temp_files.append(converted)
        audio_path = converted

    _log_step(scan_id, "video_analysis_start", has_video=bool(video_path))
    video = _safe_analyze(_analyze_video_file, video_path, "video_missing")
    _log_step(scan_id, "video_analysis_done", score=video.get("score"))

    _log_step(scan_id, "audio_analysis_start", has_audio=bool(audio_path))
    voice = _safe_analyze(_analyze_audio_file, audio_path, "audio_missing")
    _log_step(scan_id, "audio_analysis_done", score=voice.get("score"))

    _log_step(scan_id, "image_analysis_start", has_image=bool(image_path))
    camera = _safe_analyze(_analyze_face_image, image_path, "image_missing")
    _log_step(scan_id, "image_analysis_done", score=camera.get("score"))

    return {"camera": camera, "video": video, "voice": voice}, temp_files


def _mark_scan_failed(scan_id: str, reason: str, message: str | None = None) -> dict[str, str]:
    payload = _wellness_scan_update_payload(
        {
            "status": SCAN_STATUS_FAILED,
            "failure_reason": _safe_string(reason, fallback=FAILURE_REASON_ANALYSIS_EXCEPTION),
            "failure_message": _safe_string(message or failure_message(reason), max_len=500),
            "completed_at": _utc_now(),
            "ai_model_version": MODEL_VERSION,
        }
    )
    try:
        directus.update_wellness_scan(scan_id, payload)
        _log_step(scan_id, "directus_writeback_done", writeback_status={"wellness_scan": "failed_updated"})
        return {"wellness_scan": "failed_updated"}
    except Exception as exc:
        logger.exception("scan_id=%s step=directus_mark_failed_error error=%s", scan_id, exc)
        return {"wellness_scan": f"failed:{exc}"}


def _quality_failure_response(scan_id: str, quality_result: dict, diagnostics: dict, writeback_status: dict) -> dict:
    explanation = "Available media was too weak for a reliable result. A re-scan is recommended."
    if diagnostics.get("quality", {}).get("media_quality", {}).get("image", {}).get("usable") and not diagnostics.get("quality", {}).get("media_quality", {}).get("video", {}).get("usable") and not diagnostics.get("quality", {}).get("media_quality", {}).get("audio", {}).get("usable"):
        explanation = "Only thumbnail data was available, so this result has limited confidence."
    return {
        "status": "failed",
        "retake_required": True,
        "failure_reason": quality_result.get("failure_reason") or FAILURE_REASON_LOW_QUALITY_MEDIA,
        "readiness_score": 35 if (quality_result.get("failure_reason") == FAILURE_REASON_MISSING_MEDIA) else 45,
        "risk_level": None,
        "confidence": 0.2,
        "camera_confidence": None,
        "voice_confidence": None,
        "task_performance_score": None,
        "baseline_used": False,
        "confidence_drift": None,
        "face_metrics": None,
        "voice_metrics": None,
        "reaction_metrics": None,
        "explanation": explanation,
        "suggested_action": "rescan_recommended",
        "ai_model_version": MODEL_VERSION,
        "diagnostics": diagnostics,
        "writeback_status": writeback_status,
    }


def _safe_internal_analysis_text(result: dict, internal_analysis: dict) -> str:
    quality = internal_analysis.get("quality") or {}
    warnings = [str(w).replace("_", " ") for w in (quality.get("warnings") or result.get("validation_warnings") or [])[:3]]
    if quality.get("failure_reason") in {"missing_media", "low_quality_media"}:
        summary = "reliable assessment unavailable"
        if warnings:
            summary = f"{summary}: {', '.join(warnings)}"
        return sanitize_text(summary, fallback="reliable assessment unavailable", max_len=255) or "reliable assessment unavailable"
    if warnings:
        return sanitize_text(f"analysis warnings: {', '.join(warnings)}", fallback="analysis available", max_len=255) or "analysis available"
    return "analysis available"


def _build_scan_result_payload(scan_id: str, result: dict, internal_analysis: dict) -> dict:
    payload = {
        "scan_id": scan_id,
        "readiness_score": result.get("readiness_score"),
        "observed_fatigue_score": result.get("observed_fatigue_score"),
        "risk_level": result.get("risk_level"),
        "confidence": result.get("confidence"),
        "camera_confidence": result.get("camera_confidence"),
        "voice_confidence": result.get("voice_confidence"),
        "task_performance_score": result.get("task_performance_score"),
        "explanation": sanitize_text(result.get("explanation"), fallback="Analysis completed.", max_len=65535),
        "suggested_action": sanitize_text(result.get("suggested_action"), fallback="review_required", max_len=255),
        "ai_model_version": MODEL_VERSION,
        "confidence_drift": result.get("confidence_drift"),
        "fatigue_evidence_score": result.get("fatigue_evidence_score"),
        "baseline_used": result.get("baseline_used"),
        "face_metrics": result.get("face_metrics"),
        "voice_metrics": result.get("voice_metrics"),
        "reaction_metrics": result.get("reaction_metrics"),
        "spoken_transcript": sanitize_text(result.get("spoken_transcript"), max_len=65535),
        "expected_phrase": sanitize_text(result.get("expected_phrase"), max_len=65535),
        "phrase_match_score": result.get("phrase_match_score"),
        "audio_quality_score": result.get("audio_quality_score"),
        "video_quality_score": result.get("video_quality_score"),
        "image_quality_score": result.get("image_quality_score"),
        "validation_warnings": result.get("validation_warnings"),
        "result_status": result.get("result_status"),
        "capture_quality_score": result.get("capture_quality_score"),
        "measurement_reliability_score": result.get("measurement_reliability_score"),
        "personal_deviation_score": result.get("personal_deviation_score"),
        "task_completion_status": result.get("task_completion_status"),
        "baseline_status_at_inference": result.get("baseline_status_at_inference"),
        "baseline_confidence": result.get("baseline_confidence"),
        "baseline_eligible": result.get("baseline_eligible"),
        "hard_gates_triggered": result.get("hard_gates_triggered"),
        "explainable_reasons": result.get("explainable_reasons"),
    }
    supported_optional = directus.supports_fields(
        "scan_results",
        OPTIONAL_SCAN_RESULT_FIELDS
        + [
            "spoken_transcript",
            "expected_phrase",
            "phrase_match_score",
            "audio_quality_score",
            "video_quality_score",
            "image_quality_score",
            "validation_warnings",
        ],
    )
    if "analysis_metadata" in supported_optional:
        payload["analysis_metadata"] = internal_analysis
    if "media_quality" in supported_optional:
        payload["media_quality"] = internal_analysis.get("quality", {}).get("media_quality")
    if "warnings" in supported_optional:
        payload["warnings"] = internal_analysis.get("warnings")
    if "modality_scores" in supported_optional:
        payload["modality_scores"] = result.get("modality_scores")
    if "fusion_details" in supported_optional:
        payload["fusion_details"] = result.get("fusion_details")
    if "internal_analysis" in supported_optional:
        payload["internal_analysis"] = _safe_internal_analysis_text(result, internal_analysis)
    return _schema_aware_scan_result_payload(payload)


def _write_quality_failure(scan_id: str, quality_result: dict) -> dict:
    return _mark_scan_failed(
        scan_id,
        quality_result.get("failure_reason") or FAILURE_REASON_LOW_QUALITY_MEDIA,
    )


def _baseline_personal_deviation_score(result: dict) -> float | None:
    drifts = []
    for field_name in ["face_metrics", "voice_metrics", "reaction_metrics"]:
        metric_drifts = (result.get(field_name) or {}).get("baseline_drifts") or {}
        for drift_payload in metric_drifts.values():
            drift = (drift_payload or {}).get("drift")
            if drift is None:
                continue
            try:
                drifts.append(abs(float(drift)))
            except (TypeError, ValueError):
                continue
    if not drifts:
        return None
    return round(sum(drifts) / len(drifts), 4)


def _result_status_from_outcome(
    *,
    quality_result: dict,
    validation_result: dict,
    result: dict,
    baseline_eligibility: dict,
) -> str:
    if result.get("risk_level") == "high_risk":
        return "scored"
    if quality_result.get("retake_required") or result.get("retake_required") or quality_result.get("failure_reason") in {"low_quality_media", "missing_media"}:
        return "retake_required"
    task_completion_status = baseline_eligibility.get("task_completion_status")
    if task_completion_status in {"incomplete_required_speech", "incomplete_required_task"}:
        return "incomplete"
    confidence = result.get("confidence")
    if confidence is None or float(confidence) < 0.45:
        return "low_confidence"
    return "scored"


def _write_success(
    *,
    scan_id: str,
    scan_context: dict,
    identifiers: dict,
    result: dict,
    internal_analysis: dict,
) -> dict:
    if not _result_has_valid_evidence(result):
        raise ProcessingError(FAILURE_REASON_ANALYSIS_EXCEPTION, "invalid_result_classification")

    status: dict[str, Any] = {}
    try:
        scan_result_payload = _build_scan_result_payload(scan_id, result, internal_analysis)
        _log_scan_result_payload_ready(scan_id, scan_result_payload)
        write_mode, scan_result = directus.upsert_scan_result(scan_id, scan_result_payload)
        status["scan_result"] = f"{write_mode}:{_relation_id(scan_result.get('id')) or 'ok'}"
    except Exception as exc:
        logger.exception("scan_result_write_failed scan_id=%s error=%s", scan_id, exc)
        raise ProcessingError(FAILURE_REASON_WRITEBACK_FAILED, str(exc)) from exc

    try:
        directus.update_wellness_scan(
            scan_id,
            _wellness_scan_update_payload(
                {
                "status": SCAN_STATUS_COMPLETED,
                "completed_at": _utc_now(),
                "failure_reason": None,
                "failure_message": None,
                "ai_model_version": MODEL_VERSION,
                }
            ),
        )
        status["wellness_scan"] = "updated"
    except Exception as exc:
        logger.exception("wellness_scan_update_failed scan_id=%s error=%s", scan_id, exc)
        raise ProcessingError(FAILURE_REASON_WRITEBACK_FAILED, str(exc)) from exc

    member_id = identifiers.get("member_id")
    if member_id:
        try:
            directus.update_member_last_result(
                member_id,
                {
                    "last_scan_at": _utc_now(),
                    "last_readiness_score": result.get("readiness_score"),
                    "last_risk_level": result.get("risk_level"),
                },
            )
            status["member"] = "updated"
        except Exception as exc:
            logger.warning("member_update_failed member_id=%s error=%s", member_id, exc)
            status["member"] = f"failed:{exc}"

    try:
        scan_request = directus.update_scan_request_if_needed(
            request_id=None,
            scan_context=scan_context,
            scan_id=scan_id,
        )
        status["scan_request"] = "updated" if scan_request else "skipped"
    except Exception as exc:
        logger.warning("scan_request_update_failed scan_id=%s error=%s", scan_id, exc)
        status["scan_request"] = f"failed:{exc}"

    high_risk_detected = result.get("risk_level") == "high_risk"
    high_risk_evidence = _high_risk_evidence_summary(result, internal_analysis)
    logger.info("[HIGH_RISK] detected=%s", str(bool(high_risk_detected)).lower())
    logger.info("[HIGH_RISK] evidence=%s", high_risk_evidence)
    logger.info("[HIGH_RISK] final_state=%s", result.get("risk_level") or "retake_required")

    alert_result = "skipped"
    notification_dispatch = "not_available"
    alert = None
    try:
        alert = directus.create_alert_if_needed(
            risk_level=result["risk_level"],
            confidence=float(result["confidence"] or 0.0),
            scan_id=scan_id,
            member_id=identifiers.get("member_id"),
            business_profile_id=identifiers.get("business_profile_id"),
            department_id=identifiers.get("department_id"),
            user_id=identifiers.get("user_id"),
        )
        alert_result = "created" if alert else "skipped"
        status["alert"] = alert_result
    except Exception as exc:
        logger.warning("alert_write_failed scan_id=%s error=%s", scan_id, exc)
        alert_result = "failed"
        notification_dispatch = "failed" if high_risk_detected else "not_available"
        status["alert"] = f"failed:{exc}"
    if high_risk_detected and alert:
        try:
            notification_dispatch = _dispatch_high_risk_notifications(
                scan_id=scan_id,
                alert=alert,
                identifiers=identifiers,
                risk_level=result["risk_level"],
            )
            status["notification_dispatch"] = notification_dispatch
        except Exception as exc:
            logger.warning("notification_dispatch_failed scan_id=%s error=%s", scan_id, exc)
            notification_dispatch = "failed"
            status["notification_dispatch"] = f"failed:{exc}"
    logger.info("[HIGH_RISK] alert_result=%s", alert_result)
    logger.info("[HIGH_RISK] notification_dispatch=%s", notification_dispatch)
    return status


def _critical_validation_errors_allow_result(critical_errors: list[str] | None) -> bool:
    errors = set(critical_errors or [])
    if not errors:
        return True
    return errors.issubset({"missing_media", "unreadable_media"})


def _face_eye_evidence_unreliable(warnings: list[str] | None, video_details: dict | None, image_details: dict | None) -> bool:
    warning_set = set(warnings or [])
    if warning_set & {"face_not_visible", "subject_not_visible", "landmark_detection_failed", "insufficient_usable_frames"}:
        return True
    video_details = video_details or {}
    image_details = image_details or {}
    if video_details.get("reliable_eye_landmarks") is False and int(video_details.get("face_frames") or 0) <= 0:
        return True
    if image_details and image_details.get("face_detected") is False and image_details.get("avg_ear") is None:
        return True
    return False


def _required_modality_gate(
    quality_result: dict,
    *,
    timed_out_modalities: list[str],
) -> tuple[str | None, list[str]]:
    media_quality = quality_result.get("media_quality") or {}
    required_modalities: list[str] = []
    for modality, required in [
        ("video", VALIDATION_POLICY.require_video),
        ("audio", VALIDATION_POLICY.require_audio),
        ("image", VALIDATION_POLICY.require_image),
    ]:
        if not required:
            continue
        required_modalities.append(modality)
        if modality in timed_out_modalities:
            if modality == "audio":
                return FAILURE_REASON_AUDIO_VALIDATION_TIMEOUT, required_modalities
            return FAILURE_REASON_LOW_QUALITY_MEDIA, required_modalities
        modality_quality = media_quality.get(modality) or {}
        if modality == "audio" and "audio_decode_timeout" in set(modality_quality.get("warnings") or []):
            return FAILURE_REASON_AUDIO_VALIDATION_TIMEOUT, required_modalities
        if not modality_quality.get("present"):
            if modality == "audio":
                return FAILURE_REASON_AUDIO_MISSING, required_modalities
            if modality == "video":
                return FAILURE_REASON_VIDEO_MISSING, required_modalities
            if modality == "image":
                return FAILURE_REASON_IMAGE_MISSING, required_modalities
            return FAILURE_REASON_MISSING_MEDIA, required_modalities
        if not modality_quality.get("usable"):
            return FAILURE_REASON_LOW_QUALITY_MEDIA, required_modalities
    return None, required_modalities


def _process_scan_sync(scan_id: str) -> dict[str, Any]:
    total_started = time.perf_counter()
    _log_step(scan_id, "validation_start")
    scan_context = _resolve_scan_context(scan_id)
    _log_step(scan_id, "scan_context_loaded", status=scan_context.get("status"))

    media_row = scan_context.get("scan_media") or directus.get_scan_media(scan_id)
    _log_step(
        scan_id,
        "scan_media_loaded",
        media_found=media_row is not None,
        video_file_id=_relation_id((media_row or {}).get("video_file")),
        audio_file_id=_relation_id((media_row or {}).get("audio_file")),
        thumbnail_id=_relation_id((media_row or {}).get("thumbnail")),
    )

    if media_row is None:
        scan_context["scan_media"] = None
        scan_context["resolved_media"] = {"image": None, "audio": None, "video": None}
        _log_step(scan_id, "media_missing_continue_with_unknown_result")

    if ml_runtime.local_model_required() and not ml_runtime.is_loaded():
        validation_result = fail_validation(FAILURE_REASON_MODEL_NOT_LOADED)
        _log_step(scan_id, "validation_failed", reason=validation_result["failure_reason"])
        _log_step(scan_id, "directus_writeback_start")
        writeback_started = time.perf_counter()
        _mark_scan_failed(scan_id, validation_result["failure_reason"], validation_result["failure_message"])
        _log_perf(scan_id, "directus_writeback_ms", _elapsed_ms(writeback_started))
        _log_perf(scan_id, "total_process_ms", _elapsed_ms(total_started))
        return {"status": SCAN_STATUS_FAILED, **validation_result}

    media = _merge_media(scan_context)
    task = _merge_task(scan_context)
    identifiers = _identifier_payload(scan_context)
    baseline_rows = _baseline_rows_for_member(identifiers.get("member_id"), identifiers.get("business_profile_id"))
    if len(baseline_rows) > 1:
        logger.warning(
            "baseline_duplicate_rows member_id=%s business_profile_id=%s count=%s",
            identifiers.get("member_id"),
            identifiers.get("business_profile_id"),
            len(baseline_rows),
        )
    baseline = baseline_rows[0] if len(baseline_rows) == 1 else None
    baseline_status = baseline_status_payload(baseline)
    expected_phrase = _expected_phrase(scan_context)

    temp_files: list[str] = []
    analysis_executor: concurrent.futures.ThreadPoolExecutor | None = None
    transcript: str | None = None
    phrase_score: float | None = None
    phrase_status = "not_required"
    try:
        _log_step(scan_id, "media_download_start")
        stage_started = time.perf_counter()
        download_executor = concurrent.futures.ThreadPoolExecutor(max_workers=3, thread_name_prefix="media-download")
        download_futures = {
            "image": download_executor.submit(_resolve_media_input, media.image, ".jpg", "image", allow_url=False, allow_local_path=False),
            "audio": download_executor.submit(_resolve_media_input, media.audio, ".bin", "audio", allow_url=False, allow_local_path=False),
            "video": download_executor.submit(_resolve_media_input, media.video, ".mp4", "video", allow_url=False, allow_local_path=False),
        }
        done, not_done = concurrent.futures.wait(
            list(download_futures.values()),
            timeout=FAST_SCAN_DOWNLOAD_TIMEOUT_SECONDS if FAST_SCAN_MODE else None,
            return_when=concurrent.futures.ALL_COMPLETED,
        )
        if not_done:
            logger.warning(
                "media_download_timeout scan_id=%s timeout_seconds=%s pending=%s",
                scan_id,
                FAST_SCAN_DOWNLOAD_TIMEOUT_SECONDS,
                len(not_done),
            )
        image_path = audio_path = video_path = None
        image_temp = audio_temp = video_temp = False
        for name, future in download_futures.items():
            if future not in done:
                continue
            try:
                path, is_temp = future.result()
            except Exception as exc:
                logger.warning("media_download_failed scan_id=%s media=%s error=%s", scan_id, name, exc)
                path, is_temp = None, False
            if name == "image":
                image_path, image_temp = path, is_temp
            elif name == "audio":
                audio_path, audio_temp = path, is_temp
            else:
                video_path, video_temp = path, is_temp
        download_executor.shutdown(wait=False, cancel_futures=True)
        for path, is_temp in [(image_path, image_temp), (audio_path, audio_temp), (video_path, video_temp)]:
            if path and is_temp:
                temp_files.append(path)
        if audio_path and _should_convert_audio(audio_path):
            converted = _convert_audio_to_wav(audio_path)
            temp_files.append(converted)
            audio_path = converted
        resolved_media = Media(image=image_path, audio=audio_path, video=video_path)
        _log_perf(scan_id, "media_download_ms", _elapsed_ms(stage_started))
        _log_step(
            scan_id,
            "media_download_done",
            has_video=bool(video_path),
            has_audio=bool(audio_path),
            has_image=bool(image_path),
        )

        _log_step(scan_id, "video_validation_start")
        media_validation_started = time.perf_counter()
        stage_started = media_validation_started
        analysis_executor = concurrent.futures.ThreadPoolExecutor(max_workers=3, thread_name_prefix="media-analysis")
        analysis_futures = {
            "video": analysis_executor.submit(
                _timed_safe_analyze,
                scan_id,
                "video_validation_ms",
                _analyze_video_file,
                resolved_media.video,
                "video_missing",
            ),
            "image": analysis_executor.submit(
                _timed_safe_analyze,
                scan_id,
                "image_validation_ms",
                _analyze_face_image,
                resolved_media.image,
                "image_missing",
            )
            if resolved_media.image
            else None,
            "audio": analysis_executor.submit(
                _timed_safe_analyze,
                scan_id,
                "audio_validation_ms",
                _analyze_audio_file,
                resolved_media.audio,
                "audio_missing",
            ),
        }
        done, not_done = concurrent.futures.wait(
            [future for future in analysis_futures.values() if future is not None],
            timeout=MEDIA_VALIDATION_WALL_TIMEOUT_SECONDS,
            return_when=concurrent.futures.ALL_COMPLETED,
        )
        if not_done:
            wall_elapsed_ms = _elapsed_ms(media_validation_started)
            logger.warning(
                "media_analysis_timeout scan_id=%s timeout_seconds=%s pending=%s",
                scan_id,
                MEDIA_VALIDATION_WALL_TIMEOUT_SECONDS,
                len(not_done),
            )
            for name, future in analysis_futures.items():
                if future is not None and future in not_done:
                    _log_perf(scan_id, f"{name}_validation_ms", wall_elapsed_ms)
        try:
            video_result = analysis_futures["video"].result() if analysis_futures["video"] in done else _analysis_timeout_placeholder("video")
        except Exception as exc:
            logger.warning("video_analysis_failed scan_id=%s error=%s", scan_id, exc)
            video_result = _analysis_timeout_placeholder("video")
        _log_step(scan_id, "video_validation_done", quality_score=((video_result.get("details") or {}).get("visual_quality_score")))

        _log_step(scan_id, "face_validation_start")
        stage_started = time.perf_counter()
        video_details = (video_result.get("details") or {})
        if analysis_futures.get("image") and analysis_futures["image"] in done:
            try:
                image_result = analysis_futures["image"].result()
            except Exception as exc:
                logger.warning("image_analysis_failed scan_id=%s error=%s", scan_id, exc)
                image_result = _analysis_timeout_placeholder("image")
        elif analysis_futures.get("image"):
            image_result = _analysis_timeout_placeholder("image")
        else:
            image_result = None
        _log_step(
            scan_id,
            "face_validation_done",
            video_face_ratio=video_details.get("face_or_subject_visibility") or video_details.get("face_rate"),
            image_face_detected=((image_result or {}).get("details") or {}).get("face_detected"),
        )

        _log_step(scan_id, "audio_validation_start")
        stage_started = time.perf_counter()
        try:
            audio_result = analysis_futures["audio"].result() if analysis_futures["audio"] in done else _analysis_timeout_placeholder("audio")
        except Exception as exc:
            logger.warning("audio_analysis_failed scan_id=%s error=%s", scan_id, exc)
            audio_result = _analysis_timeout_placeholder("audio")
        audio_timings = ((audio_result.get("details") or {}).get("timings_ms") or {}) if isinstance(audio_result, dict) else {}
        _log_perf(scan_id, "audio_decode_ms", audio_timings.get("audio_decode_ms"))
        _log_perf(scan_id, "audio_quality_ms", audio_timings.get("audio_quality_ms"))
        _log_perf(scan_id, "voice_activity_ms", audio_timings.get("voice_activity_ms"))
        audio_remaining_ms = max(0, _elapsed_ms(stage_started) - int(audio_timings.get("audio_decode_ms") or 0) - int(audio_timings.get("audio_quality_ms") or 0) - int(audio_timings.get("voice_activity_ms") or 0))
        if audio_remaining_ms:
            logger.info("[PERF] audio_validation_overhead_ms scan_id=%s value=%s", scan_id, audio_remaining_ms)
        _log_step(scan_id, "audio_validation_done", quality_score=((audio_result.get("details") or {}).get("audio_quality_score")))
        _log_perf(scan_id, "media_validation_wall_ms", _elapsed_ms(media_validation_started))
        _shutdown_executor(analysis_executor)
        analysis_executor = None

        _log_step(scan_id, "phrase_validation_start")
        stage_started = time.perf_counter()
        if expected_phrase and resolved_media.audio:
            if VALIDATION_POLICY.require_phrase_match:
                transcript, phrase_status = _transcribe_audio_file_optional(resolved_media.audio)
            else:
                phrase_status = "skipped_optional"
        elif expected_phrase:
            phrase_status = "audio_missing"
        phrase_expected_for_validation = expected_phrase if (VALIDATION_POLICY.require_phrase_match or transcript) else None
        phrase_validation = validate_scan_inputs(
            policy=VALIDATION_POLICY,
            media=resolved_media,
            video_result=video_result,
            audio_result=audio_result,
            image_result=image_result,
            expected_phrase=phrase_expected_for_validation,
            transcript=transcript,
        )
        phrase_score = phrase_validation["quality_scores"].get("phrase_match")
        phrase_failure_reason = None
        if VALIDATION_POLICY.require_phrase_match and not transcript:
            phrase_failure_reason = phrase_validation.get("failure_reason") or FAILURE_REASON_TRANSCRIPTION_FAILED
        _log_perf(scan_id, "phrase_optional_ms", _elapsed_ms(stage_started))
        _log_step(scan_id, "phrase_validation_done", transcript_present=bool(transcript), phrase_match_score=phrase_score, phrase_status=phrase_status)

        _log_step(scan_id, "image_validation_start")
        _log_step(
            scan_id,
            "image_validation_done",
            quality_score=((image_result or {}).get("details") or {}).get("image_quality_score"),
        )

        raw_signals = {
            "camera": image_result or {"score": None, "details": {"status": "missing"}},
            "video": video_result,
            "voice": audio_result,
        }
        quality_result = assess_quality(
            raw_signals,
            task,
            speech_required=VALIDATION_POLICY.require_phrase_match or bool(expected_phrase),
        )
        combined_warnings = []
        combined_warnings.extend(quality_result.get("warnings") or [])
        combined_warnings.extend(phrase_validation.get("warnings") or [])
        combined_warnings.extend(phrase_validation.get("critical_errors") or [])
        if "unreadable_media" in (phrase_validation.get("critical_errors") or []):
            combined_warnings.append("missing_media")
        quality_result["warnings"] = list(dict.fromkeys(warning for warning in combined_warnings if warning))
        if phrase_validation.get("warnings") or phrase_validation.get("critical_errors"):
            quality_result["weak"] = True
            quality_result["status"] = "weak"
        if phrase_validation.get("failure_reason") in {"missing_media", "unreadable_media"}:
            quality_result["failure_reason"] = "missing_media"
            quality_result["retake_required"] = True
            quality_result["suggested_action"] = "rescan_recommended"

        valid_modalities = [
            modality
            for modality in ["video", "audio", "image"]
            if (quality_result.get("media_quality") or {}).get(modality, {}).get("usable")
        ]
        timed_out_modalities = [
            modality
            for modality, future in analysis_futures.items()
            if future is not None and future in not_done
        ]
        running_modalities = list(timed_out_modalities)
        all_workers_terminal = not running_modalities
        terminal_failure_reason, _required_modalities = _required_modality_gate(
            quality_result,
            timed_out_modalities=running_modalities,
        )
        terminal_reason = "validation_passed"
        if phrase_failure_reason and terminal_failure_reason is None:
            terminal_failure_reason = phrase_failure_reason
            terminal_reason = "phrase_validation_failed"
        if terminal_failure_reason == FAILURE_REASON_AUDIO_VALIDATION_TIMEOUT:
            terminal_reason = "audio_validation_timeout"
        elif terminal_failure_reason:
            terminal_reason = "validation_timeout" if timed_out_modalities else "validation_no_reliable_evidence"
        elif quality_result.get("usable_modalities", 0) <= 0:
            terminal_failure_reason = quality_result.get("failure_reason") or FAILURE_REASON_LOW_QUALITY_MEDIA
            terminal_reason = "validation_timeout" if timed_out_modalities else "validation_no_reliable_evidence"
        if not all_workers_terminal and terminal_failure_reason is None:
            terminal_failure_reason = FAILURE_REASON_ANALYSIS_EXCEPTION
            terminal_reason = "workers_not_terminal"
        _log_validation_decision(
            scan_id,
            valid_modalities=valid_modalities,
            timed_out_modalities=timed_out_modalities,
            terminal_reason=terminal_reason,
        )
        _log_validation_lifecycle(scan_id, all_workers_terminal=all_workers_terminal, running_modalities=running_modalities)

        if terminal_failure_reason:
            validation_result = fail_validation(terminal_failure_reason, warnings=quality_result.get("warnings"))
            _log_step(scan_id, "validation_failed", reason=validation_result["failure_reason"], scores=phrase_validation["quality_scores"])
            _log_step(scan_id, "directus_writeback_start")
            writeback_started = time.perf_counter()
            _shutdown_executor(analysis_executor)
            analysis_executor = None
            _mark_scan_failed(scan_id, validation_result["failure_reason"], validation_result["failure_message"])
            _log_perf(scan_id, "directus_writeback_ms", _elapsed_ms(writeback_started))
            _log_perf(scan_id, "total_process_ms", _elapsed_ms(total_started))
            return {"status": SCAN_STATUS_FAILED, **validation_result}

        critical_errors = phrase_validation.get("critical_errors") or []
        if critical_errors and not _critical_validation_errors_allow_result(critical_errors):
            validation_result = fail_validation(critical_errors[0], warnings=quality_result.get("warnings"))
            _log_step(scan_id, "validation_failed", reason=validation_result["failure_reason"], scores=phrase_validation["quality_scores"])
            _log_step(scan_id, "directus_writeback_start")
            writeback_started = time.perf_counter()
            _shutdown_executor(analysis_executor)
            analysis_executor = None
            _mark_scan_failed(scan_id, validation_result["failure_reason"], validation_result["failure_message"])
            _log_perf(scan_id, "directus_writeback_ms", _elapsed_ms(writeback_started))
            _log_perf(scan_id, "total_process_ms", _elapsed_ms(total_started))
            return {"status": SCAN_STATUS_FAILED, **validation_result}

        if terminal_failure_reason or _face_eye_evidence_unreliable(quality_result.get("warnings"), video_details, ((image_result or {}).get("details") or {})):
            quality_result["failure_reason"] = terminal_failure_reason or quality_result.get("failure_reason") or FAILURE_REASON_LOW_QUALITY_MEDIA
            quality_result["retake_required"] = True
            quality_result["suggested_action"] = "rescan_recommended"
            quality_result["status"] = "weak"
            quality_result["weak"] = True
            _log_step(
                scan_id,
                "validation_retake_required",
                reason=terminal_reason if terminal_failure_reason else "unreliable_face_eye_evidence",
                warnings=quality_result.get("warnings"),
            )

        _log_step(scan_id, "validation_completed", scores=phrase_validation["quality_scores"], warnings=quality_result.get("warnings"))
        _log_step(scan_id, "analysis_start")
        stage_started = time.perf_counter()
        feature_map, _ = features_from_signals(raw_signals, task=task)
        feature_vector = vector_from_features(feature_map)
        ml_result = ml_runtime.predict(feature_vector)
        preview_result = compute_result(
            signals=raw_signals,
            task=task,
            previous_confidence=None,
            baseline=baseline,
            baseline_used=False,
            quality=quality_result,
            ml_result=ml_result,
        )
        baseline_used = baseline_ready_for_personalized_scoring(
            baseline,
            quality_result=quality_result,
            validation_result=phrase_validation,
            result=preview_result,
            task=task,
            expected_phrase=expected_phrase,
            unique_row=len(baseline_rows) == 1,
        )
        result = compute_result(
            signals=raw_signals,
            task=task,
            previous_confidence=None,
            baseline=baseline,
            baseline_used=baseline_used,
            quality=quality_result,
            ml_result=ml_result,
        )
        result.update(
            {
                "spoken_transcript": transcript,
                "expected_phrase": expected_phrase,
                "phrase_match_score": phrase_score,
                "audio_quality_score": phrase_validation["quality_scores"].get("audio"),
                "video_quality_score": phrase_validation["quality_scores"].get("video"),
                "image_quality_score": phrase_validation["quality_scores"].get("image"),
                "validation_warnings": quality_result.get("warnings"),
            }
        )
        baseline_eligibility = evaluate_baseline_eligibility(
            quality_result=quality_result,
            validation_result=phrase_validation,
            result=result,
            signals=raw_signals,
            expected_phrase=expected_phrase,
            task=task,
            manually_unreliable=False,
        )
        result.update(
            {
                "capture_quality_score": baseline_eligibility.get("capture_quality_score"),
                "measurement_reliability_score": baseline_eligibility.get("measurement_reliability_score"),
                "personal_deviation_score": _baseline_personal_deviation_score(result),
                "task_completion_status": baseline_eligibility.get("task_completion_status"),
                "baseline_status_at_inference": baseline_status.get("baseline_status"),
                "baseline_confidence": baseline_status.get("baseline_confidence"),
                "baseline_eligible": baseline_eligibility.get("eligible"),
                "hard_gates_triggered": baseline_eligibility.get("hard_gates_triggered"),
                "explainable_reasons": baseline_eligibility.get("reasons"),
            }
        )
        result["result_status"] = _result_status_from_outcome(
            quality_result=quality_result,
            validation_result=phrase_validation,
            result=result,
            baseline_eligibility=baseline_eligibility,
        )
        if not _result_has_valid_evidence(result):
            validation_result = fail_validation(quality_result.get("failure_reason") or FAILURE_REASON_LOW_QUALITY_MEDIA, warnings=quality_result.get("warnings"))
            _log_validation_decision(
                scan_id,
                valid_modalities=valid_modalities,
                timed_out_modalities=timed_out_modalities,
                terminal_reason="result_without_real_evidence",
            )
            _log_step(scan_id, "validation_failed", reason=validation_result["failure_reason"], scores=phrase_validation["quality_scores"])
            _log_step(scan_id, "directus_writeback_start")
            writeback_started = time.perf_counter()
            _shutdown_executor(analysis_executor)
            analysis_executor = None
            _mark_scan_failed(scan_id, validation_result["failure_reason"], validation_result["failure_message"])
            _log_perf(scan_id, "directus_writeback_ms", _elapsed_ms(writeback_started))
            _log_perf(scan_id, "total_process_ms", _elapsed_ms(total_started))
            return {"status": SCAN_STATUS_FAILED, **validation_result}
        _log_perf(scan_id, "analysis_ms", _elapsed_ms(stage_started))
        _log_step(scan_id, "analysis_done", risk_level=result.get("risk_level"), confidence=result.get("confidence"))

        if identifiers.get("member_id") and identifiers.get("business_profile_id") and baseline_eligibility.get("eligible"):
            if len(baseline_rows) > 1:
                logger.warning(
                    "baseline_write_skipped_duplicate member_id=%s business_profile_id=%s scan_id=%s",
                    identifiers.get("member_id"),
                    identifiers.get("business_profile_id"),
                    scan_id,
                )
            else:
                try:
                    baseline_payload = baseline_signal_payload(
                        baseline,
                        signals=raw_signals,
                        scanned_at=_scan_timestamp(scan_context),
                    )
                    baseline_payload["member"] = identifiers["member_id"]
                    baseline_payload["business_profile"] = identifiers["business_profile_id"]
                    directus.upsert_employee_baseline(_relation_id((baseline or {}).get("id")), baseline_payload)
                except Exception as exc:
                    logger.warning("baseline_write_failed scan_id=%s optional=true error=%s", scan_id, exc)

        internal_analysis = sanitize_payload(
            {
                "quality": quality_result,
                "signals": raw_signals,
                "ml": ml_result,
                "baseline_status_before": baseline_status,
                "validation": phrase_validation,
                "phrase_optional": {
                    "status": phrase_status,
                    "timeout_seconds": OPTIONAL_PHRASE_TIMEOUT_SECONDS,
                    "transcript_present": bool(transcript),
                    "blocking_required": VALIDATION_POLICY.require_phrase_match,
                },
            }
        )

        _log_step(scan_id, "directus_writeback_start")
        writeback_started = time.perf_counter()
        writeback_status = _write_success(
            scan_id=scan_id,
            scan_context=scan_context,
            identifiers=identifiers,
            result=result,
            internal_analysis=internal_analysis,
        )
        _log_perf(scan_id, "directus_writeback_ms", _elapsed_ms(writeback_started))
        _log_step(scan_id, "directus_writeback_done", writeback_status=writeback_status)
        _log_perf(scan_id, "total_process_ms", _elapsed_ms(total_started))
        return {"status": SCAN_STATUS_COMPLETED, "failure_reason": None, "writeback_status": writeback_status}
    except ProcessingError as exc:
        validation_result = fail_validation(exc.reason)
        _log_step(scan_id, "validation_failed", reason=validation_result["failure_reason"])
        _log_step(scan_id, "directus_writeback_start")
        writeback_started = time.perf_counter()
        _mark_scan_failed(scan_id, validation_result["failure_reason"], validation_result["failure_message"])
        _log_perf(scan_id, "directus_writeback_ms", _elapsed_ms(writeback_started))
        _log_perf(scan_id, "total_process_ms", _elapsed_ms(total_started))
        return {"status": SCAN_STATUS_FAILED, **validation_result}
    finally:
        if analysis_executor is not None:
            analysis_executor.shutdown(wait=True, cancel_futures=True)
        for path in temp_files:
            remove_temp_file(path)


def process_scan_background(scan_id: str) -> None:
    try:
        result = _process_scan_sync(scan_id)
        _log_step(
            scan_id,
            "background_process_done",
            final_status=result.get("status"),
            failure_reason=result.get("failure_reason"),
        )
    except Exception as exc:
        logger.error("BACKGROUND ERROR scan_id=%s error=%s", scan_id, exc)
        logger.error(traceback.format_exc())
        _log_step(scan_id, "directus_writeback_start")
        _mark_scan_failed(scan_id, FAILURE_REASON_ANALYSIS_EXCEPTION, failure_message(FAILURE_REASON_ANALYSIS_EXCEPTION))


@app.get("/health")
def health():
    return _model_health()


def debug_scan(scan_id: str):
    scan_context = _resolve_scan_context(scan_id)
    media_row = scan_context.get("scan_media")
    baseline = _baseline_for_member(
        _relation_id(scan_context.get("member")),
        _relation_id(scan_context.get("business_profile")),
    )
    return {
        "scan_id": scan_id,
        "media_row_found": media_row is not None,
        "video_file_id": _relation_id((media_row or {}).get("video_file")),
        "audio_file_id": _relation_id((media_row or {}).get("audio_file")),
        "thumbnail_id": _relation_id((media_row or {}).get("thumbnail")),
        "member": _relation_id(scan_context.get("member")),
        "business_profile": _relation_id(scan_context.get("business_profile")),
        "user": _relation_id(scan_context.get("user")),
        "baseline_scan_count": int((baseline or {}).get("scan_count", 0)),
        "request_source": scan_context.get("request_source"),
        "status": scan_context.get("status"),
    }


if DEBUG_SCAN_ENDPOINT_ENABLED:
    app.get("/debug/scan/{scan_id}")(debug_scan)


@app.get("/baseline/status", response_model=BaselineStatusResponse)
def baseline_status(
    member_id: str = Query(...),
    business_profile_id: str = Query(...),
):
    if not directus.is_configured():
        raise HTTPException(status_code=500, detail="Directus credentials are not configured")
    baseline = _baseline_for_member(member_id, business_profile_id)
    return baseline_status_payload(baseline)


@app.post("/baseline")
def set_baseline(
    req: BaselineRequest,
    authorization: str | None = Header(default=None),
):
    if not directus.is_configured():
        raise HTTPException(status_code=500, detail="Directus credentials are not configured")
    if not req.scan_id:
        raise HTTPException(status_code=422, detail="scan_id_required")
    authenticated_user_id = _authenticate_process_user(authorization, req.scan_id.strip())
    scan_context = _resolve_scan_auth_context(req.scan_id.strip())
    _authorize_scan_access(scan_context, authenticated_user_id)
    scan_context = _resolve_scan_context(req.scan_id.strip())
    identifiers = _identifier_payload(scan_context)
    if not identifiers.get("member_id") or not identifiers.get("business_profile_id"):
        raise HTTPException(status_code=422, detail="scan_identifiers_missing")
    media = _merge_media(scan_context)
    baseline_rows = _baseline_rows_for_member(identifiers["member_id"], identifiers["business_profile_id"])
    if len(baseline_rows) > 1:
        logger.warning(
            "baseline_duplicate_rows member_id=%s business_profile_id=%s count=%s",
            identifiers["member_id"],
            identifiers["business_profile_id"],
            len(baseline_rows),
        )
        raise HTTPException(status_code=409, detail="duplicate_baseline_rows")
    baseline = baseline_rows[0] if baseline_rows else None
    if req.media:
        requested = _model_to_dict(req.media)
        trusted = _model_to_dict(media)
        for field_name in ["image", "audio", "video"]:
            candidate = requested.get(field_name)
            if candidate and str(candidate) != str(trusted.get(field_name)):
                raise HTTPException(status_code=422, detail="manual_baseline_requires_directus_media")
    expected_phrase = _expected_phrase(scan_context)
    signals, temp_files = _analyze_media("baseline", media)
    try:
        task = _merge_task(scan_context)
        quality_result = assess_quality(
            signals,
            task,
            speech_required=VALIDATION_POLICY.require_phrase_match or bool(expected_phrase),
        )
        if not quality_result["passed"]:
            raise HTTPException(status_code=422, detail=quality_result["failure_reason"] or FAILURE_REASON_LOW_QUALITY_MEDIA)
        transcript = None
        if expected_phrase:
            try:
                audio_path, audio_temp = _resolve_media_input(media.audio, ".bin", "audio", allow_url=False, allow_local_path=False)
                if audio_path and audio_temp:
                    temp_files.append(audio_path)
                if audio_path and _should_convert_audio(audio_path):
                    converted = _convert_audio_to_wav(audio_path)
                    temp_files.append(converted)
                    audio_path = converted
                transcript = _transcribe_audio_file(audio_path) if audio_path else None
            except Exception:
                transcript = None
        phrase_validation = validate_scan_inputs(
            policy=VALIDATION_POLICY,
            media=media,
            video_result=signals.get("video"),
            audio_result=signals.get("voice"),
            image_result=signals.get("camera"),
            expected_phrase=expected_phrase,
            transcript=transcript,
        )
        result = compute_result(
            signals=signals,
            task=task,
            previous_confidence=None,
            baseline=baseline,
            baseline_used=False,
            quality=quality_result,
            ml_result=None,
        )
        eligibility = evaluate_baseline_eligibility(
            quality_result=quality_result,
            validation_result=phrase_validation,
            result=result,
            signals=signals,
            expected_phrase=expected_phrase,
            task=task,
            manually_unreliable=req.manually_unreliable,
        )
        if not eligibility["eligible"]:
            raise HTTPException(status_code=422, detail={"reason": "baseline_ineligible", "reasons": eligibility["reasons"]})
        payload = baseline_signal_payload(
            baseline,
            signals=signals,
            scanned_at=_utc_now(),
        )
        payload["member"] = identifiers["member_id"]
        payload["business_profile"] = identifiers["business_profile_id"]
        updated = directus.upsert_employee_baseline(_relation_id((baseline or {}).get("id")), payload)
        return {
            "scan_id": req.scan_id,
            "member_id": identifiers["member_id"],
            "business_profile_id": identifiers["business_profile_id"],
            "baseline": updated,
            "baseline_status": baseline_status_payload(updated),
            "baseline_eligible": True,
            "model_version": MODEL_VERSION,
        }
    finally:
        for path in temp_files:
            remove_temp_file(path)


@app.post("/process")
def process_scan(
    req: ScanRequest,
    background_tasks: BackgroundTasks,
    authorization: str | None = Header(default=None),
):
    scan_id = (req.scan_id or "").strip()
    logger.info("RECEIVED /process scan_id=%s", scan_id)

    try:
        authenticated_user_id = _authenticate_process_user(authorization, scan_id)
        if not scan_id:
            return _build_scan_result_response(ok=False, scan_id=scan_id, error="invalid_scan_id", status_code=422)
        scan_context = _resolve_scan_auth_context(scan_id)
        _authorize_scan_access(scan_context, authenticated_user_id)
    except HTTPException as exc:
        error = "scan_not_found" if exc.status_code == 404 else "scan_context_failed"
        if exc.status_code == 401:
            error = "invalid_authorization"
        elif exc.status_code == 403:
            error = "active_membership_required"
        return _build_scan_result_response(ok=False, scan_id=scan_id, error=error, status_code=exc.status_code)

    current_status = (scan_context.get("status") or "").strip()
    if current_status == SCAN_STATUS_PENDING:
        return _build_scan_result_response(
            ok=False,
            scan_id=scan_id,
            error="scan_not_ready",
            current_status=current_status,
            status_code=409,
        )
    if current_status == SCAN_STATUS_PROCESSING:
        return _build_scan_result_response(ok=True, scan_id=scan_id, status="already_processing", status_code=202)
    if current_status == SCAN_STATUS_COMPLETED:
        return _build_scan_result_response(ok=True, scan_id=scan_id, status="already_completed", status_code=200)
    if current_status == SCAN_STATUS_FAILED:
        return _build_scan_result_response(ok=True, scan_id=scan_id, status="already_failed", status_code=200)
    if current_status != SCAN_STATUS_MEDIA_READY:
        return _build_scan_result_response(
            ok=False,
            scan_id=scan_id,
            error="invalid_scan_status",
            current_status=current_status,
            status_code=409,
        )

    try:
        _ensure_scan_media_ready(scan_id)
    except HTTPException as exc:
        error = exc.detail if exc.detail == "scan_media_not_ready" else "invalid_scan_status"
        return _build_scan_result_response(ok=False, scan_id=scan_id, error=error, status_code=exc.status_code)

    update_payload = _wellness_scan_update_payload(
        {
            "status": SCAN_STATUS_PROCESSING,
            "processing_started_at": _utc_now(),
            "failure_reason": None,
            "failure_message": None,
        }
    )
    if "processing_attempts" in scan_context:
        update_payload["processing_attempts"] = int(scan_context.get("processing_attempts") or 0) + 1

    try:
        directus.update_wellness_scan(scan_id, update_payload)
    except Exception as exc:
        logger.exception("processing_state_update_failed scan_id=%s error=%s", scan_id, exc)
        return _build_scan_result_response(ok=False, scan_id=scan_id, error="processing_state_update_failed", status_code=500)

    background_tasks.add_task(process_scan_background, scan_id)
    return _build_scan_result_response(ok=True, scan_id=scan_id, status="accepted", status_code=202)
