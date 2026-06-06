from __future__ import annotations

from datetime import datetime, timezone
import math
import os
import shutil
import subprocess
import tempfile
import traceback
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import requests

from baseline import baseline_ready_for_scoring, baseline_signal_payload, baseline_status_payload
from config import MAX_DOWNLOAD_BYTES, MODEL_VERSION
from directus_client import DirectusClient
from logger import get_logger
from ml.features import features_from_signals, vector_from_features
from ml.runtime import MLRuntime
from quality import assess_quality
from scoring import compute_result
from utils import directus_auth_headers, download_temp_file, is_url, remove_temp_file, sanitize_payload, sanitize_text
from validation import ValidationPolicy, fail_validation, failure_message, validate_scan_inputs


app = FastAPI()
logger = get_logger()
ml_runtime = MLRuntime()
ml_runtime.load()
directus = DirectusClient()

VALIDATION_POLICY = ValidationPolicy.from_env()

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
]

SCAN_RESULT_NUMERIC_FIELDS: dict[str, bool] = {
    "readiness_score": True,
    "confidence": False,
    "camera_confidence": False,
    "voice_confidence": False,
    "task_performance_score": True,
    "confidence_drift": False,
    "phrase_match_score": False,
    "audio_quality_score": False,
    "video_quality_score": False,
    "image_quality_score": False,
}

SCAN_RESULT_CHOICE_ALIASES: dict[str, dict[str, list[str]]] = {
    "risk_level": {
        "stable": ["Stable"],
        "low_focus": ["Low Focus"],
        "elevated_fatigue": ["Elevated Fatigue"],
        "high_risk": ["High Risk"],
        "unknown": ["Unknown"],
    },
    "suggested_action": {
        "continue_normal_activity": ["Continue Normal Activity"],
        "review_required": ["Review Required"],
        "rescan_recommended": ["Rescan Recommended"],
        "rest_advised": ["Rest Advised"],
        "manager_review": ["Manager Review"],
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
    member_id: str
    business_profile_id: str
    media: Media
    task: Task | None = None


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

    for field_name in ["risk_level", "suggested_action"]:
        if field_name in candidate:
            candidate[field_name] = _coerce_scan_result_choice(field_name, candidate.get(field_name))

    candidate["explanation"] = _safe_string(candidate.get("explanation"), max_len=500, fallback="Analysis completed.")
    candidate["ai_model_version"] = _safe_string(candidate.get("ai_model_version"), max_len=100, fallback=MODEL_VERSION)

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


def _log_step(scan_id: str, step: str, **details: Any) -> None:
    if details:
        logger.info("scan_id=%s step=%s details=%s", scan_id, step, sanitize_payload(details))
        return
    logger.info("scan_id=%s step=%s", scan_id, step)


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
    cmd = ["ffmpeg", "-y", "-i", path, "-ac", "1", "-ar", "16000", out]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return out


def _download_directus_asset(asset_id: str, suffix: str) -> str:
    if not directus.is_configured():
        raise ProcessingError(FAILURE_REASON_DIRECTUS_DOWNLOAD_FAILED, "Directus credentials are not configured")
    url = f"{directus.base_url}/assets/{asset_id}"
    response = requests.get(url, headers=directus_auth_headers(url), timeout=(10, 30), stream=True)
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
            path = download_temp_file(value, suffix)
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
    if not directus.is_configured() or not member_id or not business_profile_id:
        return None
    try:
        return directus.get_employee_baseline(member_id, business_profile_id)
    except Exception as exc:
        logger.warning("baseline_fetch_failed member_id=%s error=%s", member_id, exc)
        return None


def _identifier_payload(scan_context: dict) -> dict:
    return {
        "user_id": _relation_id(scan_context.get("user")),
        "member_id": _relation_id(scan_context.get("member")),
        "business_profile_id": _relation_id(scan_context.get("business_profile")),
        "department_id": _relation_id(scan_context.get("department")),
    }


def _expected_phrase(scan_context: dict) -> str | None:
    return _safe_string(scan_context.get("expected_phrase"), max_len=500)


def _wellness_scan_update_payload(payload: dict[str, Any]) -> dict[str, Any]:
    filtered = directus.filter_payload_fields("wellness_scans", payload)
    fallback_field = directus.first_supported_field("wellness_scans", ["failure_message", "user_message"])
    if "failure_message" in payload and "failure_message" not in filtered and fallback_field:
        filtered[fallback_field] = payload["failure_message"]
    cleaned = sanitize_payload(filtered)
    for field_name in ["failure_reason", "failure_message", "completed_at", "ai_model_version", "status"]:
        if field_name in filtered and filtered[field_name] is None:
            cleaned[field_name] = None
    if fallback_field and fallback_field in filtered and filtered[fallback_field] is None:
        cleaned[fallback_field] = None
    return cleaned


def _scan_result_payload(payload: dict[str, Any]) -> dict[str, Any]:
    filtered = directus.filter_payload_fields("scan_results", payload)
    metadata_field = directus.first_supported_field("scan_results", ["internal_analysis", "analysis_metadata"])
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
    return sanitize_payload(filtered)


def _safe_analyze(fn, path: str | None, missing_warning: str) -> dict:
    if not path:
        return {"score": None, "details": {"status": "missing", "warnings": [missing_warning]}}
    try:
        result = fn(path)
        return result if isinstance(result, dict) else {"score": None, "details": {"status": "invalid"}}
    except Exception as exc:
        logger.exception("analyzer_error path=%s error=%s", path, exc)
    return {"score": None, "details": {"status": "error", "warnings": [missing_warning]}}


def _analyze_audio_file(path: str | None) -> dict:
    from audio import analyze_audio

    return analyze_audio(path)


def _transcribe_audio_file(path: str) -> str:
    from audio import transcribe_audio

    return transcribe_audio(path)


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
        "readiness_score": None,
        "risk_level": "unknown",
        "confidence": None,
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


def _build_scan_result_payload(scan_id: str, result: dict, internal_analysis: dict) -> dict:
    payload = {
        "scan_id": scan_id,
        "readiness_score": result.get("readiness_score"),
        "risk_level": result.get("risk_level"),
        "confidence": result.get("confidence"),
        "camera_confidence": result.get("camera_confidence"),
        "voice_confidence": result.get("voice_confidence"),
        "task_performance_score": result.get("task_performance_score"),
        "explanation": _safe_string(result.get("explanation"), max_len=500, fallback="Analysis completed."),
        "suggested_action": _safe_string(result.get("suggested_action"), max_len=100, fallback="review_required"),
        "ai_model_version": _safe_string(result.get("ai_model_version"), max_len=100, fallback=MODEL_VERSION),
        "confidence_drift": result.get("confidence_drift"),
        "baseline_used": result.get("baseline_used"),
        "face_metrics": result.get("face_metrics"),
        "voice_metrics": result.get("voice_metrics"),
        "reaction_metrics": result.get("reaction_metrics"),
        "spoken_transcript": result.get("spoken_transcript"),
        "expected_phrase": result.get("expected_phrase"),
        "phrase_match_score": result.get("phrase_match_score"),
        "audio_quality_score": result.get("audio_quality_score"),
        "video_quality_score": result.get("video_quality_score"),
        "image_quality_score": result.get("image_quality_score"),
        "validation_warnings": result.get("validation_warnings"),
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
        payload["internal_analysis"] = internal_analysis
    return _schema_aware_scan_result_payload(payload)


def _write_quality_failure(scan_id: str, quality_result: dict) -> dict:
    return _mark_scan_failed(
        scan_id,
        quality_result.get("failure_reason") or FAILURE_REASON_LOW_QUALITY_MEDIA,
    )


def _write_success(
    *,
    scan_id: str,
    scan_context: dict,
    identifiers: dict,
    result: dict,
    internal_analysis: dict,
) -> dict:
    status: dict[str, Any] = {}
    try:
        write_mode, scan_result = directus.upsert_scan_result(scan_id, _build_scan_result_payload(scan_id, result, internal_analysis))
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
        status["alert"] = "created" if alert else "skipped"
    except Exception as exc:
        logger.warning("alert_write_failed scan_id=%s error=%s", scan_id, exc)
        status["alert"] = f"failed:{exc}"
    return status


def _process_scan_sync(scan_id: str) -> dict[str, Any]:
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
        validation_result = fail_validation(FAILURE_REASON_MISSING_MEDIA)
        _log_step(scan_id, "validation_failed", reason=validation_result["failure_reason"])
        _log_step(scan_id, "directus_writeback_start")
        _mark_scan_failed(scan_id, validation_result["failure_reason"], validation_result["failure_message"])
        return {"status": SCAN_STATUS_FAILED, **validation_result}

    if ml_runtime.local_model_required() and not ml_runtime.is_loaded():
        validation_result = fail_validation(FAILURE_REASON_MODEL_NOT_LOADED)
        _log_step(scan_id, "validation_failed", reason=validation_result["failure_reason"])
        _log_step(scan_id, "directus_writeback_start")
        _mark_scan_failed(scan_id, validation_result["failure_reason"], validation_result["failure_message"])
        return {"status": SCAN_STATUS_FAILED, **validation_result}

    media = _merge_media(scan_context)
    task = _merge_task(scan_context)
    identifiers = _identifier_payload(scan_context)
    baseline = _baseline_for_member(identifiers.get("member_id"), identifiers.get("business_profile_id"))
    baseline_used = baseline_ready_for_scoring(baseline)
    baseline_status = baseline_status_payload(baseline)
    expected_phrase = _expected_phrase(scan_context)

    temp_files: list[str] = []
    transcript: str | None = None
    phrase_score: float | None = None
    try:
        _log_step(scan_id, "media_download_start")
        image_path, image_temp = _resolve_media_input(media.image, ".jpg", "image", allow_url=False, allow_local_path=False)
        audio_path, audio_temp = _resolve_media_input(media.audio, ".bin", "audio", allow_url=False, allow_local_path=False)
        video_path, video_temp = _resolve_media_input(media.video, ".mp4", "video", allow_url=False, allow_local_path=False)
        for path, is_temp in [(image_path, image_temp), (audio_path, audio_temp), (video_path, video_temp)]:
            if path and is_temp:
                temp_files.append(path)
        if audio_path and _should_convert_audio(audio_path):
            converted = _convert_audio_to_wav(audio_path)
            temp_files.append(converted)
            audio_path = converted
        resolved_media = Media(image=image_path, audio=audio_path, video=video_path)
        _log_step(
            scan_id,
            "media_download_done",
            has_video=bool(video_path),
            has_audio=bool(audio_path),
            has_image=bool(image_path),
        )

        _log_step(scan_id, "video_validation_start")
        video_result = _safe_analyze(_analyze_video_file, resolved_media.video, "video_missing")
        _log_step(scan_id, "video_validation_done", quality_score=((video_result.get("details") or {}).get("visual_quality_score")))

        _log_step(scan_id, "face_validation_start")
        video_details = (video_result.get("details") or {})
        image_result = None
        if resolved_media.image:
            image_result = _safe_analyze(_analyze_face_image, resolved_media.image, "image_missing")
        _log_step(
            scan_id,
            "face_validation_done",
            video_face_ratio=video_details.get("face_or_subject_visibility") or video_details.get("face_rate"),
            image_face_detected=((image_result or {}).get("details") or {}).get("face_detected"),
        )

        _log_step(scan_id, "audio_validation_start")
        audio_result = _safe_analyze(_analyze_audio_file, resolved_media.audio, "audio_missing")
        _log_step(scan_id, "audio_validation_done", quality_score=((audio_result.get("details") or {}).get("audio_quality_score")))

        _log_step(scan_id, "phrase_validation_start")
        if VALIDATION_POLICY.require_phrase_match or expected_phrase:
            if expected_phrase:
                try:
                    transcript = _transcribe_audio_file(resolved_media.audio) if resolved_media.audio else None
                except Exception as exc:
                    logger.warning("scan_id=%s step=phrase_transcription_error error=%s", scan_id, exc)
                    transcript = None
        phrase_validation = validate_scan_inputs(
            policy=VALIDATION_POLICY,
            media=resolved_media,
            video_result=video_result,
            audio_result=audio_result,
            image_result=image_result,
            expected_phrase=expected_phrase,
            transcript=transcript,
        )
        phrase_score = phrase_validation["quality_scores"].get("phrase_match")
        _log_step(scan_id, "phrase_validation_done", transcript_present=bool(transcript), phrase_match_score=phrase_score)

        _log_step(scan_id, "image_validation_start")
        _log_step(
            scan_id,
            "image_validation_done",
            quality_score=((image_result or {}).get("details") or {}).get("image_quality_score"),
        )

        if not phrase_validation["passed"]:
            _log_step(scan_id, "validation_failed", reason=phrase_validation["failure_reason"], scores=phrase_validation["quality_scores"])
            _log_step(scan_id, "directus_writeback_start")
            _mark_scan_failed(scan_id, phrase_validation["failure_reason"], phrase_validation["failure_message"])
            return {"status": SCAN_STATUS_FAILED, **phrase_validation}

        raw_signals = {
            "camera": image_result or {"score": None, "details": {"status": "missing"}},
            "video": video_result,
            "voice": audio_result,
        }
        quality_result = assess_quality(raw_signals, task)
        if not quality_result["passed"]:
            validation_result = fail_validation(
                quality_result.get("failure_reason") or FAILURE_REASON_LOW_QUALITY_MEDIA,
                scores=phrase_validation["quality_scores"],
                warnings=quality_result.get("warnings"),
            )
            _log_step(scan_id, "validation_failed", reason=validation_result["failure_reason"], scores=validation_result["quality_scores"])
            _log_step(scan_id, "directus_writeback_start")
            _mark_scan_failed(scan_id, validation_result["failure_reason"], validation_result["failure_message"])
            return {"status": SCAN_STATUS_FAILED, **validation_result}

        _log_step(scan_id, "validation_passed", scores=phrase_validation["quality_scores"])
        _log_step(scan_id, "analysis_start")
        feature_map, _ = features_from_signals(raw_signals, task=task)
        feature_vector = vector_from_features(feature_map)
        ml_result = ml_runtime.predict(feature_vector)
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
                "validation_warnings": phrase_validation.get("warnings"),
            }
        )
        _log_step(scan_id, "analysis_done", risk_level=result.get("risk_level"), confidence=result.get("confidence"))

        if identifiers.get("member_id") and identifiers.get("business_profile_id"):
            try:
                baseline_payload = baseline_signal_payload(
                    baseline,
                    face_score=result["face_metrics"]["face_score"],
                    voice_score=result["voice_metrics"]["voice_score"],
                    reaction_score=result["reaction_metrics"]["reaction_score"],
                    scanned_at=_scan_timestamp(scan_context),
                )
                baseline_payload["member"] = identifiers["member_id"]
                baseline_payload["business_profile"] = identifiers["business_profile_id"]
                directus.upsert_employee_baseline(_relation_id((baseline or {}).get("id")), baseline_payload)
            except Exception as exc:
                logger.warning("baseline_write_failed scan_id=%s error=%s", scan_id, exc)

        internal_analysis = sanitize_payload(
            {
                "quality": quality_result,
                "signals": raw_signals,
                "ml": ml_result,
                "baseline_status_before": baseline_status,
                "validation": phrase_validation,
            }
        )

        _log_step(scan_id, "directus_writeback_start")
        writeback_status = _write_success(
            scan_id=scan_id,
            scan_context=scan_context,
            identifiers=identifiers,
            result=result,
            internal_analysis=internal_analysis,
        )
        _log_step(scan_id, "directus_writeback_done", writeback_status=writeback_status)
        return {"status": SCAN_STATUS_COMPLETED, "failure_reason": None, "writeback_status": writeback_status}
    except ProcessingError as exc:
        validation_result = fail_validation(exc.reason)
        _log_step(scan_id, "validation_failed", reason=validation_result["failure_reason"])
        _log_step(scan_id, "directus_writeback_start")
        _mark_scan_failed(scan_id, validation_result["failure_reason"], validation_result["failure_message"])
        return {"status": SCAN_STATUS_FAILED, **validation_result}
    finally:
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


@app.get("/debug/scan/{scan_id}")
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
def set_baseline(req: BaselineRequest):
    signals, temp_files = _analyze_media("baseline", req.media)
    try:
        quality_result = assess_quality(signals, req.task)
        if not quality_result["passed"]:
            raise HTTPException(status_code=422, detail=quality_result["failure_reason"] or FAILURE_REASON_LOW_QUALITY_MEDIA)
        result = compute_result(
            signals=signals,
            task=req.task,
            previous_confidence=None,
            baseline=None,
            baseline_used=False,
            quality=quality_result,
            ml_result=None,
        )
        baseline = _baseline_for_member(req.member_id, req.business_profile_id)
        payload = baseline_signal_payload(
            baseline,
            face_score=result["face_metrics"]["face_score"],
            voice_score=result["voice_metrics"]["voice_score"],
            reaction_score=result["reaction_metrics"]["reaction_score"],
            scanned_at=_utc_now(),
        )
        payload["member"] = req.member_id
        payload["business_profile"] = req.business_profile_id
        updated = directus.upsert_employee_baseline(_relation_id((baseline or {}).get("id")), payload)
        return {
            "member_id": req.member_id,
            "business_profile_id": req.business_profile_id,
            "baseline": updated,
            "baseline_status": baseline_status_payload(updated),
            "model_version": MODEL_VERSION,
        }
    finally:
        for path in temp_files:
            remove_temp_file(path)


@app.post("/process")
def process_scan(req: ScanRequest, background_tasks: BackgroundTasks):
    scan_id = (req.scan_id or "").strip()
    logger.info("RECEIVED /process scan_id=%s", scan_id)
    if not scan_id:
        return _build_scan_result_response(ok=False, scan_id=scan_id, error="invalid_scan_id", status_code=422)

    try:
        scan_context = _resolve_scan_context(scan_id)
    except HTTPException as exc:
        error = "scan_not_found" if exc.status_code == 404 else "scan_context_failed"
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
