from __future__ import annotations

from datetime import datetime
import os
import shutil
import subprocess
import tempfile
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
import requests

from audio import analyze_audio
from baseline import baseline_ready_for_scoring, baseline_signal_payload, baseline_status_payload
from config import MAX_DOWNLOAD_BYTES, MODEL_VERSION
from directus_client import DirectusClient
from logger import get_logger
from ml.features import features_from_signals, vector_from_features
from ml.runtime import MLRuntime
from quality import assess_quality
from scoring import compute_result
from utils import download_temp_file, is_url, remove_temp_file
from video import analyze_video
from vision import analyze_face


app = FastAPI()


@app.get("/")
def root():
    return {"status": "ok"}


logger = get_logger()
ml_runtime = MLRuntime()
ml_runtime.load()
directus = DirectusClient()


class Media(BaseModel):
    image: str | None = Field(None, description="Local path, URL, or Directus asset ID")
    audio: str | None = Field(None, description="Local path, URL, or Directus asset ID")
    video: str | None = Field(None, description="Local path, URL, or Directus asset ID")


class Task(BaseModel):
    reaction_time: float | None = None
    errors: int | None = None
    attempts: int | None = None


class ScanRequest(BaseModel):
    scan_id: str
    request_id: str | None = None
    media: Media | None = None
    task: Task | None = None
    previous_confidence: float | None = None
    subject_id: str | None = None
    member_id: str | None = None
    business_profile_id: str | None = None
    department_id: str | None = None


class BaselineRequest(BaseModel):
    member_id: str
    business_profile_id: str
    media: Media
    task: Task | None = None


class ProcessResponse(BaseModel):
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


def _relation_id(value: Any) -> Any:
    if isinstance(value, dict):
        return value.get("id", value.get("uuid"))
    return value


def _utc_now() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _convert_audio_to_wav(path: str) -> str:
    if shutil.which("ffmpeg") is None:
        raise HTTPException(status_code=500, detail="ffmpeg not installed")
    fd, out = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    cmd = ["ffmpeg", "-y", "-i", path, "-ac", "1", "-ar", "16000", out]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return out


def _should_convert_audio(path: str) -> bool:
    _, ext = os.path.splitext(path)
    return ext.lower() not in [".wav", ".wave"]


def _download_directus_asset(asset_id: str, suffix: str) -> str:
    if not directus.is_configured():
        raise HTTPException(status_code=500, detail="Directus credentials are not configured")

    url = f"{directus.base_url}/assets/{asset_id}"
    headers = directus._headers()
    response = requests.get(url, headers=headers, timeout=30, stream=True)
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
                raise HTTPException(status_code=400, detail=f"Downloaded file too large: {asset_id}")
            handle.write(chunk)
    return path


def _resolve_media_input(value: str | None, suffix: str):
    if not value:
        return None, False
    if os.path.exists(value):
        return value, False
    if is_url(value):
        try:
            return download_temp_file(value, suffix), True
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to download media: {value}") from exc
    try:
        return _download_directus_asset(value, suffix), True
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to resolve media: {value}") from exc


def _safe_analyze(fn, path):
    if not path:
        return {"score": None, "details": {"status": "missing"}}
    try:
        result = fn(path)
    except Exception:
        return {"score": None, "details": {"status": "error"}}
    if isinstance(result, dict) and "score" in result:
        return result
    return {"score": result, "details": {}}


def _ensure_media_present(media: Media | None) -> None:
    if not media or not any([media.image, media.audio, media.video]):
        raise HTTPException(status_code=422, detail="At least one media input is required.")


def _model_to_dict(model: BaseModel | None) -> dict:
    if not model:
        return {}
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _merge_media(request_media: Media | None, scan_context: dict) -> Media:
    request_values = _model_to_dict(request_media)
    directus_values = scan_context.get("resolved_media", {})
    return Media(
        image=request_values.get("image") or directus_values.get("image"),
        audio=request_values.get("audio") or directus_values.get("audio"),
        video=request_values.get("video") or directus_values.get("video"),
    )


def _merge_task(request_task: Task | None, scan_context: dict) -> Task | None:
    if request_task:
        return request_task
    task_metrics = scan_context.get("task_metrics")
    if not isinstance(task_metrics, dict):
        return None
    return Task(
        reaction_time=task_metrics.get("reaction_time"),
        errors=task_metrics.get("errors"),
        attempts=task_metrics.get("attempts"),
    )


def _scan_timestamp(scan_context: dict) -> str:
    return (
        scan_context.get("completed_at")
        or scan_context.get("started_at")
        or scan_context.get("date_created")
        or _utc_now()
    )


def _analyze_media(media: Media):
    _ensure_media_present(media)
    temp_files = []

    image_path, is_temp = _resolve_media_input(media.image, ".jpg")
    if is_temp:
        temp_files.append(image_path)

    audio_path, is_temp = _resolve_media_input(media.audio, ".bin")
    if is_temp:
        temp_files.append(audio_path)

    if audio_path:
        try:
            if _should_convert_audio(audio_path):
                converted = _convert_audio_to_wav(audio_path)
                temp_files.append(converted)
                audio_path = converted
        except Exception:
            return {
                "camera": {"score": None, "details": {"status": "load_failed"}},
                "video": {"score": None, "details": {"status": "load_failed"}},
                "voice": {"score": None, "details": {"status": "load_failed"}},
            }, temp_files

    video_path, is_temp = _resolve_media_input(media.video, ".mp4")
    if is_temp:
        temp_files.append(video_path)

    camera = _safe_analyze(analyze_face, image_path)
    video = _safe_analyze(analyze_video, video_path)
    voice = _safe_analyze(analyze_audio, audio_path)

    return {"camera": camera, "video": video, "voice": voice}, temp_files


def _resolve_scan_context(scan_id: str) -> dict:
    if not directus.is_configured():
        raise HTTPException(status_code=500, detail="Directus credentials are not configured")
    try:
        return directus.get_scan_context(scan_id)
    except Exception as exc:
        logger.warning("scan_context_failed scan_id=%s error=%s", scan_id, exc)
        raise HTTPException(status_code=404, detail="wellness_scans record not found") from exc


def _identifier_payload(req: ScanRequest, scan_context: dict) -> dict:
    return {
        "user_id": _relation_id(scan_context.get("user")),
        "member_id": req.member_id or _relation_id(scan_context.get("member")),
        "business_profile_id": req.business_profile_id or _relation_id(scan_context.get("business_profile")),
        "department_id": req.department_id or _relation_id(scan_context.get("department")),
    }


def _baseline_for_member(member_id: str | None, business_profile_id: str | None) -> dict | None:
    if not directus.is_configured() or not member_id or not business_profile_id:
        return None
    try:
        return directus.get_employee_baseline(member_id, business_profile_id)
    except Exception as exc:
        logger.warning("baseline_fetch_failed member_id=%s error=%s", member_id, exc)
        return None


def _quality_failure_response(scan_id: str, quality_result: dict, diagnostics: dict, writeback_status: dict) -> dict:
    return {
        "status": "failed",
        "retake_required": True,
        "failure_reason": quality_result.get("failure_reason") or "low_quality_media",
        "readiness_score": None,
        "risk_level": None,
        "confidence": None,
        "camera_confidence": None,
        "voice_confidence": None,
        "task_performance_score": None,
        "baseline_used": False,
        "confidence_drift": None,
        "face_metrics": None,
        "voice_metrics": None,
        "reaction_metrics": None,
        "explanation": "Scan quality was too weak for a reliable readiness result.",
        "suggested_action": quality_result["suggested_action"],
        "ai_model_version": MODEL_VERSION,
        "diagnostics": diagnostics,
        "writeback_status": writeback_status,
    }


def _build_scan_result_payload(scan_id: str, result: dict, internal_analysis: dict) -> dict:
    return {
        "scan_id": scan_id,
        "readiness_score": result["readiness_score"],
        "risk_level": result["risk_level"],
        "confidence": result["confidence"],
        "camera_confidence": result["camera_confidence"],
        "voice_confidence": result["voice_confidence"],
        "task_performance_score": result["task_performance_score"],
        "explanation": result["explanation"],
        "suggested_action": result["suggested_action"],
        "ai_model_version": result["ai_model_version"],
        "confidence_drift": result["confidence_drift"],
        "baseline_used": result["baseline_used"],
        "face_metrics": result["face_metrics"],
        "voice_metrics": result["voice_metrics"],
        "reaction_metrics": result["reaction_metrics"],
        "internal_analysis": internal_analysis,
    }


def _write_quality_failure(scan_id: str, quality_result: dict) -> dict:
    try:
        directus.update_wellness_scan(
            scan_id,
            {
                "status": "failed",
                "failure_reason": quality_result.get("failure_reason") or "low_quality_media",
            },
        )
        return {"wellness_scan": "failed_updated"}
    except Exception as exc:
        logger.warning("writeback_failure_failed scan_id=%s error=%s", scan_id, exc)
        return {"wellness_scan": f"failed:{exc}"}


def _write_success(
    *,
    scan_id: str,
    request_id: str | None,
    scan_context: dict,
    identifiers: dict,
    result: dict,
    internal_analysis: dict,
) -> dict:
    status: dict[str, Any] = {}

    try:
        scan_result = directus.create_scan_result(
            _build_scan_result_payload(scan_id, result, internal_analysis)
        )
        status["scan_result"] = _relation_id(scan_result.get("id")) or "created"
    except Exception as exc:
        logger.warning("scan_result_write_failed scan_id=%s error=%s", scan_id, exc)
        status["scan_result"] = f"failed:{exc}"

    try:
        directus.update_wellness_scan(
            scan_id,
            {
                "status": "completed",
                "completed_at": _utc_now(),
            },
        )
        status["wellness_scan"] = "updated"
    except Exception as exc:
        logger.warning("wellness_scan_update_failed scan_id=%s error=%s", scan_id, exc)
        status["wellness_scan"] = f"failed:{exc}"

    member_id = identifiers.get("member_id")
    if member_id:
        try:
            directus.update_member_last_result(
                member_id,
                {
                    "last_scan_at": _utc_now(),
                    "last_readiness_score": result["readiness_score"],
                    "last_risk_level": result["risk_level"],
                },
            )
            status["member"] = "updated"
        except Exception as exc:
            logger.warning("member_update_failed member_id=%s error=%s", member_id, exc)
            status["member"] = f"failed:{exc}"

    try:
        scan_request = directus.update_scan_request_if_needed(
            request_id=request_id,
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
            confidence=float(result["confidence"]),
            scan_id=scan_id,
            member_id=identifiers.get("member_id"),
            business_profile_id=identifiers.get("business_profile_id"),
            department_id=identifiers.get("department_id"),
            user_id=identifiers.get("user_id"),
        )
        status["alert"] = "created" if alert else "skipped"
        notification_user_id = None
        if alert and notification_user_id:
            notification = directus.create_notification(
                user_id=notification_user_id,
                business_profile_id=identifiers.get("business_profile_id"),
                alert_id=_relation_id(alert.get("id")),
                scan_id=scan_id,
                member_id=identifiers.get("member_id"),
                risk_level=result["risk_level"],
            )
            status["notification"] = "created" if notification else "skipped"
        else:
            status["notification"] = "skipped"
    except Exception as exc:
        logger.warning("alert_write_failed scan_id=%s error=%s", scan_id, exc)
        status["alert"] = f"failed:{exc}"
        status["notification"] = f"failed:{exc}"

    return status


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": MODEL_VERSION,
        "ml_loaded": ml_runtime.is_loaded(),
        "ml_error": ml_runtime.error,
        "directus_configured": directus.is_configured(),
    }


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
    signals, temp_files = _analyze_media(req.media)
    try:
        quality_result = assess_quality(signals, req.task)
        if not quality_result["passed"]:
            raise HTTPException(status_code=422, detail=quality_result["failure_reason"] or "low_quality_media")

        baseline = _baseline_for_member(req.member_id, req.business_profile_id)
        result = compute_result(
            camera_score=signals["camera"].get("score"),
            video_score=signals["video"].get("score"),
            voice_score=signals["voice"].get("score"),
            task=req.task,
            previous_confidence=None,
            baseline=baseline,
            baseline_used=False,
            quality=quality_result,
            ml_result=None,
        )
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


@app.post("/process", response_model=ProcessResponse)
def process_scan(req: ScanRequest):
    scan_context = _resolve_scan_context(req.scan_id)
    media = _merge_media(req.media, scan_context)
    task = _merge_task(req.task, scan_context)
    identifiers = _identifier_payload(req, scan_context)
    media_row = scan_context.get("scan_media")

    try:
        directus.update_wellness_scan(req.scan_id, {"status": "processing"})
    except Exception as exc:
        logger.warning("wellness_scan_processing_update_failed scan_id=%s error=%s", req.scan_id, exc)

    logger.info(
        "process_scan_started scan_id=%s media_row_found=%s video_file_id=%s audio_file_id=%s member_id=%s business_profile_id=%s",
        req.scan_id,
        media_row is not None,
        _relation_id((media_row or {}).get("video_file")),
        _relation_id((media_row or {}).get("audio_file")),
        identifiers.get("member_id"),
        identifiers.get("business_profile_id"),
    )

    try:
        signals, temp_files = _analyze_media(media)
    except HTTPException:
        diagnostics = {
            "scan_id": req.scan_id,
            "media_row_found": media_row is not None,
            "video_file_id": _relation_id((media_row or {}).get("video_file")),
            "audio_file_id": _relation_id((media_row or {}).get("audio_file")),
            "member": identifiers.get("member_id"),
            "business_profile": identifiers.get("business_profile_id"),
            "quality_status": "failed",
        }
        writeback_status = _write_quality_failure(
            req.scan_id,
            {
                "status": "failed",
                "failure_reason": "low_quality_media",
                "suggested_action": "Please retake the scan in better lighting with clear face, voice, and reaction input.",
            },
        )
        return _quality_failure_response(
            req.scan_id,
            {
                "status": "failed",
                "failure_reason": "low_quality_media",
                "suggested_action": "Please retake the scan in better lighting with clear face, voice, and reaction input.",
            },
            diagnostics,
            writeback_status,
        )

    try:
        quality_result = assess_quality(signals, task)
        baseline = _baseline_for_member(identifiers.get("member_id"), identifiers.get("business_profile_id"))
        baseline_status = baseline_status_payload(baseline)
        baseline_used = baseline_ready_for_scoring(baseline)

        feature_map, _ = features_from_signals(signals, task=task)
        feature_vector = vector_from_features(feature_map)
        ml_result = ml_runtime.predict(feature_vector) if ml_runtime.is_loaded() else None

        diagnostics = {
            "scan_id": req.scan_id,
            "media_row_found": media_row is not None,
            "video_file_id": _relation_id((media_row or {}).get("video_file")),
            "audio_file_id": _relation_id((media_row or {}).get("audio_file")),
            "member": identifiers.get("member_id"),
            "business_profile": identifiers.get("business_profile_id"),
            "baseline_scan_count": int((baseline or {}).get("scan_count", 0)),
            "quality_status": quality_result["status"],
            "signals": signals,
            "baseline_status": baseline_status,
            "ml": ml_result,
        }

        logger.info(
            "quality_result scan_id=%s member_id=%s baseline_scan_count=%s quality_status=%s",
            req.scan_id,
            identifiers.get("member_id"),
            diagnostics["baseline_scan_count"],
            quality_result["status"],
        )

        if not quality_result["passed"]:
            writeback_status = _write_quality_failure(req.scan_id, quality_result)
            return _quality_failure_response(req.scan_id, quality_result, diagnostics, writeback_status)

        result = compute_result(
            camera_score=signals["camera"].get("score"),
            video_score=signals["video"].get("score"),
            voice_score=signals["voice"].get("score"),
            task=task,
            previous_confidence=req.previous_confidence,
            baseline=baseline,
            baseline_used=baseline_used,
            quality=quality_result,
            ml_result=ml_result,
        )

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
                updated_baseline = directus.upsert_employee_baseline(
                    _relation_id((baseline or {}).get("id")),
                    baseline_payload,
                )
                diagnostics["baseline_status_after"] = baseline_status_payload(updated_baseline)
            except Exception as exc:
                logger.warning("baseline_write_failed scan_id=%s member_id=%s error=%s", req.scan_id, identifiers.get("member_id"), exc)
                diagnostics["baseline_write_error"] = str(exc)

        internal_analysis = {
            "quality": quality_result,
            "baseline_status_before": baseline_status,
            "ml": ml_result,
            "signals": signals,
            "scan_context": {
                "scan_id": req.scan_id,
                "member": identifiers.get("member_id"),
                "business_profile": identifiers.get("business_profile_id"),
                "department": identifiers.get("department_id"),
                "user": identifiers.get("user_id"),
                "media_row_found": media_row is not None,
                "video_file_id": _relation_id((media_row or {}).get("video_file")),
                "audio_file_id": _relation_id((media_row or {}).get("audio_file")),
            },
        }

        writeback_status = _write_success(
            scan_id=req.scan_id,
            request_id=req.request_id,
            scan_context=scan_context,
            identifiers=identifiers,
            result=result,
            internal_analysis=internal_analysis,
        )

        logger.info(
            "process_scan_completed scan_id=%s member_id=%s baseline_scan_count=%s quality_status=%s risk_level=%s confidence=%.3f result_writeback=%s",
            req.scan_id,
            identifiers.get("member_id"),
            diagnostics["baseline_scan_count"],
            quality_result["status"],
            result["risk_level"],
            float(result["confidence"]),
            writeback_status.get("scan_result"),
        )

        response = dict(result)
        response["diagnostics"] = diagnostics
        response["writeback_status"] = writeback_status
        return response
    finally:
        for path in temp_files:
            remove_temp_file(path)
