from __future__ import annotations

from datetime import datetime
from statistics import median
from typing import Any

from config import (
    BASELINE_ACTIVE_AFTER,
    BASELINE_EVENING_START_HOUR,
    BASELINE_HIGH_CONFIDENCE_AFTER,
    BASELINE_MORNING_END_HOUR,
    BASELINE_PROVISIONAL_AFTER,
    BASELINE_USE_AFTER,
    BASELINE_MAX_STORED_SAMPLES,
    BASELINE_MIN_CAPTURE_QUALITY,
    BASELINE_MIN_CONFIDENCE,
)
from utils import clean_warning_codes, safe_number


BASELINE_SCHEMA_VERSION = 2
BASELINE_VERSION = "robust_v2"
BASELINE_SOURCE = "qualified_calibration_scans"
BASELINE_SAMPLE_LIMIT = BASELINE_MAX_STORED_SAMPLES
BASELINE_MAD_FLOOR = 0.02
BASELINE_STD_TO_MAD = 1.4826
FACE_FEATURES = ["open_eye_aperture", "left_right_eye_asymmetry"]
VOICE_FEATURES = ["normalized_voice_energy", "speech_rate"]
REACTION_FEATURES: list[str] = []
PERSONALIZATION_FEATURES = (
    ("face_avg", "open_eye_aperture"),
    ("face_avg", "left_right_eye_asymmetry"),
    ("voice_avg", "normalized_voice_energy"),
    ("voice_avg", "speech_rate"),
)

BASELINE_CALIBRATION_WARNING_BLOCKLIST = {
    "audio_missing",
    "audio_too_noisy",
    "audio_too_quiet",
    "audio_too_short",
    "face_not_visible",
    "image_blurry",
    "image_missing",
    "image_too_dark",
    "insufficient_usable_frames",
    "low_quality_media",
    "missing_media",
    "phrase_mismatch",
    "sustained_eye_closure",
    "subject_not_visible",
    "transcription_failed",
    "unstable_camera",
    "unstable_video",
    "video_blurry",
    "video_missing",
    "video_too_dark",
    "video_too_short",
    "speech_not_detected",
    "expected_phrase_missing",
    "fatigue_hard_gate",
}


def _utc_now() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _to_datetime(value: str | None) -> datetime:
    if not value:
        return datetime.utcnow()
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return datetime.utcnow()


def _time_bucket(scanned_at: str | None) -> str:
    hour = _to_datetime(scanned_at).hour
    if hour < BASELINE_MORNING_END_HOUR:
        return "morning"
    if hour >= BASELINE_EVENING_START_HOUR:
        return "evening"
    return "midday"


def _baseline_count(baseline: dict | None) -> int:
    current = baseline or {}
    value = current.get("eligible_scan_count")
    if value is None:
        value = current.get("scan_count", 0)
    try:
        return max(int(value or 0), 0)
    except (TypeError, ValueError):
        return 0


def _baseline_status_from_count(count: int) -> str:
    if count >= BASELINE_ACTIVE_AFTER:
        return "active"
    if count >= BASELINE_PROVISIONAL_AFTER:
        return "provisional"
    return "collecting"


def _baseline_confidence_from_count(count: int) -> float:
    if count <= 0:
        return 0.0
    if count >= BASELINE_HIGH_CONFIDENCE_AFTER:
        return 1.0
    if count < BASELINE_PROVISIONAL_AFTER:
        return round(min(count / float(BASELINE_PROVISIONAL_AFTER), 0.34), 4)
    if count < BASELINE_ACTIVE_AFTER:
        span = max(BASELINE_ACTIVE_AFTER - BASELINE_PROVISIONAL_AFTER, 1)
        progress = (count - BASELINE_PROVISIONAL_AFTER + 1) / float(span + 1)
        return round(0.4 + min(progress * 0.25, 0.25), 4)
    span = max(BASELINE_HIGH_CONFIDENCE_AFTER - BASELINE_ACTIVE_AFTER, 1)
    progress = min((count - BASELINE_ACTIVE_AFTER + 1) / float(span + 1), 1.0)
    return round(0.65 + (progress * 0.3), 4)


def _coerce_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clean_sample_list(values: Any) -> list[float]:
    cleaned: list[float] = []
    for raw in values or []:
        value = _coerce_float(raw)
        if value is not None:
            cleaned.append(float(value))
    return cleaned[-BASELINE_SAMPLE_LIMIT:]


def _empty_robust_payload(feature_names: list[str]) -> dict[str, Any]:
    return {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "feature_stats": {
            name: {
                "median": 0.0,
                "mad": 0.0,
                "count": 0,
            }
            for name in feature_names
        },
    }


def _legacy_feature_seed(value: Any, count: int) -> list[float]:
    numeric = _coerce_float(value)
    if numeric is None or count <= 0:
        return []
    sample_count = min(max(count, 1), 5)
    return [float(numeric)] * sample_count


def _legacy_feature_stats(value: Any, std: Any) -> dict[str, float]:
    center = _coerce_float(value) or 0.0
    spread = _coerce_float(std) or 0.0
    mad = spread / BASELINE_STD_TO_MAD if spread > 0 else 0.0
    return {
        "median": round(float(center), 4),
        "mad": round(max(float(mad), 0.0), 4),
        "count": 1 if center is not None else 0,
    }


def _existing_feature_samples(baseline: dict | None, field_name: str, feature_names: list[str]) -> dict[str, list[float]]:
    current = baseline or {}
    metadata = current.get("baseline_metadata") or {}
    sample_root = metadata.get("samples") or {}
    field_samples = sample_root.get(field_name) or {}
    existing = {
        name: _clean_sample_list(field_samples.get(name))
        for name in feature_names
    }
    payload = current.get(field_name)
    if isinstance(payload, dict) and payload.get("schema_version") == BASELINE_SCHEMA_VERSION:
        for name in feature_names:
            if existing[name]:
                continue
            existing[name] = _clean_sample_list((payload.get("feature_samples") or {}).get(name))
    count = _baseline_count(current)
    if isinstance(payload, dict) and payload.get("schema_version") != BASELINE_SCHEMA_VERSION:
        legacy_value = payload.get("avg")
        if field_name == "face_avg":
            if not existing["open_eye_aperture"]:
                existing["open_eye_aperture"] = _legacy_feature_seed(legacy_value, count)
            if not existing["left_right_eye_asymmetry"]:
                existing["left_right_eye_asymmetry"] = []
        elif field_name == "voice_avg":
            if not existing["normalized_voice_energy"]:
                existing["normalized_voice_energy"] = _legacy_feature_seed(legacy_value, count)
            if not existing["speech_rate"]:
                existing["speech_rate"] = []
        elif field_name == "reaction_avg":
            existing = {}
        if field_name in {"face_avg", "voice_avg"}:
            for name in feature_names:
                existing[name] = existing.get(name, [])[-BASELINE_SAMPLE_LIMIT:]
    return existing


def _median_and_mad(samples: list[float]) -> tuple[float, float]:
    if not samples:
        return 0.0, 0.0
    med = float(median(samples))
    deviations = [abs(sample - med) for sample in samples]
    mad = float(median(deviations)) if deviations else 0.0
    return round(med, 4), round(max(mad, BASELINE_MAD_FLOOR if len(samples) > 1 else 0.0), 4)


def _build_feature_payload(
    feature_names: list[str],
    samples_by_feature: dict[str, list[float]],
) -> dict[str, Any]:
    payload = _empty_robust_payload(feature_names)
    feature_stats: dict[str, dict[str, float]] = {}
    feature_samples: dict[str, list[float]] = {}
    max_count = 0
    for name in feature_names:
        samples = _clean_sample_list(samples_by_feature.get(name))
        med, mad = _median_and_mad(samples)
        feature_stats[name] = {"median": med, "mad": mad, "count": len(samples)}
        if samples:
            feature_samples[name] = [round(value, 6) for value in samples]
            max_count = max(max_count, len(samples))
    payload["feature_stats"] = feature_stats
    if feature_samples:
        payload["feature_samples"] = feature_samples
    return payload


def _bucket_counts(baseline: dict | None) -> dict[str, int]:
    current = baseline or {}
    metadata = current.get("baseline_metadata") or {}
    counts = metadata.get("bucket_counts") or {}
    if counts:
        return {
            "morning": int(counts.get("morning") or 0),
            "midday": int(counts.get("midday") or 0),
            "evening": int(counts.get("evening") or 0),
        }
    face_avg = current.get("face_avg") or {}
    buckets = face_avg.get("buckets") or {}
    return {
        "morning": int((buckets.get("morning") or {}).get("count", 0)),
        "midday": int((buckets.get("midday") or {}).get("count", 0)),
        "evening": int((buckets.get("evening") or {}).get("count", 0)),
    }


def _append_feature_sample(sample_map: dict[str, list[float]], feature_name: str, value: float | None) -> None:
    if value is None:
        return
    samples = list(sample_map.get(feature_name) or [])
    samples.append(float(value))
    sample_map[feature_name] = samples[-BASELINE_SAMPLE_LIMIT:]


def current_baseline_features(
    *,
    signals: dict | None,
    result: dict | None = None,
) -> dict[str, dict[str, float | None]]:
    signals = signals or {}
    camera_details = ((signals.get("camera") or {}).get("details") or {}) if isinstance(signals.get("camera"), dict) else {}
    voice_details = ((signals.get("voice") or {}).get("details") or {}) if isinstance(signals.get("voice"), dict) else {}

    left_eye = _coerce_float(camera_details.get("left_eye_aperture"))
    right_eye = _coerce_float(camera_details.get("right_eye_aperture"))
    avg_eye = _coerce_float(camera_details.get("avg_ear"))
    asymmetry = _coerce_float(camera_details.get("left_right_eye_asymmetry"))
    if asymmetry is None and left_eye is not None and right_eye is not None:
        asymmetry = abs(left_eye - right_eye)

    return {
        "face_avg": {
            "open_eye_aperture": avg_eye,
            "left_right_eye_asymmetry": asymmetry,
        },
        "voice_avg": {
            "normalized_voice_energy": _coerce_float(voice_details.get("rms_energy")),
            "speech_rate": _coerce_float(voice_details.get("speech_rate")),
        },
        "reaction_avg": {},
    }


def _hard_gates_from_signals(signals: dict | None) -> list[str]:
    return []


def _feature_presence_requirements(signals: dict | None) -> tuple[bool, list[str], dict[str, dict[str, float | None]]]:
    features = current_baseline_features(signals=signals, result=None)
    missing_reasons: list[str] = []
    face = features.get("face_avg") or {}
    voice = features.get("voice_avg") or {}
    face_usable = face.get("open_eye_aperture") is not None
    voice_usable = voice.get("normalized_voice_energy") is not None
    if not face_usable:
        missing_reasons.append("missing_face_baseline_feature")
    if not voice_usable:
        missing_reasons.append("missing_voice_baseline_feature")
    return face_usable and voice_usable, missing_reasons, features


def _task_completion_status(
    *,
    validation_result: dict | None,
    quality_result: dict | None,
    expected_phrase: str | None,
    task: Any = None,
) -> str:
    warnings = set((validation_result or {}).get("warnings") or [])
    if expected_phrase and warnings.intersection({"speech_not_detected", "transcription_failed", "expected_phrase_missing", "phrase_mismatch"}):
        return "incomplete_required_speech"
    task_quality = (quality_result or {}).get("task_quality")
    task_present = False
    if isinstance(task, dict):
        task_present = any(task.get(key) is not None for key in ["reaction_time", "errors", "attempts"])
    elif task is not None:
        task_present = any(getattr(task, key, None) is not None for key in ["reaction_time", "errors", "attempts"])
    if task_present and task_quality in {"weak", "failed"}:
        return "incomplete_required_task"
    if expected_phrase or task_present:
        return "completed"
    return "not_required"


def evaluate_baseline_eligibility(
    *,
    quality_result: dict | None,
    validation_result: dict | None,
    result: dict | None,
    signals: dict | None,
    expected_phrase: str | None = None,
    task: Any = None,
    manually_unreliable: bool = False,
) -> dict[str, Any]:
    quality_result = quality_result or {}
    validation_result = validation_result or {}
    result = result or {}
    features_present, feature_reasons, _ = _feature_presence_requirements(signals)
    hard_gates_triggered = _hard_gates_from_signals(signals)
    task_completion_status = _task_completion_status(
        validation_result=validation_result,
        quality_result=quality_result,
        expected_phrase=expected_phrase,
        task=task,
    )
    reasons: list[str] = []

    quality_warnings = set(clean_warning_codes((quality_result or {}).get("warnings") or []))
    validation_warnings = set(clean_warning_codes((validation_result or {}).get("warnings") or []))
    combined_warnings = quality_warnings | validation_warnings
    capture_quality = _coerce_float((quality_result.get("media_quality") or {}).get("aggregate_quality"))
    measurement_reliability = _coerce_float(quality_result.get("confidence_multiplier"))
    baseline_confidence = _coerce_float(result.get("confidence"))

    if quality_result.get("status") != "passed" or quality_result.get("weak"):
        reasons.append("insufficient_capture_quality")
    if quality_result.get("retake_required") or result.get("retake_required"):
        reasons.append("retake_required")
    if not features_present:
        reasons.extend(feature_reasons)
    if validation_result.get("critical_errors"):
        reasons.append("validation_failure")
    if task_completion_status not in {"completed", "not_required"}:
        reasons.append(task_completion_status)
    if capture_quality is None or capture_quality < BASELINE_MIN_CAPTURE_QUALITY:
        reasons.append("capture_quality_too_low")
    if measurement_reliability is None or measurement_reliability < BASELINE_MIN_CAPTURE_QUALITY:
        reasons.append("measurement_reliability_too_low")
    if baseline_confidence is None or baseline_confidence < BASELINE_MIN_CONFIDENCE:
        reasons.append("low_confidence")
    if combined_warnings & BASELINE_CALIBRATION_WARNING_BLOCKLIST:
        reasons.append("measurement_warning")
    if hard_gates_triggered:
        reasons.append("fatigue_hard_gate")
    if str(result.get("risk_level") or "").strip().lower() != "stable":
        reasons.append("risk_not_stable")
    if manually_unreliable:
        reasons.append("manually_unreliable")
    if validation_warnings & {"speech_not_detected", "expected_phrase_missing", "phrase_mismatch", "transcription_failed"}:
        reasons.append("speech_required_not_completed")

    return {
        "eligible": len(reasons) == 0,
        "reasons": reasons,
        "hard_gates_triggered": hard_gates_triggered,
        "task_completion_status": task_completion_status,
        "capture_quality_score": safe_number(capture_quality, 4),
        "measurement_reliability_score": safe_number(measurement_reliability, 4),
    }


def is_scan_eligible_for_baseline(**kwargs: Any) -> bool:
    return bool(evaluate_baseline_eligibility(**kwargs)["eligible"])


def baseline_signal_payload(
    baseline: dict | None,
    *,
    signals: dict,
    scanned_at: str | None = None,
) -> dict:
    current = baseline or {}
    next_count = _baseline_count(current) + 1
    bucket = _time_bucket(scanned_at)
    features = current_baseline_features(signals=signals)

    face_samples = _existing_feature_samples(current, "face_avg", FACE_FEATURES)
    voice_samples = _existing_feature_samples(current, "voice_avg", VOICE_FEATURES)
    reaction_samples = _existing_feature_samples(current, "reaction_avg", REACTION_FEATURES)

    for name, value in (features.get("face_avg") or {}).items():
        _append_feature_sample(face_samples, name, value)
    for name, value in (features.get("voice_avg") or {}).items():
        _append_feature_sample(voice_samples, name, value)
    for name, value in (features.get("reaction_avg") or {}).items():
        _append_feature_sample(reaction_samples, name, value)

    bucket_counts = _bucket_counts(current)
    bucket_counts[bucket] = int(bucket_counts.get(bucket) or 0) + 1
    baseline_status = _baseline_status_from_count(next_count)
    baseline_confidence = _baseline_confidence_from_count(next_count)
    activated_at = current.get("activated_at")
    if baseline_status == "active" and not activated_at:
        activated_at = _utc_now()

    face_avg = _build_feature_payload(FACE_FEATURES, face_samples)
    voice_avg = _build_feature_payload(VOICE_FEATURES, voice_samples)
    reaction_avg = _build_feature_payload(REACTION_FEATURES, reaction_samples)
    return {
        "scan_count": next_count,
        "face_avg": face_avg,
        "voice_avg": voice_avg,
        "reaction_avg": reaction_avg,
        "is_active": baseline_status == "active",
    }


def baseline_status_payload(baseline: dict | None) -> dict:
    count = _baseline_count(baseline)
    status = _baseline_status_from_count(count)
    is_active = status == "active"
    is_provisional = status == "provisional"
    buckets = _bucket_counts(baseline)
    scans_remaining = max(BASELINE_ACTIVE_AFTER - count, 0)

    if is_active:
        message = "Baseline active. Current scans are compared against the employee's own readiness pattern."
    elif count == 0:
        message = "No baseline yet. Capture qualified scans to start calibration."
    elif is_provisional:
        message = "Baseline is provisional. Keep collecting qualified scans until activation."
    else:
        message = "Calibration started. More qualified scans are needed before personalized scoring is active."

    return {
        "is_active": is_active,
        "scan_count": count,
        "eligible_scan_count": count,
        "scans_remaining": scans_remaining,
        "is_provisional": is_provisional,
        "baseline_status": status,
        "baseline_confidence": _coerce_float((baseline or {}).get("baseline_confidence")) or _baseline_confidence_from_count(count),
        "needs_morning_scan": int(buckets.get("morning", 0)) == 0,
        "needs_evening_scan": int(buckets.get("evening", 0)) == 0,
        "message": message,
    }


def baseline_ready_for_scoring(baseline: dict | None) -> bool:
    status = baseline_status_payload(baseline)
    return bool(status["is_active"]) and int(status["scan_count"]) >= BASELINE_USE_AFTER


def baseline_has_valid_personalization_references(baseline: dict | None) -> bool:
    return any(
        baseline_feature_reference(baseline, field_name, feature_name) is not None
        for field_name, feature_name in PERSONALIZATION_FEATURES
    )


def baseline_ready_for_personalized_scoring(
    baseline: dict | None,
    *,
    quality_result: dict | None = None,
    validation_result: dict | None = None,
    result: dict | None = None,
    task: Any = None,
    expected_phrase: str | None = None,
    unique_row: bool = True,
) -> bool:
    if not unique_row:
        return False
    if not baseline_ready_for_scoring(baseline):
        return False
    if not baseline_has_valid_personalization_references(baseline):
        return False

    quality_result = quality_result or {}
    validation_result = validation_result or {}
    result = result or {}

    if quality_result.get("status") != "passed" or quality_result.get("weak") or quality_result.get("retake_required"):
        return False
    if validation_result.get("critical_errors"):
        return False
    if _task_completion_status(
        validation_result=validation_result,
        quality_result=quality_result,
        expected_phrase=expected_phrase,
        task=task,
    ) not in {"completed", "not_required"}:
        return False

    quality_warnings = set(clean_warning_codes((quality_result or {}).get("warnings") or []))
    validation_warnings = set(clean_warning_codes((validation_result or {}).get("warnings") or []))
    if quality_warnings & BASELINE_CALIBRATION_WARNING_BLOCKLIST:
        return False
    if validation_warnings & {"speech_not_detected", "expected_phrase_missing", "phrase_mismatch", "transcription_failed"}:
        return False
    if _coerce_float(quality_result.get("confidence_multiplier")) is not None and float(quality_result["confidence_multiplier"]) < BASELINE_MIN_CAPTURE_QUALITY:
        return False
    if _coerce_float((quality_result.get("media_quality") or {}).get("aggregate_quality")) is not None and float((quality_result.get("media_quality") or {}).get("aggregate_quality")) < BASELINE_MIN_CAPTURE_QUALITY:
        return False
    if _coerce_float(result.get("confidence")) is not None and float(result["confidence"]) < BASELINE_MIN_CONFIDENCE:
        return False
    return True


def legacy_baseline_stat(baseline: dict | None, key: str) -> dict[str, float] | None:
    payload = (baseline or {}).get(key)
    if not isinstance(payload, dict):
        return None
    if payload.get("schema_version") == BASELINE_SCHEMA_VERSION:
        return None
    if payload.get("avg") is None:
        return None
    return {
        "median": _coerce_float(payload.get("avg")) or 0.0,
        "mad": max((_coerce_float(payload.get("std")) or 0.0) / BASELINE_STD_TO_MAD, BASELINE_MAD_FLOOR),
    }


def baseline_feature_reference(baseline: dict | None, field_name: str, feature_name: str) -> dict[str, float] | None:
    payload = (baseline or {}).get(field_name)
    if isinstance(payload, dict) and payload.get("schema_version") == BASELINE_SCHEMA_VERSION:
        stats = (payload.get("feature_stats") or {}).get(feature_name)
        if isinstance(stats, dict):
            center = _coerce_float(stats.get("median"))
            mad = _coerce_float(stats.get("mad"))
            if center is not None:
                return {
                    "median": round(center, 4),
                    "mad": round(max(mad or 0.0, BASELINE_MAD_FLOOR), 4),
                }
    if field_name == "face_avg" and feature_name == "open_eye_aperture":
        return legacy_baseline_stat(baseline, field_name)
    if field_name == "voice_avg" and feature_name == "normalized_voice_energy":
        return legacy_baseline_stat(baseline, field_name)
    return None
