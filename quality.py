from __future__ import annotations

from typing import Any

from utils import clamp01, clean_warning_codes, safe_number


FAILURE_REASON_LOW_QUALITY_MEDIA = "low_quality_media"
FAILURE_REASON_MISSING_MEDIA = "missing_media"


def _task_value(task: Any, key: str) -> Any:
    if task is None:
        return None
    if isinstance(task, dict):
        return task.get(key)
    return getattr(task, key, None)


def _signal_summary(name: str, result: dict | None, usable_threshold: float, warning_key: str) -> dict:
    result = result or {}
    details = result.get("details", {}) if isinstance(result, dict) else {}
    score = clamp01(result.get("score"))
    warnings = clean_warning_codes(details.get(warning_key) or [])
    missing_statuses = {"missing", "open_failed", "load_failed", "empty_audio", "invalid", "invalid_image", "error", None}
    present = details.get("status") not in missing_statuses or score is not None
    usable = bool(score is not None and score >= usable_threshold and len(warnings) < 3)
    weak = bool(score is not None and not usable)
    return {
        "name": name,
        "present": present,
        "usable": usable,
        "weak": weak,
        "score": score,
        "warnings": warnings,
        "details": details,
    }


def assess_quality(signals: dict, task: Any = None) -> dict:
    video = _signal_summary("video", signals.get("video"), 0.42, "visual_warnings")
    audio = _signal_summary("audio", signals.get("voice"), 0.4, "audio_warnings")
    image = _signal_summary("image", signals.get("camera"), 0.38, "image_warnings")

    warnings = clean_warning_codes(video["warnings"] + audio["warnings"] + image["warnings"])
    weak_modalities = [signal["name"] for signal in [video, audio, image] if signal["weak"]]
    missing_modalities = [signal["name"] for signal in [video, audio, image] if not signal["present"]]
    usable_modalities = sum(1 for signal in [video, audio, image] if signal["usable"])
    present_modalities = sum(1 for signal in [video, audio, image] if signal["present"])
    strong_modalities = sum(1 for signal in [video, audio, image] if (signal["score"] or 0.0) >= 0.72)

    task_present = any(_task_value(task, key) is not None for key in ["reaction_time", "errors", "attempts"])
    task_quality = "missing"
    if task_present:
        attempts = int(_task_value(task, "attempts") or 0)
        reaction_time = _task_value(task, "reaction_time")
        errors = _task_value(task, "errors")
        valid = attempts > 0 and reaction_time is not None and float(reaction_time) > 0 and errors is not None
        task_quality = "good" if valid and attempts >= 3 else ("weak" if valid else "failed")

    quality_components = [signal["score"] for signal in [video, audio, image] if signal["score"] is not None]
    aggregate_quality = sum(quality_components) / len(quality_components) if quality_components else 0.0
    confidence_multiplier = float(
        max(
            0.15,
            min(
                1.0,
                (0.5 * aggregate_quality)
                + (0.2 * (usable_modalities / 3.0))
                + (0.15 * (strong_modalities / 3.0))
                + (0.15 * (1.0 - min(len(warnings) / 8.0, 1.0))),
            ),
        )
    )

    failure_reason = None
    status = "passed"
    retake_required = False
    if present_modalities == 0:
        failure_reason = FAILURE_REASON_MISSING_MEDIA
        status = "weak"
        retake_required = True
    elif usable_modalities == 0:
        failure_reason = FAILURE_REASON_LOW_QUALITY_MEDIA
        status = "weak"
        retake_required = True
    elif usable_modalities == 1 and strong_modalities == 0:
        failure_reason = FAILURE_REASON_LOW_QUALITY_MEDIA
        status = "weak"
        retake_required = True
    elif usable_modalities == 1 or warnings:
        status = "weak"

    if status == "failed" or retake_required:
        suggested_action = "rescan_recommended"
    elif usable_modalities == 1:
        suggested_action = "review_required"
    else:
        suggested_action = "continue_normal_activity"

    return {
        "status": status,
        "passed": True,
        "weak": status == "weak",
        "failure_reason": failure_reason,
        "reasons": [failure_reason] if failure_reason else [],
        "weak_reasons": warnings if status == "weak" else [],
        "retake_required": retake_required,
        "suggested_action": suggested_action,
        "usable_modalities": usable_modalities,
        "weak_modalities": weak_modalities,
        "missing_modalities": missing_modalities,
        "present_modalities": present_modalities,
        "confidence_multiplier": safe_number(confidence_multiplier),
        "task_quality": task_quality,
        "warnings": warnings,
        "media_quality": {
            "aggregate_quality": safe_number(aggregate_quality),
            "video": {
                "present": video["present"],
                "usable": video["usable"],
                "weak": video["weak"],
                "score": safe_number(video["score"]),
                "warnings": video["warnings"],
            },
            "audio": {
                "present": audio["present"],
                "usable": audio["usable"],
                "weak": audio["weak"],
                "score": safe_number(audio["score"]),
                "warnings": audio["warnings"],
            },
            "image": {
                "present": image["present"],
                "usable": image["usable"],
                "weak": image["weak"],
                "score": safe_number(image["score"]),
                "warnings": image["warnings"],
            },
        },
    }
