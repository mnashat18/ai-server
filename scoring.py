from __future__ import annotations

from typing import Any

from config import MODEL_VERSION
from utils import clamp01, clean_warning_codes, safe_number, sanitize_text

VALID_RISK_LEVELS = {"stable", "low_focus", "elevated_fatigue", "high_risk", "unknown"}
VALID_ACTIONS = {
    "continue_normal_activity",
    "review_required",
    "rescan_recommended",
    "rest_advised",
    "manager_review",
}
BASE_WEIGHTS = {
    "video": 0.45,
    "audio": 0.35,
    "image": 0.15,
    "task": 0.05,
}
WARNING_SCORE_PENALTIES = {
    "video_blurry": 0.18,
    "video_too_dark": 0.18,
    "video_too_short": 0.1,
    "unstable_video": 0.16,
    "unstable_camera": 0.16,
    "face_not_visible": 0.22,
    "subject_not_visible": 0.18,
    "audio_too_noisy": 0.18,
    "audio_too_quiet": 0.18,
    "speech_not_detected": 0.22,
    "audio_too_short": 0.1,
    "image_blurry": 0.14,
    "image_too_dark": 0.14,
    "low_quality_media": 0.12,
}


def clamp_confidence(value: float | None) -> float | None:
    return clamp01(value)


def _task_value(task: Any, key: str):
    if task is None:
        return None
    if isinstance(task, dict):
        return task.get(key)
    return getattr(task, key, None)


def compute_task_score(task: Any) -> float | None:
    if not task:
        return None
    reaction_time = _task_value(task, "reaction_time")
    errors = _task_value(task, "errors")
    attempts = _task_value(task, "attempts")
    scores: list[float] = []
    if reaction_time is not None:
        rt = float(reaction_time)
        scores.append(0.92 if rt <= 0.55 else 0.72 if rt <= 0.85 else 0.45 if rt <= 1.15 else 0.22)
    if errors is not None:
        err = int(errors)
        scores.append(0.95 if err == 0 else 0.75 if err <= 1 else 0.5 if err <= 3 else 0.2)
    if attempts is not None:
        att = int(attempts)
        scores.append(0.95 if att >= 3 else 0.6 if att >= 1 else 0.2)
    if not scores:
        return None
    return round(sum(scores) / len(scores), 4)


def _baseline_stat(baseline: dict | None, key: str) -> dict | None:
    value = (baseline or {}).get(key)
    return value if isinstance(value, dict) else None


def _baseline_drift(current: float | None, stat: dict | None) -> dict | None:
    if current is None or not stat or stat.get("avg") is None:
        return None
    avg = float(stat["avg"])
    std = float(stat.get("std") or 0.0)
    threshold = max(0.1, std * 1.5)
    drift = round(float(current) - avg, 4)
    return {
        "current": safe_number(current),
        "baseline_avg": safe_number(avg),
        "baseline_std": safe_number(std),
        "drift": safe_number(drift),
        "threshold": safe_number(threshold),
        "below_threshold": drift <= (-threshold),
    }


def _normalize_weights(weight_map: dict[str, float]) -> dict[str, float]:
    total = sum(max(weight, 0.0) for weight in weight_map.values())
    if total <= 0:
        return {}
    return {name: weight / total for name, weight in weight_map.items() if weight > 0}


def _signal_score(analysis: dict, score_key: str) -> float | None:
    details = analysis.get("details", {}) if isinstance(analysis, dict) else {}
    return clamp01(details.get(score_key) if score_key in details else analysis.get("score"))


def _signal_warnings(analysis: dict, key: str) -> list[str]:
    details = analysis.get("details", {}) if isinstance(analysis, dict) else {}
    return clean_warning_codes(details.get(key) or [])


def _signal_present(analysis: dict) -> bool:
    if not isinstance(analysis, dict):
        return False
    details = analysis.get("details", {})
    missing_statuses = {"missing", "open_failed", "load_failed", "empty_audio", "invalid", "invalid_image", "error", None}
    return analysis.get("score") is not None or details.get("status") not in missing_statuses


def _missing_modalities_from_profiles(profiles: dict[str, dict]) -> list[str]:
    return [name for name in ["video", "audio", "image"] if not profiles.get(name, {}).get("present")]


def _degraded_signal_score(score: float | None, quality: float | None, warnings: list[str]) -> tuple[float | None, float]:
    if score is None:
        return None, 0.0
    warning_penalty = sum(WARNING_SCORE_PENALTIES.get(warning, 0.0) for warning in warnings)
    warning_penalty = min(warning_penalty, 0.42)
    quality_penalty = 0.0
    if quality is not None and quality < 0.5:
        quality_penalty = min((0.5 - quality) * 0.35, 0.2)
    penalty = min(warning_penalty + quality_penalty, 0.55)
    return clamp01(score - penalty, 0.0), round(penalty, 4)


def _quality_warnings_for_modality(quality: dict | None, modality: str, current: list[str]) -> list[str]:
    combined = list(current)
    media_quality = (quality or {}).get("media_quality") or {}
    modality_quality = media_quality.get(modality) or {}
    combined.extend(modality_quality.get("warnings") or [])
    global_warnings = (quality or {}).get("warnings") or []
    modality_warning_map = {
        "video": {
            "video_blurry",
            "video_too_dark",
            "video_too_short",
            "unstable_video",
            "unstable_camera",
            "face_not_visible",
            "subject_not_visible",
            "low_quality_media",
        },
        "audio": {
            "audio_too_noisy",
            "audio_too_quiet",
            "speech_not_detected",
            "audio_too_short",
            "low_quality_media",
        },
        "image": {
            "image_blurry",
            "image_too_dark",
            "face_not_visible",
            "subject_not_visible",
            "low_quality_media",
        },
    }
    allowed = modality_warning_map.get(modality, set())
    combined.extend(warning for warning in global_warnings if warning in allowed)
    return clean_warning_codes(combined)


def _build_signal_profiles(signals: dict, task_score: float | None, quality: dict | None = None) -> dict[str, dict]:
    video = signals.get("video", {}) or {}
    audio = signals.get("voice", {}) or {}
    image = signals.get("camera", {}) or {}
    video_score = _signal_score(video, "visual_confidence")
    video_quality = _signal_score(video, "visual_quality_score")
    video_warnings = _quality_warnings_for_modality(quality, "video", _signal_warnings(video, "visual_warnings"))
    adjusted_video_score, video_penalty = _degraded_signal_score(video_score, video_quality, video_warnings)

    audio_score = _signal_score(audio, "audio_confidence")
    audio_quality = _signal_score(audio, "audio_quality_score")
    audio_warnings = _quality_warnings_for_modality(quality, "audio", _signal_warnings(audio, "audio_warnings"))
    adjusted_audio_score, audio_penalty = _degraded_signal_score(audio_score, audio_quality, audio_warnings)

    image_score = _signal_score(image, "image_confidence")
    image_quality = _signal_score(image, "image_quality_score")
    image_warnings = _quality_warnings_for_modality(quality, "image", _signal_warnings(image, "image_warnings"))
    adjusted_image_score, image_penalty = _degraded_signal_score(image_score, image_quality, image_warnings)
    return {
        "video": {
            "present": _signal_present(video),
            "score": adjusted_video_score,
            "raw_score": video_score,
            "quality": video_quality,
            "warnings": video_warnings,
            "score_penalty": video_penalty,
        },
        "audio": {
            "present": _signal_present(audio),
            "score": adjusted_audio_score,
            "raw_score": audio_score,
            "quality": audio_quality,
            "warnings": audio_warnings,
            "score_penalty": audio_penalty,
        },
        "image": {
            "present": _signal_present(image),
            "score": adjusted_image_score,
            "raw_score": image_score,
            "quality": image_quality,
            "warnings": image_warnings,
            "score_penalty": image_penalty,
        },
        "task": {
            "present": task_score is not None,
            "score": task_score,
            "quality": task_score,
            "warnings": [],
        },
    }


def _adaptive_weights(profiles: dict[str, dict]) -> dict[str, float]:
    weighted: dict[str, float] = {}
    for name, profile in profiles.items():
        if not profile["present"] or profile["score"] is None:
            continue
        quality = profile["quality"] if profile["quality"] is not None else profile["score"]
        warning_penalty = min(len(profile["warnings"]) * 0.12, 0.45)
        weighted[name] = BASE_WEIGHTS[name] * max(0.05, (quality or 0.0) - warning_penalty)
    return _normalize_weights(weighted)


def _agreement_factor(profiles: dict[str, dict]) -> tuple[float, float]:
    scores = [profile["score"] for profile in profiles.values() if profile["present"] and profile["score"] is not None]
    if len(scores) < 2:
        return 0.0, 0.0
    spread = max(scores) - min(scores)
    return max(0.0, 0.18 - spread * 0.18), min(spread * 0.45, 0.25)


def _confidence_from_profiles(
    *,
    fused_score: float,
    profiles: dict[str, dict],
    weights: dict[str, float],
    quality: dict,
    baseline_used: bool,
    ml_result: dict | None,
) -> tuple[float, dict]:
    available = [name for name, profile in profiles.items() if profile["present"] and profile["score"] is not None and name != "task"]
    major_available = [name for name in available if name in {"video", "audio"}]
    agreement_bonus, conflict_penalty = _agreement_factor(profiles)
    quality_multiplier = float((quality or {}).get("confidence_multiplier") or 0.3)
    coverage = len(available) / 3.0
    missing_major_penalty = 0.18 if len(major_available) == 1 else 0.32 if len(major_available) == 0 else 0.0
    warning_penalty = min(len((quality or {}).get("warnings") or []) * 0.03, 0.18)
    ml_confidence = clamp01((ml_result or {}).get("confidence"))

    confidence = (
        0.32 * fused_score
        + 0.22 * quality_multiplier
        + 0.18 * coverage
        + agreement_bonus
        + (0.06 if baseline_used else 0.0)
        + (0.05 * ml_confidence if ml_confidence is not None else 0.0)
        - missing_major_penalty
        - warning_penalty
        - conflict_penalty
    )

    ceiling = 0.98
    if len(available) == 1:
        ceiling = 0.55
        if "image" in available:
            ceiling = 0.35
    elif len(major_available) < 2:
        ceiling = 0.78
    if (quality or {}).get("weak"):
        ceiling = min(ceiling, 0.72)
    if conflict_penalty > 0.12:
        ceiling = min(ceiling, 0.68)

    confidence = min(clamp01(confidence, 0.0) or 0.0, ceiling)
    return round(confidence, 4), {
        "weights": {name: safe_number(value) for name, value in weights.items()},
        "agreement_bonus": safe_number(agreement_bonus),
        "conflict_penalty": safe_number(conflict_penalty),
        "quality_multiplier": safe_number(quality_multiplier),
        "coverage": safe_number(coverage),
        "missing_major_penalty": safe_number(missing_major_penalty),
        "warning_penalty": safe_number(warning_penalty),
        "ceiling": safe_number(ceiling),
        "ml_confidence": safe_number(ml_confidence),
    }


def _risk_level(readiness_score: int, confidence: float, baseline_flags: list[str], quality: dict) -> str:
    quality_failed = quality.get("status") == "failed"
    media_quality_too_weak = quality.get("failure_reason") in {"low_quality_media", "missing_media"}
    missing_major_media = bool({"video", "audio"} & set(quality.get("missing_modalities") or []))
    if quality_failed or media_quality_too_weak or missing_major_media or confidence < 0.45:
        return "unknown"
    if quality.get("weak") and readiness_score < 52 and not baseline_flags:
        return "unknown"
    if readiness_score < 35 and (confidence >= 0.62 or len(baseline_flags) >= 2):
        return "high_risk"
    if readiness_score < 52 and (confidence >= 0.5 or baseline_flags):
        return "elevated_fatigue"
    if readiness_score < 68:
        return "low_focus"
    return "stable"


def _suggested_action(risk_level: str, confidence: float, quality: dict) -> str:
    if quality.get("status") == "failed":
        return "rescan_recommended"
    if risk_level == "unknown":
        return "rescan_recommended"
    if risk_level == "high_risk":
        return "manager_review"
    if risk_level == "elevated_fatigue":
        return "rest_advised"
    if risk_level == "low_focus":
        return "review_required" if confidence >= 0.45 else "rescan_recommended"
    if confidence < 0.45 or quality.get("weak"):
        return "rescan_recommended"
    return "continue_normal_activity"


def _explanation(profiles: dict[str, dict], quality: dict, risk_level: str, confidence: float) -> str:
    positives: list[str] = []
    negatives: list[str] = []
    warning_reasons: list[str] = []
    if profiles["video"]["score"] is not None and (profiles["video"]["quality"] or 0.0) >= 0.55:
        positives.append("Video quality was acceptable")
    if profiles["audio"]["score"] is not None and (profiles["audio"]["quality"] or 0.0) >= 0.55:
        positives.append("voice signal was clear")
    if profiles["image"]["score"] is not None and (profiles["image"]["quality"] or 0.0) >= 0.55:
        positives.append("thumbnail quality was usable")
    if "audio_too_noisy" in quality.get("warnings", []):
        negatives.append("Audio was noisy")
        warning_reasons.append("background noise reduced voice confidence")
    if "audio_too_quiet" in quality.get("warnings", []):
        negatives.append("audio volume was low")
        warning_reasons.append("low voice volume reduced confidence")
    if "speech_not_detected" in quality.get("warnings", []):
        negatives.append("speech was not clearly detected")
        warning_reasons.append("weak speech detection reduced confidence")
    if "video_blurry" in quality.get("warnings", []) or "image_blurry" in quality.get("warnings", []):
        negatives.append("visual media was partially blurred")
        warning_reasons.append("blur reduced visual confidence")
    if "video_too_dark" in quality.get("warnings", []) or "image_too_dark" in quality.get("warnings", []):
        negatives.append("lighting was too dark")
        warning_reasons.append("low lighting reduced visual confidence")
    if "subject_not_visible" in quality.get("warnings", []):
        negatives.append("subject visibility was limited")
        warning_reasons.append("limited subject visibility reduced confidence")
    if "face_not_visible" in quality.get("warnings", []) or "unstable_video" in quality.get("warnings", []):
        negatives.append("face visibility was weak")
        warning_reasons.append("weak face visibility reduced confidence")
    if "low_quality_media" in quality.get("warnings", []) or quality.get("failure_reason") == "low_quality_media":
        warning_reasons.append("overall media quality was low")
    if "missing_media" in quality.get("warnings", []) or quality.get("failure_reason") == "missing_media":
        negatives.append("scan media was missing or unreadable")
        warning_reasons.append("missing media prevented a confident result")
    missing_modalities = clean_warning_codes(quality.get("missing_modalities") or [])
    if missing_modalities:
        label = ", ".join(missing_modalities[:3])
        negatives.append(f"{label} data was missing or unreadable")
        warning_reasons.append("missing or unreadable media reduced confidence")
    if profiles["video"]["present"] is False and profiles["audio"]["present"] is False and profiles["image"]["present"]:
        negatives.append("Only thumbnail data was available")

    lead = ". ".join(part for part in [", ".join(positives) if positives else None, ", ".join(negatives) if negatives else None] if part)
    if not lead:
        lead = "Media signals were mixed"
    if warning_reasons:
        unique_reasons = clean_warning_codes(warning_reasons)
        lead = f"{lead}. Score and confidence were reduced because {', '.join(unique_reasons[:4])}"
    if risk_level == "stable":
        tail = "Both signals suggest stable readiness with moderate confidence." if confidence >= 0.5 else "Confidence was reduced because the available signals were limited."
    elif risk_level == "low_focus":
        tail = "The fused result suggests a mild reduction in readiness."
    elif risk_level == "elevated_fatigue":
        tail = "The fused result suggests elevated fatigue and should be reviewed."
    elif risk_level == "high_risk":
        tail = "The fused result suggests a high-risk state that merits review."
    else:
        tail = "A re-scan is recommended for a more reliable result."
    return sanitize_text(f"{lead}. {tail}", fallback="A re-scan is recommended for a more reliable result.", max_len=500) or "A re-scan is recommended for a more reliable result."


def compute_result(
    *,
    signals: dict,
    task: Any = None,
    previous_confidence: float | None = None,
    baseline: dict | None = None,
    baseline_used: bool = False,
    quality: dict | None = None,
    ml_result: dict | None = None,
) -> dict:
    quality = quality or {}
    task_score = compute_task_score(task)
    profiles = _build_signal_profiles(signals, task_score, quality=quality)
    if not quality.get("missing_modalities"):
        inferred_missing = _missing_modalities_from_profiles(profiles)
        if inferred_missing:
            quality = {**quality, "missing_modalities": inferred_missing}
    weights = _adaptive_weights(profiles)

    fused_score = 0.0
    for name, weight in weights.items():
        score = profiles[name]["score"]
        if score is not None:
            fused_score += score * weight

    ml_confidence = clamp01((ml_result or {}).get("confidence"))
    if ml_confidence is not None:
        fused_score = (fused_score * 0.84) + (ml_confidence * 0.16)
    quality_warnings = clean_warning_codes((quality or {}).get("warnings") or [])
    quality_penalty = min(len(quality_warnings) * 0.025, 0.16)
    if (quality or {}).get("failure_reason") == "low_quality_media":
        quality_penalty = max(quality_penalty, 0.12)
    if (quality or {}).get("failure_reason") == "missing_media":
        quality_penalty = max(quality_penalty, 0.20)
    if (quality or {}).get("weak"):
        quality_penalty = max(quality_penalty, 0.06)
    fused_score = fused_score - quality_penalty
    fused_score = clamp01(fused_score, 0.0) or 0.0

    image_score = profiles["image"]["score"]
    video_score = profiles["video"]["score"]
    face_score = None
    if image_score is not None and video_score is not None:
        face_score = round((image_score * 0.4) + (video_score * 0.6), 4)
    else:
        face_score = image_score if image_score is not None else video_score

    face_drift = _baseline_drift(face_score, _baseline_stat(baseline, "face_avg")) if baseline_used else None
    voice_drift = _baseline_drift(profiles["audio"]["score"], _baseline_stat(baseline, "voice_avg")) if baseline_used else None
    reaction_drift = _baseline_drift(task_score, _baseline_stat(baseline, "reaction_avg")) if baseline_used else None
    baseline_flags = [
        name
        for name, drift in {
            "face": face_drift,
            "voice": voice_drift,
            "reaction": reaction_drift,
        }.items()
        if drift and drift.get("below_threshold")
    ]

    if baseline_flags:
        fused_score = max(0.0, fused_score - (0.03 * len(baseline_flags)))

    confidence, calibration = _confidence_from_profiles(
        fused_score=fused_score,
        profiles=profiles,
        weights=weights,
        quality=quality,
        baseline_used=baseline_used,
        ml_result=ml_result,
    )
    readiness_score = int(round((clamp01(fused_score, 0.0) or 0.0) * 100))
    risk_level = _risk_level(readiness_score, confidence, baseline_flags, quality)
    if risk_level not in VALID_RISK_LEVELS:
        risk_level = "unknown"
    suggested_action = _suggested_action(risk_level, confidence, quality)
    if suggested_action not in VALID_ACTIONS:
        suggested_action = "rescan_recommended"
    explanation = _explanation(profiles, quality, risk_level, confidence)

    previous = clamp01(previous_confidence, confidence)
    confidence_drift = safe_number(confidence - (previous if previous is not None else confidence))
    modality_scores = {
        "video": safe_number(profiles["video"]["score"]),
        "audio": safe_number(profiles["audio"]["score"]),
        "image": safe_number(profiles["image"]["score"]),
        "task": safe_number(task_score),
    }

    return {
        "status": "completed",
        "retake_required": False,
        "failure_reason": None,
        "readiness_score": readiness_score,
        "risk_level": risk_level,
        "confidence": safe_number(confidence),
        "camera_confidence": safe_number(face_score),
        "voice_confidence": safe_number(profiles["audio"]["score"]),
        "task_performance_score": int(round((task_score or 0.0) * 100)) if task_score is not None else None,
        "baseline_used": baseline_used,
        "confidence_drift": confidence_drift,
        "face_metrics": {
            "image_score": safe_number(image_score),
            "video_score": safe_number(video_score),
            "face_score": safe_number(face_score),
            "baseline_drift": face_drift,
        },
        "voice_metrics": {
            "voice_score": safe_number(profiles["audio"]["score"]),
            "baseline_drift": voice_drift,
        },
        "reaction_metrics": {
            "reaction_score": safe_number(task_score),
            "reaction_time": _task_value(task, "reaction_time"),
            "errors": _task_value(task, "errors"),
            "attempts": _task_value(task, "attempts"),
            "baseline_drift": reaction_drift,
        },
        "explanation": explanation,
        "suggested_action": suggested_action,
        "ai_model_version": MODEL_VERSION,
        "modality_scores": modality_scores,
        "fusion_details": {
            "signal_profiles": profiles,
            "adaptive_weights": calibration["weights"],
            "baseline_flags": baseline_flags,
            "quality_penalty": safe_number(quality_penalty),
            "fused_score": safe_number(fused_score),
            "calibration": calibration,
        },
    }
