from __future__ import annotations

from typing import Any

from config import (
    BASELINE_MIN_DRIFT,
    BASELINE_STD_MULTIPLIER,
    LOW_CONFIDENCE_THRESHOLD,
    ML_WEIGHT,
    MODEL_VERSION,
    READINESS_FACE_WEIGHT,
    READINESS_ML_BLEND,
    READINESS_REACTION_WEIGHT,
    READINESS_VOICE_WEIGHT,
    TASK_ERR_GOOD,
    TASK_ERR_MED,
    TASK_RT_GOOD,
    TASK_RT_MED,
)


def _clamp(value: float | None, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return max(0.0, min(float(value), 1.0))
    except (TypeError, ValueError):
        return default


def _weighted_average(scores: dict[str, float], weights: dict[str, float]) -> float:
    if not scores:
        return 0.0
    total_weight = sum(weights[k] for k in scores)
    if total_weight <= 0:
        return 0.0
    return sum(scores[k] * weights[k] for k in scores) / total_weight


def _task_value(task: Any, key: str):
    if task is None:
        return None
    if isinstance(task, dict):
        return task.get(key)
    return getattr(task, key, None)


def bucket_score_low(value, good, medium):
    if value <= good:
        return 1.0
    if value <= medium:
        return 0.6
    return 0.3


def compute_task_score(task) -> float | None:
    if not task:
        return None
    scores = []
    reaction_time = _task_value(task, "reaction_time")
    errors = _task_value(task, "errors")
    attempts = _task_value(task, "attempts")

    if reaction_time is not None:
        scores.append(bucket_score_low(float(reaction_time), TASK_RT_GOOD, TASK_RT_MED))
    if errors is not None:
        scores.append(bucket_score_low(int(errors), TASK_ERR_GOOD, TASK_ERR_MED))
    if attempts is not None:
        scores.append(1.0 if int(attempts) >= 3 else 0.6 if int(attempts) >= 1 else 0.3)
    if not scores:
        return None
    return round(sum(scores) / len(scores), 3)


def _combine_face_score(camera_score: float | None, video_score: float | None) -> float | None:
    present = {}
    weights = {}
    if camera_score is not None:
        present["camera"] = camera_score
        weights["camera"] = 0.45
    if video_score is not None:
        present["video"] = video_score
        weights["video"] = 0.55
    if not present:
        return None
    return round(_weighted_average(present, weights), 3)


def _baseline_stat(baseline: dict | None, key: str) -> dict | None:
    if not baseline:
        return None
    value = baseline.get(key)
    return value if isinstance(value, dict) else None


def _baseline_drift(current: float | None, stat: dict | None) -> dict | None:
    if current is None or not stat or stat.get("avg") is None:
        return None
    avg = float(stat.get("avg"))
    std = stat.get("std")
    threshold = max(BASELINE_MIN_DRIFT, float(std or 0.0) * BASELINE_STD_MULTIPLIER)
    drift = round(float(current) - avg, 3)
    return {
        "current": round(float(current), 3),
        "baseline_avg": round(avg, 3),
        "baseline_std": round(float(std), 3) if std is not None else None,
        "drift": drift,
        "threshold": round(threshold, 3),
        "below_threshold": drift <= (-threshold),
    }


def _explanation(risk_level: str, drift_flags: list[str], baseline_used: bool) -> str:
    if risk_level == "high_risk":
        if baseline_used:
            return "Readiness dropped clearly below the employee's typical pattern across multiple signals."
        return "Multiple signals point to reduced readiness with strong agreement."
    if risk_level == "elevated_fatigue":
        if baseline_used:
            return "Current signals are below the employee's usual readiness pattern and suggest elevated fatigue."
        return "Current signals suggest elevated fatigue with moderate agreement."
    if risk_level == "low_focus":
        if baseline_used and drift_flags:
            return "A mild drop from the employee's normal pattern suggests lower focus right now."
        return "Signals suggest a mild dip in focus, but not a broad readiness drop."
    return "Signals are within the expected readiness range."


def _suggested_action(risk_level: str, confidence: float, quality_weak: bool) -> str:
    if risk_level == "high_risk":
        return "Pause demanding work, retake the scan after a short break, and escalate if the next scan stays high risk."
    if risk_level == "elevated_fatigue":
        return "Take a short break and consider a follow-up scan before continuing demanding work."
    if risk_level == "low_focus":
        return "A quick reset or short break is recommended before the next demanding task."
    if quality_weak or confidence < 0.65:
        return "Readiness looks stable, but a clearer retake can improve confidence."
    return "No action needed."


def compute_result(
    *,
    camera_score: float | None,
    video_score: float | None,
    voice_score: float | None,
    task=None,
    previous_confidence: float | None = None,
    baseline: dict | None = None,
    baseline_used: bool = False,
    quality: dict | None = None,
    ml_result: dict | None = None,
) -> dict:
    camera_score = _clamp(camera_score)
    video_score = _clamp(video_score)
    voice_score = _clamp(voice_score)
    reaction_score = compute_task_score(task)
    face_score = _combine_face_score(camera_score, video_score)

    modality_scores = {}
    weights = {}
    if face_score is not None:
        modality_scores["face"] = face_score
        weights["face"] = READINESS_FACE_WEIGHT
    if voice_score is not None:
        modality_scores["voice"] = voice_score
        weights["voice"] = READINESS_VOICE_WEIGHT
    if reaction_score is not None:
        modality_scores["reaction"] = reaction_score
        weights["reaction"] = READINESS_REACTION_WEIGHT

    readiness = _weighted_average(modality_scores, weights)
    ml_confidence = ml_result.get("confidence") if isinstance(ml_result, dict) else None
    if ml_confidence is not None:
        readiness = (readiness * (1 - READINESS_ML_BLEND)) + (float(ml_confidence) * READINESS_ML_BLEND)

    face_drift = _baseline_drift(face_score, _baseline_stat(baseline, "face_avg")) if baseline_used else None
    voice_drift = _baseline_drift(voice_score, _baseline_stat(baseline, "voice_avg")) if baseline_used else None
    reaction_drift = _baseline_drift(reaction_score, _baseline_stat(baseline, "reaction_avg")) if baseline_used else None

    drift_flags = [
        name
        for name, drift in {
            "face": face_drift,
            "voice": voice_drift,
            "reaction": reaction_drift,
        }.items()
        if drift and drift.get("below_threshold")
    ]

    if baseline_used and drift_flags:
        readiness -= 0.06 * len(drift_flags)

    quality_multiplier = float((quality or {}).get("confidence_multiplier", 1.0))
    readiness = max(0.0, min(readiness, 1.0))

    coverage = len(modality_scores) / 3.0
    baseline_bonus = 0.1 if baseline_used else 0.0
    confidence = min(1.0, max(0.0, (0.45 * quality_multiplier) + (0.35 * coverage) + baseline_bonus))
    if drift_flags and baseline_used:
        confidence = min(1.0, confidence + 0.05)
    if (quality or {}).get("weak"):
        confidence = max(0.0, confidence - 0.08)
    confidence = round(confidence, 3)

    readiness_score = int(round(max(0.0, min(1.0, readiness)) * 100))
    previous = previous_confidence if previous_confidence is not None else confidence
    confidence_drift = round(confidence - float(previous), 3)

    low_modalities = sum(1 for score in modality_scores.values() if score < 0.45)
    medium_modalities = sum(1 for score in modality_scores.values() if score < 0.6)

    if baseline_used:
        if len(drift_flags) >= 2 and readiness_score < 45:
            risk_level = "high_risk"
        elif len(drift_flags) >= 2 or (len(drift_flags) == 1 and readiness_score < 55):
            risk_level = "elevated_fatigue"
        elif len(drift_flags) == 1 or (medium_modalities >= 2 and readiness_score < 65):
            risk_level = "low_focus"
        else:
            risk_level = "stable"
    else:
        if low_modalities >= 2 and readiness_score < 40 and confidence >= LOW_CONFIDENCE_THRESHOLD:
            risk_level = "high_risk"
        elif medium_modalities >= 2 and readiness_score < 52 and confidence >= LOW_CONFIDENCE_THRESHOLD:
            risk_level = "elevated_fatigue"
        elif medium_modalities >= 1 and readiness_score < 65 and confidence >= LOW_CONFIDENCE_THRESHOLD:
            risk_level = "low_focus"
        else:
            risk_level = "stable"

    explanation = _explanation(risk_level, drift_flags, baseline_used)
    suggested_action = _suggested_action(risk_level, confidence, bool((quality or {}).get("weak")))

    return {
        "status": "completed",
        "retake_required": False,
        "failure_reason": None,
        "readiness_score": readiness_score,
        "risk_level": risk_level,
        "overall_state": risk_level,
        "confidence": confidence,
        "camera_confidence": round(face_score or 0.0, 3),
        "voice_confidence": round(voice_score or 0.0, 3),
        "task_performance_score": int(round((reaction_score or 0.0) * 100)),
        "baseline_used": baseline_used,
        "confidence_drift": confidence_drift,
        "face_metrics": {
            "camera_score": camera_score,
            "video_score": video_score,
            "face_score": face_score,
            "baseline_drift": face_drift,
        },
        "voice_metrics": {
            "voice_score": voice_score,
            "baseline_drift": voice_drift,
        },
        "reaction_metrics": {
            "reaction_score": reaction_score,
            "reaction_time": _task_value(task, "reaction_time"),
            "errors": _task_value(task, "errors"),
            "attempts": _task_value(task, "attempts"),
            "baseline_drift": reaction_drift,
        },
        "explanation": explanation,
        "suggested_action": suggested_action,
        "ai_model_version": MODEL_VERSION,
        "confidence_components": {
            "modality_scores": modality_scores,
            "quality_multiplier": quality_multiplier,
            "baseline_used": baseline_used,
            "drift_flags": drift_flags,
            "ml_confidence": ml_confidence,
            "ml_weight": ML_WEIGHT if ml_confidence is not None else 0.0,
        },
    }
