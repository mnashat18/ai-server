#scoring.py
from config import (
    WEIGHT_CAMERA,
    WEIGHT_VIDEO,
    WEIGHT_VOICE,
    WEIGHT_TASK,
    ML_WEIGHT,
    MISSING_MEDIA_PENALTY,
    BASELINE_DRIFT_THRESHOLD,
    BASELINE_DRIFT_PENALTY,
    TASK_RT_GOOD,
    TASK_RT_MED,
    TASK_ERR_GOOD,
    TASK_ERR_MED,
)
from report import generate_medical_report


def bucket_score(value, good, medium):
    if value >= good:
        return 1.0
    elif value >= medium:
        return 0.6
    else:
        return 0.3


def bucket_score_low(value, good, medium):
    if value <= good:
        return 1.0
    elif value <= medium:
        return 0.6
    else:
        return 0.3


def clamp_score(value):
    try:
        return max(0.0, min(float(value), 1.0))
    except (TypeError, ValueError):
        return 0.5


def compute_drift(current, previous):
    if previous is None:
        return 0.0
    return round(current - previous, 2)


def _weighted_average(scores: dict, weights: dict) -> float:
    if not scores:
        return 0.0
    weight_sum = sum(weights[k] for k in scores)
    if weight_sum == 0:
        return 0.0
    return sum(scores[k] * weights[k] for k in scores) / weight_sum


def _get_task_value(task, key):
    if task is None:
        return None
    if isinstance(task, dict):
        return task.get(key)
    return getattr(task, key, None)


def compute_task_score(task):
    if not task:
        return None
    scores = []
    reaction_time = _get_task_value(task, "reaction_time")
    errors = _get_task_value(task, "errors")

    if reaction_time is not None:
        scores.append(bucket_score_low(reaction_time, TASK_RT_GOOD, TASK_RT_MED))
    if errors is not None:
        scores.append(bucket_score_low(errors, TASK_ERR_GOOD, TASK_ERR_MED))

    if not scores:
        return None
    return round(sum(scores) / len(scores), 2)


def _baseline_confidence(baseline: dict | None, weights: dict) -> float | None:
    if not baseline:
        return None
    scores = {k: baseline.get(k) for k in weights if baseline.get(k) is not None}
    if not scores:
        return None
    return round(_weighted_average(scores, weights), 2)


def compute_result(camera, video, voice, previous_confidence=None, task=None, baseline=None, ml_result=None):
    camera_score = clamp_score(camera) if camera is not None else None
    video_score = clamp_score(video) if video is not None else None
    voice_score = clamp_score(voice) if voice is not None else None

    signals = {
        "camera": camera_score,
        "video": video_score,
        "voice": voice_score,
    }
    weights = {
        "camera": WEIGHT_CAMERA,
        "video": WEIGHT_VIDEO,
        "voice": WEIGHT_VOICE,
    }

    present = {k: v for k, v in signals.items() if v is not None}
    missing_media = [k for k, v in signals.items() if v is None]
    missing_count = len(missing_media)

    base_confidence = _weighted_average(present, weights)
    missing_penalty = missing_count * MISSING_MEDIA_PENALTY
    confidence = max(0.0, base_confidence - missing_penalty)

    task_score = compute_task_score(task)
    if task_score is not None:
        confidence = (confidence * (1 - WEIGHT_TASK)) + (task_score * WEIGHT_TASK)

    baseline_conf = _baseline_confidence(baseline, weights)
    baseline_penalty = 0.0
    if baseline_conf is not None and (baseline_conf - confidence) >= BASELINE_DRIFT_THRESHOLD:
        baseline_penalty = BASELINE_DRIFT_PENALTY
        confidence = max(0.0, confidence - baseline_penalty)

    ml_confidence = None
    if ml_result and isinstance(ml_result, dict):
        ml_confidence = ml_result.get("confidence")
    if ml_confidence is not None and ML_WEIGHT > 0:
        confidence = (confidence * (1 - ML_WEIGHT)) + (ml_confidence * ML_WEIGHT)

    confidence = round(min(max(confidence, 0.0), 1.0), 2)
    drift = compute_drift(confidence, previous_confidence)
    baseline_drift = None if baseline_conf is None else round(confidence - baseline_conf, 2)

    if confidence >= 0.7:
        state = "Stable"
    elif confidence >= 0.55:
        state = "Low Focus"
    elif confidence >= 0.4:
        state = "Elevated Fatigue"
    else:
        state = "High Risk"

    if state == "High Risk":
        explanation = "Strong instability detected across signals; recommend immediate pause and further assessment."
    elif state == "Elevated Fatigue":
        explanation = "Reduced alertness detected, consistent with physical or cognitive fatigue."
    elif state == "Low Focus":
        explanation = "Mild attention drift detected; short rest is recommended."
    else:
        explanation = "Normal psychomotor patterns detected."

    medical = generate_medical_report(
        state,
        confidence,
        camera_score or 0.0,
        video_score or 0.0,
        voice_score or 0.0,
        missing_media=missing_media,
    )
    task_perf = int(round((task_score if task_score is not None else confidence) * 100))

    alerts = []
    if missing_media:
        alerts.append("missing_media")
    if baseline_conf is not None and baseline_drift is not None:
        if baseline_drift <= -BASELINE_DRIFT_THRESHOLD:
            alerts.append("baseline_drop")
    if confidence < 0.4:
        alerts.append("high_risk")

    return {
        "overall_state": state,
        "confidence": confidence,
        "confidence_drift": drift,
        "baseline_confidence": baseline_conf,
        "baseline_drift": baseline_drift,
        "camera_confidence": camera_score or 0.0,
        "video_confidence": video_score or 0.0,
        "voice_confidence": voice_score or 0.0,
        "task_performance_score": task_perf,
        "task_score": task_score,
        "missing_media": missing_media,
        "alerts": alerts,
        "explanation": explanation,
        "medical_report": medical,
        "confidence_components": {
            "base_confidence": round(base_confidence, 3),
            "missing_penalty": round(missing_penalty, 3),
            "task_score": task_score,
            "task_weight": WEIGHT_TASK if task_score is not None else 0.0,
            "baseline_penalty": baseline_penalty,
            "ml_confidence": ml_confidence,
            "ml_weight": ML_WEIGHT if ml_confidence is not None else 0.0,
        },
    }
