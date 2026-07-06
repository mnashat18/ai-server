from __future__ import annotations

from typing import Any

from baseline import baseline_feature_reference
from config import MODEL_VERSION
from utils import clamp01, clean_warning_codes, safe_number, sanitize_text

VALID_RISK_LEVELS = {"stable", "low_focus", "elevated_fatigue", "high_risk"}
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
    "sustained_eye_closure": 0.16,
    "subject_not_visible": 0.18,
    "audio_too_noisy": 0.18,
    "audio_too_quiet": 0.18,
    "speech_not_detected": 0.22,
    "audio_too_short": 0.1,
    "too_much_silence": 0.18,
    "audio_clipping": 0.18,
    "insufficient_usable_frames": 0.18,
    "landmark_detection_failed": 0.18,
    "image_blurry": 0.14,
    "image_too_dark": 0.14,
    "low_quality_media": 0.12,
}

EYE_CLOSURE_SUSTAINED_THRESHOLD = 0.55
EYE_CLOSURE_HIGH_THRESHOLD = 0.38
EYE_APERTURE_FATIGUE_THRESHOLD = 0.24
VOICE_FATIGUE_SPEECH_THRESHOLD = 0.58
VOICE_FATIGUE_ENERGY_THRESHOLD = 0.018
VOICE_FATIGUE_SILENCE_THRESHOLD = 0.45


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


def _baseline_drift(current: float | None, stat: dict | None) -> dict | None:
    if current is None or not stat or stat.get("median") is None:
        return None
    median_value = float(stat["median"])
    mad = float(stat.get("mad") or 0.0)
    threshold = max(0.1, mad * 2.5)
    drift = round(float(current) - median_value, 4)
    return {
        "current": safe_number(current),
        "baseline_median": safe_number(median_value),
        "baseline_mad": safe_number(mad),
        "drift": safe_number(drift),
        "threshold": safe_number(threshold),
        "below_threshold": drift <= (-threshold),
    }


def _raw_baseline_observations(signals: dict | None) -> dict[str, float | None]:
    signals = signals or {}
    camera_details = ((signals.get("camera") or {}).get("details") or {}) if isinstance(signals.get("camera"), dict) else {}
    voice_details = ((signals.get("voice") or {}).get("details") or {}) if isinstance(signals.get("voice"), dict) else {}
    return {
        "open_eye_aperture": safe_number(camera_details.get("avg_ear")),
        "left_right_eye_asymmetry": safe_number(camera_details.get("left_right_eye_asymmetry")),
        "normalized_voice_energy": safe_number(voice_details.get("rms_energy"), 6),
        "speech_rate": safe_number(voice_details.get("speech_rate")),
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
            "sustained_eye_closure",
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


def _raw_signal_details(signals: dict | None, modality: str) -> dict[str, Any]:
    signals = signals or {}
    analysis = signals.get(modality) or {}
    if not isinstance(analysis, dict):
        return {}
    details = analysis.get("details") or {}
    return details if isinstance(details, dict) else {}


def _video_fatigue_signal(
    *,
    signals: dict | None,
    quality: dict | None,
    baseline_flags: list[str] | None,
    eye_aperture_drift: dict | None,
) -> float:
    details = _raw_signal_details(signals, "video")
    if not details:
        return 0.0
    if details.get("sustained_eye_closure"):
        return 1.0

    reliable_eye_landmarks = bool(details.get("reliable_eye_landmarks"))
    if not reliable_eye_landmarks:
        return 0.0

    closed_eye_ratio = clamp01(details.get("closed_eye_ratio"), 0.0) or 0.0
    avg_eye_aperture = details.get("avg_eye_aperture")
    eye_aperture = 0.0
    if avg_eye_aperture is not None:
        eye_aperture = clamp01((EYE_APERTURE_FATIGUE_THRESHOLD - float(avg_eye_aperture)) / EYE_APERTURE_FATIGUE_THRESHOLD, 0.0) or 0.0
    closure_ratio = clamp01((closed_eye_ratio - 0.12) / max(EYE_CLOSURE_HIGH_THRESHOLD - 0.12, 1e-6), 0.0) or 0.0

    closure_streak = float(details.get("longest_eye_closure_streak") or 0.0)
    closure_window_seconds = float(details.get("eye_closure_window_seconds") or 0.0)
    closure_streak_score = clamp01(closure_streak / 4.0, 0.0) or 0.0
    closure_window_score = clamp01(closure_window_seconds / 1.5, 0.0) or 0.0
    motion_stability = clamp01(details.get("motion_stability_score"), 1.0) or 1.0
    motion_penalty = max(0.0, 1.0 - motion_stability)
    quality_score = clamp01(details.get("visual_quality_score"), 0.0) or 0.0

    fatigue = (
        0.28 * closed_eye_ratio
        + 0.16 * closure_ratio
        + 0.25 * eye_aperture
        + 0.15 * closure_streak_score
        + 0.1 * closure_window_score
        + 0.1 * motion_penalty
        + 0.05 * max(0.0, 1.0 - quality_score)
    )
    if closed_eye_ratio >= EYE_CLOSURE_SUSTAINED_THRESHOLD:
        fatigue += 0.14
    if (eye_aperture_drift or {}).get("below_threshold"):
        fatigue += 0.12
    if baseline_flags:
        fatigue += min(0.04 * len(baseline_flags), 0.12)
    if (quality or {}).get("weak"):
        fatigue += 0.04
    return round(clamp01(fatigue, 0.0) or 0.0, 4)


def _audio_fatigue_signal(
    *,
    signals: dict | None,
    quality: dict | None,
    baseline_flags: list[str] | None,
    voice_energy_drift: dict | None,
    speech_rate_drift: dict | None,
) -> float:
    details = _raw_signal_details(signals, "voice")
    if not details:
        return 0.0

    speech_presence_score = clamp01(details.get("speech_presence_score"), 0.0) or 0.0
    rms_energy = float(details.get("rms_energy") or details.get("energy") or 0.0)
    silence_ratio = clamp01(details.get("silence_ratio"), 0.0) or 0.0
    speech_rate = details.get("speech_rate")
    quiet_but_usable = bool(details.get("quiet_but_usable"))
    normalized_energy = clamp01(rms_energy / max(VOICE_FATIGUE_ENERGY_THRESHOLD * 2.0, 1e-6), 0.0) or 0.0
    speech_rate_factor = 0.0
    if speech_rate is not None:
        speech_rate_factor = 1.0 - (clamp01(float(speech_rate) / 2.1, 0.0) or 0.0)

    fatigue = (
        0.3 * (1.0 - speech_presence_score)
        + 0.24 * (1.0 - normalized_energy)
        + 0.2 * max(0.0, silence_ratio - (VOICE_FATIGUE_SILENCE_THRESHOLD - 0.1))
        + 0.14 * speech_rate_factor
    )
    if quiet_but_usable:
        fatigue = max(fatigue, 0.36)
    if speech_presence_score < VOICE_FATIGUE_SPEECH_THRESHOLD:
        fatigue += 0.06
    if (voice_energy_drift or {}).get("below_threshold"):
        fatigue += 0.1
    if (speech_rate_drift or {}).get("below_threshold"):
        fatigue += 0.06
    if baseline_flags:
        fatigue += min(0.03 * len(baseline_flags), 0.09)
    if (quality or {}).get("weak"):
        fatigue += 0.03
    return round(clamp01(fatigue, 0.0) or 0.0, 4)


def _fatigue_signal_context(
    *,
    signals: dict | None,
    quality: dict | None,
    baseline_used: bool,
    baseline_flags: list[str],
    eye_aperture_drift: dict | None,
    voice_energy_drift: dict | None,
    speech_rate_drift: dict | None,
) -> dict[str, Any]:
    video_fatigue = _video_fatigue_signal(
        signals=signals,
        quality=quality,
        baseline_flags=baseline_flags,
        eye_aperture_drift=eye_aperture_drift if baseline_used else None,
    )
    audio_fatigue = _audio_fatigue_signal(
        signals=signals,
        quality=quality,
        baseline_flags=baseline_flags,
        voice_energy_drift=voice_energy_drift if baseline_used else None,
        speech_rate_drift=speech_rate_drift if baseline_used else None,
    )
    combined = 0.62 * video_fatigue + 0.33 * audio_fatigue
    if baseline_used and baseline_flags:
        combined += min(0.03 * len(baseline_flags), 0.08)
    if (quality or {}).get("weak"):
        combined += 0.04
    combined = round(clamp01(combined, 0.0) or 0.0, 4)
    return {
        "video": safe_number(video_fatigue),
        "audio": safe_number(audio_fatigue),
        "combined": safe_number(combined),
    }


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


def _invalid_scan_outcome(readiness_score: int, confidence: float, quality: dict) -> tuple[int, float]:
    adjusted_score = min(readiness_score, 51)
    adjusted_confidence = min(confidence, 0.44)
    if quality.get("failure_reason") == "missing_media":
        adjusted_score = min(adjusted_score, 35)
    elif quality.get("failure_reason") == "low_quality_media":
        adjusted_score = min(adjusted_score, 45)
    return adjusted_score, adjusted_confidence


def _confirmed_sustained_eye_closure(quality: dict) -> bool:
    return "sustained_eye_closure" in set(quality.get("warnings") or [])


def _quality_limited_scan(quality: dict, confidence: float) -> bool:
    return bool(
        quality.get("status") == "failed"
        or quality.get("retake_required")
        or quality.get("failure_reason") in {"low_quality_media", "missing_media"}
        or bool({"video", "audio"} & set(quality.get("missing_modalities") or []))
        or confidence < 0.45
    )


def _risk_level(
    readiness_score: int,
    confidence: float,
    baseline_flags: list[str],
    quality: dict,
    *,
    fatigue_evidence: float = 0.0,
) -> str:
    sustained_eye_closure = _confirmed_sustained_eye_closure(quality)
    if sustained_eye_closure:
        return "elevated_fatigue"
    if _quality_limited_scan(quality, confidence):
        return "low_focus"
    if fatigue_evidence >= 0.82 and confidence >= 0.5:
        return "high_risk"
    if readiness_score < 35 and confidence >= 0.62:
        return "high_risk"
    if fatigue_evidence >= 0.6:
        return "elevated_fatigue"
    if quality.get("weak") and readiness_score < 52 and not baseline_flags:
        return "low_focus"
    if readiness_score < 52 and confidence >= 0.5:
        return "elevated_fatigue"
    if readiness_score < 68 and fatigue_evidence >= 0.3:
        return "low_focus"
    if readiness_score < 68:
        return "low_focus"
    return "stable"


def _suggested_action(risk_level: str | None, confidence: float, quality: dict) -> str:
    if risk_level == "high_risk":
        return "manager_review"
    if quality.get("status") == "failed" or quality.get("retake_required") or quality.get("failure_reason") in {"low_quality_media", "missing_media"}:
        return "rescan_recommended"
    if risk_level == "elevated_fatigue":
        return "rest_advised"
    if risk_level == "low_focus":
        return "review_required" if confidence >= 0.45 else "rescan_recommended"
    if confidence < 0.45 or quality.get("weak"):
        return "rescan_recommended"
    return "continue_normal_activity"


def _explanation(
    profiles: dict[str, dict],
    quality: dict,
    risk_level: str | None,
    confidence: float,
    fatigue_context: dict[str, Any] | None = None,
    baseline_notes: list[str] | None = None,
) -> str:
    positives: list[str] = []
    negatives: list[str] = []
    warning_reasons: list[str] = []
    baseline_notes = baseline_notes or []
    fatigue_context = fatigue_context or {}
    if profiles["video"]["score"] is not None and (profiles["video"]["quality"] or 0.0) >= 0.55:
        positives.append("Video quality was acceptable")
    if profiles["audio"]["score"] is not None and (profiles["audio"]["quality"] or 0.0) >= 0.55:
        positives.append("voice signal was clear")
    if profiles["image"]["score"] is not None and (profiles["image"]["quality"] or 0.0) >= 0.55:
        positives.append("thumbnail quality was usable")
    if baseline_notes:
        positives.extend(baseline_notes)
    if "audio_too_noisy" in quality.get("warnings", []):
        negatives.append("Audio was noisy")
        warning_reasons.append("background noise reduced voice confidence")
    if "audio_too_quiet" in quality.get("warnings", []):
        negatives.append("audio volume was low")
        warning_reasons.append("low voice volume reduced confidence")
    if "speech_not_detected" in quality.get("warnings", []):
        negatives.append("no usable speech was detected")
        warning_reasons.append("missing or unusable speech reduced confidence")
    if "too_much_silence" in quality.get("warnings", []):
        negatives.append("audio was mostly silence")
        warning_reasons.append("insufficient voice activity reduced confidence")
    if "audio_clipping" in quality.get("warnings", []):
        negatives.append("audio was clipped")
        warning_reasons.append("clipping reduced voice confidence")
    if "video_blurry" in quality.get("warnings", []) or "image_blurry" in quality.get("warnings", []):
        negatives.append("visual media was partially blurred")
        warning_reasons.append("blur reduced visual confidence")
    if "video_too_dark" in quality.get("warnings", []) or "image_too_dark" in quality.get("warnings", []):
        negatives.append("lighting was too dark")
        warning_reasons.append("low lighting reduced visual confidence")
    if "subject_not_visible" in quality.get("warnings", []):
        negatives.append("subject visibility was limited")
        warning_reasons.append("limited subject visibility reduced confidence")
    if "insufficient_usable_frames" in quality.get("warnings", []):
        negatives.append("too few usable video frames were available")
        warning_reasons.append("insufficient usable frames reduced confidence")
    if "landmark_detection_failed" in quality.get("warnings", []):
        negatives.append("face landmarks were not reliably detected")
        warning_reasons.append("unreliable face detection reduced confidence")
    if "face_not_visible" in quality.get("warnings", []) or "unstable_video" in quality.get("warnings", []):
        negatives.append("face visibility was weak")
        warning_reasons.append("weak face visibility reduced confidence")
    if "sustained_eye_closure" in quality.get("warnings", []):
        negatives.append("reliable sustained eye closure was observed")
        warning_reasons.append("repeated eye-closure evidence lowered the readiness result")
    if (fatigue_context.get("video") or 0.0) >= 0.45:
        negatives.append("video showed fatigue-like eye behavior")
        warning_reasons.append("eye behavior suggested the person may be tired")
    if (fatigue_context.get("audio") or 0.0) >= 0.45:
        negatives.append("voice energy and pacing suggested fatigue")
        warning_reasons.append("voice pattern suggested the person may be tired")
    if (fatigue_context.get("combined") or 0.0) >= 0.55:
        warning_reasons.append("combined video and audio fatigue cues were elevated")
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
    confirmed_sustained_eye_closure = _confirmed_sustained_eye_closure(quality)
    scan_unreliable = (
        quality.get("status") == "failed"
        or quality.get("retake_required")
        or quality.get("failure_reason") in {"low_quality_media", "missing_media"}
        or bool({"video", "audio"} & set(quality.get("missing_modalities") or []))
        or confidence < 0.45
    )
    if confirmed_sustained_eye_closure:
        scan_unreliable = False
    quality_limited = _quality_limited_scan(quality, confidence) and not confirmed_sustained_eye_closure
    fatigue_likely = (
        confirmed_sustained_eye_closure
        or (fatigue_context.get("combined") or 0.0) >= 0.55
        or (fatigue_context.get("audio") or 0.0) >= 0.45
        or (fatigue_context.get("video") or 0.0) >= 0.45
    )
    if quality_limited and not fatigue_likely:
        tail = "The result is weak because the scan quality was poor. Please retake it with clearer lighting, a steadier video, and a visible face."
    elif confirmed_sustained_eye_closure or fatigue_likely:
        tail = "The result suggests real fatigue. You look tired and should rest before continuing."
    elif scan_unreliable:
        tail = "The scan could not be assessed reliably. Please repeat it with clear face visibility, better lighting, steady video, and usable audio."
    elif risk_level == "stable":
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
    raw_features = _raw_baseline_observations(signals)
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

    eye_aperture_drift = _baseline_drift(
        raw_features.get("open_eye_aperture"),
        baseline_feature_reference(baseline, "face_avg", "open_eye_aperture"),
    ) if baseline_used else None
    eye_asymmetry_drift = _baseline_drift(
        raw_features.get("left_right_eye_asymmetry"),
        baseline_feature_reference(baseline, "face_avg", "left_right_eye_asymmetry"),
    ) if baseline_used else None
    voice_energy_drift = _baseline_drift(
        raw_features.get("normalized_voice_energy"),
        baseline_feature_reference(baseline, "voice_avg", "normalized_voice_energy"),
    ) if baseline_used else None
    speech_rate_drift = _baseline_drift(
        raw_features.get("speech_rate"),
        baseline_feature_reference(baseline, "voice_avg", "speech_rate"),
    ) if baseline_used else None
    baseline_flags = [
        name
        for name, drift in {
            "open_eye_aperture": eye_aperture_drift,
            "left_right_eye_asymmetry": eye_asymmetry_drift,
            "normalized_voice_energy": voice_energy_drift,
            "speech_rate": speech_rate_drift,
        }.items()
        if drift and drift.get("below_threshold")
    ]

    if baseline_flags:
        fused_score = max(0.0, fused_score - (0.03 * len(baseline_flags)))
    if "sustained_eye_closure" in set(quality.get("warnings") or []):
        fused_score = max(0.0, fused_score - 0.14)

    fatigue_context = _fatigue_signal_context(
        signals=signals,
        quality=quality,
        baseline_used=baseline_used,
        baseline_flags=baseline_flags,
        eye_aperture_drift=eye_aperture_drift,
        voice_energy_drift=voice_energy_drift,
        speech_rate_drift=speech_rate_drift,
    )
    if baseline_used:
        baseline_fatigue_boost = 0.0
        if (eye_aperture_drift or {}).get("below_threshold"):
            baseline_fatigue_boost += 0.08
        if (voice_energy_drift or {}).get("below_threshold"):
            baseline_fatigue_boost += 0.07
        if (speech_rate_drift or {}).get("below_threshold"):
            baseline_fatigue_boost += 0.05
        if baseline_fatigue_boost:
            fatigue_context["combined"] = safe_number(
                clamp01((fatigue_context.get("combined") or 0.0) + baseline_fatigue_boost, 0.0)
            )

    baseline_notes: list[str] = []
    if baseline_used:
        eye_aperture_reference = baseline_feature_reference(baseline, "face_avg", "open_eye_aperture")
        voice_energy_reference = baseline_feature_reference(baseline, "voice_avg", "normalized_voice_energy")
        speech_rate_reference = baseline_feature_reference(baseline, "voice_avg", "speech_rate")
        if (
            raw_features.get("open_eye_aperture") is not None
            and eye_aperture_reference is not None
            and not (eye_aperture_drift or {}).get("below_threshold")
        ):
            baseline_notes.append("Eye appearance was within this employee's established normal range")
        if (
            raw_features.get("normalized_voice_energy") is not None
            and voice_energy_reference is not None
            and not (voice_energy_drift or {}).get("below_threshold")
        ):
            baseline_notes.append("Quiet but usable speech was consistent with the employee's established baseline")
        if (
            raw_features.get("speech_rate") is not None
            and speech_rate_reference is not None
            and not (speech_rate_drift or {}).get("below_threshold")
        ):
            baseline_notes.append("Speech timing was consistent with the employee's established baseline")

    confidence, calibration = _confidence_from_profiles(
        fused_score=fused_score,
        profiles=profiles,
        weights=weights,
        quality=quality,
        baseline_used=baseline_used,
        ml_result=ml_result,
    )
    readiness_score = int(round((clamp01(fused_score, 0.0) or 0.0) * 100))
    confirmed_sustained_eye_closure = _confirmed_sustained_eye_closure(quality)
    fatigue_evidence = float(fatigue_context.get("combined") or 0.0)
    scan_unreliable = (
        quality.get("status") == "failed"
        or quality.get("retake_required")
        or quality.get("failure_reason") in {"low_quality_media", "missing_media"}
        or bool({"video", "audio"} & set(quality.get("missing_modalities") or []))
        or confidence < 0.45
        or (quality.get("weak") and readiness_score < 52 and not baseline_flags)
    )
    if confirmed_sustained_eye_closure:
        scan_unreliable = False
        readiness_score = min(readiness_score, 30)
    if scan_unreliable:
        readiness_score, confidence = _invalid_scan_outcome(readiness_score, confidence, quality)
    risk_level = _risk_level(
        readiness_score,
        confidence,
        baseline_flags,
        quality,
        fatigue_evidence=fatigue_evidence,
    )
    if scan_unreliable and risk_level is None:
        risk_level = "low_focus"
    if risk_level is not None and risk_level not in VALID_RISK_LEVELS:
        risk_level = "low_focus"
    suggested_action = _suggested_action(risk_level, confidence, quality)
    if suggested_action not in VALID_ACTIONS:
        suggested_action = "rescan_recommended"
    explanation = _explanation(
        profiles,
        quality,
        risk_level,
        confidence,
        fatigue_context=fatigue_context,
        baseline_notes=baseline_notes,
    )

    previous = clamp01(previous_confidence, confidence)
    confidence_drift = safe_number(confidence - (previous if previous is not None else confidence))
    modality_scores = {
        "video": safe_number(profiles["video"]["score"]),
        "audio": safe_number(profiles["audio"]["score"]),
        "image": safe_number(profiles["image"]["score"]),
        "task": safe_number(task_score),
    }
    observed_fatigue_score = int(round(max(
        1.0 - (clamp01(fused_score, 0.0) or 0.0),
        fatigue_evidence,
    ) * 100))
    retake_required = bool((scan_unreliable or quality.get("retake_required")) and not confirmed_sustained_eye_closure)

    return {
        "status": "completed",
        "retake_required": retake_required,
        "failure_reason": quality.get("failure_reason") if retake_required else None,
        "readiness_score": readiness_score,
        "observed_fatigue_score": observed_fatigue_score,
        "risk_level": risk_level,
        "confidence": safe_number(confidence),
        "camera_confidence": safe_number(face_score),
        "voice_confidence": safe_number(profiles["audio"]["score"]),
        "task_performance_score": int(round((task_score or 0.0) * 100)) if task_score is not None else None,
        "baseline_used": baseline_used,
        "confidence_drift": confidence_drift,
        "fatigue_evidence_score": safe_number(fatigue_evidence),
        "face_metrics": {
            "image_score": safe_number(image_score),
            "video_score": safe_number(video_score),
            "face_score": safe_number(face_score),
            "open_eye_aperture": raw_features.get("open_eye_aperture"),
            "left_right_eye_asymmetry": raw_features.get("left_right_eye_asymmetry"),
            "baseline_drifts": {
                "open_eye_aperture": eye_aperture_drift,
                "left_right_eye_asymmetry": eye_asymmetry_drift,
            },
        },
        "voice_metrics": {
            "voice_score": safe_number(profiles["audio"]["score"]),
            "normalized_voice_energy": raw_features.get("normalized_voice_energy"),
            "speech_rate": raw_features.get("speech_rate"),
            "baseline_drifts": {
                "normalized_voice_energy": voice_energy_drift,
                "speech_rate": speech_rate_drift,
            },
        },
        "reaction_metrics": {
            "reaction_score": safe_number(task_score),
            "reaction_time": _task_value(task, "reaction_time"),
            "errors": _task_value(task, "errors"),
            "attempts": _task_value(task, "attempts"),
            "baseline_drifts": {},
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
            "observed_fatigue_score": observed_fatigue_score,
            "fatigue_evidence_score": safe_number(fatigue_evidence),
            "fatigue_context": fatigue_context,
            "calibration": calibration,
        },
    }
