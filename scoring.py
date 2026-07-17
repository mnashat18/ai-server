from __future__ import annotations

import math
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
# Each warning below is documented to affect exactly one numeric channel: the
# capture-confidence score of its own modality. Fatigue evidence is derived from
# biometric measurements only and never from these capture-quality warnings.
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

FATIGUE_ELEVATED_THRESHOLD = 0.5
FATIGUE_HIGH_THRESHOLD = 0.82
INVALID_SCAN_CONFIDENCE_CAP = 0.44
CONFIDENCE_RELIABLE_FLOOR = 0.45

# Robust drift is measured in MAD units; a metric that sits this many MAD below the
# personal baseline (fatigue direction) is flagged. Zero MAD is protected by a floor.
BASELINE_MAD_FLOOR = 0.02
BASELINE_DRIFT_Z_THRESHOLD = 2.5

# Audio warnings that mean speech is not usable evidence of fatigue.
_AUDIO_NON_FATIGUE_WARNINGS = {
    "speech_not_detected",
    "audio_too_noisy",
    "audio_clipping",
    "too_much_silence",
}

_MISSING_STATUSES = {
    "missing",
    "open_failed",
    "load_failed",
    "empty_audio",
    "invalid",
    "invalid_image",
    "error",
}


# ---------------------------------------------------------------------------
# strict value coercion helpers (public-input safety)
# ---------------------------------------------------------------------------
def _finite_number(value: Any) -> float | None:
    """Return a finite float, rejecting None, bool, strings and non-finite values.

    Booleans are never treated as numbers and numeric strings are rejected so that
    freshly captured runtime evidence must arrive as a real int/float.
    """
    if value is None or isinstance(value, bool):
        return None
    if not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        return None
    return numeric


def _unit_interval(value: Any) -> float | None:
    """Return value only when it is a finite number in [0, 1]. Out-of-range is invalid,
    not clamped. Exact 0.0 is preserved as valid evidence."""
    numeric = _finite_number(value)
    if numeric is None:
        return None
    if numeric < 0.0 or numeric > 1.0:
        return None
    return numeric


def _bool_flag(value: Any) -> bool:
    """True only when value is exactly the boolean True."""
    return value is True


def _strict_int(value: Any, *, min_value: int | None = None) -> int | None:
    """Accept only a real Python int (never bool, float, string, or object)."""
    if type(value) is not int:
        return None
    if min_value is not None and value < min_value:
        return None
    return value


def clamp_confidence(value: float | None) -> float | None:
    return clamp01(value)


def _task_value(task: Any, key: str):
    """Read task evidence without letting ordinary getter failures escape.

    KeyboardInterrupt and SystemExit are intentionally not caught because they do not
    derive from Exception.
    """
    if task is None:
        return None
    try:
        if isinstance(task, dict):
            return task.get(key)
        return getattr(task, key, None)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# analyzer contract helpers
# ---------------------------------------------------------------------------
def _analysis_details(analysis: Any) -> dict[str, Any]:
    if not isinstance(analysis, dict):
        return {}
    details = analysis.get("details")
    return details if isinstance(details, dict) else {}


def _normalized_status(details: dict[str, Any]) -> str | None:
    status = details.get("status")
    if not isinstance(status, str):
        return None
    normalized = status.strip().casefold()
    return normalized or None


def _status_ok(analysis: Any) -> bool:
    """Only a normalized details.status of "ok" makes a modality readable evidence."""
    return _normalized_status(_analysis_details(analysis)) == "ok"


def _signal_score(analysis: Any, score_key: str) -> float | None:
    """Read a modality's dedicated capture-quality/confidence score.

    A modality contributes a score only when its status is "ok". The dedicated field
    is authoritative; the top-level "score" is a backward-compatibility alias used only
    when the dedicated field is absent. Values outside [0, 1] are invalid (dropped),
    never clamped. A failed/missing status can never be rescued by a top-level score.
    """
    if not _status_ok(analysis):
        return None
    details = _analysis_details(analysis)
    if score_key in details:
        return _unit_interval(details.get(score_key))
    return _unit_interval(analysis.get("score"))


def _signal_warnings(analysis: Any, key: str) -> list[str]:
    """Analyzer-emitted warnings, ignored entirely when the status is not "ok"
    (stale warnings under an unreadable status must not leak through)."""
    if not _status_ok(analysis):
        return []
    details = _analysis_details(analysis)
    return clean_warning_codes(details.get(key) or [])


def _signal_present(analysis: Any) -> bool:
    """A modality is present/readable only when its own status is "ok"."""
    return _status_ok(analysis)


def _missing_modalities_from_profiles(profiles: dict[str, dict]) -> list[str]:
    return [name for name in ("video", "audio", "image") if not profiles.get(name, {}).get("present")]


def _normalize_weights(weight_map: dict[str, float]) -> dict[str, float]:
    total = sum(weight for weight in weight_map.values() if weight > 0.0)
    if total <= 0:
        return {}
    return {name: weight / total for name, weight in weight_map.items() if weight > 0.0}


# ---------------------------------------------------------------------------
# task scoring
# ---------------------------------------------------------------------------
def compute_task_score(task: Any) -> float | None:
    """Score only a completed task with a valid positive integer attempt count.

    Missing or malformed attempts invalidate the entire task. Reaction time and errors
    are optional supporting evidence and cannot rescue an uncompleted task.
    """
    if task is None:
        return None

    attempts = _strict_int(_task_value(task, "attempts"), min_value=1)
    if attempts is None:
        return None

    reaction_time = _finite_number(_task_value(task, "reaction_time"))
    if reaction_time is not None and reaction_time <= 0.0:
        reaction_time = None
    errors = _strict_int(_task_value(task, "errors"), min_value=0)

    scores: list[float] = [0.95 if attempts >= 3 else 0.6]
    if reaction_time is not None:
        scores.append(
            0.92
            if reaction_time <= 0.55
            else 0.72
            if reaction_time <= 0.85
            else 0.45
            if reaction_time <= 1.15
            else 0.22
        )
    if errors is not None:
        scores.append(0.95 if errors == 0 else 0.75 if errors <= 1 else 0.5 if errors <= 3 else 0.2)

    return round(sum(scores) / len(scores), 4)


# ---------------------------------------------------------------------------
# baseline drift
# ---------------------------------------------------------------------------
def _baseline_drift(current: float | None, stat: dict | None) -> dict | None:
    if current is None or not isinstance(stat, dict):
        return None
    median_value = _finite_number(stat.get("median"))
    current_value = _finite_number(current)
    if median_value is None or current_value is None:
        return None
    mad = _finite_number(stat.get("mad"))
    mad = max(mad if mad is not None else 0.0, BASELINE_MAD_FLOOR)
    drift = round(current_value - median_value, 4)
    z_score = drift / mad if mad > 0 else 0.0
    return {
        "current": safe_number(current_value),
        "baseline_median": safe_number(median_value),
        "baseline_mad": safe_number(mad),
        "drift": safe_number(drift),
        "z_score": safe_number(z_score),
        "threshold": safe_number(-BASELINE_DRIFT_Z_THRESHOLD),
        "below_threshold": z_score <= -BASELINE_DRIFT_Z_THRESHOLD,
    }


def _drift_fatigue_grade(drift: dict | None) -> float:
    """Continuous 0..1 grade of how far below baseline (fatigue direction) a metric sits."""
    if not isinstance(drift, dict):
        return 0.0
    z = _finite_number(drift.get("z_score"))
    if z is None or z >= 0.0:
        return 0.0
    return min((-z) / BASELINE_DRIFT_Z_THRESHOLD, 1.0)


def _ok_details(signals: dict | None, modality: str) -> dict[str, Any]:
    """Raw analyzer details for a modality, but only when its status is "ok"; otherwise
    the metrics are stale and must be ignored."""
    signals = signals if isinstance(signals, dict) else {}
    analysis = signals.get(modality)
    if not _status_ok(analysis):
        return {}
    return _analysis_details(analysis)


def _raw_baseline_observations(signals: dict | None) -> dict[str, float | None]:
    camera_details = _ok_details(signals, "camera")
    voice_details = _ok_details(signals, "voice")
    return {
        "open_eye_aperture": safe_number(camera_details.get("avg_ear")),
        "left_right_eye_asymmetry": safe_number(camera_details.get("left_right_eye_asymmetry")),
        "normalized_voice_energy": safe_number(voice_details.get("rms_energy"), 6),
        # speech_rate is retired: never extracted, never used, always reported as None.
        "speech_rate": None,
    }


# ---------------------------------------------------------------------------
# capture-quality degradation of modality scores
# ---------------------------------------------------------------------------
def _degraded_signal_score(score: float | None, quality: float | None, warnings: list[str]) -> tuple[float | None, float]:
    if score is None:
        return None, 0.0
    # Warnings are already de-duplicated; each contributes its documented penalty once.
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


# ---------------------------------------------------------------------------
# biometric fatigue evidence (independent of capture-quality penalties)
# ---------------------------------------------------------------------------
def _confirmed_sustained_eye_closure(signals: dict | None) -> bool:
    """Confirm sustained eye closure only from complete, consistent video evidence.

    The analyzer booleans are necessary but not sufficient. Scoring validates the
    evidence package that produced them without reimplementing video.py thresholds.
    """
    details = _ok_details(signals, "video")
    if not details:
        return False
    if not _bool_flag(details.get("reliable_eye_landmarks")):
        return False
    if not _bool_flag(details.get("sustained_eye_closure")):
        return False

    motion_stability = _unit_interval(details.get("motion_stability_score"))
    sample_count = _strict_int(details.get("eye_closure_sample_count"), min_value=1)
    closed_eye_ratio = _unit_interval(details.get("closed_eye_ratio"))
    longest_streak = _strict_int(details.get("longest_eye_closure_streak"), min_value=1)
    window_ms = _finite_number(details.get("eye_closure_window_ms"))
    window_seconds = _finite_number(details.get("eye_closure_window_seconds"))
    avg_eye_aperture = _finite_number(details.get("avg_eye_aperture"))
    eye_aperture_std = _finite_number(details.get("eye_aperture_std"))

    if any(
        value is None
        for value in (
            motion_stability,
            sample_count,
            closed_eye_ratio,
            longest_streak,
            window_ms,
            window_seconds,
            avg_eye_aperture,
            eye_aperture_std,
        )
    ):
        return False
    if longest_streak > sample_count:
        return False
    if window_ms <= 0.0 or window_seconds <= 0.0:
        return False
    if avg_eye_aperture < 0.0 or eye_aperture_std < 0.0:
        return False
    if not math.isclose(window_ms / 1000.0, window_seconds, rel_tol=0.02, abs_tol=0.02):
        return False
    return True


def _video_fatigue_signal(
    *,
    signals: dict | None,
    baseline_flags: list[str] | None,
    eye_aperture_drift: dict | None,
) -> float:
    details = _ok_details(signals, "video")
    if not details:
        return 0.0
    if _confirmed_sustained_eye_closure(signals):
        return 1.0

    if not _bool_flag(details.get("reliable_eye_landmarks")):
        return 0.0

    closed_eye_ratio = clamp01(details.get("closed_eye_ratio"), 0.0) or 0.0
    avg_eye_aperture = _finite_number(details.get("avg_eye_aperture"))
    eye_aperture = 0.0
    if avg_eye_aperture is not None:
        eye_aperture = clamp01((EYE_APERTURE_FATIGUE_THRESHOLD - avg_eye_aperture) / EYE_APERTURE_FATIGUE_THRESHOLD, 0.0) or 0.0
    closure_ratio = clamp01((closed_eye_ratio - 0.12) / max(EYE_CLOSURE_HIGH_THRESHOLD - 0.12, 1e-6), 0.0) or 0.0

    closure_streak = _finite_number(details.get("longest_eye_closure_streak")) or 0.0
    closure_window_seconds = _finite_number(details.get("eye_closure_window_seconds")) or 0.0
    closure_streak_score = clamp01(closure_streak / 4.0, 0.0) or 0.0
    closure_window_score = clamp01(closure_window_seconds / 1.5, 0.0) or 0.0
    motion_stability = clamp01(details.get("motion_stability_score"), 1.0) or 1.0
    motion_penalty = max(0.0, 1.0 - motion_stability)

    fatigue = (
        0.30 * closed_eye_ratio
        + 0.18 * closure_ratio
        + 0.27 * eye_aperture
        + 0.16 * closure_streak_score
        + 0.10 * closure_window_score
        + 0.09 * motion_penalty
    )
    if closed_eye_ratio >= EYE_CLOSURE_SUSTAINED_THRESHOLD:
        fatigue += 0.14
    if (eye_aperture_drift or {}).get("below_threshold"):
        fatigue += 0.12
    if baseline_flags:
        fatigue += min(0.04 * len(baseline_flags), 0.12)
    return round(clamp01(fatigue, 0.0) or 0.0, 4)


def _audio_usable_speech(details: dict[str, Any], warnings: list[str]) -> bool:
    """Accept voice fatigue evidence only from an internally consistent speech state."""
    if not isinstance(details, dict):
        return False
    warning_codes = clean_warning_codes(warnings) if isinstance(warnings, list) else []
    if any(warning in _AUDIO_NON_FATIGUE_WARNINGS for warning in warning_codes):
        return False

    speech_state = details.get("speech_state")
    usable_speech_detected = details.get("usable_speech_detected")
    quiet_but_usable = details.get("quiet_but_usable")

    if usable_speech_detected is not True:
        return False
    if quiet_but_usable not in {None, False, True}:
        return False
    if speech_state == "usable_speech":
        return quiet_but_usable is not True
    if speech_state == "quiet_usable_speech":
        return quiet_but_usable is True
    return False


def _audio_fatigue_signal(
    *,
    signals: dict | None,
    baseline_flags: list[str] | None,
    voice_energy_drift: dict | None,
) -> float:
    details = _ok_details(signals, "voice")
    if not details:
        return 0.0
    warnings = _signal_warnings(signals.get("voice") if signals else None, "audio_warnings")
    if not _audio_usable_speech(details, warnings):
        # Noise, clipping, silence or absent speech are capture problems, not fatigue.
        return 0.0

    speech_presence_score = clamp01(details.get("speech_presence_score"), 0.0) or 0.0
    rms_energy = _finite_number(details.get("rms_energy"))
    if rms_energy is None:
        rms_energy = _finite_number(details.get("energy")) or 0.0
    silence_ratio = clamp01(details.get("silence_ratio"), 0.0) or 0.0
    normalized_energy = clamp01(rms_energy / max(VOICE_FATIGUE_ENERGY_THRESHOLD * 2.0, 1e-6), 0.0) or 0.0

    fatigue = (
        0.34 * (1.0 - speech_presence_score)
        + 0.28 * (1.0 - normalized_energy)
        + 0.24 * max(0.0, silence_ratio - (VOICE_FATIGUE_SILENCE_THRESHOLD - 0.1))
    )
    if _bool_flag(details.get("quiet_but_usable")):
        fatigue = max(fatigue, 0.36)
    if speech_presence_score < VOICE_FATIGUE_SPEECH_THRESHOLD:
        fatigue += 0.06
    if (voice_energy_drift or {}).get("below_threshold"):
        fatigue += 0.1
    if baseline_flags:
        fatigue += min(0.03 * len(baseline_flags), 0.09)
    return round(clamp01(fatigue, 0.0) or 0.0, 4)


def _fatigue_signal_context(
    *,
    signals: dict | None,
    baseline_used: bool,
    baseline_flags: list[str],
    eye_aperture_drift: dict | None,
    voice_energy_drift: dict | None,
) -> dict[str, Any]:
    video_fatigue = _video_fatigue_signal(
        signals=signals,
        baseline_flags=baseline_flags,
        eye_aperture_drift=eye_aperture_drift if baseline_used else None,
    )
    audio_fatigue = _audio_fatigue_signal(
        signals=signals,
        baseline_flags=baseline_flags,
        voice_energy_drift=voice_energy_drift if baseline_used else None,
    )
    combined = 0.68 * video_fatigue + 0.36 * audio_fatigue
    combined = round(clamp01(combined, 0.0) or 0.0, 4)
    return {
        "video": safe_number(video_fatigue),
        "audio": safe_number(audio_fatigue),
        "combined": safe_number(combined),
    }


# ---------------------------------------------------------------------------
# signal profiles & fusion
# ---------------------------------------------------------------------------
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
    decision_warning_count: int,
    baseline_used: bool,
    ml_result: dict | None,
) -> tuple[float, dict]:
    available = [name for name, profile in profiles.items() if profile["present"] and profile["score"] is not None and name != "task"]
    major_available = [name for name in available if name in {"video", "audio"}]
    agreement_bonus, conflict_penalty = _agreement_factor(profiles)
    quality_multiplier = _finite_number((quality or {}).get("confidence_multiplier"))
    if quality_multiplier is None or quality_multiplier < 0.0:
        quality_multiplier = 0.3
    quality_multiplier = min(quality_multiplier, 1.0)
    coverage = len(available) / 3.0
    missing_major_penalty = 0.18 if len(major_available) == 1 else 0.32 if len(major_available) == 0 else 0.0
    warning_penalty = min(decision_warning_count * 0.03, 0.18)
    ml_confidence = _unit_interval((ml_result or {}).get("confidence")) if isinstance(ml_result, dict) else None

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
    # Cap confidence below the reliable floor while preserving ordering: a noisier or
    # otherwise weaker scan keeps a strictly lower confidence than a cleaner one. The
    # cap never inflates an already-low confidence.
    ceiling = INVALID_SCAN_CONFIDENCE_CAP * (0.88 + 0.12 * (clamp01(confidence, 0.0) or 0.0))
    adjusted_confidence = round(min(confidence, ceiling), 4)
    if quality.get("failure_reason") == "missing_media":
        adjusted_score = min(adjusted_score, 35)
    elif quality.get("failure_reason") == "low_quality_media":
        adjusted_score = min(adjusted_score, 45)
    return adjusted_score, adjusted_confidence


def _quality_warning_set(quality: dict) -> set[str]:
    return set(clean_warning_codes(quality.get("warnings") or []))


def _quality_limited_scan(quality: dict, confidence: float) -> bool:
    return bool(
        quality.get("status") == "failed"
        or quality.get("retake_required")
        or quality.get("failure_reason") in {"low_quality_media", "missing_media"}
        or bool({"video", "audio"} & set(quality.get("missing_modalities") or []))
        or confidence < CONFIDENCE_RELIABLE_FLOOR
    )


def _risk_level(
    readiness_score: int,
    confidence: float,
    baseline_flags: list[str],
    quality: dict,
    *,
    fatigue_evidence: float = 0.0,
) -> str:
    # Sustained eye closure (surfaced into this quality view only when biometrically
    # confirmed) caps the risk at elevated fatigue; it is never rescued to stable and
    # never escalated to high_risk on the strength of the closure alone.
    sustained_eye_closure = "sustained_eye_closure" in _quality_warning_set(quality)
    if sustained_eye_closure:
        return "elevated_fatigue"
    if _quality_limited_scan(quality, confidence):
        return "low_focus"
    if fatigue_evidence >= FATIGUE_HIGH_THRESHOLD and confidence >= 0.5:
        return "high_risk"
    if readiness_score < 35 and confidence >= 0.62:
        return "high_risk"
    if fatigue_evidence >= FATIGUE_ELEVATED_THRESHOLD:
        return "elevated_fatigue"
    if quality.get("weak") and readiness_score < 52 and not baseline_flags:
        return "low_focus"
    if readiness_score < 52 and confidence >= 0.5:
        return "elevated_fatigue"
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
        return "review_required" if confidence >= CONFIDENCE_RELIABLE_FLOOR else "rescan_recommended"
    if confidence < CONFIDENCE_RELIABLE_FLOOR or quality.get("weak"):
        return "rescan_recommended"
    return "continue_normal_activity"


def _explanation(
    profiles: dict[str, dict],
    quality: dict,
    risk_level: str | None,
    confidence: float,
    *,
    confirmed_sustained_eye_closure: bool,
    fatigue_context: dict[str, Any] | None = None,
    baseline_notes: list[str] | None = None,
) -> str:
    positives: list[str] = []
    negatives: list[str] = []
    warning_reasons: list[str] = []
    baseline_notes = baseline_notes or []
    fatigue_context = fatigue_context or {}

    # Warnings that drive the narrative come from the live per-modality analyzer
    # warnings plus the quality warnings, de-duplicated. Unknown codes simply have no
    # matching narrative branch.
    warning_pool: list[str] = []
    for modality in ("video", "audio", "image"):
        warning_pool.extend(profiles.get(modality, {}).get("warnings") or [])
    warning_pool.extend(quality.get("warnings") or [])
    all_warnings = set(clean_warning_codes(warning_pool))

    if profiles["video"]["score"] is not None and (profiles["video"]["quality"] or 0.0) >= 0.55:
        positives.append("Video quality was acceptable")
    if profiles["audio"]["score"] is not None and (profiles["audio"]["quality"] or 0.0) >= 0.55:
        positives.append("voice signal was clear")
    if profiles["image"]["score"] is not None and (profiles["image"]["quality"] or 0.0) >= 0.55:
        positives.append("thumbnail quality was usable")
    if baseline_notes:
        positives.extend(baseline_notes)

    if "audio_too_noisy" in all_warnings:
        negatives.append("Audio was noisy")
        warning_reasons.append("background noise reduced voice confidence")
    if "audio_too_quiet" in all_warnings:
        negatives.append("audio volume was low")
        warning_reasons.append("low voice volume reduced confidence")
    if "speech_not_detected" in all_warnings:
        negatives.append("no usable speech was detected")
        warning_reasons.append("missing or unusable speech reduced confidence")
    if "too_much_silence" in all_warnings:
        negatives.append("audio was mostly silence")
        warning_reasons.append("insufficient voice activity reduced confidence")
    if "audio_clipping" in all_warnings:
        negatives.append("audio was clipped")
        warning_reasons.append("clipping reduced voice confidence")
    if "video_blurry" in all_warnings or "image_blurry" in all_warnings:
        negatives.append("visual media was partially blurred")
        warning_reasons.append("blur reduced visual confidence")
    if "video_too_dark" in all_warnings or "image_too_dark" in all_warnings:
        negatives.append("lighting was too dark")
        warning_reasons.append("low lighting reduced visual confidence")
    if "subject_not_visible" in all_warnings:
        negatives.append("subject visibility was limited")
        warning_reasons.append("limited subject visibility reduced confidence")
    if "insufficient_usable_frames" in all_warnings:
        negatives.append("too few usable video frames were available")
        warning_reasons.append("insufficient usable frames reduced confidence")
    if "landmark_detection_failed" in all_warnings:
        negatives.append("face landmarks were not reliably detected")
        warning_reasons.append("unreliable face detection reduced confidence")
    if "face_not_visible" in all_warnings or "unstable_video" in all_warnings:
        negatives.append("face visibility was weak")
        warning_reasons.append("weak face visibility reduced confidence")

    if confirmed_sustained_eye_closure:
        negatives.append("sustained eye closure was observed")
        warning_reasons.append("sustained eye closure lowered the readiness result")
    if (fatigue_context.get("video") or 0.0) >= 0.45:
        negatives.append("video showed fatigue-like eye behavior")
        warning_reasons.append("eye behavior suggested the person may be tired")
    if (fatigue_context.get("audio") or 0.0) >= 0.45:
        negatives.append("voice energy suggested fatigue")
        warning_reasons.append("voice pattern suggested the person may be tired")
    if (fatigue_context.get("combined") or 0.0) >= 0.55:
        warning_reasons.append("combined video and audio fatigue cues were elevated")

    if "low_quality_media" in all_warnings or quality.get("failure_reason") == "low_quality_media":
        warning_reasons.append("overall media quality was low")
    if "missing_media" in all_warnings or quality.get("failure_reason") == "missing_media":
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

    quality_limited = _quality_limited_scan(quality, confidence) and not confirmed_sustained_eye_closure
    fatigue_likely = (
        confirmed_sustained_eye_closure
        or (fatigue_context.get("combined") or 0.0) >= 0.55
        or (fatigue_context.get("audio") or 0.0) >= 0.45
        or (fatigue_context.get("video") or 0.0) >= 0.45
    )
    # A quality-limited or unreliable scan is never described as confirmed fatigue.
    if quality_limited and not confirmed_sustained_eye_closure:
        tail = "The result is weak because the scan quality was poor. Please retake it with clearer lighting, a steadier video, and a visible face."
    elif confirmed_sustained_eye_closure:
        tail = "The result shows sustained eye closure consistent with elevated fatigue and should be reviewed before continuing."
    elif fatigue_likely:
        tail = "The result suggests elevated fatigue signs and should be reviewed before continuing."
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
    signals = signals if isinstance(signals, dict) else {}
    quality = quality if isinstance(quality, dict) else {}
    baseline_used = _bool_flag(baseline_used)
    ml_result = ml_result if isinstance(ml_result, dict) else None

    task_score = compute_task_score(task)
    raw_features = _raw_baseline_observations(signals)
    profiles = _build_signal_profiles(signals, task_score, quality=quality)

    # Readability is decided by each analyzer's own status, never by the (duration-gated)
    # missing set the quality module may report. We build a decision view of quality that
    # reflects the modalities scoring can actually read, without mutating the input.
    status_missing = _missing_modalities_from_profiles(profiles)
    decision_quality = dict(quality)
    decision_quality["missing_modalities"] = status_missing

    weights = _adaptive_weights(profiles)

    fused_score = 0.0
    for name, weight in weights.items():
        score = profiles[name]["score"]
        if score is not None:
            fused_score += score * weight

    ml_confidence = _unit_interval(ml_result.get("confidence")) if ml_result is not None else None
    if ml_confidence is not None:
        fused_score = (fused_score * 0.84) + (ml_confidence * 0.16)

    quality_warnings = clean_warning_codes(decision_quality.get("warnings") or [])
    quality_penalty = min(len(quality_warnings) * 0.025, 0.16)
    if decision_quality.get("failure_reason") == "low_quality_media":
        quality_penalty = max(quality_penalty, 0.12)
    if decision_quality.get("failure_reason") == "missing_media":
        quality_penalty = max(quality_penalty, 0.20)
    if decision_quality.get("weak"):
        quality_penalty = max(quality_penalty, 0.06)
    fused_score = clamp01(fused_score - quality_penalty, 0.0) or 0.0

    image_score = profiles["image"]["score"]
    video_score = profiles["video"]["score"]
    if image_score is not None and video_score is not None:
        face_score = round((image_score * 0.4) + (video_score * 0.6), 4)
    else:
        face_score = image_score if image_score is not None else video_score

    # Baseline personalization only references the three schema-v2 personal features and
    # never mutates the baseline payload. speech_rate is retired.
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

    baseline_flags = [
        name
        for name, drift in {
            "open_eye_aperture": eye_aperture_drift,
            "left_right_eye_asymmetry": eye_asymmetry_drift,
            "normalized_voice_energy": voice_energy_drift,
        }.items()
        if drift and drift.get("below_threshold")
    ]

    if baseline_flags:
        fused_score = max(0.0, fused_score - (0.03 * len(baseline_flags)))

    confirmed_sustained_eye_closure = _confirmed_sustained_eye_closure(signals)
    if confirmed_sustained_eye_closure:
        fused_score = max(0.0, fused_score - 0.14)

    fatigue_context = _fatigue_signal_context(
        signals=signals,
        baseline_used=baseline_used,
        baseline_flags=baseline_flags,
        eye_aperture_drift=eye_aperture_drift,
        voice_energy_drift=voice_energy_drift,
    )
    if baseline_used:
        # Personal deviation strengthens fatigue continuously but is capped so that a
        # baseline deviation alone can never reach the high-risk threshold.
        baseline_fatigue_boost = (
            0.09 * _drift_fatigue_grade(eye_aperture_drift)
            + 0.07 * _drift_fatigue_grade(voice_energy_drift)
            + 0.05 * _drift_fatigue_grade(eye_asymmetry_drift)
        )
        baseline_fatigue_boost = min(baseline_fatigue_boost, 0.15)
        if baseline_fatigue_boost:
            fatigue_context["combined"] = safe_number(
                clamp01((fatigue_context.get("combined") or 0.0) + baseline_fatigue_boost, 0.0)
            )

    baseline_notes: list[str] = []
    if baseline_used:
        eye_aperture_reference = baseline_feature_reference(baseline, "face_avg", "open_eye_aperture")
        voice_energy_reference = baseline_feature_reference(baseline, "voice_avg", "normalized_voice_energy")
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

    # Confidence reflects the reliability of the fused measurement, not readiness/fatigue.
    decision_warning_count = len(
        set(clean_warning_codes(
            [w for modality in ("video", "audio", "image") for w in (profiles[modality]["warnings"] or [])]
            + list(decision_quality.get("warnings") or [])
        ))
    )
    confidence, calibration = _confidence_from_profiles(
        fused_score=fused_score,
        profiles=profiles,
        weights=weights,
        quality=decision_quality,
        decision_warning_count=decision_warning_count,
        baseline_used=baseline_used,
        ml_result=ml_result,
    )

    readiness_score = int(round((clamp01(fused_score, 0.0) or 0.0) * 100))
    fatigue_evidence = _finite_number(fatigue_context.get("combined")) or 0.0

    scan_unreliable = bool(
        decision_quality.get("status") == "failed"
        or decision_quality.get("retake_required")
        or decision_quality.get("failure_reason") in {"low_quality_media", "missing_media"}
        or bool({"video", "audio"} & set(status_missing))
        or confidence < CONFIDENCE_RELIABLE_FLOOR
        or (decision_quality.get("weak") and readiness_score < 52 and not baseline_flags)
    )
    # Confirmed sustained eye closure is biometric fatigue evidence, not a capture failure:
    # it lowers readiness but does NOT rescue an otherwise invalid scan.
    if confirmed_sustained_eye_closure:
        readiness_score = min(readiness_score, 30)
    if scan_unreliable:
        readiness_score, confidence = _invalid_scan_outcome(readiness_score, confidence, decision_quality)

    # Surface confirmed closure into the risk/explanation quality view so the documented
    # elevated-fatigue cap applies, without depending on a warning string alone.
    risk_quality = dict(decision_quality)
    if confirmed_sustained_eye_closure and not scan_unreliable:
        closure_warnings = list(risk_quality.get("warnings") or [])
        if "sustained_eye_closure" not in closure_warnings:
            closure_warnings = closure_warnings + ["sustained_eye_closure"]
        risk_quality["warnings"] = closure_warnings

    risk_level = _risk_level(
        readiness_score,
        confidence,
        baseline_flags,
        risk_quality,
        fatigue_evidence=fatigue_evidence,
    )
    if risk_level not in VALID_RISK_LEVELS:
        risk_level = "low_focus"

    suggested_action = _suggested_action(risk_level, confidence, decision_quality)
    if suggested_action not in VALID_ACTIONS:
        suggested_action = "rescan_recommended"

    explanation = _explanation(
        profiles,
        risk_quality,
        risk_level,
        confidence,
        confirmed_sustained_eye_closure=confirmed_sustained_eye_closure,
        fatigue_context=fatigue_context,
        baseline_notes=baseline_notes,
    )

    previous = _finite_number(previous_confidence)
    previous = clamp01(previous) if previous is not None else confidence
    confidence_drift = safe_number(confidence - (previous if previous is not None else confidence))

    modality_scores = {
        "video": safe_number(profiles["video"]["score"]),
        "audio": safe_number(profiles["audio"]["score"]),
        "image": safe_number(profiles["image"]["score"]),
        "task": safe_number(task_score),
    }
    # observed_fatigue_score is biometric only; capture penalties never inflate it.
    observed_fatigue_score = int(round((clamp01(fatigue_evidence, 0.0) or 0.0) * 100))
    # A genuine capture failure always requires a retake. Confirmed sustained eye closure
    # is biometric fatigue evidence, not a capture defect, so it may suppress a retake that
    # was triggered ONLY by the closure itself depressing confidence below the floor — but
    # it can never clear a capture-driven retake or invalid-capture state.
    capture_failure = bool(
        decision_quality.get("status") == "failed"
        or decision_quality.get("retake_required")
        or decision_quality.get("failure_reason") in {"low_quality_media", "missing_media"}
        or bool({"video", "audio"} & set(status_missing))
    )
    retake_required = bool(scan_unreliable or capture_failure)
    if confirmed_sustained_eye_closure and not capture_failure:
        retake_required = False
    failure_reason = decision_quality.get("failure_reason") if retake_required else None
    if failure_reason is not None and failure_reason not in {"low_quality_media", "missing_media"}:
        failure_reason = "low_quality_media"

    return {
        "status": "completed",
        "retake_required": retake_required,
        "failure_reason": failure_reason,
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
            "speech_rate": None,
            "baseline_drifts": {
                "normalized_voice_energy": voice_energy_drift,
                "speech_rate": None,
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