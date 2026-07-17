from __future__ import annotations

import math
import os
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from utils import clamp01, clean_warning_codes, safe_number, sanitize_text


FAILURE_MESSAGES = {
    "face_not_visible": "Please show your face clearly in the camera and try again.",
    "video_missing": "Video is missing. Please record the scan video again.",
    "video_too_dark": "The video is too dark. Please move to a brighter place and try again.",
    "video_blurry": "The video is blurry. Please keep the camera steady and try again.",
    "video_too_short": "The video is too short. Please complete the full scan recording.",
    "unstable_video": "Please keep your face visible and the camera steady during the scan.",
    "image_missing": "Image is missing. Please capture the image again.",
    "image_too_dark": "The image is too dark. Please move to a brighter place and try again.",
    "image_blurry": "The image is blurry. Please hold the camera steady and try again.",
    "audio_missing": "Audio is missing. Please record your voice again.",
    "audio_too_quiet": "Your voice is too quiet. Please speak clearly and try again.",
    "audio_too_noisy": "There is too much background noise. Please move to a quieter place and try again.",
    "speech_not_detected": "We could not detect your voice. Please speak clearly and try again.",
    "audio_too_short": "The voice recording is too short. Please complete the full sentence.",
    "phrase_mismatch": "Please read the exact sentence shown on the screen and try again.",
    "transcription_failed": "We could not understand the voice recording. Please speak clearly and try again.",
    "expected_phrase_missing": "The required voice phrase is missing. Please restart the scan.",
    "low_quality_media": "The scan quality is too low. Please try again with better lighting, a steady camera, and clear audio.",
    "missing_media": "Required scan media is missing. Please restart the scan.",
    "unreadable_media": "Scan media could not be read. Please restart the scan.",
    "analysis_exception": "Something went wrong while analyzing your scan. Please try again.",
    "model_not_loaded": "The AI model is not ready. Please try again later.",
    "directus_download_failed": "We could not load your scan media. Please try again.",
    "audio_validation_timeout": "Audio validation timed out. Please record the voice again.",
    "writeback_failed": "We could not save the scan result. Please try again.",
}


@dataclass(frozen=True)
class ValidationPolicy:
    require_video: bool = True
    require_audio: bool = True
    require_face: bool = True
    require_phrase_match: bool = False
    require_image: bool = True
    phrase_match_threshold: float = 0.80
    min_video_seconds: float = 3.0
    min_audio_seconds: float = 2.0
    min_face_visible_ratio: float = 0.50
    min_video_quality: float = 0.50
    min_audio_quality: float = 0.50
    min_image_quality: float = 0.50

    def __post_init__(self) -> None:
        _validate_policy_bool("require_video", self.require_video)
        _validate_policy_bool("require_audio", self.require_audio)
        _validate_policy_bool("require_face", self.require_face)
        _validate_policy_bool("require_phrase_match", self.require_phrase_match)
        _validate_policy_bool("require_image", self.require_image)
        _validate_policy_number("phrase_match_threshold", self.phrase_match_threshold, min_value=0.0, max_value=1.0)
        _validate_policy_number("min_video_seconds", self.min_video_seconds, min_value=0.0)
        _validate_policy_number("min_audio_seconds", self.min_audio_seconds, min_value=0.0)
        _validate_policy_number("min_face_visible_ratio", self.min_face_visible_ratio, min_value=0.0, max_value=1.0)
        _validate_policy_number("min_video_quality", self.min_video_quality, min_value=0.0, max_value=1.0)
        _validate_policy_number("min_audio_quality", self.min_audio_quality, min_value=0.0, max_value=1.0)
        _validate_policy_number("min_image_quality", self.min_image_quality, min_value=0.0, max_value=1.0)

    @classmethod
    def from_env(cls) -> "ValidationPolicy":
        return cls(
            require_video=_env_bool("REQUIRE_VIDEO", True),
            require_audio=_env_bool("REQUIRE_AUDIO", True),
            require_face=_env_bool("REQUIRE_FACE", True),
            require_phrase_match=_env_bool("REQUIRE_PHRASE_MATCH", False),
            require_image=_env_bool("REQUIRE_IMAGE", True),
            phrase_match_threshold=_env_float("PHRASE_MATCH_THRESHOLD", 0.80, min_value=0.0, max_value=1.0),
            min_video_seconds=_env_float("MIN_VIDEO_SECONDS", 3.0, min_value=0.0),
            min_audio_seconds=_env_float("MIN_AUDIO_SECONDS", 2.0, min_value=0.0),
            min_face_visible_ratio=_env_float("MIN_FACE_VISIBLE_RATIO", 0.50, min_value=0.0, max_value=1.0),
            min_video_quality=_env_float("MIN_VIDEO_QUALITY", 0.50, min_value=0.0, max_value=1.0),
            min_audio_quality=_env_float("MIN_AUDIO_QUALITY", 0.50, min_value=0.0, max_value=1.0),
            min_image_quality=_env_float("MIN_IMAGE_QUALITY", 0.50, min_value=0.0, max_value=1.0),
        )


_TRUE_ENV_VALUES = {"1", "true", "yes", "on"}
_FALSE_ENV_VALUES = {"0", "false", "no", "off"}
_EVIDENCE_ONLY_WARNINGS = {"sustained_eye_closure"}


def _validate_policy_bool(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _validate_policy_number(
    name: str,
    value: Any,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be a finite number")
    if min_value is not None and numeric < min_value:
        raise ValueError(f"{name} must be between {min_value} and {max_value}" if max_value is not None else f"{name} must be at least {min_value}")
    if max_value is not None and numeric > max_value:
        raise ValueError(f"{name} must be between {min_value} and {max_value}" if min_value is not None else f"{name} must be at most {max_value}")
    return numeric


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().casefold()
    if normalized in _TRUE_ENV_VALUES:
        return True
    if normalized in _FALSE_ENV_VALUES:
        return False
    raise ValueError(f"{name} must be one of: 0, 1, false, true, no, yes, off, on")


def _env_float(
    name: str,
    default: float,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    text = value.strip()
    if not text:
        raise ValueError(f"{name} must be a finite number")
    try:
        numeric = float(text)
    except ValueError as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    return _validate_policy_number(name, numeric, min_value=min_value, max_value=max_value)


def failure_message(reason: Any) -> str:
    if not isinstance(reason, str):
        return FAILURE_MESSAGES["analysis_exception"]
    try:
        return FAILURE_MESSAGES[reason]
    except (KeyError, TypeError):
        return FAILURE_MESSAGES["analysis_exception"]


def make_validation_result() -> dict[str, Any]:
    return {
        "passed": True,
        "failure_reason": None,
        "failure_message": None,
        "quality_scores": {
            "video": None,
            "audio": None,
            "image": None,
            "phrase_match": None,
        },
        "warnings": [],
        "critical_errors": [],
        "quality_penalties": [],
        "usable_modalities": [],
        "weak_modalities": [],
        "missing_modalities": [],
    }


def fail_validation(reason: str, *, scores: dict[str, Any] | None = None, warnings: list[str] | None = None) -> dict[str, Any]:
    result = make_validation_result()
    result["passed"] = False
    result["failure_reason"] = reason
    result["failure_message"] = failure_message(reason)
    result["critical_errors"] = [reason]
    if scores:
        result["quality_scores"].update(scores)
    result["warnings"] = clean_warning_codes(warnings)
    return result


QUALITY_WARNING_PENALTIES = {
    "video_too_short": 0.08,
    "video_too_dark": 0.15,
    "video_blurry": 0.15,
    "unstable_video": 0.12,
    "face_not_visible": 0.18,
    "audio_too_short": 0.08,
    "speech_not_detected": 0.18,
    "audio_too_noisy": 0.15,
    "audio_too_quiet": 0.15,
    "image_too_dark": 0.12,
    "image_blurry": 0.12,
    "low_quality_media": 0.15,
    "phrase_mismatch": 0.1,
    "transcription_failed": 0.1,
    "expected_phrase_missing": 0.08,
}


def _add_unique(target: list, value: Any) -> None:
    if value and value not in target:
        target.append(value)


def _remove_value(target: list, value: Any) -> None:
    target[:] = [item for item in target if item != value]


def _add_modality_state(result: dict[str, Any], modality: str, state: str) -> None:
    for key in ["usable_modalities", "weak_modalities", "missing_modalities"]:
        _remove_value(result[key], modality)
    if state == "usable":
        _add_unique(result["usable_modalities"], modality)
        return
    if state == "weak":
        _add_unique(result["weak_modalities"], modality)
        return
    if state == "missing":
        _add_unique(result["missing_modalities"], modality)


def _add_quality_penalties(result: dict[str, Any], modality: str, warnings: list[str]) -> None:
    for warning in clean_warning_codes(warnings):
        penalty = QUALITY_WARNING_PENALTIES.get(warning)
        if penalty is None:
            continue
        candidate = {"modality": modality, "reason": warning, "penalty": penalty}
        if candidate not in result["quality_penalties"]:
            result["quality_penalties"].append(candidate)


def _merge_validation_metadata(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key in ["critical_errors", "warnings", "usable_modalities", "weak_modalities", "missing_modalities"]:
        for value in source.get(key) or []:
            _add_unique(target[key], value)
    for penalty in source.get("quality_penalties") or []:
        if penalty not in target["quality_penalties"]:
            target["quality_penalties"].append(penalty)
    for modality in source.get("usable_modalities") or []:
        _add_modality_state(target, modality, "usable")
    for modality in source.get("weak_modalities") or []:
        _add_modality_state(target, modality, "weak")
    for modality in source.get("missing_modalities") or []:
        _add_modality_state(target, modality, "missing")


def _sanitize_state_lists(result: dict[str, Any]) -> None:
    for key in ["critical_errors", "warnings", "usable_modalities", "weak_modalities", "missing_modalities"]:
        result[key] = clean_warning_codes(result.get(key) or [])
    ordered_modalities: list[str] = []
    for key in ["usable_modalities", "weak_modalities", "missing_modalities"]:
        for modality in result[key]:
            if modality not in ordered_modalities:
                ordered_modalities.append(modality)
    for modality in ordered_modalities:
        if modality in result["missing_modalities"]:
            _add_modality_state(result, modality, "missing")
        elif modality in result["weak_modalities"]:
            _add_modality_state(result, modality, "weak")
        else:
            _add_modality_state(result, modality, "usable")


def _sanitize_quality_penalties(result: dict[str, Any]) -> None:
    raw_penalties = result.get("quality_penalties")
    if raw_penalties is None:
        items: list[Any] = []
    elif isinstance(raw_penalties, (list, tuple)):
        items = list(raw_penalties)
    else:
        items = [raw_penalties]

    sanitized: list[dict[str, Any]] = []
    seen: set[tuple[str, str, float]] = set()
    for penalty in items:
        if not isinstance(penalty, dict):
            continue
        modality = penalty.get("modality")
        reason = penalty.get("reason")
        value = penalty.get("penalty")
        if not isinstance(modality, str) or not modality.strip():
            continue
        if not isinstance(reason, str) or not reason.strip():
            continue
        numeric = _coerce_finite_number(value)
        if numeric is None or numeric < 0.0:
            continue
        key = (modality.strip(), reason.strip(), numeric)
        if key in seen:
            continue
        seen.add(key)
        sanitized.append({"modality": key[0], "reason": key[1], "penalty": numeric})
    result["quality_penalties"] = sanitized


def _analysis_status(details: dict[str, Any] | None) -> str | None:
    if not isinstance(details, dict):
        return None
    status = details.get("status")
    if not isinstance(status, str):
        return None
    normalized = status.strip().casefold()
    return normalized or None


def _analysis_has_minimum_evidence(details: dict[str, Any] | None, modality: str) -> bool:
    status = _analysis_status(details)
    if status != "ok":
        return False
    if not isinstance(details, dict):
        return False
    if modality == "video":
        quality = _coerce_finite_number(details.get("visual_quality_score"))
        duration = _coerce_finite_number(details.get("duration_seconds"))
        return quality is not None and duration is not None and duration >= 0.0
    if modality == "audio":
        quality = _coerce_finite_number(details.get("audio_quality_score"))
        duration = _coerce_finite_number(details.get("duration_seconds"))
        if duration is None:
            duration = _coerce_finite_number(details.get("duration_sec"))
        return quality is not None and duration is not None and duration >= 0.0
    if modality == "image":
        return _coerce_finite_number(details.get("image_quality_score")) is not None
    return False


def _finalize_validation_result(result: dict[str, Any]) -> dict[str, Any]:
    result["passed"] = bool(result.get("passed"))
    result["warnings"] = clean_warning_codes(result.get("warnings") or [])
    result["critical_errors"] = clean_warning_codes(result.get("critical_errors") or [])
    _sanitize_state_lists(result)
    _sanitize_quality_penalties(result)
    for key, value in (result.get("quality_scores") or {}).items():
        if value is None:
            continue
        result["quality_scores"][key] = safe_number(value)
    if result["passed"]:
        result["failure_reason"] = None
        result["failure_message"] = None
        result["critical_errors"] = []
    else:
        reason = result.get("failure_reason")
        if not isinstance(reason, str) or reason not in FAILURE_MESSAGES:
            reason = "analysis_exception"
        result["failure_reason"] = reason
        result["failure_message"] = failure_message(reason)
        if not result["critical_errors"]:
            result["critical_errors"] = [reason]
        else:
            result["critical_errors"] = clean_warning_codes(result["critical_errors"])
    return result


def _coerce_finite_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        return None
    return numeric


def _coerce_finite_int(value: Any) -> int | None:
    numeric = _coerce_finite_number(value)
    if numeric is None or not float(numeric).is_integer():
        return None
    return int(numeric)


def _analysis_details(result: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(result, dict):
        return None
    details = result.get("details")
    if not isinstance(details, dict) or not details:
        return None
    return details


def _infer_analysis_modality(details: dict[str, Any] | None) -> str | None:
    if not isinstance(details, dict):
        return None
    if any(key in details for key in ("audio_quality_score", "duration_sec", "speech_presence_score", "rms_energy", "noise_estimate", "usable_speech_detected", "audio_warnings")):
        return "audio"
    if any(key in details for key in ("visual_quality_score", "duration_seconds", "face_or_subject_visibility", "face_rate", "usable_frame_ratio", "motion_stability_score", "visual_warnings", "quality_flags", "sustained_eye_closure")):
        return "video"
    if any(key in details for key in ("image_quality_score", "image_warnings", "face_detected")):
        return "image"
    return None


def _analysis_is_unreadable(result: dict[str, Any] | None, modality: str | None = None) -> bool:
    details = _analysis_details(result)
    inferred_modality = modality or _infer_analysis_modality(details)
    if inferred_modality is None:
        return True
    return not _analysis_has_minimum_evidence(details, inferred_modality)


def _finalize_modality_result(
    result: dict[str, Any],
    *,
    modality: str,
    warnings: list[str],
    missing: bool = False,
    penalty_warnings: list[str] | None = None,
) -> dict[str, Any]:
    cleaned = clean_warning_codes(warnings)
    quality_warnings = clean_warning_codes(penalty_warnings if penalty_warnings is not None else cleaned)
    result["warnings"] = cleaned
    if missing:
        _add_modality_state(result, modality, "missing")
    elif quality_warnings:
        _add_modality_state(result, modality, "weak")
    else:
        _add_modality_state(result, modality, "usable")
    _add_quality_penalties(result, modality, quality_warnings)
    return result


def normalize_phrase(value: str | None) -> str:
    text = sanitize_text(value, fallback="") or ""
    text = unicodedata.normalize("NFKC", text).casefold().strip()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def phrase_match_score(expected_phrase: str | None, transcript: str | None) -> float:
    expected = normalize_phrase(expected_phrase)
    actual = normalize_phrase(transcript)
    if not expected or not actual:
        return 0.0
    if expected == actual:
        return 1.0

    sequence = SequenceMatcher(None, expected, actual).ratio()
    expected_tokens = expected.split()
    actual_tokens = actual.split()
    expected_counts = Counter(expected_tokens)
    actual_counts = Counter(actual_tokens)
    matched_tokens = sum(min(expected_counts[token], actual_counts[token]) for token in expected_counts)
    precision = matched_tokens / max(len(actual_tokens), 1)
    recall = matched_tokens / max(len(expected_tokens), 1)
    token_score = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    length_balance = min(len(actual_tokens), len(expected_tokens)) / max(len(actual_tokens), len(expected_tokens), 1)
    score = (0.55 * sequence) + (0.35 * token_score) + (0.10 * length_balance)
    return round(max(0.0, min(score, 1.0)), 4)


def validate_scan_inputs(
    *,
    policy: ValidationPolicy,
    media: Any,
    video_result: dict[str, Any] | None,
    audio_result: dict[str, Any] | None,
    image_result: dict[str, Any] | None,
    expected_phrase: str | None,
    transcript: str | None,
) -> dict[str, Any]:
    result = make_validation_result()
    warnings: list[str] = []
    phrase_validation: dict[str, Any] | None = None

    video_path = _media_value(media, "video")
    audio_path = _media_value(media, "audio")
    image_path = _media_value(media, "image")

    if policy.require_video and not video_path:
        warnings.append("video_missing")
        _add_modality_state(result, "video", "missing")
    if policy.require_audio and not audio_path:
        warnings.append("audio_missing")
        _add_modality_state(result, "audio", "missing")
    if policy.require_image and not image_path:
        warnings.append("image_missing")
        _add_modality_state(result, "image", "missing")
    if not video_path and not audio_path and not image_path:
        warnings.append("missing_media")

    video_details = _analysis_details(video_result) or {}
    audio_details = _analysis_details(audio_result) or {}
    image_details = _analysis_details(image_result) or {}

    result["quality_scores"]["video"] = safe_number(video_details.get("visual_quality_score"))
    result["quality_scores"]["audio"] = safe_number(audio_details.get("audio_quality_score"))
    result["quality_scores"]["image"] = safe_number(image_details.get("image_quality_score"))

    if video_path:
        video_validation = validate_video_result(policy, video_result)
        warnings.extend(video_validation["warnings"])
        _merge_quality_scores(result["quality_scores"], video_validation["quality_scores"])
        _merge_validation_metadata(result, video_validation)

    if audio_path:
        speech_required = policy.require_phrase_match or bool(expected_phrase)
        audio_validation = validate_audio_result(policy, audio_result, speech_required=speech_required)
        warnings.extend(audio_validation["warnings"])
        _merge_quality_scores(result["quality_scores"], audio_validation["quality_scores"])
        _merge_validation_metadata(result, audio_validation)

    if image_path:
        image_validation = validate_image_result(policy, image_result, image_required=policy.require_image)
        warnings.extend(image_validation["warnings"])
        _merge_quality_scores(result["quality_scores"], image_validation["quality_scores"])
        _merge_validation_metadata(result, image_validation)

    if policy.require_phrase_match or expected_phrase:
        phrase_validation = validate_phrase_result(policy, expected_phrase, transcript)
        warnings.extend(phrase_validation["warnings"])
        _merge_quality_scores(result["quality_scores"], phrase_validation["quality_scores"])
        _merge_validation_metadata(result, phrase_validation)

    result["warnings"] = clean_warning_codes(warnings)
    required_missing: list[str] = []
    if policy.require_video and not video_path:
        required_missing.append("video_missing")
    if policy.require_audio and not audio_path:
        required_missing.append("audio_missing")
    if policy.require_image and not image_path:
        required_missing.append("image_missing")
    required_unreadable = []
    if video_path and policy.require_video and _analysis_is_unreadable(video_result, "video"):
        required_unreadable.append("video")
    if audio_path and policy.require_audio and _analysis_is_unreadable(audio_result, "audio"):
        required_unreadable.append("audio")
    if image_path and policy.require_image and _analysis_is_unreadable(image_result, "image"):
        required_unreadable.append("image")

    if not video_path and not audio_path and not image_path:
        result["passed"] = False
        result["failure_reason"] = "missing_media"
        result["failure_message"] = failure_message("missing_media")
        result["critical_errors"] = ["missing_media"]
    elif required_missing:
        result["passed"] = False
        result["failure_reason"] = required_missing[0]
        result["failure_message"] = failure_message(required_missing[0])
        result["critical_errors"] = [required_missing[0]]
    elif required_unreadable or _all_present_media_unreadable(
        media={"video": video_path, "audio": audio_path, "image": image_path},
        results={"video": video_result, "audio": audio_result, "image": image_result},
    ):
        result["passed"] = False
        result["failure_reason"] = "unreadable_media"
        result["failure_message"] = failure_message("unreadable_media")
        result["critical_errors"] = ["unreadable_media"]
    elif phrase_validation and not phrase_validation.get("passed", True) and policy.require_phrase_match:
        failure_reason = phrase_validation.get("failure_reason") or "analysis_exception"
        result["passed"] = False
        result["failure_reason"] = failure_reason
        result["failure_message"] = failure_message(failure_reason)
        result["critical_errors"] = clean_warning_codes(phrase_validation.get("critical_errors") or [failure_reason])
    return _finalize_validation_result(result)


def validate_video_result(policy: ValidationPolicy, video_result: dict[str, Any] | None) -> dict[str, Any]:
    result = make_validation_result()
    details = _analysis_details(video_result)
    if details is None:
        return _finalize_modality_result(
            result,
            modality="video",
            warnings=["video_missing"],
            missing=True,
        )
    warnings = clean_warning_codes(details.get("visual_warnings") or details.get("quality_flags") or [])
    result["quality_scores"]["video"] = safe_number(details.get("visual_quality_score"))
    if not _analysis_has_minimum_evidence(details, "video"):
        return _finalize_modality_result(result, modality="video", warnings=warnings + ["video_missing"], missing=True)
    status = _analysis_status(details)
    if status != "ok":
        return _finalize_modality_result(
            result,
            modality="video",
            warnings=warnings + ["video_missing"],
            missing=True,
        )
    duration = _coerce_finite_number(details.get("duration_seconds"))
    if duration is not None and duration < policy.min_video_seconds:
        warnings.append("video_too_short")
    brightness_score = _coerce_finite_number(details.get("brightness_score"))
    if "video_too_dark" in warnings or (brightness_score is not None and brightness_score < 0.35):
        warnings.append("video_too_dark")
    sharpness_score = _coerce_finite_number(details.get("sharpness_score"))
    if "video_blurry" in warnings or (sharpness_score is not None and sharpness_score < 0.35):
        warnings.append("video_blurry")

    face_ratio = _coerce_finite_number(details.get("face_or_subject_visibility"))
    if face_ratio is None:
        face_ratio = _coerce_finite_number(details.get("face_rate"))
    face_frames = _coerce_finite_int(details.get("face_frames"))
    if policy.require_face and (face_frames is None or face_frames <= 0):
        warnings.append("face_not_visible")
    if policy.require_face and face_ratio is not None and face_ratio < policy.min_face_visible_ratio:
        warnings.append("unstable_video")

    usable_ratio = _coerce_finite_number(details.get("usable_frame_ratio"))
    motion_stability = _coerce_finite_number(details.get("motion_stability_score"))
    if (usable_ratio is not None and usable_ratio < 0.3) or (motion_stability is not None and motion_stability < 0.35) or "unstable_camera" in warnings:
        warnings.append("unstable_video")
    if details.get("sustained_eye_closure"):
        warnings.append("sustained_eye_closure")

    quality = _coerce_finite_number(details.get("visual_quality_score"))
    if quality is not None and quality < policy.min_video_quality:
        warnings.append("low_quality_media")

    quality_warnings = [warning for warning in warnings if warning not in _EVIDENCE_ONLY_WARNINGS]
    return _finalize_modality_result(result, modality="video", warnings=warnings, penalty_warnings=quality_warnings)


def validate_audio_result(
    policy: ValidationPolicy,
    audio_result: dict[str, Any] | None,
    *,
    speech_required: bool = False,
) -> dict[str, Any]:
    result = make_validation_result()
    details = _analysis_details(audio_result)
    if details is None:
        return _finalize_modality_result(
            result,
            modality="audio",
            warnings=["audio_missing"],
            missing=True,
        )
    warnings = clean_warning_codes(details.get("audio_warnings") or [])
    result["quality_scores"]["audio"] = safe_number(details.get("audio_quality_score"))
    if not _analysis_has_minimum_evidence(details, "audio"):
        return _finalize_modality_result(result, modality="audio", warnings=warnings + ["audio_missing"], missing=True)
    status = _analysis_status(details)
    if status != "ok":
        return _finalize_modality_result(
            result,
            modality="audio",
            warnings=warnings + ["audio_missing"],
            missing=True,
        )
    duration = _coerce_finite_number(details.get("duration_seconds"))
    if duration is None:
        duration = _coerce_finite_number(details.get("duration_sec"))
    quiet_but_usable = bool(details.get("quiet_but_usable"))
    if duration is not None and duration < policy.min_audio_seconds:
        warnings.append("audio_too_short")
    usable_speech_detected = details.get("usable_speech_detected")
    speech_state_raw = details.get("speech_state")
    speech_presence_score = _coerce_finite_number(details.get("speech_presence_score"))
    speech_detected = False
    if isinstance(speech_state_raw, str) and speech_state_raw.strip().casefold() == "no_speech":
        speech_detected = True
    elif usable_speech_detected is False:
        speech_detected = True
    elif speech_presence_score is not None and speech_presence_score < 0.35:
        speech_detected = True
    if speech_required:
        if "speech_not_detected" in warnings or speech_detected:
            warnings.append("speech_not_detected")
    else:
        warnings = [warning for warning in warnings if warning != "speech_not_detected"]
    noise_estimate = _coerce_finite_number(details.get("noise_estimate"))
    if "audio_too_noisy" in warnings or (noise_estimate is not None and noise_estimate > 0.72):
        warnings.append("audio_too_noisy")
    rms_energy = _coerce_finite_number(details.get("rms_energy"))
    if rms_energy is None:
        rms_energy = _coerce_finite_number(details.get("energy"))
    if not quiet_but_usable and ("audio_too_quiet" in warnings or (rms_energy is not None and rms_energy < 0.012)):
        warnings.append("audio_too_quiet")
    clipping_ratio = _coerce_finite_number(details.get("clipping_ratio"))
    if "audio_clipping" in warnings or (clipping_ratio is not None and clipping_ratio > 0.015):
        warnings.append("audio_clipping")
    silence_ratio = _coerce_finite_number(details.get("silence_ratio"))
    if "too_much_silence" in warnings or (silence_ratio is not None and silence_ratio > 0.55):
        warnings.append("too_much_silence")

    quality = _coerce_finite_number(details.get("audio_quality_score"))
    if quality is not None and quality < policy.min_audio_quality and not quiet_but_usable:
        warnings.append("low_quality_media")

    return _finalize_modality_result(result, modality="audio", warnings=warnings)


def validate_image_result(
    policy: ValidationPolicy,
    image_result: dict[str, Any] | None,
    *,
    image_required: bool,
) -> dict[str, Any]:
    result = make_validation_result()
    details = _analysis_details(image_result)
    if details is None:
        return _finalize_modality_result(
            result,
            modality="image",
            warnings=["image_missing"],
            missing=True,
        )
    warnings = clean_warning_codes(details.get("image_warnings") or [])
    result["quality_scores"]["image"] = safe_number(details.get("image_quality_score"))
    if not _analysis_has_minimum_evidence(details, "image"):
        return _finalize_modality_result(
            result,
            modality="image",
            warnings=warnings + ["image_missing"],
            missing=True,
        )
    status = _analysis_status(details)
    if status != "ok":
        return _finalize_modality_result(
            result,
            modality="image",
            warnings=warnings + ["image_missing"],
            missing=True,
        )

    brightness_score = _coerce_finite_number(details.get("brightness_score"))
    if "image_too_dark" in warnings or (brightness_score is not None and brightness_score < 0.35):
        warnings.append("image_too_dark")
    sharpness_score = _coerce_finite_number(details.get("sharpness_score"))
    if "image_blurry" in warnings or (sharpness_score is not None and sharpness_score < 0.35):
        warnings.append("image_blurry")
    if policy.require_face and not details.get("face_detected", False):
        warnings.append("face_not_visible")

    quality = _coerce_finite_number(details.get("image_quality_score"))
    if image_required and quality is not None and quality < policy.min_image_quality:
        warnings.append("low_quality_media")

    return _finalize_modality_result(result, modality="image", warnings=warnings)


def validate_phrase_result(policy: ValidationPolicy, expected_phrase: str | None, transcript: str | None) -> dict[str, Any]:
    expected = normalize_phrase(expected_phrase)
    quality_scores = {"phrase_match": None}
    if not expected:
        result = make_validation_result()
        result["quality_scores"].update(quality_scores)
        if policy.require_phrase_match:
            return fail_validation("expected_phrase_missing", warnings=["expected_phrase_missing"])
        return _finalize_validation_result(result)

    normalized_transcript = normalize_phrase(transcript)
    if not normalized_transcript:
        if policy.require_phrase_match:
            result = fail_validation("transcription_failed", warnings=["transcription_failed"])
            result["quality_scores"].update(quality_scores)
            _add_quality_penalties(result, "phrase", result["warnings"])
            return _finalize_validation_result(result)
        result = make_validation_result()
        result["quality_scores"].update(quality_scores)
        result["warnings"] = ["transcription_failed"]
        _add_quality_penalties(result, "phrase", result["warnings"])
        return _finalize_validation_result(result)

    score = phrase_match_score(expected_phrase, transcript)
    quality_scores["phrase_match"] = safe_number(score)
    result = make_validation_result()
    result["quality_scores"].update(quality_scores)
    if score < policy.phrase_match_threshold:
        if policy.require_phrase_match:
            result = fail_validation("phrase_mismatch", scores=quality_scores, warnings=["phrase_mismatch"])
            _add_quality_penalties(result, "phrase", result["warnings"])
            return _finalize_validation_result(result)
        result["warnings"] = ["phrase_mismatch"]
        _add_quality_penalties(result, "phrase", result["warnings"])
        return _finalize_validation_result(result)

    return _finalize_validation_result(result)


def _media_value(media: Any, key: str) -> Any:
    if media is None:
        return None
    if isinstance(media, dict):
        return media.get(key)
    return getattr(media, key, None)


def _merge_quality_scores(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in (source or {}).items():
        if value is not None:
            target[key] = value


def _all_present_media_unreadable(*, media: dict[str, Any], results: dict[str, Any]) -> bool:
    present_modalities = [name for name, value in media.items() if value]
    if not present_modalities:
        return False
    for name in present_modalities:
        if not _analysis_is_unreadable(results.get(name), name):
            return False
    return True
