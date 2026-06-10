from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import os
import re
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

    @classmethod
    def from_env(cls) -> "ValidationPolicy":
        return cls(
            require_video=_env_bool("REQUIRE_VIDEO", True),
            require_audio=_env_bool("REQUIRE_AUDIO", True),
            require_face=_env_bool("REQUIRE_FACE", True),
            require_phrase_match=_env_bool("REQUIRE_PHRASE_MATCH", False),
            require_image=_env_bool("REQUIRE_IMAGE", True),
            phrase_match_threshold=_env_float("PHRASE_MATCH_THRESHOLD", 0.80),
            min_video_seconds=_env_float("MIN_VIDEO_SECONDS", 3.0),
            min_audio_seconds=_env_float("MIN_AUDIO_SECONDS", 2.0),
            min_face_visible_ratio=_env_float("MIN_FACE_VISIBLE_RATIO", 0.50),
            min_video_quality=_env_float("MIN_VIDEO_QUALITY", 0.50),
            min_audio_quality=_env_float("MIN_AUDIO_QUALITY", 0.50),
            min_image_quality=_env_float("MIN_IMAGE_QUALITY", 0.50),
        )


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def failure_message(reason: str) -> str:
    return FAILURE_MESSAGES.get(reason, FAILURE_MESSAGES["analysis_exception"])


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


def _add_modality_state(result: dict[str, Any], modality: str, state: str) -> None:
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


def _finalize_modality_result(
    result: dict[str, Any],
    *,
    modality: str,
    warnings: list[str],
    missing: bool = False,
) -> dict[str, Any]:
    cleaned = clean_warning_codes(warnings)
    result["warnings"] = cleaned
    if missing:
        _add_modality_state(result, modality, "missing")
    elif cleaned:
        _add_modality_state(result, modality, "weak")
    else:
        _add_modality_state(result, modality, "usable")
    _add_quality_penalties(result, modality, cleaned)
    return result


def normalize_phrase(value: str | None) -> str:
    text = sanitize_text(value, fallback="") or ""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
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
    overlap = len(set(expected_tokens) & set(actual_tokens))
    token_ratio = overlap / max(len(set(expected_tokens)), 1)
    length_penalty = min(len(actual_tokens), len(expected_tokens)) / max(len(actual_tokens), len(expected_tokens), 1)
    return round(max(sequence, (0.65 * sequence) + (0.25 * token_ratio) + (0.1 * length_penalty)), 4)


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

    video_details = ((video_result or {}).get("details") or {}) if video_result else {}
    audio_details = ((audio_result or {}).get("details") or {}) if audio_result else {}
    image_details = ((image_result or {}).get("details") or {}) if image_result else {}

    result["quality_scores"]["video"] = safe_number(video_details.get("visual_quality_score"))
    result["quality_scores"]["audio"] = safe_number(audio_details.get("audio_quality_score"))
    result["quality_scores"]["image"] = safe_number(image_details.get("image_quality_score"))

    if video_path:
        video_validation = validate_video_result(policy, video_result)
        warnings.extend(video_validation["warnings"])
        _merge_quality_scores(result["quality_scores"], video_validation["quality_scores"])
        _merge_validation_metadata(result, video_validation)

    if audio_path:
        audio_validation = validate_audio_result(policy, audio_result)
        warnings.extend(audio_validation["warnings"])
        _merge_quality_scores(result["quality_scores"], audio_validation["quality_scores"])
        _merge_validation_metadata(result, audio_validation)

    if policy.require_phrase_match or expected_phrase:
        phrase_validation = validate_phrase_result(policy, expected_phrase, transcript)
        warnings.extend(phrase_validation["warnings"])
        _merge_quality_scores(result["quality_scores"], phrase_validation["quality_scores"])
        _merge_validation_metadata(result, phrase_validation)

    if image_path:
        image_validation = validate_image_result(policy, image_result, image_required=policy.require_image)
        warnings.extend(image_validation["warnings"])
        _merge_quality_scores(result["quality_scores"], image_validation["quality_scores"])
        _merge_validation_metadata(result, image_validation)

    result["warnings"] = clean_warning_codes(warnings)
    if not video_path and not audio_path and not image_path:
        result["passed"] = False
        result["failure_reason"] = "missing_media"
        result["failure_message"] = failure_message("missing_media")
        result["critical_errors"] = ["missing_media"]
    elif _all_present_media_unreadable(
        media={"video": video_path, "audio": audio_path, "image": image_path},
        results={"video": video_result, "audio": audio_result, "image": image_result},
    ):
        result["passed"] = False
        result["failure_reason"] = "unreadable_media"
        result["failure_message"] = failure_message("unreadable_media")
        result["critical_errors"] = ["unreadable_media"]
    return result


def validate_video_result(policy: ValidationPolicy, video_result: dict[str, Any] | None) -> dict[str, Any]:
    details = ((video_result or {}).get("details") or {}) if video_result else {}
    warnings = clean_warning_codes(details.get("visual_warnings") or details.get("quality_flags") or [])
    quality_scores = {"video": safe_number(details.get("visual_quality_score"))}
    status = details.get("status")

    result = make_validation_result()
    result["quality_scores"].update(quality_scores)
    if status in {"missing", "open_failed"}:
        return _finalize_modality_result(
            result,
            modality="video",
            warnings=warnings + ["video_missing"],
            missing=True,
        )
    duration = float(details.get("duration_seconds") or 0.0)
    if duration < policy.min_video_seconds:
        warnings.append("video_too_short")
    if "video_too_dark" in warnings or float(details.get("brightness_score") or 0.0) < 0.35:
        warnings.append("video_too_dark")
    if "video_blurry" in warnings or float(details.get("sharpness_score") or 0.0) < 0.35:
        warnings.append("video_blurry")

    face_ratio = float(details.get("face_or_subject_visibility") or details.get("face_rate") or 0.0)
    face_frames = int(details.get("face_frames") or 0)
    if policy.require_face and face_frames <= 0:
        warnings.append("face_not_visible")
    if policy.require_face and face_ratio < policy.min_face_visible_ratio:
        warnings.append("unstable_video")

    usable_ratio = float(details.get("usable_frame_ratio") or 0.0)
    motion_stability = float(details.get("motion_stability_score") or 0.0)
    if usable_ratio < 0.3 or motion_stability < 0.35 or "unstable_camera" in warnings:
        warnings.append("unstable_video")

    quality = float(details.get("visual_quality_score") or 0.0)
    if quality < policy.min_video_quality:
        warnings.append("low_quality_media")

    return _finalize_modality_result(result, modality="video", warnings=warnings)


def validate_audio_result(policy: ValidationPolicy, audio_result: dict[str, Any] | None) -> dict[str, Any]:
    details = ((audio_result or {}).get("details") or {}) if audio_result else {}
    warnings = clean_warning_codes(details.get("audio_warnings") or [])
    quality_scores = {"audio": safe_number(details.get("audio_quality_score"))}
    status = details.get("status")

    result = make_validation_result()
    result["quality_scores"].update(quality_scores)
    if status in {"missing", "load_failed", "empty_audio"}:
        return _finalize_modality_result(
            result,
            modality="audio",
            warnings=warnings + ["audio_missing"],
            missing=True,
        )
    duration = float(details.get("duration_seconds") or details.get("duration_sec") or 0.0)
    if duration < policy.min_audio_seconds:
        warnings.append("audio_too_short")
    if "speech_not_detected" in warnings or float(details.get("speech_presence_score") or 0.0) < 0.35:
        warnings.append("speech_not_detected")
    if "audio_too_noisy" in warnings or float(details.get("noise_estimate") or 0.0) > 0.72:
        warnings.append("audio_too_noisy")
    if "audio_too_quiet" in warnings or float(details.get("rms_energy") or details.get("energy") or 0.0) < 0.012:
        warnings.append("audio_too_quiet")

    quality = float(details.get("audio_quality_score") or 0.0)
    if quality < policy.min_audio_quality:
        warnings.append("low_quality_media")

    return _finalize_modality_result(result, modality="audio", warnings=warnings)


def validate_image_result(
    policy: ValidationPolicy,
    image_result: dict[str, Any] | None,
    *,
    image_required: bool,
) -> dict[str, Any]:
    details = ((image_result or {}).get("details") or {}) if image_result else {}
    warnings = clean_warning_codes(details.get("image_warnings") or [])
    quality_scores = {"image": safe_number(details.get("image_quality_score"))}
    status = details.get("status")

    result = make_validation_result()
    result["quality_scores"].update(quality_scores)
    if status in {"missing", "invalid_image"}:
        if image_required:
            warnings.append("image_missing")
        return _finalize_modality_result(
            result,
            modality="image",
            warnings=warnings,
            missing=True,
        )

    if "image_too_dark" in warnings or float(details.get("brightness_score") or 0.0) < 0.35:
        warnings.append("image_too_dark")
    if "image_blurry" in warnings or float(details.get("sharpness_score") or 0.0) < 0.35:
        warnings.append("image_blurry")
    if policy.require_face and not details.get("face_detected", False):
        warnings.append("face_not_visible")

    quality = float(details.get("image_quality_score") or 0.0)
    if image_required and quality < policy.min_image_quality:
        warnings.append("low_quality_media")

    return _finalize_modality_result(result, modality="image", warnings=warnings)


def validate_phrase_result(policy: ValidationPolicy, expected_phrase: str | None, transcript: str | None) -> dict[str, Any]:
    expected = normalize_phrase(expected_phrase)
    quality_scores = {"phrase_match": None}
    if not expected:
        result = make_validation_result()
        result["quality_scores"].update(quality_scores)
        if policy.require_phrase_match:
            result["warnings"] = ["expected_phrase_missing"]
            _add_quality_penalties(result, "phrase", result["warnings"])
        return result

    normalized_transcript = normalize_phrase(transcript)
    if not normalized_transcript:
        result = make_validation_result()
        result["quality_scores"].update(quality_scores)
        result["warnings"] = ["transcription_failed"]
        _add_quality_penalties(result, "phrase", result["warnings"])
        return result

    score = phrase_match_score(expected_phrase, transcript)
    quality_scores["phrase_match"] = safe_number(score)
    result = make_validation_result()
    result["quality_scores"].update(quality_scores)
    if score < policy.phrase_match_threshold:
        result["warnings"] = ["phrase_mismatch"]
        _add_quality_penalties(result, "phrase", result["warnings"])
        return result

    return result


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
    unreadable_statuses = {
        "missing",
        "open_failed",
        "load_failed",
        "empty_audio",
        "invalid",
        "invalid_image",
        "error",
    }
    for name in present_modalities:
        details = ((results.get(name) or {}).get("details") or {}) if isinstance(results.get(name), dict) else {}
        if details.get("status") not in unreadable_statuses:
            return False
    return True
