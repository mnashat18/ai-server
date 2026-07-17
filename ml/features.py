from __future__ import annotations

import math
from typing import Any


# IMPORTANT:
# Keep this order stable for compatibility with existing trained model artifacts.
# Any addition, removal, reordering, or semantic change requires coordinated
# model retraining/versioning and updates to ml/model.py, ml/runtime.py, and tests.
FEATURE_ORDER = [
    "camera_score",
    "face_detected",
    "avg_ear",
    "eyes_closed",
    "audio_score",
    "audio_energy",
    "audio_zcr",
    "audio_centroid",
    "audio_duration",
    "audio_silent",
    "video_score",
    "video_sway_std",
    "video_face_rate",
    "video_face_frames",
    "video_sampled_frames",
    "task_reaction_time",
    "task_errors",
    "task_present",
    "missing_camera",
    "missing_audio",
    "missing_video",
]


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _finite_number(
    value: Any,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float | None:
    """Accept only real finite int/float values.

    bool, numeric strings, arbitrary objects, NaN, and infinities are rejected.
    Values outside an optional range are rejected rather than clamped.
    """
    if isinstance(value, bool) or type(value) not in {int, float}:
        return None

    numeric = float(value)
    if not math.isfinite(numeric):
        return None
    if min_value is not None and numeric < min_value:
        return None
    if max_value is not None and numeric > max_value:
        return None
    return numeric


def _strict_non_negative_int(value: Any) -> int | None:
    if type(value) is not int or value < 0:
        return None
    return value


def _strict_positive_int(value: Any) -> int | None:
    if type(value) is not int or value <= 0:
        return None
    return value


def _bool(value: Any) -> float:
    """Encode only an actual boolean True as 1.0."""
    return 1.0 if value is True else 0.0


def _normalized_status(details: dict[str, Any]) -> str | None:
    status = details.get("status")
    if not isinstance(status, str):
        return None
    normalized = status.strip().casefold()
    return normalized or None


def _readable_details(analysis: Any) -> dict[str, Any]:
    """Return details only when the analyzer explicitly completed with status='ok'."""
    if not isinstance(analysis, dict):
        return {}

    details = analysis.get("details")
    if not isinstance(details, dict):
        return {}
    if _normalized_status(details) != "ok":
        return {}
    return details


def _signal_score(analysis: Any, dedicated_key: str) -> float | None:
    """Read a finite normalized analyzer confidence.

    The dedicated analyzer field is authoritative. The top-level score is retained
    only as a compatibility fallback for an otherwise readable legacy payload where
    the dedicated field is absent. A malformed dedicated value is not rescued by the
    top-level score.
    """
    if not isinstance(analysis, dict):
        return None

    details = _readable_details(analysis)
    if not details:
        return None

    if dedicated_key in details:
        return _finite_number(details.get(dedicated_key), min_value=0.0, max_value=1.0)
    return _finite_number(analysis.get("score"), min_value=0.0, max_value=1.0)


def _get_detail(
    details: Any,
    key: str,
    default: float = 0.0,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float:
    """Read a strict finite detail value or return a safe finite default."""
    safe_default = _finite_number(default)
    if safe_default is None:
        safe_default = 0.0

    if not isinstance(details, dict):
        return safe_default

    numeric = _finite_number(
        details.get(key),
        min_value=min_value,
        max_value=max_value,
    )
    return numeric if numeric is not None else safe_default


def _get_first_valid_detail(
    details: Any,
    keys: tuple[str, ...],
    default: float = 0.0,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float:
    """Use the first present field as authoritative.

    Legacy aliases are used only when the preferred field is absent. A malformed
    canonical field is not rescued by a legacy alias.
    """
    safe_default = _finite_number(default)
    if safe_default is None:
        safe_default = 0.0

    if not isinstance(details, dict):
        return safe_default

    for key in keys:
        if key not in details:
            continue
        numeric = _finite_number(
            details.get(key),
            min_value=min_value,
            max_value=max_value,
        )
        return numeric if numeric is not None else safe_default
    return safe_default


def _task_value(task: Any, key: str) -> Any:
    if task is None:
        return None
    if isinstance(task, dict):
        return task.get(key)

    try:
        return getattr(task, key, None)
    except Exception:
        # Malformed task objects fail closed. KeyboardInterrupt/SystemExit propagate.
        return None


def _extract_task(task: Any) -> dict[str, float]:
    """Extract only completed, numerically valid task evidence.

    A task is considered present only when attempts is an actual positive integer
    and at least one measurable task field is valid.
    """
    attempts = _strict_positive_int(_task_value(task, "attempts"))

    reaction_time = _finite_number(
        _task_value(task, "reaction_time"),
        min_value=0.0,
    )
    if reaction_time is not None and reaction_time <= 0.0:
        reaction_time = None

    errors = _strict_non_negative_int(_task_value(task, "errors"))

    task_present = attempts is not None and (
        reaction_time is not None or errors is not None
    )
    if not task_present:
        return {
            "task_reaction_time": 0.0,
            "task_errors": 0.0,
            "task_present": 0.0,
        }

    return {
        "task_reaction_time": reaction_time if reaction_time is not None else 0.0,
        "task_errors": float(errors) if errors is not None else 0.0,
        "task_present": 1.0,
    }


def features_from_signals(signals: dict, task: Any = None) -> tuple[dict, dict]:
    """Convert analyzer outputs into the stable ML feature schema.

    Only results with details.status == "ok" contribute evidence. Missing or
    failed analyzers produce zero placeholders with explicit missing_* flags.
    """
    safe_signals = _safe_dict(signals)

    camera = _safe_dict(safe_signals.get("camera"))
    audio = _safe_dict(safe_signals.get("voice"))
    video = _safe_dict(safe_signals.get("video"))

    camera_details = _readable_details(camera)
    audio_details = _readable_details(audio)
    video_details = _readable_details(video)

    camera_score = _signal_score(camera, "image_confidence")
    audio_score = _signal_score(audio, "audio_confidence")
    video_score = _signal_score(video, "visual_confidence")

    face_frames = _strict_non_negative_int(video_details.get("face_frames"))
    sampled_frames = _strict_non_negative_int(video_details.get("sampled_frames"))

    feature_map = {
        "camera_score": camera_score if camera_score is not None else 0.0,
        "face_detected": _bool(camera_details.get("face_detected")),
        "avg_ear": _get_detail(
            camera_details,
            "avg_ear",
            0.0,
            min_value=0.0,
            max_value=1.0,
        ),
        "eyes_closed": _bool(camera_details.get("eyes_closed")),
        "audio_score": audio_score if audio_score is not None else 0.0,
        "audio_energy": _get_first_valid_detail(
            audio_details,
            ("rms_energy", "energy"),
            0.0,
            min_value=0.0,
            max_value=1.0,
        ),
        "audio_zcr": _get_first_valid_detail(
            audio_details,
            ("zero_crossing_rate", "zcr"),
            0.0,
            min_value=0.0,
            max_value=1.0,
        ),
        "audio_centroid": _get_first_valid_detail(
            audio_details,
            ("spectral_centroid", "centroid"),
            0.0,
            min_value=0.0,
        ),
        "audio_duration": _get_first_valid_detail(
            audio_details,
            ("duration_seconds", "duration_sec"),
            0.0,
            min_value=0.0,
        ),
        "audio_silent": _bool(audio_details.get("silent")),
        "video_score": video_score if video_score is not None else 0.0,
        "video_sway_std": _get_detail(
            video_details,
            "sway_std",
            0.0,
            min_value=0.0,
        ),
        "video_face_rate": _get_first_valid_detail(
            video_details,
            ("face_or_subject_visibility", "face_rate"),
            0.0,
            min_value=0.0,
            max_value=1.0,
        ),
        "video_face_frames": float(face_frames) if face_frames is not None else 0.0,
        # frame_count is not a semantic substitute for sampled_frames.
        "video_sampled_frames": (
            float(sampled_frames) if sampled_frames is not None else 0.0
        ),
        "missing_camera": _bool(camera_score is None),
        "missing_audio": _bool(audio_score is None),
        "missing_video": _bool(video_score is None),
    }

    feature_map.update(_extract_task(task))
    return feature_map, safe_signals


_BINARY_FEATURES = {
    "face_detected",
    "eyes_closed",
    "audio_silent",
    "task_present",
    "missing_camera",
    "missing_audio",
    "missing_video",
}

_UNIT_INTERVAL_FEATURES = {
    "camera_score",
    "avg_ear",
    "audio_score",
    "audio_energy",
    "audio_zcr",
    "video_score",
    "video_face_rate",
}

_NON_NEGATIVE_FEATURES = {
    "audio_centroid",
    "audio_duration",
    "video_sway_std",
    "task_reaction_time",
}

_COUNT_FEATURES = {
    "video_face_frames",
    "video_sampled_frames",
    "task_errors",
}


def _validated_feature_value(key: str, value: Any) -> float:
    numeric = _finite_number(value)
    if numeric is None:
        return 0.0

    if key in _BINARY_FEATURES:
        return numeric if numeric in {0.0, 1.0} else 0.0

    if key in _UNIT_INTERVAL_FEATURES:
        return numeric if 0.0 <= numeric <= 1.0 else 0.0

    if key in _NON_NEGATIVE_FEATURES:
        return numeric if numeric >= 0.0 else 0.0

    if key in _COUNT_FEATURES:
        return numeric if numeric >= 0.0 and numeric.is_integer() else 0.0

    return numeric


def vector_from_features(feature_map: dict) -> list[float]:
    """Build a finite fixed-length vector in FEATURE_ORDER."""
    safe_features = _safe_dict(feature_map)
    return [
        _validated_feature_value(key, safe_features.get(key))
        for key in FEATURE_ORDER
    ]


def _media_value(media: Any, key: str) -> Any:
    if isinstance(media, dict):
        return media.get(key)
    if media is None:
        return None

    try:
        return getattr(media, key, None)
    except Exception:
        return None


def features_from_media(media: Any, task: Any = None) -> tuple[dict, dict]:
    """Run the existing analyzers, then extract the stable ML feature input.

    This preserves the existing analysis sequence and performs no Directus I/O.
    """
    from audio import analyze_audio
    from video import analyze_video
    from vision import analyze_face

    image_path = _media_value(media, "image")
    audio_path = _media_value(media, "audio")
    video_path = _media_value(media, "video")

    signals = {
        "camera": analyze_face(image_path),
        "voice": analyze_audio(audio_path),
        "video": analyze_video(video_path),
    }
    return features_from_signals(signals, task=task)