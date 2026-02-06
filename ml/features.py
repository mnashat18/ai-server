from audio import analyze_audio
from video import analyze_video
from vision import analyze_face


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


def _bool(value) -> float:
    return 1.0 if value else 0.0


def _get_detail(details: dict, key: str, default=0.0) -> float:
    if not details:
        return float(default)
    return float(details.get(key, default))


def _extract_task(task) -> dict:
    if not task:
        return {"task_reaction_time": 0.0, "task_errors": 0.0, "task_present": 0.0}
    reaction_time = getattr(task, "reaction_time", None)
    if isinstance(task, dict):
        reaction_time = task.get("reaction_time")
    errors = getattr(task, "errors", None)
    if isinstance(task, dict):
        errors = task.get("errors")
    return {
        "task_reaction_time": float(reaction_time) if reaction_time is not None else 0.0,
        "task_errors": float(errors) if errors is not None else 0.0,
        "task_present": 1.0,
    }


def features_from_signals(signals: dict, task=None) -> tuple[dict, dict]:
    camera = signals.get("camera", {})
    audio = signals.get("voice", {})
    video = signals.get("video", {})

    camera_score = camera.get("score")
    audio_score = audio.get("score")
    video_score = video.get("score")

    camera_details = camera.get("details", {})
    audio_details = audio.get("details", {})
    video_details = video.get("details", {})

    feature_map = {
        "camera_score": float(camera_score) if camera_score is not None else 0.0,
        "face_detected": _bool(camera_details.get("face_detected")),
        "avg_ear": _get_detail(camera_details, "avg_ear", 0.0),
        "eyes_closed": _bool(camera_details.get("eyes_closed")),
        "audio_score": float(audio_score) if audio_score is not None else 0.0,
        "audio_energy": _get_detail(audio_details, "energy", 0.0),
        "audio_zcr": _get_detail(audio_details, "zcr", 0.0),
        "audio_centroid": _get_detail(audio_details, "centroid", 0.0),
        "audio_duration": _get_detail(audio_details, "duration_sec", 0.0),
        "audio_silent": _bool(audio_details.get("silent")),
        "video_score": float(video_score) if video_score is not None else 0.0,
        "video_sway_std": _get_detail(video_details, "sway_std", 0.0),
        "video_face_rate": _get_detail(video_details, "face_rate", 0.0),
        "video_face_frames": _get_detail(video_details, "face_frames", 0.0),
        "video_sampled_frames": _get_detail(video_details, "sampled_frames", 0.0),
        "missing_camera": _bool(camera_score is None),
        "missing_audio": _bool(audio_score is None),
        "missing_video": _bool(video_score is None),
    }
    feature_map.update(_extract_task(task))
    return feature_map, signals


def vector_from_features(feature_map: dict) -> list[float]:
    return [float(feature_map.get(key, 0.0)) for key in FEATURE_ORDER]


def features_from_media(media, task=None) -> tuple[dict, dict]:
    image_path = media.get("image") if isinstance(media, dict) else getattr(media, "image", None)
    audio_path = media.get("audio") if isinstance(media, dict) else getattr(media, "audio", None)
    video_path = media.get("video") if isinstance(media, dict) else getattr(media, "video", None)

    signals = {
        "camera": analyze_face(image_path),
        "voice": analyze_audio(audio_path),
        "video": analyze_video(video_path),
    }
    return features_from_signals(signals, task=task)
