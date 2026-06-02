from __future__ import annotations

from typing import Any

from config import (
    MIN_AUDIO_DURATION_SEC,
    MIN_AUDIO_ENERGY,
    MIN_TASK_ATTEMPTS,
    MIN_VIDEO_FACE_FRAMES,
    MIN_VIDEO_SAMPLED_FRAMES,
)


def _task_value(task: Any, key: str) -> Any:
    if task is None:
        return None
    if isinstance(task, dict):
        return task.get(key)
    return getattr(task, key, None)


def assess_quality(signals: dict, task: Any = None) -> dict:
    reasons: list[str] = []
    weak_reasons: list[str] = []
    usable_modalities = 0
    confidence_multiplier = 1.0

    camera = signals.get("camera", {})
    video = signals.get("video", {})
    voice = signals.get("voice", {})

    camera_details = camera.get("details", {})
    video_details = video.get("details", {})
    voice_details = voice.get("details", {})

    visual_usable = False
    if camera.get("score") is not None and camera_details.get("face_detected"):
        visual_usable = True
    if video.get("score") is not None:
        face_frames = int(video_details.get("face_frames", 0) or 0)
        sampled_frames = int(video_details.get("sampled_frames", 0) or 0)
        if face_frames >= MIN_VIDEO_FACE_FRAMES and sampled_frames >= MIN_VIDEO_SAMPLED_FRAMES:
            visual_usable = True
        elif sampled_frames > 0 and face_frames < MIN_VIDEO_FACE_FRAMES:
            weak_reasons.append("limited_face_frames")

    if visual_usable:
        usable_modalities += 1
        if camera_details.get("low_light") or camera_details.get("blurry"):
            weak_reasons.append("image_quality_weak")
            confidence_multiplier -= 0.1
        if (video_details.get("quality_flags") or []):
            weak_reasons.append("video_quality_weak")
            confidence_multiplier -= 0.1
    elif camera.get("score") is not None or video.get("score") is not None:
        reasons.append("low_quality_visual")

    audio_duration = float(voice_details.get("duration_sec", 0.0) or 0.0)
    audio_energy = float(voice_details.get("energy", 0.0) or 0.0)
    if voice.get("score") is not None:
        if audio_duration >= MIN_AUDIO_DURATION_SEC and not voice_details.get("silent"):
            usable_modalities += 1
            if audio_energy < (MIN_AUDIO_ENERGY * 1.5):
                weak_reasons.append("audio_energy_low")
                confidence_multiplier -= 0.1
        elif audio_duration > 0:
            reasons.append("low_quality_audio")
    elif voice_details.get("status") not in {None, "missing"}:
        reasons.append("low_quality_audio")

    attempts = _task_value(task, "attempts")
    reaction_time = _task_value(task, "reaction_time")
    errors = _task_value(task, "errors")
    task_quality = "missing"
    if reaction_time is not None or errors is not None or attempts is not None:
        valid_reaction = reaction_time is not None and float(reaction_time) > 0
        valid_errors = errors is None or int(errors) >= 0
        attempt_count = int(attempts or 0)
        if valid_reaction and valid_errors and attempt_count >= MIN_TASK_ATTEMPTS:
            usable_modalities += 1
            task_quality = "good"
        elif valid_reaction and valid_errors:
            weak_reasons.append("task_attempts_low")
            confidence_multiplier -= 0.1
            task_quality = "weak"
        else:
            reasons.append("low_quality_task")
            task_quality = "failed"

    if usable_modalities == 0:
        reasons.append("no_usable_modalities")

    failed = bool(reasons)
    confidence_multiplier = max(0.3, min(confidence_multiplier, 1.0))
    weak = bool(weak_reasons) and not failed
    status = "failed" if failed else ("weak" if weak else "passed")
    suggested_action = (
        "Please retake the scan in better lighting with clear face, voice, and reaction input."
        if failed
        else (
            "Scan accepted, but a retake is recommended for stronger confidence."
            if weak
            else "Scan quality is acceptable."
        )
    )

    return {
        "status": status,
        "passed": not failed,
        "weak": weak,
        "failure_reason": reasons[0] if reasons else None,
        "reasons": reasons,
        "weak_reasons": weak_reasons,
        "confidence_multiplier": round(confidence_multiplier, 3),
        "usable_modalities": usable_modalities,
        "task_quality": task_quality,
        "suggested_action": suggested_action,
        "retake_required": failed,
    }
