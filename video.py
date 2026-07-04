import os

os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np

from utils import clamp01, clean_warning_codes, safe_number

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

try:
    import mediapipe as mp
except Exception:  # pragma: no cover
    mp = None


LOW_LIGHT_THRESHOLD = 75.0
BLUR_THRESHOLD = 65.0
MIN_DURATION_SEC = 1.5
MIN_RESOLUTION = (480, 360)
MAX_SAMPLED_FRAMES = 8
MAX_EYE_CLOSURE_WINDOWS = 2
EYE_CLOSURE_SAMPLES_PER_WINDOW = 4
MIN_EYE_CLOSURE_DURATION_SECONDS = 0.45
MAX_EYE_CLOSURE_WINDOW_SECONDS = 1.2
DEFAULT_EYE_CLOSURE_WINDOW_SPAN_SECONDS = 0.6
MIN_EYE_EVIDENCE_FRAMES = 4
SUSTAINED_EYE_CLOSURE_EAR = 0.16
SUSTAINED_EYE_CLOSURE_STRONG_EAR = 0.14
MAX_EYE_ASYMMETRY = 0.08

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

mp_face = (
    mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    if mp
    else None
)


def _resolution_score(width: int, height: int) -> float:
    if width >= 1280 and height >= 720:
        return 1.0
    if width >= MIN_RESOLUTION[0] and height >= MIN_RESOLUTION[1]:
        return 0.7
    if width > 0 and height > 0:
        return 0.4
    return 0.0


def _landmark_visibility(frame) -> tuple[bool, float | None]:
    if mp_face is None:
        return False, None
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = mp_face.process(rgb)
    if res.multi_face_landmarks:
        return True, 0.9
    return False, 0.0


def _dist(a, b) -> float:
    return float(np.hypot(a.x - b.x, a.y - b.y))


def _eye_aspect_ratio(landmarks, idxs) -> float:
    p0, p1, p2, p3, p4, p5 = [landmarks[i] for i in idxs]
    horizontal = _dist(p0, p3)
    if horizontal == 0:
        return 0.0
    vertical = _dist(p1, p5) + _dist(p2, p4)
    return vertical / (2.0 * horizontal)


def _longest_true_streak(values: list[bool]) -> int:
    longest = 0
    current = 0
    for value in values:
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _eye_closure_sample_windows(frame_count: int, fps: float, duration_seconds: float) -> list[list[dict[str, float | int]]]:
    if frame_count < EYE_CLOSURE_SAMPLES_PER_WINDOW or fps <= 0 or duration_seconds < MIN_EYE_CLOSURE_DURATION_SECONDS:
        return []

    min_window_span_frames = max(int(np.ceil(MIN_EYE_CLOSURE_DURATION_SECONDS * fps)), EYE_CLOSURE_SAMPLES_PER_WINDOW - 1)
    max_window_span_frames = min(
        int(np.floor(MAX_EYE_CLOSURE_WINDOW_SECONDS * fps)),
        frame_count - 1,
    )
    if max_window_span_frames < min_window_span_frames:
        return []

    target_span_frames = int(round(DEFAULT_EYE_CLOSURE_WINDOW_SPAN_SECONDS * fps))
    target_span_frames = max(target_span_frames, min_window_span_frames)
    target_span_frames = min(target_span_frames, max_window_span_frames)
    if target_span_frames < min_window_span_frames:
        return []

    window_count = 1
    if duration_seconds >= 1.0 and frame_count >= 8:
        window_count = 2

    windows: list[list[dict[str, float | int]]] = []
    for window_index, center_ratio in enumerate([0.25, 0.75][:window_count]):
        center_frame = int(round((frame_count - 1) * center_ratio))
        start_frame = center_frame - (target_span_frames // 2)
        max_start_frame = max(frame_count - 1 - target_span_frames, 0)
        start_frame = max(0, min(start_frame, max_start_frame))
        end_frame = start_frame + target_span_frames
        sample_frame_indices = [
            start_frame,
            start_frame + int(round(target_span_frames / 3.0)),
            start_frame + int(round((2.0 * target_span_frames) / 3.0)),
            end_frame,
        ]
        sample_frame_indices = [max(0, min(frame_count - 1, frame_index)) for frame_index in sample_frame_indices]
        if len(set(sample_frame_indices)) < EYE_CLOSURE_SAMPLES_PER_WINDOW:
            continue
        window_samples: list[dict[str, float | int]] = []
        for frame_index in sample_frame_indices:
            timestamp = frame_index / fps
            window_samples.append(
                {
                    "window_id": window_index,
                    "frame_index": frame_index,
                    "timestamp": timestamp,
                }
            )
        windows.append(window_samples)
    return windows


def _longest_temporal_eye_closure_streak(
    observations: list[dict[str, float | bool | int | None]],
) -> tuple[int, int, float]:
    grouped: dict[int, list[dict[str, float | bool | int | None]]] = {}
    for observation in observations:
        window_id = int(observation.get("window_id") or 0)
        grouped.setdefault(window_id, []).append(observation)

    longest = 0
    best_window_seconds = 0.0
    best_window_ms = 0
    for window_samples in grouped.values():
        ordered = sorted(window_samples, key=lambda item: float(item.get("timestamp") or 0.0))
        if len(ordered) < EYE_CLOSURE_SAMPLES_PER_WINDOW:
            continue
        if len({int(sample.get("frame_index") or -1) for sample in ordered}) < EYE_CLOSURE_SAMPLES_PER_WINDOW:
            continue
        all_closed = all(
            bool(sample.get("usable"))
            and bool(sample.get("bright_enough"))
            and bool(sample.get("sharp_enough"))
            and bool(sample.get("face_visible"))
            and bool(sample.get("landmark_valid"))
            and bool(sample.get("eye_closed"))
            for sample in ordered
        )
        if not all_closed:
            continue
        window_seconds = max(0.0, float(ordered[-1].get("timestamp") or 0.0) - float(ordered[0].get("timestamp") or 0.0))
        if window_seconds < MIN_EYE_CLOSURE_DURATION_SECONDS or window_seconds > MAX_EYE_CLOSURE_WINDOW_SECONDS:
            continue
        if len(ordered) > longest:
            longest = len(ordered)
            best_window_seconds = window_seconds
            best_window_ms = int(round(window_seconds * 1000))

    return longest, best_window_ms, best_window_seconds


def analyze_video(video_path: str) -> dict:
    if not video_path:
        return {
            "score": None,
            "details": {
                "status": "missing",
                "visual_warnings": ["video_missing"],
            },
        }

    if cv2 is None:
        return {
            "score": None,
            "details": {
                "status": "open_failed",
                "visual_warnings": ["video_missing"],
            },
        }

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {
            "score": None,
            "details": {
                "status": "open_failed",
                "visual_warnings": ["video_missing"],
            },
        }

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration_seconds = (frame_count / fps) if fps and frame_count else 0.0

    sampled_frames = 0
    usable_frames = 0
    face_frames = 0
    low_light_frames = 0
    blurry_frames = 0
    landmark_confidences: list[float] = []
    brightness_values: list[float] = []
    blur_values: list[float] = []
    camera_motion: list[float] = []
    eye_apertures: list[float] = []
    eye_asymmetries: list[float] = []
    eye_closure_samples: list[bool] = []
    frame_observations: list[dict[str, float | bool | int | None]] = []
    prev_gray = None
    warnings: list[str] = []
    eye_closure_windows = _eye_closure_sample_windows(frame_count, fps, duration_seconds)

    try:
        for window in eye_closure_windows:
            for sample in window:
                if not cap.isOpened():
                    break
                frame_index = int(sample["frame_index"])
                timestamp = float(sample["timestamp"])
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = cap.read()
                if not ok:
                    frame_observations.append(
                        {
                            "window_id": sample["window_id"],
                            "frame_index": frame_index,
                            "timestamp": frame_index / fps if fps else timestamp,
                            "usable": False,
                            "bright_enough": False,
                            "sharp_enough": False,
                            "face_visible": False,
                            "landmark_valid": False,
                            "eye_closed": False,
                        }
                    )
                    continue

                sampled_frames += 1
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                brightness = float(np.mean(gray))
                blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                brightness_values.append(brightness)
                blur_values.append(blur_var)

                bright_enough = brightness >= LOW_LIGHT_THRESHOLD
                sharp_enough = blur_var >= BLUR_THRESHOLD
                if not bright_enough:
                    low_light_frames += 1
                if not sharp_enough:
                    blurry_frames += 1

                if prev_gray is not None:
                    diff = cv2.absdiff(prev_gray, gray)
                    camera_motion.append(float(np.mean(diff)))
                prev_gray = gray

                face_visible = False
                landmark_confidence = 0.0 if mp_face is not None else None
                landmark_valid = False
                eye_closed = False
                if mp_face is not None:
                    try:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        res = mp_face.process(rgb)
                        if res.multi_face_landmarks:
                            face_visible = True
                            landmark_confidence = 0.9
                            landmarks = res.multi_face_landmarks[0].landmark
                            left_eye_aperture = _eye_aspect_ratio(landmarks, LEFT_EYE)
                            right_eye_aperture = _eye_aspect_ratio(landmarks, RIGHT_EYE)
                            avg_ear = (left_eye_aperture + right_eye_aperture) / 2.0
                            asymmetry = abs(left_eye_aperture - right_eye_aperture)
                            landmark_valid = True
                            if bright_enough and sharp_enough and asymmetry <= MAX_EYE_ASYMMETRY:
                                eye_apertures.append(avg_ear)
                                eye_asymmetries.append(asymmetry)
                                eye_closed = avg_ear <= SUSTAINED_EYE_CLOSURE_EAR
                    except Exception:
                        face_visible = False
                        landmark_valid = False
                        landmark_confidence = 0.0
                if landmark_confidence is not None:
                    landmark_confidences.append(float(landmark_confidence))
                if face_visible:
                    face_frames += 1

                usable = bright_enough and sharp_enough and face_visible and landmark_valid
                if usable:
                    usable_frames += 1
                eye_closure_samples.append(bool(usable and eye_closed))
                frame_observations.append(
                    {
                        "window_id": sample["window_id"],
                        "frame_index": frame_index,
                        "timestamp": frame_index / fps if fps else timestamp,
                        "usable": usable,
                        "bright_enough": bright_enough,
                        "sharp_enough": sharp_enough,
                        "face_visible": face_visible,
                        "landmark_valid": landmark_valid,
                        "eye_closed": eye_closed,
                    }
                )
    finally:
        cap.release()

    if duration_seconds and duration_seconds < MIN_DURATION_SEC:
        warnings.append("video_too_short")
    if width < MIN_RESOLUTION[0] or height < MIN_RESOLUTION[1]:
        warnings.append("video_low_resolution")

    avg_brightness = float(np.mean(brightness_values)) if brightness_values else 0.0
    avg_blur = float(np.mean(blur_values)) if blur_values else 0.0
    brightness_score = clamp01(avg_brightness / 160.0, 0.0) or 0.0
    sharpness_score = clamp01(avg_blur / 180.0, 0.0) or 0.0
    blur_score = 1.0 - sharpness_score
    low_light_ratio = (low_light_frames / sampled_frames) if sampled_frames else 1.0
    blurry_ratio = (blurry_frames / sampled_frames) if sampled_frames else 1.0
    usable_frame_ratio = (usable_frames / sampled_frames) if sampled_frames else 0.0
    face_visibility = (face_frames / sampled_frames) if sampled_frames else 0.0
    motion_mean = float(np.mean(camera_motion)) if camera_motion else 0.0
    motion_std = float(np.std(camera_motion)) if camera_motion else 0.0
    motion_stability_score = float(np.clip(1.0 - min(motion_std / 32.0, 1.0), 0.0, 1.0))
    landmark_detection_confidence = (
        float(np.mean(landmark_confidences)) if landmark_confidences else (0.0 if mp_face else None)
    )
    avg_eye_aperture = float(np.mean(eye_apertures)) if eye_apertures else 0.0
    eye_aperture_std = float(np.std(eye_apertures)) if eye_apertures else 0.0
    longest_eye_closure_streak, closure_window_ms, closure_window_seconds = _longest_temporal_eye_closure_streak(frame_observations)
    closed_eye_ratio = (sum(1 for value in eye_closure_samples if value) / len(eye_closure_samples)) if eye_closure_samples else 0.0
    reliable_eye_landmarks = bool(
        mp_face is not None
        and len(eye_apertures) >= MIN_EYE_EVIDENCE_FRAMES
        and face_visibility >= 0.65
        and motion_stability_score >= 0.45
        and sharpness_score >= 0.4
        and brightness_score >= 0.4
    )
    sustained_eye_closure = bool(
        reliable_eye_landmarks
        and longest_eye_closure_streak >= EYE_CLOSURE_SAMPLES_PER_WINDOW
        and len(eye_closure_samples) >= EYE_CLOSURE_SAMPLES_PER_WINDOW
        and closed_eye_ratio >= 0.75
        and avg_eye_aperture <= SUSTAINED_EYE_CLOSURE_STRONG_EAR
        and eye_aperture_std <= 0.025
        and closure_window_seconds >= MIN_EYE_CLOSURE_DURATION_SECONDS
        and closure_window_seconds <= MAX_EYE_CLOSURE_WINDOW_SECONDS
    )

    if avg_brightness < LOW_LIGHT_THRESHOLD:
        warnings.append("video_too_dark")
    if avg_blur < BLUR_THRESHOLD:
        warnings.append("video_blurry")
    if motion_stability_score < 0.4:
        warnings.append("unstable_camera")
    if usable_frame_ratio < 0.3:
        warnings.append("insufficient_usable_frames")
    if face_visibility < 0.25:
        warnings.append("subject_not_visible")
    if mp_face is not None and face_frames == 0:
        warnings.append("landmark_detection_failed")
    if sustained_eye_closure:
        warnings.append("sustained_eye_closure")

    visual_quality_score = float(
        np.clip(
            0.2 * brightness_score
            + 0.2 * sharpness_score
            + 0.2 * motion_stability_score
            + 0.25 * usable_frame_ratio
            + 0.15 * _resolution_score(width, height),
            0.0,
            1.0,
        )
    )
    visual_confidence = float(
        np.clip(
            0.55 * visual_quality_score
            + 0.25 * face_visibility
            + 0.2 * (landmark_detection_confidence if landmark_detection_confidence is not None else 0.35),
            0.0,
            1.0,
        )
    )

    details = {
        "status": "ok",
        "frame_count": frame_count,
        "duration_seconds": safe_number(duration_seconds, 3),
        "fps": safe_number(fps, 3),
        "resolution": {"width": width, "height": height},
        "brightness_score": safe_number(brightness_score),
        "blur_score": safe_number(blur_score),
        "sharpness_score": safe_number(sharpness_score),
        "motion_stability_score": safe_number(motion_stability_score),
        "usable_frame_ratio": safe_number(usable_frame_ratio),
        "face_or_subject_visibility": safe_number(face_visibility),
        "landmark_detection_confidence": safe_number(landmark_detection_confidence),
        "reliable_eye_landmarks": reliable_eye_landmarks,
        "sustained_eye_closure": sustained_eye_closure,
        "eye_closure_sample_count": len(eye_closure_samples),
        "closed_eye_ratio": safe_number(closed_eye_ratio),
        "longest_eye_closure_streak": longest_eye_closure_streak,
        "eye_closure_window_ms": closure_window_ms,
        "avg_eye_aperture": safe_number(avg_eye_aperture),
        "eye_aperture_std": safe_number(eye_aperture_std),
        "avg_eye_asymmetry": safe_number(float(np.mean(eye_asymmetries)) if eye_asymmetries else None),
        "visual_confidence": safe_number(visual_confidence),
        "visual_quality_score": safe_number(visual_quality_score),
        "visual_warnings": clean_warning_codes(warnings),
        "frames": frame_count,
        "sampled_frames": sampled_frames,
        "face_frames": face_frames,
        "face_rate": safe_number(face_visibility),
        "sway_std": safe_number(motion_std),
        "avg_brightness": safe_number(avg_brightness, 2),
        "avg_blur_var": safe_number(avg_blur, 2),
        "low_light_frames": low_light_frames,
        "blurry_frames": blurry_frames,
        "low_light_rate": safe_number(low_light_ratio),
        "blurry_rate": safe_number(blurry_ratio),
        "quality_flags": clean_warning_codes(warnings),
    }
    return {"score": details["visual_confidence"], "details": details}
