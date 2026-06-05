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
MAX_SAMPLED_FRAMES = 180
FRAME_STRIDE = 3

mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=False) if mp else None


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
    prev_gray = None
    warnings: list[str] = []

    try:
        while cap.isOpened() and sampled_frames < MAX_SAMPLED_FRAMES:
            ok, frame = cap.read()
            if not ok:
                break

            current_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if current_index % FRAME_STRIDE != 0:
                continue

            sampled_frames += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = float(np.mean(gray))
            blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            brightness_values.append(brightness)
            blur_values.append(blur_var)

            is_bright_enough = brightness >= LOW_LIGHT_THRESHOLD
            is_sharp_enough = blur_var >= BLUR_THRESHOLD
            if not is_bright_enough:
                low_light_frames += 1
            if not is_sharp_enough:
                blurry_frames += 1

            if prev_gray is not None:
                diff = cv2.absdiff(prev_gray, gray)
                camera_motion.append(float(np.mean(diff)))
            prev_gray = gray

            face_visible, landmark_confidence = _landmark_visibility(frame)
            if landmark_confidence is not None:
                landmark_confidences.append(float(landmark_confidence))
            if face_visible:
                face_frames += 1

            usable = is_bright_enough and is_sharp_enough and face_visible
            if usable:
                usable_frames += 1
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
