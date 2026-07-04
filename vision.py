import math
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
BLUR_THRESHOLD = 60.0
MIN_RESOLUTION = 256 * 256
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

mp_face = (
    mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
    )
    if mp
    else None
)


def _dist(a, b) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def _eye_aspect_ratio(landmarks, idxs) -> float:
    p0, p1, p2, p3, p4, p5 = [landmarks[i] for i in idxs]
    horizontal = _dist(p0, p3)
    if horizontal == 0:
        return 0.0
    vertical = _dist(p1, p5) + _dist(p2, p4)
    return vertical / (2.0 * horizontal)


def _brightness_score(gray: np.ndarray) -> float:
    mean_v = float(np.mean(gray))
    return clamp01(mean_v / 160.0, 0.0) or 0.0


def _sharpness_from_blur(blur_var: float) -> float:
    return clamp01(blur_var / 180.0, 0.0) or 0.0


def analyze_face(image_path: str) -> dict:
    if not image_path:
        return {
            "score": None,
            "details": {
                "status": "missing",
                "image_warnings": ["image_missing"],
            },
        }

    if cv2 is None:
        return {
            "score": None,
            "details": {
                "status": "invalid_image",
                "image_warnings": ["image_missing"],
            },
        }

    img = cv2.imread(image_path)
    if img is None:
        return {
            "score": None,
            "details": {
                "status": "invalid_image",
                "image_warnings": ["image_missing"],
            },
        }

    height, width = img.shape[:2]
    resolution = {"width": int(width), "height": int(height)}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness_raw = float(np.mean(gray))
    blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness_score = _brightness_score(gray)
    sharpness_score = _sharpness_from_blur(blur_var)
    blur_score = 1.0 - sharpness_score
    warnings: list[str] = []
    subject_visibility = 0.0
    face_detected = False
    landmark_confidence = None
    avg_ear = None
    left_eye_aperture = None
    right_eye_aperture = None
    left_right_eye_asymmetry = None
    eyes_closed = None

    if width * height < MIN_RESOLUTION:
        warnings.append("image_low_resolution")
    if brightness_raw < LOW_LIGHT_THRESHOLD:
        warnings.append("image_too_dark")
    if blur_var < BLUR_THRESHOLD:
        warnings.append("image_blurry")

    if mp_face is None:
        warnings.append("subject_not_visible")
    else:
        try:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = mp_face.process(rgb)
            if res.multi_face_landmarks:
                face_detected = True
                subject_visibility = 1.0
                landmark_confidence = 0.9
                landmarks = res.multi_face_landmarks[0].landmark
                left_eye_aperture = _eye_aspect_ratio(landmarks, LEFT_EYE)
                right_eye_aperture = _eye_aspect_ratio(landmarks, RIGHT_EYE)
                avg_ear = (left_eye_aperture + right_eye_aperture) / 2.0
                left_right_eye_asymmetry = abs(left_eye_aperture - right_eye_aperture)
                eyes_closed = avg_ear < 0.18
                if eyes_closed:
                    subject_visibility = 0.85
            else:
                warnings.append("subject_not_visible")
        except Exception:
            warnings.append("subject_not_visible")

    quality_components = [
        brightness_score,
        sharpness_score,
        clamp01(subject_visibility, 0.0) or 0.0,
        1.0 if width * height >= MIN_RESOLUTION else 0.45,
    ]
    image_quality_score = round(sum(quality_components) / len(quality_components), 4)
    image_confidence = round(
        max(0.0, min(image_quality_score * (0.95 if face_detected else 0.55), 1.0)),
        4,
    )

    details = {
        "status": "ok",
        "resolution": resolution,
        "brightness_score": safe_number(brightness_score),
        "blur_score": safe_number(blur_score),
        "sharpness_score": safe_number(sharpness_score),
        "subject_visibility": safe_number(subject_visibility),
        "image_quality_score": safe_number(image_quality_score),
        "image_confidence": safe_number(image_confidence),
        "image_warnings": clean_warning_codes(warnings),
        "face_detected": face_detected,
        "landmark_detection_confidence": safe_number(landmark_confidence),
        "avg_brightness": safe_number(brightness_raw, 2),
        "blur_var": safe_number(blur_var, 2),
        "avg_ear": safe_number(avg_ear),
        "left_eye_aperture": safe_number(left_eye_aperture),
        "right_eye_aperture": safe_number(right_eye_aperture),
        "left_right_eye_asymmetry": safe_number(left_right_eye_asymmetry),
        "eyes_closed": eyes_closed,
        "low_light": brightness_raw < LOW_LIGHT_THRESHOLD,
        "blurry": blur_var < BLUR_THRESHOLD,
    }
    return {"score": details["image_confidence"], "details": details}
