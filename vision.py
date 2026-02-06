#vision.py
import cv2
import math
import mediapipe as mp
import numpy as np

from config import MIN_EAR

mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LOW_LIGHT_THRESHOLD = 70.0
BLUR_THRESHOLD = 40.0


def _apply_gamma(bgr, gamma: float):
    if gamma <= 0:
        return bgr
    inv = 1.0 / gamma
    table = (np.arange(256) / 255.0) ** inv
    table = np.clip(table * 255.0, 0, 255).astype("uint8")
    return cv2.LUT(bgr, table)


def _enhance_bgr(bgr):
    # Improve low-light and contrast before face mesh.
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab = cv2.merge((l2, a, b))
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mean_v = float(np.mean(gray))
    if mean_v < LOW_LIGHT_THRESHOLD:
        bgr = _apply_gamma(bgr, 1.6)
    elif mean_v > 200:
        bgr = _apply_gamma(bgr, 0.85)
    return bgr, mean_v


def _dist(a, b) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def _eye_aspect_ratio(landmarks, idxs) -> float:
    p0, p1, p2, p3, p4, p5 = [landmarks[i] for i in idxs]
    horizontal = _dist(p0, p3)
    if horizontal == 0:
        return 0.0
    vertical = _dist(p1, p5) + _dist(p2, p4)
    return vertical / (2.0 * horizontal)


def analyze_face(image_path: str) -> dict:
    if not image_path:
        return {"score": None, "details": {"status": "missing"}}
    img = cv2.imread(image_path)
    if img is None:
        return {"score": None, "details": {"status": "invalid_image"}}

    try:
        enhanced, mean_v = _enhance_bgr(img)
        rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        res = mp_face.process(rgb)
    except Exception:
        return {"score": None, "details": {"status": "processing_error"}}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    if not res.multi_face_landmarks:
        return {
            "score": 0.4,
            "details": {
                "face_detected": False,
                "avg_brightness": round(mean_v, 2),
                "blur_var": round(blur, 2),
                "low_light": mean_v < LOW_LIGHT_THRESHOLD,
                "blurry": blur < BLUR_THRESHOLD,
            },
        }

    landmarks = res.multi_face_landmarks[0].landmark
    left_ear = _eye_aspect_ratio(landmarks, LEFT_EYE)
    right_ear = _eye_aspect_ratio(landmarks, RIGHT_EYE)
    avg_ear = (left_ear + right_ear) / 2.0
    if avg_ear < MIN_EAR:
        return {
            "score": 0.35,
            "details": {
                "face_detected": True,
                "avg_ear": round(avg_ear, 4),
                "eyes_closed": True,
                "avg_brightness": round(mean_v, 2),
                "blur_var": round(blur, 2),
                "low_light": mean_v < LOW_LIGHT_THRESHOLD,
                "blurry": blur < BLUR_THRESHOLD,
            },
        }

    return {
        "score": 0.9,
        "details": {
            "face_detected": True,
            "avg_ear": round(avg_ear, 4),
            "eyes_closed": False,
            "avg_brightness": round(mean_v, 2),
            "blur_var": round(blur, 2),
            "low_light": mean_v < LOW_LIGHT_THRESHOLD,
            "blurry": blur < BLUR_THRESHOLD,
        },
    }
