#video.py
import cv2
import mediapipe as mp
import numpy as np

from config import MIN_SWAY_STD, VIDEO_FRAME_STRIDE, VIDEO_MAX_FRAMES

mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=False)
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


def analyze_video(video_path: str) -> dict:
    if not video_path:
        return {"score": None, "details": {"status": "missing"}}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"score": None, "details": {"status": "open_failed"}}

    sway = []
    frames = 0
    sampled = 0
    face_frames = 0
    low_light_frames = 0
    blurry_frames = 0
    brightness_vals = []
    blur_vals = []
    max_frames = VIDEO_MAX_FRAMES

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frames += 1
            if frames % VIDEO_FRAME_STRIDE != 0:
                continue
            if frames > max_frames:
                break

            sampled += 1
            try:
                enhanced, mean_v = _enhance_bgr(frame)
                gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
                blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                brightness_vals.append(mean_v)
                blur_vals.append(blur)
                if mean_v < LOW_LIGHT_THRESHOLD:
                    low_light_frames += 1
                if blur < BLUR_THRESHOLD:
                    blurry_frames += 1

                rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
                res = mp_face.process(rgb)
            except Exception:
                continue

            if res.multi_face_landmarks:
                face_frames += 1
                nose = res.multi_face_landmarks[0].landmark[1]
                sway.append(nose.x)
    finally:
        cap.release()

    if len(sway) < 5:
        return {
            "score": 0.4,
            "details": {
                "frames": frames,
                "sampled_frames": sampled,
                "face_frames": face_frames,
                "avg_brightness": round(float(np.mean(brightness_vals)), 2) if brightness_vals else 0.0,
                "avg_blur_var": round(float(np.mean(blur_vals)), 2) if blur_vals else 0.0,
                "low_light_frames": low_light_frames,
                "blurry_frames": blurry_frames,
            },
        }

    instability = np.std(sway) * 3
    if instability < MIN_SWAY_STD:
        return {
            "score": 0.3,
            "details": {
                "frames": frames,
                "sampled_frames": sampled,
                "face_frames": face_frames,
                "sway_std": float(np.std(sway)),
            },
        }

    score = 1 - min(instability, 1)
    face_rate = (face_frames / sampled) if sampled else 0.0
    return {
        "score": round(max(0, score), 2),
        "details": {
            "frames": frames,
            "sampled_frames": sampled,
            "face_frames": face_frames,
            "face_rate": round(face_rate, 3),
            "sway_std": float(np.std(sway)),
            "avg_brightness": round(float(np.mean(brightness_vals)), 2) if brightness_vals else 0.0,
            "avg_blur_var": round(float(np.mean(blur_vals)), 2) if blur_vals else 0.0,
            "low_light_frames": low_light_frames,
            "blurry_frames": blurry_frames,
        },
    }
