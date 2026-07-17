from __future__ import annotations

import math
import os

os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")

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


LOW_LIGHT_MEAN_THRESHOLD = 75.0
SHARPNESS_WARNING_THRESHOLD = 0.35
MIN_DIMENSION = 256
LAPLACIAN_NORMALIZER = 120.0
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


def _coerce_finite_number(value: object) -> float | None:
    if value is None or isinstance(value, bool) or isinstance(value, str):
        return None
    if not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        return None
    return numeric


def _coerce_normalized_number(value: object) -> float | None:
    numeric = _coerce_finite_number(value)
    if numeric is None:
        return None
    if numeric < 0.0 or numeric > 1.0:
        return None
    return numeric


def _coerce_positive_int(value: object) -> int | None:
    numeric = _coerce_finite_number(value)
    if numeric is None or numeric <= 0.0 or not numeric.is_integer():
        return None
    return int(numeric)


def _dist(a, b) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def _point_coordinates(point: object) -> tuple[float, float] | None:
    if point is None:
        return None
    try:
        x_value = getattr(point, "x", None)
        y_value = getattr(point, "y", None)
    except Exception:
        return None
    x = _coerce_finite_number(x_value)
    y = _coerce_finite_number(y_value)
    if x is None or y is None:
        return None
    return x, y


def _eye_aspect_ratio(landmarks, idxs) -> float | None:
    try:
        points = [landmarks[i] for i in idxs]
    except Exception:
        return None
    if len(points) != 6:
        return None
    coords = [_point_coordinates(point) for point in points]
    if any(coord is None for coord in coords):
        return None
    p0, p1, p2, p3, p4, p5 = coords
    horizontal = math.hypot(p0[0] - p3[0], p0[1] - p3[1])
    if not math.isfinite(horizontal) or horizontal <= 0.0:
        return None
    vertical = math.hypot(p1[0] - p5[0], p1[1] - p5[1]) + math.hypot(p2[0] - p4[0], p2[1] - p4[1])
    ear = vertical / (2.0 * horizontal)
    if not math.isfinite(ear) or ear < 0.0 or ear > 1.0:
        return None
    return ear


def _brightness_score(gray: np.ndarray) -> float:
    mean_v = _coerce_finite_number(float(np.mean(gray)))
    if mean_v is None:
        return 0.0
    # Mid-range exposure is best; both very dark and overexposed frames score poorly.
    score = 1.0 - (abs(mean_v - 128.0) / 128.0)
    return clamp01(score, 0.0) or 0.0


def _sharpness_score(gray: np.ndarray) -> float:
    if cv2 is None:
        return 0.0
    blur_var = _coerce_finite_number(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
    if blur_var is None:
        return 0.0
    # Normalize Laplacian variance into a bounded score instead of exposing the raw value.
    score = blur_var / (blur_var + LAPLACIAN_NORMALIZER)
    return clamp01(score, 0.0) or 0.0


def _resolution_score(width: int, height: int) -> float:
    return clamp01(min(width, height) / float(MIN_DIMENSION), 0.0) or 0.0


def _missing_result() -> dict:
    return {
        "score": None,
        "details": {
            "status": "missing",
            "image_warnings": ["image_missing"],
        },
    }


def _invalid_result() -> dict:
    return {
        "score": None,
        "details": {
            "status": "invalid_image",
            "image_warnings": ["image_missing"],
        },
    }


def _face_mesh_factory():
    if mp is None:
        return None
    solutions = getattr(mp, "solutions", None)
    face_mesh = getattr(solutions, "face_mesh", None) if solutions is not None else None
    factory = getattr(face_mesh, "FaceMesh", None) if face_mesh is not None else None
    return factory


def _landmark_detection_confidence(result: object) -> float | None:
    for attr in ("landmark_detection_confidence", "face_landmark_detection_confidence"):
        confidence = _coerce_normalized_number(getattr(result, attr, None))
        if confidence is not None:
            return confidence
    return None


def _close_detector(detector: object) -> None:
    close = getattr(detector, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


def _rgb_for_detection(image: np.ndarray) -> np.ndarray:
    if cv2 is None:
        raise ValueError("cv2 unavailable")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.ndim != 3:
        raise ValueError("unsupported image array")
    channels = image.shape[2]
    if channels == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if channels == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    raise ValueError("unsupported image array")


def _analyze_image_array(image: np.ndarray) -> dict:
    if cv2 is None:
        return _invalid_result()
    if not isinstance(image, np.ndarray) or image.size == 0:
        return _invalid_result()
    if image.dtype.kind in {"b", "O", "c"} or image.dtype.kind not in {"u", "i", "f"}:
        return _invalid_result()
    if image.dtype.kind == "f":
        try:
            if not np.isfinite(image).all():
                return _invalid_result()
        except Exception:
            return _invalid_result()

    try:
        if image.ndim == 2:
            gray = image
        elif image.ndim == 3 and image.shape[2] in {3, 4}:
            gray_code = cv2.COLOR_BGR2GRAY if image.shape[2] == 3 else cv2.COLOR_BGRA2GRAY
            gray = cv2.cvtColor(image, gray_code)
        else:
            return _invalid_result()
    except Exception:
        return _invalid_result()

    if not isinstance(gray, np.ndarray) or gray.size == 0:
        return _invalid_result()
    try:
        if gray.dtype.kind == "f" and not np.isfinite(gray).all():
            return _invalid_result()
    except Exception:
        return _invalid_result()

    try:
        height, width = image.shape[:2]
    except Exception:
        return _invalid_result()
    if _coerce_positive_int(width) is None or _coerce_positive_int(height) is None:
        return _invalid_result()

    try:
        brightness_score = _brightness_score(gray)
        sharpness_score = _sharpness_score(gray)
        resolution_score = _resolution_score(int(width), int(height))
        blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        avg_brightness = float(np.mean(gray))
    except Exception:
        return _invalid_result()

    warnings: list[str] = []
    if width < MIN_DIMENSION or height < MIN_DIMENSION:
        warnings.append("image_low_resolution")
    low_light = avg_brightness < LOW_LIGHT_MEAN_THRESHOLD
    if low_light:
        warnings.append("image_too_dark")
    if sharpness_score < SHARPNESS_WARNING_THRESHOLD:
        warnings.append("image_blurry")

    face_detected: bool | None = None
    subject_visibility: float | None = None
    landmark_detection_confidence: float | None = None
    avg_ear: float | None = None
    left_eye_aperture: float | None = None
    right_eye_aperture: float | None = None
    left_right_eye_asymmetry: float | None = None
    eyes_closed: bool | None = None

    factory = _face_mesh_factory()
    if factory is not None:
        detector = None
        try:
            detector = factory(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=False,
                min_detection_confidence=0.5,
            )
            rgb = _rgb_for_detection(image)
            res = detector.process(rgb)
            if getattr(res, "multi_face_landmarks", None):
                face_detected = True
                subject_visibility = 1.0
                landmark_detection_confidence = _landmark_detection_confidence(res)
                landmarks = res.multi_face_landmarks[0].landmark
                left_eye_aperture = _eye_aspect_ratio(landmarks, LEFT_EYE)
                right_eye_aperture = _eye_aspect_ratio(landmarks, RIGHT_EYE)
                if left_eye_aperture is not None and right_eye_aperture is not None:
                    avg_ear = (left_eye_aperture + right_eye_aperture) / 2.0
                    left_right_eye_asymmetry = abs(left_eye_aperture - right_eye_aperture)
                    eyes_closed = avg_ear < 0.18
                else:
                    left_eye_aperture = None
                    right_eye_aperture = None
            else:
                face_detected = False
                warnings.extend(["subject_not_visible", "face_not_visible"])
        except Exception:
            face_detected = None
        finally:
            if detector is not None:
                _close_detector(detector)

    quality_components = [
        brightness_score,
        sharpness_score,
        resolution_score,
    ]
    image_quality_score = sum(quality_components) / len(quality_components)
    image_confidence = clamp01(image_quality_score, 0.0) or 0.0

    details = {
        "status": "ok",
        "resolution": {"width": int(width), "height": int(height)},
        "brightness_score": safe_number(brightness_score),
        "blur_score": safe_number(1.0 - sharpness_score),
        "sharpness_score": safe_number(sharpness_score),
        "subject_visibility": safe_number(subject_visibility),
        "image_quality_score": safe_number(image_quality_score),
        "image_confidence": safe_number(image_confidence),
        "image_warnings": clean_warning_codes(warnings),
        "face_detected": face_detected,
        "landmark_detection_confidence": safe_number(landmark_detection_confidence),
        "avg_brightness": safe_number(avg_brightness, 2),
        "blur_var": safe_number(blur_var, 2),
        "avg_ear": safe_number(avg_ear),
        "left_eye_aperture": safe_number(left_eye_aperture),
        "right_eye_aperture": safe_number(right_eye_aperture),
        "left_right_eye_asymmetry": safe_number(left_right_eye_asymmetry),
        "eyes_closed": eyes_closed,
        "low_light": low_light,
        "blurry": sharpness_score < SHARPNESS_WARNING_THRESHOLD,
    }
    return {"score": details["image_confidence"], "details": details}


def analyze_face(image_path: str | None) -> dict:
    if not isinstance(image_path, str):
        return _missing_result()

    normalized_path = image_path.strip()
    if not normalized_path:
        return _missing_result()

    if cv2 is None:
        return _invalid_result()

    try:
        if not os.path.exists(normalized_path) or os.path.isdir(normalized_path) or not os.path.isfile(normalized_path):
            return _invalid_result()
        if os.path.getsize(normalized_path) <= 0:
            return _invalid_result()
    except Exception:
        return _invalid_result()

    try:
        image = cv2.imread(normalized_path)
    except Exception:
        return _invalid_result()
    if image is None:
        return _invalid_result()
    return _analyze_image_array(image)
