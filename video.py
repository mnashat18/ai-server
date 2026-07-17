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


LOW_LIGHT_THRESHOLD = 75.0
BLUR_THRESHOLD = 65.0
LAPLACIAN_NORMALIZER = 120.0
MIN_DURATION_SEC = 1.5
MIN_RESOLUTION = (480, 360)
MAX_SAMPLED_FRAMES = 8
COVERAGE_SAMPLE_COUNT = 4
BURST_SAMPLE_COUNT = 4
MIN_EYE_CLOSURE_DURATION_SECONDS = 0.3
MAX_EYE_CLOSURE_WINDOW_SECONDS = 1.2
DEFAULT_EYE_CLOSURE_WINDOW_SPAN_SECONDS = 0.6
MIN_EYE_EVIDENCE_FRAMES = 3
SUSTAINED_EYE_CLOSURE_EAR = 0.16
SUSTAINED_EYE_CLOSURE_STRONG_EAR = 0.14
MAX_EYE_ASYMMETRY = 0.08
MIN_USABLE_FRAME_COUNT = 3

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


def _coerce_non_negative_int(value: object) -> int | None:
    numeric = _coerce_finite_number(value)
    if numeric is None or numeric < 0.0 or not numeric.is_integer():
        return None
    return int(numeric)


def _coerce_positive_int(value: object) -> int | None:
    numeric = _coerce_finite_number(value)
    if numeric is None or numeric <= 0.0 or not numeric.is_integer():
        return None
    return int(numeric)


def _coerce_normalized_number(value: object) -> float | None:
    numeric = _coerce_finite_number(value)
    if numeric is None or numeric < 0.0 or numeric > 1.0:
        return None
    return numeric


def _frame_is_valid(frame: object) -> bool:
    if not isinstance(frame, np.ndarray) or frame.size == 0:
        return False
    if frame.dtype.kind in {"b", "O", "c"}:
        return False
    if frame.dtype.kind not in {"u", "i", "f"}:
        return False
    if frame.dtype.kind == "f":
        try:
            if not np.isfinite(frame).all():
                return False
        except Exception:
            return False
    if frame.ndim == 2:
        return True
    if frame.ndim == 3 and frame.shape[2] in {1, 3, 4}:
        return frame.shape[0] > 0 and frame.shape[1] > 0
    return False


def _frame_to_gray(frame: np.ndarray) -> np.ndarray:
    if cv2 is None:
        raise ValueError("cv2 unavailable")
    if frame.ndim == 2:
        return frame
    channels = frame.shape[2]
    if channels == 1:
        return frame[:, :, 0]
    if channels == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if channels == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    raise ValueError("unsupported frame shape")


def _frame_to_rgb(frame: np.ndarray) -> np.ndarray:
    if cv2 is None:
        raise ValueError("cv2 unavailable")
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    channels = frame.shape[2]
    if channels == 1:
        return cv2.cvtColor(frame[:, :, 0], cv2.COLOR_GRAY2RGB)
    if channels == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if channels == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
    raise ValueError("unsupported frame shape")


def _frame_shape(frame: np.ndarray) -> tuple[int, int]:
    height, width = frame.shape[:2]
    if _coerce_positive_int(width) is None or _coerce_positive_int(height) is None:
        raise ValueError("invalid frame dimensions")
    return int(width), int(height)


def _resolution_score(width: int, height: int) -> float:
    if width >= 1280 and height >= 720:
        return 1.0
    if width >= MIN_RESOLUTION[0] and height >= MIN_RESOLUTION[1]:
        return 0.7
    if width > 0 and height > 0:
        return 0.4
    return 0.0


def _brightness_score_from_mean(mean_v: float) -> float:
    score = 1.0 - (abs(mean_v - 128.0) / 128.0)
    return clamp01(score, 0.0) or 0.0


def _sharpness_score_from_blur_var(blur_var: float) -> float:
    score = blur_var / (blur_var + LAPLACIAN_NORMALIZER)
    return clamp01(score, 0.0) or 0.0


def _blur_warning_threshold() -> float:
    return BLUR_THRESHOLD / (BLUR_THRESHOLD + LAPLACIAN_NORMALIZER)


def _dist(a, b) -> float:
    return float(math.hypot(float(a.x) - float(b.x), float(a.y) - float(b.y)))


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


def _face_mesh_factory():
    if mp is None:
        return None
    solutions = getattr(mp, "solutions", None)
    face_mesh = getattr(solutions, "face_mesh", None) if solutions is not None else None
    return getattr(face_mesh, "FaceMesh", None) if face_mesh is not None else None


def _close_detector(detector: object) -> None:
    close = getattr(detector, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


def _release_capture(capture: object) -> None:
    release = getattr(capture, "release", None)
    if callable(release):
        try:
            release()
        except Exception:
            pass


def _build_sample_plan(frame_count: int | None, fps: float | None) -> list[dict[str, int | str | None]]:
    plan: list[dict[str, int | str | None]] = []
    if frame_count is not None and frame_count > 0:
        if frame_count <= COVERAGE_SAMPLE_COUNT:
            coverage_indices = list(range(frame_count))
        else:
            coverage_indices = [
                0,
                int(round((frame_count - 1) / 3.0)),
                int(round((2 * (frame_count - 1)) / 3.0)),
                frame_count - 1,
            ]
        if frame_count < BURST_SAMPLE_COUNT:
            burst_indices = list(range(frame_count))
        else:
            burst_span = int(round(DEFAULT_EYE_CLOSURE_WINDOW_SPAN_SECONDS * (fps if fps and fps > 0 else 30.0)))
            burst_span = max(burst_span, BURST_SAMPLE_COUNT - 1)
            burst_span = min(burst_span, frame_count - 1)
            burst_span = max(burst_span, BURST_SAMPLE_COUNT - 1)
            burst_start = max(0, min(frame_count - 1 - burst_span, (frame_count - 1 - burst_span) // 2))
            burst_indices = [
                burst_start,
                burst_start + int(round(burst_span / 3.0)),
                burst_start + int(round((2.0 * burst_span) / 3.0)),
                burst_start + burst_span,
            ]
            burst_indices = [max(0, min(frame_count - 1, index)) for index in burst_indices]
    else:
        coverage_indices = list(range(COVERAGE_SAMPLE_COUNT))
        burst_indices = list(range(COVERAGE_SAMPLE_COUNT, MAX_SAMPLED_FRAMES))

    seen: set[int] = set()
    for index in coverage_indices:
        if index in seen:
            continue
        seen.add(index)
        plan.append({"frame_index": index, "role": "coverage", "burst_id": None})
    for index in burst_indices:
        if index in seen:
            continue
        seen.add(index)
        plan.append({"frame_index": index, "role": "burst", "burst_id": 0})
    return plan[:MAX_SAMPLED_FRAMES]


def _eye_closure_sample_windows(
    *,
    frame_count: int | None,
    fps: float | None,
    duration_seconds: float | None,
) -> list[list[dict[str, int | float | bool | None]]]:
    frame_total = _coerce_non_negative_int(frame_count)
    fps_value = _coerce_finite_number(fps)
    duration_value = _coerce_finite_number(duration_seconds)
    if frame_total is None or frame_total <= 0 or fps_value is None or fps_value <= 0.0:
        return []
    if duration_value is None or duration_value < MIN_EYE_CLOSURE_DURATION_SECONDS:
        return []

    span = int(round(DEFAULT_EYE_CLOSURE_WINDOW_SPAN_SECONDS * fps_value))
    span = max(span, BURST_SAMPLE_COUNT - 1)
    span = min(span, max(frame_total - 1, BURST_SAMPLE_COUNT - 1))
    starts = [0]
    if frame_total > (span * 2):
        starts.append(max(0, frame_total - 1 - span))

    windows: list[list[dict[str, int | float | bool | None]]] = []
    for window_id, start in enumerate(starts):
        candidate_indices = [
            start,
            start + int(round(span / 3.0)),
            start + int(round((2.0 * span) / 3.0)),
            start + span,
        ]
        unique_indices: list[int] = []
        seen_indices: set[int] = set()
        for index in candidate_indices:
            bounded = max(0, min(frame_total - 1, index))
            if bounded in seen_indices:
                continue
            seen_indices.add(bounded)
            unique_indices.append(bounded)
        if len(unique_indices) != BURST_SAMPLE_COUNT:
            continue
        window = [
            {
                "window_id": window_id,
                "frame_index": index,
                "timestamp": index / float(fps_value),
            }
            for index in unique_indices
        ]
        windows.append(window)
    return windows


def _verified_capture_timestamp(
    *,
    frame_index: int,
    fps: float | None,
    capture_timestamp_ms: float | None,
    previous_timestamp: float | None,
) -> float | None:
    fps_value = _coerce_finite_number(fps)
    if fps_value is not None and fps_value > 0.0:
        return frame_index / fps_value
    timestamp_ms = _coerce_finite_number(capture_timestamp_ms)
    if timestamp_ms is None or timestamp_ms <= 0.0:
        return None
    timestamp = timestamp_ms / 1000.0
    if previous_timestamp is not None:
        gap = timestamp - previous_timestamp
        if gap <= 0.0 or gap > 0.25:
            return None
    return timestamp


def _motion_stability_from_diffs(diffs: list[float]) -> tuple[float | None, float | None, float | None]:
    if not diffs:
        return None, None, None
    motion_magnitude = float(np.mean(diffs))
    motion_variability = float(np.std(diffs))
    motion_penalty = min((motion_magnitude / 24.0) * 0.7 + (motion_variability / 18.0) * 0.3, 1.0)
    motion_stability_score = float(clamp01(1.0 - motion_penalty, 0.0) or 0.0)
    return motion_stability_score, motion_magnitude, motion_variability


def _select_decoded_dimensions(
    frame_dimensions: list[tuple[int, int]],
    metadata_width: int | None,
    metadata_height: int | None,
) -> tuple[int | None, int | None]:
    valid_dimensions = [
        (width, height)
        for width, height in frame_dimensions
        if _coerce_positive_int(width) is not None and _coerce_positive_int(height) is not None
    ]
    if valid_dimensions:
        widths = [width for width, _ in valid_dimensions]
        heights = [height for _, height in valid_dimensions]
        return min(widths), min(heights)
    if metadata_width is not None and metadata_height is not None:
        return metadata_width, metadata_height
    return None, None


def _derive_duration_seconds(
    *,
    fps: float | None,
    frame_count: int | None,
    observed_timestamps: list[float],
    decoded_frame_indices: list[int],
) -> float | None:
    fps_value = _coerce_finite_number(fps)
    frame_total = _coerce_non_negative_int(frame_count)
    valid_indices = [index for index in decoded_frame_indices if _coerce_non_negative_int(index) is not None]
    if (
        fps_value is not None
        and fps_value > 0.0
        and frame_total is not None
        and frame_total > 0
        and valid_indices
        and max(valid_indices) < frame_total
    ):
        duration_seconds = frame_total / fps_value
        if math.isfinite(duration_seconds) and duration_seconds > 0.0:
            return duration_seconds

    valid_timestamps = [timestamp for timestamp in observed_timestamps if _coerce_finite_number(timestamp) is not None]
    if len(set(valid_timestamps)) >= 2:
        start = min(valid_timestamps)
        end = max(valid_timestamps)
        span = end - start
        if math.isfinite(span) and span > 0.0:
            return span

    if fps_value is not None and fps_value > 0.0:
        unique_indices = sorted(set(valid_indices))
        if len(unique_indices) >= 2:
            frame_span = unique_indices[-1] - unique_indices[0]
            if frame_span > 0:
                duration_seconds = frame_span / fps_value
                if math.isfinite(duration_seconds) and duration_seconds > 0.0:
                    return duration_seconds
    return None


def _eye_landmark_reliability(
    *,
    detector_present: bool,
    valid_eye_samples: int,
    burst_observation_count: int,
    face_visibility: float | None,
    motion_stability_score: float | None,
    sharpness_score: float,
    brightness_score: float,
) -> bool:
    return bool(
        detector_present
        and valid_eye_samples >= MIN_EYE_EVIDENCE_FRAMES
        and burst_observation_count >= BURST_SAMPLE_COUNT
        and face_visibility is not None
        and face_visibility >= 0.65
        and motion_stability_score is not None
        and math.isfinite(float(motion_stability_score))
        and motion_stability_score >= 0.45
        and sharpness_score >= 0.4
        and brightness_score >= 0.4
    )


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
        if len(window_samples) < BURST_SAMPLE_COUNT:
            continue
        unique: list[dict[str, float | bool | int | None]] = []
        seen_indices: set[int] = set()
        previous_timestamp: float | None = None
        previous_index: int | None = None
        chronology_valid = True
        for sample in window_samples:
            frame_index = _coerce_non_negative_int(sample.get("frame_index"))
            timestamp = _coerce_finite_number(sample.get("timestamp"))
            if frame_index is None or timestamp is None:
                chronology_valid = False
                break
            if previous_index is not None and frame_index <= previous_index:
                chronology_valid = False
                break
            if previous_timestamp is not None:
                gap = timestamp - previous_timestamp
                if gap <= 0.0 or gap > 0.25:
                    chronology_valid = False
                    break
            if frame_index in seen_indices:
                chronology_valid = False
                break
            seen_indices.add(frame_index)
            unique.append(sample)
            previous_timestamp = timestamp
            previous_index = frame_index
        if not chronology_valid or len(unique) < BURST_SAMPLE_COUNT:
            continue
        all_closed = all(
            bool(sample.get("usable"))
            and bool(sample.get("bright_enough"))
            and bool(sample.get("sharp_enough"))
            and bool(sample.get("face_visible"))
            and bool(sample.get("landmark_valid"))
            and bool(sample.get("eye_closed"))
            for sample in unique
        )
        if not all_closed:
            continue
        start_timestamp = _coerce_finite_number(unique[0].get("timestamp"))
        end_timestamp = _coerce_finite_number(unique[-1].get("timestamp"))
        if start_timestamp is None or end_timestamp is None:
            continue
        window_seconds = end_timestamp - start_timestamp
        if window_seconds < MIN_EYE_CLOSURE_DURATION_SECONDS or window_seconds > MAX_EYE_CLOSURE_WINDOW_SECONDS:
            continue
        if len(unique) > longest:
            longest = len(unique)
            best_window_seconds = window_seconds
            best_window_ms = int(round(window_seconds * 1000))
    return longest, best_window_ms, best_window_seconds


def _missing_result(status: str) -> dict:
    return {
        "score": None,
        "details": {
            "status": status,
            "visual_warnings": ["video_missing"],
        },
    }


def _read_frame_metadata(capture: object) -> dict[str, float | int | None]:
    if cv2 is None:
        return {"fps": None, "frame_count": None, "width": None, "height": None}
    try:
        fps = _coerce_finite_number(capture.get(cv2.CAP_PROP_FPS))
        frame_count = _coerce_non_negative_int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = _coerce_positive_int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = _coerce_positive_int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    except Exception:
        raise
    return {"fps": fps, "frame_count": frame_count, "width": width, "height": height}


def analyze_video(video_path: str | None) -> dict:
    if not isinstance(video_path, str):
        return _missing_result("missing")

    normalized_path = video_path.strip()
    if not normalized_path:
        return _missing_result("missing")

    if cv2 is None:
        return _missing_result("open_failed")

    try:
        if (
            not os.path.exists(normalized_path)
            or os.path.isdir(normalized_path)
            or not os.path.isfile(normalized_path)
            or os.path.getsize(normalized_path) <= 0
        ):
            return _missing_result("open_failed")
    except Exception:
        return _missing_result("open_failed")

    capture = None
    detector = None
    try:
        capture = cv2.VideoCapture(normalized_path)
    except Exception:
        return _missing_result("open_failed")

    try:
        if not capture or not capture.isOpened():
            return _missing_result("open_failed")

        try:
            metadata = _read_frame_metadata(capture)
        except Exception:
            metadata = {"fps": None, "frame_count": None, "width": None, "height": None}

        fps = metadata["fps"]
        metadata_frame_count = metadata["frame_count"]
        width = metadata["width"]
        height = metadata["height"]
        plan = _build_sample_plan(metadata_frame_count, fps)
        detector_failed = False

        detector_factory = _face_mesh_factory()
        if detector_factory is not None:
            try:
                detector = detector_factory(
                    static_image_mode=False,
                    max_num_faces=1,
                    refine_landmarks=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
            except Exception:
                detector = None
                detector_failed = True

        sampled_frames = 0
        usable_frames = 0
        face_frames = 0
        low_light_frames = 0
        blurry_frames = 0
        brightness_values: list[float] = []
        blur_values: list[float] = []
        frame_dimensions: list[tuple[int, int]] = []
        burst_motion_diffs: list[float] = []
        burst_observations: list[dict[str, float | bool | int | None]] = []
        landmark_confidences: list[float] = []
        eye_apertures: list[float] = []
        eye_asymmetries: list[float] = []
        eye_closed_flags: list[bool] = []
        observed_timestamps: list[float] = []
        decoded_frame_indices: list[int] = []
        burst_last_gray: np.ndarray | None = None
        burst_last_index: int | None = None
        burst_last_timestamp: float | None = None
        burst_last_id: int | None = None

        for sample in plan:
            frame_index = int(sample["frame_index"])
            burst_id = sample.get("burst_id")
            try:
                if not capture.isOpened():
                    break
                try:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                except Exception:
                    pass
                ok, frame = capture.read()
            except Exception:
                ok, frame = False, None
            if not ok or not _frame_is_valid(frame):
                if sample.get("role") == "burst":
                    burst_last_gray = None
                    burst_last_index = None
                    burst_last_timestamp = None
                    burst_last_id = burst_id if isinstance(burst_id, int) else None
                continue

            try:
                gray = _frame_to_gray(frame)
                rgb = _frame_to_rgb(frame)
                frame_width, frame_height = _frame_shape(frame)
            except Exception:
                if sample.get("role") == "burst":
                    burst_last_gray = None
                    burst_last_index = None
                    burst_last_id = burst_id if isinstance(burst_id, int) else None
                continue

            sampled_frames += 1
            decoded_frame_indices.append(frame_index)
            frame_dimensions.append((frame_width, frame_height))
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

            if sample.get("role") == "burst":
                if burst_last_id != burst_id:
                    burst_last_gray = None
                    burst_last_index = None
                    burst_last_timestamp = None
                capture_timestamp_ms = None
                try:
                    capture_timestamp_ms = _coerce_finite_number(capture.get(cv2.CAP_PROP_POS_MSEC))
                except Exception:
                    capture_timestamp_ms = None
                current_timestamp = _verified_capture_timestamp(
                    frame_index=frame_index,
                    fps=fps,
                    capture_timestamp_ms=capture_timestamp_ms,
                    previous_timestamp=burst_last_timestamp,
                )
                if burst_last_gray is not None and burst_last_index is not None:
                    previous_timestamp = burst_last_timestamp
                    if (
                        current_timestamp is not None
                        and previous_timestamp is not None
                        and frame_index > burst_last_index
                        and current_timestamp > previous_timestamp
                        and current_timestamp - previous_timestamp <= 0.25
                    ):
                        diff = cv2.absdiff(burst_last_gray, gray)
                        burst_motion_diffs.append(float(np.mean(diff)))
                    else:
                        burst_last_gray = None
                        burst_last_index = None
                        burst_last_timestamp = None
                burst_last_gray = gray
                burst_last_index = frame_index
                burst_last_timestamp = current_timestamp
                burst_last_id = burst_id if isinstance(burst_id, int) else 0
            else:
                burst_last_gray = None
                burst_last_index = None
                burst_last_timestamp = None
                burst_last_id = None

            face_visible = False
            landmark_valid = False
            eye_closed = False
            landmark_confidence = None
            if detector is not None:
                try:
                    res = detector.process(rgb)
                    if getattr(res, "multi_face_landmarks", None):
                        face_visible = True
                        face_frames += 1
                        landmark_confidence = _coerce_normalized_number(
                            getattr(res, "landmark_detection_confidence", None)
                        )
                        if landmark_confidence is None:
                            landmark_confidence = _coerce_normalized_number(
                                getattr(res, "face_landmark_detection_confidence", None)
                            )
                        if landmark_confidence is not None:
                            landmark_confidences.append(landmark_confidence)

                        landmarks = res.multi_face_landmarks[0].landmark
                        left_eye = _eye_aspect_ratio(landmarks, LEFT_EYE)
                        right_eye = _eye_aspect_ratio(landmarks, RIGHT_EYE)
                        if left_eye is not None and right_eye is not None:
                            avg_ear = (left_eye + right_eye) / 2.0
                            asymmetry = abs(left_eye - right_eye)
                            eye_closed = avg_ear <= SUSTAINED_EYE_CLOSURE_EAR
                            landmark_valid = True
                            if sample.get("role") == "burst":
                                eye_apertures.append(avg_ear)
                                eye_asymmetries.append(asymmetry)
                                eye_closed_flags.append(eye_closed)
                    elif detector_factory is not None:
                        # Distinguish "no face" from detector failure.
                        pass
                except Exception:
                    detector_failed = True

            usable = bool(bright_enough and sharp_enough and (frame_width > 0 and frame_height > 0))
            if usable:
                usable_frames += 1
            capture_timestamp_ms = None
            try:
                capture_timestamp_ms = _coerce_finite_number(capture.get(cv2.CAP_PROP_POS_MSEC))
            except Exception:
                capture_timestamp_ms = None
            verified_timestamp = _verified_capture_timestamp(
                frame_index=frame_index,
                fps=fps,
                capture_timestamp_ms=capture_timestamp_ms,
                previous_timestamp=observed_timestamps[-1] if observed_timestamps else None,
            )
            if verified_timestamp is not None:
                observed_timestamps.append(verified_timestamp)

            if sample.get("role") == "burst":
                burst_observation: dict[str, float | bool | int | None] | None = None
                if verified_timestamp is not None and landmark_valid:
                    burst_observation = {
                        "window_id": int(burst_id or 0),
                        "frame_index": frame_index,
                        "timestamp": verified_timestamp,
                        "usable": usable,
                        "bright_enough": bright_enough,
                        "sharp_enough": sharp_enough,
                        "face_visible": face_visible,
                        "landmark_valid": landmark_valid,
                        "eye_closed": eye_closed,
                    }
                if burst_observation is not None:
                    burst_observations.append(burst_observation)

        if sampled_frames == 0:
            return _missing_result("open_failed")

        selected_width, selected_height = _select_decoded_dimensions(frame_dimensions, width, height)
        if selected_width is None or selected_height is None:
            return _missing_result("open_failed")
        width, height = selected_width, selected_height
        reliable_frame_count = None
        if (
            metadata_frame_count is not None
            and metadata_frame_count > 0
            and decoded_frame_indices
            and max(decoded_frame_indices) < metadata_frame_count
        ):
            reliable_frame_count = metadata_frame_count

        avg_brightness = float(np.mean(brightness_values)) if brightness_values else None
        avg_blur_var = float(np.mean(blur_values)) if blur_values else None
        if avg_brightness is None or avg_blur_var is None:
            return _missing_result("open_failed")

        brightness_score = _brightness_score_from_mean(avg_brightness)
        sharpness_score = _sharpness_score_from_blur_var(avg_blur_var)
        blur_score = 1.0 - sharpness_score
        resolution_score = _resolution_score(int(width), int(height))
        usable_frame_ratio = usable_frames / float(sampled_frames)
        coverage_ratio = sampled_frames / float(len(plan) if plan else sampled_frames)
        low_light_ratio = low_light_frames / float(sampled_frames)
        blurry_ratio = blurry_frames / float(sampled_frames)

        motion_stability_score, motion_magnitude, motion_variability = _motion_stability_from_diffs(burst_motion_diffs)
        motion_std = motion_variability

        duration_seconds = _derive_duration_seconds(
            fps=fps,
            frame_count=reliable_frame_count,
            observed_timestamps=observed_timestamps,
            decoded_frame_indices=decoded_frame_indices,
        )
        if duration_seconds is None or duration_seconds < 0.0:
            return _missing_result("open_failed")

        face_visibility = (face_frames / float(sampled_frames)) if detector is not None else None
        if detector is not None and face_frames == 0 and not detector_failed:
            warnings_base = ["subject_not_visible", "face_not_visible"]
        else:
            warnings_base = []
        if detector_failed:
            warnings_base.append("landmark_detection_failed")
        if sampled_frames < MIN_USABLE_FRAME_COUNT or usable_frame_ratio < 0.3:
            warnings_base.append("insufficient_usable_frames")
        if duration_seconds < MIN_DURATION_SEC:
            warnings_base.append("video_too_short")
        if int(width) < MIN_RESOLUTION[0] or int(height) < MIN_RESOLUTION[1]:
            warnings_base.append("video_low_resolution")
        if avg_brightness < LOW_LIGHT_THRESHOLD:
            warnings_base.append("video_too_dark")
        if avg_blur_var < BLUR_THRESHOLD:
            warnings_base.append("video_blurry")
        if motion_stability_score is not None and motion_stability_score < 0.4:
            warnings_base.append("unstable_camera")

        # Eye evidence only comes from valid, contiguous burst samples.
        valid_closed_flags = [flag for flag in eye_closed_flags]
        valid_eye_samples = len(eye_apertures)
        closed_eye_ratio = (sum(1 for flag in valid_closed_flags if flag) / float(valid_eye_samples)) if valid_eye_samples else None
        avg_eye_aperture = float(np.mean(eye_apertures)) if eye_apertures else None
        eye_aperture_std = float(np.std(eye_apertures)) if eye_apertures else None
        avg_eye_asymmetry = float(np.mean(eye_asymmetries)) if eye_asymmetries else None
        longest_eye_closure_streak, eye_closure_window_ms, eye_closure_window_seconds = _longest_temporal_eye_closure_streak(
            burst_observations
        )
        reliable_eye_landmarks = _eye_landmark_reliability(
            detector_present=detector is not None,
            valid_eye_samples=valid_eye_samples,
            burst_observation_count=len(burst_observations),
            face_visibility=face_visibility,
            motion_stability_score=motion_stability_score,
            sharpness_score=sharpness_score,
            brightness_score=brightness_score,
        )
        sustained_eye_closure = bool(
            reliable_eye_landmarks
            and closed_eye_ratio is not None
            and closed_eye_ratio >= 0.75
            and longest_eye_closure_streak >= BURST_SAMPLE_COUNT
            and avg_eye_aperture is not None
            and avg_eye_aperture <= SUSTAINED_EYE_CLOSURE_STRONG_EAR
            and eye_aperture_std is not None
            and eye_aperture_std <= 0.025
            and eye_closure_window_seconds >= MIN_EYE_CLOSURE_DURATION_SECONDS
            and eye_closure_window_seconds <= MAX_EYE_CLOSURE_WINDOW_SECONDS
        )
        if sustained_eye_closure:
            warnings_base.append("sustained_eye_closure")

        warnings = clean_warning_codes(warnings_base)
        if not warnings and detector is None and not detector_failed:
            warnings = clean_warning_codes([])

        visual_quality_score = float(
            clamp01(
                0.25 * brightness_score
                + 0.25 * sharpness_score
                + 0.20 * (motion_stability_score if motion_stability_score is not None else 0.5)
                + 0.15 * usable_frame_ratio
                + 0.15 * resolution_score,
                0.0,
            )
            or 0.0
        )
        visual_confidence = float(
            clamp01(
                0.60 * visual_quality_score
                + 0.20 * coverage_ratio
                + 0.20 * (face_visibility if face_visibility is not None else 0.0),
                0.0,
            )
            or 0.0
        )

        details = {
            "status": "ok",
            "frame_count": int(reliable_frame_count) if reliable_frame_count is not None else None,
            "duration_seconds": safe_number(duration_seconds, 3),
            "fps": safe_number(fps, 3),
            "resolution": {"width": int(width), "height": int(height)},
            "brightness_score": safe_number(brightness_score),
            "blur_score": safe_number(blur_score),
            "sharpness_score": safe_number(sharpness_score),
            "motion_stability_score": safe_number(motion_stability_score),
            "usable_frame_ratio": safe_number(usable_frame_ratio),
            "face_or_subject_visibility": safe_number(face_visibility),
            "landmark_detection_confidence": safe_number(
                float(np.mean(landmark_confidences)) if landmark_confidences else None
            ),
            "reliable_eye_landmarks": reliable_eye_landmarks,
            "sustained_eye_closure": sustained_eye_closure,
            "eye_closure_sample_count": valid_eye_samples,
            "closed_eye_ratio": safe_number(closed_eye_ratio),
            "longest_eye_closure_streak": longest_eye_closure_streak,
            "eye_closure_window_ms": eye_closure_window_ms,
            "eye_closure_window_seconds": safe_number(eye_closure_window_seconds, 3),
            "avg_eye_aperture": safe_number(avg_eye_aperture),
            "eye_aperture_std": safe_number(eye_aperture_std),
            "avg_eye_asymmetry": safe_number(avg_eye_asymmetry),
            "visual_confidence": safe_number(visual_confidence),
            "visual_quality_score": safe_number(visual_quality_score),
            "visual_warnings": warnings,
            "frames": int(reliable_frame_count) if reliable_frame_count is not None else None,
            "sampled_frames": sampled_frames,
            "face_frames": face_frames,
            "face_rate": safe_number(face_visibility),
            "sway_std": safe_number(motion_std),
            "avg_brightness": safe_number(avg_brightness, 2),
            "avg_blur_var": safe_number(avg_blur_var, 2),
            "low_light_frames": low_light_frames,
            "blurry_frames": blurry_frames,
            "low_light_rate": safe_number(low_light_ratio),
            "blurry_rate": safe_number(blurry_ratio),
            "quality_flags": warnings,
        }
        return {"score": details["visual_confidence"], "details": details}
    finally:
        _close_detector(detector)
        _release_capture(capture)
