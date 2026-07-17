from __future__ import annotations

import copy
import contextlib
import importlib.util
import math
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

import quality
import validation
import video


def _frame(value: int, *, width: int = 640, height: int = 480) -> np.ndarray:
    return np.full((height, width, 3), value, dtype=np.uint8)


def _checkerboard(size: int = 480) -> np.ndarray:
    pattern = (np.indices((size, size)).sum(axis=0) % 2).astype(np.uint8) * 255
    return np.dstack([pattern, pattern, pattern])


def _landmark(x: float, y: float) -> types.SimpleNamespace:
    return types.SimpleNamespace(x=x, y=y)


def _landmarks(*, left_ear: float = 0.28, right_ear: float = 0.28) -> list[types.SimpleNamespace]:
    points = [_landmark(0.0, 0.0) for _ in range(400)]

    def _eye(idxs: list[int], ear: float) -> None:
        p0, p1, p2, p3, p4, p5 = idxs
        points[p0] = _landmark(0.0, 0.0)
        points[p3] = _landmark(1.0, 0.0)
        points[p1] = _landmark(0.5, ear / 2.0)
        points[p5] = _landmark(0.5, -ear / 2.0)
        points[p2] = _landmark(0.5, ear / 2.0)
        points[p4] = _landmark(0.5, -ear / 2.0)

    _eye(video.LEFT_EYE, left_ear)
    _eye(video.RIGHT_EYE, right_ear)
    return points


def _invalid_landmark_points(mode: str) -> list[object]:
    points: list[object] = [_landmark(0.0, 0.0) for _ in range(400)]
    if mode == "zero_horizontal":
        for idx in video.LEFT_EYE + video.RIGHT_EYE:
            points[idx] = _landmark(0.0, 0.0)
        points[video.LEFT_EYE[1]] = _landmark(0.0, 0.1)
        points[video.LEFT_EYE[2]] = _landmark(0.0, 0.1)
    elif mode == "nan":
        points[video.LEFT_EYE[0]] = _landmark(float("nan"), 0.0)
    elif mode == "inf":
        points[video.LEFT_EYE[0]] = _landmark(float("inf"), 0.0)
    elif mode == "missing_indices":
        return [_landmark(0.0, 0.0) for _ in range(20)]
    elif mode == "malformed":
        points[video.LEFT_EYE[0]] = types.SimpleNamespace()
    return points


def _success_frames(*, value: int = 128, count: int = 120, width: int = 640, height: int = 480) -> list[np.ndarray]:
    return [_frame(value, width=width, height=height) for _ in range(count)]


def _good_frames(count: int = 120) -> list[np.ndarray]:
    return [_checkerboard() for _ in range(count)]


def _video_metadata(*, frame_count: int = 120, fps: float = 30.0, width: int = 640, height: int = 480) -> dict[int, object]:
    return {
        _FakeCV2.CAP_PROP_FPS: fps,
        _FakeCV2.CAP_PROP_FRAME_COUNT: frame_count,
        _FakeCV2.CAP_PROP_FRAME_WIDTH: width,
        _FakeCV2.CAP_PROP_FRAME_HEIGHT: height,
    }


def _moving_frames(count: int = 120) -> list[np.ndarray]:
    return [_frame((index * 37) % 255) for index in range(count)]


class _DetectorInstance:
    def __init__(
        self,
        *,
        has_face: bool,
        landmarks: list[object] | None = None,
        confidence: float | None = None,
        process_raises: bool = False,
        close_raises: bool = False,
    ) -> None:
        self.has_face = has_face
        self.landmarks = landmarks or _landmarks()
        self.confidence = confidence
        self.process_raises = process_raises
        self.close_raises = close_raises
        self.process_calls = 0
        self.close_calls = 0

    def process(self, rgb):
        self.process_calls += 1
        if self.process_raises:
            raise RuntimeError("process boom")
        if not self.has_face:
            return types.SimpleNamespace(multi_face_landmarks=None)
        result = types.SimpleNamespace(
            multi_face_landmarks=[types.SimpleNamespace(landmark=self.landmarks)],
        )
        if self.confidence is not None:
            result.landmark_detection_confidence = self.confidence
        return result

    def close(self):
        self.close_calls += 1
        if self.close_raises:
            raise RuntimeError("close boom")


class _DetectorFactory:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.instances: list[_DetectorInstance] = []

    def __call__(self, **kwargs):
        instance = _DetectorInstance(**self.kwargs)
        instance.kwargs = kwargs
        self.instances.append(instance)
        return instance


def _detector_result(*, has_face: bool = True, landmarks=None, confidence: float | None = None):
    if not has_face:
        return types.SimpleNamespace(multi_face_landmarks=None)
    result = types.SimpleNamespace(
        multi_face_landmarks=[types.SimpleNamespace(landmark=landmarks or _landmarks())],
    )
    if confidence is not None:
        result.landmark_detection_confidence = confidence
    return result


class _SequenceDetector:
    def __init__(self, sequence):
        self.sequence = list(sequence)
        self.process_calls = 0
        self.close_calls = 0

    def process(self, rgb):
        index = min(self.process_calls, len(self.sequence) - 1)
        self.process_calls += 1
        item = self.sequence[index]
        if isinstance(item, Exception):
            raise item
        return item

    def close(self):
        self.close_calls += 1


class _FakeCapture:
    def __init__(
        self,
        frames: list[np.ndarray | object],
        *,
        metadata: dict[int, object] | None = None,
        opened: bool = True,
        read_raises: bool = False,
        set_raises: bool = False,
        get_raises: set[int] | None = None,
        release_raises: bool = False,
    ) -> None:
        self.frames = frames
        self.metadata = metadata or {}
        self.opened = opened
        self.read_raises = read_raises
        self.set_raises = set_raises
        self.get_raises = set(get_raises or set())
        self.release_raises = release_raises
        self.position = 0
        self.read_calls = 0
        self.set_calls: list[tuple[int, int]] = []
        self.release_calls = 0

    def isOpened(self):
        return self.opened

    def get(self, prop):
        if prop in self.get_raises:
            raise RuntimeError("metadata boom")
        return self.metadata.get(prop, 0)

    def set(self, prop, value):
        self.set_calls.append((prop, int(value)))
        if self.set_raises:
            raise RuntimeError("seek boom")
        if prop == _FakeCV2.CAP_PROP_POS_FRAMES:
            self.position = int(value)
        return True

    def read(self):
        self.read_calls += 1
        if self.read_raises:
            raise RuntimeError("read boom")
        if self.position < 0 or self.position >= len(self.frames):
            return False, None
        frame = self.frames[self.position]
        self.position += 1
        return True, frame

    def release(self):
        self.release_calls += 1
        if self.release_raises:
            raise RuntimeError("release boom")


class _CaptureFactory:
    def __init__(self, capture):
        self.capture = capture
        self.instances: list[object] = []

    def __call__(self, path):
        if callable(self.capture) and not isinstance(self.capture, _FakeCapture):
            instance = self.capture(path)
        else:
            instance = self.capture
        self.instances.append(instance)
        return instance


class _LaplacianResult:
    def __init__(self, value: float):
        self._value = value

    def var(self):
        return self._value


class _FakeCV2:
    CAP_PROP_FPS = 5
    CAP_PROP_FRAME_COUNT = 7
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_POS_FRAMES = 1
    CAP_PROP_POS_MSEC = 2
    COLOR_BGR2GRAY = 10
    COLOR_BGRA2GRAY = 11
    COLOR_GRAY2RGB = 12
    COLOR_BGR2RGB = 13
    COLOR_BGRA2RGB = 14
    CV_64F = 0

    def __init__(self, capture_factory):
        self.capture_factory = capture_factory

    def VideoCapture(self, path):
        return self.capture_factory(path)

    def cvtColor(self, image, code):
        arr = np.asarray(image)
        if code == self.COLOR_BGR2GRAY:
            return np.mean(arr[..., :3], axis=2).astype(arr.dtype)
        if code == self.COLOR_BGRA2GRAY:
            return np.mean(arr[..., :4], axis=2).astype(arr.dtype)
        if code == self.COLOR_GRAY2RGB:
            gray = np.asarray(arr)
            return np.dstack([gray, gray, gray]).astype(arr.dtype)
        if code == self.COLOR_BGR2RGB:
            return arr[..., ::-1]
        if code == self.COLOR_BGRA2RGB:
            return arr[..., [2, 1, 0]]
        raise ValueError(code)

    def Laplacian(self, gray, dtype):
        arr = np.asarray(gray, dtype=float)
        if arr.size == 0:
            return _LaplacianResult(0.0)
        vertical = np.diff(arr, axis=0)
        horizontal = np.diff(arr, axis=1)
        score = float(np.var(vertical) + np.var(horizontal))
        return _LaplacianResult(score)

    def absdiff(self, a, b):
        return np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))


def _fake_mp(detector_factory):
    return types.SimpleNamespace(
        solutions=types.SimpleNamespace(
            face_mesh=types.SimpleNamespace(FaceMesh=detector_factory),
        )
    )


class _VideoPatch(contextlib.AbstractContextManager):
    def __init__(
        self,
        capture,
        *,
        detector_factory=None,
        path_exists: bool = True,
        is_dir: bool = False,
        is_file: bool = True,
        file_size: int = 1024,
        cv2_available: bool = True,
        cv2_module=None,
        mp_module=None,
    ) -> None:
        self.capture = capture
        self.detector_factory = detector_factory
        self.path_exists = path_exists
        self.is_dir = is_dir
        self.is_file = is_file
        self.file_size = file_size
        self.cv2_available = cv2_available
        self.cv2_module = cv2_module
        self.mp_module = mp_module
        self.stack = contextlib.ExitStack()

    def __enter__(self):
        self.stack.enter_context(patch.object(video.os.path, "exists", return_value=self.path_exists))
        self.stack.enter_context(patch.object(video.os.path, "isdir", return_value=self.is_dir))
        self.stack.enter_context(patch.object(video.os.path, "isfile", return_value=self.is_file))
        self.stack.enter_context(patch.object(video.os.path, "getsize", return_value=self.file_size))
        if self.cv2_module is not None:
            self.stack.enter_context(patch.object(video, "cv2", self.cv2_module))
        else:
            self.stack.enter_context(
                patch.object(video, "cv2", _FakeCV2(_CaptureFactory(self.capture)) if self.cv2_available else None)
            )
        if self.mp_module is not None:
            self.stack.enter_context(patch.object(video, "mp", self.mp_module))
        else:
            self.stack.enter_context(
                patch.object(
                    video,
                    "mp",
                    _fake_mp(self.detector_factory)
                    if self.detector_factory is not None
                    else types.SimpleNamespace(
                        solutions=types.SimpleNamespace(face_mesh=types.SimpleNamespace(FaceMesh=None))
                    ),
                )
            )
        return self

    def __exit__(self, exc_type, exc, tb):
        return self.stack.__exit__(exc_type, exc, tb)


class VideoUnitTests(unittest.TestCase):
    def _analyze(
        self,
        capture,
        *,
        detector_factory=None,
        path: str = "video.mp4",
        path_exists: bool = True,
        is_dir: bool = False,
        is_file: bool = True,
        file_size: int = 1024,
        cv2_available: bool = True,
        cv2_module=None,
        mp_module=None,
    ):
        with _VideoPatch(
            capture,
            detector_factory=detector_factory,
            path_exists=path_exists,
            is_dir=is_dir,
            is_file=is_file,
            file_size=file_size,
            cv2_available=cv2_available,
            cv2_module=cv2_module,
            mp_module=mp_module,
        ):
            return video.analyze_video(path)

    def test_input_safety_and_failure_contracts(self):
        cases = [
            (None, "missing", {}),
            (123, "missing", {}),
            ("   ", "missing", {}),
            ("missing.mp4", "open_failed", {"path_exists": False}),
            ("dir", "open_failed", {"is_dir": True, "is_file": False}),
            ("empty.mp4", "open_failed", {"file_size": 0}),
            ("nocv2.mp4", "open_failed", {"cv2_available": False}),
        ]
        for video_path, expected_status, kwargs in cases:
            with self.subTest(video_path=video_path, expected_status=expected_status):
                capture = _FakeCapture(_success_frames())
                result = self._analyze(capture, path=video_path, **kwargs)
                self.assertIsNone(result["score"])
                self.assertEqual(result["details"]["status"], expected_status)
                self.assertEqual(result["details"]["visual_warnings"], ["video_missing"])

    def test_capture_constructor_exception_returns_open_failed(self):
        class _RaisingCaptureFactory:
            def __call__(self, path):
                raise RuntimeError("ctor boom")

        result = self._analyze(
            _RaisingCaptureFactory(),
            cv2_module=_FakeCV2(_RaisingCaptureFactory()),
        )
        self.assertIsNone(result["score"])
        self.assertEqual(result["details"]["status"], "open_failed")

    def test_capture_not_opened_returns_open_failed(self):
        capture = _FakeCapture(_success_frames(), opened=False)
        result = self._analyze(capture)
        self.assertIsNone(result["score"])
        self.assertEqual(result["details"]["status"], "open_failed")

    def test_release_is_called_on_success(self):
        capture = _FakeCapture(_good_frames(), metadata=_video_metadata())
        result = self._analyze(capture, detector_factory=_DetectorFactory(has_face=True))
        self.assertEqual(result["details"]["status"], "ok")
        self.assertEqual(capture.release_calls, 1)

    def test_release_is_called_when_no_frame_is_decoded(self):
        capture = _FakeCapture([])
        result = self._analyze(capture, detector_factory=_DetectorFactory(has_face=True))
        self.assertIsNone(result["score"])
        self.assertEqual(result["details"]["status"], "open_failed")
        self.assertEqual(capture.release_calls, 1)

    def test_release_is_called_on_metadata_exception(self):
        capture = _FakeCapture(_good_frames(), metadata=_video_metadata(), get_raises={_FakeCV2.CAP_PROP_FPS})
        result = self._analyze(capture, detector_factory=_DetectorFactory(has_face=True))
        self.assertEqual(result["details"]["status"], "open_failed")
        self.assertEqual(capture.release_calls, 1)

    def test_release_is_called_on_frame_processing_exception(self):
        class _RaisingCV2(_FakeCV2):
            def cvtColor(self, image, code):
                raise RuntimeError("convert boom")

        capture = _FakeCapture(_success_frames())
        result = self._analyze(capture, cv2_module=_RaisingCV2(_CaptureFactory(capture)), detector_factory=None)
        self.assertIsNone(result["score"])
        self.assertEqual(result["details"]["status"], "open_failed")
        self.assertEqual(capture.release_calls, 1)

    def test_release_exception_is_suppressed(self):
        capture = _FakeCapture(_good_frames(), metadata=_video_metadata(), release_raises=True)
        result = self._analyze(capture, detector_factory=_DetectorFactory(has_face=True))
        self.assertEqual(result["details"]["status"], "ok")
        self.assertEqual(capture.release_calls, 1)

    def test_sampling_plan_is_deterministic_and_unique(self):
        plan = video._build_sample_plan(120, 30.0)
        self.assertEqual(plan, video._build_sample_plan(120, 30.0))
        indices = [item["frame_index"] for item in plan]
        self.assertEqual(len(indices), len(set(indices)))
        self.assertLessEqual(len(indices), video.MAX_SAMPLED_FRAMES)
        self.assertLessEqual(len(video._build_sample_plan(None, None)), video.MAX_SAMPLED_FRAMES)

    def test_sampling_cap_is_at_most_eight_and_short_clips_remain_short(self):
        capture = _FakeCapture(_success_frames(count=120), metadata=_video_metadata())
        result = self._analyze(capture, detector_factory=_DetectorFactory(has_face=True))
        self.assertEqual(result["details"]["status"], "ok")
        self.assertLessEqual(capture.read_calls, video.MAX_SAMPLED_FRAMES)
        self.assertLess(len(video._build_sample_plan(3, 30.0)), video.MAX_SAMPLED_FRAMES)
        self.assertLessEqual(len({item["frame_index"] for item in video._build_sample_plan(4, 30.0)}), 4)

    def test_eye_closure_sample_windows_are_planning_only(self):
        windows = video._eye_closure_sample_windows(frame_count=120, fps=30.0, duration_seconds=4.0)
        self.assertGreater(len(windows), 0)
        for window in windows:
            indices = [sample["frame_index"] for sample in window]
            self.assertEqual(len(indices), len(set(indices)))
            for sample in window:
                self.assertEqual(set(sample.keys()), {"window_id", "frame_index", "timestamp"})
                self.assertAlmostEqual(sample["timestamp"], sample["frame_index"] / 30.0)

    def test_eye_closure_sample_windows_reject_invalid_inputs(self):
        self.assertEqual(video._eye_closure_sample_windows(frame_count=8, fps=30.0, duration_seconds=0.2), [])
        self.assertEqual(video._eye_closure_sample_windows(frame_count=0, fps=30.0, duration_seconds=1.0), [])
        self.assertEqual(video._eye_closure_sample_windows(frame_count=120, fps=float("nan"), duration_seconds=1.0), [])

    def test_eye_closure_sample_windows_are_unique_and_bounded(self):
        windows = video._eye_closure_sample_windows(frame_count=120, fps=30.0, duration_seconds=4.0)
        self.assertLessEqual(sum(len(window) for window in windows), video.MAX_SAMPLED_FRAMES)
        self.assertLessEqual(len(windows), 2)
        for window in windows:
            self.assertLessEqual(len(window), 4)
            self.assertEqual(len({sample["frame_index"] for sample in window}), len(window))

    def test_invalid_and_nonfinite_metadata_do_not_crash(self):
        metadata = {
            _FakeCV2.CAP_PROP_FPS: float("nan"),
            _FakeCV2.CAP_PROP_FRAME_COUNT: True,
            _FakeCV2.CAP_PROP_FRAME_WIDTH: float("inf"),
            _FakeCV2.CAP_PROP_FRAME_HEIGHT: -1,
        }
        capture = _FakeCapture(_good_frames(), metadata=metadata)
        result = self._analyze(capture, detector_factory=_DetectorFactory(has_face=True))
        details = result["details"]
        self.assertEqual(details["status"], "open_failed")
        self.assertIsNone(result["score"])

    def test_zero_decoded_frames_returns_open_failed(self):
        capture = _FakeCapture([None] * 4)
        result = self._analyze(capture, detector_factory=_DetectorFactory(has_face=True))
        self.assertIsNone(result["score"])
        self.assertEqual(result["details"]["status"], "open_failed")

    def test_malformed_frames_are_skipped(self):
        frames = [
            np.array([[1]], dtype=bool),
            np.array([[1]], dtype=object),
            np.array([[1 + 2j]], dtype=complex),
            np.array([[float("nan")]], dtype=float),
            np.array([[float("inf")]], dtype=float),
        ]
        capture = _FakeCapture(frames)
        result = self._analyze(capture, detector_factory=_DetectorFactory(has_face=True))
        self.assertEqual(result["details"]["status"], "open_failed")
        self.assertIsNone(result["details"].get("visual_quality_score"))

    def test_success_satisfies_validation_and_quality_contracts(self):
        detector_factory = _DetectorFactory(has_face=True, landmarks=_landmarks())
        capture = _FakeCapture(_good_frames(count=120), metadata=_video_metadata())
        result = self._analyze(capture, detector_factory=detector_factory)
        details = result["details"]
        self.assertEqual(details["status"], "ok")
        self.assertTrue(math.isfinite(details["visual_quality_score"]))
        self.assertGreaterEqual(details["visual_quality_score"], 0.0)
        self.assertLessEqual(details["visual_quality_score"], 1.0)
        self.assertTrue(math.isfinite(details["visual_confidence"]))
        self.assertGreaterEqual(details["visual_confidence"], 0.0)
        self.assertLessEqual(details["visual_confidence"], 1.0)
        self.assertEqual(result["score"], details["visual_confidence"])
        self.assertGreater(details["duration_seconds"], 0.0)
        validated = validation.validate_video_result(validation.ValidationPolicy(), result)
        self.assertIn("video", validated["usable_modalities"])
        self.assertNotIn("video", validated["missing_modalities"])
        assessment = quality.assess_quality({"video": result}, speech_required=False)
        self.assertTrue(assessment["media_quality"]["video"]["present"])
        self.assertTrue(assessment["media_quality"]["video"]["usable"])

    def test_success_output_schema_is_complete(self):
        detector_factory = _DetectorFactory(has_face=True, landmarks=_landmarks())
        capture = _FakeCapture(_good_frames(count=120), metadata=_video_metadata())
        result = self._analyze(capture, detector_factory=detector_factory)
        expected_keys = {
            "status",
            "frame_count",
            "duration_seconds",
            "fps",
            "resolution",
            "brightness_score",
            "blur_score",
            "sharpness_score",
            "motion_stability_score",
            "usable_frame_ratio",
            "face_or_subject_visibility",
            "landmark_detection_confidence",
            "reliable_eye_landmarks",
            "sustained_eye_closure",
            "eye_closure_sample_count",
            "closed_eye_ratio",
            "longest_eye_closure_streak",
            "eye_closure_window_ms",
            "eye_closure_window_seconds",
            "avg_eye_aperture",
            "eye_aperture_std",
            "avg_eye_asymmetry",
            "visual_confidence",
            "visual_quality_score",
            "visual_warnings",
            "frames",
            "sampled_frames",
            "face_frames",
            "face_rate",
            "sway_std",
            "avg_brightness",
            "avg_blur_var",
            "low_light_frames",
            "blurry_frames",
            "low_light_rate",
            "blurry_rate",
            "quality_flags",
        }
        self.assertEqual(set(result["details"].keys()), expected_keys)

    def test_failure_output_schema_is_complete(self):
        capture = _FakeCapture([])
        result = self._analyze(capture, detector_factory=_DetectorFactory(has_face=True))
        self.assertEqual(set(result.keys()), {"score", "details"})
        self.assertEqual(result["details"]["visual_warnings"], ["video_missing"])
        self.assertIsNone(result["score"])

    def test_top_level_score_equals_visual_confidence(self):
        detector_factory = _DetectorFactory(has_face=True, landmarks=_landmarks())
        capture = _FakeCapture(_good_frames(count=120), metadata=_video_metadata())
        result = self._analyze(capture, detector_factory=detector_factory)
        self.assertEqual(result["score"], result["details"]["visual_confidence"])

    def test_detector_reused_within_one_analysis(self):
        detector_factory = _DetectorFactory(has_face=True, landmarks=_landmarks())
        capture = _FakeCapture(_good_frames(count=120), metadata=_video_metadata())
        self._analyze(capture, detector_factory=detector_factory)
        self.assertEqual(len(detector_factory.instances), 1)
        self.assertGreater(detector_factory.instances[0].process_calls, 1)
        self.assertEqual(detector_factory.instances[0].close_calls, 1)

    def test_separate_analyses_receive_separate_detector_instances(self):
        detector_factory = _DetectorFactory(has_face=True, landmarks=_landmarks())
        self._analyze(_FakeCapture(_good_frames(count=120), metadata=_video_metadata()), detector_factory=detector_factory)
        self._analyze(_FakeCapture(_good_frames(count=120), metadata=_video_metadata()), detector_factory=detector_factory)
        self.assertEqual(len(detector_factory.instances), 2)
        self.assertIsNot(detector_factory.instances[0], detector_factory.instances[1])

    def test_detector_close_called_on_process_exception(self):
        detector_factory = _DetectorFactory(has_face=True, process_raises=True)
        capture = _FakeCapture(_good_frames(count=120), metadata=_video_metadata())
        result = self._analyze(capture, detector_factory=detector_factory)
        self.assertEqual(result["details"]["status"], "ok")
        self.assertIn("landmark_detection_failed", result["details"]["visual_warnings"])
        self.assertEqual(detector_factory.instances[0].close_calls, 1)

    def test_detector_close_exception_is_suppressed(self):
        detector_factory = _DetectorFactory(has_face=True, close_raises=True)
        capture = _FakeCapture(_good_frames(count=120), metadata=_video_metadata())
        result = self._analyze(capture, detector_factory=detector_factory)
        self.assertEqual(result["details"]["status"], "ok")
        self.assertEqual(detector_factory.instances[0].close_calls, 1)

    def test_detector_unavailable_does_not_fabricate_no_face_confidence(self):
        capture = _FakeCapture(_good_frames(count=120), metadata=_video_metadata())
        result = self._analyze(capture, detector_factory=None)
        details = result["details"]
        self.assertIsNone(details["landmark_detection_confidence"])
        self.assertIsNone(details["face_or_subject_visibility"])
        self.assertNotIn("subject_not_visible", details["visual_warnings"])
        self.assertNotIn("face_not_visible", details["visual_warnings"])
        self.assertNotIn("landmark_detection_failed", details["visual_warnings"])

    def test_no_face_does_not_emit_unstable_video(self):
        result = self._analyze(
            _FakeCapture(_good_frames(count=120), metadata=_video_metadata()),
            detector_factory=_DetectorFactory(has_face=False),
        )
        warnings = result["details"]["visual_warnings"]
        self.assertIn("subject_not_visible", warnings)
        self.assertIn("face_not_visible", warnings)
        self.assertNotIn("unstable_video", warnings)

    def test_detector_failure_does_not_emit_unstable_video(self):
        result = self._analyze(
            _FakeCapture(_good_frames(count=120), metadata=_video_metadata()),
            detector_factory=_DetectorFactory(has_face=True, process_raises=True),
        )
        warnings = result["details"]["visual_warnings"]
        self.assertIn("landmark_detection_failed", warnings)
        self.assertNotIn("subject_not_visible", warnings)
        self.assertNotIn("face_not_visible", warnings)
        self.assertNotIn("unstable_video", warnings)

    def test_no_face_is_distinguished_from_detector_failure(self):
        no_face = self._analyze(_FakeCapture(_good_frames(count=120), metadata=_video_metadata()), detector_factory=_DetectorFactory(has_face=False))
        detector_failure = self._analyze(
            _FakeCapture(_good_frames(count=120), metadata=_video_metadata()),
            detector_factory=_DetectorFactory(has_face=True, process_raises=True),
        )
        self.assertIn("subject_not_visible", no_face["details"]["visual_warnings"])
        self.assertIn("face_not_visible", no_face["details"]["visual_warnings"])
        self.assertNotIn("landmark_detection_failed", no_face["details"]["visual_warnings"])
        self.assertIn("landmark_detection_failed", detector_failure["details"]["visual_warnings"])
        self.assertNotIn("subject_not_visible", detector_failure["details"]["visual_warnings"])

    def test_landmark_confidence_is_not_fabricated(self):
        detector_factory = _DetectorFactory(has_face=True, landmarks=_landmarks())
        capture = _FakeCapture(_good_frames(count=120), metadata=_video_metadata())
        result = self._analyze(capture, detector_factory=detector_factory)
        self.assertIsNone(result["details"]["landmark_detection_confidence"])

    def test_valid_eye_geometry_produces_finite_eye_metrics(self):
        detector_factory = _DetectorFactory(has_face=True, landmarks=_landmarks(left_ear=0.28, right_ear=0.30))
        capture = _FakeCapture(_good_frames(count=120), metadata=_video_metadata())
        result = self._analyze(capture, detector_factory=detector_factory)
        details = result["details"]
        self.assertTrue(math.isfinite(details["avg_eye_aperture"]))
        self.assertTrue(math.isfinite(details["eye_aperture_std"]))
        self.assertTrue(math.isfinite(details["avg_eye_asymmetry"]))
        self.assertGreater(details["eye_closure_sample_count"], 0)

    def test_invalid_eye_geometry_leaves_eye_metrics_none(self):
        for mode in ["zero_horizontal", "nan", "inf", "missing_indices", "malformed"]:
            with self.subTest(mode=mode):
                detector_factory = _DetectorFactory(has_face=True, landmarks=_invalid_landmark_points(mode))
                capture = _FakeCapture(_good_frames(count=120), metadata=_video_metadata())
                result = self._analyze(capture, detector_factory=detector_factory)
                details = result["details"]
                self.assertIsNone(details["avg_eye_aperture"])
                self.assertIsNone(details["eye_aperture_std"])
                self.assertIsNone(details["avg_eye_asymmetry"])
                self.assertFalse(details["sustained_eye_closure"])

    def test_reliable_eye_landmarks_require_motion_evidence(self):
        self.assertFalse(
            video._eye_landmark_reliability(
                detector_present=True,
                valid_eye_samples=3,
                burst_observation_count=4,
                face_visibility=0.9,
                motion_stability_score=None,
                sharpness_score=0.8,
                brightness_score=0.8,
            )
        )
        self.assertFalse(
            video._eye_landmark_reliability(
                detector_present=True,
                valid_eye_samples=3,
                burst_observation_count=4,
                face_visibility=0.9,
                motion_stability_score=0.44,
                sharpness_score=0.8,
                brightness_score=0.8,
            )
        )
        self.assertTrue(
            video._eye_landmark_reliability(
                detector_present=True,
                valid_eye_samples=3,
                burst_observation_count=4,
                face_visibility=0.9,
                motion_stability_score=0.5,
                sharpness_score=0.8,
                brightness_score=0.8,
            )
        )

    def test_burst_observation_construction_is_frame_local(self):
        plan = [
            {"frame_index": 4, "role": "burst", "burst_id": 0},
            {"frame_index": 5, "role": "burst", "burst_id": 0},
        ]
        detector = _SequenceDetector([
            _detector_result(has_face=True, landmarks=_landmarks(left_ear=0.28, right_ear=0.30)),
            _detector_result(has_face=True, landmarks=_invalid_landmark_points("zero_horizontal")),
        ])
        captured = {}

        def _spy(observations):
            captured["observations"] = copy.deepcopy(observations)
            return 0, 0, 0.0

        capture = _FakeCapture(
            _good_frames(count=6),
            metadata={
                _FakeCV2.CAP_PROP_FPS: 30.0,
                _FakeCV2.CAP_PROP_FRAME_COUNT: 120,
                _FakeCV2.CAP_PROP_FRAME_WIDTH: 640,
                _FakeCV2.CAP_PROP_FRAME_HEIGHT: 480,
            },
        )
        with patch.object(video, "_build_sample_plan", return_value=plan), patch.object(
            video, "_longest_temporal_eye_closure_streak", side_effect=_spy
        ):
            result = self._analyze(
                capture,
                cv2_module=_FakeCV2(_CaptureFactory(capture)),
                mp_module=_fake_mp(lambda **kwargs: detector),
            )
        self.assertEqual(result["details"]["status"], "ok")
        observations = captured["observations"]
        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0]["frame_index"], 4)
        self.assertTrue(observations[0]["usable"])
        self.assertTrue(observations[0]["landmark_valid"])

    def test_burst_observation_ignores_stale_timestamp_without_mutating_previous_frame(self):
        plan = [
            {"frame_index": 4, "role": "burst", "burst_id": 0},
            {"frame_index": 5, "role": "burst", "burst_id": 0},
        ]
        detector = _SequenceDetector([
            _detector_result(has_face=True, landmarks=_landmarks(left_ear=0.28, right_ear=0.30)),
            _detector_result(has_face=True, landmarks=_invalid_landmark_points("zero_horizontal")),
        ])
        captured = {}

        def _spy(observations):
            captured["observations"] = copy.deepcopy(observations)
            return 0, 0, 0.0

        capture = _FakeCapture(
            _good_frames(count=6),
            metadata={
                _FakeCV2.CAP_PROP_FPS: 30.0,
                _FakeCV2.CAP_PROP_FRAME_COUNT: 120,
                _FakeCV2.CAP_PROP_FRAME_WIDTH: 640,
                _FakeCV2.CAP_PROP_FRAME_HEIGHT: 480,
            },
        )
        with patch.object(video, "_build_sample_plan", return_value=plan), patch.object(
            video, "_longest_temporal_eye_closure_streak", side_effect=_spy
        ):
            result = self._analyze(
                capture,
                cv2_module=_FakeCV2(_CaptureFactory(capture)),
                mp_module=_fake_mp(lambda **kwargs: detector),
            )
        self.assertEqual(result["details"]["status"], "ok")
        observations = captured["observations"]
        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0]["frame_index"], 4)
        self.assertTrue(observations[0]["usable"])

    def test_burst_observation_survives_detector_failure_without_cross_frame_mutation(self):
        plan = [
            {"frame_index": 4, "role": "burst", "burst_id": 0},
            {"frame_index": 5, "role": "burst", "burst_id": 0},
        ]
        detector = _SequenceDetector([
            _detector_result(has_face=True, landmarks=_landmarks(left_ear=0.28, right_ear=0.30)),
            RuntimeError("boom"),
        ])
        captured = {}

        def _spy(observations):
            captured["observations"] = copy.deepcopy(observations)
            return 0, 0, 0.0

        capture = _FakeCapture(
            _good_frames(count=6),
            metadata={
                _FakeCV2.CAP_PROP_FPS: 30.0,
                _FakeCV2.CAP_PROP_FRAME_COUNT: 120,
                _FakeCV2.CAP_PROP_FRAME_WIDTH: 640,
                _FakeCV2.CAP_PROP_FRAME_HEIGHT: 480,
            },
        )
        with patch.object(video, "_build_sample_plan", return_value=plan), patch.object(
            video, "_longest_temporal_eye_closure_streak", side_effect=_spy
        ):
            result = self._analyze(
                capture,
                cv2_module=_FakeCV2(_CaptureFactory(capture)),
                mp_module=_fake_mp(lambda **kwargs: detector),
            )
        self.assertEqual(result["details"]["status"], "ok")
        observations = captured["observations"]
        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0]["frame_index"], 4)
        self.assertTrue(observations[0]["usable"])

    def test_insufficient_temporal_evidence_cannot_produce_sustained_closure(self):
        observations = [
            {"window_id": 0, "timestamp": 0.0, "frame_index": 0, "usable": True, "bright_enough": True, "sharp_enough": True, "face_visible": True, "landmark_valid": True, "eye_closed": True},
            {"window_id": 0, "timestamp": 0.1, "frame_index": 1, "usable": True, "bright_enough": True, "sharp_enough": True, "face_visible": True, "landmark_valid": True, "eye_closed": True},
            {"window_id": 1, "timestamp": 0.0, "frame_index": 0, "usable": True, "bright_enough": True, "sharp_enough": True, "face_visible": True, "landmark_valid": True, "eye_closed": True},
            {"window_id": 1, "timestamp": 0.1, "frame_index": 1, "usable": True, "bright_enough": True, "sharp_enough": True, "face_visible": True, "landmark_valid": True, "eye_closed": True},
        ]
        longest, window_ms, window_seconds = video._longest_temporal_eye_closure_streak(observations)
        self.assertEqual(longest, 0)
        self.assertEqual(window_ms, 0)
        self.assertEqual(window_seconds, 0.0)

    def test_duplicate_frames_do_not_inflate_closure_streak(self):
        observations = [
            {"window_id": 0, "timestamp": 0.0, "frame_index": 10, "usable": True, "bright_enough": True, "sharp_enough": True, "face_visible": True, "landmark_valid": True, "eye_closed": True},
            {"window_id": 0, "timestamp": 0.1, "frame_index": 10, "usable": True, "bright_enough": True, "sharp_enough": True, "face_visible": True, "landmark_valid": True, "eye_closed": True},
            {"window_id": 0, "timestamp": 0.2, "frame_index": 11, "usable": True, "bright_enough": True, "sharp_enough": True, "face_visible": True, "landmark_valid": True, "eye_closed": True},
            {"window_id": 0, "timestamp": 0.3, "frame_index": 12, "usable": True, "bright_enough": True, "sharp_enough": True, "face_visible": True, "landmark_valid": True, "eye_closed": True},
            {"window_id": 0, "timestamp": 0.4, "frame_index": 13, "usable": True, "bright_enough": True, "sharp_enough": True, "face_visible": True, "landmark_valid": True, "eye_closed": True},
        ]
        longest, _, _ = video._longest_temporal_eye_closure_streak(observations)
        self.assertEqual(longest, 0)

    def test_longest_temporal_eye_closure_streak_rejects_bad_chronology(self):
        repeated_index = [
            {"window_id": 0, "timestamp": 0.0, "frame_index": 10, "usable": True, "bright_enough": True, "sharp_enough": True, "face_visible": True, "landmark_valid": True, "eye_closed": True},
            {"window_id": 0, "timestamp": 0.1, "frame_index": 10, "usable": True, "bright_enough": True, "sharp_enough": True, "face_visible": True, "landmark_valid": True, "eye_closed": True},
            {"window_id": 0, "timestamp": 0.2, "frame_index": 11, "usable": True, "bright_enough": True, "sharp_enough": True, "face_visible": True, "landmark_valid": True, "eye_closed": True},
            {"window_id": 0, "timestamp": 0.3, "frame_index": 12, "usable": True, "bright_enough": True, "sharp_enough": True, "face_visible": True, "landmark_valid": True, "eye_closed": True},
        ]
        decreasing_index = [
            {"window_id": 0, "timestamp": 0.0, "frame_index": 10, "usable": True, "bright_enough": True, "sharp_enough": True, "face_visible": True, "landmark_valid": True, "eye_closed": True},
            {"window_id": 0, "timestamp": 0.1, "frame_index": 9, "usable": True, "bright_enough": True, "sharp_enough": True, "face_visible": True, "landmark_valid": True, "eye_closed": True},
            {"window_id": 0, "timestamp": 0.2, "frame_index": 11, "usable": True, "bright_enough": True, "sharp_enough": True, "face_visible": True, "landmark_valid": True, "eye_closed": True},
            {"window_id": 0, "timestamp": 0.3, "frame_index": 12, "usable": True, "bright_enough": True, "sharp_enough": True, "face_visible": True, "landmark_valid": True, "eye_closed": True},
        ]
        repeated_timestamp = [
            {"window_id": 0, "timestamp": 0.0, "frame_index": 10, "usable": True, "bright_enough": True, "sharp_enough": True, "face_visible": True, "landmark_valid": True, "eye_closed": True},
            {"window_id": 0, "timestamp": 0.0, "frame_index": 11, "usable": True, "bright_enough": True, "sharp_enough": True, "face_visible": True, "landmark_valid": True, "eye_closed": True},
            {"window_id": 0, "timestamp": 0.2, "frame_index": 12, "usable": True, "bright_enough": True, "sharp_enough": True, "face_visible": True, "landmark_valid": True, "eye_closed": True},
            {"window_id": 0, "timestamp": 0.3, "frame_index": 13, "usable": True, "bright_enough": True, "sharp_enough": True, "face_visible": True, "landmark_valid": True, "eye_closed": True},
        ]
        decreasing_timestamp = [
            {"window_id": 0, "timestamp": 0.0, "frame_index": 10, "usable": True, "bright_enough": True, "sharp_enough": True, "face_visible": True, "landmark_valid": True, "eye_closed": True},
            {"window_id": 0, "timestamp": 0.2, "frame_index": 11, "usable": True, "bright_enough": True, "sharp_enough": True, "face_visible": True, "landmark_valid": True, "eye_closed": True},
            {"window_id": 0, "timestamp": 0.1, "frame_index": 12, "usable": True, "bright_enough": True, "sharp_enough": True, "face_visible": True, "landmark_valid": True, "eye_closed": True},
            {"window_id": 0, "timestamp": 0.3, "frame_index": 13, "usable": True, "bright_enough": True, "sharp_enough": True, "face_visible": True, "landmark_valid": True, "eye_closed": True},
        ]
        too_large_gap = [
            {"window_id": 0, "timestamp": 0.0, "frame_index": 10, "usable": True, "bright_enough": True, "sharp_enough": True, "face_visible": True, "landmark_valid": True, "eye_closed": True},
            {"window_id": 0, "timestamp": 0.1, "frame_index": 11, "usable": True, "bright_enough": True, "sharp_enough": True, "face_visible": True, "landmark_valid": True, "eye_closed": True},
            {"window_id": 0, "timestamp": 0.5, "frame_index": 12, "usable": True, "bright_enough": True, "sharp_enough": True, "face_visible": True, "landmark_valid": True, "eye_closed": True},
            {"window_id": 0, "timestamp": 0.6, "frame_index": 13, "usable": True, "bright_enough": True, "sharp_enough": True, "face_visible": True, "landmark_valid": True, "eye_closed": True},
        ]
        for case in [repeated_index, decreasing_index, repeated_timestamp, decreasing_timestamp, too_large_gap]:
            with self.subTest(case=case):
                longest, window_ms, window_seconds = video._longest_temporal_eye_closure_streak(case)
                self.assertEqual(longest, 0)
                self.assertEqual(window_ms, 0)
                self.assertEqual(window_seconds, 0.0)

    def test_duration_and_dimension_helpers_are_conservative(self):
        self.assertAlmostEqual(
            video._derive_duration_seconds(fps=30.0, frame_count=120, observed_timestamps=[0.0, 0.1], decoded_frame_indices=[0, 119]),
            4.0,
        )
        self.assertAlmostEqual(
            video._derive_duration_seconds(fps=None, frame_count=None, observed_timestamps=[1.0, 2.5, 3.0], decoded_frame_indices=[4, 20]),
            2.0,
        )
        self.assertAlmostEqual(
            video._derive_duration_seconds(fps=30.0, frame_count=None, observed_timestamps=[], decoded_frame_indices=[4, 20]),
            (20 - 4) / 30.0,
        )
        self.assertIsNone(
            video._derive_duration_seconds(fps=None, frame_count=None, observed_timestamps=[0.0, 0.0], decoded_frame_indices=[4, 20])
        )
        self.assertIsNone(video._derive_duration_seconds(fps=30.0, frame_count=None, observed_timestamps=[], decoded_frame_indices=[4]))
        self.assertIsNone(video._derive_duration_seconds(fps=30.0, frame_count=None, observed_timestamps=[], decoded_frame_indices=[4, 4]))
        self.assertIsNone(video._derive_duration_seconds(fps=30.0, frame_count=None, observed_timestamps=[], decoded_frame_indices=[True, 4]))
        self.assertEqual(video._select_decoded_dimensions([(1920, 1080), (1280, 720)], 3840, 2160), (1280, 720))
        self.assertEqual(video._select_decoded_dimensions([(1920, 1080), (640, 360)], 1920, 1080), (640, 360))
        self.assertEqual(video._select_decoded_dimensions([], 640, 360), (640, 360))

    def test_duration_priority_prefers_metadata_frame_count_when_consistent(self):
        capture = _FakeCapture(_good_frames(count=120), metadata=_video_metadata(frame_count=120, fps=30.0))
        result = self._analyze(capture, detector_factory=_DetectorFactory(has_face=True))
        self.assertEqual(result["details"]["status"], "ok")
        self.assertAlmostEqual(result["details"]["duration_seconds"], 4.0)
        self.assertEqual(result["details"]["frame_count"], 120)
        self.assertEqual(result["details"]["frames"], 120)
        self.assertLessEqual(result["details"]["sampled_frames"], video.MAX_SAMPLED_FRAMES)

    def test_duration_priority_uses_timestamp_span_when_frame_count_missing(self):
        plan = [
            {"frame_index": 0, "role": "coverage", "burst_id": None},
            {"frame_index": 1, "role": "coverage", "burst_id": None},
            {"frame_index": 2, "role": "coverage", "burst_id": None},
            {"frame_index": 3, "role": "coverage", "burst_id": None},
            {"frame_index": 4, "role": "coverage", "burst_id": None},
            {"frame_index": 5, "role": "coverage", "burst_id": None},
            {"frame_index": 6, "role": "coverage", "burst_id": None},
            {"frame_index": 7, "role": "coverage", "burst_id": None},
        ]

        class _TimestampCapture(_FakeCapture):
            def __init__(self, *args, timestamps, **kwargs):
                super().__init__(*args, **kwargs)
                self._timestamps = iter(timestamps)

            def get(self, prop):
                if prop == _FakeCV2.CAP_PROP_POS_MSEC:
                    return next(self._timestamps, 7000.0)
                return super().get(prop)

        capture = _TimestampCapture(
            _good_frames(count=8),
            timestamps=[1000.0, 1220.0, 1440.0, 1660.0, 1880.0, 2100.0, 2320.0, 2540.0],
            metadata={
                _FakeCV2.CAP_PROP_FPS: None,
                _FakeCV2.CAP_PROP_FRAME_COUNT: None,
                _FakeCV2.CAP_PROP_FRAME_WIDTH: 640,
                _FakeCV2.CAP_PROP_FRAME_HEIGHT: 480,
            },
        )
        with patch.object(video, "_build_sample_plan", return_value=plan):
            result = self._analyze(capture, detector_factory=_DetectorFactory(has_face=True))
        self.assertEqual(result["details"]["status"], "ok")
        self.assertAlmostEqual(result["details"]["duration_seconds"], 1.54)
        self.assertIsNone(result["details"]["frame_count"])
        self.assertIsNone(result["details"]["frames"])
        self.assertEqual(result["details"]["sampled_frames"], 8)

    def test_duration_priority_uses_decoded_index_span_when_fps_available(self):
        plan = [
            {"frame_index": 4, "role": "coverage", "burst_id": None},
            {"frame_index": 20, "role": "coverage", "burst_id": None},
        ]
        capture = _FakeCapture(
            [_frame(128) for _ in range(21)],
            metadata={
                _FakeCV2.CAP_PROP_FPS: 30.0,
                _FakeCV2.CAP_PROP_FRAME_COUNT: None,
                _FakeCV2.CAP_PROP_FRAME_WIDTH: 640,
                _FakeCV2.CAP_PROP_FRAME_HEIGHT: 480,
            },
        )
        with patch.object(video, "_build_sample_plan", return_value=plan):
            result = self._analyze(capture, detector_factory=_DetectorFactory(has_face=True))
        self.assertEqual(result["details"]["status"], "ok")
        self.assertAlmostEqual(result["details"]["duration_seconds"], 16.0 / 30.0, places=3)
        self.assertIsNone(result["details"]["frame_count"])
        self.assertIsNone(result["details"]["frames"])
        self.assertLessEqual(result["details"]["sampled_frames"], video.MAX_SAMPLED_FRAMES)
        self.assertNotAlmostEqual(result["details"]["duration_seconds"], result["details"]["sampled_frames"] / 30.0)

    def test_missing_frame_count_uses_timestamp_span_not_sample_count(self):
        plan = [
            {"frame_index": 0, "role": "coverage", "burst_id": None},
            {"frame_index": 1, "role": "coverage", "burst_id": None},
            {"frame_index": 2, "role": "coverage", "burst_id": None},
            {"frame_index": 3, "role": "coverage", "burst_id": None},
            {"frame_index": 4, "role": "coverage", "burst_id": None},
            {"frame_index": 5, "role": "coverage", "burst_id": None},
            {"frame_index": 6, "role": "coverage", "burst_id": None},
            {"frame_index": 7, "role": "coverage", "burst_id": None},
        ]
        capture = _FakeCapture(
            _good_frames(count=8),
            metadata={
                _FakeCV2.CAP_PROP_FPS: None,
                _FakeCV2.CAP_PROP_FRAME_COUNT: None,
                _FakeCV2.CAP_PROP_FRAME_WIDTH: 640,
                _FakeCV2.CAP_PROP_FRAME_HEIGHT: 480,
                _FakeCV2.CAP_PROP_POS_MSEC: 0.0,
            },
        )
        timestamps = iter([1000.0, 1220.0, 1440.0, 1660.0, 1880.0, 2100.0, 2320.0, 2540.0, 2760.0, 2980.0, 3200.0, 3420.0, 3640.0, 3860.0, 4080.0, 4300.0])

        class _TimestampCapture(_FakeCapture):
            def get(self, prop):
                if prop == _FakeCV2.CAP_PROP_POS_MSEC:
                    return next(timestamps, 7000.0)
                return super().get(prop)

        with patch.object(video, "_build_sample_plan", return_value=plan):
            result = self._analyze(
                _TimestampCapture(_good_frames(count=8), metadata={
                    _FakeCV2.CAP_PROP_FPS: None,
                    _FakeCV2.CAP_PROP_FRAME_COUNT: None,
                    _FakeCV2.CAP_PROP_FRAME_WIDTH: 640,
                    _FakeCV2.CAP_PROP_FRAME_HEIGHT: 480,
                }),
                detector_factory=_DetectorFactory(has_face=True),
            )
        self.assertEqual(result["details"]["status"], "ok")
        self.assertAlmostEqual(result["details"]["duration_seconds"], 1.54)
        self.assertIsNone(result["details"]["frame_count"])
        self.assertIsNone(result["details"]["frames"])
        self.assertLessEqual(result["details"]["sampled_frames"], video.MAX_SAMPLED_FRAMES)

    def test_missing_frame_count_with_valid_fps_uses_decoded_span_not_sample_count(self):
        plan = [
            {"frame_index": 4, "role": "coverage", "burst_id": None},
            {"frame_index": 20, "role": "coverage", "burst_id": None},
        ]
        capture = _FakeCapture(
            [_frame(128) for _ in range(21)],
            metadata={
                _FakeCV2.CAP_PROP_FPS: 30.0,
                _FakeCV2.CAP_PROP_FRAME_COUNT: None,
                _FakeCV2.CAP_PROP_FRAME_WIDTH: 640,
                _FakeCV2.CAP_PROP_FRAME_HEIGHT: 480,
            },
        )
        with patch.object(video, "_build_sample_plan", return_value=plan):
            result = self._analyze(capture, detector_factory=_DetectorFactory(has_face=True))
        self.assertEqual(result["details"]["status"], "ok")
        self.assertAlmostEqual(result["details"]["duration_seconds"], 16.0 / 30.0, places=3)
        self.assertIsNone(result["details"]["frame_count"])
        self.assertIsNone(result["details"]["frames"])
        self.assertNotAlmostEqual(result["details"]["duration_seconds"], 8.0 / 30.0)

    def test_contradictory_frame_count_is_rejected_for_duration(self):
        plan = [
            {"frame_index": 0, "role": "coverage", "burst_id": None},
            {"frame_index": 6, "role": "coverage", "burst_id": None},
        ]
        capture = _FakeCapture(
            [_frame(128) for _ in range(7)],
            metadata={
                _FakeCV2.CAP_PROP_FPS: 30.0,
                _FakeCV2.CAP_PROP_FRAME_COUNT: 5,
                _FakeCV2.CAP_PROP_FRAME_WIDTH: 640,
                _FakeCV2.CAP_PROP_FRAME_HEIGHT: 480,
            },
        )
        with patch.object(video, "_build_sample_plan", return_value=plan):
            result = self._analyze(capture, detector_factory=_DetectorFactory(has_face=True))
        self.assertEqual(result["details"]["status"], "ok")
        self.assertAlmostEqual(result["details"]["duration_seconds"], 6.0 / 30.0, places=3)
        self.assertIsNone(result["details"]["frame_count"])
        self.assertIsNone(result["details"]["frames"])

    def test_no_reliable_source_returns_open_failed(self):
        plan = [
            {"frame_index": 0, "role": "coverage", "burst_id": None},
            {"frame_index": 1, "role": "coverage", "burst_id": None},
        ]
        capture = _FakeCapture(
            [_frame(128) for _ in range(2)],
            metadata={
                _FakeCV2.CAP_PROP_FPS: None,
                _FakeCV2.CAP_PROP_FRAME_COUNT: None,
                _FakeCV2.CAP_PROP_FRAME_WIDTH: 640,
                _FakeCV2.CAP_PROP_FRAME_HEIGHT: 480,
                _FakeCV2.CAP_PROP_POS_MSEC: 0.0,
            },
        )
        with patch.object(video, "_build_sample_plan", return_value=plan):
            result = self._analyze(capture, detector_factory=_DetectorFactory(has_face=True))
        self.assertEqual(result["details"]["status"], "open_failed")
        self.assertIsNone(result["score"])
        self.assertEqual(result["details"]["visual_warnings"], ["video_missing"])

    def test_repeated_zero_timestamps_do_not_establish_duration(self):
        plan = [
            {"frame_index": 0, "role": "coverage", "burst_id": None},
            {"frame_index": 1, "role": "coverage", "burst_id": None},
        ]
        class _ZeroTimestampCapture(_FakeCapture):
            def get(self, prop):
                if prop == _FakeCV2.CAP_PROP_POS_MSEC:
                    return 0.0
                return super().get(prop)

        capture = _ZeroTimestampCapture(
            _good_frames(count=2),
            metadata={
                _FakeCV2.CAP_PROP_FPS: None,
                _FakeCV2.CAP_PROP_FRAME_COUNT: None,
                _FakeCV2.CAP_PROP_FRAME_WIDTH: 640,
                _FakeCV2.CAP_PROP_FRAME_HEIGHT: 480,
            },
        )
        with patch.object(video, "_build_sample_plan", return_value=plan):
            result = self._analyze(capture, detector_factory=_DetectorFactory(has_face=True))
        self.assertEqual(result["details"]["status"], "open_failed")
        self.assertIsNone(result["score"])

    def test_decoded_dimensions_override_inflated_metadata_end_to_end(self):
        plan = [
            {"frame_index": 0, "role": "coverage", "burst_id": None},
            {"frame_index": 1, "role": "coverage", "burst_id": None},
        ]
        decoded_frames = [_frame(128, width=320, height=240), _frame(128, width=320, height=240)]

        with patch.object(video, "_build_sample_plan", return_value=plan):
            inflated = self._analyze(
                _FakeCapture(
                    decoded_frames,
                    metadata={
                        _FakeCV2.CAP_PROP_FPS: 30.0,
                        _FakeCV2.CAP_PROP_FRAME_COUNT: 120,
                        _FakeCV2.CAP_PROP_FRAME_WIDTH: 1920,
                        _FakeCV2.CAP_PROP_FRAME_HEIGHT: 1080,
                    },
                ),
                detector_factory=_DetectorFactory(has_face=True),
            )
        self.assertEqual(inflated["details"]["status"], "ok")
        self.assertEqual(inflated["details"]["resolution"], {"width": 320, "height": 240})
        self.assertIn("video_low_resolution", inflated["details"]["visual_warnings"])

        with patch.object(video, "_build_sample_plan", return_value=plan):
            decoded_only = self._analyze(
                _FakeCapture(
                    decoded_frames,
                    metadata={
                        _FakeCV2.CAP_PROP_FPS: 30.0,
                        _FakeCV2.CAP_PROP_FRAME_COUNT: 120,
                        _FakeCV2.CAP_PROP_FRAME_WIDTH: 320,
                        _FakeCV2.CAP_PROP_FRAME_HEIGHT: 240,
                    },
                ),
                detector_factory=_DetectorFactory(has_face=True),
            )
        self.assertEqual(decoded_only["details"]["resolution"], {"width": 320, "height": 240})
        self.assertEqual(decoded_only["details"]["visual_warnings"], inflated["details"]["visual_warnings"])
        self.assertAlmostEqual(decoded_only["details"]["visual_quality_score"], inflated["details"]["visual_quality_score"])

    def test_mixed_decoded_dimensions_use_conservative_selection(self):
        plan = [
            {"frame_index": 0, "role": "coverage", "burst_id": None},
            {"frame_index": 1, "role": "coverage", "burst_id": None},
        ]
        capture = _FakeCapture(
            [_frame(128, width=640, height=480), _frame(128, width=320, height=240)],
            metadata={
                _FakeCV2.CAP_PROP_FPS: 30.0,
                _FakeCV2.CAP_PROP_FRAME_COUNT: 120,
                _FakeCV2.CAP_PROP_FRAME_WIDTH: 1920,
                _FakeCV2.CAP_PROP_FRAME_HEIGHT: 1080,
            },
        )
        with patch.object(video, "_build_sample_plan", return_value=plan):
            result = self._analyze(capture, detector_factory=_DetectorFactory(has_face=True))
        self.assertEqual(result["details"]["status"], "ok")
        self.assertEqual(result["details"]["resolution"], {"width": 320, "height": 240})
        self.assertIn("video_low_resolution", result["details"]["visual_warnings"])
        self.assertLessEqual(result["details"]["visual_quality_score"], 1.0)

    def test_successful_contiguous_closure_evidence_can_set_observable_flag(self):
        detector_factory = _DetectorFactory(has_face=True, landmarks=_landmarks(left_ear=0.12, right_ear=0.12))
        capture = _FakeCapture(_good_frames(count=120), metadata=_video_metadata())
        result = self._analyze(capture, detector_factory=detector_factory)
        self.assertTrue(result["details"]["sustained_eye_closure"])
        self.assertIn("sustained_eye_closure", result["details"]["visual_warnings"])

    def test_brightness_and_blur_behaviour(self):
        dark = self._analyze(_FakeCapture(_success_frames(value=0, count=120), metadata=_video_metadata()), detector_factory=_DetectorFactory(has_face=True))
        mid = self._analyze(_FakeCapture(_success_frames(value=128, count=120), metadata=_video_metadata()), detector_factory=_DetectorFactory(has_face=True))
        bright = self._analyze(_FakeCapture(_success_frames(value=255, count=120), metadata=_video_metadata()), detector_factory=_DetectorFactory(has_face=True))
        self.assertIn("video_too_dark", dark["details"]["visual_warnings"])
        self.assertNotIn("video_too_dark", mid["details"]["visual_warnings"])
        self.assertNotIn("video_too_dark", bright["details"]["visual_warnings"])
        self.assertGreater(mid["details"]["brightness_score"], dark["details"]["brightness_score"])
        self.assertLess(bright["details"]["brightness_score"], mid["details"]["brightness_score"])
        self.assertLess(bright["details"]["visual_quality_score"], mid["details"]["visual_quality_score"])

    def test_low_resolution_warning_is_deterministic(self):
        capture = _FakeCapture(_success_frames(count=120, width=320, height=240), metadata=_video_metadata(width=320, height=240))
        result = self._analyze(capture, detector_factory=_DetectorFactory(has_face=True))
        self.assertIn("video_low_resolution", result["details"]["visual_warnings"])

    def test_sharpness_score_is_normalized(self):
        sharp = self._analyze(_FakeCapture(_good_frames(count=120), metadata=_video_metadata()), detector_factory=_DetectorFactory(has_face=True))
        blurry = self._analyze(_FakeCapture(_success_frames(value=128, count=120), metadata=_video_metadata()), detector_factory=_DetectorFactory(has_face=True))
        self.assertGreaterEqual(sharp["details"]["sharpness_score"], 0.0)
        self.assertLessEqual(sharp["details"]["sharpness_score"], 1.0)
        self.assertLess(blurry["details"]["sharpness_score"], sharp["details"]["sharpness_score"])
        self.assertIn("video_blurry", blurry["details"]["visual_warnings"])

    def test_motion_stability_reacts_to_constant_large_movement(self):
        stable = self._analyze(_FakeCapture(_good_frames(count=120), metadata=_video_metadata()), detector_factory=_DetectorFactory(has_face=True))
        moving = self._analyze(_FakeCapture(_moving_frames(count=120), metadata=_video_metadata()), detector_factory=_DetectorFactory(has_face=True))
        self.assertGreater(stable["details"]["motion_stability_score"], moving["details"]["motion_stability_score"])
        self.assertLess(moving["details"]["motion_stability_score"], 1.0)
        self.assertIn("unstable_camera", moving["details"]["visual_warnings"])

    def test_warning_order_and_deduplication_are_deterministic(self):
        detector_factory = _DetectorFactory(has_face=True)
        result = self._analyze(
            _FakeCapture(_success_frames(value=0, count=120, width=320, height=240), metadata=_video_metadata(width=320, height=240)),
            detector_factory=detector_factory,
        )
        warnings = result["details"]["visual_warnings"]
        self.assertEqual(warnings, list(dict.fromkeys(warnings)))

    def test_cuda_visible_devices_is_not_modified_on_import(self):
        original = os.environ.get("CUDA_VISIBLE_DEVICES")
        marker = "video-test-marker"
        os.environ["CUDA_VISIBLE_DEVICES"] = marker
        module_name = "video_import_isolation_test"
        try:
            sys.modules.pop(module_name, None)
            spec = importlib.util.spec_from_file_location(module_name, Path(video.__file__).resolve())
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)
            self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), marker)
        finally:
            if original is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = original
            sys.modules.pop(module_name, None)


if __name__ == "__main__":
    unittest.main()
