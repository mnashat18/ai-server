from __future__ import annotations

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
import vision


def _image(value: int = 128, *, width: int = 320, height: int = 320) -> np.ndarray:
    return np.full((height, width, 3), value, dtype=np.uint8)


def _checkerboard(size: int = 320) -> np.ndarray:
    pattern = (np.indices((size, size)).sum(axis=0) % 2).astype(np.uint8) * 255
    return np.dstack([pattern, pattern, pattern])


def _landmark(x: float, y: float):
    return types.SimpleNamespace(x=x, y=y)


def _landmarks(*, left_ear: float = 0.28, right_ear: float = 0.28):
    points = [_landmark(0.0, 0.0) for _ in range(400)]

    def _eye(idxs, ear):
        p0, p1, p2, p3, p4, p5 = idxs
        points[p0] = _landmark(0.0, 0.0)
        points[p3] = _landmark(1.0, 0.0)
        points[p1] = _landmark(0.5, ear / 2.0)
        points[p5] = _landmark(0.5, -ear / 2.0)
        points[p2] = _landmark(0.5, ear / 2.0)
        points[p4] = _landmark(0.5, -ear / 2.0)

    _eye(vision.LEFT_EYE, left_ear)
    _eye(vision.RIGHT_EYE, right_ear)
    return points


def _fake_detector(
    *,
    has_face: bool,
    left_ear: float = 0.28,
    right_ear: float = 0.28,
    raises: bool = False,
    landmarks=None,
    close_raises: bool = False,
):
    class _Detector:
        last_instance = None

        def __init__(self, **kwargs):
            self.closed = False
            self.close_called = False
            _Detector.last_instance = self

        def process(self, rgb):
            if raises:
                raise RuntimeError("detector boom")
            if not has_face:
                return types.SimpleNamespace(multi_face_landmarks=None)
            face_landmarks = landmarks if landmarks is not None else _landmarks(left_ear=left_ear, right_ear=right_ear)
            return types.SimpleNamespace(
                multi_face_landmarks=[types.SimpleNamespace(landmark=face_landmarks)]
            )

        def close(self):
            self.closed = True
            self.close_called = True
            if close_raises:
                raise RuntimeError("close boom")

    return _Detector


def _invalid_landmarks(*, mode: str):
    points = [_landmark(0.0, 0.0) for _ in range(400)]
    if mode == "zero_horizontal":
        for idx in vision.LEFT_EYE + vision.RIGHT_EYE:
            points[idx] = _landmark(0.0, 0.0)
        points[vision.LEFT_EYE[1]] = _landmark(0.0, 0.1)
        points[vision.LEFT_EYE[2]] = _landmark(0.0, 0.1)
    elif mode == "nan":
        points[vision.LEFT_EYE[0]] = _landmark(float("nan"), 0.0)
    elif mode == "inf":
        points[vision.LEFT_EYE[0]] = _landmark(float("inf"), 0.0)
    elif mode == "malformed":
        return types.SimpleNamespace()
    return points


def _fake_mp(detector_cls):
    return types.SimpleNamespace(
        solutions=types.SimpleNamespace(
            face_mesh=types.SimpleNamespace(FaceMesh=detector_cls),
        )
    )


def _fake_cv2(*, imread_result=None, imread_raises: bool = False):
    class _LaplacianResult:
        def __init__(self, value: float):
            self._value = value

        def var(self):
            return self._value

    class _FakeCV2:
        COLOR_BGR2GRAY = 0
        COLOR_BGRA2GRAY = 1
        COLOR_GRAY2RGB = 2
        COLOR_BGR2RGB = 3
        COLOR_BGRA2RGB = 4
        CV_64F = 0

        def imread(self, path):
            if imread_raises:
                raise RuntimeError("imread boom")
            return imread_result

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

    return _FakeCV2()


class VisionUnitTests(unittest.TestCase):
    def _analyze_with_image(self, image: np.ndarray, detector_cls, path: str = "image.jpg") -> dict:
        with patch.object(vision, "mp", _fake_mp(detector_cls)), patch.object(vision, "cv2", _fake_cv2(imread_result=image)), patch.object(vision.os.path, "exists", return_value=True), patch.object(vision.os.path, "isdir", return_value=False), patch.object(vision.os.path, "isfile", return_value=True), patch.object(vision.os.path, "getsize", return_value=10):
            return vision.analyze_face(path)

    def _assert_success_schema(self, result: dict):
        self.assertEqual(set(result.keys()), {"score", "details"})
        self.assertTrue(math.isfinite(result["score"]))
        self.assertGreaterEqual(result["score"], 0.0)
        self.assertLessEqual(result["score"], 1.0)
        details = result["details"]
        expected_keys = {
            "status",
            "resolution",
            "brightness_score",
            "blur_score",
            "sharpness_score",
            "subject_visibility",
            "image_quality_score",
            "image_confidence",
            "image_warnings",
            "face_detected",
            "landmark_detection_confidence",
            "avg_brightness",
            "blur_var",
            "avg_ear",
            "left_eye_aperture",
            "right_eye_aperture",
            "left_right_eye_asymmetry",
            "eyes_closed",
            "low_light",
            "blurry",
        }
        self.assertEqual(set(details.keys()), expected_keys)
        self.assertEqual(details["status"], "ok")
        self.assertIsInstance(details["image_warnings"], list)
        self.assertEqual(details["image_warnings"], list(dict.fromkeys(details["image_warnings"])))
        self.assertIsInstance(details["resolution"]["width"], int)
        self.assertIsInstance(details["resolution"]["height"], int)
        self.assertGreater(details["resolution"]["width"], 0)
        self.assertGreater(details["resolution"]["height"], 0)
        for key in [
            "brightness_score",
            "blur_score",
            "sharpness_score",
            "image_quality_score",
            "image_confidence",
            "avg_brightness",
            "blur_var",
            "subject_visibility",
            "landmark_detection_confidence",
            "avg_ear",
            "left_eye_aperture",
            "right_eye_aperture",
            "left_right_eye_asymmetry",
        ]:
            if details[key] is not None:
                self.assertTrue(math.isfinite(details[key]))
                self.assertGreaterEqual(details[key], 0.0)
                self.assertLessEqual(details[key], 1.0 if key not in {"avg_brightness", "blur_var"} else details[key])

    def _patch_input(self, *, path_exists: bool = True, is_dir: bool = False, is_file: bool = True, size: int = 1024, imread=None):
        patches = [
            patch.object(vision.os.path, "exists", return_value=path_exists),
            patch.object(vision.os.path, "isdir", return_value=is_dir),
            patch.object(vision.os.path, "isfile", return_value=is_file),
            patch.object(vision.os.path, "getsize", return_value=size),
        ]
        patches.append(patch.object(vision, "cv2", _fake_cv2(imread_result=imread)))
        return patches

    def test_none_path_returns_missing(self):
        result = vision.analyze_face(None)
        self.assertEqual(result, {"score": None, "details": {"status": "missing", "image_warnings": ["image_missing"]}})

    def test_non_string_path_returns_missing(self):
        result = vision.analyze_face(123)
        self.assertEqual(result["details"]["status"], "missing")
        self.assertEqual(result["details"]["image_warnings"], ["image_missing"])

    def test_empty_path_returns_missing(self):
        result = vision.analyze_face("   ")
        self.assertEqual(result["details"]["status"], "missing")

    def test_missing_file_returns_invalid_image(self):
        with patch.object(vision.os.path, "exists", return_value=False):
            result = vision.analyze_face("missing.jpg")
        self.assertEqual(result["details"]["status"], "invalid_image")
        self.assertIsNone(result["score"])

    def test_directory_path_is_rejected(self):
        with patch.object(vision.os.path, "exists", return_value=True), patch.object(vision.os.path, "isdir", return_value=True):
            result = vision.analyze_face("folder")
        self.assertEqual(result["details"]["status"], "invalid_image")

    def test_empty_file_is_invalid_image(self):
        with patch.object(vision.os.path, "exists", return_value=True), patch.object(vision.os.path, "isdir", return_value=False), patch.object(vision.os.path, "isfile", return_value=True), patch.object(vision.os.path, "getsize", return_value=0):
            result = vision.analyze_face("empty.jpg")
        self.assertEqual(result["details"]["status"], "invalid_image")

    def test_cv2_imread_none_returns_invalid_image(self):
        with patch.object(vision, "cv2", _fake_cv2(imread_result=None)), patch.object(vision.os.path, "exists", return_value=True), patch.object(vision.os.path, "isdir", return_value=False), patch.object(vision.os.path, "isfile", return_value=True), patch.object(vision.os.path, "getsize", return_value=10):
            result = vision.analyze_face("corrupt.jpg")
        self.assertEqual(result["details"]["status"], "invalid_image")
        self.assertNotIn("image_quality_score", result["details"])

    def test_corrupt_input_does_not_raise(self):
        result = vision._analyze_image_array(np.array([], dtype=np.uint8))
        self.assertEqual(result["details"]["status"], "invalid_image")

    def test_failure_result_has_no_stale_valid_score(self):
        with patch.object(vision, "cv2", _fake_cv2(imread_result=None)), patch.object(vision.os.path, "exists", return_value=True), patch.object(vision.os.path, "isdir", return_value=False), patch.object(vision.os.path, "isfile", return_value=True), patch.object(vision.os.path, "getsize", return_value=10):
            result = vision.analyze_face("broken.jpg")
        self.assertIsNone(result["score"])
        self.assertEqual(result["details"], {"status": "invalid_image", "image_warnings": ["image_missing"]})

    def test_successful_result_status_is_ok(self):
        img = _image(128)
        with patch.object(vision, "mp", _fake_mp(_fake_detector(has_face=True))), patch.object(vision, "cv2", _fake_cv2(imread_result=img)), patch.object(vision.os.path, "exists", return_value=True), patch.object(vision.os.path, "isdir", return_value=False), patch.object(vision.os.path, "isfile", return_value=True), patch.object(vision.os.path, "getsize", return_value=10):
            result = vision.analyze_face("ok.jpg")
        self._assert_success_schema(result)
        self.assertEqual(result["details"]["status"], "ok")
        self.assertIsNone(result["details"]["landmark_detection_confidence"])

    def test_successful_scores_are_finite_and_normalized(self):
        img = _image(128)
        with patch.object(vision, "mp", _fake_mp(_fake_detector(has_face=True))), patch.object(vision, "cv2", _fake_cv2(imread_result=img)), patch.object(vision.os.path, "exists", return_value=True), patch.object(vision.os.path, "isdir", return_value=False), patch.object(vision.os.path, "isfile", return_value=True), patch.object(vision.os.path, "getsize", return_value=10):
            result = vision.analyze_face("ok.jpg")
        details = result["details"]
        for key in ["brightness_score", "blur_score", "sharpness_score", "image_quality_score", "image_confidence"]:
            self.assertGreaterEqual(details[key], 0.0)
            self.assertLessEqual(details[key], 1.0)
            self.assertTrue(math.isfinite(details[key]))

    def test_dark_image_scores_poorly_and_warns(self):
        img = _image(5)
        with patch.object(vision, "mp", _fake_mp(_fake_detector(has_face=True))), patch.object(vision, "cv2", _fake_cv2(imread_result=img)), patch.object(vision.os.path, "exists", return_value=True), patch.object(vision.os.path, "isdir", return_value=False), patch.object(vision.os.path, "isfile", return_value=True), patch.object(vision.os.path, "getsize", return_value=10):
            result = vision.analyze_face("dark.jpg")
        self.assertIn("image_too_dark", result["details"]["image_warnings"])
        self.assertTrue(result["details"]["low_light"])
        self.assertLess(result["details"]["image_quality_score"], 0.5)

    def test_mid_exposure_is_not_low_light(self):
        img = _image(128)
        with patch.object(vision, "mp", _fake_mp(_fake_detector(has_face=True))), patch.object(vision, "cv2", _fake_cv2(imread_result=img)), patch.object(vision.os.path, "exists", return_value=True), patch.object(vision.os.path, "isdir", return_value=False), patch.object(vision.os.path, "isfile", return_value=True), patch.object(vision.os.path, "getsize", return_value=10):
            result = vision.analyze_face("mid.jpg")
        self.assertNotIn("image_too_dark", result["details"]["image_warnings"])
        self.assertFalse(result["details"]["low_light"])

    def test_mid_exposure_beats_very_dark(self):
        dark = _image(5)
        mid = _image(128)
        dark_result = self._analyze_with_image(dark, _fake_detector(has_face=True), "dark.jpg")
        mid_result = self._analyze_with_image(mid, _fake_detector(has_face=True), "mid.jpg")
        self.assertGreater(mid_result["details"]["image_quality_score"], dark_result["details"]["image_quality_score"])

    def test_overexposed_image_does_not_get_perfect_quality(self):
        img = _image(250)
        with patch.object(vision, "mp", _fake_mp(_fake_detector(has_face=True))), patch.object(vision, "cv2", _fake_cv2(imread_result=img)), patch.object(vision.os.path, "exists", return_value=True), patch.object(vision.os.path, "isdir", return_value=False), patch.object(vision.os.path, "isfile", return_value=True), patch.object(vision.os.path, "getsize", return_value=10):
            result = vision.analyze_face("bright.jpg")
        self.assertLess(result["details"]["brightness_score"], 1.0)
        self.assertLess(result["details"]["image_quality_score"], 1.0)
        self.assertNotIn("image_too_dark", result["details"]["image_warnings"])
        self.assertFalse(result["details"]["low_light"])
        self.assertLess(result["details"]["brightness_score"], vision._brightness_score(_image(128)))

    def test_blurry_image_scores_worse_than_sharp_image(self):
        sharp = _checkerboard()
        gradient = np.tile(np.linspace(0, 255, sharp.shape[1], dtype=np.uint8), (sharp.shape[0], 1))
        blurry = np.dstack([gradient, gradient, gradient])
        sharp_result = self._analyze_with_image(sharp, _fake_detector(has_face=True), "sharp.jpg")
        blurry_result = self._analyze_with_image(blurry, _fake_detector(has_face=True), "blurry.jpg")
        self.assertGreater(sharp_result["details"]["sharpness_score"], blurry_result["details"]["sharpness_score"])
        self.assertLessEqual(sharp_result["details"]["sharpness_score"], 1.0)
        self.assertLessEqual(blurry_result["details"]["sharpness_score"], 1.0)

    def test_sharpness_score_is_normalized_not_raw_variance(self):
        img = _checkerboard()
        with patch.object(vision, "mp", _fake_mp(_fake_detector(has_face=True))), patch.object(vision, "cv2", _fake_cv2(imread_result=img)), patch.object(vision.os.path, "exists", return_value=True), patch.object(vision.os.path, "isdir", return_value=False), patch.object(vision.os.path, "isfile", return_value=True), patch.object(vision.os.path, "getsize", return_value=10):
            result = vision.analyze_face("sharp.jpg")
        self.assertLessEqual(result["details"]["sharpness_score"], 1.0)
        self.assertGreater(result["details"]["blur_var"], result["details"]["sharpness_score"])

    def test_tiny_or_invalid_dimensions_do_not_crash(self):
        result = vision._analyze_image_array(np.zeros((0, 0, 3), dtype=np.uint8))
        self.assertEqual(result["details"]["status"], "invalid_image")
        result = vision._analyze_image_array(np.ones((4, 4, 5), dtype=np.uint8))
        self.assertEqual(result["details"]["status"], "invalid_image")

    def test_valid_dimensions_are_positive_integers(self):
        img = _image(128, width=300, height=200)
        with patch.object(vision, "mp", _fake_mp(_fake_detector(has_face=True))), patch.object(vision, "cv2", _fake_cv2(imread_result=img)), patch.object(vision.os.path, "exists", return_value=True), patch.object(vision.os.path, "isdir", return_value=False), patch.object(vision.os.path, "isfile", return_value=True), patch.object(vision.os.path, "getsize", return_value=10):
            result = vision.analyze_face("dims.jpg")
        self.assertEqual(result["details"]["resolution"], {"width": 300, "height": 200})
        self.assertGreater(result["details"]["resolution"]["width"], 0)
        self.assertGreater(result["details"]["resolution"]["height"], 0)

    def test_valid_eye_geometry_produces_finite_metrics(self):
        img = _image(128)
        with patch.object(vision, "mp", _fake_mp(_fake_detector(has_face=True))), patch.object(vision, "cv2", _fake_cv2(imread_result=img)), patch.object(vision.os.path, "exists", return_value=True), patch.object(vision.os.path, "isdir", return_value=False), patch.object(vision.os.path, "isfile", return_value=True), patch.object(vision.os.path, "getsize", return_value=10):
            result = vision.analyze_face("eyes.jpg")
        details = result["details"]
        self.assertTrue(math.isfinite(details["avg_ear"]))
        self.assertTrue(math.isfinite(details["left_eye_aperture"]))
        self.assertTrue(math.isfinite(details["right_eye_aperture"]))
        self.assertTrue(math.isfinite(details["left_right_eye_asymmetry"]))
        self.assertIsInstance(details["eyes_closed"], bool)
        self.assertIsNone(details["landmark_detection_confidence"])

    def test_eye_aspect_ratio_rejects_invalid_geometry(self):
        cases = {
            "zero_horizontal": _invalid_landmarks(mode="zero_horizontal"),
            "nan_coordinate": _invalid_landmarks(mode="nan"),
            "inf_coordinate": _invalid_landmarks(mode="inf"),
            "missing_indices": [_landmark(0.0, 0.0) for _ in range(3)],
            "malformed_object": _invalid_landmarks(mode="malformed"),
        }
        for name, landmarks in cases.items():
            with self.subTest(name=name):
                self.assertIsNone(vision._eye_aspect_ratio(landmarks, vision.LEFT_EYE))

    def test_invalid_eye_geometry_leaves_face_metrics_none(self):
        img = _image(128)
        with patch.object(vision, "mp", _fake_mp(_fake_detector(has_face=True, landmarks=_invalid_landmarks(mode="zero_horizontal")))), patch.object(vision, "cv2", _fake_cv2(imread_result=img)), patch.object(vision.os.path, "exists", return_value=True), patch.object(vision.os.path, "isdir", return_value=False), patch.object(vision.os.path, "isfile", return_value=True), patch.object(vision.os.path, "getsize", return_value=10):
            result = vision.analyze_face("invalid-eyes.jpg")
        details = result["details"]
        self.assertTrue(details["face_detected"])
        self.assertIsNone(details["avg_ear"])
        self.assertIsNone(details["left_eye_aperture"])
        self.assertIsNone(details["right_eye_aperture"])
        self.assertIsNone(details["left_right_eye_asymmetry"])
        self.assertIsNone(details["eyes_closed"])
        self.assertIsNone(details["landmark_detection_confidence"])

    def test_face_detector_no_face_does_not_fabricate_face_metrics(self):
        img = _image(128)
        with patch.object(vision, "mp", _fake_mp(_fake_detector(has_face=False))), patch.object(vision, "cv2", _fake_cv2(imread_result=img)), patch.object(vision.os.path, "exists", return_value=True), patch.object(vision.os.path, "isdir", return_value=False), patch.object(vision.os.path, "isfile", return_value=True), patch.object(vision.os.path, "getsize", return_value=10):
            result = vision.analyze_face("noface.jpg")
        details = result["details"]
        self.assertFalse(details["face_detected"])
        self.assertIn("face_not_visible", details["image_warnings"])
        self.assertIn("subject_not_visible", details["image_warnings"])
        self.assertIsNone(details["avg_ear"])
        self.assertIsNone(details["left_eye_aperture"])
        self.assertIsNone(details["right_eye_aperture"])
        self.assertIsNone(details["left_right_eye_asymmetry"])
        self.assertIsNone(details["landmark_detection_confidence"])

    def test_detector_exception_does_not_crash_image_analysis(self):
        img = _image(128)
        with patch.object(vision, "mp", _fake_mp(_fake_detector(has_face=True, raises=True))), patch.object(vision, "cv2", _fake_cv2(imread_result=img)), patch.object(vision.os.path, "exists", return_value=True), patch.object(vision.os.path, "isdir", return_value=False), patch.object(vision.os.path, "isfile", return_value=True), patch.object(vision.os.path, "getsize", return_value=10):
            result = vision.analyze_face("boom.jpg")
        self.assertEqual(result["details"]["status"], "ok")
        self.assertIsNone(result["details"]["face_detected"])
        self.assertNotIn("face_not_visible", result["details"]["image_warnings"])
        self.assertIsNone(result["details"]["landmark_detection_confidence"])

    def test_face_dependent_metrics_remain_none_when_unavailable(self):
        img = _image(128)
        with patch.object(vision, "mp", _fake_mp(_fake_detector(has_face=False))), patch.object(vision, "cv2", _fake_cv2(imread_result=img)), patch.object(vision.os.path, "exists", return_value=True), patch.object(vision.os.path, "isdir", return_value=False), patch.object(vision.os.path, "isfile", return_value=True), patch.object(vision.os.path, "getsize", return_value=10):
            result = vision.analyze_face("noface.jpg")
        for key in ["avg_ear", "left_eye_aperture", "right_eye_aperture", "left_right_eye_asymmetry", "landmark_detection_confidence"]:
            self.assertIsNone(result["details"][key])

    def test_behavioral_evidence_does_not_change_capture_quality(self):
        img = _image(128)
        with patch.object(vision, "mp", _fake_mp(_fake_detector(has_face=True, left_ear=0.12, right_ear=0.34))), patch.object(vision, "cv2", _fake_cv2(imread_result=img)), patch.object(vision.os.path, "exists", return_value=True), patch.object(vision.os.path, "isdir", return_value=False), patch.object(vision.os.path, "isfile", return_value=True), patch.object(vision.os.path, "getsize", return_value=10):
            left = vision.analyze_face("left.jpg")
        with patch.object(vision, "mp", _fake_mp(_fake_detector(has_face=True, left_ear=0.40, right_ear=0.41))), patch.object(vision, "cv2", _fake_cv2(imread_result=img)), patch.object(vision.os.path, "exists", return_value=True), patch.object(vision.os.path, "isdir", return_value=False), patch.object(vision.os.path, "isfile", return_value=True), patch.object(vision.os.path, "getsize", return_value=10):
            right = vision.analyze_face("right.jpg")
        self.assertEqual(left["details"]["image_quality_score"], right["details"]["image_quality_score"])
        self.assertEqual(left["score"], right["score"])

    def test_nan_and_infinity_cannot_survive_result_details(self):
        self.assertIsNone(vision._coerce_finite_number(float("nan")))
        self.assertIsNone(vision._coerce_finite_number(float("inf")))
        self.assertIsNone(vision._coerce_normalized_number(True))
        self.assertIsNone(vision._coerce_normalized_number(float("nan")))
        self.assertIsNone(vision._coerce_normalized_number(float("inf")))

    def test_warning_order_is_deterministic(self):
        img = _image(5, width=128, height=128)
        with patch.object(vision, "mp", _fake_mp(_fake_detector(has_face=False))), patch.object(vision, "cv2", _fake_cv2(imread_result=img)), patch.object(vision.os.path, "exists", return_value=True), patch.object(vision.os.path, "isdir", return_value=False), patch.object(vision.os.path, "isfile", return_value=True), patch.object(vision.os.path, "getsize", return_value=10):
            result = vision.analyze_face("warn.jpg")
        self.assertEqual(result["details"]["image_warnings"], ["image_low_resolution", "image_too_dark", "image_blurry", "subject_not_visible", "face_not_visible"])

    def test_duplicate_warning_codes_are_removed(self):
        self.assertEqual(vision.clean_warning_codes(["image_blurry", "image_blurry", "image_too_dark"]), ["image_blurry", "image_too_dark"])

    def test_established_warning_codes_are_preserved(self):
        img = _image(5, width=128, height=128)
        with patch.object(vision, "mp", _fake_mp(_fake_detector(has_face=False))), patch.object(vision, "cv2", _fake_cv2(imread_result=img)), patch.object(vision.os.path, "exists", return_value=True), patch.object(vision.os.path, "isdir", return_value=False), patch.object(vision.os.path, "isfile", return_value=True), patch.object(vision.os.path, "getsize", return_value=10):
            result = vision.analyze_face("warn.jpg")
        self.assertTrue({"image_low_resolution", "image_too_dark", "image_blurry"} <= set(result["details"]["image_warnings"]))

    def test_minimum_evidence_contract_matches_validation_and_quality(self):
        img = _checkerboard()
        with patch.object(vision, "mp", _fake_mp(_fake_detector(has_face=True))), patch.object(vision, "cv2", _fake_cv2(imread_result=img)), patch.object(vision.os.path, "exists", return_value=True), patch.object(vision.os.path, "isdir", return_value=False), patch.object(vision.os.path, "isfile", return_value=True), patch.object(vision.os.path, "getsize", return_value=10):
            image_result = vision.analyze_face("ok.jpg")
        validation_result = validation.validate_image_result(validation.ValidationPolicy(require_face=False), image_result, image_required=False)
        self.assertTrue(validation_result["passed"])
        quality_result = quality.assess_quality(
            {
                "video": {"score": 0.9, "details": {"status": "ok", "duration_seconds": 5.0, "visual_quality_score": 0.9, "visual_warnings": []}},
                "voice": {"score": 0.9, "details": {"status": "ok", "duration_seconds": 4.0, "audio_quality_score": 0.9, "audio_warnings": []}},
                "camera": image_result,
            }
        )
        self.assertEqual(quality_result["media_quality"]["image"]["present"], True)
        self.assertEqual(quality_result["media_quality"]["image"]["usable"], True)

    def test_complete_output_schema_is_always_returned(self):
        img = _image(128)
        with patch.object(vision, "mp", _fake_mp(_fake_detector(has_face=True))), patch.object(vision, "cv2", _fake_cv2(imread_result=img)), patch.object(vision.os.path, "exists", return_value=True), patch.object(vision.os.path, "isdir", return_value=False), patch.object(vision.os.path, "isfile", return_value=True), patch.object(vision.os.path, "getsize", return_value=10):
            result = vision.analyze_face("ok.jpg")
        self.assertEqual(set(result.keys()), {"score", "details"})
        self._assert_success_schema(result)

    def test_bool_values_are_not_accepted_as_numeric_evidence(self):
        self.assertIsNone(vision._coerce_finite_number(True))
        self.assertIsNone(vision._coerce_normalized_number(True))
        self.assertIsNone(vision._coerce_positive_int(True))

    def test_top_level_score_matches_image_confidence_on_success(self):
        img = _image(128)
        with patch.object(vision, "mp", _fake_mp(_fake_detector(has_face=True))), patch.object(vision, "cv2", _fake_cv2(imread_result=img)), patch.object(vision.os.path, "exists", return_value=True), patch.object(vision.os.path, "isdir", return_value=False), patch.object(vision.os.path, "isfile", return_value=True), patch.object(vision.os.path, "getsize", return_value=10):
            result = vision.analyze_face("ok.jpg")
        self.assertEqual(result["score"], result["details"]["image_confidence"])

    def test_stale_metrics_do_not_survive_invalid_image(self):
        with patch.object(vision, "cv2", _fake_cv2(imread_result=None)), patch.object(vision.os.path, "exists", return_value=True), patch.object(vision.os.path, "isdir", return_value=False), patch.object(vision.os.path, "isfile", return_value=True), patch.object(vision.os.path, "getsize", return_value=10):
            result = vision.analyze_face("stale.jpg")
        self.assertEqual(result["details"], {"status": "invalid_image", "image_warnings": ["image_missing"]})
        self.assertIsNone(result["score"])

    def test_nonfinite_and_non_numeric_arrays_are_invalid_image(self):
        cases = {
            "nan": np.array([[[np.nan, 0.0, 0.0]]], dtype=float),
            "pos_inf": np.array([[[np.inf, 0.0, 0.0]]], dtype=float),
            "neg_inf": np.array([[[-np.inf, 0.0, 0.0]]], dtype=float),
            "object": np.array([[["bad", "bad", "bad"]]], dtype=object),
            "complex": np.array([[[1 + 2j, 0j, 0j]]], dtype=complex),
            "bool": np.array([[[True, False, True]]], dtype=bool),
        }
        with patch.object(vision, "cv2", _fake_cv2(imread_result=_image(128))):
            for name, array in cases.items():
                with self.subTest(name=name):
                    result = vision._analyze_image_array(array)
                    self.assertEqual(result["details"]["status"], "invalid_image")

    def test_detector_close_is_called_on_success(self):
        detector_cls = _fake_detector(has_face=True)
        img = _image(128)
        with patch.object(vision, "mp", _fake_mp(detector_cls)), patch.object(vision, "cv2", _fake_cv2(imread_result=img)), patch.object(vision.os.path, "exists", return_value=True), patch.object(vision.os.path, "isdir", return_value=False), patch.object(vision.os.path, "isfile", return_value=True), patch.object(vision.os.path, "getsize", return_value=10):
            vision.analyze_face("close-success.jpg")
        self.assertTrue(detector_cls.last_instance.close_called)

    def test_detector_close_is_called_when_no_face_is_found(self):
        detector_cls = _fake_detector(has_face=False)
        img = _image(128)
        with patch.object(vision, "mp", _fake_mp(detector_cls)), patch.object(vision, "cv2", _fake_cv2(imread_result=img)), patch.object(vision.os.path, "exists", return_value=True), patch.object(vision.os.path, "isdir", return_value=False), patch.object(vision.os.path, "isfile", return_value=True), patch.object(vision.os.path, "getsize", return_value=10):
            vision.analyze_face("close-noface.jpg")
        self.assertTrue(detector_cls.last_instance.close_called)

    def test_detector_close_is_called_when_process_raises(self):
        detector_cls = _fake_detector(has_face=True, raises=True)
        img = _image(128)
        with patch.object(vision, "mp", _fake_mp(detector_cls)), patch.object(vision, "cv2", _fake_cv2(imread_result=img)), patch.object(vision.os.path, "exists", return_value=True), patch.object(vision.os.path, "isdir", return_value=False), patch.object(vision.os.path, "isfile", return_value=True), patch.object(vision.os.path, "getsize", return_value=10):
            result = vision.analyze_face("close-error.jpg")
        self.assertTrue(detector_cls.last_instance.close_called)
        self.assertEqual(result["details"]["status"], "ok")
        self.assertIsNone(result["details"]["face_detected"])

    def test_detector_close_exception_is_suppressed(self):
        detector_cls = _fake_detector(has_face=True, close_raises=True)
        img = _image(128)
        with patch.object(vision, "mp", _fake_mp(detector_cls)), patch.object(vision, "cv2", _fake_cv2(imread_result=img)), patch.object(vision.os.path, "exists", return_value=True), patch.object(vision.os.path, "isdir", return_value=False), patch.object(vision.os.path, "isfile", return_value=True), patch.object(vision.os.path, "getsize", return_value=10):
            result = vision.analyze_face("close-raises.jpg")
        self.assertTrue(detector_cls.last_instance.close_called)
        self.assertEqual(result["details"]["status"], "ok")

    def test_import_does_not_mutate_cuda_visible_devices(self):
        module_name = "vision_cuda_probe"
        module_path = Path(vision.__file__).resolve()
        original = os.environ.get("CUDA_VISIBLE_DEVICES")
        try:
            os.environ["CUDA_VISIBLE_DEVICES"] = "sentinel-gpu"
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            self.assertIsNotNone(spec)
            self.assertIsNotNone(spec.loader)
            probe = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = probe
            try:
                spec.loader.exec_module(probe)
                self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), "sentinel-gpu")
            finally:
                sys.modules.pop(module_name, None)

            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            self.assertIsNotNone(spec)
            self.assertIsNotNone(spec.loader)
            probe = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = probe
            try:
                spec.loader.exec_module(probe)
                self.assertIsNone(os.environ.get("CUDA_VISIBLE_DEVICES"))
            finally:
                sys.modules.pop(module_name, None)
        finally:
            if original is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = original


if __name__ == "__main__":
    unittest.main()
