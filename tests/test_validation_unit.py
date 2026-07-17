import math
import os
from contextlib import contextmanager
from unittest import TestCase

import validation


_MISSING = object()
_VALIDATION_ENV_VARS = [
    "REQUIRE_VIDEO",
    "REQUIRE_AUDIO",
    "REQUIRE_FACE",
    "REQUIRE_PHRASE_MATCH",
    "REQUIRE_IMAGE",
    "PHRASE_MATCH_THRESHOLD",
    "MIN_VIDEO_SECONDS",
    "MIN_AUDIO_SECONDS",
    "MIN_FACE_VISIBLE_RATIO",
    "MIN_VIDEO_QUALITY",
    "MIN_AUDIO_QUALITY",
    "MIN_IMAGE_QUALITY",
]


@contextmanager
def isolated_validation_env(**overrides):
    original = {name: os.environ.get(name, _MISSING) for name in _VALIDATION_ENV_VARS}
    try:
        for name, value in overrides.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        yield
    finally:
        for name, value in original.items():
            if value is _MISSING:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _video_result(**overrides):
    details = {
        "status": "ok",
        "duration_seconds": 5.0,
        "brightness_score": 0.8,
        "sharpness_score": 0.8,
        "visual_quality_score": 0.82,
        "face_or_subject_visibility": 0.8,
        "face_frames": 20,
        "usable_frame_ratio": 0.8,
        "motion_stability_score": 0.8,
        "visual_warnings": [],
    }
    details.update(overrides)
    return {"score": 0.82, "details": details}


def _audio_result(**overrides):
    details = {
        "status": "ok",
        "duration_seconds": 4.0,
        "rms_energy": 0.03,
        "noise_estimate": 0.2,
        "speech_presence_score": 0.9,
        "usable_speech_detected": True,
        "audio_quality_score": 0.82,
        "audio_warnings": [],
    }
    details.update(overrides)
    return {"score": 0.84, "details": details}


def _image_result(**overrides):
    details = {
        "status": "ok",
        "brightness_score": 0.82,
        "sharpness_score": 0.84,
        "image_quality_score": 0.83,
        "face_detected": True,
        "image_warnings": [],
    }
    details.update(overrides)
    return {"score": 0.81, "details": details}


class ValidationPolicyUnitTests(TestCase):
    def _assert_result_invariants(self, result):
        self.assertIsInstance(result["passed"], bool)
        if result["passed"]:
            self.assertIsNone(result["failure_reason"])
            self.assertIsNone(result["failure_message"])
        else:
            self.assertIn(result["failure_reason"], validation.FAILURE_MESSAGES)
            self.assertEqual(result["failure_message"], validation.failure_message(result["failure_reason"]))

        for key in ["warnings", "critical_errors", "usable_modalities", "weak_modalities", "missing_modalities"]:
            values = result[key]
            self.assertEqual(values, list(dict.fromkeys(values)))

        combined_modalities = []
        for key in ["usable_modalities", "weak_modalities", "missing_modalities"]:
            combined_modalities.extend(result[key])
        self.assertEqual(len(combined_modalities), len(set(combined_modalities)))

        for value in (result.get("quality_scores") or {}).values():
            if value is None:
                continue
            self.assertTrue(math.isfinite(value))

        for penalty in result.get("quality_penalties") or []:
            self.assertTrue(math.isfinite(penalty["penalty"]))
            self.assertGreaterEqual(penalty["penalty"], 0.0)
        penalty_keys = [
            (penalty["modality"], penalty["reason"], penalty["penalty"])
            for penalty in result.get("quality_penalties") or []
        ]
        self.assertEqual(penalty_keys, list(dict.fromkeys(penalty_keys)))

    def test_default_policy_values(self):
        policy = validation.ValidationPolicy()
        self.assertTrue(policy.require_video)
        self.assertTrue(policy.require_audio)
        self.assertTrue(policy.require_face)
        self.assertFalse(policy.require_phrase_match)
        self.assertTrue(policy.require_image)
        self.assertEqual(policy.phrase_match_threshold, 0.80)
        self.assertEqual(policy.min_video_seconds, 3.0)
        self.assertEqual(policy.min_audio_seconds, 2.0)
        self.assertEqual(policy.min_face_visible_ratio, 0.50)
        self.assertEqual(policy.min_video_quality, 0.50)
        self.assertEqual(policy.min_audio_quality, 0.50)
        self.assertEqual(policy.min_image_quality, 0.50)

    def test_from_env_accepts_true_and_false_values(self):
        with isolated_validation_env(
            REQUIRE_VIDEO=" TRUE ",
            REQUIRE_AUDIO="off",
            REQUIRE_FACE=" Yes ",
            REQUIRE_PHRASE_MATCH=" no ",
            REQUIRE_IMAGE="1",
            PHRASE_MATCH_THRESHOLD=" 0.75 ",
            MIN_VIDEO_SECONDS="3",
            MIN_AUDIO_SECONDS="2.5",
            MIN_FACE_VISIBLE_RATIO="0.4",
            MIN_VIDEO_QUALITY="0.65",
            MIN_AUDIO_QUALITY="0.55",
            MIN_IMAGE_QUALITY="0.45",
        ):
            policy = validation.ValidationPolicy.from_env()
        self.assertTrue(policy.require_video)
        self.assertFalse(policy.require_audio)
        self.assertTrue(policy.require_face)
        self.assertFalse(policy.require_phrase_match)
        self.assertTrue(policy.require_image)
        self.assertEqual(policy.phrase_match_threshold, 0.75)
        self.assertEqual(policy.min_video_seconds, 3.0)
        self.assertEqual(policy.min_audio_seconds, 2.5)
        self.assertEqual(policy.min_face_visible_ratio, 0.4)
        self.assertEqual(policy.min_video_quality, 0.65)
        self.assertEqual(policy.min_audio_quality, 0.55)
        self.assertEqual(policy.min_image_quality, 0.45)

    def test_env_context_restores_original_values(self):
        original = os.environ.get("REQUIRE_VIDEO", _MISSING)
        with isolated_validation_env(REQUIRE_VIDEO="false"):
            self.assertEqual(os.environ.get("REQUIRE_VIDEO"), "false")
        restored = os.environ.get("REQUIRE_VIDEO", _MISSING)
        self.assertEqual(restored, original)

    def test_from_env_rejects_invalid_boolean_values(self):
        with isolated_validation_env(REQUIRE_VIDEO="maybe"):
            with self.assertRaises(ValueError):
                validation.ValidationPolicy.from_env()

    def test_from_env_rejects_invalid_numeric_values(self):
        cases = [
            {"PHRASE_MATCH_THRESHOLD": "1.1"},
            {"MIN_VIDEO_SECONDS": "-0.1"},
            {"MIN_AUDIO_SECONDS": "nan"},
            {"MIN_FACE_VISIBLE_RATIO": "infinity"},
            {"MIN_VIDEO_QUALITY": ""},
            {"MIN_AUDIO_QUALITY": "abc"},
        ]
        for overrides in cases:
            with self.subTest(overrides=overrides):
                with isolated_validation_env(**overrides):
                    with self.assertRaises(ValueError):
                        validation.ValidationPolicy.from_env()

    def test_direct_policy_construction_validates_types_and_ranges(self):
        cases = [
            {"require_video": "yes"},
            {"require_audio": 1},
            {"phrase_match_threshold": 1.1},
            {"phrase_match_threshold": -0.1},
            {"min_video_seconds": True},
            {"min_audio_seconds": float("nan")},
            {"min_face_visible_ratio": 1.5},
            {"min_video_quality": -0.01},
            {"min_audio_quality": float("inf")},
            {"min_image_quality": "0.5"},
        ]
        for overrides in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    validation.ValidationPolicy(**overrides)

    def test_all_media_missing_fails_with_missing_media(self):
        result = validation.validate_scan_inputs(
            policy=validation.ValidationPolicy(),
            media={"video": None, "audio": None, "image": None},
            video_result=None,
            audio_result=None,
            image_result=None,
            expected_phrase=None,
            transcript=None,
        )
        self.assertFalse(result["passed"])
        self.assertEqual(result["failure_reason"], "missing_media")
        self.assertEqual(result["critical_errors"], ["missing_media"])
        self.assertCountEqual(result["missing_modalities"], ["video", "audio", "image"])
        self._assert_result_invariants(result)

    def test_each_required_missing_modality_fails(self):
        cases = [
            ("video", {"video": None, "audio": "audio.wav", "image": "image.jpg"}, "video_missing"),
            ("audio", {"video": "video.mp4", "audio": None, "image": "image.jpg"}, "audio_missing"),
            ("image", {"video": "video.mp4", "audio": "audio.wav", "image": None}, "image_missing"),
        ]
        for required_modality, media, expected_reason in cases:
            with self.subTest(required_modality=required_modality):
                policy_kwargs = {"require_video": False, "require_audio": False, "require_image": False}
                policy_kwargs[f"require_{required_modality}"] = True
                result = validation.validate_scan_inputs(
                    policy=validation.ValidationPolicy(**policy_kwargs),
                    media=media,
                    video_result=_video_result(),
                    audio_result=_audio_result(),
                    image_result=_image_result(),
                    expected_phrase=None,
                    transcript=None,
                )
                self.assertFalse(result["passed"])
                self.assertEqual(result["failure_reason"], expected_reason)
                self.assertEqual(result["critical_errors"], [expected_reason])
                self._assert_result_invariants(result)

    def test_optional_missing_modality_does_not_fail(self):
        result = validation.validate_scan_inputs(
            policy=validation.ValidationPolicy(require_audio=False),
            media={"video": "video.mp4", "audio": None, "image": "image.jpg"},
            video_result=_video_result(),
            audio_result=None,
            image_result=_image_result(),
            expected_phrase=None,
            transcript=None,
        )
        self.assertTrue(result["passed"])
        self.assertIsNone(result["failure_reason"])
        self.assertEqual(result["critical_errors"], [])
        self._assert_result_invariants(result)

    def test_required_analyzer_result_none_fails_as_unreadable_media(self):
        result = validation.validate_scan_inputs(
            policy=validation.ValidationPolicy(),
            media={"video": "video.mp4", "audio": "audio.wav", "image": "image.jpg"},
            video_result=None,
            audio_result=_audio_result(),
            image_result=_image_result(),
            expected_phrase=None,
            transcript=None,
        )
        self.assertFalse(result["passed"])
        self.assertEqual(result["failure_reason"], "unreadable_media")
        self.assertEqual(result["critical_errors"], ["unreadable_media"])
        self._assert_result_invariants(result)

    def test_required_failure_status_with_stale_score_is_not_usable(self):
        result = validation.validate_video_result(
            validation.ValidationPolicy(),
            {"score": 0.9, "details": {"status": "missing", "visual_quality_score": 0.9, "visual_warnings": []}},
        )
        self.assertIn("video", result["missing_modalities"])
        self.assertNotIn("video", result["usable_modalities"])
        self.assertEqual(result["quality_scores"]["video"], 0.9)
        self._assert_result_invariants(result)

    def test_optional_unreadable_modality_is_recorded_without_failure(self):
        result = validation.validate_image_result(
            validation.ValidationPolicy(require_image=False),
            {"score": None, "details": {"status": "invalid_image", "image_warnings": []}},
            image_required=False,
        )
        self.assertTrue(result["passed"])
        self.assertIsNone(result["failure_reason"])
        self.assertIn("image", result["missing_modalities"])
        self.assertNotIn("image", result["usable_modalities"])
        self._assert_result_invariants(result)

    def test_malformed_numeric_analyzer_values_do_not_crash(self):
        result = validation.validate_video_result(
            validation.ValidationPolicy(),
            {
                "score": 0.3,
                "details": {
                    "status": "ok",
                    "duration_seconds": "not-a-number",
                    "brightness_score": "bad",
                    "sharpness_score": None,
                    "visual_quality_score": "also-bad",
                    "face_or_subject_visibility": "bad",
                    "face_frames": "bad",
                    "usable_frame_ratio": "bad",
                    "motion_stability_score": "bad",
                    "visual_warnings": [],
                },
            },
        )
        self.assertIsInstance(result["passed"], bool)
        self.assertIsNone(result["quality_scores"]["video"])
        self._assert_result_invariants(result)

    def test_nan_and_infinity_do_not_survive_quality_scores(self):
        result = validation.validate_audio_result(
            validation.ValidationPolicy(),
            {
                "score": 0.5,
                "details": {
                    "status": "ok",
                    "audio_quality_score": float("inf"),
                    "duration_seconds": 4.0,
                    "rms_energy": 0.03,
                    "speech_presence_score": 0.9,
                    "usable_speech_detected": True,
                    "audio_warnings": [],
                },
            },
            speech_required=False,
        )
        self.assertIsNone(result["quality_scores"]["audio"])
        self._assert_result_invariants(result)

    def test_unreadable_status_contract_is_fail_closed(self):
        cases = [
            (
                "video unknown status with stale score",
                validation.validate_video_result,
                validation.ValidationPolicy(),
                _video_result(status="corrupt", score=0.99, visual_quality_score=0.99, duration_seconds=5.0),
                "video",
            ),
            (
                "audio failed status",
                validation.validate_audio_result,
                validation.ValidationPolicy(),
                _audio_result(status="failed", score=0.99, audio_quality_score=0.99, duration_seconds=4.0),
                "audio",
            ),
            (
                "image processing status",
                validation.validate_image_result,
                validation.ValidationPolicy(),
                _image_result(status="processing", score=0.99, image_quality_score=0.99),
                "image",
            ),
            (
                "normalized ok status remains accepted",
                validation.validate_video_result,
                validation.ValidationPolicy(),
                _video_result(status=" OK ", visual_quality_score=0.82, duration_seconds=5.0),
                "video",
            ),
        ]
        for label, validator, policy, payload, modality in cases:
            with self.subTest(label=label):
                if validator is validation.validate_image_result:
                    result = validator(policy, payload, image_required=False)
                elif validator is validation.validate_audio_result:
                    result = validator(policy, payload, speech_required=False)
                else:
                    result = validator(policy, payload)
                if label == "normalized ok status remains accepted":
                    self.assertIn(modality, result["usable_modalities"])
                    self.assertNotIn(modality, result["missing_modalities"])
                else:
                    self.assertIn(modality, result["missing_modalities"])
                    self.assertNotIn(modality, result["usable_modalities"])
                self._assert_result_invariants(result)

    def test_video_minimum_evidence_contract(self):
        cases = [
            (
                "usable evidence passes",
                _video_result(status="ok", visual_quality_score=0.82, duration_seconds=5.0),
                True,
            ),
            (
                "missing quality is unreadable",
                _video_result(status="ok", visual_quality_score=None, duration_seconds=5.0, score=0.99),
                False,
            ),
            (
                "nan quality is unreadable",
                _video_result(status="ok", visual_quality_score=float("nan"), duration_seconds=5.0),
                False,
            ),
            (
                "missing duration is unreadable",
                _video_result(status="ok", visual_quality_score=0.82, duration_seconds=None),
                False,
            ),
            (
                "negative duration is unreadable",
                _video_result(status="ok", visual_quality_score=0.82, duration_seconds=-1.0),
                False,
            ),
        ]
        for label, payload, expected_usable in cases:
            with self.subTest(label=label):
                result = validation.validate_video_result(validation.ValidationPolicy(), payload)
                if expected_usable:
                    self.assertIn("video", result["usable_modalities"])
                    self.assertNotIn("video", result["missing_modalities"])
                else:
                    self.assertIn("video", result["missing_modalities"])
                    self.assertNotIn("video", result["usable_modalities"])
                self._assert_result_invariants(result)

    def test_audio_minimum_evidence_contract(self):
        cases = [
            (
                "usable evidence passes",
                _audio_result(status="ok", audio_quality_score=0.82, duration_seconds=None, duration_sec=4.0),
                True,
            ),
            (
                "missing quality is unreadable",
                _audio_result(status="ok", audio_quality_score=None, duration_seconds=4.0),
                False,
            ),
            (
                "nan quality is unreadable",
                _audio_result(status="ok", audio_quality_score=float("nan"), duration_seconds=4.0),
                False,
            ),
            (
                "missing duration is unreadable",
                _audio_result(status="ok", audio_quality_score=0.82, duration_seconds=None, duration_sec=None),
                False,
            ),
            (
                "negative duration is unreadable",
                _audio_result(status="ok", audio_quality_score=0.82, duration_seconds=-1.0),
                False,
            ),
        ]
        for label, payload, expected_usable in cases:
            with self.subTest(label=label):
                result = validation.validate_audio_result(validation.ValidationPolicy(), payload, speech_required=False)
                if expected_usable:
                    self.assertIn("audio", result["usable_modalities"])
                    self.assertNotIn("audio", result["missing_modalities"])
                else:
                    self.assertIn("audio", result["missing_modalities"])
                    self.assertNotIn("audio", result["usable_modalities"])
                self._assert_result_invariants(result)

    def test_image_minimum_evidence_contract(self):
        cases = [
            (
                "usable evidence passes",
                _image_result(status="ok", image_quality_score=0.83),
                True,
            ),
            (
                "missing quality is unreadable",
                _image_result(status="ok", image_quality_score=None, score=0.99),
                False,
            ),
            (
                "nan quality is unreadable",
                _image_result(status="ok", image_quality_score=float("nan")),
                False,
            ),
            (
                "infinity quality is unreadable",
                _image_result(status="ok", image_quality_score=float("inf")),
                False,
            ),
        ]
        for label, payload, expected_usable in cases:
            with self.subTest(label=label):
                result = validation.validate_image_result(validation.ValidationPolicy(), payload, image_required=False)
                if expected_usable:
                    self.assertIn("image", result["usable_modalities"])
                    self.assertNotIn("image", result["missing_modalities"])
                else:
                    self.assertIn("image", result["missing_modalities"])
                    self.assertNotIn("image", result["usable_modalities"])
                self._assert_result_invariants(result)

    def test_finalize_validation_result_normalizes_failure_reason(self):
        cases = [
            (None, "analysis_exception"),
            ([], "analysis_exception"),
            ({}, "analysis_exception"),
            (object(), "analysis_exception"),
            ("unknown_string", "analysis_exception"),
            ("missing_media", "missing_media"),
        ]
        for reason, expected in cases:
            with self.subTest(reason=reason):
                result = validation.make_validation_result()
                result["passed"] = False
                result["failure_reason"] = reason
                finalized = validation._finalize_validation_result(result)
                self.assertEqual(finalized["failure_reason"], expected)
                self.assertEqual(finalized["failure_message"], validation.failure_message(expected))

    def test_finalize_validation_result_sanitizes_quality_penalties(self):
        result = validation.make_validation_result()
        result["passed"] = False
        result["failure_reason"] = "missing_media"
        result["quality_penalties"] = [
            {"modality": "video", "reason": "audio_too_noisy", "penalty": 0.15},
            {"modality": "video", "reason": "audio_too_noisy", "penalty": 0.15},
            {"modality": "audio", "reason": "low_quality_media", "penalty": 0.1},
            {"modality": "audio", "reason": "negative", "penalty": -0.1},
            {"modality": "audio", "reason": "nan", "penalty": float("nan")},
            {"modality": "audio", "reason": "infinity", "penalty": float("inf")},
            {"modality": "audio", "reason": "bool", "penalty": True},
            {"modality": "audio", "reason": "", "penalty": 0.1},
            {"reason": "missing modality", "penalty": 0.1},
            "not-a-dict",
            {"modality": "image", "reason": "image_blurry", "penalty": 0.12},
        ]
        finalized = validation._finalize_validation_result(result)
        self.assertEqual(
            finalized["quality_penalties"],
            [
                {"modality": "video", "reason": "audio_too_noisy", "penalty": 0.15},
                {"modality": "audio", "reason": "low_quality_media", "penalty": 0.1},
                {"modality": "image", "reason": "image_blurry", "penalty": 0.12},
            ],
        )
        self._assert_result_invariants(finalized)

    def test_modality_state_transitions_are_exclusive(self):
        result = validation.make_validation_result()
        validation._add_modality_state(result, "video", "usable")
        validation._add_modality_state(result, "video", "weak")
        self.assertEqual(result["usable_modalities"], [])
        self.assertEqual(result["weak_modalities"], ["video"])
        validation._add_modality_state(result, "video", "missing")
        self.assertEqual(result["usable_modalities"], [])
        self.assertEqual(result["weak_modalities"], [])
        self.assertEqual(result["missing_modalities"], ["video"])

    def test_merged_sub_results_keep_states_exclusive(self):
        target = validation.make_validation_result()
        source = {
            "warnings": [],
            "critical_errors": [],
            "quality_penalties": [],
            "usable_modalities": ["audio"],
            "weak_modalities": ["audio"],
            "missing_modalities": ["audio"],
        }
        validation._merge_validation_metadata(target, source)
        self.assertEqual(target["usable_modalities"], [])
        self.assertEqual(target["weak_modalities"], [])
        self.assertEqual(target["missing_modalities"], ["audio"])

    def test_duplicate_warnings_and_penalties_are_removed(self):
        result = validation.validate_audio_result(
            validation.ValidationPolicy(),
            {
                "score": 0.9,
                "details": {
                    "status": "ok",
                    "audio_quality_score": 0.9,
                    "duration_seconds": 4.0,
                    "rms_energy": 0.03,
                    "speech_presence_score": 0.9,
                    "usable_speech_detected": True,
                    "noise_estimate": 0.95,
                    "audio_warnings": ["audio_too_noisy", "audio_too_noisy"],
                },
            },
            speech_required=False,
        )
        self.assertEqual(result["warnings"], ["audio_too_noisy"])
        self.assertEqual(result["quality_penalties"], [{"modality": "audio", "reason": "audio_too_noisy", "penalty": 0.15}])
        self._assert_result_invariants(result)

    def test_sustained_eye_closure_alone_does_not_weaken_video(self):
        result = validation.validate_video_result(
            validation.ValidationPolicy(),
            {
                "score": 0.9,
                "details": {
                    "status": "ok",
                    "duration_seconds": 5.0,
                    "visual_quality_score": 0.9,
                    "face_or_subject_visibility": 0.9,
                    "face_frames": 20,
                    "usable_frame_ratio": 0.9,
                    "motion_stability_score": 0.9,
                    "sustained_eye_closure": True,
                    "visual_warnings": [],
                },
            },
        )
        self.assertIn("sustained_eye_closure", result["warnings"])
        self.assertEqual(result["quality_penalties"], [])
        self.assertIn("video", result["usable_modalities"])
        self.assertNotIn("video", result["weak_modalities"])
        self._assert_result_invariants(result)

    def test_capture_warnings_still_weaken_video(self):
        result = validation.validate_video_result(
            validation.ValidationPolicy(),
            {
                "score": 0.2,
                "details": {
                    "status": "ok",
                    "duration_seconds": 1.0,
                    "brightness_score": 0.2,
                    "sharpness_score": 0.2,
                    "visual_quality_score": 0.2,
                    "face_or_subject_visibility": 0.2,
                    "face_frames": 0,
                    "usable_frame_ratio": 0.1,
                    "motion_stability_score": 0.1,
                    "visual_warnings": [],
                },
            },
        )
        self.assertIn("video", result["weak_modalities"])
        self.assertTrue(result["quality_penalties"])
        self._assert_result_invariants(result)

    def test_speech_required_detects_missing_speech(self):
        result = validation.validate_audio_result(
            validation.ValidationPolicy(),
            {
                "score": 0.8,
                "details": {
                    "status": "ok",
                    "duration_seconds": 4.0,
                    "audio_quality_score": 0.8,
                    "speech_presence_score": 0.0,
                    "usable_speech_detected": False,
                    "audio_warnings": [],
                },
            },
            speech_required=True,
        )
        self.assertIn("speech_not_detected", result["warnings"])
        self.assertIn("audio", result["weak_modalities"])
        self._assert_result_invariants(result)

    def test_optional_speech_absence_is_not_penalized(self):
        result = validation.validate_audio_result(
            validation.ValidationPolicy(),
            {
                "score": 0.8,
                "details": {
                    "status": "ok",
                    "duration_seconds": 4.0,
                    "audio_quality_score": 0.8,
                    "speech_presence_score": 0.0,
                    "usable_speech_detected": False,
                    "audio_warnings": ["speech_not_detected"],
                },
            },
            speech_required=False,
        )
        self.assertNotIn("speech_not_detected", result["warnings"])
        self.assertNotIn("audio", result["weak_modalities"])
        self._assert_result_invariants(result)

    def test_optional_audio_retains_technical_warnings(self):
        result = validation.validate_audio_result(
            validation.ValidationPolicy(),
            {
                "score": 0.4,
                "details": {
                    "status": "ok",
                    "duration_seconds": 4.0,
                    "audio_quality_score": 0.4,
                    "noise_estimate": 0.95,
                    "clipping_ratio": 0.02,
                    "silence_ratio": 0.7,
                    "audio_warnings": [],
                },
            },
            speech_required=False,
        )
        self.assertIn("audio_too_noisy", result["warnings"])
        self.assertIn("audio_clipping", result["warnings"])
        self.assertIn("too_much_silence", result["warnings"])
        self.assertIn("audio", result["weak_modalities"])
        self._assert_result_invariants(result)

    def test_required_expected_phrase_missing_is_critical(self):
        result = validation.validate_phrase_result(validation.ValidationPolicy(require_phrase_match=True), None, None)
        self.assertFalse(result["passed"])
        self.assertEqual(result["failure_reason"], "expected_phrase_missing")
        self.assertEqual(result["critical_errors"], ["expected_phrase_missing"])
        self._assert_result_invariants(result)

    def test_required_transcript_missing_is_critical(self):
        result = validation.validate_phrase_result(
            validation.ValidationPolicy(require_phrase_match=True),
            "please say continuity ready",
            None,
        )
        self.assertFalse(result["passed"])
        self.assertEqual(result["failure_reason"], "transcription_failed")
        self.assertEqual(result["critical_errors"], ["transcription_failed"])
        self._assert_result_invariants(result)

    def test_required_phrase_mismatch_is_critical(self):
        result = validation.validate_phrase_result(
            validation.ValidationPolicy(require_phrase_match=True),
            "please say continuity ready",
            "something else entirely",
        )
        self.assertFalse(result["passed"])
        self.assertEqual(result["failure_reason"], "phrase_mismatch")
        self.assertEqual(result["critical_errors"], ["phrase_mismatch"])
        self._assert_result_invariants(result)

    def test_successful_phrase_match_passes(self):
        result = validation.validate_phrase_result(
            validation.ValidationPolicy(require_phrase_match=True),
            "Please say continuity ready",
            "please say continuity ready",
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["quality_scores"]["phrase_match"], 1.0)
        self._assert_result_invariants(result)

    def test_phrase_match_score_exact_match_is_one(self):
        self.assertEqual(
            validation.phrase_match_score("Please say continuity ready", "please say continuity ready"),
            1.0,
        )

    def test_phrase_match_score_empty_input_is_zero(self):
        self.assertEqual(validation.phrase_match_score("", "hello"), 0.0)
        self.assertEqual(validation.phrase_match_score("hello", ""), 0.0)

    def test_phrase_match_score_stays_in_unit_interval(self):
        score = validation.phrase_match_score("please say continuity ready", "please say continuity ready please")
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_phrase_match_score_repeated_unrelated_words_do_not_pass_threshold(self):
        score = validation.phrase_match_score(
            "please say continuity ready",
            "please say continuity ready unrelated unrelated unrelated unrelated",
        )
        self.assertLess(score, validation.ValidationPolicy().phrase_match_threshold)

    def test_failure_message_handles_invalid_reason_types(self):
        expected = validation.FAILURE_MESSAGES["analysis_exception"]
        self.assertEqual(validation.failure_message(None), expected)
        self.assertEqual(validation.failure_message(object()), expected)
        self.assertEqual(validation.failure_message([]), expected)

    def test_public_validators_satisfy_invariants(self):
        results = [
            validation.make_validation_result(),
            validation.fail_validation("missing_media"),
            validation.validate_video_result(validation.ValidationPolicy(), _video_result()),
            validation.validate_audio_result(validation.ValidationPolicy(), _audio_result(), speech_required=True),
            validation.validate_image_result(validation.ValidationPolicy(), _image_result(), image_required=True),
            validation.validate_phrase_result(validation.ValidationPolicy(require_phrase_match=True), "please say continuity ready", "please say continuity ready"),
            validation.validate_scan_inputs(
                policy=validation.ValidationPolicy(),
                media={"video": "video.mp4", "audio": "audio.wav", "image": "image.jpg"},
                video_result=_video_result(),
                audio_result=_audio_result(),
                image_result=_image_result(),
                expected_phrase="please say continuity ready",
                transcript="please say continuity ready",
            ),
        ]
        for result in results:
            with self.subTest(result=result):
                self._assert_result_invariants(result)
