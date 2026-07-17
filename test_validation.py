from __future__ import annotations

import unittest
from unittest.mock import ANY, MagicMock, patch

from fastapi import HTTPException
from fastapi.testclient import TestClient

from validation import ValidationPolicy, validate_scan_inputs
import scoring

import main
import audio
import video


def _video_result(**overrides):
    details = {
        "status": "ok",
        "duration_seconds": 5.0,
        "brightness_score": 0.8,
        "sharpness_score": 0.8,
        "visual_quality_score": 0.8,
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
        "speech_state": "usable_speech",
        "quiet_but_usable": False,
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


def _analysis_dispatcher(*, video_result, image_result, audio_result):
    results = {
        "video_missing": video_result,
        "image_missing": image_result,
        "audio_missing": audio_result,
    }

    def dispatch(_analyzer, _path, missing_warning):
        return results[missing_warning]

    return dispatch


def _media_input_dispatcher():
    paths = {
        "image": ("image.jpg", False),
        "audio": ("audio.wav", False),
        "video": ("video.mp4", False),
    }

    def dispatch(_value, _suffix, media_kind, **_kwargs):
        return paths[media_kind]

    return dispatch


class ValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = ValidationPolicy()
        self.media = {"video": "video.mp4", "audio": "audio.wav", "image": "image.jpg"}

    def test_missing_video(self):
        result = validate_scan_inputs(
            policy=self.policy,
            media={"video": None, "audio": "audio.wav", "image": None},
            video_result=None,
            audio_result=_audio_result(),
            image_result=None,
            expected_phrase="hello world",
            transcript="hello world",
        )
        self.assertFalse(result["passed"])
        self.assertEqual(result["failure_reason"], "video_missing")
        self.assertEqual(result["critical_errors"], ["video_missing"])
        self.assertIn("video_missing", result["warnings"])

    def test_missing_audio(self):
        result = validate_scan_inputs(
            policy=self.policy,
            media={"video": "video.mp4", "audio": None, "image": None},
            video_result=_video_result(),
            audio_result=None,
            image_result=None,
            expected_phrase="hello world",
            transcript=None,
        )
        self.assertFalse(result["passed"])
        self.assertEqual(result["failure_reason"], "audio_missing")
        self.assertEqual(result["critical_errors"], ["audio_missing"])
        self.assertIn("audio_missing", result["warnings"])

    def test_missing_face(self):
        result = validate_scan_inputs(
            policy=self.policy,
            media=self.media,
            video_result=_video_result(face_frames=0, face_or_subject_visibility=0.0),
            audio_result=_audio_result(),
            image_result=_image_result(face_detected=False),
            expected_phrase="hello world",
            transcript="hello world",
        )
        self.assertTrue(result["passed"])
        self.assertIn("face_not_visible", result["warnings"])

    def test_missing_image(self):
        result = validate_scan_inputs(
            policy=self.policy,
            media={"video": "video.mp4", "audio": "audio.wav", "image": None},
            video_result=_video_result(),
            audio_result=_audio_result(),
            image_result=None,
            expected_phrase="hello world",
            transcript="hello world",
        )
        self.assertFalse(result["passed"])
        self.assertEqual(result["failure_reason"], "image_missing")
        self.assertEqual(result["critical_errors"], ["image_missing"])
        self.assertIn("image_missing", result["warnings"])

    def test_blurry_video(self):
        result = validate_scan_inputs(
            policy=self.policy,
            media=self.media,
            video_result=_video_result(sharpness_score=0.2, visual_warnings=["video_blurry"]),
            audio_result=_audio_result(),
            image_result=_image_result(),
            expected_phrase="hello world",
            transcript="hello world",
        )
        self.assertTrue(result["passed"])
        self.assertIn("video_blurry", result["warnings"])
        self.assertIn("video", result["weak_modalities"])
        self.assertIn(
            {"modality": "video", "reason": "video_blurry", "penalty": 0.15},
            result["quality_penalties"],
        )

    def test_dark_video(self):
        result = validate_scan_inputs(
            policy=self.policy,
            media=self.media,
            video_result=_video_result(brightness_score=0.2, visual_warnings=["video_too_dark"]),
            audio_result=_audio_result(),
            image_result=_image_result(),
            expected_phrase="hello world",
            transcript="hello world",
        )
        self.assertTrue(result["passed"])
        self.assertIn("video_too_dark", result["warnings"])
        self.assertIn("video", result["weak_modalities"])

    def test_noisy_audio(self):
        result = validate_scan_inputs(
            policy=self.policy,
            media=self.media,
            video_result=_video_result(),
            audio_result=_audio_result(noise_estimate=0.9, audio_warnings=["audio_too_noisy"]),
            image_result=_image_result(),
            expected_phrase="hello world",
            transcript="hello world",
        )
        self.assertTrue(result["passed"])
        self.assertIn("audio_too_noisy", result["warnings"])
        self.assertIn("audio", result["weak_modalities"])
        self.assertIn(
            {"modality": "audio", "reason": "audio_too_noisy", "penalty": 0.15},
            result["quality_penalties"],
        )

    def test_phrase_mismatch(self):
        result = validate_scan_inputs(
            policy=self.policy,
            media=self.media,
            video_result=_video_result(),
            audio_result=_audio_result(),
            image_result=_image_result(),
            expected_phrase="please say continuity ready",
            transcript="something else entirely",
        )
        self.assertTrue(result["passed"])
        self.assertIn("phrase_mismatch", result["warnings"])

    def test_valid_phrase_match(self):
        result = validate_scan_inputs(
            policy=self.policy,
            media=self.media,
            video_result=_video_result(),
            audio_result=_audio_result(),
            image_result=_image_result(),
            expected_phrase="Please say continuity ready",
            transcript="please say continuity ready",
        )
        self.assertTrue(result["passed"])
        self.assertGreaterEqual(result["quality_scores"]["phrase_match"], 0.8)

    def test_low_quality_media_warning(self):
        result = validate_scan_inputs(
            policy=self.policy,
            media=self.media,
            video_result=_video_result(visual_quality_score=0.2, usable_frame_ratio=0.2, motion_stability_score=0.2),
            audio_result=_audio_result(),
            image_result=_image_result(),
            expected_phrase="hello world",
            transcript="hello world",
        )
        self.assertTrue(result["passed"])
        self.assertTrue({"unstable_video", "low_quality_media"} & set(result["warnings"]))
        self.assertEqual(result["critical_errors"], [])
        self.assertIn("video", result["weak_modalities"])

    def test_quiet_but_usable_audio_is_not_marked_low_quality(self):
        result = validate_scan_inputs(
            policy=self.policy,
            media=self.media,
            video_result=_video_result(),
            audio_result=_audio_result(rms_energy=0.0105, energy=0.0105, speech_presence_score=0.65, audio_quality_score=0.58, audio_warnings=[], quiet_but_usable=True),
            image_result=_image_result(),
            expected_phrase="hello world",
            transcript="hello world",
        )
        self.assertTrue(result["passed"])
        self.assertNotIn("audio_too_quiet", result["warnings"])
        self.assertNotIn("low_quality_media", result["warnings"])

    def test_no_usable_speech_in_required_scan_fails(self):
        scan_context = {
            "status": "media_ready",
            "scan_media": {"video_file": "vid-1", "audio_file": "aud-1", "thumbnail": "img-1"},
            "resolved_media": {"video": "vid-1", "audio": "aud-1", "image": "img-1"},
            "task_metrics": {},
            "expected_phrase": "please say continuity ready",
        }
        with patch.object(main, "_resolve_scan_context", return_value=scan_context), patch.object(
            main.directus,
            "get_scan_media",
            return_value=scan_context["scan_media"],
        ), patch.object(
            main.ml_runtime,
            "is_loaded",
            return_value=False,
        ), patch.object(
            main.ml_runtime,
            "local_model_required",
            return_value=False,
        ), patch.object(
            main,
            "_resolve_media_input",
            side_effect=_media_input_dispatcher(),
        ), patch.object(
            main,
            "_safe_analyze",
            side_effect=_analysis_dispatcher(
                video_result=_video_result(),
                image_result=_image_result(),
                audio_result=_audio_result(
                    speech_presence_score=0.1,
                    usable_speech_detected=False,
                    speech_state="no_usable_speech",
                    audio_quality_score=0.2,
                    audio_warnings=["speech_not_detected"],
                ),
            ),
        ), patch.object(
            main,
            "_expected_phrase",
            return_value="please say continuity ready",
        ), patch.object(
            main,
            "_transcribe_audio_file",
            return_value=None,
        ), patch.object(
            main,
            "_baseline_rows_for_member",
            return_value=[],
        ), patch.object(
            main,
            "_write_success",
            return_value={"scan_result": "created:1"},
        ) as write_mock:
            result = main._process_scan_sync("scan-123")

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["failure_reason"], "low_quality_media")
        write_mock.assert_not_called()

    def test_validation_passed(self):
        result = validate_scan_inputs(
            policy=self.policy,
            media=self.media,
            video_result=_video_result(),
            audio_result=_audio_result(),
            image_result=_image_result(),
            expected_phrase="hello world",
            transcript="hello world",
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["failure_reason"], None)
        self.assertEqual(result["critical_errors"], [])
        self.assertCountEqual(result["usable_modalities"], ["video", "audio", "image"])
        self.assertEqual(result["weak_modalities"], [])
        self.assertEqual(result["missing_modalities"], [])
        self.assertEqual(result["quality_penalties"], [])

    def test_all_media_missing_is_critical(self):
        result = validate_scan_inputs(
            policy=self.policy,
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

    def test_all_present_media_unreadable_is_critical(self):
        result = validate_scan_inputs(
            policy=self.policy,
            media=self.media,
            video_result={"score": None, "details": {"status": "open_failed", "visual_warnings": []}},
            audio_result={"score": None, "details": {"status": "load_failed", "audio_warnings": []}},
            image_result={"score": None, "details": {"status": "invalid_image", "image_warnings": []}},
            expected_phrase=None,
            transcript=None,
        )

        self.assertFalse(result["passed"])
        self.assertEqual(result["failure_reason"], "unreadable_media")
        self.assertEqual(result["critical_errors"], ["unreadable_media"])
        self.assertCountEqual(result["missing_modalities"], ["video", "audio", "image"])

    def test_expected_phrase_missing_is_allowed_when_not_required(self):
        policy = ValidationPolicy(require_phrase_match=False)
        result = validate_scan_inputs(
            policy=policy,
            media={"video": "video.mp4", "audio": "audio.wav", "image": "image.jpg"},
            video_result=_video_result(),
            audio_result=_audio_result(),
            image_result=_image_result(),
            expected_phrase=None,
            transcript=None,
        )
        self.assertTrue(result["passed"])

    def test_speech_absence_is_not_penalized_when_not_required(self):
        result = validate_scan_inputs(
            policy=ValidationPolicy(require_phrase_match=False),
            media=self.media,
            video_result=_video_result(),
            audio_result=_audio_result(speech_presence_score=0.05, audio_quality_score=0.82, audio_warnings=["speech_not_detected"]),
            image_result=_image_result(),
            expected_phrase=None,
            transcript=None,
        )

        self.assertTrue(result["passed"])
        self.assertNotIn("speech_not_detected", result["warnings"])
        self.assertNotIn("audio", result["weak_modalities"])

    def test_audio_timeout_placeholder_is_fail_closed(self):
        result = main._analysis_timeout_placeholder("audio")

        self.assertIsNone(result["score"])
        self.assertEqual(result["details"]["status"], "load_failed")
        self.assertIn("audio_timeout", result["details"]["audio_warnings"])


class MainPayloadTests(unittest.TestCase):
    def _run_scan_with_analysis(self, *, video_result, image_result, audio_result):
        scan_context = {
            "status": "media_ready",
            "scan_media": {"video_file": "vid-1", "audio_file": "aud-1", "thumbnail": "img-1"},
            "resolved_media": {"video": "vid-1", "audio": "aud-1", "image": "img-1"},
            "task_metrics": {},
        }
        with patch.object(main, "_resolve_scan_context", return_value=scan_context), patch.object(
            main.directus,
            "get_scan_media",
            return_value=scan_context["scan_media"],
        ), patch.object(
            main.ml_runtime,
            "is_loaded",
            return_value=False,
        ), patch.object(
            main.ml_runtime,
            "local_model_required",
            return_value=False,
        ), patch.object(
            main,
            "_resolve_media_input",
            side_effect=_media_input_dispatcher(),
        ), patch.object(
            main,
            "_safe_analyze",
            side_effect=_analysis_dispatcher(
                video_result=video_result,
                image_result=image_result,
                audio_result=audio_result,
            ),
        ), patch.object(
            main,
            "_expected_phrase",
            return_value=None,
        ), patch.object(
            main,
            "_baseline_rows_for_member",
            return_value=[],
        ), patch.object(
            main,
            "_write_success",
            return_value={"scan_result": "created:1"},
        ) as write_mock:
            result = main._process_scan_sync("scan-123")
        return result, write_mock

    def test_blurry_video_still_creates_result(self):
        result, write_mock = self._run_scan_with_analysis(
            video_result=_video_result(sharpness_score=0.2, visual_quality_score=0.25, visual_warnings=["video_blurry"]),
            image_result=_image_result(),
            audio_result=_audio_result(),
        )

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["failure_reason"], "low_quality_media")
        write_mock.assert_not_called()

    def test_dark_video_still_creates_result(self):
        result, write_mock = self._run_scan_with_analysis(
            video_result=_video_result(brightness_score=0.2, visual_quality_score=0.25, visual_warnings=["video_too_dark"]),
            image_result=_image_result(),
            audio_result=_audio_result(),
        )

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["failure_reason"], "low_quality_media")
        write_mock.assert_not_called()

    def test_noisy_audio_fails_when_audio_is_required(self):
        result, write_mock = self._run_scan_with_analysis(
            video_result=_video_result(),
            image_result=_image_result(),
            audio_result=_audio_result(
                noise_estimate=0.9,
                audio_quality_score=0.25,
                audio_warnings=["audio_too_noisy"],
                usable_speech_detected=False,
                speech_state="no_usable_speech",
                quiet_but_usable=False,
            ),
        )

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["failure_reason"], "low_quality_media")
        write_mock.assert_not_called()

    def test_weak_face_visibility_still_creates_result(self):
        result, write_mock = self._run_scan_with_analysis(
            video_result=_video_result(face_frames=1, face_or_subject_visibility=0.1, usable_frame_ratio=0.25),
            image_result=_image_result(face_detected=False),
            audio_result=_audio_result(),
        )

        self.assertEqual(result["status"], "completed")
        write_mock.assert_called_once()
        written_result = write_mock.call_args.kwargs["result"]
        self.assertTrue({"face_not_visible", "unstable_video"} & set(written_result["validation_warnings"]))
        self.assertIn("face visibility", written_result["explanation"].lower())

    def test_single_poor_frame_does_not_reduce_valid_scan(self):
        result, write_mock = self._run_scan_with_analysis(
            video_result=_video_result(
                visual_quality_score=0.82,
                visual_warnings=[],
                sampled_frames=8,
                blurry_frames=1,
                low_light_frames=0,
                reliable_eye_landmarks=True,
                sustained_eye_closure=False,
            ),
            image_result=_image_result(),
            audio_result=_audio_result(),
        )

        self.assertEqual(result["status"], "completed")
        written_result = write_mock.call_args.kwargs["result"]
        self.assertIn(written_result["risk_level"], {"stable", "low_focus", "elevated_fatigue", "high_risk"})
        self.assertNotIn("sustained_eye_closure", written_result["validation_warnings"])

    def test_sustained_eye_closure_with_reliable_evidence_is_not_stable(self):
        result, write_mock = self._run_scan_with_analysis(
            video_result=_video_result(
                visual_quality_score=0.8,
                visual_warnings=["sustained_eye_closure"],
                reliable_eye_landmarks=True,
                sustained_eye_closure=True,
                motion_stability_score=0.8,
                eye_closure_sample_count=8,
                closed_eye_ratio=0.75,
                longest_eye_closure_streak=4,
                eye_closure_window_ms=600,
                eye_closure_window_seconds=0.6,
                avg_eye_aperture=0.16,
                eye_aperture_std=0.02,
            ),
            image_result=_image_result(),
            audio_result=_audio_result(),
        )

        self.assertEqual(result["status"], "completed")
        written_result = write_mock.call_args.kwargs["result"]
        self.assertNotEqual(written_result["risk_level"], "stable")
        self.assertIn("eye closure", written_result["explanation"].lower())

    def test_completely_missing_media_fails_without_writing_scan_result(self):
        scan_context = {
            "status": "media_ready",
            "scan_media": {"video_file": None, "audio_file": None, "thumbnail": None},
            "resolved_media": {"video": None, "audio": None, "image": None},
            "task_metrics": {},
        }
        with patch.object(main, "_resolve_scan_context", return_value=scan_context), patch.object(
            main.directus,
            "get_scan_media",
            return_value=scan_context["scan_media"],
        ), patch.object(
            main.ml_runtime,
            "is_loaded",
            return_value=False,
        ), patch.object(
            main.ml_runtime,
            "local_model_required",
            return_value=False,
        ), patch.object(
            main,
            "_baseline_rows_for_member",
            return_value=[],
        ), patch.object(
            main.directus,
            "upsert_scan_result",
            return_value=("created", {"id": "result-1"}),
        ) as upsert_mock, patch.object(
            main.directus,
            "update_wellness_scan",
            return_value={},
        ) as update_scan_mock, patch.object(
            main,
            "_write_success",
            return_value={"scan_result": "created:1"},
        ) as write_mock:
            result = main._process_scan_sync("scan-123")

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["failure_reason"], "video_missing")
        write_mock.assert_not_called()
        update_scan_mock.assert_called_once()
        self.assertEqual(update_scan_mock.call_args[0][1]["status"], "failed")

    def test_degraded_audio_scan_fails_before_scan_result_write(self):
        scan_context = {
            "status": "media_ready",
            "scan_media": {"video_file": "vid-1", "audio_file": "aud-1", "thumbnail": "img-1"},
            "resolved_media": {"video": "vid-1", "audio": "aud-1", "image": "img-1"},
            "task_metrics": {},
        }
        with patch.object(main, "_resolve_scan_context", return_value=scan_context), patch.object(
            main.directus,
            "get_scan_media",
            return_value=scan_context["scan_media"],
        ), patch.object(
            main.ml_runtime,
            "is_loaded",
            return_value=False,
        ), patch.object(
            main.ml_runtime,
            "local_model_required",
            return_value=False,
        ), patch.object(
            main,
            "_resolve_media_input",
            side_effect=_media_input_dispatcher(),
        ), patch.object(
            main,
            "_safe_analyze",
            side_effect=_analysis_dispatcher(
                video_result=_video_result(
                    sharpness_score=0.2,
                    visual_quality_score=0.25,
                    visual_warnings=["video_blurry"],
                ),
                image_result=_image_result(),
                audio_result=_audio_result(
                    noise_estimate=0.9,
                    usable_speech_detected=False,
                    speech_state="no_usable_speech",
                    audio_quality_score=0.25,
                    audio_warnings=["audio_too_noisy"],
                ),
            ),
        ), patch.object(
            main,
            "_expected_phrase",
            return_value=None,
        ), patch.object(
            main,
            "_baseline_rows_for_member",
            return_value=[],
        ), patch.object(
            main.directus,
            "supports_fields",
            return_value=set(),
        ), patch.object(
            main.directus,
            "filter_payload_fields",
            side_effect=lambda collection, payload: payload,
        ), patch.object(
            main.directus,
            "first_supported_field",
            return_value=None,
        ), patch.object(
            main.directus,
            "get_field_choices",
            return_value=[],
        ), patch.object(
            main.directus,
            "is_field_required",
            return_value=False,
        ), patch.object(
            main.directus,
            "upsert_scan_result",
            return_value=("created", {"id": "result-1"}),
        ) as upsert_mock, patch.object(
            main.directus,
            "update_wellness_scan",
            return_value={},
        ) as update_scan_mock, patch.object(
            main.directus,
            "update_scan_request_if_needed",
            return_value=None,
        ), patch.object(
            main.directus,
            "create_alert_if_needed",
            return_value=None,
        ):
            result = main._process_scan_sync("scan-123")

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["failure_reason"], "low_quality_media")
        completed_payload = update_scan_mock.call_args[0][1]
        self.assertEqual(completed_payload["status"], "failed")
        self.assertEqual(completed_payload["failure_reason"], "low_quality_media")

    def test_missing_media_fails_before_scan_result_write(self):
        scan_context = {
            "status": "media_ready",
            "scan_media": {"video_file": None, "audio_file": None, "thumbnail": None},
            "resolved_media": {"video": None, "audio": None, "image": None},
            "task_metrics": {},
        }
        with patch.object(main, "_resolve_scan_context", return_value=scan_context), patch.object(
            main.directus,
            "get_scan_media",
            return_value=scan_context["scan_media"],
        ), patch.object(
            main.ml_runtime,
            "is_loaded",
            return_value=False,
        ), patch.object(
            main.ml_runtime,
            "local_model_required",
            return_value=False,
        ), patch.object(
            main,
            "_baseline_rows_for_member",
            return_value=[],
        ), patch.object(
            main.directus,
            "supports_fields",
            return_value=set(),
        ), patch.object(
            main.directus,
            "filter_payload_fields",
            side_effect=lambda collection, payload: payload,
        ), patch.object(
            main.directus,
            "first_supported_field",
            return_value=None,
        ), patch.object(
            main.directus,
            "get_field_choices",
            return_value=[],
        ), patch.object(
            main.directus,
            "is_field_required",
            return_value=False,
        ), patch.object(
            main.directus,
            "upsert_scan_result",
            return_value=("created", {"id": "result-1"}),
        ) as upsert_mock, patch.object(
            main.directus,
            "update_wellness_scan",
            return_value={},
        ) as update_scan_mock, patch.object(
            main.directus,
            "update_scan_request_if_needed",
            return_value=None,
        ), patch.object(
            main.directus,
            "create_alert_if_needed",
            return_value=None,
        ):
            result = main._process_scan_sync("scan-123")

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["failure_reason"], "video_missing")
        update_scan_mock.assert_called_once()
        completed_payload = update_scan_mock.call_args[0][1]
        self.assertEqual(completed_payload["status"], "failed")

    def test_required_audio_timeout_fails_before_scoring(self):
        scan_context = {
            "status": "media_ready",
            "scan_media": {"video_file": "vid-1", "audio_file": "aud-1", "thumbnail": "img-1"},
            "resolved_media": {"video": "vid-1", "audio": "aud-1", "image": "img-1"},
            "task_metrics": {},
        }
        with patch.object(main, "_resolve_scan_context", return_value=scan_context), patch.object(
            main.directus,
            "get_scan_media",
            return_value=scan_context["scan_media"],
        ), patch.object(
            main.ml_runtime,
            "is_loaded",
            return_value=False,
        ), patch.object(
            main.ml_runtime,
            "local_model_required",
            return_value=False,
        ), patch.object(
            main,
            "_resolve_media_input",
            side_effect=_media_input_dispatcher(),
        ), patch.object(
            main,
            "_safe_analyze",
            side_effect=_analysis_dispatcher(
                video_result=_video_result(),
                image_result=_image_result(),
                audio_result={
                    "score": None,
                    "details": {
                        "status": "load_failed",
                        "audio_warnings": ["audio_decode_timeout"],
                    },
                },
            ),
        ), patch.object(
            main,
            "_expected_phrase",
            return_value=None,
        ), patch.object(
            main,
            "_baseline_rows_for_member",
            return_value=[],
        ), patch.object(
            main.directus,
            "update_wellness_scan",
            return_value={},
        ) as update_scan_mock, patch.object(
            main,
            "_write_success",
            return_value={"scan_result": "created:1"},
        ) as write_mock:
            result = main._process_scan_sync("scan-123")

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["failure_reason"], "audio_validation_timeout")
        write_mock.assert_not_called()
        update_scan_mock.assert_called_once()
        update_payload = update_scan_mock.call_args[0][1]
        self.assertEqual(update_payload["status"], "failed")
        self.assertEqual(update_payload["failure_reason"], "audio_validation_timeout")
        self.assertEqual(update_payload["failure_message"], main.failure_message("audio_validation_timeout"))
        self.assertEqual(update_payload["ai_model_version"], main.MODEL_VERSION)
        self.assertIn("completed_at", update_payload)

    def test_health_uses_configured_model_version_and_optional_local_model(self):
        with patch.object(main, "MODEL_VERSION", "cie_v1_2"), patch.object(
            main.ml_runtime,
            "is_loaded",
            return_value=False,
        ), patch.object(
            main.ml_runtime,
            "local_model_required",
            return_value=False,
        ), patch.object(
            main.ml_runtime,
            "error",
            None,
            create=True,
        ), patch.object(
            main.ml_runtime,
            "model_path",
            "models/latest.pt",
            create=True,
        ), patch("main.os.path.exists", return_value=False), patch.object(
            main.directus,
            "is_configured",
            return_value=True,
        ):
            health = main.health()

        self.assertEqual(health["model_version"], "cie_v1_2")
        self.assertFalse(health["ml_loaded"])
        self.assertIsNone(health["ml_error"])
        self.assertFalse(health["model_file_exists"])
        self.assertFalse(health["local_model_required"])
        self.assertTrue(health["directus_configured"])

    def test_process_does_not_fail_when_local_model_is_optional_and_missing(self):
        scan_context = {
            "status": "media_ready",
            "scan_media": {"video_file": "vid-1", "audio_file": "aud-1", "thumbnail": "img-1"},
            "resolved_media": {"video": "vid-1", "audio": "aud-1", "image": "img-1"},
            "task_metrics": {},
        }
        with patch.object(main, "_resolve_scan_context", return_value=scan_context), patch.object(
            main.directus,
            "get_scan_media",
            return_value=scan_context["scan_media"],
        ), patch.object(
            main.ml_runtime,
            "is_loaded",
            return_value=False,
        ), patch.object(
            main.ml_runtime,
            "local_model_required",
            return_value=False,
        ), patch.object(
            main,
            "_resolve_media_input",
            side_effect=_media_input_dispatcher(),
        ), patch.object(
            main,
            "_safe_analyze",
            side_effect=_analysis_dispatcher(
                video_result=_video_result(),
                image_result=_image_result(),
                audio_result=_audio_result(),
            ),
        ), patch.object(
            main,
            "_transcribe_audio_file",
            return_value="hello world",
        ), patch.object(
            main,
            "_expected_phrase",
            return_value=None,
        ), patch.object(
            main,
            "_baseline_rows_for_member",
            return_value=[],
        ), patch.object(
            main,
            "_write_success",
            return_value={"scan_result": "created:1"},
        ):
            result = main._process_scan_sync("scan-123")

        self.assertEqual(result["status"], "completed")

    def test_no_undefined_values_in_scan_results_payload(self):
        with patch.object(main.directus, "supports_fields", return_value=set()), patch.object(
            main.directus,
            "filter_payload_fields",
            side_effect=lambda collection, payload: payload,
        ), patch.object(main.directus, "first_supported_field", return_value=None), patch.object(
            main.directus,
            "get_field_choices",
            return_value=[],
        ), patch.object(
            main.directus,
            "is_field_required",
            return_value=False,
        ):
            payload = main._build_scan_result_payload(
                "scan-1",
                {
                    "readiness_score": 75,
                    "risk_level": "stable",
                    "confidence": 0.8,
                    "camera_confidence": None,
                    "voice_confidence": 0.7,
                    "task_performance_score": None,
                    "explanation": "ok",
                    "suggested_action": "continue_normal_activity",
                    "ai_model_version": "1.2.0",
                    "spoken_transcript": "hello",
                    "expected_phrase": "hello",
                    "phrase_match_score": 1.0,
                    "audio_quality_score": 0.8,
                    "video_quality_score": 0.8,
                    "image_quality_score": 0.8,
                    "validation_warnings": [],
                },
                {"quality": {}},
            )
        serialized = str(payload)
        self.assertNotIn("undefined", serialized)

    def test_scan_results_payload_contains_only_supported_fields(self):
        supported_fields = {
            "scan_id",
            "readiness_score",
            "observed_fatigue_score",
            "risk_level",
            "confidence",
            "explanation",
            "suggested_action",
            "ai_model_version",
        }
        with patch.object(main.directus, "supports_fields", return_value=set()), patch.object(
            main.directus,
            "filter_payload_fields",
            side_effect=lambda collection, payload: {key: value for key, value in payload.items() if key in supported_fields},
        ), patch.object(main.directus, "first_supported_field", return_value=None), patch.object(
            main.directus,
            "get_field_choices",
            return_value=[],
        ), patch.object(
            main.directus,
            "is_field_required",
            return_value=False,
        ):
            payload = main._build_scan_result_payload(
                "scan-1",
                {
                    "readiness_score": 75,
                    "observed_fatigue_score": 25,
                    "risk_level": "stable",
                    "confidence": 0.8,
                    "camera_confidence": 0.7,
                    "voice_confidence": 0.7,
                    "task_performance_score": 90,
                    "explanation": "ok",
                    "suggested_action": "continue_normal_activity",
                    "ai_model_version": "cie_v1_2",
                    "validation_warnings": ["video_blurry"],
                },
                {"quality": {}},
            )

        self.assertEqual(set(payload.keys()), supported_fields)

    def test_scan_results_ai_model_version_is_forced_to_server_identifier(self):
        with patch.object(main.directus, "supports_fields", return_value=set()), patch.object(
            main.directus,
            "filter_payload_fields",
            side_effect=lambda collection, payload: payload,
        ), patch.object(main.directus, "first_supported_field", return_value=None), patch.object(
            main.directus,
            "get_field_choices",
            return_value=[],
        ), patch.object(
            main.directus,
            "is_field_required",
            return_value=False,
        ), patch.object(
            main.directus,
            "get_field_max_length",
            side_effect=lambda collection, field: 100 if field == "ai_model_version" else None,
        ):
            payload = main._build_scan_result_payload(
                "scan-1",
                {
                    "readiness_score": 75,
                    "observed_fatigue_score": 25,
                    "risk_level": "stable",
                    "confidence": 0.8,
                    "explanation": "ok",
                    "suggested_action": "continue_normal_activity",
                    "ai_model_version": "some-ui-display-name",
                },
                {"quality": {}},
            )

        self.assertEqual(payload["ai_model_version"], "cie_v1_2")

    def test_scan_results_payload_includes_new_fields_only_when_schema_supports_them(self):
        supported_fields = {
            "scan_id",
            "readiness_score",
            "observed_fatigue_score",
            "risk_level",
            "confidence",
            "explanation",
            "suggested_action",
            "ai_model_version",
            "result_status",
            "capture_quality_score",
            "measurement_reliability_score",
            "personal_deviation_score",
            "task_completion_status",
            "baseline_status_at_inference",
            "baseline_confidence",
            "baseline_eligible",
            "hard_gates_triggered",
            "explainable_reasons",
        }
        with patch.object(main.directus, "supports_fields", return_value=supported_fields), patch.object(
            main.directus,
            "filter_payload_fields",
            side_effect=lambda collection, payload: {key: value for key, value in payload.items() if key in supported_fields},
        ), patch.object(main.directus, "first_supported_field", return_value=None), patch.object(
            main.directus,
            "get_field_choices",
            return_value=[],
        ), patch.object(
            main.directus,
            "is_field_required",
            return_value=False,
        ):
            payload = main._build_scan_result_payload(
                "scan-1",
                {
                    "readiness_score": 75,
                    "observed_fatigue_score": 25,
                    "risk_level": "stable",
                    "confidence": 0.8,
                    "explanation": "ok",
                    "suggested_action": "continue_normal_activity",
                    "result_status": "scored",
                    "capture_quality_score": 0.83,
                    "measurement_reliability_score": 0.77,
                    "personal_deviation_score": 0.04,
                    "task_completion_status": "completed",
                    "baseline_status_at_inference": "provisional",
                    "baseline_confidence": 0.6,
                    "baseline_eligible": True,
                    "hard_gates_triggered": [],
                    "explainable_reasons": [],
                },
                {"quality": {}},
            )

        self.assertIn("result_status", payload)
        self.assertIn("baseline_eligible", payload)
        self.assertIn("hard_gates_triggered", payload)
        self.assertIn("explainable_reasons", payload)

    def test_scan_results_payload_omits_new_fields_when_schema_does_not_support_them(self):
        supported_fields = {
            "scan_id",
            "readiness_score",
            "risk_level",
            "confidence",
            "explanation",
            "suggested_action",
            "ai_model_version",
        }
        with patch.object(main.directus, "supports_fields", return_value=supported_fields), patch.object(
            main.directus,
            "filter_payload_fields",
            side_effect=lambda collection, payload: {key: value for key, value in payload.items() if key in supported_fields},
        ), patch.object(main.directus, "first_supported_field", return_value=None), patch.object(
            main.directus,
            "get_field_choices",
            return_value=[],
        ), patch.object(
            main.directus,
            "is_field_required",
            return_value=False,
        ):
            payload = main._build_scan_result_payload(
                "scan-1",
                {
                    "readiness_score": 75,
                    "observed_fatigue_score": 25,
                    "risk_level": "stable",
                    "confidence": 0.8,
                    "explanation": "ok",
                    "suggested_action": "continue_normal_activity",
                    "result_status": "scored",
                    "capture_quality_score": 0.83,
                    "measurement_reliability_score": 0.77,
                    "personal_deviation_score": 0.04,
                    "task_completion_status": "completed",
                    "baseline_status_at_inference": "provisional",
                    "baseline_confidence": 0.6,
                    "baseline_eligible": True,
                    "hard_gates_triggered": [],
                    "explainable_reasons": [],
                },
                {"quality": {}},
            )

        self.assertNotIn("result_status", payload)
        self.assertNotIn("baseline_eligible", payload)
        self.assertNotIn("hard_gates_triggered", payload)

    def test_optional_overlong_string_field_is_skipped_with_warning(self):
        with patch.object(main.directus, "supports_fields", return_value=set()), patch.object(
            main.directus,
            "filter_payload_fields",
            side_effect=lambda collection, payload: payload,
        ), patch.object(main.directus, "first_supported_field", return_value=None), patch.object(
            main.directus,
            "get_field_choices",
            return_value=[],
        ), patch.object(
            main.directus,
            "is_field_required",
            return_value=False,
        ), patch.object(
            main.directus,
            "get_field_max_length",
            side_effect=lambda collection, field: 10 if field == "spoken_transcript" else None,
        ):
            with self.assertLogs("ai-server", level="WARNING") as logs:
                payload = main._build_scan_result_payload(
                    "scan-1",
                    {
                        "readiness_score": 75,
                        "risk_level": "stable",
                        "confidence": 0.8,
                        "explanation": "ok",
                        "suggested_action": "continue_normal_activity",
                        "ai_model_version": "cie_v1_2",
                        "spoken_transcript": "this transcript is too long",
                    },
                    {"quality": {}},
                )

        self.assertNotIn("spoken_transcript", payload)
        self.assertIn("field=spoken_transcript", "\n".join(logs.output))

    def test_required_overlong_string_field_is_truncated_before_post(self):
        with patch.object(main.directus, "supports_fields", return_value=set()), patch.object(
            main.directus,
            "filter_payload_fields",
            side_effect=lambda collection, payload: payload,
        ), patch.object(main.directus, "first_supported_field", return_value=None), patch.object(
            main.directus,
            "get_field_choices",
            return_value=[],
        ), patch.object(
            main.directus,
            "is_field_required",
            return_value=True,
        ), patch.object(
            main.directus,
            "get_field_max_length",
            side_effect=lambda collection, field: 10 if field == "explanation" else None,
        ):
            with self.assertLogs("ai-server", level="WARNING") as logs:
                payload = main._build_scan_result_payload(
                    "scan-1",
                    {
                        "readiness_score": 75,
                        "risk_level": "stable",
                        "confidence": 0.8,
                        "explanation": "explanation that is too long",
                        "suggested_action": "continue_normal_activity",
                        "ai_model_version": "ignored-ui-value",
                    },
                    {"quality": {}},
                )

        self.assertEqual(payload["explanation"], "explanatio")
        self.assertIn("directus_field_truncated", "\n".join(logs.output))

    def test_invalid_numeric_values_are_removed_from_scan_results_payload(self):
        with patch.object(main.directus, "supports_fields", return_value=set()), patch.object(
            main.directus,
            "filter_payload_fields",
            side_effect=lambda collection, payload: payload,
        ), patch.object(main.directus, "first_supported_field", return_value=None), patch.object(
            main.directus,
            "get_field_choices",
            return_value=[],
        ), patch.object(
            main.directus,
            "is_field_required",
            return_value=False,
        ):
            payload = main._build_scan_result_payload(
                "scan-1",
                {
                    "readiness_score": "80",
                    "risk_level": "stable",
                    "confidence": "bad",
                    "camera_confidence": float("nan"),
                    "voice_confidence": float("inf"),
                    "task_performance_score": 80.4,
                    "explanation": "ok",
                    "suggested_action": "continue_normal_activity",
                    "ai_model_version": "cie_v1_2",
                    "confidence_drift": "-0.25",
                },
                {"quality": {}},
            )

        self.assertNotIn("readiness_score", payload)
        self.assertNotIn("task_performance_score", payload)
        self.assertNotIn("confidence_drift", payload)
        self.assertNotIn("confidence", payload)
        self.assertNotIn("camera_confidence", payload)
        self.assertNotIn("voice_confidence", payload)

    def test_risk_level_mapping_uses_directus_choices(self):
        def fake_choices(collection, field_name):
            if field_name == "risk_level":
                return [{"value": "Stable", "label": "Stable"}]
            return []

        with patch.object(main.directus, "supports_fields", return_value=set()), patch.object(
            main.directus,
            "filter_payload_fields",
            side_effect=lambda collection, payload: payload,
        ), patch.object(main.directus, "first_supported_field", return_value=None), patch.object(
            main.directus,
            "get_field_choices",
            side_effect=fake_choices,
        ), patch.object(
            main.directus,
            "is_field_required",
            return_value=False,
        ):
            payload = main._build_scan_result_payload(
                "scan-1",
                {
                    "readiness_score": 75,
                    "risk_level": "stable",
                    "confidence": 0.8,
                    "explanation": "ok",
                    "suggested_action": "continue_normal_activity",
                    "ai_model_version": "cie_v1_2",
                },
                {"quality": {}},
            )

        self.assertEqual(payload["risk_level"], "Stable")

    def test_missing_optional_scan_result_fields_are_skipped(self):
        supported_optional = {"internal_analysis"}
        with patch.object(main.directus, "supports_fields", return_value=supported_optional), patch.object(
            main.directus,
            "filter_payload_fields",
            side_effect=lambda collection, payload: payload,
        ), patch.object(main.directus, "first_supported_field", return_value="internal_analysis"), patch.object(
            main.directus,
            "get_field_choices",
            return_value=[],
        ), patch.object(
            main.directus,
            "is_field_required",
            return_value=False,
        ):
            payload = main._build_scan_result_payload(
                "scan-1",
                {
                    "readiness_score": 75,
                    "risk_level": "stable",
                    "confidence": 0.8,
                    "explanation": "ok",
                    "suggested_action": "continue_normal_activity",
                    "ai_model_version": "cie_v1_2",
                },
                {"quality": {}, "validation": {"warnings": []}},
            )

        self.assertIn("internal_analysis", payload)
        self.assertIsInstance(payload["internal_analysis"], str)
        self.assertLessEqual(len(payload["internal_analysis"]), 255)
        self.assertNotIn("audio_quality_score", payload)
        self.assertNotIn("video_quality_score", payload)
        self.assertNotIn("image_quality_score", payload)
        self.assertNotIn("validation_warnings", payload)

    def test_internal_analysis_never_receives_structured_dictionary(self):
        supported_optional = {"internal_analysis"}
        with patch.object(main.directus, "supports_fields", return_value=supported_optional), patch.object(
            main.directus,
            "filter_payload_fields",
            side_effect=lambda collection, payload: payload,
        ), patch.object(main.directus, "first_supported_field", return_value="internal_analysis"), patch.object(
            main.directus,
            "get_field_choices",
            return_value=[],
        ), patch.object(
            main.directus,
            "is_field_required",
            return_value=False,
        ):
            payload = main._build_scan_result_payload(
                "scan-1",
                {
                    "readiness_score": 45,
                    "risk_level": "low_focus",
                    "confidence": 0.3,
                    "explanation": "retake needed",
                    "suggested_action": "rescan_recommended",
                    "ai_model_version": "cie_v1_2",
                    "validation_warnings": ["missing_media"],
                },
                {"quality": {"failure_reason": "missing_media", "warnings": ["missing_media"]}, "signals": {"video": {"details": {"status": "missing"}}}},
            )

        self.assertIn("internal_analysis", payload)
        self.assertIsInstance(payload["internal_analysis"], str)
        self.assertNotIn("{", payload["internal_analysis"])
        self.assertLessEqual(len(payload["internal_analysis"]), 255)

    def test_no_unsupported_directus_risk_values_are_introduced(self):
        supported = {"stable", "low_focus", "elevated_fatigue", "high_risk"}
        self.assertEqual(set(main.SCAN_RESULT_CHOICE_ALIASES["risk_level"].keys()), supported)
        self.assertEqual(scoring.VALID_RISK_LEVELS, supported)

    def test_idempotency_already_processing(self):
        client = TestClient(main.app)
        with patch.object(main, "_authenticate_process_user", return_value="user-1"), patch.object(
            main,
            "_resolve_scan_auth_context",
            return_value={"id": "scan-123", "status": "processing", "user": "user-1", "business_profile": "bp-1"},
        ), patch.object(main, "_authorize_scan_access", return_value=None):
            response = client.post("/process", json={"scan_id": "scan-123"}, headers={"Authorization": "Bearer test-token"})
        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json()["status"], "already_processing")

    def test_process_requires_bearer_token_before_scan_lookup(self):
        client = TestClient(main.app)
        with patch.object(main, "_resolve_scan_auth_context") as scan_lookup_mock:
            response = client.post("/process", json={"scan_id": "scan-123"})

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["error"], "invalid_authorization")
        scan_lookup_mock.assert_not_called()

    def test_process_hides_scan_owned_by_another_user(self):
        client = TestClient(main.app)
        with patch.object(main, "_authenticate_process_user", return_value="user-1"), patch.object(
            main,
            "_resolve_scan_auth_context",
            return_value={"id": "scan-123", "status": "media_ready", "user": "user-2", "business_profile": "bp-1"},
        ), patch.object(
            main,
            "_authorize_scan_access",
            side_effect=HTTPException(status_code=404, detail="wellness_scans record not found"),
        ):
            response = client.post("/process", json={"scan_id": "scan-123"}, headers={"Authorization": "Bearer test-token"})

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["error"], "scan_not_found")

    def test_process_rejects_missing_active_membership(self):
        client = TestClient(main.app)
        with patch.object(main, "_authenticate_process_user", return_value="user-1"), patch.object(
            main,
            "_resolve_scan_auth_context",
            return_value={"id": "scan-123", "status": "media_ready", "user": "user-1", "business_profile": "bp-1"},
        ), patch.object(
            main,
            "_authorize_scan_access",
            side_effect=HTTPException(status_code=403, detail="active_membership_required"),
        ):
            response = client.post("/process", json={"scan_id": "scan-123"}, headers={"Authorization": "Bearer test-token"})

        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json()["error"], "active_membership_required")

    def test_process_rejects_non_active_directus_user_status(self):
        client = TestClient(main.app)
        with patch.object(main.directus, "get_current_user", return_value={"id": "user-1", "status": "suspended"}), patch.object(
            main,
            "_resolve_scan_auth_context",
        ) as scan_lookup_mock:
            response = client.post("/process", json={"scan_id": "scan-123"}, headers={"Authorization": "Bearer test-token"})

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["error"], "invalid_authorization")
        scan_lookup_mock.assert_not_called()

    def test_process_rejects_media_ready_scan_with_missing_required_scan_media(self):
        client = TestClient(main.app)
        scan_context = {
            "id": "scan-123",
            "status": "media_ready",
            "user": "user-1",
            "business_profile": "bp-1",
        }
        policy = ValidationPolicy(require_video=True, require_audio=True, require_image=True)
        with patch.object(main, "VALIDATION_POLICY", policy), patch.object(
            main,
            "_authenticate_process_user",
            return_value="user-1",
        ), patch.object(
            main,
            "_resolve_scan_auth_context",
            return_value=scan_context,
        ), patch.object(main, "_authorize_scan_access", return_value=None), patch.object(
            main.directus,
            "get_scan_media",
            return_value={"id": "media-1", "scan_id": "scan-123", "video_file": "video-1", "audio_file": None, "thumbnail": "thumb-1"},
        ):
            response = client.post("/process", json={"scan_id": "scan-123"}, headers={"Authorization": "Bearer test-token"})

        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["error"], "scan_media_not_ready")

    def test_process_scan_media_not_ready_does_not_update_or_schedule_background(self):
        client = TestClient(main.app)
        scan_context = {
            "id": "scan-123",
            "status": "media_ready",
            "processing_attempts": 1,
            "user": "user-1",
            "business_profile": "bp-1",
        }
        policy = ValidationPolicy(require_video=True, require_audio=False, require_image=False)
        with patch.object(main, "VALIDATION_POLICY", policy), patch.object(
            main,
            "_authenticate_process_user",
            return_value="user-1",
        ), patch.object(
            main,
            "_resolve_scan_auth_context",
            return_value=scan_context,
        ), patch.object(main, "_authorize_scan_access", return_value=None), patch.object(
            main.directus,
            "get_scan_media",
            return_value=None,
        ), patch.object(
            main.directus,
            "update_wellness_scan",
            return_value={},
        ) as update_mock, patch.object(
            main,
            "process_scan_background",
            return_value=None,
        ) as background_mock:
            response = client.post("/process", json={"scan_id": "scan-123"}, headers={"Authorization": "Bearer test-token"})

        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["error"], "scan_media_not_ready")
        update_mock.assert_not_called()
        background_mock.assert_not_called()

    def test_process_accepts_media_ready_scan_and_ignores_extra_request_fields(self):
        client = TestClient(main.app)
        scan_context = {
            "id": "scan-123",
            "status": "media_ready",
            "processing_attempts": 1,
            "user": "user-1",
            "business_profile": "bp-1",
        }
        with patch.object(main, "_authenticate_process_user", return_value="user-1"), patch.object(
            main,
            "_resolve_scan_auth_context",
            return_value=scan_context,
        ), patch.object(main, "_authorize_scan_access", return_value=None), patch.object(
            main.directus,
            "get_scan_media",
            return_value={"id": "media-1", "scan_id": "scan-123", "video_file": "video-1", "audio_file": "audio-1", "thumbnail": "thumb-1"},
        ), patch.object(
            main.directus,
            "update_wellness_scan",
            return_value={},
        ) as update_mock, patch.object(
            main,
            "process_scan_background",
            return_value=None,
        ):
            response = client.post(
                "/process",
                json={
                    "scan_id": "scan-123",
                    "media": {"video": "https://evil.example/video.mp4"},
                    "previous_confidence": 0.99,
                    "task": {"reaction_time": 0.1},
                },
                headers={"Authorization": "Bearer test-token"},
            )
        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json()["status"], "accepted")
        update_mock.assert_called_once()
        args = update_mock.call_args[0]
        self.assertEqual(args[0], "scan-123")
        self.assertEqual(args[1]["status"], "processing")
        self.assertEqual(args[1]["processing_attempts"], 2)
        self.assertEqual(args[1]["processing_started_at"], ANY)

    def test_write_success_updates_scan_request_when_match_exists(self):
        with patch.object(main.directus, "upsert_scan_result", return_value=("created", {"id": "result-1"})), patch.object(
            main.directus,
            "update_wellness_scan",
            return_value={},
        ), patch.object(
            main.directus,
            "update_member_last_result",
            return_value={},
        ), patch.object(
            main.directus,
            "update_scan_request_if_needed",
            return_value={"id": "request-1"},
        ) as scan_request_mock, patch.object(
            main.directus,
            "create_alert_if_needed",
            return_value=None,
        ):
            status = main._write_success(
                scan_id="scan-123",
                scan_context={"request_source": "manager_request", "member": "member-1", "business_profile": "bp-1"},
                identifiers={"member_id": "member-1", "business_profile_id": "bp-1", "department_id": None, "user_id": None},
                result={
                    "readiness_score": 80,
                    "risk_level": "stable",
                    "confidence": 0.8,
                    "camera_confidence": 0.8,
                    "voice_confidence": 0.8,
                    "task_performance_score": 80,
                    "explanation": "ok",
                    "suggested_action": "continue_normal_activity",
                    "ai_model_version": "1.2.0",
                    "confidence_drift": 0.0,
                    "baseline_used": False,
                    "face_metrics": {"face_score": 0.8},
                    "voice_metrics": {"voice_score": 0.8},
                    "reaction_metrics": {"reaction_score": 0.8},
                },
                internal_analysis={"quality": {}},
            )

        self.assertEqual(status["scan_request"], "updated")
        scan_request_mock.assert_called_once_with(
            request_id=None,
            scan_context={"request_source": "manager_request", "member": "member-1", "business_profile": "bp-1"},
            scan_id="scan-123",
        )

    def test_write_success_marks_wellness_scan_completed_after_scan_result_write(self):
        with patch.object(main.directus, "upsert_scan_result", return_value=("created", {"id": "result-1"})), patch.object(
            main.directus,
            "update_wellness_scan",
            return_value={},
        ) as update_scan_mock, patch.object(
            main.directus,
            "update_member_last_result",
            return_value={},
        ), patch.object(
            main.directus,
            "update_scan_request_if_needed",
            return_value=None,
        ), patch.object(
            main.directus,
            "create_alert_if_needed",
            return_value=None,
        ):
            status = main._write_success(
                scan_id="scan-123",
                scan_context={},
                identifiers={"member_id": None, "business_profile_id": None, "department_id": None, "user_id": None},
                result={
                    "readiness_score": 80,
                    "risk_level": "stable",
                    "confidence": 0.8,
                    "camera_confidence": 0.8,
                    "voice_confidence": 0.8,
                    "task_performance_score": 80,
                    "explanation": "ok",
                    "suggested_action": "continue_normal_activity",
                    "ai_model_version": "cie_v1_2",
                    "confidence_drift": 0.0,
                    "baseline_used": False,
                    "face_metrics": {"face_score": 0.8},
                    "voice_metrics": {"voice_score": 0.8},
                    "reaction_metrics": {"reaction_score": 0.8},
                },
                internal_analysis={"quality": {}},
            )

        self.assertEqual(status["wellness_scan"], "updated")
        update_payload = update_scan_mock.call_args[0][1]
        self.assertEqual(update_payload["status"], "completed")
        self.assertEqual(update_payload["failure_reason"], None)
        self.assertEqual(update_payload["ai_model_version"], "cie_v1_2")
        self.assertIn("completed_at", update_payload)

    def test_write_success_rejects_all_null_evidence_before_scan_result_write(self):
        with patch.object(main.directus, "upsert_scan_result", return_value=("created", {"id": "result-1"})) as upsert_mock, patch.object(
            main.directus,
            "update_wellness_scan",
            return_value={},
        ):
            with self.assertRaises(main.ProcessingError):
                main._write_success(
                    scan_id="scan-123",
                    scan_context={},
                    identifiers={"member_id": None, "business_profile_id": None, "department_id": None, "user_id": None},
                    result={
                        "readiness_score": None,
                        "risk_level": None,
                        "confidence": None,
                        "camera_confidence": None,
                        "voice_confidence": None,
                        "task_performance_score": None,
                        "explanation": "invalid",
                        "suggested_action": "rescan_recommended",
                        "ai_model_version": "cie_v1_2",
                        "confidence_drift": None,
                        "baseline_used": False,
                        "face_metrics": None,
                        "voice_metrics": None,
                        "reaction_metrics": None,
                        "modality_scores": None,
                    },
                    internal_analysis={"quality": {}},
                )


    def test_wellness_scan_completed_update_keeps_short_ai_model_version(self):
        with patch.object(main.directus, "filter_payload_fields", side_effect=lambda collection, payload: payload), patch.object(
            main.directus,
            "first_supported_field",
            return_value=None,
        ), patch.object(
            main.directus,
            "get_field_max_length",
            side_effect=lambda collection, field: 10 if field == "ai_model_version" else None,
        ):
            payload = main._wellness_scan_update_payload(
                {
                    "status": "completed",
                    "completed_at": "2026-06-06T12:00:00Z",
                    "failure_reason": None,
                    "failure_message": None,
                    "ai_model_version": "too-long-ui-display-name",
                }
            )

        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["completed_at"], "2026-06-06T12:00:00Z")
        self.assertEqual(payload["ai_model_version"], "cie_v1_2")

    def test_wellness_scan_completed_update_skips_ai_model_version_when_field_missing(self):
        def filter_missing_field(collection, payload):
            return {key: value for key, value in payload.items() if key != "ai_model_version"}

        with patch.object(main.directus, "filter_payload_fields", side_effect=filter_missing_field), patch.object(
            main.directus,
            "first_supported_field",
            return_value="failure_message",
        ), patch.object(
            main.directus,
            "get_field_max_length",
            return_value=None,
        ):
            payload = main._wellness_scan_update_payload(
                {
                    "status": "completed",
                    "completed_at": "2026-06-06T12:00:00Z",
                    "failure_reason": None,
                    "failure_message": None,
                    "ai_model_version": "cie_v1_2",
                }
            )

        self.assertEqual(
            payload,
            {
                "status": "completed",
                "completed_at": "2026-06-06T12:00:00Z",
                "failure_reason": None,
                "failure_message": None,
            },
        )

    def test_baseline_403_does_not_fail_scan_completion(self):
        scan_context = {
            "status": "media_ready",
            "scan_media": {"video_file": "vid-1", "audio_file": "aud-1", "thumbnail": "img-1"},
            "resolved_media": {"video": "vid-1", "audio": "aud-1", "image": "img-1"},
            "task_metrics": {},
            "member": "member-1",
            "business_profile": "bp-1",
        }
        response = MagicMock()
        response.status_code = 403
        response._content = b'{"errors":[{"message":"forbidden"}]}'
        baseline_error = main.requests.HTTPError(response=response)

        with patch.object(main, "_resolve_scan_context", return_value=scan_context), patch.object(
            main.directus,
            "get_scan_media",
            return_value=scan_context["scan_media"],
        ), patch.object(
            main.ml_runtime,
            "is_loaded",
            return_value=False,
        ), patch.object(
            main.ml_runtime,
            "local_model_required",
            return_value=False,
        ), patch.object(
            main,
            "_resolve_media_input",
            side_effect=_media_input_dispatcher(),
        ), patch.object(
            main,
            "_safe_analyze",
            side_effect=_analysis_dispatcher(
                video_result=_video_result(),
                image_result=_image_result(),
                audio_result=_audio_result(),
            ),
        ), patch.object(
            main,
            "_transcribe_audio_file",
            return_value="hello world",
        ), patch.object(
            main,
            "_expected_phrase",
            return_value=None,
        ), patch.object(
            main,
            "_baseline_rows_for_member",
            return_value=[{"id": "baseline-1"}],
        ), patch.object(
            main.directus,
            "upsert_employee_baseline",
            side_effect=baseline_error,
        ), patch.object(
            main,
            "_write_success",
            return_value={"scan_result": "created:1", "wellness_scan": "updated"},
        ):
            result = main._process_scan_sync("scan-123")

        self.assertEqual(result["status"], "completed")

    def test_manual_baseline_route_cannot_fetch_arbitrary_remote_url(self):
        client = TestClient(main.app)
        scan_context = {
            "id": "scan-123",
            "status": "media_ready",
            "member": "member-1",
            "business_profile": "bp-1",
            "resolved_media": {"image": "img-1", "audio": "aud-1", "video": "vid-1"},
        }
        with patch.object(main.directus, "is_configured", return_value=True), patch.object(
            main,
            "_authenticate_process_user",
            return_value="user-1",
        ), patch.object(
            main,
            "_resolve_scan_auth_context",
            return_value={"id": "scan-123", "status": "media_ready", "user": "user-1", "business_profile": "bp-1", "member": "member-1"},
        ), patch.object(
            main,
            "_authorize_scan_access",
            return_value=None,
        ), patch.object(
            main,
            "_resolve_scan_context",
            return_value=scan_context,
        ):
            response = client.post(
                "/baseline",
                json={
                    "scan_id": "scan-123",
                    "media": {"video": "https://evil.example/video.mp4"},
                },
                headers={"Authorization": "Bearer test-token"},
            )

        self.assertEqual(response.status_code, 422)

    def test_baseline_requires_authentication(self):
        client = TestClient(main.app)
        with patch.object(main.directus, "is_configured", return_value=True), patch.object(
            main,
            "_authenticate_process_user",
            side_effect=HTTPException(status_code=401, detail="invalid_authorization"),
        ):
            response = client.post("/baseline", json={"scan_id": "scan-123"})

        self.assertEqual(response.status_code, 401)

    def test_baseline_requires_ownership_and_active_membership(self):
        client = TestClient(main.app)
        with patch.object(main.directus, "is_configured", return_value=True), patch.object(
            main,
            "_authenticate_process_user",
            return_value="user-1",
        ), patch.object(
            main,
            "_resolve_scan_auth_context",
            return_value={"id": "scan-123", "status": "media_ready", "user": "user-2", "business_profile": "bp-1"},
        ), patch.object(
            main,
            "_authorize_scan_access",
            side_effect=HTTPException(status_code=403, detail="active_membership_required"),
        ):
            response = client.post("/baseline", json={"scan_id": "scan-123"}, headers={"Authorization": "Bearer test-token"})

        self.assertEqual(response.status_code, 403)

    def test_baseline_rejects_duplicate_rows_before_analysis(self):
        client = TestClient(main.app)
        scan_context = {
            "id": "scan-123",
            "status": "media_ready",
            "member": "member-1",
            "business_profile": "bp-1",
            "resolved_media": {"image": "img-1", "audio": "aud-1", "video": "vid-1"},
        }
        duplicate_rows = [{"id": "baseline-1"}, {"id": "baseline-2"}]
        with patch.object(main.directus, "is_configured", return_value=True), patch.object(
            main,
            "_authenticate_process_user",
            return_value="user-1",
        ), patch.object(
            main,
            "_resolve_scan_auth_context",
            return_value=scan_context,
        ), patch.object(
            main,
            "_authorize_scan_access",
            return_value=None,
        ), patch.object(
            main,
            "_resolve_scan_context",
            return_value=scan_context,
        ), patch.object(
            main,
            "_baseline_rows_for_member",
            return_value=duplicate_rows,
        ) as baseline_rows_mock:
            response = client.post("/baseline", json={"scan_id": "scan-123"}, headers={"Authorization": "Bearer test-token"})

        self.assertEqual(response.status_code, 409)
        baseline_rows_mock.assert_called_once()

    def test_baseline_route_does_not_use_current_scan_as_personalization(self):
        client = TestClient(main.app)
        scan_context = {
            "id": "scan-123",
            "status": "media_ready",
            "member": "member-1",
            "business_profile": "bp-1",
            "resolved_media": {"image": "img-1", "audio": "aud-1", "video": "vid-1"},
            "task_metrics": {},
        }
        signals = {
            "camera": _image_result(),
            "video": _video_result(),
            "voice": _audio_result(),
        }
        captured_baseline_used = []

        def fake_compute_result(**kwargs):
            captured_baseline_used.append(kwargs["baseline_used"])
            return {
                "status": "completed",
                "retake_required": False,
                "failure_reason": None,
                "readiness_score": 80,
                "observed_fatigue_score": 20,
                "risk_level": "stable",
                "confidence": 0.8,
                "camera_confidence": 0.8,
                "voice_confidence": 0.8,
                "task_performance_score": 80,
                "baseline_used": kwargs["baseline_used"],
                "confidence_drift": 0.0,
                "face_metrics": {"face_score": 0.8, "baseline_drifts": {}},
                "voice_metrics": {"voice_score": 0.8, "baseline_drifts": {}},
                "reaction_metrics": {"reaction_score": 0.8, "baseline_drifts": {}},
                "explanation": "ok",
                "suggested_action": "continue_normal_activity",
                "ai_model_version": "cie_v1_2",
                "modality_scores": {},
                "fusion_details": {},
            }

        with patch.object(main.directus, "is_configured", return_value=True), patch.object(
            main,
            "_authenticate_process_user",
            return_value="user-1",
        ), patch.object(
            main,
            "_resolve_scan_auth_context",
            return_value={"id": "scan-123", "status": "media_ready", "user": "user-1", "business_profile": "bp-1", "member": "member-1"},
        ), patch.object(
            main,
            "_authorize_scan_access",
            return_value=None,
        ), patch.object(
            main,
            "_resolve_scan_context",
            return_value=scan_context,
        ), patch.object(
            main,
            "_analyze_media",
            return_value=(signals, []),
        ), patch.object(
            main,
            "_transcribe_audio_file",
            return_value="hello world",
        ), patch.object(
            main,
            "_expected_phrase",
            return_value=None,
        ), patch.object(
            main,
            "_baseline_rows_for_member",
            return_value=[{"id": "baseline-1", "scan_count": 4, "is_active": True}],
        ), patch.object(
            main,
            "compute_result",
            side_effect=fake_compute_result,
        ), patch.object(
            main,
            "evaluate_baseline_eligibility",
            return_value={
                "eligible": True,
                "task_completion_status": "not_required",
                "reasons": [],
            },
        ), patch.object(
            main.directus,
            "upsert_employee_baseline",
            return_value={"id": "baseline-1"},
        ):
            response = client.post("/baseline", json={"scan_id": "scan-123"}, headers={"Authorization": "Bearer test-token"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(captured_baseline_used, [False])

    def test_baseline_write_failure_does_not_corrupt_completed_scan_result(self):
        scan_context = {
            "status": "media_ready",
            "scan_media": {"video_file": "vid-1", "audio_file": "aud-1", "thumbnail": "img-1"},
            "resolved_media": {"video": "vid-1", "audio": "aud-1", "image": "img-1"},
            "task_metrics": {},
            "member": "member-1",
            "business_profile": "bp-1",
        }
        response = MagicMock()
        response.status_code = 403
        response._content = b'{"errors":[{"message":"forbidden"}]}'
        baseline_error = main.requests.HTTPError(response=response)

        with patch.object(main, "_resolve_scan_context", return_value=scan_context), patch.object(
            main.directus,
            "get_scan_media",
            return_value=scan_context["scan_media"],
        ), patch.object(
            main.ml_runtime,
            "is_loaded",
            return_value=False,
        ), patch.object(
            main.ml_runtime,
            "local_model_required",
            return_value=False,
        ), patch.object(
            main,
            "_resolve_media_input",
            side_effect=_media_input_dispatcher(),
        ), patch.object(
            main,
            "_safe_analyze",
            side_effect=_analysis_dispatcher(
                video_result=_video_result(),
                image_result=_image_result(),
                audio_result=_audio_result(),
            ),
        ), patch.object(
            main,
            "_expected_phrase",
            return_value=None,
        ), patch.object(
            main,
            "_baseline_rows_for_member",
            return_value=[{"id": "baseline-1", "scan_count": 4, "is_active": False}],
        ), patch.object(
            main.directus,
            "upsert_employee_baseline",
            side_effect=baseline_error,
        ), patch.object(
            main,
            "_write_success",
            return_value={"scan_result": "created:1", "wellness_scan": "updated"},
        ) as write_mock:
            result = main._process_scan_sync("scan-123")

        self.assertEqual(result["status"], "completed")
        write_mock.assert_called_once()

    def test_duplicate_baseline_rows_skip_baseline_update_and_continue_scan(self):
        scan_context = {
            "status": "media_ready",
            "scan_media": {"video_file": "vid-1", "audio_file": "aud-1", "thumbnail": "img-1"},
            "resolved_media": {"video": "vid-1", "audio": "aud-1", "image": "img-1"},
            "task_metrics": {},
            "member": "member-1",
            "business_profile": "bp-1",
        }
        duplicate_rows = [{"id": "baseline-1"}, {"id": "baseline-2"}]
        with patch.object(main, "_resolve_scan_context", return_value=scan_context), patch.object(
            main.directus,
            "get_scan_media",
            return_value=scan_context["scan_media"],
        ), patch.object(
            main.ml_runtime,
            "is_loaded",
            return_value=False,
        ), patch.object(
            main.ml_runtime,
            "local_model_required",
            return_value=False,
        ), patch.object(
            main,
            "_resolve_media_input",
            side_effect=_media_input_dispatcher(),
        ), patch.object(
            main,
            "_safe_analyze",
            side_effect=_analysis_dispatcher(
                video_result=_video_result(),
                image_result=_image_result(),
                audio_result=_audio_result(),
            ),
        ), patch.object(
            main,
            "_expected_phrase",
            return_value=None,
        ), patch.object(
            main,
            "_baseline_rows_for_member",
            return_value=duplicate_rows,
        ), patch.object(
            main.directus,
            "upsert_employee_baseline",
        ) as baseline_upsert_mock, patch.object(
            main,
            "_write_success",
            return_value={"scan_result": "created:1", "wellness_scan": "updated"},
        ):
            result = main._process_scan_sync("scan-123")

        self.assertEqual(result["status"], "completed")
        baseline_upsert_mock.assert_not_called()

    def test_low_quality_scan_does_not_call_baseline_upsert(self):
        scan_context = {
            "status": "media_ready",
            "scan_media": {"video_file": "vid-1", "audio_file": "aud-1", "thumbnail": "img-1"},
            "resolved_media": {"video": "vid-1", "audio": "aud-1", "image": "img-1"},
            "task_metrics": {},
            "member": "member-1",
            "business_profile": "bp-1",
        }
        with patch.object(main, "_resolve_scan_context", return_value=scan_context), patch.object(
            main.directus,
            "get_scan_media",
            return_value=scan_context["scan_media"],
        ), patch.object(
            main.ml_runtime,
            "is_loaded",
            return_value=False,
        ), patch.object(
            main.ml_runtime,
            "local_model_required",
            return_value=False,
        ), patch.object(
            main,
            "_resolve_media_input",
            side_effect=_media_input_dispatcher(),
        ), patch.object(
            main,
            "_safe_analyze",
            side_effect=_analysis_dispatcher(
                video_result=_video_result(
                    visual_quality_score=0.2,
                    visual_warnings=["video_blurry"],
                ),
                image_result=_image_result(
                    image_quality_score=0.2,
                    image_warnings=["image_blurry"],
                ),
                audio_result=_audio_result(
                    usable_speech_detected=False,
                    speech_state="no_usable_speech",
                    audio_quality_score=0.2,
                    audio_warnings=["audio_too_noisy"],
                ),
            ),
        ), patch.object(
            main,
            "_expected_phrase",
            return_value=None,
        ), patch.object(
            main,
            "_baseline_rows_for_member",
            return_value=[{"id": "baseline-1"}],
        ), patch.object(
            main.directus,
            "upsert_employee_baseline",
        ) as baseline_upsert_mock, patch.object(
            main,
            "_write_success",
            return_value={"scan_result": "created:1", "wellness_scan": "updated"},
        ):
            result = main._process_scan_sync("scan-123")

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["failure_reason"], "low_quality_media")
        baseline_upsert_mock.assert_not_called()

    def test_malformed_baseline_json_is_not_used_for_personalization(self):
        malformed_baseline = {
            "id": "baseline-1",
            "scan_count": "bad",
            "face_avg": "broken",
            "voice_avg": 123,
            "reaction_avg": [],
        }
        quality_result = {
            "status": "passed",
            "passed": True,
            "weak": False,
            "retake_required": False,
            "media_quality": {"aggregate_quality": 0.9},
            "confidence_multiplier": 0.9,
            "warnings": [],
        }
        validation_result = {"critical_errors": [], "warnings": []}
        result = {
            "confidence": 0.8,
            "risk_level": "stable",
            "retake_required": False,
        }

        self.assertFalse(
            main.baseline_ready_for_personalized_scoring(
                malformed_baseline,
                quality_result=quality_result,
                validation_result=validation_result,
                result=result,
                unique_row=True,
            )
        )

    def test_missing_required_speech_does_not_call_baseline_upsert(self):
        scan_context = {
            "status": "media_ready",
            "scan_media": {"video_file": "vid-1", "audio_file": "aud-1", "thumbnail": "img-1"},
            "resolved_media": {"video": "vid-1", "audio": "aud-1", "image": "img-1"},
            "task_metrics": {},
            "member": "member-1",
            "business_profile": "bp-1",
            "expected_phrase": "please say continuity ready",
        }
        with patch.object(main, "_resolve_scan_context", return_value=scan_context), patch.object(
            main.directus,
            "get_scan_media",
            return_value=scan_context["scan_media"],
        ), patch.object(
            main.ml_runtime,
            "is_loaded",
            return_value=False,
        ), patch.object(
            main.ml_runtime,
            "local_model_required",
            return_value=False,
        ), patch.object(
            main,
            "_resolve_media_input",
            side_effect=_media_input_dispatcher(),
        ), patch.object(
            main,
            "_safe_analyze",
            side_effect=_analysis_dispatcher(
                video_result=_video_result(),
                image_result=_image_result(),
                audio_result=_audio_result(),
            ),
        ), patch.object(
            main,
            "_transcribe_audio_file",
            return_value="something else entirely",
        ), patch.object(
            main,
            "_baseline_rows_for_member",
            return_value=[{"id": "baseline-1"}],
        ), patch.object(
            main.directus,
            "upsert_employee_baseline",
        ) as baseline_upsert_mock, patch.object(
            main,
            "_write_success",
            return_value={"scan_result": "created:1", "wellness_scan": "updated"},
        ):
            result = main._process_scan_sync("scan-123")

        self.assertEqual(result["status"], "completed")
        baseline_upsert_mock.assert_not_called()

    def test_elevated_fatigue_and_high_risk_do_not_call_baseline_upsert(self):
        scan_context = {
            "status": "media_ready",
            "scan_media": {"video_file": "vid-1", "audio_file": "aud-1", "thumbnail": "img-1"},
            "resolved_media": {"video": "vid-1", "audio": "aud-1", "image": "img-1"},
            "task_metrics": {},
            "member": "member-1",
            "business_profile": "bp-1",
        }
        for risk_level in ["elevated_fatigue", "high_risk"]:
            with self.subTest(risk_level=risk_level):
                with patch.object(main, "_resolve_scan_context", return_value=scan_context), patch.object(
                    main.directus,
                    "get_scan_media",
                    return_value=scan_context["scan_media"],
                ), patch.object(
                    main.ml_runtime,
                    "is_loaded",
                    return_value=False,
                ), patch.object(
                    main.ml_runtime,
                    "local_model_required",
                    return_value=False,
                ), patch.object(
                    main,
                    "_resolve_media_input",
                    side_effect=_media_input_dispatcher(),
                ), patch.object(
                    main,
                    "_safe_analyze",
                    side_effect=_analysis_dispatcher(
                video_result=_video_result(),
                image_result=_image_result(),
                audio_result=_audio_result(),
            ),
                ), patch.object(
                    main,
                    "_expected_phrase",
                    return_value=None,
                ), patch.object(
                    main,
                    "_baseline_rows_for_member",
                    return_value=[{"id": "baseline-1"}],
                ), patch.object(
                    main,
                    "compute_result",
                    return_value={
                        "status": "completed",
                        "retake_required": False,
                        "failure_reason": None,
                        "readiness_score": 40,
                        "observed_fatigue_score": 60,
                        "risk_level": risk_level,
                        "confidence": 0.8,
                        "camera_confidence": 0.8,
                        "voice_confidence": 0.8,
                        "task_performance_score": None,
                        "baseline_used": False,
                        "confidence_drift": 0.0,
                        "face_metrics": {"face_score": 0.8, "baseline_drifts": {}},
                        "voice_metrics": {"voice_score": 0.8, "baseline_drifts": {}},
                        "reaction_metrics": {"reaction_score": None, "baseline_drifts": {}},
                        "explanation": "ok",
                        "suggested_action": "continue_normal_activity",
                        "ai_model_version": "cie_v1_2",
                        "modality_scores": {},
                        "fusion_details": {},
                    },
                ), patch.object(
                    main.directus,
                    "upsert_employee_baseline",
                ) as baseline_upsert_mock, patch.object(
                    main,
                    "_write_success",
                    return_value={"scan_result": "created:1", "wellness_scan": "updated"},
                ):
                    result = main._process_scan_sync("scan-123")

                self.assertEqual(result["status"], "completed")
                baseline_upsert_mock.assert_not_called()

    def test_result_status_uses_only_approved_values(self):
        self.assertEqual(
            main._result_status_from_outcome(
                quality_result={"retake_required": False, "failure_reason": None},
                validation_result={},
                result={"confidence": 0.8, "retake_required": False},
                baseline_eligibility={"task_completion_status": "completed"},
            ),
            "scored",
        )
        self.assertEqual(
            main._result_status_from_outcome(
                quality_result={"retake_required": True, "failure_reason": "low_quality_media"},
                validation_result={},
                result={"confidence": 0.8, "retake_required": True},
                baseline_eligibility={"task_completion_status": "completed"},
            ),
            "retake_required",
        )
        self.assertEqual(
            main._result_status_from_outcome(
                quality_result={"retake_required": False, "failure_reason": None},
                validation_result={},
                result={"confidence": 0.8, "retake_required": False},
                baseline_eligibility={"task_completion_status": "incomplete_required_speech"},
            ),
            "incomplete",
        )
        self.assertEqual(
            main._result_status_from_outcome(
                quality_result={"retake_required": False, "failure_reason": None},
                validation_result={},
                result={"confidence": 0.3, "retake_required": False},
                baseline_eligibility={"task_completion_status": "completed"},
            ),
            "low_confidence",
        )

    def test_migration_contains_no_member_null_mutation(self):
        from pathlib import Path

        candidates = [
            Path("sql") / "2026_07_01_phase2_baseline_foundation.sql",
            Path(__file__).resolve().parent.parent / "2026_07_01_phase2_baseline_foundation.sql",
        ]
        migration_path = next((path for path in candidates if path.is_file()), None)
        if migration_path is None:
            self.skipTest("baseline migration file is not available in this test checkout")

        migration = migration_path.read_text(encoding="utf-8")
        self.assertNotIn("SET member = NULL", migration)

    def test_missing_wellness_scan_ai_model_version_does_not_fail_completion(self):
        def filter_missing_wellness_ai_model_version(collection, payload):
            if collection == "wellness_scans":
                return {key: value for key, value in payload.items() if key != "ai_model_version"}
            return payload

        with patch.object(main.directus, "upsert_scan_result", return_value=("created", {"id": "result-1"})) as upsert_mock, patch.object(
            main.directus,
            "update_wellness_scan",
            return_value={},
        ) as update_scan_mock, patch.object(
            main.directus,
            "update_scan_request_if_needed",
            return_value=None,
        ), patch.object(
            main.directus,
            "create_alert_if_needed",
            return_value=None,
        ), patch.object(
            main.directus,
            "filter_payload_fields",
            side_effect=filter_missing_wellness_ai_model_version,
        ), patch.object(
            main.directus,
            "first_supported_field",
            side_effect=lambda collection, candidates: "failure_message" if collection == "wellness_scans" and "failure_message" in candidates else None,
        ), patch.object(
            main.directus,
            "get_field_max_length",
            return_value=None,
        ):
            status = main._write_success(
                scan_id="scan-123",
                scan_context={},
                identifiers={"member_id": None, "business_profile_id": None, "department_id": None, "user_id": None},
                result={
                    "readiness_score": 80,
                    "risk_level": "stable",
                    "confidence": 0.8,
                    "camera_confidence": 0.8,
                    "voice_confidence": 0.8,
                    "task_performance_score": 80,
                    "explanation": "ok",
                    "suggested_action": "continue_normal_activity",
                    "ai_model_version": "cie_v1_2",
                    "confidence_drift": 0.0,
                    "baseline_used": False,
                    "face_metrics": {"face_score": 0.8},
                    "voice_metrics": {"voice_score": 0.8},
                    "reaction_metrics": {"reaction_score": 0.8},
                },
                internal_analysis={"quality": {}},
            )

        self.assertEqual(status["scan_result"], "created:result-1")
        self.assertEqual(status["wellness_scan"], "updated")
        upsert_payload = upsert_mock.call_args[0][1]
        self.assertEqual(upsert_payload["ai_model_version"], "cie_v1_2")
        update_payload = update_scan_mock.call_args[0][1]
        self.assertEqual(update_payload["status"], "completed")
        self.assertEqual(update_payload["failure_reason"], None)
        self.assertNotIn("ai_model_version", update_payload)


if __name__ == "__main__":
    unittest.main()