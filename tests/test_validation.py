from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import ANY, MagicMock, patch

from fastapi.testclient import TestClient

from validation import ValidationPolicy, validate_scan_inputs


if "torch" not in sys.modules:
    fake_torch = types.ModuleType("torch")
    fake_torch.manual_seed = lambda seed: None
    sys.modules["torch"] = fake_torch

if "audio" not in sys.modules:
    fake_audio = types.ModuleType("audio")
    fake_audio.analyze_audio = lambda path: {"score": 0.8, "details": {"status": "ok", "audio_quality_score": 0.8, "audio_warnings": []}}
    fake_audio.transcribe_audio = lambda path: "hello world"
    sys.modules["audio"] = fake_audio

if "video" not in sys.modules:
    fake_video = types.ModuleType("video")
    fake_video.analyze_video = lambda path: {"score": 0.8, "details": {"status": "ok", "visual_quality_score": 0.8, "visual_warnings": []}}
    sys.modules["video"] = fake_video

if "vision" not in sys.modules:
    fake_vision = types.ModuleType("vision")
    fake_vision.analyze_face = lambda path: {"score": 0.8, "details": {"status": "ok", "image_quality_score": 0.8, "image_warnings": [], "face_detected": True}}
    sys.modules["vision"] = fake_vision

if "ml.runtime" not in sys.modules:
    fake_runtime = types.ModuleType("ml.runtime")

    class _FakeMLRuntime:
        def __init__(self, model_path=None):
            self.model_path = model_path or "models/latest.pt"
            self.error = None
            self.require_local_model = False

        def load(self):
            return True

        def is_loaded(self):
            return True

        def local_model_required(self):
            return self.require_local_model

        def predict(self, features):
            return {"confidence": 0.8, "label": "Stable", "model_version": "Conntinuity Intelligence Engine v1.2"}

    fake_runtime.MLRuntime = _FakeMLRuntime
    sys.modules["ml.runtime"] = fake_runtime

if "ml.features" not in sys.modules:
    fake_features = types.ModuleType("ml.features")
    fake_features.features_from_signals = lambda signals, task=None: ({}, signals)
    fake_features.vector_from_features = lambda feature_map: []
    sys.modules["ml.features"] = fake_features

import main


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
        self.assertFalse(result["passed"])
        self.assertEqual(result["failure_reason"], "face_not_visible")

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
        self.assertFalse(result["passed"])
        self.assertEqual(result["failure_reason"], "video_blurry")

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
        self.assertFalse(result["passed"])
        self.assertEqual(result["failure_reason"], "video_too_dark")

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
        self.assertFalse(result["passed"])
        self.assertEqual(result["failure_reason"], "audio_too_noisy")

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
        self.assertFalse(result["passed"])
        self.assertEqual(result["failure_reason"], "phrase_mismatch")

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

    def test_low_quality_media_failure(self):
        result = validate_scan_inputs(
            policy=self.policy,
            media=self.media,
            video_result=_video_result(visual_quality_score=0.2, usable_frame_ratio=0.2, motion_stability_score=0.2),
            audio_result=_audio_result(),
            image_result=_image_result(),
            expected_phrase="hello world",
            transcript="hello world",
        )
        self.assertFalse(result["passed"])
        self.assertIn(result["failure_reason"], {"unstable_video", "low_quality_media"})

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


class MainPayloadTests(unittest.TestCase):
    def test_health_uses_configured_model_version_and_optional_local_model(self):
        with patch.object(main, "MODEL_VERSION", "Conntinuity Intelligence Engine v1.2"), patch.object(
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

        self.assertEqual(health["model_version"], "Conntinuity Intelligence Engine v1.2")
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
            side_effect=[("image.jpg", False), ("audio.wav", False), ("video.mp4", False)],
        ), patch.object(
            main,
            "_safe_analyze",
            side_effect=[_video_result(), _image_result(), _audio_result()],
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
            "_baseline_for_member",
            return_value=None,
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
                    "risk_level": "stable",
                    "confidence": 0.8,
                    "camera_confidence": 0.7,
                    "voice_confidence": 0.7,
                    "task_performance_score": 90,
                    "explanation": "ok",
                    "suggested_action": "continue_normal_activity",
                    "ai_model_version": "Conntinuity Intelligence Engine v1.2",
                    "validation_warnings": ["video_blurry"],
                },
                {"quality": {}},
            )

        self.assertEqual(set(payload.keys()), supported_fields)

    def test_scan_results_ai_model_version_full_value_preserved_when_schema_allows(self):
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
                    "risk_level": "stable",
                    "confidence": 0.8,
                    "explanation": "ok",
                    "suggested_action": "continue_normal_activity",
                    "ai_model_version": "Conntinuity Intelligence Engine v1.2",
                },
                {"quality": {}},
            )

        self.assertEqual(payload["ai_model_version"], "Conntinuity Intelligence Engine v1.2")

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
                        "ai_model_version": "Conntinuity Intelligence Engine v1.2",
                        "spoken_transcript": "this transcript is too long",
                    },
                    {"quality": {}},
                )

        self.assertNotIn("spoken_transcript", payload)
        self.assertIn("field=spoken_transcript", "\n".join(logs.output))

    def test_required_overlong_string_field_raises_clear_schema_error_before_post(self):
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
            side_effect=lambda collection, field: 10 if field == "ai_model_version" else None,
        ):
            with self.assertRaises(main.SchemaValidationError) as ctx:
                main._build_scan_result_payload(
                    "scan-1",
                    {
                        "readiness_score": 75,
                        "risk_level": "stable",
                        "confidence": 0.8,
                        "explanation": "ok",
                        "suggested_action": "continue_normal_activity",
                        "ai_model_version": "Conntinuity Intelligence Engine v1.2",
                    },
                    {"quality": {}},
                )

        self.assertIn("scan_results.ai_model_version exceeds Directus max length", str(ctx.exception))

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
                    "ai_model_version": "Conntinuity Intelligence Engine v1.2",
                    "confidence_drift": "-0.25",
                },
                {"quality": {}},
            )

        self.assertEqual(payload["readiness_score"], 80)
        self.assertEqual(payload["task_performance_score"], 80)
        self.assertEqual(payload["confidence_drift"], -0.25)
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
                    "ai_model_version": "Conntinuity Intelligence Engine v1.2",
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
                    "ai_model_version": "Conntinuity Intelligence Engine v1.2",
                },
                {"quality": {}, "validation": {"warnings": []}},
            )

        self.assertIn("internal_analysis", payload)
        self.assertNotIn("audio_quality_score", payload)
        self.assertNotIn("video_quality_score", payload)
        self.assertNotIn("image_quality_score", payload)
        self.assertNotIn("validation_warnings", payload)

    def test_idempotency_already_processing(self):
        client = TestClient(main.app)
        with patch.object(main, "_resolve_scan_context", return_value={"status": "processing"}):
            response = client.post("/process", json={"scan_id": "scan-123"})
        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json()["status"], "already_processing")

    def test_process_accepts_media_ready_scan_and_ignores_extra_request_fields(self):
        client = TestClient(main.app)
        scan_context = {"status": "media_ready", "processing_attempts": 1}
        with patch.object(main, "_resolve_scan_context", return_value=scan_context), patch.object(
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
                    "ai_model_version": "Conntinuity Intelligence Engine v1.2",
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
        self.assertEqual(update_payload["ai_model_version"], "Conntinuity Intelligence Engine v1.2")
        self.assertIn("completed_at", update_payload)

    def test_wellness_scan_completed_update_skips_optional_overlong_ai_model_version(self):
        with patch.object(main.directus, "filter_payload_fields", side_effect=lambda collection, payload: payload), patch.object(
            main.directus,
            "first_supported_field",
            return_value=None,
        ), patch.object(
            main.directus,
            "get_field_max_length",
            side_effect=lambda collection, field: 10 if field == "ai_model_version" else None,
        ):
            with self.assertLogs("ai-server", level="WARNING") as logs:
                payload = main._wellness_scan_update_payload(
                    {
                        "status": "completed",
                        "completed_at": "2026-06-06T12:00:00Z",
                        "failure_reason": None,
                        "failure_message": None,
                        "ai_model_version": "Conntinuity Intelligence Engine v1.2",
                    }
                )

        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["completed_at"], "2026-06-06T12:00:00Z")
        self.assertNotIn("ai_model_version", payload)
        self.assertIn("field=ai_model_version", "\n".join(logs.output))

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
            side_effect=[("image.jpg", False), ("audio.wav", False), ("video.mp4", False)],
        ), patch.object(
            main,
            "_safe_analyze",
            side_effect=[_video_result(), _image_result(), _audio_result()],
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
            "_baseline_for_member",
            return_value={"id": "baseline-1"},
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


if __name__ == "__main__":
    unittest.main()
