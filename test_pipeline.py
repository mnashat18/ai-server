import unittest
from unittest.mock import MagicMock

from directus_client import DirectusClient
from quality import assess_quality
from scoring import clamp_confidence, compute_result
from utils import sanitize_payload
import requests


def _signal(score, details):
    return {"score": score, "details": details}


class PipelineTests(unittest.TestCase):
    def test_confidence_clamping(self):
        self.assertEqual(clamp_confidence(1.5), 1.0)
        self.assertEqual(clamp_confidence(-0.2), 0.0)
        self.assertIsNone(clamp_confidence(None))

    def test_missing_media_handling(self):
        result = assess_quality({"camera": {}, "video": {}, "voice": {}})
        self.assertTrue(result["passed"])
        self.assertTrue(result["weak"])
        self.assertEqual(result["failure_reason"], "missing_media")
        self.assertEqual(result["suggested_action"], "rescan_recommended")

    def test_low_quality_media_warning(self):
        signals = {
            "camera": _signal(0.2, {"status": "ok", "image_warnings": ["image_blurry"]}),
            "video": _signal(0.25, {"status": "ok", "visual_warnings": ["video_blurry"]}),
            "voice": _signal(0.22, {"status": "ok", "audio_warnings": ["audio_too_noisy"]}),
        }
        result = assess_quality(signals)
        self.assertTrue(result["passed"])
        self.assertTrue(result["weak"])
        self.assertEqual(result["failure_reason"], "low_quality_media")

    def test_no_undefined_values_in_payload(self):
        payload = sanitize_payload(
            {
                "ok": True,
                "text": "undefined",
                "nested": {"value": float("nan"), "good": 1.234567},
            }
        )
        self.assertNotIn("text", payload)
        self.assertNotIn("value", payload["nested"])
        self.assertEqual(payload["nested"]["good"], 1.234567)

    def test_fusion_with_only_audio(self):
        signals = {
            "camera": _signal(None, {"status": "missing", "image_warnings": ["image_missing"]}),
            "video": _signal(None, {"status": "missing", "visual_warnings": ["video_missing"]}),
            "voice": _signal(0.74, {"status": "ok", "audio_confidence": 0.74, "audio_quality_score": 0.7, "audio_warnings": []}),
        }
        quality = assess_quality(signals)
        result = compute_result(signals=signals, quality=quality)
        self.assertLessEqual(result["confidence"], 0.55)

    def test_fusion_with_only_video(self):
        signals = {
            "camera": _signal(None, {"status": "missing", "image_warnings": ["image_missing"]}),
            "video": _signal(0.81, {"status": "ok", "visual_confidence": 0.81, "visual_quality_score": 0.79, "visual_warnings": []}),
            "voice": _signal(None, {"status": "missing", "audio_warnings": ["audio_missing"]}),
        }
        quality = assess_quality(signals)
        result = compute_result(signals=signals, quality=quality)
        self.assertLessEqual(result["confidence"], 0.55)

    def test_fusion_with_all_modalities(self):
        signals = {
            "camera": _signal(0.76, {"status": "ok", "image_confidence": 0.76, "image_quality_score": 0.73, "image_warnings": []}),
            "video": _signal(0.82, {"status": "ok", "visual_confidence": 0.82, "visual_quality_score": 0.8, "visual_warnings": []}),
            "voice": _signal(0.78, {"status": "ok", "audio_confidence": 0.78, "audio_quality_score": 0.76, "audio_warnings": []}),
        }
        quality = assess_quality(signals)
        result = compute_result(signals=signals, quality=quality)
        self.assertGreater(result["confidence"], 0.5)
        self.assertNotEqual(result["risk_level"], "unknown")

    def test_low_media_quality_reduces_confidence_and_score(self):
        good_signals = {
            "camera": _signal(0.76, {"status": "ok", "image_confidence": 0.76, "image_quality_score": 0.73, "image_warnings": []}),
            "video": _signal(0.82, {"status": "ok", "visual_confidence": 0.82, "visual_quality_score": 0.8, "visual_warnings": []}),
            "voice": _signal(0.78, {"status": "ok", "audio_confidence": 0.78, "audio_quality_score": 0.76, "audio_warnings": []}),
        }
        weak_signals = {
            "camera": _signal(0.22, {"status": "ok", "image_confidence": 0.22, "image_quality_score": 0.2, "image_warnings": ["image_blurry"]}),
            "video": _signal(0.25, {"status": "ok", "visual_confidence": 0.25, "visual_quality_score": 0.2, "visual_warnings": ["video_blurry"]}),
            "voice": _signal(0.24, {"status": "ok", "audio_confidence": 0.24, "audio_quality_score": 0.2, "audio_warnings": ["audio_too_noisy"]}),
        }
        good = compute_result(signals=good_signals, quality=assess_quality(good_signals))
        weak = compute_result(signals=weak_signals, quality=assess_quality(weak_signals))

        self.assertLess(weak["confidence"], good["confidence"])
        self.assertLess(weak["readiness_score"], good["readiness_score"])
        self.assertIn("reduced", weak["explanation"].lower())

    def test_blurry_video_reduces_confidence(self):
        clean_signals = {
            "camera": _signal(0.76, {"status": "ok", "image_confidence": 0.76, "image_quality_score": 0.73, "image_warnings": []}),
            "video": _signal(0.82, {"status": "ok", "visual_confidence": 0.82, "visual_quality_score": 0.8, "visual_warnings": []}),
            "voice": _signal(0.78, {"status": "ok", "audio_confidence": 0.78, "audio_quality_score": 0.76, "audio_warnings": []}),
        }
        blurry_signals = {
            "camera": clean_signals["camera"],
            "video": _signal(0.82, {"status": "ok", "visual_confidence": 0.82, "visual_quality_score": 0.8, "visual_warnings": ["video_blurry"]}),
            "voice": clean_signals["voice"],
        }

        clean = compute_result(signals=clean_signals, quality=assess_quality(clean_signals))
        blurry = compute_result(signals=blurry_signals, quality=assess_quality(blurry_signals))

        self.assertLess(blurry["confidence"], clean["confidence"])
        self.assertLess(blurry["modality_scores"]["video"], clean["modality_scores"]["video"])
        self.assertIn("blur", blurry["explanation"].lower())

    def test_noisy_audio_reduces_confidence(self):
        clean_signals = {
            "camera": _signal(0.76, {"status": "ok", "image_confidence": 0.76, "image_quality_score": 0.73, "image_warnings": []}),
            "video": _signal(0.82, {"status": "ok", "visual_confidence": 0.82, "visual_quality_score": 0.8, "visual_warnings": []}),
            "voice": _signal(0.78, {"status": "ok", "audio_confidence": 0.78, "audio_quality_score": 0.76, "audio_warnings": []}),
        }
        noisy_signals = {
            "camera": clean_signals["camera"],
            "video": clean_signals["video"],
            "voice": _signal(0.78, {"status": "ok", "audio_confidence": 0.78, "audio_quality_score": 0.76, "audio_warnings": ["audio_too_noisy"]}),
        }

        clean = compute_result(signals=clean_signals, quality=assess_quality(clean_signals))
        noisy = compute_result(signals=noisy_signals, quality=assess_quality(noisy_signals))

        self.assertLess(noisy["confidence"], clean["confidence"])
        self.assertLess(noisy["modality_scores"]["audio"], clean["modality_scores"]["audio"])
        self.assertIn("noise", noisy["explanation"].lower())

    def test_weak_face_visibility_reduces_confidence(self):
        clean_signals = {
            "camera": _signal(0.76, {"status": "ok", "image_confidence": 0.76, "image_quality_score": 0.73, "image_warnings": []}),
            "video": _signal(0.82, {"status": "ok", "visual_confidence": 0.82, "visual_quality_score": 0.8, "visual_warnings": []}),
            "voice": _signal(0.78, {"status": "ok", "audio_confidence": 0.78, "audio_quality_score": 0.76, "audio_warnings": []}),
        }
        weak_face_quality = assess_quality(clean_signals)
        weak_face_quality["warnings"] = ["face_not_visible"]
        weak_face_quality["weak"] = True
        weak_face_quality["status"] = "weak"

        clean = compute_result(signals=clean_signals, quality=assess_quality(clean_signals))
        weak_face = compute_result(signals=clean_signals, quality=weak_face_quality)

        self.assertLess(weak_face["confidence"], clean["confidence"])
        self.assertLess(weak_face["modality_scores"]["video"], clean["modality_scores"]["video"])
        self.assertIn("face visibility", weak_face["explanation"].lower())

    def test_low_confidence_does_not_return_stable(self):
        signals = {
            "camera": _signal(None, {"status": "missing", "image_warnings": ["image_missing"]}),
            "video": _signal(None, {"status": "missing", "visual_warnings": ["video_missing"]}),
            "voice": _signal(None, {"status": "missing", "audio_warnings": ["audio_missing"]}),
        }
        result = compute_result(signals=signals, quality=assess_quality(signals))

        self.assertLess(result["confidence"], 0.45)
        self.assertEqual(result["risk_level"], "unknown")
        self.assertEqual(result["suggested_action"], "rescan_recommended")

    def test_missing_video_with_good_audio_image_is_unknown_rescan(self):
        signals = {
            "camera": _signal(0.76, {"status": "ok", "image_confidence": 0.76, "image_quality_score": 0.73, "image_warnings": []}),
            "video": _signal(None, {"status": "missing", "visual_warnings": ["video_missing"]}),
            "voice": _signal(0.78, {"status": "ok", "audio_confidence": 0.78, "audio_quality_score": 0.76, "audio_warnings": []}),
        }
        result = compute_result(signals=signals, quality=assess_quality(signals))

        self.assertEqual(result["risk_level"], "unknown")
        self.assertEqual(result["suggested_action"], "rescan_recommended")
        self.assertIn("video", result["explanation"].lower())

    def test_missing_audio_with_good_video_image_is_unknown_rescan(self):
        signals = {
            "camera": _signal(0.76, {"status": "ok", "image_confidence": 0.76, "image_quality_score": 0.73, "image_warnings": []}),
            "video": _signal(0.82, {"status": "ok", "visual_confidence": 0.82, "visual_quality_score": 0.8, "visual_warnings": []}),
            "voice": _signal(None, {"status": "missing", "audio_warnings": ["audio_missing"]}),
        }
        result = compute_result(signals=signals, quality=assess_quality(signals))

        self.assertEqual(result["risk_level"], "unknown")
        self.assertEqual(result["suggested_action"], "rescan_recommended")
        self.assertIn("audio", result["explanation"].lower())

    def test_missing_major_media_is_unknown_without_quality_metadata(self):
        signals = {
            "camera": _signal(0.76, {"status": "ok", "image_confidence": 0.76, "image_quality_score": 0.73, "image_warnings": []}),
            "video": _signal(None, {"status": "open_failed", "visual_warnings": []}),
            "voice": _signal(0.78, {"status": "ok", "audio_confidence": 0.78, "audio_quality_score": 0.76, "audio_warnings": []}),
        }
        result = compute_result(signals=signals, quality={})

        self.assertEqual(result["risk_level"], "unknown")
        self.assertEqual(result["suggested_action"], "rescan_recommended")
        self.assertIn("video", result["explanation"].lower())

    def test_low_media_quality_creates_explanation_warnings(self):
        signals = {
            "camera": _signal(0.3, {"status": "ok", "image_confidence": 0.3, "image_quality_score": 0.3, "image_warnings": ["image_blurry"]}),
            "video": _signal(0.3, {"status": "ok", "visual_confidence": 0.3, "visual_quality_score": 0.3, "visual_warnings": ["video_too_dark"]}),
            "voice": _signal(0.3, {"status": "ok", "audio_confidence": 0.3, "audio_quality_score": 0.3, "audio_warnings": ["audio_too_quiet"]}),
        }
        result = compute_result(signals=signals, quality=assess_quality(signals))

        explanation = result["explanation"].lower()
        self.assertEqual(result["risk_level"], "unknown")
        self.assertEqual(result["suggested_action"], "rescan_recommended")
        self.assertIn("reduced", explanation)
        self.assertTrue("lighting" in explanation or "volume" in explanation or "blur" in explanation)

    def test_duplicate_scan_result_prevention(self):
        client = DirectusClient(base_url="http://example.com", token="x")
        client.get_scan_result_by_scan_id = MagicMock(return_value={"id": "existing"})
        client.update_item = MagicMock(return_value={"id": "existing"})
        client.create_item = MagicMock(return_value={"id": "new"})
        mode, payload = client.upsert_scan_result("scan-1", {"scan_id": "scan-1"})
        self.assertEqual(mode, "updated")
        self.assertEqual(payload["id"], "existing")
        client.create_item.assert_not_called()

    def test_duplicate_scan_result_conflict_recovers_by_update(self):
        client = DirectusClient(base_url="http://example.com", token="x")
        response = requests.Response()
        response.status_code = 409
        response._content = b'{"errors":[{"message":"duplicate key value violates unique constraint scan_results_scan_id_unique"}]}'
        conflict = requests.HTTPError(response=response)
        client.get_scan_result_by_scan_id = MagicMock(side_effect=[None, {"id": "existing"}])
        client.create_item = MagicMock(side_effect=conflict)
        client.update_item = MagicMock(return_value={"id": "existing"})

        mode, payload = client.upsert_scan_result("scan-1", {"scan_id": "scan-1"})

        self.assertEqual(mode, "updated_after_conflict")
        self.assertEqual(payload["id"], "existing")
        client.create_item.assert_called_once()
        client.update_item.assert_called_once()

    def test_directus_400_logs_response_body(self):
        client = DirectusClient(base_url="http://example.com", token="x")
        response = requests.Response()
        response.status_code = 400
        response.url = "http://example.com/items/scan_results"
        response._content = b'{"errors":[{"message":"Invalid payload. Invalid one-to-many update structure: risk_level"}]}'

        with unittest.mock.patch("directus_client.requests.request", return_value=response):
            with self.assertLogs("ai-server", level="ERROR") as logs:
                with self.assertRaises(requests.HTTPError):
                    client.create_item(
                        "scan_results",
                        {
                            "scan_id": "scan-1",
                            "risk_level": "stable",
                            "confidence": 0.8,
                        },
                    )

        combined = "\n".join(logs.output)
        self.assertIn("status_code=400", combined)
        self.assertIn("/items/scan_results", combined)
        self.assertIn("payload_keys=['confidence', 'risk_level', 'scan_id']", combined)
        self.assertIn("Invalid payload", combined)

    def test_directus_value_too_long_body_is_logged(self):
        client = DirectusClient(base_url="http://example.com", token="x")
        response = requests.Response()
        response.status_code = 400
        response.url = "http://example.com/items/scan_results"
        response._content = (
            b'{"errors":[{"message":"Value \\"Conntinuity Intelligence Engine v1.2\\" for field '
            b'\\"ai_model_version\\" in collection \\"scan_results\\" is too long.",'
            b'"extensions":{"collection":"scan_results","field":"ai_model_version",'
            b'"value":"Conntinuity Intelligence Engine v1.2","code":"VALUE_TOO_LONG"}}]}'
        )

        with unittest.mock.patch("directus_client.requests.request", return_value=response):
            with self.assertLogs("ai-server", level="ERROR") as logs:
                with self.assertRaises(requests.HTTPError):
                    client.create_item(
                        "scan_results",
                        {
                            "scan_id": "scan-1",
                            "ai_model_version": "Conntinuity Intelligence Engine v1.2",
                        },
                    )

        combined = "\n".join(logs.output)
        self.assertIn("VALUE_TOO_LONG", combined)
        self.assertIn("ai_model_version", combined)
        self.assertIn("Conntinuity Intelligence Engine v1.2", combined)


if __name__ == "__main__":
    unittest.main()
