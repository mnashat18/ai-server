import unittest
from unittest.mock import MagicMock

from baseline import (
    baseline_ready_for_personalized_scoring,
    baseline_signal_payload,
    baseline_status_payload,
    is_scan_eligible_for_baseline,
)
from directus_client import DirectusClient
from quality import assess_quality
from scoring import clamp_confidence, compute_result
from utils import sanitize_payload
import requests
import scoring
import video


def _signal(score, details):
    return {"score": score, "details": details}


class PipelineTests(unittest.TestCase):
    def test_two_qualified_scans_remain_inactive_baseline(self):
        baseline = None
        signals = {
            "camera": _signal(0.8, {"status": "ok", "avg_ear": 0.31, "left_right_eye_asymmetry": 0.02}),
            "video": _signal(0.82, {"status": "ok"}),
            "voice": _signal(0.81, {"status": "ok", "rms_energy": 0.03}),
        }
        for _ in range(2):
            baseline = baseline_signal_payload(baseline, signals=signals, scanned_at="2026-07-01T08:00:00Z")

        status = baseline_status_payload(baseline)
        self.assertEqual(baseline["scan_count"], 2)
        self.assertFalse(baseline["is_active"])
        self.assertFalse(status["is_active"])
        self.assertTrue(status["is_provisional"])

    def test_three_qualified_scans_create_active_baseline(self):
        baseline = None
        signals = {
            "camera": _signal(0.8, {"status": "ok", "avg_ear": 0.31, "left_right_eye_asymmetry": 0.02}),
            "video": _signal(0.82, {"status": "ok"}),
            "voice": _signal(0.81, {"status": "ok", "rms_energy": 0.03}),
        }
        for _ in range(3):
            baseline = baseline_signal_payload(baseline, signals=signals, scanned_at="2026-07-01T08:00:00Z")

        status = baseline_status_payload(baseline)
        self.assertEqual(baseline["scan_count"], 3)
        self.assertTrue(baseline["is_active"])
        self.assertTrue(status["is_active"])
        self.assertFalse(status["is_provisional"])

    def test_baseline_uses_robust_median_and_mad(self):
        baseline = None
        values = [0.1, 0.1, 0.1, 0.9]
        for value in values:
            baseline = baseline_signal_payload(
                baseline,
                signals={
                    "camera": _signal(value, {"status": "ok", "avg_ear": value, "left_right_eye_asymmetry": 0.01}),
                    "video": _signal(0.82, {"status": "ok"}),
                    "voice": _signal(0.81, {"status": "ok", "rms_energy": 0.03}),
                },
                scanned_at="2026-07-01T08:00:00Z",
            )

        face_stats = baseline["face_avg"]["feature_stats"]["open_eye_aperture"]
        self.assertEqual(face_stats["median"], 0.1)
        self.assertGreaterEqual(face_stats["mad"], 0.02)
        self.assertEqual(face_stats["count"], 4)

    def test_baseline_sample_history_is_bounded(self):
        baseline = None
        for index in range(12):
            baseline = baseline_signal_payload(
                baseline,
                signals={
                    "camera": _signal(0.2 + (index * 0.01), {"status": "ok", "avg_ear": 0.2 + (index * 0.01), "left_right_eye_asymmetry": 0.01}),
                    "video": _signal(0.82, {"status": "ok"}),
                    "voice": _signal(0.81, {"status": "ok", "rms_energy": 0.03 + (index * 0.001)}),
                },
                scanned_at="2026-07-01T08:00:00Z",
            )

        self.assertLessEqual(len(baseline["face_avg"]["feature_samples"]["open_eye_aperture"]), 9)
        self.assertLessEqual(len(baseline["voice_avg"]["feature_samples"]["normalized_voice_energy"]), 9)

    def test_baseline_signal_payload_uses_only_existing_fields(self):
        payload = baseline_signal_payload(
            None,
            signals={
                "camera": _signal(0.8, {"status": "ok", "avg_ear": 0.3, "left_right_eye_asymmetry": 0.01}),
                "video": _signal(0.82, {"status": "ok"}),
                "voice": _signal(0.81, {"status": "ok", "rms_energy": 0.03, "speech_rate": 1.8}),
            },
            scanned_at="2026-07-01T08:00:00Z",
        )

        self.assertEqual(set(payload.keys()), {"scan_count", "face_avg", "voice_avg", "reaction_avg", "is_active"})

    def test_low_quality_scan_is_not_baseline_eligible(self):
        eligible = is_scan_eligible_for_baseline(
            quality_result={"status": "weak", "weak": True, "retake_required": False, "media_quality": {"aggregate_quality": 0.2}, "confidence_multiplier": 0.3},
            validation_result={"critical_errors": [], "warnings": []},
            result={"confidence": 0.8, "risk_level": "stable", "retake_required": False},
            signals={"camera": {"details": {}}},
        )
        self.assertFalse(eligible)

    def test_missing_required_speech_is_not_baseline_eligible(self):
        eligible = is_scan_eligible_for_baseline(
            quality_result={"status": "passed", "weak": False, "retake_required": False, "media_quality": {"aggregate_quality": 0.9}, "confidence_multiplier": 0.9},
            validation_result={"critical_errors": [], "warnings": ["speech_not_detected"]},
            result={"confidence": 0.8, "risk_level": "stable", "retake_required": False},
            signals={"camera": {"details": {}}},
            expected_phrase="please say continuity ready",
        )
        self.assertFalse(eligible)

    def test_elevated_fatigue_scan_is_not_baseline_eligible(self):
        eligible = is_scan_eligible_for_baseline(
            quality_result={"status": "passed", "weak": False, "retake_required": False, "media_quality": {"aggregate_quality": 0.9}, "confidence_multiplier": 0.9},
            validation_result={"critical_errors": [], "warnings": []},
            result={"confidence": 0.8, "risk_level": "elevated_fatigue", "retake_required": False},
            signals={"camera": {"details": {}}},
        )
        self.assertFalse(eligible)

    def test_high_risk_scan_is_not_baseline_eligible(self):
        eligible = is_scan_eligible_for_baseline(
            quality_result={"status": "passed", "weak": False, "retake_required": False, "media_quality": {"aggregate_quality": 0.9}, "confidence_multiplier": 0.9},
            validation_result={"critical_errors": [], "warnings": []},
            result={"confidence": 0.8, "risk_level": "high_risk", "retake_required": False},
            signals={"camera": {"details": {}}},
        )
        self.assertFalse(eligible)

    def test_retake_required_scan_is_not_baseline_eligible(self):
        eligible = is_scan_eligible_for_baseline(
            quality_result={"status": "passed", "weak": False, "retake_required": True, "media_quality": {"aggregate_quality": 0.9}, "confidence_multiplier": 0.9},
            validation_result={"critical_errors": [], "warnings": []},
            result={"confidence": 0.8, "risk_level": "stable", "retake_required": True},
            signals={"camera": {"details": {}}},
        )
        self.assertFalse(eligible)

    def test_existing_legacy_baseline_json_remains_readable(self):
        signals = {
            "camera": _signal(0.76, {"status": "ok", "image_confidence": 0.76, "image_quality_score": 0.73, "image_warnings": []}),
            "video": _signal(0.82, {"status": "ok", "visual_confidence": 0.82, "visual_quality_score": 0.8, "visual_warnings": []}),
            "voice": _signal(0.78, {"status": "ok", "audio_confidence": 0.78, "audio_quality_score": 0.76, "audio_warnings": []}),
        }
        quality = assess_quality(signals)
        legacy_baseline = {
            "scan_count": 6,
            "is_active": True,
            "face_avg": {"avg": 0.8, "std": 0.04},
            "voice_avg": {"avg": 0.78, "std": 0.05},
            "reaction_avg": {"avg": 0.9, "std": 0.02},
        }
        result = compute_result(signals=signals, quality=quality, baseline=legacy_baseline, baseline_used=True)
        self.assertTrue(result["baseline_used"])
        self.assertIn("baseline_drifts", result["face_metrics"])
        self.assertIn("open_eye_aperture", result["face_metrics"]["baseline_drifts"])
        self.assertIn("normalized_voice_energy", result["voice_metrics"]["baseline_drifts"])

    def test_static_eyes_closed_is_not_sustained_eye_closure(self):
        eligible = is_scan_eligible_for_baseline(
            quality_result={"status": "passed", "weak": False, "retake_required": False, "media_quality": {"aggregate_quality": 0.9}, "confidence_multiplier": 0.9},
            validation_result={"critical_errors": [], "warnings": []},
            result={"confidence": 0.8, "risk_level": "stable", "retake_required": False},
            signals={"camera": {"details": {"avg_ear": 0.22, "left_right_eye_asymmetry": 0.01, "eyes_closed": True}}, "voice": {"details": {"rms_energy": 0.03}}},
        )
        self.assertTrue(eligible)

    def test_four_closed_frames_adjacent_at_30_fps_do_not_sustain(self):
        observations = [
            {"window_id": 0, "frame_index": 10, "timestamp": 0.00, "usable": True, "bright_enough": True, "sharp_enough": True, "face_visible": True, "landmark_valid": True, "eye_closed": True},
            {"window_id": 0, "frame_index": 11, "timestamp": 0.03, "usable": True, "bright_enough": True, "sharp_enough": True, "face_visible": True, "landmark_valid": True, "eye_closed": True},
            {"window_id": 0, "frame_index": 12, "timestamp": 0.07, "usable": True, "bright_enough": True, "sharp_enough": True, "face_visible": True, "landmark_valid": True, "eye_closed": True},
            {"window_id": 0, "frame_index": 13, "timestamp": 0.10, "usable": True, "bright_enough": True, "sharp_enough": True, "face_visible": True, "landmark_valid": True, "eye_closed": True},
        ]
        longest, window_ms, window_seconds = video._longest_temporal_eye_closure_streak(observations)

        self.assertEqual(longest, 0)
        self.assertEqual(window_ms, 0)
        self.assertEqual(window_seconds, 0.0)

    def test_four_closed_samples_spanning_real_time_can_sustain(self):
        observations = [
            {"window_id": 0, "frame_index": 20, "timestamp": 0.00, "usable": True, "bright_enough": True, "sharp_enough": True, "face_visible": True, "landmark_valid": True, "eye_closed": True},
            {"window_id": 0, "frame_index": 25, "timestamp": 0.18, "usable": True, "bright_enough": True, "sharp_enough": True, "face_visible": True, "landmark_valid": True, "eye_closed": True},
            {"window_id": 0, "frame_index": 30, "timestamp": 0.36, "usable": True, "bright_enough": True, "sharp_enough": True, "face_visible": True, "landmark_valid": True, "eye_closed": True},
            {"window_id": 0, "frame_index": 36, "timestamp": 0.54, "usable": True, "bright_enough": True, "sharp_enough": True, "face_visible": True, "landmark_valid": True, "eye_closed": True},
        ]
        longest, window_ms, window_seconds = video._longest_temporal_eye_closure_streak(observations)

        self.assertEqual(longest, 4)
        self.assertGreaterEqual(window_seconds, 0.45)
        self.assertLessEqual(window_seconds, 1.2)
        self.assertGreater(window_ms, 0)

    def test_isolated_closed_samples_from_different_bursts_do_not_combine(self):
        observations = [
            {"window_id": 0, "frame_index": 2, "timestamp": 0.00, "usable": True, "bright_enough": True, "sharp_enough": True, "face_visible": True, "landmark_valid": True, "eye_closed": True},
            {"window_id": 0, "frame_index": 4, "timestamp": 0.18, "usable": True, "bright_enough": True, "sharp_enough": True, "face_visible": True, "landmark_valid": True, "eye_closed": False},
            {"window_id": 0, "frame_index": 6, "timestamp": 0.36, "usable": True, "bright_enough": True, "sharp_enough": True, "face_visible": True, "landmark_valid": True, "eye_closed": True},
            {"window_id": 0, "frame_index": 8, "timestamp": 0.54, "usable": True, "bright_enough": True, "sharp_enough": True, "face_visible": True, "landmark_valid": True, "eye_closed": True},
            {"window_id": 1, "frame_index": 40, "timestamp": 1.00, "usable": True, "bright_enough": True, "sharp_enough": True, "face_visible": True, "landmark_valid": True, "eye_closed": True},
            {"window_id": 1, "frame_index": 44, "timestamp": 1.18, "usable": True, "bright_enough": True, "sharp_enough": True, "face_visible": True, "landmark_valid": True, "eye_closed": False},
            {"window_id": 1, "frame_index": 48, "timestamp": 1.36, "usable": True, "bright_enough": True, "sharp_enough": True, "face_visible": True, "landmark_valid": True, "eye_closed": True},
            {"window_id": 1, "frame_index": 52, "timestamp": 1.54, "usable": True, "bright_enough": True, "sharp_enough": True, "face_visible": True, "landmark_valid": True, "eye_closed": True},
        ]
        longest, window_ms, window_seconds = video._longest_temporal_eye_closure_streak(observations)

        self.assertEqual(longest, 0)
        self.assertEqual(window_ms, 0)
        self.assertEqual(window_seconds, 0.0)

    def test_invalid_sample_breaks_the_sequence(self):
        break_cases = [
            {"eye_closed": False},
            {"sharp_enough": False},
            {"bright_enough": False},
            {"face_visible": False},
            {"landmark_valid": False},
            {"usable": False},
        ]
        for case in break_cases:
            with self.subTest(case=case):
                observations = [
                    {"window_id": 0, "frame_index": 20, "timestamp": 0.00, "usable": True, "bright_enough": True, "sharp_enough": True, "face_visible": True, "landmark_valid": True, "eye_closed": True},
                    {"window_id": 0, "frame_index": 25, "timestamp": 0.18, "usable": True, "bright_enough": True, "sharp_enough": True, "face_visible": True, "landmark_valid": True, "eye_closed": True},
                    {"window_id": 0, "frame_index": 30, "timestamp": 0.36, "usable": case.get("usable", True), "bright_enough": case.get("bright_enough", True), "sharp_enough": case.get("sharp_enough", True), "face_visible": case.get("face_visible", True), "landmark_valid": case.get("landmark_valid", True), "eye_closed": case.get("eye_closed", True)},
                    {"window_id": 0, "frame_index": 36, "timestamp": 0.54, "usable": True, "bright_enough": True, "sharp_enough": True, "face_visible": True, "landmark_valid": True, "eye_closed": True},
                ]
                longest, window_ms, window_seconds = video._longest_temporal_eye_closure_streak(observations)

                self.assertEqual(longest, 0)
                self.assertEqual(window_ms, 0)
                self.assertEqual(window_seconds, 0.0)

    def test_eye_closure_sampling_windows_are_bounded_and_spread_out(self):
        windows = video._eye_closure_sample_windows(frame_count=120, fps=30.0, duration_seconds=4.0)

        self.assertLessEqual(sum(len(window) for window in windows), 8)
        self.assertLessEqual(len(windows), 2)
        for window in windows:
            self.assertLessEqual(len(window), 4)
            self.assertEqual(len({sample["frame_index"] for sample in window}), 4)
            timestamps = [sample["timestamp"] for sample in window]
            self.assertGreaterEqual(timestamps[-1] - timestamps[0], 0.45)
            self.assertLessEqual(timestamps[-1] - timestamps[0], 1.2)

    def test_eye_closure_sampling_rejects_short_videos(self):
        windows = video._eye_closure_sample_windows(frame_count=8, fps=30.0, duration_seconds=0.2)
        self.assertEqual(windows, [])

    def test_naturally_narrow_eye_user_can_create_baseline_without_temporal_gate(self):
        eligible = is_scan_eligible_for_baseline(
            quality_result={"status": "passed", "weak": False, "retake_required": False, "media_quality": {"aggregate_quality": 0.9}, "confidence_multiplier": 0.9},
            validation_result={"critical_errors": [], "warnings": []},
            result={"confidence": 0.8, "risk_level": "stable", "retake_required": False},
            signals={"camera": {"details": {"avg_ear": 0.16, "left_right_eye_asymmetry": 0.01}}, "voice": {"details": {"rms_energy": 0.03}}},
        )
        self.assertTrue(eligible)

    def test_baseline_drift_compares_matching_raw_metrics_only(self):
        signals = {
            "camera": _signal(0.76, {"status": "ok", "avg_ear": 0.26, "left_right_eye_asymmetry": 0.03, "image_confidence": 0.76, "image_quality_score": 0.73, "image_warnings": []}),
            "video": _signal(0.82, {"status": "ok", "visual_confidence": 0.82, "visual_quality_score": 0.8, "visual_warnings": []}),
            "voice": _signal(0.78, {"status": "ok", "rms_energy": 0.024, "audio_confidence": 0.78, "audio_quality_score": 0.76, "audio_warnings": []}),
        }
        quality = assess_quality(signals)
        baseline = {
            "scan_count": 6,
            "is_active": True,
            "face_avg": {
                "schema_version": 2,
                "feature_stats": {
                    "open_eye_aperture": {"median": 0.30, "mad": 0.01},
                    "left_right_eye_asymmetry": {"median": 0.01, "mad": 0.005},
                },
            },
            "voice_avg": {
                "schema_version": 2,
                "feature_stats": {
                    "normalized_voice_energy": {"median": 0.03, "mad": 0.002},
                    "speech_rate": {"median": 2.1, "mad": 0.2},
                },
            },
        }
        result = compute_result(signals=signals, quality=quality, baseline=baseline, baseline_used=True)
        self.assertIsNotNone(result["face_metrics"]["baseline_drifts"]["open_eye_aperture"])
        self.assertIsNotNone(result["face_metrics"]["baseline_drifts"]["left_right_eye_asymmetry"])
        self.assertIsNotNone(result["voice_metrics"]["baseline_drifts"]["normalized_voice_energy"])
        self.assertIsNone(result["voice_metrics"]["baseline_drifts"]["speech_rate"])

    def test_inactive_baseline_does_not_influence_scoring(self):
        signals = {
            "camera": _signal(0.76, {"status": "ok", "image_confidence": 0.76, "image_quality_score": 0.73, "image_warnings": [], "avg_ear": 0.18, "left_right_eye_asymmetry": 0.01}),
            "video": _signal(0.82, {"status": "ok", "visual_confidence": 0.82, "visual_quality_score": 0.8, "visual_warnings": []}),
            "voice": _signal(0.78, {"status": "ok", "audio_confidence": 0.78, "audio_quality_score": 0.76, "audio_warnings": [], "rms_energy": 0.014, "speech_rate": 1.1}),
        }
        quality = assess_quality(signals)
        inactive_baseline = {
            "scan_count": 4,
            "is_active": False,
            "face_avg": {"schema_version": 2, "feature_stats": {"open_eye_aperture": {"median": 0.18, "mad": 0.02, "count": 4}}},
            "voice_avg": {"schema_version": 2, "feature_stats": {"normalized_voice_energy": {"median": 0.014, "mad": 0.02, "count": 4}}},
            "reaction_avg": {"schema_version": 2, "feature_stats": {}},
        }

        without_baseline = compute_result(signals=signals, quality=quality, baseline=None, baseline_used=False)
        ignored_baseline = compute_result(signals=signals, quality=quality, baseline=inactive_baseline, baseline_used=False)

        self.assertEqual(ignored_baseline["risk_level"], without_baseline["risk_level"])
        self.assertEqual(ignored_baseline["confidence"], without_baseline["confidence"])
        self.assertFalse(ignored_baseline["baseline_used"])

    def test_active_baseline_can_relax_static_false_penalties(self):
        signals = {
            "camera": _signal(0.78, {"status": "ok", "image_confidence": 0.78, "image_quality_score": 0.75, "image_warnings": [], "avg_ear": 0.18, "left_right_eye_asymmetry": 0.01}),
            "video": _signal(0.8, {"status": "ok", "visual_confidence": 0.8, "visual_quality_score": 0.78, "visual_warnings": []}),
            "voice": _signal(0.77, {"status": "ok", "audio_confidence": 0.77, "audio_quality_score": 0.75, "audio_warnings": [], "rms_energy": 0.014, "speech_rate": 1.1}),
        }
        quality = assess_quality(signals)
        active_baseline = {
            "scan_count": 4,
            "is_active": True,
            "face_avg": {
                "schema_version": 2,
                "feature_stats": {
                    "open_eye_aperture": {"median": 0.18, "mad": 0.02, "count": 4},
                    "left_right_eye_asymmetry": {"median": 0.01, "mad": 0.02, "count": 4},
                },
            },
            "voice_avg": {
                "schema_version": 2,
                "feature_stats": {
                    "normalized_voice_energy": {"median": 0.014, "mad": 0.02, "count": 4},
                    "speech_rate": {"median": 1.1, "mad": 0.2, "count": 4},
                },
            },
            "reaction_avg": {"schema_version": 2, "feature_stats": {}},
        }

        baseline_free = compute_result(signals=signals, quality=quality, baseline=None, baseline_used=False)
        personalized = compute_result(signals=signals, quality=quality, baseline=active_baseline, baseline_used=True)

        self.assertGreater(personalized["confidence"], baseline_free["confidence"])
        self.assertIn("baseline", personalized["explanation"].lower())
        self.assertIn("normal range", personalized["explanation"].lower())
        self.assertFalse(personalized["risk_level"] == "high_risk")

    def test_personalized_gate_requires_active_valid_baseline(self):
        quality_result = {"status": "passed", "weak": False, "retake_required": False, "media_quality": {"aggregate_quality": 0.9}, "confidence_multiplier": 0.9, "warnings": []}
        validation_result = {"critical_errors": [], "warnings": []}
        result = {"confidence": 0.8, "risk_level": "stable", "retake_required": False}
        active_baseline = {
            "scan_count": 4,
            "is_active": True,
            "face_avg": {"schema_version": 2, "feature_stats": {"open_eye_aperture": {"median": 0.18, "mad": 0.02, "count": 4}}},
            "voice_avg": {"schema_version": 2, "feature_stats": {"normalized_voice_energy": {"median": 0.014, "mad": 0.02, "count": 4}}},
            "reaction_avg": {"schema_version": 2, "feature_stats": {}},
        }
        inactive_baseline = dict(active_baseline, is_active=False)
        malformed_baseline = {
            "scan_count": 4,
            "is_active": True,
            "face_avg": {"schema_version": 2, "feature_stats": {"open_eye_aperture": {"median": None, "mad": 0.02, "count": 4}}},
            "voice_avg": {"schema_version": 2, "feature_stats": {"normalized_voice_energy": {"median": 0.014, "mad": 0.02, "count": 4}}},
            "reaction_avg": {"schema_version": 2, "feature_stats": {}},
        }

        self.assertTrue(
            baseline_ready_for_personalized_scoring(
                active_baseline,
                quality_result=quality_result,
                validation_result=validation_result,
                result=result,
                unique_row=True,
            )
        )
        self.assertFalse(
            baseline_ready_for_personalized_scoring(
                inactive_baseline,
                quality_result=quality_result,
                validation_result=validation_result,
                result=result,
                unique_row=True,
            )
        )
        self.assertFalse(
            baseline_ready_for_personalized_scoring(
                malformed_baseline,
                quality_result=quality_result,
                validation_result=validation_result,
                result=result,
                unique_row=True,
            )
        )

    def test_personal_deviation_alone_cannot_become_high_risk(self):
        signals = {
            "camera": _signal(0.82, {"status": "ok", "image_confidence": 0.82, "image_quality_score": 0.8, "image_warnings": [], "avg_ear": 0.17, "left_right_eye_asymmetry": 0.01}),
            "video": _signal(0.84, {"status": "ok", "visual_confidence": 0.84, "visual_quality_score": 0.82, "visual_warnings": []}),
            "voice": _signal(0.81, {"status": "ok", "audio_confidence": 0.81, "audio_quality_score": 0.8, "audio_warnings": [], "rms_energy": 0.03, "speech_rate": 2.0}),
        }
        quality = assess_quality(signals)
        active_baseline = {
            "scan_count": 4,
            "is_active": True,
            "face_avg": {
                "schema_version": 2,
                "feature_stats": {
                    "open_eye_aperture": {"median": 0.32, "mad": 0.02, "count": 4},
                    "left_right_eye_asymmetry": {"median": 0.01, "mad": 0.02, "count": 4},
                },
            },
            "voice_avg": {
                "schema_version": 2,
                "feature_stats": {
                    "normalized_voice_energy": {"median": 0.03, "mad": 0.02, "count": 4},
                    "speech_rate": {"median": 2.0, "mad": 0.2, "count": 4},
                },
            },
            "reaction_avg": {"schema_version": 2, "feature_stats": {}},
        }

        result = compute_result(signals=signals, quality=quality, baseline=active_baseline, baseline_used=True)

        self.assertNotEqual(result["risk_level"], "high_risk")
        self.assertIn("open_eye_aperture", result["fusion_details"]["baseline_flags"])

    def test_confidence_clamping(self):
        self.assertEqual(clamp_confidence(1.5), 1.0)
        self.assertEqual(clamp_confidence(-0.2), 0.0)
        self.assertIsNone(clamp_confidence(None))

    def test_risk_level_contract_is_exact(self):
        expected = {"stable", "low_focus", "elevated_fatigue", "high_risk"}
        self.assertEqual(scoring.VALID_RISK_LEVELS, expected)
        self.assertEqual(set(main.SCAN_RESULT_CHOICE_ALIASES["risk_level"].keys()), expected)

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
        self.assertIn(result["risk_level"], {"stable", "low_focus", "elevated_fatigue", "high_risk"})

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

    def test_single_eye_blink_does_not_reduce_valid_scan(self):
        clean_signals = {
            "camera": _signal(0.8, {"status": "ok", "image_confidence": 0.8, "image_quality_score": 0.8, "image_warnings": []}),
            "video": _signal(
                0.84,
                {
                    "status": "ok",
                    "visual_confidence": 0.84,
                    "visual_quality_score": 0.82,
                    "visual_warnings": [],
                    "reliable_eye_landmarks": True,
                    "sustained_eye_closure": False,
                },
            ),
            "voice": _signal(0.8, {"status": "ok", "audio_confidence": 0.8, "audio_quality_score": 0.79, "audio_warnings": []}),
        }
        blink_signals = {
            **clean_signals,
            "video": _signal(
                0.84,
                {
                    "status": "ok",
                    "visual_confidence": 0.84,
                    "visual_quality_score": 0.82,
                    "visual_warnings": [],
                    "reliable_eye_landmarks": True,
                    "sustained_eye_closure": False,
                },
            ),
        }

        clean = compute_result(signals=clean_signals, quality=assess_quality(clean_signals))
        blink = compute_result(signals=blink_signals, quality=assess_quality(blink_signals))

        self.assertEqual(blink["risk_level"], clean["risk_level"])
        self.assertGreaterEqual(blink["confidence"], 0.45)

    def test_sustained_eye_closure_with_reliable_evidence_cannot_return_stable(self):
        signals = {
            "camera": _signal(0.82, {"status": "ok", "image_confidence": 0.82, "image_quality_score": 0.82, "image_warnings": []}),
            "video": _signal(
                0.8,
                {
                    "status": "ok",
                    "visual_confidence": 0.8,
                    "visual_quality_score": 0.8,
                    "visual_warnings": ["sustained_eye_closure"],
                    "reliable_eye_landmarks": True,
                    "sustained_eye_closure": True,
                },
            ),
            "voice": _signal(0.8, {"status": "ok", "audio_confidence": 0.8, "audio_quality_score": 0.8, "audio_warnings": []}),
        }
        baseline = {
            "scan_count": 4,
            "is_active": True,
            "face_avg": {"schema_version": 2, "feature_stats": {"open_eye_aperture": {"median": 0.22, "mad": 0.02, "count": 4}}},
            "voice_avg": {"schema_version": 2, "feature_stats": {"normalized_voice_energy": {"median": 0.03, "mad": 0.02, "count": 4}}},
            "reaction_avg": {"schema_version": 2, "feature_stats": {}},
        }
        result = compute_result(signals=signals, quality=assess_quality(signals), baseline=baseline, baseline_used=True)

        self.assertNotEqual(result["risk_level"], "stable")
        self.assertIn("eye closure", result["explanation"].lower())

    def test_hard_safety_signals_are_not_rescued_by_baseline(self):
        baseline = {
            "scan_count": 4,
            "is_active": True,
            "face_avg": {"schema_version": 2, "feature_stats": {"open_eye_aperture": {"median": 0.18, "mad": 0.02, "count": 4}}},
            "voice_avg": {"schema_version": 2, "feature_stats": {"normalized_voice_energy": {"median": 0.014, "mad": 0.02, "count": 4}}},
            "reaction_avg": {"schema_version": 2, "feature_stats": {}},
        }
        cases = [
            {"warnings": ["video_too_dark"], "failure_reason": None, "label": "dark"},
            {"warnings": ["video_blurry"], "failure_reason": None, "label": "blur"},
            {"warnings": ["face_not_visible"], "failure_reason": None, "label": "face"},
            {"warnings": ["audio_missing"], "failure_reason": "missing_media", "label": "missing_media"},
            {"warnings": ["audio_too_noisy"], "failure_reason": None, "label": "audio"},
            {"warnings": ["speech_not_detected"], "failure_reason": None, "label": "speech"},
        ]
        for case in cases:
            with self.subTest(case=case["label"]):
                signals = {
                    "camera": _signal(0.82, {"status": "ok", "image_confidence": 0.82, "image_quality_score": 0.82, "image_warnings": []}),
                    "video": _signal(
                        0.8,
                        {
                            "status": "ok",
                            "visual_confidence": 0.8,
                            "visual_quality_score": 0.8,
                            "visual_warnings": case["warnings"],
                        },
                    ),
                    "voice": _signal(
                        0.8,
                        {
                            "status": "ok",
                            "audio_confidence": 0.8,
                            "audio_quality_score": 0.8,
                            "audio_warnings": case["warnings"],
                        },
                    ),
                }
                quality = assess_quality(signals)
                quality["status"] = "failed"
                quality["weak"] = True
                quality["failure_reason"] = case["failure_reason"] or "low_quality_media"
                result = compute_result(signals=signals, quality=quality, baseline=baseline, baseline_used=True)
                self.assertNotEqual(result["risk_level"], "stable")
                self.assertNotEqual(result["suggested_action"], "continue_normal_activity")

    def test_eye_closure_only_caps_at_elevated_fatigue_before_baseline(self):
        self.assertEqual(
            scoring._risk_level(
                30,
                0.72,
                ["open_eye_aperture"],
                {"status": "passed", "warnings": ["sustained_eye_closure"], "weak": False},
            ),
            "elevated_fatigue",
        )

    def test_non_eye_high_risk_behavior_is_still_available(self):
        self.assertEqual(
            scoring._risk_level(
                30,
                0.72,
                [],
                {"status": "passed", "warnings": [], "weak": False},
            ),
            "high_risk",
        )

    def test_low_confidence_does_not_return_stable(self):
        signals = {
            "camera": _signal(None, {"status": "missing", "image_warnings": ["image_missing"]}),
            "video": _signal(None, {"status": "missing", "visual_warnings": ["video_missing"]}),
            "voice": _signal(None, {"status": "missing", "audio_warnings": ["audio_missing"]}),
        }
        result = compute_result(signals=signals, quality=assess_quality(signals))

        self.assertLess(result["confidence"], 0.45)
        self.assertEqual(result["risk_level"], "low_focus")
        self.assertEqual(result["suggested_action"], "rescan_recommended")
        self.assertLess(result["readiness_score"], 52)

    def test_missing_video_with_good_audio_image_is_unknown_rescan(self):
        signals = {
            "camera": _signal(0.76, {"status": "ok", "image_confidence": 0.76, "image_quality_score": 0.73, "image_warnings": []}),
            "video": _signal(None, {"status": "missing", "visual_warnings": ["video_missing"]}),
            "voice": _signal(0.78, {"status": "ok", "audio_confidence": 0.78, "audio_quality_score": 0.76, "audio_warnings": []}),
        }
        result = compute_result(signals=signals, quality=assess_quality(signals))

        self.assertEqual(result["risk_level"], "low_focus")
        self.assertEqual(result["suggested_action"], "rescan_recommended")
        self.assertIn("video", result["explanation"].lower())
        self.assertNotEqual(result["risk_level"], "high_risk")

    def test_missing_audio_with_good_video_image_is_unknown_rescan(self):
        signals = {
            "camera": _signal(0.76, {"status": "ok", "image_confidence": 0.76, "image_quality_score": 0.73, "image_warnings": []}),
            "video": _signal(0.82, {"status": "ok", "visual_confidence": 0.82, "visual_quality_score": 0.8, "visual_warnings": []}),
            "voice": _signal(None, {"status": "missing", "audio_warnings": ["audio_missing"]}),
        }
        result = compute_result(signals=signals, quality=assess_quality(signals))

        self.assertEqual(result["risk_level"], "low_focus")
        self.assertEqual(result["suggested_action"], "rescan_recommended")
        self.assertIn("audio", result["explanation"].lower())

    def test_missing_major_media_is_unknown_without_quality_metadata(self):
        signals = {
            "camera": _signal(0.76, {"status": "ok", "image_confidence": 0.76, "image_quality_score": 0.73, "image_warnings": []}),
            "video": _signal(None, {"status": "open_failed", "visual_warnings": []}),
            "voice": _signal(0.78, {"status": "ok", "audio_confidence": 0.78, "audio_quality_score": 0.76, "audio_warnings": []}),
        }
        result = compute_result(signals=signals, quality={})

        self.assertEqual(result["risk_level"], "low_focus")
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
        self.assertEqual(result["risk_level"], "low_focus")
        self.assertEqual(result["suggested_action"], "rescan_recommended")
        self.assertIn("reduced", explanation)
        self.assertTrue("lighting" in explanation or "volume" in explanation or "blur" in explanation)

    def test_quiet_but_usable_speech_is_not_automatically_treated_as_fatigue(self):
        signals = {
            "camera": _signal(0.78, {"status": "ok", "image_confidence": 0.78, "image_quality_score": 0.77, "image_warnings": []}),
            "video": _signal(0.81, {"status": "ok", "visual_confidence": 0.81, "visual_quality_score": 0.8, "visual_warnings": []}),
            "voice": _signal(
                0.74,
                {
                    "status": "ok",
                    "audio_confidence": 0.74,
                    "audio_quality_score": 0.62,
                    "audio_warnings": [],
                    "speech_presence_score": 0.64,
                    "quiet_but_usable": True,
                },
            ),
        }
        result = compute_result(signals=signals, quality=assess_quality(signals))

        self.assertIn(result["risk_level"], {"stable", "low_focus"})
        self.assertNotEqual(result["risk_level"], "high_risk")

    def test_poor_audio_quality_alone_cannot_become_high_risk(self):
        signals = {
            "camera": _signal(0.86, {"status": "ok", "image_confidence": 0.86, "image_quality_score": 0.85, "image_warnings": []}),
            "video": _signal(0.88, {"status": "ok", "visual_confidence": 0.88, "visual_quality_score": 0.87, "visual_warnings": []}),
            "voice": _signal(0.22, {"status": "ok", "audio_confidence": 0.22, "audio_quality_score": 0.2, "audio_warnings": ["audio_too_noisy", "speech_not_detected"]}),
        }
        result = compute_result(signals=signals, quality=assess_quality(signals))

        self.assertNotEqual(result["risk_level"], "high_risk")
        self.assertEqual(result["suggested_action"], "rescan_recommended")

    def test_invalid_capture_quality_alone_does_not_become_high_risk(self):
        signals = {
            "camera": _signal(0.95, {"status": "ok", "image_confidence": 0.95, "image_quality_score": 0.95, "image_warnings": []}),
            "video": _signal(0.95, {"status": "ok", "visual_confidence": 0.95, "visual_quality_score": 0.95, "visual_warnings": []}),
            "voice": _signal(0.95, {"status": "missing", "audio_warnings": ["audio_missing"]}),
        }
        result = compute_result(signals=signals, quality=assess_quality(signals))

        self.assertEqual(result["risk_level"], "low_focus")
        self.assertNotEqual(result["risk_level"], "high_risk")
        self.assertLess(result["confidence"], 0.45)

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

    def test_one_business_profile_member_baseline_is_enforced_by_lookup_order(self):
        client = DirectusClient(base_url="http://example.com", token="x")
        client.list_items = MagicMock(return_value=[])
        client.get_employee_baseline("member-1", "bp-1")
        _, kwargs = client.list_items.call_args
        self.assertEqual(kwargs["sort"], "-scan_count,-date_updated")

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
