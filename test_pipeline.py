import unittest
import json
import time
from contextlib import ExitStack
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
import main


def _signal(score, details):
    return {"score": score, "details": details}


class _FakePipeEndpoint:
    def __init__(self, shared_state):
        self._shared_state = shared_state
        self.closed = False

    def send(self, payload):
        self._shared_state["payload"] = payload
        self._shared_state["ready"] = True

    def poll(self, timeout=None):
        return bool(self._shared_state.get("ready"))

    def recv(self):
        self._shared_state["ready"] = False
        return self._shared_state.get("payload")

    def close(self):
        self.closed = True


class _FakeProcess:
    def __init__(self, target, args, daemon, behavior):
        self.target = target
        self.args = args
        self.daemon = daemon
        self.behavior = behavior or {}
        self.sentinel = object()
        self.started = False
        self.terminated = False
        self.killed = False
        self.join_calls = []
        self.closed = False
        self.exitcode = None
        self._alive = bool(self.behavior.get("alive_after_start", False))

    def start(self):
        self.started = True
        if self.behavior.get("invoke_target", True) and self.target is not None:
            self.target(*self.args)
            if not self.behavior.get("alive_after_target", False):
                self._alive = False
                self.exitcode = 0
        else:
            self._alive = bool(self.behavior.get("alive_after_start", True))

    def is_alive(self):
        return self._alive

    def terminate(self):
        self.terminated = True
        if not self.behavior.get("requires_kill", False):
            self._alive = False
            self.exitcode = -15

    def kill(self):
        self.killed = True
        self._alive = False
        self.exitcode = -9

    def join(self, timeout=None):
        self.join_calls.append(timeout)
        if self.terminated and self.behavior.get("requires_kill", False) and self.killed:
            self._alive = False

    def close(self):
        self.closed = True


class _FakeContext:
    def __init__(self, behaviors=None):
        self.behaviors = list(behaviors or [])
        self.processes = []

    def Pipe(self, duplex=False):
        shared_state = {"payload": None, "ready": False}
        return _FakePipeEndpoint(shared_state), _FakePipeEndpoint(shared_state)

    def Process(self, target, args, daemon):
        behavior = self.behaviors.pop(0) if self.behaviors else {}
        process = _FakeProcess(target, args, daemon, behavior)
        self.processes.append(process)
        return process


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
            quality_result={"status": "passed", "passed": True, "weak": False, "retake_required": False, "media_quality": {"aggregate_quality": 0.9}, "confidence_multiplier": 0.9},
            validation_result={"passed": True, "critical_errors": [], "warnings": []},
            result={"confidence": 0.8, "risk_level": "stable", "retake_required": False},
            signals={"camera": {"details": {"status": "ok", "avg_ear": 0.22, "left_right_eye_asymmetry": 0.01, "eyes_closed": True}}, "voice": {"details": {"status": "ok", "rms_energy": 0.03}}},
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
            quality_result={"status": "passed", "passed": True, "weak": False, "retake_required": False, "media_quality": {"aggregate_quality": 0.9}, "confidence_multiplier": 0.9},
            validation_result={"passed": True, "critical_errors": [], "warnings": []},
            result={"confidence": 0.8, "risk_level": "stable", "retake_required": False},
            signals={"camera": {"details": {"status": "ok", "avg_ear": 0.16, "left_right_eye_asymmetry": 0.01}}, "voice": {"details": {"status": "ok", "rms_energy": 0.03}}},
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
                    "open_eye_aperture": {"median": 0.30, "mad": 0.01, "count": 6},
                    "left_right_eye_asymmetry": {"median": 0.01, "mad": 0.005, "count": 6},
                },
            },
            "voice_avg": {
                "schema_version": 2,
                "feature_stats": {
                    "normalized_voice_energy": {"median": 0.03, "mad": 0.002, "count": 6},
                    "speech_rate": {"median": 2.1, "mad": 0.2, "count": 6},
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
        quality_result = {"status": "passed", "passed": True, "weak": False, "retake_required": False, "media_quality": {"aggregate_quality": 0.9}, "confidence_multiplier": 0.9, "warnings": []}
        validation_result = {"passed": True, "critical_errors": [], "warnings": []}
        result = {"confidence": 0.8, "risk_level": "stable", "retake_required": False}
        active_baseline = {
            "scan_count": 4,
            "is_active": True,
            "face_avg": {"schema_version": 2, "feature_stats": {"open_eye_aperture": {"median": 0.18, "mad": 0.02, "count": 4}}},
            "voice_avg": {"schema_version": 2, "feature_stats": {"normalized_voice_energy": {"median": 0.014, "mad": 0.02, "count": 4}}},
            "reaction_avg": {"schema_version": 2, "feature_stats": {}},
        }
        inactive_baseline = dict(active_baseline, scan_count=2, is_active=False)
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
        # A malformed face reference does not cancel a valid supported voice
        # reference. Personalization requires at least one valid supported reference.
        self.assertTrue(
            baseline_ready_for_personalized_scoring(
                malformed_baseline,
                quality_result=quality_result,
                validation_result=validation_result,
                result=result,
                unique_row=True,
            )
        )

        no_valid_reference_baseline = {
            "scan_count": 4,
            "is_active": True,
            "face_avg": {
                "schema_version": 2,
                "feature_stats": {
                    "open_eye_aperture": {"median": None, "mad": 0.02, "count": 4}
                },
            },
            "voice_avg": {
                "schema_version": 2,
                "feature_stats": {
                    "normalized_voice_energy": {"median": None, "mad": 0.02, "count": 4}
                },
            },
            "reaction_avg": {"schema_version": 2, "feature_stats": {}},
        }
        self.assertFalse(
            baseline_ready_for_personalized_scoring(
                no_valid_reference_baseline,
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
            "camera": _signal(
                0.2,
                {
                    "status": "ok",
                    "image_quality_score": 0.2,
                    "image_warnings": ["image_blurry"],
                },
            ),
            "video": _signal(
                0.25,
                {
                    "status": "ok",
                    "duration_seconds": 5.0,
                    "visual_quality_score": 0.25,
                    "visual_warnings": ["video_blurry"],
                },
            ),
            "voice": _signal(
                0.22,
                {
                    "status": "ok",
                    "duration_seconds": 4.0,
                    "audio_quality_score": 0.22,
                    "audio_warnings": ["audio_too_noisy"],
                },
            ),
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
                    "motion_stability_score": 0.88,
                    "eye_closure_sample_count": 8,
                    "closed_eye_ratio": 0.75,
                    "longest_eye_closure_streak": 4,
                    "eye_closure_window_ms": 720.0,
                    "eye_closure_window_seconds": 0.72,
                    "avg_eye_aperture": 0.12,
                    "eye_aperture_std": 0.025,
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

    def test_video_and_audio_fatigue_signals_raise_fatigue_level(self):
        signals = {
            "camera": _signal(0.84, {"status": "ok", "image_confidence": 0.84, "image_quality_score": 0.84, "image_warnings": [], "face_detected": True}),
            "video": _signal(
                0.78,
                {
                    "status": "ok",
                    "visual_confidence": 0.78,
                    "visual_quality_score": 0.78,
                    "visual_warnings": [],
                    "reliable_eye_landmarks": True,
                    "sustained_eye_closure": False,
                    "closed_eye_ratio": 0.58,
                    "avg_eye_aperture": 0.16,
                    "longest_eye_closure_streak": 5,
                    "eye_closure_window_seconds": 1.0,
                    "motion_stability_score": 0.74,
                    "eye_closure_sample_count": 12,
                },
            ),
            "voice": _signal(
                0.72,
                {
                    "status": "ok",
                    "audio_confidence": 0.72,
                    "audio_quality_score": 0.71,
                    "audio_warnings": [],
                    "speech_presence_score": 0.46,
                    "rms_energy": 0.011,
                    "silence_ratio": 0.61,
                    "speech_rate": None,
                    "speech_state": "quiet_usable_speech",
                    "usable_speech_detected": True,
                    "quiet_but_usable": True,
                },
            ),
        }
        result = compute_result(signals=signals, quality=assess_quality(signals))

        self.assertIn(result["risk_level"], {"elevated_fatigue", "high_risk"})
        self.assertGreaterEqual(result["observed_fatigue_score"], 55)
        self.assertGreaterEqual(result["fatigue_evidence_score"], 0.45)
        self.assertIn("fatigue", result["explanation"].lower())

    def test_baseline_strengthens_fatigue_detection(self):
        signals = {
            "camera": _signal(0.84, {"status": "ok", "image_confidence": 0.84, "image_quality_score": 0.84, "image_warnings": [], "face_detected": True}),
            "video": _signal(
                0.78,
                {
                    "status": "ok",
                    "visual_confidence": 0.78,
                    "visual_quality_score": 0.78,
                    "visual_warnings": [],
                    "reliable_eye_landmarks": True,
                    "sustained_eye_closure": False,
                    "closed_eye_ratio": 0.52,
                    "avg_eye_aperture": 0.17,
                    "longest_eye_closure_streak": 4,
                    "eye_closure_window_seconds": 0.9,
                    "motion_stability_score": 0.76,
                    "eye_closure_sample_count": 10,
                },
            ),
            "voice": _signal(
                0.72,
                {
                    "status": "ok",
                    "audio_confidence": 0.72,
                    "audio_quality_score": 0.71,
                    "audio_warnings": [],
                    "speech_presence_score": 0.5,
                    "rms_energy": 0.012,
                    "silence_ratio": 0.58,
                    "speech_rate": None,
                    "speech_state": "quiet_usable_speech",
                    "usable_speech_detected": True,
                    "quiet_but_usable": True,
                },
            ),
        }
        baseline = {
            "scan_count": 4,
            "is_active": True,
            "face_avg": {
                "schema_version": 2,
                "feature_stats": {
                    "open_eye_aperture": {"median": 0.27, "mad": 0.02, "count": 4},
                    "left_right_eye_asymmetry": {"median": 0.02, "mad": 0.01, "count": 4},
                },
            },
            "voice_avg": {
                "schema_version": 2,
                "feature_stats": {
                    "normalized_voice_energy": {"median": 0.03, "mad": 0.01, "count": 4},
                    "speech_rate": {"median": 1.9, "mad": 0.1, "count": 4},
                },
            },
            "reaction_avg": {"schema_version": 2, "feature_stats": {}},
        }

        without_baseline = compute_result(signals=signals, quality=assess_quality(signals))
        with_baseline = compute_result(signals=signals, quality=assess_quality(signals), baseline=baseline, baseline_used=True)

        self.assertGreater(with_baseline["fatigue_evidence_score"], without_baseline["fatigue_evidence_score"])
        self.assertIn(with_baseline["risk_level"], {"elevated_fatigue", "high_risk"})
        self.assertNotEqual(with_baseline["risk_level"], "stable")

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
                    "speech_state": "quiet_usable_speech",
                    "usable_speech_detected": True,
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

    def test_task_score_requires_exact_positive_integer_attempts(self):
        invalid_attempts = [None, 0, -1, 3.0, "3", True]
        for attempts in invalid_attempts:
            with self.subTest(attempts=attempts):
                self.assertIsNone(
                    scoring.compute_task_score(
                        {
                            "reaction_time": 0.5,
                            "errors": 0,
                            "attempts": attempts,
                        }
                    )
                )

        self.assertIsNotNone(
            scoring.compute_task_score(
                {"reaction_time": 0.5, "errors": 0, "attempts": 3}
            )
        )

    def test_bool_only_eye_closure_is_not_confirmed(self):
        signals = {
            "video": _signal(
                0.8,
                {
                    "status": "ok",
                    "reliable_eye_landmarks": True,
                    "sustained_eye_closure": True,
                },
            )
        }
        self.assertFalse(scoring._confirmed_sustained_eye_closure(signals))

    def test_complete_consistent_eye_closure_is_confirmed(self):
        signals = {
            "video": _signal(
                0.8,
                {
                    "status": "ok",
                    "reliable_eye_landmarks": True,
                    "sustained_eye_closure": True,
                    "motion_stability_score": 0.9,
                    "eye_closure_sample_count": 8,
                    "closed_eye_ratio": 0.75,
                    "longest_eye_closure_streak": 4,
                    "eye_closure_window_ms": 720.0,
                    "eye_closure_window_seconds": 0.72,
                    "avg_eye_aperture": 0.12,
                    "eye_aperture_std": 0.02,
                },
            )
        }
        self.assertTrue(scoring._confirmed_sustained_eye_closure(signals))

    def test_audio_usable_speech_requires_consistent_state_and_flags(self):
        self.assertTrue(
            scoring._audio_usable_speech(
                {
                    "speech_state": "usable_speech",
                    "usable_speech_detected": True,
                    "quiet_but_usable": False,
                },
                [],
            )
        )
        self.assertTrue(
            scoring._audio_usable_speech(
                {
                    "speech_state": "quiet_usable_speech",
                    "usable_speech_detected": True,
                    "quiet_but_usable": True,
                },
                [],
            )
        )

        rejected = [
            ({"speech_state": "usable_speech"}, []),
            ({"usable_speech_detected": True}, []),
            (
                {
                    "speech_state": "quiet_usable_speech",
                    "usable_speech_detected": True,
                    "quiet_but_usable": False,
                },
                [],
            ),
            (
                {
                    "speech_state": "usable_speech",
                    "usable_speech_detected": True,
                    "quiet_but_usable": False,
                },
                ["audio_too_noisy"],
            ),
        ]
        for details, warnings in rejected:
            with self.subTest(details=details, warnings=warnings):
                self.assertFalse(
                    scoring._audio_usable_speech(details, warnings)
                )

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

    def test_directus_400_logs_sanitized_summary_without_response_body(self):
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
        self.assertIn("response_summary={'status_code': 400", combined)
        self.assertIn("/items/scan_results", combined)
        self.assertIn("payload_keys=['confidence', 'risk_level', 'scan_id']", combined)
        self.assertNotIn("Invalid payload", combined)
        self.assertNotIn("one-to-many", combined)


    def test_directus_value_too_long_logs_only_sanitized_error_code(self):
        client = DirectusClient(base_url="http://example.com", token="x")
        response = requests.Response()
        response.status_code = 400
        response.url = "http://example.com/items/scan_results"
        response._content = json.dumps(
            {
                "errors": [
                    {
                        "message": (
                            "Value 'Conntinuity Intelligence Engine v1.2' for field "
                            "'ai_model_version' in collection 'scan_results' is too long."
                        ),
                        "extensions": {
                            "collection": "scan_results",
                            "field": "ai_model_version",
                            "value": "Conntinuity Intelligence Engine v1.2",
                            "code": "VALUE_TOO_LONG",
                        },
                    }
                ]
            }
        ).encode("utf-8")

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
        self.assertIn("'code': 'VALUE_TOO_LONG'", combined)
        self.assertIn("ai_model_version", combined)
        self.assertNotIn("for field", combined)
        self.assertNotIn("is too long", combined)

    def test_run_parallel_analysis_returns_real_results_before_deadline(self):
        fake_context = _FakeContext()
        sentinels = []

        def fake_safe_analyze(fn, path, missing_warning):
            return {"score": 0.9, "details": {"status": "ok", "analyzer": fn.__name__}}

        with unittest.mock.patch.object(main.multiprocessing, "get_context", return_value=fake_context):
            with unittest.mock.patch.object(main, "_safe_analyze", side_effect=fake_safe_analyze):
                with unittest.mock.patch.object(main, "multiprocessing_wait", side_effect=lambda handles, timeout=None: set(handles) if not sentinels else set()):
                    results, worker_states = main._run_parallel_analysis(
                        "scan-1",
                        main.Media(image="image.jpg", audio="audio.wav", video="video.mp4"),
                    )

        self.assertEqual(results["video"]["details"]["analyzer"], "analyze_video")
        self.assertEqual(results["audio"]["details"]["analyzer"], "analyze_audio")
        self.assertEqual(results["image"]["details"]["analyzer"], "analyze_face")
        self.assertTrue(all(not state["timed_out"] for state in worker_states.values()))
        self.assertTrue(all(not state["alive"] for state in worker_states.values()))
        self.assertTrue(all(not state["terminated"] for state in worker_states.values()))
        self.assertTrue(all(not state["killed"] for state in worker_states.values()))

    def test_finalize_analysis_worker_reads_completed_pipe_payload(self):
        shared_state = {
            "payload": {
                "ok": True,
                "result": {"score": 0.94, "details": {"status": "ok"}},
                "metrics": {"child_boot_ms": 11, "analyzer_execution_ms": 7},
            },
            "ready": True,
        }
        conn = _FakePipeEndpoint(shared_state)
        process = _FakeProcess(lambda *args: None, (), False, {"invoke_target": False, "alive_after_start": False})

        result, state = main._finalize_analysis_worker(
            scan_id="scan-1",
            analyzer_name="video",
            process=process,
            result_conn=conn,
            started_at=time.perf_counter() - 0.01,
            parent_process_start_ms=3,
            timeout_seconds=main.MEDIA_VALIDATION_WALL_TIMEOUT_SECONDS,
            timed_out=False,
        )

        self.assertEqual(result["score"], 0.94)
        self.assertEqual(state["child_boot_ms"], 11)
        self.assertFalse(state["timed_out"])
        self.assertFalse(state["alive"])

    def test_finalize_analysis_process_kills_still_alive_worker(self):
        process = _FakeProcess(lambda *args: None, (), False, {"invoke_target": False, "alive_after_start": True, "requires_kill": True})

        state = main._finalize_analysis_process(process)

        self.assertTrue(state["terminated"])
        self.assertTrue(state["killed"])
        self.assertFalse(state["alive"])
        self.assertEqual(process.join_calls, [1.0, 1.0])

    def test_finalize_analysis_worker_terminates_timed_out_worker(self):
        conn = _FakePipeEndpoint({"payload": None, "ready": False})
        process = _FakeProcess(lambda *args: None, (), False, {"invoke_target": False, "alive_after_start": True})

        result, state = main._finalize_analysis_worker(
            scan_id="scan-1",
            analyzer_name="image",
            process=process,
            result_conn=conn,
            started_at=time.perf_counter() - 0.01,
            parent_process_start_ms=4,
            timeout_seconds=main.MEDIA_VALIDATION_WALL_TIMEOUT_SECONDS,
            timed_out=True,
        )

        self.assertTrue(state["terminated"])
        self.assertFalse(state["alive"])
        self.assertFalse(state["killed"])
        self.assertIn("image_timeout", result["details"]["image_warnings"])

    def test_timeout_placeholder_is_not_valid_fatigue_evidence(self):
        placeholder = main._analysis_timeout_placeholder("video")

        self.assertFalse(main._result_has_valid_evidence(placeholder))
        self.assertIn("video_timeout", placeholder["details"]["visual_warnings"])

    def test_partial_modality_timeout_preserves_success_path(self):
        with ExitStack() as stack:
            stack.enter_context(
                unittest.mock.patch.object(
                    main,
                    "_resolve_scan_context",
                    return_value={
                        "status": main.SCAN_STATUS_MEDIA_READY,
                        "scan_media": {"id": "media-1"},
                        "resolved_media": {},
                        "task_metrics": None,
                        "expected_phrase": None,
                        "user": None,
                        "member": None,
                        "business_profile": None,
                        "department": None,
                    },
                )
            )
            stack.enter_context(unittest.mock.patch.object(main, "_merge_media", return_value=main.Media(image="image.jpg", audio="audio.wav", video="video.mp4")))
            stack.enter_context(unittest.mock.patch.object(main, "_merge_task", return_value=None))
            stack.enter_context(unittest.mock.patch.object(main, "_identifier_payload", return_value={"user_id": None, "member_id": None, "business_profile_id": None, "department_id": None}))
            stack.enter_context(unittest.mock.patch.object(main, "_baseline_rows_for_member", return_value=[]))
            stack.enter_context(unittest.mock.patch.object(main, "baseline_status_payload", return_value={"baseline_status": "inactive", "baseline_confidence": None}))
            stack.enter_context(unittest.mock.patch.object(main, "_expected_phrase", return_value=None))
            stack.enter_context(unittest.mock.patch.object(main.ml_runtime, "local_model_required", return_value=False))
            stack.enter_context(unittest.mock.patch.object(main, "_resolve_media_input", side_effect=lambda path, *args, **kwargs: (path, False)))
            stack.enter_context(unittest.mock.patch.object(main, "_should_convert_audio", return_value=False))
            stack.enter_context(
                unittest.mock.patch.object(
                    main,
                    "_run_parallel_analysis",
                    return_value=(
                        {
                            "video": {"score": 0.8, "details": {"status": "ok", "visual_quality_score": 0.8, "visual_warnings": []}},
                            "audio": {"score": 0.8, "details": {"status": "ok", "audio_quality_score": 0.8, "audio_warnings": [], "timings_ms": {}}},
                            "image": {"score": 0.8, "details": {"status": "ok", "image_quality_score": 0.8, "image_warnings": []}},
                        },
                        {
                            "video": {"timed_out": False, "alive": False, "terminated": False, "killed": False, "process_exitcode": 0},
                            "audio": {"timed_out": True, "alive": False, "terminated": True, "killed": False, "process_exitcode": -15},
                            "image": {"timed_out": False, "alive": False, "terminated": False, "killed": False, "process_exitcode": 0},
                        },
                    ),
                )
            )
            stack.enter_context(
                unittest.mock.patch.object(
                    main,
                    "validate_scan_inputs",
                    return_value={
                        "quality_scores": {"phrase_match": 0.9, "audio": 0.8, "video": 0.8, "image": 0.8},
                        "warnings": [],
                        "critical_errors": [],
                        "failure_reason": None,
                    },
                )
            )
            stack.enter_context(
                unittest.mock.patch.object(
                    main,
                    "assess_quality",
                    return_value={
                        "warnings": [],
                        "media_quality": {
                            "video": {"usable": True, "present": True},
                            "audio": {"usable": False, "present": False},
                            "image": {"usable": True, "present": True},
                        },
                        "usable_modalities": 2,
                        "failure_reason": None,
                        "status": "passed",
                        "weak": False,
                        "retake_required": False,
                    },
                )
            )
            stack.enter_context(unittest.mock.patch.object(main, "_required_modality_gate", return_value=(None, ["video", "audio", "image"])))
            stack.enter_context(unittest.mock.patch.object(main, "_face_eye_evidence_unreliable", return_value=False))
            stack.enter_context(unittest.mock.patch.object(main, "_result_has_valid_evidence", return_value=True))
            stack.enter_context(unittest.mock.patch.object(main, "features_from_signals", return_value=({}, {})))
            stack.enter_context(unittest.mock.patch.object(main, "vector_from_features", return_value=[0.0] * 21))
            stack.enter_context(unittest.mock.patch.object(main.ml_runtime, "predict", return_value={"score": 0.5}))
            stack.enter_context(
                unittest.mock.patch.object(
                    main,
                    "compute_result",
                    return_value={
                        "readiness_score": 80,
                        "confidence": 0.8,
                        "risk_level": "stable",
                        "suggested_action": "continue_normal_activity",
                        "explanation": "ok",
                        "retake_required": False,
                        "baseline_used": False,
                        "fusion_details": {"baseline_flags": {}},
                        "modality_scores": {},
                        "face_metrics": {"baseline_drifts": {}},
                        "voice_metrics": {"baseline_drifts": {}},
                        "reaction_metrics": {"baseline_drifts": {}},
                    },
                )
            )
            stack.enter_context(unittest.mock.patch.object(main, "baseline_ready_for_personalized_scoring", return_value=False))
            stack.enter_context(
                unittest.mock.patch.object(
                    main,
                    "evaluate_baseline_eligibility",
                    return_value={
                        "eligible": False,
                        "capture_quality_score": 0.0,
                        "measurement_reliability_score": 0.0,
                        "task_completion_status": "not_required",
                        "hard_gates_triggered": [],
                        "reasons": [],
                    },
                )
            )
            write_success = MagicMock(return_value={"scan_result": "updated:scan-1", "wellness_scan": "updated"})
            stack.enter_context(unittest.mock.patch.object(main, "_write_success", write_success))
            stack.enter_context(unittest.mock.patch.object(main, "_log_step", side_effect=lambda *args, **kwargs: None))
            stack.enter_context(unittest.mock.patch.object(main, "_log_perf", side_effect=lambda *args, **kwargs: None))
            stack.enter_context(unittest.mock.patch.object(main, "_log_validation_decision", side_effect=lambda *args, **kwargs: None))
            stack.enter_context(unittest.mock.patch.object(main, "_log_validation_lifecycle", side_effect=lambda *args, **kwargs: None))

            result = main._process_scan_sync("scan-1")

        self.assertEqual(result["status"], main.SCAN_STATUS_COMPLETED)
        write_success.assert_called_once()

    def test_process_scan_sync_timeout_path_finalizes_failed_scan(self):
        with ExitStack() as stack:
            stack.enter_context(
                unittest.mock.patch.object(
                    main,
                    "_resolve_scan_context",
                    return_value={
                        "status": main.SCAN_STATUS_MEDIA_READY,
                        "scan_media": {"id": "media-1"},
                        "resolved_media": {},
                        "task_metrics": None,
                        "expected_phrase": None,
                        "user": None,
                        "member": None,
                        "business_profile": None,
                        "department": None,
                    },
                )
            )
            stack.enter_context(unittest.mock.patch.object(main, "_merge_media", return_value=main.Media(image="image.jpg", audio="audio.wav", video="video.mp4")))
            stack.enter_context(unittest.mock.patch.object(main, "_merge_task", return_value=None))
            stack.enter_context(unittest.mock.patch.object(main, "_identifier_payload", return_value={"user_id": None, "member_id": None, "business_profile_id": None, "department_id": None}))
            stack.enter_context(unittest.mock.patch.object(main, "_baseline_rows_for_member", return_value=[]))
            stack.enter_context(unittest.mock.patch.object(main, "baseline_status_payload", return_value={"baseline_status": "inactive", "baseline_confidence": None}))
            stack.enter_context(unittest.mock.patch.object(main, "_expected_phrase", return_value=None))
            stack.enter_context(unittest.mock.patch.object(main.ml_runtime, "local_model_required", return_value=False))
            stack.enter_context(unittest.mock.patch.object(main, "_resolve_media_input", side_effect=lambda path, *args, **kwargs: (path, False)))
            stack.enter_context(unittest.mock.patch.object(main, "_should_convert_audio", return_value=False))
            stack.enter_context(
                unittest.mock.patch.object(
                    main,
                    "_run_parallel_analysis",
                    return_value=(
                        {
                            "video": main._analysis_timeout_placeholder("video"),
                            "audio": main._analysis_timeout_placeholder("audio"),
                            "image": main._analysis_timeout_placeholder("image"),
                        },
                        {
                            "video": {"timed_out": True, "alive": False, "terminated": True, "killed": False, "process_exitcode": -15},
                            "audio": {"timed_out": True, "alive": False, "terminated": True, "killed": False, "process_exitcode": -15},
                            "image": {"timed_out": True, "alive": False, "terminated": True, "killed": False, "process_exitcode": -15},
                        },
                    ),
                )
            )
            stack.enter_context(
                unittest.mock.patch.object(
                    main,
                    "validate_scan_inputs",
                    return_value={
                        "quality_scores": {"phrase_match": None, "audio": 0.0, "video": 0.0, "image": 0.0},
                        "warnings": [],
                        "critical_errors": [],
                        "failure_reason": None,
                    },
                )
            )
            stack.enter_context(
                unittest.mock.patch.object(
                    main,
                    "assess_quality",
                    return_value={
                        "warnings": ["audio_timeout"],
                        "media_quality": {
                            "video": {"usable": True, "present": True},
                            "audio": {"usable": False, "present": False, "warnings": ["audio_timeout"]},
                            "image": {"usable": True, "present": True},
                        },
                        "usable_modalities": 2,
                        "failure_reason": None,
                        "status": "passed",
                        "weak": False,
                        "retake_required": False,
                    },
                )
            )
            stack.enter_context(unittest.mock.patch.object(main, "_required_modality_gate", return_value=(main.FAILURE_REASON_LOW_QUALITY_MEDIA, ["video", "audio", "image"])))
            mark_failed_terminal = stack.enter_context(unittest.mock.patch.object(main, "_mark_scan_failed_terminal", return_value={"wellness_scan": "failed_updated"}))
            stack.enter_context(unittest.mock.patch.object(main, "_log_step", side_effect=lambda *args, **kwargs: None))
            stack.enter_context(unittest.mock.patch.object(main, "_log_perf", side_effect=lambda *args, **kwargs: None))
            stack.enter_context(unittest.mock.patch.object(main, "_log_validation_decision", side_effect=lambda *args, **kwargs: None))
            stack.enter_context(unittest.mock.patch.object(main, "_log_validation_lifecycle", side_effect=lambda *args, **kwargs: None))
            stack.enter_context(unittest.mock.patch.object(main, "_write_success", side_effect=AssertionError("write_success should not run on timeout failure")))

            result = main._process_scan_sync("scan-1")

        self.assertEqual(result["status"], main.SCAN_STATUS_FAILED)
        self.assertNotEqual(result["status"], main.SCAN_STATUS_PROCESSING)
        self.assertEqual(result["failure_reason"], "validation_timeout")
        mark_failed_terminal.assert_called_once()

    def test_process_scan_sync_success_path_returns_completed(self):
        with ExitStack() as stack:
            stack.enter_context(
                unittest.mock.patch.object(
                    main,
                    "_resolve_scan_context",
                    return_value={
                        "status": main.SCAN_STATUS_MEDIA_READY,
                        "scan_media": {"id": "media-1"},
                        "resolved_media": {},
                        "task_metrics": None,
                        "expected_phrase": None,
                        "user": None,
                        "member": None,
                        "business_profile": None,
                        "department": None,
                    },
                )
            )
            stack.enter_context(unittest.mock.patch.object(main, "_merge_media", return_value=main.Media(image="image.jpg", audio="audio.wav", video="video.mp4")))
            stack.enter_context(unittest.mock.patch.object(main, "_merge_task", return_value=None))
            stack.enter_context(unittest.mock.patch.object(main, "_identifier_payload", return_value={"user_id": None, "member_id": None, "business_profile_id": None, "department_id": None}))
            stack.enter_context(unittest.mock.patch.object(main, "_baseline_rows_for_member", return_value=[]))
            stack.enter_context(unittest.mock.patch.object(main, "baseline_status_payload", return_value={"baseline_status": "inactive", "baseline_confidence": None}))
            stack.enter_context(unittest.mock.patch.object(main, "_expected_phrase", return_value=None))
            stack.enter_context(unittest.mock.patch.object(main.ml_runtime, "local_model_required", return_value=False))
            stack.enter_context(unittest.mock.patch.object(main, "_resolve_media_input", side_effect=lambda path, *args, **kwargs: (path, False)))
            stack.enter_context(unittest.mock.patch.object(main, "_should_convert_audio", return_value=False))
            stack.enter_context(
                unittest.mock.patch.object(
                    main,
                    "_run_parallel_analysis",
                    return_value=(
                        {
                            "video": {"score": 0.8, "details": {"status": "ok", "visual_quality_score": 0.8, "visual_warnings": []}},
                            "audio": {"score": 0.8, "details": {"status": "ok", "audio_quality_score": 0.8, "audio_warnings": [], "timings_ms": {}}},
                            "image": {"score": 0.8, "details": {"status": "ok", "image_quality_score": 0.8, "image_warnings": []}},
                        },
                        {
                            "video": {"timed_out": False, "alive": False, "terminated": False, "killed": False, "process_exitcode": 0},
                            "audio": {"timed_out": False, "alive": False, "terminated": False, "killed": False, "process_exitcode": 0},
                            "image": {"timed_out": False, "alive": False, "terminated": False, "killed": False, "process_exitcode": 0},
                        },
                    ),
                )
            )
            stack.enter_context(
                unittest.mock.patch.object(
                    main,
                    "validate_scan_inputs",
                    return_value={
                        "quality_scores": {"phrase_match": 0.9, "audio": 0.8, "video": 0.8, "image": 0.8},
                        "warnings": [],
                        "critical_errors": [],
                        "failure_reason": None,
                    },
                )
            )
            stack.enter_context(
                unittest.mock.patch.object(
                    main,
                    "assess_quality",
                    return_value={
                        "warnings": [],
                        "media_quality": {
                            "video": {"usable": True, "present": True},
                            "audio": {"usable": True, "present": True},
                            "image": {"usable": True, "present": True},
                        },
                        "usable_modalities": 3,
                        "failure_reason": None,
                        "status": "passed",
                        "weak": False,
                        "retake_required": False,
                    },
                )
            )
            stack.enter_context(unittest.mock.patch.object(main, "_required_modality_gate", return_value=(None, ["video", "audio", "image"])))
            stack.enter_context(unittest.mock.patch.object(main, "_face_eye_evidence_unreliable", return_value=False))
            stack.enter_context(unittest.mock.patch.object(main, "_result_has_valid_evidence", return_value=True))
            stack.enter_context(unittest.mock.patch.object(main, "features_from_signals", return_value=({}, {})))
            stack.enter_context(unittest.mock.patch.object(main, "vector_from_features", return_value=[0.0] * 21))
            stack.enter_context(unittest.mock.patch.object(main.ml_runtime, "predict", return_value={"score": 0.5}))
            stack.enter_context(
                unittest.mock.patch.object(
                    main,
                    "compute_result",
                    return_value={
                        "readiness_score": 80,
                        "confidence": 0.8,
                        "risk_level": "stable",
                        "suggested_action": "continue_normal_activity",
                        "explanation": "ok",
                        "retake_required": False,
                        "baseline_used": False,
                        "fusion_details": {"baseline_flags": {}},
                        "modality_scores": {},
                        "face_metrics": {"baseline_drifts": {}},
                        "voice_metrics": {"baseline_drifts": {}},
                        "reaction_metrics": {"baseline_drifts": {}},
                    },
                )
            )
            stack.enter_context(
                unittest.mock.patch.object(
                    main,
                    "baseline_ready_for_personalized_scoring",
                    return_value=False,
                )
            )
            stack.enter_context(
                unittest.mock.patch.object(
                    main,
                    "evaluate_baseline_eligibility",
                    return_value={
                        "eligible": False,
                        "capture_quality_score": 0.0,
                        "measurement_reliability_score": 0.0,
                        "task_completion_status": "not_required",
                        "hard_gates_triggered": [],
                        "reasons": [],
                    },
                )
            )
            write_success = MagicMock(return_value={"scan_result": "updated:scan-1", "wellness_scan": "updated"})
            stack.enter_context(unittest.mock.patch.object(main, "_write_success", write_success))
            stack.enter_context(unittest.mock.patch.object(main, "_log_step", side_effect=lambda *args, **kwargs: None))
            stack.enter_context(unittest.mock.patch.object(main, "_log_perf", side_effect=lambda *args, **kwargs: None))
            stack.enter_context(unittest.mock.patch.object(main, "_log_validation_decision", side_effect=lambda *args, **kwargs: None))
            stack.enter_context(unittest.mock.patch.object(main, "_log_validation_lifecycle", side_effect=lambda *args, **kwargs: None))
            upsert_baseline = stack.enter_context(unittest.mock.patch.object(main.directus, "upsert_employee_baseline", MagicMock()))

            result = main._process_scan_sync("scan-1")

        self.assertEqual(result["status"], main.SCAN_STATUS_COMPLETED)
        self.assertNotEqual(result["status"], main.SCAN_STATUS_PROCESSING)
        write_success.assert_called_once()
        upsert_baseline.assert_not_called()

    def test_process_scan_sync_writeback_failure_without_recovery_attempts_terminal_failed_update(self):
        with ExitStack() as stack:
            stack.enter_context(
                unittest.mock.patch.object(
                    main,
                    "_resolve_scan_context",
                    return_value={
                        "status": main.SCAN_STATUS_MEDIA_READY,
                        "scan_media": {"id": "media-1"},
                        "resolved_media": {},
                        "task_metrics": None,
                        "expected_phrase": None,
                        "user": None,
                        "member": None,
                        "business_profile": None,
                        "department": None,
                    },
                )
            )
            stack.enter_context(unittest.mock.patch.object(main, "_merge_media", return_value=main.Media(image="image.jpg", audio="audio.wav", video="video.mp4")))
            stack.enter_context(unittest.mock.patch.object(main, "_merge_task", return_value=None))
            stack.enter_context(unittest.mock.patch.object(main, "_identifier_payload", return_value={"user_id": None, "member_id": None, "business_profile_id": None, "department_id": None}))
            stack.enter_context(unittest.mock.patch.object(main, "_baseline_rows_for_member", return_value=[]))
            stack.enter_context(unittest.mock.patch.object(main, "baseline_status_payload", return_value={"baseline_status": "inactive", "baseline_confidence": None}))
            stack.enter_context(unittest.mock.patch.object(main, "_expected_phrase", return_value=None))
            stack.enter_context(unittest.mock.patch.object(main.ml_runtime, "local_model_required", return_value=False))
            stack.enter_context(unittest.mock.patch.object(main, "_resolve_media_input", side_effect=lambda path, *args, **kwargs: (path, False)))
            stack.enter_context(unittest.mock.patch.object(main, "_should_convert_audio", return_value=False))
            stack.enter_context(
                unittest.mock.patch.object(
                    main,
                    "_run_parallel_analysis",
                    return_value=(
                        {
                            "video": {"score": 0.8, "details": {"status": "ok", "visual_quality_score": 0.8, "visual_warnings": []}},
                            "audio": {"score": 0.8, "details": {"status": "ok", "audio_quality_score": 0.8, "audio_warnings": [], "timings_ms": {}}},
                            "image": {"score": 0.8, "details": {"status": "ok", "image_quality_score": 0.8, "image_warnings": []}},
                        },
                        {
                            "video": {"timed_out": False, "alive": False, "terminated": False, "killed": False, "process_exitcode": 0},
                            "audio": {"timed_out": False, "alive": False, "terminated": False, "killed": False, "process_exitcode": 0},
                            "image": {"timed_out": False, "alive": False, "terminated": False, "killed": False, "process_exitcode": 0},
                        },
                    ),
                )
            )
            stack.enter_context(
                unittest.mock.patch.object(
                    main,
                    "validate_scan_inputs",
                    return_value={
                        "quality_scores": {"phrase_match": 0.9, "audio": 0.8, "video": 0.8, "image": 0.8},
                        "warnings": [],
                        "critical_errors": [],
                        "failure_reason": None,
                    },
                )
            )
            stack.enter_context(
                unittest.mock.patch.object(
                    main,
                    "assess_quality",
                    return_value={
                        "warnings": [],
                        "media_quality": {
                            "video": {"usable": True, "present": True},
                            "audio": {"usable": True, "present": True},
                            "image": {"usable": True, "present": True},
                        },
                        "usable_modalities": 3,
                        "failure_reason": None,
                        "status": "passed",
                        "weak": False,
                        "retake_required": False,
                    },
                )
            )
            stack.enter_context(unittest.mock.patch.object(main, "_required_modality_gate", return_value=(None, ["video", "audio", "image"])))
            stack.enter_context(unittest.mock.patch.object(main, "_face_eye_evidence_unreliable", return_value=False))
            stack.enter_context(unittest.mock.patch.object(main, "_result_has_valid_evidence", return_value=True))
            stack.enter_context(unittest.mock.patch.object(main, "features_from_signals", return_value=({}, {})))
            stack.enter_context(unittest.mock.patch.object(main, "vector_from_features", return_value=[0.0] * 21))
            stack.enter_context(unittest.mock.patch.object(main.ml_runtime, "predict", return_value={"score": 0.5}))
            stack.enter_context(
                unittest.mock.patch.object(
                    main,
                    "compute_result",
                    return_value={
                        "readiness_score": 80,
                        "confidence": 0.8,
                        "risk_level": "stable",
                        "suggested_action": "continue_normal_activity",
                        "explanation": "ok",
                        "retake_required": False,
                        "baseline_used": False,
                        "fusion_details": {"baseline_flags": {}},
                        "modality_scores": {},
                        "face_metrics": {"baseline_drifts": {}},
                        "voice_metrics": {"baseline_drifts": {}},
                        "reaction_metrics": {"baseline_drifts": {}},
                    },
                )
            )
            stack.enter_context(
                unittest.mock.patch.object(
                    main,
                    "baseline_ready_for_personalized_scoring",
                    return_value=False,
                )
            )
            stack.enter_context(
                unittest.mock.patch.object(
                    main,
                    "evaluate_baseline_eligibility",
                    return_value={
                        "eligible": False,
                        "capture_quality_score": 0.0,
                        "measurement_reliability_score": 0.0,
                        "task_completion_status": "not_required",
                        "hard_gates_triggered": [],
                        "reasons": [],
                    },
                )
            )
            write_failure = main.ProcessingError(main.FAILURE_REASON_WRITEBACK_FAILED, "RuntimeError")
            write_success = MagicMock(side_effect=write_failure)
            stack.enter_context(unittest.mock.patch.object(main, "_write_success", write_success))
            mark_failed_terminal = MagicMock(return_value={"wellness_scan": "failed_updated"})
            stack.enter_context(unittest.mock.patch.object(main, "_mark_scan_failed_terminal", mark_failed_terminal))
            stack.enter_context(unittest.mock.patch.object(main, "_log_step", side_effect=lambda *args, **kwargs: None))
            stack.enter_context(unittest.mock.patch.object(main, "_log_perf", side_effect=lambda *args, **kwargs: None))
            stack.enter_context(unittest.mock.patch.object(main, "_log_validation_decision", side_effect=lambda *args, **kwargs: None))
            stack.enter_context(unittest.mock.patch.object(main, "_log_validation_lifecycle", side_effect=lambda *args, **kwargs: None))
            upsert_baseline = stack.enter_context(unittest.mock.patch.object(main.directus, "upsert_employee_baseline", MagicMock()))

            result = main._process_scan_sync("scan-1")

        self.assertEqual(result["status"], main.SCAN_STATUS_FAILED)
        self.assertNotEqual(result["status"], main.SCAN_STATUS_PROCESSING)
        self.assertEqual(result["writeback_status"]["wellness_scan"], "failed_updated")
        self.assertEqual(result["writeback_status"]["recovery"], "unavailable")
        write_success.assert_called_once()
        mark_failed_terminal.assert_called_once()
        upsert_baseline.assert_not_called()

    def test_recover_completed_scan_if_result_exists_finalizes_completed_without_duplicate_result_write(self):
        main.directus.get_scan_result_by_scan_id = MagicMock(return_value={"id": "scan-result-1"})
        main.directus.update_wellness_scan = MagicMock(return_value={"id": "scan-1"})
        main.directus.upsert_scan_result = MagicMock()

        recovery = main._recover_completed_scan_if_result_exists("scan-1")

        self.assertTrue(recovery)
        main.directus.get_scan_result_by_scan_id.assert_called_once_with("scan-1")
        main.directus.update_wellness_scan.assert_called_once()
        main.directus.upsert_scan_result.assert_not_called()

    def test_mark_scan_failed_terminal_logs_unconfirmed_failure(self):
        with unittest.mock.patch.object(main, "_mark_scan_failed", return_value={"wellness_scan": "failed:RuntimeError"}):
            with self.assertLogs("ai-server", level="ERROR") as logs:
                status = main._mark_scan_failed_terminal("scan-1", "writeback_failed", "sanitized message")

        self.assertEqual(status["wellness_scan"], "failed:RuntimeError")
        self.assertIn("terminal_failure_writeback_unconfirmed", "\n".join(logs.output))

    def test_mark_scan_failed_sanitizes_exception_details(self):
        with unittest.mock.patch.object(main.directus, "update_wellness_scan", side_effect=RuntimeError("token=secret")):
            with self.assertLogs("ai-server", level="ERROR") as logs:
                status = main._mark_scan_failed("scan-1", "writeback_failed", "sanitized message")

        self.assertEqual(status["wellness_scan"], "failed:RuntimeError")
        combined = "\n".join(logs.output)
        self.assertIn("error_type=RuntimeError", combined)
        self.assertNotIn("token=secret", combined)



if __name__ == "__main__":
    unittest.main()
