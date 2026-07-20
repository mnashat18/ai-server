from __future__ import annotations

import math
import os
import shutil
import tempfile
import threading
import wave
from contextlib import suppress
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import MagicMock, patch

import numpy as np

import audio


def _float_to_pcm16(samples: np.ndarray) -> bytes:
    clipped = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def _write_wav(path: str, samples: np.ndarray, *, sample_rate: int = 16000, channels: int = 1, sample_width: int = 2) -> None:
    if channels > 1:
        samples = np.asarray(samples, dtype=np.float32).reshape(-1, channels)
        pcm = _float_to_pcm16(samples.reshape(-1))
    else:
        pcm = _float_to_pcm16(np.asarray(samples, dtype=np.float32))
    with wave.open(path, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(sample_width)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)


def _sine(duration_sec: float = 3.0, *, sample_rate: int = 16000, frequency: float = 220.0, amplitude: float = 0.2) -> np.ndarray:
    t = np.arange(int(round(duration_sec * sample_rate)), dtype=np.float32) / float(sample_rate)
    return (amplitude * np.sin(2.0 * np.pi * frequency * t)).astype(np.float32)


def _speech_like(duration_sec: float = 3.0, *, sample_rate: int = 16000, amplitude: float = 0.015) -> np.ndarray:
    t = np.arange(int(round(duration_sec * sample_rate)), dtype=np.float32) / float(sample_rate)
    carrier = np.sin(2.0 * np.pi * 180.0 * t)
    mod = 0.55 + 0.45 * np.sin(2.0 * np.pi * 2.5 * t)
    return (amplitude * carrier * mod).astype(np.float32)


def _noise(duration_sec: float = 3.0, *, sample_rate: int = 16000, amplitude: float = 0.4, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.uniform(-amplitude, amplitude, int(round(duration_sec * sample_rate)))).astype(np.float32)


def _frame_count_for(samples: np.ndarray, hop_length: int, n_fft: int) -> int:
    sample_count = int(np.asarray(samples).reshape(-1).size)
    if sample_count <= 0:
        return 1
    effective = max(sample_count, n_fft)
    return max(1, 1 + max(0, effective - n_fft) // max(1, hop_length))


def _signal_profile(samples: np.ndarray, sample_rate: int) -> tuple[float, float]:
    y = np.asarray(samples, dtype=np.float32).reshape(-1)
    if y.size == 0:
        return 0.0, 0.0
    window = np.hanning(y.size).astype(np.float32, copy=False) if y.size > 1 else np.ones(1, dtype=np.float32)
    spectrum = np.abs(np.fft.rfft(y * window))
    if spectrum.size == 0:
        return 0.0, 0.0
    total = float(np.sum(spectrum))
    centroid = 0.0
    if total > 0.0:
        freqs = np.fft.rfftfreq(y.size, d=1.0 / float(sample_rate)) if y.size > 1 else np.asarray([0.0], dtype=np.float64)
        centroid = float(np.sum(freqs * spectrum) / total)
    positive = spectrum[spectrum > 0.0]
    flatness = 0.0
    if positive.size:
        arithmetic = float(np.mean(positive))
        geometric = float(np.exp(np.mean(np.log(positive)))) if arithmetic > 0.0 else 0.0
        if arithmetic > 0.0 and math.isfinite(geometric):
            flatness = float(np.clip(geometric / arithmetic, 0.0, 1.0))
    return centroid, flatness


def _build_fake_librosa(samples: np.ndarray, sample_rate: int, *, mfcc_matrix: np.ndarray | None = None, centroid: float | None = None, flatness: float | None = None):
    def _stft(y, n_fft, hop_length, win_length=None, window="hann", center=True, pad_mode="constant"):
        prepared = np.asarray(y, dtype=np.float32).reshape(-1)
        win_length = win_length or n_fft
        pad = n_fft // 2 if center else 0
        padded = np.pad(prepared, (pad, pad), mode=pad_mode)
        if padded.size < n_fft:
            padded = np.pad(padded, (0, n_fft - padded.size), mode=pad_mode)
        frames = np.ascontiguousarray(np.lib.stride_tricks.sliding_window_view(padded, n_fft)[::hop_length])
        if frames.size == 0:
            frames = padded[-n_fft:][np.newaxis, :]
        window_values = np.hanning(win_length).astype(np.float32, copy=False)
        if win_length != n_fft:
            window_values = np.pad(window_values, (0, n_fft - win_length), mode="constant")
        return np.fft.rfft(np.ascontiguousarray(frames * window_values), axis=-1).T

    def _power_to_db(power, ref=np.max):
        matrix = np.asarray(power, dtype=np.float64)
        reference = ref(matrix) if callable(ref) else ref
        reference = max(float(reference), 1e-10)
        return 10.0 * np.log10(np.maximum(matrix, 1e-10) / reference)

    def _centroid(y, sr, hop_length, n_fft):
        frames = _frame_count_for(samples, hop_length, n_fft)
        value = centroid if centroid is not None else _signal_profile(samples, sample_rate)[0]
        return np.asarray([np.full(frames, value, dtype=np.float32)], dtype=np.float32)

    def _flatness(y, hop_length, n_fft):
        frames = _frame_count_for(samples, hop_length, n_fft)
        value = flatness if flatness is not None else _signal_profile(samples, sample_rate)[1]
        return np.asarray([np.full(frames, value, dtype=np.float32)], dtype=np.float32)

    def _mfcc(y, sr, n_mfcc, hop_length, n_fft):
        frames = _frame_count_for(samples, hop_length, n_fft)
        if mfcc_matrix is None:
            base = float(np.log1p(max(float(np.mean(np.abs(np.asarray(samples, dtype=np.float32)))), 0.0)))
            matrix = np.vstack([np.full(frames, base + idx, dtype=np.float32) for idx in range(n_mfcc)])
        else:
            matrix = np.asarray(mfcc_matrix, dtype=np.float32)
        return matrix

    def _centroid_from_s(*, y=None, sr=None, hop_length=512, n_fft=2048, S=None):
        return _centroid(samples if y is None else y, sr or sample_rate, hop_length, n_fft)

    def _flatness_from_s(*, y=None, hop_length=512, n_fft=2048, S=None, power=2.0):
        return _flatness(samples if y is None else y, hop_length, n_fft)

    def _melspectrogram(*, y=None, sr=None, S=None, n_mels=128, n_fft=2048, hop_length=512):
        if S is None:
            magnitude = np.abs(_stft(samples if y is None else y, n_fft=n_fft, hop_length=hop_length))
            S = magnitude * magnitude
        power = np.asarray(S, dtype=np.float64)
        bins = power.shape[0]
        edges = np.linspace(0, bins - 1, n_mels + 2)
        basis = np.zeros((n_mels, bins), dtype=np.float64)
        for row in range(n_mels):
            left = int(round(edges[row]))
            center = max(int(round(edges[row + 1])), left + 1)
            right = max(int(round(edges[row + 2])), center + 1)
            for col in range(left, min(center + 1, bins)):
                basis[row, col] = (col - left) / max(center - left, 1)
            for col in range(center, min(right + 1, bins)):
                basis[row, col] = max(basis[row, col], (right - col) / max(right - center, 1))
        return basis @ power

    def _mfcc_from_s(*, y=None, sr=None, S=None, n_mfcc=5, hop_length=512, n_fft=2048):
        if S is not None and y is None:
            frames = np.asarray(S).shape[-1] if np.asarray(S).ndim >= 2 else _frame_count_for(samples, hop_length, n_fft)
            if mfcc_matrix is None:
                base = float(np.log1p(max(float(np.mean(np.abs(np.asarray(samples, dtype=np.float32)))), 0.0)))
                return np.vstack([np.full(frames, base + idx, dtype=np.float32) for idx in range(n_mfcc)])
            return np.asarray(mfcc_matrix, dtype=np.float32)
        return _mfcc(samples if y is None else y, sr or sample_rate, n_mfcc, hop_length, n_fft)

    return SimpleNamespace(
        stft=_stft,
        power_to_db=_power_to_db,
        feature=SimpleNamespace(
            spectral_centroid=_centroid_from_s,
            spectral_flatness=_flatness_from_s,
            melspectrogram=_melspectrogram,
            mfcc=_mfcc_from_s,
        ),
    )


def _fake_librosa_with_feature(feature) -> SimpleNamespace:
    base = _build_fake_librosa(_speech_like(), audio.TARGET_SAMPLE_RATE)
    base.feature = feature
    if not hasattr(base.feature, "melspectrogram"):
        base.feature.melspectrogram = _build_fake_librosa(_speech_like(), audio.TARGET_SAMPLE_RATE).feature.melspectrogram
    return base


class FakeQueue:
    def __init__(self):
        self.items = []

    def put(self, payload):
        self.items.append(payload)


class FakeModel:
    def __init__(self, text="hello world"):
        self.text = text
        self.transcribe_calls = []

    def transcribe(self, path, language="en", fp16=False):
        self.transcribe_calls.append((path, language, fp16))
        return {"text": self.text}


class AudioUnitTests(TestCase):
    def setUp(self):
        audio._WHISPER_MODEL = None
        audio._WHISPER_MODEL_NAME = None

    def tearDown(self):
        audio._WHISPER_MODEL = None
        audio._WHISPER_MODEL_NAME = None

    def _analyze(self, samples: np.ndarray, *, sample_rate: int = 16000, path: str = "clip.wav", source_duration: float | None = None):
        prepared = np.asarray(samples, dtype=np.float32).reshape(-1)
        analyzed_duration = source_duration if source_duration is not None else prepared.size / float(sample_rate)
        fake_librosa = _build_fake_librosa(prepared, sample_rate)
        with patch.object(audio, "librosa", fake_librosa), patch.object(audio, "_prepare_audio_source", return_value=("ok", path)), patch.object(
            audio,
            "_decode_audio_once",
            return_value=(prepared, sample_rate, analyzed_duration, analyzed_duration, "test", 1),
        ):
            return audio.analyze_audio(path)

    def test_none_input_returns_missing(self):
        result = audio.analyze_audio(None)
        self.assertEqual(result["details"]["status"], "missing")
        self.assertEqual(result["details"]["audio_warnings"], ["audio_missing"])
        self.assertIsNone(result["score"])

    def test_non_string_input_returns_missing(self):
        result = audio.analyze_audio(123)
        self.assertEqual(result["details"]["status"], "missing")
        self.assertEqual(result["details"]["audio_warnings"], ["audio_missing"])

    def test_whitespace_input_returns_missing(self):
        result = audio.analyze_audio("   ")
        self.assertEqual(result["details"]["status"], "missing")
        self.assertEqual(result["details"]["audio_warnings"], ["audio_missing"])

    def test_missing_file_returns_load_failed(self):
        result = audio.analyze_audio("does-not-exist.wav")
        self.assertEqual(result["details"]["status"], "load_failed")
        self.assertEqual(result["details"]["audio_warnings"], ["audio_decode_failed"])
        self.assertIsNone(result["score"])

    def test_directory_returns_load_failed(self):
        tmpdir = tempfile.mkdtemp(prefix="audio-dir-")
        self.addCleanup(lambda: with_suppress_rmtree(tmpdir))
        result = audio.analyze_audio(tmpdir)
        self.assertEqual(result["details"]["status"], "load_failed")
        self.assertEqual(result["details"]["audio_warnings"], ["audio_decode_failed"])

    def test_empty_file_returns_empty_audio(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            path = handle.name
        self.addCleanup(lambda: with_suppress_remove(path))
        result = audio.analyze_audio(path)
        self.assertEqual(result["details"]["status"], "empty_audio")
        self.assertEqual(result["details"]["audio_warnings"], ["audio_decode_failed"])
        self.assertIsNone(result["score"])

    def test_corrupt_file_returns_load_failed(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            handle.write(b"not-a-valid-wav")
            path = handle.name
        self.addCleanup(lambda: with_suppress_remove(path))
        with patch.object(audio, "librosa", object()), patch.object(audio, "_decode_audio_once", side_effect=RuntimeError("audio_decode_failed")):
            result = audio.analyze_audio(path)
        self.assertEqual(result["details"]["status"], "load_failed")
        self.assertEqual(result["details"]["audio_warnings"], ["audio_decode_failed"])
        self.assertEqual(set(result["details"].keys()), {"status", "audio_warnings"})

    def test_failure_output_has_no_stale_metrics(self):
        with patch.object(audio, "librosa", object()), patch.object(audio, "_prepare_audio_source", return_value=("ok", "clip.wav")), patch.object(
            audio, "_decode_audio_once", side_effect=RuntimeError("audio_decode_failed")
        ):
            result = audio.analyze_audio("clip.wav")
        self.assertEqual(result["details"], {"status": "load_failed", "audio_warnings": ["audio_decode_failed"]})
        self.assertIsNone(result["score"])

    def test_librosa_unavailable_returns_load_failed(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            handle.write(b"RIFF")
            path = handle.name
        self.addCleanup(lambda: with_suppress_remove(path))
        with patch.object(audio, "librosa", None):
            result = audio.analyze_audio(path)
        self.assertEqual(result["details"]["status"], "load_failed")
        self.assertEqual(result["details"]["audio_warnings"], ["audio_decode_failed"])

    def test_strict_timeout_parser_handles_valid_and_invalid_values(self):
        self.assertEqual(audio.FFMPEG_CONVERSION_TIMEOUT_SECONDS, 3.5)
        cases = [
            (None, 3.5),
            (" 4.25 ", 4.25),
            ("7", 7.0),
            ("0.5", 0.5),
        ]
        for raw, expected in cases:
            with self.subTest(raw=raw):
                with patch.dict(os.environ, {}, clear=True):
                    if raw is not None:
                        os.environ["AUDIO_FFMPEG_CONVERSION_TIMEOUT_SECONDS"] = raw
                    value = audio._parse_positive_timeout_env("AUDIO_FFMPEG_CONVERSION_TIMEOUT_SECONDS", 3.5)
                    self.assertEqual(value, expected)
        invalid = ["", "abc", "0", "-1", "nan", "inf", "-inf"]
        for raw in invalid:
            with self.subTest(raw=raw):
                with patch.dict(os.environ, {"AUDIO_FFMPEG_CONVERSION_TIMEOUT_SECONDS": raw}, clear=True):
                    with self.assertRaises(ValueError):
                        audio._parse_positive_timeout_env("AUDIO_FFMPEG_CONVERSION_TIMEOUT_SECONDS", 3.5)

    def test_mfcc_summary_is_genuine_and_uses_five_coefficients(self):
        samples = _speech_like()
        prepared = np.asarray(samples, dtype=np.float32)
        mfcc_matrix = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0],
            ],
            dtype=np.float32,
        )
        fake_mfcc = MagicMock(return_value=mfcc_matrix)
        fake_centroid = MagicMock(return_value=np.array([[111.0, 111.0, 111.0]], dtype=np.float32))
        fake_flatness = MagicMock(return_value=np.array([[0.25, 0.25, 0.25]], dtype=np.float32))
        fake_librosa = _fake_librosa_with_feature(SimpleNamespace(mfcc=fake_mfcc, spectral_centroid=fake_centroid, spectral_flatness=fake_flatness))
        with patch.object(audio, "librosa", fake_librosa), patch.object(
            audio,
            "_prepare_audio_source",
            return_value=("ok", "clip.wav"),
        ), patch.object(audio, "_decode_audio_once", return_value=(prepared, audio.TARGET_SAMPLE_RATE, 3.0, 3.0, "test", 1)):
            result = audio.analyze_audio("clip.wav")
        self.assertEqual(fake_mfcc.call_count, 1)
        _, kwargs = fake_mfcc.call_args
        self.assertEqual(kwargs["n_mfcc"], 5)
        self.assertEqual(result["details"]["mfcc_summary"], [2.0, 5.0, 8.0, 11.0, 14.0])
        self.assertEqual(len(result["details"]["mfcc_summary"]), 5)
        self.assertEqual(result["details"]["spectral_centroid"], 111.0)
        self.assertEqual(result["details"]["spectral_flatness"], 0.25)

    def test_mfcc_nonfinite_and_bad_shape_fail_closed(self):
        samples = _speech_like()
        prepared = np.asarray(samples, dtype=np.float32)
        bad_cases = [
            np.array([[1.0, np.nan]] * 5, dtype=np.float32),
            np.array([[1.0, np.inf]] * 5, dtype=np.float32),
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            np.ones((4, 3), dtype=np.float32),
        ]
        for matrix in bad_cases:
            with self.subTest(shape=getattr(matrix, "shape", None)):
                fake_librosa = _fake_librosa_with_feature(
                    SimpleNamespace(
                        mfcc=MagicMock(return_value=matrix),
                        spectral_centroid=MagicMock(return_value=np.array([[111.0, 111.0]], dtype=np.float32)),
                        spectral_flatness=MagicMock(return_value=np.array([[0.25, 0.25]], dtype=np.float32)),
                    )
                )
                with patch.object(audio, "librosa", fake_librosa), patch.object(audio, "_prepare_audio_source", return_value=("ok", "clip.wav")), patch.object(
                    audio,
                    "_decode_audio_once",
                    return_value=(prepared, audio.TARGET_SAMPLE_RATE, 3.0, 3.0, "test", 1),
                ):
                    result = audio.analyze_audio("clip.wav")
                self.assertEqual(result["details"]["status"], "load_failed")
                self.assertEqual(result["details"]["audio_warnings"], ["audio_decode_failed"])

    def test_spectral_centroid_and_flatness_failures_return_load_failed(self):
        prepared = np.asarray(_speech_like(), dtype=np.float32)
        for centroid_side_effect, flatness_side_effect in [
            (RuntimeError("boom"), None),
            (None, RuntimeError("boom")),
        ]:
            with self.subTest(case=(centroid_side_effect, flatness_side_effect)):
                fake_librosa = _fake_librosa_with_feature(
                    SimpleNamespace(
                        mfcc=MagicMock(return_value=np.ones((5, 3), dtype=np.float32)),
                        spectral_centroid=MagicMock(side_effect=centroid_side_effect) if centroid_side_effect else MagicMock(return_value=np.array([[111.0, 111.0]], dtype=np.float32)),
                        spectral_flatness=MagicMock(side_effect=flatness_side_effect) if flatness_side_effect else MagicMock(return_value=np.array([[0.25, 0.25]], dtype=np.float32)),
                    )
                )
                with patch.object(audio, "librosa", fake_librosa), patch.object(audio, "_prepare_audio_source", return_value=("ok", "clip.wav")), patch.object(
                    audio,
                    "_decode_audio_once",
                    return_value=(prepared, audio.TARGET_SAMPLE_RATE, 3.0, 3.0, "test", 1),
                ):
                    result = audio.analyze_audio("clip.wav")
                self.assertEqual(result["details"]["status"], "load_failed")

    def test_feature_calls_receive_bounded_samples(self):
        captured = {}

        samples = _speech_like()
        fake_librosa = _build_fake_librosa(samples, audio.TARGET_SAMPLE_RATE)
        original_stft = fake_librosa.stft

        def _recording_stft(y, *args, **kwargs):
            captured["stft_samples"] = len(np.asarray(y).reshape(-1))
            return original_stft(y, *args, **kwargs)

        def _recording_mfcc(*, S, n_mfcc, **_kwargs):
            captured["mfcc_from_mel_bins"] = np.asarray(S).shape[0]
            return np.ones((5, 3), dtype=np.float32)

        def _recording_centroid(*, S, sr, **_kwargs):
            captured["centroid_bins"] = np.asarray(S).shape[0]
            return np.array([[111.0, 111.0, 111.0]], dtype=np.float32)

        def _recording_flatness(*, S, **_kwargs):
            captured["flatness_bins"] = np.asarray(S).shape[0]
            return np.array([[0.2, 0.2, 0.2]], dtype=np.float32)

        fake_librosa.stft = _recording_stft
        fake_librosa.feature.mfcc = _recording_mfcc
        fake_librosa.feature.spectral_centroid = _recording_centroid
        fake_librosa.feature.spectral_flatness = _recording_flatness
        with patch.object(audio, "librosa", fake_librosa), patch.object(audio, "_prepare_audio_source", return_value=("ok", "clip.wav")), patch.object(
            audio,
            "_decode_audio_once",
            return_value=(samples, audio.TARGET_SAMPLE_RATE, 3.0, 3.0, "test", 1),
        ):
            result = audio.analyze_audio("clip.wav")
        self.assertEqual(result["details"]["status"], "ok")
        self.assertLessEqual(captured["stft_samples"], audio.MAX_AUDIO_SAMPLES)
        self.assertGreater(captured["mfcc_from_mel_bins"], 0)
        self.assertGreater(captured["centroid_bins"], 0)
        self.assertGreater(captured["flatness_bins"], 0)

    def test_high_source_rate_resampling_is_bounded_and_contiguous(self):
        for sample_rate in (48000, 96000, 192000):
            with self.subTest(sample_rate=sample_rate):
                samples = _sine(duration_sec=3.0, sample_rate=sample_rate, amplitude=0.2)
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    path = tmp.name
                self.addCleanup(lambda p=path: with_suppress_remove(p))
                _write_wav(path, samples, sample_rate=sample_rate)
                y, sr, _, analyzed_duration, _, _ = audio._decode_wav_slice(path)
                self.assertEqual(sr, audio.TARGET_SAMPLE_RATE)
                self.assertLessEqual(len(y), audio.MAX_AUDIO_SAMPLES)
                self.assertTrue(y.flags["C_CONTIGUOUS"])
                self.assertTrue(np.isfinite(y).all())
                self.assertEqual(y.dtype, np.float32)
                self.assertLessEqual(analyzed_duration, audio.MAX_AUDIO_ANALYSIS_SEC)

    def test_timing_fields_are_owned_by_expected_stages(self):
        counter = iter([idx / 1000.0 for idx in range(1, 200)])

        def fake_perf_counter():
            return next(counter)

        with patch.object(audio.time, "perf_counter", side_effect=fake_perf_counter):
            result = self._analyze(_speech_like())
        details = result["details"]
        self.assertIsInstance(details["timings_ms"]["audio_decode_ms"], int)
        self.assertIsInstance(details["timings_ms"]["audio_quality_ms"], int)
        self.assertIsInstance(details["timings_ms"]["voice_activity_ms"], int)
        for key in ("rms_ms", "zcr_ms", "spectral_centroid_ms", "spectral_flatness_ms", "mfcc_ms", "derived_metrics_ms"):
            self.assertIsInstance(details["audio_quality_timings_ms"][key], int)
        self.assertGreaterEqual(details["timings_ms"]["audio_decode_ms"], 0)
        self.assertGreaterEqual(details["timings_ms"]["audio_quality_ms"], 0)
        self.assertGreaterEqual(details["timings_ms"]["voice_activity_ms"], 0)

    def test_tone_inputs_are_not_usable_speech(self):
        for frequency in (220.0, 440.0, 1200.0):
            with self.subTest(frequency=frequency):
                result = self._analyze(_sine(frequency=frequency, amplitude=0.2))
                details = result["details"]
                self.assertFalse(details["usable_speech_detected"])
                self.assertIn("speech_not_detected", details["audio_warnings"])
                self.assertIsNone(details["speech_rate"])
                self.assertIn(details["speech_state"], {"no_speech", "unusable_quality"})

    def test_speech_like_modulation_remains_usable(self):
        result = self._analyze(_speech_like(amplitude=0.02))
        self.assertIn(result["details"]["speech_state"], {"quiet_usable_speech", "usable_speech"})
        self.assertTrue(result["details"]["usable_speech_detected"])

    def test_no_speech_state_is_not_quiet_but_usable(self):
        warnings, speech_state, quiet_but_usable, usable_speech_detected = audio._speech_state_and_warnings(
            duration_seconds=3.0,
            rms_energy=audio.MIN_RMS_ENERGY * 0.7,
            noise_estimate=0.2,
            silence_ratio=0.5,
            speech_presence_score=0.13,
            clipping_ratio=0.0,
            tonal_concentration=0.2,
            rms_variation=0.5,
        )
        self.assertIn("speech_not_detected", warnings)
        self.assertEqual(speech_state, "no_speech")
        self.assertFalse(quiet_but_usable)
        self.assertFalse(usable_speech_detected)

    def test_silence_tone_noise_and_clipping_confidence_are_gated(self):
        speech = self._analyze(_speech_like(amplitude=0.02))
        silence = self._analyze(np.zeros(int(audio.TARGET_SAMPLE_RATE * 3), dtype=np.float32))
        tone = self._analyze(_sine(amplitude=0.2))
        noise = self._analyze(_noise(amplitude=0.9))
        clipped = self._analyze(np.ones(int(audio.TARGET_SAMPLE_RATE * 3), dtype=np.float32))
        self.assertGreater(speech["details"]["voice_clarity_score"], silence["details"]["voice_clarity_score"])
        self.assertGreater(speech["details"]["voice_clarity_score"], tone["details"]["voice_clarity_score"])
        self.assertGreater(speech["details"]["voice_clarity_score"], noise["details"]["voice_clarity_score"])
        self.assertGreater(speech["details"]["voice_clarity_score"], clipped["details"]["voice_clarity_score"])
        self.assertGreater(speech["details"]["audio_confidence"], silence["details"]["audio_confidence"])
        self.assertGreater(speech["details"]["audio_confidence"], tone["details"]["audio_confidence"])
        self.assertGreater(speech["details"]["audio_confidence"], noise["details"]["audio_confidence"])
        self.assertGreater(speech["details"]["audio_confidence"], clipped["details"]["audio_confidence"])

    def test_decode_before_whisper_loading_and_transcription_failure_semantics(self):
        fake_model = FakeModel()
        load_model = MagicMock(return_value=fake_model)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            handle.write(b"not-a-valid-wav")
            corrupt_path = handle.name
        self.addCleanup(lambda: with_suppress_remove(corrupt_path))
        with patch.object(audio, "whisper", SimpleNamespace(load_model=load_model)), patch.object(audio, "_decode_audio_once", side_effect=RuntimeError("audio_decode_failed")):
            with self.assertRaisesRegex(RuntimeError, "audio_decode_failed"):
                audio.transcribe_audio(corrupt_path)
        load_model.assert_not_called()

        samples = _speech_like()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            path = handle.name
        self.addCleanup(lambda: with_suppress_remove(path))
        _write_wav(path, samples, sample_rate=audio.TARGET_SAMPLE_RATE)
        load_model = MagicMock(return_value=fake_model)
        fake_model.transcribe = MagicMock(return_value={"text": "hello"})
        with patch.object(audio, "whisper", SimpleNamespace(load_model=load_model)):
            text = audio.transcribe_audio(path)
        self.assertEqual(text, "hello")
        load_model.assert_called_once_with("tiny.en")
        fake_model.transcribe.assert_called_once()

    def test_transcription_decode_timeout_and_model_failure_messages(self):
        samples = _speech_like()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            path = handle.name
        self.addCleanup(lambda: with_suppress_remove(path))
        _write_wav(path, samples, sample_rate=audio.TARGET_SAMPLE_RATE)
        with patch.object(audio, "_decode_audio_once", side_effect=TimeoutError("audio_decode_timeout")), patch.object(audio, "whisper", SimpleNamespace(load_model=MagicMock())):
            with self.assertRaisesRegex(RuntimeError, "audio_decode_failed") as exc:
                audio.transcribe_audio(path)
            self.assertNotIn(path, str(exc.exception))
        fake_model = FakeModel()
        fake_model.transcribe = MagicMock(side_effect=RuntimeError("boom"))
        with patch.object(audio, "_decode_audio_once", return_value=(np.asarray(samples, dtype=np.float32), audio.TARGET_SAMPLE_RATE, 3.0, 3.0, "test", 1)), patch.object(
            audio, "whisper", SimpleNamespace(load_model=MagicMock(return_value=fake_model))
        ):
            with self.assertRaisesRegex(RuntimeError, "transcription_failed") as exc:
                audio.transcribe_audio(path)
        self.assertNotIn(path, str(exc.exception))

    def test_success_result_satisfies_schema_and_audio_rate_is_none(self):
        result = self._analyze(_speech_like())
        details = result["details"]
        self.assertEqual(details["status"], "ok")
        self.assertIn("audio_quality_score", details)
        self.assertIn("audio_confidence", details)
        self.assertIsNone(details["speech_rate"])
        self.assertIsNone(details["pitch_stability_score"])
        self.assertGreaterEqual(details["audio_confidence"], 0.0)
        self.assertLessEqual(details["audio_confidence"], 1.0)
        self.assertAlmostEqual(result["score"], details["audio_confidence"])
        self.assertGreaterEqual(details["duration_seconds"], 0.0)
        self.assertLessEqual(details["duration_seconds"], audio.MAX_AUDIO_ANALYSIS_SEC)
        self.assertIsInstance(details["audio_warnings"], list)
        self.assertEqual(details["audio_warnings"], list(dict.fromkeys(details["audio_warnings"])))
        self.assertTrue(math.isfinite(details["audio_quality_score"]))
        self.assertTrue(math.isfinite(details["audio_confidence"]))

    def test_pure_silence_is_not_noisy_and_emits_no_speech(self):
        result = self._analyze(np.zeros(int(audio.TARGET_SAMPLE_RATE * 3), dtype=np.float32))
        details = result["details"]
        self.assertNotIn("audio_too_noisy", details["audio_warnings"])
        self.assertIn("speech_not_detected", details["audio_warnings"])
        self.assertEqual(details["speech_state"], "no_speech")
        self.assertLess(details["audio_confidence"], 0.5)
        self.assertIsNone(details["speech_rate"])

    def test_tone_and_noise_keep_speech_rate_none(self):
        cases = [
            _sine(amplitude=0.12),
            _noise(amplitude=0.4),
            _speech_like(),
        ]
        for samples in cases:
            with self.subTest(kind=samples.shape[0]):
                result = self._analyze(samples)
                self.assertIsNone(result["details"]["speech_rate"])

    def test_quiet_but_usable_signal_sets_quiet_but_usable(self):
        result = self._analyze(_speech_like(amplitude=0.02))
        self.assertIn(result["details"]["speech_state"], {"quiet_usable_speech", "usable_speech"})
        if result["details"]["speech_state"] == "quiet_usable_speech":
            self.assertTrue(result["details"]["quiet_but_usable"])
            self.assertTrue(result["details"]["usable_speech_detected"])

    def test_too_quiet_signal_is_not_quiet_but_usable(self):
        result = self._analyze(_speech_like(amplitude=0.0015))
        self.assertFalse(result["details"]["quiet_but_usable"])
        self.assertIn("audio_too_quiet", result["details"]["audio_warnings"])

    def test_noise_produces_audio_too_noisy(self):
        result = self._analyze(_noise(amplitude=0.9))
        self.assertIn("audio_too_noisy", result["details"]["audio_warnings"])
        self.assertLess(result["details"]["speech_presence_score"], 0.7)

    def test_clipping_produces_audio_clipping(self):
        result = self._analyze(np.ones(int(audio.TARGET_SAMPLE_RATE * 3), dtype=np.float32))
        self.assertIn("audio_clipping", result["details"]["audio_warnings"])
        self.assertLess(result["details"]["audio_confidence"], 1.0)

    def test_warning_order_is_deterministic(self):
        samples = np.zeros(int(audio.TARGET_SAMPLE_RATE * 3), dtype=np.float32)
        samples[:1000] = 1.0
        result = self._analyze(samples)
        self.assertEqual(result["details"]["audio_warnings"], list(dict.fromkeys(result["details"]["audio_warnings"])))

    def test_duration_factor_reaches_one_at_analysis_window(self):
        quality = audio._quality_from_features(
            duration_seconds=audio.MAX_AUDIO_ANALYSIS_SEC,
            rms_energy=0.05,
            silence_ratio=0.0,
            noise_estimate=0.0,
            peak_volume=0.4,
            clipping_ratio=0.0,
        )
        self.assertAlmostEqual(quality, 1.0, places=6)

    def test_duration_factor_is_normalized_below_analysis_window(self):
        quality = audio._quality_from_features(
            duration_seconds=1.5,
            rms_energy=0.05,
            silence_ratio=0.0,
            noise_estimate=0.0,
            peak_volume=0.4,
            clipping_ratio=0.0,
        )
        self.assertLess(quality, 1.0)
        self.assertGreaterEqual(quality, 0.0)

    def test_rms_numpy_returns_finite_nonnegative_values(self):
        rms = audio._rms_numpy(np.zeros(1024, dtype=np.float32), frame_length=256, hop_length=64)
        self.assertTrue(np.isfinite(rms).all())
        self.assertTrue((rms >= 0).all())
        self.assertTrue(np.allclose(rms, 0.0))

    def test_rms_numpy_rejects_invalid_inputs(self):
        for bad in [
            np.array([], dtype=np.float32),
            np.array([1, 2], dtype=object),
            np.array([1 + 1j], dtype=np.complex64),
            np.array([True, False], dtype=bool),
        ]:
            with self.subTest(dtype=getattr(bad, "dtype", None)):
                with self.assertRaises(ValueError):
                    audio._rms_numpy(bad, frame_length=256, hop_length=64)

    def test_zcr_numpy_bounds_and_signal_behavior(self):
        zero = audio._zero_crossing_rate_numpy(np.zeros(1024, dtype=np.float32), frame_length=256, hop_length=64)
        alt = audio._zero_crossing_rate_numpy(np.tile(np.array([1.0, -1.0], dtype=np.float32), 512), frame_length=256, hop_length=64)
        self.assertTrue(np.isfinite(zero).all())
        self.assertTrue(np.isfinite(alt).all())
        self.assertTrue((zero >= 0.0).all() and (zero <= 1.0).all())
        self.assertTrue((alt >= 0.0).all() and (alt <= 1.0).all())
        self.assertLess(np.mean(zero), 0.05)
        self.assertGreater(np.mean(alt), 0.4)

    def test_zcr_numpy_rejects_invalid_inputs(self):
        with self.assertRaises(ValueError):
            audio._zero_crossing_rate_numpy(np.array([], dtype=np.float32), frame_length=256, hop_length=64)
        with self.assertRaises(ValueError):
            audio._zero_crossing_rate_numpy(np.array([True, False]), frame_length=256, hop_length=64)

    def test_decode_wav_slice_closes_handle(self):
        handle = MagicMock()
        handle.__enter__.return_value = handle
        handle.__exit__.return_value = False
        handle.getframerate.return_value = 16000
        handle.getnchannels.return_value = 1
        handle.getsampwidth.return_value = 2
        handle.getnframes.return_value = 4
        handle.readframes.return_value = _float_to_pcm16(np.array([0.0, 0.1, -0.1, 0.0], dtype=np.float32))
        with patch.object(audio.wave, "open", return_value=handle):
            y, sr, source_duration, analyzed_duration, backend, count = audio._decode_wav_slice("clip.wav")
        self.assertEqual(backend, "wave")
        self.assertEqual(sr, 16000)
        self.assertEqual(count, 4)
        self.assertTrue(handle.__exit__.called)
        self.assertEqual(len(y), 4)
        self.assertGreater(source_duration, 0.0)
        self.assertGreater(analyzed_duration, 0.0)

    def test_decode_wav_slice_rejects_unsupported_metadata(self):
        handle = MagicMock()
        handle.__enter__.return_value = handle
        handle.__exit__.return_value = False
        handle.getframerate.return_value = 0
        handle.getnchannels.return_value = 1
        handle.getsampwidth.return_value = 2
        handle.getnframes.return_value = 4
        handle.readframes.return_value = b"\x00" * 8
        with patch.object(audio.wave, "open", return_value=handle):
            with self.assertRaises(ValueError):
                audio._decode_wav_slice("clip.wav")

    def test_decode_wav_slice_resamples_to_target_rate_and_bounded_length(self):
        samples = _sine(duration_sec=4.0, sample_rate=8000)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            path = tmp.name
        self.addCleanup(lambda: with_suppress_remove(path))
        _write_wav(path, samples, sample_rate=8000)
        y, sr, _, analyzed_duration, _, _ = audio._decode_wav_slice(path)
        self.assertEqual(sr, audio.TARGET_SAMPLE_RATE)
        self.assertLessEqual(len(y), audio.MAX_AUDIO_SAMPLES)
        self.assertLessEqual(analyzed_duration, audio.MAX_AUDIO_ANALYSIS_SEC)

    def test_resample_if_needed_uses_target_rate(self):
        y, sr = audio._resample_if_needed(np.ones(8000, dtype=np.float32), 8000)
        self.assertEqual(sr, audio.TARGET_SAMPLE_RATE)
        self.assertLessEqual(len(y), audio.MAX_AUDIO_SAMPLES)

    def test_ffmpeg_cleanup_on_success(self):
        fd, path = tempfile.mkstemp(suffix=".wav")
        self.addCleanup(lambda: with_suppress_remove(path))
        with patch.object(audio.shutil, "which", return_value="ffmpeg"), patch.object(audio.tempfile, "mkstemp", return_value=(fd, path)), patch.object(
            audio.subprocess,
            "run",
            return_value=None,
        ), patch.object(audio, "_decode_wav_slice", return_value=(np.zeros(4, dtype=np.float32), 16000, 0.1, 0.1, "wave", 1)):
            result = audio._decode_with_ffmpeg("clip.m4a")
        self.assertEqual(result[1], 16000)
        with self.assertRaises(OSError):
            os.fstat(fd)
        self.assertFalse(os.path.exists(path))

    def test_ffmpeg_cleanup_on_timeout(self):
        fd, path = tempfile.mkstemp(suffix=".wav")
        self.addCleanup(lambda: with_suppress_remove(path))
        with patch.object(audio.shutil, "which", return_value="ffmpeg"), patch.object(audio.tempfile, "mkstemp", return_value=(fd, path)), patch.object(
            audio.subprocess,
            "run",
            side_effect=audio.subprocess.TimeoutExpired(cmd=["ffmpeg"], timeout=3.5),
        ):
            with self.assertRaises(TimeoutError):
                audio._decode_with_ffmpeg("clip.m4a")
        with self.assertRaises(OSError):
            os.fstat(fd)
        self.assertFalse(os.path.exists(path))

    def test_ffmpeg_cleanup_on_failure(self):
        fd, path = tempfile.mkstemp(suffix=".wav")
        self.addCleanup(lambda: with_suppress_remove(path))
        with patch.object(audio.shutil, "which", return_value="ffmpeg"), patch.object(audio.tempfile, "mkstemp", return_value=(fd, path)), patch.object(
            audio.subprocess,
            "run",
            side_effect=audio.subprocess.CalledProcessError(returncode=1, cmd=["ffmpeg"]),
        ):
            with self.assertRaises(RuntimeError):
                audio._decode_with_ffmpeg("clip.m4a")
        with self.assertRaises(OSError):
            os.fstat(fd)
        self.assertFalse(os.path.exists(path))

    def test_ffmpeg_cleanup_on_wav_decode_failure(self):
        fd, path = tempfile.mkstemp(suffix=".wav")
        self.addCleanup(lambda: with_suppress_remove(path))
        with patch.object(audio.shutil, "which", return_value="ffmpeg"), patch.object(audio.tempfile, "mkstemp", return_value=(fd, path)), patch.object(
            audio.subprocess,
            "run",
            return_value=None,
        ), patch.object(audio, "_decode_wav_slice", side_effect=RuntimeError("audio_decode_failed")):
            with self.assertRaises(RuntimeError):
                audio._decode_with_ffmpeg("clip.m4a")
        with self.assertRaises(OSError):
            os.fstat(fd)
        self.assertFalse(os.path.exists(path))

    def test_transcribe_audio_missing_input_raises_audio_missing(self):
        with self.assertRaisesRegex(RuntimeError, "audio_missing"):
            audio.transcribe_audio(None)

    def test_transcribe_audio_invalid_file_raises_decode_failure(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            path = handle.name
        self.addCleanup(lambda: with_suppress_remove(path))
        with self.assertRaisesRegex(RuntimeError, "audio_decode_failed") as exc:
            audio.transcribe_audio(path)
        self.assertNotIn(path, str(exc.exception))

    def test_transcribe_audio_uses_lazy_model_and_reuses_cache(self):
        fake_model = FakeModel()
        load_model = MagicMock(return_value=fake_model)
        with patch.object(audio, "whisper", MagicMock(load_model=load_model)):
            first = audio._load_whisper_model()
            second = audio._load_whisper_model()
        self.assertIs(first, fake_model)
        self.assertIs(second, fake_model)
        load_model.assert_called_once_with("tiny.en")

    def test_transcribe_audio_concurrent_first_loads_initialize_once(self):
        fake_model = FakeModel()
        load_model = MagicMock(return_value=fake_model)
        barrier = threading.Barrier(2)

        def worker():
            barrier.wait()
            audio._load_whisper_model()

        with patch.object(audio, "whisper", MagicMock(load_model=load_model)):
            threads = [threading.Thread(target=worker) for _ in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        load_model.assert_called_once_with("tiny.en")

    def test_transcribe_audio_empty_text_fails(self):
        fake_model = MagicMock()
        fake_model.transcribe.return_value = {"text": "   "}
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            path = handle.name
        self.addCleanup(lambda: with_suppress_remove(path))
        samples = _speech_like()
        with patch.object(audio, "_prepare_audio_source", return_value=("ok", path)), patch.object(
            audio,
            "_decode_audio_once",
            return_value=(np.asarray(samples, dtype=np.float32), audio.TARGET_SAMPLE_RATE, 3.0, 3.0, "test", 1),
        ), patch.object(
            audio,
            "whisper",
            MagicMock(load_model=MagicMock(return_value=fake_model)),
        ):
            with self.assertRaisesRegex(RuntimeError, "transcription_failed"):
                audio.transcribe_audio(path)

    def test_transcribe_audio_malformed_model_result_fails(self):
        fake_model = MagicMock()
        fake_model.transcribe.return_value = ["not-a-dict"]
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            path = handle.name
        self.addCleanup(lambda: with_suppress_remove(path))
        samples = _speech_like()
        with patch.object(audio, "_prepare_audio_source", return_value=("ok", path)), patch.object(
            audio,
            "_decode_audio_once",
            return_value=(np.asarray(samples, dtype=np.float32), audio.TARGET_SAMPLE_RATE, 3.0, 3.0, "test", 1),
        ), patch.object(
            audio,
            "whisper",
            MagicMock(load_model=MagicMock(return_value=fake_model)),
        ):
            with self.assertRaisesRegex(RuntimeError, "transcription_failed"):
                audio.transcribe_audio(path)

    def test_transcribe_audio_returns_trimmed_text(self):
        fake_model = FakeModel(text="  hello world  ")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            path = handle.name
        self.addCleanup(lambda: with_suppress_remove(path))
        samples = _speech_like()
        with patch.object(audio, "_prepare_audio_source", return_value=("ok", path)), patch.object(
            audio,
            "_decode_audio_once",
            return_value=(np.asarray(samples, dtype=np.float32), audio.TARGET_SAMPLE_RATE, 3.0, 3.0, "test", 1),
        ), patch.object(
            audio,
            "whisper",
            MagicMock(load_model=MagicMock(return_value=fake_model)),
        ):
            self.assertEqual(audio.transcribe_audio(path), "hello world")

    def test_worker_success_queue_payload_is_serializable(self):
        q = FakeQueue()
        with patch.object(audio, "analyze_audio", return_value={"score": 0.5, "details": {"status": "ok"}}):
            audio.analyze_audio_worker("clip.wav", q)
        self.assertEqual(q.items, [{"ok": True, "result": {"score": 0.5, "details": {"status": "ok"}}}])

    def test_worker_failure_payload_hides_message_and_path(self):
        q = FakeQueue()
        with patch.object(audio, "analyze_audio", side_effect=RuntimeError("boom /tmp/secret.wav")):
            audio.analyze_audio_worker("clip.wav", q)
        self.assertEqual(q.items, [{"ok": False, "error": "RuntimeError"}])

    def test_worker_keyboard_interrupt_is_not_misreported(self):
        q = FakeQueue()
        with patch.object(audio, "analyze_audio", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                audio.analyze_audio_worker("clip.wav", q)
        self.assertEqual(q.items, [])


def with_suppress_remove(path: str) -> None:
    with suppress(OSError):
        os.remove(path)


def with_suppress_rmtree(path: str) -> None:
    with suppress(OSError):
        shutil.rmtree(path, ignore_errors=True)
