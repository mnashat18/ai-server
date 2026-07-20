from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave
from contextlib import suppress
from typing import Any, Callable

import numpy as np

from logger import get_logger
from utils import clean_warning_codes, clamp01, safe_number

try:  # pragma: no cover - optional dependency
    import librosa
except Exception:  # pragma: no cover - optional dependency
    librosa = None

try:  # pragma: no cover - optional dependency
    import whisper
except Exception:  # pragma: no cover - optional dependency
    whisper = None

try:  # pragma: no cover - optional dependency
    from scipy import signal
except Exception:  # pragma: no cover - optional dependency
    signal = None


MIN_AUDIO_DURATION_SEC = 1.5
MIN_RMS_ENERGY = 0.012
MAX_REASONABLE_PEAK = 0.98
TARGET_SAMPLE_RATE = 16000
MAX_AUDIO_ANALYSIS_SEC = 3.0
MAX_AUDIO_SAMPLES = int(TARGET_SAMPLE_RATE * MAX_AUDIO_ANALYSIS_SEC)
CLIPPING_SAMPLE_THRESHOLD = 0.98
MAX_CLIPPING_RATIO = 0.015
MIN_SOURCE_SAMPLE_RATE = 8000
MAX_SOURCE_SAMPLE_RATE = 192000
MAX_SOURCE_CHANNELS = 8
ALLOWED_SAMPLE_WIDTHS = {1, 2, 3, 4}


def _parse_positive_timeout_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    if not isinstance(raw, str):
        raise ValueError(f"{name} must be a finite positive number")
    text = raw.strip()
    if not text:
        raise ValueError(f"{name} must be a finite positive number")
    try:
        numeric = float(text)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a finite positive number") from None
    if not math.isfinite(numeric) or numeric <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return numeric


FFMPEG_CONVERSION_TIMEOUT_SECONDS = _parse_positive_timeout_env(
    "AUDIO_FFMPEG_CONVERSION_TIMEOUT_SECONDS",
    3.5,
)

_WHISPER_MODEL = None
_WHISPER_MODEL_NAME = None
_WHISPER_LOCK = threading.RLock()
logger = get_logger()


class _AudioEmptyError(RuntimeError):
    pass


class AudioPrewarmError(RuntimeError):
    def __init__(self, terminal_reason: str, *, failure_stage: str, metrics: dict[str, Any] | None = None):
        super().__init__(terminal_reason)
        self.terminal_reason = terminal_reason
        self.failure_stage = failure_stage
        self.metrics = metrics or {}


def _elapsed_ms(started_at: float) -> int:
    return int(round((time.perf_counter() - started_at) * 1000))


def _log_audio_perf(scan_id: str | None, metric: str, started_at: float, *, status: str = "ok") -> None:
    value = 0 if status == "skipped" else _elapsed_ms(started_at)
    logger.info("[AUDIO_PERF] metric=%s scan_id=%s value=%s status=%s", metric, scan_id, value, status)
    for handler in getattr(logger, "handlers", []):
        with suppress(Exception):
            handler.flush()


def _write_deterministic_prewarm_wav(path: str) -> None:
    sr = TARGET_SAMPLE_RATE
    duration_seconds = 5.0
    sample_count = int(sr * duration_seconds)
    t = np.arange(sample_count, dtype=np.float32) / float(sr)
    envelope = np.where((t > 0.35) & (t < 4.65), 1.0, 0.0).astype(np.float32)
    slow_modulation = 0.65 + 0.35 * np.sin(2.0 * np.pi * 2.3 * t).astype(np.float32)
    tone = (
        0.035 * np.sin(2.0 * np.pi * 180.0 * t)
        + 0.022 * np.sin(2.0 * np.pi * 360.0 * t)
        + 0.012 * np.sin(2.0 * np.pi * 720.0 * t)
    ).astype(np.float32)
    noise = np.random.default_rng(42).normal(0.0, 0.003, sample_count).astype(np.float32)
    samples = np.clip(envelope * slow_modulation * tone + envelope * noise, -0.95, 0.95)
    pcm = np.asarray(np.round(samples * 32767.0), dtype="<i2")
    with wave.open(path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sr)
        handle.writeframes(pcm.tobytes())


def _prewarm_result_summary(result: dict | None) -> dict[str, Any]:
    details = result.get("details") if isinstance(result, dict) else {}
    if not isinstance(details, dict):
        details = {}
    return {
        "status": details.get("status"),
        "audio_warnings": list(details.get("audio_warnings") or []),
        "audio_confidence": details.get("audio_confidence"),
        "duration_seconds": details.get("duration_seconds"),
        "rms_energy": details.get("rms_energy"),
        "spectral_centroid": details.get("spectral_centroid"),
        "zero_crossing_rate": details.get("zero_crossing_rate"),
        "mfcc_length": len(details.get("mfcc_summary") or []) if isinstance(details.get("mfcc_summary"), list) else 0,
    }


def _raise_audio_prewarm_error(terminal_reason: str, *, failure_stage: str, metrics: dict[str, Any]) -> None:
    metrics["audio_prewarm_terminal_reason"] = terminal_reason
    metrics["audio_prewarm_failure_stage"] = failure_stage
    raise AudioPrewarmError(terminal_reason, failure_stage=failure_stage, metrics=dict(metrics))


def prewarm_audio_analyzer(progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    fd = None
    path = None
    first_result: dict | None = None
    second_result: dict | None = None
    metrics: dict[str, Any] = {
        "audio_prewarm_first_call_ms": 0,
        "audio_prewarm_second_call_ms": 0,
        "audio_prewarm_decode_ms": 0,
        "audio_prewarm_vad_ms": 0,
        "audio_prewarm_pitch_ms": 0,
        "audio_prewarm_spectral_ms": 0,
        "audio_prewarm_mel_ms": 0,
        "audio_prewarm_mfcc_ms": 0,
        "audio_prewarm_total_ms": 0,
        "audio_warm_benchmark_passed": False,
        "audio_prewarm_temp_deleted": False,
        "audio_prewarm_terminal_reason": None,
        "audio_prewarm_failure_stage": None,
    }
    total_started = time.perf_counter()
    def _progress(stage: str) -> None:
        if progress is None:
            return
        progress(stage)

    try:
        try:
            fd, path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            fd = None
            _write_deterministic_prewarm_wav(path)
            _progress("prewarm_file_created")
        except Exception:
            _raise_audio_prewarm_error(
                "audio_prewarm_file_generation_failed",
                failure_stage="file_generation",
                metrics=metrics,
            )

        first_started = time.perf_counter()
        _progress("first_call_started")
        try:
            first_result = analyze_audio(path, scan_id=None)
        except Exception:
            metrics["audio_prewarm_first_call_ms"] = _elapsed_ms(first_started)
            _raise_audio_prewarm_error(
                "audio_prewarm_first_call_failed",
                failure_stage="first_call",
                metrics=metrics,
            )
        metrics["audio_prewarm_first_call_ms"] = _elapsed_ms(first_started)
        _progress("first_call_completed")
        metrics["audio_prewarm_first_result"] = _prewarm_result_summary(first_result)
        if not _audio_prewarm_result_valid(first_result):
            _raise_audio_prewarm_error(
                "audio_prewarm_first_result_invalid",
                failure_stage="first_result_validation",
                metrics=metrics,
            )
        _progress("first_result_validated")

        second_started = time.perf_counter()
        _progress("second_call_started")
        try:
            second_result = analyze_audio(path, scan_id=None)
        except Exception:
            metrics["audio_prewarm_second_call_ms"] = _elapsed_ms(second_started)
            _raise_audio_prewarm_error(
                "audio_prewarm_second_call_failed",
                failure_stage="second_call",
                metrics=metrics,
            )
        metrics["audio_prewarm_second_call_ms"] = _elapsed_ms(second_started)
        _progress("second_call_completed")
        metrics["audio_prewarm_second_result"] = _prewarm_result_summary(second_result)
        if not _audio_prewarm_result_valid(second_result):
            _raise_audio_prewarm_error(
                "audio_prewarm_second_result_invalid",
                failure_stage="second_result_validation",
                metrics=metrics,
            )
        _progress("second_result_validated")

        details = second_result.get("details") if isinstance(second_result, dict) else {}
        timings = details.get("timings_ms") if isinstance(details, dict) else {}
        quality_timings = details.get("audio_quality_timings_ms") if isinstance(details, dict) else {}
        metrics["audio_prewarm_decode_ms"] = int(timings.get("audio_decode_ms") or 0)
        metrics["audio_prewarm_vad_ms"] = int(timings.get("voice_activity_ms") or 0)
        metrics["audio_prewarm_pitch_ms"] = int(quality_timings.get("pitch_ms") or 0)
        metrics["audio_prewarm_spectral_ms"] = int(quality_timings.get("spectral_total_ms") or 0)
        metrics["audio_prewarm_mel_ms"] = int(quality_timings.get("mel_ms") or 0)
        metrics["audio_prewarm_mfcc_ms"] = int(quality_timings.get("mfcc_transform_ms") or quality_timings.get("mfcc_ms") or 0)
        metrics["audio_warm_benchmark_passed"] = True
        return metrics
    finally:
        metrics["audio_prewarm_total_ms"] = _elapsed_ms(total_started)
        cleanup_failed = False
        if fd is not None:
            try:
                os.close(fd)
            except Exception:
                cleanup_failed = True
        if path:
            try:
                os.remove(path)
                metrics["audio_prewarm_temp_deleted"] = True
            except FileNotFoundError:
                metrics["audio_prewarm_temp_deleted"] = True
            except Exception:
                cleanup_failed = True
        if not cleanup_failed:
            _progress("cleanup_completed")
        if cleanup_failed:
            metrics["audio_prewarm_terminal_reason"] = "audio_prewarm_cleanup_failed"
            metrics["audio_prewarm_failure_stage"] = "cleanup"
            if sys.exc_info()[0] is None:
                raise AudioPrewarmError(
                    "audio_prewarm_cleanup_failed",
                    failure_stage="cleanup",
                    metrics=dict(metrics),
                )


def _audio_prewarm_result_valid(result: dict | None) -> bool:
    if not isinstance(result, dict):
        return False
    details = result.get("details")
    if not isinstance(details, dict) or details.get("status") != "ok":
        return False
    required_numbers = [
        details.get("audio_confidence"),
        details.get("audio_quality_score"),
        details.get("duration_seconds"),
        details.get("rms_energy"),
        details.get("spectral_centroid"),
        details.get("zero_crossing_rate"),
    ]
    if not all(type(value) in {int, float} and math.isfinite(float(value)) for value in required_numbers):
        return False
    mfcc = details.get("mfcc_summary")
    return isinstance(mfcc, list) and len(mfcc) == 5 and all(type(value) in {int, float} and math.isfinite(float(value)) for value in mfcc)


def _is_string_like(value) -> bool:
    return isinstance(value, str)


def _normalize_audio_path(audio_path) -> str | None:
    if not isinstance(audio_path, str):
        return None
    normalized = audio_path.strip()
    if not normalized:
        return None
    return normalized


def _failure_result(status: str, warning: str) -> dict:
    return {"score": None, "details": {"status": status, "audio_warnings": [warning]}}


def _is_supported_pcm_width(sample_width: int) -> bool:
    return isinstance(sample_width, int) and not isinstance(sample_width, bool) and sample_width in ALLOWED_SAMPLE_WIDTHS


def _is_safe_sample_rate(sample_rate: int) -> bool:
    return (
        isinstance(sample_rate, int)
        and not isinstance(sample_rate, bool)
        and MIN_SOURCE_SAMPLE_RATE <= sample_rate <= MAX_SOURCE_SAMPLE_RATE
    )


def _is_safe_channel_count(channels: int) -> bool:
    return isinstance(channels, int) and not isinstance(channels, bool) and 1 <= channels <= MAX_SOURCE_CHANNELS


def _is_safe_frame_count(frame_count: int) -> bool:
    return isinstance(frame_count, int) and not isinstance(frame_count, bool) and frame_count >= 0


def _coerce_float32_array(values) -> np.ndarray:
    if values is None:
        raise ValueError("audio_array_missing")
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError("audio_array_invalid")
    arr = np.asarray(values)
    if arr.dtype.kind in {"b", "O", "U", "S", "V", "c"}:
        raise ValueError("audio_array_invalid")
    if arr.ndim != 1:
        raise ValueError("audio_array_invalid")
    if arr.size == 0:
        raise ValueError("audio_array_empty")
    arr = np.ascontiguousarray(arr.astype(np.float32, copy=False).reshape(-1))
    if arr.size > MAX_AUDIO_SAMPLES:
        raise ValueError("audio_array_too_long")
    if not np.isfinite(arr).all():
        raise ValueError("audio_array_nonfinite")
    return arr


def _ensure_1d_float32(values) -> np.ndarray:
    arr = _coerce_float32_array(values)
    if arr.dtype != np.float32:
        arr = np.ascontiguousarray(arr.astype(np.float32, copy=False))
    return arr


def _frame_matrix(samples: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    if isinstance(frame_length, bool) or not isinstance(frame_length, int) or frame_length <= 0:
        raise ValueError("frame_length_must_be_positive")
    if isinstance(hop_length, bool) or not isinstance(hop_length, int) or hop_length <= 0:
        raise ValueError("hop_length_must_be_positive")
    prepared = _ensure_1d_float32(samples)
    if prepared.size == 0:
        raise ValueError("audio_signal_empty")
    pad = frame_length // 2
    padded = np.pad(prepared, (pad, pad), mode="constant")
    if padded.size < frame_length:
        padded = np.pad(padded, (0, frame_length - padded.size), mode="constant")
    frames = np.lib.stride_tricks.sliding_window_view(padded, frame_length)[::hop_length]
    if frames.size == 0:
        frames = padded[-frame_length:][np.newaxis, :]
    return np.ascontiguousarray(frames)


def _safe_mean(values) -> float | None:
    if values is None:
        return None
    arr = np.asarray(values)
    if arr.size == 0:
        return None
    if arr.dtype.kind in {"b", "O", "U", "S", "V", "c"}:
        raise ValueError("invalid_numeric_array")
    numeric = arr.astype(np.float64, copy=False).reshape(-1)
    if not np.isfinite(numeric).all():
        raise ValueError("nonfinite_numeric_array")
    return float(np.mean(numeric))


def _finite_row_means(matrix) -> list[float | None]:
    arr = np.asarray(matrix)
    if arr.ndim != 2:
        raise ValueError("feature_matrix_must_be_2d")
    if arr.shape[0] != 5:
        raise ValueError("feature_matrix_shape_mismatch")
    if arr.dtype.kind in {"b", "O", "U", "S", "V", "c"}:
        raise ValueError("feature_matrix_invalid")
    numeric = arr.astype(np.float64, copy=False)
    if not np.isfinite(numeric).all():
        raise ValueError("feature_matrix_nonfinite")
    return [float(np.mean(row)) for row in numeric]


def _require_librosa_feature_matrix(name: str, matrix, *, expected_rows: int | None = None) -> np.ndarray:
    arr = np.asarray(matrix)
    if arr.ndim != 2:
        raise RuntimeError(f"{name}_feature_failed")
    if expected_rows is not None and arr.shape[0] != expected_rows:
        raise RuntimeError(f"{name}_feature_failed")
    if arr.size == 0:
        raise RuntimeError(f"{name}_feature_failed")
    if arr.dtype.kind in {"b", "O", "U", "S", "V", "c"}:
        raise RuntimeError(f"{name}_feature_failed")
    numeric = arr.astype(np.float64, copy=False)
    if not np.isfinite(numeric).all():
        raise RuntimeError(f"{name}_feature_failed")
    return arr


def _rms_numpy(y: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    if isinstance(frame_length, bool) or not isinstance(frame_length, int) or frame_length <= 0:
        raise ValueError("frame_length_must_be_positive")
    if isinstance(hop_length, bool) or not isinstance(hop_length, int) or hop_length <= 0:
        raise ValueError("hop_length_must_be_positive")
    samples = _ensure_1d_float32(y)
    if samples.size == 0:
        raise ValueError("audio_signal_empty")
    pad = frame_length // 2
    padded = np.pad(samples, (pad, pad), mode="constant")
    if padded.size < frame_length:
        padded = np.pad(padded, (0, frame_length - padded.size), mode="constant")
    windows = np.lib.stride_tricks.sliding_window_view(padded, frame_length)[::hop_length]
    if windows.size == 0:
        windows = padded[-frame_length:][np.newaxis, :]
    rms = np.sqrt(np.mean(windows * windows, axis=-1, dtype=np.float64))
    rms = np.ascontiguousarray(np.asarray(rms, dtype=np.float32).reshape(-1))
    if not np.isfinite(rms).all():
        raise ValueError("rms_nonfinite")
    return rms


def _zero_crossing_rate_numpy(y: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    if isinstance(frame_length, bool) or not isinstance(frame_length, int) or frame_length <= 0:
        raise ValueError("frame_length_must_be_positive")
    if isinstance(hop_length, bool) or not isinstance(hop_length, int) or hop_length <= 0:
        raise ValueError("hop_length_must_be_positive")
    samples = _ensure_1d_float32(y)
    if samples.size == 0:
        raise ValueError("audio_signal_empty")
    pad = frame_length // 2
    padded = np.pad(samples, (pad, pad), mode="constant")
    if padded.size < frame_length:
        padded = np.pad(padded, (0, frame_length - padded.size), mode="constant")
    windows = np.lib.stride_tricks.sliding_window_view(padded, frame_length)[::hop_length]
    if windows.size == 0:
        windows = padded[-frame_length:][np.newaxis, :]
    signs = np.signbit(windows)
    zcr = np.mean(signs[:, 1:] != signs[:, :-1], axis=-1, dtype=np.float64)
    zcr = np.ascontiguousarray(np.asarray(zcr, dtype=np.float32).reshape(-1))
    if not np.isfinite(zcr).all():
        raise ValueError("zcr_nonfinite")
    return zcr


def _frame_spectrum(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    samples = _ensure_1d_float32(y)
    if isinstance(sr, bool) or not isinstance(sr, int) or sr <= 0:
        raise ValueError("sample_rate_must_be_positive")
    window = np.hanning(samples.size).astype(np.float32, copy=False) if samples.size > 1 else np.ones(1, dtype=np.float32)
    weighted = samples * window
    spectrum = np.abs(np.fft.rfft(weighted))
    freqs = np.fft.rfftfreq(samples.size, d=1.0 / float(sr)) if samples.size > 1 else np.asarray([0.0], dtype=np.float64)
    spectrum = np.asarray(spectrum, dtype=np.float64).reshape(-1)
    freqs = np.asarray(freqs, dtype=np.float64).reshape(-1)
    return freqs, spectrum, weighted.astype(np.float32, copy=False)


def _spectral_centroid(y: np.ndarray, sr: int, *, hop_length: int, n_fft: int) -> float:
    if librosa is None:
        raise RuntimeError("spectral_centroid_feature_failed")
    try:
        matrix = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length, n_fft=n_fft)
        arr = _require_librosa_feature_matrix("spectral_centroid", matrix)
        return float(np.mean(arr, axis=1, dtype=np.float64)[0])
    except KeyboardInterrupt:
        raise
    except SystemExit:
        raise
    except Exception as exc:
        raise RuntimeError("spectral_centroid_feature_failed") from exc


def _spectral_flatness(y: np.ndarray, sr: int, *, hop_length: int, n_fft: int) -> float:
    if librosa is None:
        raise RuntimeError("spectral_flatness_feature_failed")
    try:
        matrix = librosa.feature.spectral_flatness(y=y, hop_length=hop_length, n_fft=n_fft)
        arr = _require_librosa_feature_matrix("spectral_flatness", matrix)
        return float(np.mean(arr, axis=1, dtype=np.float64)[0])
    except KeyboardInterrupt:
        raise
    except SystemExit:
        raise
    except Exception as exc:
        raise RuntimeError("spectral_flatness_feature_failed") from exc


def _mfcc_summary_like(y: np.ndarray, sr: int, *, hop_length: int, n_fft: int) -> list[float | None]:
    if librosa is None:
        raise RuntimeError("mfcc_feature_failed")
    try:
        matrix = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=5, hop_length=hop_length, n_fft=n_fft)
        arr = _require_librosa_feature_matrix("mfcc", matrix, expected_rows=5)
        return _finite_row_means(arr)
    except KeyboardInterrupt:
        raise
    except SystemExit:
        raise
    except Exception as exc:
        raise RuntimeError("mfcc_feature_failed") from exc


def _mfcc_summary_like_from_power_with_timings(power: np.ndarray, sr: int, *, n_fft: int, scan_id: str | None = None) -> tuple[list[float | None], int, int]:
    if librosa is None:
        raise RuntimeError("mfcc_feature_failed")
    try:
        power_matrix = np.asarray(power, dtype=np.float64)
        if power_matrix.ndim != 2:
            raise RuntimeError("mfcc_feature_failed")
        mel_started = time.perf_counter()
        mel_power = librosa.feature.melspectrogram(S=power_matrix, sr=sr, n_mels=128, n_fft=n_fft)
        mel_db = librosa.power_to_db(mel_power, ref=np.max)
        _log_audio_perf(scan_id, "audio_mel_ms", mel_started)
        mfcc_started = time.perf_counter()
        mfcc_matrix = librosa.feature.mfcc(S=mel_db, n_mfcc=5)
        _log_audio_perf(scan_id, "audio_mfcc_ms", mfcc_started)
        mfcc_transform_ms = _elapsed_ms(mfcc_started)
        arr = np.asarray(mfcc_matrix, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] != 5 or arr.size == 0:
            raise RuntimeError("mfcc_feature_failed")
        if not np.isfinite(arr).all():
            raise RuntimeError("mfcc_feature_failed")
        return [float(np.mean(row)) for row in arr], _elapsed_ms(mel_started), mfcc_transform_ms
    except KeyboardInterrupt:
        raise
    except SystemExit:
        raise
    except Exception as exc:
        raise RuntimeError("mfcc_feature_failed") from exc


def _mfcc_summary_like_from_power(power: np.ndarray, sr: int, *, n_fft: int, scan_id: str | None = None) -> list[float | None]:
    summary, _, _ = _mfcc_summary_like_from_power_with_timings(power, sr, n_fft=n_fft, scan_id=scan_id)
    return summary


def _resample_if_needed(y: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    if sr == TARGET_SAMPLE_RATE:
        return np.ascontiguousarray(y.astype(np.float32, copy=False).reshape(-1)), sr
    if y.size == 0:
        raise _AudioEmptyError("audio_signal_empty")
    if signal is not None and hasattr(signal, "resample_poly"):
        gcd = int(np.gcd(sr, TARGET_SAMPLE_RATE))
        up = TARGET_SAMPLE_RATE // gcd
        down = sr // gcd
        resampled = signal.resample_poly(y, up, down)
    else:
        target_count = int(round(y.size * TARGET_SAMPLE_RATE / float(sr)))
        if target_count <= 0:
            raise _AudioEmptyError("audio_signal_empty")
        source_x = np.linspace(0.0, 1.0, num=y.size, endpoint=True, dtype=np.float64)
        target_x = np.linspace(0.0, 1.0, num=target_count, endpoint=True, dtype=np.float64)
        resampled = np.interp(target_x, source_x, y.astype(np.float64, copy=False))
    resampled = np.ascontiguousarray(np.asarray(resampled, dtype=np.float32).reshape(-1))
    if resampled.size == 0:
        raise _AudioEmptyError("audio_signal_empty")
    if resampled.size > MAX_AUDIO_SAMPLES:
        raise ValueError("resampled_audio_too_long")
    if not np.isfinite(resampled).all():
        raise ValueError("resampled_audio_nonfinite")
    return resampled, TARGET_SAMPLE_RATE


def _pcm_bytes_to_float32(raw: bytes, sample_width: int) -> np.ndarray:
    if sample_width == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        return (data - 128.0) / 128.0
    if sample_width == 2:
        return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if sample_width == 3:
        data = np.frombuffer(raw, dtype=np.uint8)
        if data.size % 3 != 0:
            raise ValueError("invalid_wav_payload")
        bytes_view = data.reshape(-1, 3)
        values = (
            bytes_view[:, 0].astype(np.int32)
            | (bytes_view[:, 1].astype(np.int32) << 8)
            | (bytes_view[:, 2].astype(np.int32) << 16)
        )
        values = np.where(values & 0x800000, values - 0x1000000, values)
        return values.astype(np.float32) / 8388608.0
    if sample_width == 4:
        return np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    raise ValueError("unsupported_wav_sample_width")


def _decode_wav_slice(audio_path: str, *, scan_id: str | None = None) -> tuple[np.ndarray, int, float, float, str, int]:
    open_started = time.perf_counter()
    try:
        with wave.open(audio_path, "rb") as handle:
            source_sr = handle.getframerate()
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            frame_count = handle.getnframes()
            if not _is_safe_sample_rate(source_sr):
                raise ValueError("unsafe_sample_rate")
            if not _is_safe_channel_count(channels):
                raise ValueError("unsafe_channel_count")
            if not _is_supported_pcm_width(sample_width):
                raise ValueError("unsupported_wav_sample_width")
            if not _is_safe_frame_count(frame_count):
                raise ValueError("unsafe_frame_count")
            if frame_count == 0:
                raise _AudioEmptyError("empty_wav")
            source_duration_seconds = frame_count / float(source_sr)
            frames_to_read = min(frame_count, int(math.ceil(MAX_AUDIO_ANALYSIS_SEC * source_sr)))
            raw = handle.readframes(frames_to_read)
        _log_audio_perf(scan_id, "audio_open_ms", open_started)
    except _AudioEmptyError:
        raise
    except (wave.Error, EOFError, OSError) as exc:
        raise RuntimeError("audio_decode_failed") from exc

    if not raw:
        raise _AudioEmptyError("empty_wav")
    decode_started = time.perf_counter()
    decoded = _pcm_bytes_to_float32(raw, sample_width)
    _log_audio_perf(scan_id, "audio_decode_ms", decode_started)
    expected_samples = frames_to_read * channels
    if decoded.size != expected_samples:
        raise RuntimeError("audio_decode_failed")
    mono_started = time.perf_counter()
    if channels > 1:
        if decoded.size % channels != 0:
            raise RuntimeError("audio_decode_failed")
        decoded = decoded.reshape(-1, channels).mean(axis=1)
    _log_audio_perf(scan_id, "audio_mono_ms", mono_started)
    decoded = np.ascontiguousarray(decoded.astype(np.float32, copy=False).reshape(-1))
    if decoded.size == 0:
        raise _AudioEmptyError("empty_wav")
    if not np.isfinite(decoded).all():
        raise RuntimeError("audio_decode_failed")
    resample_started = time.perf_counter()
    decoded, sr = _resample_if_needed(decoded, source_sr)
    _log_audio_perf(scan_id, "audio_resample_ms", resample_started)
    analysis_duration_seconds = decoded.size / float(sr)
    return decoded, sr, source_duration_seconds, analysis_duration_seconds, "wave", frames_to_read


def _decode_with_ffmpeg(audio_path: str, *, scan_id: str | None = None) -> tuple[np.ndarray, int, float, float, str, int]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("audio_decode_failed")
    fd = None
    converted_path = None
    try:
        fd, converted_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        fd = None
        cmd = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            audio_path,
            "-t",
            str(MAX_AUDIO_ANALYSIS_SEC),
            "-ac",
            "1",
            "-ar",
            str(TARGET_SAMPLE_RATE),
            "-f",
            "wav",
            converted_path,
        ]
        conversion_started = time.perf_counter()
        try:
            subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
                timeout=FFMPEG_CONVERSION_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError("audio_decode_timeout") from exc
        _log_audio_perf(scan_id, "audio_conversion_ms", conversion_started)
        return _decode_wav_slice(converted_path, scan_id=scan_id)
    except _AudioEmptyError:
        raise
    except TimeoutError:
        raise
    except Exception as exc:
        raise RuntimeError("audio_decode_failed") from exc
    finally:
        if fd is not None:
            with suppress(OSError):
                os.close(fd)
        if converted_path:
            with suppress(OSError):
                os.remove(converted_path)


def _decode_audio_once(audio_path: str, *, scan_id: str | None = None) -> tuple[np.ndarray, int, float, float, str, int]:
    _, ext = os.path.splitext(audio_path)
    if ext.lower() in {".wav", ".wave"}:
        try:
            _log_audio_perf(scan_id, "audio_conversion_ms", time.perf_counter(), status="skipped")
            return _decode_wav_slice(audio_path, scan_id=scan_id)
        except _AudioEmptyError:
            raise
        except Exception:
            return _decode_with_ffmpeg(audio_path, scan_id=scan_id)
    return _decode_with_ffmpeg(audio_path, scan_id=scan_id)


def _prepare_audio_source(audio_path: str) -> tuple[str, str | None]:
    normalized = _normalize_audio_path(audio_path)
    if normalized is None:
        return "missing", None
    try:
        if not os.path.exists(normalized) or not os.path.isfile(normalized):
            return "load_failed", normalized
        if os.path.getsize(normalized) == 0:
            return "empty_audio", normalized
    except OSError:
        return "load_failed", normalized
    return "ok", normalized


def _quality_from_features(
    *,
    duration_seconds: float,
    rms_energy: float,
    silence_ratio: float,
    noise_estimate: float,
    peak_volume: float,
    clipping_ratio: float,
) -> float:
    duration_factor = clamp01(duration_seconds / MAX_AUDIO_ANALYSIS_SEC, 0.0) or 0.0
    level_factor = clamp01(rms_energy / max(MIN_RMS_ENERGY * 2.5, 1e-6), 0.0) or 0.0
    activity_factor = clamp01(1.0 - silence_ratio, 0.0) or 0.0
    noise_factor = clamp01(1.0 - noise_estimate, 0.0) or 0.0
    headroom_loss = max(0.0, peak_volume - MAX_REASONABLE_PEAK) / max(1.0 - MAX_REASONABLE_PEAK, 1e-6)
    headroom_factor = clamp01(1.0 - headroom_loss - min(clipping_ratio * 6.0, 0.6), 0.0) or 0.0
    quality = (
        0.24 * duration_factor
        + 0.22 * level_factor
        + 0.20 * activity_factor
        + 0.19 * noise_factor
        + 0.15 * headroom_factor
    )
    return float(np.clip(quality, 0.0, 1.0))


def _presence_from_features(
    *,
    rms_energy: float,
    silence_ratio: float,
    noise_estimate: float,
    centroid: float,
    zcr_mean: float,
    tonal_concentration: float,
    rms_variation: float,
) -> float:
    activity_factor = clamp01(1.0 - silence_ratio, 0.0) or 0.0
    level_factor = clamp01(rms_energy / max(MIN_RMS_ENERGY * 1.75, 1e-6), 0.0) or 0.0
    centroid_factor = clamp01(1.0 - abs(centroid - 1700.0) / 1700.0, 0.0) or 0.0
    silence_band = clamp01(1.0 - abs(silence_ratio - 0.45) / 0.45, 0.0) or 0.0
    tone_penalty = clamp01((tonal_concentration - 0.58) / 0.22, 0.0) or 0.0
    variation_gate = clamp01(rms_variation / 0.30, 0.0) or 0.0
    cleanliness = clamp01(1.0 - noise_estimate, 0.0) or 0.0
    structure = 0.30 * silence_band + 0.30 * cleanliness + 0.20 * centroid_factor + 0.20 * level_factor
    speech_gate = activity_factor * (0.35 + 0.65 * variation_gate) * (1.0 - min(tone_penalty, 0.95))
    score = speech_gate * structure
    return float(np.clip(score, 0.0, 1.0))


def _voice_clarity_from_features(
    *,
    speech_presence_score: float,
    rms_energy: float,
    noise_estimate: float,
) -> float:
    speech_gate = clamp01(speech_presence_score, 0.0) or 0.0
    activity_factor = clamp01(rms_energy / max(MIN_RMS_ENERGY * 1.5, 1e-6), 0.0) or 0.0
    cleanliness = clamp01(1.0 - noise_estimate, 0.0) or 0.0
    score = speech_gate * (0.35 + 0.25 * activity_factor + 0.40 * cleanliness)
    return float(np.clip(score, 0.0, 1.0))


def _audio_confidence_from_components(*, audio_quality_score: float, voice_clarity_score: float) -> float:
    return float(np.clip(0.45 * audio_quality_score + 0.55 * voice_clarity_score, 0.0, 1.0))


def _feature_pipeline(y: np.ndarray, sr: int, *, scan_id: str | None = None) -> dict:
    if librosa is None:
        raise RuntimeError("audio_decode_failed")
    frame_length = min(2048, max(512, int(sr * 0.032)))
    hop_length = max(256, int(frame_length / 4))

    frame_started = time.perf_counter()
    frames = _frame_matrix(y, frame_length, hop_length)
    _log_audio_perf(scan_id, "audio_frame_build_ms", frame_started)
    step_started = time.perf_counter()
    rms = np.sqrt(np.mean(frames * frames, axis=-1, dtype=np.float64))
    rms = np.ascontiguousarray(np.asarray(rms, dtype=np.float32).reshape(-1))
    if not np.isfinite(rms).all():
        raise ValueError("rms_nonfinite")
    zcr = np.mean(np.signbit(frames[:, 1:]) != np.signbit(frames[:, :-1]), axis=-1, dtype=np.float64)
    zcr = np.ascontiguousarray(np.asarray(zcr, dtype=np.float32).reshape(-1))
    if not np.isfinite(zcr).all():
        raise ValueError("zcr_nonfinite")
    rms_ms = _elapsed_ms(step_started)
    _log_audio_perf(scan_id, "audio_energy_ms", step_started)

    step_started = time.perf_counter()
    stft_started = time.perf_counter()
    stft_matrix = librosa.stft(
        y=y,
        n_fft=frame_length,
        hop_length=hop_length,
        win_length=frame_length,
        window="hann",
        center=True,
        pad_mode="constant",
    )
    stft_ms = _elapsed_ms(stft_started)
    _log_audio_perf(scan_id, "audio_stft_ms", stft_started)
    magnitude = np.asarray(np.abs(stft_matrix), dtype=np.float64)
    power = magnitude * magnitude
    centroid_started = time.perf_counter()
    centroid_matrix = librosa.feature.spectral_centroid(S=magnitude, sr=sr)
    centroid_arr = _require_librosa_feature_matrix("spectral_centroid", centroid_matrix)
    centroid = float(np.mean(centroid_arr, axis=1, dtype=np.float64)[0])
    centroid_ms = _elapsed_ms(centroid_started)

    flatness_started = time.perf_counter()
    flatness_matrix = librosa.feature.spectral_flatness(S=magnitude, power=2.0)
    flatness_arr = _require_librosa_feature_matrix("spectral_flatness", flatness_matrix)
    flatness = float(np.mean(flatness_arr, axis=1, dtype=np.float64)[0])
    flatness_ms = _elapsed_ms(flatness_started)

    mfcc_started = time.perf_counter()
    mfcc_summary, mel_ms, mfcc_transform_ms = _mfcc_summary_like_from_power_with_timings(power, sr, n_fft=frame_length, scan_id=scan_id)
    mfcc_ms = _elapsed_ms(mfcc_started)
    spectral_ms = _elapsed_ms(step_started)
    _log_audio_perf(scan_id, "audio_spectral_ms", step_started)
    _log_audio_perf(scan_id, "audio_pitch_ms", time.perf_counter(), status="skipped")

    derived_started = time.perf_counter()
    peak_volume = float(np.max(np.abs(y))) if y.size else 0.0
    clipping_ratio = float(np.mean(np.abs(y) >= CLIPPING_SAMPLE_THRESHOLD)) if y.size else 0.0
    rms_energy = float(np.mean(rms)) if rms.size else 0.0
    silence_ratio = float(np.mean(rms < max(MIN_RMS_ENERGY * 0.6, rms_energy * 0.35))) if rms.size else 1.0
    zcr_mean = float(np.mean(zcr)) if zcr.size else 0.0
    rms_variation = float(np.std(rms) / max(rms_energy, 1e-6)) if rms.size else 0.0
    full_spectrum = np.abs(np.fft.rfft(y * np.hanning(y.size).astype(np.float32, copy=False))) if y.size else np.asarray([], dtype=np.float32)
    dominant_concentration = float(np.max(full_spectrum) / max(float(np.sum(full_spectrum)), 1e-6)) if full_spectrum.size else 0.0
    tonal_concentration = float(np.clip(0.55 * dominant_concentration + 0.25 * (1.0 - flatness) + 0.20 * clamp01(1.0 - rms_variation / 1.5, 0.0), 0.0, 1.0))
    noise_estimate = float(np.clip(0.65 * flatness + 0.35 * zcr_mean, 0.0, 1.0))
    if rms_energy < MIN_RMS_ENERGY * 0.2 and silence_ratio > 0.95:
        noise_estimate_value = 0.0
    else:
        noise_estimate_value = noise_estimate
    derived_ms = _elapsed_ms(derived_started)
    speech_presence_score = _presence_from_features(
        rms_energy=rms_energy,
        silence_ratio=silence_ratio,
        noise_estimate=noise_estimate_value,
        centroid=centroid,
        zcr_mean=zcr_mean,
        tonal_concentration=tonal_concentration,
        rms_variation=rms_variation,
    )
    voice_clarity_score = _voice_clarity_from_features(
        speech_presence_score=speech_presence_score,
        rms_energy=rms_energy,
        noise_estimate=noise_estimate_value,
    )
    _log_audio_perf(scan_id, "audio_quality_features_ms", derived_started)
    return {
        "frame_length": frame_length,
        "hop_length": hop_length,
        "rms": rms,
        "zcr": zcr,
        "centroid": centroid,
        "flatness": flatness,
        "mfcc_summary": mfcc_summary,
        "peak_volume": peak_volume,
        "clipping_ratio": clipping_ratio,
        "rms_energy": rms_energy,
        "silence_ratio": silence_ratio,
        "noise_estimate": noise_estimate_value,
        "speech_presence_score": speech_presence_score,
        "voice_clarity_score": voice_clarity_score,
        "tonal_concentration": tonal_concentration,
        "rms_variation": rms_variation,
        "zcr_mean": zcr_mean,
        "timings_ms": {
            "rms_ms": rms_ms,
            "zcr_ms": rms_ms,
            "spectral_centroid_ms": centroid_ms,
            "spectral_flatness_ms": flatness_ms,
            "stft_ms": stft_ms,
            "mfcc_ms": mfcc_ms,
            "mel_ms": mel_ms,
            "mfcc_transform_ms": mfcc_transform_ms,
            "pitch_ms": 0,
            "derived_metrics_ms": derived_ms,
            "spectral_total_ms": spectral_ms,
        },
    }


def _speech_state_and_warnings(
    *,
    duration_seconds: float,
    rms_energy: float,
    noise_estimate: float,
    silence_ratio: float,
    speech_presence_score: float,
    clipping_ratio: float,
    tonal_concentration: float,
    rms_variation: float,
) -> tuple[list[str], str, bool, bool]:
    minimum_usable_energy = MIN_RMS_ENERGY * 0.75
    quiet_but_usable = bool(
        duration_seconds >= MIN_AUDIO_DURATION_SEC
        and MIN_RMS_ENERGY * 0.65 <= rms_energy <= MIN_RMS_ENERGY * 1.15
        and speech_presence_score >= 0.12
        and noise_estimate <= 0.68
        and silence_ratio <= 0.55
    )
    obvious_tone_like = bool(
        tonal_concentration >= 0.42
        and rms_variation <= 0.35
        and noise_estimate <= 0.35
        and silence_ratio <= 0.75
    )
    no_usable_speech = bool(
        obvious_tone_like
        or silence_ratio > 0.80
        or (speech_presence_score < 0.15 and rms_energy < minimum_usable_energy and noise_estimate < 0.55 and clipping_ratio <= MAX_CLIPPING_RATIO)
    )
    quiet_but_usable = quiet_but_usable and not no_usable_speech
    warnings: list[str] = []
    if duration_seconds < MIN_AUDIO_DURATION_SEC:
        warnings.append("audio_too_short")
    if rms_energy < minimum_usable_energy and not quiet_but_usable:
        warnings.append("audio_too_quiet")
    if noise_estimate > 0.72 and not no_usable_speech:
        warnings.append("audio_too_noisy")
    if silence_ratio > 0.55:
        warnings.append("too_much_silence")
    if no_usable_speech:
        warnings.append("speech_not_detected")
    if clipping_ratio > MAX_CLIPPING_RATIO:
        warnings.append("audio_clipping")
    if obvious_tone_like and "speech_not_detected" not in warnings:
        warnings.append("speech_not_detected")
    warnings = clean_warning_codes(warnings)
    speech_state = "usable_speech"
    if "audio_clipping" in warnings or "audio_too_noisy" in warnings or ("audio_too_quiet" in warnings and not quiet_but_usable and not no_usable_speech):
        speech_state = "unusable_quality"
    elif no_usable_speech:
        speech_state = "no_speech"
    elif quiet_but_usable:
        speech_state = "quiet_usable_speech"
    usable_speech_detected = speech_state in {"usable_speech", "quiet_usable_speech"}
    return warnings, speech_state, quiet_but_usable, usable_speech_detected


def _build_success_details(
    *,
    y: np.ndarray,
    sr: int,
    source_duration_seconds: float,
    analyzed_duration_seconds: float,
    scan_id: str | None = None,
) -> dict:
    prepared = _ensure_1d_float32(y)
    if prepared.size == 0:
        raise _AudioEmptyError("audio_signal_empty")
    if prepared.size > MAX_AUDIO_SAMPLES:
        raise RuntimeError("audio_decode_failed")
    quality_started = time.perf_counter()
    features = _feature_pipeline(prepared, sr, scan_id=scan_id)

    duration_seconds = float(prepared.size / float(sr))
    if not math.isfinite(duration_seconds) or duration_seconds < 0.0:
        raise RuntimeError("audio_decode_failed")
    voice_started = time.perf_counter()
    warnings, speech_state, quiet_but_usable, usable_speech_detected = _speech_state_and_warnings(
        duration_seconds=duration_seconds,
        rms_energy=features["rms_energy"],
        noise_estimate=features["noise_estimate"],
        silence_ratio=features["silence_ratio"],
        speech_presence_score=features["speech_presence_score"],
        clipping_ratio=features["clipping_ratio"],
        tonal_concentration=features["tonal_concentration"],
        rms_variation=features["rms_variation"],
    )
    _log_audio_perf(scan_id, "audio_vad_ms", voice_started)
    voice_activity_ms = _elapsed_ms(voice_started)
    pitch_stability_score = None
    duration_factor = clamp01(duration_seconds / MAX_AUDIO_ANALYSIS_SEC, 0.0) or 0.0
    level_factor = clamp01(features["rms_energy"] / max(MIN_RMS_ENERGY * 2.5, 1e-6), 0.0) or 0.0
    headroom_loss = max(0.0, features["peak_volume"] - MAX_REASONABLE_PEAK) / max(1.0 - MAX_REASONABLE_PEAK, 1e-6)
    headroom_factor = clamp01(1.0 - headroom_loss - min(features["clipping_ratio"] * 6.0, 0.6), 0.0) or 0.0
    noise_factor = clamp01(1.0 - features["noise_estimate"], 0.0) or 0.0
    activity_factor = clamp01(1.0 - features["silence_ratio"], 0.0) or 0.0
    audio_quality_score = float(
        np.clip(
            0.24 * duration_factor
            + 0.22 * level_factor
            + 0.20 * activity_factor
            + 0.19 * noise_factor
            + 0.15 * headroom_factor,
            0.0,
            1.0,
        )
    )
    voice_clarity_score = features["voice_clarity_score"]
    audio_confidence = _audio_confidence_from_components(
        audio_quality_score=audio_quality_score,
        voice_clarity_score=voice_clarity_score,
    )
    audio_quality_ms = _elapsed_ms(quality_started)
    silent = bool(features["silence_ratio"] > 0.80 or features["rms_energy"] < (MIN_RMS_ENERGY * 0.5))
    details = {
        "status": "ok",
        "duration_seconds": safe_number(duration_seconds, 3),
        "duration_sec": safe_number(duration_seconds, 3),
        "analyzed_duration_seconds": safe_number(analyzed_duration_seconds, 3),
        "analysis_sample_limit_seconds": safe_number(MAX_AUDIO_ANALYSIS_SEC, 3),
        "sample_rate": int(sr),
        "rms_energy": safe_number(features["rms_energy"], 6),
        "energy": safe_number(features["rms_energy"], 6),
        "peak_volume": safe_number(features["peak_volume"], 6),
        "clipping_ratio": safe_number(features["clipping_ratio"], 6),
        "silence_ratio": safe_number(features["silence_ratio"], 6),
        "noise_estimate": safe_number(features["noise_estimate"], 6),
        "speech_presence_score": safe_number(features["speech_presence_score"], 6),
        "speech_rate": None,
        "voice_clarity_score": safe_number(voice_clarity_score, 6),
        "pitch_stability_score": safe_number(pitch_stability_score, 6),
        "speech_state": speech_state,
        "usable_speech_detected": usable_speech_detected,
        "quiet_but_usable": quiet_but_usable,
        "spectral_centroid": safe_number(features["centroid"], 2),
        "centroid": safe_number(features["centroid"], 2),
        "spectral_flatness": safe_number(features["flatness"], 6),
        "zero_crossing_rate": safe_number(features["zcr_mean"], 6),
        "zcr": safe_number(features["zcr_mean"], 6),
        "mfcc_summary": [safe_number(value, 4) if value is not None else None for value in features["mfcc_summary"]],
        "audio_quality_score": safe_number(audio_quality_score, 6),
        "audio_confidence": safe_number(audio_confidence, 6),
        "audio_warnings": warnings,
        "silent": silent,
        "timings_ms": {
            "audio_decode_ms": None,
            "audio_quality_ms": audio_quality_ms,
            "voice_activity_ms": voice_activity_ms,
        },
        "audio_quality_timings_ms": features["timings_ms"],
    }
    logger.info(
        "[AUDIO_CONFIDENCE_COMPONENTS] audio_quality_score=%.6f voice_clarity_score=%.6f speech_presence_score=%.6f "
        "audio_confidence=%.6f speech_state=%s usable_speech_detected=%s quiet_but_usable=%s warning_count=%s",
        audio_quality_score,
        voice_clarity_score,
        features["speech_presence_score"],
        audio_confidence,
        speech_state,
        usable_speech_detected,
        quiet_but_usable,
        len(warnings),
    )
    _log_audio_perf(scan_id, "audio_result_build_ms", quality_started)
    return details


def analyze_audio(audio_path: str, *, scan_id: str | None = None) -> dict:
    total_started = time.perf_counter()
    if _normalize_audio_path(audio_path) is None:
        _log_audio_perf(scan_id, "audio_path_validation_ms", total_started)
        _log_audio_perf(scan_id, "audio_total_ms", total_started, status="missing")
        return _failure_result("missing", "audio_missing")

    validation_started = time.perf_counter()
    path_state, normalized_path = _prepare_audio_source(audio_path)
    _log_audio_perf(scan_id, "audio_path_validation_ms", validation_started, status=path_state)
    _log_audio_perf(scan_id, "audio_source_prepare_ms", validation_started, status=path_state)
    if path_state == "missing":
        _log_audio_perf(scan_id, "audio_total_ms", total_started, status="missing")
        return _failure_result("missing", "audio_missing")
    if path_state == "empty_audio":
        _log_audio_perf(scan_id, "audio_total_ms", total_started, status="empty_audio")
        return _failure_result("empty_audio", "audio_decode_failed")
    if path_state == "load_failed":
        _log_audio_perf(scan_id, "audio_total_ms", total_started, status="load_failed")
        return _failure_result("load_failed", "audio_decode_failed")

    if librosa is None:
        _log_audio_perf(scan_id, "audio_total_ms", total_started, status="load_failed")
        return _failure_result("load_failed", "audio_decode_failed")

    try:
        decode_started = time.perf_counter()
        y, sr, source_duration_seconds, analyzed_duration_seconds, decode_backend, decode_count = _decode_audio_once(normalized_path, scan_id=scan_id)
        decode_ms = _elapsed_ms(decode_started)
        logger.info(
            "[AUDIO_DECODE_DETAIL] backend=%s decode_count=%s source_duration_ms=%s analysis_duration_ms=%s",
            decode_backend,
            decode_count,
            int(round(source_duration_seconds * 1000)),
            int(round(analyzed_duration_seconds * 1000)),
        )
        details = _build_success_details(
            y=y,
            sr=sr,
            source_duration_seconds=source_duration_seconds,
            analyzed_duration_seconds=analyzed_duration_seconds,
            scan_id=scan_id,
        )
        details["timings_ms"]["audio_decode_ms"] = decode_ms
        _log_audio_perf(scan_id, "audio_total_ms", total_started)
        return {"score": details["audio_confidence"], "details": details}
    except TimeoutError:
        _log_audio_perf(scan_id, "audio_total_ms", total_started, status="timeout")
        return _failure_result("load_failed", "audio_decode_timeout")
    except _AudioEmptyError:
        _log_audio_perf(scan_id, "audio_total_ms", total_started, status="empty_audio")
        return _failure_result("empty_audio", "audio_decode_failed")
    except Exception:
        _log_audio_perf(scan_id, "audio_total_ms", total_started, status="load_failed")
        return _failure_result("load_failed", "audio_decode_failed")


def _load_whisper_model():
    global _WHISPER_MODEL, _WHISPER_MODEL_NAME
    if whisper is None:
        raise RuntimeError("whisper_not_installed")
    model_name = "tiny.en"
    with _WHISPER_LOCK:
        if _WHISPER_MODEL is not None and _WHISPER_MODEL_NAME == model_name:
            return _WHISPER_MODEL
        _WHISPER_MODEL = whisper.load_model(model_name)
        _WHISPER_MODEL_NAME = model_name
        return _WHISPER_MODEL


def transcribe_audio(audio_path: str) -> str:
    normalized_path = _normalize_audio_path(audio_path)
    if normalized_path is None:
        raise RuntimeError("audio_missing")
    path_state, _ = _prepare_audio_source(normalized_path)
    if path_state == "missing":
        raise RuntimeError("audio_missing")
    if path_state in {"empty_audio", "load_failed"}:
        raise RuntimeError("audio_decode_failed")

    try:
        _decode_audio_once(normalized_path)
    except (KeyboardInterrupt, SystemExit):
        raise
    except TimeoutError as exc:
        raise RuntimeError("audio_decode_failed") from exc
    except _AudioEmptyError as exc:
        raise RuntimeError("audio_decode_failed") from exc
    except Exception as exc:
        raise RuntimeError("audio_decode_failed") from exc

    with _WHISPER_LOCK:
        model = _load_whisper_model()
        try:
            result = model.transcribe(normalized_path, language="en", fp16=False)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            raise RuntimeError("transcription_failed") from exc

    if not isinstance(result, dict):
        raise RuntimeError("transcription_failed")
    text = result.get("text")
    if not isinstance(text, str):
        raise RuntimeError("transcription_failed")
    normalized_text = text.strip()
    if not normalized_text:
        raise RuntimeError("transcription_failed")
    return normalized_text


def analyze_audio_worker(audio_path, result_queue):
    try:
        result = analyze_audio(audio_path)
        payload = {"ok": True, "result": result}
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        payload = {"ok": False, "error": type(exc).__name__}
    try:
        result_queue.put(payload)
    except Exception:
        logger.warning("[AUDIO_WORKER_QUEUE_FAILED] payload_not_delivered")
