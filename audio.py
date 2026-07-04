import time
import wave

import numpy as np

from logger import get_logger
from utils import clamp01, clean_warning_codes, safe_number

try:
    import librosa
except Exception:  # pragma: no cover
    librosa = None

try:
    import whisper
except Exception:  # pragma: no cover
    whisper = None

try:
    from scipy import signal
except Exception:  # pragma: no cover
    signal = None

MIN_AUDIO_DURATION_SEC = 1.5
MIN_RMS_ENERGY = 0.012
MAX_REASONABLE_PEAK = 0.98
TARGET_SAMPLE_RATE = 16000
MAX_AUDIO_ANALYSIS_SEC = 6.0
CLIPPING_SAMPLE_THRESHOLD = 0.98
MAX_CLIPPING_RATIO = 0.015
_WHISPER_MODEL = None
_WHISPER_MODEL_NAME = None
logger = get_logger()


def _safe_mean(values) -> float:
    return float(np.mean(values)) if values is not None and len(values) else 0.0


def _elapsed_ms(started_at: float) -> int:
    return int(round((time.perf_counter() - started_at) * 1000))


def _pcm_bytes_to_float32(raw: bytes, sample_width: int) -> np.ndarray:
    if sample_width == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        return (data - 128.0) / 128.0
    if sample_width == 2:
        return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if sample_width == 3:
        bytes_view = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
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


def _resample_if_needed(y: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    if sr == TARGET_SAMPLE_RATE:
        return y.astype(np.float32, copy=False), sr
    if signal is None:
        return y.astype(np.float32, copy=False), sr
    gcd = int(np.gcd(sr, TARGET_SAMPLE_RATE))
    resampled = signal.resample_poly(y, TARGET_SAMPLE_RATE // gcd, sr // gcd)
    return resampled.astype(np.float32, copy=False), TARGET_SAMPLE_RATE


def _decode_wav_slice(audio_path: str) -> tuple[np.ndarray, int, float, float, str, int]:
    with wave.open(audio_path, "rb") as handle:
        source_sr = int(handle.getframerate())
        channels = int(handle.getnchannels())
        sample_width = int(handle.getsampwidth())
        frame_count = int(handle.getnframes())
        source_duration_seconds = frame_count / float(source_sr) if source_sr else 0.0
        frames_to_read = min(frame_count, int(round(MAX_AUDIO_ANALYSIS_SEC * source_sr)))
        raw = handle.readframes(frames_to_read)

    if not raw or source_sr <= 0 or channels <= 0:
        return np.array([], dtype=np.float32), source_sr, source_duration_seconds, 0.0, "wave", 1

    y = _pcm_bytes_to_float32(raw, sample_width)
    if channels > 1:
        y = y.reshape(-1, channels).mean(axis=1)
    y, sr = _resample_if_needed(y, source_sr)
    analysis_duration_seconds = len(y) / float(sr) if sr else 0.0
    return y, sr, source_duration_seconds, analysis_duration_seconds, "wave", 1


def _decode_audio_once(audio_path: str) -> tuple[np.ndarray, int, float, float, str, int]:
    try:
        return _decode_wav_slice(audio_path)
    except Exception:
        if librosa is None:
            raise
        y, sr = librosa.load(audio_path, sr=TARGET_SAMPLE_RATE, mono=True, duration=MAX_AUDIO_ANALYSIS_SEC)
        duration_seconds = len(y) / float(sr) if sr else 0.0
        return y, sr, duration_seconds, duration_seconds, "librosa", 1


def _load_whisper_model():
    global _WHISPER_MODEL, _WHISPER_MODEL_NAME
    if whisper is None:
        raise RuntimeError("whisper_not_installed")
    model_name = "tiny.en"
    if _WHISPER_MODEL is not None and _WHISPER_MODEL_NAME == model_name:
        return _WHISPER_MODEL
    _WHISPER_MODEL = whisper.load_model(model_name)
    _WHISPER_MODEL_NAME = model_name
    return _WHISPER_MODEL


def transcribe_audio(audio_path: str) -> str:
    if not audio_path:
        raise RuntimeError("audio_missing")
    model = _load_whisper_model()
    result = model.transcribe(audio_path, language="en", fp16=False)
    text = (result or {}).get("text")
    if not text:
        raise RuntimeError("transcription_failed")
    return str(text).strip()


def analyze_audio(audio_path: str) -> dict:
    timings_ms: dict[str, int] = {"audio_decode_ms": 0, "audio_quality_ms": 0, "voice_activity_ms": 0}
    if not audio_path:
        return {
            "score": None,
            "details": {
                "status": "missing",
                "audio_warnings": ["audio_missing"],
            },
        }

    if librosa is None:
        return {
            "score": None,
            "details": {
                "status": "load_failed",
                "audio_warnings": ["audio_decode_failed"],
            },
        }

    try:
        decode_started = time.perf_counter()
        y, sr, source_duration_seconds, analyzed_duration_seconds, decode_backend, decode_count = _decode_audio_once(audio_path)
        timings_ms["audio_decode_ms"] = int(round((time.perf_counter() - decode_started) * 1000))
        logger.info(
            "[AUDIO_DECODE_DETAIL] backend=%s decode_count=%s source_duration_ms=%s analysis_duration_ms=%s",
            decode_backend,
            decode_count,
            int(round(source_duration_seconds * 1000)),
            int(round(analyzed_duration_seconds * 1000)),
        )
    except Exception:
        return {
            "score": None,
            "details": {
                "status": "load_failed",
                "audio_warnings": ["audio_decode_failed"],
            },
        }

    if y is None or len(y) == 0:
        return {
            "score": None,
            "details": {
                "status": "empty_audio",
                "audio_warnings": ["audio_decode_failed"],
            },
        }

    quality_started = time.perf_counter()
    quality_steps_ms: dict[str, int] = {}
    duration_seconds = source_duration_seconds
    frame_length = min(2048, max(512, int(sr * 0.032)))
    hop_length = max(256, int(frame_length / 4))
    step_started = time.perf_counter()
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    quality_steps_ms["rms_ms"] = _elapsed_ms(step_started)
    step_started = time.perf_counter()
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length)[0]
    quality_steps_ms["zcr_ms"] = _elapsed_ms(step_started)
    step_started = time.perf_counter()
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    quality_steps_ms["spectral_centroid_ms"] = _elapsed_ms(step_started)
    step_started = time.perf_counter()
    flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop_length)[0]
    quality_steps_ms["spectral_flatness_ms"] = _elapsed_ms(step_started)
    step_started = time.perf_counter()
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=5, hop_length=hop_length)
    quality_steps_ms["mfcc_ms"] = _elapsed_ms(step_started)
    step_started = time.perf_counter()
    peak_volume = float(np.max(np.abs(y))) if len(y) else 0.0
    clipping_ratio = float(np.mean(np.abs(y) >= CLIPPING_SAMPLE_THRESHOLD)) if len(y) else 0.0
    rms_energy = _safe_mean(rms)
    silence_ratio = float(np.mean(rms < max(MIN_RMS_ENERGY * 0.6, rms_energy * 0.35))) if len(rms) else 1.0
    noise_estimate = float(np.clip((_safe_mean(flatness) * 0.75) + (silence_ratio * 0.25), 0.0, 1.0))
    quality_steps_ms["derived_metrics_ms"] = _elapsed_ms(step_started)
    timings_ms["audio_quality_ms"] = int(round((time.perf_counter() - quality_started) * 1000))
    logger.info(
        "[AUDIO_QUALITY_DETAIL] rms_ms=%s zcr_ms=%s spectral_centroid_ms=%s spectral_flatness_ms=%s mfcc_ms=%s derived_metrics_ms=%s",
        quality_steps_ms["rms_ms"],
        quality_steps_ms["zcr_ms"],
        quality_steps_ms["spectral_centroid_ms"],
        quality_steps_ms["spectral_flatness_ms"],
        quality_steps_ms["mfcc_ms"],
        quality_steps_ms["derived_metrics_ms"],
    )

    voice_started = time.perf_counter()
    speech_presence_score = float(
        np.clip(
            0.5 * (1.0 - silence_ratio)
            + 0.2 * min(rms_energy / max(MIN_RMS_ENERGY, 1e-6), 1.0)
            + 0.2 * (1.0 - min(noise_estimate, 1.0))
            + 0.1 * (1.0 - min(abs(_safe_mean(centroid) - 1700.0) / 1700.0, 1.0)),
            0.0,
            1.0,
        )
    )
    speech_frames = int(np.sum(rms >= max(MIN_RMS_ENERGY * 0.6, rms_energy * 0.35))) if len(rms) else 0
    speech_rate = float(speech_frames / max(analyzed_duration_seconds, 1e-6)) if analyzed_duration_seconds else None
    minimum_usable_energy = MIN_RMS_ENERGY * 0.75
    quiet_but_usable = bool(
        rms_energy < MIN_RMS_ENERGY
        and rms_energy >= minimum_usable_energy
        and duration_seconds >= MIN_AUDIO_DURATION_SEC
        and speech_presence_score >= 0.58
        and noise_estimate <= 0.68
        and silence_ratio <= 0.45
    )
    no_usable_speech = bool(
        speech_presence_score < 0.3
        or silence_ratio > 0.8
        or (rms_energy < minimum_usable_energy and silence_ratio > 0.6)
    )

    pitch_stability_score = None
    timings_ms["voice_activity_ms"] = int(round((time.perf_counter() - voice_started) * 1000))

    voice_clarity_score = float(
        np.clip(
            0.45 * speech_presence_score
            + 0.25 * min(rms_energy / max(MIN_RMS_ENERGY * 1.5, 1e-6), 1.0)
            + 0.2 * (1.0 - noise_estimate)
            + 0.1 * (pitch_stability_score if pitch_stability_score is not None else 0.5),
            0.0,
            1.0,
        )
    )

    warnings: list[str] = []
    if duration_seconds < MIN_AUDIO_DURATION_SEC:
        warnings.append("audio_too_short")
    if rms_energy < minimum_usable_energy and not quiet_but_usable:
        warnings.append("audio_too_quiet")
    if noise_estimate > 0.72:
        warnings.append("audio_too_noisy")
    if silence_ratio > 0.55:
        warnings.append("too_much_silence")
    if no_usable_speech:
        warnings.append("speech_not_detected")
    if clipping_ratio > MAX_CLIPPING_RATIO:
        warnings.append("audio_clipping")

    speech_state = "usable_speech"
    if no_usable_speech:
        speech_state = "no_speech"
    elif "audio_clipping" in warnings or "audio_too_noisy" in warnings or ("audio_too_quiet" in warnings and not quiet_but_usable):
        speech_state = "unusable_quality"
    elif quiet_but_usable:
        speech_state = "quiet_usable_speech"

    duration_factor = clamp01(duration_seconds / 4.0, 0.0) or 0.0
    level_factor = clamp01(rms_energy / (MIN_RMS_ENERGY * 2.5), 0.0) or 0.0
    headroom_factor = max(0.0, 1.0 - max(0.0, peak_volume - MAX_REASONABLE_PEAK) - min(clipping_ratio * 12.0, 0.4))
    audio_quality_score = float(
        np.clip(
            0.25 * duration_factor
            + 0.2 * level_factor
            + 0.2 * (1.0 - silence_ratio)
            + 0.2 * (1.0 - noise_estimate)
            + 0.15 * headroom_factor,
            0.0,
            1.0,
        )
    )
    audio_confidence = float(
        np.clip(
            0.55 * audio_quality_score + 0.45 * voice_clarity_score,
            0.0,
            1.0,
        )
    )

    details = {
        "status": "ok",
        "duration_seconds": safe_number(duration_seconds, 3),
        "duration_sec": safe_number(duration_seconds, 3),
        "analyzed_duration_seconds": safe_number(analyzed_duration_seconds, 3),
        "analysis_sample_limit_seconds": safe_number(MAX_AUDIO_ANALYSIS_SEC, 3),
        "sample_rate": int(sr),
        "rms_energy": safe_number(rms_energy, 6),
        "energy": safe_number(rms_energy, 6),
        "peak_volume": safe_number(peak_volume, 6),
        "clipping_ratio": safe_number(clipping_ratio, 6),
        "silence_ratio": safe_number(silence_ratio, 4),
        "noise_estimate": safe_number(noise_estimate, 4),
        "speech_presence_score": safe_number(speech_presence_score, 4),
        "speech_rate": safe_number(speech_rate, 4),
        "voice_clarity_score": safe_number(voice_clarity_score, 4),
        "pitch_stability_score": safe_number(pitch_stability_score, 4),
        "speech_state": speech_state,
        "usable_speech_detected": speech_state in {"usable_speech", "quiet_usable_speech"},
        "quiet_but_usable": quiet_but_usable,
        "spectral_centroid": safe_number(_safe_mean(centroid), 2),
        "centroid": safe_number(_safe_mean(centroid), 2),
        "zero_crossing_rate": safe_number(_safe_mean(zcr), 6),
        "zcr": safe_number(_safe_mean(zcr), 6),
        "mfcc_summary": [safe_number(v, 4) for v in np.mean(mfcc, axis=1).tolist()],
        "audio_quality_score": safe_number(audio_quality_score, 4),
        "audio_confidence": safe_number(audio_confidence, 4),
        "audio_warnings": clean_warning_codes(warnings),
        "silent": silence_ratio > 0.8 or rms_energy < (MIN_RMS_ENERGY * 0.5),
        "timings_ms": timings_ms,
        "audio_quality_timings_ms": quality_steps_ms,
    }
    return {"score": details["audio_confidence"], "details": details}
