import numpy as np

from utils import clamp01, clean_warning_codes, safe_number

try:
    import librosa
except Exception:  # pragma: no cover
    librosa = None

try:
    import whisper
except Exception:  # pragma: no cover
    whisper = None

MIN_AUDIO_DURATION_SEC = 1.5
MIN_RMS_ENERGY = 0.012
MAX_REASONABLE_PEAK = 0.98
_WHISPER_MODEL = None
_WHISPER_MODEL_NAME = None


def _safe_mean(values) -> float:
    return float(np.mean(values)) if values is not None and len(values) else 0.0


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
        y, sr = librosa.load(audio_path, sr=None, mono=True)
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

    duration_seconds = len(y) / float(sr) if sr else 0.0
    frame_length = min(2048, max(512, int(sr * 0.032)))
    hop_length = max(256, int(frame_length / 4))
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop_length)[0]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=5, hop_length=hop_length)
    peak_volume = float(np.max(np.abs(y))) if len(y) else 0.0
    rms_energy = _safe_mean(rms)
    silence_ratio = float(np.mean(rms < max(MIN_RMS_ENERGY * 0.6, rms_energy * 0.35))) if len(rms) else 1.0
    noise_estimate = float(np.clip((_safe_mean(flatness) * 0.75) + (silence_ratio * 0.25), 0.0, 1.0))
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
    try:
        f0 = librosa.yin(y, fmin=75, fmax=400, sr=sr, frame_length=frame_length, hop_length=hop_length)
        voiced = f0[np.isfinite(f0)]
        if len(voiced):
            pitch_cv = float(np.std(voiced) / (np.mean(voiced) + 1e-6))
            pitch_stability_score = float(np.clip(1.0 - min(pitch_cv, 1.0), 0.0, 1.0))
    except Exception:
        pitch_stability_score = None

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

    speech_state = "usable_speech"
    if no_usable_speech:
        speech_state = "no_speech"
    elif "audio_too_noisy" in warnings or ("audio_too_quiet" in warnings and not quiet_but_usable):
        speech_state = "unusable_quality"
    elif quiet_but_usable:
        speech_state = "quiet_usable_speech"

    duration_factor = clamp01(duration_seconds / 4.0, 0.0) or 0.0
    level_factor = clamp01(rms_energy / (MIN_RMS_ENERGY * 2.5), 0.0) or 0.0
    headroom_factor = 1.0 - max(0.0, peak_volume - MAX_REASONABLE_PEAK)
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
        "sample_rate": int(sr),
        "rms_energy": safe_number(rms_energy, 6),
        "energy": safe_number(rms_energy, 6),
        "peak_volume": safe_number(peak_volume, 6),
        "silence_ratio": safe_number(silence_ratio, 4),
        "noise_estimate": safe_number(noise_estimate, 4),
        "speech_presence_score": safe_number(speech_presence_score, 4),
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
    }
    return {"score": details["audio_confidence"], "details": details}
