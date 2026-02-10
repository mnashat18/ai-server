#audio.py
import librosa
import numpy as np

from config import MIN_AUDIO_ENERGY

NEUTRAL_SCORE = 0.5


def _blend_with_neutral(score: float, quality_weight: float) -> float:
    weight = max(0.0, min(quality_weight, 1.0))
    return NEUTRAL_SCORE + (score - NEUTRAL_SCORE) * weight


def analyze_audio(audio_path: str) -> dict:
    if not audio_path:
        return {"score": None, "details": {"status": "missing"}}
    try:
        y, sr = librosa.load(audio_path, sr=None)
    except Exception:
        return {"score": None, "details": {"status": "load_failed"}}

    if y is None or len(y) == 0:
        return {"score": None, "details": {"status": "empty_audio"}}

    energy = float(np.mean(librosa.feature.rms(y=y)))
    quality_weight = 1.0
    quality_flag = None
    y_feat = y
    if MIN_AUDIO_ENERGY > 0 and energy < MIN_AUDIO_ENERGY:
        quality_weight = max(0.0, min(energy / MIN_AUDIO_ENERGY, 1.0))
        quality_flag = "low_energy"
        if energy > 0:
            y_feat = y / (energy + 1e-8) * MIN_AUDIO_ENERGY

    zcr = np.mean(librosa.feature.zero_crossing_rate(y_feat))
    centroid = np.mean(librosa.feature.spectral_centroid(y=y_feat, sr=sr))

    slur_score = min(centroid / 3000, 1)
    instability = min(zcr * 5, 1)

    voice_confidence = 1 - ((slur_score * 0.6) + (instability * 0.4))
    raw_score = max(0, min(voice_confidence, 1))
    if quality_flag:
        score = _blend_with_neutral(raw_score, quality_weight)
    else:
        score = raw_score
    score = round(score, 2)
    duration_sec = len(y) / float(sr) if sr else 0.0
    details = {
        "energy": float(energy),
        "zcr": float(zcr),
        "centroid": float(centroid),
        "duration_sec": round(duration_sec, 3),
        "silent": energy < MIN_AUDIO_ENERGY,
    }
    if quality_flag:
        details["quality_flag"] = quality_flag
        details["quality_weight"] = round(quality_weight, 3)
        details["raw_score"] = round(raw_score, 2)
    return {"score": score, "details": details}
