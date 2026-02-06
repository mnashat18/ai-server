#audio.py
import librosa
import numpy as np

from config import MIN_AUDIO_ENERGY


def analyze_audio(audio_path: str) -> dict:
    if not audio_path:
        return {"score": None, "details": {"status": "missing"}}
    try:
        y, sr = librosa.load(audio_path, sr=None)
    except Exception:
        return {"score": None, "details": {"status": "load_failed"}}

    if y is None or len(y) == 0:
        return {"score": None, "details": {"status": "empty_audio"}}

    energy = np.mean(librosa.feature.rms(y=y))
    if energy < MIN_AUDIO_ENERGY:
        return {
            "score": 0.3,
            "details": {"energy": float(energy), "silent": True},
        }

    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

    slur_score = min(centroid / 3000, 1)
    instability = min(zcr * 5, 1)

    voice_confidence = 1 - ((slur_score * 0.6) + (instability * 0.4))
    score = round(max(0, min(voice_confidence, 1)), 2)
    duration_sec = len(y) / float(sr) if sr else 0.0
    return {
        "score": score,
        "details": {
            "energy": float(energy),
            "zcr": float(zcr),
            "centroid": float(centroid),
            "duration_sec": round(duration_sec, 3),
            "silent": False,
        },
    }
