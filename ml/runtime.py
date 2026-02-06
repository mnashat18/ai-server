import os
from datetime import datetime

import numpy as np

from config import LABELS, LABEL_SCORES, ML_MODEL_PATH, MODEL_VERSION
from ml.model import load_bundle, predict_proba


class MLRuntime:
    def __init__(self, model_path: str | None = None):
        self.model_path = model_path or ML_MODEL_PATH
        self.bundle = None
        self.loaded_at = None
        self.error = None

    def load(self) -> bool:
        if not self.model_path or not os.path.exists(self.model_path):
            self.error = "model_path_missing"
            return False
        try:
            self.bundle = load_bundle(self.model_path)
            self.loaded_at = datetime.utcnow().isoformat() + "Z"
            self.error = None
            return True
        except Exception as exc:
            self.bundle = None
            self.error = str(exc)
            return False

    def is_loaded(self) -> bool:
        return self.bundle is not None

    def predict(self, features: list[float]) -> dict | None:
        if not self.bundle:
            return None
        probs = predict_proba(self.bundle, features)
        idx = int(np.argmax(probs))
        label_names = self.bundle.label_names or LABELS
        label = label_names[idx] if idx < len(label_names) else str(idx)
        scores = LABEL_SCORES if len(LABEL_SCORES) == len(probs) else [0.25] * len(probs)
        confidence = float(np.sum(probs * np.array(scores)))
        return {
            "label": label,
            "confidence": round(confidence, 3),
            "probs": [round(float(p), 4) for p in probs],
            "model_path": self.model_path,
            "model_version": MODEL_VERSION,
            "loaded_at": self.loaded_at,
        }
