from __future__ import annotations

import math
import os
import threading
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from config import (
    LABELS,
    LABEL_SCORES,
    ML_MODEL_PATH,
    MODEL_VERSION,
    REQUIRE_LOCAL_MODEL,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _normalized_labels(value: Any, *, field_name: str) -> list[str]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a list of strings")

    labels: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"{field_name} must contain strings only")
        label = item.strip()
        if not label:
            raise ValueError(f"{field_name} cannot contain blank labels")
        labels.append(label)

    if not labels:
        raise ValueError(f"{field_name} cannot be empty")
    if len(set(labels)) != len(labels):
        raise ValueError(f"{field_name} cannot contain duplicate labels")
    return labels


def _finite_unit_value(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool) or type(value) not in {int, float}:
        raise ValueError(f"{field_name} must be a real number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{field_name} must be finite")
    if numeric < 0.0 or numeric > 1.0:
        raise ValueError(f"{field_name} must be in the range 0..1")
    return numeric


def _configured_label_score_map() -> dict[str, float]:
    labels = _normalized_labels(LABELS, field_name="LABELS")

    if isinstance(LABEL_SCORES, Mapping):
        score_map: dict[str, float] = {}
        for label in labels:
            if label not in LABEL_SCORES:
                raise ValueError(f"LABEL_SCORES is missing label {label!r}")
            score_map[label] = _finite_unit_value(
                LABEL_SCORES[label],
                field_name=f"LABEL_SCORES[{label!r}]",
            )

        unknown = set(LABEL_SCORES) - set(labels)
        if unknown:
            raise ValueError(
                "LABEL_SCORES contains unsupported labels: "
                + ", ".join(sorted(str(item) for item in unknown))
            )
        return score_map

    if not isinstance(LABEL_SCORES, Sequence) or isinstance(
        LABEL_SCORES,
        (str, bytes, bytearray),
    ):
        raise ValueError("LABEL_SCORES must be a sequence or label mapping")
    if len(LABEL_SCORES) != len(labels):
        raise ValueError("LABEL_SCORES length must match LABELS")

    return {
        label: _finite_unit_value(
            LABEL_SCORES[index],
            field_name=f"LABEL_SCORES[{index}]",
        )
        for index, label in enumerate(labels)
    }


def _safe_error_code(prefix: str, exc: BaseException) -> str:
    return f"{prefix}:{type(exc).__name__}"


class MLRuntime:
    def __init__(self, model_path: str | None = None):
        self.model_path = model_path or ML_MODEL_PATH
        self.require_local_model = bool(REQUIRE_LOCAL_MODEL)
        self.bundle = None
        self.loaded_at: str | None = None
        self.error: str | None = None
        self._lock = threading.RLock()

    def _clear_state(self, *, error: str | None) -> None:
        with self._lock:
            self.bundle = None
            self.loaded_at = None
            self.error = error

    def load(self) -> bool:
        raw_path = self.model_path
        if not isinstance(raw_path, (str, os.PathLike)) or not str(raw_path).strip():
            self._clear_state(
                error="model_path_missing" if self.require_local_model else None
            )
            return False

        model_path = Path(raw_path)
        if not model_path.exists():
            self._clear_state(
                error="model_path_missing" if self.require_local_model else None
            )
            return False
        if not model_path.is_file():
            self._clear_state(error="model_path_not_file")
            return False

        try:
            if model_path.stat().st_size <= 0:
                self._clear_state(error="model_file_empty")
                return False
        except OSError as exc:
            self._clear_state(error=_safe_error_code("model_stat_failed", exc))
            return False

        try:
            from ml.features import FEATURE_ORDER
            from ml.model import load_bundle

            bundle = load_bundle(str(model_path))

            expected_feature_order = list(FEATURE_ORDER)
            actual_feature_order = list(bundle.feature_order)
            if actual_feature_order != expected_feature_order:
                raise ValueError(
                    "model feature_order does not match the active feature schema"
                )

            configured_labels = _normalized_labels(
                LABELS,
                field_name="LABELS",
            )
            bundle_labels = _normalized_labels(
                bundle.label_names,
                field_name="bundle.label_names",
            )
            if set(bundle_labels) != set(configured_labels):
                raise ValueError(
                    "model labels do not match the configured runtime labels"
                )

            # Validate the configured readiness values before exposing the bundle.
            _configured_label_score_map()
        except Exception as exc:
            self._clear_state(error=_safe_error_code("model_load_failed", exc))
            return False

        with self._lock:
            self.bundle = bundle
            self.loaded_at = _utc_now_iso()
            self.error = None
        return True

    def is_loaded(self) -> bool:
        with self._lock:
            return self.bundle is not None

    def local_model_required(self) -> bool:
        return self.require_local_model

    def predict(self, features: list[float]) -> dict | None:
        with self._lock:
            bundle = self.bundle
            loaded_at = self.loaded_at

        if bundle is None:
            return None

        try:
            from ml.model import predict_proba

            probabilities = np.asarray(
                predict_proba(bundle, features),
                dtype=np.float64,
            )
            if probabilities.ndim != 1:
                raise ValueError("model probabilities must be one-dimensional")
            if probabilities.size != len(bundle.label_names):
                raise ValueError(
                    "model probability count does not match bundle labels"
                )
            if probabilities.size == 0:
                raise ValueError("model probabilities cannot be empty")
            if not np.all(np.isfinite(probabilities)):
                raise ValueError("model probabilities contain NaN or Infinity")
            if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
                raise ValueError("model probabilities must be in the range 0..1")

            total = float(np.sum(probabilities, dtype=np.float64))
            if not math.isclose(total, 1.0, rel_tol=1e-5, abs_tol=1e-6):
                raise ValueError("model probabilities do not sum to one")

            label_names = _normalized_labels(
                bundle.label_names,
                field_name="bundle.label_names",
            )
            score_map = _configured_label_score_map()

            unsupported = [label for label in label_names if label not in score_map]
            if unsupported:
                raise ValueError(
                    "model returned unsupported labels: "
                    + ", ".join(unsupported)
                )

            index = int(np.argmax(probabilities))
            label = label_names[index]

            readiness_values = np.asarray(
                [score_map[item] for item in label_names],
                dtype=np.float64,
            )
            readiness_confidence = float(
                np.dot(probabilities, readiness_values)
            )
            if not math.isfinite(readiness_confidence):
                raise ValueError("model readiness confidence is not finite")
            readiness_confidence = min(max(readiness_confidence, 0.0), 1.0)

            result = {
                "label": label,
                # Preserve the established public meaning used by scoring.py:
                # probability-weighted readiness value, not max class probability.
                "confidence": round(readiness_confidence, 3),
                "probs": [
                    round(float(probability), 4)
                    for probability in probabilities
                ],
                "model_path": str(self.model_path),
                "model_version": MODEL_VERSION,
                "loaded_at": loaded_at,
            }
        except Exception as exc:
            with self._lock:
                self.error = _safe_error_code("prediction_failed", exc)
            return None

        with self._lock:
            self.error = None
        return result