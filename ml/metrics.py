from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


__all__ = ["accuracy", "macro_f1"]


def _positive_class_count(value: Any) -> int:
    """Validate the declared number of classes."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError("num_classes must be a positive integer")

    num_classes = int(value)
    if num_classes <= 0:
        raise ValueError("num_classes must be greater than zero")
    return num_classes


def _label_array(values: Any, *, field_name: str) -> np.ndarray:
    """Return a one-dimensional int64 class-label array.

    Accepted labels are real integer values (Python/NumPy integers). Finite
    integer-valued floats are accepted for compatibility with datasets that
    deserialize class indices as floating-point values. Booleans, strings,
    NaN, Infinity, non-integral floats, nested arrays, and arbitrary objects
    are rejected.
    """
    if values is None:
        raise TypeError(f"{field_name} cannot be None")
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError(f"{field_name} must be a one-dimensional label sequence")

    try:
        array = np.asarray(values)
    except Exception as exc:
        raise TypeError(
            f"{field_name} must be convertible to a NumPy array"
        ) from exc

    if array.ndim != 1:
        raise ValueError(f"{field_name} must be one-dimensional")

    if array.size == 0:
        return np.empty(0, dtype=np.int64)

    if np.issubdtype(array.dtype, np.bool_):
        raise TypeError(f"{field_name} cannot contain boolean labels")

    if np.issubdtype(array.dtype, np.integer):
        return array.astype(np.int64, copy=True)

    if np.issubdtype(array.dtype, np.floating):
        numeric = array.astype(np.float64, copy=False)
        if not np.all(np.isfinite(numeric)):
            raise ValueError(f"{field_name} contains NaN or Infinity")
        if not np.all(numeric == np.floor(numeric)):
            raise ValueError(f"{field_name} contains non-integral labels")
        return numeric.astype(np.int64)

    # Object arrays need element-by-element validation so numeric strings,
    # booleans, and arbitrary objects cannot be treated as valid labels.
    if array.dtype == object:
        normalized: list[int] = []
        for index, value in enumerate(array.tolist()):
            if isinstance(value, (bool, np.bool_)):
                raise TypeError(
                    f"{field_name}[{index}] cannot be a boolean label"
                )
            if isinstance(value, (int, np.integer)):
                normalized.append(int(value))
                continue
            if isinstance(value, (float, np.floating)):
                numeric = float(value)
                if not np.isfinite(numeric):
                    raise ValueError(
                        f"{field_name}[{index}] contains NaN or Infinity"
                    )
                if not numeric.is_integer():
                    raise ValueError(
                        f"{field_name}[{index}] is not an integral label"
                    )
                normalized.append(int(numeric))
                continue
            raise TypeError(
                f"{field_name}[{index}] must be an integer class label"
            )
        return np.asarray(normalized, dtype=np.int64)

    raise TypeError(f"{field_name} must contain numeric class labels")


def _paired_labels(y_true: Any, y_pred: Any) -> tuple[np.ndarray, np.ndarray]:
    true_labels = _label_array(y_true, field_name="y_true")
    predicted_labels = _label_array(y_pred, field_name="y_pred")

    if true_labels.shape[0] != predicted_labels.shape[0]:
        raise ValueError(
            "y_true and y_pred must contain the same number of labels"
        )

    return true_labels, predicted_labels


def accuracy(y_true: Any, y_pred: Any) -> float:
    """Return exact classification accuracy.

    Empty, equally sized inputs return 0.0. Mismatched lengths and malformed
    labels fail explicitly instead of relying on NumPy broadcasting.
    """
    true_labels, predicted_labels = _paired_labels(y_true, y_pred)

    if true_labels.size == 0:
        return 0.0

    correct = np.count_nonzero(true_labels == predicted_labels)
    result = correct / int(true_labels.size)
    return float(result)


def macro_f1(y_true: Any, y_pred: Any, num_classes: int) -> float:
    """Return unweighted macro F1 across class indices 0..num_classes-1.

    Classes absent from both y_true and y_pred contribute 0.0, preserving the
    previous public behavior. Labels outside the declared class range are
    rejected so unsupported classes cannot be silently ignored.
    """
    class_count = _positive_class_count(num_classes)
    true_labels, predicted_labels = _paired_labels(y_true, y_pred)

    if true_labels.size == 0:
        return 0.0

    if np.any(true_labels < 0) or np.any(true_labels >= class_count):
        raise ValueError("y_true contains labels outside the declared class range")
    if np.any(predicted_labels < 0) or np.any(predicted_labels >= class_count):
        raise ValueError("y_pred contains labels outside the declared class range")

    f1_scores = np.zeros(class_count, dtype=np.float64)

    for class_index in range(class_count):
        true_is_class = true_labels == class_index
        predicted_is_class = predicted_labels == class_index

        true_positive = int(np.count_nonzero(true_is_class & predicted_is_class))
        false_positive = int(
            np.count_nonzero((~true_is_class) & predicted_is_class)
        )
        false_negative = int(
            np.count_nonzero(true_is_class & (~predicted_is_class))
        )

        denominator = (2 * true_positive) + false_positive + false_negative
        if denominator > 0:
            f1_scores[class_index] = (2.0 * true_positive) / denominator

    result = float(np.mean(f1_scores, dtype=np.float64))
    if not np.isfinite(result):
        raise RuntimeError("macro F1 calculation produced a non-finite result")
    return result