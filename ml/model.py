from __future__ import annotations

import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


_BUNDLE_FORMAT_VERSION = 2
_SCALER_STD_EPSILON = 1e-8


def _strict_positive_int(value: Any, *, field_name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _validate_string_list(
    value: Any,
    *,
    field_name: str,
    allow_empty: bool = False,
) -> list[str]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a list of strings")

    normalized: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"{field_name} must contain strings only")
        text = item.strip()
        if not text:
            raise ValueError(f"{field_name} cannot contain blank values")
        normalized.append(text)

    if not allow_empty and not normalized:
        raise ValueError(f"{field_name} cannot be empty")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} cannot contain duplicates")
    return normalized


def _plain_metadata(value: Any, *, path: str = "metadata") -> Any:
    """Normalize metadata to simple, finite, weights-only-safe Python values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return value

    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{path} keys must be non-empty strings")
            result[key] = _plain_metadata(item, path=f"{path}.{key}")
        return result

    if isinstance(value, (list, tuple)):
        return [
            _plain_metadata(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]

    raise ValueError(
        f"{path} contains unsupported value type: {type(value).__name__}"
    )


def _numeric_array(
    value: Any,
    *,
    field_name: str,
    allowed_ndim: tuple[int, ...],
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{field_name} must be a NumPy array")
    if value.ndim not in allowed_ndim:
        expected = " or ".join(str(item) for item in allowed_ndim)
        raise ValueError(f"{field_name} must have {expected} dimensions")
    if value.size == 0:
        raise ValueError(f"{field_name} cannot be empty")
    if np.issubdtype(value.dtype, np.bool_) or not np.issubdtype(
        value.dtype,
        np.number,
    ):
        raise TypeError(f"{field_name} must contain numeric non-boolean values")

    array = np.asarray(value, dtype=np.float32)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{field_name} contains NaN or Infinity")
    return array


class StandardScaler:
    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    @property
    def is_fitted(self) -> bool:
        return self.mean is not None and self.std is not None

    @property
    def feature_count(self) -> int | None:
        if not self.is_fitted:
            return None
        assert self.mean is not None
        return int(self.mean.shape[0])

    def fit(self, x: np.ndarray) -> None:
        array = _numeric_array(
            x,
            field_name="x",
            allowed_ndim=(2,),
        )
        if array.shape[0] < 1 or array.shape[1] < 1:
            raise ValueError("x must contain at least one row and one feature")

        mean = np.mean(array, axis=0, dtype=np.float64)
        std = np.std(array, axis=0, dtype=np.float64)

        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
            raise ValueError("scaler statistics contain NaN or Infinity")

        std = np.where(np.abs(std) < _SCALER_STD_EPSILON, 1.0, std)

        self.mean = mean.astype(np.float32, copy=False)
        self.std = std.astype(np.float32, copy=False)

    def transform(self, x: np.ndarray) -> np.ndarray:
        array = _numeric_array(
            x,
            field_name="x",
            allowed_ndim=(1, 2),
        )

        if self.mean is None and self.std is None:
            return array.copy()
        if self.mean is None or self.std is None:
            raise RuntimeError("scaler is only partially initialized")

        expected_features = int(self.mean.shape[0])
        actual_features = int(array.shape[-1])
        if actual_features != expected_features:
            raise ValueError(
                "feature count mismatch: "
                f"expected {expected_features}, got {actual_features}"
            )

        transformed = (array - self.mean) / self.std
        transformed = np.asarray(transformed, dtype=np.float32)

        if not np.all(np.isfinite(transformed)):
            raise ValueError("scaled features contain NaN or Infinity")
        return transformed

    def state_dict(self) -> dict[str, list[float] | None]:
        if self.mean is None and self.std is None:
            return {"mean": None, "std": None}
        if self.mean is None or self.std is None:
            raise RuntimeError("scaler is only partially initialized")

        return {
            "mean": self.mean.astype(np.float32, copy=False).tolist(),
            "std": self.std.astype(np.float32, copy=False).tolist(),
        }

    def load_state_dict(self, state: dict) -> None:
        if not isinstance(state, dict):
            raise TypeError("scaler state must be a dictionary")

        mean = state.get("mean")
        std = state.get("std")

        if mean is None and std is None:
            self.mean = None
            self.std = None
            return
        if mean is None or std is None:
            raise ValueError("scaler mean and std must either both exist or both be None")
        if not isinstance(mean, (list, tuple)) or not isinstance(std, (list, tuple)):
            raise TypeError("scaler mean and std must be lists")
        if not mean or not std or len(mean) != len(std):
            raise ValueError("scaler mean and std must be non-empty and equal in length")

        mean_array = np.asarray(mean, dtype=np.float32)
        std_array = np.asarray(std, dtype=np.float32)

        if mean_array.ndim != 1 or std_array.ndim != 1:
            raise ValueError("scaler mean and std must be one-dimensional")
        if not np.all(np.isfinite(mean_array)) or not np.all(np.isfinite(std_array)):
            raise ValueError("scaler state contains NaN or Infinity")
        if np.any(std_array <= 0.0):
            raise ValueError("scaler std values must be strictly positive")

        self.mean = mean_array.copy()
        self.std = std_array.copy()


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()

        self.input_dim = _strict_positive_int(input_dim, field_name="input_dim")
        self.hidden_dim = _strict_positive_int(hidden_dim, field_name="hidden_dim")
        self.num_classes = _strict_positive_int(
            num_classes,
            field_name="num_classes",
        )

        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(self.hidden_dim, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a torch.Tensor")
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.ndim != 2:
            raise ValueError("x must be a one-dimensional feature vector or 2D batch")
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"expected {self.input_dim} features, got {x.shape[-1]}"
            )

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.fc2(x)


@dataclass
class ModelBundle:
    model: MLP
    scaler: StandardScaler
    feature_order: list[str]
    label_names: list[str]
    metadata: dict[str, Any]


def _validate_bundle(bundle: ModelBundle) -> tuple[list[str], list[str], dict[str, Any]]:
    if not isinstance(bundle, ModelBundle):
        raise TypeError("bundle must be a ModelBundle")
    if not isinstance(bundle.model, MLP):
        raise TypeError("bundle.model must be an MLP")
    if not isinstance(bundle.scaler, StandardScaler):
        raise TypeError("bundle.scaler must be a StandardScaler")

    feature_order = _validate_string_list(
        bundle.feature_order,
        field_name="feature_order",
    )
    label_names = _validate_string_list(
        bundle.label_names,
        field_name="label_names",
    )

    if len(feature_order) != bundle.model.input_dim:
        raise ValueError(
            "feature_order length must match model.input_dim: "
            f"{len(feature_order)} != {bundle.model.input_dim}"
        )
    if len(label_names) != bundle.model.num_classes:
        raise ValueError(
            "label_names length must match model.num_classes: "
            f"{len(label_names)} != {bundle.model.num_classes}"
        )

    scaler_feature_count = bundle.scaler.feature_count
    if scaler_feature_count is not None and scaler_feature_count != bundle.model.input_dim:
        raise ValueError(
            "scaler feature count must match model.input_dim: "
            f"{scaler_feature_count} != {bundle.model.input_dim}"
        )

    metadata = _plain_metadata(bundle.metadata)
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a dictionary")

    return feature_order, label_names, metadata


def _cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for name, tensor in model.state_dict().items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"model state {name!r} is not a tensor")
        detached = tensor.detach().cpu()
        if not torch.isfinite(detached).all().item():
            raise ValueError(f"model state {name!r} contains NaN or Infinity")
        state[name] = detached
    return state


def save_bundle(path: str, bundle: ModelBundle) -> None:
    feature_order, label_names, metadata = _validate_bundle(bundle)

    target = Path(path)
    if not target.name:
        raise ValueError("path must identify a file")
    target.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "bundle_format_version": _BUNDLE_FORMAT_VERSION,
        "state_dict": _cpu_state_dict(bundle.model),
        "scaler": bundle.scaler.state_dict(),
        "feature_order": feature_order,
        "label_names": label_names,
        "metadata": metadata,
        "input_dim": bundle.model.input_dim,
        "hidden_dim": bundle.model.hidden_dim,
        "num_classes": bundle.model.num_classes,
    }

    file_descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=str(target.parent),
    )
    os.close(file_descriptor)

    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, target)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def _torch_load_payload(path: Path) -> Any:
    """Load a local model bundle using PyTorch's restricted loader when supported."""
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        # Compatibility fallback for older PyTorch releases that do not support
        # weights_only. Model artifacts must still come from a trusted local source.
        return torch.load(path, map_location="cpu")


def _validate_loaded_state_dict(state_dict: Any) -> dict[str, torch.Tensor]:
    if not isinstance(state_dict, Mapping):
        raise TypeError("state_dict must be a mapping")

    validated: dict[str, torch.Tensor] = {}
    for name, tensor in state_dict.items():
        if not isinstance(name, str) or not name:
            raise ValueError("state_dict keys must be non-empty strings")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"state_dict value {name!r} must be a tensor")
        tensor = tensor.detach().cpu()
        if not torch.isfinite(tensor).all().item():
            raise ValueError(f"state_dict value {name!r} contains NaN or Infinity")
        validated[name] = tensor
    return validated


def load_bundle(path: str) -> ModelBundle:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"model bundle not found: {source}")

    data = _torch_load_payload(source)
    if not isinstance(data, dict):
        raise TypeError("model bundle payload must be a dictionary")

    feature_order = _validate_string_list(
        data.get("feature_order"),
        field_name="feature_order",
    )
    label_names = _validate_string_list(
        data.get("label_names"),
        field_name="label_names",
    )

    raw_input_dim = data.get("input_dim", len(feature_order))
    raw_hidden_dim = data.get("hidden_dim", 64)
    raw_num_classes = data.get("num_classes", len(label_names))

    input_dim = _strict_positive_int(raw_input_dim, field_name="input_dim")
    hidden_dim = _strict_positive_int(raw_hidden_dim, field_name="hidden_dim")
    num_classes = _strict_positive_int(
        raw_num_classes,
        field_name="num_classes",
    )

    if len(feature_order) != input_dim:
        raise ValueError(
            "saved feature_order length does not match input_dim: "
            f"{len(feature_order)} != {input_dim}"
        )
    if len(label_names) != num_classes:
        raise ValueError(
            "saved label_names length does not match num_classes: "
            f"{len(label_names)} != {num_classes}"
        )

    model = MLP(input_dim, hidden_dim, num_classes)
    state_dict = _validate_loaded_state_dict(data.get("state_dict"))
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    scaler = StandardScaler()
    scaler_state = data.get("scaler", {})
    scaler.load_state_dict(scaler_state if isinstance(scaler_state, dict) else {})

    if scaler.feature_count is not None and scaler.feature_count != input_dim:
        raise ValueError(
            "saved scaler feature count does not match input_dim: "
            f"{scaler.feature_count} != {input_dim}"
        )

    metadata = _plain_metadata(data.get("metadata", {}))
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a dictionary")

    return ModelBundle(
        model=model,
        scaler=scaler,
        feature_order=feature_order,
        label_names=label_names,
        metadata=metadata,
    )


def _feature_vector(features: Any, *, expected_length: int) -> np.ndarray:
    if isinstance(features, np.ndarray):
        if features.ndim != 1:
            raise ValueError("features must be a one-dimensional vector")
        if np.issubdtype(features.dtype, np.bool_) or not np.issubdtype(
            features.dtype,
            np.number,
        ):
            raise TypeError("features must contain numeric non-boolean values")
        vector = np.asarray(features, dtype=np.float32)
    elif isinstance(features, Sequence) and not isinstance(
        features,
        (str, bytes, bytearray),
    ):
        normalized: list[float] = []
        for value in features:
            if isinstance(value, bool) or type(value) not in {int, float}:
                raise TypeError(
                    "features must contain actual int/float values only"
                )
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError("features contain NaN or Infinity")
            normalized.append(numeric)
        vector = np.asarray(normalized, dtype=np.float32)
    else:
        raise TypeError("features must be a one-dimensional numeric sequence")

    if vector.size != expected_length:
        raise ValueError(
            f"expected {expected_length} features, got {vector.size}"
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError("features contain NaN or Infinity")
    return vector


def predict_proba(bundle: ModelBundle, features: list[float]) -> np.ndarray:
    feature_order, label_names, _ = _validate_bundle(bundle)

    vector = _feature_vector(
        features,
        expected_length=len(feature_order),
    )
    transformed = bundle.scaler.transform(vector)

    if transformed.ndim != 1 or transformed.shape[0] != bundle.model.input_dim:
        raise ValueError("scaled feature vector has an invalid shape")
    if not np.all(np.isfinite(transformed)):
        raise ValueError("scaled feature vector contains NaN or Infinity")

    x_tensor = torch.from_numpy(
        np.asarray(transformed, dtype=np.float32),
    ).unsqueeze(0)

    bundle.model.eval()
    with torch.inference_mode():
        logits = bundle.model(x_tensor)

        expected_shape = (1, len(label_names))
        if tuple(logits.shape) != expected_shape:
            raise RuntimeError(
                f"model returned shape {tuple(logits.shape)}, "
                f"expected {expected_shape}"
            )
        if not torch.isfinite(logits).all().item():
            raise RuntimeError("model logits contain NaN or Infinity")

        probabilities = torch.softmax(logits, dim=1).squeeze(0)

    if not torch.isfinite(probabilities).all().item():
        raise RuntimeError("model probabilities contain NaN or Infinity")

    result = probabilities.detach().cpu().numpy().astype(np.float32, copy=True)
    if result.shape != (len(label_names),):
        raise RuntimeError("model probabilities have an invalid shape")

    total = float(np.sum(result, dtype=np.float64))
    if not math.isfinite(total) or not math.isclose(
        total,
        1.0,
        rel_tol=1e-5,
        abs_tol=1e-6,
    ):
        raise RuntimeError("model probabilities do not sum to one")

    return result