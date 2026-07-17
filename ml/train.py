from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from config import LABELS, MODEL_VERSION
from ml.features import FEATURE_ORDER, features_from_media, vector_from_features
from ml.metrics import accuracy, macro_f1
from ml.model import MLP, ModelBundle, StandardScaler, save_bundle


_DEFAULT_VALIDATION_FRACTION = 0.20
_DEFAULT_HIDDEN_DIM = 64
_GRADIENT_CLIP_NORM = 5.0


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _configured_labels() -> list[str]:
    if not isinstance(LABELS, (list, tuple)):
        raise ValueError("LABELS must be a list of strings")

    labels: list[str] = []
    for item in LABELS:
        if not isinstance(item, str):
            raise ValueError("LABELS must contain strings only")
        label = item.strip()
        if not label:
            raise ValueError("LABELS cannot contain blank values")
        labels.append(label)

    if not labels:
        raise ValueError("LABELS cannot be empty")
    if len(set(labels)) != len(labels):
        raise ValueError("LABELS cannot contain duplicates")
    return labels


def _strict_positive_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{field_name} must be a positive integer")
    numeric = int(value)
    if numeric <= 0:
        raise ValueError(f"{field_name} must be greater than zero")
    return numeric


def _strict_seed(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError("seed must be a non-negative integer")
    seed = int(value)
    if seed < 0:
        raise ValueError("seed must be non-negative")
    return seed


def _strict_learning_rate(value: Any) -> float:
    if isinstance(value, bool) or type(value) not in {int, float}:
        raise TypeError("lr must be a positive finite number")
    learning_rate = float(value)
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("lr must be a positive finite number")
    return learning_rate


def _safe_dict(value: Any, *, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be a dictionary")
    return value


def load_manifest(path: str) -> list[dict]:
    """Load a JSONL training manifest with line-specific validation errors."""
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")

    samples: list[dict] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                sample = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON in manifest at line {line_number}: {exc.msg}"
                ) from exc

            if not isinstance(sample, dict):
                raise ValueError(
                    f"manifest line {line_number} must contain a JSON object"
                )
            samples.append(sample)

    return samples


def label_to_index(label: Any) -> int:
    """Map a configured label name or valid integer class index to an index."""
    labels = _configured_labels()

    if isinstance(label, bool):
        raise ValueError("boolean values are not valid labels")

    if isinstance(label, (int, np.integer)):
        index = int(label)
        if index < 0 or index >= len(labels):
            raise ValueError(
                f"label index {index} is outside the range 0..{len(labels) - 1}"
            )
        return index

    if isinstance(label, str):
        normalized = label.strip()
        if normalized in labels:
            return labels.index(normalized)

    raise ValueError(f"unknown label: {label!r}")


def _validated_feature_vector(feature_map: dict, *, sample_index: int) -> list[float]:
    vector = vector_from_features(feature_map)

    if not isinstance(vector, list):
        raise TypeError(
            f"sample {sample_index}: vector_from_features must return a list"
        )
    if len(vector) != len(FEATURE_ORDER):
        raise ValueError(
            f"sample {sample_index}: expected {len(FEATURE_ORDER)} features, "
            f"got {len(vector)}"
        )

    normalized: list[float] = []
    for feature_index, value in enumerate(vector):
        if isinstance(value, bool) or type(value) not in {int, float}:
            raise TypeError(
                f"sample {sample_index}: feature {feature_index} "
                "must be a real number"
            )
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(
                f"sample {sample_index}: feature {feature_index} "
                "contains NaN or Infinity"
            )
        normalized.append(numeric)

    return normalized


def build_dataset(samples: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Run the established analyzers and build a finite fixed-width dataset.

    Unlabelled rows are ignored for backward compatibility. Malformed labelled
    rows fail explicitly with their sample index so training cannot silently use
    corrupted evidence.
    """
    if not isinstance(samples, list):
        raise TypeError("samples must be a list")

    feature_rows: list[list[float]] = []
    labels: list[int] = []

    for sample_index, raw_sample in enumerate(samples):
        sample = _safe_dict(
            raw_sample,
            field_name=f"sample {sample_index}",
        )

        label = sample.get("label")
        if label is None:
            continue

        media = sample.get("media", {})
        if not isinstance(media, dict):
            raise TypeError(f"sample {sample_index}: media must be a dictionary")

        task = sample.get("task")
        if task is not None and not isinstance(task, dict):
            raise TypeError(
                f"sample {sample_index}: task must be a dictionary or null"
            )

        try:
            feature_map, _signals = features_from_media(media, task=task)
        except Exception as exc:
            raise RuntimeError(
                f"sample {sample_index}: media feature extraction failed"
            ) from exc

        if not isinstance(feature_map, dict):
            raise TypeError(
                f"sample {sample_index}: features_from_media "
                "must return a feature dictionary"
            )

        feature_rows.append(
            _validated_feature_vector(
                feature_map,
                sample_index=sample_index,
            )
        )
        labels.append(label_to_index(label))

    if not feature_rows:
        return (
            np.empty((0, len(FEATURE_ORDER)), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    x = np.asarray(feature_rows, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)

    if x.ndim != 2 or x.shape[1] != len(FEATURE_ORDER):
        raise RuntimeError("constructed feature matrix has an invalid shape")
    if y.ndim != 1 or y.shape[0] != x.shape[0]:
        raise RuntimeError("constructed label vector has an invalid shape")
    if not np.all(np.isfinite(x)):
        raise ValueError("constructed feature matrix contains NaN or Infinity")

    return x, y


def _validated_training_arrays(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("x and y must be NumPy arrays")
    if x.ndim != 2:
        raise ValueError("x must be a two-dimensional feature matrix")
    if y.ndim != 1:
        raise ValueError("y must be a one-dimensional label vector")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must contain the same number of samples")
    if x.shape[0] == 0:
        raise ValueError("training data cannot be empty")
    if x.shape[1] != len(FEATURE_ORDER):
        raise ValueError(
            f"x must contain exactly {len(FEATURE_ORDER)} features"
        )
    if np.issubdtype(x.dtype, np.bool_) or not np.issubdtype(
        x.dtype,
        np.number,
    ):
        raise TypeError("x must contain numeric non-boolean values")
    if np.issubdtype(y.dtype, np.bool_) or not np.issubdtype(
        y.dtype,
        np.integer,
    ):
        raise TypeError("y must contain integer class indices")

    features = np.asarray(x, dtype=np.float32)
    labels = np.asarray(y, dtype=np.int64)

    if not np.all(np.isfinite(features)):
        raise ValueError("x contains NaN or Infinity")

    class_count = len(_configured_labels())
    if np.any(labels < 0) or np.any(labels >= class_count):
        raise ValueError("y contains labels outside the configured class range")

    return features, labels


def _validate_class_coverage(labels: np.ndarray) -> dict[int, int]:
    configured = _configured_labels()
    counts = Counter(int(value) for value in labels.tolist())

    missing = [
        configured[class_index]
        for class_index in range(len(configured))
        if counts.get(class_index, 0) == 0
    ]
    if missing:
        raise ValueError(
            "training data is missing configured classes: "
            + ", ".join(missing)
        )

    too_small = [
        f"{configured[class_index]}={counts[class_index]}"
        for class_index in range(len(configured))
        if counts[class_index] < 2
    ]
    if too_small:
        raise ValueError(
            "every class needs at least two samples for train/validation "
            "coverage: "
            + ", ".join(too_small)
        )

    return dict(counts)


def _stratified_split_indices(
    labels: np.ndarray,
    *,
    seed: int,
    validation_fraction: float = _DEFAULT_VALIDATION_FRACTION,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")

    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    validation_parts: list[np.ndarray] = []

    for class_index in range(len(_configured_labels())):
        class_indices = np.flatnonzero(labels == class_index)
        if class_indices.size < 2:
            raise ValueError(
                f"class {class_index} needs at least two samples"
            )

        shuffled = rng.permutation(class_indices)
        validation_count = max(
            1,
            int(round(class_indices.size * validation_fraction)),
        )
        validation_count = min(validation_count, class_indices.size - 1)

        validation_parts.append(shuffled[:validation_count])
        train_parts.append(shuffled[validation_count:])

    train_indices = rng.permutation(np.concatenate(train_parts))
    validation_indices = rng.permutation(np.concatenate(validation_parts))

    if train_indices.size == 0 or validation_indices.size == 0:
        raise RuntimeError("stratified split produced an empty partition")

    return train_indices.astype(np.int64), validation_indices.astype(np.int64)


def _set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _class_weights(train_labels: np.ndarray, class_count: int) -> torch.Tensor:
    counts = np.bincount(train_labels, minlength=class_count).astype(np.float64)
    if np.any(counts <= 0):
        raise ValueError("every class must be represented in the training split")

    inverse = 1.0 / counts
    normalized = inverse / np.mean(inverse)
    return torch.tensor(normalized, dtype=torch.float32)


def _evaluate_model(
    model: MLP,
    data_loader: DataLoader,
    loss_fn: torch.nn.Module,
    *,
    class_count: int,
) -> tuple[float, float, float]:
    model.eval()

    total_loss = 0.0
    total_samples = 0
    predicted_labels: list[int] = []
    true_labels: list[int] = []

    with torch.inference_mode():
        for batch_x, batch_y in data_loader:
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)

            if not torch.isfinite(loss).item():
                raise RuntimeError("validation loss became NaN or Infinity")

            batch_size = int(batch_y.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size

            predictions = torch.argmax(logits, dim=1)
            predicted_labels.extend(int(value) for value in predictions.tolist())
            true_labels.extend(int(value) for value in batch_y.tolist())

    if total_samples <= 0:
        raise RuntimeError("validation loader produced no samples")

    validation_loss = total_loss / total_samples
    validation_accuracy = accuracy(true_labels, predicted_labels)
    validation_f1 = macro_f1(
        true_labels,
        predicted_labels,
        class_count,
    )
    return validation_loss, validation_accuracy, validation_f1


def train_model(
    x: np.ndarray,
    y: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
):
    """Train the established MLP using a stratified leakage-safe split.

    The scaler is fitted on the training partition only. The best validation
    checkpoint is restored before returning. This function does not claim or
    guarantee any particular model accuracy.
    """
    features, labels = _validated_training_arrays(x, y)

    epoch_count = _strict_positive_int(epochs, field_name="epochs")
    requested_batch_size = _strict_positive_int(
        batch_size,
        field_name="batch_size",
    )
    learning_rate = _strict_learning_rate(lr)
    normalized_seed = _strict_seed(seed)

    class_counts = _validate_class_coverage(labels)
    _set_reproducible_seed(normalized_seed)

    train_indices, validation_indices = _stratified_split_indices(
        labels,
        seed=normalized_seed,
    )

    x_train = features[train_indices]
    y_train = labels[train_indices]
    x_validation = features[validation_indices]
    y_validation = labels[validation_indices]

    scaler = StandardScaler()
    scaler.fit(x_train)

    x_train_scaled = scaler.transform(x_train)
    x_validation_scaled = scaler.transform(x_validation)

    train_dataset = TensorDataset(
        torch.from_numpy(
            np.asarray(x_train_scaled, dtype=np.float32),
        ),
        torch.from_numpy(
            np.asarray(y_train, dtype=np.int64),
        ),
    )
    validation_dataset = TensorDataset(
        torch.from_numpy(
            np.asarray(x_validation_scaled, dtype=np.float32),
        ),
        torch.from_numpy(
            np.asarray(y_validation, dtype=np.int64),
        ),
    )

    effective_batch_size = min(
        requested_batch_size,
        len(train_dataset),
    )

    generator = torch.Generator()
    generator.manual_seed(normalized_seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=min(requested_batch_size, len(validation_dataset)),
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    class_count = len(_configured_labels())
    model = MLP(
        input_dim=features.shape[1],
        hidden_dim=_DEFAULT_HIDDEN_DIM,
        num_classes=class_count,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )
    loss_fn = torch.nn.CrossEntropyLoss(
        weight=_class_weights(y_train, class_count),
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_validation_loss = math.inf
    best_validation_accuracy = -1.0
    best_validation_f1 = -1.0

    for epoch_index in range(epoch_count):
        model.train()

        running_loss = 0.0
        trained_samples = 0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad(set_to_none=True)

            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            if not torch.isfinite(loss).item():
                raise RuntimeError(
                    f"training loss became NaN or Infinity at epoch "
                    f"{epoch_index + 1}"
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=_GRADIENT_CLIP_NORM,
            )
            optimizer.step()

            current_batch_size = int(batch_y.shape[0])
            running_loss += float(loss.item()) * current_batch_size
            trained_samples += current_batch_size

        if trained_samples <= 0:
            raise RuntimeError("training loader produced no samples")

        training_loss = running_loss / trained_samples
        validation_loss, validation_accuracy, validation_f1 = _evaluate_model(
            model,
            validation_loader,
            loss_fn,
            class_count=class_count,
        )

        print(
            f"epoch={epoch_index + 1} "
            f"train_loss={training_loss:.4f} "
            f"val_loss={validation_loss:.4f} "
            f"val_acc={validation_accuracy:.4f} "
            f"val_f1={validation_f1:.4f}"
        )

        candidate = (
            validation_f1,
            validation_accuracy,
            -validation_loss,
        )
        best = (
            best_validation_f1,
            best_validation_accuracy,
            -best_validation_loss,
        )
        if candidate > best:
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            best_epoch = epoch_index + 1
            best_validation_loss = validation_loss
            best_validation_accuracy = validation_accuracy
            best_validation_f1 = validation_f1

    if best_state is None:
        raise RuntimeError("training did not produce a valid checkpoint")

    model.load_state_dict(best_state, strict=True)
    model.eval()

    model.training_summary = {
        "best_epoch": int(best_epoch),
        "validation_loss": float(best_validation_loss),
        "validation_accuracy": float(best_validation_accuracy),
        "validation_macro_f1": float(best_validation_f1),
        "train_samples": int(train_indices.size),
        "validation_samples": int(validation_indices.size),
        "class_counts": {
            _configured_labels()[class_index]: int(class_counts[class_index])
            for class_index in range(class_count)
        },
        "seed": int(normalized_seed),
    }

    return model, scaler


def _training_metadata(
    *,
    model: MLP,
    total_samples: int,
) -> dict[str, Any]:
    summary = getattr(model, "training_summary", {})
    if not isinstance(summary, dict):
        summary = {}

    return {
        "created_at": _utc_now_iso(),
        "model_version": MODEL_VERSION,
        "feature_order": list(FEATURE_ORDER),
        "train_samples": int(total_samples),
        "training_summary": copy.deepcopy(summary),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the local readiness classification model.",
    )
    parser.add_argument("--manifest", default="data/manifest.jsonl")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="models/latest.pt")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        raise SystemExit(f"Manifest not found: {manifest_path}")

    try:
        samples = load_manifest(str(manifest_path))
        if not samples:
            raise ValueError("manifest is empty")

        x, y = build_dataset(samples)
        if len(y) < 10:
            raise ValueError("need at least 10 labeled samples to train")

        model, scaler = train_model(
            x,
            y,
            args.epochs,
            args.batch_size,
            args.lr,
            args.seed,
        )

        output_path = Path(args.out)
        if not output_path.name:
            raise ValueError("output path must identify a file")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        labels = _configured_labels()
        bundle = ModelBundle(
            model=model,
            scaler=scaler,
            feature_order=list(FEATURE_ORDER),
            label_names=labels,
            metadata=_training_metadata(
                model=model,
                total_samples=len(y),
            ),
        )
        save_bundle(str(output_path), bundle)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"Training failed: {exc}") from exc

    print(f"saved_model={output_path}")


if __name__ == "__main__":
    main()