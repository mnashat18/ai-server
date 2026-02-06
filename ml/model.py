from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn


class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, x: np.ndarray) -> None:
        self.mean = np.mean(x, axis=0)
        self.std = np.std(x, axis=0)
        self.std[self.std == 0] = 1.0

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            return x
        return (x - self.mean) / self.std

    def state_dict(self) -> dict:
        return {
            "mean": self.mean.tolist() if self.mean is not None else None,
            "std": self.std.tolist() if self.std is not None else None,
        }

    def load_state_dict(self, state: dict) -> None:
        mean = state.get("mean")
        std = state.get("std")
        self.mean = np.array(mean, dtype=np.float32) if mean is not None else None
        self.std = np.array(std, dtype=np.float32) if std is not None else None


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
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


def save_bundle(path: str, bundle: ModelBundle) -> None:
    payload = {
        "state_dict": bundle.model.state_dict(),
        "scaler": bundle.scaler.state_dict(),
        "feature_order": bundle.feature_order,
        "label_names": bundle.label_names,
        "metadata": bundle.metadata,
        "input_dim": bundle.model.input_dim,
        "hidden_dim": bundle.model.hidden_dim,
    }
    torch.save(payload, path)


def load_bundle(path: str) -> ModelBundle:
    data = torch.load(path, map_location="cpu")
    input_dim = int(data.get("input_dim", len(data.get("feature_order", []))))
    label_names = data.get("label_names", [])
    hidden_dim = int(data.get("hidden_dim", 64))
    model = MLP(input_dim, hidden_dim, len(label_names))
    model.load_state_dict(data["state_dict"])
    model.eval()

    scaler = StandardScaler()
    scaler.load_state_dict(data.get("scaler", {}))

    return ModelBundle(
        model=model,
        scaler=scaler,
        feature_order=data.get("feature_order", []),
        label_names=label_names,
        metadata=data.get("metadata", {}),
    )


def predict_proba(bundle: ModelBundle, features: list[float]) -> np.ndarray:
    x = np.array(features, dtype=np.float32)
    x = bundle.scaler.transform(x)
    x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits = bundle.model(x_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).numpy()
    return probs
