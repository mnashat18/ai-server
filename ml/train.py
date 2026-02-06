import argparse
import json
import os
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from config import LABELS, MODEL_VERSION
from ml.features import FEATURE_ORDER, features_from_media, vector_from_features
from ml.metrics import accuracy, macro_f1
from ml.model import MLP, ModelBundle, StandardScaler, save_bundle


def load_manifest(path: str) -> list[dict]:
    samples = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples


def label_to_index(label) -> int:
    if isinstance(label, int):
        return label
    if label in LABELS:
        return LABELS.index(label)
    raise ValueError(f"Unknown label: {label}")


def build_dataset(samples: list[dict]):
    features = []
    labels = []
    for sample in samples:
        media = sample.get("media", {})
        task = sample.get("task")
        label = sample.get("label")
        if label is None:
            continue
        feature_map, _signals = features_from_media(media, task=task)
        features.append(vector_from_features(feature_map))
        labels.append(label_to_index(label))
    x = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)
    return x, y


def train_model(x: np.ndarray, y: np.ndarray, epochs: int, batch_size: int, lr: float, seed: int):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    split = int(len(y) * 0.8)
    train_idx = idx[:split]
    val_idx = idx[split:]

    x_train = x[train_idx]
    y_train = y[train_idx]
    x_val = x[val_idx]
    y_val = y[val_idx]

    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_val = scaler.transform(x_val)

    train_ds = TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
    val_ds = TensorDataset(torch.tensor(x_val), torch.tensor(y_val))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = MLP(input_dim=x.shape[1], hidden_dim=64, num_classes=len(LABELS))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()
            running += float(loss.item())
        avg_loss = running / max(1, len(train_loader))

        model.eval()
        val_preds = []
        val_true = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                logits = model(batch_x)
                preds = torch.argmax(logits, dim=1)
                val_preds.extend(preds.tolist())
                val_true.extend(batch_y.tolist())

        acc = accuracy(val_true, val_preds)
        f1 = macro_f1(val_true, val_preds, len(LABELS))
        print(f"epoch={epoch + 1} loss={avg_loss:.4f} val_acc={acc:.4f} val_f1={f1:.4f}")

    return model, scaler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifest.jsonl")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="models/latest.pt")
    args = parser.parse_args()

    if not os.path.exists(args.manifest):
        raise SystemExit(f"Manifest not found: {args.manifest}")

    samples = load_manifest(args.manifest)
    if not samples:
        raise SystemExit("Manifest is empty.")
    x, y = build_dataset(samples)
    if len(y) < 10:
        raise SystemExit("Need at least 10 labeled samples to train.")

    model, scaler = train_model(x, y, args.epochs, args.batch_size, args.lr, args.seed)
    metadata = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "model_version": MODEL_VERSION,
        "feature_order": FEATURE_ORDER,
        "train_samples": int(len(y)),
    }
    bundle = ModelBundle(
        model=model,
        scaler=scaler,
        feature_order=FEATURE_ORDER,
        label_names=LABELS,
        metadata=metadata,
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    save_bundle(args.out, bundle)
    print(f"saved_model={args.out}")


if __name__ == "__main__":
    main()
