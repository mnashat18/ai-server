import json
import os
from datetime import datetime

from config import BASELINE_ALPHA, BASELINE_PATH


def _ensure_dir(path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def load_baselines() -> dict:
    if not os.path.exists(BASELINE_PATH):
        return {}
    try:
        with open(BASELINE_PATH, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}


def save_baselines(data: dict) -> None:
    _ensure_dir(BASELINE_PATH)
    with open(BASELINE_PATH, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def get_baseline(subject_id: str) -> dict | None:
    data = load_baselines()
    return data.get(subject_id)


def update_baseline(subject_id: str, scores: dict) -> dict:
    data = load_baselines()
    entry = data.get(subject_id, {"count": 0})

    for key, value in scores.items():
        if value is None:
            continue
        prior = entry.get(key)
        if prior is None:
            entry[key] = value
        else:
            entry[key] = round((prior * (1 - BASELINE_ALPHA)) + (value * BASELINE_ALPHA), 4)

    entry["count"] = int(entry.get("count", 0)) + 1
    entry["updated_at"] = datetime.utcnow().isoformat() + "Z"
    data[subject_id] = entry
    save_baselines(data)
    return entry
