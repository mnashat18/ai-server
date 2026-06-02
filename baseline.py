from __future__ import annotations

from datetime import datetime
from typing import Any

from config import (
    BASELINE_ACTIVE_AFTER,
    BASELINE_EVENING_START_HOUR,
    BASELINE_MORNING_END_HOUR,
    BASELINE_PROVISIONAL_AFTER,
    BASELINE_USE_AFTER,
)


def _to_datetime(value: str | None) -> datetime:
    if not value:
        return datetime.utcnow()
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return datetime.utcnow()


def _time_bucket(scanned_at: str | None) -> str:
    hour = _to_datetime(scanned_at).hour
    if hour < BASELINE_MORNING_END_HOUR:
        return "morning"
    if hour >= BASELINE_EVENING_START_HOUR:
        return "evening"
    return "midday"


def _empty_stat() -> dict[str, Any]:
    return {
        "avg": None,
        "count": 0,
        "m2": 0.0,
        "std": None,
        "buckets": {
            "morning": {"avg": None, "count": 0, "m2": 0.0, "std": None},
            "midday": {"avg": None, "count": 0, "m2": 0.0, "std": None},
            "evening": {"avg": None, "count": 0, "m2": 0.0, "std": None},
        },
    }


def _update_running_stat(stat: dict[str, Any], value: float) -> dict[str, Any]:
    current = dict(stat or _empty_stat())
    count = int(current.get("count", 0)) + 1
    avg = current.get("avg")
    m2 = float(current.get("m2", 0.0))

    if avg is None:
        avg = float(value)
        m2 = 0.0
    else:
        delta = float(value) - float(avg)
        avg = float(avg) + (delta / count)
        delta2 = float(value) - avg
        m2 = m2 + (delta * delta2)

    std = (m2 / (count - 1)) ** 0.5 if count > 1 else None
    current["avg"] = round(avg, 4)
    current["count"] = count
    current["m2"] = round(m2, 6)
    current["std"] = round(std, 4) if std is not None else None
    return current


def _update_bucket(stat: dict[str, Any], bucket: str, value: float) -> dict[str, Any]:
    buckets = dict(stat.get("buckets") or _empty_stat()["buckets"])
    buckets[bucket] = _update_running_stat(buckets.get(bucket) or _empty_stat()["buckets"][bucket], value)
    stat["buckets"] = buckets
    return stat


def baseline_signal_payload(
    baseline: dict | None,
    *,
    face_score: float | None,
    voice_score: float | None,
    reaction_score: float | None,
    scanned_at: str | None = None,
) -> dict:
    bucket = _time_bucket(scanned_at)
    current = baseline or {}

    face_avg = dict(current.get("face_avg") or _empty_stat())
    voice_avg = dict(current.get("voice_avg") or _empty_stat())
    reaction_avg = dict(current.get("reaction_avg") or _empty_stat())

    if face_score is not None:
        face_avg = _update_bucket(_update_running_stat(face_avg, face_score), bucket, face_score)
    if voice_score is not None:
        voice_avg = _update_bucket(_update_running_stat(voice_avg, voice_score), bucket, voice_score)
    if reaction_score is not None:
        reaction_avg = _update_bucket(_update_running_stat(reaction_avg, reaction_score), bucket, reaction_score)

    next_count = int(current.get("scan_count", 0)) + 1
    is_active = bool(current.get("is_active")) or next_count >= BASELINE_ACTIVE_AFTER
    activated_at = current.get("activated_at")
    if is_active and not activated_at:
        activated_at = datetime.utcnow().isoformat() + "Z"

    return {
        "scan_count": next_count,
        "face_avg": face_avg,
        "voice_avg": voice_avg,
        "reaction_avg": reaction_avg,
        "is_active": is_active,
        "activated_at": activated_at,
        "date_updated": datetime.utcnow().isoformat() + "Z",
    }


def baseline_status_payload(baseline: dict | None) -> dict:
    count = int((baseline or {}).get("scan_count", 0))
    is_active = bool((baseline or {}).get("is_active")) and count >= BASELINE_USE_AFTER
    is_provisional = count >= BASELINE_PROVISIONAL_AFTER and not is_active

    face_avg = (baseline or {}).get("face_avg") or {}
    buckets = face_avg.get("buckets") or {}
    needs_morning_scan = int((buckets.get("morning") or {}).get("count", 0)) == 0
    needs_evening_scan = int((buckets.get("evening") or {}).get("count", 0)) == 0
    scans_remaining = max(BASELINE_ACTIVE_AFTER - count, 0)

    if is_active:
        message = "Baseline active. Current scans are compared against the employee's own readiness pattern."
    elif count == 0:
        message = "No baseline yet. Capture morning and evening scans to start calibration."
    elif is_provisional:
        message = "Baseline is provisional. Keep collecting scans across the day until activation."
    else:
        message = "Calibration started. More valid scans are needed before personalized scoring is active."

    return {
        "is_active": is_active,
        "scan_count": count,
        "scans_remaining": scans_remaining,
        "is_provisional": is_provisional,
        "needs_morning_scan": needs_morning_scan,
        "needs_evening_scan": needs_evening_scan,
        "message": message,
    }


def baseline_ready_for_scoring(baseline: dict | None) -> bool:
    status = baseline_status_payload(baseline)
    return bool(status["is_active"])
