from __future__ import annotations

import time
from typing import Any


def _elapsed_ms(started_at: float) -> int:
    return int(round((time.perf_counter() - started_at) * 1000))


def _warning_key(analyzer_name: str) -> str:
    return {
        "video": "visual_warnings",
        "audio": "audio_warnings",
        "image": "image_warnings",
    }.get(analyzer_name, "warnings")


def _missing_placeholder(analyzer_name: str, missing_warning: str) -> dict[str, Any]:
    return {
        "score": None,
        "details": {
            "status": "missing",
            _warning_key(analyzer_name): [missing_warning],
        },
    }


def run_analysis_worker(
    result_conn: Any,
    analyzer_name: str,
    path: str | None,
    missing_warning: str,
    scan_id: str,
    parent_started_at: float,
) -> None:
    metrics: dict[str, Any] = {
        "child_entry_ms": _elapsed_ms(parent_started_at),
        "analyzer_import_ms": None,
        "analyzer_execution_ms": None,
        "result_send_ms": None,
        "total_worker_ms": None,
    }

    try:
        if not path:
            payload = {
                "ok": True,
                "result": _missing_placeholder(analyzer_name, missing_warning),
                "metrics": metrics,
            }
        else:
            analyzer_import_started_at = time.perf_counter()
            if analyzer_name == "video":
                from video import analyze_video as analyzer
            elif analyzer_name == "audio":
                from audio import analyze_audio as analyzer
            elif analyzer_name == "image":
                from vision import analyze_face as analyzer
            else:
                raise ValueError("invalid_analyzer")

            metrics["analyzer_import_ms"] = _elapsed_ms(analyzer_import_started_at)
            analyzer_execution_started_at = time.perf_counter()
            try:
                result = analyzer(path)
            except Exception as exc:
                payload = {
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "metrics": metrics,
                }
            else:
                metrics["analyzer_execution_ms"] = _elapsed_ms(analyzer_execution_started_at)
                payload = {
                    "ok": True,
                    "result": result if isinstance(result, dict) else _missing_placeholder(analyzer_name, missing_warning),
                    "metrics": metrics,
                }
    except Exception as exc:
        payload = {
            "ok": False,
            "error_type": type(exc).__name__,
            "metrics": metrics,
        }

    result_send_started_at = time.perf_counter()
    try:
        result_conn.send(payload)
    except Exception:
        pass
    finally:
        metrics["result_send_ms"] = _elapsed_ms(result_send_started_at)
        metrics["total_worker_ms"] = _elapsed_ms(parent_started_at)
        try:
            result_conn.close()
        except Exception:
            pass
