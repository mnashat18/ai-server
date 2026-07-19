from __future__ import annotations

import inspect
import time
from typing import Any

from logger import get_logger


logger = get_logger()


def _elapsed_ms(started_at: float) -> int:
    return int(round((time.perf_counter() - started_at) * 1000))


def _log_worker_perf(scan_id: str | None, analyzer_name: str, metric: str, started_at: float, *, status: str = "ok") -> None:
    value = _elapsed_ms(started_at)
    logger.info(
        "[WORKER_PERF] analyzer=%s metric=%s scan_id=%s value=%s status=%s",
        analyzer_name,
        metric,
        scan_id,
        value,
        status,
    )
    for handler in getattr(logger, "handlers", []):
        try:
            handler.flush()
        except Exception:
            pass


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


def _load_analyzer_callable(analyzer_name: str):
    if analyzer_name == "video":
        from video import analyze_video as analyzer
    elif analyzer_name == "audio":
        from audio import analyze_audio as analyzer
    elif analyzer_name == "image":
        from vision import analyze_face as analyzer
    else:
        raise ValueError("invalid_analyzer")
    return analyzer


def _prewarm_analyzer(analyzer_name: str) -> dict[str, Any]:
    if analyzer_name != "audio":
        return {}
    from audio import prewarm_audio_analyzer

    return prewarm_audio_analyzer()


def _ready_payload(
    analyzer_name: str,
    worker_generation: int,
    started_at: float,
    analyzer_import_ms: int,
    prewarm_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metrics = {
        "child_entry_ms": _elapsed_ms(started_at),
        "analyzer_import_ms": analyzer_import_ms,
        "total_worker_ms": _elapsed_ms(started_at),
    }
    if analyzer_name == "audio":
        metrics["audio_import_ms"] = metrics["analyzer_import_ms"]
        metrics.update(prewarm_metrics or {})
    return {
        "type": "ready",
        "ok": True,
        "analyzer": analyzer_name,
        "worker_generation": worker_generation,
        "metrics": metrics,
    }


def _structured_error_payload(
    *,
    analyzer_name: str,
    worker_generation: int,
    job_id: str | None = None,
    scan_id: str | None = None,
    error_type: str,
    started_at: float,
    execution_started_at: float | None = None,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "child_entry_ms": _elapsed_ms(started_at),
        "analyzer_execution_ms": None,
        "response_send_ms": None,
        "total_worker_ms": None,
    }
    if execution_started_at is not None:
        metrics["analyzer_execution_ms"] = _elapsed_ms(execution_started_at)
    return {
        "type": "result",
        "job_id": job_id,
        "scan_id": scan_id,
        "worker_generation": worker_generation,
        "ok": False,
        "error_type": error_type,
        "metrics": metrics,
    }


def _structured_success_payload(
    *,
    analyzer_name: str,
    worker_generation: int,
    job_id: str | None = None,
    scan_id: str | None = None,
    result: Any,
    started_at: float,
    execution_started_at: float,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "child_entry_ms": _elapsed_ms(started_at),
        "analyzer_execution_ms": _elapsed_ms(execution_started_at),
        "response_send_ms": None,
        "total_worker_ms": None,
    }
    return {
        "type": "result",
        "job_id": job_id,
        "scan_id": scan_id,
        "worker_generation": worker_generation,
        "ok": True,
        "result": result if isinstance(result, dict) else _missing_placeholder(analyzer_name, f"{analyzer_name}_missing"),
        "metrics": metrics,
    }


def _invoke_analyzer(analyzer: Any, analyzer_name: str, path: str, scan_id: str | None) -> Any:
    if analyzer_name == "audio":
        try:
            signature = inspect.signature(analyzer)
        except (TypeError, ValueError):
            signature = None
        if signature is not None:
            parameters = signature.parameters
            if "scan_id" in parameters or any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
                return analyzer(path, scan_id=scan_id)
    return analyzer(path)


def worker_main(result_conn: Any, analyzer_name: str, worker_generation: int) -> None:
    started_at = time.perf_counter()
    analyzer = None
    try:
        import_started_at = time.perf_counter()
        analyzer = _load_analyzer_callable(analyzer_name)
        analyzer_import_ms = _elapsed_ms(import_started_at)
        prewarm_metrics = _prewarm_analyzer(analyzer_name)
        if analyzer_name == "audio" and prewarm_metrics.get("audio_warm_benchmark_passed") is not True:
            raise RuntimeError("audio_prewarm_failed")
        ready = _ready_payload(analyzer_name, worker_generation, started_at, analyzer_import_ms, prewarm_metrics)
        try:
            result_conn.send(ready)
        except Exception:
            return
    except Exception as exc:
        error_type = type(exc).__name__
        logger.error(
            "analysis_worker_start_failed analyzer=%s error_type=%s",
            analyzer_name,
            error_type,
        )
        try:
            result_conn.send(
                {
                    "type": "ready",
                    "ok": False,
                    "analyzer": analyzer_name,
                    "worker_generation": worker_generation,
                    "error_type": error_type,
                    "metrics": {
                        "child_entry_ms": _elapsed_ms(started_at),
                        "analyzer_import_ms": None,
                        "total_worker_ms": _elapsed_ms(started_at),
                    },
                }
            )
        except Exception:
            pass
        try:
            result_conn.close()
        except Exception:
            pass
        return

    try:
        while True:
            receive_started_at = time.perf_counter()
            try:
                message = result_conn.recv()
            except EOFError:
                break
            except Exception as exc:
                logger.error(
                    "analysis_worker_recv_failed analyzer=%s error_type=%s",
                    analyzer_name,
                    type(exc).__name__,
                )
                break

            if not isinstance(message, dict):
                continue

            message_type = message.get("type")
            if message_type == "shutdown":
                break
            if message_type != "job":
                continue

            job_started_at = time.perf_counter()
            job_id = message.get("job_id")
            scan_id = message.get("scan_id")
            path = message.get("path")
            _log_worker_perf(scan_id, analyzer_name, f"{analyzer_name}_worker_receive_ms", receive_started_at)
            response_started_at = time.perf_counter()
            if not path:
                payload = {
                    "type": "result",
                    "job_id": job_id,
                    "scan_id": scan_id,
                    "worker_generation": worker_generation,
                    "ok": True,
                    "result": _missing_placeholder(analyzer_name, f"{analyzer_name}_missing"),
                    "metrics": {
                        "child_entry_ms": _elapsed_ms(started_at),
                        "analyzer_execution_ms": 0,
                        "response_send_ms": None,
                        "total_worker_ms": None,
                    },
                }
            else:
                try:
                    execution_started_at = time.perf_counter()
                    result = _invoke_analyzer(analyzer, analyzer_name, path, scan_id)
                except Exception as exc:
                    payload = _structured_error_payload(
                        analyzer_name=analyzer_name,
                        worker_generation=worker_generation,
                        job_id=job_id,
                        scan_id=scan_id,
                        error_type=type(exc).__name__,
                        started_at=job_started_at,
                        execution_started_at=execution_started_at,
                    )
                else:
                    payload = _structured_success_payload(
                        analyzer_name=analyzer_name,
                        worker_generation=worker_generation,
                        job_id=job_id,
                        scan_id=scan_id,
                        result=result,
                        started_at=job_started_at,
                        execution_started_at=execution_started_at,
                    )

            try:
                payload.setdefault("metrics", {})["response_send_ms"] = _elapsed_ms(response_started_at)
                payload["metrics"]["total_worker_ms"] = _elapsed_ms(job_started_at)
                result_conn.send(payload)
                _log_worker_perf(scan_id, analyzer_name, f"{analyzer_name}_result_publish_ms", response_started_at)
            except Exception:
                break
    finally:
        try:
            result_conn.close()
        except Exception:
            pass


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
