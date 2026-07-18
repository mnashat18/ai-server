from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FuturesTimeoutError, wait as futures_wait
from dataclasses import dataclass
import multiprocessing
import queue
import threading
import time
import uuid
from typing import Any, Callable
from multiprocessing.connection import wait as multiprocessing_wait

from logger import get_logger

import analysis_worker


logger = get_logger()


def _elapsed_ms(started_at: float) -> int:
    return int(round((time.perf_counter() - started_at) * 1000))


def _safe_ms(value: Any) -> int:
    if isinstance(value, bool) or type(value) not in {int, float}:
        return 0
    numeric = float(value)
    if numeric < 0.0:
        return 0
    return int(round(numeric))


def _timeout_placeholder(analyzer_name: str) -> dict[str, Any]:
    warning_key = {
        "video": "visual_warnings",
        "audio": "audio_warnings",
        "image": "image_warnings",
    }.get(analyzer_name, "warnings")
    status = "load_failed" if analyzer_name in {"video", "audio"} else "invalid_image"
    return {
        "score": None,
        "details": {
            "status": status,
            warning_key: [f"{analyzer_name}_timeout"],
        },
    }


def _error_placeholder(analyzer_name: str) -> dict[str, Any]:
    warning_key = {
        "video": "visual_warnings",
        "audio": "audio_warnings",
        "image": "image_warnings",
    }.get(analyzer_name, "warnings")
    return {
        "score": None,
        "details": {
            "status": "error",
            warning_key: [f"{analyzer_name}_analysis_error"],
        },
    }


def _missing_placeholder(analyzer_name: str) -> dict[str, Any]:
    warning_key = {
        "video": "visual_warnings",
        "audio": "audio_warnings",
        "image": "image_warnings",
    }.get(analyzer_name, "warnings")
    return {
        "score": None,
        "details": {
            "status": "missing",
            warning_key: [f"{analyzer_name}_missing"],
        },
    }


def _job_warning_key(analyzer_name: str) -> str:
    return {
        "video": "visual_warnings",
        "audio": "audio_warnings",
        "image": "image_warnings",
    }.get(analyzer_name, "warnings")


def _log_perf(scan_id: str, metric: str, value: Any) -> None:
    if isinstance(value, bool):
        numeric: Any = int(value)
    elif value is None:
        numeric = 0
    elif type(value) in {int, float}:
        numeric = int(round(float(value))) if float(value) >= 0.0 else 0
    else:
        numeric = 0
    logger.info("[PERF] %s scan_id=%s value=%s", metric, scan_id, numeric)


def _wait_handles(handles: list[Any], timeout: float | None = None) -> list[Any]:
    return list(multiprocessing_wait(handles, timeout=timeout))


def _smoke_worker_main(result_conn: Any, analyzer_name: str, worker_generation: int) -> None:
    started_at = time.perf_counter()
    try:
        result_conn.send(
            {
                "type": "ready",
                "ok": True,
                "analyzer": analyzer_name,
                "worker_generation": worker_generation,
                "metrics": {
                    "child_entry_ms": 1,
                    "analyzer_import_ms": 1,
                    "total_worker_ms": 1,
                },
            }
        )
        while True:
            try:
                message = result_conn.recv()
            except EOFError:
                break
            if not isinstance(message, dict):
                continue
            if message.get("type") == "shutdown":
                break
            if message.get("type") != "job":
                continue
            job_id = message.get("job_id")
            scan_id = message.get("scan_id")
            path = message.get("path")
            if not path:
                payload = {
                    "type": "result",
                    "job_id": job_id,
                    "scan_id": scan_id,
                    "worker_generation": worker_generation,
                    "ok": True,
                    "result": _missing_placeholder(analyzer_name),
                    "metrics": {
                        "child_entry_ms": 1,
                        "analyzer_execution_ms": 1,
                        "response_send_ms": 1,
                        "total_worker_ms": 1,
                    },
                }
            else:
                payload = {
                    "type": "result",
                    "job_id": job_id,
                    "scan_id": scan_id,
                    "worker_generation": worker_generation,
                    "ok": True,
                    "result": {
                        "score": 0.9,
                        "details": {
                            "status": "ok",
                            "analyzer": analyzer_name,
                        },
                    },
                    "metrics": {
                        "child_entry_ms": 1,
                        "analyzer_execution_ms": 1,
                        "response_send_ms": 1,
                        "total_worker_ms": 1,
                    },
                }
            try:
                result_conn.send(payload)
            except Exception:
                break
    finally:
        try:
            result_conn.close()
        except Exception:
            pass


class AnalysisRuntimeError(RuntimeError):
    pass


class AnalysisRuntimeStartupError(AnalysisRuntimeError):
    pass


class AnalysisRuntimeUnavailable(AnalysisRuntimeError):
    pass


class AnalysisRuntimeBusyError(AnalysisRuntimeError):
    pass


@dataclass
class _WorkerJob:
    job_id: str
    scan_id: str
    path: str | None
    deadline_at: float
    submitted_at: float
    future: Future


_SHUTDOWN_JOB = object()


class WorkerSupervisor:
    def __init__(
        self,
        analyzer_name: str,
        *,
        queue_capacity: int = 4,
        startup_timeout_seconds: float = 6.5,
        recovery_timeout_seconds: float = 1.5,
        poll_interval_seconds: float = 0.05,
        context_factory: Callable[[], Any] | None = None,
        worker_entry: Callable[..., Any] | None = None,
    ) -> None:
        self.analyzer_name = analyzer_name
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=queue_capacity)
        self._startup_timeout_seconds = startup_timeout_seconds
        self._recovery_timeout_seconds = recovery_timeout_seconds
        self._poll_interval_seconds = poll_interval_seconds
        self._context_factory = context_factory or (lambda: multiprocessing.get_context("spawn"))
        self._worker_entry = worker_entry or analysis_worker.worker_main
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._dispatcher = threading.Thread(
            target=self._dispatch_loop,
            name=f"{analyzer_name}-analysis-dispatcher",
            daemon=True,
        )
        self._dispatcher_started = False
        self._started = False
        self._generation = 0
        self._process = None
        self._parent_conn = None
        self._child_conn = None
        self._ready = False
        self._ready_metrics: dict[str, Any] = {}
        self._last_spawn_metrics: dict[str, Any] = {}

    def _spawn_worker_locked(self) -> dict[str, Any]:
        ctx = self._context_factory()
        parent_conn, child_conn = ctx.Pipe(duplex=True)
        generation = self._generation + 1
        process_started_at = time.perf_counter()
        process = ctx.Process(
            target=self._worker_entry,
            args=(child_conn, self.analyzer_name, generation),
            daemon=True,
        )
        process.start()
        process_start_return_ms = _elapsed_ms(process_started_at)
        try:
            child_conn.close()
        except Exception:
            pass

        ready_deadline = time.perf_counter() + self._startup_timeout_seconds
        ready_payload: dict[str, Any] | None = None
        while time.perf_counter() < ready_deadline:
            timeout = max(0.0, ready_deadline - time.perf_counter())
            ready_handles = _wait_handles([parent_conn, process.sentinel], timeout=timeout)
            if parent_conn in ready_handles:
                try:
                    message = parent_conn.recv()
                except EOFError:
                    message = None
                if isinstance(message, dict) and message.get("type") == "ready":
                    ready_payload = message
                break
            if process.sentinel in ready_handles and not process.is_alive():
                break
        if not (
            isinstance(ready_payload, dict)
            and ready_payload.get("ok") is True
            and ready_payload.get("worker_generation") == generation
        ):
            self._cleanup_process_locked(process, parent_conn, timeout_seconds=self._recovery_timeout_seconds)
            raise AnalysisRuntimeStartupError(f"{self.analyzer_name}_worker_start_failed")

        self._generation = generation
        self._process = process
        self._parent_conn = parent_conn
        self._child_conn = child_conn
        self._ready = True

        ready_metrics = ready_payload.get("metrics") if isinstance(ready_payload.get("metrics"), dict) else {}
        self._last_spawn_metrics = {
            "process_start_return_ms": process_start_return_ms,
            "analyzer_import_ms": ready_metrics.get("analyzer_import_ms"),
            "child_entry_ms": ready_metrics.get("child_entry_ms"),
            "total_worker_ms": ready_metrics.get("total_worker_ms"),
        }
        return self._last_spawn_metrics

    def _cleanup_process_locked(self, process: Any, parent_conn: Any, *, timeout_seconds: float) -> dict[str, Any]:
        state = {
            "terminated": False,
            "killed": False,
            "process_exitcode": getattr(process, "exitcode", None),
            "process_exited": False,
            "alive": bool(getattr(process, "is_alive", lambda: False)()),
        }
        try:
            if getattr(process, "is_alive", lambda: False)():
                process.terminate()
                state["terminated"] = True
                process.join(timeout_seconds)
        except Exception:
            pass
        try:
            if getattr(process, "is_alive", lambda: False)():
                process.kill()
                state["killed"] = True
                process.join(timeout_seconds)
        except Exception:
            pass
        state["alive"] = bool(getattr(process, "is_alive", lambda: False)())
        state["process_exited"] = not state["alive"]
        state["process_exitcode"] = getattr(process, "exitcode", None)
        try:
            parent_conn.close()
        except Exception:
            pass
        try:
            process.close()
        except Exception:
            pass
        return state

    def _reset_worker_locked(self) -> dict[str, Any]:
        process = self._process
        parent_conn = self._parent_conn
        child_conn = self._child_conn
        self._process = None
        self._parent_conn = None
        self._child_conn = None
        self._ready = False
        if process is None or parent_conn is None:
            return {"terminated": False, "killed": False, "process_exitcode": None, "alive": False, "process_exited": True}
        return self._cleanup_process_locked(process, parent_conn, timeout_seconds=self._recovery_timeout_seconds)

    def start(self) -> dict[str, Any]:
        with self._lock:
            if self._started and self._ready:
                return {
                    "worker_generation": self._generation,
                    "spawn_metrics": dict(self._last_spawn_metrics),
                }
            if not self._dispatcher_started:
                self._dispatcher.start()
                self._dispatcher_started = True
            if self._process is not None or self._parent_conn is not None:
                self._reset_worker_locked()

        startup_started_at = time.perf_counter()
        with self._lock:
            spawn_metrics = self._spawn_worker_locked()
            self._started = True
        logger.info("[PERF] analyzer_runtime_start_ms scan_id=runtime value=%s", _elapsed_ms(startup_started_at))
        _log_perf("runtime", f"{self.analyzer_name}_worker_spawn_ms", spawn_metrics.get("process_start_return_ms"))
        _log_perf("runtime", f"{self.analyzer_name}_worker_import_ms", spawn_metrics.get("analyzer_import_ms"))
        return {
            "worker_generation": self._generation,
            "spawn_metrics": spawn_metrics,
            "runtime_start_ms": _elapsed_ms(startup_started_at),
        }

    def _set_future_result(self, job: _WorkerJob, result: dict[str, Any], state: dict[str, Any]) -> None:
        if job.future.done():
            return
        job.future.set_result({
            "result": result,
            "state": state,
        })

    def _build_state(
        self,
        *,
        job: _WorkerJob,
        timed_out: bool,
        analyzer_error: bool,
        result_received: bool,
        process_state: dict[str, Any] | None = None,
        worker_restarted: bool = False,
        child_metrics: dict[str, Any] | None = None,
        queue_wait_ms: int = 0,
        dispatch_ms: int = 0,
        response_ms: int = 0,
    ) -> dict[str, Any]:
        process_state = process_state or {}
        child_metrics = child_metrics or {}
        finalized = True
        state = {
            "state": "finalized" if finalized else "running",
            "child_started": True,
            "result_ready": result_received,
            "result_received": result_received,
            "analyzer_error": analyzer_error,
            "process_exited": bool(process_state.get("process_exited", False)),
            "timed_out": timed_out,
            "alive": bool(process_state.get("alive", True)),
            "terminated": bool(process_state.get("terminated", False)),
            "killed": bool(process_state.get("killed", False)),
            "process_exitcode": process_state.get("process_exitcode"),
            "final_alive": False,
            "finalized": finalized,
            "worker_generation": self._generation,
            "parent_process_start_ms": None,
            "process_start_begin_ms": None,
            "process_start_return_ms": self._last_spawn_metrics.get("process_start_return_ms"),
            "child_entry_ms": child_metrics.get("child_entry_ms"),
            "analyzer_import_ms": child_metrics.get("analyzer_import_ms"),
            "analyzer_execution_ms": child_metrics.get("analyzer_execution_ms"),
            "result_send_ms": child_metrics.get("response_send_ms"),
            "result_receive_ms": response_ms,
            "process_exit_ms": child_metrics.get("process_exit_ms"),
            "child_boot_ms": child_metrics.get("child_boot_ms"),
            "total_worker_ms": child_metrics.get("total_worker_ms"),
            "worker_queue_wait_ms": queue_wait_ms,
            "worker_dispatch_ms": dispatch_ms,
            "worker_response_ms": response_ms,
            "modality_total_ms": _elapsed_ms(job.submitted_at),
            "worker_restarted": worker_restarted,
        }
        return state

    def _dispatch_job(self, job: _WorkerJob) -> dict[str, Any]:
        queue_wait_ms = _elapsed_ms(job.submitted_at)
        if time.perf_counter() >= job.deadline_at:
            state = self._build_state(
                job=job,
                timed_out=True,
                analyzer_error=False,
                result_received=False,
                process_state={"process_exited": False, "alive": True, "terminated": False, "killed": False, "process_exitcode": None},
                worker_restarted=False,
                child_metrics={},
                queue_wait_ms=queue_wait_ms,
                dispatch_ms=0,
                response_ms=0,
            )
            _log_perf(job.scan_id, f"{self.analyzer_name}_worker_queue_wait_ms", queue_wait_ms)
            _log_perf(job.scan_id, f"{self.analyzer_name}_worker_dispatch_ms", 0)
            _log_perf(job.scan_id, f"{self.analyzer_name}_worker_response_ms", 0)
            _log_perf(job.scan_id, f"{self.analyzer_name}_modality_total_ms", state["modality_total_ms"])
            _log_perf(job.scan_id, f"{self.analyzer_name}_worker_generation", state["worker_generation"])
            _log_perf(job.scan_id, f"{self.analyzer_name}_result_received", state["result_received"])
            _log_perf(job.scan_id, f"{self.analyzer_name}_process_exitcode", state["process_exitcode"])
            _log_perf(job.scan_id, f"{self.analyzer_name}_timed_out", state["timed_out"])
            _log_perf(job.scan_id, f"{self.analyzer_name}_analyzer_error", state["analyzer_error"])
            _log_perf(job.scan_id, f"{self.analyzer_name}_worker_restarted", state["worker_restarted"])
            return {
                "result": _timeout_placeholder(self.analyzer_name),
                "state": state,
            }

        with self._lock:
            process = self._process
            parent_conn = self._parent_conn
            generation = self._generation

        if process is None or parent_conn is None:
            process_state = self._reset_worker_locked()
            state = self._build_state(
                job=job,
                timed_out=False,
                analyzer_error=True,
                result_received=False,
                process_state=process_state,
                worker_restarted=True,
                child_metrics={},
                queue_wait_ms=queue_wait_ms,
                dispatch_ms=0,
                response_ms=0,
            )
            _log_perf(job.scan_id, f"{self.analyzer_name}_worker_queue_wait_ms", queue_wait_ms)
            _log_perf(job.scan_id, f"{self.analyzer_name}_worker_dispatch_ms", 0)
            _log_perf(job.scan_id, f"{self.analyzer_name}_worker_response_ms", 0)
            _log_perf(job.scan_id, f"{self.analyzer_name}_modality_total_ms", state["modality_total_ms"])
            _log_perf(job.scan_id, f"{self.analyzer_name}_worker_generation", state["worker_generation"])
            _log_perf(job.scan_id, f"{self.analyzer_name}_result_received", state["result_received"])
            _log_perf(job.scan_id, f"{self.analyzer_name}_process_exitcode", state["process_exitcode"])
            _log_perf(job.scan_id, f"{self.analyzer_name}_timed_out", state["timed_out"])
            _log_perf(job.scan_id, f"{self.analyzer_name}_analyzer_error", state["analyzer_error"])
            _log_perf(job.scan_id, f"{self.analyzer_name}_worker_restarted", state["worker_restarted"])
            return {
                "result": _error_placeholder(self.analyzer_name),
                "state": state,
            }

        dispatch_started_at = time.perf_counter()
        try:
            parent_conn.send(
                {
                    "type": "job",
                    "job_id": job.job_id,
                    "scan_id": job.scan_id,
                    "path": job.path,
                    "worker_generation": generation,
                    "deadline_at": job.deadline_at,
                }
            )
        except Exception:
            process_state = self._reset_worker_locked()
            state = self._build_state(
                job=job,
                timed_out=False,
                analyzer_error=True,
                result_received=False,
                process_state=process_state,
                worker_restarted=True,
                child_metrics={},
                queue_wait_ms=queue_wait_ms,
                dispatch_ms=_elapsed_ms(dispatch_started_at),
                response_ms=0,
            )
            _log_perf(job.scan_id, f"{self.analyzer_name}_worker_queue_wait_ms", queue_wait_ms)
            _log_perf(job.scan_id, f"{self.analyzer_name}_worker_dispatch_ms", _elapsed_ms(dispatch_started_at))
            _log_perf(job.scan_id, f"{self.analyzer_name}_worker_response_ms", 0)
            _log_perf(job.scan_id, f"{self.analyzer_name}_modality_total_ms", state["modality_total_ms"])
            _log_perf(job.scan_id, f"{self.analyzer_name}_worker_generation", state["worker_generation"])
            _log_perf(job.scan_id, f"{self.analyzer_name}_result_received", state["result_received"])
            _log_perf(job.scan_id, f"{self.analyzer_name}_process_exitcode", state["process_exitcode"])
            _log_perf(job.scan_id, f"{self.analyzer_name}_timed_out", state["timed_out"])
            _log_perf(job.scan_id, f"{self.analyzer_name}_analyzer_error", state["analyzer_error"])
            _log_perf(job.scan_id, f"{self.analyzer_name}_worker_restarted", state["worker_restarted"])
            return {
                "result": _error_placeholder(self.analyzer_name),
                "state": state,
            }

        dispatch_ms = _elapsed_ms(dispatch_started_at)
        response_started_at = time.perf_counter()
        payload: dict[str, Any] | None = None
        timed_out = False
        analyzer_error = False
        worker_restarted = False
        child_metrics: dict[str, Any] = {}
        process_state: dict[str, Any] = {
            "alive": True,
            "terminated": False,
            "killed": False,
            "process_exited": False,
            "process_exitcode": getattr(process, "exitcode", None),
        }

        while True:
            remaining = job.deadline_at - time.perf_counter()
            if remaining <= 0:
                timed_out = True
                break
            ready_handles = _wait_handles([parent_conn, process.sentinel], timeout=min(self._poll_interval_seconds, remaining))
            if parent_conn in ready_handles:
                try:
                    payload = parent_conn.recv()
                except EOFError:
                    payload = None
                break
            if process.sentinel in ready_handles and not process.is_alive():
                break

        response_ms = _elapsed_ms(response_started_at)
        if payload is not None and isinstance(payload, dict) and payload.get("type") == "result":
            if payload.get("worker_generation") != generation or payload.get("job_id") != job.job_id:
                analyzer_error = True
                payload = None
            elif payload.get("ok") is True and isinstance(payload.get("result"), dict):
                child_metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
                process_state["alive"] = bool(process.is_alive())
                process_state["process_exited"] = not process_state["alive"]
                process_state["process_exitcode"] = process.exitcode
                state = self._build_state(
                    job=job,
                    timed_out=False,
                    analyzer_error=False,
                    result_received=True,
                    process_state=process_state,
                    worker_restarted=False,
                    child_metrics=child_metrics,
                    queue_wait_ms=queue_wait_ms,
                    dispatch_ms=dispatch_ms,
                    response_ms=response_ms,
                )
                _log_perf(job.scan_id, f"{self.analyzer_name}_worker_queue_wait_ms", queue_wait_ms)
                _log_perf(job.scan_id, f"{self.analyzer_name}_worker_dispatch_ms", dispatch_ms)
                _log_perf(job.scan_id, f"{self.analyzer_name}_analyzer_execution_ms", child_metrics.get("analyzer_execution_ms"))
                _log_perf(job.scan_id, f"{self.analyzer_name}_worker_response_ms", response_ms)
                _log_perf(job.scan_id, f"{self.analyzer_name}_modality_total_ms", state["modality_total_ms"])
                _log_perf(job.scan_id, f"{self.analyzer_name}_worker_generation", state["worker_generation"])
                _log_perf(job.scan_id, f"{self.analyzer_name}_result_received", state["result_received"])
                _log_perf(job.scan_id, f"{self.analyzer_name}_process_exitcode", state["process_exitcode"])
                _log_perf(job.scan_id, f"{self.analyzer_name}_timed_out", state["timed_out"])
                _log_perf(job.scan_id, f"{self.analyzer_name}_analyzer_error", state["analyzer_error"])
                _log_perf(job.scan_id, f"{self.analyzer_name}_worker_restarted", state["worker_restarted"])
                return {
                    "result": payload["result"],
                    "state": state,
                }
            else:
                analyzer_error = True
        else:
            analyzer_error = True

        if not timed_out:
            process_state = self._cleanup_process_locked(process, parent_conn, timeout_seconds=self._recovery_timeout_seconds)
            if not process_state.get("alive"):
                worker_restarted = True
            with self._lock:
                self._process = None
                self._parent_conn = None
                self._child_conn = None
                self._ready = False
            if not self._stop_event.is_set():
                try:
                    self._spawn_worker_locked()
                    worker_restarted = True
                except Exception as exc:
                    logger.error(
                        "analysis_worker_restart_failed analyzer=%s error_type=%s",
                        self.analyzer_name,
                        type(exc).__name__,
                    )

        if timed_out:
            process_state = self._cleanup_process_locked(process, parent_conn, timeout_seconds=self._recovery_timeout_seconds)
            worker_restarted = True
            with self._lock:
                self._process = None
                self._parent_conn = None
                self._child_conn = None
                self._ready = False
            if not self._stop_event.is_set():
                try:
                    self._spawn_worker_locked()
                except Exception as exc:
                    logger.error(
                        "analysis_worker_restart_failed analyzer=%s error_type=%s",
                        self.analyzer_name,
                        type(exc).__name__,
                    )

        state = self._build_state(
            job=job,
            timed_out=timed_out,
            analyzer_error=analyzer_error,
            result_received=False,
            process_state=process_state,
            worker_restarted=worker_restarted,
            child_metrics=child_metrics,
            queue_wait_ms=queue_wait_ms,
            dispatch_ms=dispatch_ms,
            response_ms=response_ms,
        )
        _log_perf(job.scan_id, f"{self.analyzer_name}_worker_queue_wait_ms", queue_wait_ms)
        _log_perf(job.scan_id, f"{self.analyzer_name}_worker_dispatch_ms", dispatch_ms)
        _log_perf(job.scan_id, f"{self.analyzer_name}_analyzer_execution_ms", child_metrics.get("analyzer_execution_ms"))
        _log_perf(job.scan_id, f"{self.analyzer_name}_worker_response_ms", response_ms)
        _log_perf(job.scan_id, f"{self.analyzer_name}_modality_total_ms", state["modality_total_ms"])
        _log_perf(job.scan_id, f"{self.analyzer_name}_worker_generation", state["worker_generation"])
        _log_perf(job.scan_id, f"{self.analyzer_name}_result_received", state["result_received"])
        _log_perf(job.scan_id, f"{self.analyzer_name}_process_exitcode", state["process_exitcode"])
        _log_perf(job.scan_id, f"{self.analyzer_name}_timed_out", state["timed_out"])
        _log_perf(job.scan_id, f"{self.analyzer_name}_analyzer_error", state["analyzer_error"])
        _log_perf(job.scan_id, f"{self.analyzer_name}_worker_restarted", state["worker_restarted"])
        if timed_out:
            return {
                "result": _timeout_placeholder(self.analyzer_name),
                "state": state,
            }
        return {
            "result": _error_placeholder(self.analyzer_name),
            "state": state,
        }

    def _dispatch_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                job = self._queue.get(timeout=self._poll_interval_seconds)
            except queue.Empty:
                with self._lock:
                    if self._started and self._process is not None and not self._process.is_alive() and not self._stop_event.is_set():
                        try:
                            self._reset_worker_locked()
                        except Exception:
                            pass
                        try:
                            self._spawn_worker_locked()
                        except Exception:
                            pass
                continue
            if job is _SHUTDOWN_JOB:
                break
            if not isinstance(job, _WorkerJob):
                continue
            try:
                completion = self._dispatch_job(job)
            except Exception as exc:
                completion = {
                    "result": _error_placeholder(self.analyzer_name),
                    "state": self._build_state(
                        job=job,
                        timed_out=False,
                        analyzer_error=True,
                        result_received=False,
                        process_state={"alive": False, "terminated": False, "killed": False, "process_exited": True, "process_exitcode": None},
                        worker_restarted=False,
                        child_metrics={},
                    ),
                }
                logger.error(
                    "analysis_worker_dispatch_failed analyzer=%s error_type=%s",
                    self.analyzer_name,
                    type(exc).__name__,
                )
            if not job.future.done():
                job.future.set_result(completion)

    def submit(self, *, scan_id: str, path: str | None, deadline_at: float) -> Future:
        if self._stop_event.is_set():
            raise AnalysisRuntimeUnavailable(f"{self.analyzer_name}_runtime_stopped")
        future: Future = Future()
        job = _WorkerJob(
            job_id=uuid.uuid4().hex,
            scan_id=scan_id,
            path=path,
            deadline_at=deadline_at,
            submitted_at=time.perf_counter(),
            future=future,
        )
        try:
            self._queue.put_nowait(job)
        except queue.Full as exc:
            raise AnalysisRuntimeBusyError(f"{self.analyzer_name}_queue_full") from exc
        return future

    def health(self) -> dict[str, Any]:
        with self._lock:
            process = self._process
            ready = self._ready
            generation = self._generation
        return {
            "ready": ready,
            "worker_generation": generation,
            "process_alive": bool(process.is_alive()) if process is not None else False,
            "process_exitcode": getattr(process, "exitcode", None) if process is not None else None,
        }

    def shutdown(self) -> None:
        self._stop_event.set()
        try:
            self._queue.put_nowait(_SHUTDOWN_JOB)
        except queue.Full:
            pass
        if self._dispatcher_started:
            self._dispatcher.join(timeout=self._recovery_timeout_seconds)
        with self._lock:
            process = self._process
            parent_conn = self._parent_conn
            self._process = None
            self._parent_conn = None
            self._child_conn = None
            self._ready = False
        if process is not None and parent_conn is not None:
            try:
                if process.is_alive():
                    parent_conn.send({"type": "shutdown"})
                    process.join(self._recovery_timeout_seconds)
            except Exception:
                pass
            try:
                if process.is_alive():
                    process.terminate()
                    process.join(self._recovery_timeout_seconds)
            except Exception:
                pass
            try:
                if process.is_alive():
                    process.kill()
                    process.join(self._recovery_timeout_seconds)
            except Exception:
                pass
            try:
                parent_conn.close()
            except Exception:
                pass
            try:
                process.close()
            except Exception:
                pass


class WarmAnalyzerRuntime:
    def __init__(
        self,
        *,
        context_factory: Callable[[], Any] | None = None,
        worker_entry_map: dict[str, Callable[..., Any]] | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._started = False
        self._stopped = False
        self._startup_started_at: float | None = None
        self._startup_ready_at: float | None = None
        worker_entry_map = worker_entry_map or {}
        self._supervisors = {
            "video": WorkerSupervisor("video", context_factory=context_factory, worker_entry=worker_entry_map.get("video")),
            "audio": WorkerSupervisor("audio", context_factory=context_factory, worker_entry=worker_entry_map.get("audio")),
            "image": WorkerSupervisor("image", context_factory=context_factory, worker_entry=worker_entry_map.get("image")),
        }

    def start(self) -> dict[str, Any]:
        with self._lock:
            if self._started and not self._stopped and all(supervisor.health()["ready"] for supervisor in self._supervisors.values()):
                return self.health()
            if self._stopped:
                raise AnalysisRuntimeUnavailable("analysis_runtime_stopped")
            self._startup_started_at = time.perf_counter()

        startup_started_at = self._startup_started_at
        assert startup_started_at is not None

        startup_results: dict[str, Any] = {}
        errors: list[Exception] = []
        with ThreadPoolExecutor(max_workers=len(self._supervisors)) as executor:
            futures = {
                analyzer_name: executor.submit(supervisor.start)
                for analyzer_name, supervisor in self._supervisors.items()
            }
            for analyzer_name, future in futures.items():
                try:
                    startup_results[analyzer_name] = future.result()
                except Exception as exc:
                    errors.append(exc)
                    startup_results[analyzer_name] = {"error_type": type(exc).__name__}

        if errors:
            if not self._started:
                self.shutdown()
                raise AnalysisRuntimeStartupError("analysis_runtime_start_failed") from errors[0]
            logger.warning("analysis_runtime_partial_start_failed error_type=%s", type(errors[0]).__name__)

        with self._lock:
            self._started = True
            self._stopped = False
            self._startup_ready_at = time.perf_counter()

        _log_perf("runtime", "analyzer_runtime_start_ms", _elapsed_ms(startup_started_at))
        _log_perf("runtime", "analyzer_runtime_ready_ms", _elapsed_ms(startup_started_at))
        return {
            "analyzer_runtime_start_ms": _elapsed_ms(startup_started_at),
            "analyzer_runtime_ready_ms": _elapsed_ms(startup_started_at),
            "workers": startup_results,
        }

    def run_scan(self, scan_id: str, media: Any, *, deadline_seconds: float) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
        if not self._started:
            self.start()

        deadline_at = time.perf_counter() + deadline_seconds
        jobs: dict[str, Future] = {}
        for analyzer_name, supervisor, path in [
            ("video", self._supervisors["video"], getattr(media, "video", None)),
            ("audio", self._supervisors["audio"], getattr(media, "audio", None)),
            ("image", self._supervisors["image"], getattr(media, "image", None)),
        ]:
            try:
                jobs[analyzer_name] = supervisor.submit(scan_id=scan_id, path=path, deadline_at=deadline_at)
            except AnalysisRuntimeBusyError:
                future: Future = Future()
                future.set_result(
                    {
                        "result": _error_placeholder(analyzer_name),
                        "state": {
                            "state": "finalized",
                            "child_started": True,
                            "result_ready": False,
                            "result_received": False,
                            "analyzer_error": True,
                            "process_exited": False,
                            "timed_out": False,
                            "alive": True,
                            "terminated": False,
                            "killed": False,
                            "process_exitcode": None,
                            "final_alive": False,
                            "finalized": True,
                            "worker_generation": self._supervisors[analyzer_name]._generation,
                            "worker_queue_wait_ms": 0,
                            "worker_dispatch_ms": 0,
                            "worker_response_ms": 0,
                            "modality_total_ms": 0,
                            "worker_restarted": False,
                            "process_start_return_ms": None,
                            "child_entry_ms": None,
                            "analyzer_import_ms": None,
                            "analyzer_execution_ms": None,
                            "result_send_ms": None,
                            "result_receive_ms": None,
                            "process_exit_ms": None,
                            "child_boot_ms": None,
                            "total_worker_ms": None,
                        },
                    }
                )
                jobs[analyzer_name] = future

        results: dict[str, dict[str, Any]] = {}
        worker_states: dict[str, dict[str, Any]] = {}
        for analyzer_name, future in jobs.items():
            remaining = max(0.0, deadline_at - time.perf_counter())
            try:
                completion = future.result(timeout=remaining)
            except FuturesTimeoutError:
                completion = {
                    "result": _timeout_placeholder(analyzer_name),
                    "state": {
                        "state": "finalized",
                        "child_started": True,
                        "result_ready": False,
                        "result_received": False,
                        "analyzer_error": False,
                        "process_exited": False,
                        "timed_out": True,
                        "alive": False,
                        "terminated": False,
                        "killed": False,
                        "process_exitcode": None,
                        "final_alive": False,
                        "finalized": True,
                        "worker_generation": self._supervisors[analyzer_name]._generation,
                        "worker_queue_wait_ms": 0,
                        "worker_dispatch_ms": 0,
                        "worker_response_ms": 0,
                        "modality_total_ms": 0,
                        "worker_restarted": False,
                    },
                }
            except Exception:
                completion = {
                    "result": _timeout_placeholder(analyzer_name),
                    "state": {
                        "state": "finalized",
                        "child_started": True,
                        "result_ready": False,
                        "result_received": False,
                        "analyzer_error": True,
                        "process_exited": False,
                        "timed_out": True,
                        "alive": False,
                        "terminated": False,
                        "killed": False,
                        "process_exitcode": None,
                        "final_alive": False,
                        "finalized": True,
                        "worker_generation": self._supervisors[analyzer_name]._generation,
                        "worker_queue_wait_ms": 0,
                        "worker_dispatch_ms": 0,
                        "worker_response_ms": 0,
                        "modality_total_ms": 0,
                        "worker_restarted": False,
                    },
                }
            results[analyzer_name] = completion["result"]
            worker_states[analyzer_name] = completion["state"]
        return results, worker_states

    def is_ready(self) -> bool:
        with self._lock:
            return self._started and not self._stopped and all(supervisor.health()["ready"] for supervisor in self._supervisors.values())

    def health(self) -> dict[str, Any]:
        return {
            "ready": self.is_ready(),
            "started": self._started,
            "stopped": self._stopped,
            "workers": {name: supervisor.health() for name, supervisor in self._supervisors.items()},
            "startup_started_at": self._startup_started_at,
            "startup_ready_at": self._startup_ready_at,
        }

    def shutdown(self) -> None:
        with self._lock:
            if self._stopped:
                return
            self._stopped = True
        for supervisor in self._supervisors.values():
            supervisor.shutdown()
        with self._lock:
            self._started = False


_RUNTIME: WarmAnalyzerRuntime | None = None
_RUNTIME_LOCK = threading.Lock()


def get_runtime() -> WarmAnalyzerRuntime:
    global _RUNTIME
    with _RUNTIME_LOCK:
        if _RUNTIME is None:
            _RUNTIME = WarmAnalyzerRuntime()
        return _RUNTIME
