from __future__ import annotations

import concurrent.futures
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FuturesTimeoutError, wait as futures_wait
from collections import OrderedDict
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
    worker_generation: int | None = None


_SHUTDOWN_JOB = object()


class WorkerSupervisor:
    def __init__(
        self,
        analyzer_name: str,
        *,
        queue_capacity: int = 4,
        startup_timeout_seconds: float = 15.0,
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
        self._jobs_by_id: dict[str, _WorkerJob] = {}
        self._completed_jobs: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._expired_jobs: OrderedDict[str, float] = OrderedDict()
        self._active_job_id: str | None = None
        self._active_worker_generation: int | None = None
        self._completed_job_limit = 2048
        self._expired_job_limit = 2048
        self._job_retention_seconds = 300.0
        self._restart_thread: threading.Thread | None = None
        self._restart_generation: int | None = None
        self._restart_started_at: float | None = None
        self._restart_in_progress = False
        self._restart_ready = False
        self._restart_error_type: str | None = None
        self._restart_token = 0
        self._active_restart_token: int | None = None
        self._restart_candidate_process = None
        self._restart_candidate_parent_conn = None

    def _prune_job_registries_locked(self) -> None:
        while len(self._completed_jobs) > self._completed_job_limit:
            self._completed_jobs.popitem(last=False)
        while len(self._expired_jobs) > self._expired_job_limit:
            self._expired_jobs.popitem(last=False)

    def _snapshot_process_locked(self) -> dict[str, Any]:
        process = self._process
        if process is None:
            return {
                "terminated": False,
                "killed": False,
                "process_exitcode": None,
                "process_exited": True,
                "alive": False,
            }
        alive = bool(process.is_alive())
        return {
            "terminated": False,
            "killed": False,
            "process_exitcode": getattr(process, "exitcode", None),
            "process_exited": not alive,
            "alive": alive,
        }

    def _spawn_worker_candidate(self, generation: int, *, restart_token: int | None = None) -> tuple[Any, Any, Any, dict[str, Any]]:
        ctx = self._context_factory()
        parent_conn, child_conn = ctx.Pipe(duplex=True)
        process_started_at = time.perf_counter()
        process = ctx.Process(
            target=self._worker_entry,
            args=(child_conn, self.analyzer_name, generation),
            daemon=True,
        )
        process.start()
        process_start_return_ms = _elapsed_ms(process_started_at)
        ready_wait_started_at = time.perf_counter()
        try:
            child_conn.close()
        except Exception:
            pass
        if restart_token is not None:
            with self._lock:
                if self._stop_event.is_set() or self._active_restart_token != restart_token:
                    process_state = None
                else:
                    self._restart_candidate_process = process
                    self._restart_candidate_parent_conn = parent_conn
                    process_state = True
            if process_state is None:
                self._cleanup_process_locked(process, parent_conn, timeout_seconds=self._recovery_timeout_seconds)
                raise AnalysisRuntimeStartupError(f"{self.analyzer_name}_worker_start_stale")

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
            process_state = self._cleanup_process_locked(process, parent_conn, timeout_seconds=self._recovery_timeout_seconds)
            logger.error(
                "analysis_worker_start_failed analyzer=%s worker_generation=%s error_type=%s process_exitcode=%s elapsed_startup_ms=%s",
                self.analyzer_name,
                generation,
                "startup_timeout" if ready_payload is None else str(ready_payload.get("error_type") or "invalid_ready_payload"),
                process_state.get("process_exitcode"),
                _elapsed_ms(process_started_at),
            )
            raise AnalysisRuntimeStartupError(f"{self.analyzer_name}_worker_start_failed")

        ready_metrics = ready_payload.get("metrics") if isinstance(ready_payload.get("metrics"), dict) else {}
        spawn_metrics = {
            "process_start_return_ms": process_start_return_ms,
            "analyzer_import_ms": ready_metrics.get("analyzer_import_ms"),
            "child_entry_ms": ready_metrics.get("child_entry_ms"),
            "total_worker_ms": ready_metrics.get("total_worker_ms"),
            "ready_wait_ms": _elapsed_ms(ready_wait_started_at),
        }
        return process, parent_conn, child_conn, spawn_metrics

    def _publish_spawned_worker_locked(self, generation: int, process: Any, parent_conn: Any, child_conn: Any, spawn_metrics: dict[str, Any]) -> None:
        self._generation = generation
        self._process = process
        self._parent_conn = parent_conn
        self._child_conn = child_conn
        self._ready = True
        self._last_spawn_metrics = spawn_metrics

    def _spawn_worker_locked(self) -> dict[str, Any]:
        generation = self._generation + 1
        process, parent_conn, child_conn, spawn_metrics = self._spawn_worker_candidate(generation)
        self._publish_spawned_worker_locked(generation, process, parent_conn, child_conn, spawn_metrics)
        return self._last_spawn_metrics

    def _cleanup_process_locked(self, process: Any, parent_conn: Any, *, timeout_seconds: float) -> dict[str, Any]:
        cleanup_started_at = time.perf_counter()
        terminate_ms = 0
        join_ms = 0
        kill_ms = 0
        final_join_ms = 0
        state = {
            "terminated": False,
            "killed": False,
            "process_exitcode": getattr(process, "exitcode", None),
            "process_exited": False,
            "alive": bool(getattr(process, "is_alive", lambda: False)()),
        }
        try:
            if getattr(process, "is_alive", lambda: False)():
                terminate_started_at = time.perf_counter()
                process.terminate()
                terminate_ms = _elapsed_ms(terminate_started_at)
                state["terminated"] = True
                join_started_at = time.perf_counter()
                process.join(timeout_seconds)
                join_ms = _elapsed_ms(join_started_at)
        except Exception:
            pass
        try:
            if getattr(process, "is_alive", lambda: False)():
                kill_started_at = time.perf_counter()
                process.kill()
                kill_ms = _elapsed_ms(kill_started_at)
                state["killed"] = True
                final_join_started_at = time.perf_counter()
                process.join(timeout_seconds)
                final_join_ms = _elapsed_ms(final_join_started_at)
        except Exception:
            pass
        state["alive"] = bool(getattr(process, "is_alive", lambda: False)())
        state["process_exited"] = not state["alive"]
        state["process_exitcode"] = getattr(process, "exitcode", None)
        state["old_worker_cleanup_ms"] = _elapsed_ms(cleanup_started_at)
        state["old_worker_terminate_ms"] = terminate_ms
        state["old_worker_join_ms"] = join_ms
        state["old_worker_kill_ms"] = kill_ms
        state["old_worker_final_join_ms"] = final_join_ms
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

    def _restart_worker_async(self, expected_generation: int, scheduled_at: float) -> None:
        schedule_ms = _elapsed_ms(scheduled_at)
        restart_token: int | None
        with self._lock:
            restart_token = self._active_restart_token
            if (
                self._stop_event.is_set()
                or restart_token is None
                or self._restart_generation != expected_generation
                or self._generation != expected_generation
                or self._process is not None
            ):
                self._restart_in_progress = False
                self._restart_ready = self._ready
                return
        candidate_process = None
        candidate_parent_conn = None
        try:
            next_generation = expected_generation + 1
            candidate_process, candidate_parent_conn, candidate_child_conn, spawn_metrics = self._spawn_worker_candidate(
                next_generation,
                restart_token=restart_token,
            )
            publish_candidate = False
            with self._lock:
                if (
                    not self._stop_event.is_set()
                    and self._active_restart_token == restart_token
                    and self._restart_generation == expected_generation
                    and self._generation == expected_generation
                    and self._process is None
                ):
                    self._publish_spawned_worker_locked(
                        next_generation,
                        candidate_process,
                        candidate_parent_conn,
                        candidate_child_conn,
                        spawn_metrics,
                    )
                    self._restart_candidate_process = None
                    self._restart_candidate_parent_conn = None
                    publish_candidate = True
                    self._restart_ready = True
                    self._restart_error_type = None
                    restart_generation = self._generation
                else:
                    self._restart_candidate_process = None
                    self._restart_candidate_parent_conn = None
                    self._restart_ready = self._ready
                    restart_generation = self._generation
            if not publish_candidate:
                self._cleanup_process_locked(candidate_process, candidate_parent_conn, timeout_seconds=self._recovery_timeout_seconds)
                candidate_process = None
                candidate_parent_conn = None
                return
            _log_perf("runtime", f"{self.analyzer_name}_replacement_worker_schedule_ms", schedule_ms)
            _log_perf("runtime", f"{self.analyzer_name}_replacement_worker_spawn_ms", spawn_metrics.get("process_start_return_ms"))
            _log_perf("runtime", f"{self.analyzer_name}_replacement_worker_import_ms", spawn_metrics.get("analyzer_import_ms"))
            _log_perf("runtime", f"{self.analyzer_name}_replacement_worker_ready_wait_ms", spawn_metrics.get("ready_wait_ms"))
            _log_perf("runtime", f"{self.analyzer_name}_worker_restart_generation", restart_generation)
        except Exception as exc:
            with self._lock:
                if self._active_restart_token == restart_token:
                    self._restart_ready = False
                    self._restart_error_type = type(exc).__name__
            logger.error(
                "analysis_worker_async_restart_failed analyzer=%s worker_generation=%s error_type=%s",
                self.analyzer_name,
                expected_generation + 1,
                type(exc).__name__,
            )
        finally:
            with self._lock:
                if self._active_restart_token == restart_token:
                    self._active_restart_token = None
                    self._restart_in_progress = False
                if self._restart_candidate_process is candidate_process:
                    self._restart_candidate_process = None
                    self._restart_candidate_parent_conn = None

    def _schedule_restart_locked(self, expected_generation: int) -> bool:
        if self._stop_event.is_set():
            return False
        if self._restart_in_progress and self._restart_generation == expected_generation:
            return False
        if self._process is not None:
            return False
        scheduled_at = time.perf_counter()
        self._restart_token += 1
        self._restart_generation = expected_generation
        self._active_restart_token = self._restart_token
        self._restart_started_at = scheduled_at
        self._restart_in_progress = True
        self._restart_ready = False
        self._restart_error_type = None
        self._ready = False
        self._restart_thread = threading.Thread(
            target=self._restart_worker_async,
            args=(expected_generation, scheduled_at),
            name=f"{self.analyzer_name}-analysis-restart",
            daemon=True,
        )
        return True

    def _start_restart_thread(self, thread: threading.Thread | None) -> None:
        if thread is None:
            return
        try:
            thread.start()
        except RuntimeError:
            pass

    def _wait_for_ready_until_deadline(self, deadline_at: float) -> bool:
        while time.perf_counter() < deadline_at:
            restart_thread = None
            remaining = max(0.0, deadline_at - time.perf_counter())
            acquired = self._lock.acquire(timeout=min(self._poll_interval_seconds, remaining))
            if not acquired:
                continue
            try:
                if self._ready and self._process is not None and self._parent_conn is not None:
                    return True
                if not self._restart_in_progress and self._process is None:
                    if self._schedule_restart_locked(self._generation):
                        restart_thread = self._restart_thread
            finally:
                self._lock.release()
            self._start_restart_thread(restart_thread)
            time.sleep(min(self._poll_interval_seconds, max(0.0, deadline_at - time.perf_counter())))
        return False

    def _finalize_job_locked(
        self,
        job: _WorkerJob,
        *,
        timed_out: bool,
        analyzer_error: bool,
        result_received: bool,
        process_state: dict[str, Any] | None = None,
        worker_restarted: bool = False,
        child_metrics: dict[str, Any] | None = None,
        queue_wait_ms: int = 0,
        dispatch_ms: int = 0,
        response_ms: int = 0,
        cleanup_worker: bool = True,
        restart_worker: bool = True,
        clear_active: bool = True,
    ) -> dict[str, Any]:
        process_to_cleanup = None
        parent_conn_to_cleanup = None
        restart_thread = None
        with self._lock:
            cached = self._completed_jobs.get(job.job_id)
            if cached is not None:
                return cached

            process_state = process_state or {}
            child_metrics = child_metrics or {}
            if timed_out:
                self._expired_jobs[job.job_id] = time.perf_counter()
                self._expired_jobs.move_to_end(job.job_id)
                self._prune_job_registries_locked()

            process = self._process
            parent_conn = self._parent_conn

            if cleanup_worker and process is not None and parent_conn is not None:
                old_generation = self._generation
                process_to_cleanup = process
                parent_conn_to_cleanup = parent_conn
                worker_restarted = True
                self._process = None
                self._parent_conn = None
                self._child_conn = None
                self._ready = False
            elif not process_state:
                process_state = self._snapshot_process_locked()

            if clear_active and self._active_job_id == job.job_id:
                self._active_job_id = None
                self._active_worker_generation = None

        if process_to_cleanup is not None and parent_conn_to_cleanup is not None:
            process_state = self._cleanup_process_locked(
                process_to_cleanup,
                parent_conn_to_cleanup,
                timeout_seconds=self._recovery_timeout_seconds,
            )

        with self._lock:
            cached = self._completed_jobs.get(job.job_id)
            if cached is not None:
                return cached
            if process_to_cleanup is not None and restart_worker:
                if self._schedule_restart_locked(old_generation):
                    restart_thread = self._restart_thread

            state = self._build_state(
                job=job,
                timed_out=timed_out,
                analyzer_error=analyzer_error,
                result_received=result_received,
                process_state=process_state,
                worker_restarted=worker_restarted,
                child_metrics=child_metrics,
                queue_wait_ms=queue_wait_ms,
                dispatch_ms=dispatch_ms,
                response_ms=response_ms,
            )
            completion = {
                "result": _timeout_placeholder(self.analyzer_name) if timed_out else _error_placeholder(self.analyzer_name),
                "state": state,
            }
            if not job.future.done():
                job.future.set_result(completion)
            self._completed_jobs[job.job_id] = completion
            self._jobs_by_id.pop(job.job_id, None)
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
            _log_perf(job.scan_id, f"{self.analyzer_name}_old_worker_cleanup_ms", state.get("old_worker_cleanup_ms"))
            _log_perf(job.scan_id, f"{self.analyzer_name}_old_worker_terminate_ms", state.get("old_worker_terminate_ms"))
            _log_perf(job.scan_id, f"{self.analyzer_name}_old_worker_join_ms", state.get("old_worker_join_ms"))
            _log_perf(job.scan_id, f"{self.analyzer_name}_old_worker_kill_ms", state.get("old_worker_kill_ms"))
            _log_perf(job.scan_id, f"{self.analyzer_name}_old_worker_final_join_ms", state.get("old_worker_final_join_ms"))
            self._completed_jobs[job.job_id] = completion
            self._completed_jobs.move_to_end(job.job_id)
            self._prune_job_registries_locked()
            self._jobs_by_id.pop(job.job_id, None)
            if clear_active and self._active_job_id == job.job_id:
                self._active_job_id = None
                self._active_worker_generation = None
        self._start_restart_thread(restart_thread)
        return completion

    def start(self) -> dict[str, Any]:
        process_to_reset = None
        parent_conn_to_reset = None
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
                process_to_reset = self._process
                parent_conn_to_reset = self._parent_conn
                self._process = None
                self._parent_conn = None
                self._child_conn = None
                self._ready = False

        startup_started_at = time.perf_counter()
        if process_to_reset is not None and parent_conn_to_reset is not None:
            self._cleanup_process_locked(process_to_reset, parent_conn_to_reset, timeout_seconds=self._recovery_timeout_seconds)
        generation = None
        process = parent_conn = child_conn = None
        spawn_metrics: dict[str, Any] = {}
        with self._lock:
            generation = self._generation + 1
        process, parent_conn, child_conn, spawn_metrics = self._spawn_worker_candidate(generation)
        with self._lock:
            if self._stop_event.is_set():
                cleanup_process = process
                cleanup_parent_conn = parent_conn
            else:
                cleanup_process = None
                cleanup_parent_conn = None
                self._publish_spawned_worker_locked(generation, process, parent_conn, child_conn, spawn_metrics)
            self._started = True
        if cleanup_process is not None and cleanup_parent_conn is not None:
            self._cleanup_process_locked(cleanup_process, cleanup_parent_conn, timeout_seconds=self._recovery_timeout_seconds)
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
            "worker_process_alive": bool(process_state.get("alive", True)),
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
            "old_worker_cleanup_ms": process_state.get("old_worker_cleanup_ms"),
            "old_worker_terminate_ms": process_state.get("old_worker_terminate_ms"),
            "old_worker_join_ms": process_state.get("old_worker_join_ms"),
            "old_worker_kill_ms": process_state.get("old_worker_kill_ms"),
            "old_worker_final_join_ms": process_state.get("old_worker_final_join_ms"),
        }
        state.update(self._restart_state_snapshot())
        return state

    def _restart_state_snapshot(self) -> dict[str, Any]:
        return {
            "worker_restart_scheduled": self._restart_started_at is not None,
            "worker_restart_in_progress": self._restart_in_progress,
            "worker_restart_ready": self._restart_ready,
            "worker_restart_generation": self._restart_generation,
            "worker_restart_error_type": self._restart_error_type,
        }

    def _dispatch_job(self, job: _WorkerJob) -> dict[str, Any]:
        queue_wait_ms = _elapsed_ms(job.submitted_at)
        if time.perf_counter() >= job.deadline_at:
            return self._finalize_job_locked(
                job,
                timed_out=True,
                analyzer_error=False,
                result_received=False,
                process_state={"process_exited": False, "alive": True, "terminated": False, "killed": False, "process_exitcode": None},
                worker_restarted=False,
                child_metrics={},
                queue_wait_ms=queue_wait_ms,
                dispatch_ms=0,
                response_ms=0,
                cleanup_worker=False,
            )

        if not self._wait_for_ready_until_deadline(job.deadline_at):
            return self._finalize_job_locked(
                job,
                timed_out=True,
                analyzer_error=False,
                result_received=False,
                process_state=self._snapshot_process_locked(),
                worker_restarted=False,
                child_metrics={},
                queue_wait_ms=queue_wait_ms,
                dispatch_ms=0,
                response_ms=0,
                cleanup_worker=False,
                clear_active=False,
            )

        with self._lock:
            process = self._process
            parent_conn = self._parent_conn
            generation = self._generation

        if process is None or parent_conn is None:
            process_state = self._reset_worker_locked()
            return self._finalize_job_locked(
                job,
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

        with self._lock:
            if job.future.done() or (job.job_id in self._expired_jobs and job.job_id != self._active_job_id):
                cached = self._completed_jobs.get(job.job_id)
                if cached is not None:
                    return cached
                if job.future.done():
                    try:
                        completion = job.future.result()
                        if isinstance(completion, dict):
                            return completion
                    except Exception:
                        pass
            self._active_job_id = job.job_id
            self._active_worker_generation = generation
            job.worker_generation = generation

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
            return self._finalize_job_locked(
                job,
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
            if job.job_id in self._expired_jobs:
                timed_out = True
                break

        response_ms = _elapsed_ms(response_started_at)
        if payload is not None and isinstance(payload, dict) and payload.get("type") == "result":
            if payload.get("worker_generation") != generation or payload.get("job_id") != job.job_id:
                if not timed_out and job.job_id not in self._expired_jobs:
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
                with self._lock:
                    if self._active_job_id == job.job_id:
                        self._active_job_id = None
                        self._active_worker_generation = None
                return {
                    "result": payload["result"],
                    "state": state,
                }
            else:
                if not timed_out and job.job_id not in self._expired_jobs:
                    analyzer_error = True
        elif not timed_out and job.job_id not in self._expired_jobs:
            analyzer_error = True

        return self._finalize_job_locked(
            job,
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

    def _dispatch_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                job = self._queue.get(timeout=self._poll_interval_seconds)
            except queue.Empty:
                process_to_reset = None
                parent_conn_to_reset = None
                restart_thread = None
                with self._lock:
                    if self._started and self._process is not None and not self._process.is_alive() and not self._stop_event.is_set():
                        process_to_reset = self._process
                        parent_conn_to_reset = self._parent_conn
                        old_generation = self._generation
                        self._process = None
                        self._parent_conn = None
                        self._child_conn = None
                        self._ready = False
                        if self._schedule_restart_locked(old_generation):
                            restart_thread = self._restart_thread
                if process_to_reset is not None and parent_conn_to_reset is not None:
                    try:
                        self._cleanup_process_locked(process_to_reset, parent_conn_to_reset, timeout_seconds=self._recovery_timeout_seconds)
                    except Exception:
                        pass
                self._start_restart_thread(restart_thread)
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
        setattr(future, "_analysis_job_id", job.job_id)
        setattr(future, "_analysis_analyzer_name", self.analyzer_name)
        with self._lock:
            self._prune_job_registries_locked()
            self._jobs_by_id[job.job_id] = job
        try:
            self._queue.put_nowait(job)
        except queue.Full as exc:
            with self._lock:
                self._jobs_by_id.pop(job.job_id, None)
            raise AnalysisRuntimeBusyError(f"{self.analyzer_name}_queue_full") from exc
        return future

    def finalize_timed_out_job(self, job_id: str) -> dict[str, Any]:
        finalize_args: dict[str, Any] | None = None
        with self._lock:
            cached = self._completed_jobs.get(job_id)
            if cached is not None:
                return cached
            job = self._jobs_by_id.get(job_id)
            if job is None:
                cached_state = self._completed_jobs.get(job_id)
                if cached_state is not None:
                    return cached_state
                return {
                    "result": _timeout_placeholder(self.analyzer_name),
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
                        "worker_process_alive": False,
                        "finalized": True,
                        "worker_generation": self._generation,
                        "worker_queue_wait_ms": 0,
                        "worker_dispatch_ms": 0,
                        "worker_response_ms": 0,
                        "modality_total_ms": 0,
                        "worker_restarted": False,
                    },
                }
            self._expired_jobs[job_id] = time.perf_counter()
            self._expired_jobs.move_to_end(job_id)
            self._prune_job_registries_locked()
            is_active_job = job_id == self._active_job_id and job.worker_generation == self._active_worker_generation
            if not is_active_job:
                process_state = self._snapshot_process_locked()
                finalize_args = {
                    "job": job,
                    "process_state": process_state,
                    "cleanup_worker": False,
                    "clear_active": False,
                }
            else:
                finalize_args = {
                    "job": job,
                    "process_state": {"process_exited": False, "alive": True, "terminated": False, "killed": False, "process_exitcode": None},
                    "cleanup_worker": True,
                    "clear_active": True,
                }
        assert finalize_args is not None
        return self._finalize_job_locked(
            finalize_args["job"],
            timed_out=True,
            analyzer_error=False,
            result_received=False,
            process_state=finalize_args["process_state"],
            worker_restarted=False,
            child_metrics={},
            queue_wait_ms=_elapsed_ms(finalize_args["job"].submitted_at),
            dispatch_ms=0,
            response_ms=0,
            cleanup_worker=finalize_args["cleanup_worker"],
            restart_worker=True,
            clear_active=finalize_args["clear_active"],
        )

    def forget_job(self, job_id: str) -> None:
        with self._lock:
            self._completed_jobs.pop(job_id, None)
            self._expired_jobs.pop(job_id, None)
            self._jobs_by_id.pop(job_id, None)
            if self._active_job_id == job_id:
                self._active_job_id = None
                self._active_worker_generation = None

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
            "restart_in_progress": self._restart_in_progress,
            "restart_ready": self._restart_ready,
            "restart_generation": self._restart_generation,
            "restart_error_type": self._restart_error_type,
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
            self._restart_token += 1
            self._active_restart_token = None
            candidate_process = self._restart_candidate_process
            candidate_parent_conn = self._restart_candidate_parent_conn
            self._restart_candidate_process = None
            self._restart_candidate_parent_conn = None
            restart_thread = self._restart_thread
            process = self._process
            parent_conn = self._parent_conn
            self._process = None
            self._parent_conn = None
            self._child_conn = None
            self._ready = False
            self._restart_in_progress = False
            self._restart_ready = False
        if candidate_process is not None and candidate_parent_conn is not None:
            self._cleanup_process_locked(candidate_process, candidate_parent_conn, timeout_seconds=self._recovery_timeout_seconds)
        if restart_thread is not None and restart_thread.is_alive():
            restart_thread.join(timeout=self._recovery_timeout_seconds)
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
        with self._lock:
            self._jobs_by_id.clear()
            self._completed_jobs.clear()
            self._expired_jobs.clear()
            self._active_job_id = None
            self._active_worker_generation = None
            self._restart_thread = None
            self._restart_in_progress = False
            self._active_restart_token = None
            self._restart_candidate_process = None
            self._restart_candidate_parent_conn = None


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
        runtime_started_at = time.perf_counter()
        if not self._started:
            self.start()

        deadline_at = time.perf_counter() + deadline_seconds
        jobs: dict[str, Future] = {}
        submission_started_at = time.perf_counter()
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
                            "worker_process_alive": False,
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
        _log_perf(scan_id, "analyzer_submission_ms", _elapsed_ms(submission_started_at))

        results: dict[str, dict[str, Any]] = {}
        worker_states: dict[str, dict[str, Any]] = {}
        pending = {future: analyzer_name for analyzer_name, future in jobs.items()}
        deadline_wait_started_at = time.perf_counter()
        while pending:
            remaining = deadline_at - time.perf_counter()
            if remaining <= 0:
                break
            done, _ = futures_wait(list(pending.keys()), timeout=remaining, return_when=concurrent.futures.FIRST_COMPLETED)
            if not done:
                break
            for future in done:
                analyzer_name = pending.pop(future, None)
                if analyzer_name is None:
                    continue
                try:
                    completion = future.result()
                except Exception:
                    completion = {
                        "result": _error_placeholder(analyzer_name),
                        "state": {
                            "state": "finalized",
                            "child_started": True,
                            "result_ready": False,
                            "result_received": False,
                            "analyzer_error": True,
                            "process_exited": False,
                            "timed_out": False,
                            "alive": False,
                            "terminated": False,
                            "killed": False,
                            "process_exitcode": None,
                            "final_alive": False,
                            "worker_process_alive": False,
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
                job_id = getattr(future, "_analysis_job_id", None)
                if isinstance(job_id, str):
                    self._supervisors[analyzer_name].forget_job(job_id)
        _log_perf(scan_id, "media_deadline_wait_ms", _elapsed_ms(deadline_wait_started_at))
        if pending:
            timeout_finalization_started_at = time.perf_counter()
            with ThreadPoolExecutor(max_workers=len(pending)) as timeout_executor:
                finalizers: dict[Future, tuple[Future, str, str]] = {}
                for future, analyzer_name in pending.items():
                    job_id = getattr(future, "_analysis_job_id", None)
                    if not isinstance(job_id, str):
                        job_id = uuid.uuid4().hex
                    modality_finalization_started_at = time.perf_counter()
                    finalizers[
                        timeout_executor.submit(self._supervisors[analyzer_name].finalize_timed_out_job, job_id)
                    ] = (future, analyzer_name, job_id, modality_finalization_started_at)
                for finalizer in concurrent.futures.as_completed(finalizers):
                    pending_future, analyzer_name, job_id, modality_finalization_started_at = finalizers[finalizer]
                    try:
                        completion = finalizer.result()
                    except Exception:
                        completion = {
                            "result": _error_placeholder(analyzer_name),
                            "state": {
                                "state": "finalized",
                                "child_started": True,
                                "result_ready": False,
                                "result_received": False,
                                "analyzer_error": True,
                                "process_exited": False,
                                "timed_out": False,
                                "alive": False,
                                "terminated": False,
                                "killed": False,
                                "process_exitcode": None,
                                "final_alive": False,
                                "worker_process_alive": False,
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
                    finalization_ms = _elapsed_ms(modality_finalization_started_at)
                    _log_perf(scan_id, f"{analyzer_name}_timeout_finalization_ms", finalization_ms)
                    self._supervisors[analyzer_name].forget_job(job_id)
                    if pending_future.done():
                        try:
                            pending_future.result()
                        except Exception:
                            pass
            _log_perf(scan_id, "timeout_finalization_ms", _elapsed_ms(timeout_finalization_started_at))
        _log_perf(scan_id, "analyzer_runtime_total_ms", _elapsed_ms(runtime_started_at))
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
