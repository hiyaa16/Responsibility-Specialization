"""Isolated, resumable runner for the latency and timeout-sensitivity reruns.

This module deliberately does not use the historical runners' output locations or
RunRecord serialization. Schedule preparation is offline; model execution requires an
explicit --pilot or --full flag.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import itertools
import json
import os
import platform
import random
import socket
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
RERUN_ROOT = (ROOT / "results" / "reruns").resolve()
sys.path.insert(0, str(Path(__file__).resolve().parent))

from conditions import build_agents  # noqa: E402
from evaluator import evaluate  # noqa: E402
from failures import classify  # noqa: E402
from llm import LLMResponse, OllamaClient  # noqa: E402
from pipeline import run_pipeline  # noqa: E402
from schema import load_tasks  # noqa: E402

SCHEMA_VERSION = 1
CONFIG_FILES = {
    "latency": ROOT / "config" / "reruns" / "latency_interleaved_v1.yaml",
    "timeout": ROOT / "config" / "reruns" / "timeout_sensitivity_v1.yaml",
}
CONDITION_PERMUTATIONS = (
    ("generalist", "specialized", "specialized_length_matched"),
    ("generalist", "specialized_length_matched", "specialized"),
    ("specialized", "generalist", "specialized_length_matched"),
    ("specialized", "specialized_length_matched", "generalist"),
    ("specialized_length_matched", "generalist", "specialized"),
    ("specialized_length_matched", "specialized", "generalist"),
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def load_experiment_config(experiment: str) -> dict[str, Any]:
    if experiment not in CONFIG_FILES:
        raise ValueError(f"unknown experiment: {experiment}")
    with CONFIG_FILES[experiment].open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if config.get("kind") != ("latency" if experiment == "latency" else "timeout_sensitivity"):
        raise ValueError(f"invalid kind in {CONFIG_FILES[experiment]}")
    return config


def task_order(tasks: list[Any], task_ids: list[str] | None) -> list[Any]:
    if task_ids is None:
        return tasks
    by_id = {task.task_id: task for task in tasks}
    missing = sorted(set(task_ids) - set(by_id))
    if missing:
        raise ValueError(f"pilot task IDs not found: {missing}")
    return [by_id[task_id] for task_id in task_ids]


def _balanced_permutations(n_triplets: int, rng: random.Random) -> list[tuple[str, ...]]:
    complete_cycles, remainder = divmod(n_triplets, len(CONDITION_PERMUTATIONS))
    base = list(CONDITION_PERMUTATIONS) * complete_cycles
    if remainder:
        balanced_remainders = []
        for candidate in itertools.combinations(CONDITION_PERMUTATIONS, remainder):
            counts = {
                condition: [
                    sum(order[position] == condition for order in candidate)
                    for position in range(len(CONDITION_PERMUTATIONS[0]))
                ]
                for condition in CONDITION_PERMUTATIONS[0]
            }
            if all(max(positions) - min(positions) <= 1 for positions in counts.values()):
                balanced_remainders.append(candidate)
        if not balanced_remainders:
            raise ValueError(f"cannot balance {remainder} remaining condition orders")
        base.extend(rng.choice(balanced_remainders))
    rng.shuffle(base)
    return base


def build_schedule(
    config: dict[str, Any],
    tasks: list[Any],
    model_tag: str,
    pilot: bool,
) -> list[dict[str, Any]]:
    """Build a deterministic order, balancing all six A/B/C orders across triplets."""
    selected_tasks = task_order(
        tasks, config["pilot_task_ids"] if pilot else None
    )
    repetitions = config["pilot_repetitions"] if pilot else config["runs_per_task"]
    rng = random.Random(config["schedule_seed"])
    triplets = [(task, repetition) for task in selected_tasks for repetition in range(repetitions)]
    permutations = _balanced_permutations(len(triplets), rng)
    paired = list(zip(triplets, permutations))
    rng.shuffle(paired)
    arm_orders = []
    if config["kind"] == "timeout_sensitivity":
        pair_count = len(triplets) * len(config["conditions"])
        arm_orders = (
            [(600, 1800)] * (pair_count // 2)
            + [(1800, 600)] * (pair_count - pair_count // 2)
        )
        rng.shuffle(arm_orders)

    jobs = []
    order = 0
    for (task, repetition), condition_order in paired:
        for condition in condition_order:
            base = {
                "experiment_version": config["experiment_version"],
                "model_tag": model_tag,
                "task_id": task.task_id,
                "condition": condition,
                "repetition": repetition,
                "task_seed": config["task_seeds"][repetition],
                "schedule_seed": config["schedule_seed"],
            }
            if config["kind"] == "timeout_sensitivity":
                rng.shuffle(arm_orders)
                arm_order = arm_orders.pop()
                for timeout_s in arm_order:
                    jobs.append(
                        {
                            **base,
                            "configured_request_timeout_s": timeout_s,
                            "execution_order": order,
                        }
                    )
                    order += 1
            else:
                jobs.append(
                    {
                        **base,
                        "configured_request_timeout_s": config["request_timeout_s"],
                        "execution_order": order,
                    }
                )
                order += 1
    return jobs


def experiment_paths(experiment: str, model_tag: str, pilot: bool) -> tuple[Path, Path]:
    """Return (schedule directory, output root) while enforcing new-path isolation."""
    if experiment == "latency":
        if pilot:
            schedule_dir = RERUN_ROOT / "latency_interleaved_v1" / "pilot" / model_tag
        else:
            schedule_dir = RERUN_ROOT / "latency_interleaved_v1" / model_tag
        output_root = schedule_dir
    elif experiment == "timeout":
        if pilot:
            schedule_dir = RERUN_ROOT / "timeout_sensitivity_v1" / "pilot" / "schedules" / model_tag
            output_root = RERUN_ROOT / "timeout_sensitivity_v1"
        else:
            schedule_dir = RERUN_ROOT / "timeout_sensitivity_v1" / "schedules" / model_tag
            output_root = RERUN_ROOT / "timeout_sensitivity_v1"
    else:
        raise ValueError(f"unknown experiment: {experiment}")
    assert_new_path(schedule_dir)
    assert_new_path(output_root)
    return schedule_dir, output_root


def assert_new_path(path: Path) -> None:
    resolved = path.resolve()
    try:
        resolved.relative_to(RERUN_ROOT)
    except ValueError as exc:
        raise ValueError(f"refusing output outside isolated rerun root: {resolved}") from exc


def _write_immutable(path: Path, content: bytes) -> None:
    assert_new_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f"refusing to overwrite frozen artifact: {path}")
        return
    with path.open("xb") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        for row in rows
    )


def prepare_schedule(experiment: str, model_tag: str, pilot: bool) -> tuple[list[dict[str, Any]], Path, Path]:
    config = load_experiment_config(experiment)
    if model_tag not in config["model_tags"]:
        raise ValueError(f"model tag not configured: {model_tag}")
    tasks = load_tasks(str(ROOT / config["tasks_path"]))
    if len(tasks) != 50:
        raise ValueError(f"expected 50 V2 tasks; loaded {len(tasks)}")
    schedule = build_schedule(config, tasks, model_tag, pilot)
    schedule_dir, output_root = experiment_paths(experiment, model_tag, pilot)

    schedule_path = schedule_dir / "schedule.jsonl"
    frozen_path = schedule_dir / "frozen_config.json"
    schedule_content = _jsonl_bytes(schedule)
    _write_immutable(schedule_path, schedule_content)
    frozen = {
        **config,
        "model_tag": model_tag,
        "ollama_model": config["model_tags"][model_tag]["ollama_model"],
        "pilot": pilot,
        "planned_runs": len(schedule),
        "schedule_sha256": hashlib.sha256(schedule_content).hexdigest(),
        "task_data_sha256": hashlib.sha256((ROOT / config["tasks_path"]).read_bytes()).hexdigest(),
        "source_config_sha256": hashlib.sha256(CONFIG_FILES[experiment].read_bytes()).hexdigest(),
    }
    _write_immutable(frozen_path, _json_bytes(frozen))

    if experiment == "timeout":
        for timeout_s in config["timeout_arms"]:
            arm_root = output_root / f"timeout_{timeout_s}s" / model_tag
            if pilot:
                arm_root = output_root / "pilot" / f"timeout_{timeout_s}s" / model_tag
            assert_new_path(arm_root)
            arm_jobs = [job for job in schedule if job["configured_request_timeout_s"] == timeout_s]
            arm_schedule = _jsonl_bytes(arm_jobs)
            arm_config = {
                **frozen,
                "request_timeout_s": timeout_s,
                "planned_runs": len(arm_jobs),
                "schedule_sha256": hashlib.sha256(arm_schedule).hexdigest(),
            }
            _write_immutable(arm_root / "schedule.jsonl", arm_schedule)
            _write_immutable(arm_root / "frozen_config.json", _json_bytes(arm_config))
    else:
        _write_immutable(output_root / "frozen_config.json", _json_bytes(frozen))
    return schedule, schedule_dir, output_root


def read_schedule(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"schedule must be prepared first: {path}")
    jobs = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                job = json.loads(line)
                if job.get("execution_order") != len(jobs):
                    raise ValueError(f"non-contiguous schedule order at line {line_no}")
                jobs.append(job)
    if len({job["execution_order"] for job in jobs}) != len(jobs):
        raise ValueError("duplicate schedule execution_order")
    return jobs


def validate_model_available(config: dict[str, Any], model_tag: str) -> dict[str, Any]:
    """Read installed Ollama tag metadata; does not issue an inference request."""
    base_url = config["ollama_url"].rstrip("/")
    with urllib.request.urlopen(f"{base_url}/api/tags", timeout=5) as response:
        payload = json.loads(response.read().decode("utf-8"))
    model_name = config["model_tags"][model_tag]["ollama_model"]
    for model in payload.get("models", []):
        if model.get("name") == model_name:
            return model
    raise RuntimeError(f"required Ollama model is not installed: {model_name}")


def environment_metadata() -> dict[str, Any]:
    free_memory_bytes = None
    total_memory_bytes = None
    if os.name == "nt":
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            free_memory_bytes = int(status.ullAvailPhys)
            total_memory_bytes = int(status.ullTotalPhys)
    return {
        "captured_at": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "total_memory_bytes": total_memory_bytes,
        "available_memory_bytes": free_memory_bytes,
        "gpu_detection": "not inferred by runner; record local Ollama diagnostics separately",
    }


def _is_timeout(error: BaseException) -> bool:
    if isinstance(error, (socket.timeout, TimeoutError)):
        return True
    reason = getattr(error, "reason", None)
    if isinstance(error, urllib.error.URLError) and isinstance(reason, (socket.timeout, TimeoutError)):
        return True
    return "timed out" in str(error).lower()


def _is_model_error(error: BaseException) -> bool:
    return isinstance(
        error,
        (
            urllib.error.URLError,
            urllib.error.HTTPError,
            json.JSONDecodeError,
            ModelResponseError,
        ),
    )


class ModelResponseError(RuntimeError):
    """An Ollama response was received but did not contain usable model output."""


class MeasuredOllamaClient(OllamaClient):
    """Capture request-level wall time and partial counters without changing legacy clients."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.reset_metrics()

    def reset_metrics(self) -> None:
        self.request_attempts = 0
        self.request_successes = 0
        self.model_call_seconds = 0.0
        self.input_tokens = 0
        self.output_tokens = 0
        self.truncated_any = False

    def chat(self, messages: list[dict], seed: int) -> LLMResponse:
        self.request_attempts += 1
        start = time.perf_counter()
        try:
            response = super().chat(messages, seed)
            self.request_successes += 1
            self.input_tokens += response.prompt_tokens
            self.output_tokens += response.output_tokens
            self.truncated_any = self.truncated_any or response.done_reason == "length"
            if not response.content.strip():
                raise ModelResponseError("Ollama response contained no message content")
            return response
        finally:
            self.model_call_seconds += time.perf_counter() - start


@contextmanager
def model_run_lock(model_tag: str):
    """One OS-released lock serializes all rerun models and timeout arms."""
    RERUN_ROOT.mkdir(parents=True, exist_ok=True)
    path = RERUN_ROOT / ".rerun-model.lock"
    with path.open("a+b") as f:
        if os.name == "nt":
            import msvcrt

            f.seek(0, os.SEEK_END)
            if f.tell() == 0:
                f.write(b"\0")
                f.flush()
            f.seek(0)
            try:
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError(
                    f"another rerun holds the global model lock; requested {model_tag}"
                ) from exc
            try:
                yield
            finally:
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError(
                    f"another rerun holds the global model lock; requested {model_tag}"
                ) from exc
            try:
                yield
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _jsonl_append(path: Path, row: dict[str, Any]) -> None:
    assert_new_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def open_store(output_dir: Path) -> sqlite3.Connection:
    assert_new_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(output_dir / "experiments.sqlite")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            experiment_id TEXT PRIMARY KEY,
            schema_version INTEGER NOT NULL,
            experiment_version TEXT NOT NULL,
            model_tag TEXT NOT NULL,
            timeout_arm INTEGER NOT NULL,
            task_id TEXT NOT NULL,
            condition TEXT NOT NULL,
            repetition INTEGER NOT NULL,
            execution_order INTEGER NOT NULL,
            outcome_category TEXT NOT NULL,
            elapsed_pipeline_seconds REAL NOT NULL,
            record_json TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS attempt_events (
            event_id TEXT PRIMARY KEY,
            experiment_id TEXT NOT NULL,
            attempt_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            payload_json TEXT NOT NULL
        )
        """
    )
    connection.commit()
    return connection


def load_records(path: Path) -> dict[str, dict[str, Any]]:
    records = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            run_id = record["experiment_id"]
            if run_id in records:
                raise ValueError(f"duplicate experiment_id in {path} line {line_no}: {run_id}")
            records[run_id] = record
    return records


def sync_store_from_jsonl(connection: sqlite3.Connection, records: dict[str, dict[str, Any]]) -> None:
    """Repair only missing SQLite mirror rows after a crash; conflicting rows are an error."""
    existing = {
        row[0]: row[1]
        for row in connection.execute("SELECT experiment_id, record_json FROM runs")
    }
    unexpected = set(existing) - set(records)
    if unexpected:
        raise ValueError(
            f"SQLite contains records absent from JSONL: {sorted(unexpected)[:5]}"
        )
    for run_id, record in records.items():
        serialized = json.dumps(record, ensure_ascii=False, sort_keys=True)
        if run_id in existing:
            if json.loads(existing[run_id]) != record:
                raise ValueError(f"JSONL/SQLite conflict for run {run_id}")
            continue
        connection.execute(
            """
            INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            _sqlite_record_values(record, serialized),
        )
    connection.commit()


def _sqlite_record_values(record: dict[str, Any], serialized: str | None = None) -> tuple[Any, ...]:
    if serialized is None:
        serialized = json.dumps(record, ensure_ascii=False, sort_keys=True)
    return (
        record["experiment_id"],
        record["schema_version"],
        record["experiment_version"],
        record["model_tag"],
        record["configured_request_timeout_s"],
        record["task_id"],
        record["condition"],
        record["repetition"],
        record["execution_order"],
        record["outcome_category"],
        record["elapsed_pipeline_seconds"],
        serialized,
    )


def append_attempt_event(
    connection: sqlite3.Connection,
    output_dir: Path,
    experiment_id: str,
    attempt_id: str,
    event_type: str,
    payload: dict[str, Any],
) -> None:
    event = {
        "event_id": f"{attempt_id}:{event_type}",
        "experiment_id": experiment_id,
        "attempt_id": attempt_id,
        "event_type": event_type,
        "timestamp": utc_now(),
        **payload,
    }
    _jsonl_append(output_dir / "attempts.jsonl", event)
    connection.execute(
        "INSERT OR IGNORE INTO attempt_events VALUES (?, ?, ?, ?, ?, ?)",
        (
            event["event_id"],
            experiment_id,
            attempt_id,
            event_type,
            event["timestamp"],
            json.dumps(event, ensure_ascii=False, sort_keys=True),
        ),
    )
    connection.commit()


def sync_attempt_events_from_jsonl(
    connection: sqlite3.Connection, output_dir: Path
) -> list[dict[str, Any]]:
    path = output_dir / "attempts.jsonl"
    if not path.exists():
        return []
    events = []
    existing = {
        row[0]: row[1]
        for row in connection.execute(
            "SELECT event_id, payload_json FROM attempt_events"
        )
    }
    journal_ids = set()
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            event = json.loads(line)
            if event["event_id"] in journal_ids:
                raise ValueError(f"duplicate attempt event in journal: {event['event_id']}")
            events.append(event)
            journal_ids.add(event["event_id"])
            serialized = json.dumps(event, ensure_ascii=False, sort_keys=True)
            if event["event_id"] in existing:
                if json.loads(existing[event["event_id"]]) != event:
                    raise ValueError(
                        f"attempt journal/SQLite conflict at {path}:{line_no}"
                    )
                continue
            connection.execute(
                "INSERT INTO attempt_events VALUES (?, ?, ?, ?, ?, ?)",
                (
                    event["event_id"],
                    event["experiment_id"],
                    event["attempt_id"],
                    event["event_type"],
                    event["timestamp"],
                    serialized,
                ),
            )
            existing[event["event_id"]] = serialized
    unexpected = set(existing) - journal_ids
    if unexpected:
        raise ValueError(
            f"attempt SQLite contains events absent from journal: {sorted(unexpected)[:5]}"
        )
    connection.commit()
    return events


def recover_interrupted_attempts(
    connection: sqlite3.Connection,
    output_dir: Path,
    completed_records: dict[str, dict[str, Any]] | set[str],
) -> dict[str, int]:
    events_path = output_dir / "attempts.jsonl"
    if not events_path.exists():
        return {}
    events = sync_attempt_events_from_jsonl(connection, output_dir)
    starts = {}
    finished = set()
    for event in events:
        if event["event_type"] == "attempt_started":
            starts[event["attempt_id"]] = event
        elif event["event_type"] in ("attempt_finished", "attempt_interrupted"):
            finished.add(event["attempt_id"])
    attempts_by_run: dict[str, int] = {}
    for event in events:
        run_id = event["experiment_id"]
        attempts_by_run[run_id] = max(attempts_by_run.get(run_id, 0), event.get("attempt_number", 0))
    completed_ids = (
        set(completed_records)
        if isinstance(completed_records, dict)
        else completed_records
    )
    for attempt_id, event in starts.items():
        if attempt_id in finished:
            continue
        if event["experiment_id"] in completed_ids:
            record = (
                completed_records[event["experiment_id"]]
                if isinstance(completed_records, dict)
                else {}
            )
            append_attempt_event(
                connection,
                output_dir,
                event["experiment_id"],
                attempt_id,
                "attempt_finished",
                {
                    "attempt_number": event["attempt_number"],
                    "outcome_category": record.get("outcome_category", "unknown"),
                    "elapsed_pipeline_seconds": record.get("elapsed_pipeline_seconds"),
                    "timestamp_end": record.get("run_end_utc"),
                    "recovered_from_run_record": True,
                },
            )
            continue
        append_attempt_event(
            connection,
            output_dir,
            event["experiment_id"],
            attempt_id,
            "attempt_interrupted",
            {
                "attempt_number": event["attempt_number"],
                "reason": "start event exists without a finish event; elapsed duration is unknown",
            },
        )
    return attempts_by_run


def _timeout_exception(error: BaseException) -> bool:
    return _is_timeout(error)


def execute_job(
    job: dict[str, Any],
    task: Any,
    agents: list[Any],
    client: MeasuredOllamaClient,
    model_info: dict[str, Any],
    experiment_version: str,
    attempt_number: int,
) -> dict[str, Any]:
    """Execute one scheduled run and always retain actual elapsed time, including errors."""
    timeout_s = job["configured_request_timeout_s"]
    experiment_id = make_experiment_id(job, experiment_version)
    attempt_id = f"{experiment_id}::attempt{attempt_number}"
    started_at = utc_now()
    pipeline_start = time.perf_counter()
    pipeline_elapsed = None
    client.reset_metrics()
    output = None
    error_details = None
    correct = None
    evaluated = False
    pipeline_completed = False
    failure = None
    failure_stage = None
    final_answer = ""
    failure_types: list[str] = []
    corrections = 0
    repeated_work = 0.0
    try:
        output = run_pipeline(agents, task, job["task_seed"])
        pipeline_elapsed = time.perf_counter() - pipeline_start
        pipeline_completed = len(output["turns"]) == 3
        failure_stage = "evaluation"
        final_answer, correct = evaluate(output["final_text"], task)
        evaluated = True
        failure_stage = "failure_classification"
        failure_types, corrections, repeated_work = classify(
            task, output["turns"], output["final_text"], correct
        )
        failure_stage = None
    except Exception as exc:
        failure = exc
        if pipeline_elapsed is None:
            pipeline_elapsed = time.perf_counter() - pipeline_start
        error_details = {
            "type": type(exc).__name__,
            "message": str(exc)[:1000],
            "is_timeout": _timeout_exception(exc),
            "stage": failure_stage or "pipeline",
        }
    finally:
        ended_at = utc_now()

    truncated = client.truncated_any or bool(output and output["truncated"])
    if failure is not None:
        if _timeout_exception(failure):
            outcome = "timeout"
        elif _is_model_error(failure):
            outcome = "model_error"
        else:
            outcome = "generic_error"
    elif truncated:
        outcome = "truncation"
    elif pipeline_completed and evaluated and final_answer.strip():
        outcome = "success"
    else:
        outcome = "generic_error"
        error_details = {
            "type": "MissingFinalAnswer" if pipeline_completed and evaluated else "IncompletePipeline",
            "message": (
                "pipeline completed but no final answer was extracted"
                if pipeline_completed and evaluated
                else "pipeline did not complete all three agent turns and evaluation"
            ),
            "is_timeout": False,
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "experiment_version": experiment_version,
        "model_tag": job["model_tag"],
        "ollama_model": model_info["name"],
        "model_digest": model_info.get("digest"),
        "model_modified_at": model_info.get("modified_at"),
        "task_id": job["task_id"],
        "condition": job["condition"],
        "repetition": job["repetition"],
        "task_seed": job["task_seed"],
        "schedule_seed": job["schedule_seed"],
        "execution_order": job["execution_order"],
        "configured_request_timeout_s": timeout_s,
        "attempt_id": attempt_id,
        "attempt_number": attempt_number,
        "run_start_utc": started_at,
        "run_end_utc": ended_at,
        "elapsed_pipeline_seconds": pipeline_elapsed,
        "accumulated_model_call_seconds": client.model_call_seconds,
        "llm_calls_attempted": client.request_attempts,
        "llm_calls_succeeded": client.request_successes,
        "llm_call_count": client.request_attempts,
        "input_tokens": client.input_tokens,
        "output_tokens": client.output_tokens,
        "total_tokens": client.input_tokens + client.output_tokens,
        "pipeline_completed": pipeline_completed,
        "evaluation_completed": evaluated,
        "outcome_category": outcome,
        "execution_successful": outcome == "success",
        "timed_out": outcome == "timeout",
        "model_error": outcome == "model_error",
        "generic_error": outcome == "generic_error",
        "truncated": truncated,
        "correct": correct,
        "final_answer": final_answer,
        "failure_types": failure_types,
        "corrections": corrections,
        "repeated_work": repeated_work,
        "error_details": error_details,
        "outcome_definition": (
            "success means all three turns and deterministic evaluation completed, a final answer "
            "was extracted, and no timeout/error/truncation occurred; correctness is reported "
            "separately and may be false"
        ),
        "latency_definition": (
            "pipeline wall time from immediately before run_pipeline through return/exception; "
            "excludes agent construction, evaluation, and file I/O"
        ),
        "model_call_latency_definition": (
            "sum of per-request wall time around Ollama HTTP request, response read, and JSON decode; "
            "includes failed HTTP attempts"
        ),
    }


def make_experiment_id(job: dict[str, Any], experiment_version: str) -> str:
    prefix = experiment_version
    if "configured_request_timeout_s" in job and experiment_version == "timeout_sensitivity_v1":
        prefix += f"::timeout{job['configured_request_timeout_s']}"
    return (
        f"{prefix}::{job['model_tag']}::{job['condition']}::{job['task_id']}"
        f"::rep{job['repetition']}"
    )


def pending_jobs(
    jobs: list[dict[str, Any]],
    experiment_version: str,
    completed_records: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        job
        for job in jobs
        if make_experiment_id(job, experiment_version) not in completed_records
    ]


def _write_record(connection: sqlite3.Connection, output_dir: Path, record: dict[str, Any]) -> None:
    path = output_dir / "runs.jsonl"
    existing = load_records(path)
    if record["experiment_id"] in existing:
        raise ValueError(f"refusing duplicate run record: {record['experiment_id']}")
    sqlite_existing = connection.execute(
        "SELECT record_json FROM runs WHERE experiment_id = ?",
        (record["experiment_id"],),
    ).fetchone()
    if sqlite_existing is not None:
        raise ValueError(f"refusing run ID already present in SQLite: {record['experiment_id']}")
    _jsonl_append(path, record)
    connection.execute(
        "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        _sqlite_record_values(record),
    )
    connection.commit()


def _arm_directory(output_root: Path, experiment: str, job: dict[str, Any], pilot: bool) -> Path:
    if experiment == "latency":
        return output_root
    prefix = "pilot" if pilot else None
    parts = [output_root]
    if prefix:
        parts.append(Path(prefix))
    parts.extend([Path(f"timeout_{job['configured_request_timeout_s']}s"), Path(job["model_tag"])])
    result = parts[0].joinpath(*parts[1:])
    assert_new_path(result)
    return result


def _validate_schedule_jobs(
    jobs: list[dict[str, Any]],
    experiment: str,
    pilot: bool = False,
    config: dict[str, Any] | None = None,
) -> None:
    if not jobs:
        raise ValueError("schedule is empty")
    config = config or load_experiment_config(experiment)
    ids = [make_experiment_id(job, job["experiment_version"]) for job in jobs]
    if len(ids) != len(set(ids)):
        raise ValueError("schedule contains duplicate experiment IDs")
    if len({job["model_tag"] for job in jobs}) != 1:
        raise ValueError("schedule must contain exactly one model tag")
    if config.get("model_tag") is not None and jobs[0]["model_tag"] != config["model_tag"]:
        raise ValueError("schedule model tag does not match frozen config")
    positions = Counter(job["execution_order"] for job in jobs)
    if set(positions) != set(range(len(jobs))) or any(count != 1 for count in positions.values()):
        raise ValueError("schedule order must be unique and contiguous")
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for job in jobs:
        grouped.setdefault((job["task_id"], job["repetition"]), []).append(job)
    tasks = load_tasks(str(ROOT / config["tasks_path"]))
    selected = task_order(tasks, config["pilot_task_ids"] if pilot else None)
    repetitions = config["pilot_repetitions"] if pilot else config["runs_per_task"]
    expected_triplets = {
        (task.task_id, repetition)
        for task in selected
        for repetition in range(repetitions)
    }
    if set(grouped) != expected_triplets:
        raise ValueError("schedule task/repetition set does not match the frozen design")
    expected_conditions = set(config["conditions"])
    expected_arms = set(
        config["timeout_arms"]
        if "timeout_arms" in config
        else [config["request_timeout_s"]]
    )
    for key, triplet_jobs in grouped.items():
        conditions = [job["condition"] for job in triplet_jobs]
        if (
            set(conditions) != expected_conditions
            or len(conditions) != len(expected_conditions) * len(expected_arms)
        ):
            raise ValueError(f"task/repetition {key} does not contain the expected conditions/arms")
        for job in triplet_jobs:
            task, repetition = key
            if job["task_seed"] != config["task_seeds"][repetition]:
                raise ValueError(f"wrong task seed in schedule for {task} repetition {repetition}")
            if job["schedule_seed"] != config["schedule_seed"]:
                raise ValueError(f"wrong schedule seed in schedule for {task} repetition {repetition}")
            if job["model_tag"] not in config["model_tags"]:
                raise ValueError(f"unexpected model tag in schedule: {job['model_tag']}")
            if job["experiment_version"] != config["experiment_version"]:
                raise ValueError(f"unexpected experiment version in schedule: {job['experiment_version']}")
        by_condition: dict[str, list[dict[str, Any]]] = {}
        for job in triplet_jobs:
            by_condition.setdefault(job["condition"], []).append(job)
        for condition, condition_jobs in by_condition.items():
            arms = [job["configured_request_timeout_s"] for job in condition_jobs]
            if len(arms) != len(expected_arms) or set(arms) != expected_arms:
                raise ValueError(
                    f"task/repetition {key} condition {condition} has wrong timeout arms"
                )
    planned = len(selected) * repetitions * len(expected_conditions) * len(expected_arms)
    if len(jobs) != planned:
        raise ValueError(f"schedule has {len(jobs)} jobs; expected {planned}")


def _capture_run_metadata(output_dir: Path, config: dict[str, Any], model_info: dict[str, Any]) -> None:
    metadata_path = output_dir / "environment.json"
    model_path = output_dir / "model.json"
    if not metadata_path.exists():
        _write_immutable(metadata_path, _json_bytes(environment_metadata()))
    _write_immutable(model_path, _json_bytes(model_info))
    frozen_config = output_dir / "frozen_config.json"
    if not frozen_config.is_file():
        raise FileNotFoundError(f"frozen config missing; prepare schedule first: {frozen_config}")
    recorded = json.loads(frozen_config.read_text(encoding="utf-8"))
    for key in ("experiment_version", "model_tag", "ollama_model", "request_timeout_s"):
        if key in config and recorded.get(key) != config[key]:
            raise ValueError(f"frozen config mismatch for {key} in {frozen_config}")
    schedule_path = output_dir / "schedule.jsonl"
    if recorded.get("schedule_sha256") != hashlib.sha256(schedule_path.read_bytes()).hexdigest():
        raise ValueError(f"arm schedule has changed since it was frozen: {schedule_path}")


def run_schedule(experiment: str, model_tag: str, pilot: bool) -> None:
    schedule_dir, output_root = experiment_paths(experiment, model_tag, pilot)
    jobs = read_schedule(schedule_dir / "schedule.jsonl")
    frozen_path = schedule_dir / "frozen_config.json"
    if not frozen_path.is_file():
        raise FileNotFoundError(f"frozen config missing: {frozen_path}")
    config = json.loads(frozen_path.read_text(encoding="utf-8"))
    if config.get("model_tag") != model_tag:
        raise ValueError("frozen schedule belongs to a different model tag")
    if config.get("kind") != ("latency" if experiment == "latency" else "timeout_sensitivity"):
        raise ValueError("frozen config experiment type does not match requested experiment")
    _validate_schedule_jobs(jobs, experiment, pilot=pilot, config=config)
    if config.get("task_data_sha256") != hashlib.sha256(
        (ROOT / config["tasks_path"]).read_bytes()
    ).hexdigest():
        raise ValueError("task data has changed since schedule preparation")
    if config.get("schedule_sha256") != hashlib.sha256(
        (schedule_dir / "schedule.jsonl").read_bytes()
    ).hexdigest():
        raise ValueError("schedule has changed since it was frozen")

    with model_run_lock(model_tag):
        model_info = validate_model_available(config, model_tag)
        task_map = {task.task_id: task for task in load_tasks(str(ROOT / config["tasks_path"]))}
        client = MeasuredOllamaClient(
            url=config["ollama_url"].rstrip("/") + "/api/chat",
            model=config["model_tags"][model_tag]["ollama_model"],
            temperature=config["temperature"],
            top_p=config["top_p"],
            num_ctx=config["num_ctx"],
            num_predict=config["num_predict"],
            timeout_s=jobs[0]["configured_request_timeout_s"],
        )
        agents_by_key: dict[tuple[str, int], list[Any]] = {}
        for job in jobs:
            key = (job["condition"], job["configured_request_timeout_s"])
            if key not in agents_by_key:
                client.timeout_s = job["configured_request_timeout_s"]
                agents_by_key[key] = build_agents(job["condition"], client, config["max_tool_iters"])

        output_dirs = {
            timeout_s: _arm_directory(
                output_root,
                experiment,
                next(job for job in jobs if job["configured_request_timeout_s"] == timeout_s),
                pilot,
            )
            for timeout_s in sorted({job["configured_request_timeout_s"] for job in jobs})
        }
        stores = {}
        try:
            for timeout_s, output_dir in output_dirs.items():
                assert_new_path(output_dir)
                if experiment == "timeout":
                    arm_config_path = output_dir / "frozen_config.json"
                    if not arm_config_path.is_file():
                        raise FileNotFoundError(
                            f"arm config missing; prepare schedule first: {arm_config_path}"
                        )
                    arm_config = json.loads(arm_config_path.read_text(encoding="utf-8"))
                else:
                    arm_config = config
                _write_immutable(output_dir / "schedule.jsonl", _jsonl_bytes(
                    [row for row in jobs if row["configured_request_timeout_s"] == timeout_s]
                ))
                _capture_run_metadata(output_dir, arm_config, model_info)
                connection = open_store(output_dir)
                records_path = output_dir / "runs.jsonl"
                records = load_records(records_path)
                sync_store_from_jsonl(connection, records)
                sync_attempt_events_from_jsonl(connection, output_dir)
                attempts = recover_interrupted_attempts(connection, output_dir, records)
                stores[timeout_s] = {
                    "connection": connection,
                    "records": records,
                    "attempts": attempts,
                    "output_dir": output_dir,
                }

            done = 0
            completed_records = {}
            for state in stores.values():
                completed_records.update(state["records"])
            pending = pending_jobs(jobs, config["experiment_version"], completed_records)
            for job in pending:
                timeout_s = job["configured_request_timeout_s"]
                state = stores[timeout_s]
                output_dir = state["output_dir"]
                connection = state["connection"]
                run_id = make_experiment_id(job, config["experiment_version"])
                attempt_number = state["attempts"].get(run_id, 0) + 1
                attempt_id = f"{run_id}::attempt{attempt_number}"
                append_attempt_event(
                    connection,
                    output_dir,
                    run_id,
                    attempt_id,
                    "attempt_started",
                    {
                        "attempt_number": attempt_number,
                        "execution_order": job["execution_order"],
                        "configured_request_timeout_s": timeout_s,
                        "timestamp_start": utc_now(),
                    },
                )
                client.timeout_s = timeout_s
                record = execute_job(
                    job,
                    task_map[job["task_id"]],
                    agents_by_key[(job["condition"], timeout_s)],
                    client,
                    model_info,
                    config["experiment_version"],
                    attempt_number,
                )
                _write_record(connection, output_dir, record)
                state["records"][run_id] = record
                state["attempts"][run_id] = attempt_number
                append_attempt_event(
                    connection,
                    output_dir,
                    run_id,
                    record["attempt_id"],
                    "attempt_finished",
                    {
                        "attempt_number": attempt_number,
                        "outcome_category": record["outcome_category"],
                        "elapsed_pipeline_seconds": record["elapsed_pipeline_seconds"],
                        "timestamp_end": record["run_end_utc"],
                    },
                )
                done += 1
                print(
                    f"[{job['execution_order'] + 1}/{len(jobs)}] {run_id} "
                    f"{record['outcome_category']} wall={record['elapsed_pipeline_seconds']:.3f}s "
                    f"model={record['accumulated_model_call_seconds']:.3f}s "
                    f"correct={record['correct']}"
                )
            print(f"Schedule complete or resumed: {len(jobs)} planned, {done} executed this invocation.")
        finally:
            for state in stores.values():
                state["connection"].close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("latency", "timeout"), required=True)
    parser.add_argument(
        "--model",
        choices=("qwen2_5_7b", "llama3_1_8b", "mistral_7b"),
        required=True,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true", help="freeze a schedule; no model calls")
    mode.add_argument(
        "--prepare-pilot-only", action="store_true", help="freeze the bounded pilot schedule only"
    )
    mode.add_argument("--pilot", action="store_true", help="execute only the separately scheduled pilot")
    mode.add_argument("--full", action="store_true", help="execute the full frozen schedule")
    args = parser.parse_args()

    if args.prepare_only or args.prepare_pilot_only:
        pilot = args.prepare_pilot_only
        jobs, schedule_dir, _ = prepare_schedule(args.experiment, args.model, pilot=pilot)
        print(f"Prepared {len(jobs)} frozen jobs at {schedule_dir}; no model calls made.")
        return 0
    pilot = args.pilot
    schedule_dir, _ = experiment_paths(args.experiment, args.model, pilot)
    if not (schedule_dir / "schedule.jsonl").is_file():
        raise FileNotFoundError(
            f"frozen schedule missing at {schedule_dir}; prepare it explicitly before execution"
        )
    run_schedule(args.experiment, args.model, pilot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
