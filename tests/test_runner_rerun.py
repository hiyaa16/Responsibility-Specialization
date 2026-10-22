from __future__ import annotations

import json
import sys
import tempfile
import time
import urllib.error
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import runner_rerun as rerun
import validate_rerun_results as result_validator
import summarize_rerun as result_summary
import manifest_rerun
from schema import Task


class ScheduleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks = rerun.load_tasks(
            str(rerun.ROOT / "data" / "tasks_v2.jsonl")
        )

    def test_latency_schedule_is_reproducible_and_balanced(self):
        config = rerun.load_experiment_config("latency")
        first = rerun.build_schedule(config, self.tasks, "qwen2_5_7b", pilot=False)
        second = rerun.build_schedule(config, self.tasks, "qwen2_5_7b", pilot=False)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 750)
        rerun._validate_schedule_jobs(first, "latency")

        positions = [0, 0, 0]
        position_counts = {
            condition: [0, 0, 0] for condition in config["conditions"]
        }
        orders = {}
        for index in range(0, len(first), 3):
            triplet = first[index : index + 3]
            self.assertEqual(len({job["task_id"] for job in triplet}), 1)
            self.assertEqual(len({job["repetition"] for job in triplet}), 1)
            self.assertEqual(
                {job["condition"] for job in triplet},
                set(config["conditions"]),
            )
            order = tuple(job["condition"] for job in triplet)
            orders[order] = orders.get(order, 0) + 1
            for position, job in enumerate(triplet):
                positions[position] += 1
                position_counts[job["condition"]][position] += 1
                self.assertEqual(job["execution_order"], index + position)
        self.assertEqual(len(orders), 6)
        self.assertLessEqual(max(orders.values()) - min(orders.values()), 1)
        self.assertEqual(positions, [250, 250, 250])
        self.assertTrue(
            all(max(counts) - min(counts) <= 1 for counts in position_counts.values()),
            position_counts,
        )

    def test_timeout_schedule_pairs_arms_with_balanced_first_order(self):
        config = rerun.load_experiment_config("timeout")
        jobs = rerun.build_schedule(config, self.tasks, "qwen2_5_7b", pilot=False)
        self.assertEqual(len(jobs), 1500)
        rerun._validate_schedule_jobs(jobs, "timeout")
        self.assertEqual(
            {arm: sum(job["configured_request_timeout_s"] == arm for job in jobs)
             for arm in config["timeout_arms"]},
            {600: 750, 1800: 750},
        )

        pair_first_arms = []
        for index in range(0, len(jobs), 2):
            pair = jobs[index : index + 2]
            self.assertEqual(len(pair), 2)
            self.assertEqual(
                (pair[0]["task_id"], pair[0]["condition"], pair[0]["repetition"]),
                (pair[1]["task_id"], pair[1]["condition"], pair[1]["repetition"]),
            )
            self.assertEqual(
                {job["configured_request_timeout_s"] for job in pair},
                {600, 1800},
            )
            pair_first_arms.append(pair[0]["configured_request_timeout_s"])
        self.assertEqual(pair_first_arms.count(600), 375)
        self.assertEqual(pair_first_arms.count(1800), 375)

    def test_pilot_schedule_sizes(self):
        latency = rerun.load_experiment_config("latency")
        timeout = rerun.load_experiment_config("timeout")
        self.assertEqual(
            len(rerun.build_schedule(latency, self.tasks, "mistral_7b", pilot=True)),
            9,
        )
        self.assertEqual(
            len(rerun.build_schedule(timeout, self.tasks, "mistral_7b", pilot=True)),
            18,
        )

    def test_prepare_writes_immutable_schedules_without_run_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(rerun, "RERUN_ROOT", Path(temp).resolve()):
                schedule, schedule_dir, output_root = rerun.prepare_schedule(
                    "timeout", "qwen2_5_7b", pilot=True
                )
                self.assertEqual(len(schedule), 18)
                self.assertTrue((schedule_dir / "schedule.jsonl").is_file())
                self.assertTrue((schedule_dir / "frozen_config.json").is_file())
                for arm in (600, 1800):
                    arm_dir = output_root / "pilot" / f"timeout_{arm}s" / "qwen2_5_7b"
                    self.assertTrue((arm_dir / "schedule.jsonl").is_file())
                    self.assertTrue((arm_dir / "frozen_config.json").is_file())
                    self.assertFalse((arm_dir / "runs.jsonl").exists())
                    self.assertEqual(
                        len((arm_dir / "schedule.jsonl").read_text(encoding="utf-8").splitlines()),
                        9,
                    )

                with self.assertRaises(FileExistsError):
                    rerun._write_immutable(
                        schedule_dir / "schedule.jsonl",
                        b'{"different": true}\n',
                    )


class ClientMetricsTests(unittest.TestCase):
    def test_wrapped_client_passes_each_configured_timeout_to_http(self):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps(
                    {
                        "message": {"content": "ok"},
                        "prompt_eval_count": 4,
                        "eval_count": 2,
                        "done_reason": "stop",
                    }
                ).encode("utf-8")

        client = rerun.MeasuredOllamaClient(
            url="http://localhost:11434/api/chat",
            model="test-model",
            temperature=0.7,
            top_p=0.9,
            num_ctx=1024,
            num_predict=32,
            timeout_s=600,
        )
        observed_timeouts = []

        def fake_urlopen(_request, timeout):
            observed_timeouts.append(timeout)
            return FakeResponse()

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            for timeout_s in (600, 1800):
                client.timeout_s = timeout_s
                response = client.chat([], seed=11)
                self.assertEqual(response.content, "ok")
        self.assertEqual(observed_timeouts, [600, 1800])
        self.assertEqual(client.request_attempts, 2)
        self.assertEqual(client.request_successes, 2)
        self.assertEqual(client.input_tokens, 8)
        self.assertEqual(client.output_tokens, 4)

    def test_request_success_and_failure_timing_are_accumulated(self):
        client = rerun.MeasuredOllamaClient(
            url="http://localhost:11434/api/chat",
            model="test-model",
            temperature=0.7,
            top_p=0.9,
            num_ctx=1024,
            num_predict=32,
            timeout_s=1,
        )
        response = rerun.LLMResponse("answer", 7, 3, "stop")
        with patch.object(rerun.OllamaClient, "chat", return_value=response):
            client.chat([], seed=11)
        self.assertEqual(client.request_attempts, 1)
        self.assertEqual(client.request_successes, 1)
        self.assertEqual(client.input_tokens, 7)
        self.assertEqual(client.output_tokens, 3)
        self.assertGreaterEqual(client.model_call_seconds, 0)

        client.reset_metrics()
        def fail_after_delay(*_args, **_kwargs):
            time.sleep(0.01)
            raise TimeoutError("timed out")

        with patch.object(
            rerun.OllamaClient, "chat", side_effect=fail_after_delay
        ):
            with self.assertRaises(TimeoutError):
                client.chat([], seed=11)
        self.assertEqual(client.request_attempts, 1)
        self.assertEqual(client.request_successes, 0)
        self.assertGreater(client.model_call_seconds, 0)


class OutcomeTests(unittest.TestCase):
    def setUp(self):
        self.task = Task(
            task_id="T1",
            question="What is the answer?",
            context="The answer is 1.",
            required_info=[],
            answer=1,
            answer_type="numeric",
            tolerance=0,
        )
        self.job = {
            "experiment_version": "latency_interleaved_v1",
            "model_tag": "test_model",
            "task_id": "T1",
            "condition": "generalist",
            "repetition": 0,
            "task_seed": 11,
            "schedule_seed": 123,
            "execution_order": 0,
            "configured_request_timeout_s": 5,
        }
        self.model_info = {"name": "test-model", "digest": "sha256:test"}
        self.client = rerun.MeasuredOllamaClient(
            url="http://localhost:11434/api/chat",
            model="test-model",
            temperature=0.7,
            top_p=0.9,
            num_ctx=1024,
            num_predict=32,
            timeout_s=5,
        )
        self.agents = []

    def test_success_is_execution_status_not_correctness(self):
        turn = SimpleNamespace(
            output_text="FINAL_ANSWER: 1",
            role="generalist",
            tool_calls=[],
        )
        pipeline_output = {
            "turns": [turn, turn, turn],
            "final_text": "FINAL_ANSWER: 1",
            "truncated": False,
        }
        with patch("runner_rerun.run_pipeline", return_value=pipeline_output):
            record = rerun.execute_job(
                self.job, self.task, self.agents, self.client, self.model_info,
                "latency_interleaved_v1", 1
            )
        self.assertEqual(record["outcome_category"], "success")
        self.assertTrue(record["pipeline_completed"])
        self.assertTrue(record["evaluation_completed"])
        self.assertTrue(record["correct"])
        self.assertGreaterEqual(record["elapsed_pipeline_seconds"], 0)

    def test_clean_incorrect_answer_is_not_conflated_with_correctness(self):
        turn = SimpleNamespace(
            output_text="FINAL_ANSWER: 2",
            role="generalist",
            tool_calls=[],
        )
        pipeline_output = {
            "turns": [turn, turn, turn],
            "final_text": "FINAL_ANSWER: 2",
            "truncated": False,
        }
        with patch("runner_rerun.run_pipeline", return_value=pipeline_output):
            record = rerun.execute_job(
                self.job, self.task, self.agents, self.client, self.model_info,
                "latency_interleaved_v1", 1
            )
        self.assertEqual(record["outcome_category"], "success")
        self.assertTrue(record["execution_successful"])
        self.assertFalse(record["correct"])

    def test_timeout_keeps_measured_failure_elapsed_time(self):
        def fail_after_delay(*_args, **_kwargs):
            time.sleep(0.01)
            raise TimeoutError("request timed out")

        with patch(
            "runner_rerun.run_pipeline",
            side_effect=fail_after_delay,
        ):
            record = rerun.execute_job(
                self.job, self.task, self.agents, self.client, self.model_info,
                "latency_interleaved_v1", 1
            )
        self.assertEqual(record["outcome_category"], "timeout")
        self.assertTrue(record["timed_out"])
        self.assertIsNone(record["correct"])
        self.assertGreater(record["elapsed_pipeline_seconds"], 0)
        self.assertNotEqual(record["error_details"]["message"], "")

    def test_pipeline_failure_preserves_partial_request_metrics(self):
        def fail_after_call(*_args, **_kwargs):
            self.client.chat([], seed=11)
            time.sleep(0.01)
            raise TimeoutError("request timed out")

        with patch.object(
            rerun.OllamaClient,
            "chat",
            return_value=rerun.LLMResponse("partial", 5, 2, "stop"),
        ), patch("runner_rerun.run_pipeline", side_effect=fail_after_call):
            record = rerun.execute_job(
                self.job,
                self.task,
                self.agents,
                self.client,
                self.model_info,
                "latency_interleaved_v1",
                1,
            )
        self.assertEqual(record["outcome_category"], "timeout")
        self.assertEqual(record["llm_calls_attempted"], 1)
        self.assertEqual(record["llm_calls_succeeded"], 1)
        self.assertEqual(record["input_tokens"], 5)
        self.assertEqual(record["output_tokens"], 2)
        self.assertGreater(record["elapsed_pipeline_seconds"], 0)
        self.assertGreaterEqual(record["accumulated_model_call_seconds"], 0)

    def test_generic_error_and_truncation_are_separate_outcomes(self):
        with patch(
            "runner_rerun.run_pipeline",
            side_effect=ValueError("bad response"),
        ):
            error_record = rerun.execute_job(
                self.job, self.task, self.agents, self.client, self.model_info,
                "latency_interleaved_v1", 1
            )
        self.assertEqual(error_record["outcome_category"], "generic_error")
        self.assertTrue(error_record["generic_error"])

        with patch(
            "runner_rerun.run_pipeline",
            side_effect=urllib.error.URLError("connection refused"),
        ):
            model_error_record = rerun.execute_job(
                self.job, self.task, self.agents, self.client, self.model_info,
                "latency_interleaved_v1", 1
            )
        self.assertEqual(model_error_record["outcome_category"], "model_error")
        self.assertTrue(model_error_record["model_error"])

        turn = SimpleNamespace(
            output_text="FINAL_ANSWER: 1",
            role="generalist",
            tool_calls=[],
        )
        pipeline_output = {
            "turns": [turn, turn, turn],
            "final_text": "FINAL_ANSWER: 1",
            "truncated": True,
        }
        with patch("runner_rerun.run_pipeline", return_value=pipeline_output):
            truncated_record = rerun.execute_job(
                self.job, self.task, self.agents, self.client, self.model_info,
                "latency_interleaved_v1", 1
            )
        self.assertEqual(truncated_record["outcome_category"], "truncation")
        self.assertTrue(truncated_record["correct"])
        self.assertTrue(truncated_record["truncated"])


class PersistenceTests(unittest.TestCase):
    def test_jsonl_sqlite_resume_and_interrupted_attempt_recovery(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(rerun, "RERUN_ROOT", Path(temp).resolve()):
                output_dir = Path(temp) / "latency_interleaved_v1" / "pilot" / "qwen"
                connection = rerun.open_store(output_dir)
                experiment_id = "run::one"
                rerun.append_attempt_event(
                    connection,
                    output_dir,
                    experiment_id,
                    "run::one::attempt1",
                    "attempt_started",
                    {"attempt_number": 1},
                )
                interrupted = rerun.recover_interrupted_attempts(
                    connection, output_dir, set()
                )
                self.assertEqual(interrupted[experiment_id], 1)
                events = [
                    json.loads(line)
                    for line in (output_dir / "attempts.jsonl").read_text(encoding="utf-8").splitlines()
                ]
                self.assertEqual(
                    [event["event_type"] for event in events],
                    ["attempt_started", "attempt_interrupted"],
                )
                self.assertEqual(
                    connection.execute("SELECT COUNT(*) FROM attempt_events").fetchone()[0],
                    2,
                )
                record = {
                    "experiment_id": "run::finished",
                    "schema_version": rerun.SCHEMA_VERSION,
                    "experiment_version": "latency_interleaved_v1",
                    "model_tag": "qwen",
                    "configured_request_timeout_s": 1800,
                    "task_id": "T1",
                    "condition": "generalist",
                    "repetition": 0,
                    "execution_order": 0,
                    "outcome_category": "success",
                    "elapsed_pipeline_seconds": 1.25,
                }
                rerun._write_record(connection, output_dir, record)
                connection.close()

                rows = rerun.load_records(output_dir / "runs.jsonl")
                self.assertEqual(set(rows), {"run::finished"})
                self.assertEqual(
                    rerun.pending_jobs(
                        [{
                            "experiment_version": "latency_interleaved_v1",
                            "model_tag": "qwen",
                            "condition": "generalist",
                            "task_id": "T1",
                            "repetition": 0,
                        }],
                        "latency_interleaved_v1",
                        {
                            "latency_interleaved_v1::qwen::generalist::T1::rep0": record
                        },
                    ),
                    [],
                )
                connection = rerun.open_store(output_dir)
                rerun.sync_store_from_jsonl(connection, rows)
                db_rows = connection.execute("SELECT experiment_id, record_json FROM runs").fetchall()
                self.assertEqual(len(db_rows), 1)
                self.assertEqual(json.loads(db_rows[0][1]), record)
                connection.close()

    def test_runner_resume_and_timeout_arm_isolation_with_fake_pipeline(self):
        with tempfile.TemporaryDirectory() as temp:
            rerun_root = Path(temp).resolve()
            with patch.object(rerun, "RERUN_ROOT", rerun_root):
                rerun.prepare_schedule("timeout", "qwen2_5_7b", pilot=True)

                def fake_pipeline(_agents, task, _seed):
                    answer = str(task.answer)
                    text = f"FINAL_ANSWER: {answer}"
                    turns = [
                        SimpleNamespace(
                            output_text=text,
                            role="generalist",
                            tool_calls=[],
                        )
                        for _ in range(3)
                    ]
                    return {
                        "turns": turns,
                        "final_text": text,
                        "truncated": False,
                    }

                with (
                    patch.object(
                        rerun,
                        "validate_model_available",
                        return_value={"name": "qwen2.5:7b", "digest": "sha256:test"},
                    ),
                    patch.object(rerun, "build_agents", return_value=[object()] * 3),
                    patch.object(rerun, "run_pipeline", side_effect=fake_pipeline),
                ):
                    rerun.run_schedule("timeout", "qwen2_5_7b", pilot=True)

                    output_root = rerun_root / "timeout_sensitivity_v1" / "pilot"
                    arm_records = {}
                    for timeout_s in (600, 1800):
                        arm_dir = output_root / f"timeout_{timeout_s}s" / "qwen2_5_7b"
                        records = rerun.load_records(arm_dir / "runs.jsonl")
                        self.assertEqual(len(records), 9)
                        self.assertTrue(
                            all(
                                row["configured_request_timeout_s"] == timeout_s
                                for row in records.values()
                            )
                        )
                        arm_records[timeout_s] = records

                    arm_keys = {
                        timeout_s: {
                            (row["task_id"], row["condition"], row["repetition"])
                            for row in records.values()
                        }
                        for timeout_s, records in arm_records.items()
                    }
                    self.assertEqual(arm_keys[600], arm_keys[1800])
                    self.assertFalse(
                        (output_root / "runs.jsonl").exists(),
                        "timeout-arm records must not be combined in a parent JSONL",
                    )
                    report = result_validator.validate_experiment(
                        "timeout", "qwen2_5_7b", pilot=True
                    )
                    self.assertTrue(report["valid"], report["errors"])
                    self.assertEqual(report["planned"], 18)
                    self.assertEqual(report["completed"], 18)
                    summary = result_summary.summarize(
                        "timeout", "qwen2_5_7b", pilot=True
                    )
                    self.assertTrue(summary["validation_passed"])
                    self.assertEqual(summary["completed_runs"], 18)
                    for arm in summary["arms"]:
                        self.assertEqual(arm["completed"], 9)
                        self.assertEqual(
                            sum(
                                counts["success"]
                                for counts in arm["counts_by_condition"].values()
                            ),
                            9,
                        )

                    test_report = rerun_root / "offline_test_report.txt"
                    test_report.write_text("offline tests passed\n", encoding="utf-8")
                    manifest_path = manifest_rerun.create_manifest(
                        "timeout",
                        "qwen2_5_7b",
                        pilot=True,
                        output=output_root / "pilot" / "SHA256SUMS",
                        test_report=test_report,
                    )
                    manifest_text = manifest_path.read_text(encoding="utf-8")
                    self.assertIn("runs.jsonl", manifest_text)
                    self.assertIn("schedule.jsonl", manifest_text)
                    self.assertIn("offline_test_report.txt", manifest_text)
                    with self.assertRaises(FileExistsError):
                        manifest_rerun.create_manifest(
                            "timeout",
                            "qwen2_5_7b",
                            pilot=True,
                            output=manifest_path,
                            test_report=test_report,
                        )

                    rerun.run_schedule("timeout", "qwen2_5_7b", pilot=True)
                    for timeout_s in (600, 1800):
                        arm_dir = output_root / f"timeout_{timeout_s}s" / "qwen2_5_7b"
                        self.assertEqual(
                            len(rerun.load_records(arm_dir / "runs.jsonl")),
                            9,
                            "resume must skip all terminal scheduled run IDs",
                        )

    def test_output_path_guard_rejects_historical_locations(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(rerun, "RERUN_ROOT", (Path(temp) / "reruns").resolve()):
                rerun.assert_new_path(Path(temp) / "reruns" / "latency")
                with self.assertRaises(ValueError):
                    rerun.assert_new_path(Path(temp) / "results" / "v2")

    def test_store_refuses_sqlite_only_records(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(rerun, "RERUN_ROOT", Path(temp).resolve()):
                output_dir = Path(temp) / "latency"
                connection = rerun.open_store(output_dir)
                phantom = {
                    "experiment_id": "sqlite-only",
                    "schema_version": 1,
                    "experiment_version": "latency_interleaved_v1",
                    "model_tag": "qwen",
                    "configured_request_timeout_s": 1800,
                    "task_id": "T1",
                    "condition": "generalist",
                    "repetition": 0,
                    "execution_order": 0,
                    "outcome_category": "success",
                    "elapsed_pipeline_seconds": 1.0,
                }
                connection.execute(
                    "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    rerun._sqlite_record_values(phantom),
                )
                connection.commit()
                with self.assertRaisesRegex(ValueError, "absent from JSONL"):
                    rerun.sync_store_from_jsonl(connection, {})
                connection.close()

    def test_global_lock_prevents_concurrent_model_tags(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(rerun, "RERUN_ROOT", Path(temp).resolve()):
                with rerun.model_run_lock("qwen2_5_7b"):
                    with self.assertRaisesRegex(RuntimeError, "global model lock"):
                        with rerun.model_run_lock("mistral_7b"):
                            self.fail("global model lock allowed concurrent model tag")


if __name__ == "__main__":
    unittest.main()
