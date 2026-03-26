from __future__ import annotations

import asyncio
import json
import sqlite3
import sys
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace


CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = CURRENT_FILE.parent if (CURRENT_FILE.parent / "src").exists() else CURRENT_FILE.parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from SQL_agent.benchmark import (
    AgentRunnerCache,
    BirdBenchmarkExample,
    collect_prediction_from_events,
    extract_sql_from_final_response,
    generate_benchmark_predictions,
    write_prediction_artifacts,
)
from SQL_agent.pipeline import run_nl_to_sql_pipeline


def create_pipeline_fixture(db_path: Path) -> None:
    connection = sqlite3.connect(db_path)
    connection.executescript(
        """
        CREATE TABLE patients (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            sex TEXT NOT NULL,
            age INTEGER
        );

        INSERT INTO patients (name, sex, age) VALUES
            ('Anya', 'female', 33),
            ('Ben', 'male', 40),
            ('Clare', 'female', 44),
            ('Drew', 'male', 55);
        """
    )
    connection.commit()
    connection.close()


def create_benchmark_fixture(db_path: Path) -> None:
    connection = sqlite3.connect(db_path)
    connection.executescript(
        """
        CREATE TABLE items (
            id INTEGER PRIMARY KEY,
            label TEXT NOT NULL
        );

        INSERT INTO items (label) VALUES
            ('alpha'),
            ('beta');
        """
    )
    connection.commit()
    connection.close()


class FakeEvent:
    def __init__(self, *, parts, final_response: bool = False, author: str = "model") -> None:
        self.author = author
        self.content = SimpleNamespace(parts=parts)
        self._final_response = final_response

    def is_final_response(self) -> bool:
        return self._final_response


class FakeSessionService:
    def __init__(self) -> None:
        self.created_sessions: list[dict[str, str]] = []

    async def create_session(self, *, app_name: str, user_id: str, session_id: str):
        record = {
            "app_name": app_name,
            "user_id": user_id,
            "session_id": session_id,
        }
        self.created_sessions.append(record)
        return record


class FakeRunner:
    def __init__(
        self,
        *,
        app_name: str,
        db_path: Path,
        emit_tool_response: bool = True,
    ) -> None:
        self.app_name = app_name
        self.db_path = Path(db_path)
        self.emit_tool_response = emit_tool_response
        self.session_service = FakeSessionService()
        self.calls: list[dict[str, str]] = []

    async def _event_stream(self, question: str, sql: str):
        if self.emit_tool_response:
            yield FakeEvent(
                parts=[
                    SimpleNamespace(
                        text=None,
                        function_response=SimpleNamespace(
                            name="execute_sqlite_read_only",
                            response={
                                "status": "success",
                                "sql": sql,
                                "error": None,
                            },
                        ),
                    )
                ]
            )

        yield FakeEvent(
            parts=[
                SimpleNamespace(
                    text=(
                        "Generated SQL:\n"
                        f"```sql\n{sql}\n```\n\n"
                        "Result summary:\nFound 1 matching row."
                    ),
                    function_response=None,
                )
            ],
            final_response=True,
        )

    def run_async(self, *, user_id: str, session_id: str, new_message):
        question = new_message.parts[0].text
        sql = f"SELECT '{self.db_path.stem}:{question}' AS answer"
        self.calls.append(
            {
                "user_id": user_id,
                "session_id": session_id,
                "question": question,
                "db_path": str(self.db_path),
                "sql": sql,
            }
        )
        return self._event_stream(question, sql)


class SQLPipelineTestCase(unittest.TestCase):
    def setUp(self) -> None:
        temp_root = REPO_ROOT / ".tmp_test_runs"
        temp_root.mkdir(exist_ok=True)
        self.db_path = temp_root / f"{uuid.uuid4().hex}.sqlite"
        create_pipeline_fixture(self.db_path)

    def tearDown(self) -> None:
        if self.db_path.exists():
            self.db_path.unlink()

    def test_run_nl_to_sql_pipeline_success(self) -> None:
        def generator(question: str, schema_summary: dict) -> dict:
            self.assertIn("patients", schema_summary["schema_text"])
            self.assertEqual(question, "How many female patients are below 45 years old?")
            return {
                "sql": """
                    SELECT COUNT(*) AS patient_count
                    FROM patients
                    WHERE LOWER(sex) = 'female' AND age < 45
                """,
                "explanation": "Counts female patients younger than 45.",
            }

        result = run_nl_to_sql_pipeline(
            "How many female patients are below 45 years old?",
            str(self.db_path),
            generator,
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["generated_sql"], "SELECT COUNT(*) AS patient_count FROM patients WHERE LOWER(sex) = 'female' AND age < 45")
        self.assertEqual(result["execution"]["rows"][0]["patient_count"], 2)
        self.assertEqual(result["summary"], "Found 1 matching row.")

    def test_run_nl_to_sql_pipeline_handles_invalid_sql_gracefully(self) -> None:
        def generator(_: str, __: dict) -> str:
            return "SELECT missing FROM patients"

        result = run_nl_to_sql_pipeline(
            "Show the missing field",
            str(self.db_path),
            generator,
        )

        self.assertEqual(result["status"], "error")
        self.assertIn("no such column", result["execution"]["error"].lower())
        self.assertIn("no such column", result["summary"].lower())

    def test_run_nl_to_sql_pipeline_handles_empty_generation(self) -> None:
        def generator(_: str, __: dict) -> None:
            return None

        result = run_nl_to_sql_pipeline(
            "Answer something ambiguous",
            str(self.db_path),
            generator,
        )

        self.assertEqual(result["status"], "error")
        self.assertIsNone(result["generated_sql"])
        self.assertIn("No SQL was generated", result["summary"])


class SQLBenchmarkWrapperTestCase(unittest.TestCase):
    def setUp(self) -> None:
        temp_root = REPO_ROOT / ".tmp_test_runs"
        temp_root.mkdir(exist_ok=True)
        self.temp_dir = temp_root / f"benchmark_{uuid.uuid4().hex}"
        self.temp_dir.mkdir()

        self.db_root = self.temp_dir / "db_root"
        self.db_a_dir = self.db_root / "alpha"
        self.db_b_dir = self.db_root / "beta"
        self.db_a_dir.mkdir(parents=True)
        self.db_b_dir.mkdir(parents=True)
        self.db_a_path = self.db_a_dir / "alpha.sqlite"
        self.db_b_path = self.db_b_dir / "beta.sqlite"
        create_benchmark_fixture(self.db_a_path)
        create_benchmark_fixture(self.db_b_path)

    def tearDown(self) -> None:
        for path in sorted(self.temp_dir.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()

    def test_extract_sql_from_final_response_code_block(self) -> None:
        response_text = (
            "Generated SQL:\n"
            "```sql\n"
            "SELECT COUNT(*) AS matching_count FROM items\n"
            "```\n\n"
            "Result summary:\nFound 1 matching row."
        )

        extracted = extract_sql_from_final_response(response_text)

        self.assertEqual(extracted, "SELECT COUNT(*) AS matching_count FROM items")

    def test_collect_prediction_prefers_tool_response_sql(self) -> None:
        async def event_stream():
            yield FakeEvent(
                parts=[
                    SimpleNamespace(
                        text=None,
                        function_response=SimpleNamespace(
                            name="execute_sqlite_read_only",
                            response={
                                "status": "success",
                                "sql": "SELECT 1 AS answer",
                                "error": None,
                            },
                        ),
                    )
                ]
            )
            yield FakeEvent(
                parts=[
                    SimpleNamespace(
                        text="Generated SQL:\n```sql\nSELECT 2 AS answer\n```",
                        function_response=None,
                    )
                ],
                final_response=True,
            )

        capture = asyncio.run(collect_prediction_from_events(event_stream()))

        self.assertEqual(capture.sql, "SELECT 1 AS answer")
        self.assertEqual(capture.status, "success")
        self.assertIsNone(capture.error)

    def test_collect_prediction_falls_back_to_final_response_sql(self) -> None:
        async def event_stream():
            yield FakeEvent(
                parts=[
                    SimpleNamespace(
                        text="Generated SQL:\n```sql\nSELECT 99 AS answer\n```",
                        function_response=None,
                    )
                ],
                final_response=True,
            )

        capture = asyncio.run(collect_prediction_from_events(event_stream()))

        self.assertEqual(capture.sql, "SELECT 99 AS answer")
        self.assertEqual(capture.status, "success")

    def test_runner_cache_builds_db_specific_settings_and_reuses_per_db(self) -> None:
        captured_paths: list[str] = []

        def runner_factory(db_id, settings):
            captured_paths.append(str(settings.db_path))
            return FakeRunner(app_name=f"app_{db_id}", db_path=settings.db_path)

        cache = AgentRunnerCache(model="test-model", runner_factory=runner_factory)

        alpha_bundle = cache.get_bundle("alpha", self.db_a_path)
        beta_bundle = cache.get_bundle("beta", self.db_b_path)
        alpha_bundle_again = cache.get_bundle("alpha", self.db_a_path)

        self.assertEqual(alpha_bundle.db_path, self.db_a_path.resolve())
        self.assertEqual(beta_bundle.db_path, self.db_b_path.resolve())
        self.assertIs(alpha_bundle, alpha_bundle_again)
        self.assertEqual(captured_paths, [str(self.db_a_path.resolve()), str(self.db_b_path.resolve())])

    def test_generate_predictions_uses_fresh_sessions_and_correct_db_routing(self) -> None:
        created_runners: dict[str, FakeRunner] = {}

        def runner_factory(db_id, settings):
            runner = FakeRunner(app_name=f"app_{db_id}", db_path=settings.db_path)
            created_runners[db_id] = runner
            return runner

        examples = [
            BirdBenchmarkExample(example_id=0, db_id="alpha", question="first question"),
            BirdBenchmarkExample(example_id=1, db_id="alpha", question="second question"),
            BirdBenchmarkExample(example_id=2, db_id="beta", question="third question"),
        ]

        results = asyncio.run(
            generate_benchmark_predictions(
                examples,
                db_root=self.db_root,
                runner_cache=AgentRunnerCache(model="test-model", runner_factory=runner_factory),
            )
        )

        alpha_sessions = [entry["session_id"] for entry in created_runners["alpha"].session_service.created_sessions]
        beta_sessions = [entry["session_id"] for entry in created_runners["beta"].session_service.created_sessions]

        self.assertEqual(len(results), 3)
        self.assertEqual(len(alpha_sessions), 2)
        self.assertEqual(len(beta_sessions), 1)
        self.assertEqual(len(set(alpha_sessions + beta_sessions)), 3)
        self.assertEqual(created_runners["alpha"].calls[0]["db_path"], str(self.db_a_path.resolve()))
        self.assertEqual(created_runners["beta"].calls[0]["db_path"], str(self.db_b_path.resolve()))

    def test_integration_writes_bird_prediction_json_and_debug_trace(self) -> None:
        created_runners: dict[str, FakeRunner] = {}

        def runner_factory(db_id, settings):
            runner = FakeRunner(app_name=f"app_{db_id}", db_path=settings.db_path)
            created_runners[db_id] = runner
            return runner

        examples = [
            BirdBenchmarkExample(
                example_id=0,
                db_id="alpha",
                question="first question",
                gold_sql="SELECT 'gold alpha'",
                difficulty="simple",
                question_id=101,
            ),
            BirdBenchmarkExample(
                example_id=1,
                db_id="beta",
                question="second question",
                gold_sql="SELECT 'gold beta'",
                difficulty="moderate",
                question_id=202,
            ),
        ]

        results = asyncio.run(
            generate_benchmark_predictions(
                examples,
                db_root=self.db_root,
                runner_cache=AgentRunnerCache(model="test-model", runner_factory=runner_factory),
            )
        )
        prediction_path, debug_trace_path = write_prediction_artifacts(results, self.temp_dir / "output")

        payload = json.loads(prediction_path.read_text(encoding="utf-8"))
        debug_rows = [json.loads(line) for line in debug_trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]

        self.assertEqual(sorted(payload.keys()), ["0", "1"])
        self.assertIn("\t----- bird -----\talpha", payload["0"])
        self.assertIn("\t----- bird -----\tbeta", payload["1"])
        self.assertEqual(len(debug_rows), 2)
        self.assertEqual(debug_rows[0]["db_id"], "alpha")
        self.assertEqual(debug_rows[1]["db_id"], "beta")
        self.assertNotEqual(debug_rows[0]["session_id"], debug_rows[1]["session_id"])


if __name__ == "__main__":
    unittest.main()
