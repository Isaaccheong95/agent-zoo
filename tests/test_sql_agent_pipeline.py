from __future__ import annotations

import asyncio
import sqlite3
import unittest
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, patch

from google.genai import types


REPO_ROOT = Path(__file__).resolve().parents[1]

from agent_zoo.base import BaseAgent
from agent_zoo.sql_agent import SQLAgent, SQLAgentSettings
from agent_zoo.sql_agent import runtime as sql_runtime
from agent_zoo.sql_agent.pipeline import run_nl_to_sql_pipeline


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


def create_missing_pipeline_fixture(db_path: Path) -> None:
    connection = sqlite3.connect(db_path)
    connection.executescript(
        """
        CREATE TABLE patients (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            sex TEXT,
            age TEXT
        );

        INSERT INTO patients (name, sex, age) VALUES
            ('Anya', 'female', '14'),
            ('Ben', 'male', '42'),
            ('Cara', NULL, NULL),
            ('Drew', '', ''),
            ('Eli', ' ', ' '),
            ('Fay', 'female', '33');
        """
    )
    connection.commit()
    connection.close()


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

    def test_namespace_exports_instantiable_sqlagent(self) -> None:
        agent = SQLAgent(
            settings=SQLAgentSettings(
                db_path=self.db_path,
                model="openai/test-model",
            )
        )

        self.assertIsInstance(agent, BaseAgent)
        self.assertEqual(agent.get_name(), "sql_agent")
        self.assertEqual(
            agent.get_description(),
            "Converts natural language questions into safe read-only SQLite queries and explains the results.",
        )

    def test_sqlagent_ask_delegates_to_runtime(self) -> None:
        settings = SQLAgentSettings(
            db_path=self.db_path,
            model="openai/test-model",
        )
        agent = SQLAgent(settings=settings)
        mocked_ask_question = AsyncMock(return_value="Found 2 matching rows.")

        with patch("agent_zoo.sql_agent.runtime.ask_question", new=mocked_ask_question):
            response = asyncio.run(
                agent.ask(
                    "How many female patients are below 45 years old?",
                    session_id="test-session",
                )
            )

        self.assertEqual(response, "Found 2 matching rows.")
        mocked_ask_question.assert_awaited_once_with(
            "How many female patients are below 45 years old?",
            settings,
            session_id="test-session",
        )

    def test_ask_question_reuses_cached_runner_for_same_session(self) -> None:
        settings = SQLAgentSettings(
            db_path=self.db_path,
            model="openai/test-model",
            app_name=f"test-app-{uuid.uuid4().hex}",
            session_id=f"test-session-{uuid.uuid4().hex}",
        )

        created_runners: list[object] = []

        class FakeSessionService:
            def __init__(self) -> None:
                self.created_sessions: list[dict[str, str]] = []

            async def create_session(self, **kwargs) -> None:
                self.created_sessions.append(kwargs)

        class FakeFinalEvent:
            def __init__(self, text: str) -> None:
                self.author = "model"
                self.content = types.Content(role="model", parts=[types.Part(text=text)])

            def is_final_response(self) -> bool:
                return True

        class FakeRunner:
            def __init__(self, agent, app_name: str) -> None:
                self.agent = agent
                self.app_name = app_name
                self.session_service = FakeSessionService()
                created_runners.append(self)

            async def run_async(self, **kwargs):
                yield FakeFinalEvent("ok")

        sql_runtime._RUNNER_CACHE.clear()
        sql_runtime._INITIALIZED_SESSION_KEYS.clear()

        with patch("agent_zoo.sql_agent.runtime.InMemoryRunner", FakeRunner), patch(
            "agent_zoo.sql_agent.agent.build_root_agent",
            return_value=object(),
        ):
            first_response = asyncio.run(sql_runtime.ask_question("first", settings))
            second_response = asyncio.run(sql_runtime.ask_question("second", settings))

        self.assertEqual(first_response, "ok")
        self.assertEqual(second_response, "ok")
        self.assertEqual(len(created_runners), 1)
        self.assertEqual(len(created_runners[0].session_service.created_sessions), 1)

        sql_runtime._RUNNER_CACHE.clear()
        sql_runtime._INITIALIZED_SESSION_KEYS.clear()


class SQLPipelineMissingGroupTestCase(unittest.TestCase):
    def setUp(self) -> None:
        temp_root = REPO_ROOT / ".tmp_test_runs"
        temp_root.mkdir(exist_ok=True)
        self.db_path = temp_root / f"{uuid.uuid4().hex}.sqlite"
        create_missing_pipeline_fixture(self.db_path)

    def tearDown(self) -> None:
        if self.db_path.exists():
            self.db_path.unlink()

    def test_run_nl_to_sql_pipeline_surfaces_null_bucket_for_case_groups(self) -> None:
        def generator(_: str, __: dict) -> dict:
            return {
                "sql": """
                    SELECT CASE
                        WHEN CAST(age AS INTEGER) <= 17 THEN '0-17'
                        WHEN CAST(age AS INTEGER) BETWEEN 18 AND 39 THEN '18-39'
                        WHEN CAST(age AS INTEGER) >= 40 THEN '40+'
                    END AS age_category,
                    COUNT(*) AS patient_count
                    FROM patients
                    GROUP BY age_category
                    ORDER BY age_category
                """,
                "explanation": "Buckets patients into age ranges.",
            }

        result = run_nl_to_sql_pipeline(
            "Split patients by age categories.",
            str(self.db_path),
            generator,
        )

        self.assertEqual(result["status"], "success")
        self.assertIn("'Null'", result["execution"].get("display_sql", ""))
        self.assertEqual(
            {row["age_category"]: row["patient_count"] for row in result["execution"]["rows"]},
            {
                "0-17": 1,
                "18-39": 1,
                "40+": 1,
                "Null": 3,
            },
        )


if __name__ == "__main__":
    unittest.main()
