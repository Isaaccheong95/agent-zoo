from __future__ import annotations

import asyncio
import json
import sqlite3
import sys
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch


CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = CURRENT_FILE.parent if (CURRENT_FILE.parent / "src").exists() else CURRENT_FILE.parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agent_zoo.base import BaseAgent
from agent_zoo.sql_agent import SQLAgent, SQLAgentSettings
from agent_zoo.sql_agent.pipeline import run_nl_to_sql_pipeline
from agent_zoo.sql_agent.runtime import ask_question, ask_question_structured


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

    def test_ask_question_returns_user_message_from_structured_envelope(self) -> None:
        structured_response = json.dumps(
            {
                "schema_version": 1,
                "response_type": "sql_result",
                "user_message": "Found 216 matching rows.",
                "sql_result": {
                    "status": "success",
                    "rows": [{"matching_count": 216}],
                },
            }
        )

        class FakeRunner:
            def __init__(self, final_text: str) -> None:
                self.app_name = "test-app"
                self.session_service = SimpleNamespace(create_session=AsyncMock())
                self._final_text = final_text

            async def run_async(self, **kwargs):
                yield SimpleNamespace(
                    author="sql_agent",
                    content=SimpleNamespace(parts=[SimpleNamespace(text=self._final_text)]),
                    is_final_response=lambda: True,
                )

        response = asyncio.run(
            ask_question(
                "How many drinkers?",
                SQLAgentSettings(db_path=self.db_path, model="openai/test-model"),
                runner=FakeRunner(structured_response),
                session_id="test-session",
            )
        )

        self.assertEqual(response, "Found 216 matching rows.")

    def test_ask_question_structured_returns_response_envelope(self) -> None:
        structured_response = json.dumps(
            {
                "schema_version": 1,
                "response_type": "clarification",
                "user_message": "Which drinking frequency do you mean?",
                "options": ["Occasionally", "Regularly"],
            }
        )

        class FakeRunner:
            def __init__(self, final_text: str) -> None:
                self.app_name = "test-app"
                self.session_service = SimpleNamespace(create_session=AsyncMock())
                self._final_text = final_text

            async def run_async(self, **kwargs):
                yield SimpleNamespace(
                    author="sql_agent",
                    content=SimpleNamespace(parts=[SimpleNamespace(text=self._final_text)]),
                    is_final_response=lambda: True,
                )

        response = asyncio.run(
            ask_question_structured(
                "How many drinkers?",
                SQLAgentSettings(db_path=self.db_path, model="openai/test-model"),
                runner=FakeRunner(structured_response),
                session_id="test-session",
            )
        )

        self.assertIsNotNone(response)
        self.assertEqual(response["response_type"], "clarification")
        self.assertEqual(response["options"], ["Occasionally", "Regularly"])


if __name__ == "__main__":
    unittest.main()
