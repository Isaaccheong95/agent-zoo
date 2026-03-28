from __future__ import annotations

import sqlite3
import sys
import unittest
import uuid
from pathlib import Path


CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = CURRENT_FILE.parent if (CURRENT_FILE.parent / "src").exists() else CURRENT_FILE.parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agent_zoo.sql_agent import SQLAgent
from agent_zoo.sql_agent.agent import build_root_agent
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


    def test_namespace_exports_sqlagent_alias(self) -> None:
        self.assertIs(SQLAgent, build_root_agent)

if __name__ == "__main__":
    unittest.main()



