from __future__ import annotations

import sqlite3
import unittest
import uuid
from pathlib import Path

from agent_zoo.sql_agent.db import execute_sqlite_query

from tests._fixtures import (
    REPO_ROOT,
    create_missing_group_fixture_database,
)


def create_patients_fixture(db_path: Path) -> None:
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


class ExecuteSqliteQueryTestCase(unittest.TestCase):
    def setUp(self) -> None:
        temp_root = REPO_ROOT / ".tmp_test_runs"
        temp_root.mkdir(exist_ok=True)
        self.db_path = temp_root / f"{uuid.uuid4().hex}.sqlite"
        create_patients_fixture(self.db_path)

    def tearDown(self) -> None:
        if self.db_path.exists():
            self.db_path.unlink()

    def test_executes_grounded_select_and_returns_rows(self) -> None:
        sql = (
            "SELECT COUNT(*) AS patient_count\n"
            "                    FROM patients\n"
            "                    WHERE LOWER(sex) = 'female' AND age < 45"
        )

        result = execute_sqlite_query(str(self.db_path), sql)

        self.assertEqual(result["status"], "success")
        self.assertEqual(
            result["sql"],
            "SELECT COUNT(*) AS patient_count FROM patients WHERE LOWER(sex) = 'female' AND age < 45",
        )
        self.assertEqual(result["rows"][0]["patient_count"], 2)
        self.assertEqual(result["row_count"], 1)

    def test_reports_schema_error_without_raising(self) -> None:
        result = execute_sqlite_query(str(self.db_path), "SELECT missing FROM patients")

        self.assertEqual(result["status"], "error")
        self.assertIn("no such column", (result.get("error") or "").lower())


class ExecuteSqliteQueryNullBucketTestCase(unittest.TestCase):
    def setUp(self) -> None:
        temp_root = REPO_ROOT / ".tmp_test_runs"
        temp_root.mkdir(exist_ok=True)
        self.db_path = temp_root / f"{uuid.uuid4().hex}.sqlite"
        create_missing_group_fixture_database(self.db_path)

    def tearDown(self) -> None:
        if self.db_path.exists():
            self.db_path.unlink()

    def test_case_group_surfaces_null_bucket_in_display_sql_and_rows(self) -> None:
        sql = (
            "SELECT CASE\n"
            "    WHEN CAST(age AS INTEGER) <= 17 THEN '0-17'\n"
            "    WHEN CAST(age AS INTEGER) BETWEEN 18 AND 39 THEN '18-39'\n"
            "    WHEN CAST(age AS INTEGER) >= 40 THEN '40+'\n"
            "END AS age_category,\n"
            "COUNT(*) AS patient_count\n"
            "FROM patients\n"
            "GROUP BY age_category\n"
            "ORDER BY age_category"
        )

        result = execute_sqlite_query(str(self.db_path), sql)

        self.assertEqual(result["status"], "success")
        self.assertIn("'Null'", result.get("display_sql", ""))
        self.assertEqual(
            {row["age_category"]: row["patient_count"] for row in result["rows"]},
            {
                "0-17": 1,
                "18-39": 1,
                "40+": 1,
                "Null": 3,
            },
        )


if __name__ == "__main__":
    unittest.main()
