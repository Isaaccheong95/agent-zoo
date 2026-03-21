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

from SQL_agent.db import execute_sqlite_query, get_schema_summary, validate_sql_read_only


def create_fixture_database(db_path: Path) -> None:
    connection = sqlite3.connect(db_path)
    connection.executescript(
        """
        CREATE TABLE people (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            sex TEXT,
            age INTEGER
        );

        CREATE TABLE visits (
            id INTEGER PRIMARY KEY,
            person_id INTEGER NOT NULL,
            city TEXT NOT NULL,
            FOREIGN KEY(person_id) REFERENCES people(id)
        );

        CREATE TABLE __shadow (
            id INTEGER PRIMARY KEY,
            ignored TEXT
        );

        INSERT INTO people (name, sex, age) VALUES
            ('Alice', 'female', 30),
            ('Bob', 'male', 44),
            ('Cara', 'female', 19),
            ('Dana', 'female', 52);

        INSERT INTO visits (person_id, city) VALUES
            (1, 'Singapore'),
            (2, 'Tokyo'),
            (3, 'Singapore'),
            (4, 'Paris');
        """
    )
    connection.commit()
    connection.close()


class SQLiteHelpersTestCase(unittest.TestCase):
    def setUp(self) -> None:
        temp_root = REPO_ROOT / ".tmp_test_runs"
        temp_root.mkdir(exist_ok=True)
        self.db_path = temp_root / f"{uuid.uuid4().hex}.sqlite"
        create_fixture_database(self.db_path)

    def tearDown(self) -> None:
        if self.db_path.exists():
            self.db_path.unlink()

    def test_get_schema_summary_excludes_internal_tables(self) -> None:
        summary = get_schema_summary(self.db_path)

        self.assertEqual(summary["status"], "success")
        table_names = [table["name"] for table in summary["tables"]]
        self.assertEqual(table_names, ["people", "visits"])
        self.assertIn("people(id INTEGER PRIMARY KEY", summary["schema_text"])
        self.assertNotIn("__shadow", summary["schema_text"])

    def test_validate_sql_read_only_allows_select_and_with(self) -> None:
        select_validation = validate_sql_read_only(
            "SELECT name FROM people WHERE age > 20 ORDER BY name",
            self.db_path,
        )
        with_validation = validate_sql_read_only(
            """
            WITH filtered AS (
                SELECT name FROM people WHERE sex = 'female'
            )
            SELECT name FROM filtered ORDER BY name
            """,
            self.db_path,
        )

        self.assertTrue(select_validation["is_valid"])
        self.assertTrue(with_validation["is_valid"])

    def test_validate_sql_read_only_blocks_unsafe_and_multi_statement_sql(self) -> None:
        unsafe_validation = validate_sql_read_only("DELETE FROM people", self.db_path)
        multi_statement_validation = validate_sql_read_only(
            "SELECT * FROM people; DROP TABLE people;",
            self.db_path,
        )

        self.assertFalse(unsafe_validation["is_valid"])
        self.assertIn("Only read-only SELECT and WITH queries are allowed", unsafe_validation["reason"])
        self.assertFalse(multi_statement_validation["is_valid"])
        self.assertIn("single SQL statement", multi_statement_validation["reason"])

    def test_validate_sql_read_only_catches_hallucinated_columns(self) -> None:
        validation = validate_sql_read_only(
            "SELECT imaginary_column FROM people",
            self.db_path,
        )

        self.assertFalse(validation["is_valid"])
        self.assertIn("no such column", validation["reason"].lower())

    def test_execute_sqlite_query_returns_truncated_preview_and_total_count(self) -> None:
        result = execute_sqlite_query(
            self.db_path,
            "SELECT name FROM people ORDER BY name",
            preview_rows=2,
        )

        self.assertEqual(result["status"], "success")
        self.assertTrue(result["truncated"])
        self.assertEqual(result["row_count"], 4)
        self.assertEqual(result["preview_row_count"], 2)
        self.assertEqual(result["rows"][0]["name"], "Alice")

    def test_execute_sqlite_query_returns_structured_error_for_invalid_sql(self) -> None:
        result = execute_sqlite_query(
            self.db_path,
            "SELECT missing_column FROM people",
        )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["row_count"], 0)
        self.assertIn("no such column", result["error"].lower())


if __name__ == "__main__":
    unittest.main()
