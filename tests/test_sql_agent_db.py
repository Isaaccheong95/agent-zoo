from __future__ import annotations

import io
import os
import sqlite3
import sys
import unittest
import uuid
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = CURRENT_FILE.parent if (CURRENT_FILE.parent / "src").exists() else CURRENT_FILE.parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from SQL_agent.callbacks import (
    SQL_INTERNAL_QUERY_RESULT_STATE_KEY,
    SQL_INTERNAL_RESULT_REF_STATE_KEY,
    SQL_PUBLIC_RESULT_STATE_KEY,
    build_format_final_agent_response_callback,
    build_remember_query_result_callback,
)
from SQL_agent.config import SQLAgentSettings, load_settings
from SQL_agent.db import execute_sqlite_query, get_schema_summary, validate_sql_read_only
from SQL_agent.runtime import _print_debug_event


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


def make_query_result(
    rows: list[dict],
    *,
    columns: list[str] | None = None,
    row_count: int | None = None,
    truncated: bool = False,
    status: str = "success",
    error: str | None = None,
) -> dict:
    return {
        "status": status,
        "db_path": "fixture.sqlite",
        "sql": "SELECT ...",
        "columns": columns if columns is not None else (list(rows[0].keys()) if rows else []),
        "rows": rows,
        "row_count": len(rows) if row_count is None else row_count,
        "preview_row_count": len(rows),
        "truncated": truncated,
        "error": error,
    }


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


class SQLAgentPrivacyTestCase(unittest.TestCase):
    def _settings(self, **overrides) -> SQLAgentSettings:
        settings = SQLAgentSettings(
            db_path=REPO_ROOT / "dataset" / "titantic" / "titanic.sqlite",
            model="test-model",
        )
        for key, value in overrides.items():
            setattr(settings, key, value)
        return settings

    def _invoke_after_tool(self, settings: SQLAgentSettings, tool_response: dict) -> dict:
        callback = build_remember_query_result_callback(settings)
        tool = SimpleNamespace(name="execute_sqlite_read_only")
        tool_context = SimpleNamespace(state={})
        callback(tool, {}, tool_context, tool_response)
        return tool_context.state

    def test_load_settings_defaults_and_env_overrides(self) -> None:
        settings = load_settings(
            {
                "db_path": str(REPO_ROOT / "dataset" / "titantic" / "titanic.sqlite"),
                "model": "test-model",
            }
        )

        self.assertTrue(settings.count_aggregates_only)
        self.assertEqual(settings.minimum_aggregate_count, 3)
        self.assertFalse(settings.capture_internal_rows)

        with patch.dict(
            os.environ,
            {
                "SQL_AGENT_COUNT_AGGREGATES_ONLY": "false",
                "SQL_AGENT_MINIMUM_AGGREGATE_COUNT": "7",
                "SQL_AGENT_CAPTURE_INTERNAL_ROWS": "true",
            },
            clear=False,
        ):
            overridden = load_settings(
                {
                    "db_path": str(REPO_ROOT / "dataset" / "titantic" / "titanic.sqlite"),
                    "model": "test-model",
                }
            )

        self.assertFalse(overridden.count_aggregates_only)
        self.assertEqual(overridden.minimum_aggregate_count, 7)
        self.assertTrue(overridden.capture_internal_rows)

    def test_detail_rows_become_public_matching_count_by_default(self) -> None:
        state = self._invoke_after_tool(
            self._settings(),
            make_query_result(
                [
                    {"name": "Alice"},
                    {"name": "Bob"},
                ],
                columns=["name"],
                row_count=4,
                truncated=True,
            ),
        )

        public_result = state[SQL_PUBLIC_RESULT_STATE_KEY]
        self.assertEqual(public_result["status"], "success")
        self.assertEqual(public_result["rows"], [{"matching_count": 4}])
        self.assertEqual(public_result["matched_row_count"], 4)
        self.assertNotIn(SQL_INTERNAL_RESULT_REF_STATE_KEY, state)
        self.assertNotIn(SQL_INTERNAL_QUERY_RESULT_STATE_KEY, state)

    def test_capture_internal_rows_stores_raw_result_reference(self) -> None:
        raw_result = make_query_result(
            [
                {"name": "Alice"},
                {"name": "Bob"},
            ],
            columns=["name"],
            row_count=2,
        )
        state = self._invoke_after_tool(
            self._settings(capture_internal_rows=True),
            raw_result,
        )

        self.assertEqual(state[SQL_INTERNAL_RESULT_REF_STATE_KEY], SQL_INTERNAL_QUERY_RESULT_STATE_KEY)
        self.assertEqual(state[SQL_INTERNAL_QUERY_RESULT_STATE_KEY], raw_result)

    def test_scalar_counts_below_threshold_are_blocked(self) -> None:
        state = self._invoke_after_tool(
            self._settings(minimum_aggregate_count=3),
            make_query_result(
                [{"matching_count": 2}],
                columns=["matching_count"],
            ),
        )

        public_result = state[SQL_PUBLIC_RESULT_STATE_KEY]
        self.assertEqual(public_result["status"], "error")
        self.assertTrue(public_result["privacy_blocked"])
        self.assertEqual(public_result["matched_row_count"], 2)
        self.assertIn("minimum threshold (3)", public_result["error"])

    def test_grouped_counts_below_threshold_are_blocked(self) -> None:
        state = self._invoke_after_tool(
            self._settings(minimum_aggregate_count=3),
            make_query_result(
                [
                    {"sex": "female", "matching_count": 5},
                    {"sex": "male", "matching_count": 1},
                ],
                columns=["sex", "matching_count"],
                row_count=2,
            ),
        )

        public_result = state[SQL_PUBLIC_RESULT_STATE_KEY]
        self.assertEqual(public_result["status"], "error")
        self.assertTrue(public_result["privacy_blocked"])
        self.assertIn("group count is below the minimum threshold", public_result["error"])

    def test_grouped_counts_above_threshold_are_kept_publicly(self) -> None:
        state = self._invoke_after_tool(
            self._settings(minimum_aggregate_count=3),
            make_query_result(
                [
                    {"sex": "female", "matching_count": 5},
                    {"sex": "male", "matching_count": 4},
                ],
                columns=["sex", "matching_count"],
                row_count=2,
            ),
        )

        public_result = state[SQL_PUBLIC_RESULT_STATE_KEY]
        self.assertEqual(public_result["status"], "success")
        self.assertEqual(public_result["rows"][0]["sex"], "female")
        self.assertEqual(public_result["matched_row_count"], 9)

    def test_count_mode_off_preserves_public_passthrough(self) -> None:
        raw_result = make_query_result(
            [{"name": "Alice"}],
            columns=["name"],
            row_count=1,
        )
        state = self._invoke_after_tool(
            self._settings(count_aggregates_only=False),
            raw_result,
        )

        self.assertEqual(state[SQL_PUBLIC_RESULT_STATE_KEY], raw_result)

    def test_formatted_final_response_omits_raw_rows_in_privacy_mode(self) -> None:
        settings = self._settings()
        tool_state = self._invoke_after_tool(
            settings,
            make_query_result(
                [
                    {"name": "Alice"},
                    {"name": "Bob"},
                ],
                columns=["name"],
                row_count=4,
            ),
        )
        callback = build_format_final_agent_response_callback(settings)
        content = callback(SimpleNamespace(state=tool_state))

        self.assertIsNotNone(content)
        response_text = content.parts[0].text
        self.assertIn("Found 4 matching rows.", response_text)
        self.assertNotIn("Alice", response_text)
        self.assertNotIn('"name"', response_text)

    def test_debug_output_redacts_tool_response_in_privacy_mode(self) -> None:
        settings = self._settings()
        event = SimpleNamespace(
            author="tool",
            content=SimpleNamespace(
                parts=[
                    SimpleNamespace(
                        text=None,
                        function_call=None,
                        function_response=SimpleNamespace(
                            name="execute_sqlite_read_only",
                            response={"rows": [{"name": "Alice"}]},
                        ),
                    )
                ]
            ),
        )

        output = io.StringIO()
        with redirect_stdout(output):
            _print_debug_event(event, settings)

        debug_text = output.getvalue()
        self.assertIn("<redacted in privacy mode>", debug_text)
        self.assertNotIn("Alice", debug_text)


if __name__ == "__main__":
    unittest.main()

