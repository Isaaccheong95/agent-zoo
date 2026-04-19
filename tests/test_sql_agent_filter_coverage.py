from __future__ import annotations

import unittest
import uuid
from types import SimpleNamespace

from agent_zoo.sql_agent.callbacks import (
    SQL_LAST_USER_TEXT_STATE_KEY,
    SQL_PUBLIC_RESULT_STATE_KEY,
    build_remember_query_result_callback,
)
from agent_zoo.sql_agent.config import SQLAgentSettings
from agent_zoo.sql_agent.db import execute_sqlite_query

from tests._fixtures import (
    REPO_ROOT,
    create_missing_group_fixture_database,
)


class SQLAgentFilterCoverageTestCase(unittest.TestCase):
    def setUp(self) -> None:
        temp_root = REPO_ROOT / ".tmp_test_runs"
        temp_root.mkdir(exist_ok=True)
        self.db_path = temp_root / f"{uuid.uuid4().hex}.sqlite"
        create_missing_group_fixture_database(self.db_path)

    def tearDown(self) -> None:
        if self.db_path.exists():
            self.db_path.unlink()

    def _settings(self, **overrides) -> SQLAgentSettings:
        settings = SQLAgentSettings(
            db_path=self.db_path,
            model="test-model",
        )
        for key, value in overrides.items():
            setattr(settings, key, value)
        return settings

    def test_remember_query_result_adds_missing_blank_row_count_to_filter_coverage_context(self) -> None:
        sql = "SELECT COUNT(*) AS matching_count FROM patients WHERE sex IN ('female', 'male')"
        tool_response = execute_sqlite_query(self.db_path, sql)
        callback = build_remember_query_result_callback(self._settings(minimum_aggregate_count=1))
        tool = SimpleNamespace(name="execute_sqlite_read_only")
        tool_context = SimpleNamespace(state={SQL_LAST_USER_TEXT_STATE_KEY: "How many patients have a recorded sex value?"})

        callback(tool, {"sql": sql, "is_final": True}, tool_context, tool_response)

        public_result = tool_context.state[SQL_PUBLIC_RESULT_STATE_KEY]
        categorical_filters = public_result["query_summary_context"]["categorical_filters"]
        self.assertEqual(len(categorical_filters), 1)
        self.assertEqual(
            categorical_filters[0],
            {
                "column": "sex",
                "selected_values": ["female", "male"],
                "available_values": ["female", "male"],
                "dataset_missing_or_blank_count": 3,
                "dataset_missing_or_blank_count_unit": "rows",
            },
        )

    def test_remember_query_result_counts_whole_column_missing_values_at_patient_level(self) -> None:
        sql = "SELECT COUNT(*) AS matching_count FROM patients WHERE sex IN ('female', 'male')"
        tool_response = execute_sqlite_query(self.db_path, sql)
        callback = build_remember_query_result_callback(
            self._settings(minimum_aggregate_count=1, object_id_column="patient_id")
        )
        tool = SimpleNamespace(name="execute_sqlite_read_only")
        tool_context = SimpleNamespace(state={SQL_LAST_USER_TEXT_STATE_KEY: "How many patients have a recorded sex value?"})

        callback(tool, {"sql": sql, "is_final": True}, tool_context, tool_response)

        public_result = tool_context.state[SQL_PUBLIC_RESULT_STATE_KEY]
        self.assertEqual(
            public_result["query_summary_context"]["categorical_filters"],
            [
                {
                    "column": "sex",
                    "selected_values": ["female", "male"],
                    "available_values": ["female", "male"],
                    "dataset_missing_or_blank_count": 2,
                    "dataset_missing_or_blank_count_unit": "patients",
                }
            ],
        )

    def test_remember_query_result_normalizes_semicolon_sql_before_missing_blank_count(self) -> None:
        sql = "SELECT COUNT(*) AS matching_count FROM patients WHERE sex IN ('female', 'male');"
        tool_response = execute_sqlite_query(self.db_path, sql)
        callback = build_remember_query_result_callback(self._settings(minimum_aggregate_count=1))
        tool = SimpleNamespace(name="execute_sqlite_read_only")
        tool_context = SimpleNamespace(state={SQL_LAST_USER_TEXT_STATE_KEY: "How many patients have a recorded sex value?"})

        callback(tool, {"sql": sql, "is_final": True}, tool_context, tool_response)

        public_result = tool_context.state[SQL_PUBLIC_RESULT_STATE_KEY]
        query_summary_context = public_result["query_summary_context"]
        self.assertEqual(
            query_summary_context["sql"],
            "SELECT COUNT(*) AS matching_count FROM patients WHERE sex IN ('female', 'male')",
        )
        self.assertEqual(
            query_summary_context["categorical_filters"],
            [
                {
                    "column": "sex",
                    "selected_values": ["female", "male"],
                    "available_values": ["female", "male"],
                    "dataset_missing_or_blank_count": 3,
                    "dataset_missing_or_blank_count_unit": "rows",
                }
            ],
        )

    def test_remember_query_result_keeps_exact_small_missing_blank_count_below_threshold(self) -> None:
        sql = "SELECT COUNT(*) AS matching_count FROM patients WHERE sex IN ('female', 'male')"
        tool_response = execute_sqlite_query(self.db_path, sql)
        callback = build_remember_query_result_callback(self._settings(minimum_aggregate_count=5))
        tool = SimpleNamespace(name="execute_sqlite_read_only")
        tool_context = SimpleNamespace(state={SQL_LAST_USER_TEXT_STATE_KEY: "How many patients have a recorded sex value?"})

        callback(tool, {"sql": sql, "is_final": True}, tool_context, tool_response)

        public_result = tool_context.state[SQL_PUBLIC_RESULT_STATE_KEY]
        self.assertEqual(
            public_result["query_summary_context"]["categorical_filters"],
            [
                {
                    "column": "sex",
                    "selected_values": ["female", "male"],
                    "available_values": ["female", "male"],
                    "dataset_missing_or_blank_count": 3,
                    "dataset_missing_or_blank_count_unit": "rows",
                }
            ],
        )

if __name__ == "__main__":
    unittest.main()
