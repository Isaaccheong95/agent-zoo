from __future__ import annotations

import unittest
import uuid
from types import SimpleNamespace

from agent_zoo.sql_agent.callbacks import (
    SQL_PUBLIC_RESULT_STATE_KEY,
    build_format_final_agent_response_callback,
    build_remember_query_result_callback,
)
from agent_zoo.sql_agent.config import SQLAgentSettings
from agent_zoo.sql_agent.db import execute_sqlite_query

from tests._fixtures import (
    REPO_ROOT,
    create_object_mode_fixture_database,
)


class SQLAgentObjectModeTestCase(unittest.TestCase):
    def setUp(self) -> None:
        temp_root = REPO_ROOT / ".tmp_test_runs"
        temp_root.mkdir(exist_ok=True)
        self.db_path = temp_root / f"{uuid.uuid4().hex}.sqlite"
        create_object_mode_fixture_database(self.db_path)

    def tearDown(self) -> None:
        if self.db_path.exists():
            self.db_path.unlink()

    def _object_settings(self, **overrides) -> SQLAgentSettings:
        settings = SQLAgentSettings(
            db_path=self.db_path,
            model="test-model",
            object_id_column="person_id",
            object_order_column="event_rank",
        )
        for key, value in overrides.items():
            setattr(settings, key, value)
        return settings

    def test_execute_sqlite_query_preserves_row_level_behavior_without_object_id(self) -> None:
        result = execute_sqlite_query(
            self.db_path,
            "SELECT COUNT(*) AS total_rows FROM records",
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["rows"][0]["total_rows"], 5)

    def test_execute_sqlite_query_counts_distinct_objects_when_object_id_is_configured(self) -> None:
        result = execute_sqlite_query(
            self.db_path,
            "SELECT COUNT(*) AS total_rows FROM records",
            object_id_column="person_id",
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["rows"][0]["total_rows"], 3)

    def test_execute_sqlite_query_selects_canonical_rows_using_object_order_column(self) -> None:
        result = execute_sqlite_query(
            self.db_path,
            "SELECT person_id, city, score FROM records ORDER BY person_id",
            object_id_column="person_id",
            object_order_column="event_rank",
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["row_count"], 3)
        self.assertEqual(
            result["rows"],
            [
                {"person_id": "p1", "city": "Tokyo", "score": 20.0},
                {"person_id": "p2", "city": "Paris", "score": 30.0},
                {"person_id": "p3", "city": "Tokyo", "score": 50.0},
            ],
        )

    def test_execute_sqlite_query_uses_first_occurrence_when_object_order_column_is_omitted(self) -> None:
        result = execute_sqlite_query(
            self.db_path,
            "SELECT person_id, city, score FROM records ORDER BY person_id",
            object_id_column="person_id",
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["row_count"], 3)
        self.assertEqual(
            result["rows"],
            [
                {"person_id": "p1", "city": "Singapore", "score": 10.0},
                {"person_id": "p2", "city": "Paris", "score": 30.0},
                {"person_id": "p3", "city": "Paris", "score": 40.0},
            ],
        )

    def test_execute_sqlite_query_returns_clear_error_for_missing_object_id_column(self) -> None:
        result = execute_sqlite_query(
            self.db_path,
            "SELECT COUNT(*) AS total_rows FROM records",
            object_id_column="missing_person_id",
        )

        self.assertEqual(result["status"], "error")
        self.assertIn("object_id_column 'missing_person_id'", result["error"])

    def test_execute_sqlite_query_returns_clear_error_for_unorderable_object_order_column(self) -> None:
        result = execute_sqlite_query(
            self.db_path,
            "SELECT COUNT(*) AS total_rows FROM records",
            object_id_column="person_id",
            object_order_column="payload",
        )

        self.assertEqual(result["status"], "error")
        self.assertIn("object_order_column 'payload'", result["error"])

    def test_grouped_aggregate_object_mode_uses_canonical_rows(self) -> None:
        tool_response = execute_sqlite_query(
            self.db_path,
            "SELECT city, AVG(score) AS average_score FROM records GROUP BY city ORDER BY city",
            object_id_column="person_id",
            object_order_column="event_rank",
        )
        callback = build_remember_query_result_callback(self._object_settings(minimum_aggregate_count=1))
        tool = SimpleNamespace(name="execute_sqlite_read_only")
        tool_context = SimpleNamespace(state={})

        callback(tool, {}, tool_context, tool_response)

        public_result = tool_context.state[SQL_PUBLIC_RESULT_STATE_KEY]
        self.assertEqual(public_result["status"], "success")
        self.assertEqual(public_result["matched_row_count"], 3)
        self.assertEqual(public_result["public_result_kind"], "safe_aggregate")
        self.assertEqual(
            public_result["rows"],
            [
                {"city": "Paris", "matching_count": 1, "average_score": 30.0},
                {"city": "Tokyo", "matching_count": 2, "average_score": 35.0},
            ],
        )

    def test_execute_sqlite_query_supports_top_n_subquery_aggregate_in_object_mode(self) -> None:
        result = execute_sqlite_query(
            self.db_path,
            "SELECT AVG(score) AS average_score FROM (SELECT score FROM records ORDER BY score DESC LIMIT 2)",
            object_id_column="person_id",
            object_order_column="event_rank",
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["row_count"], 1)
        self.assertEqual(result["rows"], [{"average_score": 40.0}])
        self.assertIn("__az_object_canonical", result["sql"])
        self.assertIn("LIMIT 2", result["sql"])

    def test_object_mode_final_render_shows_original_query_not_canonicalized_sql(self) -> None:
        simple_sql = "SELECT COUNT(*) AS matching_count FROM records WHERE city = 'Tokyo'"
        tool_response = execute_sqlite_query(
            self.db_path,
            simple_sql,
            object_id_column="person_id",
            object_order_column="event_rank",
        )

        self.assertEqual(tool_response["status"], "success")
        self.assertIn("__az_object_source", tool_response["sql"])

        callback = build_remember_query_result_callback(self._object_settings(minimum_aggregate_count=1))
        tool = SimpleNamespace(name="execute_sqlite_read_only")
        tool_context = SimpleNamespace(state={})

        callback(
            tool,
            {"sql": simple_sql, "is_final": True},
            tool_context,
            tool_response,
        )

        public_result = tool_context.state[SQL_PUBLIC_RESULT_STATE_KEY]
        self.assertEqual(public_result["display_sql"], simple_sql)
        self.assertIn("__az_object_source", public_result["sql"])

        content = build_format_final_agent_response_callback()(SimpleNamespace(state=tool_context.state))
        self.assertIsNotNone(content)
        self.assertIn(simple_sql, content.parts[0].text)
        self.assertNotIn("__az_object_source", content.parts[0].text)

if __name__ == "__main__":
    unittest.main()
