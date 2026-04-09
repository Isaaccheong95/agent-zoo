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

from google.genai import types


CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = CURRENT_FILE.parent if (CURRENT_FILE.parent / "src").exists() else CURRENT_FILE.parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agent_zoo.sql_agent.callbacks import (
    SQL_ACTIVE_QUERY_TOPIC_STATE_KEY,
    SQL_INTERNAL_QUERY_RESULT_STATE_KEY,
    SQL_INTERNAL_RESULT_REF_STATE_KEY,
    SQL_LAST_USER_TEXT_STATE_KEY,
    SQL_LAST_QUERY_FRAME_STATE_KEY,
    SQL_PENDING_CLARIFICATION_STATE_KEY,
    SQL_PUBLIC_RESULT_STATE_KEY,
    SQL_PUBLIC_RESULT_RENDERED_STATE_KEY,
    build_combined_before_model_callback,
    build_finalize_after_query_before_model_callback,
    build_format_final_agent_response_callback,
    build_normalize_clarification_after_model_callback,
    build_remember_query_result_callback,
)
from agent_zoo.sql_agent.config import SQLAgentSettings, load_settings
from agent_zoo.sql_agent.db import execute_sqlite_query, get_schema_summary, validate_sql_read_only
from agent_zoo.sql_agent.formatting import (
    build_sql_result_view_model,
    render_sql_result_view_model,
)
from agent_zoo.sql_agent.instructions import build_agent_instruction
from agent_zoo.sql_agent.runtime import _print_debug_event
from agent_zoo.sql_agent.tools import build_sql_tools
from agent_zoo.scope_guard import (
    build_llm_clarification_resolver,
    build_llm_result_refinement_resolver,
    build_llm_scope_gate,
)


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


def create_object_mode_fixture_database(db_path: Path) -> None:
    connection = sqlite3.connect(db_path)
    connection.executescript(
        """
        CREATE TABLE records (
            id INTEGER PRIMARY KEY,
            person_id TEXT NOT NULL,
            event_rank INTEGER NOT NULL,
            city TEXT NOT NULL,
            score REAL NOT NULL,
            payload BLOB
        );

        INSERT INTO records (person_id, event_rank, city, score, payload) VALUES
            ('p1', 2, 'Singapore', 10.0, x'01'),
            ('p1', 1, 'Tokyo', 20.0, x'02'),
            ('p2', 1, 'Paris', 30.0, x'03'),
            ('p3', 2, 'Paris', 40.0, x'04'),
            ('p3', 1, 'Tokyo', 50.0, x'05');
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
    sql: str = "SELECT ...",
) -> dict:
    return {
        "status": status,
        "db_path": "fixture.sqlite",
        "sql": sql,
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

    def test_get_schema_summary_includes_categorical_value_guidance(self) -> None:
        summary = get_schema_summary(
            self.db_path,
            include_categorical_value_guidance=True,
        )

        self.assertEqual(summary["status"], "success")
        guidance = {
            (entry["table"], entry["column"]): entry["values"]
            for entry in summary["categorical_value_guidance"]
        }
        self.assertEqual(guidance[("people", "sex")], ["female", "male"])
        self.assertEqual(guidance[("visits", "city")], ["Paris", "Singapore", "Tokyo"])
        self.assertNotIn(("people", "id"), guidance)
        self.assertNotIn(("people", "name"), guidance)
        self.assertIn(
            'people.sex: Stored SQLite values seen in the dataset: "female", "male"',
            summary["categorical_value_guidance_text"],
        )

        people_table = next(table for table in summary["tables"] if table["name"] == "people")
        people_columns = {column["name"]: column for column in people_table["columns"]}
        self.assertEqual(people_columns["sex"]["categorical_values"], ["female", "male"])
        self.assertNotIn("categorical_values", people_columns["name"])

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

    def test_build_agent_instruction_mentions_clarifying_near_matches(self) -> None:
        instruction = build_agent_instruction(
            SQLAgentSettings(
                db_path=self.db_path,
                model="test-model",
            )
        )

        self.assertIn("approximate, colloquial, or partially incorrect dataset terminology", instruction)
        self.assertIn("nearby schema concept", instruction)
        self.assertIn('"response_type":"clarification"', instruction)
        self.assertIn("available category values", instruction)
        self.assertIn("choose one or more options or describe their own rule", instruction)

    def test_build_agent_instruction_includes_categorical_value_guidance_and_exploratory_flag(self) -> None:
        instruction = build_agent_instruction(
            SQLAgentSettings(
                db_path=self.db_path,
                model="test-model",
            )
        )

        self.assertIn("Relevant Categorical Value Guidance", instruction)
        self.assertIn('people.sex: Stored SQLite values seen in the dataset: "female", "male"', instruction)
        self.assertIn("is_final=False", instruction)
        self.assertIn("Do not stop after an exploratory query", instruction)

    def test_inspect_sqlite_schema_tool_reuses_categorical_value_guidance(self) -> None:
        inspect_tool = next(
            tool
            for tool in build_sql_tools(
                SQLAgentSettings(
                    db_path=self.db_path,
                    model="test-model",
                )
            )
            if tool.__name__ == "inspect_sqlite_schema"
        )

        summary = inspect_tool()
        guidance = {
            (entry["table"], entry["column"]): entry["values"]
            for entry in summary["categorical_value_guidance"]
        }

        self.assertEqual(guidance[("people", "sex")], ["female", "male"])
        self.assertEqual(guidance[("visits", "city")], ["Paris", "Singapore", "Tokyo"])

    def test_scope_gate_prompt_keeps_schema_adjacent_requests_in_scope(self) -> None:
        captured: dict[str, str] = {}

        def completion(**kwargs):
            captured["system_prompt"] = kwargs["messages"][0]["content"]
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="IN_SCOPE"))]
            )

        classifier = build_llm_scope_gate(
            "test-model",
            "titanic_passengers(pclass INTEGER, fare REAL)",
        )

        with patch.dict(sys.modules, {"litellm": SimpleNamespace(completion=completion)}):
            allow, refusal = classifier("how many ppl in each fare class")

        self.assertTrue(allow)
        self.assertIsNone(refusal)
        self.assertIn("schema-adjacent wording", captured["system_prompt"])
        self.assertIn("near-match", captured["system_prompt"])

    def test_clarification_resolver_includes_current_topic_context(self) -> None:
        captured: dict[str, str] = {}

        def completion(**kwargs):
            captured["system_prompt"] = kwargs["messages"][0]["content"]
            captured["user_prompt"] = kwargs["messages"][1]["content"]
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content='{"resolution_type":"selected_options","selected_options":["Occasionally","Regularly"],"custom_rule":""}'
                        )
                    )
                ]
            )

        classifier = build_llm_clarification_resolver("test-model")

        with patch.dict(sys.modules, {"litellm": SimpleNamespace(completion=completion)}):
            resolution = classifier(
                "how many females are alcoholics",
                "Which category should I use for 'alcoholics'?",
                ["Never", "Occasionally", "Regularly", "Unknown"],
                "occasionally and regularly",
            )

        self.assertEqual(
            resolution,
            {
                "resolution_type": "selected_options",
                "selected_options": ["Occasionally", "Regularly"],
                "custom_rule": "",
            },
        )
        self.assertIn('"resolution_type"', captured["system_prompt"])
        self.assertIn("selected_options", captured["system_prompt"])
        self.assertIn("topic_change", captured["system_prompt"])
        self.assertIn("Current dataset question/topic", captured["user_prompt"])
        self.assertIn("how many females are alcoholics", captured["user_prompt"])
        self.assertIn("Pending clarification question", captured["user_prompt"])
        self.assertIn("Available clarification options", captured["user_prompt"])
        self.assertIn("Latest user reply", captured["user_prompt"])

    def test_clarification_resolver_fails_open_on_invalid_output(self) -> None:
        def completion(**kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="not json"))]
            )

        classifier = build_llm_clarification_resolver("test-model")

        with patch.dict(sys.modules, {"litellm": SimpleNamespace(completion=completion)}):
            resolution = classifier(
                "how many females are alcoholics",
                "Which category should I use for 'alcoholics'?",
                ["Never", "Occasionally", "Regularly", "Unknown"],
                "tell me a joke",
            )

        self.assertEqual(
            resolution,
            {
                "resolution_type": "custom_rule",
                "selected_options": [],
                "custom_rule": "tell me a joke",
            },
        )

    def test_result_refinement_resolver_includes_last_query_frame_context(self) -> None:
        captured: dict[str, str] = {}

        def completion(**kwargs):
            captured["system_prompt"] = kwargs["messages"][0]["content"]
            captured["user_prompt"] = kwargs["messages"][1]["content"]
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content='{"resolution_type":"needs_clarification","target_column":"sococc","selected_values":[],"refinement_request":""}'
                        )
                    )
                ]
            )

        resolver = build_llm_result_refinement_resolver("test-model")

        with patch.dict(sys.modules, {"litellm": SimpleNamespace(completion=completion)}):
            resolution = resolver(
                {
                    "question": "how many males dont work",
                    "sql": "SELECT COUNT(*) AS matching_count FROM filtered_dataset WHERE gender = 'Male' AND sococc = 'Unemployed'",
                    "categorical_filters": [
                        {"column": "gender", "selected_values": ["Male"], "available_values": ["Male", "Female"]},
                        {
                            "column": "sococc",
                            "selected_values": ["Unemployed"],
                            "available_values": ["Employed", "Retired", "Student", "Unemployed", "Unknown"],
                        },
                    ],
                },
                "i want to add other categories to dont work",
            )

        self.assertEqual(
            resolution,
            {
                "resolution_type": "needs_clarification",
                "target_column": "sococc",
                "selected_values": [],
                "refinement_request": "",
            },
        )
        self.assertIn("refine_query|needs_clarification|topic_change", captured["system_prompt"])
        self.assertIn("Previous dataset question", captured["user_prompt"])
        self.assertIn("how many males dont work", captured["user_prompt"])
        self.assertIn("Previous SQL query", captured["user_prompt"])
        self.assertIn("Categorical filters from the previous query", captured["user_prompt"])
        self.assertIn("Latest user reply", captured["user_prompt"])

    def test_result_refinement_resolver_fails_closed_on_invalid_output(self) -> None:
        def completion(**kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="not json"))]
            )

        resolver = build_llm_result_refinement_resolver("test-model")

        with patch.dict(sys.modules, {"litellm": SimpleNamespace(completion=completion)}):
            resolution = resolver(
                {
                    "question": "how many males dont work",
                    "sql": "SELECT COUNT(*) AS matching_count FROM filtered_dataset WHERE gender = 'Male' AND sococc = 'Unemployed'",
                    "categorical_filters": [
                        {
                            "column": "sococc",
                            "selected_values": ["Unemployed"],
                            "available_values": ["Employed", "Retired", "Student", "Unemployed", "Unknown"],
                        }
                    ],
                },
                "i want to add other categories to dont work",
            )

        self.assertEqual(
            resolution,
            {
                "resolution_type": "topic_change",
                "target_column": "",
                "selected_values": [],
                "refinement_request": "",
            },
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

    def _invoke_after_tool_with_state(
        self,
        settings: SQLAgentSettings,
        tool_response: dict,
        state,
        args: dict | None = None,
    ):
        callback = build_remember_query_result_callback(settings)
        tool = SimpleNamespace(name="execute_sqlite_read_only")
        tool_context = SimpleNamespace(state=state)
        callback(tool, args or {}, tool_context, tool_response)
        return tool_context.state

    def _invoke_after_tool_with_args(
        self,
        settings: SQLAgentSettings,
        tool_response: dict,
        args: dict | None = None,
    ) -> tuple[dict, dict | None]:
        callback = build_remember_query_result_callback(settings)
        tool = SimpleNamespace(name="execute_sqlite_read_only")
        tool_context = SimpleNamespace(state={})
        returned_result = callback(tool, args or {}, tool_context, tool_response)
        return tool_context.state, returned_result

    def test_load_settings_defaults_and_env_overrides(self) -> None:
        settings = load_settings(
            {
                "db_path": str(REPO_ROOT / "dataset" / "titantic" / "titanic.sqlite"),
                "model": "test-model",
            }
        )

        self.assertTrue(settings.count_aggregates_only)
        self.assertEqual(settings.minimum_aggregate_count, 5)
        self.assertFalse(settings.capture_internal_rows)
        self.assertTrue(settings.include_categorical_value_guidance)
        self.assertEqual(settings.max_categorical_values, 12)

        with patch.dict(
            os.environ,
            {
                "SQL_AGENT_COUNT_AGGREGATES_ONLY": "false",
                "SQL_AGENT_MINIMUM_AGGREGATE_COUNT": "7",
                "SQL_AGENT_CAPTURE_INTERNAL_ROWS": "true",
                "SQL_AGENT_INCLUDE_CATEGORICAL_VALUE_GUIDANCE": "false",
                "SQL_AGENT_MAX_CATEGORICAL_VALUES": "7",
                "OPENAI_API_BASE": "http://127.0.0.1:9000/v1",
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
        self.assertFalse(overridden.include_categorical_value_guidance)
        self.assertEqual(overridden.max_categorical_values, 7)
        self.assertEqual(overridden.openai_api_base, "http://127.0.0.1:9000/v1")

    def test_load_settings_supports_object_level_configuration(self) -> None:
        settings = load_settings(
            {
                "db_path": str(REPO_ROOT / "dataset" / "titantic" / "titanic.sqlite"),
                "model": "test-model",
            }
        )

        self.assertIsNone(settings.object_id_column)
        self.assertIsNone(settings.object_order_column)

        with patch.dict(
            os.environ,
            {
                "SQL_AGENT_OBJECT_ID_COLUMN": "person_id",
                "SQL_AGENT_OBJECT_ORDER_COLUMN": "event_rank",
            },
            clear=False,
        ):
            overridden = load_settings(
                {
                    "db_path": str(REPO_ROOT / "dataset" / "titantic" / "titanic.sqlite"),
                    "model": "test-model",
                }
            )

        self.assertEqual(overridden.object_id_column, "person_id")
        self.assertEqual(overridden.object_order_column, "event_rank")
        with self.assertRaises(ValueError):
            load_settings(
                {
                    "db_path": str(REPO_ROOT / "dataset" / "titantic" / "titanic.sqlite"),
                    "model": "test-model",
                    "object_order_column": "event_rank",
                }
            )

    def test_build_root_agent_applies_openai_api_base_before_model_construction(self) -> None:
        from agent_zoo.sql_agent.agent import build_root_agent

        settings = SQLAgentSettings(
            db_path=REPO_ROOT / "dataset" / "titantic" / "titanic.sqlite",
            model="openai/test-model",
            openai_api_base="http://127.0.0.1:9001/v1",
        )

        def construct_model(*args, **kwargs):
            self.assertEqual(os.environ["OPENAI_API_BASE"], settings.openai_api_base)
            self.assertEqual(os.environ["OPENAI_API_KEY"], "local-openai-compatible-key")
            return SimpleNamespace()

        with patch.dict(os.environ, {}, clear=True):
            with patch("agent_zoo.sql_agent.agent.LiteLlm", side_effect=construct_model) as mocked_model:
                with patch("agent_zoo.sql_agent.agent.LlmAgent", return_value=SimpleNamespace()) as mocked_agent:
                    with patch("agent_zoo.sql_agent.agent.build_agent_instruction", return_value="instructions"):
                        with patch("agent_zoo.sql_agent.agent.build_sql_tools", return_value=[]):
                            build_root_agent(settings)

        mocked_model.assert_called_once_with(model="openai/test-model")
        self.assertIsNotNone(mocked_agent.call_args.kwargs["after_model_callback"])

    def test_detail_rows_become_public_matching_count_by_default(self) -> None:
        state = self._invoke_after_tool(
            self._settings(minimum_aggregate_count=1),
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
        self.assertEqual(public_result["public_result_kind"], "detail_count_fallback")
        self.assertNotIn(SQL_INTERNAL_RESULT_REF_STATE_KEY, state)
        self.assertNotIn(SQL_INTERNAL_QUERY_RESULT_STATE_KEY, state)

    def test_privacy_mode_returns_public_result_to_model(self) -> None:
        callback = build_remember_query_result_callback(self._settings(minimum_aggregate_count=3))
        tool = SimpleNamespace(name="execute_sqlite_read_only")
        tool_context = SimpleNamespace(state={})

        returned_result = callback(
            tool,
            {},
            tool_context,
            make_query_result(
                [{"matching_count": 2}],
                columns=["matching_count"],
            ),
        )

        self.assertEqual(returned_result, tool_context.state[SQL_PUBLIC_RESULT_STATE_KEY])
        self.assertEqual(returned_result["status"], "error")
        self.assertTrue(returned_result["privacy_blocked"])

    def test_exploratory_query_skips_public_state_and_finalization(self) -> None:
        state, returned_result = self._invoke_after_tool_with_args(
            self._settings(minimum_aggregate_count=3),
            make_query_result(
                [
                    {"sex": "female"},
                    {"sex": "male"},
                ],
                columns=["sex"],
                row_count=2,
                sql="SELECT DISTINCT sex FROM people",
            ),
            args={"is_final": False},
        )

        self.assertIsNone(returned_result)
        self.assertNotIn(SQL_PUBLIC_RESULT_STATE_KEY, state)

        finalize = build_finalize_after_query_before_model_callback()
        final_response = finalize(
            callback_context=SimpleNamespace(state=state),
            llm_request=SimpleNamespace(),
        )

        self.assertIsNone(final_response)

    def test_remember_query_result_stores_last_query_frame_for_final_query(self) -> None:
        with patch(
            "agent_zoo.sql_agent.callbacks.get_schema_summary",
            return_value={
                "status": "success",
                "categorical_value_guidance": [
                    {"column": "gender", "values": ["Male", "Female"]},
                    {"column": "sococc", "values": ["Employed", "Retired", "Student", "Unemployed", "Unknown"]},
                ],
            },
        ):
            callback = build_remember_query_result_callback(self._settings(minimum_aggregate_count=1))

        tool = SimpleNamespace(name="execute_sqlite_read_only")
        tool_context = SimpleNamespace(
            state={
                SQL_ACTIVE_QUERY_TOPIC_STATE_KEY: "how many males dont work",
            }
        )

        callback(
            tool,
            {
                "sql": "SELECT COUNT(*) AS matching_count FROM filtered_dataset WHERE gender = 'Male' AND sococc = 'Unemployed'",
                "is_final": True,
            },
            tool_context,
            make_query_result(
                [{"matching_count": 33}],
                columns=["matching_count"],
                sql="SELECT COUNT(*) AS matching_count FROM filtered_dataset WHERE gender = 'Male' AND sococc = 'Unemployed'",
            ),
        )

        query_frame = tool_context.state[SQL_LAST_QUERY_FRAME_STATE_KEY]
        self.assertEqual(query_frame["question"], "how many males dont work")
        self.assertEqual(
            query_frame["sql"],
            "SELECT COUNT(*) AS matching_count FROM filtered_dataset WHERE gender = 'Male' AND sococc = 'Unemployed'",
        )
        self.assertEqual(
            query_frame["categorical_filters"],
            [
                {"column": "gender", "selected_values": ["Male"], "available_values": ["Male", "Female"]},
                {
                    "column": "sococc",
                    "selected_values": ["Unemployed"],
                    "available_values": ["Employed", "Retired", "Student", "Unemployed", "Unknown"],
                },
            ],
        )
        public_result = tool_context.state[SQL_PUBLIC_RESULT_STATE_KEY]
        self.assertEqual(public_result["query_summary_context"], query_frame)

    def test_remember_query_result_includes_non_categorical_comparison_filters(self) -> None:
        with patch(
            "agent_zoo.sql_agent.callbacks.get_schema_summary",
            return_value={
                "status": "success",
                "categorical_value_guidance": [
                    {"column": "gender", "values": ["Male", "Female"]},
                ],
            },
        ):
            callback = build_remember_query_result_callback(self._settings(minimum_aggregate_count=1))

        tool = SimpleNamespace(name="execute_sqlite_read_only")
        tool_context = SimpleNamespace(
            state={
                SQL_ACTIVE_QUERY_TOPIC_STATE_KEY: "how many females are older than 46",
            }
        )

        callback(
            tool,
            {
                "sql": "SELECT COUNT(*) AS matching_count FROM filtered_dataset WHERE gender = 'Female' AND age > 46",
                "is_final": True,
            },
            tool_context,
            make_query_result(
                [{"matching_count": 12}],
                columns=["matching_count"],
                sql="SELECT COUNT(*) AS matching_count FROM filtered_dataset WHERE gender = 'Female' AND age > 46",
            ),
        )

        query_frame = tool_context.state[SQL_LAST_QUERY_FRAME_STATE_KEY]
        self.assertEqual(
            query_frame["comparison_filters"],
            [{"column": "age", "operator": ">", "value": "46"}],
        )

    def test_remember_query_result_includes_cast_comparison_filters_in_final_render(self) -> None:
        with patch(
            "agent_zoo.sql_agent.callbacks.get_schema_summary",
            return_value={
                "status": "success",
                "categorical_value_guidance": [
                    {"column": "gender", "values": ["Male", "Female"]},
                    {"column": "socsmk", "values": ["No", "Unknown", "Yes"]},
                    {"column": "Centre", "values": ["HospitalA", "HospitalB"]},
                ],
            },
        ):
            callback = build_remember_query_result_callback(self._settings(minimum_aggregate_count=1))

        tool = SimpleNamespace(name="execute_sqlite_read_only")
        tool_context = SimpleNamespace(
            state={
                SQL_ACTIVE_QUERY_TOPIC_STATE_KEY: "how many females are smokers from hosp A and under 45",
            }
        )
        simple_sql = (
            "SELECT COUNT(*) AS matching_count FROM filtered_dataset "
            "WHERE gender = 'Female' AND socsmk = 'Yes' AND Centre = 'HospitalA' "
            "AND CAST(age AS INTEGER) < 45"
        )

        callback(
            tool,
            {
                "sql": simple_sql,
                "is_final": True,
            },
            tool_context,
            make_query_result(
                [{"matching_count": 11}],
                columns=["matching_count"],
                sql=simple_sql,
            ),
        )

        query_frame = tool_context.state[SQL_LAST_QUERY_FRAME_STATE_KEY]
        self.assertEqual(
            query_frame["comparison_filters"],
            [{"column": "age", "operator": "<", "value": "45"}],
        )

        content = build_format_final_agent_response_callback()(SimpleNamespace(state=tool_context.state))
        self.assertIsNotNone(content)
        self.assertIn("- age < 45", content.parts[0].text)

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

    def test_callback_handles_adk_state_objects_without_pop(self) -> None:
        class FakeState:
            def __init__(self, initial: dict | None = None) -> None:
                self._value = dict(initial or {})

            def __getitem__(self, key):
                return self._value[key]

            def __setitem__(self, key, value) -> None:
                self._value[key] = value

            def __contains__(self, key) -> bool:
                return key in self._value

            def get(self, key, default=None):
                return self._value.get(key, default)

        state = self._invoke_after_tool_with_state(
            self._settings(minimum_aggregate_count=1),
            make_query_result(
                [{"matching_count": 4}],
                columns=["matching_count"],
            ),
            FakeState(
                {
                    SQL_PUBLIC_RESULT_STATE_KEY: {"status": "success"},
                    SQL_INTERNAL_RESULT_REF_STATE_KEY: "ref",
                    SQL_INTERNAL_QUERY_RESULT_STATE_KEY: {"rows": [{"name": "Alice"}]},
                }
            ),
        )

        public_result = state[SQL_PUBLIC_RESULT_STATE_KEY]
        self.assertEqual(public_result["status"], "success")
        self.assertEqual(public_result["rows"], [{"matching_count": 4}])
        self.assertIsNone(state.get(SQL_INTERNAL_RESULT_REF_STATE_KEY))
        self.assertIsNone(state.get(SQL_INTERNAL_QUERY_RESULT_STATE_KEY))

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
        self.assertIn("minimum threshold", public_result["error"])

    def test_scalar_count_with_non_count_alias_uses_actual_value(self) -> None:
        state = self._invoke_after_tool(
            self._settings(minimum_aggregate_count=3),
            make_query_result(
                [{"total_passengers": 4}],
                columns=["total_passengers"],
                sql="SELECT COUNT(*) AS total_passengers FROM people",
            ),
        )

        public_result = state[SQL_PUBLIC_RESULT_STATE_KEY]
        self.assertEqual(public_result["status"], "success")
        self.assertEqual(public_result["rows"], [{"matching_count": 4}])
        self.assertEqual(public_result["matched_row_count"], 4)
        self.assertEqual(public_result["public_result_kind"], "count_aggregate")

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
                sql="SELECT sex, COUNT(*) AS matching_count FROM people GROUP BY sex",
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
                sql="SELECT sex, COUNT(*) AS matching_count FROM people GROUP BY sex",
            ),
        )

        public_result = state[SQL_PUBLIC_RESULT_STATE_KEY]
        self.assertEqual(public_result["status"], "success")
        self.assertEqual(public_result["rows"][0]["sex"], "female")
        self.assertEqual(public_result["matched_row_count"], 9)
        self.assertEqual(public_result["public_result_kind"], "count_aggregate")

    def test_scalar_average_with_matching_count_is_kept_publicly(self) -> None:
        state = self._invoke_after_tool(
            self._settings(minimum_aggregate_count=3),
            make_query_result(
                [{"matching_count": 4, "average_age": 36.25}],
                columns=["matching_count", "average_age"],
                row_count=1,
            ),
        )

        public_result = state[SQL_PUBLIC_RESULT_STATE_KEY]
        self.assertEqual(public_result["status"], "success")
        self.assertEqual(public_result["rows"][0]["average_age"], 36.25)
        self.assertEqual(public_result["matched_row_count"], 4)
        self.assertEqual(public_result["public_result_kind"], "safe_aggregate")

    def test_scalar_average_below_threshold_is_blocked(self) -> None:
        state = self._invoke_after_tool(
            self._settings(minimum_aggregate_count=3),
            make_query_result(
                [{"matching_count": 2, "average_age": 36.25}],
                columns=["matching_count", "average_age"],
                row_count=1,
            ),
        )

        public_result = state[SQL_PUBLIC_RESULT_STATE_KEY]
        self.assertEqual(public_result["status"], "error")
        self.assertTrue(public_result["privacy_blocked"])
        self.assertIn("minimum threshold", public_result["error"])

    def test_grouped_average_above_threshold_is_kept_publicly(self) -> None:
        state = self._invoke_after_tool(
            self._settings(minimum_aggregate_count=3),
            make_query_result(
                [
                    {"sex": "female", "matching_count": 5, "average_age": 31.4},
                    {"sex": "male", "matching_count": 4, "average_age": 44.0},
                ],
                columns=["sex", "matching_count", "average_age"],
                row_count=2,
                sql="SELECT sex, COUNT(*) AS matching_count, AVG(age) AS average_age FROM people GROUP BY sex",
            ),
        )

        public_result = state[SQL_PUBLIC_RESULT_STATE_KEY]
        self.assertEqual(public_result["status"], "success")
        self.assertEqual(public_result["matched_row_count"], 9)
        self.assertEqual(public_result["public_result_kind"], "safe_aggregate")
        self.assertEqual(public_result["rows"][0]["average_age"], 31.4)

    def test_grouped_average_without_matching_count_derives_group_counts(self) -> None:
        count_result = make_query_result(
            [
                {"sex": "female", "matching_count": 5},
                {"sex": "male", "matching_count": 4},
            ],
            columns=["sex", "matching_count"],
            row_count=2,
            sql="SELECT sex, COUNT(*) AS matching_count FROM people GROUP BY sex",
        )

        with patch("agent_zoo.sql_agent.callbacks.execute_sqlite_query", return_value=count_result):
            state = self._invoke_after_tool(
                self._settings(minimum_aggregate_count=3),
                make_query_result(
                    [
                        {"sex": "female", "average_age": 31.4},
                        {"sex": "male", "average_age": 44.0},
                    ],
                    columns=["sex", "average_age"],
                    row_count=2,
                    sql="SELECT sex, AVG(age) AS average_age FROM people GROUP BY sex",
                ),
            )

        public_result = state[SQL_PUBLIC_RESULT_STATE_KEY]
        self.assertEqual(public_result["status"], "success")
        self.assertEqual(public_result["rows"][0]["matching_count"], 5)
        self.assertEqual(public_result["matched_row_count"], 9)
        self.assertEqual(public_result["public_result_kind"], "safe_aggregate")

    def test_grouped_average_below_threshold_is_blocked(self) -> None:
        state = self._invoke_after_tool(
            self._settings(minimum_aggregate_count=3),
            make_query_result(
                [
                    {"sex": "female", "matching_count": 5, "average_age": 31.4},
                    {"sex": "male", "matching_count": 1, "average_age": 44.0},
                ],
                columns=["sex", "matching_count", "average_age"],
                row_count=2,
                sql="SELECT sex, COUNT(*) AS matching_count, AVG(age) AS average_age FROM people GROUP BY sex",
            ),
        )

        public_result = state[SQL_PUBLIC_RESULT_STATE_KEY]
        self.assertEqual(public_result["status"], "error")
        self.assertTrue(public_result["privacy_blocked"])
        self.assertIn("group count is below the minimum threshold", public_result["error"])

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
        settings = self._settings(minimum_aggregate_count=1)
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
        callback = build_format_final_agent_response_callback()
        content = callback(SimpleNamespace(state=tool_state))

        self.assertIsNotNone(content)
        response_text = content.parts[0].text
        self.assertIn("What I matched:", response_text)
        self.assertIn("- Matched rows in the current filtered dataset and returned only the safe count.", response_text)
        self.assertIn("```", response_text)
        self.assertIn("4", response_text)
        self.assertNotIn("Alice", response_text)
        self.assertNotIn('"name"', response_text)

    def test_formatted_final_response_includes_matched_categories(self) -> None:
        callback = build_format_final_agent_response_callback()
        content = callback(
            SimpleNamespace(
                state={
                    SQL_PUBLIC_RESULT_STATE_KEY: {
                        "status": "success",
                        "db_path": "fixture.sqlite",
                        "sql": "SELECT COUNT(*) AS matching_count FROM filtered_dataset WHERE gender = 'Male' AND sococc IN ('Retired', 'Student', 'Unemployed')",
                        "columns": ["matching_count"],
                        "rows": [{"matching_count": 33}],
                        "row_count": 1,
                        "preview_row_count": 1,
                        "truncated": False,
                        "error": None,
                        "matched_row_count": 33,
                        "public_result_kind": "count_aggregate",
                        "query_summary_context": {
                            "question": "how many males are not working",
                            "categorical_filters": [
                                {"column": "gender", "selected_values": ["Male"], "available_values": ["Male", "Female"]},
                                {
                                    "column": "sococc",
                                    "selected_values": ["Retired", "Student", "Unemployed"],
                                    "available_values": ["Employed", "Retired", "Student", "Unemployed", "Unknown"],
                                },
                            ],
                        },
                    }
                }
            )
        )

        self.assertIsNotNone(content)
        response_text = content.parts[0].text
        self.assertIn("What I matched:", response_text)
        self.assertIn("- gender = Male", response_text)
        self.assertIn("- sococc in Retired, Student, Unemployed", response_text)
        self.assertNotIn("- Counted matching rows in the current filtered dataset.", response_text)
        self.assertLess(response_text.index("What I matched:"), response_text.index("Result:"))

    def test_formatted_final_response_includes_comparison_filters(self) -> None:
        callback = build_format_final_agent_response_callback()
        content = callback(
            SimpleNamespace(
                state={
                    SQL_PUBLIC_RESULT_STATE_KEY: {
                        "status": "success",
                        "db_path": "fixture.sqlite",
                        "sql": "SELECT COUNT(*) AS matching_count FROM filtered_dataset WHERE gender = 'Female' AND age > 46",
                        "columns": ["matching_count"],
                        "rows": [{"matching_count": 12}],
                        "row_count": 1,
                        "preview_row_count": 1,
                        "truncated": False,
                        "error": None,
                        "matched_row_count": 12,
                        "public_result_kind": "count_aggregate",
                        "query_summary_context": {
                            "categorical_filters": [
                                {"column": "gender", "selected_values": ["Female"], "available_values": ["Female", "Male"]},
                            ],
                            "comparison_filters": [
                                {"column": "age", "operator": ">", "value": "46"},
                            ],
                        },
                    }
                }
            )
        )

        self.assertIsNotNone(content)
        response_text = content.parts[0].text
        self.assertIn("What I matched:", response_text)
        self.assertIn("- gender = Female", response_text)
        self.assertIn("- age > 46", response_text)
        self.assertNotIn("- Counted matching rows in the current filtered dataset.", response_text)

    def test_before_model_callback_short_circuits_when_public_result_exists(self) -> None:
        settings = self._settings()
        callback = build_finalize_after_query_before_model_callback()
        state = {
            SQL_PUBLIC_RESULT_STATE_KEY: {
                "status": "success",
                "db_path": "fixture.sqlite",
                "sql": "SELECT COUNT(*) AS matching_count FROM people",
                "columns": ["matching_count"],
                "rows": [{"matching_count": 4}],
                "row_count": 1,
                "preview_row_count": 1,
                "truncated": False,
                "error": None,
                "matched_row_count": 4,
                "public_result_kind": "count_aggregate",
                "query_summary_context": {
                    "categorical_filters": [
                        {"column": "gender", "selected_values": ["Female"], "available_values": ["Female", "Male"]},
                    ]
                },
            }
        }
        result = callback(
            callback_context=SimpleNamespace(state=state),
            llm_request=SimpleNamespace(),
        )

        self.assertIsNotNone(result)
        response_text = result.content.parts[0].text
        self.assertIn("SELECT COUNT(*) AS matching_count FROM people", response_text)
        self.assertIn("What I matched:", response_text)
        self.assertIn("- gender = Female", response_text)
        self.assertNotIn("- Counted matching rows in the current filtered dataset.", response_text)
        self.assertIn("```", response_text)
        self.assertIn("4", response_text)
        self.assertLess(response_text.index("What I matched:"), response_text.index("Result:"))
        self.assertTrue(state[SQL_PUBLIC_RESULT_RENDERED_STATE_KEY])

    def test_after_agent_callback_is_fallback_once_before_model_has_rendered(self) -> None:
        settings = self._settings()
        finalize = build_finalize_after_query_before_model_callback()
        after_agent = build_format_final_agent_response_callback()
        state = {
            SQL_PUBLIC_RESULT_STATE_KEY: {
                "status": "success",
                "db_path": "fixture.sqlite",
                "sql": "SELECT COUNT(*) AS matching_count FROM people",
                "columns": ["matching_count"],
                "rows": [{"matching_count": 4}],
                "row_count": 1,
                "preview_row_count": 1,
                "truncated": False,
                "error": None,
                "matched_row_count": 4,
                "public_result_kind": "count_aggregate",
                "query_summary_context": {
                    "categorical_filters": [
                        {"column": "gender", "selected_values": ["Female"], "available_values": ["Female", "Male"]},
                    ]
                },
            }
        }

        result = finalize(
            callback_context=SimpleNamespace(state=state),
            llm_request=SimpleNamespace(),
        )

        self.assertIsNotNone(result)
        self.assertTrue(state[SQL_PUBLIC_RESULT_RENDERED_STATE_KEY])
        self.assertIsNone(after_agent(SimpleNamespace(state=state)))

    def test_build_sql_result_view_model_keeps_critical_fields(self) -> None:
        tool_result = {
            "status": "success",
            "sql": "SELECT COUNT(*) AS matching_count FROM filtered_dataset WHERE gender = 'Female' AND age > 46",
            "columns": ["matching_count"],
            "rows": [{"matching_count": 12}],
            "row_count": 1,
            "preview_row_count": 1,
            "truncated": False,
            "error": None,
            "matched_row_count": 12,
            "public_result_kind": "detail_count_fallback",
            "query_summary_context": {
                "categorical_filters": [
                    {
                        "column": "gender",
                        "selected_values": ["Female"],
                        "available_values": ["Female", "Male"],
                    },
                ],
                "comparison_filters": [
                    {"column": "age", "operator": ">", "value": "46"},
                ],
            },
        }

        model = build_sql_result_view_model(tool_result)

        self.assertEqual(model.status, "success")
        self.assertEqual(model.sql, tool_result["sql"])
        self.assertEqual(model.public_result_kind, "detail_count_fallback")
        self.assertEqual(model.matched_row_count, 12)
        self.assertEqual(model.query_summary_context, tool_result["query_summary_context"])
        self.assertIn("privacy guardrails", model.note or "")

    def test_build_sql_result_view_model_prefers_display_sql_when_present(self) -> None:
        tool_result = {
            "status": "success",
            "sql": (
                "WITH __az_object_source AS (SELECT * FROM filtered_dataset WHERE gender = 'Female') "
                "SELECT COUNT(*) AS matching_count FROM __az_object_canonical AS filtered_dataset"
            ),
            "display_sql": "SELECT COUNT(*) AS matching_count FROM filtered_dataset WHERE gender = 'Female'",
            "columns": ["matching_count"],
            "rows": [{"matching_count": 12}],
            "row_count": 1,
            "preview_row_count": 1,
            "truncated": False,
            "error": None,
            "matched_row_count": 12,
            "public_result_kind": "count_aggregate",
            "query_summary_context": {
                "categorical_filters": [
                    {
                        "column": "gender",
                        "selected_values": ["Female"],
                        "available_values": ["Female", "Male"],
                    },
                ],
            },
        }

        model = build_sql_result_view_model(tool_result)

        self.assertEqual(model.sql, tool_result["display_sql"])

    def test_render_sql_result_view_model_matches_existing_formatter_output(self) -> None:
        tool_result = {
            "status": "success",
            "sql": "SELECT COUNT(*) AS matching_count FROM filtered_dataset WHERE gender = 'Female'",
            "columns": ["matching_count"],
            "rows": [{"matching_count": 12}],
            "row_count": 1,
            "preview_row_count": 1,
            "truncated": False,
            "error": None,
            "matched_row_count": 12,
            "public_result_kind": "count_aggregate",
            "query_summary_context": {
                "categorical_filters": [
                    {
                        "column": "gender",
                        "selected_values": ["Female"],
                        "available_values": ["Female", "Male"],
                    },
                ],
            },
        }

        model = build_sql_result_view_model(tool_result)
        rendered = render_sql_result_view_model(model)

        callback = build_format_final_agent_response_callback()
        content = callback(SimpleNamespace(state={SQL_PUBLIC_RESULT_STATE_KEY: tool_result}))
        self.assertIsNotNone(content)
        self.assertEqual(rendered, content.parts[0].text)

    def test_after_model_callback_formats_structured_clarification_options(self) -> None:
        callback = build_normalize_clarification_after_model_callback(self._settings())
        state: dict[str, object] = {
            SQL_LAST_USER_TEXT_STATE_KEY: "how many females are alcoholics",
        }

        result = callback(
            callback_context=SimpleNamespace(state=state),
            llm_response=SimpleNamespace(
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part(
                            text=(
                                '{"response_type":"clarification","user_message":'
                                '"Which occupation status should I count as not working?",'
                                '"options":["Employed","Retired","Unemployed","Unknown","Student"]}'
                            )
                        )
                    ],
                )
            ),
        )

        self.assertIsNotNone(result)
        response_text = result.content.parts[0].text
        self.assertIn("Which occupation status should I count as not working?", response_text)
        self.assertIn("Choose one or more options, or describe your own rule.", response_text)
        self.assertIn("You can reply with option numbers like 2 or 2 and 3.", response_text)
        self.assertIn("1. Employed", response_text)
        self.assertIn("5. Student", response_text)
        self.assertNotIn("Available categories:", response_text)
        self.assertEqual(
            state[SQL_PENDING_CLARIFICATION_STATE_KEY]["options"],
            ["Employed", "Retired", "Unemployed", "Unknown", "Student"],
        )
        self.assertEqual(
            state[SQL_PENDING_CLARIFICATION_STATE_KEY]["topic_context"],
            "how many females are alcoholics",
        )

    def test_after_model_callback_prunes_topic_context_echo_from_options(self) -> None:
        callback = build_normalize_clarification_after_model_callback(self._settings())
        state: dict[str, object] = {
            SQL_LAST_USER_TEXT_STATE_KEY: "how many females are alcoholics",
        }

        result = callback(
            callback_context=SimpleNamespace(state=state),
            llm_response=SimpleNamespace(
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part(
                            text=(
                                '{"response_type":"clarification","user_message":'
                                '"Which category should I use for \'alcoholics\'?",'
                                '"options":["how many females are alcoholics","Never",'
                                '"Occasionally","Regularly","Unknown"]}'
                            )
                        )
                    ],
                )
            ),
        )

        self.assertIsNotNone(result)
        response_text = result.content.parts[0].text
        self.assertIn("Which category should I use for 'alcoholics'?", response_text)
        self.assertIn("Choose one or more options, or describe your own rule.", response_text)
        self.assertIn("1. Never", response_text)
        self.assertIn("2. Occasionally", response_text)
        self.assertIn("3. Regularly", response_text)
        self.assertIn("4. Unknown", response_text)
        self.assertNotRegex(response_text, r"\d+\.\s+how many females are alcoholics\b")
        self.assertEqual(
            state[SQL_PENDING_CLARIFICATION_STATE_KEY]["options"],
            ["Never", "Occasionally", "Regularly", "Unknown"],
        )

    def test_after_model_callback_strips_reasoning_from_clarification_path(self) -> None:
        callback = build_normalize_clarification_after_model_callback(self._settings())
        raw_response = """The user is asking about patients who are \"not working\". Looking at the sococc field (occupation status), I need to identify which categories represent people who are not working.

From the schema and guidance:

sococc: Occupation status
Stored SQLite values seen: \"Employed\", \"Retired\", \"Unemployed\", \"Unknown\", \"Student\"
The question \"not working\" is ambiguous. It could mean:

Unemployed (no job)
Retired (no longer working)
Student (not working for income)
Unknown - uncertain status

I need clarification on your question. Could you please specify which category or combination of categories you'd like me to count?"""

        result = callback(
            callback_context=SimpleNamespace(state={}),
            llm_response=SimpleNamespace(
                content=types.Content(
                    role="model",
                    parts=[types.Part(text=raw_response)],
                )
            ),
        )

        self.assertIsNotNone(result)
        response_text = result.content.parts[0].text
        self.assertNotIn("The user is asking", response_text)
        self.assertNotIn("Looking at the sococc field", response_text)
        self.assertIn("Could you please specify which category or combination of categories you'd like me to count?", response_text)
        self.assertIn("Choose one or more options, or describe your own rule.", response_text)
        self.assertIn("1. Employed", response_text)
        self.assertIn("2. Retired", response_text)
        self.assertIn("3. Unemployed", response_text)
        self.assertIn("4. Unknown", response_text)
        self.assertIn("5. Student", response_text)
        self.assertNotRegex(response_text, r"\d+\.\s+sococc\b")

    def test_after_model_callback_ignores_non_clarification_plain_text(self) -> None:
        callback = build_normalize_clarification_after_model_callback(self._settings())

        result = callback(
            callback_context=SimpleNamespace(state={}),
            llm_response=SimpleNamespace(
                content=types.Content(
                    role="model",
                    parts=[types.Part(text="Found 4 matching rows.")],
                )
            ),
        )

        self.assertIsNone(result)

    def test_after_model_callback_falls_back_for_unparseable_clarification_like_text(self) -> None:
        callback = build_normalize_clarification_after_model_callback(self._settings())
        state: dict[str, object] = {
            SQL_LAST_USER_TEXT_STATE_KEY: "how many females are alcoholics",
        }

        result = callback(
            callback_context=SimpleNamespace(state=state),
            llm_response=SimpleNamespace(
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part(
                            text=(
                                "I need clarification before querying because the request is ambiguous. "
                                "Please specify the exact category, value, or rule you want me to use."
                            )
                        )
                    ],
                )
            ),
        )

        self.assertIsNotNone(result)
        response_text = result.content.parts[0].text
        self.assertEqual(
            response_text,
            "I need clarification before I can run the query. Please specify the exact category, value, or rule you want me to use.",
        )
        self.assertEqual(state[SQL_PENDING_CLARIFICATION_STATE_KEY]["options"], [])
        self.assertEqual(
            state[SQL_PENDING_CLARIFICATION_STATE_KEY]["topic_context"],
            "how many females are alcoholics",
        )

    def test_after_model_callback_falls_back_for_heading_only_option_extraction(self) -> None:
        callback = build_normalize_clarification_after_model_callback(self._settings())
        state: dict[str, object] = {
            SQL_LAST_USER_TEXT_STATE_KEY: "how many males are not working",
        }

        result = callback(
            callback_context=SimpleNamespace(state=state),
            llm_response=SimpleNamespace(
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part(
                            text=(
                                "Available categories:\n"
                                "- Employed\n"
                                "- Retired\n"
                                "- Unemployed\n"
                                "- Student\n"
                            )
                        )
                    ],
                )
            ),
        )

        self.assertIsNotNone(result)
        response_text = result.content.parts[0].text
        self.assertEqual(
            response_text,
            "I need clarification before I can run the query. Please specify the exact category, value, or rule you want me to use.",
        )
        self.assertEqual(state[SQL_PENDING_CLARIFICATION_STATE_KEY]["options"], [])
        self.assertEqual(
            state[SQL_PENDING_CLARIFICATION_STATE_KEY]["topic_context"],
            "how many males are not working",
        )

    def test_after_model_callback_recovers_from_truncated_clarification_json(self) -> None:
        callback = build_normalize_clarification_after_model_callback(self._settings())
        raw_response = """Which category do you mean by 'alcoholics'?\",\"options\":[\"Regularly (regular alcohol consumption)\",\"Occasionally (occasional drinking)\",\"Never (no alcohol)\",\"Unknown (missing data)\"]}

- alcoholics
- Never
- Occasionally
- Regularly
- Unknown
- Alcoholic
- response_type
- clarification
- user_message
- ,
"""

        result = callback(
            callback_context=SimpleNamespace(state={}),
            llm_response=SimpleNamespace(
                content=types.Content(
                    role="model",
                    parts=[types.Part(text=raw_response)],
                )
            ),
        )

        self.assertIsNotNone(result)
        response_text = result.content.parts[0].text
        self.assertIn("Which category do you mean by 'alcoholics'?", response_text)
        self.assertIn("Choose one or more options, or describe your own rule.", response_text)
        self.assertIn("1. Regularly", response_text)
        self.assertIn("2. Occasionally", response_text)
        self.assertIn("3. Never", response_text)
        self.assertIn("4. Unknown", response_text)
        self.assertNotIn("response_type", response_text)
        self.assertNotIn("clarification", response_text)
        self.assertNotIn("user_message", response_text)
        self.assertNotRegex(response_text, r"\d+\.\s+alcoholics\b")
        self.assertNotRegex(response_text, r"\d+\.\s+Alcoholic\b")

    def test_after_model_callback_prefers_embedded_clarification_json_over_fallback_regex(self) -> None:
        with patch(
            "agent_zoo.sql_agent.callbacks.get_schema_summary",
            return_value={
                "status": "success",
                "categorical_value_guidance": [
                    {"column": "gender", "values": ["Female", "Male"]},
                    {"column": "socalc", "values": ["Never", "Occasionally", "Regularly", "Unknown"]},
                ],
            },
        ):
            callback = build_normalize_clarification_after_model_callback(self._settings())
        raw_response = """The user is asking about \"females who are alcoholics\". Let me map this to the schema:

1. \"females\" → gender = \"Female\"
2. \"alcoholics\" → This is tricky. Looking at the socalc field, the stored values are: \"Never\", \"Occasionally\", \"Regularly\", \"Unknown\"

\"Alcoholic\" is not a direct match to any of these stored values. The closest interpretation would be \"Regularly\" (regular alcohol consumption), but this is an assumption.

I should ask for clarification about what the user means by \"alcoholics\" since the schema doesn't have a direct \"alcoholic\" category.
{"response_type":"clarification","user_message":"I need clarification on what you mean by 'alcoholics'. The socalc field (alcohol consumption status) has these stored values: 'Never', 'Occasionally', 'Regularly', 'Unknown'. Which category should I use for 'alcoholics'?\\n\\nOptions:\\n- 'Regularly' (regular alcohol consumption)\\n- 'Occasionally' (occasional alcohol consumption)\\n- Or do you have a different interpretation in mind?","options":["Regularly","Occasionally","Other interpretation"]}
"""

        result = callback(
            callback_context=SimpleNamespace(state={}),
            llm_response=SimpleNamespace(
                content=types.Content(
                    role="model",
                    parts=[types.Part(text=raw_response)],
                )
            ),
        )

        self.assertIsNotNone(result)
        response_text = result.content.parts[0].text
        self.assertIn("I need clarification on what you mean by 'alcoholics'.", response_text)
        self.assertIn("Which category should I use for 'alcoholics'?", response_text)
        self.assertIn("Choose one or more options, or describe your own rule.", response_text)
        self.assertIn("1. Never", response_text)
        self.assertIn("2. Occasionally", response_text)
        self.assertIn("3. Regularly", response_text)
        self.assertIn("4. Unknown", response_text)
        self.assertNotRegex(response_text, r"\d+\.\s+Other interpretation\b")
        self.assertNotIn("\\n\\nOptions:", response_text)
        self.assertNotIn("Options:", response_text)
        self.assertFalse(response_text.startswith("ionally'"))

    def test_after_model_callback_drops_subject_echo_from_fallback_options(self) -> None:
        callback = build_normalize_clarification_after_model_callback(self._settings())
        raw_response = (
            'The phrase "alcoholics" is ambiguous for this dataset. '
            "Which category should I use for 'alcoholics'? "
            'Please choose from "alcoholics", "Never", "Occasionally", "Regularly", "Unknown".'
        )

        result = callback(
            callback_context=SimpleNamespace(state={}),
            llm_response=SimpleNamespace(
                content=types.Content(
                    role="model",
                    parts=[types.Part(text=raw_response)],
                )
            ),
        )

        self.assertIsNotNone(result)
        response_text = result.content.parts[0].text
        self.assertIn("Which category should I use for 'alcoholics'?", response_text)
        self.assertIn("Choose one or more options, or describe your own rule.", response_text)
        self.assertIn("1. Never", response_text)
        self.assertIn("2. Occasionally", response_text)
        self.assertIn("3. Regularly", response_text)
        self.assertIn("4. Unknown", response_text)
        self.assertNotRegex(response_text, r"\d+\.\s+alcoholics\b")

    def test_after_model_callback_ignores_quoted_prose_outside_option_sections(self) -> None:
        with patch(
            "agent_zoo.sql_agent.callbacks.get_schema_summary",
            return_value={
                "status": "success",
                "categorical_value_guidance": [
                    {"column": "socalc", "values": ["Never", "Occasionally", "Regularly", "Unknown"]},
                ],
            },
        ):
            callback = build_normalize_clarification_after_model_callback(self._settings())
        raw_response = """The user is asking about \"alcoholics\" which is a colloquial term.
The available categories for alcohol consumption are:
- \"Never\"
- \"Occasionally\"
- \"Regularly\"
- \"Unknown\"
\"Alcoholic\" is not an exact match to any stored value. The closest interpretation would be \"Regularly\".
Could you please specify which category you'd like me to use?"""

        result = callback(
            callback_context=SimpleNamespace(state={}),
            llm_response=SimpleNamespace(
                content=types.Content(
                    role="model",
                    parts=[types.Part(text=raw_response)],
                )
            ),
        )

        self.assertIsNotNone(result)
        response_text = result.content.parts[0].text
        self.assertIn("Could you please specify which category you'd like me to use?", response_text)
        self.assertIn("1. Never", response_text)
        self.assertIn("2. Occasionally", response_text)
        self.assertIn("3. Regularly", response_text)
        self.assertIn("4. Unknown", response_text)
        self.assertNotRegex(response_text, r"\d+\.\s+alcoholics\b")
        self.assertNotRegex(response_text, r"\d+\.\s+Alcoholic\b")

    def test_after_model_callback_clamps_options_to_exact_dataset_categories(self) -> None:
        with patch(
            "agent_zoo.sql_agent.callbacks.get_schema_summary",
            return_value={
                "status": "success",
                "categorical_value_guidance": [
                    {"column": "gender", "values": ["Female", "Male"]},
                    {"column": "socalc", "values": ["Never", "Occasionally", "Regularly", "Unknown"]},
                ],
            },
        ):
            callback = build_normalize_clarification_after_model_callback(self._settings())
        raw_response = """The user is asking about \"alcoholics\" but the socalc field has categories: \"Never\", \"Occasionally\", \"Regularly\", \"Unknown\".

\"Alcoholic\" is not a direct match to any stored value. I need to clarify what the user means by \"alcoholic\" since it could map to:
- \"Regularly\" (heavy drinkers)
- \"Occasionally\" (anyone who drinks)
- Both combined

I need clarification on what you mean by \"alcoholics\" since the socalc field has these categories: \"Never\", \"Occasionally\", \"Regularly\", \"Unknown\".

Which category or combination should I use for \"alcoholic\"?

- Regularly
- Occasionally
- Both Regularly and Occasionally
- Something else"""

        result = callback(
            callback_context=SimpleNamespace(state={}),
            llm_response=SimpleNamespace(
                content=types.Content(
                    role="model",
                    parts=[types.Part(text=raw_response)],
                )
            ),
        )

        self.assertIsNotNone(result)
        response_text = result.content.parts[0].text
        self.assertIn("1. Never", response_text)
        self.assertIn("2. Occasionally", response_text)
        self.assertIn("3. Regularly", response_text)
        self.assertIn("4. Unknown", response_text)
        self.assertNotRegex(response_text, r"\d+\.\s+Both combined\b")
        self.assertNotRegex(response_text, r"\d+\.\s+Both Regularly and Occasionally\b")
        self.assertNotRegex(response_text, r"\d+\.\s+Something else\b")

    def test_combined_before_model_callback_rewrites_pending_clarification_followup(self) -> None:
        scope_gate_calls: list[str] = []
        resolver_calls: list[tuple[str, str, list[str], str]] = []

        def fake_scope_gate(user_text: str) -> tuple[bool, str | None]:
            scope_gate_calls.append(user_text)
            return False, "blocked"

        def fake_resolver(topic_context: str, clarification_question: str, options: list[str], user_reply: str) -> dict[str, object]:
            resolver_calls.append((topic_context, clarification_question, options, user_reply))
            return {"resolution_type": "custom_rule", "selected_options": [], "custom_rule": user_reply}

        with patch("agent_zoo.sql_agent.callbacks.build_llm_scope_gate", return_value=fake_scope_gate), patch(
            "agent_zoo.sql_agent.callbacks.build_llm_clarification_resolver",
            return_value=fake_resolver,
        ):
            callback = build_combined_before_model_callback(self._settings())

        state = {
            SQL_PENDING_CLARIFICATION_STATE_KEY: {
                "topic_context": "how many females are alcoholics",
                "user_message": "Which category should I use for 'alcoholics'?",
                "options": ["Never", "Occasionally", "Regularly", "Unknown"],
            }
        }
        llm_request = SimpleNamespace(
            contents=[types.Content(role="user", parts=[types.Part(text="occasionally and regularly")])]
        )

        result = callback(
            callback_context=SimpleNamespace(state=state),
            llm_request=llm_request,
        )

        self.assertIsNone(result)
        self.assertEqual(scope_gate_calls, [])
        self.assertEqual(resolver_calls, [])
        rewritten_text = llm_request.contents[-1].parts[0].text
        self.assertIn("The user is replying to the previous clarification", rewritten_text)
        self.assertIn("Matched options from the reply: Occasionally, Regularly", rewritten_text)
        self.assertIn("User clarification reply: occasionally and regularly", rewritten_text)
        self.assertNotIn(SQL_PENDING_CLARIFICATION_STATE_KEY, state)

    def test_combined_before_model_callback_rewrites_numeric_pending_clarification_followups(self) -> None:
        scope_gate_calls: list[str] = []
        resolver_calls: list[tuple[str, str, list[str], str]] = []

        def fake_scope_gate(user_text: str) -> tuple[bool, str | None]:
            scope_gate_calls.append(user_text)
            return False, "blocked"

        def fake_resolver(topic_context: str, clarification_question: str, options: list[str], user_reply: str) -> dict[str, object]:
            resolver_calls.append((topic_context, clarification_question, options, user_reply))
            return {"resolution_type": "custom_rule", "selected_options": [], "custom_rule": user_reply}

        with patch("agent_zoo.sql_agent.callbacks.build_llm_scope_gate", return_value=fake_scope_gate), patch(
            "agent_zoo.sql_agent.callbacks.build_llm_clarification_resolver",
            return_value=fake_resolver,
        ):
            callback = build_combined_before_model_callback(self._settings())

        for reply_text, matched_text in [
            ("2", "Occasionally"),
            ("2 and 3", "Occasionally, Regularly"),
            ("2,3", "Occasionally, Regularly"),
        ]:
            with self.subTest(reply_text=reply_text):
                state = {
                    SQL_PENDING_CLARIFICATION_STATE_KEY: {
                        "topic_context": "how many females are alcoholics",
                        "user_message": "Which category should I use for 'alcoholics'?",
                        "options": ["Never", "Occasionally", "Regularly", "Unknown"],
                    }
                }
                llm_request = SimpleNamespace(
                    contents=[types.Content(role="user", parts=[types.Part(text=reply_text)])]
                )
                scope_gate_calls.clear()
                resolver_calls.clear()

                result = callback(
                    callback_context=SimpleNamespace(state=state),
                    llm_request=llm_request,
                )

                self.assertIsNone(result)
                self.assertEqual(scope_gate_calls, [])
                self.assertEqual(resolver_calls, [])
                rewritten_text = llm_request.contents[-1].parts[0].text
                self.assertIn("The user is replying to the previous clarification", rewritten_text)
                self.assertIn(f"Matched options from the reply: {matched_text}", rewritten_text)
                self.assertIn(f"User clarification reply: {reply_text}", rewritten_text)
                self.assertNotIn(SQL_PENDING_CLARIFICATION_STATE_KEY, state)

    def test_combined_before_model_callback_skips_scope_gate_for_selection_like_followup(self) -> None:
        scope_gate_calls: list[str] = []
        resolver_calls: list[tuple[str, str, list[str], str]] = []

        def fake_scope_gate(user_text: str) -> tuple[bool, str | None]:
            scope_gate_calls.append(user_text)
            return False, "blocked"

        def fake_resolver(topic_context: str, clarification_question: str, options: list[str], user_reply: str) -> dict[str, object]:
            resolver_calls.append((topic_context, clarification_question, options, user_reply))
            return {
                "resolution_type": "selected_options",
                "selected_options": ["Occasionally", "Regularly"],
                "custom_rule": "",
            }

        with patch("agent_zoo.sql_agent.callbacks.build_llm_scope_gate", return_value=fake_scope_gate), patch(
            "agent_zoo.sql_agent.callbacks.build_llm_clarification_resolver",
            return_value=fake_resolver,
        ):
            callback = build_combined_before_model_callback(self._settings())

        state = {
            SQL_PENDING_CLARIFICATION_STATE_KEY: {
                "topic_context": "how many females are alcoholics",
                "user_message": "Which category should I use for 'alcoholics'?",
                "options": ["Never", "Occasionally", "Regularly", "Unknown"],
            }
        }
        llm_request = SimpleNamespace(
            contents=[types.Content(role="user", parts=[types.Part(text="count both drinking categories")])]
        )

        result = callback(
            callback_context=SimpleNamespace(state=state),
            llm_request=llm_request,
        )

        self.assertIsNone(result)
        self.assertEqual(scope_gate_calls, [])
        self.assertEqual(
            resolver_calls,
            [
                (
                    "how many females are alcoholics",
                    "Which category should I use for 'alcoholics'?",
                    ["Never", "Occasionally", "Regularly", "Unknown"],
                    "count both drinking categories",
                )
            ],
        )
        rewritten_text = llm_request.contents[-1].parts[0].text
        self.assertIn("Clarification question: Which category should I use for 'alcoholics'?", rewritten_text)
        self.assertIn("Matched options from the reply: Occasionally, Regularly", rewritten_text)
        self.assertIn("User clarification reply: count both drinking categories", rewritten_text)
        self.assertNotIn(SQL_PENDING_CLARIFICATION_STATE_KEY, state)

    def test_combined_before_model_callback_stores_terminal_user_question_as_topic_context(self) -> None:
        def fake_scope_gate(user_text: str) -> tuple[bool, str | None]:
            return True, None

        with patch("agent_zoo.sql_agent.callbacks.build_llm_scope_gate", return_value=fake_scope_gate), patch(
            "agent_zoo.sql_agent.callbacks.build_llm_clarification_resolver",
            return_value=lambda *args, **kwargs: {"resolution_type": "custom_rule", "selected_options": [], "custom_rule": ""},
        ):
            callback = build_combined_before_model_callback(self._settings())

        llm_request = SimpleNamespace(
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            text=(
                                "The SQLite database is a snapshot of the current filtered cohort from the web app.\n"
                                "Use only the table `filtered_dataset`.\n"
                                "\n"
                                "Relevant categorical value guidance:\n"
                                "- socalc: Stored SQLite values seen in the current cohort: \"Never\", \"Occasionally\", \"Regularly\", \"Unknown\"\n"
                                "\n"
                                "User question:\n"
                                "how many females are alcoholics"
                            )
                        )
                    ],
                )
            ]
        )
        state: dict[str, object] = {}

        result = callback(
            callback_context=SimpleNamespace(state=state),
            llm_request=llm_request,
        )

        self.assertIsNone(result)
        self.assertEqual(state[SQL_LAST_USER_TEXT_STATE_KEY], "how many females are alcoholics")

    def test_pending_clarification_state_survives_invocation_boundary(self) -> None:
        scope_gate_calls: list[str] = []

        def fake_scope_gate(user_text: str) -> tuple[bool, str | None]:
            scope_gate_calls.append(user_text)
            return False, "blocked"

        with patch("agent_zoo.sql_agent.callbacks.build_llm_scope_gate", return_value=fake_scope_gate), patch(
            "agent_zoo.sql_agent.callbacks.build_llm_clarification_resolver",
            return_value=lambda *args, **kwargs: {"resolution_type": "custom_rule", "selected_options": [], "custom_rule": ""},
        ):
            before_callback = build_combined_before_model_callback(self._settings())

        after_callback = build_normalize_clarification_after_model_callback(self._settings())
        state: dict[str, object] = {
            SQL_LAST_USER_TEXT_STATE_KEY: "how many males are alcoholics",
        }

        clarification_response = after_callback(
            callback_context=SimpleNamespace(state=state),
            llm_response=SimpleNamespace(
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part(
                            text=(
                                '{"response_type":"clarification","user_message":'
                                '"Which category should I use for \'alcoholics\'?",'
                                '"options":["Never","Occasionally","Regularly","Unknown"]}'
                            )
                        )
                    ],
                )
            ),
        )

        self.assertIsNotNone(clarification_response)
        self.assertIn(SQL_PENDING_CLARIFICATION_STATE_KEY, state)

        persisted_state = {
            key: value
            for key, value in state.items()
            if not key.startswith("temp:")
        }
        llm_request = SimpleNamespace(
            contents=[types.Content(role="user", parts=[types.Part(text="occasionally and regularly")])]
        )

        result = before_callback(
            callback_context=SimpleNamespace(state=persisted_state),
            llm_request=llm_request,
        )

        self.assertIsNone(result)
        self.assertEqual(scope_gate_calls, [])
        rewritten_text = llm_request.contents[-1].parts[0].text
        self.assertIn("The user is replying to the previous clarification", rewritten_text)
        self.assertIn("Matched options from the reply: Occasionally, Regularly", rewritten_text)
        self.assertNotIn(SQL_PENDING_CLARIFICATION_STATE_KEY, persisted_state)

    def test_combined_before_model_callback_keeps_custom_rule_followup_in_clarification_flow(self) -> None:
        scope_gate_calls: list[str] = []
        resolver_calls: list[tuple[str, str, list[str], str]] = []

        def fake_scope_gate(user_text: str) -> tuple[bool, str | None]:
            scope_gate_calls.append(user_text)
            return False, "blocked"

        def fake_resolver(topic_context: str, clarification_question: str, options: list[str], user_reply: str) -> dict[str, object]:
            resolver_calls.append((topic_context, clarification_question, options, user_reply))
            return {
                "resolution_type": "custom_rule",
                "selected_options": [],
                "custom_rule": "match any drinking category",
            }

        with patch("agent_zoo.sql_agent.callbacks.build_llm_scope_gate", return_value=fake_scope_gate), patch(
            "agent_zoo.sql_agent.callbacks.build_llm_clarification_resolver",
            return_value=fake_resolver,
        ):
            callback = build_combined_before_model_callback(self._settings())

        state = {
            SQL_PENDING_CLARIFICATION_STATE_KEY: {
                "topic_context": "how many females are alcoholics",
                "user_message": "Which category should I use for 'alcoholics'?",
                "options": ["Never", "Occasionally", "Regularly", "Unknown"],
            }
        }
        llm_request = SimpleNamespace(
            contents=[types.Content(role="user", parts=[types.Part(text="match any drinking category")])]
        )

        result = callback(
            callback_context=SimpleNamespace(state=state),
            llm_request=llm_request,
        )

        self.assertIsNone(result)
        self.assertEqual(scope_gate_calls, [])
        self.assertEqual(
            resolver_calls,
            [
                (
                    "how many females are alcoholics",
                    "Which category should I use for 'alcoholics'?",
                    ["Never", "Occasionally", "Regularly", "Unknown"],
                    "match any drinking category",
                )
            ],
        )
        rewritten_text = llm_request.contents[-1].parts[0].text
        self.assertIn("Resolved custom rule from the reply: match any drinking category", rewritten_text)
        self.assertIn("User clarification reply: match any drinking category", rewritten_text)
        self.assertNotIn(SQL_PENDING_CLARIFICATION_STATE_KEY, state)

    def test_combined_before_model_callback_turns_result_refinement_into_clarification(self) -> None:
        scope_gate_calls: list[str] = []

        def fake_scope_gate(user_text: str) -> tuple[bool, str | None]:
            scope_gate_calls.append(user_text)
            return False, "blocked"

        with patch("agent_zoo.sql_agent.callbacks.build_llm_scope_gate", return_value=fake_scope_gate), patch(
            "agent_zoo.sql_agent.callbacks.build_llm_clarification_resolver",
            return_value=lambda *args, **kwargs: {"resolution_type": "custom_rule", "selected_options": [], "custom_rule": ""},
        ), patch(
            "agent_zoo.sql_agent.callbacks.build_llm_result_refinement_resolver",
            return_value=lambda frame, user_text: {
                "resolution_type": "needs_clarification",
                "target_column": "sococc",
                "selected_values": [],
                "refinement_request": "",
            },
        ):
            callback = build_combined_before_model_callback(self._settings())

        state = {
            SQL_LAST_QUERY_FRAME_STATE_KEY: {
                "question": "how many males dont work",
                "sql": "SELECT COUNT(*) AS matching_count FROM filtered_dataset WHERE gender = 'Male' AND sococc = 'Unemployed'",
                "categorical_filters": [
                    {"column": "gender", "selected_values": ["Male"], "available_values": ["Male", "Female"]},
                    {
                        "column": "sococc",
                        "selected_values": ["Unemployed"],
                        "available_values": ["Employed", "Retired", "Student", "Unemployed", "Unknown"],
                    },
                ],
            }
        }
        llm_request = SimpleNamespace(
            contents=[types.Content(role="user", parts=[types.Part(text="i want to add other categories to dont work")])]
        )

        result = callback(
            callback_context=SimpleNamespace(state=state),
            llm_request=llm_request,
        )

        self.assertIsNotNone(result)
        self.assertEqual(scope_gate_calls, [])
        response_text = result.content.parts[0].text
        self.assertIn("I previously used sococc = Unemployed.", response_text)
        self.assertIn("Which values from sococc should I include now?", response_text)
        self.assertIn("You can reply with option numbers like 2 or 2 and 3.", response_text)
        self.assertIn("1. Employed", response_text)
        self.assertIn("2. Retired", response_text)
        self.assertIn("3. Student", response_text)
        self.assertIn("4. Unemployed", response_text)
        self.assertIn("5. Unknown", response_text)
        self.assertEqual(
            state[SQL_PENDING_CLARIFICATION_STATE_KEY]["options"],
            ["Employed", "Retired", "Student", "Unemployed", "Unknown"],
        )

    def test_combined_before_model_callback_rewrites_result_refinement_followup(self) -> None:
        scope_gate_calls: list[str] = []

        def fake_scope_gate(user_text: str) -> tuple[bool, str | None]:
            scope_gate_calls.append(user_text)
            return False, "blocked"

        with patch("agent_zoo.sql_agent.callbacks.build_llm_scope_gate", return_value=fake_scope_gate), patch(
            "agent_zoo.sql_agent.callbacks.build_llm_clarification_resolver",
            return_value=lambda *args, **kwargs: {"resolution_type": "custom_rule", "selected_options": [], "custom_rule": ""},
        ), patch(
            "agent_zoo.sql_agent.callbacks.build_llm_result_refinement_resolver",
            return_value=lambda frame, user_text: {
                "resolution_type": "refine_query",
                "target_column": "sococc",
                "selected_values": ["Retired", "Student", "Unemployed"],
                "refinement_request": "",
            },
        ):
            callback = build_combined_before_model_callback(self._settings())

        state = {
            SQL_LAST_QUERY_FRAME_STATE_KEY: {
                "question": "how many males dont work",
                "sql": "SELECT COUNT(*) AS matching_count FROM filtered_dataset WHERE gender = 'Male' AND sococc = 'Unemployed'",
                "categorical_filters": [
                    {"column": "gender", "selected_values": ["Male"], "available_values": ["Male", "Female"]},
                    {
                        "column": "sococc",
                        "selected_values": ["Unemployed"],
                        "available_values": ["Employed", "Retired", "Student", "Unemployed", "Unknown"],
                    },
                ],
            }
        }
        llm_request = SimpleNamespace(
            contents=[types.Content(role="user", parts=[types.Part(text="include retired and student too")])]
        )

        result = callback(
            callback_context=SimpleNamespace(state=state),
            llm_request=llm_request,
        )

        self.assertIsNone(result)
        self.assertEqual(scope_gate_calls, [])
        rewritten_text = llm_request.contents[-1].parts[0].text
        self.assertIn("The user is refining the previous dataset request", rewritten_text)
        self.assertIn("Previous dataset question: how many males dont work", rewritten_text)
        self.assertIn("Use this updated value set for sococc: Retired, Student, Unemployed", rewritten_text)

    def test_combined_before_model_callback_applies_scope_gate_for_topic_change(self) -> None:
        scope_gate_calls: list[str] = []
        resolver_calls: list[tuple[str, str, list[str], str]] = []

        def fake_scope_gate(user_text: str) -> tuple[bool, str | None]:
            scope_gate_calls.append(user_text)
            return False, "blocked"

        def fake_resolver(topic_context: str, clarification_question: str, options: list[str], user_reply: str) -> dict[str, object]:
            resolver_calls.append((topic_context, clarification_question, options, user_reply))
            return {
                "resolution_type": "topic_change",
                "selected_options": [],
                "custom_rule": "",
            }

        with patch("agent_zoo.sql_agent.callbacks.build_llm_scope_gate", return_value=fake_scope_gate), patch(
            "agent_zoo.sql_agent.callbacks.build_llm_clarification_resolver",
            return_value=fake_resolver,
        ):
            callback = build_combined_before_model_callback(self._settings())

        state = {
            SQL_PENDING_CLARIFICATION_STATE_KEY: {
                "topic_context": "how many females are alcoholics",
                "user_message": "Which category should I use for 'alcoholics'?",
                "options": ["Never", "Occasionally", "Regularly", "Unknown"],
            }
        }
        llm_request = SimpleNamespace(
            contents=[types.Content(role="user", parts=[types.Part(text="tell me a joke")])]
        )

        result = callback(
            callback_context=SimpleNamespace(state=state),
            llm_request=llm_request,
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.content.parts[0].text, "blocked")
        self.assertEqual(
            resolver_calls,
            [
                (
                    "how many females are alcoholics",
                    "Which category should I use for 'alcoholics'?",
                    ["Never", "Occasionally", "Regularly", "Unknown"],
                    "tell me a joke",
                )
            ],
        )
        self.assertEqual(scope_gate_calls, ["tell me a joke"])
        self.assertNotIn(SQL_PENDING_CLARIFICATION_STATE_KEY, state)

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

