from __future__ import annotations

import sqlite3
import sys
import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import patch

from google.genai import types

from agent_zoo.scope_guard import (
    build_llm_clarification_resolver,
    build_llm_fresh_topic_relevance_router,
    build_llm_result_refinement_resolver,
    build_llm_schema_grounding_resolver,
    build_llm_scope_gate,
)
from agent_zoo.sql_agent.callbacks import (
    _collect_grounding_match_evidence,
    build_combined_before_model_callback,
)
from agent_zoo.sql_agent.config import SQLAgentSettings
from agent_zoo.sql_agent.db import execute_sqlite_query, get_schema_summary, validate_sql_read_only
from agent_zoo.sql_agent.instructions import build_agent_instruction
from agent_zoo.sql_agent.tools import build_sql_tools

from tests._fixtures import (
    REPO_ROOT,
    create_fixture_database,
    create_missing_group_fixture_database,
    set_sql_current_query_frame,
)


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
        self.assertNotIn("Lookup tables:", summary["schema_text"])

    def test_get_schema_summary_separates_lookup_tables_from_primary_tables(self) -> None:
        connection = sqlite3.connect(self.db_path)
        try:
            connection.executescript(
                """
                CREATE TABLE schema_columns (
                    table_name TEXT NOT NULL,
                    column_name TEXT NOT NULL,
                    field_label TEXT
                );

                CREATE TABLE schema_categories (
                    table_name TEXT NOT NULL,
                    column_name TEXT NOT NULL,
                    category_value TEXT NOT NULL
                );
                """
            )
            connection.commit()
        finally:
            connection.close()

        summary = get_schema_summary(self.db_path)

        self.assertEqual(summary["status"], "success")
        schema_text = summary["schema_text"]
        self.assertIn("Primary tables:\n", schema_text)
        self.assertIn("\n\nLookup tables:\n", schema_text)

        primary_text, lookup_text = schema_text.split("\n\nLookup tables:\n", 1)
        self.assertTrue(primary_text.startswith("Primary tables:\n"))
        self.assertIn("people(id INTEGER PRIMARY KEY", primary_text)
        self.assertIn("visits(id INTEGER PRIMARY KEY", primary_text)
        self.assertNotIn("schema_columns(", primary_text)
        self.assertNotIn("schema_categories(", primary_text)
        self.assertIn("schema_categories(table_name TEXT NOT NULL", lookup_text)
        self.assertIn("schema_columns(table_name TEXT NOT NULL", lookup_text)

        table_names = [table["name"] for table in summary["tables"]]
        self.assertIn("schema_columns", table_names)
        self.assertIn("schema_categories", table_names)

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
        self.assertEqual(people_columns["sex"]["source_header"], "Sex")
        self.assertEqual(people_columns["sex"]["categorical_values"], ["female", "male"])
        self.assertNotIn("categorical_values", people_columns["name"])

    def test_get_schema_summary_includes_column_glossary_from_mapping_table(self) -> None:
        summary = get_schema_summary(self.db_path)

        self.assertEqual(summary["status"], "success")
        people_table = next(table for table in summary["tables"] if table["name"] == "people")
        visits_table = next(table for table in summary["tables"] if table["name"] == "visits")
        people_columns = {column["name"]: column for column in people_table["columns"]}
        visits_columns = {column["name"]: column for column in visits_table["columns"]}
        self.assertEqual(people_columns["name"]["source_header"], "Name")
        self.assertEqual(people_columns["sex"]["source_header"], "Sex")
        self.assertEqual(visits_columns["person_id"]["source_header"], "PersonId")
        self.assertEqual(visits_columns["city"]["source_header"], "City")

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

    def test_validate_sql_read_only_allows_case_expressions(self) -> None:
        validation = validate_sql_read_only(
            """
            SELECT CASE
                WHEN age <= 17 THEN '0-17'
                WHEN age >= 18 AND age <= 39 THEN '18-39'
                WHEN age >= 40 AND age <= 59 THEN '40-59'
                WHEN age >= 60 AND age <= 79 THEN '60-79'
                WHEN age >= 80 THEN '80+'
                ELSE 'Unknown / Null'
            END AS age_category,
            COUNT(*) AS patient_count
            FROM people
            GROUP BY age_category
            ORDER BY age_category
            """,
            self.db_path,
        )

        self.assertTrue(validation["is_valid"])

    def test_validate_sql_read_only_allows_with_union_query(self) -> None:
        validation = validate_sql_read_only(
            """
            WITH person_names AS (
                SELECT name AS label FROM people
            )
            SELECT label FROM person_names
            UNION
            SELECT city AS label FROM visits
            """,
            self.db_path,
        )

        self.assertTrue(validation["is_valid"])

    def test_validate_sql_read_only_allows_bracket_quoted_keyword_aliases(self) -> None:
        validation = validate_sql_read_only(
            "SELECT [name] AS [UPDATE] FROM people ORDER BY [UPDATE]",
            self.db_path,
        )

        self.assertTrue(validation["is_valid"])

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

    def test_validate_sql_read_only_blocks_with_delete_and_update(self) -> None:
        delete_validation = validate_sql_read_only(
            """
            WITH filtered AS (
                SELECT id FROM people WHERE sex = 'female'
            )
            DELETE FROM people WHERE id IN (SELECT id FROM filtered)
            """,
            self.db_path,
        )
        update_validation = validate_sql_read_only(
            """
            WITH filtered AS (
                SELECT id FROM people WHERE sex = 'female'
            )
            UPDATE people SET age = 0 WHERE id IN (SELECT id FROM filtered)
            """,
            self.db_path,
        )

        self.assertFalse(delete_validation["is_valid"])
        self.assertIn("Only read-only SELECT and WITH queries are allowed", delete_validation["reason"])
        self.assertFalse(update_validation["is_valid"])
        self.assertIn("Only read-only SELECT and WITH queries are allowed", update_validation["reason"])

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

    def test_execute_sqlite_query_rewrites_categorical_negation_to_explicit_values(self) -> None:
        result = execute_sqlite_query(
            self.db_path,
            "SELECT COUNT(*) AS matching_count FROM visits WHERE city != 'Tokyo'",
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["rows"], [{"matching_count": 3}])
        self.assertEqual(
            result["sql"],
            "SELECT COUNT(*) AS matching_count FROM visits WHERE city IN ('Paris', 'Singapore')",
        )
        self.assertEqual(result.get("display_sql"), result["sql"])
        self.assertNotIn("!=", result["display_sql"])

    def test_execute_sqlite_query_rewrites_multiple_categorical_negations_to_explicit_values(self) -> None:
        result = execute_sqlite_query(
            self.db_path,
            "SELECT COUNT(*) AS matching_count FROM visits WHERE city != 'Tokyo' AND city != 'Paris'",
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["rows"], [{"matching_count": 2}])
        self.assertNotIn("!=", result["display_sql"])
        self.assertIn("city IN", result["display_sql"])

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
        self.assertIn("do not express complements with `!=`, `<>`, or `NOT IN`", instruction)

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

    def test_combined_before_model_callback_builds_richer_scope_gate_context(self) -> None:
        captured: dict[str, str] = {}

        def fake_build_llm_scope_gate(model: str, schema_context: str, refusal_message: str | None = None):
            captured["schema_context"] = schema_context

            def fake_scope_gate(user_text: str) -> tuple[bool, str | None]:
                return True, None

            return fake_scope_gate

        with patch(
            "agent_zoo.sql_agent.callbacks.get_schema_summary",
            return_value={
                "status": "success",
                "schema_text": "filtered_dataset(gender TEXT, socalc TEXT)",
                "tables": [
                    {
                        "name": "filtered_dataset",
                        "columns": [
                            {
                                "name": "gender",
                                "type": "TEXT",
                                "source_header": "Sex of patient",
                                "categorical_values": ["Female", "Male"],
                            },
                            {
                                "name": "socalc",
                                "type": "TEXT",
                                "source_header": "Alcohol consumption status",
                                "categorical_values": ["Never", "Occasionally", "Regularly", "Unknown"],
                            },
                        ],
                    }
                ],
                "categorical_value_guidance": [
                    {"table": "filtered_dataset", "column": "gender", "values": ["Female", "Male"]},
                    {
                        "table": "filtered_dataset",
                        "column": "socalc",
                        "values": ["Never", "Occasionally", "Regularly", "Unknown"],
                    },
                ],
                "categorical_value_guidance_text": (
                    '- filtered_dataset.gender: Stored SQLite values seen in the dataset: "Female", "Male"\n'
                    '- filtered_dataset.socalc: Stored SQLite values seen in the dataset: "Never", "Occasionally", "Regularly", "Unknown"'
                ),
            },
        ), patch(
            "agent_zoo.sql_agent.callbacks.build_llm_scope_gate",
            side_effect=fake_build_llm_scope_gate,
        ), patch(
            "agent_zoo.sql_agent.callbacks.build_llm_clarification_resolver",
            return_value=lambda *args, **kwargs: {"resolution_type": "custom_rule", "selected_options": [], "custom_rule": ""},
        ), patch(
            "agent_zoo.sql_agent.callbacks.build_llm_result_refinement_resolver",
            return_value=lambda *args, **kwargs: {"resolution_type": "topic_change", "target_column": "", "selected_values": [], "refinement_request": ""},
        ), patch(
            "agent_zoo.sql_agent.callbacks.build_llm_schema_grounding_resolver",
            return_value=lambda *args, **kwargs: {"resolution_type": "proceed", "grounded_filters": {}, "candidate_columns": []},
        ):
            build_combined_before_model_callback(
                SQLAgentSettings(
                    db_path=self.db_path,
                    model="test-model",
                )
            )

        self.assertIn("filtered_dataset(gender TEXT, socalc TEXT)", captured["schema_context"])
        self.assertIn("Schema columns and semantic labels:", captured["schema_context"])
        self.assertIn("field label = Sex of patient", captured["schema_context"])
        self.assertIn("field label = Alcohol consumption status", captured["schema_context"])
        self.assertIn("categorical values = Female, Male", captured["schema_context"])
        self.assertIn("Stored categorical value guidance:", captured["schema_context"])

    def test_combined_before_model_callback_topic_change_uses_fresh_topic_router_before_schema_grounding(self) -> None:
        scope_gate_calls: list[str] = []
        fresh_topic_router_calls: list[tuple[str, str | None]] = []
        schema_grounding_calls: list[tuple[str, str, list[dict[str, object]], list[str]]] = []

        def fake_scope_gate(user_text: str) -> tuple[bool, str | None]:
            scope_gate_calls.append(user_text)
            return False, "blocked"

        def fake_fresh_topic_router(user_text: str, current_topic: str | None = None) -> dict[str, str]:
            fresh_topic_router_calls.append((user_text, current_topic))
            return {"resolution_type": "dataset_question"}

        def fake_schema_grounding_resolver(
            user_text: str,
            schema_context: str,
            grounding_candidates: list[dict[str, object]],
            candidate_columns: list[str],
        ) -> dict[str, object]:
            schema_grounding_calls.append((user_text, schema_context, grounding_candidates, candidate_columns))
            return {
                "resolution_type": "needs_clarification",
                "grounded_filters": {"gender": ["Female"]},
                "candidate_columns": ["socalc", "socsmk"],
            }

        with patch(
            "agent_zoo.sql_agent.callbacks.get_schema_summary",
            return_value={
                "status": "success",
                "schema_text": "filtered_dataset(gender TEXT, socsmk TEXT, socalc TEXT)",
                "tables": [
                    {
                        "name": "filtered_dataset",
                        "columns": [
                            {
                                "name": "gender",
                                "type": "TEXT",
                                "source_header": "Sex of patient",
                                "categorical_values": ["Female", "Male"],
                            },
                            {
                                "name": "socsmk",
                                "type": "TEXT",
                                "source_header": "Smoking status",
                                "categorical_values": ["No", "Unknown", "Yes"],
                            },
                            {
                                "name": "socalc",
                                "type": "TEXT",
                                "source_header": "Alcohol consumption status",
                                "categorical_values": ["Never", "Occasionally", "Regularly", "Unknown"],
                            },
                        ],
                    }
                ],
                "categorical_value_guidance": [
                    {"column": "gender", "values": ["Female", "Male"]},
                    {"column": "socsmk", "values": ["No", "Unknown", "Yes"]},
                    {
                        "column": "socalc",
                        "values": ["Never", "Occasionally", "Regularly", "Unknown"],
                    },
                ],
                "categorical_value_guidance_text": "",
            },
        ), patch(
            "agent_zoo.sql_agent.callbacks.build_llm_scope_gate",
            return_value=fake_scope_gate,
        ), patch(
            "agent_zoo.sql_agent.callbacks.build_llm_clarification_resolver",
            return_value=lambda *args, **kwargs: {"resolution_type": "custom_rule", "selected_options": [], "custom_rule": ""},
        ), patch(
            "agent_zoo.sql_agent.callbacks.build_llm_fresh_topic_relevance_router",
            return_value=fake_fresh_topic_router,
        ), patch(
            "agent_zoo.sql_agent.callbacks.build_llm_result_refinement_resolver",
            return_value=lambda *args, **kwargs: {"resolution_type": "topic_change", "target_column": "", "selected_values": [], "refinement_request": ""},
        ), patch(
            "agent_zoo.sql_agent.callbacks.build_llm_schema_grounding_resolver",
            return_value=fake_schema_grounding_resolver,
        ):
            callback = build_combined_before_model_callback(
                SQLAgentSettings(
                    db_path=self.db_path,
                    model="test-model",
                )
            )

        state: dict[str, object] = {}
        set_sql_current_query_frame(
            state,
            {
                "question": "what is the max age of working professionals",
                "sql": "SELECT MAX(age) AS maximum_age, COUNT(*) AS matching_count FROM filtered_dataset WHERE sococc = 'Employed'",
                "categorical_filters": [
                    {
                        "column": "sococc",
                        "selected_values": ["Employed"],
                        "available_values": ["Employed", "Retired", "Student", "Unemployed", "Unknown"],
                    }
                ],
            },
        )
        llm_request = SimpleNamespace(
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            text=(
                                "The SQLite database is a snapshot of the current filtered cohort from the web app.\n"
                                "Use only the table `filtered_dataset`.\n\n"
                                "Field glossary:\n"
                                "- gender: Sex of patient (categorical)\n"
                                "- socsmk: Smoking status (categorical)\n"
                                "- socalc: Alcohol consumption status (categorical)\n\n"
                                "User question:\n"
                                "how many ladies drink"
                            )
                        )
                    ],
                )
            ]
        )

        result = callback(
            callback_context=SimpleNamespace(state=state),
            llm_request=llm_request,
        )

        self.assertIsNotNone(result)
        self.assertEqual(scope_gate_calls, [])
        self.assertEqual(len(fresh_topic_router_calls), 1)
        self.assertEqual(
            fresh_topic_router_calls[0][1],
            "what is the max age of working professionals",
        )
        self.assertIn("Field glossary:", fresh_topic_router_calls[0][0])
        self.assertIn("User question:\nhow many ladies drink", fresh_topic_router_calls[0][0])
        self.assertEqual(len(schema_grounding_calls), 1)
        response_text = result.content.parts[0].text
        self.assertIn("Which one do you mean?", response_text)
        self.assertIn("1. Alcohol consumption status", response_text)
        self.assertIn("2. Smoking status", response_text)

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
        self.assertIn("Current committed dataset question/topic", captured["user_prompt"])
        self.assertIn("how many males dont work", captured["user_prompt"])
        self.assertIn("Current committed SQL query", captured["user_prompt"])
        self.assertIn("Categorical filters from the current committed query", captured["user_prompt"])
        self.assertIn("Latest user reply", captured["user_prompt"])

    def test_result_refinement_resolver_includes_recent_refinement_history(self) -> None:
        captured: dict[str, str] = {}

        def completion(**kwargs):
            captured["user_prompt"] = kwargs["messages"][1]["content"]
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content='{"resolution_type":"topic_change","target_column":"","selected_values":[],"refinement_request":""}'
                        )
                    )
                ]
            )

        resolver = build_llm_result_refinement_resolver("test-model")

        with patch.dict(sys.modules, {"litellm": SimpleNamespace(completion=completion)}):
            resolver(
                {
                    "question": "how many males work",
                    "topic_context": (
                        "Current committed dataset question/topic: how many males work\n\n"
                        "Current committed categorical filters:\n- gender = Female\n- sococc = Employed"
                    ),
                    "sql": "SELECT COUNT(*) AS matching_count FROM filtered_dataset WHERE gender = 'Female' AND sococc = 'Employed'",
                    "categorical_filters": [
                        {"column": "gender", "selected_values": ["Female"], "available_values": ["Male", "Female"]},
                        {
                            "column": "sococc",
                            "selected_values": ["Employed"],
                            "available_values": ["Employed", "Retired", "Student", "Unemployed", "Unknown"],
                        },
                    ],
                    "recent_refinement": {
                        "changes": [
                            {
                                "column": "gender",
                                "previous_values": ["Male"],
                                "selected_values": ["Female"],
                                "added_values": ["Female"],
                                "removed_values": ["Male"],
                            }
                        ]
                    },
                },
                "number of working adults",
            )

        self.assertIn("Current committed query context", captured["user_prompt"])
        self.assertIn("Current committed categorical filters", captured["user_prompt"])
        self.assertIn("Recent categorical refinement history", captured["user_prompt"])
        self.assertIn(
            "gender: previous = Male; current = Female; added = Female; removed = Male",
            captured["user_prompt"],
        )

    def test_result_refinement_resolver_preserves_grouping_clarification_request(self) -> None:
        captured: dict[str, str] = {}

        def completion(**kwargs):
            captured["system_prompt"] = kwargs["messages"][0]["content"]
            captured["user_prompt"] = kwargs["messages"][1]["content"]
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=(
                                '{"resolution_type":"needs_clarification","target_column":"",'
                                '"selected_values":[],"refinement_request":"use other categories instead"}'
                            )
                        )
                    )
                ]
            )

        resolver = build_llm_result_refinement_resolver("test-model")

        with patch.dict(sys.modules, {"litellm": SimpleNamespace(completion=completion)}):
            resolution = resolver(
                {
                    "question": "number of males for each diagnosed cancer type",
                    "sql": (
                        "SELECT parent_category, COUNT(*) AS matching_count "
                        "FROM filtered_dataset WHERE gender = 'Male' GROUP BY parent_category"
                    ),
                    "categorical_filters": [
                        {"column": "gender", "selected_values": ["Male"], "available_values": ["Male", "Female"]},
                    ],
                    "is_grouped": True,
                    "group_columns": ["parent_category"],
                },
                "use other categories instead",
            )

        self.assertEqual(
            resolution,
            {
                "resolution_type": "needs_clarification",
                "target_column": "",
                "selected_values": [],
                "refinement_request": "use other categories instead",
            },
        )
        self.assertIn("change the grouping dimension", captured["system_prompt"])
        self.assertIn("Grouping columns from the current committed query", captured["user_prompt"])
        self.assertIn("parent_category", captured["user_prompt"])

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

    def test_result_refinement_resolver_marks_standalone_dataset_question_as_topic_change(self) -> None:
        captured: dict[str, str] = {}

        def completion(**kwargs):
            captured["system_prompt"] = kwargs["messages"][0]["content"]
            captured["user_prompt"] = kwargs["messages"][1]["content"]
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content='{"resolution_type":"topic_change","target_column":"","selected_values":[],"refinement_request":""}'
                        )
                    )
                ]
            )

        resolver = build_llm_result_refinement_resolver("test-model")

        with patch.dict(sys.modules, {"litellm": SimpleNamespace(completion=completion)}):
            resolution = resolver(
                {
                    "question": "total non-working",
                    "sql": "SELECT COUNT(*) AS matching_count FROM filtered_dataset WHERE sococc = 'Unemployed'",
                    "categorical_filters": [
                        {
                            "column": "sococc",
                            "selected_values": ["Unemployed"],
                            "available_values": ["Employed", "Retired", "Student", "Unemployed", "Unknown"],
                        }
                    ],
                },
                "give me the avg age of females",
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
        self.assertIn("fresh dataset question", captured["system_prompt"])
        self.assertIn("give me the avg age of females", captured["user_prompt"])

    def test_result_refinement_resolver_preserves_mixed_selected_values_and_refinement_request(self) -> None:
        captured: dict[str, str] = {}

        def completion(**kwargs):
            captured["system_prompt"] = kwargs["messages"][0]["content"]
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=(
                                '{"resolution_type":"refine_query","target_column":"gender",'
                                '"selected_values":["Female"],"refinement_request":"average age only below 34"}'
                            )
                        )
                    )
                ]
            )

        resolver = build_llm_result_refinement_resolver("test-model")

        with patch.dict(sys.modules, {"litellm": SimpleNamespace(completion=completion)}):
            resolution = resolver(
                {
                    "question": "how many males work",
                    "sql": "SELECT COUNT(*) AS matching_count FROM filtered_dataset WHERE gender = 'Male' AND sococc = 'Employed'",
                    "categorical_filters": [
                        {"column": "gender", "selected_values": ["Male"], "available_values": ["Male", "Female"]},
                        {
                            "column": "sococc",
                            "selected_values": ["Employed"],
                            "available_values": ["Employed", "Retired", "Student", "Unemployed", "Unknown"],
                        },
                    ],
                },
                "make it females and give me the average age only below 34",
            )

        self.assertEqual(
            resolution,
            {
                "resolution_type": "refine_query",
                "target_column": "gender",
                "selected_values": ["Female"],
                "refinement_request": "average age only below 34",
            },
        )
        self.assertIn("selected_values may coexist with refinement_request", captured["system_prompt"])

    def test_fresh_topic_relevance_router_includes_current_topic_context(self) -> None:
        captured: dict[str, str] = {}

        def completion(**kwargs):
            captured["system_prompt"] = kwargs["messages"][0]["content"]
            captured["user_prompt"] = kwargs["messages"][1]["content"]
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content='{"resolution_type":"dataset_question"}'
                        )
                    )
                ]
            )

        router = build_llm_fresh_topic_relevance_router(
            "test-model",
            "- gender (TEXT): field label = Sex of patient; categorical values = Female, Male",
        )

        with patch.dict(sys.modules, {"litellm": SimpleNamespace(completion=completion)}):
            resolution = router(
                "how many females are there who are below 45 and smoke and ddrink",
                "number of males for each diagnosed cancer type",
            )

        self.assertEqual(resolution, {"resolution_type": "dataset_question"})
        self.assertIn("dataset_question|meta_or_conversational|out_of_scope", captured["system_prompt"])
        self.assertIn("Current committed dataset topic", captured["user_prompt"])
        self.assertIn("number of males for each diagnosed cancer type", captured["user_prompt"])
        self.assertIn("Latest user reply already classified as a topic change", captured["user_prompt"])
        self.assertIn(
            "how many females are there who are below 45 and smoke and ddrink",
            captured["user_prompt"],
        )

    def test_fresh_topic_relevance_router_fails_open_on_invalid_output(self) -> None:
        def completion(**kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="not json"))]
            )

        router = build_llm_fresh_topic_relevance_router(
            "test-model",
            "- gender (TEXT): field label = Sex of patient; categorical values = Female, Male",
        )

        with patch.dict(sys.modules, {"litellm": SimpleNamespace(completion=completion)}):
            resolution = router("how many females drink", "how many males work")

        self.assertEqual(resolution, {"resolution_type": "dataset_question"})

    def test_schema_grounding_resolver_includes_schema_context(self) -> None:
        captured_system_prompts: list[str] = []
        captured_user_prompts: list[str] = []

        def completion(**kwargs):
            captured_system_prompts.append(kwargs["messages"][0]["content"])
            captured_user_prompts.append(kwargs["messages"][1]["content"])
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=(
                                '{"resolution_type":"needs_clarification",'
                                '"grounded_filters":[{"column":"gender","selected_values":["Female"]}],'
                                '"candidate_columns":["socalc","socsmk"]}'
                            )
                        )
                    )
                ]
            )

        resolver = build_llm_schema_grounding_resolver("test-model")

        with patch.dict(sys.modules, {"litellm": SimpleNamespace(completion=completion)}):
            resolution = resolver(
                "how many females drink",
                "- gender (TEXT): categorical values = Female, Male\n- socalc (TEXT): categorical values = Never, Occasionally, Regularly, Unknown\n- socsmk (TEXT): categorical values = No, Unknown, Yes",
                [
                    {
                        "column": "gender",
                        "candidate_value": "Female",
                        "field_label": "Gender",
                        "matched_user_phrase": "females",
                        "confidence": 0.95,
                        "evidence_sources": ["candidate_value"],
                    },
                    {
                        "column": "gender",
                        "candidate_value": "Male",
                        "field_label": "Gender",
                        "matched_user_phrase": "",
                        "confidence": 0.0,
                        "evidence_sources": ["candidate_value"],
                    },
                ],
                ["gender", "socalc", "socsmk"],
            )

        self.assertEqual(
            resolution,
            {
                "resolution_type": "needs_clarification",
                "grounded_filters": {"gender": ["Female"]},
                "candidate_columns": ["socalc", "socsmk"],
                "resolution_items": [
                    {
                        "matched_phrase": "females",
                        "ambiguity_kind": "grounded_filter",
                        "selected_column": "gender",
                        "selected_values": ["Female"],
                        "candidate_columns": [],
                        "candidate_values": [],
                    },
                    {
                        "matched_phrase": "how many females drink",
                        "ambiguity_kind": "field_ambiguity",
                        "selected_column": "",
                        "selected_values": [],
                        "candidate_columns": ["socalc", "socsmk"],
                        "candidate_values": [],
                    },
                ],
            },
        )
        self.assertIn("proceed|needs_clarification", captured_system_prompts[0])
        self.assertIn("grounded_filters", captured_system_prompts[0])
        self.assertIn("Schema column identifiers you may return", captured_user_prompts[0])
        self.assertIn("Categorical grounding candidates", captured_user_prompts[0])
        self.assertIn("socalc", captured_user_prompts[0])
        self.assertIn("socsmk", captured_user_prompts[0])
        self.assertIn("Latest user request", captured_user_prompts[0])
        self.assertIn("how many females drink", captured_user_prompts[0])
        self.assertIn("strict structured schema-grounding planner", captured_system_prompts[1])

    def test_schema_grounding_resolver_fails_open_on_invalid_output(self) -> None:
        def completion(**kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="not json"))]
            )

        resolver = build_llm_schema_grounding_resolver("test-model")

        with patch.dict(sys.modules, {"litellm": SimpleNamespace(completion=completion)}):
            resolution = resolver(
                "how many females drink",
                "- socalc (TEXT): categorical values = Never, Occasionally, Regularly, Unknown",
                [
                    {
                        "column": "socalc",
                        "candidate_value": "Never",
                        "field_label": "Alcohol consumption",
                        "matched_user_phrase": "",
                        "confidence": 0.0,
                        "evidence_sources": ["candidate_value"],
                    }
                ],
                ["socalc", "socsmk"],
            )

        self.assertEqual(
            resolution,
            {
                "resolution_type": "proceed",
                "grounded_filters": {},
                "candidate_columns": [],
                "resolution_items": [],
            },
        )

    def test_schema_grounding_resolver_proceeds_after_review_when_one_field_is_clearly_best(self) -> None:
        captured_system_prompts: list[str] = []
        captured_user_prompts: list[str] = []
        responses = iter(
            [
                '{"resolution_type":"proceed","grounded_filters":[{"column":"gender","selected_values":["Female"]}],"candidate_columns":[]}',
                '{"resolution_type":"needs_clarification","candidate_columns":["socalc","socsmk"]}',
                '{"resolution_type":"proceed","selected_column":"socalc"}',
                '{"items":[{"matched_phrase":"women","ambiguity_kind":"grounded_filter","selected_column":"gender","selected_values":["Female"],"candidate_columns":[],"candidate_values":[]},{"matched_phrase":"drink","ambiguity_kind":"value_ambiguity","selected_column":"socalc","selected_values":[],"candidate_columns":[],"candidate_values":[]}]}',
            ]
        )

        def completion(**kwargs):
            captured_system_prompts.append(kwargs["messages"][0]["content"])
            captured_user_prompts.append(kwargs["messages"][1]["content"])
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=next(responses))
                    )
                ]
            )

        resolver = build_llm_schema_grounding_resolver("test-model")

        with patch.dict(sys.modules, {"litellm": SimpleNamespace(completion=completion)}):
            resolution = resolver(
                "how many women drink",
                "- gender (TEXT): field label = Sex of patient; categorical values = Female, Male\n"
                "- socalc (TEXT): field label = Alcohol consumption status; categorical values = Never, Occasionally, Regularly, Unknown\n"
                "- socsmk (TEXT): field label = Smoking status; categorical values = No, Unknown, Yes",
                [
                    {
                        "column": "gender",
                        "candidate_value": "Female",
                        "field_label": "Sex of patient",
                        "matched_user_phrase": "",
                        "confidence": 0.0,
                        "evidence_sources": ["candidate_value"],
                    },
                    {
                        "column": "gender",
                        "candidate_value": "Male",
                        "field_label": "Sex of patient",
                        "matched_user_phrase": "",
                        "confidence": 0.0,
                        "evidence_sources": ["candidate_value"],
                    },
                    {
                        "column": "socalc",
                        "candidate_value": "Occasionally",
                        "field_label": "Alcohol consumption status",
                        "matched_user_phrase": "drink",
                        "confidence": 0.6,
                        "evidence_sources": ["source_header"],
                    },
                    {
                        "column": "socsmk",
                        "candidate_value": "Yes",
                        "field_label": "Smoking status",
                        "matched_user_phrase": "smoke",
                        "confidence": 0.6,
                        "evidence_sources": ["source_header"],
                    },
                ],
                ["gender", "socalc", "socsmk"],
            )

        self.assertEqual(resolution["resolution_type"], "proceed")
        self.assertEqual(resolution["grounded_filters"], {"gender": ["Female"]})
        self.assertEqual(resolution["candidate_columns"], [])
        self.assertEqual(resolution["resolved_columns"], ["socalc"])
        self.assertTrue(
            any(
                item.get("ambiguity_kind") == "grounded_filter"
                and item.get("selected_column") == "gender"
                and item.get("selected_values") == ["Female"]
                for item in resolution.get("resolution_items") or []
            )
        )
        self.assertTrue(
            all(
                item.get("ambiguity_kind") != "field_ambiguity"
                for item in resolution.get("resolution_items") or []
            )
        )
        self.assertEqual(len(captured_system_prompts), 4)
        self.assertIn("Do not return proceed merely because one part of the request was grounded", captured_system_prompts[0])
        self.assertIn("strict unresolved-request reviewer", captured_system_prompts[1])
        self.assertIn("Already grounded filters:", captured_user_prompts[1])
        self.assertIn("- gender = Female", captured_user_prompts[1])
        self.assertIn("strict final field-resolution judge", captured_system_prompts[2])
        self.assertIn("Candidate columns to judge:", captured_user_prompts[2])
        self.assertIn("strict structured schema-grounding planner", captured_system_prompts[3])

    def test_schema_grounding_resolver_reduces_overbroad_review_candidates(self) -> None:
        captured_system_prompts: list[str] = []
        responses = iter(
            [
                '{"resolution_type":"proceed","grounded_filters":[{"column":"gender","selected_values":["Female"]}],"candidate_columns":[]}',
                '{"resolution_type":"needs_clarification","candidate_columns":["socsmk","socalc","sococc"]}',
                '{"candidate_columns":["socalc","socsmk"]}',
                '{"resolution_type":"needs_clarification","selected_column":""}',
                '{"items":[{"matched_phrase":"women","ambiguity_kind":"grounded_filter","selected_column":"gender","selected_values":["Female"],"candidate_columns":[],"candidate_values":[]},{"matched_phrase":"drink","ambiguity_kind":"field_ambiguity","selected_column":"","selected_values":[],"candidate_columns":["socalc","socsmk"],"candidate_values":[]}]}',
            ]
        )

        def completion(**kwargs):
            captured_system_prompts.append(kwargs["messages"][0]["content"])
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=next(responses))
                    )
                ]
            )

        resolver = build_llm_schema_grounding_resolver("test-model")

        with patch.dict(sys.modules, {"litellm": SimpleNamespace(completion=completion)}):
            resolution = resolver(
                "how many women drink",
                "- gender (TEXT): field label = Sex of patient; categorical values = Female, Male\n"
                "- socalc (TEXT): field label = Alcohol consumption status; categorical values = Never, Occasionally, Regularly, Unknown\n"
                "- socsmk (TEXT): field label = Smoking status; categorical values = No, Unknown, Yes\n"
                "- sococc (TEXT): field label = Occupation status; categorical values = Employed, Retired, Student, Unemployed, Unknown",
                [
                    {"column": "gender", "candidate_value": "Female", "field_label": "Sex of patient", "matched_user_phrase": "", "confidence": 0.0, "evidence_sources": ["candidate_value"]},
                    {"column": "gender", "candidate_value": "Male", "field_label": "Sex of patient", "matched_user_phrase": "", "confidence": 0.0, "evidence_sources": ["candidate_value"]},
                    {"column": "socalc", "candidate_value": "Occasionally", "field_label": "Alcohol consumption status", "matched_user_phrase": "drink", "confidence": 0.6, "evidence_sources": ["source_header"]},
                    {"column": "socsmk", "candidate_value": "Yes", "field_label": "Smoking status", "matched_user_phrase": "smoke", "confidence": 0.6, "evidence_sources": ["source_header"]},
                    {"column": "sococc", "candidate_value": "Employed", "field_label": "Occupation status", "matched_user_phrase": "job", "confidence": 0.6, "evidence_sources": ["source_header"]},
                ],
                ["gender", "socalc", "socsmk", "sococc"],
            )

        self.assertEqual(
            resolution,
            {
                "resolution_type": "needs_clarification",
                "grounded_filters": {"gender": ["Female"]},
                "candidate_columns": ["socalc", "socsmk"],
                "resolution_items": [
                    {
                        "matched_phrase": "women",
                        "ambiguity_kind": "grounded_filter",
                        "selected_column": "gender",
                        "selected_values": ["Female"],
                        "candidate_columns": [],
                        "candidate_values": [],
                    },
                    {
                        "matched_phrase": "drink",
                        "ambiguity_kind": "field_ambiguity",
                        "selected_column": "",
                        "selected_values": [],
                        "candidate_columns": ["socalc", "socsmk"],
                        "candidate_values": [],
                    },
                ],
            },
        )
        self.assertEqual(len(captured_system_prompts), 5)
        self.assertIn("strict clarification-set reducer", captured_system_prompts[2])
        self.assertIn("strict final field-resolution judge", captured_system_prompts[3])
        self.assertIn("strict structured schema-grounding planner", captured_system_prompts[4])

    def test_schema_grounding_resolver_filters_zero_evidence_columns_from_second_pass_review(self) -> None:
        captured_user_prompts: list[str] = []
        responses = iter(
            [
                '{"resolution_type":"proceed","grounded_filters":[{"column":"gender","selected_values":["Female"]}],"candidate_columns":[]}',
                '{"resolution_type":"needs_clarification","candidate_columns":["socalc","socsmk","bsymp","bsymp1"]}',
                '{"resolution_type":"needs_clarification","selected_column":""}',
                '{"items":[{"matched_phrase":"women","ambiguity_kind":"grounded_filter","selected_column":"gender","selected_values":["Female"],"candidate_columns":[],"candidate_values":[]},{"matched_phrase":"drink or smoke","ambiguity_kind":"field_ambiguity","selected_column":"","selected_values":[],"candidate_columns":["socalc","socsmk"],"candidate_values":[]}]}',
            ]
        )

        def completion(**kwargs):
            captured_user_prompts.append(kwargs["messages"][1]["content"])
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=next(responses))
                    )
                ]
            )

        resolver = build_llm_schema_grounding_resolver("test-model")

        with patch.dict(sys.modules, {"litellm": SimpleNamespace(completion=completion)}):
            resolution = resolver(
                "how many women drink and smoke",
                "- gender (TEXT): field label = Sex of patient; categorical values = Female, Male\n"
                "- socalc (TEXT): field label = Alcohol consumption status; categorical values = Never, Occasionally, Regularly, Unknown\n"
                "- socsmk (TEXT): field label = Smoking status; categorical values = No, Unknown, Yes\n"
                "- bsymp (TEXT): field label = B symptom status; categorical values = No, Unknown, Yes\n"
                "- bsymp1 (TEXT): field label = B symptom severity; categorical values = Mild, Moderate, Severe, Unknown",
                [
                    {"column": "gender", "candidate_value": "Female", "field_label": "Sex of patient", "matched_user_phrase": "", "confidence": 0.0, "evidence_sources": []},
                    {"column": "gender", "candidate_value": "Male", "field_label": "Sex of patient", "matched_user_phrase": "", "confidence": 0.0, "evidence_sources": []},
                    {"column": "socalc", "candidate_value": "Occasionally", "field_label": "Alcohol consumption status", "matched_user_phrase": "drink", "confidence": 0.6, "evidence_sources": ["source_header"]},
                    {"column": "socsmk", "candidate_value": "Yes", "field_label": "Smoking status", "matched_user_phrase": "smoke", "confidence": 0.6, "evidence_sources": ["source_header"]},
                    {"column": "bsymp", "candidate_value": "Yes", "field_label": "B symptom status", "matched_user_phrase": "", "confidence": 0.0, "evidence_sources": []},
                    {"column": "bsymp1", "candidate_value": "Moderate", "field_label": "B symptom severity", "matched_user_phrase": "", "confidence": 0.0, "evidence_sources": []},
                ],
                ["gender", "socalc", "socsmk", "bsymp", "bsymp1"],
            )

        self.assertEqual(
            resolution,
            {
                "resolution_type": "needs_clarification",
                "grounded_filters": {"gender": ["Female"]},
                "candidate_columns": ["socalc", "socsmk"],
                "resolution_items": [
                    {
                        "matched_phrase": "women",
                        "ambiguity_kind": "grounded_filter",
                        "selected_column": "gender",
                        "selected_values": ["Female"],
                        "candidate_columns": [],
                        "candidate_values": [],
                    },
                    {
                        "matched_phrase": "drink or smoke",
                        "ambiguity_kind": "field_ambiguity",
                        "selected_column": "",
                        "selected_values": [],
                        "candidate_columns": ["socalc", "socsmk"],
                        "candidate_values": [],
                    },
                ],
            },
        )
        self.assertEqual(len(captured_user_prompts), 4)
        self.assertIn("Remaining schema column identifiers you may return:", captured_user_prompts[1])
        self.assertIn("- socalc", captured_user_prompts[1])
        self.assertIn("- socsmk", captured_user_prompts[1])
        self.assertNotIn("- bsymp", captured_user_prompts[1])
        self.assertNotIn("- bsymp1", captured_user_prompts[1])
        self.assertIn("- socalc (TEXT)", captured_user_prompts[1])
        self.assertIn("- socsmk (TEXT)", captured_user_prompts[1])
        self.assertNotIn("- bsymp (TEXT)", captured_user_prompts[1])
        self.assertNotIn("- bsymp1 (TEXT)", captured_user_prompts[1])

    def test_schema_grounding_resolver_proceeds_when_no_evidence_backed_columns_remain(self) -> None:
        captured_user_prompts: list[str] = []
        responses = iter(
            [
                '{"resolution_type":"proceed","grounded_filters":[{"column":"socalc","selected_values":["Binge Drinking"]}],"candidate_columns":[]}',
                '{"items":[{"matched_phrase":"drink alot","ambiguity_kind":"grounded_filter","selected_column":"socalc","selected_values":["Binge Drinking"],"candidate_columns":[],"candidate_values":[]}]}',
            ]
        )

        def completion(**kwargs):
            captured_user_prompts.append(kwargs["messages"][1]["content"])
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=next(responses))
                    )
                ]
            )

        resolver = build_llm_schema_grounding_resolver("test-model")

        with patch.dict(sys.modules, {"litellm": SimpleNamespace(completion=completion)}):
            resolution = resolver(
                "how many drink alot",
                "- socalc (TEXT): field label = Alcohol intake; categorical values = Binge Drinking, Ex-Binge Drinker, Ex-Social Drinker, Never, Social Drinking, Unknown\n"
                "- bsymp (TEXT): field label = B symptoms; categorical values = No, No response, Yes\n"
                "- bsymp1 (TEXT): field label = bsymp1; categorical values = No, No response, Yes\n"
                "- bsymp2 (TEXT): field label = bsymp2; categorical values = No, No response, Yes\n"
                "- bsymp3 (TEXT): field label = bsymp3; categorical values = No, No response, Yes",
                [
                    {"column": "socalc", "candidate_value": "Binge Drinking", "field_label": "Alcohol intake", "matched_user_phrase": "drink alot", "confidence": 0.95, "evidence_sources": ["candidate_value"]},
                    {"column": "socalc", "candidate_value": "Social Drinking", "field_label": "Alcohol intake", "matched_user_phrase": "drink", "confidence": 0.6, "evidence_sources": ["source_header"]},
                    {"column": "bsymp", "candidate_value": "Yes", "field_label": "B symptoms", "matched_user_phrase": "", "confidence": 0.0, "evidence_sources": []},
                    {"column": "bsymp1", "candidate_value": "Yes", "field_label": "bsymp1", "matched_user_phrase": "", "confidence": 0.0, "evidence_sources": []},
                    {"column": "bsymp2", "candidate_value": "Yes", "field_label": "bsymp2", "matched_user_phrase": "", "confidence": 0.0, "evidence_sources": []},
                    {"column": "bsymp3", "candidate_value": "Yes", "field_label": "bsymp3", "matched_user_phrase": "", "confidence": 0.0, "evidence_sources": []},
                ],
                ["socalc", "bsymp", "bsymp1", "bsymp2", "bsymp3"],
            )

        self.assertEqual(
            resolution,
            {
                "resolution_type": "proceed",
                "grounded_filters": {"socalc": ["Binge Drinking"]},
                "candidate_columns": [],
                "resolution_items": [
                    {
                        "matched_phrase": "drink alot",
                        "ambiguity_kind": "grounded_filter",
                        "selected_column": "socalc",
                        "selected_values": ["Binge Drinking"],
                        "candidate_columns": [],
                        "candidate_values": [],
                    },
                ],
            },
        )
        self.assertEqual(len(captured_user_prompts), 2)

    def test_schema_grounding_resolver_reviews_multi_value_filter_for_drink_excluding_unknown(self) -> None:
        captured_system_prompts: list[str] = []
        responses = iter(
            [
                '{"resolution_type":"proceed","grounded_filters":[{"column":"gender","selected_values":["Female"]},{"column":"socalc","selected_values":["Never","Occasionally","Regularly"]}],"candidate_columns":[]}',
                '{"selected_values":["Never","Occasionally","Regularly"]}',
                '{"resolution_type":"proceed","candidate_columns":[]}',
                '{"items":[{"matched_phrase":"women","ambiguity_kind":"grounded_filter","selected_column":"gender","selected_values":["Female"],"candidate_columns":[],"candidate_values":[]},{"matched_phrase":"drink","ambiguity_kind":"value_ambiguity","selected_column":"socalc","selected_values":[],"candidate_columns":[],"candidate_values":[]}]}',
            ]
        )

        def completion(**kwargs):
            captured_system_prompts.append(kwargs["messages"][0]["content"])
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=next(responses))
                    )
                ]
            )

        resolver = build_llm_schema_grounding_resolver("test-model")

        with patch.dict(sys.modules, {"litellm": SimpleNamespace(completion=completion)}):
            resolution = resolver(
                "how many women drink, exclude unk",
                "- gender (TEXT): field label = Sex of patient; categorical values = Female, Male\n"
                "- socalc (TEXT): field label = Alcohol consumption status; categorical values = Never, Occasionally, Regularly, Unknown\n"
                "- socsmk (TEXT): field label = Smoking status; categorical values = No, Unknown, Yes",
                [
                    {
                        "column": "gender",
                        "candidate_value": "Female",
                        "field_label": "Sex of patient",
                        "matched_user_phrase": "",
                        "confidence": 0.0,
                        "evidence_sources": ["candidate_value"],
                    },
                    {
                        "column": "gender",
                        "candidate_value": "Male",
                        "field_label": "Sex of patient",
                        "matched_user_phrase": "",
                        "confidence": 0.0,
                        "evidence_sources": ["candidate_value"],
                    },
                    {
                        "column": "socalc",
                        "candidate_value": "Never",
                        "field_label": "Alcohol consumption status",
                        "matched_user_phrase": "",
                        "confidence": 0.0,
                        "evidence_sources": ["candidate_value"],
                    },
                    {
                        "column": "socalc",
                        "candidate_value": "Occasionally",
                        "field_label": "Alcohol consumption status",
                        "matched_user_phrase": "",
                        "confidence": 0.0,
                        "evidence_sources": ["candidate_value"],
                    },
                    {
                        "column": "socalc",
                        "candidate_value": "Regularly",
                        "field_label": "Alcohol consumption status",
                        "matched_user_phrase": "",
                        "confidence": 0.0,
                        "evidence_sources": ["candidate_value"],
                    },
                    {
                        "column": "socalc",
                        "candidate_value": "Unknown",
                        "field_label": "Alcohol consumption status",
                        "matched_user_phrase": "",
                        "confidence": 0.0,
                        "evidence_sources": ["candidate_value"],
                    },
                ],
                ["gender", "socalc", "socsmk"],
            )

        self.assertEqual(resolution["resolution_type"], "proceed")
        self.assertEqual(resolution["grounded_filters"], {"gender": ["Female"]})
        self.assertEqual(resolution["candidate_columns"], [])
        self.assertEqual(resolution["resolved_columns"], ["socalc"])
        self.assertTrue(
            any(
                item.get("ambiguity_kind") == "grounded_filter"
                and item.get("selected_column") == "gender"
                and item.get("selected_values") == ["Female"]
                for item in resolution.get("resolution_items") or []
            )
        )
        self.assertTrue(
            any(
                item.get("ambiguity_kind") == "value_ambiguity"
                and item.get("selected_column") == "socalc"
                and item.get("candidate_values") == ["Never", "Occasionally", "Regularly", "Unknown"]
                for item in resolution.get("resolution_items") or []
            )
        )
        self.assertEqual(len(captured_system_prompts), 3)
        self.assertIn("strict categorical-value grounding reviewer", captured_system_prompts[1])
        self.assertIn("strict structured schema-grounding planner", captured_system_prompts[2])

    def test_collect_grounding_match_evidence_prefers_value_match_over_generic_label_tokens(self) -> None:
        matched_phrase, confidence, evidence_sources = _collect_grounding_match_evidence(
            "number of males for each diagnosed cancer type",
            [
                ("request_field_label", "Sex of patient"),
                ("candidate_value", "Male"),
            ],
        )

        self.assertEqual(matched_phrase, "males")
        self.assertEqual(confidence, 0.95)
        self.assertEqual(evidence_sources, ["candidate_value"])


class SQLiteMissingGroupRewriteTestCase(unittest.TestCase):
    def setUp(self) -> None:
        temp_root = REPO_ROOT / ".tmp_test_runs"
        temp_root.mkdir(exist_ok=True)
        self.db_path = temp_root / f"{uuid.uuid4().hex}.sqlite"
        create_missing_group_fixture_database(self.db_path)

    def tearDown(self) -> None:
        if self.db_path.exists():
            self.db_path.unlink()

    def test_execute_sqlite_query_groups_null_and_blank_values_under_null_label(self) -> None:
        result = execute_sqlite_query(
            self.db_path,
            "SELECT sex, COUNT(*) AS matching_count FROM patients GROUP BY sex ORDER BY sex",
        )

        self.assertEqual(result["status"], "success")
        self.assertIn("'Null'", result.get("display_sql", ""))
        self.assertEqual(
            {row["sex"]: row["matching_count"] for row in result["rows"]},
            {
                "Null": 3,
                "female": 2,
                "male": 1,
            },
        )

    def test_execute_sqlite_query_rewrites_case_buckets_for_missing_source_values(self) -> None:
        result = execute_sqlite_query(
            self.db_path,
            """
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
        )

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
