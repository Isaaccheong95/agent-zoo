from __future__ import annotations

import io
import os
import sys
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

from google.genai import types

from agent_zoo.sql_agent.callbacks import (
    SQL_ACTIVE_QUERY_TOPIC_STATE_KEY,
    SQL_FRESH_TOPIC_CLARIFICATION_STATE_KEY,
    SQL_INTERNAL_QUERY_RESULT_STATE_KEY,
    SQL_INTERNAL_RESULT_REF_STATE_KEY,
    SQL_LAST_USER_TEXT_STATE_KEY,
    SQL_PUBLIC_RESULT_RENDERED_STATE_KEY,
    SQL_PUBLIC_RESULT_STATE_KEY,
    SQL_REFINEMENT_SOURCE_QUERY_FRAME_STATE_KEY,
    build_combined_before_model_callback,
    build_finalize_after_query_before_model_callback,
    build_format_final_agent_response_callback,
    build_normalize_clarification_after_model_callback,
    build_remember_query_result_callback,
)
from agent_zoo.sql_agent.config import SQLAgentSettings, load_settings
from agent_zoo.sql_agent.formatting import (
    build_sql_result_view_model,
    render_sql_result_view_model,
)
from agent_zoo.sql_agent.runtime import _print_debug_event
from agent_zoo.working_memory import get_agent_working_memory_value

from tests._fixtures import (
    REPO_ROOT,
    get_sql_current_query_frame,
    get_sql_pending_clarification,
    get_sql_working_memory,
    make_query_result,
    set_sql_current_query_frame,
    set_sql_pending_clarification,
)


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

    def test_remember_query_result_prefers_tool_response_display_sql(self) -> None:
        rewritten_sql = (
            "SELECT CASE WHEN NULLIF(TRIM(CAST((sex) AS TEXT)), '') IS NULL THEN 'Unknown / Null' "
            "ELSE (sex) END AS sex, COUNT(*) AS matching_count FROM people "
            "GROUP BY CASE WHEN NULLIF(TRIM(CAST((sex) AS TEXT)), '') IS NULL THEN 'Unknown / Null' "
            "ELSE (sex) END ORDER BY sex"
        )
        state = self._invoke_after_tool_with_state(
            self._settings(minimum_aggregate_count=1),
            make_query_result(
                [
                    {"sex": "Unknown / Null", "matching_count": 3},
                    {"sex": "female", "matching_count": 2},
                    {"sex": "male", "matching_count": 1},
                ],
                columns=["sex", "matching_count"],
                row_count=3,
                sql=rewritten_sql,
                display_sql=rewritten_sql,
            ),
            state={},
            args={"sql": "SELECT sex, COUNT(*) AS matching_count FROM people GROUP BY sex ORDER BY sex"},
        )

        public_result = state[SQL_PUBLIC_RESULT_STATE_KEY]
        self.assertEqual(public_result["display_sql"], rewritten_sql)
        self.assertIn("Unknown / Null", public_result["display_sql"])

    def test_remember_query_result_uses_rewritten_display_sql_for_categorical_filters(self) -> None:
        with patch(
            "agent_zoo.sql_agent.callbacks.get_schema_summary",
            return_value={
                "status": "success",
                "categorical_value_guidance": [
                    {"column": "gender", "values": ["Female", "Male"]},
                    {"column": "socsmk", "values": ["No", "Unknown", "Yes"]},
                    {"column": "socalc", "values": ["Never", "Occasionally", "Regularly", "Unknown"]},
                ],
            },
        ):
            callback = build_remember_query_result_callback(self._settings(minimum_aggregate_count=1))

        tool = SimpleNamespace(name="execute_sqlite_read_only")
        tool_context = SimpleNamespace(
            state={
                SQL_ACTIVE_QUERY_TOPIC_STATE_KEY: "how many ladeis smoke and drink alcohol",
            }
        )
        rewritten_sql = (
            "SELECT COUNT(*) AS matching_count FROM filtered_dataset "
            "WHERE gender = 'Female' AND socsmk = 'Yes' "
            "AND socalc IN ('Occasionally', 'Regularly', 'Unknown')"
        )

        callback(
            tool,
            {
                "sql": "SELECT COUNT(*) AS matching_count FROM filtered_dataset WHERE gender = 'Female' AND socsmk = 'Yes' AND socalc != 'Never'",
                "is_final": True,
            },
            tool_context,
            make_query_result(
                [{"matching_count": 32}],
                columns=["matching_count"],
                sql=rewritten_sql,
                display_sql=rewritten_sql,
            ),
        )

        public_result = tool_context.state[SQL_PUBLIC_RESULT_STATE_KEY]
        self.assertEqual(public_result["display_sql"], rewritten_sql)

        query_frame = get_sql_current_query_frame(tool_context.state)
        self.assertIsNotNone(query_frame)
        self.assertEqual(
            query_frame.get("categorical_filters"),
            [
                {
                    "column": "gender",
                    "selected_values": ["Female"],
                    "available_values": ["Female", "Male"],
                },
                {
                    "column": "socsmk",
                    "selected_values": ["Yes"],
                    "available_values": ["No", "Unknown", "Yes"],
                },
                {
                    "column": "socalc",
                    "selected_values": ["Occasionally", "Regularly", "Unknown"],
                    "available_values": ["Never", "Occasionally", "Regularly", "Unknown"],
                },
            ],
        )
        self.assertNotIn("comparison_filters", query_frame)

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

        query_frame = get_sql_current_query_frame(tool_context.state)
        self.assertIsNotNone(query_frame)
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

    def test_remember_query_result_stores_current_query_frame_in_shared_memory(self) -> None:
        with patch(
            "agent_zoo.sql_agent.callbacks.get_schema_summary",
            return_value={
                "status": "success",
                "categorical_value_guidance": [
                    {"column": "gender", "values": ["Male", "Female"]},
                    {
                        "column": "sococc",
                        "values": ["Employed", "Retired", "Student", "Unemployed", "Unknown"],
                    },
                ],
            },
        ):
            callback = build_remember_query_result_callback(self._settings(minimum_aggregate_count=1))

        tool = SimpleNamespace(name="execute_sqlite_read_only")
        previous_query_frame = {
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
        }
        tool_context = SimpleNamespace(
            state={
                SQL_ACTIVE_QUERY_TOPIC_STATE_KEY: "females?",
                SQL_REFINEMENT_SOURCE_QUERY_FRAME_STATE_KEY: previous_query_frame,
            }
        )
        set_sql_current_query_frame(tool_context.state, previous_query_frame)

        callback(
            tool,
            {
                "sql": "SELECT COUNT(*) AS matching_count FROM filtered_dataset WHERE gender = 'Female' AND sococc = 'Employed'",
                "is_final": True,
            },
            tool_context,
            make_query_result(
                [{"matching_count": 12}],
                columns=["matching_count"],
                sql="SELECT COUNT(*) AS matching_count FROM filtered_dataset WHERE gender = 'Female' AND sococc = 'Employed'",
            ),
        )

        working_memory = get_sql_working_memory(tool_context.state)
        self.assertIn("current_query_frame", working_memory)
        self.assertEqual(
            get_agent_working_memory_value(tool_context.state, "sql_agent", "current_query_frame"),
            get_sql_current_query_frame(tool_context.state),
        )
        self.assertEqual(
            get_sql_current_query_frame(tool_context.state)["question"],
            "how many females work",
        )
        self.assertIn(
            "Current committed dataset question/topic: how many females work",
            get_sql_current_query_frame(tool_context.state)["topic_context"],
        )
        self.assertEqual(
            get_sql_current_query_frame(tool_context.state)["recent_refinement"]["changes"],
            [
                {
                    "column": "gender",
                    "previous_values": ["Male"],
                    "selected_values": ["Female"],
                    "available_values": ["Male", "Female"],
                    "added_values": ["Female"],
                    "removed_values": ["Male"],
                }
            ],
        )

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

        query_frame = get_sql_current_query_frame(tool_context.state)
        self.assertIsNotNone(query_frame)
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

        query_frame = get_sql_current_query_frame(tool_context.state)
        self.assertIsNotNone(query_frame)
        self.assertEqual(
            query_frame["comparison_filters"],
            [{"column": "age", "operator": "<", "value": "45"}],
        )

        content = build_format_final_agent_response_callback()(SimpleNamespace(state=tool_context.state))
        self.assertIsNotNone(content)
        self.assertIn("- age < 45", content.parts[0].text)

    def test_remember_query_result_ignores_case_bucket_comparisons_in_query_summary(self) -> None:
        with patch(
            "agent_zoo.sql_agent.callbacks.get_schema_summary",
            return_value={
                "status": "success",
                "categorical_value_guidance": [
                    {"column": "parent_category", "values": ["B-cell lymphoma", "T-cell lymphoma"]},
                ],
            },
        ):
            callback = build_remember_query_result_callback(self._settings(minimum_aggregate_count=1))

        tool = SimpleNamespace(name="execute_sqlite_read_only")
        tool_context = SimpleNamespace(
            state={
                SQL_ACTIVE_QUERY_TOPIC_STATE_KEY: "split b cell patients by age categories",
            }
        )
        sql = (
            "SELECT CASE "
            "WHEN CAST(age AS INTEGER) >= 70 THEN '70+' "
            "WHEN CAST(age AS INTEGER) BETWEEN 60 AND 69 THEN '60-69' "
            "ELSE 'Unknown / Null' END AS age_category, "
            "COUNT(*) AS matching_count FROM filtered_dataset "
            "WHERE parent_category = 'B-cell lymphoma' "
            "GROUP BY age_category ORDER BY age_category"
        )

        callback(
            tool,
            {
                "sql": sql,
                "is_final": True,
            },
            tool_context,
            make_query_result(
                [
                    {"age_category": "70+", "matching_count": 51},
                    {"age_category": "Unknown / Null", "matching_count": 12},
                ],
                columns=["age_category", "matching_count"],
                row_count=2,
                sql=sql,
            ),
        )

        query_frame = get_sql_current_query_frame(tool_context.state)
        self.assertIsNotNone(query_frame)
        self.assertEqual(
            query_frame.get("categorical_filters"),
            [
                {
                    "column": "parent_category",
                    "selected_values": ["B-cell lymphoma"],
                    "available_values": ["B-cell lymphoma", "T-cell lymphoma"],
                }
            ],
        )
        self.assertNotIn("comparison_filters", query_frame)

        content = build_format_final_agent_response_callback()(SimpleNamespace(state=tool_context.state))
        self.assertIsNotNone(content)
        self.assertIn(
            "- parent_category:\n  Matched categories: B-cell lymphoma\n  other stored values: T-cell lymphoma",
            content.parts[0].text,
        )
        self.assertNotIn("age >= 70", content.parts[0].text)

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

    def test_grouped_counts_below_threshold_are_suppressed(self) -> None:
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
        self.assertEqual(public_result["status"], "success")
        self.assertEqual(public_result["rows"], [{"sex": "female", "matching_count": 5}])
        self.assertEqual(public_result["row_count"], 1)
        self.assertEqual(public_result["preview_row_count"], 1)
        self.assertEqual(public_result["public_result_kind"], "count_aggregate")
        self.assertTrue(public_result["grouped_result_suppressed"])
        self.assertIsNone(public_result.get("matched_row_count"))

    def test_grouped_counts_all_below_threshold_are_blocked(self) -> None:
        state = self._invoke_after_tool(
            self._settings(minimum_aggregate_count=3),
            make_query_result(
                [
                    {"sex": "female", "matching_count": 2},
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

    def test_grouped_counts_truncated_preview_reloads_full_groups(self) -> None:
        sql = "SELECT gender, trmttype, COUNT(*) AS matching_count FROM people GROUP BY gender, trmttype"
        full_result = make_query_result(
            [
                {"gender": "Female", "trmttype": "Chemo", "matching_count": 5},
                {"gender": "Male", "trmttype": "Chemo", "matching_count": 4},
                {"gender": "Null", "trmttype": "Others", "matching_count": 1},
            ],
            columns=["gender", "trmttype", "matching_count"],
            row_count=3,
            sql=sql,
        )

        with patch("agent_zoo.sql_agent.callbacks.execute_sqlite_query", return_value=full_result) as mocked_query:
            state = self._invoke_after_tool(
                self._settings(minimum_aggregate_count=3),
                make_query_result(
                    [
                        {"gender": "Female", "trmttype": "Chemo", "matching_count": 5},
                        {"gender": "Male", "trmttype": "Chemo", "matching_count": 4},
                    ],
                    columns=["gender", "trmttype", "matching_count"],
                    row_count=3,
                    truncated=True,
                    sql=sql,
                ),
            )

        public_result = state[SQL_PUBLIC_RESULT_STATE_KEY]
        self.assertEqual(public_result["status"], "success")
        self.assertEqual(
            public_result["rows"],
            [
                {"gender": "Female", "trmttype": "Chemo", "matching_count": 5},
                {"gender": "Male", "trmttype": "Chemo", "matching_count": 4},
            ],
        )
        self.assertTrue(public_result["grouped_result_suppressed"])
        self.assertIsNone(public_result.get("matched_row_count"))
        mocked_query.assert_called_once_with("fixture.sqlite", sql, preview_rows=3)

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
        model = build_sql_result_view_model(public_result)
        self.assertIn('"average_age": "36.25"', model.result_payload)
        self.assertIn('"matching_count": 4', model.result_payload)
        self.assertNotIn('"matching_count": "4.00"', model.result_payload)

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
        self.assertEqual(public_result["rows"][1]["average_age"], 44.0)

        callback = build_format_final_agent_response_callback()
        content = callback(SimpleNamespace(state=state))

        self.assertIsNotNone(content)
        response_text = content.parts[0].text
        self.assertIn('"average_age": "31.40"', response_text)
        self.assertIn('"average_age": "44.00"', response_text)
        self.assertIn('"matching_count": 5', response_text)
        self.assertIn('"matching_count": 4', response_text)
        self.assertNotIn('"matching_count": "5.00"', response_text)

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

    def test_grouped_average_without_matching_count_suppresses_unsafe_groups(self) -> None:
        count_result = make_query_result(
            [
                {"sex": "female", "matching_count": 5},
                {"sex": "male", "matching_count": 1},
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
        self.assertEqual(
            public_result["rows"],
            [{"sex": "female", "matching_count": 5, "average_age": 31.4}],
        )
        self.assertEqual(public_result["public_result_kind"], "safe_aggregate")
        self.assertTrue(public_result["grouped_result_suppressed"])
        self.assertIsNone(public_result.get("matched_row_count"))

    def test_grouped_average_truncated_preview_reloads_full_groups(self) -> None:
        sql = "SELECT sex, AVG(age) AS average_age FROM people GROUP BY sex"
        full_grouped_result = make_query_result(
            [
                {"sex": "female", "average_age": 31.4},
                {"sex": "male", "average_age": 44.0},
                {"sex": "Null", "average_age": 52.0},
            ],
            columns=["sex", "average_age"],
            row_count=3,
            sql=sql,
        )
        count_result = make_query_result(
            [
                {"sex": "female", "matching_count": 5},
                {"sex": "male", "matching_count": 4},
                {"sex": "Null", "matching_count": 1},
            ],
            columns=["sex", "matching_count"],
            row_count=3,
            sql="SELECT sex, COUNT(*) AS matching_count FROM people GROUP BY sex",
        )

        with patch(
            "agent_zoo.sql_agent.callbacks.execute_sqlite_query",
            side_effect=[full_grouped_result, count_result],
        ) as mocked_query:
            state = self._invoke_after_tool(
                self._settings(minimum_aggregate_count=3),
                make_query_result(
                    [
                        {"sex": "female", "average_age": 31.4},
                        {"sex": "male", "average_age": 44.0},
                    ],
                    columns=["sex", "average_age"],
                    row_count=3,
                    truncated=True,
                    sql=sql,
                ),
            )

        public_result = state[SQL_PUBLIC_RESULT_STATE_KEY]
        self.assertEqual(public_result["status"], "success")
        self.assertEqual(
            public_result["rows"],
            [
                {"sex": "female", "matching_count": 5, "average_age": 31.4},
                {"sex": "male", "matching_count": 4, "average_age": 44.0},
            ],
        )
        self.assertTrue(public_result["grouped_result_suppressed"])
        self.assertIsNone(public_result.get("matched_row_count"))
        self.assertEqual(mocked_query.call_count, 2)
        self.assertEqual(mocked_query.call_args_list[0].args, ("fixture.sqlite", sql))
        self.assertEqual(mocked_query.call_args_list[0].kwargs, {"preview_rows": 3})

    def test_grouped_average_below_threshold_is_suppressed(self) -> None:
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
        self.assertEqual(public_result["status"], "success")
        self.assertEqual(
            public_result["rows"],
            [{"sex": "female", "matching_count": 5, "average_age": 31.4}],
        )
        self.assertEqual(public_result["row_count"], 1)
        self.assertEqual(public_result["preview_row_count"], 1)
        self.assertEqual(public_result["public_result_kind"], "safe_aggregate")
        self.assertTrue(public_result["grouped_result_suppressed"])
        self.assertIsNone(public_result.get("matched_row_count"))

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
        self.assertIn("Cohort filter summary:", response_text)
        self.assertIn("- Matched rows in the current filtered dataset and returned only the safe count.", response_text)
        self.assertIn("```", response_text)
        self.assertIn("4", response_text)
        self.assertNotIn("Alice", response_text)
        self.assertNotIn('"name"', response_text)

    def test_formatted_grouped_response_notes_suppressed_groups(self) -> None:
        callback = build_format_final_agent_response_callback()
        content = callback(
            SimpleNamespace(
                state={
                    SQL_PUBLIC_RESULT_STATE_KEY: {
                        "status": "success",
                        "db_path": "fixture.sqlite",
                        "sql": "SELECT sex, COUNT(*) AS matching_count FROM people GROUP BY sex",
                        "columns": ["sex", "matching_count"],
                        "rows": [{"sex": "female", "matching_count": 5}],
                        "row_count": 1,
                        "preview_row_count": 1,
                        "truncated": False,
                        "error": None,
                        "public_result_kind": "count_aggregate",
                        "grouped_result_suppressed": True,
                    }
                }
            )
        )

        self.assertIsNotNone(content)
        response_text = content.parts[0].text
        self.assertIn(
            "- Counted rows for the privacy-safe matched groups in the current filtered dataset.",
            response_text,
        )
        self.assertIn(
            "Note: Some grouped results were omitted due to privacy guardrails.",
            response_text,
        )

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
                                {
                                    "column": "gender",
                                    "selected_values": ["Male"],
                                    "available_values": ["Male", "Female"],
                                    "dataset_missing_or_blank_count": 7,
                                    "dataset_missing_or_blank_count_unit": "patients",
                                },
                                {
                                    "column": "sococc",
                                    "selected_values": ["Retired", "Student", "Unemployed"],
                                    "available_values": ["Employed", "Retired", "Student", "Unemployed", "Unknown"],
                                    "dataset_missing_or_blank_count": 9,
                                    "dataset_missing_or_blank_count_unit": "patients",
                                },
                            ],
                        },
                    }
                }
            )
        )

        self.assertIsNotNone(content)
        response_text = content.parts[0].text
        self.assertIn("Cohort filter summary:", response_text)
        self.assertIn(
            "- gender:\n  Matched categories: Male\n  other stored values: Female\n  total patients with missing / blank values in this column: 7 patients",
            response_text,
        )
        self.assertIn(
            "- sococc:\n  Matched categories: Retired, Student, Unemployed\n  other stored values: Employed, Unknown\n  total patients with missing / blank values in this column: 9 patients",
            response_text,
        )
        self.assertNotIn("- Counted matching rows in the current filtered dataset.", response_text)
        self.assertLess(response_text.index("Cohort filter summary:"), response_text.index("Result:"))

    def test_formatted_final_response_shows_full_non_null_coverage_when_all_values_are_selected(self) -> None:
        callback = build_format_final_agent_response_callback()
        content = callback(
            SimpleNamespace(
                state={
                    SQL_PUBLIC_RESULT_STATE_KEY: {
                        "status": "success",
                        "db_path": "fixture.sqlite",
                        "sql": "SELECT COUNT(*) AS matching_count FROM filtered_dataset WHERE gender IN ('Female', 'Male')",
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
                                    "selected_values": ["Female", "Male"],
                                    "available_values": ["Female", "Male"],
                                    "dataset_missing_or_blank_count": 7,
                                    "dataset_missing_or_blank_count_unit": "patients",
                                },
                            ],
                        },
                    }
                }
            )
        )

        self.assertIsNotNone(content)
        response_text = content.parts[0].text
        self.assertIn(
            "- gender:\n  Matched categories: Female, Male\n  coverage: all non-null stored values for this column\n  total patients with missing / blank values in this column: 7 patients",
            response_text,
        )
        self.assertNotIn("other stored values:", response_text)

    def test_formatted_final_response_includes_small_missing_blank_exclusion_counts(self) -> None:
        callback = build_format_final_agent_response_callback()
        content = callback(
            SimpleNamespace(
                state={
                    SQL_PUBLIC_RESULT_STATE_KEY: {
                        "status": "success",
                        "db_path": "fixture.sqlite",
                        "sql": "SELECT COUNT(*) AS matching_count FROM filtered_dataset WHERE gender IN ('Female', 'Male')",
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
                                    "selected_values": ["Female", "Male"],
                                    "available_values": ["Female", "Male"],
                                    "dataset_missing_or_blank_count": 2,
                                    "dataset_missing_or_blank_count_unit": "patients",
                                },
                            ],
                        },
                    }
                }
            )
        )

        self.assertIsNotNone(content)
        response_text = content.parts[0].text
        self.assertIn("total patients with missing / blank values in this column: 2 patients", response_text)

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
                                {
                                    "column": "gender",
                                    "selected_values": ["Female"],
                                    "available_values": ["Female", "Male"],
                                    "dataset_missing_or_blank_count": 7,
                                    "dataset_missing_or_blank_count_unit": "patients",
                                },
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
        self.assertIn("Cohort filter summary:", response_text)
        self.assertIn(
            "- gender:\n  Matched categories: Female\n  other stored values: Male\n  total patients with missing / blank values in this column: 7 patients",
            response_text,
        )
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
                        {
                            "column": "gender",
                            "selected_values": ["Female"],
                            "available_values": ["Female", "Male"],
                            "dataset_missing_or_blank_count": 7,
                            "dataset_missing_or_blank_count_unit": "patients",
                        },
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
        self.assertIn("Cohort filter summary:", response_text)
        self.assertIn(
            "- gender:\n  Matched categories: Female\n  other stored values: Male\n  total patients with missing / blank values in this column: 7 patients",
            response_text,
        )
        self.assertNotIn("- Counted matching rows in the current filtered dataset.", response_text)
        self.assertIn("```", response_text)
        self.assertIn("4", response_text)
        self.assertLess(response_text.index("Cohort filter summary:"), response_text.index("Result:"))
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
                        {
                            "column": "gender",
                            "selected_values": ["Female"],
                            "available_values": ["Female", "Male"],
                            "dataset_missing_or_blank_count": 7,
                            "dataset_missing_or_blank_count_unit": "patients",
                        },
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
                        "dataset_missing_or_blank_count": 7,
                        "dataset_missing_or_blank_count_unit": "patients",
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
        self.assertIn("Cohort filter summary:", model.query_summary_section)
        self.assertIn(
            "- gender:\n  Matched categories: Female\n  other stored values: Male\n  total patients with missing / blank values in this column: 7 patients",
            model.query_summary_section,
        )
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

    def test_build_sql_result_view_model_formats_scalar_safe_aggregate_to_two_decimals(self) -> None:
        tool_result = {
            "status": "success",
            "sql": "SELECT AVG(age) AS average_age FROM filtered_dataset WHERE gender = 'Female'",
            "columns": ["average_age"],
            "rows": [{"average_age": 44.0}],
            "row_count": 1,
            "preview_row_count": 1,
            "truncated": False,
            "error": None,
            "matched_row_count": 12,
            "public_result_kind": "safe_aggregate",
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

        self.assertEqual(model.result_payload, "44.00")

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

        self.assertEqual(model.result_payload, "12")

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
        pending_clarification = get_sql_pending_clarification(state)
        self.assertIsNotNone(pending_clarification)
        self.assertEqual(
            pending_clarification["options"],
            ["Employed", "Retired", "Unemployed", "Unknown", "Student"],
        )
        self.assertEqual(
            pending_clarification["topic_context"],
            "how many females are alcoholics",
        )
        self.assertEqual(
            pending_clarification["clarification_kind"],
            "generic",
        )

    def test_after_model_callback_attaches_base_query_frame_context_when_available(self) -> None:
        callback = build_normalize_clarification_after_model_callback(self._settings())
        base_query_frame = {
            "question": "how many females work",
            "sql": "SELECT COUNT(*) AS matching_count FROM filtered_dataset WHERE gender = 'Female' AND sococc = 'Employed'",
            "categorical_filters": [
                {"column": "gender", "selected_values": ["Female"], "available_values": ["Female", "Male"]},
                {
                    "column": "sococc",
                    "selected_values": ["Employed"],
                    "available_values": ["Employed", "Retired", "Student", "Unemployed", "Unknown"],
                },
            ],
            "topic_context": (
                "Current committed dataset question/topic: how many females work\n\n"
                "Current committed categorical filters:\n- gender = Female\n- sococc = Employed"
            ),
        }
        state: dict[str, object] = {
            SQL_LAST_USER_TEXT_STATE_KEY: "number of working adults",
        }
        set_sql_current_query_frame(state, base_query_frame)

        result = callback(
            callback_context=SimpleNamespace(state=state),
            llm_response=SimpleNamespace(
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part(
                            text=(
                                '{"response_type":"clarification","user_message":'
                                '"Which occupation status should I count as working?",'
                                '"options":["Employed","Retired","Student"]}'
                            )
                        )
                    ],
                )
            ),
        )

        self.assertIsNotNone(result)
        pending_clarification = get_sql_pending_clarification(state)
        self.assertIsNotNone(pending_clarification)
        self.assertEqual(
            pending_clarification["topic_context"],
            "number of working adults",
        )
        self.assertEqual(
            pending_clarification["query_context"],
            base_query_frame["topic_context"],
        )
        self.assertEqual(
            pending_clarification["base_query_frame"],
            base_query_frame,
        )

    def test_after_model_callback_skips_base_query_frame_context_for_fresh_topic(self) -> None:
        callback = build_normalize_clarification_after_model_callback(self._settings())
        base_query_frame = {
            "question": "number of lades who smoke and drink",
            "sql": "SELECT COUNT(*) AS matching_count FROM filtered_dataset WHERE socsmk = 'Yes' AND socalc = 'Occasionally' AND gender = 'Female'",
            "categorical_filters": [
                {"column": "gender", "selected_values": ["Female"], "available_values": ["Female", "Male"]},
                {"column": "socsmk", "selected_values": ["Yes"], "available_values": ["No", "Unknown", "Yes"]},
                {
                    "column": "socalc",
                    "selected_values": ["Occasionally"],
                    "available_values": ["Never", "Occasionally", "Regularly", "Unknown"],
                },
            ],
            "topic_context": (
                "Current committed dataset question/topic: number of lades who smoke and drink\n\n"
                "Current committed categorical filters:\n- gender = Female\n- socsmk = Yes\n- socalc = Occasionally"
            ),
        }
        state: dict[str, object] = {
            SQL_LAST_USER_TEXT_STATE_KEY: "how many men are working",
            SQL_FRESH_TOPIC_CLARIFICATION_STATE_KEY: True,
        }
        set_sql_current_query_frame(state, base_query_frame)

        result = callback(
            callback_context=SimpleNamespace(state=state),
            llm_response=SimpleNamespace(
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part(
                            text=(
                                '{"response_type":"clarification","user_message":'
                                '"I will use the following filters as a starting point: - gender = Male - sococc = Employed",'
                                '"options":["gender = Male","sococc = Employed"]}'
                            )
                        )
                    ],
                )
            ),
        )

        self.assertIsNotNone(result)
        pending_clarification = get_sql_pending_clarification(state)
        self.assertIsNotNone(pending_clarification)
        self.assertEqual(
            pending_clarification["topic_context"],
            "how many men are working",
        )
        self.assertNotIn("query_context", pending_clarification)
        self.assertNotIn("base_query_frame", pending_clarification)
        self.assertNotIn(SQL_FRESH_TOPIC_CLARIFICATION_STATE_KEY, state)

    def test_after_model_callback_preserves_interpretation_clarification_options(self) -> None:
        with patch(
            "agent_zoo.sql_agent.callbacks.get_schema_summary",
            return_value={
                "status": "success",
                "categorical_value_guidance": [
                    {"column": "socalc", "values": ["Never", "Occasionally", "Regularly", "Unknown"]},
                    {"column": "socsmk", "values": ["No", "Unknown", "Yes"]},
                ],
            },
        ):
            callback = build_normalize_clarification_after_model_callback(self._settings())
        state: dict[str, object] = {
            SQL_LAST_USER_TEXT_STATE_KEY: "how many males drink",
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
                                '"Do you mean alcohol consumption (socalc) or smoking status (socsmk)?",'
                                '"options":["Alcohol consumption (socalc)","Smoking status (socsmk)"]}'
                            )
                        )
                    ],
                )
            ),
        )

        self.assertIsNotNone(result)
        response_text = result.content.parts[0].text
        self.assertIn("Do you mean alcohol consumption (socalc) or smoking status (socsmk)?", response_text)
        self.assertIn(
            "You can reply with one field, multiple fields, no fields, or describe the field you mean in your own words.",
            response_text,
        )
        self.assertNotIn("Choose one or more options, or describe your own rule.", response_text)
        self.assertNotIn("You can reply with option numbers like 2 or 2 and 3.", response_text)
        self.assertIn("1. Alcohol consumption (socalc)", response_text)
        self.assertIn("2. Smoking status (socsmk)", response_text)
        self.assertNotIn("1. Never", response_text)
        self.assertNotIn("1. No", response_text)
        pending_clarification = get_sql_pending_clarification(state)
        self.assertIsNotNone(pending_clarification)
        self.assertEqual(
            pending_clarification["options"],
            ["Alcohol consumption (socalc)", "Smoking status (socsmk)"],
        )
        self.assertEqual(
            pending_clarification["clarification_kind"],
            "interpretation",
        )

    def test_after_model_callback_preserves_embedded_interpretation_options_despite_reasoning_values(self) -> None:
        with patch(
            "agent_zoo.sql_agent.callbacks.get_schema_summary",
            return_value={
                "status": "success",
                "categorical_value_guidance": [
                    {
                        "column": "parent_category",
                        "values": [
                            "B-cell lymphoma",
                            "Hodgkin lymphoma",
                            "Other/Unclassified",
                            "T/NK-cell lymphoma",
                        ],
                    },
                    {
                        "column": "lymsub",
                        "values": [
                            "Classical Hodgkin lymphoma",
                            "Diffuse large B-cell lymphoma (DLBCL)",
                        ],
                    },
                ],
            },
        ):
            callback = build_normalize_clarification_after_model_callback(self._settings())
        state: dict[str, object] = {
            SQL_LAST_USER_TEXT_STATE_KEY: "number of women for each diagnosed cancer type",
        }

        result = callback(
            callback_context=SimpleNamespace(state=state),
            llm_response=SimpleNamespace(
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part(
                            text=(
                                "The previous query used parent_category and produced rows such as B-cell lymphoma and Hodgkin lymphoma. "
                                "If the user means a more specific grouping, lymsub would be more granular.\n"
                                '{"response_type":"clarification","user_message":"When you refer to \'cancer types\', are you asking to group the results by the broader \'Parent lymphoma category\' or the more specific \'Lymphoma subtype\'?","options":["Parent lymphoma category","Lymphoma subtype"]}'
                            )
                        )
                    ],
                )
            ),
        )

        self.assertIsNotNone(result)
        response_text = result.content.parts[0].text
        self.assertIn("When you refer to 'cancer types'", response_text)
        self.assertIn(
            "You can reply with one field, multiple fields, no fields, or describe the field you mean in your own words.",
            response_text,
        )
        self.assertNotIn("Choose one or more options, or describe your own rule.", response_text)
        self.assertNotIn("You can reply with option numbers like 2 or 2 and 3.", response_text)
        self.assertIn("1. Parent lymphoma category", response_text)
        self.assertIn("2. Lymphoma subtype", response_text)
        self.assertNotIn("1. B-cell lymphoma", response_text)
        self.assertNotIn("2. Hodgkin lymphoma", response_text)
        pending_clarification = get_sql_pending_clarification(state)
        self.assertIsNotNone(pending_clarification)
        self.assertEqual(
            pending_clarification["options"],
            ["Parent lymphoma category", "Lymphoma subtype"],
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
        pending_clarification = get_sql_pending_clarification(state)
        self.assertIsNotNone(pending_clarification)
        self.assertEqual(
            pending_clarification["options"],
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
        pending_clarification = get_sql_pending_clarification(state)
        self.assertIsNotNone(pending_clarification)
        self.assertEqual(pending_clarification["options"], [])
        self.assertEqual(
            pending_clarification["topic_context"],
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
        pending_clarification = get_sql_pending_clarification(state)
        self.assertIsNotNone(pending_clarification)
        self.assertEqual(pending_clarification["options"], [])
        self.assertEqual(
            pending_clarification["topic_context"],
            "how many males are not working",
        )

    def test_after_model_callback_falls_back_for_grouping_prose_option_debris(self) -> None:
        callback = build_normalize_clarification_after_model_callback(self._settings())
        state: dict[str, object] = {
            SQL_LAST_USER_TEXT_STATE_KEY: "use other categories instead",
        }

        result = callback(
            callback_context=SimpleNamespace(state=state),
            llm_response=SimpleNamespace(
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part(
                            text=(
                                "Clarification needed.\n"
                                "- previous query grouped by parent_category\n"
                                "- SELECT parent_category, COUNT(*) FROM filtered_dataset GROUP BY parent_category\n"
                                "- latest user reply"
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
        pending_clarification = get_sql_pending_clarification(state)
        self.assertIsNotNone(pending_clarification)
        self.assertEqual(pending_clarification["options"], [])

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

        state: dict[str, object] = {}

        result = callback(
            callback_context=SimpleNamespace(state=state),
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
        pending_clarification = get_sql_pending_clarification(state)
        self.assertIsNotNone(pending_clarification)
        self.assertEqual(
            pending_clarification["clarification_kind"],
            "categorical_values",
        )

    def test_combined_before_model_callback_rewrites_pending_interpretation_followup(self) -> None:
        scope_gate_calls: list[str] = []
        resolver_calls: list[tuple[str, str, list[str], str]] = []

        def fake_scope_gate(user_text: str) -> tuple[bool, str | None]:
            scope_gate_calls.append(user_text)
            return False, "blocked"

        def fake_resolver(topic_context: str, clarification_question: str, options: list[str], user_reply: str) -> dict[str, object]:
            resolver_calls.append((topic_context, clarification_question, options, user_reply))
            return {"resolution_type": "selected_options", "selected_options": ["Smoking status (socsmk)"], "custom_rule": ""}

        with patch("agent_zoo.sql_agent.callbacks.build_llm_scope_gate", return_value=fake_scope_gate), patch(
            "agent_zoo.sql_agent.callbacks.build_llm_clarification_resolver",
            return_value=fake_resolver,
        ):
            callback = build_combined_before_model_callback(self._settings())

        state: dict[str, object] = {}
        set_sql_pending_clarification(
            state,
            {
                "topic_context": "how many males drink",
                "user_message": "Do you mean alcohol consumption (socalc) or smoking status (socsmk)?",
                "options": ["Alcohol consumption (socalc)", "Smoking status (socsmk)"],
                "clarification_kind": "interpretation",
            },
        )
        llm_request = SimpleNamespace(
            contents=[types.Content(role="user", parts=[types.Part(text="2")])]
        )

        result = callback(
            callback_context=SimpleNamespace(state=state),
            llm_request=llm_request,
        )

        self.assertIsNone(result)
        self.assertEqual(scope_gate_calls, [])
        self.assertEqual(resolver_calls, [])
        rewritten_text = llm_request.contents[-1].parts[0].text
        self.assertIn("Clarification question: Do you mean alcohol consumption (socalc) or smoking status (socsmk)?", rewritten_text)
        self.assertIn("Matched options from the reply: Smoking status (socsmk)", rewritten_text)
        self.assertIsNone(get_sql_pending_clarification(state))

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

        state: dict[str, object] = {}
        set_sql_pending_clarification(
            state,
            {
                "topic_context": "how many females are alcoholics",
                "user_message": "Which category should I use for 'alcoholics'?",
                "options": ["Never", "Occasionally", "Regularly", "Unknown"],
            },
        )
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
        self.assertIsNone(get_sql_pending_clarification(state))

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
                state: dict[str, object] = {}
                set_sql_pending_clarification(
                    state,
                    {
                        "topic_context": "how many females are alcoholics",
                        "user_message": "Which category should I use for 'alcoholics'?",
                        "options": ["Never", "Occasionally", "Regularly", "Unknown"],
                    },
                )
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
                self.assertIsNone(get_sql_pending_clarification(state))

    def test_combined_before_model_callback_rewrites_wrapped_numeric_interpretation_followup(self) -> None:
        scope_gate_calls: list[str] = []
        resolver_calls: list[tuple[str, str, list[str], str]] = []

        def fake_scope_gate(user_text: str) -> tuple[bool, str | None]:
            scope_gate_calls.append(user_text)
            return False, "blocked"

        def fake_resolver(topic_context: str, clarification_question: str, options: list[str], user_reply: str) -> dict[str, object]:
            resolver_calls.append((topic_context, clarification_question, options, user_reply))
            return {"resolution_type": "topic_change", "selected_options": [], "custom_rule": ""}

        with patch("agent_zoo.sql_agent.callbacks.build_llm_scope_gate", return_value=fake_scope_gate), patch(
            "agent_zoo.sql_agent.callbacks.build_llm_clarification_resolver",
            return_value=fake_resolver,
        ):
            callback = build_combined_before_model_callback(self._settings())

        state: dict[str, object] = {}
        set_sql_pending_clarification(
            state,
            {
                "topic_context": "how many guys drink",
                "user_message": "I found more than one nearby schema field for this request. Which one do you mean?",
                "options": ["Gender", "Smoking status", "Alcohol consumption status"],
                "clarification_kind": "interpretation",
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
                                "User question:\n"
                                "3"
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

        self.assertIsNone(result)
        self.assertEqual(scope_gate_calls, [])
        self.assertEqual(resolver_calls, [])
        rewritten_text = llm_request.contents[-1].parts[0].text
        self.assertIn("Clarification question: I found more than one nearby schema field for this request. Which one do you mean?", rewritten_text)
        self.assertIn("Matched options from the reply: Alcohol consumption status", rewritten_text)
        self.assertIn("User clarification reply: 3", rewritten_text)
        self.assertIsNone(get_sql_pending_clarification(state))

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

        state: dict[str, object] = {}
        set_sql_pending_clarification(
            state,
            {
                "topic_context": "how many females are alcoholics",
                "user_message": "Which category should I use for 'alcoholics'?",
                "options": ["Never", "Occasionally", "Regularly", "Unknown"],
            },
        )
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
        self.assertIsNone(get_sql_pending_clarification(state))

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
        self.assertIsNotNone(get_sql_pending_clarification(state))

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
        self.assertIsNone(get_sql_pending_clarification(persisted_state))

    def test_combined_before_model_callback_prefers_query_context_for_pending_clarification_resolution(self) -> None:
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

        state: dict[str, object] = {}
        set_sql_pending_clarification(
            state,
            {
                "topic_context": "add it back",
                "query_context": (
                    "Current committed dataset question/topic: how many females work\n\n"
                    "Current committed categorical filters:\n- gender = Female\n- sococc = Employed"
                ),
                "user_message": "Which occupation status should I count as working?",
                "options": ["Employed", "Retired", "Student"],
            },
        )
        expected_query_context = get_sql_pending_clarification(state)["query_context"]
        llm_request = SimpleNamespace(
            contents=[types.Content(role="user", parts=[types.Part(text="count only people who are currently working")])]
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
                    expected_query_context,
                    "Which occupation status should I count as working?",
                    ["Employed", "Retired", "Student"],
                    "count only people who are currently working",
                )
            ],
        )

    def test_combined_before_model_callback_uses_rewritten_current_state_for_broader_followup(self) -> None:
        scope_gate_calls: list[str] = []
        fresh_topic_router_calls: list[tuple[str, str | None]] = []
        refinement_resolver_calls: list[tuple[dict[str, object], str]] = []

        def fake_scope_gate(user_text: str) -> tuple[bool, str | None]:
            scope_gate_calls.append(user_text)
            return False, "blocked"

        def fake_result_refinement_resolver(frame: dict[str, object], user_text: str) -> dict[str, object]:
            refinement_resolver_calls.append((frame, user_text))
            return {
                "resolution_type": "topic_change",
                "target_column": "",
                "selected_values": [],
                "refinement_request": "",
            }

        def fake_fresh_topic_router(user_text: str, current_topic: str | None = None) -> dict[str, str]:
            fresh_topic_router_calls.append((user_text, current_topic))
            return {"resolution_type": "dataset_question"}

        with patch(
            "agent_zoo.sql_agent.callbacks.get_schema_summary",
            return_value={
                "status": "success",
                "categorical_value_guidance": [
                    {"column": "gender", "values": ["Male", "Female"]},
                    {
                        "column": "sococc",
                        "values": ["Employed", "Retired", "Student", "Unemployed", "Unknown"],
                    },
                ],
            },
        ):
            remember_callback = build_remember_query_result_callback(self._settings(minimum_aggregate_count=1))

        with patch("agent_zoo.sql_agent.callbacks.build_llm_scope_gate", return_value=fake_scope_gate), patch(
            "agent_zoo.sql_agent.callbacks.build_llm_clarification_resolver",
            return_value=lambda *args, **kwargs: {"resolution_type": "custom_rule", "selected_options": [], "custom_rule": ""},
        ), patch(
            "agent_zoo.sql_agent.callbacks.build_llm_fresh_topic_relevance_router",
            return_value=fake_fresh_topic_router,
        ), patch(
            "agent_zoo.sql_agent.callbacks.build_llm_result_refinement_resolver",
            return_value=fake_result_refinement_resolver,
        ):
            before_callback = build_combined_before_model_callback(self._settings())

        previous_query_frame = {
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
        }
        state = {
            SQL_ACTIVE_QUERY_TOPIC_STATE_KEY: "females?",
            SQL_REFINEMENT_SOURCE_QUERY_FRAME_STATE_KEY: previous_query_frame,
        }
        set_sql_current_query_frame(state, previous_query_frame)
        remember_callback(
            SimpleNamespace(name="execute_sqlite_read_only"),
            {
                "sql": "SELECT COUNT(*) AS matching_count FROM filtered_dataset WHERE gender = 'Female' AND sococc = 'Employed'",
                "is_final": True,
            },
            SimpleNamespace(state=state),
            make_query_result(
                [{"matching_count": 12}],
                columns=["matching_count"],
                sql="SELECT COUNT(*) AS matching_count FROM filtered_dataset WHERE gender = 'Female' AND sococc = 'Employed'",
            ),
        )

        llm_request = SimpleNamespace(
            contents=[types.Content(role="user", parts=[types.Part(text="number of working adults")])]
        )

        result = before_callback(
            callback_context=SimpleNamespace(state=state),
            llm_request=llm_request,
        )

        self.assertIsNone(result)
        self.assertEqual(fresh_topic_router_calls, [("number of working adults", "how many females work")])
        self.assertEqual(scope_gate_calls, [])
        self.assertEqual(len(refinement_resolver_calls), 1)
        captured_frame, captured_reply = refinement_resolver_calls[0]
        self.assertEqual(captured_reply, "number of working adults")
        self.assertEqual(captured_frame["question"], "how many females work")
        self.assertIn("recent_refinement", captured_frame)
        self.assertIn(
            "Current committed dataset question/topic: how many females work",
            captured_frame["topic_context"],
        )

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

        state: dict[str, object] = {}
        set_sql_pending_clarification(
            state,
            {
                "topic_context": "how many females are alcoholics",
                "user_message": "Which category should I use for 'alcoholics'?",
                "options": ["Never", "Occasionally", "Regularly", "Unknown"],
            },
        )
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
        self.assertIsNone(get_sql_pending_clarification(state))

    def test_combined_before_model_callback_uses_effective_wrapped_reply_for_custom_rule(self) -> None:
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
                "custom_rule": user_reply,
            }

        with patch("agent_zoo.sql_agent.callbacks.build_llm_scope_gate", return_value=fake_scope_gate), patch(
            "agent_zoo.sql_agent.callbacks.build_llm_clarification_resolver",
            return_value=fake_resolver,
        ):
            callback = build_combined_before_model_callback(self._settings())

        state: dict[str, object] = {}
        set_sql_pending_clarification(
            state,
            {
                "topic_context": "how many guys drink",
                "user_message": "I found more than one nearby schema field for this request. Which one do you mean?",
                "options": ["Smoking status", "Alcohol consumption status"],
                "clarification_kind": "interpretation",
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
                                "User question:\n"
                                "only alcohol"
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

        self.assertIsNone(result)
        self.assertEqual(scope_gate_calls, [])
        self.assertEqual(
            resolver_calls,
            [
                (
                    "how many guys drink",
                    "I found more than one nearby schema field for this request. Which one do you mean?",
                    ["Smoking status", "Alcohol consumption status"],
                    "only alcohol",
                )
            ],
        )
        rewritten_text = llm_request.contents[-1].parts[0].text
        self.assertIn("Resolved custom rule from the reply: only alcohol", rewritten_text)
        self.assertIn("User clarification reply: only alcohol", rewritten_text)
        self.assertIsNone(get_sql_pending_clarification(state))

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

        state: dict[str, object] = {}
        set_sql_current_query_frame(
            state,
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
        )
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
        pending_clarification = get_sql_pending_clarification(state)
        self.assertIsNotNone(pending_clarification)
        self.assertEqual(
            pending_clarification["options"],
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

        state: dict[str, object] = {}
        set_sql_current_query_frame(
            state,
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
        )
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

    def test_combined_before_model_callback_rewrites_result_refinement_followup_with_mixed_request(self) -> None:
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
                "target_column": "gender",
                "selected_values": ["Female"],
                "refinement_request": "average age only below 34",
            },
        ):
            callback = build_combined_before_model_callback(self._settings())

        state: dict[str, object] = {}
        set_sql_current_query_frame(
            state,
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
        )
        llm_request = SimpleNamespace(
            contents=[types.Content(role="user", parts=[types.Part(text="make it females and average age only below 34")])]
        )

        result = callback(
            callback_context=SimpleNamespace(state=state),
            llm_request=llm_request,
        )

        self.assertIsNone(result)
        self.assertEqual(scope_gate_calls, [])
        rewritten_text = llm_request.contents[-1].parts[0].text
        self.assertIn("Use this updated value set for gender: Female", rewritten_text)
        self.assertIn("Latest same-query refinement request: average age only below 34", rewritten_text)

    def test_combined_before_model_callback_restores_recently_removed_value_without_llm(self) -> None:
        scope_gate_calls: list[str] = []
        refinement_resolver_calls: list[tuple[dict[str, object], str]] = []

        def fake_scope_gate(user_text: str) -> tuple[bool, str | None]:
            scope_gate_calls.append(user_text)
            return False, "blocked"

        def fake_result_refinement_resolver(frame: dict[str, object], user_text: str) -> dict[str, object]:
            refinement_resolver_calls.append((frame, user_text))
            return {
                "resolution_type": "topic_change",
                "target_column": "",
                "selected_values": [],
                "refinement_request": "",
            }

        with patch("agent_zoo.sql_agent.callbacks.build_llm_scope_gate", return_value=fake_scope_gate), patch(
            "agent_zoo.sql_agent.callbacks.build_llm_clarification_resolver",
            return_value=lambda *args, **kwargs: {"resolution_type": "custom_rule", "selected_options": [], "custom_rule": ""},
        ), patch(
            "agent_zoo.sql_agent.callbacks.build_llm_result_refinement_resolver",
            return_value=fake_result_refinement_resolver,
        ):
            callback = build_combined_before_model_callback(self._settings())

        state: dict[str, object] = {}
        set_sql_current_query_frame(
            state,
            {
                "question": "how many females are smokers and drink alcohol",
                "sql": "SELECT COUNT(*) AS matching_count FROM filtered_dataset WHERE socalc = 'Regularly'",
                "categorical_filters": [
                    {
                        "column": "socalc",
                        "selected_values": ["Regularly"],
                        "available_values": ["Never", "Occasionally", "Regularly", "Unknown"],
                    }
                ],
                "recent_refinement": {
                    "changes": [
                        {
                            "column": "socalc",
                            "previous_values": ["Occasionally", "Regularly"],
                            "selected_values": ["Regularly"],
                            "available_values": ["Never", "Occasionally", "Regularly", "Unknown"],
                            "removed_values": ["Occasionally"],
                        }
                    ]
                },
            },
        )
        llm_request = SimpleNamespace(
            contents=[types.Content(role="user", parts=[types.Part(text="add it back")])]
        )

        result = callback(
            callback_context=SimpleNamespace(state=state),
            llm_request=llm_request,
        )

        self.assertIsNone(result)
        self.assertEqual(scope_gate_calls, [])
        self.assertEqual(refinement_resolver_calls, [])
        rewritten_text = llm_request.contents[-1].parts[0].text
        self.assertIn("Use this updated value set for socalc: Occasionally, Regularly", rewritten_text)

    def test_combined_before_model_callback_applies_scope_gate_for_topic_change(self) -> None:
        scope_gate_calls: list[str] = []
        resolver_calls: list[tuple[str, str, list[str], str]] = []
        fresh_topic_router_calls: list[tuple[str, str | None]] = []

        def fake_scope_gate(user_text: str) -> tuple[bool, str | None]:
            scope_gate_calls.append(user_text)
            return False, "blocked"

        def fake_fresh_topic_router(user_text: str, current_topic: str | None = None) -> dict[str, str]:
            fresh_topic_router_calls.append((user_text, current_topic))
            return {"resolution_type": "out_of_scope"}

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
        ), patch(
            "agent_zoo.sql_agent.callbacks.build_llm_fresh_topic_relevance_router",
            return_value=fake_fresh_topic_router,
        ):
            callback = build_combined_before_model_callback(self._settings())

        state: dict[str, object] = {}
        set_sql_pending_clarification(
            state,
            {
                "topic_context": "how many females are alcoholics",
                "user_message": "Which category should I use for 'alcoholics'?",
                "options": ["Never", "Occasionally", "Regularly", "Unknown"],
            },
        )
        llm_request = SimpleNamespace(
            contents=[types.Content(role="user", parts=[types.Part(text="tell me a joke")])]
        )

        result = callback(
            callback_context=SimpleNamespace(state=state),
            llm_request=llm_request,
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.content.parts[0].text, "I'm a dataset SQL agent. I can only help with questions about the current dataset, its schema, filters, SQL queries, and aggregated results derived from it. I can't answer general non-dataset questions.")
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
        self.assertEqual(fresh_topic_router_calls, [("tell me a joke", "how many females are alcoholics")])
        self.assertEqual(scope_gate_calls, [])
        self.assertIsNone(get_sql_pending_clarification(state))

    def test_combined_before_model_callback_creates_schema_grounding_clarification(self) -> None:
        scope_gate_calls: list[str] = []
        schema_grounding_calls: list[tuple[str, str, list[dict[str, object]], list[str]]] = []

        def fake_scope_gate(user_text: str) -> tuple[bool, str | None]:
            scope_gate_calls.append(user_text)
            return True, None

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
                "schema_text": "filtered_dataset(gender TEXT, socalc TEXT, socsmk TEXT)",
                "tables": [
                    {
                        "name": "filtered_dataset",
                        "columns": [
                            {
                                "name": "gender",
                                "type": "TEXT",
                                "source_header": "Gender",
                                "categorical_values": ["Female", "Male"],
                            },
                            {
                                "name": "socalc",
                                "type": "TEXT",
                                "source_header": "Alcohol consumption",
                                "categorical_values": ["Never", "Occasionally", "Regularly", "Unknown"],
                            },
                            {
                                "name": "socsmk",
                                "type": "TEXT",
                                "source_header": "Smoking status",
                                "categorical_values": ["No", "Unknown", "Yes"],
                            },
                        ],
                    }
                ],
                "categorical_value_guidance": [],
                "categorical_value_guidance_text": "",
            },
        ), patch(
            "agent_zoo.sql_agent.callbacks.build_llm_scope_gate",
            return_value=fake_scope_gate,
        ), patch(
            "agent_zoo.sql_agent.callbacks.build_llm_clarification_resolver",
            return_value=lambda *args, **kwargs: {"resolution_type": "custom_rule", "selected_options": [], "custom_rule": ""},
        ), patch(
            "agent_zoo.sql_agent.callbacks.build_llm_result_refinement_resolver",
            return_value=lambda *args, **kwargs: {"resolution_type": "topic_change", "target_column": "", "selected_values": [], "refinement_request": ""},
        ), patch(
            "agent_zoo.sql_agent.callbacks.build_llm_schema_grounding_resolver",
            return_value=fake_schema_grounding_resolver,
        ):
            callback = build_combined_before_model_callback(self._settings())

        state: dict[str, object] = {}
        llm_request = SimpleNamespace(
            contents=[types.Content(role="user", parts=[types.Part(text="how many females drink")])]
        )

        result = callback(
            callback_context=SimpleNamespace(state=state),
            llm_request=llm_request,
        )

        self.assertIsNotNone(result)
        self.assertEqual(scope_gate_calls, [])
        self.assertEqual(len(schema_grounding_calls), 1)
        self.assertEqual(schema_grounding_calls[0][3], ["gender", "socalc", "socsmk"])
        self.assertTrue(
            any(
                entry.get("column") == "gender" and entry.get("candidate_value") == "Female"
                for entry in schema_grounding_calls[0][2]
            )
        )
        response_text = result.content.parts[0].text
        self.assertIn("I found more than one nearby schema field for this request. Which one do you mean?", response_text)
        self.assertIn(
            "You can reply with one field, multiple fields, no fields, or describe the field you mean in your own words.",
            response_text,
        )
        self.assertNotIn("Choose one or more options, or describe your own rule.", response_text)
        self.assertNotIn("You can reply with option numbers like 2 or 2 and 3.", response_text)
        self.assertIn("Already matched from your request:", response_text)
        self.assertIn("- gender = Female", response_text)
        self.assertIn("1. Alcohol consumption", response_text)
        self.assertIn("2. Smoking status", response_text)
        self.assertIn("3. None of these / another field", response_text)
        self.assertLess(
            response_text.index("1. Alcohol consumption"),
            response_text.index("Already matched from your request:"),
        )
        self.assertNotIn("socalc (values:", response_text)
        self.assertNotIn("socsmk (values:", response_text)
        pending_clarification = get_sql_pending_clarification(state)
        self.assertIsNotNone(pending_clarification)
        self.assertEqual(
            pending_clarification["clarification_kind"],
            "interpretation",
        )
        self.assertEqual(
            pending_clarification["topic_context"],
            "how many females drink",
        )
        self.assertEqual(
            pending_clarification["options"],
            [
                "Alcohol consumption",
                "Smoking status",
                "None of these / another field",
            ],
        )
        self.assertEqual(
            pending_clarification["option_columns"],
            {
                "Alcohol consumption": "socalc",
                "Smoking status": "socsmk",
            },
        )
        self.assertEqual(
            pending_clarification["grounded_filters"],
            {"gender": ["Female"]},
        )

    def test_combined_before_model_callback_preserves_authoritative_schema_labels_in_structured_grounding(self) -> None:
        def fake_scope_gate(user_text: str) -> tuple[bool, str | None]:
            return True, None

        def fake_schema_grounding_resolver(
            user_text: str,
            schema_context: str,
            grounding_candidates: list[dict[str, object]],
            candidate_columns: list[str],
        ) -> dict[str, object]:
            return {
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
            }

        with patch(
            "agent_zoo.sql_agent.callbacks.get_schema_summary",
            return_value={
                "status": "success",
                "schema_text": "filtered_dataset(gender TEXT, socalc TEXT, socsmk TEXT)",
                "tables": [
                    {
                        "name": "filtered_dataset",
                        "columns": [
                            {
                                "name": "gender",
                                "type": "TEXT",
                                "source_header": "Gender",
                                "categorical_values": ["Female", "Male"],
                            },
                            {
                                "name": "socalc",
                                "type": "TEXT",
                                "source_header": "Alcohol consumption?",
                                "categorical_values": ["Never", "Social Drinking", "Unknown"],
                            },
                            {
                                "name": "socsmk",
                                "type": "TEXT",
                                "source_header": "Smoking status?",
                                "categorical_values": ["No", "Yes"],
                            },
                        ],
                    }
                ],
                "categorical_value_guidance": [],
                "categorical_value_guidance_text": "",
            },
        ), patch(
            "agent_zoo.sql_agent.callbacks.build_llm_scope_gate",
            return_value=fake_scope_gate,
        ), patch(
            "agent_zoo.sql_agent.callbacks.build_llm_clarification_resolver",
            return_value=lambda *args, **kwargs: {"resolution_type": "custom_rule", "selected_options": [], "custom_rule": ""},
        ), patch(
            "agent_zoo.sql_agent.callbacks.build_llm_result_refinement_resolver",
            return_value=lambda *args, **kwargs: {"resolution_type": "topic_change", "target_column": "", "selected_values": [], "refinement_request": ""},
        ), patch(
            "agent_zoo.sql_agent.callbacks.build_llm_schema_grounding_resolver",
            return_value=fake_schema_grounding_resolver,
        ):
            callback = build_combined_before_model_callback(self._settings())

        state: dict[str, object] = {}
        llm_request = SimpleNamespace(
            contents=[types.Content(role="user", parts=[types.Part(text="how many women drink")])]
        )

        result = callback(
            callback_context=SimpleNamespace(state=state),
            llm_request=llm_request,
        )

        self.assertIsNotNone(result)
        response_text = result.content.parts[0].text
        self.assertIn("1. Alcohol consumption?", response_text)
        self.assertIn("2. Smoking status?", response_text)
        pending_clarification = get_sql_pending_clarification(state)
        self.assertIsNotNone(pending_clarification)
        self.assertEqual(
            pending_clarification["options"],
            [
                "Alcohol consumption?",
                "Smoking status?",
                "None of these / another field",
            ],
        )
        self.assertEqual(
            pending_clarification["option_columns"],
            {
                "Alcohol consumption?": "socalc",
                "Smoking status?": "socsmk",
            },
        )

    def test_combined_before_model_callback_sequences_structured_grounding_items(self) -> None:
        def fake_scope_gate(user_text: str) -> tuple[bool, str | None]:
            return True, None

        def fake_schema_grounding_resolver(
            user_text: str,
            schema_context: str,
            grounding_candidates: list[dict[str, object]],
            candidate_columns: list[str],
        ) -> dict[str, object]:
            return {
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
                        "ambiguity_kind": "value_ambiguity",
                        "selected_column": "socalc",
                        "selected_values": [],
                        "candidate_columns": [],
                        "candidate_values": ["Never", "Social Drinking", "Unknown"],
                    },
                    {
                        "matched_phrase": "smoke all day",
                        "ambiguity_kind": "value_ambiguity",
                        "selected_column": "socsmk",
                        "selected_values": [],
                        "candidate_columns": [],
                        "candidate_values": ["No", "Yes"],
                    },
                ],
            }

        with patch(
            "agent_zoo.sql_agent.callbacks.get_schema_summary",
            return_value={
                "status": "success",
                "schema_text": "filtered_dataset(gender TEXT, socalc TEXT, socsmk TEXT)",
                "tables": [
                    {
                        "name": "filtered_dataset",
                        "columns": [
                            {
                                "name": "gender",
                                "type": "TEXT",
                                "source_header": "Gender",
                                "categorical_values": ["Female", "Male"],
                            },
                            {
                                "name": "socalc",
                                "type": "TEXT",
                                "source_header": "Alcohol consumption?",
                                "categorical_values": ["Never", "Social Drinking", "Unknown"],
                            },
                            {
                                "name": "socsmk",
                                "type": "TEXT",
                                "source_header": "Smoking status?",
                                "categorical_values": ["No", "Yes"],
                            },
                        ],
                    }
                ],
                "categorical_value_guidance": [],
                "categorical_value_guidance_text": "",
            },
        ), patch(
            "agent_zoo.sql_agent.callbacks.build_llm_scope_gate",
            return_value=fake_scope_gate,
        ), patch(
            "agent_zoo.sql_agent.callbacks.build_llm_clarification_resolver",
            return_value=lambda *args, **kwargs: {"resolution_type": "custom_rule", "selected_options": [], "custom_rule": ""},
        ), patch(
            "agent_zoo.sql_agent.callbacks.build_llm_result_refinement_resolver",
            return_value=lambda *args, **kwargs: {"resolution_type": "topic_change", "target_column": "", "selected_values": [], "refinement_request": ""},
        ), patch(
            "agent_zoo.sql_agent.callbacks.build_llm_schema_grounding_resolver",
            return_value=fake_schema_grounding_resolver,
        ):
            callback = build_combined_before_model_callback(self._settings())

        state: dict[str, object] = {}
        initial_request = SimpleNamespace(
            contents=[types.Content(role="user", parts=[types.Part(text="how many women drink and smoke all day")])]
        )
        first_result = callback(
            callback_context=SimpleNamespace(state=state),
            llm_request=initial_request,
        )

        self.assertIsNotNone(first_result)
        first_text = first_result.content.parts[0].text
        self.assertIn("Already matched from your request:", first_text)
        self.assertIn("- gender = Female", first_text)
        self.assertIn("Which values from Alcohol consumption? should I include?", first_text)

        second_request = SimpleNamespace(
            contents=[types.Content(role="user", parts=[types.Part(text="Social Drinking")])]
        )
        second_result = callback(
            callback_context=SimpleNamespace(state=state),
            llm_request=second_request,
        )

        self.assertIsNotNone(second_result)
        second_text = second_result.content.parts[0].text
        self.assertIn("Which values from Smoking status? should I include?", second_text)
        pending_clarification = get_sql_pending_clarification(state)
        self.assertIsNotNone(pending_clarification)
        self.assertEqual(pending_clarification["options"], ["No", "Yes"])

        third_request = SimpleNamespace(
            contents=[types.Content(role="user", parts=[types.Part(text="Yes")])]
        )
        third_result = callback(
            callback_context=SimpleNamespace(state=state),
            llm_request=third_request,
        )

        self.assertIsNone(third_result)
        rewritten_text = third_request.contents[-1].parts[0].text
        self.assertIn("Current dataset question: how many women drink and smoke all day", rewritten_text)
        self.assertIn("- gender = Female", rewritten_text)
        self.assertIn("- socalc = Social Drinking", rewritten_text)
        self.assertIn("- socsmk = Yes", rewritten_text)
        self.assertIsNone(get_sql_pending_clarification(state))

    def test_combined_before_model_callback_resolves_partially_grounded_drink_request_without_clarification(self) -> None:
        scope_gate_calls: list[str] = []
        litellm_responses = iter(
            [
                '{"resolution_type":"proceed","grounded_filters":[{"column":"gender","selected_values":["Female"]}],"candidate_columns":[]}',
                '{"resolution_type":"needs_clarification","candidate_columns":["age","race","socsmk","socalc"]}',
                '{"candidate_columns":["socalc","socsmk"]}',
                '{"resolution_type":"proceed","selected_column":"socalc"}',
                '{"items":[{"matched_phrase":"women","ambiguity_kind":"grounded_filter","selected_column":"gender","selected_values":["Female"],"candidate_columns":[],"candidate_values":[]},{"matched_phrase":"drink","ambiguity_kind":"value_ambiguity","selected_column":"socalc","selected_values":[],"candidate_columns":[],"candidate_values":[]}]}',
            ]
        )

        def fake_scope_gate(user_text: str) -> tuple[bool, str | None]:
            scope_gate_calls.append(user_text)
            return True, None

        def completion(**kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=next(litellm_responses))
                    )
                ]
            )

        with patch(
            "agent_zoo.sql_agent.callbacks.get_schema_summary",
            return_value={
                "status": "success",
                "schema_text": "filtered_dataset(age TEXT, gender TEXT, race TEXT, socalc TEXT, socsmk TEXT, sococc TEXT)",
                "tables": [
                    {
                        "name": "filtered_dataset",
                        "columns": [
                            {
                                "name": "age",
                                "type": "TEXT",
                                "source_header": "Age at diagnosis in years",
                                "categorical_values": [],
                            },
                            {
                                "name": "gender",
                                "type": "TEXT",
                                "source_header": "Sex of patient",
                                "categorical_values": ["Female", "Male"],
                            },
                            {
                                "name": "race",
                                "type": "TEXT",
                                "source_header": "Race/Ethnicity",
                                "categorical_values": ["Chinese", "Indian", "Malay", "Others"],
                            },
                            {
                                "name": "socalc",
                                "type": "TEXT",
                                "source_header": "Drink frequency",
                                "categorical_values": ["Never", "Occasionally", "Regularly", "Unknown"],
                            },
                            {
                                "name": "socsmk",
                                "type": "TEXT",
                                "source_header": "Smoking status",
                                "categorical_values": ["No", "Unknown", "Yes"],
                            },
                            {
                                "name": "sococc",
                                "type": "TEXT",
                                "source_header": "Occupation status",
                                "categorical_values": ["Employed", "Retired", "Student", "Unemployed", "Unknown"],
                            },
                        ],
                    }
                ],
                "categorical_value_guidance": [],
                "categorical_value_guidance_text": "",
            },
        ), patch(
            "agent_zoo.sql_agent.callbacks.build_llm_scope_gate",
            return_value=fake_scope_gate,
        ), patch(
            "agent_zoo.sql_agent.callbacks.build_llm_clarification_resolver",
            return_value=lambda *args, **kwargs: {"resolution_type": "custom_rule", "selected_options": [], "custom_rule": ""},
        ), patch(
            "agent_zoo.sql_agent.callbacks.build_llm_result_refinement_resolver",
            return_value=lambda *args, **kwargs: {"resolution_type": "topic_change", "target_column": "", "selected_values": [], "refinement_request": ""},
        ), patch.dict(sys.modules, {"litellm": SimpleNamespace(completion=completion)}):
            callback = build_combined_before_model_callback(self._settings())

            state: dict[str, object] = {}
            llm_request = SimpleNamespace(
                contents=[types.Content(role="user", parts=[types.Part(text="how many women drink")])]
            )

            result = callback(
                callback_context=SimpleNamespace(state=state),
                llm_request=llm_request,
            )

            self.assertIsNotNone(result)
            self.assertEqual(scope_gate_calls, [])
            response_text = result.content.parts[0].text
            self.assertIn("Which values from Drink frequency should I include?", response_text)
            self.assertIn("Already matched from your request:", response_text)
            self.assertIn("- gender = Female", response_text)
            self.assertLess(
                response_text.index("1. Never"),
                response_text.index("Already matched from your request:"),
            )
            pending_clarification = get_sql_pending_clarification(state)
            self.assertIsNotNone(pending_clarification)
            self.assertEqual(
                pending_clarification["options"],
                ["Never", "Occasionally", "Regularly", "Unknown"],
            )

    def test_combined_before_model_callback_omits_zero_evidence_fields_from_schema_grounding_clarification(self) -> None:
        scope_gate_calls: list[str] = []
        litellm_responses = iter(
            [
                '{"resolution_type":"proceed","grounded_filters":[{"column":"gender","selected_values":["Female"]}],"candidate_columns":[]}',
                '{"resolution_type":"needs_clarification","candidate_columns":["socalc","socsmk","bsymp","bsymp1"]}',
                '{"resolution_type":"needs_clarification","selected_column":""}',
                '{"items":[{"matched_phrase":"women","ambiguity_kind":"grounded_filter","selected_column":"gender","selected_values":["Female"],"candidate_columns":[],"candidate_values":[]},{"matched_phrase":"drink or smoke","ambiguity_kind":"field_ambiguity","selected_column":"","selected_values":[],"candidate_columns":["socalc","socsmk"],"candidate_values":[]}]}',
            ]
        )

        def fake_scope_gate(user_text: str) -> tuple[bool, str | None]:
            scope_gate_calls.append(user_text)
            return True, None

        def completion(**kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=next(litellm_responses))
                    )
                ]
            )

        with patch(
            "agent_zoo.sql_agent.callbacks.get_schema_summary",
            return_value={
                "status": "success",
                "schema_text": "filtered_dataset(gender TEXT, socalc TEXT, socsmk TEXT, bsymp TEXT, bsymp1 TEXT)",
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
                                "source_header": "Drink frequency",
                                "categorical_values": ["Never", "Occasionally", "Regularly", "Unknown"],
                            },
                            {
                                "name": "socsmk",
                                "type": "TEXT",
                                "source_header": "Smoke frequency",
                                "categorical_values": ["No", "Unknown", "Yes"],
                            },
                            {
                                "name": "bsymp",
                                "type": "TEXT",
                                "source_header": "B symptom status",
                                "categorical_values": ["No", "Unknown", "Yes"],
                            },
                            {
                                "name": "bsymp1",
                                "type": "TEXT",
                                "source_header": "B symptom severity",
                                "categorical_values": ["Mild", "Moderate", "Severe", "Unknown"],
                            },
                        ],
                    }
                ],
                "categorical_value_guidance": [],
                "categorical_value_guidance_text": "",
            },
        ), patch(
            "agent_zoo.sql_agent.callbacks.build_llm_scope_gate",
            return_value=fake_scope_gate,
        ), patch(
            "agent_zoo.sql_agent.callbacks.build_llm_clarification_resolver",
            return_value=lambda *args, **kwargs: {"resolution_type": "custom_rule", "selected_options": [], "custom_rule": ""},
        ), patch(
            "agent_zoo.sql_agent.callbacks.build_llm_result_refinement_resolver",
            return_value=lambda *args, **kwargs: {"resolution_type": "topic_change", "target_column": "", "selected_values": [], "refinement_request": ""},
        ), patch.dict(sys.modules, {"litellm": SimpleNamespace(completion=completion)}):
            callback = build_combined_before_model_callback(self._settings())

            state: dict[str, object] = {}
            llm_request = SimpleNamespace(
                contents=[types.Content(role="user", parts=[types.Part(text="how many women drink and smoke")])]
            )

            result = callback(
                callback_context=SimpleNamespace(state=state),
                llm_request=llm_request,
            )

        self.assertIsNotNone(result)
        self.assertEqual(scope_gate_calls, [])
        response_text = result.content.parts[0].text
        self.assertIn("1. Drink frequency", response_text)
        self.assertIn("2. Smoke frequency", response_text)
        self.assertNotIn("B symptom status", response_text)
        self.assertNotIn("B symptom severity", response_text)
        pending_clarification = get_sql_pending_clarification(state)
        self.assertIsNotNone(pending_clarification)
        self.assertEqual(
            pending_clarification["options"],
            ["Drink frequency", "Smoke frequency", "None of these / another field"],
        )

    def test_combined_before_model_callback_proceeds_with_grounded_drink_filter_when_no_other_field_has_evidence(self) -> None:
        scope_gate_calls: list[str] = []
        litellm_responses = iter(
            [
                '{"resolution_type":"proceed","grounded_filters":[{"column":"socalc","selected_values":["Binge Drinking"]}],"candidate_columns":[]}',
                '{"items":[{"matched_phrase":"drink alot","ambiguity_kind":"grounded_filter","selected_column":"socalc","selected_values":["Binge Drinking"],"candidate_columns":[],"candidate_values":[]}]}',
            ]
        )

        def fake_scope_gate(user_text: str) -> tuple[bool, str | None]:
            scope_gate_calls.append(user_text)
            return True, None

        def completion(**kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=next(litellm_responses))
                    )
                ]
            )

        with patch(
            "agent_zoo.sql_agent.callbacks.get_schema_summary",
            return_value={
                "status": "success",
                "schema_text": "filtered_dataset(socalc TEXT, bsymp TEXT, bsymp1 TEXT, bsymp2 TEXT, bsymp3 TEXT)",
                "tables": [
                    {
                        "name": "filtered_dataset",
                        "columns": [
                            {
                                "name": "socalc",
                                "type": "TEXT",
                                "source_header": "Alcohol intake",
                                "categorical_values": ["Binge Drinking", "Ex-Binge Drinker", "Ex-Social Drinker", "Never", "Social Drinking", "Unknown"],
                            },
                            {
                                "name": "bsymp",
                                "type": "TEXT",
                                "source_header": "B symptoms",
                                "categorical_values": ["No", "No response", "Yes"],
                            },
                            {
                                "name": "bsymp1",
                                "type": "TEXT",
                                "source_header": "bsymp1",
                                "categorical_values": ["No", "No response", "Yes"],
                            },
                            {
                                "name": "bsymp2",
                                "type": "TEXT",
                                "source_header": "bsymp2",
                                "categorical_values": ["No", "No response", "Yes"],
                            },
                            {
                                "name": "bsymp3",
                                "type": "TEXT",
                                "source_header": "bsymp3",
                                "categorical_values": ["No", "No response", "Yes"],
                            },
                        ],
                    }
                ],
                "categorical_value_guidance": [],
                "categorical_value_guidance_text": "",
            },
        ), patch(
            "agent_zoo.sql_agent.callbacks.build_llm_scope_gate",
            return_value=fake_scope_gate,
        ), patch(
            "agent_zoo.sql_agent.callbacks.build_llm_clarification_resolver",
            return_value=lambda *args, **kwargs: {"resolution_type": "custom_rule", "selected_options": [], "custom_rule": ""},
        ), patch(
            "agent_zoo.sql_agent.callbacks.build_llm_result_refinement_resolver",
            return_value=lambda *args, **kwargs: {"resolution_type": "topic_change", "target_column": "", "selected_values": [], "refinement_request": ""},
        ), patch.dict(sys.modules, {"litellm": SimpleNamespace(completion=completion)}):
            callback = build_combined_before_model_callback(self._settings())

            state: dict[str, object] = {}
            llm_request = SimpleNamespace(
                contents=[types.Content(role="user", parts=[types.Part(text="how many drink alot")])]
            )

            result = callback(
                callback_context=SimpleNamespace(state=state),
                llm_request=llm_request,
            )

        self.assertIsNone(result)
        self.assertEqual(scope_gate_calls, [])
        rewritten_text = llm_request.contents[-1].parts[0].text
        self.assertIn("Current dataset question: how many drink alot", rewritten_text)
        self.assertIn("- socalc = Binge Drinking", rewritten_text)
        self.assertIsNone(get_sql_pending_clarification(state))

    def test_combined_before_model_callback_prunes_never_from_drink_exclude_unknown(self) -> None:
        scope_gate_calls: list[str] = []
        litellm_responses = iter(
            [
                '{"resolution_type":"proceed","grounded_filters":[{"column":"gender","selected_values":["Female"]},{"column":"socalc","selected_values":["Never","Occasionally","Regularly"]}],"candidate_columns":[]}',
                '{"selected_values":["Never","Occasionally","Regularly"]}',
                '{"resolution_type":"proceed","candidate_columns":[]}',
                '{"items":[{"matched_phrase":"women","ambiguity_kind":"grounded_filter","selected_column":"gender","selected_values":["Female"],"candidate_columns":[],"candidate_values":[]},{"matched_phrase":"drink","ambiguity_kind":"value_ambiguity","selected_column":"socalc","selected_values":[],"candidate_columns":[],"candidate_values":[]}]}',
            ]
        )

        def fake_scope_gate(user_text: str) -> tuple[bool, str | None]:
            scope_gate_calls.append(user_text)
            return True, None

        def completion(**kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=next(litellm_responses))
                    )
                ]
            )

        with patch(
            "agent_zoo.sql_agent.callbacks.get_schema_summary",
            return_value={
                "status": "success",
                "schema_text": "filtered_dataset(gender TEXT, socalc TEXT, socsmk TEXT)",
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
                            {
                                "name": "socsmk",
                                "type": "TEXT",
                                "source_header": "Smoking status",
                                "categorical_values": ["No", "Unknown", "Yes"],
                            },
                        ],
                    }
                ],
                "categorical_value_guidance": [],
                "categorical_value_guidance_text": "",
            },
        ), patch(
            "agent_zoo.sql_agent.callbacks.build_llm_scope_gate",
            return_value=fake_scope_gate,
        ), patch(
            "agent_zoo.sql_agent.callbacks.build_llm_clarification_resolver",
            return_value=lambda *args, **kwargs: {"resolution_type": "custom_rule", "selected_options": [], "custom_rule": ""},
        ), patch(
            "agent_zoo.sql_agent.callbacks.build_llm_result_refinement_resolver",
            return_value=lambda *args, **kwargs: {"resolution_type": "topic_change", "target_column": "", "selected_values": [], "refinement_request": ""},
        ), patch.dict(sys.modules, {"litellm": SimpleNamespace(completion=completion)}):
            callback = build_combined_before_model_callback(self._settings())

            state: dict[str, object] = {}
            llm_request = SimpleNamespace(
                contents=[types.Content(role="user", parts=[types.Part(text="how many women drink, exclude unk")])]
            )

            result = callback(
                callback_context=SimpleNamespace(state=state),
                llm_request=llm_request,
            )

            self.assertIsNotNone(result)
            self.assertEqual(scope_gate_calls, [])
            response_text = result.content.parts[0].text
            self.assertIn("Which values from Alcohol consumption status should I include?", response_text)
            self.assertIn("Already matched from your request:", response_text)
            self.assertIn("- gender = Female", response_text)
            pending_clarification = get_sql_pending_clarification(state)
            self.assertIsNotNone(pending_clarification)
            self.assertEqual(
                pending_clarification["options"],
                ["Never", "Occasionally", "Regularly", "Unknown"],
            )

    def test_combined_before_model_callback_uses_request_glossary_and_excludes_grounded_gender(self) -> None:
        scope_gate_calls: list[str] = []
        schema_grounding_calls: list[tuple[str, str, list[dict[str, object]], list[str]]] = []

        def fake_scope_gate(user_text: str) -> tuple[bool, str | None]:
            scope_gate_calls.append(user_text)
            return True, None

        def fake_schema_grounding_resolver(
            user_text: str,
            schema_context: str,
            grounding_candidates: list[dict[str, object]],
            candidate_columns: list[str],
        ) -> dict[str, object]:
            schema_grounding_calls.append((user_text, schema_context, grounding_candidates, candidate_columns))
            return {
                "resolution_type": "needs_clarification",
                "grounded_filters": {"gender": ["Male"]},
                "candidate_columns": ["socsmk", "socalc"],
            }

        wrapped_prompt = (
            "The SQLite database is a snapshot of the current filtered cohort from the web app.\n"
            "Use only the table `filtered_dataset`.\n\n"
            "Field glossary:\n"
            "- gender: Sex of patient (categorical)\n"
            "- socsmk: Smoking status (categorical)\n"
            "- socalc: Alcohol consumption status (categorical)\n\n"
            "When the question uses clinician-facing labels, map them to the matching SQLite column names.\n"
            "Use actual schema column names in the SQL you generate.\n\n"
            "User question:\n"
            "how many guys drink"
        )

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
                                "categorical_values": ["Female", "Male"],
                            },
                            {
                                "name": "socsmk",
                                "type": "TEXT",
                                "categorical_values": ["No", "Unknown", "Yes"],
                            },
                            {
                                "name": "socalc",
                                "type": "TEXT",
                                "categorical_values": ["Never", "Occasionally", "Regularly", "Unknown"],
                            },
                        ],
                    }
                ],
                "categorical_value_guidance": [],
                "categorical_value_guidance_text": "",
            },
        ), patch(
            "agent_zoo.sql_agent.callbacks.build_llm_scope_gate",
            return_value=fake_scope_gate,
        ), patch(
            "agent_zoo.sql_agent.callbacks.build_llm_clarification_resolver",
            return_value=lambda *args, **kwargs: {"resolution_type": "custom_rule", "selected_options": [], "custom_rule": ""},
        ), patch(
            "agent_zoo.sql_agent.callbacks.build_llm_result_refinement_resolver",
            return_value=lambda *args, **kwargs: {"resolution_type": "topic_change", "target_column": "", "selected_values": [], "refinement_request": ""},
        ), patch(
            "agent_zoo.sql_agent.callbacks.build_llm_schema_grounding_resolver",
            return_value=fake_schema_grounding_resolver,
        ):
            callback = build_combined_before_model_callback(self._settings())

        state: dict[str, object] = {}
        llm_request = SimpleNamespace(
            contents=[types.Content(role="user", parts=[types.Part(text=wrapped_prompt)])]
        )

        result = callback(
            callback_context=SimpleNamespace(state=state),
            llm_request=llm_request,
        )

        self.assertIsNotNone(result)
        self.assertEqual(scope_gate_calls, [])
        self.assertEqual(len(schema_grounding_calls), 1)
        self.assertEqual(schema_grounding_calls[0][0], "how many guys drink")
        self.assertEqual(schema_grounding_calls[0][3], ["gender", "socsmk", "socalc"])
        self.assertTrue(
            any(
                entry.get("column") == "gender" and entry.get("candidate_value") == "Male"
                for entry in schema_grounding_calls[0][2]
            )
        )
        response_text = result.content.parts[0].text
        self.assertIn("1. Smoking status", response_text)
        self.assertIn("2. Alcohol consumption status", response_text)
        self.assertIn("3. None of these / another field", response_text)
        self.assertIn("Already matched from your request:", response_text)
        self.assertIn("- gender = Male", response_text)
        self.assertNotIn("1. gender", response_text)
        pending_clarification = get_sql_pending_clarification(state)
        self.assertIsNotNone(pending_clarification)
        self.assertEqual(
            pending_clarification["grounded_filters"],
            {"gender": ["Male"]},
        )
        self.assertEqual(
            pending_clarification["options"],
            [
                "Smoking status",
                "Alcohol consumption status",
                "None of these / another field",
            ],
        )
        self.assertEqual(
            pending_clarification["option_columns"],
            {
                "Smoking status": "socsmk",
                "Alcohol consumption status": "socalc",
            },
        )

    def test_combined_before_model_callback_rewrites_grounded_filters_without_clarification(self) -> None:
        scope_gate_calls: list[str] = []
        schema_grounding_calls: list[tuple[str, str, list[dict[str, object]], list[str]]] = []

        def fake_scope_gate(user_text: str) -> tuple[bool, str | None]:
            scope_gate_calls.append(user_text)
            return True, None

        def fake_schema_grounding_resolver(
            user_text: str,
            schema_context: str,
            grounding_candidates: list[dict[str, object]],
            candidate_columns: list[str],
        ) -> dict[str, object]:
            schema_grounding_calls.append((user_text, schema_context, grounding_candidates, candidate_columns))
            return {
                "resolution_type": "proceed",
                "grounded_filters": {"gender": ["Male"]},
                "candidate_columns": [],
            }

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
                                "source_header": "Gender",
                                "categorical_values": ["Female", "Male"],
                            },
                            {
                                "name": "socalc",
                                "type": "TEXT",
                                "source_header": "Alcohol consumption",
                                "categorical_values": ["Never", "Occasionally", "Regularly", "Unknown"],
                            },
                        ],
                    }
                ],
                "categorical_value_guidance": [],
                "categorical_value_guidance_text": "",
            },
        ), patch(
            "agent_zoo.sql_agent.callbacks.build_llm_scope_gate",
            return_value=fake_scope_gate,
        ), patch(
            "agent_zoo.sql_agent.callbacks.build_llm_clarification_resolver",
            return_value=lambda *args, **kwargs: {"resolution_type": "custom_rule", "selected_options": [], "custom_rule": ""},
        ), patch(
            "agent_zoo.sql_agent.callbacks.build_llm_result_refinement_resolver",
            return_value=lambda *args, **kwargs: {"resolution_type": "topic_change", "target_column": "", "selected_values": [], "refinement_request": ""},
        ), patch(
            "agent_zoo.sql_agent.callbacks.build_llm_schema_grounding_resolver",
            return_value=fake_schema_grounding_resolver,
        ):
            callback = build_combined_before_model_callback(self._settings())

        state: dict[str, object] = {}
        llm_request = SimpleNamespace(
            contents=[types.Content(role="user", parts=[types.Part(text="how many guys drink")])]
        )

        result = callback(
            callback_context=SimpleNamespace(state=state),
            llm_request=llm_request,
        )

        self.assertIsNone(result)
        self.assertEqual(scope_gate_calls, [])
        self.assertEqual(len(schema_grounding_calls), 1)
        rewritten_text = llm_request.contents[-1].parts[0].text
        self.assertIn(
            "Answer the user's current dataset question below. This is the current question for this turn, not background context.",
            rewritten_text,
        )
        self.assertIn("Current dataset question: how many guys drink", rewritten_text)
        self.assertIn("- gender = Male", rewritten_text)
        self.assertIsNone(get_sql_pending_clarification(state))

    def test_combined_before_model_callback_runs_schema_grounding_before_scope_gate_for_men_drink(self) -> None:
        scope_gate_calls: list[str] = []
        schema_grounding_calls: list[tuple[str, str, list[dict[str, object]], list[str]]] = []

        def fake_scope_gate(user_text: str) -> tuple[bool, str | None]:
            scope_gate_calls.append(user_text)
            return False, "blocked"

        def fake_schema_grounding_resolver(
            user_text: str,
            schema_context: str,
            grounding_candidates: list[dict[str, object]],
            candidate_columns: list[str],
        ) -> dict[str, object]:
            schema_grounding_calls.append((user_text, schema_context, grounding_candidates, candidate_columns))
            return {
                "resolution_type": "needs_clarification",
                "grounded_filters": {"gender": ["Male"]},
                "candidate_columns": ["socalc", "socsmk"],
            }

        with patch(
            "agent_zoo.sql_agent.callbacks.get_schema_summary",
            return_value={
                "status": "success",
                "schema_text": "filtered_dataset(gender TEXT, socalc TEXT, socsmk TEXT)",
                "tables": [
                    {
                        "name": "filtered_dataset",
                        "columns": [
                            {
                                "name": "gender",
                                "type": "TEXT",
                                "source_header": "Gender",
                                "categorical_values": ["Female", "Male"],
                            },
                            {
                                "name": "socalc",
                                "type": "TEXT",
                                "source_header": "Alcohol consumption",
                                "categorical_values": ["Never", "Occasionally", "Regularly", "Unknown"],
                            },
                            {
                                "name": "socsmk",
                                "type": "TEXT",
                                "source_header": "Smoking status",
                                "categorical_values": ["No", "Unknown", "Yes"],
                            },
                        ],
                    }
                ],
                "categorical_value_guidance": [],
                "categorical_value_guidance_text": "",
            },
        ), patch(
            "agent_zoo.sql_agent.callbacks.build_llm_scope_gate",
            return_value=fake_scope_gate,
        ), patch(
            "agent_zoo.sql_agent.callbacks.build_llm_clarification_resolver",
            return_value=lambda *args, **kwargs: {"resolution_type": "custom_rule", "selected_options": [], "custom_rule": ""},
        ), patch(
            "agent_zoo.sql_agent.callbacks.build_llm_result_refinement_resolver",
            return_value=lambda *args, **kwargs: {"resolution_type": "topic_change", "target_column": "", "selected_values": [], "refinement_request": ""},
        ), patch(
            "agent_zoo.sql_agent.callbacks.build_llm_schema_grounding_resolver",
            return_value=fake_schema_grounding_resolver,
        ):
            callback = build_combined_before_model_callback(self._settings())

        state: dict[str, object] = {}
        llm_request = SimpleNamespace(
            contents=[types.Content(role="user", parts=[types.Part(text="how many men drink")])]
        )

        result = callback(
            callback_context=SimpleNamespace(state=state),
            llm_request=llm_request,
        )

        self.assertIsNotNone(result)
        self.assertEqual(scope_gate_calls, [])
        self.assertEqual(len(schema_grounding_calls), 1)
        response_text = result.content.parts[0].text
        self.assertIn("I found more than one nearby schema field for this request. Which one do you mean?", response_text)
        self.assertIn("1. Alcohol consumption", response_text)
        self.assertIn("2. Smoking status", response_text)
        pending_clarification = get_sql_pending_clarification(state)
        self.assertIsNotNone(pending_clarification)
        self.assertEqual(
            pending_clarification["grounded_filters"],
            {"gender": ["Male"]},
        )

    def test_schema_grounded_clarification_followup_commits_new_topic_instead_of_old_query(self) -> None:
        def fake_scope_gate(user_text: str) -> tuple[bool, str | None]:
            return True, None

        def fake_result_refinement_resolver(frame: dict[str, object], user_text: str) -> dict[str, object]:
            return {
                "resolution_type": "topic_change",
                "target_column": "",
                "selected_values": [],
                "refinement_request": "",
            }

        def fake_schema_grounding_resolver(
            user_text: str,
            schema_context: str,
            grounding_candidates: list[dict[str, object]],
            candidate_columns: list[str],
        ) -> dict[str, object]:
            return {
                "resolution_type": "proceed",
                "grounded_filters": {
                    "gender": ["Male"],
                    "sococc": ["Employed"],
                },
                "candidate_columns": [],
            }

        with patch(
            "agent_zoo.sql_agent.callbacks.get_schema_summary",
            return_value={
                "status": "success",
                "schema_text": "filtered_dataset(gender TEXT, sococc TEXT, socsmk TEXT, socalc TEXT)",
                "tables": [
                    {
                        "name": "filtered_dataset",
                        "columns": [
                            {
                                "name": "gender",
                                "type": "TEXT",
                                "source_header": "Gender",
                                "categorical_values": ["Female", "Male"],
                            },
                            {
                                "name": "sococc",
                                "type": "TEXT",
                                "source_header": "Occupation status",
                                "categorical_values": ["Employed", "Retired", "Student", "Unemployed", "Unknown"],
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
                                "source_header": "Alcohol consumption",
                                "categorical_values": ["Never", "Occasionally", "Regularly", "Unknown"],
                            },
                        ],
                    }
                ],
                "categorical_value_guidance": [
                    {"column": "gender", "values": ["Female", "Male"]},
                    {"column": "sococc", "values": ["Employed", "Retired", "Student", "Unemployed", "Unknown"]},
                    {"column": "socsmk", "values": ["No", "Unknown", "Yes"]},
                    {"column": "socalc", "values": ["Never", "Occasionally", "Regularly", "Unknown"]},
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
            "agent_zoo.sql_agent.callbacks.build_llm_result_refinement_resolver",
            return_value=fake_result_refinement_resolver,
        ), patch(
            "agent_zoo.sql_agent.callbacks.build_llm_schema_grounding_resolver",
            return_value=fake_schema_grounding_resolver,
        ):
            before_callback = build_combined_before_model_callback(self._settings())
            after_callback = build_normalize_clarification_after_model_callback(self._settings())
            remember_callback = build_remember_query_result_callback(self._settings(minimum_aggregate_count=1))

        previous_query_frame = {
            "question": "number of lades who smoke and drink",
            "sql": "SELECT COUNT(*) AS matching_count FROM filtered_dataset WHERE socsmk = 'Yes' AND socalc = 'Occasionally' AND gender = 'Female'",
            "categorical_filters": [
                {"column": "gender", "selected_values": ["Female"], "available_values": ["Female", "Male"]},
                {"column": "socsmk", "selected_values": ["Yes"], "available_values": ["No", "Unknown", "Yes"]},
                {
                    "column": "socalc",
                    "selected_values": ["Occasionally"],
                    "available_values": ["Never", "Occasionally", "Regularly", "Unknown"],
                },
            ],
            "topic_context": (
                "Current committed dataset question/topic: number of lades who smoke and drink\n\n"
                "Current committed categorical filters:\n- gender = Female\n- socsmk = Yes\n- socalc = Occasionally"
            ),
        }
        state: dict[str, object] = {}
        set_sql_current_query_frame(state, previous_query_frame)

        first_request = SimpleNamespace(
            contents=[types.Content(role="user", parts=[types.Part(text="how many men are working")])]
        )
        first_result = before_callback(
            callback_context=SimpleNamespace(state=state),
            llm_request=first_request,
        )

        self.assertIsNone(first_result)
        self.assertIn(
            "Current dataset question: how many men are working",
            first_request.contents[-1].parts[0].text,
        )

        clarification_result = after_callback(
            callback_context=SimpleNamespace(state=state),
            llm_response=SimpleNamespace(
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part(
                            text=(
                                '{"response_type":"clarification","user_message":'
                                '"I will use the following filters as a starting point: - gender = Male - sococc = Employed",'
                                '"options":["gender = Male","sococc = Employed"]}'
                            )
                        )
                    ],
                )
            ),
        )

        self.assertIsNotNone(clarification_result)
        pending_clarification = get_sql_pending_clarification(state)
        self.assertIsNotNone(pending_clarification)
        self.assertEqual(pending_clarification["topic_context"], "how many men are working")
        self.assertNotIn("base_query_frame", pending_clarification)

        followup_request = SimpleNamespace(
            contents=[types.Content(role="user", parts=[types.Part(text="1 2")])]
        )
        followup_result = before_callback(
            callback_context=SimpleNamespace(state=state),
            llm_request=followup_request,
        )

        self.assertIsNone(followup_result)
        self.assertNotIn(SQL_REFINEMENT_SOURCE_QUERY_FRAME_STATE_KEY, state)
        self.assertEqual(
            state[SQL_ACTIVE_QUERY_TOPIC_STATE_KEY],
            "how many men are working",
        )

        remember_callback(
            SimpleNamespace(name="execute_sqlite_read_only"),
            {
                "sql": "SELECT COUNT(*) AS matching_count FROM filtered_dataset WHERE gender = 'Male' AND sococc = 'Employed'",
                "is_final": True,
            },
            SimpleNamespace(state=state),
            make_query_result(
                [{"matching_count": 116}],
                columns=["matching_count"],
                sql="SELECT COUNT(*) AS matching_count FROM filtered_dataset WHERE gender = 'Male' AND sococc = 'Employed'",
            ),
        )

        self.assertEqual(
            get_sql_current_query_frame(state)["question"],
            "how many men are working",
        )
        self.assertIn(
            "Current committed dataset question/topic: how many men are working",
            get_sql_current_query_frame(state)["topic_context"],
        )

    def test_combined_before_model_callback_reuses_interpretation_options_after_result_for_numeric_reply(self) -> None:
        scope_gate_calls: list[str] = []
        refinement_resolver_calls: list[tuple[dict[str, object], str]] = []
        schema_grounding_calls: list[tuple[str, str, list[dict[str, object]], list[str]]] = []

        def fake_scope_gate(user_text: str) -> tuple[bool, str | None]:
            scope_gate_calls.append(user_text)
            return True, None

        def fake_result_refinement_resolver(frame: dict[str, object], user_text: str) -> dict[str, object]:
            refinement_resolver_calls.append((frame, user_text))
            return {
                "resolution_type": "topic_change",
                "target_column": "",
                "selected_values": [],
                "refinement_request": "",
            }

        def fake_schema_grounding_resolver(
            user_text: str,
            schema_context: str,
            grounding_candidates: list[dict[str, object]],
            candidate_columns: list[str],
        ) -> dict[str, object]:
            schema_grounding_calls.append((user_text, schema_context, grounding_candidates, candidate_columns))
            return {
                "resolution_type": "topic_change",
                "grounded_filters": {},
                "candidate_columns": [],
            }

        with patch(
            "agent_zoo.sql_agent.callbacks.get_schema_summary",
            return_value={
                "status": "success",
                "schema_text": "filtered_dataset(gender TEXT, lymsub TEXT, parent_category TEXT)",
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
                                "name": "lymsub",
                                "type": "TEXT",
                                "source_header": "Lymphoma subtype",
                                "categorical_values": [
                                    "Classical Hodgkin lymphoma",
                                    "Diffuse large B-cell lymphoma (DLBCL)",
                                ],
                            },
                            {
                                "name": "parent_category",
                                "type": "TEXT",
                                "source_header": "Parent lymphoma category",
                                "categorical_values": [
                                    "B-cell lymphoma",
                                    "Hodgkin lymphoma",
                                    "Other/Unclassified",
                                    "T/NK-cell lymphoma",
                                ],
                            },
                        ],
                    }
                ],
                "categorical_value_guidance": [
                    {"column": "gender", "values": ["Female", "Male"]},
                    {
                        "column": "lymsub",
                        "values": [
                            "Classical Hodgkin lymphoma",
                            "Diffuse large B-cell lymphoma (DLBCL)",
                        ],
                    },
                    {
                        "column": "parent_category",
                        "values": [
                            "B-cell lymphoma",
                            "Hodgkin lymphoma",
                            "Other/Unclassified",
                            "T/NK-cell lymphoma",
                        ],
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
            "agent_zoo.sql_agent.callbacks.build_llm_result_refinement_resolver",
            return_value=fake_result_refinement_resolver,
        ), patch(
            "agent_zoo.sql_agent.callbacks.build_llm_schema_grounding_resolver",
            return_value=fake_schema_grounding_resolver,
        ):
            before_callback = build_combined_before_model_callback(self._settings())
            remember_callback = build_remember_query_result_callback(self._settings(minimum_aggregate_count=1))

        state: dict[str, object] = {}
        set_sql_pending_clarification(
            state,
            {
                "topic_context": "number of women for each diagnosed cancer type",
                "user_message": "I found more than one nearby schema field for this request. Which one do you mean?",
                "options": ["Sex of patient", "Lymphoma subtype", "Parent lymphoma category"],
                "clarification_kind": "interpretation",
                "option_columns": {
                    "Sex of patient": "gender",
                    "Lymphoma subtype": "lymsub",
                    "Parent lymphoma category": "parent_category",
                },
            },
        )

        first_request = SimpleNamespace(
            contents=[types.Content(role="user", parts=[types.Part(text="1 and 3")])]
        )
        first_result = before_callback(
            callback_context=SimpleNamespace(state=state),
            llm_request=first_request,
        )

        self.assertIsNone(first_result)
        self.assertEqual(scope_gate_calls, [])
        self.assertEqual(refinement_resolver_calls, [])
        self.assertEqual(schema_grounding_calls, [])

        remember_callback(
            SimpleNamespace(name="execute_sqlite_read_only"),
            {
                "sql": "SELECT parent_category, COUNT(*) AS matching_count FROM filtered_dataset WHERE gender = 'Female' GROUP BY parent_category",
                "is_final": True,
            },
            SimpleNamespace(state=state),
            make_query_result(
                [{"parent_category": "B-cell lymphoma", "matching_count": 102}],
                columns=["parent_category", "matching_count"],
                sql="SELECT parent_category, COUNT(*) AS matching_count FROM filtered_dataset WHERE gender = 'Female' GROUP BY parent_category",
            ),
        )

        current_query_frame = get_sql_current_query_frame(state)
        self.assertIsNotNone(current_query_frame)
        self.assertTrue(current_query_frame["is_grouped"])
        self.assertEqual(current_query_frame["group_columns"], ["parent_category"])
        self.assertIn(
            "Current committed grouping columns:\n- parent_category",
            current_query_frame["topic_context"],
        )

        second_request = SimpleNamespace(
            contents=[types.Content(role="user", parts=[types.Part(text="1 and 2")])]
        )
        second_result = before_callback(
            callback_context=SimpleNamespace(state=state),
            llm_request=second_request,
        )

        self.assertIsNone(second_result)
        self.assertEqual(scope_gate_calls, [])
        self.assertEqual(refinement_resolver_calls, [])
        self.assertEqual(schema_grounding_calls, [])
        rewritten_text = second_request.contents[-1].parts[0].text
        self.assertIn(
            "Clarification question: I found more than one nearby schema field for this request. Which one do you mean?",
            rewritten_text,
        )
        self.assertIn(
            "Matched options from the reply: Sex of patient, Lymphoma subtype",
            rewritten_text,
        )
        self.assertIn(
            "Resolved schema fields from the reply: gender, lymsub",
            rewritten_text,
        )
        self.assertIn("User clarification reply: 1 and 2", rewritten_text)

    def test_combined_before_model_callback_returns_grouping_change_clarification(self) -> None:
        scope_gate_calls: list[str] = []
        refinement_resolver_calls: list[tuple[dict[str, object], str]] = []

        def fake_scope_gate(user_text: str) -> tuple[bool, str | None]:
            scope_gate_calls.append(user_text)
            return True, None

        def fake_result_refinement_resolver(frame: dict[str, object], user_text: str) -> dict[str, object]:
            refinement_resolver_calls.append((frame, user_text))
            return {
                "resolution_type": "needs_clarification",
                "target_column": "",
                "selected_values": [],
                "refinement_request": "use other categories instead",
            }

        with patch(
            "agent_zoo.sql_agent.callbacks.get_schema_summary",
            return_value={
                "status": "success",
                "schema_text": "filtered_dataset(gender TEXT, parent_category TEXT, lymsub TEXT)",
                "tables": [],
                "categorical_value_guidance": [
                    {"column": "gender", "values": ["Female", "Male"]},
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
            "agent_zoo.sql_agent.callbacks.build_llm_result_refinement_resolver",
            return_value=fake_result_refinement_resolver,
        ):
            callback = build_combined_before_model_callback(self._settings())

        state: dict[str, object] = {}
        set_sql_current_query_frame(
            state,
            {
                "question": "number of males for each diagnosed cancer type",
                "sql": (
                    "SELECT parent_category, COUNT(*) AS matching_count "
                    "FROM filtered_dataset WHERE gender = 'Male' GROUP BY parent_category"
                ),
                "categorical_filters": [
                    {"column": "gender", "selected_values": ["Male"], "available_values": ["Female", "Male"]},
                ],
                "is_grouped": True,
                "group_columns": ["parent_category"],
                "topic_context": (
                    "Current committed dataset question/topic: number of males for each diagnosed cancer type\n\n"
                    "Current committed categorical filters:\n- gender = Male\n\n"
                    "Current committed grouping columns:\n- parent_category"
                ),
            },
        )
        llm_request = SimpleNamespace(
            contents=[types.Content(role="user", parts=[types.Part(text="use other categories instead")])]
        )

        result = callback(
            callback_context=SimpleNamespace(state=state),
            llm_request=llm_request,
        )

        self.assertIsNotNone(result)
        self.assertEqual(scope_gate_calls, [])
        self.assertEqual(len(refinement_resolver_calls), 1)
        response_text = result.content.parts[0].text
        self.assertIn("Which exact field or category should I group by instead?", response_text)
        pending_clarification = get_sql_pending_clarification(state)
        self.assertIsNotNone(pending_clarification)
        self.assertEqual(pending_clarification["options"], [])
        self.assertEqual(
            pending_clarification["grouping_change_request"],
            "use other categories instead",
        )

    def test_combined_before_model_callback_skips_scope_gate_when_fresh_topic_router_marks_dataset_question(self) -> None:
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
                "resolution_type": "proceed",
                "grounded_filters": {},
                "candidate_columns": [],
            }

        with patch(
            "agent_zoo.sql_agent.callbacks.get_schema_summary",
            return_value={
                "status": "success",
                "schema_text": "filtered_dataset(gender TEXT, socalc TEXT, socsmk TEXT)",
                "tables": [
                    {
                        "name": "filtered_dataset",
                        "columns": [
                            {
                                "name": "gender",
                                "type": "TEXT",
                                "source_header": "Gender",
                                "categorical_values": ["Female", "Male"],
                            },
                            {
                                "name": "socalc",
                                "type": "TEXT",
                                "source_header": "Alcohol consumption",
                                "categorical_values": ["Never", "Occasionally", "Regularly", "Unknown"],
                            },
                            {
                                "name": "socsmk",
                                "type": "TEXT",
                                "source_header": "Smoking status",
                                "categorical_values": ["No", "Unknown", "Yes"],
                            },
                        ],
                    }
                ],
                "categorical_value_guidance": [],
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
            callback = build_combined_before_model_callback(self._settings())

        state: dict[str, object] = {}
        set_sql_current_query_frame(
            state,
            {
                "question": "avg age of working females",
                "sql": "SELECT AVG(age) AS average_age, COUNT(*) AS matching_count FROM filtered_dataset WHERE gender = 'Female' AND sococc IN ('Employed', 'Student')",
                "categorical_filters": [
                    {"column": "gender", "selected_values": ["Female"], "available_values": ["Female", "Male"]},
                    {
                        "column": "sococc",
                        "selected_values": ["Employed", "Student"],
                        "available_values": ["Employed", "Retired", "Student", "Unemployed", "Unknown"],
                    },
                ],
            },
        )
        llm_request = SimpleNamespace(
            contents=[types.Content(role="user", parts=[types.Part(text="tell me a joke")])]
        )

        result = callback(
            callback_context=SimpleNamespace(state=state),
            llm_request=llm_request,
        )

        self.assertIsNone(result)
        self.assertEqual(fresh_topic_router_calls, [("tell me a joke", "avg age of working females")])
        self.assertEqual(len(schema_grounding_calls), 1)
        self.assertEqual(scope_gate_calls, [])

    def test_combined_before_model_callback_blocks_topic_change_before_schema_grounding(self) -> None:
        scope_gate_calls: list[str] = []
        fresh_topic_router_calls: list[tuple[str, str | None]] = []
        schema_grounding_calls: list[tuple[str, str, list[dict[str, object]], list[str]]] = []

        def fake_scope_gate(user_text: str) -> tuple[bool, str | None]:
            scope_gate_calls.append(user_text)
            return False, "blocked"

        def fake_fresh_topic_router(user_text: str, current_topic: str | None = None) -> dict[str, str]:
            fresh_topic_router_calls.append((user_text, current_topic))
            return {"resolution_type": "out_of_scope"}

        def fake_schema_grounding_resolver(
            user_text: str,
            schema_context: str,
            grounding_candidates: list[dict[str, object]],
            candidate_columns: list[str],
        ) -> dict[str, object]:
            schema_grounding_calls.append((user_text, schema_context, grounding_candidates, candidate_columns))
            return {
                "resolution_type": "needs_clarification",
                "grounded_filters": {},
                "candidate_columns": ["Redcap_SLS_ID", "dtfol", "dtodhfol1", "dtr2fol1"],
            }

        with patch(
            "agent_zoo.sql_agent.callbacks.get_schema_summary",
            return_value={
                "status": "success",
                "schema_text": "filtered_dataset(Redcap_SLS_ID TEXT, dtfol TEXT, dtodhfol1 TEXT, dtr2fol1 TEXT, gender TEXT, sococc TEXT)",
                "tables": [
                    {
                        "name": "filtered_dataset",
                        "columns": [
                            {"name": "Redcap_SLS_ID", "type": "TEXT", "source_header": "Redcap SLS ID"},
                            {"name": "dtfol", "type": "TEXT", "source_header": "Date of last follow-up"},
                            {"name": "dtodhfol1", "type": "TEXT", "source_header": "Date of death"},
                            {"name": "dtr2fol1", "type": "TEXT", "source_header": "Date of relapse event"},
                            {
                                "name": "gender",
                                "type": "TEXT",
                                "source_header": "Sex of patient",
                                "categorical_values": ["Female", "Male"],
                            },
                            {
                                "name": "sococc",
                                "type": "TEXT",
                                "source_header": "Occupation status",
                                "categorical_values": ["Employed", "Retired", "Student", "Unemployed", "Unknown"],
                            },
                        ],
                    }
                ],
                "categorical_value_guidance": [
                    {"column": "gender", "values": ["Female", "Male"]},
                    {"column": "sococc", "values": ["Employed", "Retired", "Student", "Unemployed", "Unknown"]},
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
            callback = build_combined_before_model_callback(self._settings())

        state: dict[str, object] = {}
        set_sql_current_query_frame(
            state,
            {
                "question": "avg age of working females",
                "sql": "SELECT AVG(age) AS average_age, COUNT(*) AS matching_count FROM filtered_dataset WHERE gender = 'Female' AND sococc IN ('Employed', 'Student')",
                "categorical_filters": [
                    {"column": "gender", "selected_values": ["Female"], "available_values": ["Female", "Male"]},
                    {
                        "column": "sococc",
                        "selected_values": ["Employed", "Student"],
                        "available_values": ["Employed", "Retired", "Student", "Unemployed", "Unknown"],
                    },
                ],
            },
        )
        llm_request = SimpleNamespace(
            contents=[types.Content(role="user", parts=[types.Part(text="what was my first qn")])]
        )

        result = callback(
            callback_context=SimpleNamespace(state=state),
            llm_request=llm_request,
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.content.parts[0].text, "I'm a dataset SQL agent. I can only help with questions about the current dataset, its schema, filters, SQL queries, and aggregated results derived from it. I can't answer general non-dataset questions.")
        self.assertEqual(fresh_topic_router_calls, [("what was my first qn", "avg age of working females")])
        self.assertEqual(scope_gate_calls, [])
        self.assertEqual(schema_grounding_calls, [])

    def test_combined_before_model_callback_routes_fresh_dataset_query_to_schema_grounding(self) -> None:
        scope_gate_calls: list[str] = []
        fresh_topic_router_calls: list[tuple[str, str | None]] = []
        refinement_resolver_calls: list[tuple[dict[str, object], str]] = []
        schema_grounding_calls: list[tuple[str, str, list[dict[str, object]], list[str]]] = []

        def fake_scope_gate(user_text: str) -> tuple[bool, str | None]:
            scope_gate_calls.append(user_text)
            return True, None

        def fake_result_refinement_resolver(frame: dict[str, object], user_text: str) -> dict[str, object]:
            refinement_resolver_calls.append((frame, user_text))
            return {
                "resolution_type": "topic_change",
                "target_column": "",
                "selected_values": [],
                "refinement_request": "",
            }

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
                "resolution_type": "topic_change",
                "grounded_filters": {},
                "candidate_columns": [],
            }

        with patch(
            "agent_zoo.sql_agent.callbacks.get_schema_summary",
            return_value={
                "status": "success",
                "schema_text": "filtered_dataset(age TEXT, gender TEXT, sococc TEXT)",
                "tables": [
                    {
                        "name": "filtered_dataset",
                        "columns": [
                            {"name": "age", "type": "TEXT", "source_header": "Age"},
                            {
                                "name": "gender",
                                "type": "TEXT",
                                "source_header": "Sex of patient",
                                "categorical_values": ["Female", "Male"],
                            },
                            {
                                "name": "sococc",
                                "type": "TEXT",
                                "source_header": "Occupation status",
                                "categorical_values": ["Employed", "Retired", "Student", "Unemployed", "Unknown"],
                            },
                        ],
                    }
                ],
                "categorical_value_guidance": [
                    {"column": "gender", "values": ["Female", "Male"]},
                    {"column": "sococc", "values": ["Employed", "Retired", "Student", "Unemployed", "Unknown"]},
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
            return_value=fake_result_refinement_resolver,
        ), patch(
            "agent_zoo.sql_agent.callbacks.build_llm_schema_grounding_resolver",
            return_value=fake_schema_grounding_resolver,
        ):
            callback = build_combined_before_model_callback(self._settings())

        state: dict[str, object] = {}
        set_sql_current_query_frame(
            state,
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
        )
        llm_request = SimpleNamespace(
            contents=[types.Content(role="user", parts=[types.Part(text="give me the avg age of females")])]
        )

        result = callback(
            callback_context=SimpleNamespace(state=state),
            llm_request=llm_request,
        )

        self.assertIsNone(result)
        self.assertEqual(len(refinement_resolver_calls), 1)
        self.assertEqual(fresh_topic_router_calls, [("give me the avg age of females", "total non-working")])
        self.assertEqual(scope_gate_calls, [])
        self.assertEqual(len(schema_grounding_calls), 1)
        self.assertEqual(schema_grounding_calls[0][0], "give me the avg age of females")
        self.assertEqual(
            llm_request.contents[-1].parts[0].text,
            "give me the avg age of females",
        )

    def test_combined_before_model_callback_rewrites_grounded_fresh_dataset_question_as_current_question(self) -> None:
        scope_gate_calls: list[str] = []
        fresh_topic_router_calls: list[tuple[str, str | None]] = []
        schema_grounding_calls: list[tuple[str, str, list[dict[str, object]], list[str]]] = []

        def fake_scope_gate(user_text: str) -> tuple[bool, str | None]:
            scope_gate_calls.append(user_text)
            return True, None

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
                "resolution_type": "proceed",
                "grounded_filters": {"gender": ["Female"]},
                "candidate_columns": [],
            }

        with patch(
            "agent_zoo.sql_agent.callbacks.get_schema_summary",
            return_value={
                "status": "success",
                "schema_text": "filtered_dataset(age TEXT, gender TEXT, sococc TEXT)",
                "tables": [
                    {
                        "name": "filtered_dataset",
                        "columns": [
                            {"name": "age", "type": "TEXT", "source_header": "Age"},
                            {
                                "name": "gender",
                                "type": "TEXT",
                                "source_header": "Sex of patient",
                                "categorical_values": ["Female", "Male"],
                            },
                            {
                                "name": "sococc",
                                "type": "TEXT",
                                "source_header": "Occupation status",
                                "categorical_values": ["Employed", "Retired", "Student", "Unemployed", "Unknown"],
                            },
                        ],
                    }
                ],
                "categorical_value_guidance": [
                    {"column": "gender", "values": ["Female", "Male"]},
                    {"column": "sococc", "values": ["Employed", "Retired", "Student", "Unemployed", "Unknown"]},
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
            callback = build_combined_before_model_callback(self._settings())

        state: dict[str, object] = {}
        set_sql_current_query_frame(
            state,
            {
                "question": "total non-working",
                "sql": "SELECT COUNT(*) AS matching_count FROM filtered_dataset WHERE sococc IN ('Retired', 'Student', 'Unemployed')",
                "categorical_filters": [
                    {
                        "column": "sococc",
                        "selected_values": ["Retired", "Student", "Unemployed"],
                        "available_values": ["Employed", "Retired", "Student", "Unemployed", "Unknown"],
                    }
                ],
            },
        )
        llm_request = SimpleNamespace(
            contents=[types.Content(role="user", parts=[types.Part(text="give me the avg age of females")])]
        )

        result = callback(
            callback_context=SimpleNamespace(state=state),
            llm_request=llm_request,
        )

        self.assertIsNone(result)
        self.assertEqual(fresh_topic_router_calls, [("give me the avg age of females", "total non-working")])
        self.assertEqual(scope_gate_calls, [])
        self.assertEqual(len(schema_grounding_calls), 1)
        rewritten_text = llm_request.contents[-1].parts[0].text
        self.assertIn("Current dataset question: give me the avg age of females", rewritten_text)
        self.assertIn("- gender = Female", rewritten_text)
        self.assertNotIn("Original dataset request", rewritten_text)

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
