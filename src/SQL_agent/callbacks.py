"""Define ADK callbacks used by the SQL agent runtime.

This module stores SQL tool results in session state with a public/internal
split and rewrites the agent's final answer into a deterministic, structured
response. It is used internally by the agent and is not intended to be run
directly.
"""

from __future__ import annotations

from typing import Any

from google.adk.models import LlmResponse
from google.genai import types

from .config import SQLAgentSettings, load_settings
from .formatting import format_structured_response


SQL_PUBLIC_RESULT_STATE_KEY = "temp:sql_public_result"
SQL_INTERNAL_RESULT_REF_STATE_KEY = "temp:sql_internal_result_ref"
SQL_INTERNAL_QUERY_RESULT_STATE_KEY = "temp:sql_internal_query_result"


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _normalize_count_value(value: Any) -> Any:
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _clear_private_result_state(state: Any) -> None:
    for key in (
        SQL_PUBLIC_RESULT_STATE_KEY,
        SQL_INTERNAL_RESULT_REF_STATE_KEY,
        SQL_INTERNAL_QUERY_RESULT_STATE_KEY,
    ):
        if hasattr(state, "pop"):
            state.pop(key, None)
        elif key in state:
            state[key] = None


def _build_privacy_error_result(
    tool_response: dict[str, Any],
    reason: str,
    *,
    matched_row_count: Any = None,
) -> dict[str, Any]:
    return {
        "status": "error",
        "db_path": tool_response.get("db_path", ""),
        "sql": tool_response.get("sql", ""),
        "columns": [],
        "rows": [],
        "row_count": 0,
        "preview_row_count": 0,
        "truncated": False,
        "error": reason,
        "matched_row_count": matched_row_count,
        "privacy_blocked": True,
    }


def _build_scalar_public_result(
    tool_response: dict[str, Any],
    matching_count: Any,
    minimum_aggregate_count: int,
) -> dict[str, Any]:
    normalized_count = _normalize_count_value(matching_count)
    if normalized_count < minimum_aggregate_count:
        return _build_privacy_error_result(
            tool_response,
            (
                "Privacy guardrail blocked this result because the matching count "
                f"({normalized_count}) is below the minimum threshold "
                f"({minimum_aggregate_count})."
            ),
            matched_row_count=normalized_count,
        )

    return {
        "status": "success",
        "db_path": tool_response.get("db_path", ""),
        "sql": tool_response.get("sql", ""),
        "columns": ["matching_count"],
        "rows": [{"matching_count": normalized_count}],
        "row_count": 1,
        "preview_row_count": 1,
        "truncated": False,
        "error": None,
        "matched_row_count": normalized_count,
    }


def _extract_scalar_numeric_value(rows: list[dict[str, Any]]) -> Any | None:
    if len(rows) != 1:
        return None

    row = rows[0]
    if len(row) != 1:
        return None

    value = next(iter(row.values()))
    if not _is_numeric(value):
        return None

    return value


def _detect_grouped_count_column(
    rows: list[dict[str, Any]],
    columns: list[str],
) -> str | None:
    if not rows or len(columns) < 2:
        return None

    numeric_columns = [
        column
        for column in columns
        if "count" in column.lower()
        if all(_is_numeric(row.get(column)) for row in rows)
    ]
    if len(numeric_columns) != 1:
        return None

    return numeric_columns[0]


def _build_grouped_public_result(
    tool_response: dict[str, Any],
    minimum_aggregate_count: int,
) -> dict[str, Any] | None:
    rows = tool_response.get("rows") or []
    columns = tool_response.get("columns") or []
    count_column = _detect_grouped_count_column(rows, columns)
    if count_column is None:
        return None

    if tool_response.get("truncated"):
        return _build_privacy_error_result(
            tool_response,
            (
                "Privacy guardrail blocked this grouped result because only a preview "
                "was available, so not every group could be checked safely."
            ),
        )

    count_values = [row[count_column] for row in rows]
    if any(value < minimum_aggregate_count for value in count_values):
        return _build_privacy_error_result(
            tool_response,
            (
                "Privacy guardrail blocked this grouped result because at least one "
                f"group count is below the minimum threshold ({minimum_aggregate_count})."
            ),
            matched_row_count=_normalize_count_value(sum(count_values)),
        )

    public_result = dict(tool_response)
    public_result["matched_row_count"] = _normalize_count_value(sum(count_values))
    return public_result


def _build_public_query_result(
    tool_response: dict[str, Any],
    settings: SQLAgentSettings,
) -> dict[str, Any]:
    if tool_response.get("status") != "success":
        return dict(tool_response)

    if not settings.count_aggregates_only:
        return dict(tool_response)

    rows = tool_response.get("rows") or []
    scalar_value = _extract_scalar_numeric_value(rows)
    if scalar_value is not None:
        return _build_scalar_public_result(
            tool_response,
            scalar_value,
            settings.minimum_aggregate_count,
        )

    grouped_result = _build_grouped_public_result(
        tool_response,
        settings.minimum_aggregate_count,
    )
    if grouped_result is not None:
        return grouped_result

    return _build_scalar_public_result(
        tool_response,
        tool_response.get("row_count", 0),
        settings.minimum_aggregate_count,
    )


def build_remember_query_result_callback(
    settings: SQLAgentSettings | None = None,
):
    active_settings = settings or load_settings()

    def remember_query_result(tool, args: dict, tool_context, tool_response: dict, **kwargs) -> dict | None:
        tool_name = getattr(tool, "name", "")
        if tool_name != "execute_sqlite_read_only":
            return None

        _clear_private_result_state(tool_context.state)
        public_result = _build_public_query_result(
            tool_response,
            active_settings,
        )
        tool_context.state[SQL_PUBLIC_RESULT_STATE_KEY] = public_result

        if active_settings.capture_internal_rows and tool_response.get("status") == "success":
            tool_context.state[SQL_INTERNAL_QUERY_RESULT_STATE_KEY] = tool_response
            tool_context.state[SQL_INTERNAL_RESULT_REF_STATE_KEY] = SQL_INTERNAL_QUERY_RESULT_STATE_KEY

        if active_settings.count_aggregates_only:
            return public_result

        return None

    return remember_query_result


def build_format_final_agent_response_callback(
    settings: SQLAgentSettings | None = None,
):
    def format_final_agent_response(callback_context=None, **kwargs) -> types.Content | None:
        context = callback_context
        if context is None:
            return None

        public_query_result = context.state.get(SQL_PUBLIC_RESULT_STATE_KEY)
        if not isinstance(public_query_result, dict):
            return None

        return types.Content(
            role="model",
            parts=[types.Part(text=format_structured_response(public_query_result))],
        )

    return format_final_agent_response


def build_finalize_after_query_before_model_callback(
    settings: SQLAgentSettings | None = None,
):
    def finalize_after_query(callback_context=None, llm_request=None, **kwargs) -> LlmResponse | None:
        context = callback_context
        if context is None:
            return None

        public_query_result = context.state.get(SQL_PUBLIC_RESULT_STATE_KEY)
        if not isinstance(public_query_result, dict):
            return None

        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part(text=format_structured_response(public_query_result))],
            )
        )

    return finalize_after_query


remember_query_result = build_remember_query_result_callback()
format_final_agent_response = build_format_final_agent_response_callback()
finalize_after_query_before_model = build_finalize_after_query_before_model_callback()
