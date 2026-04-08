"""Define ADK callbacks used by the SQL agent runtime.

This module stores SQL tool results in session state with a public/internal
split and rewrites the agent's final answer into a deterministic, structured
response. It is used internally by the agent and is not intended to be run
directly.
"""

from __future__ import annotations

import re
from typing import Any

from google.adk.models import LlmResponse
from google.genai import types

from .config import SQLAgentSettings, load_settings
from .db import count_subset_rows, execute_sqlite_query, get_schema_summary
from .formatting import (
    format_clarification_response,
    format_structured_response,
    normalize_clarification_response,
)
try:
    from ..scope_guard import DEFAULT_REFUSAL_MESSAGE, build_llm_scope_gate
except ImportError:  # Support ADK loading this package as top-level `sql_agent`.
    from scope_guard import DEFAULT_REFUSAL_MESSAGE, build_llm_scope_gate  # type: ignore[no-redef]


SQL_PUBLIC_RESULT_STATE_KEY = "temp:sql_public_result"
SQL_INTERNAL_RESULT_REF_STATE_KEY = "temp:sql_internal_result_ref"
SQL_INTERNAL_QUERY_RESULT_STATE_KEY = "temp:sql_internal_query_result"
SQL_PENDING_CLARIFICATION_STATE_KEY = "temp:sql_pending_clarification"
SAFE_AGGREGATE_COLUMN_PATTERNS = (
    "avg",
    "average",
    "min",
    "minimum",
    "max",
    "maximum",
)


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _normalize_count_value(value: Any) -> Any:
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _normalize_column_name(value: str) -> str:
    return re.sub(r"\s+", "_", str(value).strip().lower())


def _is_count_column(column_name: str) -> bool:
    normalized = _normalize_column_name(column_name)
    return "count" in normalized


def _is_safe_aggregate_column(column_name: str) -> bool:
    normalized = _normalize_column_name(column_name)
    if _is_count_column(normalized):
        return False
    return any(pattern in normalized for pattern in SAFE_AGGREGATE_COLUMN_PATTERNS)


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


def _clear_pending_clarification_state(state: Any) -> None:
    if hasattr(state, "pop"):
        state.pop(SQL_PENDING_CLARIFICATION_STATE_KEY, None)
    elif SQL_PENDING_CLARIFICATION_STATE_KEY in state:
        state[SQL_PENDING_CLARIFICATION_STATE_KEY] = None


def _get_pending_clarification(state: Any) -> dict[str, Any] | None:
    if state is None or not hasattr(state, "get"):
        return None
    pending = state.get(SQL_PENDING_CLARIFICATION_STATE_KEY)
    if not isinstance(pending, dict):
        return None
    return pending


def _request_ends_with_tool_response(llm_request) -> bool:
    contents = getattr(llm_request, "contents", None) or []
    if not contents:
        return False

    last_content = contents[-1]
    if getattr(last_content, "role", None) == "tool":
        return True

    last_parts = getattr(last_content, "parts", None) or []
    return any(getattr(part, "function_response", None) is not None for part in last_parts)


def _llm_response_has_function_call(llm_response: LlmResponse) -> bool:
    content = getattr(llm_response, "content", None)
    parts = getattr(content, "parts", None) or []
    return any(getattr(part, "function_call", None) is not None for part in parts)


def _extract_llm_response_text(llm_response: LlmResponse) -> str:
    content = getattr(llm_response, "content", None)
    parts = getattr(content, "parts", None) or []
    return "".join(getattr(part, "text", "") or "" for part in parts).strip()


def _normalize_match_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _extract_matching_clarification_options(
    user_text: str,
    options: list[str],
) -> list[str]:
    normalized_user_text = _normalize_match_text(user_text)
    if not normalized_user_text:
        return []

    matches: list[str] = []
    for option in options:
        normalized_option = _normalize_match_text(option)
        if not normalized_option:
            continue
        if re.search(rf"(?<!\w){re.escape(normalized_option)}(?!\w)", normalized_user_text):
            matches.append(option)
    return matches


def _looks_like_new_query(user_text: str) -> bool:
    normalized_user_text = user_text.strip().casefold()
    if not normalized_user_text:
        return False
    if "?" in normalized_user_text:
        return True
    return normalized_user_text.startswith(
        (
            "how ",
            "what ",
            "which ",
            "show ",
            "count ",
            "list ",
            "find ",
            "give ",
            "calculate ",
            "compute ",
            "filter ",
            "sql ",
            "who ",
            "where ",
            "when ",
        )
    )


def _replace_last_user_text(llm_request, replacement_text: str) -> bool:
    contents = getattr(llm_request, "contents", None) or []
    for index in range(len(contents) - 1, -1, -1):
        content = contents[index]
        if getattr(content, "role", None) != "user":
            continue
        contents[index] = types.Content(
            role="user",
            parts=[types.Part(text=replacement_text)],
        )
        return True
    return False


def _apply_pending_clarification_followup(llm_request, clarification: dict[str, Any]) -> bool:
    user_text = _extract_last_user_text(llm_request)
    if not user_text:
        return False

    options = [
        option
        for option in clarification.get("options") or []
        if isinstance(option, str) and option.strip()
    ]
    matched_options = _extract_matching_clarification_options(user_text, options)
    word_count = len(re.findall(r"\w+", user_text))
    if not matched_options and (word_count > 8 or _looks_like_new_query(user_text)):
        return False

    rewritten_sections = [
        "The user is replying to the previous clarification for the same dataset request.",
        f"Clarification question: {clarification.get('user_message') or ''}",
    ]
    if options:
        rewritten_sections.append(
            "Available options:\n" + "\n".join(f"- {option}" for option in options)
        )
    if matched_options:
        rewritten_sections.append(
            "Matched options from the reply: " + ", ".join(matched_options)
        )
    rewritten_sections.append(f"User clarification reply: {user_text}")
    rewritten_sections.append(
        "Interpret this as clarification for the prior dataset question and continue from there."
    )

    return _replace_last_user_text(llm_request, "\n\n".join(rewritten_sections))


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


def _build_detail_fallback_public_result(
    tool_response: dict[str, Any],
    minimum_aggregate_count: int,
) -> dict[str, Any]:
    return _build_scalar_public_result(
        tool_response,
        tool_response.get("row_count", 0),
        minimum_aggregate_count,
        public_result_kind="detail_count_fallback",
    )


def _build_scalar_public_result(
    tool_response: dict[str, Any],
    matching_count: Any,
    minimum_aggregate_count: int,
    *,
    public_result_kind: str = "count_aggregate",
) -> dict[str, Any]:
    normalized_count = _normalize_count_value(matching_count)
    if normalized_count < minimum_aggregate_count:
        return _build_privacy_error_result(
            tool_response,
            "Privacy guardrail blocked this result because the matching count is below the minimum threshold.",
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
        "public_result_kind": public_result_kind,
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


def _detect_count_column(
    rows: list[dict[str, Any]],
    columns: list[str],
) -> str | None:
    if not rows or not columns:
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


def _detect_safe_aggregate_columns(
    rows: list[dict[str, Any]],
    columns: list[str],
) -> list[str]:
    if not rows:
        return []

    aggregate_columns = []
    for column in columns:
        if not _is_safe_aggregate_column(column):
            continue
        if all(_is_numeric(row.get(column)) for row in rows):
            aggregate_columns.append(column)
    return aggregate_columns


def _sql_has_top_level_group_by(sql: str) -> bool:
    state = "normal"
    depth = 0
    tokens: list[str] = []
    token_chars: list[str] = []
    index = 0
    upper_sql = sql.upper()

    while index < len(upper_sql):
        char = upper_sql[index]
        next_char = upper_sql[index + 1] if index + 1 < len(upper_sql) else ""

        if state == "line_comment":
            if char == "\n":
                state = "normal"
            index += 1
            continue

        if state == "block_comment":
            if char == "*" and next_char == "/":
                state = "normal"
                index += 2
                continue
            index += 1
            continue

        if state == "single_quote":
            if char == "'" and next_char == "'":
                index += 2
                continue
            if char == "'":
                state = "normal"
            index += 1
            continue

        if state == "double_quote":
            if char == '"':
                state = "normal"
            index += 1
            continue

        if char == "-" and next_char == "-":
            state = "line_comment"
            index += 2
            continue

        if char == "/" and next_char == "*":
            state = "block_comment"
            index += 2
            continue

        if char == "'":
            state = "single_quote"
            index += 1
            continue

        if char == '"':
            state = "double_quote"
            index += 1
            continue

        if char == "(":
            depth += 1
            if token_chars:
                tokens.append("".join(token_chars))
                token_chars = []
            index += 1
            continue

        if char == ")":
            depth = max(0, depth - 1)
            if token_chars:
                tokens.append("".join(token_chars))
                token_chars = []
            index += 1
            continue

        if depth > 0:
            if token_chars:
                tokens.append("".join(token_chars))
                token_chars = []
            index += 1
            continue

        if char.isalpha() or char == "_":
            token_chars.append(char)
        else:
            if token_chars:
                tokens.append("".join(token_chars))
                token_chars = []
        index += 1

    if token_chars:
        tokens.append("".join(token_chars))

    for first, second in zip(tokens, tokens[1:]):
        if first == "GROUP" and second == "BY":
            return True
    return False


def _find_top_level_keyword(sql: str, keyword: str) -> int | None:
    """Return the start position of the first top-level occurrence of keyword in sql.

    Top-level means not inside parentheses, quoted strings, or comments.
    Returns None if not found.
    """
    upper = sql.upper()
    keyword_upper = keyword.upper()
    keyword_len = len(keyword_upper)
    state = "normal"
    depth = 0
    index = 0
    while index < len(upper):
        char = upper[index]
        next_char = upper[index + 1] if index + 1 < len(upper) else ""
        if state == "line_comment":
            if char == "\n":
                state = "normal"
            index += 1
            continue
        if state == "block_comment":
            if char == "*" and next_char == "/":
                state = "normal"
                index += 2
                continue
            index += 1
            continue
        if state == "single_quote":
            if char == "'" and next_char == "'":
                index += 2
                continue
            if char == "'":
                state = "normal"
            index += 1
            continue
        if state == "double_quote":
            if char == '"':
                state = "normal"
            index += 1
            continue
        if char == "-" and next_char == "-":
            state = "line_comment"
            index += 2
            continue
        if char == "/" and next_char == "*":
            state = "block_comment"
            index += 2
            continue
        if char == "'":
            state = "single_quote"
            index += 1
            continue
        if char == '"':
            state = "double_quote"
            index += 1
            continue
        if char == "(":
            depth += 1
            index += 1
            continue
        if char == ")":
            depth = max(0, depth - 1)
            index += 1
            continue
        if depth == 0 and upper[index:index + keyword_len] == keyword_upper:
            before = upper[index - 1] if index > 0 else " "
            after = upper[index + keyword_len] if index + keyword_len < len(upper) else " "
            if (not before.isalnum() and before != "_") and (not after.isalnum() and after != "_"):
                return index
        index += 1
    return None


def _build_count_sql(sql: str) -> str | None:
    """Convert a scalar aggregate SELECT to SELECT COUNT(*) over the same FROM/WHERE.

    For example:
        SELECT MIN(age) FROM t WHERE sex = 'male'
        → SELECT COUNT(*) FROM t WHERE sex = 'male'

    Returns None if the FROM clause cannot be found.
    """
    select_pos = _find_top_level_keyword(sql, "SELECT")
    from_pos = _find_top_level_keyword(sql, "FROM")
    if select_pos is None or from_pos is None:
        return None
    prefix = sql[:select_pos].rstrip()
    rest = sql[from_pos:]
    cut_pos = len(rest)
    for terminal in ("GROUP BY", "HAVING", "ORDER BY", "LIMIT"):
        pos = _find_top_level_keyword(rest, terminal)
        if pos is not None and pos < cut_pos:
            cut_pos = pos
    from_clause = rest[:cut_pos].rstrip()
    count_sql = f"SELECT COUNT(*) {from_clause}"
    return f"{prefix} {count_sql}".strip()


def _build_grouped_count_sql(sql: str) -> str | None:
    select_pos = _find_top_level_keyword(sql, "SELECT")
    from_pos = _find_top_level_keyword(sql, "FROM")
    group_by_pos = _find_top_level_keyword(sql, "GROUP BY")
    if select_pos is None or from_pos is None or group_by_pos is None or group_by_pos <= from_pos:
        return None
    prefix = sql[:select_pos].rstrip()

    from_rest = sql[from_pos:]
    from_cut_pos = len(from_rest)
    for terminal in ("ORDER BY", "LIMIT"):
        pos = _find_top_level_keyword(from_rest, terminal)
        if pos is not None and pos < from_cut_pos:
            from_cut_pos = pos
    from_clause = from_rest[:from_cut_pos].rstrip()

    group_by_rest = sql[group_by_pos + len("GROUP BY"):]
    group_by_cut_pos = len(group_by_rest)
    for terminal in ("HAVING", "ORDER BY", "LIMIT"):
        pos = _find_top_level_keyword(group_by_rest, terminal)
        if pos is not None and pos < group_by_cut_pos:
            group_by_cut_pos = pos
    group_by_clause = group_by_rest[:group_by_cut_pos].strip()
    if not group_by_clause:
        return None

    count_sql = f"SELECT {group_by_clause}, COUNT(*) AS matching_count {from_clause}"
    return f"{prefix} {count_sql}".strip()


def _is_top_level_scalar_count_sql(sql: str) -> bool:
    select_pos = _find_top_level_keyword(sql, "SELECT")
    from_pos = _find_top_level_keyword(sql, "FROM")
    if select_pos is None or from_pos is None or from_pos <= select_pos:
        return False

    select_clause = sql[select_pos + len("SELECT"):from_pos].strip()
    return bool(
        re.match(
            r'^COUNT\s*\((?:[^()]|\([^()]*\))*\)\s*(?:AS\s+"?[A-Za-z_][A-Za-z0-9_]*"?)?$',
            select_clause,
            flags=re.IGNORECASE,
        )
    )


def _build_aggregate_public_result(
    tool_response: dict[str, Any],
    minimum_aggregate_count: int,
) -> dict[str, Any] | None:
    rows = tool_response.get("rows") or []
    columns = tool_response.get("columns") or []
    sql = tool_response.get("sql") or ""
    has_group_by = _sql_has_top_level_group_by(sql)
    count_column = _detect_count_column(rows, columns)
    if count_column is None:
        scalar_numeric_value = _extract_scalar_numeric_value(rows)
        if (
            scalar_numeric_value is not None
            and not has_group_by
            and _is_top_level_scalar_count_sql(sql)
        ):
            return _build_scalar_public_result(
                tool_response,
                scalar_numeric_value,
                minimum_aggregate_count,
            )

        # No COUNT column — check if this is a pure scalar aggregate (MIN/MAX/AVG etc.).
        # If so, run a COUNT(*) over the same FROM/WHERE to get the actual subset size
        # and use that for the privacy threshold check.
        aggregate_columns = _detect_safe_aggregate_columns(rows, columns)
        if not aggregate_columns:
            return None
        if has_group_by:
            db_path = tool_response.get("db_path") or ""
            count_sql = _build_grouped_count_sql(sql)
            count_result = (
                execute_sqlite_query(db_path, count_sql, preview_rows=max(len(rows), 1))
                if (count_sql and db_path)
                else {"status": "error"}
            )
            if count_result.get("status") != "success" or count_result.get("truncated"):
                return None

            count_rows = count_result.get("rows") or []
            count_columns = count_result.get("columns") or []
            group_columns = [
                column
                for column in columns
                if column not in aggregate_columns and column in count_columns
            ]
            if not group_columns or len(count_rows) != len(rows):
                return None

            count_by_group: dict[tuple[Any, ...], Any] = {}
            for count_row in count_rows:
                matching_count = count_row.get("matching_count")
                if not _is_numeric(matching_count):
                    return None
                key = tuple(count_row.get(column) for column in group_columns)
                if key in count_by_group:
                    return None
                count_by_group[key] = matching_count

            count_values = []
            public_rows = []
            remaining_columns = [column for column in columns if column not in group_columns]
            for row in rows:
                key = tuple(row.get(column) for column in group_columns)
                matching_count = count_by_group.get(key)
                if not _is_numeric(matching_count):
                    return None
                normalized_count = _normalize_count_value(matching_count)
                count_values.append(normalized_count)

                public_row = {column: row.get(column) for column in group_columns}
                public_row["matching_count"] = normalized_count
                for column in remaining_columns:
                    public_row[column] = row.get(column)
                public_rows.append(public_row)

            if any(value < minimum_aggregate_count for value in count_values):
                return _build_privacy_error_result(
                    tool_response,
                    "Privacy guardrail blocked this grouped result because at least one group count is below the minimum threshold.",
                    matched_row_count=_normalize_count_value(sum(count_values)),
                )

            public_result = dict(tool_response)
            public_result["columns"] = [*group_columns, "matching_count", *remaining_columns]
            public_result["rows"] = public_rows
            public_result["matched_row_count"] = _normalize_count_value(sum(count_values))
            public_result["aggregate_columns"] = aggregate_columns
            public_result["public_result_kind"] = "safe_aggregate"
            return public_result

        db_path = tool_response.get("db_path") or ""
        count_sql = _build_count_sql(sql)
        subset_count = count_subset_rows(db_path, count_sql) if (count_sql and db_path) else None
        if subset_count is not None and subset_count < minimum_aggregate_count:
            return _build_privacy_error_result(
                tool_response,
                "Privacy guardrail blocked this result because the matching count is below the minimum threshold.",
                matched_row_count=subset_count,
            )
        public_result = dict(tool_response)
        public_result["aggregate_columns"] = aggregate_columns
        public_result["public_result_kind"] = "safe_aggregate"
        if subset_count is not None:
            public_result["matched_row_count"] = subset_count
        return public_result

    aggregate_columns = _detect_safe_aggregate_columns(rows, columns)
    is_scalar = len(rows) == 1 and not has_group_by
    is_grouped = has_group_by

    if not is_scalar and not is_grouped:
        return None

    if tool_response.get("truncated"):
        return _build_privacy_error_result(
            tool_response,
            (
                "Privacy guardrail blocked this aggregate result because only a preview "
                "was available, so not every cohort or group could be checked safely."
            ),
        )

    count_values = [row[count_column] for row in rows]
    if any(value < minimum_aggregate_count for value in count_values):
        if not is_grouped:
            matching_count = _normalize_count_value(count_values[0])
            return _build_privacy_error_result(
                tool_response,
                "Privacy guardrail blocked this result because the matching count is below the minimum threshold.",
                matched_row_count=matching_count,
            )
        return _build_privacy_error_result(
            tool_response,
            "Privacy guardrail blocked this grouped result because at least one group count is below the minimum threshold.",
            matched_row_count=_normalize_count_value(sum(count_values)),
        )

    public_result = dict(tool_response)
    public_result["matched_row_count"] = _normalize_count_value(sum(count_values))
    public_result["aggregate_columns"] = aggregate_columns
    public_result["public_result_kind"] = (
        "safe_aggregate" if aggregate_columns else "count_aggregate"
    )
    return public_result


def _build_public_query_result(
    tool_response: dict[str, Any],
    settings: SQLAgentSettings,
) -> dict[str, Any]:
    if tool_response.get("status") != "success":
        return dict(tool_response)

    if not settings.count_aggregates_only:
        return dict(tool_response)

    aggregate_result = _build_aggregate_public_result(
        tool_response,
        settings.minimum_aggregate_count,
    )
    if aggregate_result is not None:
        return aggregate_result

    return _build_detail_fallback_public_result(
        tool_response,
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


def build_normalize_clarification_after_model_callback(
    settings: SQLAgentSettings | None = None,
):
    def normalize_clarification_after_model(
        callback_context=None,
        llm_response: LlmResponse | None = None,
        **kwargs,
    ) -> LlmResponse | None:
        if llm_response is None or _llm_response_has_function_call(llm_response):
            return None

        response_text = _extract_llm_response_text(llm_response)
        if not response_text:
            return None

        clarification = normalize_clarification_response(response_text)
        if clarification is None:
            return None

        if callback_context is not None:
            callback_context.state[SQL_PENDING_CLARIFICATION_STATE_KEY] = clarification

        formatted_response = format_clarification_response(clarification)
        if not formatted_response:
            return None

        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part(text=formatted_response)],
            )
        )

    return normalize_clarification_after_model


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


def _extract_last_user_text(llm_request) -> str:
    """Return the text of the most recent user turn from an LlmRequest."""
    contents = getattr(llm_request, "contents", None) or []
    for content in reversed(contents):
        if getattr(content, "role", None) == "user":
            parts = getattr(content, "parts", None) or []
            return " ".join(
                getattr(part, "text", "") or "" for part in parts
            ).strip()
    return ""


def build_scope_gate_callback(
    classifier,
    refusal_message: str = DEFAULT_REFUSAL_MESSAGE,
):
    """Return a before_model_callback that refuses prompts judged out-of-scope.

    classifier is a callable (user_text) -> (allow: bool, refusal: str | None),
    e.g. the return value of build_llm_scope_gate().
    Returns None when the prompt is in scope or when a SQL result is already in
    state (i.e. the agent is mid-execution past the first LLM call).
    """
    def scope_gate(callback_context=None, llm_request=None, **kwargs) -> LlmResponse | None:
        # Skip the check if the most recent content in the request is a tool
        # response — that means we're on the second LLM call within the same
        # turn (after the SQL tool ran), not at the start of a new user message.
        if _request_ends_with_tool_response(llm_request):
            return None

        user_text = _extract_last_user_text(llm_request) if llm_request is not None else ""
        allow, refusal = classifier(user_text)
        if not allow:
            return LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[types.Part(text=refusal)],
                )
            )
        return None

    return scope_gate


def build_combined_before_model_callback(
    settings: SQLAgentSettings | None = None,
):
    """Chain LLM scope gate → finalize-after-query into a single before_model_callback."""
    active_settings = settings or load_settings()

    schema_summary = get_schema_summary(active_settings.db_path)
    schema_text = schema_summary.get("schema_text") or ""
    classifier = build_llm_scope_gate(active_settings.model, schema_text)

    scope_gate = build_scope_gate_callback(classifier)
    finalize = build_finalize_after_query_before_model_callback(active_settings)

    def combined(callback_context=None, llm_request=None, **kwargs) -> LlmResponse | None:
        if callback_context is not None and not _request_ends_with_tool_response(llm_request):
            _clear_private_result_state(callback_context.state)
            pending_clarification = _get_pending_clarification(callback_context.state)
            clarification_followup = False
            if pending_clarification is not None:
                clarification_followup = _apply_pending_clarification_followup(
                    llm_request,
                    pending_clarification,
                )
                _clear_pending_clarification_state(callback_context.state)
            if clarification_followup:
                return finalize(callback_context=callback_context, llm_request=llm_request, **kwargs)

        result = scope_gate(callback_context=callback_context, llm_request=llm_request, **kwargs)
        if result is not None:
            return result
        return finalize(callback_context=callback_context, llm_request=llm_request, **kwargs)

    return combined


remember_query_result = build_remember_query_result_callback()
normalize_clarification_after_model = build_normalize_clarification_after_model_callback()
format_final_agent_response = build_format_final_agent_response_callback()
