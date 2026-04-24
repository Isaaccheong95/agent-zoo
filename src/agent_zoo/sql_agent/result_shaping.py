"""Shape SQL tool output into privacy-filtered public results and query-frame context.

This module owns the after_tool pipeline: helpers that (a) build the
``last_query_frame`` stored in working memory for refinement memory, and (b)
transform raw tool responses into the public result the agent renders. It is
used internally by ``callbacks.py`` and is not meant to be run directly.
"""

from __future__ import annotations

import re
from typing import Any


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


def _normalize_public_display_sql(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized_sql = re.sub(r"\s+", " ", value).strip().rstrip(";").strip()
    return normalized_sql or None


def _is_count_column(column_name: str) -> bool:
    normalized = _normalize_column_name(column_name)
    return "count" in normalized


def _is_safe_aggregate_column(column_name: str) -> bool:
    normalized = _normalize_column_name(column_name)
    if _is_count_column(normalized):
        return False
    return any(pattern in normalized for pattern in SAFE_AGGREGATE_COLUMN_PATTERNS)


def _ordered_unique_values(values: list[str]) -> list[str]:
    ordered_values: list[str] = []
    seen_values: set[str] = set()
    for value in values:
        normalized_value = str(value or "").strip()
        if not normalized_value or normalized_value in seen_values:
            continue
        seen_values.add(normalized_value)
        ordered_values.append(normalized_value)
    return ordered_values


def _get_query_frame_question_text(query_frame: dict[str, Any]) -> str:
    if not isinstance(query_frame, dict):
        return ""

    question = str(query_frame.get("question") or "").strip()
    if question:
        return question

    topic_context = str(query_frame.get("topic_context") or "").strip()
    if not topic_context:
        return ""

    first_line = topic_context.splitlines()[0].strip()
    prefix = "Current committed dataset question/topic:"
    if first_line.startswith(prefix):
        return first_line[len(prefix) :].strip()
    return first_line
