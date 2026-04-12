"""Define ADK callbacks used by the SQL agent runtime.

This module stores SQL tool results in session state with a public/internal
split and rewrites the agent's final answer into a deterministic, structured
response. It is used internally by the agent and is not intended to be run
directly.
"""

from __future__ import annotations

import copy
import re
from typing import Any

from google.adk.models import LlmResponse
from google.genai import types

from .config import SQLAgentSettings, load_settings
from .db import count_subset_rows, execute_sqlite_query, get_schema_summary
from .formatting import (
    CLARIFICATION_KIND_CATEGORICAL_VALUES,
    CLARIFICATION_KIND_GENERIC,
    CLARIFICATION_KIND_INTERPRETATION,
    build_fallback_clarification_response,
    build_clarification_response,
    clarification_requires_deterministic_fallback,
    format_clarification_response,
    format_public_query_result,
    looks_like_clarification_attempt,
    normalize_clarification_response,
)
try:
    from ..working_memory import (
        get_agent_working_memory_value,
        set_agent_working_memory_value,
    )
except ImportError:  # Support ADK loading this package as top-level `sql_agent`.
    from working_memory import (  # type: ignore[no-redef]
        get_agent_working_memory_value,
        set_agent_working_memory_value,
    )
try:
    from ..scope_guard import (
        DEFAULT_REFUSAL_MESSAGE,
        build_llm_clarification_resolver,
        build_llm_result_refinement_resolver,
        build_llm_schema_grounding_resolver,
        build_llm_scope_gate,
    )
except ImportError:  # Support ADK loading this package as top-level `sql_agent`.
    from scope_guard import (  # type: ignore[no-redef]
        DEFAULT_REFUSAL_MESSAGE,
        build_llm_clarification_resolver,
        build_llm_result_refinement_resolver,
        build_llm_schema_grounding_resolver,
        build_llm_scope_gate,
    )


SQL_PUBLIC_RESULT_STATE_KEY = "temp:sql_public_result"
SQL_PUBLIC_RESULT_RENDERED_STATE_KEY = "temp:sql_public_result_rendered"
SQL_INTERNAL_RESULT_REF_STATE_KEY = "temp:sql_internal_result_ref"
SQL_INTERNAL_QUERY_RESULT_STATE_KEY = "temp:sql_internal_query_result"
SQL_LAST_USER_TEXT_STATE_KEY = "temp:sql_last_user_text"
SQL_ACTIVE_QUERY_TOPIC_STATE_KEY = "temp:sql_active_query_topic"
SQL_REFINEMENT_SOURCE_QUERY_FRAME_STATE_KEY = "temp:sql_refinement_source_query_frame"
SQL_FRESH_TOPIC_CLARIFICATION_STATE_KEY = "temp:sql_fresh_topic_clarification"
SQL_WORKING_MEMORY_NAMESPACE = "sql_agent"
SQL_WORKING_MEMORY_PENDING_CLARIFICATION_KEY = "pending_clarification"
SQL_WORKING_MEMORY_CURRENT_QUERY_FRAME_KEY = "current_query_frame"
SQL_WORKING_MEMORY_RECENT_INTERPRETATION_CLARIFICATION_KEY = "recent_interpretation_clarification"
SAFE_AGGREGATE_COLUMN_PATTERNS = (
    "avg",
    "average",
    "min",
    "minimum",
    "max",
    "maximum",
)
_GENERIC_ADD_HINT_TOKENS = frozenset({"add", "again", "also", "back", "include", "put", "restore", "too"})
_GENERIC_REMOVE_HINT_TOKENS = frozenset({"drop", "exclude", "remove", "without"})


def _print_clarification_debug(settings: SQLAgentSettings, stage: str, payload: Any) -> None:
    if not settings.debug:
        return
    if isinstance(payload, str):
        print(f"[debug][sql-clarification][{stage}]\n{payload}")
        return
    print(f"[debug][sql-clarification][{stage}] {payload}")


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


def _clear_private_result_state(state: Any) -> None:
    for key in (
        SQL_PUBLIC_RESULT_STATE_KEY,
        SQL_PUBLIC_RESULT_RENDERED_STATE_KEY,
        SQL_INTERNAL_RESULT_REF_STATE_KEY,
        SQL_INTERNAL_QUERY_RESULT_STATE_KEY,
    ):
        if hasattr(state, "pop"):
            state.pop(key, None)
        elif key in state:
            state[key] = None


def _clear_pending_clarification_state(state: Any) -> None:
    _set_pending_clarification_state(state, None)


def _clear_fresh_topic_clarification_state(state: Any) -> None:
    if state is None:
        return
    if hasattr(state, "pop"):
        state.pop(SQL_FRESH_TOPIC_CLARIFICATION_STATE_KEY, None)
    elif SQL_FRESH_TOPIC_CLARIFICATION_STATE_KEY in state:
        state[SQL_FRESH_TOPIC_CLARIFICATION_STATE_KEY] = None


def _mark_fresh_topic_clarification_state(state: Any) -> None:
    if state is None:
        return
    state[SQL_FRESH_TOPIC_CLARIFICATION_STATE_KEY] = True


def _consume_fresh_topic_clarification_state(state: Any) -> bool:
    if state is None or not hasattr(state, "get"):
        return False
    is_fresh_topic = bool(state.get(SQL_FRESH_TOPIC_CLARIFICATION_STATE_KEY))
    _clear_fresh_topic_clarification_state(state)
    return is_fresh_topic


def _get_pending_clarification(state: Any) -> dict[str, Any] | None:
    pending = _get_sql_working_memory_value(state, SQL_WORKING_MEMORY_PENDING_CLARIFICATION_KEY)
    return pending if isinstance(pending, dict) else None


def _get_last_query_frame(state: Any) -> dict[str, Any] | None:
    query_frame = _get_sql_working_memory_value(state, SQL_WORKING_MEMORY_CURRENT_QUERY_FRAME_KEY)
    return query_frame if isinstance(query_frame, dict) else None


def _get_recent_interpretation_clarification(state: Any) -> dict[str, Any] | None:
    clarification = _get_sql_working_memory_value(
        state,
        SQL_WORKING_MEMORY_RECENT_INTERPRETATION_CLARIFICATION_KEY,
    )
    return clarification if isinstance(clarification, dict) else None


def _get_sql_working_memory_value(state: Any, field_name: str) -> Any:
    return get_agent_working_memory_value(state, SQL_WORKING_MEMORY_NAMESPACE, field_name)


def _set_sql_working_memory_value(state: Any, field_name: str, value: Any) -> None:
    set_agent_working_memory_value(state, SQL_WORKING_MEMORY_NAMESPACE, field_name, value)


def _set_pending_clarification_state(
    state: Any,
    clarification: dict[str, Any] | None,
) -> None:
    _set_sql_working_memory_value(state, SQL_WORKING_MEMORY_PENDING_CLARIFICATION_KEY, clarification)


def _set_last_query_frame_state(
    state: Any,
    query_frame: dict[str, Any] | None,
) -> None:
    _set_sql_working_memory_value(state, SQL_WORKING_MEMORY_CURRENT_QUERY_FRAME_KEY, query_frame)


def _set_recent_interpretation_clarification_state(
    state: Any,
    clarification: dict[str, Any] | None,
) -> None:
    _set_sql_working_memory_value(
        state,
        SQL_WORKING_MEMORY_RECENT_INTERPRETATION_CLARIFICATION_KEY,
        clarification,
    )


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


def _replace_phrase_case_insensitive(text: str, source_phrase: str, target_phrase: str) -> tuple[str, bool]:
    if not text or not source_phrase or not target_phrase:
        return text, False

    pattern = rf"(?<!\w){re.escape(source_phrase)}(?!\w)"
    if re.search(pattern, text, flags=re.IGNORECASE) is None:
        return text, False

    def _replacement(match: re.Match[str]) -> str:
        matched_text = match.group(0)
        if matched_text.islower():
            return target_phrase.lower()
        if matched_text.isupper():
            return target_phrase.upper()
        if matched_text[:1].isupper() and matched_text[1:].islower():
            return target_phrase.capitalize()
        return target_phrase

    return re.sub(pattern, _replacement, text, flags=re.IGNORECASE), True


def _pluralize_phrase(value: str) -> str:
    normalized_value = str(value or "").strip()
    if not normalized_value:
        return ""

    tokens = normalized_value.split()
    last_token = tokens[-1]
    if len(last_token) <= 1:
        return normalized_value
    if re.search(r"(?:s|x|z|ch|sh)$", last_token, flags=re.IGNORECASE):
        plural_token = last_token + "es"
    elif last_token.endswith(("y", "Y")) and len(last_token) > 1 and last_token[-2].lower() not in {"a", "e", "i", "o", "u"}:
        plural_token = last_token[:-1] + ("IES" if last_token[-1].isupper() else "ies")
    else:
        plural_token = last_token + "s"

    tokens[-1] = plural_token
    return " ".join(tokens)


def _rewrite_question_from_recent_refinement(
    source_query_frame: dict[str, Any],
    recent_refinement: dict[str, Any] | None,
) -> str:
    source_question = _get_query_frame_question_text(source_query_frame)
    if not source_question or not isinstance(recent_refinement, dict):
        return source_question

    rewritten_question = source_question
    replaced_any_value = False
    for change in recent_refinement.get("changes") or []:
        if not isinstance(change, dict):
            continue

        previous_values = _ordered_unique_values(
            [
                value
                for value in change.get("previous_values") or []
                if isinstance(value, str) and value.strip()
            ]
        )
        current_values = _ordered_unique_values(
            [
                value
                for value in change.get("selected_values") or []
                if isinstance(value, str) and value.strip()
            ]
        )
        if len(previous_values) != 1 or len(current_values) != 1:
            continue

        candidate_pairs = [
            (previous_values[0], current_values[0]),
            (_pluralize_phrase(previous_values[0]), _pluralize_phrase(current_values[0])),
        ]
        for source_phrase, target_phrase in candidate_pairs:
            rewritten_question, replaced_value = _replace_phrase_case_insensitive(
                rewritten_question,
                source_phrase,
                target_phrase,
            )
            replaced_any_value = replaced_any_value or replaced_value
            if replaced_value:
                break

    if replaced_any_value:
        return rewritten_question
    return source_question


def _build_query_frame_topic_context(query_frame: dict[str, Any]) -> str:
    if not isinstance(query_frame, dict):
        return ""

    context_sections: list[str] = []
    question = str(query_frame.get("question") or "").strip()
    if question:
        context_sections.append(f"Current committed dataset question/topic: {question}")

    categorical_lines: list[str] = []
    for entry in query_frame.get("categorical_filters") or []:
        if not isinstance(entry, dict):
            continue
        column_name = str(entry.get("column") or "").strip()
        selected_values = [
            value
            for value in entry.get("selected_values") or []
            if isinstance(value, str) and value.strip()
        ]
        if not column_name or not selected_values:
            continue
        categorical_lines.append(f"- {column_name} = {', '.join(selected_values)}")
    if categorical_lines:
        context_sections.append(
            "Current committed categorical filters:\n" + "\n".join(categorical_lines)
        )

    comparison_lines: list[str] = []
    for entry in query_frame.get("comparison_filters") or []:
        if not isinstance(entry, dict):
            continue
        column_name = str(entry.get("column") or "").strip()
        operator = str(entry.get("operator") or "").strip()
        value = str(entry.get("value") or "").strip()
        if not column_name or not operator or not value:
            continue
        comparison_lines.append(f"- {column_name} {operator} {value}")
    if comparison_lines:
        context_sections.append(
            "Current committed comparison filters:\n" + "\n".join(comparison_lines)
        )

    return "\n\n".join(context_sections).strip()


def _select_clarification_query_context(state: Any) -> str | None:
    query_frame = _get_last_query_frame(state)
    if not isinstance(query_frame, dict):
        return None
    topic_context = str(query_frame.get("topic_context") or "").strip()
    return topic_context or None


def _select_pending_clarification_topic_text(clarification: dict[str, Any]) -> str | None:
    if not isinstance(clarification, dict):
        return None

    base_query_frame = clarification.get("base_query_frame")
    if isinstance(base_query_frame, dict):
        base_question = _get_query_frame_question_text(base_query_frame)
        if base_question:
            return base_question

    topic_context = clarification.get("topic_context")
    if isinstance(topic_context, str) and topic_context.strip():
        return topic_context.strip()
    return None


def _build_recent_interpretation_clarification(
    clarification: dict[str, Any],
) -> dict[str, Any] | None:
    if not isinstance(clarification, dict):
        return None

    clarification_kind = str(clarification.get("clarification_kind") or "").strip().lower()
    if clarification_kind != CLARIFICATION_KIND_INTERPRETATION:
        return None

    options = [
        option
        for option in clarification.get("options") or []
        if isinstance(option, str) and option.strip()
    ]
    if not options:
        return None

    recent_clarification: dict[str, Any] = {
        "options": options,
        "clarification_kind": CLARIFICATION_KIND_INTERPRETATION,
    }

    user_message = str(clarification.get("user_message") or "").strip()
    if user_message:
        recent_clarification["user_message"] = user_message

    topic_context = _select_pending_clarification_topic_text(clarification)
    if topic_context:
        recent_clarification["topic_context"] = topic_context

    query_context = clarification.get("query_context")
    if isinstance(query_context, str) and query_context.strip():
        recent_clarification["query_context"] = query_context.strip()

    option_columns = clarification.get("option_columns")
    if isinstance(option_columns, dict) and option_columns:
        recent_clarification["option_columns"] = copy.deepcopy(option_columns)

    grounded_filters = clarification.get("grounded_filters")
    if isinstance(grounded_filters, dict) and grounded_filters:
        recent_clarification["grounded_filters"] = copy.deepcopy(grounded_filters)

    base_query_frame = clarification.get("base_query_frame")
    if isinstance(base_query_frame, dict):
        recent_clarification["base_query_frame"] = copy.deepcopy(base_query_frame)

    return recent_clarification


def _build_recent_refinement(
    previous_query_frame: dict[str, Any],
    current_query_frame: dict[str, Any],
) -> dict[str, Any] | None:
    if not isinstance(previous_query_frame, dict) or not isinstance(current_query_frame, dict):
        return None

    previous_filters = {
        str(entry.get("column") or "").strip(): entry
        for entry in previous_query_frame.get("categorical_filters") or []
        if isinstance(entry, dict) and str(entry.get("column") or "").strip()
    }
    current_filters = {
        str(entry.get("column") or "").strip(): entry
        for entry in current_query_frame.get("categorical_filters") or []
        if isinstance(entry, dict) and str(entry.get("column") or "").strip()
    }
    changed_columns = previous_filters.keys() | current_filters.keys()

    changes: list[dict[str, Any]] = []
    for column_name in sorted(changed_columns):
        previous_entry = previous_filters.get(column_name) or {}
        current_entry = current_filters.get(column_name) or {}
        previous_values = _ordered_unique_values(
            [
                value
                for value in previous_entry.get("selected_values") or []
                if isinstance(value, str) and value.strip()
            ]
        )
        current_values = _ordered_unique_values(
            [
                value
                for value in current_entry.get("selected_values") or []
                if isinstance(value, str) and value.strip()
            ]
        )
        if previous_values == current_values:
            continue

        available_values = _ordered_unique_values(
            [
                value
                for value in [
                    *(current_entry.get("available_values") or []),
                    *(previous_entry.get("available_values") or []),
                ]
                if isinstance(value, str) and value.strip()
            ]
        )
        added_values = [value for value in current_values if value not in previous_values]
        removed_values = [value for value in previous_values if value not in current_values]
        change = {
            "column": column_name,
            "previous_values": previous_values,
            "selected_values": current_values,
            "available_values": available_values,
        }
        if added_values:
            change["added_values"] = added_values
        if removed_values:
            change["removed_values"] = removed_values
        changes.append(change)

    if not changes:
        return None

    recent_refinement = {
        "changes": changes,
    }
    previous_topic_context = str(previous_query_frame.get("topic_context") or "").strip()
    if previous_topic_context:
        recent_refinement["previous_topic_context"] = previous_topic_context
    return recent_refinement


def _resolve_recent_refinement_followup(
    last_query_frame: dict[str, Any],
    user_text: str,
) -> dict[str, Any] | None:
    if not isinstance(last_query_frame, dict) or not user_text or not user_text.strip():
        return None

    recent_refinement = last_query_frame.get("recent_refinement")
    if not isinstance(recent_refinement, dict):
        return None

    changes = [
        entry
        for entry in recent_refinement.get("changes") or []
        if isinstance(entry, dict)
    ]
    if len(changes) != 1:
        return None

    change = changes[0]
    target_column = str(change.get("column") or "").strip()
    if not target_column:
        return None

    filter_entry = next(
        (
            entry
            for entry in last_query_frame.get("categorical_filters") or []
            if isinstance(entry, dict) and str(entry.get("column") or "").strip() == target_column
        ),
        None,
    )
    if filter_entry is None:
        return None

    current_selected_values = _ordered_unique_values(
        [
            value
            for value in filter_entry.get("selected_values") or []
            if isinstance(value, str) and value.strip()
        ]
    )
    available_values = _ordered_unique_values(
        [
            value
            for value in filter_entry.get("available_values") or []
            if isinstance(value, str) and value.strip()
        ]
    )
    if not current_selected_values or not available_values:
        return None

    reply_tokens = set(re.findall(r"[a-z]+", user_text.casefold()))
    has_add_hint = bool(reply_tokens & _GENERIC_ADD_HINT_TOKENS)
    has_remove_hint = bool(reply_tokens & _GENERIC_REMOVE_HINT_TOKENS)
    explicit_values = _extract_matching_clarification_options(user_text, available_values)

    updated_values: list[str] | None = None
    if explicit_values:
        if has_add_hint:
            updated_values = _ordered_unique_values(current_selected_values + explicit_values)
        elif has_remove_hint:
            updated_values = [
                value for value in current_selected_values if value not in explicit_values
            ]
    elif has_add_hint:
        removed_values = _ordered_unique_values(
            [
                value
                for value in change.get("removed_values") or []
                if isinstance(value, str) and value.strip()
            ]
        )
        if removed_values:
            updated_values = _ordered_unique_values(current_selected_values + removed_values)
    elif has_remove_hint:
        added_values = _ordered_unique_values(
            [
                value
                for value in change.get("added_values") or []
                if isinstance(value, str) and value.strip()
            ]
        )
        if added_values:
            updated_values = [
                value for value in current_selected_values if value not in added_values
            ]

    if not updated_values:
        return None

    normalized_updated_values = [
        value for value in available_values if value in updated_values
    ]
    if not normalized_updated_values or normalized_updated_values == current_selected_values:
        return None

    return {
        "resolution_type": "refine_query",
        "target_column": target_column,
        "selected_values": normalized_updated_values,
        "refinement_request": "",
    }


def _mark_public_query_result_rendered(state: Any) -> None:
    if state is None:
        return
    state[SQL_PUBLIC_RESULT_RENDERED_STATE_KEY] = True


def _public_query_result_was_rendered(state: Any) -> bool:
    if state is None or not hasattr(state, "get"):
        return False
    return bool(state.get(SQL_PUBLIC_RESULT_RENDERED_STATE_KEY))


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


def _dedupe_preserving_order(values: list[str]) -> list[str]:
    deduped_values: list[str] = []
    seen_values: set[str] = set()
    for value in values:
        normalized_value = str(value or "").strip().casefold()
        if not normalized_value or normalized_value in seen_values:
            continue
        seen_values.add(normalized_value)
        deduped_values.append(str(value).strip())
    return deduped_values


def _singularize_grounding_token(token: str) -> str:
    normalized_token = token.strip()
    if len(normalized_token) <= 3:
        return normalized_token
    if normalized_token.endswith("ies") and len(normalized_token) > 4:
        return normalized_token[:-3] + "y"
    if normalized_token.endswith("ses") and len(normalized_token) > 4:
        return normalized_token[:-2]
    if normalized_token.endswith("s") and not normalized_token.endswith("ss"):
        return normalized_token[:-1]
    return normalized_token


def _build_grounding_variants(value: str) -> list[str]:
    normalized_value = _normalize_match_text(value)
    if not normalized_value:
        return []

    tokens = [token for token in normalized_value.split() if token]
    singular_tokens = [_singularize_grounding_token(token) for token in tokens]
    variants = [normalized_value]
    if tokens:
        variants.extend(tokens)
    if singular_tokens:
        variants.extend(singular_tokens)
        singular_phrase = " ".join(token for token in singular_tokens if token)
        if singular_phrase:
            variants.append(singular_phrase)
    return _dedupe_preserving_order([variant for variant in variants if variant])


def _collect_grounding_match_evidence(
    user_text: str,
    evidence_texts: list[tuple[str, str]],
) -> tuple[str, float, list[str]]:
    normalized_user_text = _normalize_match_text(user_text)
    if not normalized_user_text:
        return "", 0.0, []

    matched_phrase = ""
    confidence = 0.0
    evidence_sources: list[str] = []
    for source_name, evidence_text in evidence_texts:
        if not evidence_text:
            continue
        for variant in _build_grounding_variants(evidence_text):
            if not variant:
                continue
            if re.search(rf"(?<!\w){re.escape(variant)}(?!\w)", normalized_user_text):
                if not matched_phrase:
                    matched_phrase = variant
                if source_name not in evidence_sources:
                    evidence_sources.append(source_name)
                if source_name == "candidate_value":
                    confidence = max(confidence, 0.95)
                else:
                    confidence = max(confidence, 0.6)
                break
    return matched_phrase, confidence, evidence_sources


def _humanize_schema_label(value: str) -> str:
    candidate = str(value or "").strip()
    if not candidate:
        return ""
    candidate = candidate.replace("_", " ")
    candidate = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", candidate)
    candidate = re.sub(r"\bId\b", "ID", candidate)
    candidate = re.sub(r"\s+", " ", candidate).strip()
    return candidate


def _normalize_allowed_values(values: list[str], allowed_values: list[str]) -> list[str]:
    value_lookup: dict[str, str] = {}
    for allowed_value in allowed_values:
        normalized_allowed_value = _normalize_match_text(allowed_value)
        if normalized_allowed_value and normalized_allowed_value not in value_lookup:
            value_lookup[normalized_allowed_value] = allowed_value

    normalized_values: list[str] = []
    seen_values: set[str] = set()
    for value in values:
        canonical_value = value_lookup.get(_normalize_match_text(value))
        if canonical_value and canonical_value not in seen_values:
            seen_values.add(canonical_value)
            normalized_values.append(canonical_value)
    return normalized_values


def _extract_request_field_glossary(user_text: str) -> dict[str, str]:
    glossary: dict[str, str] = {}
    in_field_glossary = False

    for raw_line in user_text.splitlines():
        line = raw_line.strip()
        if not in_field_glossary:
            if line.casefold() == "field glossary:":
                in_field_glossary = True
            continue

        if not line:
            if glossary:
                break
            continue

        bullet_match = re.match(r"^(?:[-*•])\s*(.+)$", line)
        if bullet_match is None:
            if glossary:
                break
            continue

        entry_text = bullet_match.group(1).strip()
        column_name = ""
        label_text = ""
        if ":" in entry_text:
            raw_column_name, raw_label_text = entry_text.split(":", 1)
            column_name = raw_column_name.strip()
            label_text = raw_label_text.strip()
        else:
            name_match = re.match(r"^(?P<column>[A-Za-z_][A-Za-z0-9_]*)", entry_text)
            if name_match is not None:
                column_name = name_match.group("column").strip()
                label_text = column_name

        if not column_name:
            continue

        normalized_label = re.sub(
            r"\s*\((?:categorical|date|numeric|other)\)\s*$",
            "",
            label_text,
            flags=re.IGNORECASE,
        ).strip()
        glossary[column_name.casefold()] = normalized_label or column_name

    return glossary


def _extract_user_turn_texts(llm_request) -> tuple[str, str]:
    raw_user_text = _extract_last_user_text(llm_request) if llm_request is not None else ""
    effective_user_text = _extract_topic_context_text(raw_user_text)
    return raw_user_text, effective_user_text or raw_user_text.strip()


def _iter_schema_grounding_columns(schema_summary: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
    tables = schema_summary.get("tables") or []
    if not isinstance(tables, list):
        return []

    name_counts: dict[str, int] = {}
    for table in tables:
        if not isinstance(table, dict):
            continue
        for column in table.get("columns") or []:
            if not isinstance(column, dict):
                continue
            column_name = str(column.get("name") or "").strip()
            if not column_name:
                continue
            normalized_column_name = column_name.casefold()
            name_counts[normalized_column_name] = name_counts.get(normalized_column_name, 0) + 1

    schema_columns: list[tuple[str, str, dict[str, Any]]] = []
    for table in tables:
        if not isinstance(table, dict):
            continue
        table_name = str(table.get("name") or "").strip()
        for column in table.get("columns") or []:
            if not isinstance(column, dict):
                continue
            column_name = str(column.get("name") or "").strip()
            if not column_name:
                continue

            identifier = column_name
            if name_counts.get(column_name.casefold(), 0) > 1 and table_name:
                identifier = f"{table_name}.{column_name}"
            schema_columns.append((identifier, column_name, column))

    return schema_columns


def _normalize_schema_grounding_filters(
    grounded_filters: Any,
    candidate_values_by_column: dict[str, list[str]],
) -> dict[str, list[str]]:
    if not isinstance(candidate_values_by_column, dict):
        return {}

    identifier_lookup = {
        identifier.casefold(): identifier
        for identifier in candidate_values_by_column
        if isinstance(identifier, str) and identifier.strip()
    }

    filter_items: list[tuple[Any, Any]] = []
    if isinstance(grounded_filters, dict):
        filter_items = list(grounded_filters.items())
    elif isinstance(grounded_filters, list):
        for entry in grounded_filters:
            if not isinstance(entry, dict):
                continue
            filter_items.append((entry.get("column"), entry.get("selected_values")))

    normalized_filters: dict[str, list[str]] = {}
    for column_name, selected_values in filter_items:
        if not isinstance(column_name, str):
            continue
        canonical_identifier = identifier_lookup.get(column_name.strip().casefold())
        if not canonical_identifier:
            continue

        allowed_values = [
            value
            for value in candidate_values_by_column.get(canonical_identifier) or []
            if isinstance(value, str) and value.strip()
        ]
        if not allowed_values:
            continue

        raw_values = [
            value
            for value in (selected_values or [])
            if isinstance(value, str) and value.strip()
        ]
        normalized_values = _normalize_allowed_values(raw_values, allowed_values)
        if not normalized_values or len(normalized_values) >= len(allowed_values):
            continue
        normalized_filters[canonical_identifier] = normalized_values

    return normalized_filters


def _extract_sql_string_literals(sql_fragment: str) -> list[str]:
    return [
        match.group(1).replace("''", "'")
        for match in re.finditer(r"'((?:''|[^'])*)'", sql_fragment)
    ]


def _mask_char_for_sql_filter_extraction(char: str) -> str:
    return "\n" if char == "\n" else " "


def _mask_case_expressions(sql: str) -> str:
    masked = list(sql)
    state = "normal"
    case_depth = 0
    index = 0

    while index < len(sql):
        char = sql[index]
        next_char = sql[index + 1] if index + 1 < len(sql) else ""

        if state == "line_comment":
            if case_depth > 0:
                masked[index] = _mask_char_for_sql_filter_extraction(char)
            if char == "\n":
                state = "normal"
            index += 1
            continue

        if state == "block_comment":
            if case_depth > 0:
                masked[index] = _mask_char_for_sql_filter_extraction(char)
            if char == "*" and next_char == "/":
                if case_depth > 0:
                    masked[index + 1] = _mask_char_for_sql_filter_extraction(next_char)
                state = "normal"
                index += 2
                continue
            index += 1
            continue

        if state == "single_quote":
            if case_depth > 0:
                masked[index] = _mask_char_for_sql_filter_extraction(char)
            if char == "'" and next_char == "'":
                if case_depth > 0:
                    masked[index + 1] = _mask_char_for_sql_filter_extraction(next_char)
                index += 2
                continue
            if char == "'":
                state = "normal"
            index += 1
            continue

        if state == "double_quote":
            if case_depth > 0:
                masked[index] = _mask_char_for_sql_filter_extraction(char)
            if char == '"':
                state = "normal"
            index += 1
            continue

        if state == "backtick":
            if case_depth > 0:
                masked[index] = _mask_char_for_sql_filter_extraction(char)
            if char == "`":
                state = "normal"
            index += 1
            continue

        if state == "bracket":
            if case_depth > 0:
                masked[index] = _mask_char_for_sql_filter_extraction(char)
            if char == "]":
                state = "normal"
            index += 1
            continue

        if char == "-" and next_char == "-":
            if case_depth > 0:
                masked[index] = _mask_char_for_sql_filter_extraction(char)
                masked[index + 1] = _mask_char_for_sql_filter_extraction(next_char)
            state = "line_comment"
            index += 2
            continue

        if char == "/" and next_char == "*":
            if case_depth > 0:
                masked[index] = _mask_char_for_sql_filter_extraction(char)
                masked[index + 1] = _mask_char_for_sql_filter_extraction(next_char)
            state = "block_comment"
            index += 2
            continue

        if char == "'":
            if case_depth > 0:
                masked[index] = _mask_char_for_sql_filter_extraction(char)
            state = "single_quote"
            index += 1
            continue

        if char == '"':
            if case_depth > 0:
                masked[index] = _mask_char_for_sql_filter_extraction(char)
            state = "double_quote"
            index += 1
            continue

        if char == "`":
            if case_depth > 0:
                masked[index] = _mask_char_for_sql_filter_extraction(char)
            state = "backtick"
            index += 1
            continue

        if char == "[":
            if case_depth > 0:
                masked[index] = _mask_char_for_sql_filter_extraction(char)
            state = "bracket"
            index += 1
            continue

        if char.isalpha() or char == "_":
            token_start = index
            index += 1
            while index < len(sql) and (sql[index].isalnum() or sql[index] == "_"):
                index += 1
            token = sql[token_start:index]
            token_upper = token.upper()
            if token_upper == "CASE":
                case_depth += 1
            if case_depth > 0:
                for mask_index in range(token_start, index):
                    masked[mask_index] = _mask_char_for_sql_filter_extraction(sql[mask_index])
            if token_upper == "END" and case_depth > 0:
                case_depth -= 1
            continue

        if case_depth > 0:
            masked[index] = _mask_char_for_sql_filter_extraction(char)

        index += 1

    return "".join(masked)


def _normalize_sql_identifier(value: str) -> str:
    identifier = value.strip()
    if "." in identifier:
        identifier = identifier.rsplit(".", 1)[-1]
    if identifier.startswith('"') and identifier.endswith('"') and len(identifier) >= 2:
        identifier = identifier[1:-1]
    return identifier.strip()


def _normalize_sql_literal(value: str) -> str:
    literal = value.strip()
    if literal.startswith("'") and literal.endswith("'") and len(literal) >= 2:
        literal = literal[1:-1].replace("''", "'")
    return literal.strip()


def _extract_categorical_filters_from_sql(
    sql: str,
    categorical_value_guidance: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    searchable_sql = _mask_case_expressions(sql)
    filters: list[dict[str, Any]] = []
    for entry in categorical_value_guidance:
        column_name = str(entry.get("column") or "").strip()
        available_values = [
            value.strip()
            for value in entry.get("values") or []
            if isinstance(value, str) and value.strip()
        ]
        if not column_name or not available_values:
            continue

        quoted_or_bare_column = rf'(?<!\w)(?:"{re.escape(column_name)}"|{re.escape(column_name)})(?!\w)'
        selected_values: list[str] = []

        in_match = re.search(
            rf"{quoted_or_bare_column}\s+IN\s*\((?P<values>[^)]*)\)",
            searchable_sql,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if in_match is not None:
            selected_values = _normalize_allowed_values(
                _extract_sql_string_literals(in_match.group("values")),
                available_values,
            )

        if not selected_values:
            equality_values = [
                match.group(1).replace("''", "'")
                for match in re.finditer(
                    rf"{quoted_or_bare_column}\s*=\s*'((?:''|[^'])*)'",
                    searchable_sql,
                    flags=re.IGNORECASE,
                )
            ]
            selected_values = _normalize_allowed_values(equality_values, available_values)

        if not selected_values:
            continue

        filters.append(
            {
                "column": column_name,
                "selected_values": selected_values,
                "available_values": available_values,
            }
        )

    return filters


def _extract_comparison_filters_from_sql(
    sql: str,
    *,
    excluded_columns: set[str] | None = None,
) -> list[dict[str, str]]:
    searchable_sql = _mask_case_expressions(sql)
    comparison_filters: list[dict[str, str]] = []
    seen_filters: set[tuple[str, str, str]] = set()
    skipped_columns = {column.casefold() for column in (excluded_columns or set()) if column}
    identifier_pattern = r'(?:"[^"]+"|[A-Za-z_][A-Za-z0-9_]*)(?:\.(?:"[^"]+"|[A-Za-z_][A-Za-z0-9_]*))?'
    value_pattern = r"-?\d+(?:\.\d+)?|NULL|'(?:(?:'')|[^'])*'"

    for match in re.finditer(
        rf'(?:(?:CAST\(\s*(?P<cast_column>{identifier_pattern})\s+AS\s+[A-Za-z_][A-Za-z0-9_]*\s*\))|(?P<column>{identifier_pattern}))\s*'
        rf'(?P<operator>>=|<=|<>|!=|=|>|<)\s*'
        rf'(?P<value>{value_pattern})',
        searchable_sql,
        flags=re.IGNORECASE,
    ):
        column_name = _normalize_sql_identifier(match.group("cast_column") or match.group("column") or "")
        if not column_name or column_name.casefold() in skipped_columns:
            continue

        normalized_filter = (
            column_name,
            match.group("operator"),
            _normalize_sql_literal(match.group("value")),
        )
        if normalized_filter in seen_filters:
            continue
        seen_filters.add(normalized_filter)
        comparison_filters.append(
            {
                "column": normalized_filter[0],
                "operator": normalized_filter[1],
                "value": normalized_filter[2],
            }
        )

    return comparison_filters


def _build_last_query_frame(
    state: Any,
    args: dict[str, Any],
    tool_response: dict[str, Any],
    categorical_value_guidance: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if state is None or not hasattr(state, "get"):
        return None

    active_query_topic = state.get(SQL_ACTIVE_QUERY_TOPIC_STATE_KEY)
    if not isinstance(active_query_topic, str) or not active_query_topic.strip():
        active_query_topic = state.get(SQL_LAST_USER_TEXT_STATE_KEY)
    if not isinstance(active_query_topic, str) or not active_query_topic.strip():
        return None

    raw_sql = args.get("sql") if isinstance(args, dict) else None
    if not isinstance(raw_sql, str) or not raw_sql.strip():
        raw_sql = tool_response.get("sql")
    if not isinstance(raw_sql, str) or not raw_sql.strip():
        return None

    query_frame = {
        "question": active_query_topic.strip(),
        "sql": raw_sql.strip(),
    }
    categorical_filters = _extract_categorical_filters_from_sql(raw_sql, categorical_value_guidance)
    if categorical_filters:
        query_frame["categorical_filters"] = categorical_filters
    comparison_filters = _extract_comparison_filters_from_sql(
        raw_sql,
        excluded_columns={
            str(entry.get("column") or "").strip()
            for entry in categorical_filters
            if isinstance(entry, dict)
        },
    )
    if comparison_filters:
        query_frame["comparison_filters"] = comparison_filters
    topic_context = _build_query_frame_topic_context(query_frame)
    if topic_context:
        query_frame["topic_context"] = topic_context
    return query_frame


def _coerce_bool_argument(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if not normalized:
            return default
        return normalized not in {"0", "false", "no", "off"}
    return bool(value)


def _tool_call_is_final(args: dict[str, Any] | None) -> bool:
    if not isinstance(args, dict):
        return True
    return _coerce_bool_argument(args.get("is_final"), default=True)


def _extract_matching_clarification_options(
    user_text: str,
    options: list[str],
) -> list[str]:
    normalized_user_text = _normalize_match_text(user_text)
    if not normalized_user_text:
        return []

    matches: list[str] = []
    seen_matches: set[str] = set()
    for option in options:
        normalized_option = _normalize_match_text(option)
        if not normalized_option:
            continue
        if re.search(rf"(?<!\w){re.escape(normalized_option)}(?!\w)", normalized_user_text):
            seen_matches.add(option)
            matches.append(option)

    if options and re.search(r"\d", user_text):
        allowed_words = {
            "all",
            "and",
            "both",
            "choose",
            "for",
            "go",
            "i",
            "include",
            "just",
            "number",
            "numbers",
            "only",
            "option",
            "options",
            "or",
            "pick",
            "please",
            "select",
            "too",
            "use",
            "value",
            "values",
            "want",
            "with",
        }
        reply_words = re.findall(r"[a-z]+", user_text.casefold())
        if all(word in allowed_words for word in reply_words):
            selected_indexes: list[int] = []
            seen_indexes: set[int] = set()
            for raw_number in re.findall(r"\d+", user_text):
                index = int(raw_number)
                if index < 1 or index > len(options):
                    selected_indexes = []
                    break
                if index not in seen_indexes:
                    seen_indexes.add(index)
                    selected_indexes.append(index)
            for index in selected_indexes:
                option = options[index - 1]
                if option not in seen_matches:
                    seen_matches.add(option)
                    matches.append(option)
    return matches


def _prune_topic_context_option(
    clarification: dict[str, Any],
    topic_context: str | None,
) -> dict[str, Any]:
    if not topic_context or not topic_context.strip():
        return clarification

    normalized_topic_context = _normalize_match_text(topic_context)
    if not normalized_topic_context:
        return clarification

    options = clarification.get("options") or []
    pruned_options = [
        option
        for option in options
        if isinstance(option, str) and _normalize_match_text(option) != normalized_topic_context
    ]
    if len(pruned_options) == len(options):
        return clarification

    normalized_clarification = dict(clarification)
    normalized_clarification["options"] = pruned_options
    return normalized_clarification


def _build_clarification_guidance_match_text(clarification: dict[str, Any]) -> str:
    if not isinstance(clarification, dict):
        return ""

    text_parts: list[str] = []
    user_message = str(clarification.get("user_message") or "").strip()
    if user_message:
        text_parts.append(user_message)

    text_parts.extend(
        option.strip()
        for option in clarification.get("options") or []
        if isinstance(option, str) and option.strip()
    )
    return "\n".join(text_parts).strip()


def _match_clarification_values_to_schema_guidance(
    clarification: dict[str, Any],
    raw_text: str,
    categorical_value_guidance: list[dict[str, Any]],
) -> dict[str, Any]:
    clarification_kind = str(clarification.get("clarification_kind") or "").strip().lower()
    if clarification_kind == CLARIFICATION_KIND_INTERPRETATION:
        return clarification

    if not categorical_value_guidance:
        if clarification_kind:
            return clarification
        normalized_clarification = dict(clarification)
        normalized_clarification["clarification_kind"] = CLARIFICATION_KIND_GENERIC
        return normalized_clarification

    normalized_raw_text = _normalize_match_text(raw_text)
    normalized_option_keys = {
        _normalize_match_text(option)
        for option in clarification.get("options") or []
        if isinstance(option, str) and option.strip()
    }

    best_values: list[str] | None = None
    best_score = (0, 0)
    ambiguous_best_match = False
    for entry in categorical_value_guidance:
        values = [
            value.strip()
            for value in entry.get("values") or []
            if isinstance(value, str) and value.strip()
        ]
        if not values:
            continue

        overlap_count = 0
        for value in values:
            normalized_value = _normalize_match_text(value)
            if not normalized_value:
                continue
            if normalized_value in normalized_option_keys:
                overlap_count += 1
                continue
            if re.search(rf"(?<!\w){re.escape(normalized_value)}(?!\w)", normalized_raw_text):
                overlap_count += 1

        if overlap_count == 0:
            continue

        normalized_column_name = _normalize_match_text(str(entry.get("column") or ""))
        column_mentioned = bool(
            normalized_column_name
            and re.search(rf"(?<!\w){re.escape(normalized_column_name)}(?!\w)", normalized_raw_text)
        )
        score = (overlap_count, int(column_mentioned))
        if score > best_score:
            best_score = score
            best_values = values
            ambiguous_best_match = False
        elif score == best_score:
            ambiguous_best_match = True

    if best_values is None or ambiguous_best_match:
        if clarification_kind:
            return clarification
        normalized_clarification = dict(clarification)
        normalized_clarification["clarification_kind"] = CLARIFICATION_KIND_GENERIC
        return normalized_clarification

    normalized_clarification = dict(clarification)
    normalized_clarification["options"] = best_values
    normalized_clarification["clarification_kind"] = CLARIFICATION_KIND_CATEGORICAL_VALUES
    return normalized_clarification


def _build_schema_grounding_catalog(
    schema_summary: dict[str, Any],
    *,
    request_field_glossary: dict[str, str] | None = None,
    grounded_filters: dict[str, list[str]] | None = None,
    user_text: str = "",
) -> dict[str, Any]:
    schema_columns = _iter_schema_grounding_columns(schema_summary)
    if not schema_columns:
        return {
            "context_text": "",
            "candidate_columns": [],
            "option_labels": {},
            "grounding_candidates": [],
            "candidate_values_by_column": {},
        }

    context_lines: list[str] = []
    active_glossary = request_field_glossary or {}
    active_grounded_filters = grounded_filters or {}
    if active_grounded_filters:
        context_lines.append("Already grounded categorical filters from the user wording:")
        for identifier, selected_values in active_grounded_filters.items():
            normalized_values = [
                value.strip()
                for value in selected_values
                if isinstance(value, str) and value.strip()
            ]
            if not normalized_values:
                continue
            context_lines.append(f"- {identifier} = {', '.join(normalized_values)}")

    candidate_columns: list[str] = []
    option_labels: dict[str, str] = {}
    grounding_candidates: list[dict[str, Any]] = []
    candidate_values_by_column: dict[str, list[str]] = {}
    used_option_label_keys: set[str] = set()
    for identifier, column_name, column in schema_columns:
        declared_type = str(column.get("type") or "TEXT").strip() or "TEXT"
        categorical_values = [
            value.strip()
            for value in column.get("categorical_values") or []
            if isinstance(value, str) and value.strip()
        ]
        candidate_values_by_column[identifier] = categorical_values
        request_field_label = _humanize_schema_label(
            active_glossary.get(column_name.casefold())
            or active_glossary.get(identifier.casefold())
            or ""
        )
        source_header = _humanize_schema_label(str(column.get("source_header") or ""))
        column_label = _humanize_schema_label(column_name) or identifier

        if categorical_values:
            for categorical_value in categorical_values:
                matched_phrase, confidence, lexical_sources = _collect_grounding_match_evidence(
                    user_text,
                    [
                        ("column_label", column_label),
                        ("request_field_label", request_field_label),
                        ("source_header", source_header),
                        ("candidate_value", categorical_value),
                    ],
                )
                evidence_sources = _dedupe_preserving_order(
                    [
                        source_name
                        for source_name, evidence_text in (
                            ("column_label", column_label),
                            ("request_field_label", request_field_label),
                            ("source_header", source_header),
                            ("candidate_value", categorical_value),
                        )
                        if evidence_text
                    ]
                    + lexical_sources
                )
                grounding_candidates.append(
                    {
                        "column": identifier,
                        "column_name": column_name,
                        "field_label": request_field_label or source_header or column_label,
                        "source_header": source_header,
                        "request_field_label": request_field_label,
                        "candidate_value": categorical_value,
                        "candidate_value_variants": _build_grounding_variants(categorical_value),
                        "matched_user_phrase": matched_phrase,
                        "confidence": confidence,
                        "evidence_sources": evidence_sources,
                    }
                )

        if identifier in active_grounded_filters:
            continue

        candidate_columns.append(identifier)
        line = f"- {identifier} ({declared_type})"
        line_details: list[str] = []
        if request_field_label:
            line_details.append(f"field label = {request_field_label}")
        elif source_header:
            line_details.append(f"dataset label = {source_header}")
        if categorical_values:
            preview_values = categorical_values[:4]
            preview_text = ", ".join(preview_values)
            if len(categorical_values) > len(preview_values):
                preview_text += ", ..."
            line_details.append(f"categorical values = {preview_text}")
        if line_details:
            line += ": " + "; ".join(line_details)

        option_label = ""
        if request_field_label:
            option_label = request_field_label
            if option_label.casefold() in used_option_label_keys:
                option_label = f"{request_field_label} ({identifier})"
        elif source_header:
            option_label = source_header
            if option_label.casefold() in used_option_label_keys:
                option_label = f"{source_header} ({identifier})"
        elif categorical_values:
            option_label = f"{identifier} (values: {preview_text})"
        else:
            option_label = identifier

        option_labels[identifier] = option_label
        used_option_label_keys.add(option_label.casefold())
        context_lines.append(line)

    return {
        "context_text": "\n".join(context_lines),
        "candidate_columns": candidate_columns,
        "option_labels": option_labels,
        "grounding_candidates": grounding_candidates,
        "candidate_values_by_column": candidate_values_by_column,
    }


def _build_schema_grounding_clarification(
    topic_context: str,
    candidate_columns: list[str],
    schema_grounding_catalog: dict[str, Any],
    *,
    grounded_filters: dict[str, list[str]] | None = None,
) -> dict[str, Any] | None:
    option_labels = schema_grounding_catalog.get("option_labels") or {}
    selected_options: list[str] = []
    option_columns: dict[str, str] = {}
    for candidate_column in candidate_columns:
        option_label = option_labels.get(candidate_column)
        if not isinstance(option_label, str) or not option_label.strip():
            continue
        if option_label in option_columns:
            continue
        selected_options.append(option_label)
        option_columns[option_label] = candidate_column

    if len(selected_options) < 2:
        return None

    clarification = build_clarification_response(
        "I found more than one nearby schema field for this request. Which one do you mean?",
        selected_options,
        clarification_kind=CLARIFICATION_KIND_INTERPRETATION,
    )
    if topic_context:
        clarification["topic_context"] = topic_context
    clarification["option_columns"] = option_columns
    if grounded_filters:
        clarification["grounded_filters"] = {
            identifier: [
                value
                for value in selected_values
                if isinstance(value, str) and value.strip()
            ]
            for identifier, selected_values in grounded_filters.items()
        }
    return clarification


def _resolve_pending_clarification_reply(
    resolver,
    clarification: dict[str, Any],
    user_text: str,
) -> dict[str, Any]:
    if not user_text or not user_text.strip():
        return {
            "resolution_type": "custom_rule",
            "selected_options": [],
            "custom_rule": "",
        }

    topic_context = clarification.get("query_context") or clarification.get("topic_context")
    clarification_question = clarification.get("user_message")
    options = clarification.get("options") or []
    return resolver(
        topic_context if isinstance(topic_context, str) else "",
        clarification_question if isinstance(clarification_question, str) else "",
        [
            option
            for option in options
            if isinstance(option, str) and option.strip()
        ],
        user_text,
    )


def _build_result_refinement_clarification(
    last_query_frame: dict[str, Any],
    target_column: str,
) -> dict[str, Any] | None:
    if not target_column:
        return None

    filter_entry = next(
        (
            entry
            for entry in (last_query_frame.get("categorical_filters") or [])
            if isinstance(entry, dict) and str(entry.get("column") or "").strip() == target_column
        ),
        None,
    )
    if filter_entry is None:
        return None

    available_values = [
        value
        for value in filter_entry.get("available_values") or []
        if isinstance(value, str) and value.strip()
    ]
    if not available_values:
        return None

    selected_values = [
        value
        for value in filter_entry.get("selected_values") or []
        if isinstance(value, str) and value.strip()
    ]
    previous_question = _get_query_frame_question_text(last_query_frame)

    message_parts = []
    if previous_question:
        message_parts.append(f"Your previous question was: {previous_question}.")
    if selected_values:
        message_parts.append(
            f"I previously used {target_column} = {', '.join(selected_values)}."
        )
    message_parts.append(f"Which values from {target_column} should I include now?")

    clarification = build_clarification_response(
        " ".join(message_parts),
        available_values,
        clarification_kind=CLARIFICATION_KIND_CATEGORICAL_VALUES,
    )
    if previous_question:
        clarification["topic_context"] = previous_question
    query_context = str(last_query_frame.get("topic_context") or "").strip()
    if query_context:
        clarification["query_context"] = query_context
    clarification["base_query_frame"] = copy.deepcopy(last_query_frame)
    return clarification


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


def _apply_last_query_refinement_followup(
    llm_request,
    last_query_frame: dict[str, Any],
    user_text: str,
    *,
    target_column: str | None = None,
    selected_values: list[str] | None = None,
    refinement_request: str | None = None,
) -> bool:
    previous_question = str(last_query_frame.get("question") or "").strip()
    previous_sql = str(last_query_frame.get("sql") or "").strip()
    if not previous_question and not previous_sql:
        return False

    rewritten_sections = [
        "The user is refining the previous dataset request rather than starting a new question.",
    ]
    if previous_question:
        rewritten_sections.append(f"Previous dataset question: {previous_question}")
    if previous_sql:
        rewritten_sections.append(f"Previous SQL:\n{previous_sql}")

    categorical_filters = [
        entry
        for entry in (last_query_frame.get("categorical_filters") or [])
        if isinstance(entry, dict)
    ]
    if categorical_filters:
        filter_lines = []
        for entry in categorical_filters:
            column_name = str(entry.get("column") or "").strip()
            if not column_name:
                continue
            selected_text = ", ".join(entry.get("selected_values") or []) or "[none]"
            available_text = ", ".join(entry.get("available_values") or []) or "[none]"
            filter_lines.append(
                f"- {column_name}: selected values = {selected_text}; available dataset values = {available_text}"
            )
        if filter_lines:
            rewritten_sections.append(
                "Categorical filters from the previous query:\n" + "\n".join(filter_lines)
            )

    if target_column and selected_values:
        rewritten_sections.append(
            f"Use this updated value set for {target_column}: {', '.join(selected_values)}"
        )

    normalized_refinement_request = refinement_request.strip() if isinstance(refinement_request, str) else ""
    rewritten_sections.append(
        f"Latest same-query refinement request: {normalized_refinement_request or user_text}"
    )
    rewritten_sections.append(
        "Apply the refinement to the previous dataset question and continue from there."
    )

    return _replace_last_user_text(llm_request, "\n\n".join(rewritten_sections))


def _apply_pending_clarification_followup(llm_request, clarification: dict[str, Any]) -> bool:
    _raw_user_text, user_text = _extract_user_turn_texts(llm_request)
    if not user_text:
        return False

    return _apply_pending_clarification_followup_with_resolution(llm_request, clarification, user_text)


def _apply_pending_clarification_followup_with_resolution(
    llm_request,
    clarification: dict[str, Any],
    user_text: str,
    *,
    matched_options: list[str] | None = None,
    custom_rule: str | None = None,
) -> bool:
    if not user_text:
        return False

    options = [
        option
        for option in clarification.get("options") or []
        if isinstance(option, str) and option.strip()
    ]
    resolved_options = [
        option for option in (matched_options or []) if option in options
    ] or _extract_matching_clarification_options(user_text, options)
    normalized_custom_rule = custom_rule.strip() if isinstance(custom_rule, str) else ""

    rewritten_sections = [
        "The user is replying to the previous clarification for the same dataset request.",
        f"Clarification question: {clarification.get('user_message') or ''}",
    ]
    if options:
        rewritten_sections.append(
            "Available options:\n" + "\n".join(f"- {option}" for option in options)
        )
    grounded_filters = clarification.get("grounded_filters")
    if isinstance(grounded_filters, dict):
        grounded_lines = []
        for column_name, selected_values in grounded_filters.items():
            normalized_values = [
                value
                for value in selected_values or []
                if isinstance(value, str) and value.strip()
            ]
            if not isinstance(column_name, str) or not column_name.strip() or not normalized_values:
                continue
            grounded_lines.append(f"- {column_name} = {', '.join(normalized_values)}")
        if grounded_lines:
            rewritten_sections.append(
                "Already grounded categorical filters from the original request:\n"
                + "\n".join(grounded_lines)
            )
    if resolved_options:
        rewritten_sections.append(
            "Matched options from the reply: " + ", ".join(resolved_options)
        )
        option_columns = clarification.get("option_columns")
        if isinstance(option_columns, dict):
            resolved_columns: list[str] = []
            seen_columns: set[str] = set()
            for resolved_option in resolved_options:
                resolved_column = str(option_columns.get(resolved_option) or "").strip()
                if not resolved_column or resolved_column in seen_columns:
                    continue
                seen_columns.add(resolved_column)
                resolved_columns.append(resolved_column)
            if resolved_columns:
                label = "Resolved schema field from the reply: "
                if len(resolved_columns) > 1:
                    label = "Resolved schema fields from the reply: "
                rewritten_sections.append(label + ", ".join(resolved_columns))
    if normalized_custom_rule:
        rewritten_sections.append(
            "Resolved custom rule from the reply: " + normalized_custom_rule
        )
    rewritten_sections.append(f"User clarification reply: {user_text}")
    rewritten_sections.append(
        "Interpret this as clarification for the prior dataset question and continue from there."
    )

    return _replace_last_user_text(llm_request, "\n\n".join(rewritten_sections))


def _apply_grounded_filter_followup(
    llm_request,
    user_text: str,
    grounded_filters: dict[str, list[str]],
) -> bool:
    if not user_text:
        return False

    grounded_lines: list[str] = []
    for column_name, selected_values in grounded_filters.items():
        normalized_values = [
            value
            for value in selected_values or []
            if isinstance(value, str) and value.strip()
        ]
        if not isinstance(column_name, str) or not column_name.strip() or not normalized_values:
            continue
        grounded_lines.append(f"- {column_name} = {', '.join(normalized_values)}")

    if not grounded_lines:
        return False

    rewritten_sections = [
        "Answer the user's current dataset question below. This is the current question for this turn, not background context.",
        f"Current dataset question: {user_text}",
        "Grounded categorical filters already implied by the same question:\n" + "\n".join(grounded_lines),
        "Use these grounded filters when answering the current dataset question.",
    ]
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


def _build_grouped_public_result(
    tool_response: dict[str, Any],
    columns: list[str],
    public_rows: list[dict[str, Any]],
    normalized_count_values: list[int],
    minimum_aggregate_count: int,
    *,
    public_result_kind: str,
    aggregate_columns: list[str] | None = None,
) -> dict[str, Any]:
    safe_rows = [
        row
        for row, matching_count in zip(public_rows, normalized_count_values)
        if matching_count >= minimum_aggregate_count
    ]
    if not safe_rows:
        return _build_privacy_error_result(
            tool_response,
            "Privacy guardrail blocked this grouped result because at least one group count is below the minimum threshold.",
            matched_row_count=_normalize_count_value(sum(normalized_count_values)),
        )

    public_result = dict(tool_response)
    public_result["columns"] = columns
    public_result["rows"] = safe_rows
    public_result["row_count"] = len(safe_rows)
    public_result["preview_row_count"] = len(safe_rows)
    public_result["public_result_kind"] = public_result_kind
    if aggregate_columns is not None:
        public_result["aggregate_columns"] = aggregate_columns

    if len(safe_rows) == len(public_rows):
        public_result["matched_row_count"] = _normalize_count_value(sum(normalized_count_values))
    else:
        public_result.pop("matched_row_count", None)
        public_result["grouped_result_suppressed"] = True

    return public_result


def _reload_complete_grouped_result(tool_response: dict[str, Any]) -> dict[str, Any] | None:
    db_path = str(tool_response.get("db_path") or "").strip()
    sql = str(tool_response.get("sql") or "").strip()
    if not db_path or not sql:
        return None

    row_count = _normalize_count_value(tool_response.get("row_count"))
    if not isinstance(row_count, int) or row_count < 1:
        return None

    complete_result = execute_sqlite_query(
        db_path,
        sql,
        preview_rows=row_count,
    )
    if complete_result.get("status") != "success" or complete_result.get("truncated"):
        return None
    return complete_result


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


def _split_top_level_sql_expressions(sql_fragment: str) -> list[str]:
    expressions: list[str] = []
    current: list[str] = []
    state = "normal"
    depth = 0
    index = 0

    while index < len(sql_fragment):
        char = sql_fragment[index]
        next_char = sql_fragment[index + 1] if index + 1 < len(sql_fragment) else ""

        if state == "line_comment":
            current.append(char)
            if char == "\n":
                state = "normal"
            index += 1
            continue

        if state == "block_comment":
            current.append(char)
            if char == "*" and next_char == "/":
                current.append(next_char)
                state = "normal"
                index += 2
                continue
            index += 1
            continue

        if state == "single_quote":
            current.append(char)
            if char == "'" and next_char == "'":
                current.append(next_char)
                index += 2
                continue
            if char == "'":
                state = "normal"
            index += 1
            continue

        if state == "double_quote":
            current.append(char)
            if char == '"':
                state = "normal"
            index += 1
            continue

        if char == "-" and next_char == "-":
            current.append(char)
            current.append(next_char)
            state = "line_comment"
            index += 2
            continue

        if char == "/" and next_char == "*":
            current.append(char)
            current.append(next_char)
            state = "block_comment"
            index += 2
            continue

        if char == "'":
            current.append(char)
            state = "single_quote"
            index += 1
            continue

        if char == '"':
            current.append(char)
            state = "double_quote"
            index += 1
            continue

        if char == "(":
            depth += 1
            current.append(char)
            index += 1
            continue

        if char == ")":
            depth = max(0, depth - 1)
            current.append(char)
            index += 1
            continue

        if char == "," and depth == 0:
            expression = "".join(current).strip()
            if expression:
                expressions.append(expression)
            current = []
            index += 1
            continue

        current.append(char)
        index += 1

    trailing_expression = "".join(current).strip()
    if trailing_expression:
        expressions.append(trailing_expression)

    return expressions


def _find_last_top_level_keyword(sql: str, keyword: str) -> int | None:
    last_position = None
    search_start = 0
    while search_start < len(sql):
        next_position = _find_top_level_keyword(sql[search_start:], keyword)
        if next_position is None:
            break
        absolute_position = search_start + next_position
        last_position = absolute_position
        search_start = absolute_position + len(keyword)
    return last_position


def _extract_select_item_parts(select_item: str) -> tuple[str, str | None]:
    item = select_item.strip()
    if not item:
        return "", None

    as_pos = _find_last_top_level_keyword(item, "AS")
    if as_pos is None:
        return item, None

    expression = item[:as_pos].strip()
    alias_sql = item[as_pos + len("AS"):].strip()
    if not expression or not alias_sql:
        return item, None
    return expression, alias_sql


def _find_group_projection_select_items(sql: str, group_column_names: list[str]) -> list[str] | None:
    select_pos = _find_top_level_keyword(sql, "SELECT")
    from_pos = _find_top_level_keyword(sql, "FROM")
    if select_pos is None or from_pos is None or from_pos <= select_pos:
        return None

    select_clause = sql[select_pos + len("SELECT"):from_pos].strip()
    select_items = _split_top_level_sql_expressions(select_clause)
    if not select_items:
        return None

    projected_items: list[str] = []
    for column_name in group_column_names:
        normalized_column_name = str(column_name or "").strip().casefold()
        if not normalized_column_name:
            return None

        matched_item = None
        for select_item in select_items:
            expression, alias_sql = _extract_select_item_parts(select_item)
            candidate_names = {
                _normalize_sql_identifier(expression).casefold(),
            }
            if alias_sql:
                candidate_names.add(_normalize_sql_identifier(alias_sql).casefold())
            if normalized_column_name in candidate_names:
                matched_item = select_item.strip()
                break

        if matched_item is None:
            return None
        projected_items.append(matched_item)

    return projected_items


def _build_grouped_count_sql(sql: str, group_column_names: list[str] | None = None) -> str | None:
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

    grouped_select_items = None
    if group_column_names:
        grouped_select_items = _find_group_projection_select_items(sql, group_column_names)

    projected_group_sql = ", ".join(grouped_select_items) if grouped_select_items else group_by_clause
    count_sql = f"SELECT {projected_group_sql}, COUNT(*) AS matching_count {from_clause}"
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

    if has_group_by and tool_response.get("truncated"):
        complete_grouped_result = _reload_complete_grouped_result(tool_response)
        if complete_grouped_result is None:
            return _build_privacy_error_result(
                tool_response,
                (
                    "Privacy guardrail blocked this aggregate result because only a preview "
                    "was available, so not every cohort or group could be checked safely."
                ),
            )
        tool_response = complete_grouped_result
        rows = tool_response.get("rows") or []
        columns = tool_response.get("columns") or []
        sql = tool_response.get("sql") or sql

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
            group_column_names = [
                column
                for column in columns
                if column not in aggregate_columns
            ]
            count_sql = _build_grouped_count_sql(sql, group_column_names)
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

            return _build_grouped_public_result(
                tool_response,
                [*group_columns, "matching_count", *remaining_columns],
                public_rows,
                count_values,
                minimum_aggregate_count,
                public_result_kind="safe_aggregate",
                aggregate_columns=aggregate_columns,
            )

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

    normalized_count_values = [_normalize_count_value(row[count_column]) for row in rows]
    if any(value < minimum_aggregate_count for value in normalized_count_values):
        if not is_grouped:
            matching_count = normalized_count_values[0]
            return _build_privacy_error_result(
                tool_response,
                "Privacy guardrail blocked this result because the matching count is below the minimum threshold.",
                matched_row_count=matching_count,
            )
        return _build_grouped_public_result(
            tool_response,
            columns,
            rows,
            normalized_count_values,
            minimum_aggregate_count,
            public_result_kind=(
                "safe_aggregate" if aggregate_columns else "count_aggregate"
            ),
            aggregate_columns=aggregate_columns,
        )

    public_result = dict(tool_response)
    public_result["matched_row_count"] = _normalize_count_value(sum(normalized_count_values))
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
    schema_summary = get_schema_summary(
        active_settings.db_path,
        include_categorical_value_guidance=active_settings.include_categorical_value_guidance,
        max_categorical_values=active_settings.max_categorical_values,
    )
    categorical_value_guidance = schema_summary.get("categorical_value_guidance") or []

    def remember_query_result(tool, args: dict, tool_context, tool_response: dict, **kwargs) -> dict | None:
        tool_name = getattr(tool, "name", "")
        if tool_name != "execute_sqlite_read_only":
            return None

        is_final = _tool_call_is_final(args)
        _clear_private_result_state(tool_context.state)
        last_query_frame: dict[str, Any] | None = None

        if active_settings.capture_internal_rows and tool_response.get("status") == "success":
            tool_context.state[SQL_INTERNAL_QUERY_RESULT_STATE_KEY] = tool_response
            tool_context.state[SQL_INTERNAL_RESULT_REF_STATE_KEY] = SQL_INTERNAL_QUERY_RESULT_STATE_KEY

        if not is_final:
            return None

        if tool_response.get("status") == "success":
            last_query_frame = _build_last_query_frame(
                tool_context.state,
                args,
                tool_response,
                categorical_value_guidance,
            )
            if last_query_frame is not None:
                source_query_frame = tool_context.state.get(SQL_REFINEMENT_SOURCE_QUERY_FRAME_STATE_KEY)
                recent_refinement = None
                if isinstance(source_query_frame, dict):
                    recent_refinement = _build_recent_refinement(source_query_frame, last_query_frame)
                if recent_refinement is not None:
                    rewritten_question = _rewrite_question_from_recent_refinement(
                        source_query_frame,
                        recent_refinement,
                    )
                    if rewritten_question:
                        last_query_frame["question"] = rewritten_question
                    last_query_frame["recent_refinement"] = recent_refinement
                    refreshed_topic_context = _build_query_frame_topic_context(last_query_frame)
                    if refreshed_topic_context:
                        last_query_frame["topic_context"] = refreshed_topic_context
                _set_last_query_frame_state(tool_context.state, last_query_frame)
                _print_clarification_debug(
                    active_settings,
                    "after-tool-last-query-frame",
                    last_query_frame,
                )
                if hasattr(tool_context.state, "pop"):
                    tool_context.state.pop(SQL_REFINEMENT_SOURCE_QUERY_FRAME_STATE_KEY, None)
                elif SQL_REFINEMENT_SOURCE_QUERY_FRAME_STATE_KEY in tool_context.state:
                    tool_context.state[SQL_REFINEMENT_SOURCE_QUERY_FRAME_STATE_KEY] = None

        public_result = _build_public_query_result(
            tool_response,
            active_settings,
        )
        display_sql = _normalize_public_display_sql(tool_response.get("display_sql"))
        if display_sql is None:
            display_sql = _normalize_public_display_sql(args.get("sql") if isinstance(args, dict) else None)
        if display_sql is not None:
            public_result["display_sql"] = display_sql
        if last_query_frame is not None:
            public_result["query_summary_context"] = {
                key: value
                for key, value in last_query_frame.items()
                if key != "recent_refinement"
            }
        tool_context.state[SQL_PUBLIC_RESULT_STATE_KEY] = public_result

        if active_settings.count_aggregates_only:
            return public_result

        return None

    return remember_query_result


def build_format_final_agent_response_callback():
    def format_final_agent_response(callback_context=None, **kwargs) -> types.Content | None:
        context = callback_context
        if context is None:
            return None

        if _public_query_result_was_rendered(context.state):
            return None

        public_query_result = context.state.get(SQL_PUBLIC_RESULT_STATE_KEY)
        if not isinstance(public_query_result, dict):
            return None

        _mark_public_query_result_rendered(context.state)
        return types.Content(
            role="model",
            parts=[types.Part(text=format_public_query_result(public_query_result))],
        )

    return format_final_agent_response


def build_normalize_clarification_after_model_callback(
    settings: SQLAgentSettings | None = None,
):
    active_settings = settings or load_settings()
    schema_summary = get_schema_summary(
        active_settings.db_path,
        include_categorical_value_guidance=active_settings.include_categorical_value_guidance,
        max_categorical_values=active_settings.max_categorical_values,
    )
    categorical_value_guidance = schema_summary.get("categorical_value_guidance") or []

    def normalize_clarification_after_model(
        callback_context=None,
        llm_response: LlmResponse | None = None,
        **kwargs,
    ) -> LlmResponse | None:
        is_fresh_topic_clarification = False
        if callback_context is not None:
            is_fresh_topic_clarification = _consume_fresh_topic_clarification_state(
                callback_context.state
            )

        if llm_response is None or _llm_response_has_function_call(llm_response):
            return None

        response_text = _extract_llm_response_text(llm_response)
        if not response_text:
            return None
        _print_clarification_debug(active_settings, "after-model-raw-response", response_text)

        clarification = normalize_clarification_response(response_text)
        if clarification is None:
            if not looks_like_clarification_attempt(response_text):
                return None
            clarification = build_fallback_clarification_response()
            _print_clarification_debug(
                active_settings,
                "after-model-fallback-reason",
                "normalization_failed",
            )
        elif clarification_requires_deterministic_fallback(clarification):
            clarification = build_fallback_clarification_response()
            _print_clarification_debug(
                active_settings,
                "after-model-fallback-reason",
                "low_confidence_normalization",
            )

        clarification_match_text = _build_clarification_guidance_match_text(clarification) or response_text
        clarification = _match_clarification_values_to_schema_guidance(
            clarification,
            clarification_match_text,
            categorical_value_guidance,
        )

        topic_context = None
        query_context = None
        if callback_context is not None:
            topic_context = callback_context.state.get(SQL_LAST_USER_TEXT_STATE_KEY)
            if not isinstance(topic_context, str):
                topic_context = None
            if not is_fresh_topic_clarification:
                query_context = _select_clarification_query_context(callback_context.state)
        clarification = _prune_topic_context_option(clarification, topic_context)
        _print_clarification_debug(active_settings, "after-model-normalized", clarification)

        if callback_context is not None:
            pending_clarification = dict(clarification)
            if isinstance(topic_context, str) and topic_context.strip():
                pending_clarification["topic_context"] = topic_context.strip()
            if isinstance(query_context, str) and query_context.strip():
                pending_clarification["query_context"] = query_context.strip()
            base_query_frame = None
            if not is_fresh_topic_clarification:
                base_query_frame = _get_last_query_frame(callback_context.state)
            if isinstance(base_query_frame, dict):
                pending_clarification["base_query_frame"] = copy.deepcopy(base_query_frame)
            _set_pending_clarification_state(callback_context.state, pending_clarification)
            _print_clarification_debug(
                active_settings,
                "after-model-pending-state",
                pending_clarification,
            )

        formatted_response = format_clarification_response(clarification)
        if not formatted_response:
            return None
        _print_clarification_debug(active_settings, "after-model-formatted-response", formatted_response)

        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part(text=formatted_response)],
            )
        )

    return normalize_clarification_after_model


def build_finalize_after_query_before_model_callback():
    def finalize_after_query(callback_context=None, llm_request=None, **kwargs) -> LlmResponse | None:
        context = callback_context
        if context is None:
            return None

        public_query_result = context.state.get(SQL_PUBLIC_RESULT_STATE_KEY)
        if not isinstance(public_query_result, dict):
            return None

        _mark_public_query_result_rendered(context.state)
        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part(text=format_public_query_result(public_query_result))],
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


def _extract_topic_context_text(user_text: str) -> str:
    normalized_text = user_text.strip()
    if not normalized_text:
        return ""

    non_empty_lines = [
        re.sub(r"\s+", " ", line).strip()
        for line in normalized_text.splitlines()
        if line.strip()
    ]
    if len(non_empty_lines) <= 1:
        return non_empty_lines[0] if non_empty_lines else ""

    for line in reversed(non_empty_lines):
        if line.endswith(":"):
            continue
        if re.match(r"^(?:[-*•]|\d+[.)])\s*", line):
            continue
        return line

    return re.sub(r"\s+", " ", normalized_text).strip()


def build_scope_gate_callback(
    classifier,
    refusal_message: str = DEFAULT_REFUSAL_MESSAGE,
    *,
    debug: bool = False,
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

        _raw_user_text, user_text = _extract_user_turn_texts(llm_request)
        if debug:
            print(f"[debug][sql-scope-gate][user-prompt]\n{user_text}")
        allow, refusal = classifier(user_text)
        if debug:
            verdict = "IN_SCOPE" if allow else "OUT_OF_SCOPE"
            print(f"[debug][sql-scope-gate][verdict] {verdict}")
            if refusal:
                print(f"[debug][sql-scope-gate][refusal]\n{refusal}")
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

    schema_summary = get_schema_summary(
        active_settings.db_path,
        include_categorical_value_guidance=active_settings.include_categorical_value_guidance,
        max_categorical_values=active_settings.max_categorical_values,
    )
    schema_text = schema_summary.get("schema_text") or ""
    classifier = build_llm_scope_gate(active_settings.model, schema_text)
    clarification_resolver = build_llm_clarification_resolver(
        active_settings.model,
        debug=active_settings.debug,
    )
    result_refinement_resolver = build_llm_result_refinement_resolver(
        active_settings.model,
        debug=active_settings.debug,
    )
    schema_grounding_resolver = build_llm_schema_grounding_resolver(
        active_settings.model,
        debug=active_settings.debug,
    )

    scope_gate = build_scope_gate_callback(classifier, debug=active_settings.debug)
    finalize = build_finalize_after_query_before_model_callback()

    def combined(callback_context=None, llm_request=None, **kwargs) -> LlmResponse | None:
        scope_gate_prechecked = False
        if callback_context is not None and not _request_ends_with_tool_response(llm_request):
            _clear_private_result_state(callback_context.state)
            _clear_fresh_topic_clarification_state(callback_context.state)
            raw_user_text, user_text = _extract_user_turn_texts(llm_request)
            topic_text = ""
            if user_text:
                topic_text = _extract_topic_context_text(user_text)
                callback_context.state[SQL_LAST_USER_TEXT_STATE_KEY] = topic_text
                callback_context.state[SQL_ACTIVE_QUERY_TOPIC_STATE_KEY] = topic_text
                _print_clarification_debug(active_settings, "before-model-user-prompt", raw_user_text or user_text)
            pending_clarification = _get_pending_clarification(callback_context.state)
            clarification_followup = False
            if pending_clarification is not None:
                _print_clarification_debug(
                    active_settings,
                    "before-model-pending-state",
                    pending_clarification,
                )
                pending_options = [
                    option
                    for option in pending_clarification.get("options") or []
                    if isinstance(option, str) and option.strip()
                ]
                matched_options = _extract_matching_clarification_options(user_text, pending_options)
                has_option_match = bool(matched_options)
                if matched_options:
                    _print_clarification_debug(
                        active_settings,
                        "before-model-option-matches",
                        matched_options,
                    )
                if has_option_match:
                    clarification_followup = _apply_pending_clarification_followup(
                        llm_request,
                        pending_clarification,
                    )
                    if clarification_followup:
                        base_query_frame = pending_clarification.get("base_query_frame")
                        if isinstance(base_query_frame, dict):
                            callback_context.state[SQL_REFINEMENT_SOURCE_QUERY_FRAME_STATE_KEY] = copy.deepcopy(
                                base_query_frame
                            )
                        topic_context = _select_pending_clarification_topic_text(pending_clarification)
                        if isinstance(topic_context, str) and topic_context.strip():
                            callback_context.state[SQL_ACTIVE_QUERY_TOPIC_STATE_KEY] = topic_context.strip()
                        _print_clarification_debug(
                            active_settings,
                            "before-model-followup-rewritten",
                            _extract_last_user_text(llm_request),
                        )
                        recent_interpretation_clarification = _build_recent_interpretation_clarification(
                            pending_clarification
                        )
                        if recent_interpretation_clarification is not None:
                            _set_recent_interpretation_clarification_state(
                                callback_context.state,
                                recent_interpretation_clarification,
                            )
                        _clear_pending_clarification_state(callback_context.state)
                else:
                    clarification_resolution = _resolve_pending_clarification_reply(
                        clarification_resolver,
                        pending_clarification,
                        user_text,
                    )
                    _print_clarification_debug(
                        active_settings,
                        "before-model-clarification-resolution",
                        clarification_resolution,
                    )
                    if clarification_resolution.get("resolution_type") == "topic_change":
                        _print_clarification_debug(
                            active_settings,
                            "before-model-topic-router-decision",
                            "TOPIC_CHANGE",
                        )
                        _mark_fresh_topic_clarification_state(callback_context.state)
                        _clear_pending_clarification_state(callback_context.state)
                    else:
                        _print_clarification_debug(
                            active_settings,
                            "before-model-topic-router-decision",
                            clarification_resolution.get("resolution_type") or "custom_rule",
                        )
                        clarification_followup = _apply_pending_clarification_followup_with_resolution(
                            llm_request,
                            pending_clarification,
                            user_text,
                            matched_options=clarification_resolution.get("selected_options"),
                            custom_rule=clarification_resolution.get("custom_rule"),
                        )
                        if clarification_followup:
                            base_query_frame = pending_clarification.get("base_query_frame")
                            if isinstance(base_query_frame, dict):
                                callback_context.state[SQL_REFINEMENT_SOURCE_QUERY_FRAME_STATE_KEY] = copy.deepcopy(
                                    base_query_frame
                                )
                            topic_context = _select_pending_clarification_topic_text(pending_clarification)
                            if isinstance(topic_context, str) and topic_context.strip():
                                callback_context.state[SQL_ACTIVE_QUERY_TOPIC_STATE_KEY] = topic_context.strip()
                            _print_clarification_debug(
                                active_settings,
                                "before-model-followup-rewritten",
                                _extract_last_user_text(llm_request),
                            )
                            if clarification_resolution.get("selected_options"):
                                recent_interpretation_clarification = _build_recent_interpretation_clarification(
                                    pending_clarification
                                )
                                if recent_interpretation_clarification is not None:
                                    _set_recent_interpretation_clarification_state(
                                        callback_context.state,
                                        recent_interpretation_clarification,
                                    )
                            _clear_pending_clarification_state(callback_context.state)
            if clarification_followup:
                _print_clarification_debug(
                    active_settings,
                    "before-model-branch",
                    "continuing clarification flow without scope gate",
                )
                return finalize(callback_context=callback_context, llm_request=llm_request, **kwargs)

            recent_interpretation_clarification = _get_recent_interpretation_clarification(
                callback_context.state
            )
            if recent_interpretation_clarification is not None:
                recent_options = [
                    option
                    for option in recent_interpretation_clarification.get("options") or []
                    if isinstance(option, str) and option.strip()
                ]
                matched_recent_options = _extract_matching_clarification_options(
                    user_text,
                    recent_options,
                )
                if matched_recent_options:
                    _print_clarification_debug(
                        active_settings,
                        "before-model-option-matches",
                        matched_recent_options,
                    )
                    clarification_followup = _apply_pending_clarification_followup_with_resolution(
                        llm_request,
                        recent_interpretation_clarification,
                        user_text,
                        matched_options=matched_recent_options,
                    )
                    if clarification_followup:
                        base_query_frame = recent_interpretation_clarification.get("base_query_frame")
                        if isinstance(base_query_frame, dict):
                            callback_context.state[SQL_REFINEMENT_SOURCE_QUERY_FRAME_STATE_KEY] = copy.deepcopy(
                                base_query_frame
                            )
                        topic_context = _select_pending_clarification_topic_text(
                            recent_interpretation_clarification
                        )
                        if isinstance(topic_context, str) and topic_context.strip():
                            callback_context.state[SQL_ACTIVE_QUERY_TOPIC_STATE_KEY] = topic_context.strip()
                        _print_clarification_debug(
                            active_settings,
                            "before-model-followup-rewritten",
                            _extract_last_user_text(llm_request),
                        )
                        _set_recent_interpretation_clarification_state(callback_context.state, None)
                        _print_clarification_debug(
                            active_settings,
                            "before-model-branch",
                            "continuing clarification flow without scope gate",
                        )
                        return finalize(callback_context=callback_context, llm_request=llm_request, **kwargs)
                _set_recent_interpretation_clarification_state(callback_context.state, None)

            last_query_frame = _get_last_query_frame(callback_context.state)
            if last_query_frame is not None:
                _print_clarification_debug(
                    active_settings,
                    "before-model-last-query-frame",
                    last_query_frame,
                )
                refinement_resolution = _resolve_recent_refinement_followup(last_query_frame, user_text)
                if refinement_resolution is None:
                    refinement_resolution = result_refinement_resolver(last_query_frame, user_text)
                _print_clarification_debug(
                    active_settings,
                    "before-model-result-refinement-resolution",
                    refinement_resolution,
                )
                refinement_type = refinement_resolution.get("resolution_type")
                if refinement_type == "topic_change":
                    _mark_fresh_topic_clarification_state(callback_context.state)
                if refinement_type == "needs_clarification":
                    clarification = _build_result_refinement_clarification(
                        last_query_frame,
                        str(refinement_resolution.get("target_column") or "").strip(),
                    )
                    if clarification is not None:
                        _set_pending_clarification_state(callback_context.state, clarification)
                        topic_context = clarification.get("topic_context")
                        if isinstance(topic_context, str) and topic_context.strip():
                            callback_context.state[SQL_ACTIVE_QUERY_TOPIC_STATE_KEY] = topic_context.strip()
                        _print_clarification_debug(
                            active_settings,
                            "before-model-result-refinement-clarification",
                            clarification,
                        )
                        formatted_response = format_clarification_response(clarification)
                        if formatted_response:
                            return LlmResponse(
                                content=types.Content(
                                    role="model",
                                    parts=[types.Part(text=formatted_response)],
                                )
                            )
                elif refinement_type == "refine_query":
                    refinement_followup = _apply_last_query_refinement_followup(
                        llm_request,
                        last_query_frame,
                        user_text,
                        target_column=str(refinement_resolution.get("target_column") or "").strip() or None,
                        selected_values=[
                            value
                            for value in refinement_resolution.get("selected_values") or []
                            if isinstance(value, str) and value.strip()
                        ],
                        refinement_request=str(refinement_resolution.get("refinement_request") or "").strip(),
                    )
                    if refinement_followup:
                        callback_context.state[SQL_REFINEMENT_SOURCE_QUERY_FRAME_STATE_KEY] = copy.deepcopy(
                            last_query_frame
                        )
                        previous_question = _get_query_frame_question_text(last_query_frame)
                        if previous_question:
                            callback_context.state[SQL_ACTIVE_QUERY_TOPIC_STATE_KEY] = previous_question
                        _print_clarification_debug(
                            active_settings,
                            "before-model-followup-rewritten",
                            _extract_last_user_text(llm_request),
                        )
                        _print_clarification_debug(
                            active_settings,
                            "before-model-branch",
                            "continuing result refinement flow without scope gate",
                        )
                        return finalize(callback_context=callback_context, llm_request=llm_request, **kwargs)

            if bool(callback_context.state.get(SQL_FRESH_TOPIC_CLARIFICATION_STATE_KEY)):
                scope_result = scope_gate(
                    callback_context=callback_context,
                    llm_request=llm_request,
                    **kwargs,
                )
                scope_gate_prechecked = True
                if scope_result is not None:
                    return scope_result

        if callback_context is not None and not _request_ends_with_tool_response(llm_request):
            if user_text:
                topic_text = str(
                    callback_context.state.get(SQL_ACTIVE_QUERY_TOPIC_STATE_KEY)
                    or callback_context.state.get(SQL_LAST_USER_TEXT_STATE_KEY)
                    or ""
                ).strip()
                request_field_glossary = _extract_request_field_glossary(raw_user_text)
                schema_grounding_catalog = _build_schema_grounding_catalog(
                    schema_summary,
                    request_field_glossary=request_field_glossary,
                    user_text=user_text,
                )
                grounding_resolution = schema_grounding_resolver(
                    user_text,
                    str(schema_grounding_catalog.get("context_text") or ""),
                    [
                        entry
                        for entry in schema_grounding_catalog.get("grounding_candidates") or []
                        if isinstance(entry, dict)
                    ],
                    [
                        identifier
                        for identifier in schema_grounding_catalog.get("candidate_columns") or []
                        if isinstance(identifier, str) and identifier.strip()
                    ],
                )
                grounded_filters = _normalize_schema_grounding_filters(
                    grounding_resolution.get("grounded_filters"),
                    {
                        str(identifier): [
                            value
                            for value in values or []
                            if isinstance(value, str) and value.strip()
                        ]
                        for identifier, values in (schema_grounding_catalog.get("candidate_values_by_column") or {}).items()
                        if isinstance(identifier, str) and identifier.strip()
                    },
                )
                filtered_schema_grounding_catalog = _build_schema_grounding_catalog(
                    schema_summary,
                    request_field_glossary=request_field_glossary,
                    grounded_filters=grounded_filters,
                    user_text=user_text,
                )
                resolved_candidate_columns = [
                    identifier
                    for identifier in grounding_resolution.get("candidate_columns") or []
                    if isinstance(identifier, str)
                    and identifier.strip()
                    and identifier in (filtered_schema_grounding_catalog.get("candidate_columns") or [])
                ]
                _print_clarification_debug(
                    active_settings,
                    "before-model-schema-grounding-resolution",
                    {
                        **grounding_resolution,
                        "grounded_filters": grounded_filters,
                        "candidate_columns": resolved_candidate_columns,
                    },
                )
                if grounding_resolution.get("resolution_type") == "needs_clarification" and len(resolved_candidate_columns) >= 2:
                    clarification = _build_schema_grounding_clarification(
                        topic_text,
                        resolved_candidate_columns,
                        filtered_schema_grounding_catalog,
                        grounded_filters=grounded_filters,
                    )
                    if clarification is not None:
                        _set_pending_clarification_state(callback_context.state, clarification)
                        _print_clarification_debug(
                            active_settings,
                            "before-model-schema-grounding-clarification",
                            clarification,
                        )
                        formatted_response = format_clarification_response(clarification)
                        if formatted_response:
                            return LlmResponse(
                                content=types.Content(
                                    role="model",
                                    parts=[types.Part(text=formatted_response)],
                                )
                            )
                elif grounded_filters:
                    grounded_followup = _apply_grounded_filter_followup(
                        llm_request,
                        user_text,
                        grounded_filters,
                    )
                    if grounded_followup:
                        _print_clarification_debug(
                            active_settings,
                            "before-model-followup-rewritten",
                            _extract_last_user_text(llm_request),
                        )
                        _print_clarification_debug(
                            active_settings,
                            "before-model-branch",
                            "continuing schema grounding flow without scope gate",
                        )
                        return finalize(callback_context=callback_context, llm_request=llm_request, **kwargs)

        if not scope_gate_prechecked:
            result = scope_gate(callback_context=callback_context, llm_request=llm_request, **kwargs)
            if result is not None:
                return result
        return finalize(callback_context=callback_context, llm_request=llm_request, **kwargs)

    return combined
