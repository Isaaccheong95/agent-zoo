"""Shape SQL tool output into privacy-filtered public results and query-frame context.

This module owns the after_tool pipeline: helpers that (a) build the
``last_query_frame`` stored in working memory for refinement memory, and (b)
transform raw tool responses into the public result the agent renders. It is
used internally by ``callbacks.py`` and is not meant to be run directly.
"""

from __future__ import annotations

import re
from typing import Any

from .config import SQLAgentSettings
from .db import (
    _extract_sql_string_literals,
    _find_last_top_level_keyword,
    _find_top_level_keyword,
    _normalize_sql_literal,
    _quote_identifier as _quote_sql_identifier,
    _split_top_level_expressions,
    count_subset_rows,
    execute_sqlite_query,
)


SQL_LAST_USER_TEXT_STATE_KEY = "temp:sql_last_user_text"
SQL_ACTIVE_QUERY_TOPIC_STATE_KEY = "temp:sql_active_query_topic"


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


def _build_related_identifier_sql(reference_sql: str, identifier_name: str) -> str | None:
    normalized_identifier_name = str(identifier_name or "").strip()
    if not normalized_identifier_name:
        return None

    quoted_identifier = _quote_sql_identifier(normalized_identifier_name)
    normalized_reference_sql = str(reference_sql or "").strip()
    if "." not in normalized_reference_sql:
        return quoted_identifier

    qualifier, _, _ = normalized_reference_sql.rpartition(".")
    qualifier = qualifier.strip()
    if not qualifier:
        return quoted_identifier
    return f"{qualifier}.{quoted_identifier}"


def _normalize_match_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


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

        quoted_or_bare_column = rf'(?<!\w)(?P<column_sql>(?:(?:"[^"]+"|[A-Za-z_][A-Za-z0-9_]*)\s*\.\s*)?(?:"{re.escape(column_name)}"|{re.escape(column_name)}))(?!\w)'
        selected_values: list[str] = []
        matched_column_sql = ""

        in_match = re.search(
            rf"{quoted_or_bare_column}\s+IN\s*\((?P<values>[^)]*)\)",
            searchable_sql,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if in_match is not None:
            matched_column_sql = str(in_match.group("column_sql") or "").strip()
            selected_values = _normalize_allowed_values(
                _extract_sql_string_literals(in_match.group("values")),
                available_values,
            )

        if not selected_values:
            equality_values: list[str] = []
            for match in re.finditer(
                rf"{quoted_or_bare_column}\s*=\s*'((?:''|[^'])*)'",
                searchable_sql,
                flags=re.IGNORECASE,
            ):
                if not matched_column_sql:
                    matched_column_sql = str(match.group("column_sql") or "").strip()
                equality_values.append(match.group(2).replace("''", "'"))
            selected_values = _normalize_allowed_values(equality_values, available_values)

        if not selected_values:
            continue

        filter_entry = {
            "column": column_name,
            "selected_values": selected_values,
            "available_values": available_values,
        }
        if matched_column_sql:
            filter_entry["column_sql"] = matched_column_sql
        filters.append(filter_entry)

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


def _build_missing_or_blank_value_condition(column_sql: str) -> str:
    return f"NULLIF(TRIM(CAST({column_sql} AS TEXT)), '') IS NULL"


def _build_dataset_missing_or_blank_count_sql(
    sql: str,
    filter_entry: dict[str, Any],
    *,
    object_id_column: str | None = None,
) -> tuple[str | None, str | None]:
    select_pos = _find_top_level_keyword(sql, "SELECT")
    from_pos = _find_top_level_keyword(sql, "FROM")
    if select_pos is None or from_pos is None:
        return None, None

    prefix = sql[:select_pos].rstrip()
    rest = sql[from_pos:]
    cut_pos = len(rest)
    for terminal in ("WHERE", "GROUP BY", "HAVING", "ORDER BY", "LIMIT"):
        pos = _find_top_level_keyword(rest, terminal)
        if pos is not None and pos < cut_pos:
            cut_pos = pos
    source_clause = rest[:cut_pos].rstrip()
    if not source_clause:
        return None, None

    column_sql = str(filter_entry.get("column_sql") or "").strip()
    if not column_sql:
        column_name = str(filter_entry.get("column") or "").strip()
        if not column_name:
            return None, None
        column_sql = _quote_sql_identifier(column_name)

    missing_condition = _build_missing_or_blank_value_condition(column_sql)
    count_expression = "COUNT(*)"
    count_unit = "rows"
    normalized_object_id_column = str(object_id_column or "").strip()
    if normalized_object_id_column:
        object_id_sql = _build_related_identifier_sql(column_sql, normalized_object_id_column)
        if object_id_sql:
            count_expression = f"COUNT(DISTINCT {object_id_sql})"
            count_unit = "patients"

    count_sql = (
        f"SELECT {count_expression} AS missing_or_blank_count "
        f"{source_clause} WHERE {missing_condition}"
    )
    normalized_count_sql = " ".join(part for part in (prefix, count_sql) if part).strip()
    return normalized_count_sql, count_unit


def _annotate_categorical_filter_dataset_missing_counts(
    db_path: Any,
    sql: str,
    categorical_filters: list[dict[str, Any]],
    *,
    object_id_column: str | None = None,
) -> None:
    for filter_entry in categorical_filters:
        count_sql, count_unit = _build_dataset_missing_or_blank_count_sql(
            sql,
            filter_entry,
            object_id_column=object_id_column,
        )
        if not count_sql:
            continue
        missing_count = count_subset_rows(db_path, count_sql)
        if missing_count is None or missing_count < 0:
            continue
        filter_entry["dataset_missing_or_blank_count"] = missing_count
        if count_unit:
            filter_entry["dataset_missing_or_blank_count_unit"] = count_unit


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
    select_items = _split_top_level_expressions(select_clause)
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

    group_columns = [
        column
        for column in query_frame.get("group_columns") or []
        if isinstance(column, str) and column.strip()
    ]
    if group_columns:
        context_sections.append(
            "Current committed grouping columns:\n" + "\n".join(
                f"- {column}" for column in group_columns
            )
        )

    return "\n\n".join(context_sections).strip()


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


def _build_last_query_frame(
    state: Any,
    args: dict[str, Any],
    tool_response: dict[str, Any],
    categorical_value_guidance: list[dict[str, Any]],
    *,
    db_path: Any = None,
    object_id_column: str | None = None,
) -> dict[str, Any] | None:
    if state is None or not hasattr(state, "get"):
        return None

    active_query_topic = state.get(SQL_ACTIVE_QUERY_TOPIC_STATE_KEY)
    if not isinstance(active_query_topic, str) or not active_query_topic.strip():
        active_query_topic = state.get(SQL_LAST_USER_TEXT_STATE_KEY)
    if not isinstance(active_query_topic, str) or not active_query_topic.strip():
        return None

    raw_sql = tool_response.get("display_sql")
    if not isinstance(raw_sql, str) or not raw_sql.strip():
        raw_sql = args.get("sql") if isinstance(args, dict) else None
    if not isinstance(raw_sql, str) or not raw_sql.strip():
        raw_sql = tool_response.get("sql")
    if not isinstance(raw_sql, str) or not raw_sql.strip():
        return None
    normalized_sql = _normalize_public_display_sql(raw_sql) or raw_sql.strip()

    query_frame = {
        "question": active_query_topic.strip(),
        "sql": normalized_sql,
    }
    categorical_filters = _extract_categorical_filters_from_sql(normalized_sql, categorical_value_guidance)
    if categorical_filters and db_path:
        _annotate_categorical_filter_dataset_missing_counts(
            db_path,
            normalized_sql,
            categorical_filters,
            object_id_column=object_id_column,
        )
    for entry in categorical_filters:
        if isinstance(entry, dict):
            entry.pop("column_sql", None)
    if categorical_filters:
        query_frame["categorical_filters"] = categorical_filters
    comparison_filters = _extract_comparison_filters_from_sql(
        normalized_sql,
        excluded_columns={
            str(entry.get("column") or "").strip()
            for entry in categorical_filters
            if isinstance(entry, dict)
        },
    )
    if comparison_filters:
        query_frame["comparison_filters"] = comparison_filters
    group_columns = [
        str(column).strip()
        for column in (tool_response.get("group_columns") or [])
        if isinstance(column, str) and str(column).strip()
    ]
    if not group_columns and _sql_has_top_level_group_by(normalized_sql):
        group_columns = [
            column
            for column in (tool_response.get("columns") or [])
            if isinstance(column, str)
            and column.strip()
            and not _is_count_column(column)
            and not _is_safe_aggregate_column(column)
        ]
    if group_columns:
        query_frame["is_grouped"] = True
        query_frame["group_columns"] = _ordered_unique_values(group_columns)
    topic_context = _build_query_frame_topic_context(query_frame)
    if topic_context:
        query_frame["topic_context"] = topic_context
    return query_frame


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
