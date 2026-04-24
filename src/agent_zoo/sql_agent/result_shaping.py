"""Shape SQL tool output into privacy-filtered public results and query-frame context.

This module owns the after_tool pipeline: helpers that (a) build the
``last_query_frame`` stored in working memory for refinement memory, and (b)
transform raw tool responses into the public result the agent renders. It is
used internally by ``callbacks.py`` and is not meant to be run directly.
"""

from __future__ import annotations

import re
from typing import Any

from .db import (
    _extract_sql_string_literals,
    _find_last_top_level_keyword,
    _find_top_level_keyword,
    _normalize_sql_literal,
    _quote_identifier as _quote_sql_identifier,
    _split_top_level_expressions,
    count_subset_rows,
)


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
