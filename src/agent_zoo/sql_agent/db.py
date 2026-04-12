"""Provide read-only SQLite helpers for schema inspection and query execution.

This module contains the database-facing safety layer for the SQL agent. It
opens SQLite files in read-only mode, validates generated SQL, summarizes
schema information, and returns structured execution results.

It is designed to be imported by tools and tests rather than executed directly.
The easiest way to use it end-to-end is through `uv run run-sql-agent`.
"""

from __future__ import annotations

from contextlib import closing
import re
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from .config import (
    DEFAULT_MAX_CATEGORICAL_VALUES,
    DEFAULT_PREVIEW_ROWS,
    resolve_repo_path,
)


INTERNAL_TABLE_PREFIXES = ("sqlite_", "__")
READ_ONLY_ROOT_STATEMENT_KEYWORDS = ("SELECT", "INSERT", "UPDATE", "DELETE", "REPLACE")
UNKNOWN_GROUP_VALUE_LABEL = "Null"
SQL_IDENTIFIER_EXPRESSION_PATTERN = re.compile(
    r'^(?:"[^"]+"|`[^`]+`|\[[^\]]+\]|[A-Za-z_][A-Za-z0-9_]*)(?:\s*\.\s*(?:"[^"]+"|`[^`]+`|\[[^\]]+\]|[A-Za-z_][A-Za-z0-9_]*))*$'
)
SQL_IDENTIFIER_TOKEN_PATTERN = re.compile(
    r'(?:"[^"]+"|`[^`]+`|\[[^\]]+\]|[A-Za-z_][A-Za-z0-9_]*)(?:\s*\.\s*(?:"[^"]+"|`[^`]+`|\[[^\]]+\]|[A-Za-z_][A-Za-z0-9_]*))*'
)
CASE_MISSING_SOURCE_IGNORED_IDENTIFIERS = {
    "AND",
    "AS",
    "BETWEEN",
    "BLOB",
    "CASE",
    "CAST",
    "COALESCE",
    "ELSE",
    "END",
    "FALSE",
    "GLOB",
    "IN",
    "INTEGER",
    "IS",
    "LIKE",
    "LOWER",
    "NOT",
    "NULL",
    "NULLIF",
    "NUMERIC",
    "OR",
    "REAL",
    "REGEXP",
    "TEXT",
    "THEN",
    "TRIM",
    "TRUE",
    "UPPER",
    "WHEN",
}
OBJECT_SOURCE_CTE_NAME = "__az_object_source"
OBJECT_RANKED_CTE_NAME = "__az_object_ranked"
OBJECT_CANONICAL_CTE_NAME = "__az_object_canonical"
OBJECT_SOURCE_ORDINAL_COLUMN = "__az_source_ordinal"
OBJECT_ROW_NUMBER_COLUMN = "__az_object_row_number"
UNORDERABLE_DECLARED_TYPE_TOKENS = ("BLOB",)
CATEGORICAL_VALUE_GUIDANCE_EXCLUDED_NAME_PATTERNS = (
    re.compile(r"(^|_)(id|uuid)($|_)"),
    re.compile(r"(^|_)(date|time|timestamp)($|_)"),
    re.compile(r"(^|_)(created|updated|deleted)($|_)"),
)
CATEGORICAL_VALUE_GUIDANCE_EXCLUDED_NAME_TOKENS = (
    "name",
    "description",
    "comment",
    "note",
    "text",
    "address",
    "email",
    "phone",
    "payload",
)


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _as_path(db_path: str | Path) -> Path:
    resolved = resolve_repo_path(db_path)
    if resolved is None:
        raise FileNotFoundError("No SQLite database path was provided.")
    return resolved


def _ensure_database_exists(db_path: str | Path) -> Path:
    path = _as_path(db_path)
    if not path.exists():
        raise FileNotFoundError(f"SQLite database not found: {path}")
    return path


def _read_only_uri(db_path: str | Path) -> str:
    return _ensure_database_exists(db_path).resolve().as_uri() + "?mode=ro"


def _connect_read_only(db_path: str | Path) -> sqlite3.Connection:
    connection = sqlite3.connect(_read_only_uri(db_path), uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _scan_sql(sql: str) -> tuple[str, str, list[str]]:
    cleaned: list[str] = []
    token_text: list[str] = []
    statements: list[str] = []
    current_statement: list[str] = []
    state = "normal"
    index = 0

    while index < len(sql):
        char = sql[index]
        next_char = sql[index + 1] if index + 1 < len(sql) else ""

        if state == "line_comment":
            if char == "\n":
                cleaned.append("\n")
                token_text.append(" ")
                current_statement.append("\n")
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
            cleaned.append(char)
            current_statement.append(char)
            token_text.append(" ")
            if char == "'" and next_char == "'":
                cleaned.append(next_char)
                current_statement.append(next_char)
                token_text.append(" ")
                index += 2
                continue
            if char == "'":
                state = "normal"
            index += 1
            continue

        if state == "double_quote":
            cleaned.append(char)
            current_statement.append(char)
            token_text.append(" ")
            if char == '"':
                state = "normal"
            index += 1
            continue

        if state == "backtick":
            cleaned.append(char)
            current_statement.append(char)
            token_text.append(" ")
            if char == "`":
                state = "normal"
            index += 1
            continue

        if state == "bracket":
            cleaned.append(char)
            current_statement.append(char)
            token_text.append(" ")
            if char == "]":
                state = "normal"
            index += 1
            continue

        if char == "-" and next_char == "-":
            cleaned.append(" ")
            current_statement.append(" ")
            token_text.append(" ")
            state = "line_comment"
            index += 2
            continue

        if char == "/" and next_char == "*":
            cleaned.append(" ")
            current_statement.append(" ")
            token_text.append(" ")
            state = "block_comment"
            index += 2
            continue

        cleaned.append(char)
        current_statement.append(char)
        token_text.append(char)

        if char == "'":
            state = "single_quote"
        elif char == '"':
            state = "double_quote"
        elif char == "`":
            state = "backtick"
        elif char == "[":
            state = "bracket"
        elif char == ";":
            statement = "".join(current_statement[:-1]).strip()
            if statement:
                statements.append(statement)
            current_statement = []

        index += 1

    final_statement = "".join(current_statement).strip()
    if final_statement:
        statements.append(final_statement)

    return "".join(cleaned), "".join(token_text), statements


def _normalized_statement(sql: str) -> str:
    return _normalize_whitespace(sql).rstrip(";").strip()


def _normalize_declared_type(value: str | None) -> str:
    return _normalize_whitespace(str(value or "")).upper()


def _is_orderable_declared_type(value: str | None) -> bool:
    normalized = _normalize_declared_type(value)
    return not any(token in normalized for token in UNORDERABLE_DECLARED_TYPE_TOKENS)


def _should_collect_categorical_values(column_name: str, declared_type: str | None) -> bool:
    normalized_name = _normalize_whitespace(str(column_name or "")).lower()
    if not normalized_name:
        return False
    if not _is_orderable_declared_type(declared_type):
        return False
    if any(pattern.search(normalized_name) for pattern in CATEGORICAL_VALUE_GUIDANCE_EXCLUDED_NAME_PATTERNS):
        return False
    return not any(token in normalized_name for token in CATEGORICAL_VALUE_GUIDANCE_EXCLUDED_NAME_TOKENS)


def _categorical_value_expression(column_name: str) -> str:
    quoted_column = _quote_identifier(column_name)
    return f"NULLIF(TRIM(CAST({quoted_column} AS TEXT)), '')"


def _collect_categorical_values(
    connection: sqlite3.Connection,
    table_name: str,
    column_name: str,
    declared_type: str | None,
    max_values: int,
) -> list[str]:
    if not _should_collect_categorical_values(column_name, declared_type):
        return []

    max_values = max(2, int(max_values))
    value_expression = _categorical_value_expression(column_name)
    quoted_table_name = _quote_identifier(table_name)
    stats_sql = (
        f"SELECT COUNT({value_expression}) AS non_null_count, "
        f"COUNT(DISTINCT {value_expression}) AS distinct_count "
        f"FROM {quoted_table_name}"
    )
    stats_row = connection.execute(stats_sql).fetchone()
    if stats_row is None:
        return []

    non_null_count = int(stats_row["non_null_count"] or 0)
    distinct_count = int(stats_row["distinct_count"] or 0)
    if non_null_count < 2 or distinct_count < 2:
        return []
    if distinct_count > max_values or distinct_count >= non_null_count:
        return []

    value_sql = (
        f"SELECT DISTINCT {value_expression} AS value "
        f"FROM {quoted_table_name} "
        f"WHERE {value_expression} IS NOT NULL "
        f"LIMIT ?"
    )
    values = [
        str(row["value"])
        for row in connection.execute(value_sql, (max_values,))
        if row["value"] is not None
    ]
    return sorted(values, key=str.casefold)


def _format_categorical_value_guidance(guidance_entries: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for entry in guidance_entries:
        values = entry.get("values") or []
        if not values:
            continue
        values_text = ", ".join(f'"{value}"' for value in values)
        lines.append(
            f'- {entry["table"]}.{entry["column"]}: Stored SQLite values seen in the dataset: {values_text}'
        )
    return "\n".join(lines)


def _find_top_level_keyword_positions(sql: str, keyword: str) -> list[int]:
    upper = sql.upper()
    keyword_upper = keyword.upper()
    keyword_len = len(keyword_upper)
    positions: list[int] = []
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

        if state == "backtick":
            if char == "`":
                state = "normal"
            index += 1
            continue

        if state == "bracket":
            if char == "]":
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

        if char == "`":
            state = "backtick"
            index += 1
            continue

        if char == "[":
            state = "bracket"
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
                positions.append(index)
                index += keyword_len
                continue

        index += 1

    return positions


def _find_top_level_keyword(sql: str, keyword: str) -> int | None:
    positions = _find_top_level_keyword_positions(sql, keyword)
    if not positions:
        return None
    return positions[0]


def _find_last_top_level_keyword(sql: str, keyword: str) -> int | None:
    positions = _find_top_level_keyword_positions(sql, keyword)
    if not positions:
        return None
    return positions[-1]


def _resolve_effective_root_statement(sql: str) -> str | None:
    normalized_sql = _normalized_statement(sql)
    if not normalized_sql:
        return None

    first_keyword_match = re.match(r"^([A-Za-z]+)", normalized_sql)
    if first_keyword_match is None:
        return None

    first_keyword = first_keyword_match.group(1).upper()
    if first_keyword != "WITH":
        return first_keyword

    candidate_positions = [
        (position, keyword)
        for keyword in READ_ONLY_ROOT_STATEMENT_KEYWORDS
        if (position := _find_top_level_keyword(normalized_sql, keyword)) is not None
    ]
    if not candidate_positions:
        return None
    return min(candidate_positions, key=lambda item: item[0])[1]


def _extract_top_level_query_sections(sql: str) -> dict[str, str] | None:
    select_pos = _find_top_level_keyword(sql, "SELECT")
    from_pos = _find_top_level_keyword(sql, "FROM")
    if select_pos is None or from_pos is None or from_pos <= select_pos:
        return None

    cut_pos = len(sql)
    for terminal in ("GROUP BY", "HAVING", "ORDER BY", "LIMIT"):
        pos = _find_top_level_keyword(sql, terminal)
        if pos is not None and pos > from_pos and pos < cut_pos:
            cut_pos = pos

    return {
        "prefix": sql[:select_pos].rstrip(),
        "select_clause": sql[select_pos + len("SELECT"):from_pos].strip(),
        "from_clause": sql[from_pos:cut_pos].rstrip(),
        "suffix": sql[cut_pos:].strip(),
    }


def _extract_top_level_clause_body_range(
    sql: str,
    clause_keyword: str,
    following_keywords: tuple[str, ...],
) -> tuple[int, int] | None:
    clause_pos = _find_top_level_keyword(sql, clause_keyword)
    if clause_pos is None:
        return None

    body_start = clause_pos + len(clause_keyword)
    body_end = len(sql)
    for keyword in following_keywords:
        keyword_pos = _find_top_level_keyword(sql, keyword)
        if keyword_pos is not None and keyword_pos > clause_pos and keyword_pos < body_end:
            body_end = keyword_pos

    return body_start, body_end


def _split_top_level_expressions(sql_fragment: str) -> list[str]:
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

        if state == "backtick":
            current.append(char)
            if char == "`":
                state = "normal"
            index += 1
            continue

        if state == "bracket":
            current.append(char)
            if char == "]":
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

        if char == "`":
            current.append(char)
            state = "backtick"
            index += 1
            continue

        if char == "[":
            current.append(char)
            state = "bracket"
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


def _normalize_sql_reference(value: str) -> str:
    normalized = _normalize_whitespace(str(value or ""))
    if not normalized:
        return ""

    parts = re.split(r"\s*\.\s*", normalized)
    candidate = parts[-1].strip() if parts else normalized
    if candidate.startswith('"') and candidate.endswith('"') and len(candidate) >= 2:
        candidate = candidate[1:-1]
    elif candidate.startswith("`") and candidate.endswith("`") and len(candidate) >= 2:
        candidate = candidate[1:-1]
    elif candidate.startswith("[") and candidate.endswith("]") and len(candidate) >= 2:
        candidate = candidate[1:-1]
    return candidate.strip()


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


def _infer_select_item_alias_sql(expression: str) -> str | None:
    normalized_expression = expression.strip()
    if not SQL_IDENTIFIER_EXPRESSION_PATTERN.fullmatch(normalized_expression):
        return None

    return re.split(r"\s*\.\s*", normalized_expression)[-1].strip()


def _reference_match_keys(value: str) -> set[str]:
    normalized_value = _normalize_whitespace(value)
    if not normalized_value:
        return set()

    keys = {normalized_value.casefold()}
    normalized_reference = _normalize_sql_reference(normalized_value)
    if normalized_reference:
        keys.add(normalized_reference.casefold())
    return keys


def _escape_sql_string_literal(value: str) -> str:
    return value.replace("'", "''")


def _extract_sql_string_literals(sql_fragment: str) -> list[str]:
    return [
        match.group(1).replace("''", "'")
        for match in re.finditer(r"'((?:''|[^'])*)'", sql_fragment)
    ]


def _normalize_sql_literal(value: str) -> str:
    literal = value.strip()
    if literal.startswith("'") and literal.endswith("'") and len(literal) >= 2:
        literal = literal[1:-1].replace("''", "'")
    return literal.strip()


def _normalize_categorical_values(
    values: Iterable[str],
    allowed_values: list[str],
) -> list[str]:
    value_lookup: dict[str, str] = {}
    for allowed_value in allowed_values:
        normalized_allowed_value = _normalize_whitespace(str(allowed_value or "")).casefold()
        if normalized_allowed_value and normalized_allowed_value not in value_lookup:
            value_lookup[normalized_allowed_value] = str(allowed_value)

    normalized_values: list[str] = []
    seen_values: set[str] = set()
    for value in values:
        canonical_value = value_lookup.get(_normalize_whitespace(str(value or "")).casefold())
        if canonical_value and canonical_value not in seen_values:
            seen_values.add(canonical_value)
            normalized_values.append(canonical_value)
    return normalized_values


def _render_explicit_categorical_predicate(column_sql: str, selected_values: list[str]) -> str:
    literal_values = [f"'{_escape_sql_string_literal(value)}'" for value in selected_values]
    if len(literal_values) == 1:
        return f"{column_sql} = {literal_values[0]}"
    return f"{column_sql} IN ({', '.join(literal_values)})"


def _rewrite_categorical_negation_sql(
    sql: str,
    categorical_value_guidance: list[dict[str, Any]],
) -> str | None:
    normalized_sql = _normalized_statement(sql)
    if not normalized_sql or not re.search(r"!=|<>|\bNOT\s+IN\b", normalized_sql, flags=re.IGNORECASE):
        return None

    allowed_values_by_column: dict[str, list[str]] = {}
    for entry in categorical_value_guidance:
        column_name = str(entry.get("column") or "").strip()
        allowed_values = [
            value.strip()
            for value in entry.get("values") or []
            if isinstance(value, str) and value.strip()
        ]
        if column_name and allowed_values:
            allowed_values_by_column[column_name.casefold()] = allowed_values

    if not allowed_values_by_column:
        return None

    identifier_pattern = r'(?:"[^"]+"|`[^`]+`|\[[^\]]+\]|[A-Za-z_][A-Za-z0-9_]*)(?:\s*\.\s*(?:"[^"]+"|`[^`]+`|\[[^\]]+\]|[A-Za-z_][A-Za-z0-9_]*))*'
    replacements: list[tuple[int, int, str]] = []

    for match in re.finditer(
        rf'(?P<column>{identifier_pattern})\s+NOT\s+IN\s*\((?P<values>[^)]*)\)',
        normalized_sql,
        flags=re.IGNORECASE,
    ):
        column_sql = match.group("column")
        column_name = _normalize_sql_reference(column_sql)
        allowed_values = allowed_values_by_column.get(column_name.casefold())
        if not allowed_values:
            continue

        excluded_values = _normalize_categorical_values(
            _extract_sql_string_literals(match.group("values")),
            allowed_values,
        )
        if not excluded_values or len(excluded_values) >= len(allowed_values):
            continue

        selected_values = [value for value in allowed_values if value not in excluded_values]
        if not selected_values:
            continue
        replacements.append(
            (match.start(), match.end(), _render_explicit_categorical_predicate(column_sql, selected_values))
        )

    for match in re.finditer(
        rf'(?P<column>{identifier_pattern})\s*(?P<operator><>|!=)\s*(?P<value>\'(?:(?:\'\')|[^\'])*\')',
        normalized_sql,
        flags=re.IGNORECASE,
    ):
        column_sql = match.group("column")
        column_name = _normalize_sql_reference(column_sql)
        allowed_values = allowed_values_by_column.get(column_name.casefold())
        if not allowed_values:
            continue

        excluded_values = _normalize_categorical_values(
            [_normalize_sql_literal(match.group("value"))],
            allowed_values,
        )
        if len(excluded_values) != 1:
            continue

        selected_values = [value for value in allowed_values if value not in excluded_values]
        if not selected_values:
            continue
        replacements.append(
            (match.start(), match.end(), _render_explicit_categorical_predicate(column_sql, selected_values))
        )

    if not replacements:
        return None

    rewritten_sql = _replace_sql_ranges(normalized_sql, replacements)
    return _normalized_statement(rewritten_sql)


def _infer_case_missing_probe_expression(expression: str) -> str | None:
    if not re.match(r"^\s*CASE\b", expression, flags=re.IGNORECASE):
        return None

    cleaned_expression, _, _ = _scan_sql(expression)
    candidates: dict[str, str] = {}
    for match in SQL_IDENTIFIER_TOKEN_PATTERN.finditer(cleaned_expression):
        candidate_sql = match.group(0).strip()
        candidate_name = _normalize_sql_reference(candidate_sql)
        if not candidate_name:
            continue
        if candidate_name.upper() in CASE_MISSING_SOURCE_IGNORED_IDENTIFIERS:
            continue
        candidates.setdefault(candidate_name.casefold(), candidate_sql)

    if len(candidates) != 1:
        return None
    return next(iter(candidates.values()))


def _build_missing_group_expression(expression: str) -> str:
    stripped_expression = expression.strip()
    probe_expression = _infer_case_missing_probe_expression(stripped_expression) or stripped_expression
    label_literal = _escape_sql_string_literal(UNKNOWN_GROUP_VALUE_LABEL)
    return (
        "CASE WHEN NULLIF(TRIM(CAST(("
        + probe_expression
        + ") AS TEXT)), '') IS NULL "
        + f"THEN '{label_literal}' ELSE ({stripped_expression}) END"
    )


def _render_missing_group_select_item(expression: str, alias_sql: str | None) -> str:
    rewritten_expression = _build_missing_group_expression(expression)
    if alias_sql:
        return f"{rewritten_expression} AS {alias_sql}"
    return rewritten_expression


def _replace_sql_ranges(sql: str, replacements: list[tuple[int, int, str]]) -> str:
    rewritten_sql = sql
    for start, end, replacement in sorted(replacements, key=lambda item: item[0], reverse=True):
        rewritten_sql = rewritten_sql[:start] + replacement + rewritten_sql[end:]
    return rewritten_sql


def _rewrite_grouped_missing_category_sql(sql: str) -> str | None:
    normalized_sql = _normalized_statement(sql)
    group_by_range = _extract_top_level_clause_body_range(
        normalized_sql,
        "GROUP BY",
        ("HAVING", "ORDER BY", "LIMIT"),
    )
    if group_by_range is None:
        return None

    select_pos = _find_top_level_keyword(normalized_sql, "SELECT")
    from_pos = _find_top_level_keyword(normalized_sql, "FROM")
    if select_pos is None or from_pos is None or from_pos <= select_pos:
        return None

    select_clause = normalized_sql[select_pos + len("SELECT"):from_pos].strip()
    group_by_clause = normalized_sql[group_by_range[0]:group_by_range[1]].strip()
    select_items = _split_top_level_expressions(select_clause)
    group_by_items = _split_top_level_expressions(group_by_clause)
    if not select_items or not group_by_items:
        return None

    select_metadata: list[dict[str, Any]] = []
    rewritten_select_items = list(select_items)
    rewritten_group_by_items: list[str] = []
    any_rewrite = False

    for select_item in select_items:
        expression, alias_sql = _extract_select_item_parts(select_item)
        inferred_alias_sql = alias_sql or _infer_select_item_alias_sql(expression)
        match_keys = _reference_match_keys(expression)
        if alias_sql:
            match_keys.update(_reference_match_keys(alias_sql))
        elif inferred_alias_sql:
            match_keys.update(_reference_match_keys(inferred_alias_sql))
        select_metadata.append(
            {
                "expression": expression,
                "alias_sql": alias_sql,
                "display_alias_sql": inferred_alias_sql,
                "match_keys": match_keys,
            }
        )

    for group_by_item in group_by_items:
        stripped_group_by_item = group_by_item.strip()
        if not stripped_group_by_item:
            continue

        if re.fullmatch(r"\d+", stripped_group_by_item):
            ordinal = int(stripped_group_by_item)
            if 1 <= ordinal <= len(select_metadata):
                select_entry = select_metadata[ordinal - 1]
                rewritten_select_items[ordinal - 1] = _render_missing_group_select_item(
                    select_entry["expression"],
                    select_entry["display_alias_sql"],
                )
                any_rewrite = True
            rewritten_group_by_items.append(stripped_group_by_item)
            continue

        group_match_keys = _reference_match_keys(stripped_group_by_item)
        matched_index = None
        for index, select_entry in enumerate(select_metadata):
            if select_entry["match_keys"] & group_match_keys:
                matched_index = index
                break

        if matched_index is None:
            rewritten_group_by_items.append(_build_missing_group_expression(stripped_group_by_item))
            any_rewrite = True
            continue

        matched_entry = select_metadata[matched_index]
        rewritten_select_items[matched_index] = _render_missing_group_select_item(
            matched_entry["expression"],
            matched_entry["display_alias_sql"],
        )
        rewritten_group_by_items.append(_build_missing_group_expression(matched_entry["expression"]))
        any_rewrite = True

    if not any_rewrite:
        return None

    rewritten_sql = _replace_sql_ranges(
        normalized_sql,
        [
            (select_pos + len("SELECT"), from_pos, f" {', '.join(rewritten_select_items)} "),
            (group_by_range[0], group_by_range[1], f" {', '.join(rewritten_group_by_items)} "),
        ],
    )
    return _normalized_statement(rewritten_sql)


def _attach_display_sql(result: dict[str, Any], display_sql: str | None) -> dict[str, Any]:
    if display_sql:
        result["display_sql"] = display_sql
    return result


def _extract_source_segment(from_clause: str) -> str:
    from_body = from_clause[len("FROM"):].strip()
    where_pos = _find_top_level_keyword(from_body, "WHERE")
    if where_pos is None:
        return from_body
    return from_body[:where_pos].rstrip()


def _is_supported_object_level_source(source_segment: str) -> bool:
    if not source_segment or source_segment.startswith("("):
        return False
    if re.search(r"\bJOIN\b", source_segment, flags=re.IGNORECASE):
        return False
    return "," not in source_segment


def _extract_source_reference_name(source_segment: str) -> str | None:
    match = re.match(
        r'^(?P<source>"[^"]+"|[A-Za-z_][A-Za-z0-9_\.]*)'
        r'(?:\s+(?:AS\s+)?(?P<alias>"[^"]+"|[A-Za-z_][A-Za-z0-9_]*))?$',
        source_segment,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None

    alias = match.group("alias")
    if alias:
        return alias

    source = match.group("source")
    if source.startswith('"'):
        return source
    return source.rsplit(".", 1)[-1]


def _has_top_level_set_operation(sql: str) -> bool:
    return any(
        _find_top_level_keyword(sql, keyword) is not None
        for keyword in ("UNION", "INTERSECT", "EXCEPT")
    )


def _build_object_mode_sql(
    sql: str,
    object_id_column: str,
    object_order_column: str | None,
) -> tuple[str | None, str | None]:
    if _has_top_level_set_operation(sql):
        return None, "Object-level mode does not support UNION, INTERSECT, or EXCEPT queries."

    sections = _extract_top_level_query_sections(sql)
    if sections is None:
        return None, "Object-level mode requires a top-level SELECT or WITH query with a FROM clause."

    source_segment = _extract_source_segment(sections["from_clause"])
    if not _is_supported_object_level_source(source_segment):
        return None, (
            "Object-level mode currently supports only single-source top-level queries "
            "without joins, comma-separated sources, or subqueries in FROM."
        )

    source_reference_name = _extract_source_reference_name(source_segment)
    if source_reference_name is None:
        return None, "Object-level mode could not determine a stable source alias for canonicalization."

    order_expression = OBJECT_SOURCE_ORDINAL_COLUMN
    if object_order_column:
        order_expression = (
            f'{_quote_identifier(object_order_column)} ASC, '
            f"{OBJECT_SOURCE_ORDINAL_COLUMN} ASC"
        )

    ctes = ", ".join(
        [
            f"{OBJECT_SOURCE_CTE_NAME} AS (SELECT * {sections['from_clause']})",
            (
                f"{OBJECT_RANKED_CTE_NAME} AS ("
                f"SELECT {OBJECT_SOURCE_CTE_NAME}.*, "
                f"ROW_NUMBER() OVER () AS {OBJECT_SOURCE_ORDINAL_COLUMN} "
                f"FROM {OBJECT_SOURCE_CTE_NAME}"
                f")"
            ),
            (
                f"{OBJECT_CANONICAL_CTE_NAME} AS ("
                f"SELECT * FROM ("
                f"SELECT {OBJECT_RANKED_CTE_NAME}.*, "
                f"ROW_NUMBER() OVER ("
                f"PARTITION BY {_quote_identifier(object_id_column)} "
                f"ORDER BY {order_expression}"
                f") AS {OBJECT_ROW_NUMBER_COLUMN} "
                f"FROM {OBJECT_RANKED_CTE_NAME}"
                f") WHERE {OBJECT_ROW_NUMBER_COLUMN} = 1"
                f")"
            ),
        ]
    )

    prefix = sections["prefix"].strip()
    with_prefix = f"{prefix}, {ctes}" if prefix.upper().startswith("WITH") else f"WITH {ctes}"
    suffix = f" {sections['suffix']}" if sections["suffix"] else ""
    return (
        f"{with_prefix} SELECT {sections['select_clause']} "
        f"FROM {OBJECT_CANONICAL_CTE_NAME} AS {source_reference_name}{suffix}",
        None,
    )


def _schema_columns_by_name(schema_summary: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    columns_by_name: dict[str, list[dict[str, Any]]] = {}
    for table in schema_summary.get("tables", []):
        for column in table.get("columns", []):
            normalized_name = str(column.get("name", "")).strip().lower()
            if not normalized_name:
                continue
            columns_by_name.setdefault(normalized_name, []).append(column)
    return columns_by_name


def validate_object_level_columns(
    db_path: str | Path,
    object_id_column: str | None,
    object_order_column: str | None = None,
) -> str | None:
    normalized_object_id = str(object_id_column or "").strip()
    normalized_object_order = str(object_order_column or "").strip()

    if normalized_object_order and not normalized_object_id:
        return "Object-level mode requires object_id_column when object_order_column is configured."
    if not normalized_object_id:
        return None

    schema_summary = get_schema_summary(db_path)
    if schema_summary.get("status") != "success":
        return schema_summary.get("error") or "Unable to inspect the database schema."

    columns_by_name = _schema_columns_by_name(schema_summary)
    if normalized_object_id.lower() not in columns_by_name:
        return f"Configured object_id_column '{normalized_object_id}' was not found in the database schema."

    if normalized_object_order:
        order_columns = columns_by_name.get(normalized_object_order.lower())
        if not order_columns:
            return (
                f"Configured object_order_column '{normalized_object_order}' was not found in the database schema."
            )
        if all(not _is_orderable_declared_type(column.get("type")) for column in order_columns):
            return (
                f"Configured object_order_column '{normalized_object_order}' is not declared as an orderable SQLite type."
            )

    return None


def _iter_user_tables(connection: sqlite3.Connection) -> Iterable[str]:
    query = """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
          AND name NOT LIKE 'sqlite_%'
          AND substr(name, 1, 2) != '__'
        ORDER BY name
    """
    for row in connection.execute(query):
        yield row["name"]


def _load_column_mapping_lookup(connection: sqlite3.Connection) -> dict[tuple[str, str], str]:
    table_exists = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = '__column_mapping' LIMIT 1"
    ).fetchone()
    if table_exists is None:
        return {}

    lookup: dict[tuple[str, str], str] = {}
    try:
        rows = connection.execute(
            "SELECT table_name, csv_header, sqlite_column FROM __column_mapping"
        )
    except sqlite3.Error:
        return {}

    for row in rows:
        table_name = _normalize_whitespace(str(row["table_name"] or ""))
        csv_header = _normalize_whitespace(str(row["csv_header"] or ""))
        sqlite_column = _normalize_whitespace(str(row["sqlite_column"] or ""))
        if not table_name or not csv_header or not sqlite_column:
            continue
        lookup[(table_name.casefold(), sqlite_column.casefold())] = csv_header
    return lookup


def get_schema_summary(
    db_path: str | Path,
    *,
    table_names: Iterable[str] | None = None,
    max_tables: int | None = None,
    include_categorical_value_guidance: bool = False,
    max_categorical_values: int = DEFAULT_MAX_CATEGORICAL_VALUES,
) -> dict[str, Any]:
    try:
        path = _ensure_database_exists(db_path)
        with closing(_connect_read_only(path)) as connection:
            available_tables = list(_iter_user_tables(connection))
            column_mapping_lookup = _load_column_mapping_lookup(connection)
            if table_names is not None:
                allowed = {name for name in table_names}
                selected_tables = [name for name in available_tables if name in allowed]
            else:
                selected_tables = available_tables

            if max_tables is not None:
                selected_tables = selected_tables[:max_tables]

            tables: list[dict[str, Any]] = []
            categorical_value_guidance: list[dict[str, Any]] = []
            for table_name in selected_tables:
                pragma_sql = f"PRAGMA table_info({_quote_identifier(table_name)})"
                columns = []
                for column in connection.execute(pragma_sql):
                    column_definition = {
                        "name": column["name"],
                        "type": column["type"] or "TEXT",
                        "not_null": bool(column["notnull"]),
                        "default_value": column["dflt_value"],
                        "primary_key": bool(column["pk"]),
                    }
                    source_header = column_mapping_lookup.get(
                        (table_name.casefold(), column_definition["name"].casefold())
                    )
                    if source_header:
                        column_definition["source_header"] = source_header
                    if include_categorical_value_guidance:
                        categorical_values = _collect_categorical_values(
                            connection,
                            table_name,
                            column_definition["name"],
                            column_definition["type"],
                            max_categorical_values,
                        )
                        if categorical_values:
                            column_definition["categorical_values"] = categorical_values
                            guidance_entry = {
                                "table": table_name,
                                "column": column_definition["name"],
                                "values": categorical_values,
                            }
                            if source_header:
                                guidance_entry["source_header"] = source_header
                            categorical_value_guidance.append(guidance_entry)
                    columns.append(column_definition)
                tables.append({"name": table_name, "columns": columns})

        if not tables:
            schema_text = "No user-facing tables were found in the database."
        else:
            formatted_tables = []
            for table in tables:
                formatted_columns = []
                for column in table["columns"]:
                    details = [column["type"]]
                    if column["primary_key"]:
                        details.append("PRIMARY KEY")
                    if column["not_null"]:
                        details.append("NOT NULL")
                    formatted_columns.append(f'{column["name"]} {" ".join(details)}')
                formatted_tables.append(f'{table["name"]}({", ".join(formatted_columns)})')
            schema_text = "\n".join(formatted_tables)

        categorical_value_guidance_text = _format_categorical_value_guidance(
            categorical_value_guidance
        )

        return {
            "status": "success",
            "db_path": str(path),
            "schema_text": schema_text,
            "tables": tables,
            "table_count": len(tables),
            "categorical_value_guidance": categorical_value_guidance,
            "categorical_value_guidance_text": categorical_value_guidance_text,
        }
    except (FileNotFoundError, sqlite3.Error) as exc:
        return {
            "status": "error",
            "db_path": str(_as_path(db_path)),
            "schema_text": "",
            "tables": [],
            "table_count": 0,
            "categorical_value_guidance": [],
            "categorical_value_guidance_text": "",
            "error": str(exc),
        }


def validate_sql_read_only(sql: str, db_path: str | Path) -> dict[str, Any]:
    candidate = sql or ""
    try:
        path = _ensure_database_exists(db_path)
    except FileNotFoundError as exc:
        return {
            "is_valid": False,
            "reason": str(exc),
            "normalized_sql": "",
            "db_path": str(_as_path(db_path)),
        }

    _, _, statements = _scan_sql(candidate)

    if not statements:
        return {
            "is_valid": False,
            "reason": "The SQL query is empty after removing comments and whitespace.",
            "normalized_sql": "",
            "db_path": str(path),
        }

    if len(statements) > 1:
        return {
            "is_valid": False,
            "reason": "Only a single SQL statement is allowed.",
            "normalized_sql": "",
            "db_path": str(path),
        }

    normalized_sql = _normalized_statement(statements[0])
    if not normalized_sql:
        return {
            "is_valid": False,
            "reason": "The SQL query is empty after normalization.",
            "normalized_sql": "",
            "db_path": str(path),
        }

    first_keyword_match = re.match(r"^([A-Za-z]+)", normalized_sql)
    first_keyword = first_keyword_match.group(1).upper() if first_keyword_match else ""
    if first_keyword not in {"SELECT", "WITH"}:
        return {
            "is_valid": False,
            "reason": "Only read-only SELECT and WITH queries are allowed.",
            "normalized_sql": normalized_sql,
            "db_path": str(path),
        }

    effective_root_statement = _resolve_effective_root_statement(normalized_sql)
    if effective_root_statement != "SELECT":
        return {
            "is_valid": False,
            "reason": "Only read-only SELECT and WITH queries are allowed.",
            "normalized_sql": normalized_sql,
            "db_path": str(path),
        }

    try:
        with closing(_connect_read_only(path)) as connection:
            connection.execute(f"EXPLAIN QUERY PLAN {normalized_sql}")
    except sqlite3.Error as exc:
        return {
            "is_valid": False,
            "reason": f"SQLite could not validate the query against this schema: {exc}",
            "normalized_sql": normalized_sql,
            "db_path": str(path),
        }

    return {
        "is_valid": True,
        "reason": "",
        "normalized_sql": normalized_sql,
        "db_path": str(path),
    }


def execute_sqlite_query(
    db_path: str | Path,
    sql: str,
    *,
    preview_rows: int = DEFAULT_PREVIEW_ROWS,
    object_id_column: str | None = None,
    object_order_column: str | None = None,
) -> dict[str, Any]:
    preview_rows = max(1, preview_rows)
    validation = validate_sql_read_only(sql, db_path)
    normalized_sql = validation.get("normalized_sql") or _normalized_statement(sql or "")
    display_sql: str | None = None

    if not validation["is_valid"]:
        return _attach_display_sql({
            "status": "error",
            "db_path": validation["db_path"],
            "sql": normalized_sql,
            "columns": [],
            "rows": [],
            "row_count": 0,
            "preview_row_count": 0,
            "truncated": False,
            "error": validation["reason"],
        }, display_sql)

    effective_sql = normalized_sql
    rewritten_group_sql = _rewrite_grouped_missing_category_sql(normalized_sql)
    if rewritten_group_sql and rewritten_group_sql != normalized_sql:
        rewritten_validation = validate_sql_read_only(rewritten_group_sql, db_path)
        if not rewritten_validation["is_valid"]:
            return _attach_display_sql({
                "status": "error",
                "db_path": rewritten_validation["db_path"],
                "sql": normalized_sql,
                "columns": [],
                "rows": [],
                "row_count": 0,
                "preview_row_count": 0,
                "truncated": False,
                "error": (
                    "Grouped missing-category normalization could not rewrite the query safely: "
                    f"{rewritten_validation['reason']}"
                ),
            }, display_sql)
        effective_sql = rewritten_validation.get("normalized_sql") or _normalized_statement(rewritten_group_sql)
        display_sql = effective_sql

    rewritten_categorical_sql: str | None = None
    if re.search(r"!=|<>|\bNOT\s+IN\b", effective_sql, flags=re.IGNORECASE):
        schema_summary = get_schema_summary(
            db_path,
            include_categorical_value_guidance=True,
            max_categorical_values=DEFAULT_MAX_CATEGORICAL_VALUES,
        )
        if schema_summary.get("status") == "success":
            rewritten_categorical_sql = _rewrite_categorical_negation_sql(
                effective_sql,
                [
                    entry
                    for entry in schema_summary.get("categorical_value_guidance") or []
                    if isinstance(entry, dict)
                ],
            )
    if rewritten_categorical_sql and rewritten_categorical_sql != effective_sql:
        rewritten_validation = validate_sql_read_only(rewritten_categorical_sql, db_path)
        if not rewritten_validation["is_valid"]:
            return _attach_display_sql({
                "status": "error",
                "db_path": rewritten_validation["db_path"],
                "sql": normalized_sql,
                "columns": [],
                "rows": [],
                "row_count": 0,
                "preview_row_count": 0,
                "truncated": False,
                "error": (
                    "Categorical filter normalization could not rewrite the query safely: "
                    f"{rewritten_validation['reason']}"
                ),
            }, display_sql)
        effective_sql = rewritten_validation.get("normalized_sql") or _normalized_statement(rewritten_categorical_sql)
        display_sql = effective_sql

    if object_order_column and not object_id_column:
        return _attach_display_sql({
            "status": "error",
            "db_path": validation["db_path"],
            "sql": normalized_sql,
            "columns": [],
            "rows": [],
            "row_count": 0,
            "preview_row_count": 0,
            "truncated": False,
            "error": "Object-level mode requires object_id_column when object_order_column is configured.",
        }, display_sql)

    if object_id_column:
        object_mode_error = validate_object_level_columns(
            db_path,
            object_id_column,
            object_order_column,
        )
        if object_mode_error is not None:
            return _attach_display_sql({
                "status": "error",
                "db_path": validation["db_path"],
                "sql": normalized_sql,
                "columns": [],
                "rows": [],
                "row_count": 0,
                "preview_row_count": 0,
                "truncated": False,
                "error": object_mode_error,
            }, display_sql)

        object_sql, object_sql_error = _build_object_mode_sql(
            effective_sql,
            object_id_column,
            object_order_column,
        )
        if object_sql_error is not None or object_sql is None:
            return _attach_display_sql({
                "status": "error",
                "db_path": validation["db_path"],
                "sql": normalized_sql,
                "columns": [],
                "rows": [],
                "row_count": 0,
                "preview_row_count": 0,
                "truncated": False,
                "error": object_sql_error or "Object-level mode could not rewrite the query safely.",
            }, display_sql)

        rewritten_validation = validate_sql_read_only(object_sql, db_path)
        if not rewritten_validation["is_valid"]:
            return _attach_display_sql({
                "status": "error",
                "db_path": rewritten_validation["db_path"],
                "sql": normalized_sql,
                "columns": [],
                "rows": [],
                "row_count": 0,
                "preview_row_count": 0,
                "truncated": False,
                "error": (
                    "Object-level mode could not canonicalize the query safely: "
                    f"{rewritten_validation['reason']}"
                ),
            }, display_sql)
        effective_sql = rewritten_validation.get("normalized_sql") or _normalized_statement(object_sql)

    try:
        with closing(_connect_read_only(db_path)) as connection:
            cursor = connection.execute(effective_sql)
            columns = [description[0] for description in cursor.description or []]
            fetched_rows = cursor.fetchmany(preview_rows + 1)
            truncated = len(fetched_rows) > preview_rows
            preview = [dict(row) for row in fetched_rows[:preview_rows]]

            if truncated:
                count_sql = f"SELECT COUNT(*) AS total_count FROM ({effective_sql}) AS result_set"
                row_count = int(connection.execute(count_sql).fetchone()[0])
            else:
                row_count = len(preview)

        return _attach_display_sql({
            "status": "success",
            "db_path": str(_ensure_database_exists(db_path)),
            "sql": effective_sql,
            "columns": columns,
            "rows": preview,
            "row_count": row_count,
            "preview_row_count": len(preview),
            "truncated": truncated,
            "error": None,
        }, display_sql)
    except sqlite3.Error as exc:
        return _attach_display_sql({
            "status": "error",
            "db_path": str(_as_path(db_path)),
            "sql": effective_sql,
            "columns": [],
            "rows": [],
            "row_count": 0,
            "preview_row_count": 0,
            "truncated": False,
            "error": f"SQLite execution failed: {exc}",
        }, display_sql)


def count_subset_rows(db_path: str | Path, count_sql: str) -> int | None:
    """Execute a pre-built COUNT(*) query and return the integer result, or None on failure."""
    try:
        path = _ensure_database_exists(db_path)
        with closing(_connect_read_only(path)) as connection:
            row = connection.execute(count_sql).fetchone()
            if row is None:
                return None
            value = row[0]
            if isinstance(value, int):
                return value
            if isinstance(value, float) and value.is_integer():
                return int(value)
            return None
    except Exception:
        return None
