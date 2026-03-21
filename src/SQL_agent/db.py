from __future__ import annotations

from contextlib import closing
import re
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from .config import DEFAULT_PREVIEW_ROWS, resolve_repo_path


INTERNAL_TABLE_PREFIXES = ("sqlite_", "__")
UNSAFE_SQL_TOKENS = {
    "ALTER",
    "ANALYZE",
    "ATTACH",
    "BEGIN",
    "COMMIT",
    "CREATE",
    "DELETE",
    "DETACH",
    "DROP",
    "END",
    "INSERT",
    "PRAGMA",
    "REINDEX",
    "RELEASE",
    "REPLACE",
    "ROLLBACK",
    "SAVEPOINT",
    "UPDATE",
    "VACUUM",
}


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


def get_schema_summary(
    db_path: str | Path,
    *,
    table_names: Iterable[str] | None = None,
    max_tables: int | None = None,
) -> dict[str, Any]:
    try:
        path = _ensure_database_exists(db_path)
        with closing(_connect_read_only(path)) as connection:
            available_tables = list(_iter_user_tables(connection))
            if table_names is not None:
                allowed = {name for name in table_names}
                selected_tables = [name for name in available_tables if name in allowed]
            else:
                selected_tables = available_tables

            if max_tables is not None:
                selected_tables = selected_tables[:max_tables]

            tables: list[dict[str, Any]] = []
            for table_name in selected_tables:
                pragma_sql = f"PRAGMA table_info({_quote_identifier(table_name)})"
                columns = []
                for column in connection.execute(pragma_sql):
                    columns.append(
                        {
                            "name": column["name"],
                            "type": column["type"] or "TEXT",
                            "not_null": bool(column["notnull"]),
                            "default_value": column["dflt_value"],
                            "primary_key": bool(column["pk"]),
                        }
                    )
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

        return {
            "status": "success",
            "db_path": str(path),
            "schema_text": schema_text,
            "tables": tables,
            "table_count": len(tables),
        }
    except (FileNotFoundError, sqlite3.Error) as exc:
        return {
            "status": "error",
            "db_path": str(_as_path(db_path)),
            "schema_text": "",
            "tables": [],
            "table_count": 0,
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

    _, token_text, statements = _scan_sql(candidate)

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

    normalized_token_text = _normalize_whitespace(token_text).upper()
    unsafe_match = re.search(
        r"\b(" + "|".join(sorted(UNSAFE_SQL_TOKENS)) + r")\b",
        normalized_token_text,
    )
    if unsafe_match:
        return {
            "is_valid": False,
            "reason": f"Unsafe SQL token detected: {unsafe_match.group(1)}.",
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
) -> dict[str, Any]:
    preview_rows = max(1, preview_rows)
    validation = validate_sql_read_only(sql, db_path)
    normalized_sql = validation.get("normalized_sql") or _normalized_statement(sql or "")

    if not validation["is_valid"]:
        return {
            "status": "error",
            "db_path": validation["db_path"],
            "sql": normalized_sql,
            "columns": [],
            "rows": [],
            "row_count": 0,
            "preview_row_count": 0,
            "truncated": False,
            "error": validation["reason"],
        }

    try:
        with closing(_connect_read_only(db_path)) as connection:
            cursor = connection.execute(normalized_sql)
            columns = [description[0] for description in cursor.description or []]
            fetched_rows = cursor.fetchmany(preview_rows + 1)
            truncated = len(fetched_rows) > preview_rows
            preview = [dict(row) for row in fetched_rows[:preview_rows]]

            if truncated:
                count_sql = f"SELECT COUNT(*) AS total_count FROM ({normalized_sql}) AS result_set"
                row_count = int(connection.execute(count_sql).fetchone()[0])
            else:
                row_count = len(preview)

        return {
            "status": "success",
            "db_path": str(_ensure_database_exists(db_path)),
            "sql": normalized_sql,
            "columns": columns,
            "rows": preview,
            "row_count": row_count,
            "preview_row_count": len(preview),
            "truncated": truncated,
            "error": None,
        }
    except sqlite3.Error as exc:
        return {
            "status": "error",
            "db_path": str(_as_path(db_path)),
            "sql": normalized_sql,
            "columns": [],
            "rows": [],
            "row_count": 0,
            "preview_row_count": 0,
            "truncated": False,
            "error": f"SQLite execution failed: {exc}",
        }
