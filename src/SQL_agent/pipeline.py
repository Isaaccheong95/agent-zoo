from __future__ import annotations

from typing import Any, Callable

from .db import execute_sqlite_query, get_schema_summary, validate_sql_read_only


SQLGenerator = Callable[[str, dict[str, Any]], str | dict[str, Any] | None]


def _extract_sql(generator_result: str | dict[str, Any] | None) -> tuple[str | None, str | None]:
    if generator_result is None:
        return None, None
    if isinstance(generator_result, str):
        return generator_result, None
    return generator_result.get("sql"), generator_result.get("explanation")


def summarize_execution_result(execution_result: dict[str, Any]) -> str:
    if execution_result["status"] != "success":
        return execution_result.get("error") or "The query failed."

    row_count = execution_result["row_count"]
    if row_count == 0:
        return "No matching rows were found."
    if row_count == 1:
        return "Found 1 matching row."
    if execution_result["truncated"]:
        return f"Found {row_count} matching rows. Returning a preview."
    return f"Found {row_count} matching rows."


def run_nl_to_sql_pipeline(
    question: str,
    db_path: str,
    sql_generator: SQLGenerator,
    *,
    preview_rows: int = 20,
) -> dict[str, Any]:
    schema_summary = get_schema_summary(db_path)
    if schema_summary["status"] != "success":
        return {
            "status": "error",
            "question": question,
            "generated_sql": None,
            "schema": schema_summary,
            "validation": None,
            "execution": None,
            "summary": schema_summary.get("error", "Unable to inspect schema."),
            "explanation": "Schema inspection failed before SQL generation.",
        }

    generated_sql, generator_explanation = _extract_sql(sql_generator(question, schema_summary))
    if not generated_sql or not generated_sql.strip():
        return {
            "status": "error",
            "question": question,
            "generated_sql": None,
            "schema": schema_summary,
            "validation": None,
            "execution": None,
            "summary": "No SQL was generated for the request.",
            "explanation": generator_explanation or "The SQL generator returned an empty result.",
        }

    validation = validate_sql_read_only(generated_sql, db_path)
    execution = execute_sqlite_query(db_path, validation.get("normalized_sql") or generated_sql, preview_rows=preview_rows)

    return {
        "status": execution["status"],
        "question": question,
        "generated_sql": validation.get("normalized_sql") or generated_sql,
        "schema": schema_summary,
        "validation": validation,
        "execution": execution,
        "summary": summarize_execution_result(execution),
        "explanation": generator_explanation,
    }
