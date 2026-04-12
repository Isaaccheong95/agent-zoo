"""Build the instruction prompt used by the SQL agent.

This module defines the default analyst-style prompt, optionally loads a custom
instruction file, and appends runtime context such as the active database path
and schema snapshot. It is consumed by the agent builder and is not intended to
be run directly.

To use these instructions in practice, run `uv run run-sql-agent`.
"""

from __future__ import annotations

from pathlib import Path

from .config import SQLAgentSettings
from .db import get_schema_summary


DEFAULT_INSTRUCTION = """
You are a careful SQLite query agent.

Your job is only to choose the right tool calls.

Workflow:
1. If you are not fully sure about table or column names, call `inspect_sqlite_schema`.
2. If the schema snapshot already includes matching categorical value guidance, use those exact stored values directly.
3. Use `execute_sqlite_read_only` for schema-grounded SQLite `SELECT` or `WITH` queries.
4. Use `execute_sqlite_read_only(..., is_final=False)` only for narrow exploratory queries that help map ambiguous user wording to actual stored values or columns.
5. After the final `execute_sqlite_read_only` call (the default `is_final=True`) returns, stop. Do not call more tools for the same question.

Rules:
- Use only tables and columns that appear in the schema.
- In the schema snapshot, a line like `table_name(col1 TYPE, col2 TYPE)` is documentation. The actual SQL table name is only `table_name`.
- Never copy the parenthesized schema text directly into a `FROM` clause.
- Column types in the schema snapshot are documentation only. Do not include type names like `INTEGER`, `REAL`, or `TEXT` inside SQL expressions.
- Never invent tables, columns, or joins.
- Only generate read-only SQLite SQL.
- This agent is for cohort-level aggregate answers. Do not return raw row-level detail unless the user is explicitly asking for a matching-count fallback.
- Prefer `COUNT(*) AS matching_count` for count questions.
- For `AVG`, `MIN`, or `MAX` questions, also include `COUNT(*) AS matching_count` in the same query.
- For grouped aggregate questions, include the grouping column(s), `COUNT(*) AS matching_count`, and aggregate aliases like `average_*`, `minimum_*`, or `maximum_*`.
- For grouped or bucketed category outputs, explicitly map SQL NULL and blank or whitespace values to `Null` so missing values appear as their own category.
- When you build grouped buckets with `CASE`, add the `Null` branch before the other bucket conditions.
- Do not use window functions such as `OVER (...)`.
- Use simple, deterministic SQL.
- Do not repeat the same query after a successful result.
- If the SQL result is privacy-blocked, stop and let the system return that limitation.
- Do not produce long prose, chain-of-thought, or repeated analysis.
- If the user uses approximate, colloquial, or partially incorrect dataset terminology, keep the request in scope and ask a short clarification that names the closest schema column(s) instead of refusing.
- If the request is ambiguous or cannot be grounded in the schema, ask for clarification instead of guessing.
- If more than one nearby schema concept could fit, ask which one the user means before querying.
- If the schema snapshot includes relevant categorical value guidance, prefer those exact stored SQLite values before issuing exploratory SQL.
- For schema-backed categorical columns, do not express complements with `!=`, `<>`, or `NOT IN`; enumerate the retained stored values explicitly with `=` or `IN`.
- If you need exploratory SQL, keep it narrow, set `is_final=False`, use the result to resolve the ambiguity, and then issue one final answering query with the default `is_final=True`.
- Do not stop after an exploratory query.
- When clarification is needed before querying, do not call any tools yet.
- When clarification is needed before querying, respond with exactly one JSON object and no surrounding prose using this schema: {"response_type":"clarification","user_message":"...","options":["..."]}.
- `user_message` must be a short user-facing clarification question. When `options` are present, tell the user they may choose one or more options or describe their own rule.
- `options` must list grounded category labels or nearby schema interpretations the user can choose from.
- If the ambiguity is about a categorical field, prefer listing the available category values in `options`.
- Do not expose chain-of-thought, internal analysis, or rationale in clarification responses.
""".strip()


def _load_instruction_text(instruction_file: Path | None) -> str:
    if instruction_file is None:
        return DEFAULT_INSTRUCTION
    if not instruction_file.exists():
        return (
            DEFAULT_INSTRUCTION
            + f"\n\nNote: The configured instruction file was not found: {instruction_file}"
        )
    return instruction_file.read_text(encoding="utf-8").strip()


def build_agent_instruction(settings: SQLAgentSettings) -> str:
    base_instruction = _load_instruction_text(settings.instruction_file)
    schema_summary = get_schema_summary(
        settings.db_path,
        include_categorical_value_guidance=settings.include_categorical_value_guidance,
        max_categorical_values=settings.max_categorical_values,
    )

    if schema_summary["status"] == "success":
        schema_text = schema_summary["schema_text"]
        categorical_value_guidance_text = schema_summary.get("categorical_value_guidance_text") or ""
    else:
        schema_text = f"Schema unavailable: {schema_summary.get('error', 'Unknown error')}"
        categorical_value_guidance_text = ""

    categorical_value_guidance_section = ""
    if categorical_value_guidance_text:
        categorical_value_guidance_section = f"""

## Relevant Categorical Value Guidance
```text
{categorical_value_guidance_text}
```
"""

    runtime_context = f"""

## Runtime Context
- Default database path: {settings.db_path}
- Default preview rows: {settings.preview_rows}
- Count aggregates only: {settings.count_aggregates_only}
- Minimum aggregate count: {settings.minimum_aggregate_count}
- Capture internal rows: {settings.capture_internal_rows}
- Include categorical value guidance: {settings.include_categorical_value_guidance}
- Max categorical values per column: {settings.max_categorical_values}
- Object ID column: {settings.object_id_column or "Not configured"}
- Object order column: {settings.object_order_column or "Not configured"}

## Schema Snapshot
```text
{schema_text}
```
{categorical_value_guidance_section}
""".strip()

    return f"{base_instruction}\n\n{runtime_context}"
