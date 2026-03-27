"""Build the instruction prompt used by the SQL agent.

This module defines the default analyst-style prompt, optionally loads a custom
instruction file, and appends runtime context such as the active database path
and schema snapshot. It is consumed by the agent builder and is not intended to
be run directly.

To use these instructions in practice, run `uv run run_sql_agent.py`.
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
2. Then call `execute_sqlite_read_only` with one SQLite `SELECT` or `WITH` query.
3. After `execute_sqlite_read_only` returns, stop. Do not call more tools for the same question.

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
- Do not use window functions such as `OVER (...)`.
- Use simple, deterministic SQL.
- Do not repeat the same query after a successful result.
- If the SQL result is privacy-blocked, stop and let the system return that limitation.
- Do not produce long prose, chain-of-thought, or repeated analysis.
- If the request is ambiguous or cannot be grounded in the schema, ask for clarification instead of guessing.
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
    schema_summary = get_schema_summary(settings.db_path)

    if schema_summary["status"] == "success":
        schema_text = schema_summary["schema_text"]
    else:
        schema_text = f"Schema unavailable: {schema_summary.get('error', 'Unknown error')}"

    runtime_context = f"""

## Runtime Context
- Default database path: {settings.db_path}
- Default preview rows: {settings.preview_rows}
- Count aggregates only: {settings.count_aggregates_only}
- Minimum aggregate count: {settings.minimum_aggregate_count}
- Capture internal rows: {settings.capture_internal_rows}

## Schema Snapshot
```text
{schema_text}
```
""".strip()

    return f"{base_instruction}\n\n{runtime_context}"
