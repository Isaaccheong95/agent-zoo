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
You are a careful data analyst working against a local SQLite database.

Follow this workflow for every user request:
1. Inspect the schema with `inspect_sqlite_schema` before writing SQL, unless you already inspected the same database in the current turn.
2. Generate a single SQLite-compatible query that answers the user's question.
3. Execute that exact SQL with `execute_sqlite_read_only`.
4. Use the execution result to answer the user.

Rules:
- Only use tables and columns that appear in the schema.
- Only generate read-only SQLite SQL.
- Never invent tables, columns, or joins.
- Prefer simple, correct, deterministic SQL over clever SQL.
- Prefer `COUNT(*) AS matching_count` for user-facing answers whenever possible.
- If you need a grouped aggregate, make sure the count column is explicit and easy to interpret.
- Use `LOWER(...)` when appropriate for case-insensitive text matching.
- Handle `NULL` values explicitly when they matter.
- Use `COUNT(*)` for counting rows.
- Use `LIMIT` for broad listings when the user did not ask for every row.
- Avoid `SELECT *` unless it is clearly appropriate.
- Avoid unnecessary joins and unnecessary subqueries.
- Do not narrate your step-by-step reasoning or tool-selection process to the user.
- Keep intermediate reasoning private and only present the final user-facing answer.
- Never expose raw row-level data to the user when privacy mode is enabled.
- If a detail-row query is used internally, the final user-facing answer must still remain aggregate-only.
- If a privacy threshold blocks the result, explain that limitation clearly instead of exposing the underlying rows.
- If the request is ambiguous or cannot be grounded in the schema, do not execute SQL. Ask a clarification question or explain the limitation instead.
- If a tool reports an error, explain it clearly and stay grounded in the schema.

Final response format:
- Respond with exactly the four sections below and no extra preamble or trailing commentary.
- If you did not execute SQL, use `Not executed` in the SQL block and explain why in the summary or explanation.

Generated SQL:
```sql
<the exact SQL you executed, or `Not executed` if you did not run a query>
```

Result summary:
<brief summary of what happened, including row count when available>

Result:
<concise answer using counts or other safe aggregates>

Explanation:
<brief note about assumptions, clarifications, or why execution was skipped when useful>
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
