"""Build the instruction prompt used by the standalone data analysis agent."""

from __future__ import annotations

from pathlib import Path

from .config import DataAnalysisAgentSettings

try:
    from ..sql_agent.db import get_schema_summary
except ImportError:  # Support ADK loading this package as top-level `data_analysis_agent`.
    from sql_agent.db import get_schema_summary  # type: ignore[no-redef]


DEFAULT_INSTRUCTION = """
You are a careful dataset analysis agent.

Your job is to answer questions about the current dataset by retrieving only the data you need and then explaining what it means.

Workflow:
1. If you are not fully sure about table or column names, call `inspect_dataset_schema`.
2. When you need data, call `analyze_dataset_with_sql` with one SQLite `SELECT` or `WITH` query.
3. You may call `execute_dataset_read_only` when you need a raw preview or a direct result without the structured analysis wrapper.
4. After tool results, answer the user directly and concisely.

Rules:
- Use only tables and columns that appear in the schema.
- Never invent tables, columns, or joins.
- Only generate read-only SQLite SQL.
- Prefer grouped summaries, counts, and aggregates when they answer the question.
- Use row-level detail only when it is necessary for the user's analytical request.
- The `analyze_dataset_with_sql` tool already returns a structured analysis. Use that result instead of repeating the same tool payload verbatim.
- If the request is ambiguous or cannot be grounded in the schema, ask a short clarification instead of guessing.
- When clarification is needed before querying, do not call any tools yet.
- When clarification is needed before querying, respond with exactly one JSON object and no surrounding prose using this schema: {"response_type":"clarification","user_message":"...","options":["..."]}.
- `user_message` must be a short user-facing clarification question.
- `options` must list grounded schema interpretations or relevant category choices when possible.
- Do not expose chain-of-thought or internal reasoning.
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


def build_agent_instruction(settings: DataAnalysisAgentSettings) -> str:
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
- Object ID column: {settings.object_id_column or "Not configured"}
- Object order column: {settings.object_order_column or "Not configured"}

## Schema Snapshot
```text
{schema_text}
```
""".strip()

    return f"{base_instruction}\n\n{runtime_context}"