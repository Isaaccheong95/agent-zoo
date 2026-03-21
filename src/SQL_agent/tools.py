"""Expose the SQL agent's schema and query helpers as ADK tools.

This module wraps the low-level database helpers in tool functions that the
model can call through ADK. The tools are registered by `agent.py` and are not
meant to be run directly.

To use them through the full agent flow, run `uv run run_sql_agent.py`.
"""

from __future__ import annotations

from typing import Callable

from .config import SQLAgentSettings
from .db import execute_sqlite_query, get_schema_summary


def build_sql_tools(settings: SQLAgentSettings) -> list[Callable]:
    def inspect_sqlite_schema(db_path: str | None = None) -> dict:
        """Inspect the SQLite schema for the configured database or an override path.

        Args:
            db_path: Optional SQLite database path. If omitted, the configured default path is used.

        Returns:
            A dictionary containing the schema text and structured table metadata.
        """

        return get_schema_summary(db_path or settings.db_path)

    def execute_sqlite_read_only(
        sql: str,
        db_path: str | None = None,
        preview_rows: int | None = None,
    ) -> dict:
        """Validate and execute a read-only SQLite query.

        Args:
            sql: The SQLite SELECT or WITH query to execute.
            db_path: Optional SQLite database path. If omitted, the configured default path is used.
            preview_rows: Optional preview limit. If omitted, the configured default is used.

        Returns:
            A dictionary containing the SQL, rows, columns, row count, and any safe error message.
        """

        effective_preview_rows = settings.preview_rows if preview_rows is None else max(1, preview_rows)
        return execute_sqlite_query(db_path or settings.db_path, sql, preview_rows=effective_preview_rows)

    inspect_sqlite_schema.__name__ = "inspect_sqlite_schema"
    execute_sqlite_read_only.__name__ = "execute_sqlite_read_only"
    return [inspect_sqlite_schema, execute_sqlite_read_only]
