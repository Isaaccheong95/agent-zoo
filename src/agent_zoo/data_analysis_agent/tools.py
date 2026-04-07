"""Expose direct dataset-analysis tools for the standalone data analysis agent."""

from __future__ import annotations

from typing import Callable

from .config import DataAnalysisAgentSettings

try:
    from ..sql_agent.db import execute_sqlite_query, get_schema_summary
except ImportError:  # Support ADK loading this package as top-level `data_analysis_agent`.
    from sql_agent.db import execute_sqlite_query, get_schema_summary  # type: ignore[no-redef]


def build_data_analysis_tools(settings: DataAnalysisAgentSettings) -> list[Callable]:
    try:
        from .agent import analyze_query_result
    except ImportError:  # Support ADK loading this package as top-level `data_analysis_agent`.
        from agent import analyze_query_result  # type: ignore[no-redef]

    def inspect_dataset_schema() -> dict:
        """Inspect the SQLite schema for the configured dataset."""

        return get_schema_summary(settings.db_path)

    def execute_dataset_read_only(sql: str) -> dict:
        """Validate and execute one read-only SQLite query against the configured dataset."""

        return execute_sqlite_query(
            settings.db_path,
            sql,
            preview_rows=settings.preview_rows,
            object_id_column=settings.object_id_column,
            object_order_column=settings.object_order_column,
        )

    def analyze_dataset_with_sql(question: str, sql: str) -> dict:
        """Execute one read-only query and return both the query result and a structured analysis."""

        schema_summary = get_schema_summary(settings.db_path)
        schema_text = schema_summary.get("schema_text") if schema_summary.get("status") == "success" else None
        query_result = execute_dataset_read_only(sql=sql)
        if query_result.get("status") != "success":
            return {
                "status": "error",
                "question": question,
                "sql": query_result.get("sql") or sql,
                "instructions": None,
                "query_result": query_result,
                "analysis": None,
                "error": query_result.get("error"),
            }

        analysis_result = analyze_query_result(
            query_result,
            question=question,
            schema_text=schema_text,
            instructions=None,
        )
        return {
            "status": "success",
            "question": question,
            "sql": query_result.get("sql") or sql,
            "instructions": None,
            "query_result": query_result,
            "analysis": analysis_result.to_dict(),
            "error": None,
        }

    inspect_dataset_schema.__name__ = "inspect_dataset_schema"
    execute_dataset_read_only.__name__ = "execute_dataset_read_only"
    analyze_dataset_with_sql.__name__ = "analyze_dataset_with_sql"
    return [inspect_dataset_schema, execute_dataset_read_only, analyze_dataset_with_sql]