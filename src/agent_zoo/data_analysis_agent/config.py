"""Load and normalize configuration for the data analysis agent."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from ..sql_agent.config import SQLAgentSettings, load_settings as load_sql_settings
except ImportError:  # Support ADK loading this package as top-level `data_analysis_agent`.
    from sql_agent.config import SQLAgentSettings, load_settings as load_sql_settings  # type: ignore[no-redef]


@dataclass(slots=True)
class DataAnalysisAgentSettings:
    db_path: Path
    model: str
    openai_api_base: str | None = None
    preview_rows: int = 20
    object_id_column: str | None = None
    object_order_column: str | None = None
    instruction_file: Path | None = None
    app_name: str = "data_analysis_agent"
    user_id: str = "local_user"
    session_id: str = "data_analysis_agent_session"


def _from_sql_settings(settings: SQLAgentSettings) -> DataAnalysisAgentSettings:
    return DataAnalysisAgentSettings(
        db_path=settings.db_path,
        model=settings.model,
        openai_api_base=settings.openai_api_base,
        preview_rows=settings.preview_rows,
        object_id_column=settings.object_id_column,
        object_order_column=settings.object_order_column,
        instruction_file=settings.instruction_file,
    )


def load_settings(overrides: dict[str, Any] | None = None) -> DataAnalysisAgentSettings:
    return _from_sql_settings(load_sql_settings(overrides))