"""Load and normalize configuration for the SQL agent.

This module resolves database paths, model settings, debug flags, and other
runtime defaults used across the SQL agent package. It is normally imported by
the CLI entrypoint and helper modules rather than run directly.

To exercise these settings in the full agent flow, run
`uv run run_sql_agent.py --db dataset\\titantic\\titanic.sqlite`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DB_PATH = PROJECT_ROOT / "dataset" / "titantic" / "titanic.sqlite"
DEFAULT_MODEL = "openai/Qwen3.5-0.8B-GGUF"
DEFAULT_PREVIEW_ROWS = 20
DEFAULT_COUNT_AGGREGATES_ONLY = True
DEFAULT_MINIMUM_AGGREGATE_COUNT = 3
DEFAULT_CAPTURE_INTERNAL_ROWS = False


def _ensure_local_openai_api_key() -> None:
    if os.getenv("OPENAI_API_BASE") and not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = "local-openai-compatible-key"


def _parse_bool(value: str | bool | None, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return value.strip().lower() in {"1", "true", "yes", "on"}


def resolve_repo_path(raw_path: str | Path | None) -> Path | None:
    if raw_path is None:
        return None

    path = raw_path if isinstance(raw_path, Path) else Path(raw_path)
    if path.is_absolute():
        return path

    cwd_candidate = (Path.cwd() / path).resolve()
    project_candidate = (PROJECT_ROOT / path).resolve()

    if cwd_candidate.exists():
        return cwd_candidate
    if project_candidate.exists():
        return project_candidate
    return project_candidate


@dataclass(slots=True)
class SQLAgentSettings:
    db_path: Path
    model: str
    debug: bool = False
    instruction_file: Path | None = None
    preview_rows: int = DEFAULT_PREVIEW_ROWS
    count_aggregates_only: bool = DEFAULT_COUNT_AGGREGATES_ONLY
    minimum_aggregate_count: int = DEFAULT_MINIMUM_AGGREGATE_COUNT
    capture_internal_rows: bool = DEFAULT_CAPTURE_INTERNAL_ROWS
    app_name: str = "sql_agent"
    user_id: str = "local_user"
    session_id: str = "sql_agent_session"


def load_settings(overrides: dict[str, Any] | None = None) -> SQLAgentSettings:
    overrides = overrides or {}
    _ensure_local_openai_api_key()

    raw_db_path = overrides.get("db_path")
    if raw_db_path is None:
        raw_db_path = os.getenv("SQL_AGENT_DB_PATH", str(DEFAULT_DB_PATH))

    raw_model = overrides.get("model")
    if raw_model is None:
        raw_model = os.getenv("SQL_AGENT_MODEL", DEFAULT_MODEL)

    raw_debug = overrides.get("debug")
    if raw_debug is None:
        raw_debug = os.getenv("SQL_AGENT_DEBUG")

    raw_instruction_file = overrides.get("instruction_file")
    if raw_instruction_file is None:
        raw_instruction_file = os.getenv("SQL_AGENT_INSTRUCTION_FILE")

    raw_preview_rows = overrides.get("preview_rows")
    if raw_preview_rows is None:
        raw_preview_rows = os.getenv("SQL_AGENT_PREVIEW_ROWS")

    raw_count_aggregates_only = overrides.get("count_aggregates_only")
    if raw_count_aggregates_only is None:
        raw_count_aggregates_only = os.getenv("SQL_AGENT_COUNT_AGGREGATES_ONLY")

    raw_minimum_aggregate_count = overrides.get("minimum_aggregate_count")
    if raw_minimum_aggregate_count is None:
        raw_minimum_aggregate_count = os.getenv("SQL_AGENT_MINIMUM_AGGREGATE_COUNT")

    raw_capture_internal_rows = overrides.get("capture_internal_rows")
    if raw_capture_internal_rows is None:
        raw_capture_internal_rows = os.getenv("SQL_AGENT_CAPTURE_INTERNAL_ROWS")

    resolved_instruction = resolve_repo_path(raw_instruction_file)
    preview_rows = DEFAULT_PREVIEW_ROWS
    if raw_preview_rows not in (None, ""):
        preview_rows = max(1, int(raw_preview_rows))

    minimum_aggregate_count = DEFAULT_MINIMUM_AGGREGATE_COUNT
    if raw_minimum_aggregate_count not in (None, ""):
        minimum_aggregate_count = max(1, int(raw_minimum_aggregate_count))

    return SQLAgentSettings(
        db_path=resolve_repo_path(raw_db_path) or DEFAULT_DB_PATH,
        model=str(raw_model),
        debug=_parse_bool(raw_debug, default=False),
        instruction_file=resolved_instruction,
        preview_rows=preview_rows,
        count_aggregates_only=_parse_bool(
            raw_count_aggregates_only,
            default=DEFAULT_COUNT_AGGREGATES_ONLY,
        ),
        minimum_aggregate_count=minimum_aggregate_count,
        capture_internal_rows=_parse_bool(
            raw_capture_internal_rows,
            default=DEFAULT_CAPTURE_INTERNAL_ROWS,
        ),
    )
