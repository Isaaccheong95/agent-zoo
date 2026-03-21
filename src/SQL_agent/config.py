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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "dataset" / "titantic" / "titanic.sqlite"
DEFAULT_MODEL = "openai/Qwen3.5-0.8B-GGUF"
DEFAULT_PREVIEW_ROWS = 20


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

    resolved_instruction = resolve_repo_path(raw_instruction_file)
    preview_rows = DEFAULT_PREVIEW_ROWS
    if raw_preview_rows not in (None, ""):
        preview_rows = max(1, int(raw_preview_rows))

    return SQLAgentSettings(
        db_path=resolve_repo_path(raw_db_path) or DEFAULT_DB_PATH,
        model=str(raw_model),
        debug=_parse_bool(raw_debug, default=False),
        instruction_file=resolved_instruction,
        preview_rows=preview_rows,
    )
