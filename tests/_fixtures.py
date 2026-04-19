"""Shared test fixtures for the SQL agent test suite."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from google.genai import types

from agent_zoo.working_memory import (
    get_agent_working_memory,
    get_agent_working_memory_value,
    set_agent_working_memory_value,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def create_fixture_database(db_path: Path) -> None:
    connection = sqlite3.connect(db_path)
    connection.executescript(
        """
        CREATE TABLE people (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            sex TEXT,
            age INTEGER
        );

        CREATE TABLE visits (
            id INTEGER PRIMARY KEY,
            person_id INTEGER NOT NULL,
            city TEXT NOT NULL,
            FOREIGN KEY(person_id) REFERENCES people(id)
        );

        CREATE TABLE __shadow (
            id INTEGER PRIMARY KEY,
            ignored TEXT
        );

        CREATE TABLE __column_mapping (
            table_name TEXT NOT NULL,
            csv_header TEXT NOT NULL,
            sqlite_column TEXT NOT NULL
        );

        INSERT INTO people (name, sex, age) VALUES
            ('Alice', 'female', 30),
            ('Bob', 'male', 44),
            ('Cara', 'female', 19),
            ('Dana', 'female', 52);

        INSERT INTO visits (person_id, city) VALUES
            (1, 'Singapore'),
            (2, 'Tokyo'),
            (3, 'Singapore'),
            (4, 'Paris');

        INSERT INTO __column_mapping (table_name, csv_header, sqlite_column) VALUES
            ('people', 'Name', 'name'),
            ('people', 'Sex', 'sex'),
            ('people', 'Age', 'age'),
            ('visits', 'PersonId', 'person_id'),
            ('visits', 'City', 'city');
        """
    )
    connection.commit()
    connection.close()


def create_object_mode_fixture_database(db_path: Path) -> None:
    connection = sqlite3.connect(db_path)
    connection.executescript(
        """
        CREATE TABLE records (
            id INTEGER PRIMARY KEY,
            person_id TEXT NOT NULL,
            event_rank INTEGER NOT NULL,
            city TEXT NOT NULL,
            score REAL NOT NULL,
            payload BLOB
        );

        INSERT INTO records (person_id, event_rank, city, score, payload) VALUES
            ('p1', 2, 'Singapore', 10.0, x'01'),
            ('p1', 1, 'Tokyo', 20.0, x'02'),
            ('p2', 1, 'Paris', 30.0, x'03'),
            ('p3', 2, 'Paris', 40.0, x'04'),
            ('p3', 1, 'Tokyo', 50.0, x'05');
        """
    )
    connection.commit()
    connection.close()


def create_missing_group_fixture_database(db_path: Path) -> None:
    connection = sqlite3.connect(db_path)
    connection.executescript(
        """
        CREATE TABLE patients (
            id INTEGER PRIMARY KEY,
            patient_id TEXT NOT NULL,
            name TEXT NOT NULL,
            sex TEXT,
            age TEXT
        );

        INSERT INTO patients (patient_id, name, sex, age) VALUES
            ('p1', 'Anya', 'female', '14'),
            ('p2', 'Ben', 'male', '42'),
            ('p3', 'Cara', NULL, NULL),
            ('p3', 'Drew', '', ''),
            ('p4', 'Eli', ' ', ' '),
            ('p5', 'Fay', 'female', '33');
        """
    )
    connection.commit()
    connection.close()


def make_query_result(
    rows: list[dict],
    *,
    columns: list[str] | None = None,
    row_count: int | None = None,
    truncated: bool = False,
    status: str = "success",
    error: str | None = None,
    sql: str = "SELECT ...",
    display_sql: str | None = None,
) -> dict:
    result = {
        "status": status,
        "db_path": "fixture.sqlite",
        "sql": sql,
        "columns": columns if columns is not None else (list(rows[0].keys()) if rows else []),
        "rows": rows,
        "row_count": len(rows) if row_count is None else row_count,
        "preview_row_count": len(rows),
        "truncated": truncated,
        "error": error,
    }
    if display_sql is not None:
        result["display_sql"] = display_sql
    return result


def get_sql_working_memory(state: dict[str, object]) -> dict[str, object]:
    return get_agent_working_memory(state, "sql_agent")


def get_sql_working_memory_value(state: dict[str, object], field_name: str) -> object:
    return get_agent_working_memory_value(state, "sql_agent", field_name)


def set_sql_working_memory_value(state: dict[str, object], field_name: str, value: object) -> None:
    set_agent_working_memory_value(state, "sql_agent", field_name, value)


def get_sql_current_query_frame(state: dict[str, object]) -> dict[str, object] | None:
    value = get_sql_working_memory_value(state, "current_query_frame")
    return value if isinstance(value, dict) else None


def set_sql_current_query_frame(state: dict[str, object], query_frame: dict[str, object]) -> None:
    set_sql_working_memory_value(state, "current_query_frame", query_frame)


def get_sql_pending_clarification(state: dict[str, object]) -> dict[str, object] | None:
    value = get_sql_working_memory_value(state, "pending_clarification")
    return value if isinstance(value, dict) else None


def set_sql_pending_clarification(state: dict[str, object], clarification: dict[str, object]) -> None:
    set_sql_working_memory_value(state, "pending_clarification", clarification)


class FakeSessionService:
    """Stand-in for ADK session services: records create_session calls."""

    def __init__(self) -> None:
        self.created_sessions: list[dict[str, str]] = []

    async def create_session(self, **kwargs) -> None:
        self.created_sessions.append(kwargs)


class FakeFinalEvent:
    """Mimics an ADK final model event with a fixed text payload."""

    def __init__(self, text: str) -> None:
        self.author = "model"
        self.content = types.Content(role="model", parts=[types.Part(text=text)])

    def is_final_response(self) -> bool:
        return True


class FakeRunner:
    """Stand-in for ADK InMemoryRunner. Tracks instances on the class."""

    instances: list["FakeRunner"] = []

    def __init__(self, agent, app_name: str) -> None:
        self.agent = agent
        self.app_name = app_name
        self.session_service = FakeSessionService()
        FakeRunner.instances.append(self)

    async def run_async(self, **kwargs):
        yield FakeFinalEvent("ok")
