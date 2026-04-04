"""Run the SQL agent in single-question or interactive in-memory sessions.

This module provides async helpers that create an ADK `InMemoryRunner`, manage
sessions, stream events, and return the final agent response. It powers the CLI
entrypoint in `agent_zoo.sql_agent.cli` and can also be imported by other code.

To run the interactive CLI, use `uv run run-sql-agent`.
To ask one question and exit, use
`uv run run-sql-agent --question "How many passengers survived?"`.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Iterable

from google.adk.runners import InMemoryRunner
from google.genai.types import Content, Part

from .callbacks import (
    SQL_INTERNAL_QUERY_RESULT_STATE_KEY,
    SQL_INTERNAL_RESULT_REF_STATE_KEY,
    SQL_PUBLIC_RESULT_STATE_KEY,
)
from .config import SQLAgentSettings
from .db import get_schema_summary
from .result import SQLAgentStructuredResult, build_structured_result


def _text_from_parts(parts: Iterable[Part]) -> str:
    return "".join(part.text for part in parts if part.text)


def _print_debug_event(event, settings: SQLAgentSettings) -> None:
    if event.content and event.content.parts:
        text = _text_from_parts(event.content.parts)
        if text:
            print(f"[debug][{event.author}] {text}")
        for part in event.content.parts:
            function_call = getattr(part, "function_call", None)
            function_response = getattr(part, "function_response", None)
            if function_call is not None:
                print(f"[debug][tool-call] {function_call.name}: {function_call.args}")
            if function_response is not None:
                if (
                    settings.count_aggregates_only
                    and function_response.name == "execute_sqlite_read_only"
                ):
                    print(f"[debug][tool-response] {function_response.name}: <redacted in privacy mode>")
                else:
                    print(f"[debug][tool-response] {function_response.name}: {function_response.response}")


async def _run_question(
    question: str,
    settings: SQLAgentSettings,
    *,
    runner: InMemoryRunner | None = None,
    session_id: str | None = None,
) -> tuple[str, dict[str, object]]:
    from .agent import build_root_agent

    local_runner = runner or InMemoryRunner(agent=build_root_agent(settings), app_name=settings.app_name)
    active_session_id = session_id or settings.session_id

    if runner is None:
        await local_runner.session_service.create_session(
            app_name=local_runner.app_name,
            user_id=settings.user_id,
            session_id=active_session_id,
        )

    content = Content(role="user", parts=[Part(text=question)])
    final_response = "No final response was received from the agent."

    async for event in local_runner.run_async(
        user_id=settings.user_id,
        session_id=active_session_id,
        new_message=content,
    ):
        if settings.debug:
            _print_debug_event(event, settings)

        if event.is_final_response() and event.author != "user" and event.content and event.content.parts:
            final_response = _text_from_parts(event.content.parts)

    session = await local_runner.session_service.get_session(
        app_name=local_runner.app_name,
        user_id=settings.user_id,
        session_id=active_session_id,
    )
    return final_response, dict(getattr(session, "state", {}) or {})


async def ask_question(
    question: str,
    settings: SQLAgentSettings,
    *,
    runner: InMemoryRunner | None = None,
    session_id: str | None = None,
) -> str:
    final_response, _ = await _run_question(
        question,
        settings,
        runner=runner,
        session_id=session_id,
    )
    return final_response


async def ask_question_result(
    question: str,
    settings: SQLAgentSettings,
    *,
    runner: InMemoryRunner | None = None,
    session_id: str | None = None,
    capture_internal_rows: bool = True,
) -> SQLAgentStructuredResult:
    active_settings = settings
    if capture_internal_rows and runner is None and not settings.capture_internal_rows:
        active_settings = replace(settings, capture_internal_rows=True)

    final_response, state = await _run_question(
        question,
        active_settings,
        runner=runner,
        session_id=session_id,
    )

    schema_summary = get_schema_summary(active_settings.db_path)
    schema_text = schema_summary.get("schema_text") if schema_summary.get("status") == "success" else None

    internal_result = None
    if state.get(SQL_INTERNAL_RESULT_REF_STATE_KEY) == SQL_INTERNAL_QUERY_RESULT_STATE_KEY:
        internal_result = state.get(SQL_INTERNAL_QUERY_RESULT_STATE_KEY)

    return build_structured_result(
        question=question,
        final_response=final_response,
        public_result=state.get(SQL_PUBLIC_RESULT_STATE_KEY),
        internal_result=internal_result,
        schema_text=schema_text,
    )


async def run_interactive_loop(settings: SQLAgentSettings) -> None:
    from .agent import build_root_agent

    runner = InMemoryRunner(agent=build_root_agent(settings), app_name=settings.app_name)
    await runner.session_service.create_session(
        app_name=runner.app_name,
        user_id=settings.user_id,
        session_id=settings.session_id,
    )

    print(f"Running SQL agent against {settings.db_path}")
    print("Type 'exit' or 'quit' to stop.")

    while True:
        question = input("\nYou > ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        response = await ask_question(
            question,
            settings,
            runner=runner,
            session_id=settings.session_id,
        )
        print(f"\nAgent > {response}")


def run_interactive_loop_sync(settings: SQLAgentSettings) -> None:
    asyncio.run(run_interactive_loop(settings))
