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
from typing import Iterable

from google.adk.runners import InMemoryRunner
from google.genai.types import Content, Part

from .config import SQLAgentSettings


_RUNNER_CACHE: dict[tuple[object, ...], InMemoryRunner] = {}
_INITIALIZED_SESSION_KEYS: set[tuple[tuple[object, ...], str]] = set()


def _runner_cache_key(settings: SQLAgentSettings) -> tuple[object, ...]:
    return (
        settings.app_name,
        settings.user_id,
        str(settings.db_path),
        settings.model,
        settings.openai_api_base,
        str(settings.instruction_file) if settings.instruction_file is not None else None,
        settings.preview_rows,
        settings.count_aggregates_only,
        settings.minimum_aggregate_count,
        settings.capture_internal_rows,
        settings.object_id_column,
        settings.object_order_column,
    )


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


async def ask_question(
    question: str,
    settings: SQLAgentSettings,
    *,
    runner: InMemoryRunner | None = None,
    session_id: str | None = None,
) -> str:
    from .agent import build_root_agent

    active_session_id = session_id or settings.session_id
    if runner is None:
        cache_key = _runner_cache_key(settings)
        local_runner = _RUNNER_CACHE.get(cache_key)
        if local_runner is None:
            local_runner = InMemoryRunner(agent=build_root_agent(settings), app_name=settings.app_name)
            _RUNNER_CACHE[cache_key] = local_runner

        session_key = (cache_key, active_session_id)
        if session_key not in _INITIALIZED_SESSION_KEYS:
            await local_runner.session_service.create_session(
                app_name=local_runner.app_name,
                user_id=settings.user_id,
                session_id=active_session_id,
            )
            _INITIALIZED_SESSION_KEYS.add(session_key)
    else:
        local_runner = runner

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

    return final_response


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
