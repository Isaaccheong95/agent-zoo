from __future__ import annotations

from google.genai import types

from .formatting import format_structured_response


LAST_QUERY_RESULT_STATE_KEY = "temp:last_execute_sqlite_read_only_result"


def remember_query_result(tool, args: dict, tool_context, tool_response: dict, **kwargs) -> dict | None:
    tool_name = getattr(tool, "name", "")
    if tool_name == "execute_sqlite_read_only":
        tool_context.state[LAST_QUERY_RESULT_STATE_KEY] = tool_response
    return None


def format_final_agent_response(callback_context=None, **kwargs) -> types.Content | None:
    context = callback_context
    if context is None:
        return None

    last_query_result = context.state.get(LAST_QUERY_RESULT_STATE_KEY)
    if not isinstance(last_query_result, dict):
        return None

    return types.Content(
        role="model",
        parts=[types.Part(text=format_structured_response(last_query_result))],
    )
