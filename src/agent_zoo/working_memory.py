"""Shared helpers for per-agent session-scoped working memory."""

from __future__ import annotations

import copy
from typing import Any


AGENT_WORKING_MEMORY_STATE_KEY = "agent_working_memory"


def get_agent_working_memory(state: Any, agent_name: str) -> dict[str, Any]:
    if state is None or not hasattr(state, "get"):
        return {}
    root_memory = state.get(AGENT_WORKING_MEMORY_STATE_KEY)
    if not isinstance(root_memory, dict):
        return {}
    agent_memory = root_memory.get(agent_name)
    if not isinstance(agent_memory, dict):
        return {}
    return copy.deepcopy(agent_memory)


def set_agent_working_memory(state: Any, agent_name: str, memory: dict[str, Any] | None) -> None:
    if state is None:
        return

    root_memory = state.get(AGENT_WORKING_MEMORY_STATE_KEY)
    normalized_root_memory = copy.deepcopy(root_memory) if isinstance(root_memory, dict) else {}

    if isinstance(memory, dict) and memory:
        normalized_root_memory[agent_name] = copy.deepcopy(memory)
        state[AGENT_WORKING_MEMORY_STATE_KEY] = normalized_root_memory
        return

    normalized_root_memory.pop(agent_name, None)
    if normalized_root_memory:
        state[AGENT_WORKING_MEMORY_STATE_KEY] = normalized_root_memory
        return

    if hasattr(state, "pop"):
        state.pop(AGENT_WORKING_MEMORY_STATE_KEY, None)
    elif AGENT_WORKING_MEMORY_STATE_KEY in state:
        state[AGENT_WORKING_MEMORY_STATE_KEY] = None


def get_agent_working_memory_value(
    state: Any,
    agent_name: str,
    field_name: str,
) -> Any:
    return get_agent_working_memory(state, agent_name).get(field_name)


def set_agent_working_memory_value(
    state: Any,
    agent_name: str,
    field_name: str,
    value: Any,
) -> None:
    agent_memory = get_agent_working_memory(state, agent_name)
    if value is None:
        agent_memory.pop(field_name, None)
    else:
        agent_memory[field_name] = copy.deepcopy(value)
    set_agent_working_memory(state, agent_name, agent_memory)