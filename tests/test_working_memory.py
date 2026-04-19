from __future__ import annotations

import unittest

from agent_zoo.working_memory import (
    AGENT_WORKING_MEMORY_STATE_KEY,
    get_agent_working_memory,
    get_agent_working_memory_value,
    set_agent_working_memory_value,
)


class WorkingMemoryTestCase(unittest.TestCase):
    def test_agent_namespaces_are_isolated(self) -> None:
        state: dict[str, object] = {}

        set_agent_working_memory_value(state, "sql_agent", "current_query", {"sql": "SELECT 1"})
        set_agent_working_memory_value(state, "analysis_agent", "current_query", {"sql": "SELECT 2"})

        self.assertEqual(
            get_agent_working_memory_value(state, "sql_agent", "current_query"),
            {"sql": "SELECT 1"},
        )
        self.assertEqual(
            get_agent_working_memory_value(state, "analysis_agent", "current_query"),
            {"sql": "SELECT 2"},
        )

    def test_setting_none_clears_empty_agent_namespace(self) -> None:
        state: dict[str, object] = {}

        set_agent_working_memory_value(state, "sql_agent", "current_query", {"sql": "SELECT 1"})
        set_agent_working_memory_value(state, "sql_agent", "current_query", None)

        self.assertNotIn(AGENT_WORKING_MEMORY_STATE_KEY, state)

    def test_reads_return_copies(self) -> None:
        state: dict[str, object] = {}
        set_agent_working_memory_value(state, "sql_agent", "current_query", {"sql": "SELECT 1"})

        snapshot = get_agent_working_memory(state, "sql_agent")
        snapshot["current_query"]["sql"] = "SELECT 2"

        self.assertEqual(
            get_agent_working_memory_value(state, "sql_agent", "current_query"),
            {"sql": "SELECT 1"},
        )