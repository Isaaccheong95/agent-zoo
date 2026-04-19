from __future__ import annotations

import asyncio
import unittest
import uuid
from unittest.mock import AsyncMock, patch

from agent_zoo.base import BaseAgent
from agent_zoo.sql_agent import SQLAgent, SQLAgentSettings
from agent_zoo.sql_agent import runtime as sql_runtime

from tests._fixtures import (
    REPO_ROOT,
    FakeRunner,
    create_fixture_database,
)


class SQLAgentRuntimeTestCase(unittest.TestCase):
    def setUp(self) -> None:
        temp_root = REPO_ROOT / ".tmp_test_runs"
        temp_root.mkdir(exist_ok=True)
        self.db_path = temp_root / f"{uuid.uuid4().hex}.sqlite"
        create_fixture_database(self.db_path)

    def tearDown(self) -> None:
        if self.db_path.exists():
            self.db_path.unlink()

    def test_namespace_exports_instantiable_sqlagent(self) -> None:
        agent = SQLAgent(
            settings=SQLAgentSettings(
                db_path=self.db_path,
                model="openai/test-model",
            )
        )

        self.assertIsInstance(agent, BaseAgent)
        self.assertEqual(agent.get_name(), "sql_agent")
        self.assertEqual(
            agent.get_description(),
            "Converts natural language questions into safe read-only SQLite queries and explains the results.",
        )

    def test_sqlagent_ask_delegates_to_runtime(self) -> None:
        settings = SQLAgentSettings(
            db_path=self.db_path,
            model="openai/test-model",
        )
        agent = SQLAgent(settings=settings)
        mocked_ask_question = AsyncMock(return_value="Found 2 matching rows.")

        with patch("agent_zoo.sql_agent.runtime.ask_question", new=mocked_ask_question):
            response = asyncio.run(
                agent.ask(
                    "How many female patients are below 45 years old?",
                    session_id="test-session",
                )
            )

        self.assertEqual(response, "Found 2 matching rows.")
        mocked_ask_question.assert_awaited_once_with(
            "How many female patients are below 45 years old?",
            settings,
            session_id="test-session",
        )

    def test_ask_question_reuses_cached_runner_for_same_session(self) -> None:
        settings = SQLAgentSettings(
            db_path=self.db_path,
            model="openai/test-model",
            app_name=f"test-app-{uuid.uuid4().hex}",
            session_id=f"test-session-{uuid.uuid4().hex}",
        )

        sql_runtime._RUNNER_CACHE.clear()
        sql_runtime._INITIALIZED_SESSION_KEYS.clear()
        FakeRunner.instances.clear()

        with patch("agent_zoo.sql_agent.runtime.InMemoryRunner", FakeRunner), patch(
            "agent_zoo.sql_agent.agent.build_root_agent",
            return_value=object(),
        ):
            first_response = asyncio.run(sql_runtime.ask_question("first", settings))
            second_response = asyncio.run(sql_runtime.ask_question("second", settings))

        self.assertEqual(first_response, "ok")
        self.assertEqual(second_response, "ok")
        self.assertEqual(len(FakeRunner.instances), 1)
        self.assertEqual(len(FakeRunner.instances[0].session_service.created_sessions), 1)

        sql_runtime._RUNNER_CACHE.clear()
        sql_runtime._INITIALIZED_SESSION_KEYS.clear()
        FakeRunner.instances.clear()


if __name__ == "__main__":
    unittest.main()
