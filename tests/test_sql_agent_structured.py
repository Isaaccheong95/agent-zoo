from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch


CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = CURRENT_FILE.parent if (CURRENT_FILE.parent / "src").exists() else CURRENT_FILE.parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agent_zoo.sql_agent import SQLAgent, SQLAgentSettings
from agent_zoo.sql_agent.result import SQLAgentStructuredResult, build_structured_result
from agent_zoo.tabular import TabularPayload


class SQLAgentStructuredResultTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.db_path = REPO_ROOT / "dataset" / "titantic" / "titanic.sqlite"

    def test_tabular_payload_from_sql_result_preserves_provenance(self) -> None:
        payload = TabularPayload.from_sql_result(
            {
                "status": "success",
                "sql": "SELECT city, COUNT(*) AS matching_count FROM trips GROUP BY city",
                "columns": ["city", "matching_count"],
                "rows": [{"city": "Tokyo", "matching_count": 10}],
                "row_count": 1,
                "preview_row_count": 1,
                "truncated": False,
                "error": None,
                "public_result_kind": "count_aggregate",
            },
            question="Which city has the highest count?",
            schema_text="trips(city TEXT)",
        )

        self.assertEqual(payload.question, "Which city has the highest count?")
        self.assertEqual(payload.sql, "SELECT city, COUNT(*) AS matching_count FROM trips GROUP BY city")
        self.assertEqual(payload.schema_text, "trips(city TEXT)")
        self.assertEqual(payload.metadata["public_result_kind"], "count_aggregate")

    def test_build_structured_result_exposes_public_and_internal_payloads(self) -> None:
        structured = build_structured_result(
            question="Show the matching people",
            final_response="Found 4 matching rows.",
            public_result={
                "status": "success",
                "sql": "SELECT COUNT(*) AS matching_count FROM people",
                "columns": ["matching_count"],
                "rows": [{"matching_count": 4}],
                "row_count": 1,
                "preview_row_count": 1,
                "truncated": False,
                "error": None,
                "public_result_kind": "detail_count_fallback",
            },
            internal_result={
                "status": "success",
                "sql": "SELECT name FROM people",
                "columns": ["name"],
                "rows": [{"name": "Alice"}, {"name": "Bob"}],
                "row_count": 4,
                "preview_row_count": 2,
                "truncated": True,
                "error": None,
            },
            schema_text="people(name TEXT)",
        )

        self.assertEqual(structured.public_result_kind, "detail_count_fallback")
        self.assertEqual(structured.get_tabular_payload().rows, [{"matching_count": 4}])
        self.assertEqual(
            structured.get_tabular_payload(prefer_internal=True).rows,
            [{"name": "Alice"}, {"name": "Bob"}],
        )

    def test_sqlagent_query_delegates_to_runtime(self) -> None:
        settings = SQLAgentSettings(
            db_path=self.db_path,
            model="openai/test-model",
        )
        agent = SQLAgent(settings=settings)
        structured_result = SQLAgentStructuredResult(
            question="How many rows are there?",
            final_response="Found 10 matching rows.",
        )
        mocked_ask_question_result = AsyncMock(return_value=structured_result)

        with patch("agent_zoo.sql_agent.runtime.ask_question_result", new=mocked_ask_question_result):
            response = asyncio.run(agent.query("How many rows are there?", session_id="test-session"))

        self.assertIs(response, structured_result)
        mocked_ask_question_result.assert_awaited_once_with(
            "How many rows are there?",
            settings,
            session_id="test-session",
        )


if __name__ == "__main__":
    unittest.main()