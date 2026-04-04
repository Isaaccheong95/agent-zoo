from __future__ import annotations

import asyncio
import importlib
import sys
import unittest
from pathlib import Path


CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = CURRENT_FILE.parent if (CURRENT_FILE.parent / "src").exists() else CURRENT_FILE.parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agent_zoo.base import BaseAgent
from agent_zoo.data_analysis_agent import DataAnalysisAgent, analyze_tabular_payload
from agent_zoo.tabular import TabularPayload
from google.adk.agents import SequentialAgent


class DataAnalysisAgentTestCase(unittest.TestCase):
    def test_top_level_package_exposes_adk_root_agent(self) -> None:
        package_root = REPO_ROOT / "src" / "agent_zoo"
        original_sys_path = list(sys.path)
        sys.path.insert(0, str(package_root))
        try:
            if "data_analysis_agent" in sys.modules:
                del sys.modules["data_analysis_agent"]
            module = importlib.import_module("data_analysis_agent")
        finally:
            sys.path[:] = original_sys_path
            sys.modules.pop("data_analysis_agent", None)

        self.assertTrue(hasattr(module, "root_agent"))
        self.assertIsInstance(module.root_agent, SequentialAgent)

    def test_namespace_exports_instantiable_agent(self) -> None:
        agent = DataAnalysisAgent()

        self.assertIsInstance(agent, BaseAgent)
        self.assertEqual(agent.get_name(), "data_analysis_agent")
        self.assertEqual(
            agent.get_description(),
            "Interprets structured tabular results, surfaces findings and caveats, and suggests next analytical steps.",
        )

    def test_analyze_tabular_payload_highlights_ranked_groups(self) -> None:
        payload = TabularPayload.from_rows(
            [
                {"city": "Tokyo", "matching_count": 10},
                {"city": "Paris", "matching_count": 4},
                {"city": "Singapore", "matching_count": 7},
            ],
            question="Which city has the highest count?",
            sql="SELECT city, COUNT(*) AS matching_count FROM trips GROUP BY city",
        )

        result = analyze_tabular_payload(payload)

        self.assertIn("3 rows", result.summary)
        self.assertIn("Highest matching_count is 10 for city Tokyo.", result.findings)
        self.assertIn("Lowest matching_count is 4 for city Paris.", result.findings)
        self.assertTrue(result.next_steps)
        self.assertEqual(result.question, "Which city has the highest count?")

    def test_analyze_tabular_payload_calls_out_truncation(self) -> None:
        payload = TabularPayload.from_rows(
            [
                {"name": "Alice", "age": 30},
                {"name": "Bob", "age": 44},
            ],
            row_count=12,
            preview_row_count=2,
            truncated=True,
        )

        result = analyze_tabular_payload(payload)

        self.assertIn("preview of 2 rows out of 12", result.summary)
        self.assertIn(
            "Only a preview of the matching rows is available, so the full result set may change the pattern.",
            result.caveats,
        )

    def test_ask_returns_formatted_analysis_text(self) -> None:
        agent = DataAnalysisAgent()
        payload = TabularPayload.from_rows(
            [{"total_patients": 8}],
            question="How many patients are there?",
        )

        response = asyncio.run(agent.ask(data=payload))

        self.assertIn("Summary:", response)
        self.assertIn("Findings:", response)
        self.assertIn("Next steps:", response)


if __name__ == "__main__":
    unittest.main()