from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path


CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = CURRENT_FILE.parent if (CURRENT_FILE.parent / "src").exists() else CURRENT_FILE.parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agent_zoo.data_analysis_agent import DataAnalysisAgent
from agent_zoo.orchestrator_agent import (
    HandoffPolicy,
    OrchestrationError,
    OrchestratorAgent,
    WorkflowDefinition,
    WorkflowStep,
    build_question_inputs,
    build_tabular_analysis_inputs,
)
from agent_zoo.sql_agent.result import SQLAgentStructuredResult, build_structured_result
from agent_zoo.tabular import TabularPayload


class FakeSQLAgent:
    def __init__(self, result: SQLAgentStructuredResult) -> None:
        self.result = result
        self.questions: list[str] = []

    async def query(self, question: str):
        self.questions.append(question)
        return self.result


class EchoAgent:
    async def ask(self, question: str) -> str:
        return f"echo:{question}"


class OrchestratorAgentTestCase(unittest.TestCase):
    def test_single_agent_workflow_returns_text(self) -> None:
        orchestrator = OrchestratorAgent(
            agents={"echo": EchoAgent()},
            workflows={
                "echo_workflow": WorkflowDefinition(
                    steps=[
                        WorkflowStep(
                            agent_name="echo",
                            method_name="ask",
                            output_key="reply",
                            input_builder=build_question_inputs(),
                        )
                    ],
                    final_output_key="reply",
                )
            },
            default_workflow="echo_workflow",
        )

        result = asyncio.run(orchestrator.ask("hello"))

        self.assertEqual(result, "echo:hello")

    def test_sql_to_analysis_workflow_prefers_internal_payload(self) -> None:
        sql_result = build_structured_result(
            question="Analyze patient rows",
            final_response="Found 3 matching rows.",
            public_result={
                "status": "success",
                "sql": "SELECT COUNT(*) AS matching_count FROM patients",
                "columns": ["matching_count"],
                "rows": [{"matching_count": 3}],
                "row_count": 1,
                "preview_row_count": 1,
                "truncated": False,
                "error": None,
                "public_result_kind": "detail_count_fallback",
            },
            internal_result={
                "status": "success",
                "sql": "SELECT name, age FROM patients",
                "columns": ["name", "age"],
                "rows": [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 44}],
                "row_count": 3,
                "preview_row_count": 2,
                "truncated": True,
                "error": None,
            },
            schema_text="patients(name TEXT, age INTEGER)",
        )
        sql_agent = FakeSQLAgent(sql_result)
        analysis_agent = DataAnalysisAgent()
        orchestrator = OrchestratorAgent(
            agents={"sql": sql_agent, "analysis": analysis_agent},
            workflows={
                "sql_then_analysis": WorkflowDefinition(
                    steps=[
                        WorkflowStep(
                            agent_name="sql",
                            method_name="query",
                            output_key="sql_result",
                            input_builder=build_question_inputs(),
                        ),
                        WorkflowStep(
                            agent_name="analysis",
                            method_name="analyze",
                            output_key="analysis_result",
                            input_builder=build_tabular_analysis_inputs(
                                "sql_result",
                                handoff_policy=HandoffPolicy.INTERNAL_PREFERRED,
                            ),
                        ),
                    ],
                    final_output_key="analysis_result",
                )
            },
            default_workflow="sql_then_analysis",
        )

        result = asyncio.run(orchestrator.orchestrate("Analyze patient rows"))

        self.assertEqual(sql_agent.questions, ["Analyze patient rows"])
        self.assertIn("preview of 2 rows out of 3", result.final_text)

    def test_internal_required_handoff_fails_when_internal_payload_missing(self) -> None:
        sql_result = build_structured_result(
            question="Analyze count",
            final_response="Found 1 matching row.",
            public_result={
                "status": "success",
                "sql": "SELECT COUNT(*) AS matching_count FROM patients",
                "columns": ["matching_count"],
                "rows": [{"matching_count": 3}],
                "row_count": 1,
                "preview_row_count": 1,
                "truncated": False,
                "error": None,
                "public_result_kind": "detail_count_fallback",
            },
            internal_result=None,
            schema_text="patients(name TEXT, age INTEGER)",
        )
        orchestrator = OrchestratorAgent(
            agents={"sql": FakeSQLAgent(sql_result), "analysis": DataAnalysisAgent()},
            workflows={
                "sql_then_analysis": WorkflowDefinition(
                    steps=[
                        WorkflowStep(
                            agent_name="sql",
                            method_name="query",
                            output_key="sql_result",
                            input_builder=build_question_inputs(),
                        ),
                        WorkflowStep(
                            agent_name="analysis",
                            method_name="analyze",
                            output_key="analysis_result",
                            input_builder=build_tabular_analysis_inputs(
                                "sql_result",
                                handoff_policy=HandoffPolicy.INTERNAL_REQUIRED,
                            ),
                        ),
                    ],
                    final_output_key="analysis_result",
                )
            },
            default_workflow="sql_then_analysis",
        )

        with self.assertRaises(OrchestrationError) as context:
            asyncio.run(orchestrator.orchestrate("Analyze count"))

        self.assertIn("requires internal tabular data", str(context.exception))

    def test_internal_preferred_falls_back_to_public_payload(self) -> None:
        sql_result = build_structured_result(
            question="Analyze count",
            final_response="Found 1 matching row.",
            public_result={
                "status": "success",
                "sql": "SELECT COUNT(*) AS matching_count FROM patients",
                "columns": ["matching_count"],
                "rows": [{"matching_count": 3}],
                "row_count": 1,
                "preview_row_count": 1,
                "truncated": False,
                "error": None,
                "public_result_kind": "detail_count_fallback",
            },
            internal_result=None,
            schema_text="patients(name TEXT, age INTEGER)",
        )
        orchestrator = OrchestratorAgent(
            agents={"sql": FakeSQLAgent(sql_result), "analysis": DataAnalysisAgent()},
            workflows={
                "sql_then_analysis": WorkflowDefinition(
                    steps=[
                        WorkflowStep(
                            agent_name="sql",
                            method_name="query",
                            output_key="sql_result",
                            input_builder=build_question_inputs(),
                        ),
                        WorkflowStep(
                            agent_name="analysis",
                            method_name="analyze",
                            output_key="analysis_result",
                            input_builder=build_tabular_analysis_inputs(
                                "sql_result",
                                handoff_policy=HandoffPolicy.INTERNAL_PREFERRED,
                            ),
                        ),
                    ],
                    final_output_key="analysis_result",
                )
            },
            default_workflow="sql_then_analysis",
        )

        result = asyncio.run(orchestrator.orchestrate("Analyze count"))

        self.assertIn("single matching_count value", result.final_text)

    def test_missing_registered_agent_fails_clearly(self) -> None:
        orchestrator = OrchestratorAgent(
            agents={},
            workflows={
                "missing": WorkflowDefinition(
                    steps=[WorkflowStep(agent_name="sql", method_name="query", output_key="sql_result")]
                )
            },
            default_workflow="missing",
        )

        with self.assertRaises(OrchestrationError) as context:
            asyncio.run(orchestrator.orchestrate("test"))

        self.assertIn("is not registered", str(context.exception))


if __name__ == "__main__":
    unittest.main()