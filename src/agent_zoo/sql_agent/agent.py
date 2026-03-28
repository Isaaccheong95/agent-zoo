"""Build the top-level ADK SQL agent for this project.

This module assembles the model, instructions, tools, and callbacks into a
single `LlmAgent` instance. It is typically imported by the runtime helpers or
`run_sql_agent.py` rather than executed directly.

To run the agent from the command line, use `uv run run_sql_agent.py`.
"""

from __future__ import annotations

from typing import Any

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.genai import types

from ..base import BaseAgent
from .callbacks import (
    build_finalize_after_query_before_model_callback,
    build_format_final_agent_response_callback,
    build_remember_query_result_callback,
)
from .config import SQLAgentSettings, load_settings
from .instructions import build_agent_instruction
from .tools import build_sql_tools

SQL_AGENT_NAME = "sql_agent"
SQL_AGENT_DESCRIPTION = (
    "Converts natural language questions into safe read-only SQLite queries "
    "and explains the results."
)


class SQLAgent(BaseAgent):
    name = SQL_AGENT_NAME
    description = SQL_AGENT_DESCRIPTION

    def __init__(self, settings: SQLAgentSettings | None = None) -> None:
        self.settings = settings or load_settings()

    async def ask(self, question: str, **kwargs: Any) -> str:
        from .runtime import ask_question

        return await ask_question(question, self.settings, **kwargs)


def build_root_agent(settings: SQLAgentSettings | None = None) -> LlmAgent:
    active_settings = settings or load_settings()
    return LlmAgent(
        model=LiteLlm(model=active_settings.model),
        name=SQL_AGENT_NAME,
        description=SQL_AGENT_DESCRIPTION,
        instruction=build_agent_instruction(active_settings),
        tools=build_sql_tools(active_settings),
        generate_content_config=types.GenerateContentConfig(temperature=0.0),
        before_model_callback=build_finalize_after_query_before_model_callback(active_settings),
        after_tool_callback=build_remember_query_result_callback(active_settings),
        after_agent_callback=build_format_final_agent_response_callback(active_settings),
    )


root_agent = build_root_agent()
