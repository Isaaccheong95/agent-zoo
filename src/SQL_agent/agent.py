from __future__ import annotations

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.genai import types

from .callbacks import format_final_agent_response, remember_query_result
from .config import SQLAgentSettings, load_settings
from .instructions import build_agent_instruction
from .tools import build_sql_tools


def build_root_agent(settings: SQLAgentSettings | None = None) -> LlmAgent:
    active_settings = settings or load_settings()
    return LlmAgent(
        model=LiteLlm(model=active_settings.model),
        name="sql_agent",
        description="Converts natural language questions into safe read-only SQLite queries and explains the results.",
        instruction=build_agent_instruction(active_settings),
        tools=build_sql_tools(active_settings),
        generate_content_config=types.GenerateContentConfig(temperature=0.0),
        after_tool_callback=remember_query_result,
        after_agent_callback=format_final_agent_response,
    )


root_agent = build_root_agent()
