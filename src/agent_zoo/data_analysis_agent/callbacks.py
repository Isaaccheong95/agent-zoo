"""Define ADK callbacks used by the standalone data analysis agent."""

from __future__ import annotations

from google.adk.models import LlmResponse
from google.genai import types

try:
    from ..clarification import format_clarification_response, normalize_clarification_response
    from ..request_guard import build_llm_request_guard, format_request_guard_decision
    from ..sql_agent.db import get_schema_summary
except ImportError:  # Support ADK loading this package as top-level `data_analysis_agent`.
    from clarification import format_clarification_response, normalize_clarification_response  # type: ignore[no-redef]
    from request_guard import build_llm_request_guard, format_request_guard_decision  # type: ignore[no-redef]
    from sql_agent.db import get_schema_summary  # type: ignore[no-redef]

from .config import DataAnalysisAgentSettings, load_settings


DATASET_ANALYSIS_REFUSAL_MESSAGE = (
    "I'm a dataset analysis agent. I can answer questions about the current dataset, inspect schema, "
    "retrieve read-only results, and explain grounded patterns from that data. I can't answer unrelated "
    "general questions."
)
DATASET_ANALYSIS_CLARIFICATION_GUIDANCE = (
    "If the request could refer to more than one table, column, metric, grouping, or analysis angle, ask a short "
    "clarification question and suggest grounded options based on the schema."
)


def _request_ends_with_tool_response(llm_request) -> bool:
    contents = getattr(llm_request, "contents", None) or []
    if not contents:
        return False

    last_content = contents[-1]
    if getattr(last_content, "role", None) == "tool":
        return True

    last_parts = getattr(last_content, "parts", None) or []
    return any(getattr(part, "function_response", None) is not None for part in last_parts)


def _extract_last_user_text(llm_request) -> str:
    contents = getattr(llm_request, "contents", None) or []
    for content in reversed(contents):
        if getattr(content, "role", None) != "user":
            continue
        parts = getattr(content, "parts", None) or []
        return " ".join(getattr(part, "text", "") or "" for part in parts).strip()
    return ""


def _llm_response_has_function_call(llm_response: LlmResponse) -> bool:
    content = getattr(llm_response, "content", None)
    parts = getattr(content, "parts", None) or []
    return any(getattr(part, "function_call", None) is not None for part in parts)


def _extract_llm_response_text(llm_response: LlmResponse) -> str:
    content = getattr(llm_response, "content", None)
    parts = getattr(content, "parts", None) or []
    return "".join(getattr(part, "text", "") or "" for part in parts).strip()


def _build_dataset_domain_text(settings: DataAnalysisAgentSettings) -> str:
    schema_summary = get_schema_summary(settings.db_path)
    if schema_summary.get("status") == "success":
        schema_text = str(schema_summary.get("schema_text") or "")
    else:
        schema_text = f"Schema unavailable: {schema_summary.get('error', 'Unknown error')}"

    lines = [
        "This agent answers questions about the current dataset.",
        "It can inspect schema, run read-only SQLite queries, and analyze the resulting tables.",
        f"Default database path: {settings.db_path}",
        f"Default preview rows: {settings.preview_rows}",
        "Schema snapshot:",
        schema_text[:4000],
    ]
    return "\n".join(lines)


def build_request_guard_before_model_callback(
    settings: DataAnalysisAgentSettings | None = None,
):
    active_settings = settings or load_settings()
    guard = build_llm_request_guard(
        active_settings.model,
        agent_label="a dataset analysis agent",
        refusal_message=DATASET_ANALYSIS_REFUSAL_MESSAGE,
        clarification_guidance=DATASET_ANALYSIS_CLARIFICATION_GUIDANCE,
        openai_api_base=active_settings.openai_api_base,
    )
    domain_text = _build_dataset_domain_text(active_settings)

    def request_guard_before_model(callback_context=None, llm_request=None, **kwargs) -> LlmResponse | None:
        if _request_ends_with_tool_response(llm_request):
            return None

        user_text = _extract_last_user_text(llm_request) if llm_request is not None else ""
        decision = guard(user_text, domain_text)
        if decision.is_allow:
            return None

        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part(text=format_request_guard_decision(decision))],
            )
        )

    return request_guard_before_model


def build_normalize_clarification_after_model_callback(
    settings: DataAnalysisAgentSettings | None = None,
):
    def normalize_clarification_after_model(
        callback_context=None,
        llm_response: LlmResponse | None = None,
        **kwargs,
    ) -> LlmResponse | None:
        if llm_response is None or _llm_response_has_function_call(llm_response):
            return None

        response_text = _extract_llm_response_text(llm_response)
        if not response_text:
            return None

        clarification = normalize_clarification_response(response_text)
        if clarification is None:
            return None

        formatted_response = format_clarification_response(clarification)
        if not formatted_response:
            return None

        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part(text=formatted_response)],
            )
        )

    return normalize_clarification_after_model