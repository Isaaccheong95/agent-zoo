"""Minimal reusable agent for interpreting tabular query results."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import fmean
from typing import Any

from google.adk.agents import SequentialAgent
from google.genai import types

try:
    from ..base import BaseAgent
    from ..request_guard import (
        DEFAULT_REQUEST_GUARD_MODEL,
        RequestGuard,
        RequestGuardDecision,
        build_llm_request_guard,
        format_request_guard_decision,
    )
    from ..tabular import TabularPayload
except ImportError:  # Support ADK loading this package as top-level `data_analysis_agent`.
    from base import BaseAgent  # type: ignore[no-redef]
    from request_guard import (  # type: ignore[no-redef]
        DEFAULT_REQUEST_GUARD_MODEL,
        RequestGuard,
        RequestGuardDecision,
        build_llm_request_guard,
        format_request_guard_decision,
    )
    from tabular import TabularPayload  # type: ignore[no-redef]


DATA_ANALYSIS_AGENT_NAME = "data_analysis_agent"
DATA_ANALYSIS_AGENT_DESCRIPTION = (
    "Interprets structured tabular results, surfaces findings and caveats, "
    "and suggests next analytical steps."
)
DATA_ANALYSIS_REFUSAL_MESSAGE = (
    "I'm a data analysis agent. I can interpret the provided dataset result, summarize patterns, "
    "note caveats, and suggest next analytical steps. I can't answer unrelated general questions."
)
DATA_ANALYSIS_CLARIFICATION_GUIDANCE = (
    "If the request could refer to more than one column, grouping, metric, or analysis angle, ask a short "
    "clarification question and suggest grounded options based on the available columns or result context."
)


def _default_root_agent_message(callback_context=None, **kwargs) -> types.Content:
    return types.Content(
        role="model",
        parts=[
            types.Part(
                text=(
                    "The data_analysis_agent package loaded successfully, but it is intended to be used "
                    "programmatically with structured TabularPayload inputs. Instantiate DataAnalysisAgent in "
                    "Python code or add a custom ADK root_agent if you want a direct ADK web workflow."
                )
            )
        ],
    )


def build_root_agent() -> SequentialAgent:
    return SequentialAgent(
        name=DATA_ANALYSIS_AGENT_NAME,
        description=DATA_ANALYSIS_AGENT_DESCRIPTION,
        before_agent_callback=_default_root_agent_message,
    )


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return str(value)


def _numeric_columns(payload: TabularPayload) -> list[str]:
    numeric_columns: list[str] = []
    for column in payload.columns:
        values = [row.get(column) for row in payload.rows if row.get(column) is not None]
        if values and all(_is_numeric(value) for value in values):
            numeric_columns.append(column)
    return numeric_columns


def _dimension_columns(payload: TabularPayload, excluded: set[str] | None = None) -> list[str]:
    excluded = excluded or set()
    dimensions: list[str] = []
    for column in payload.columns:
        if column in excluded:
            continue
        values = [row.get(column) for row in payload.rows if row.get(column) not in (None, "")]
        if values and not all(_is_numeric(value) for value in values):
            dimensions.append(column)
    return dimensions


def _measure_column(payload: TabularPayload, numeric_columns: list[str]) -> str | None:
    for column in numeric_columns:
        lowered = column.lower()
        if any(token in lowered for token in ("count", "avg", "average", "sum", "total", "rate")):
            return column
    return numeric_columns[0] if numeric_columns else None


def _summary(payload: TabularPayload) -> str:
    if payload.row_count == 0:
        return "No rows are available to analyze."
    if len(payload.rows) == 1 and len(payload.columns) == 1:
        return f"The result returns a single {payload.columns[0]} value."
    if payload.truncated:
        return (
            f"This result is a preview of {payload.preview_row_count} rows out of "
            f"{payload.row_count} matching rows."
        )
    return f"This result contains {payload.row_count} rows across {len(payload.columns)} columns."


def _findings(payload: TabularPayload) -> list[str]:
    if payload.row_count == 0 or not payload.rows:
        return []

    if len(payload.rows) == 1 and len(payload.columns) == 1:
        column = payload.columns[0]
        return [f"{column} is {_format_value(payload.rows[0].get(column))}."]

    numeric_columns = _numeric_columns(payload)
    measure_column = _measure_column(payload, numeric_columns)
    dimension_columns = _dimension_columns(payload, excluded={measure_column} if measure_column else set())
    findings: list[str] = []

    if measure_column and dimension_columns and len(payload.rows) > 1:
        dimension_column = dimension_columns[0]
        sortable_rows = [row for row in payload.rows if _is_numeric(row.get(measure_column))]
        if sortable_rows:
            sorted_rows = sorted(sortable_rows, key=lambda row: row.get(measure_column), reverse=True)
            top_row = sorted_rows[0]
            findings.append(
                f"Highest {measure_column} is {_format_value(top_row.get(measure_column))} "
                f"for {dimension_column} {_format_value(top_row.get(dimension_column))}."
            )
            if len(sorted_rows) > 1:
                bottom_row = sorted_rows[-1]
                findings.append(
                    f"Lowest {measure_column} is {_format_value(bottom_row.get(measure_column))} "
                    f"for {dimension_column} {_format_value(bottom_row.get(dimension_column))}."
                )
            return findings

    if measure_column:
        values = [row.get(measure_column) for row in payload.rows if _is_numeric(row.get(measure_column))]
        if values:
            findings.append(
                f"{measure_column} ranges from {_format_value(min(values))} to {_format_value(max(values))} "
                f"with an average of {_format_value(fmean(values))} across the available rows."
            )

    if not findings and payload.columns:
        findings.append(f"The available columns are {', '.join(payload.columns)}.")
    return findings


def _caveats(payload: TabularPayload) -> list[str]:
    caveats: list[str] = []
    if payload.truncated:
        caveats.append(
            "Only a preview of the matching rows is available, so the full result set may change the pattern."
        )
    if 0 < payload.row_count <= 2:
        caveats.append("There are very few rows in this result, so broad conclusions are limited.")
    if payload.row_count > 0 and not _numeric_columns(payload):
        caveats.append("There are no clearly numeric measure columns, so this analysis is mostly descriptive.")
    return caveats


def _next_steps(payload: TabularPayload) -> list[str]:
    if payload.row_count == 0:
        return ["Run a broader query or relax the filters so there is data to interpret."]

    if len(payload.rows) == 1 and len(payload.columns) == 1:
        return ["Break this result down by a relevant category or time bucket to add context."]

    numeric_columns = _numeric_columns(payload)
    measure_column = _measure_column(payload, numeric_columns)
    dimension_columns = _dimension_columns(payload, excluded={measure_column} if measure_column else set())

    next_steps: list[str] = []
    if dimension_columns and measure_column:
        next_steps.append(
            f"Break the result down further around {dimension_columns[0]} to see whether the {measure_column} ranking holds across subgroups."
        )
    elif payload.truncated:
        next_steps.append("Re-run the query with narrower filters or stronger aggregation to avoid preview-only analysis.")
    else:
        next_steps.append("Group the result by a meaningful category or time field to make comparisons easier.")
    return next_steps


def _build_analysis_domain_text(payload: TabularPayload) -> str:
    lines = [
        "This agent only analyzes the already-provided tabular result. It can summarize patterns, note caveats, and suggest next analytical steps.",
        f"Columns: {', '.join(payload.columns) if payload.columns else 'No columns provided'}",
        f"Row count: {payload.row_count}",
        f"Preview row count: {payload.preview_row_count}",
        f"Truncated: {payload.truncated}",
    ]
    if payload.sql:
        lines.append(f"SQL provenance: {payload.sql}")
    if payload.schema_text:
        lines.append(f"Schema context: {payload.schema_text[:2000]}")
    if payload.metadata:
        lines.append(f"Metadata keys: {', '.join(sorted(str(key) for key in payload.metadata.keys()))}")
    return "\n".join(lines)


def _format_analysis_result(result: AnalysisResult) -> str:
    sections = [f"Summary: {result.summary}"]

    if result.findings:
        sections.append("Findings:\n" + "\n".join(f"- {finding}" for finding in result.findings))
    if result.caveats:
        sections.append("Caveats:\n" + "\n".join(f"- {caveat}" for caveat in result.caveats))
    if result.next_steps:
        sections.append("Next steps:\n" + "\n".join(f"- {step}" for step in result.next_steps))

    return "\n\n".join(sections)


@dataclass(slots=True)
class AnalysisResult:
    summary: str
    findings: list[str]
    caveats: list[str]
    next_steps: list[str]
    final_text: str
    question: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    response_type: str = "analysis"
    clarification_options: list[str] = field(default_factory=list)


def analyze_tabular_payload(
    payload: TabularPayload,
    *,
    question: str | None = None,
) -> AnalysisResult:
    effective_question = question or payload.question
    summary = _summary(payload)
    findings = _findings(payload)
    caveats = _caveats(payload)
    next_steps = _next_steps(payload)
    result = AnalysisResult(
        summary=summary,
        findings=findings,
        caveats=caveats,
        next_steps=next_steps,
        final_text="",
        question=effective_question,
        metadata={
            "row_count": payload.row_count,
            "preview_row_count": payload.preview_row_count,
            "truncated": payload.truncated,
            "columns": list(payload.columns),
        },
    )
    result.final_text = _format_analysis_result(result)
    return result


def _build_guarded_analysis_result(
    decision: RequestGuardDecision,
    *,
    question: str | None,
    payload: TabularPayload,
) -> AnalysisResult:
    return AnalysisResult(
        summary=decision.user_message or "",
        findings=[],
        caveats=[],
        next_steps=[],
        final_text=format_request_guard_decision(decision),
        question=question,
        metadata={
            "row_count": payload.row_count,
            "preview_row_count": payload.preview_row_count,
            "truncated": payload.truncated,
            "columns": list(payload.columns),
        },
        response_type=decision.response_type,
        clarification_options=list(decision.options),
    )


class DataAnalysisAgent(BaseAgent):
    name = DATA_ANALYSIS_AGENT_NAME
    description = DATA_ANALYSIS_AGENT_DESCRIPTION

    def __init__(
        self,
        *,
        request_guard: RequestGuard | None = None,
        request_guard_model: str | None = None,
        openai_api_base: str | None = None,
        enable_request_guard: bool = True,
    ) -> None:
        if request_guard is not None:
            self.request_guard = request_guard
        elif enable_request_guard:
            self.request_guard = build_llm_request_guard(
                request_guard_model or DEFAULT_REQUEST_GUARD_MODEL,
                agent_label="a data analysis agent",
                refusal_message=DATA_ANALYSIS_REFUSAL_MESSAGE,
                clarification_guidance=DATA_ANALYSIS_CLARIFICATION_GUIDANCE,
                openai_api_base=openai_api_base,
            )
        else:
            self.request_guard = None

    def _guard_request(
        self,
        question: str | None,
        payload: TabularPayload,
    ) -> RequestGuardDecision | None:
        if self.request_guard is None:
            return None

        effective_question = (question or "").strip()
        if not effective_question:
            return None

        decision = self.request_guard(
            effective_question,
            _build_analysis_domain_text(payload),
        )
        if decision.is_allow:
            return None
        return decision

    async def analyze(
        self,
        payload: TabularPayload,
        *,
        question: str | None = None,
    ) -> AnalysisResult:
        effective_question = question or payload.question
        decision = self._guard_request(effective_question, payload)
        if decision is not None:
            return _build_guarded_analysis_result(
                decision,
                question=effective_question,
                payload=payload,
            )

        return analyze_tabular_payload(payload, question=effective_question)

    async def ask(
        self,
        question: str | None = None,
        *,
        data: TabularPayload,
    ) -> str:
        result = await self.analyze(data, question=question)
        return result.final_text


root_agent = build_root_agent()