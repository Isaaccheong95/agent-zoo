"""Small workflow models for the generic orchestrator agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable, Mapping

try:
    from ..sql_agent.result import SQLAgentStructuredResult
    from ..tabular import TabularPayload
except ImportError:  # Support ADK loading this package as top-level `orchestrator_agent`.
    from sql_agent.result import SQLAgentStructuredResult  # type: ignore[no-redef]
    from tabular import TabularPayload  # type: ignore[no-redef]


class HandoffPolicy(StrEnum):
    INTERNAL_PREFERRED = "internal_preferred"
    INTERNAL_REQUIRED = "internal_required"
    PUBLIC_ONLY = "public_only"


class OrchestrationError(RuntimeError):
    """Raised when an orchestration workflow cannot be executed safely."""


InputBuilder = Callable[["OrchestrationContext"], dict[str, Any]]
InstructionBuilder = Callable[["OrchestrationContext"], str | None]
FinalTextBuilder = Callable[["OrchestrationContext"], str]
Router = Callable[[str, Mapping[str, Any], Mapping[str, "WorkflowDefinition"]], str | None]


@dataclass(slots=True)
class WorkflowStep:
    agent_name: str
    method_name: str
    output_key: str
    input_builder: InputBuilder | None = None


@dataclass(slots=True)
class WorkflowDefinition:
    steps: list[WorkflowStep]
    final_output_key: str | None = None
    final_text_builder: FinalTextBuilder | None = None
    description: str | None = None


@dataclass(slots=True)
class StepExecutionResult:
    agent_name: str
    method_name: str
    output_key: str
    value: Any


@dataclass(slots=True)
class OrchestrationContext:
    question: str
    workflow_name: str
    artifacts: dict[str, Any] = field(default_factory=dict)
    step_results: list[StepExecutionResult] = field(default_factory=list)

    def get_artifact(self, key: str) -> Any:
        if key not in self.artifacts:
            raise OrchestrationError(f"Workflow artifact {key!r} is not available.")
        return self.artifacts[key]

    def set_artifact(self, key: str, value: Any) -> None:
        self.artifacts[key] = value

    def resolve_tabular_payload(
        self,
        source_key: str,
        *,
        handoff_policy: HandoffPolicy = HandoffPolicy.INTERNAL_PREFERRED,
    ) -> TabularPayload:
        artifact = self.get_artifact(source_key)

        if isinstance(artifact, TabularPayload):
            return artifact

        if isinstance(artifact, SQLAgentStructuredResult):
            if handoff_policy == HandoffPolicy.PUBLIC_ONLY:
                payload = artifact.public_data
            elif handoff_policy == HandoffPolicy.INTERNAL_REQUIRED:
                payload = artifact.internal_data
                if payload is None:
                    raise OrchestrationError(
                        f"Workflow requires internal tabular data from {source_key!r}, but no internal payload is available."
                    )
            else:
                payload = artifact.internal_data or artifact.public_data

            if payload is None:
                raise OrchestrationError(
                    f"Workflow artifact {source_key!r} does not contain a usable tabular payload."
                )
            return payload

        get_tabular_payload = getattr(artifact, "get_tabular_payload", None)
        if callable(get_tabular_payload):
            prefer_internal = handoff_policy != HandoffPolicy.PUBLIC_ONLY
            payload = get_tabular_payload(prefer_internal=prefer_internal)
            if payload is not None:
                return payload

        raise OrchestrationError(
            f"Workflow artifact {source_key!r} does not support tabular handoff."
        )


@dataclass(slots=True)
class OrchestrationResult:
    question: str
    workflow_name: str
    final_output: Any
    final_text: str
    step_results: list[StepExecutionResult]
    artifacts: dict[str, Any]
    response_type: str = "final"
    clarification_options: list[str] = field(default_factory=list)


def build_question_inputs(
    *,
    question_arg: str = "question",
) -> InputBuilder:
    def builder(context: OrchestrationContext) -> dict[str, Any]:
        return {question_arg: context.question}

    return builder


def build_tabular_analysis_inputs(
    source_key: str,
    *,
    payload_arg: str = "payload",
    include_question: bool = True,
    handoff_policy: HandoffPolicy = HandoffPolicy.INTERNAL_PREFERRED,
    instructions_arg: str = "instructions",
    instructions_builder: InstructionBuilder | None = None,
) -> InputBuilder:
    def builder(context: OrchestrationContext) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            payload_arg: context.resolve_tabular_payload(source_key, handoff_policy=handoff_policy),
        }
        if include_question:
            kwargs["question"] = context.question
        if instructions_builder is not None:
            instructions = instructions_builder(context)
            if instructions not in (None, ""):
                kwargs[instructions_arg] = str(instructions)
        return kwargs

    return builder