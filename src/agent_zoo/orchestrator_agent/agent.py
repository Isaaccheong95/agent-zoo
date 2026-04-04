"""Generic registry-based orchestrator for composing reusable agents."""

from __future__ import annotations

import inspect
from typing import Any, Mapping

from google.adk.agents import SequentialAgent
from google.genai import types

try:
    from ..base import BaseAgent
except ImportError:  # Support ADK loading this package as top-level `orchestrator_agent`.
    from base import BaseAgent  # type: ignore[no-redef]

from .models import (
    OrchestrationContext,
    OrchestrationError,
    OrchestrationResult,
    Router,
    StepExecutionResult,
    WorkflowDefinition,
)


ORCHESTRATOR_AGENT_NAME = "orchestrator_agent"
ORCHESTRATOR_AGENT_DESCRIPTION = (
    "Coordinates registered sub-agents through explicit workflows and passes structured artifacts between them."
)


def _default_root_agent_message(callback_context=None, **kwargs) -> types.Content:
    return types.Content(
        role="model",
        parts=[
            types.Part(
                text=(
                    "The generic orchestrator package loaded successfully, but it does not ship with a default "
                    "workflow registry for ADK web. Instantiate OrchestratorAgent programmatically with registered "
                    "agents and workflows, or add a custom ADK root_agent tailored to your application."
                )
            )
        ],
    )


def build_root_agent() -> SequentialAgent:
    return SequentialAgent(
        name=ORCHESTRATOR_AGENT_NAME,
        description=ORCHESTRATOR_AGENT_DESCRIPTION,
        before_agent_callback=_default_root_agent_message,
    )


def _coerce_final_text(value: Any) -> str:
    if isinstance(value, str):
        return value

    for attribute_name in ("final_text", "final_response"):
        attribute_value = getattr(value, attribute_name, None)
        if isinstance(attribute_value, str) and attribute_value:
            return attribute_value

    return str(value)


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


class OrchestratorAgent(BaseAgent):
    name = ORCHESTRATOR_AGENT_NAME
    description = ORCHESTRATOR_AGENT_DESCRIPTION

    def __init__(
        self,
        *,
        agents: Mapping[str, Any],
        workflows: Mapping[str, WorkflowDefinition],
        default_workflow: str | None = None,
        router: Router | None = None,
    ) -> None:
        self.agents = dict(agents)
        self.workflows = dict(workflows)
        self.default_workflow = default_workflow
        self.router = router

    def _select_workflow(self, question: str, workflow_name: str | None) -> str:
        if workflow_name is not None:
            if workflow_name not in self.workflows:
                raise OrchestrationError(f"Workflow {workflow_name!r} is not registered.")
            return workflow_name

        if self.router is not None:
            selected = self.router(question, self.agents, self.workflows)
            if selected is None:
                raise OrchestrationError("The router did not select a workflow.")
            if selected not in self.workflows:
                raise OrchestrationError(f"Workflow {selected!r} selected by the router is not registered.")
            return selected

        if self.default_workflow is None:
            raise OrchestrationError("No workflow was provided and no default workflow is configured.")
        if self.default_workflow not in self.workflows:
            raise OrchestrationError(f"Default workflow {self.default_workflow!r} is not registered.")
        return self.default_workflow

    async def orchestrate(
        self,
        question: str,
        *,
        workflow_name: str | None = None,
        initial_artifacts: Mapping[str, Any] | None = None,
    ) -> OrchestrationResult:
        selected_workflow = self._select_workflow(question, workflow_name)
        workflow = self.workflows[selected_workflow]
        context = OrchestrationContext(
            question=question,
            workflow_name=selected_workflow,
            artifacts=dict(initial_artifacts or {}),
        )

        for step in workflow.steps:
            agent = self.agents.get(step.agent_name)
            if agent is None:
                raise OrchestrationError(f"Workflow requires agent {step.agent_name!r}, but it is not registered.")

            method = getattr(agent, step.method_name, None)
            if method is None or not callable(method):
                raise OrchestrationError(
                    f"Registered agent {step.agent_name!r} does not provide callable method {step.method_name!r}."
                )

            kwargs = step.input_builder(context) if step.input_builder is not None else {"question": context.question}
            value = await _maybe_await(method(**kwargs))

            context.set_artifact(step.output_key, value)
            context.step_results.append(
                StepExecutionResult(
                    agent_name=step.agent_name,
                    method_name=step.method_name,
                    output_key=step.output_key,
                    value=value,
                )
            )

        if workflow.final_text_builder is not None:
            final_text = workflow.final_text_builder(context)
            final_output = context.get_artifact(workflow.final_output_key) if workflow.final_output_key else final_text
        elif workflow.final_output_key is not None:
            final_output = context.get_artifact(workflow.final_output_key)
            final_text = _coerce_final_text(final_output)
        elif context.step_results:
            final_output = context.step_results[-1].value
            final_text = _coerce_final_text(final_output)
        else:
            raise OrchestrationError(f"Workflow {selected_workflow!r} does not define any steps.")

        return OrchestrationResult(
            question=question,
            workflow_name=selected_workflow,
            final_output=final_output,
            final_text=final_text,
            step_results=list(context.step_results),
            artifacts=dict(context.artifacts),
        )

    async def ask(
        self,
        question: str,
        *,
        workflow_name: str | None = None,
        initial_artifacts: Mapping[str, Any] | None = None,
    ) -> str:
        result = await self.orchestrate(
            question,
            workflow_name=workflow_name,
            initial_artifacts=initial_artifacts,
        )
        return result.final_text


root_agent = build_root_agent()


__all__ = ["OrchestrationError", "OrchestratorAgent", "build_root_agent", "root_agent"]