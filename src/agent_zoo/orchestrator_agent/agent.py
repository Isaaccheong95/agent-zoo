"""Generic registry-based orchestrator for composing reusable agents."""

from __future__ import annotations

import inspect
from typing import AsyncGenerator
from typing import Any, Mapping

from google.adk.agents import BaseAgent as AdkBaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.events import Event
from google.adk.agents.invocation_context import InvocationContext
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
except ImportError:  # Support ADK loading this package as top-level `orchestrator_agent`.
    from base import BaseAgent  # type: ignore[no-redef]
    from request_guard import (  # type: ignore[no-redef]
        DEFAULT_REQUEST_GUARD_MODEL,
        RequestGuard,
        RequestGuardDecision,
        build_llm_request_guard,
        format_request_guard_decision,
    )

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
ORCHESTRATOR_REFUSAL_MESSAGE = (
    "I'm an orchestrator agent. I can coordinate the registered data agents and supported workflows, "
    "but I can't answer unrelated general questions."
)
ORCHESTRATOR_CLARIFICATION_GUIDANCE = (
    "If the request could map to more than one workflow or the user intent is underspecified, ask a short "
    "clarification question and offer grounded options based on the registered agents or workflows."
)


def _default_root_agent_message(callback_context: CallbackContext) -> types.Content:
    _ = callback_context

    return types.Content(
        role="model",
        parts=[
            types.Part(
                text=(
                    "The generic orchestrator package loaded successfully, but it does not provide a default "
                    "workflow registry for ADK web. Use OrchestratorAgent programmatically with explicit agents "
                    "and workflows, or define an application-specific root_agent for ADK execution."
                )
            )
        ],
    )


class _OrchestratorPlaceholderAgent(AdkBaseAgent):
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        if False:
            yield Event(invocation_id=ctx.invocation_id, author=self.name)

    async def _run_live_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        if False:
            yield Event(invocation_id=ctx.invocation_id, author=self.name)


def build_root_agent() -> AdkBaseAgent:
    return _OrchestratorPlaceholderAgent(
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


def _describe_agent(agent_name: str, agent: Any) -> str:
    description = ""
    get_description = getattr(agent, "get_description", None)
    if callable(get_description):
        try:
            description = str(get_description() or "")
        except Exception:
            description = ""
    if not description:
        description = str(getattr(agent, "description", "") or "")
    if not description:
        description = f"Agent type: {agent.__class__.__name__}"
    return f"- {agent_name}: {description}"


def _describe_workflow(workflow_name: str, workflow: WorkflowDefinition) -> str:
    if workflow.description:
        return f"- {workflow_name}: {workflow.description}"
    if workflow.steps:
        steps_text = " -> ".join(f"{step.agent_name}.{step.method_name}" for step in workflow.steps)
        return f"- {workflow_name}: {steps_text}"
    return f"- {workflow_name}: no steps configured"


def _build_orchestrator_domain_text(
    agents: Mapping[str, Any],
    workflows: Mapping[str, WorkflowDefinition],
) -> str:
    agent_lines = [
        _describe_agent(agent_name, agent)
        for agent_name, agent in agents.items()
    ] or ["- No registered agents"]
    workflow_lines = [
        _describe_workflow(workflow_name, workflow)
        for workflow_name, workflow in workflows.items()
    ] or ["- No registered workflows"]
    return "\n".join(
        [
            "This agent only coordinates registered agents through explicit workflows.",
            "Registered agents:",
            *agent_lines,
            "Registered workflows:",
            *workflow_lines,
        ]
    )


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
        request_guard: RequestGuard | None = None,
        request_guard_model: str | None = None,
        openai_api_base: str | None = None,
        enable_request_guard: bool = True,
    ) -> None:
        self.agents = dict(agents)
        self.workflows = dict(workflows)
        self.default_workflow = default_workflow
        self.router = router
        if request_guard is not None:
            self.request_guard = request_guard
        elif enable_request_guard:
            self.request_guard = build_llm_request_guard(
                request_guard_model or DEFAULT_REQUEST_GUARD_MODEL,
                agent_label="an orchestrator agent",
                refusal_message=ORCHESTRATOR_REFUSAL_MESSAGE,
                clarification_guidance=ORCHESTRATOR_CLARIFICATION_GUIDANCE,
                openai_api_base=openai_api_base,
            )
        else:
            self.request_guard = None

    def _guard_request(self, question: str) -> RequestGuardDecision | None:
        if self.request_guard is None or not question.strip():
            return None

        decision = self.request_guard(
            question,
            _build_orchestrator_domain_text(self.agents, self.workflows),
        )
        if decision.is_allow:
            return None
        return decision

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
        artifacts = dict(initial_artifacts or {})
        guard_decision = self._guard_request(question)
        if guard_decision is not None:
            return OrchestrationResult(
                question=question,
                workflow_name=workflow_name or "preflight_guard",
                final_output=guard_decision,
                final_text=format_request_guard_decision(guard_decision),
                step_results=[],
                artifacts=artifacts,
                response_type=guard_decision.response_type,
                clarification_options=list(guard_decision.options),
            )

        selected_workflow = self._select_workflow(question, workflow_name)
        workflow = self.workflows[selected_workflow]
        context = OrchestrationContext(
            question=question,
            workflow_name=selected_workflow,
            artifacts=artifacts,
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