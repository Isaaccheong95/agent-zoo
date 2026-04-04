# Generic Orchestrator Agent

This package contains a thin, registry-based orchestrator for composing other agents through explicit workflows.

The orchestrator is intentionally small. It does not plan arbitrarily, pick tools directly, or hard-code specific child agents. Instead, the caller registers the agents they want to use and defines the workflows they want to allow.

## What it is for

Use the orchestrator when:

- you want one entry point over multiple agents
- a request spans more than one agent
- one agent's structured output needs to be passed into another
- your app wants workflow routing without embedding all coordination logic in the UI layer

Do not use it when one direct agent call is enough.

## Public API

The main exports are:

- `OrchestratorAgent`
- `WorkflowDefinition`
- `WorkflowStep`
- `HandoffPolicy`
- `build_question_inputs(...)`
- `build_tabular_analysis_inputs(...)`

`OrchestratorAgent` exposes two surfaces:

- `ask(...)` returns final user-facing text
- `orchestrate(...)` returns a structured `OrchestrationResult`

## Request guard behavior

The orchestrator now uses the shared request-guard layer by default.

That means it can:

- refuse clearly out-of-scope requests before any workflow runs
- ask the user to clarify ambiguous intent before workflow selection
- allow normal execution when the request is in scope

The request guard runs before workflow selection and before any child agent is invoked. If it returns refusal or clarification, the orchestrator executes zero workflow steps.

## Disable the default request guard

If you want routing only and do not want guard behavior, disable it explicitly:

```python
from agent_zoo.orchestrator_agent import OrchestratorAgent

orchestrator = OrchestratorAgent(
    agents={...},
    workflows={...},
    default_workflow="sql_then_analysis",
    enable_request_guard=False,
)
```

## Inject a custom request guard

You can inject a custom guard callable if you want deterministic tests or app-specific routing constraints.

```python
from agent_zoo.orchestrator_agent import OrchestratorAgent
from agent_zoo.request_guard import allow_request, clarification_request


def custom_guard(question: str, domain_text: str):
    lowered = question.lower()
    if "analyze" in lowered and "chart" in lowered:
        return clarification_request(
            "Do you want analysis only or analysis plus a chart workflow?",
            ["analysis only", "analysis plus chart"],
        )
    return allow_request()


orchestrator = OrchestratorAgent(
    agents={...},
    workflows={...},
    default_workflow="sql_then_analysis",
    request_guard=custom_guard,
)
```

Common reasons to inject a custom guard:

- deterministic unit tests
- stricter app-specific scope boundaries
- custom refusal messages
- custom clarification rules before workflow selection

## How registration works

You pass in:

- an `agents` mapping of names to agent instances
- a `workflows` mapping of names to workflow definitions
- optionally a `default_workflow`
- optionally a `router` callable if you want light workflow selection logic

The orchestrator does not care whether the registered agent is a SQL agent, analysis agent, figure agent, or something else. It only cares that the named agent exposes the method the workflow step asks for.

The request guard follows the same principle: it uses the registered agents and workflows to build its domain summary, but the orchestrator itself stays generic.

## Example: register a SQL agent and a data analysis agent

```python
import asyncio
from pathlib import Path

from agent_zoo.data_analysis_agent import DataAnalysisAgent
from agent_zoo.orchestrator_agent import (
    HandoffPolicy,
    OrchestratorAgent,
    WorkflowDefinition,
    WorkflowStep,
    build_question_inputs,
    build_tabular_analysis_inputs,
)
from agent_zoo.sql_agent import SQLAgent, load_settings

sql_settings = load_settings(
    {
        "db_path": Path("/path/to/my.sqlite"),
        "openai_api_base": "http://127.0.0.1:8080/v1",
    }
)

sql_agent = SQLAgent(settings=sql_settings)
analysis_agent = DataAnalysisAgent()

orchestrator = OrchestratorAgent(
    agents={
        "sql": sql_agent,
        "analysis": analysis_agent,
    },
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

result = asyncio.run(
    orchestrator.orchestrate("Show the age distribution and explain the pattern")
)

print(result.final_text)
print(result.workflow_name)
print(result.artifacts.keys())
```

`OrchestrationResult` also carries guarded outcomes. If the request is refused or needs clarification before execution:

- `response_type` is set to `out_of_scope` or `clarification`
- `clarification_options` contains any suggested choices
- `step_results` is empty because no workflow steps ran

## Handoff policy

The orchestrator supports three handoff modes when a downstream step needs tabular data:

- `INTERNAL_PREFERRED`: use internal structured data if available, otherwise fall back to public data
- `INTERNAL_REQUIRED`: fail if internal structured data is unavailable
- `PUBLIC_ONLY`: always use the public structured payload

For SQL-to-analysis workflows, the normal choice is `INTERNAL_PREFERRED` or `INTERNAL_REQUIRED` so the analysis agent can work on the fuller queried dataset instead of only a privacy-collapsed public summary.

## Important SQL-agent note

`SQLAgent.query(...)` uses the structured SQL path. When it builds its own runner, it enables internal row capture by default so downstream workflows can receive internal structured results.

If you pass in your own ADK runner, internal data availability still depends on how that runner was configured.

## Workflow design guidance

Keep workflows explicit and small.

Good first workflows:

- one-agent passthrough
- SQL query then analysis
- analysis then figure

Avoid turning the orchestrator into a planner or adding app-specific UI behavior to it. Let the app choose which workflows exist and when to invoke them.