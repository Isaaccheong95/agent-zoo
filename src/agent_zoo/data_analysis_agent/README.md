# Data Analysis Agent

This package now supports two complementary modes:

- a standalone ADK-web agent that can inspect the dataset directly and answer dataset questions
- a lightweight programmatic analysis surface for interpreting already-available tabular results

Its analysis core is still focused on looking at a table or summary result and producing:

- a short summary
- findings
- caveats
- suggested next analytical steps

The agent stays deliberately narrow so it remains reusable and easy to compose with other agents.

## What it does

The data analysis agent is designed for dataset-grounded descriptive analysis, such as:

- answering user questions about the current dataset through direct read-only dataset access
- interpreting grouped result tables
- calling out the highest and lowest values in a ranking
- noting truncation or small-sample caveats
- suggesting sensible next breakdowns or follow-up analyses

## What it does not do

The first version does not:

- modify the database
- generate SQL
- build charts
- perform advanced statistical inference

Write-capable database actions, charting, and richer statistical workflows should stay with other specialized agents.

## Public API

The package exports:

- `DataAnalysisAgent`
- `AnalysisResult`
- `DataAnalysisAgentSettings`
- `analyze_tabular_payload(...)`
- `analyze_query_result(...)`
- `load_settings(...)`
- `build_root_agent(...)`

There are two usage surfaces:

- `analyze(...)`: structured programmatic result
- `ask(...)`: user-facing convenience string

There is also a standalone ADK `root_agent` surface for direct dataset-aware chat usage.

This mirrors the intended split between machine-facing composition, human-facing display, and direct ADK-web use.

## Standalone ADK-web usage

When loaded in ADK web, `data_analysis_agent` now exposes a real `root_agent`.

That standalone agent can:

- inspect the dataset schema
- run read-only dataset queries
- analyze the resulting table in one tool call
- answer dataset-grounded user questions without requiring a pre-supplied `TabularPayload`

Its direct dataset-access tools are:

- `inspect_dataset_schema`
- `execute_dataset_read_only`
- `analyze_dataset_with_sql`

Those tools are intentionally fixed to the configured dataset and preview settings. The ADK-exposed tool surface does not ask the model to choose runtime overrides such as database paths or preview limits.

The standalone ADK path is intended for direct dataset question-answering. The programmatic `DataAnalysisAgent` class remains the reusable payload-analysis surface for downstream orchestration.

## Request guard behavior

The analysis agent now uses the shared request-guard layer by default.

That means it can:

- accept in-scope analysis requests
- refuse clearly out-of-scope requests
- ask for clarification when the user's intent is ambiguous

This guard runs before analysis starts. It uses the user question plus the available tabular context such as columns, SQL provenance, schema text, and metadata.

The guard fails open on LLM errors, which means analysis is still allowed to proceed if the judge call itself breaks.

## Disable the default request guard

If you do not want request-guard behavior, disable it explicitly:

```python
from agent_zoo.data_analysis_agent import DataAnalysisAgent

agent = DataAnalysisAgent(enable_request_guard=False)
```

## Inject a custom request guard

You can inject your own guard callable instead of using the default LLM-backed guard.

The callable signature is:

```python
(question: str, domain_text: str) -> RequestGuardDecision
```

Example:

```python
import asyncio

from agent_zoo.data_analysis_agent import DataAnalysisAgent
from agent_zoo.request_guard import allow_request, clarification_request
from agent_zoo.tabular import TabularPayload


def custom_guard(question: str, domain_text: str):
    if "pattern" in question.lower():
        return clarification_request(
            "Which pattern do you want me to focus on?",
            ["distribution", "ranking", "outliers"],
        )
    return allow_request()


payload = TabularPayload.from_rows([
    {"city": "Tokyo", "matching_count": 10},
    {"city": "Paris", "matching_count": 4},
])

agent = DataAnalysisAgent(request_guard=custom_guard)
result = asyncio.run(agent.analyze(payload, question="Explain the pattern"))

print(result.response_type)
print(result.final_text)
```

In normal app code, the most common custom-guard cases are:

- testing without real LLM calls
- stricter scope rules for a specific app
- custom refusal wording
- custom clarification choices

## Standalone programmatic payload usage

You can use the analysis agent directly with a `TabularPayload`.

```python
import asyncio

from agent_zoo.data_analysis_agent import DataAnalysisAgent
from agent_zoo.tabular import TabularPayload

payload = TabularPayload.from_rows(
    [
        {"city": "Tokyo", "matching_count": 10},
        {"city": "Paris", "matching_count": 4},
        {"city": "Singapore", "matching_count": 7},
    ],
    question="Which city has the highest count?",
)

agent = DataAnalysisAgent()
result = asyncio.run(agent.analyze(payload))

print(result.summary)
print(result.findings)
print(result.caveats)
print(result.next_steps)
```

If you just want a final string for display, use `ask(...)` instead:

```python
text = asyncio.run(agent.ask(data=payload))
print(text)
```

## Orchestrated SQL-fed usage

The preferred composed path is now:

1. the SQL agent queries the dataset and returns a structured artifact
2. the orchestrator resolves the payload from that artifact
3. the orchestrator passes the payload plus explicit instructions into `DataAnalysisAgent.analyze(...)`

Not every SQL-backed request needs the analysis step. Direct lookup or aggregate prompts such as "how many women are smokers?" can stop at a `sql_only` workflow and return the SQL result directly. Use `sql_then_analysis` only when the user is asking for interpretation, explanation, or analytical framing.

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

settings = load_settings(
    {
        "db_path": Path("/path/to/my.sqlite"),
        "openai_api_base": "http://127.0.0.1:8080/v1",
    }
)

sql_agent = SQLAgent(settings=settings)
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
                        instructions_builder=lambda context: (
                            f"Explain the strongest pattern for: {context.question}"
                        ),
                    ),
                ),
            ],
            final_output_key="analysis_result",
        )
    },
    default_workflow="sql_then_analysis",
)

result = asyncio.run(
    orchestrator.orchestrate("Show the count by city and explain the pattern")
)

print(result.final_text)
```

## Input contract

The analysis agent works on `TabularPayload`, which can carry:

- columns
- rows
- row count
- preview row count
- truncation flag
- original question
- SQL provenance
- optional metadata

This lets the same agent work in both standalone ADK-web mode and orchestrated payload-analysis mode.

## Output contract

`AnalysisResult` contains:

- `summary`
- `findings`
- `caveats`
- `next_steps`
- `final_text`
- `question`
- `metadata`
- `response_type`
- `clarification_options`
- `instructions_received`

For orchestration and app integration, the structured result is the preferred surface. For direct display, `final_text` is usually enough.

When the request guard short-circuits, `AnalysisResult` still comes back in the same structured shape, but:

- `response_type` will be `out_of_scope` or `clarification`
- `findings`, `caveats`, and `next_steps` will be empty
- `final_text` will contain the refusal or clarification text

## Current behavior notes

The standalone ADK root agent uses an LLM plus direct read-only dataset tools.

The programmatic payload-analysis surface remains intentionally heuristic and lightweight. That keeps it deterministic, concise, and easy to test.

The payload-analysis path is best suited for:

- ranked tables
- simple grouped summaries
- scalar results
- preview tables where caveat handling matters

If you later want richer downstream narrative analysis, you can add a model-backed layer on top of the same `TabularPayload` input contract without changing how the orchestrator hands data into this package.