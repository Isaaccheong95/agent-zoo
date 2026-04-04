# Data Analysis Agent

This package contains a lightweight agent for interpreting already-available tabular results.

Its job is not to query data. Its job is to look at a table or summary result and produce:

- a short summary
- findings
- caveats
- suggested next analytical steps

The agent is deliberately narrow so it stays reusable and easy to compose with other agents.

## What it does

The data analysis agent is designed for descriptive downstream analysis, such as:

- interpreting grouped result tables
- calling out the highest and lowest values in a ranking
- noting truncation or small-sample caveats
- suggesting sensible next breakdowns or follow-up analyses

## What it does not do

The first version does not:

- query the database directly
- generate SQL
- build charts
- perform advanced statistical inference

Those responsibilities should stay with the SQL agent, a future figure builder, or other specialized agents.

## Public API

The package exports:

- `DataAnalysisAgent`
- `AnalysisResult`
- `analyze_tabular_payload(...)`

There are two usage surfaces:

- `analyze(...)`: structured programmatic result
- `ask(...)`: user-facing convenience string

This mirrors the intended split between machine-facing composition and human-facing display.

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

## Standalone usage

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

## SQL-fed usage

The intended SQL-fed path is:

1. call `SQLAgent.query(...)`
2. resolve the tabular payload from the structured SQL result
3. pass that payload into `DataAnalysisAgent.analyze(...)`

```python
import asyncio
from pathlib import Path

from agent_zoo.data_analysis_agent import DataAnalysisAgent
from agent_zoo.sql_agent import SQLAgent, load_settings

settings = load_settings(
    {
        "db_path": Path("/path/to/my.sqlite"),
        "openai_api_base": "http://127.0.0.1:8080/v1",
    }
)

sql_agent = SQLAgent(settings=settings)
analysis_agent = DataAnalysisAgent()

sql_result = asyncio.run(
    sql_agent.query("Show the count by city and give me the raw result")
)

payload = sql_result.get_tabular_payload(prefer_internal=True)
analysis = asyncio.run(
    analysis_agent.analyze(payload, question="Explain the pattern in the result")
)

print(analysis.final_text)
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

This lets the same agent work in both standalone mode and SQL-fed mode.

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

For orchestration and app integration, the structured result is the preferred surface. For direct display, `final_text` is usually enough.

When the request guard short-circuits, `AnalysisResult` still comes back in the same structured shape, but:

- `response_type` will be `out_of_scope` or `clarification`
- `findings`, `caveats`, and `next_steps` will be empty
- `final_text` will contain the refusal or clarification text

## Current behavior notes

The current implementation is intentionally heuristic and lightweight. It does not use an LLM. That keeps it deterministic, concise, and easy to test.

That also means it is best suited for:

- ranked tables
- simple grouped summaries
- scalar results
- preview tables where caveat handling matters

If you later want richer narrative analysis, you can add a model-backed layer on top of the same `TabularPayload` input contract without changing how other agents hand data into this package.