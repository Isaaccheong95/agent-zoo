# agent-zoo

## Contents

- [What You'll Find](#what-youll-find)
- [Project Goals](#project-goals)
- [Structure](#structure)
- [Getting Started](#getting-started)
- [Why This Repo Exists](#why-this-repo-exists)

`agent-zoo` is a collection of reusable agents built with Google ADK.

The goal of this repository is simple: provide practical, plug-and-play agents that developers can run locally, study, adapt, and integrate into their own projects. Each agent is designed to solve a specific problem while keeping its logic transparent, modular, and easy to reuse.

## What You'll Find

This repository is intended to grow into a library of task-focused agents, such as:

- SQL data query agent
- Data analysis agent
- Figure and chart maker agent
- Other utility and workflow agent

## Project Goals

- Reusable: agents should be easy to plug into other repositories
- Transparent: prompts, tools, and execution flow should be easy to understand
- Modular: each agent should be self-contained and easy to extend
- Practical: focused on real developer, data, and automation workflows

## Structure

Each agent lives in its own folder and should include its own:

- agent definition
- tools
- runtime or entrypoint
- tests
- local documentation

## Getting Started

You can install `agent-zoo` into your own project and start using the SQL agent through the CLI or from Python code. For deeper implementation details, see each agent's local README.

### Install in your own project

If you are not using `uv`, create and activate your virtual environment first, then install `agent-zoo` from the Git tag, branch, or commit you want:

```bash
python -m venv .venv
source .venv/bin/activate
pip install "git+https://github.com/Isaaccheong95/agent-zoo.git@v0.1.0"
```

This plain `pip install` path can take a long time because `pip` has to resolve the full Google ADK dependency tree from scratch when installing from the Git repo. If you want a faster install experience, prefer `uv`.

If you are using `uv` and do not already have a virtual environment created for this project, create one and install `agent-zoo` like this:

```bash
uv venv
uv pip install --python .venv/bin/python "git+https://github.com/Isaaccheong95/agent-zoo.git@v0.1.0"
```

If you already have a `uv`-managed virtual environment for the project, skip `uv venv` and just run the `uv pip install ...` command.

### Run the SQL agent CLI

Once installed, the SQL agent is available as the `run-sql-agent` command. If you activated the environment, run it directly:

```bash
run-sql-agent --help
run-sql-agent --db /path/to/my_database.sqlite --question "How many rows are in this dataset?"
```

If you are using `uv` without activating the environment, use `uv run`:

```bash
uv run run-sql-agent --help
uv run run-sql-agent --db /path/to/my_database.sqlite --question "How many rows are in this dataset?"
```

If you want the same SQLite file to be the default for future runs, set `SQL_AGENT_DB_PATH` in your environment:

```bash
export SQL_AGENT_DB_PATH=/path/to/my_database.sqlite
uv run run-sql-agent --question "How many rows are in this dataset?"
```

### Use it in your own code

Import the package from `agent_zoo` in your own code:

```python
import asyncio
from pathlib import Path

from agent_zoo.sql_agent import SQLAgent, load_settings

settings = load_settings(
    {
        "db_path": Path("/path/to/my_database.sqlite"),
        "openai_api_base": "http://127.0.0.1:8080/v1",
    }
)
agent = SQLAgent(settings=settings)
response = asyncio.run(agent.ask("How many rows are in this dataset?"))
print(response)
```

### Use your own OpenAI-compatible LLM server

`agent-zoo` expects an OpenAI-compatible LLM endpoint. That can be a local `llama.cpp` server, a `vLLM` server, or another compatible backend that exposes a `/v1` API.

Point the package at that server with `OPENAI_API_BASE`. `SQL_AGENT_MODEL` controls the model name sent to the server:

```bash
export OPENAI_API_BASE="http://127.0.0.1:8080/v1"
export SQL_AGENT_MODEL="openai/Qwen3.5-0.8B-GGUF"
uv run run-sql-agent --db /path/to/my_database.sqlite --question "How many rows are in this dataset?"
```

Programmatic configuration uses the same setting:

```python
from pathlib import Path

from agent_zoo.sql_agent import load_settings

settings = load_settings(
    {
        "db_path": Path("/path/to/my_database.sqlite"),
        "openai_api_base": "http://127.0.0.1:8080/v1",
    }
)
```

## Why This Repo Exists

This repository is my personal collection of Google ADK agents: a place to experiment, refine patterns, and publish useful agents that others can learn from and use in their own work.
