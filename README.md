# agent-zoo

## Contents

- [What You'll Find](#what-youll-find)
- [Project Goals](#project-goals)
- [Structure](#structure)
- [Getting Started](#getting-started)
- [Set Up With uv](#set-up-with-uv)
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

Browse the available agent folders and follow the instructions in each agent's local README to run or integrate it.

### Set Up With uv

This repository uses `uv` for dependency and environment management.

#### Install uv

If you do not already have `uv` installed, you can use one of these common methods:

Windows (PowerShell):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

macOS and Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

With `pipx`:

```bash
pipx install uv
```

With `pip`:

```bash
pip install uv
```

Verify the installation:

```bash
uv --version
```

#### Create the environment

From the repository root, create and sync the virtual environment:

```powershell
uv sync
```

#### Run commands with uv

Run agent commands directly through `uv`:

```powershell
uv run run-sql-agent --help
```

Programmatic imports use the `agent_zoo` namespace:

```python
import asyncio

from agent_zoo.sql_agent import SQLAgent

agent = SQLAgent()
response = asyncio.run(agent.ask("How many passengers survived?"))
```

If you prefer activating the environment manually, `uv sync` will create a local `.venv` for you.

## Why This Repo Exists

This repository is my personal collection of Google ADK agents: a place to experiment, refine patterns, and publish useful agents that others can learn from and use in their own work.
