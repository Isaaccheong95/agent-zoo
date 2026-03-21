# agent-zoo

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

## Why This Repo Exists

This repository is my personal collection of Google ADK agents: a place to experiment, refine patterns, and publish useful agents that others can learn from and use in their own work.
