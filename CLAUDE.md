# CLAUDE.md

Operating manual for Claude coding sessions in the `agent-zoo` repo.
Not a user-facing README — the top-level [README.md](README.md) and [src/agent_zoo/sql_agent/README.md](src/agent_zoo/sql_agent/README.md) already cover that.

## Repo Overview

- Python package `agent-zoo` (`>=3.11`, setuptools, installable via `pip` or `uv`).
- Intended to be a collection of reusable Google ADK agents.
- Only the SQL agent under [src/agent_zoo/sql_agent/](src/agent_zoo/sql_agent/) is an actual working agent. The top-level package goal (a "zoo") is aspirational — most of the code lives there.
- Build/runtime layer is Google ADK (`google-adk[extensions]`); LLM calls go through `LiteLlm` and expect an **OpenAI-compatible endpoint** (e.g. `llama.cpp`, `vLLM`) via `OPENAI_API_BASE`.

## Current State Of The Repo

Substantive:
- [src/agent_zoo/sql_agent/](src/agent_zoo/sql_agent/) — complete, callback-driven NL→SQLite agent.
- [src/agent_zoo/scope_guard.py](src/agent_zoo/scope_guard.py) — reusable LLM-based scope gate + several LLM resolvers (clarification, fresh-topic router, refinement, schema grounding). Used by the SQL agent today; designed to be reusable by future agents.
- [src/agent_zoo/working_memory.py](src/agent_zoo/working_memory.py) — tiny helper for per-agent namespaced session state.
- [src/agent_zoo/base.py](src/agent_zoo/base.py) — minimal `BaseAgent` ABC (`name`, `description`, `ask`).
- [tests/](tests/) — mostly SQL-agent behavior (`test_sql_agent_db.py` has ~145 tests), plus pipeline and working-memory tests.
- [scripts/csv_to_sqlite.py](scripts/csv_to_sqlite.py) — stdlib-only CSV→SQLite builder used to produce the bundled Titanic dataset.

Placeholder / not implemented:
- [src/agent_zoo/data_analysis_agent/](src/agent_zoo/data_analysis_agent/) and [src/agent_zoo/orchestrator_agent/](src/agent_zoo/orchestrator_agent/) — empty apart from `.adk/session.db` and `__pycache__`. No `.py` source. Treat as reserved names, not working agents. Do not reference them in examples or docs as if they exist.

## Where The Real Logic Lives

The live SQL-agent runtime path is **`agent.py` + `runtime.py` + the callbacks wired into the `LlmAgent`**, not `pipeline.py`.

For a common change, look here first:

| Task | File |
| --- | --- |
| Change how the `LlmAgent` is assembled (model, tools, callbacks) | [src/agent_zoo/sql_agent/agent.py](src/agent_zoo/sql_agent/agent.py) |
| Change session handling, runner cache, CLI question flow | [src/agent_zoo/sql_agent/runtime.py](src/agent_zoo/sql_agent/runtime.py) |
| Change clarification, refinement, grounding, privacy, final-answer rendering behavior | [src/agent_zoo/sql_agent/callbacks.py](src/agent_zoo/sql_agent/callbacks.py) (~4.5k lines, the real control plane) |
| Change DB safety / read-only enforcement / SQL validation / schema summary / object-mode rewrite | [src/agent_zoo/sql_agent/db.py](src/agent_zoo/sql_agent/db.py) |
| Change user-visible response shape (clarifications, filter summaries, result payloads) | [src/agent_zoo/sql_agent/formatting.py](src/agent_zoo/sql_agent/formatting.py) |
| Change the system prompt / analyst rules / runtime context appended to it | [src/agent_zoo/sql_agent/instructions.py](src/agent_zoo/sql_agent/instructions.py) |
| Add/modify ADK tools exposed to the model | [src/agent_zoo/sql_agent/tools.py](src/agent_zoo/sql_agent/tools.py) |
| Change LLM scope/grounding/refinement resolver behavior | [src/agent_zoo/scope_guard.py](src/agent_zoo/scope_guard.py) |
| Change config/env-var handling, defaults | [src/agent_zoo/sql_agent/config.py](src/agent_zoo/sql_agent/config.py) |

[src/agent_zoo/sql_agent/pipeline.py](src/agent_zoo/sql_agent/pipeline.py) is **not** the live path. It is a lightweight NL→SQL helper used by tests (`tests/test_sql_agent_pipeline.py`). It does not go through ADK, callbacks, clarifications, privacy shaping, or scope gating. Do not "fix" behavior by editing pipeline.py when the live ADK path is what's running.

There is also a **deep, accurate SQL-agent README** at [src/agent_zoo/sql_agent/README.md](src/agent_zoo/sql_agent/README.md). It is the best single reference for workflow stages, state keys, and resolver contracts — skim its Contents and jump into the relevant section before diving into `callbacks.py`.

## Setup / Environment

- Python 3.11+ required by [pyproject.toml](pyproject.toml). (`.python-version` pins `3.14`; the installed `.venv/` is what the test command below actually uses.)
- Two supported install flows per the README:
  - `uv` (preferred): `uv venv` then `uv pip install --python .venv/bin/python -e .` for local dev, or install from Git.
  - Plain `venv` + `pip`: slower full ADK dep resolution.
- `.env` is loaded by [src/agent_zoo/sql_agent/config.py](src/agent_zoo/sql_agent/config.py) via `python-dotenv`. Do not commit real secrets; do not print the existing `.env` contents in chat.
- This package expects an OpenAI-compatible LLM server. Point it at one with `OPENAI_API_BASE` (e.g. `http://127.0.0.1:8080/v1`). `OPENAI_API_KEY` is auto-filled with a placeholder when missing so LiteLLM does not reject the call.
- LLM model names are LiteLLM identifiers. Default is `openai/Qwen3.5-0.8B-GGUF`; override with `SQL_AGENT_MODEL`.

## Run / Test / Validate

Run the SQL agent (confirmed console entry point `run-sql-agent` → `agent_zoo.sql_agent.cli:main`):

```bash
# One-shot
uv run run-sql-agent --db dataset/titantic/titanic.sqlite --question "How many passengers survived?"

# Interactive
uv run run-sql-agent --db dataset/titantic/titanic.sqlite

# ADK-native dev UI (supported via ImportError fallbacks in agent.py/callbacks.py)
cd src && adk run agent_zoo/sql_agent
```

`--debug` prints the full ADK event stream and callback-level clarification tracing.

Tests — configured via `[tool.pytest.ini_options]` (`testpaths = ["tests"]`). The SQL-agent README's verified command uses `unittest`:

```bash
.venv/bin/python -m unittest tests.test_sql_agent_db tests.test_sql_agent_pipeline -v
```

A `pytest` invocation over `tests/` is the conventional alternative, but prefer the verified `unittest` command when reproducing the repo's own workflow.

After any non-trivial behavior change in the SQL agent:
1. Run the full test suite. The test file `tests/test_sql_agent_db.py` covers DB safety, callbacks, clarification normalization, privacy, object-mode rewriting, and filter coverage — most regressions show up here.
2. If UI/response text changed, run at least one `run-sql-agent --db dataset/titantic/titanic.sqlite --question ...` to sanity-check the rendered output. The CLI prints deterministic rendered text; don't rely on the test suite alone for wording changes.

## SQL Agent Architecture Notes

- **Entry**: `SQLAgent.ask()` → `runtime.ask_question()` → builds `LlmAgent` via `build_root_agent()` → `InMemoryRunner` → ADK event loop.
- **Runner caching**: `runtime._RUNNER_CACHE` / `_INITIALIZED_SESSION_KEYS` keyed on a tuple of settings. Re-using the same settings reuses the runner and its session. If you change settings fields, update `runtime._runner_cache_key`.
- **Callbacks wired into the agent** (see `agent.py`):
  - `before_model_callback`: `build_combined_before_model_callback` — scope gate, clarification follow-up, fresh-topic routing, schema grounding, and finalize-after-query short-circuit.
  - `after_model_callback`: `build_normalize_clarification_after_model_callback` — collapses free-form clarifications into the canonical `{"response_type":"clarification", ...}` shape and stores a pending clarification in state.
  - `after_tool_callback`: `build_remember_query_result_callback` — persists a public/internal split of tool output into session state, builds `last_query_frame` for refinement memory, and (when `count_aggregates_only`) returns the shaped result directly as the tool response.
  - `after_agent_callback`: `build_format_final_agent_response_callback` — renders the deterministic final answer from the stored public query result.
- **State keys** live in `callbacks.py` as `SQL_*_STATE_KEY` constants (all `temp:` prefixed, e.g. `temp:sql_public_result`, `temp:sql_last_user_text`). Cross-turn working memory uses the `agent_working_memory` namespace via [working_memory.py](src/agent_zoo/working_memory.py).
- **Prompts**: `instructions.py::DEFAULT_INSTRUCTION` is the analyst-style system prompt; `build_agent_instruction()` appends the resolved DB path, runtime-config block, and live schema snapshot (including optional categorical-value guidance).
- **DB safety (`db.py`)**:
  - SQLite connections use `sqlite3.connect(<uri>, uri=True)` with a URI of the form `file://…?mode=ro`. Non-SELECT statements at runtime would fail, but the code also blocks them up front.
  - `validate_sql_read_only()` rejects empty SQL, multiple statements, and anything whose effective root keyword is not `SELECT`/`WITH` → `SELECT`. It then runs `EXPLAIN QUERY PLAN` against the read-only connection to catch schema-level errors before executing.
  - `execute_sqlite_query()` is the user-facing wrapper; it calls the validator, optionally rewrites CASE groups to surface a `'Null'` bucket, and can route through object-mode CTEs when `object_id_column` + `object_order_column` are set.
- **Scope gate / resolvers** (`scope_guard.py`): a set of `build_llm_*` factories returning callables that make a single low-token LiteLLM call and **fail open** on any error — do not change that fail-open semantics without a clear reason.

## Change Guidance

- Prefer minimal diffs. The SQL agent has a lot of invariants encoded across `callbacks.py`, `db.py`, and `formatting.py`; the test suite will usually catch accidental changes but subtle wording shifts can still escape.
- Preserve existing determinism: structured clarification JSON, result rendering via `format_public_query_result`, read-only validation, fail-open classifiers. Do not replace deterministic branches with free-form LLM calls.
- If you change any session-state schema, update both the writer (in `callbacks.py`) and any reader across `callbacks.py`/`scope_guard.py`, and update the tests.
- If you add a new ADK tool, register it in [tools.py](src/agent_zoo/sql_agent/tools.py) and add it to `build_sql_tools(...)`. Keep the tool docstring precise — ADK surfaces it to the model.
- Schema-driven behavior: categorical-value guidance, grounding, object-mode, and Null-bucket rewrites are all computed from `get_schema_summary(...)`. Do not hardcode dataset-specific columns. If a new dataset needs special handling, extend the schema-driven mechanism rather than special-casing.
- When changing the rendered response shape, keep `format_public_query_result` as the single rendering path. The `after_agent_callback` and the `after_tool_callback` short-circuit both end up there.
- Update docs/tests in the same change:
  - [src/agent_zoo/sql_agent/README.md](src/agent_zoo/sql_agent/README.md) is code-grounded — keep it truthful.
  - Top-level [README.md](README.md) is user-facing; do not surface internal state keys there.

## ADK-Specific Rules

- The repo-level instruction at [.github/copilot-instructions.md](.github/copilot-instructions.md) says to consult the `adk-docs-mcp` server as the primary source of truth for ADK behavior. Apply the same rule to Claude: if a question is about ADK callback lifecycle, `LlmAgent`/`LlmResponse` semantics, runner/session APIs, or MCP/tool plumbing, fetch ADK docs via the `adk-docs` MCP (`list_doc_sources` → `fetch_docs`) before editing.
- If the ADK docs MCP is unavailable, say so briefly, then fall back to local code inspection and mark any ADK-specific conclusion as tentative. Do not guess ADK behavior from training data when the repo evidence is thin.
- Note that [src/agent_zoo/sql_agent/callbacks.py](src/agent_zoo/sql_agent/callbacks.py) and [src/agent_zoo/sql_agent/agent.py](src/agent_zoo/sql_agent/agent.py) both have `try/except ImportError` blocks that support being loaded either as `agent_zoo.sql_agent...` (package) or as top-level `sql_agent` (ADK dev-UI run mode). Preserve both paths when adding new imports to these files.

## Common Pitfalls

- Editing [pipeline.py](src/agent_zoo/sql_agent/pipeline.py) thinking it affects the live CLI. It doesn't — it's test/helper code only.
- Mocking the LLM in ADK path tests: the existing approach (see `test_ask_question_reuses_cached_runner_for_same_session` in `test_sql_agent_pipeline.py`) patches `InMemoryRunner` + `build_root_agent` and clears `runtime._RUNNER_CACHE` / `_INITIALIZED_SESSION_KEYS`. Re-use that pattern instead of inventing a new fixture.
- `count_aggregates_only` is `True` by default. That changes the tool-response pathway: the `after_tool_callback` returns the shaped public result and the model never sees raw rows. If you're debugging "why did the model not see the rows?", check that flag first.
- The default DB path is literally `dataset/titantic/titanic.sqlite` — the misspelled directory `titantic` is load-bearing. Don't "fix" it as a typo without updating `config.DEFAULT_DB_PATH` and every reference in `README.md`, the SQL-agent README, and the scripts.
- `config.resolve_repo_path(...)` resolves relative paths against CWD first, then the project root. Absolute paths pass through unchanged. Keep this behavior when adding new path-valued settings.
- The scope gate and all LLM resolvers fail open — if a resolver throws or returns malformed JSON, behavior falls through to the main model path. Don't assume a classifier verdict is always present.
- Secrets live in `.env` (git-ignored via `.gitignore`). Never echo, cat, or paste the contents of `.env` into messages, commits, or PR descriptions.
