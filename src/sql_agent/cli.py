from __future__ import annotations

import argparse
import asyncio

from .config import load_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local ADK SQLite NL-to-SQL agent.")
    parser.add_argument("--db", dest="db_path", help="Path to the SQLite database file.")
    parser.add_argument("--model", help="LiteLLM model identifier to use.")
    parser.add_argument("--debug", action="store_true", help="Print intermediate ADK events while the agent is running.")
    parser.add_argument("--instruction-file", help="Optional path to a custom base instruction file.")
    parser.add_argument("--question", help="Run a single question and exit. If omitted, an interactive loop starts.")
    return parser.parse_args()


async def _main_async() -> int:
    args = parse_args()
    from .runtime import ask_question, run_interactive_loop

    settings = load_settings(
        {
            "db_path": args.db_path,
            "model": args.model,
            "debug": args.debug,
            "instruction_file": args.instruction_file,
        }
    )

    if args.question:
        response = await ask_question(args.question, settings)
        print(response)
        return 0

    await run_interactive_loop(settings)
    return 0


def main() -> int:
    return asyncio.run(_main_async())


if __name__ == "__main__":
    raise SystemExit(main())
