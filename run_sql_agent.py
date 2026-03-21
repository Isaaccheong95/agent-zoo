from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from SQL_agent.config import load_settings
from SQL_agent.runtime import ask_question, run_interactive_loop


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local ADK SQLite NL-to-SQL agent.")
    parser.add_argument("--db", dest="db_path", help="Path to the SQLite database file.")
    parser.add_argument("--model", help="LiteLLM model identifier to use.")
    parser.add_argument("--debug", action="store_true", help="Print intermediate ADK events while the agent is running.")
    parser.add_argument("--instruction-file", help="Optional path to a custom base instruction file.")
    parser.add_argument("--question", help="Run a single question and exit. If omitted, an interactive loop starts.")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
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
        return

    await run_interactive_loop(settings)


if __name__ == "__main__":
    asyncio.run(main())
