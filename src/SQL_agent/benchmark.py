from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import re
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Iterable, Sequence

from google.adk.runners import InMemoryRunner
from google.genai.types import Content, Part

from .agent import build_root_agent
from .config import PROJECT_ROOT, SQLAgentSettings, load_settings


DEFAULT_QUESTIONS_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "mini_bird_bench"
    / "finetuning"
    / "inference"
    / "mini_dev_prompt.jsonl"
)
DEFAULT_DB_ROOT = PROJECT_ROOT / "dataset" / "mini_bird_bench" / "databases"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "mini_bird_bench"
OFFICIAL_EVAL_DIR = PROJECT_ROOT / "dataset" / "mini_bird_bench" / "evaluation"
SQL_CODE_BLOCK_PATTERN = re.compile(
    r"Generated SQL:\s*```(?:sql|sqlite|SQLite)?\s*(.*?)```",
    re.DOTALL,
)
GENERIC_SQL_CODE_BLOCK_PATTERN = re.compile(
    r"```(?:sql|sqlite|SQLite)?\s*(.*?)```",
    re.DOTALL,
)


@dataclass(slots=True)
class BirdBenchmarkExample:
    example_id: int
    db_id: str
    question: str
    evidence: str | None = None
    gold_sql: str | None = None
    difficulty: str | None = None
    question_id: int | None = None


@dataclass(slots=True)
class BenchmarkPredictionResult:
    example_id: int
    db_id: str
    question: str
    sql: str
    final_response: str
    status: str
    error: str | None = None
    session_id: str | None = None
    question_id: int | None = None


@dataclass(slots=True)
class RunnerBundle:
    db_id: str
    db_path: Path
    settings: SQLAgentSettings
    runner: InMemoryRunner


@dataclass(slots=True)
class StreamCapture:
    sql: str
    final_response: str
    status: str
    error: str | None = None


RunnerFactory = Callable[[str, SQLAgentSettings], InMemoryRunner]
SessionIdFactory = Callable[[BirdBenchmarkExample], str]


def _text_from_parts(parts: Iterable[Part]) -> str:
    return "".join(part.text for part in parts if getattr(part, "text", None))


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
    return rows


def _normalize_questions_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        if all(str(key).isdigit() for key in payload):
            return [payload[key] for key in sorted(payload, key=lambda value: int(value))]
        raise ValueError(
            "Question JSON must be a list of examples or a numeric-keyed object."
        )

    raise ValueError("Unsupported questions payload format.")


def load_bird_examples(
    questions_path: str | Path,
    *,
    limit: int | None = None,
) -> list[BirdBenchmarkExample]:
    path = Path(questions_path)
    if not path.exists():
        raise FileNotFoundError(f"Benchmark questions file not found: {path}")

    if path.suffix.lower() == ".jsonl":
        raw_examples = _load_jsonl(path)
    elif path.suffix.lower() == ".json":
        raw_examples = _normalize_questions_payload(_load_json(path))
    else:
        raise ValueError(
            f"Unsupported questions file extension for {path}. Use .json or .jsonl."
        )

    examples: list[BirdBenchmarkExample] = []
    for index, raw in enumerate(raw_examples):
        db_id = str(raw.get("db_id", "")).strip()
        question = str(raw.get("question", "")).strip()
        if not db_id or not question:
            raise ValueError(
                f"Example {index} is missing a required 'db_id' or 'question' field."
            )
        examples.append(
            BirdBenchmarkExample(
                example_id=index,
                db_id=db_id,
                question=question,
                evidence=raw.get("evidence"),
                gold_sql=raw.get("SQL") or raw.get("sql"),
                difficulty=raw.get("difficulty"),
                question_id=raw.get("question_id"),
            )
        )

    if limit is not None:
        return examples[: max(0, limit)]

    return examples


def discover_sqlite_databases(db_root: str | Path) -> dict[str, Path]:
    root = Path(db_root)
    if not root.exists():
        raise FileNotFoundError(f"Database root not found: {root}")

    mapping: dict[str, Path] = {}
    duplicates: set[str] = set()
    for sqlite_path in root.rglob("*.sqlite"):
        db_id = sqlite_path.stem
        if db_id in mapping and mapping[db_id] != sqlite_path:
            duplicates.add(db_id)
            continue
        mapping.setdefault(db_id, sqlite_path.resolve())

    if not mapping:
        raise FileNotFoundError(
            f"No SQLite database files were found beneath {root.resolve()}."
        )

    if duplicates:
        duplicate_names = ", ".join(sorted(duplicates))
        raise ValueError(
            f"Multiple SQLite files were found for these db_ids: {duplicate_names}"
        )

    return mapping


def extract_sql_from_final_response(response_text: str) -> str:
    if not response_text:
        return ""

    for pattern in (SQL_CODE_BLOCK_PATTERN, GENERIC_SQL_CODE_BLOCK_PATTERN):
        match = pattern.search(response_text)
        if match:
            return match.group(1).strip()

    return ""


def _coerce_tool_response_payload(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    if hasattr(payload, "model_dump"):
        return payload.model_dump()
    if hasattr(payload, "dict"):
        return payload.dict()
    return {}


async def collect_prediction_from_events(events: Any) -> StreamCapture:
    predicted_sql = ""
    final_response = ""
    status = "error"
    error: str | None = None

    async for event in events:
        content = getattr(event, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            function_response = getattr(part, "function_response", None)
            if function_response is None:
                continue
            if getattr(function_response, "name", "") != "execute_sqlite_read_only":
                continue

            payload = _coerce_tool_response_payload(getattr(function_response, "response", None))
            sql = str(payload.get("sql") or "").strip()
            if sql:
                predicted_sql = sql
            tool_status = str(payload.get("status") or "").strip()
            if tool_status:
                status = tool_status
            payload_error = payload.get("error")
            if payload_error:
                error = str(payload_error)

        is_final_response = getattr(event, "is_final_response", None)
        if callable(is_final_response) and is_final_response() and getattr(event, "author", "") != "user":
            final_response = _text_from_parts(parts)

    if not predicted_sql and final_response:
        predicted_sql = extract_sql_from_final_response(final_response)
        if predicted_sql and status == "error" and error is None:
            status = "success"

    if not predicted_sql and error is None:
        error = "No SQL could be extracted from the agent response."

    return StreamCapture(
        sql=predicted_sql,
        final_response=final_response,
        status=status if predicted_sql else "error",
        error=error,
    )


def default_session_id_factory(example: BirdBenchmarkExample) -> str:
    return f"mini_bird_{example.example_id}_{uuid.uuid4().hex}"


def default_runner_factory(db_id: str, settings: SQLAgentSettings) -> InMemoryRunner:
    return InMemoryRunner(
        agent=build_root_agent(settings),
        app_name=f"{settings.app_name}_{db_id}",
    )


class AgentRunnerCache:
    def __init__(
        self,
        *,
        model: str | None = None,
        debug: bool = False,
        instruction_file: str | Path | None = None,
        runner_factory: RunnerFactory | None = None,
    ) -> None:
        self._model = model
        self._debug = debug
        self._instruction_file = str(instruction_file) if instruction_file else None
        self._runner_factory = runner_factory or default_runner_factory
        self._bundles: dict[str, RunnerBundle] = {}

    def get_bundle(self, db_id: str, db_path: Path) -> RunnerBundle:
        bundle = self._bundles.get(db_id)
        resolved_path = db_path.resolve()
        if bundle is not None and bundle.db_path == resolved_path:
            return bundle

        overrides: dict[str, Any] = {
            "db_path": str(resolved_path),
            "debug": self._debug,
        }
        if self._model:
            overrides["model"] = self._model
        if self._instruction_file:
            overrides["instruction_file"] = self._instruction_file

        settings = load_settings(overrides)
        runner = self._runner_factory(db_id, settings)
        bundle = RunnerBundle(
            db_id=db_id,
            db_path=resolved_path,
            settings=settings,
            runner=runner,
        )
        self._bundles[db_id] = bundle
        return bundle


async def run_benchmark_example(
    example: BirdBenchmarkExample,
    *,
    bundle: RunnerBundle,
    session_id_factory: SessionIdFactory = default_session_id_factory,
) -> BenchmarkPredictionResult:
    session_id = session_id_factory(example)
    await bundle.runner.session_service.create_session(
        app_name=bundle.runner.app_name,
        user_id=bundle.settings.user_id,
        session_id=session_id,
    )

    content = Content(role="user", parts=[Part(text=example.question)])
    events = bundle.runner.run_async(
        user_id=bundle.settings.user_id,
        session_id=session_id,
        new_message=content,
    )
    capture = await collect_prediction_from_events(events)

    return BenchmarkPredictionResult(
        example_id=example.example_id,
        db_id=example.db_id,
        question=example.question,
        sql=capture.sql,
        final_response=capture.final_response,
        status=capture.status,
        error=capture.error,
        session_id=session_id,
        question_id=example.question_id,
    )


async def generate_benchmark_predictions(
    examples: Sequence[BirdBenchmarkExample],
    *,
    db_root: str | Path,
    model: str | None = None,
    debug: bool = False,
    instruction_file: str | Path | None = None,
    runner_cache: AgentRunnerCache | None = None,
    session_id_factory: SessionIdFactory = default_session_id_factory,
) -> list[BenchmarkPredictionResult]:
    database_paths = discover_sqlite_databases(db_root)
    active_cache = runner_cache or AgentRunnerCache(
        model=model,
        debug=debug,
        instruction_file=instruction_file,
    )

    results: list[BenchmarkPredictionResult] = []
    for example in examples:
        if example.db_id not in database_paths:
            raise FileNotFoundError(
                f"No SQLite database was found for db_id '{example.db_id}' beneath {Path(db_root).resolve()}."
            )
        bundle = active_cache.get_bundle(example.db_id, database_paths[example.db_id])
        result = await run_benchmark_example(
            example,
            bundle=bundle,
            session_id_factory=session_id_factory,
        )
        results.append(result)

    return results


def _serialize_prediction(result: BenchmarkPredictionResult) -> str:
    return f"{result.sql}\t----- bird -----\t{result.db_id}"


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_prediction_artifacts(
    results: Sequence[BenchmarkPredictionResult],
    output_dir: str | Path,
    *,
    prediction_filename: str = "predict_mini_dev_sql_agent_sqlite.json",
) -> tuple[Path, Path]:
    directory = ensure_directory(output_dir)
    prediction_path = directory / prediction_filename
    debug_trace_path = directory / f"{prediction_path.stem}.debug.jsonl"

    payload = {
        str(index): _serialize_prediction(result)
        for index, result in enumerate(results)
    }
    prediction_path.write_text(json.dumps(payload, indent=4), encoding="utf-8")

    with debug_trace_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(asdict(result), ensure_ascii=True) + "\n")

    return prediction_path, debug_trace_path


def materialize_gold_sql_file(
    examples: Sequence[BirdBenchmarkExample],
    output_dir: str | Path,
    *,
    filename: str = "mini_dev_sqlite_gold.sql",
) -> Path:
    if any(not example.gold_sql for example in examples):
        raise ValueError(
            "Unable to materialize a gold SQL file because at least one example is missing the gold SQL."
        )

    path = ensure_directory(output_dir) / filename
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(f"{example.gold_sql}\t{example.db_id}\n")
    return path


def materialize_difficulty_file(
    examples: Sequence[BirdBenchmarkExample],
    output_dir: str | Path,
    *,
    filename: str = "mini_dev_sqlite_difficulty.jsonl",
) -> Path:
    if any(not example.difficulty for example in examples):
        raise ValueError(
            "Unable to materialize a difficulty file because at least one example is missing the difficulty."
        )

    path = ensure_directory(output_dir) / filename
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            payload = {
                "question_id": example.question_id,
                "db_id": example.db_id,
                "question": example.question,
                "difficulty": example.difficulty,
            }
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
    return path


def _path_for_evaluator(path: str | Path, *, directory: bool = False) -> str:
    resolved = Path(path).resolve()
    as_posix = resolved.as_posix()
    if directory and not as_posix.endswith("/"):
        return as_posix + "/"
    return as_posix


def _require_benchmark_dependency(module_name: str) -> None:
    if importlib.util.find_spec(module_name) is None:
        raise RuntimeError(
            "Mini-BIRD evaluation requires the optional benchmark dependencies. "
            "Install them with `uv sync --extra benchmark` and rerun."
        )


def ensure_evaluation_dependencies() -> None:
    for module_name in ("func_timeout", "pymysql", "psycopg2"):
        _require_benchmark_dependency(module_name)


def _run_official_evaluator(
    script_name: str,
    *,
    prediction_path: Path,
    gold_path: Path,
    difficulty_path: Path,
    db_root: Path,
    output_log_path: Path,
    num_cpus: int,
    meta_time_out: float,
) -> subprocess.CompletedProcess[str]:
    ensure_directory(output_log_path.parent)
    command = [
        sys.executable,
        str((OFFICIAL_EVAL_DIR / script_name).resolve()),
        "--predicted_sql_path",
        _path_for_evaluator(prediction_path),
        "--ground_truth_path",
        _path_for_evaluator(gold_path),
        "--db_root_path",
        _path_for_evaluator(db_root, directory=True),
        "--num_cpus",
        str(num_cpus),
        "--meta_time_out",
        str(meta_time_out),
        "--diff_json_path",
        _path_for_evaluator(difficulty_path),
        "--sql_dialect",
        "SQLite",
        "--output_log_path",
        _path_for_evaluator(output_log_path),
    ]
    completed = subprocess.run(
        command,
        cwd=str(OFFICIAL_EVAL_DIR.resolve()),
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{script_name} failed with exit code {completed.returncode}.\n"
            f"STDOUT:\n{completed.stdout}\n\nSTDERR:\n{completed.stderr}"
        )
    return completed


def _parse_metric_total(output_text: str, metric_name: str) -> float | None:
    pattern = re.compile(
        rf"^{re.escape(metric_name)}\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*$",
        re.MULTILINE,
    )
    match = pattern.search(output_text)
    if not match:
        return None
    return float(match.group(4))


def run_official_sqlite_evaluations(
    *,
    prediction_path: str | Path,
    gold_path: str | Path,
    difficulty_path: str | Path,
    db_root: str | Path,
    output_dir: str | Path,
    num_cpus: int,
    ex_timeout: float = 30.0,
    f1_timeout: float = 30.0,
    ves_timeout: float = 30.0,
    ves_repeats: int = 1,
) -> dict[str, Any]:
    ensure_evaluation_dependencies()

    prediction_path = Path(prediction_path)
    gold_path = Path(gold_path)
    difficulty_path = Path(difficulty_path)
    db_root = Path(db_root)
    evaluation_output_dir = ensure_directory(output_dir)

    summary: dict[str, Any] = {}

    ex_completed = _run_official_evaluator(
        "evaluation_ex.py",
        prediction_path=prediction_path,
        gold_path=gold_path,
        difficulty_path=difficulty_path,
        db_root=db_root,
        output_log_path=evaluation_output_dir / "execution_accuracy.txt",
        num_cpus=num_cpus,
        meta_time_out=ex_timeout,
    )
    summary["EX"] = {
        "total": _parse_metric_total(ex_completed.stdout, "EX"),
        "stdout": ex_completed.stdout,
        "stderr": ex_completed.stderr,
        "log_path": str((evaluation_output_dir / "execution_accuracy.txt").resolve()),
    }

    f1_completed = _run_official_evaluator(
        "evaluation_f1.py",
        prediction_path=prediction_path,
        gold_path=gold_path,
        difficulty_path=difficulty_path,
        db_root=db_root,
        output_log_path=evaluation_output_dir / "soft_f1.txt",
        num_cpus=num_cpus,
        meta_time_out=f1_timeout,
    )
    summary["Soft-F1"] = {
        "total": _parse_metric_total(f1_completed.stdout, "Soft-F1"),
        "stdout": f1_completed.stdout,
        "stderr": f1_completed.stderr,
        "log_path": str((evaluation_output_dir / "soft_f1.txt").resolve()),
    }

    ves_runs: list[dict[str, Any]] = []
    repeat_count = max(1, ves_repeats)
    for index in range(repeat_count):
        log_path = evaluation_output_dir / f"r_ves_run_{index + 1}.txt"
        completed = _run_official_evaluator(
            "evaluation_ves.py",
            prediction_path=prediction_path,
            gold_path=gold_path,
            difficulty_path=difficulty_path,
            db_root=db_root,
            output_log_path=log_path,
            num_cpus=num_cpus,
            meta_time_out=ves_timeout,
        )
        ves_runs.append(
            {
                "run_index": index + 1,
                "total": _parse_metric_total(completed.stdout, "R-VES"),
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "log_path": str(log_path.resolve()),
            }
        )

    totals = [run["total"] for run in ves_runs if run["total"] is not None]
    summary["R-VES"] = {
        "runs": ves_runs,
        "best_total": max(totals) if totals else None,
        "average_total": mean(totals) if totals else None,
    }

    summary_path = evaluation_output_dir / "metrics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path.resolve())
    return summary


def _resolve_active_evaluation_path(
    *,
    explicit_path: str | Path | None,
    generated_path_factory: Callable[[], Path],
) -> Path:
    if explicit_path is not None:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"Evaluation file not found: {path}")
        return path
    return generated_path_factory()


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the ADK SQL agent on Mini-BIRD SQLite and optionally execute the official evaluators.",
    )
    parser.add_argument("--questions-path", default=str(DEFAULT_QUESTIONS_PATH))
    parser.add_argument("--db-root", default=str(DEFAULT_DB_ROOT))
    parser.add_argument("--gold-path")
    parser.add_argument("--difficulty-path")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--model",
        help="Model to use. Supports ADK-native model strings like `gemini-2.5-flash` and LiteLLM-style identifiers like `openai/gpt-4o`.",
    )
    parser.add_argument("--instruction-file")
    parser.add_argument("--limit", type=int, help="Optional number of benchmark examples to run.")
    parser.add_argument("--num-cpus", type=int, default=max(1, min(16, os.cpu_count() or 1)))
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--debug", action="store_true", help="Enable ADK runtime debug output.")
    parser.add_argument("--ves-timeout", type=float, default=30.0)
    parser.add_argument("--ves-repeats", type=int, default=1)
    parser.add_argument("--ex-timeout", type=float, default=30.0)
    parser.add_argument("--f1-timeout", type=float, default=30.0)
    return parser


async def _run_cli_async(args: argparse.Namespace) -> int:
    examples = load_bird_examples(args.questions_path, limit=args.limit)
    output_dir = ensure_directory(args.output_dir)

    results = await generate_benchmark_predictions(
        examples,
        db_root=args.db_root,
        model=args.model,
        debug=args.debug,
        instruction_file=args.instruction_file,
    )
    prediction_path, debug_trace_path = write_prediction_artifacts(results, output_dir)

    print(f"Wrote {len(results)} predictions to {prediction_path}")
    print(f"Wrote debug traces to {debug_trace_path}")

    if args.skip_eval:
        return 0

    generated_eval_dir = ensure_directory(output_dir / "generated_eval_inputs")
    gold_path = _resolve_active_evaluation_path(
        explicit_path=args.gold_path,
        generated_path_factory=lambda: materialize_gold_sql_file(examples, generated_eval_dir),
    )
    difficulty_path = _resolve_active_evaluation_path(
        explicit_path=args.difficulty_path,
        generated_path_factory=lambda: materialize_difficulty_file(examples, generated_eval_dir),
    )

    evaluation_summary = run_official_sqlite_evaluations(
        prediction_path=prediction_path,
        gold_path=gold_path,
        difficulty_path=difficulty_path,
        db_root=args.db_root,
        output_dir=output_dir / "evaluation",
        num_cpus=max(1, args.num_cpus),
        ex_timeout=args.ex_timeout,
        f1_timeout=args.f1_timeout,
        ves_timeout=args.ves_timeout,
        ves_repeats=max(1, args.ves_repeats),
    )

    ex_total = evaluation_summary["EX"]["total"]
    f1_total = evaluation_summary["Soft-F1"]["total"]
    ves_best = evaluation_summary["R-VES"]["best_total"]
    print(f"EX total: {ex_total}")
    print(f"Soft-F1 total: {f1_total}")
    print(f"R-VES best total across {max(1, args.ves_repeats)} run(s): {ves_best}")
    print(f"Saved evaluation summary to {evaluation_summary['summary_path']}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    return asyncio.run(_run_cli_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
