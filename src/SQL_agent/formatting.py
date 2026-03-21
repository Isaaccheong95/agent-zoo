from __future__ import annotations

import json

from .pipeline import summarize_execution_result


def format_result_payload(tool_result: dict) -> str:
    if tool_result["status"] != "success":
        return tool_result.get("error") or "Execution failed."

    rows = tool_result.get("rows", [])
    if not rows:
        return "No rows returned."

    if len(rows) == 1 and len(rows[0]) == 1:
        return str(next(iter(rows[0].values())))

    return json.dumps(rows, indent=2)


def build_default_explanation(tool_result: dict) -> str:
    if tool_result["status"] != "success":
        return tool_result.get("error") or "The query could not be executed safely."

    row_count = tool_result.get("row_count", 0)
    if row_count == 0:
        return "The query executed successfully but returned no matching rows."
    if tool_result.get("truncated"):
        return f"The query executed successfully and returned {row_count} rows, so this response shows a preview."
    return f"The query executed successfully and returned {row_count} row(s)."


def format_structured_response(tool_result: dict, explanation: str | None = None) -> str:
    sql = tool_result.get("sql") or "Not executed"
    summary = summarize_execution_result(tool_result)
    result_payload = format_result_payload(tool_result)
    detail = explanation.strip() if explanation and explanation.strip() else build_default_explanation(tool_result)

    code_block = "json" if result_payload.startswith("[") or result_payload.startswith("{") else ""
    result_block = f"```{code_block}\n{result_payload}\n```".strip()

    return (
        "Generated SQL:\n"
        f"```sql\n{sql}\n```\n\n"
        "Result summary:\n"
        f"{summary}\n\n"
        "Result:\n"
        f"{result_block}\n\n"
        "Explanation:\n"
        f"{detail}"
    )
