"""Format SQL execution results into the agent's final user-facing response.

This module converts structured tool output into the four-section response
format expected by the SQL agent. It is used by callbacks and is not meant to
be run as a standalone script.
"""

from __future__ import annotations

import json

def format_result_payload(tool_result: dict) -> str:
    if tool_result["status"] != "success":
        return tool_result.get("error") or "Execution failed."

    rows = tool_result.get("rows", [])
    if not rows:
        return "No rows returned."

    if len(rows) == 1 and len(rows[0]) == 1:
        return str(next(iter(rows[0].values())))

    return json.dumps(rows, indent=2)


def _get_matching_row_count(tool_result: dict) -> int | float | None:
    matched_row_count = tool_result.get("matched_row_count")
    if isinstance(matched_row_count, (int, float)) and not isinstance(matched_row_count, bool):
        return matched_row_count
    return None


def _pluralize(value: int | float, singular: str, plural: str) -> str:
    return singular if value == 1 else plural


def summarize_execution_result(tool_result: dict) -> str:
    if tool_result["status"] != "success":
        return tool_result.get("error") or "The query failed."

    public_result_kind = tool_result.get("public_result_kind")
    matched_row_count = _get_matching_row_count(tool_result)
    aggregate_columns = tool_result.get("aggregate_columns") or []
    if public_result_kind == "safe_aggregate":
        if tool_result.get("row_count", 0) > 1:
            if matched_row_count is not None:
                group_count = tool_result["row_count"]
                return (
                    "Computed grouped cohort-level aggregates across "
                    f"{group_count} {_pluralize(group_count, 'group', 'groups')} "
                    f"covering {matched_row_count} matching rows."
                )
            return "Computed grouped cohort-level aggregates."
        if matched_row_count is not None:
            return f"Computed cohort-level aggregate values for {matched_row_count} matching rows."
        if aggregate_columns:
            return "Computed cohort-level aggregate values."
        return "Computed a cohort-level aggregate value."

    if matched_row_count is not None:
        if matched_row_count == 0:
            return "No matching rows were found."
        if matched_row_count == 1:
            return "Found 1 matching row."
        return f"Found {matched_row_count} matching rows."

    row_count = tool_result["row_count"]
    if row_count == 0:
        return "No matching rows were found."
    if row_count == 1:
        return "Found 1 matching row."
    if tool_result["truncated"]:
        return f"Found {row_count} matching rows. Returning a preview."
    return f"Found {row_count} matching rows."


def build_default_explanation(tool_result: dict) -> str:
    if tool_result["status"] != "success":
        return tool_result.get("error") or "The query could not be executed safely."

    public_result_kind = tool_result.get("public_result_kind")
    matched_row_count = _get_matching_row_count(tool_result)
    if public_result_kind == "safe_aggregate":
        if tool_result.get("row_count", 0) > 1 and matched_row_count is not None:
            return (
                "The query executed successfully and returned grouped cohort-level "
                f"aggregate values spanning {matched_row_count} matching row(s)."
            )
        if matched_row_count is not None:
            return (
                "The query executed successfully and returned cohort-level aggregate "
                f"values computed over {matched_row_count} matching row(s)."
            )
        return "The query executed successfully and returned cohort-level aggregate values."

    if matched_row_count is not None:
        if matched_row_count == 0:
            return "The query executed successfully but returned no matching rows."
        if public_result_kind == "detail_count_fallback":
            return (
                "The query matched rows successfully, but detailed row output is "
                "suppressed in privacy mode, so only the matching count is shown."
            )
        if tool_result.get("rows") and len(tool_result["rows"][0]) > 1:
            return (
                "The query executed successfully and returned grouped counts covering "
                f"{matched_row_count} matching row(s)."
            )
        return (
            "The query executed successfully and the public response reports "
            f"{matched_row_count} matching row(s)."
        )

    row_count = tool_result.get("row_count", 0)
    if row_count == 0:
        return "The query executed successfully but returned no matching rows."
    if tool_result.get("truncated"):
        return f"The query executed successfully and returned {row_count} rows, so this response shows a preview."
    return f"The query executed successfully and returned {row_count} row(s)."


def format_structured_response(tool_result: dict, explanation: str | None = None) -> str:
    sql = tool_result.get("sql") or "Not executed"
    result_payload = format_result_payload(tool_result)

    code_block = "json" if result_payload.startswith("[") or result_payload.startswith("{") else ""
    result_block = f"```{code_block}\n{result_payload}\n```".strip()

    return (
        "Generated SQL:\n"
        f"```sql\n{sql}\n```\n\n"
        "Result:\n"
        f"{result_block}"
    )
