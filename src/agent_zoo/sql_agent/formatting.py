"""Format SQL execution results into the agent's final user-facing response.

This module converts structured tool output into the four-section response
format expected by the SQL agent. It is used by callbacks and is not meant to
be run as a standalone script.
"""

from __future__ import annotations

import json
from typing import Any


RESPONSE_SCHEMA_VERSION = 1
DEFAULT_CLARIFICATION_USER_MESSAGE = (
    "I need clarification before I can run a query. Please restate the request more specifically."
)
INVALID_CLARIFICATION_PAYLOAD_ERROR = "invalid_clarification_payload"


def serialize_response_envelope(response: dict[str, Any]) -> str:
    return json.dumps(response, indent=2)


def build_sql_result_response(tool_result: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": RESPONSE_SCHEMA_VERSION,
        "response_type": "sql_result",
        "user_message": summarize_execution_result(tool_result),
        "sql_result": dict(tool_result),
    }


def build_clarification_response(
    user_message: str,
    options: list[str] | None = None,
    *,
    error: str | None = None,
) -> dict[str, Any]:
    response = {
        "schema_version": RESPONSE_SCHEMA_VERSION,
        "response_type": "clarification",
        "user_message": user_message,
        "options": options or [],
    }
    if error:
        response["error"] = error
    return response


def _unwrap_json_code_fence(value: str) -> str:
    stripped = value.strip()
    if not stripped.startswith("```") or not stripped.endswith("```"):
        return stripped

    lines = stripped.splitlines()
    if len(lines) < 2 or lines[-1].strip() != "```":
        return stripped
    return "\n".join(lines[1:-1]).strip()


def parse_clarification_response(raw_text: str) -> dict[str, Any] | None:
    candidate = _unwrap_json_code_fence(raw_text)
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return None

    if not isinstance(payload, dict):
        return None
    if payload.get("response_type") != "clarification":
        return None

    user_message = payload.get("user_message")
    if not isinstance(user_message, str) or not user_message.strip():
        return None

    raw_options = payload.get("options", [])
    if raw_options is None:
        options: list[str] = []
    elif not isinstance(raw_options, list):
        return None
    else:
        options = []
        for option in raw_options:
            if not isinstance(option, str) or not option.strip():
                return None
            options.append(option.strip())

    return build_clarification_response(user_message.strip(), options)


def normalize_clarification_response(raw_text: str) -> dict[str, Any]:
    clarification = parse_clarification_response(raw_text)
    if clarification is not None:
        return clarification
    return build_clarification_response(
        DEFAULT_CLARIFICATION_USER_MESSAGE,
        [],
        error=INVALID_CLARIFICATION_PAYLOAD_ERROR,
    )

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
    return serialize_response_envelope(build_sql_result_response(tool_result))
