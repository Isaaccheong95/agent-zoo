"""Format SQL execution results into the agent's final user-facing response.

This module converts structured tool output into the four-section response
format expected by the SQL agent. It is used by callbacks and is not meant to
be run as a standalone script.
"""

from __future__ import annotations

import json
import re
from typing import Any


_CLARIFICATION_METADATA_KEYS = {
    "clarification",
    "options",
    "response_type",
    "user_message",
}

_CLARIFICATION_REPLY_GUIDANCE = "Choose one or more options, or describe your own rule."


def _normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(value)
    return deduped


def _unwrap_json_code_fence(value: str) -> str:
    stripped = value.strip()
    if not stripped.startswith("```") or not stripped.endswith("```"):
        return stripped

    lines = stripped.splitlines()
    if len(lines) < 2 or lines[-1].strip() != "```":
        return stripped
    return "\n".join(lines[1:-1]).strip()


def _clean_option_text(value: str) -> str | None:
    candidate = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", value).strip()
    candidate = candidate.strip("`\"'")
    candidate = _normalize_whitespace(candidate)
    if not candidate:
        return None
    if candidate.casefold() in _CLARIFICATION_METADATA_KEYS:
        return None

    if " - " in candidate:
        candidate = candidate.split(" - ", 1)[0].strip()
    if " – " in candidate:
        candidate = candidate.split(" – ", 1)[0].strip()

    parenthetical_match = re.match(r"^(.*?)(?:\s*\([^)]*\))$", candidate)
    if parenthetical_match:
        base_candidate = parenthetical_match.group(1).strip()
        if 0 < len(base_candidate.split()) <= 6:
            candidate = base_candidate

    if any(char in candidate for char in "{}[]"):
        return None
    if candidate.endswith(("?", ".", ":", ";")):
        return None
    if len(candidate.split()) > 6 or len(candidate) > 60:
        return None
    return candidate


def _extract_jsonish_options(raw_text: str) -> list[str]:
    options: list[str] = []
    for match in re.finditer(r'"options"\s*:\s*\[(?P<values>[^\]]*)\]', raw_text, re.DOTALL):
        values = match.group("values")
        options.extend(
            quoted_match.group(1)
            for quoted_match in re.finditer(r'"([^"\n]{1,60})"', values)
        )
    return options


def _is_quoted_option_source_line(line: str) -> bool:
    normalized_line = _normalize_whitespace(line)
    if not normalized_line:
        return False
    if re.search(r"\bchoose\s+from\b", normalized_line, flags=re.IGNORECASE):
        return True
    if ":" not in normalized_line:
        return False
    label_text = normalized_line.split(":", 1)[0]
    return bool(re.search(r"\b(?:categories|options|values)\b", label_text, flags=re.IGNORECASE))


def _extract_embedded_clarification_json(raw_text: str) -> str | None:
    marker_index = raw_text.find('"response_type"')
    if marker_index == -1:
        return None

    start_index = raw_text.rfind("{", 0, marker_index)
    if start_index == -1:
        return None

    depth = 0
    in_string = False
    is_escaped = False
    for index in range(start_index, len(raw_text)):
        character = raw_text[index]

        if is_escaped:
            is_escaped = False
            continue
        if character == "\\" and in_string:
            is_escaped = True
            continue
        if character == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if character == "{":
            depth += 1
            continue
        if character == "}":
            depth -= 1
            if depth == 0:
                return raw_text[start_index : index + 1]

    return None


def _strip_inline_options_from_user_message(user_message: str, options: list[str]) -> str:
    normalized_message = _normalize_whitespace(user_message)
    if not normalized_message or not options:
        return normalized_message

    stripped_message = re.sub(
        r"\b(?:available\s+)?options\s*:\s*.*$",
        "",
        normalized_message,
        flags=re.IGNORECASE,
    ).strip()
    return stripped_message or normalized_message


def build_clarification_response(
    user_message: str | None,
    options: list[str] | None = None,
) -> dict[str, Any]:
    cleaned_options = [
        option
        for option in (
            _clean_option_text(value)
            for value in (options or [])
        )
        if option is not None
    ]
    normalized_message = _strip_inline_options_from_user_message(
        user_message or "",
        cleaned_options,
    )
    response = {
        "options": _dedupe_preserve_order(cleaned_options)[:10],
    }
    if normalized_message:
        response["user_message"] = normalized_message
    return response


def parse_clarification_response(raw_text: str) -> dict[str, Any] | None:
    candidate = _unwrap_json_code_fence(raw_text)
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return None

    if not isinstance(payload, dict):
        return None

    response_type = payload.get("response_type")
    if response_type not in {None, "clarification"}:
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
            if not isinstance(option, str):
                return None
            options.append(option)

    return build_clarification_response(user_message, options)


def _extract_quoted_options(raw_text: str) -> list[str]:
    jsonish_options = _extract_jsonish_options(raw_text)
    if jsonish_options:
        return jsonish_options

    options: list[str] = []
    for line in raw_text.splitlines():
        if not _is_quoted_option_source_line(line):
            continue
        quoted_values = [match.group(1) for match in re.finditer(r'"([^"\n]{1,60})"', line)]
        if len(quoted_values) >= 2:
            options.extend(quoted_values)
    return options


def _extract_line_options(raw_text: str) -> list[str]:
    options: list[str] = []
    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        match = re.match(r"^(?:[-*•]|\d+[.)])\s*(.+)$", line)
        if match:
            options.append(match.group(1))
            continue

        match = re.match(r"^([A-Za-z][A-Za-z0-9 /&_+-]{0,60})\s*[-–]\s+.+$", line)
        if match:
            options.append(match.group(1))
            continue

        match = re.match(r"^([A-Za-z][A-Za-z0-9 /&_+-]{0,60})\s*\([^)]*\)\s*$", line)
        if match:
            options.append(match.group(1))

    return options


def _extract_user_message(raw_text: str, options: list[str]) -> str | None:
    quoted_message_match = re.search(r'"user_message"\s*:\s*"(?P<message>[^"\n]{1,240})"', raw_text)
    if quoted_message_match:
        return _normalize_whitespace(quoted_message_match.group("message"))

    truncated_message_match = re.search(
        r'(?P<message>[^"\n]{3,240}\?)"\s*,\s*"options"\s*:',
        raw_text,
    )
    if truncated_message_match:
        return _normalize_whitespace(truncated_message_match.group("message"))

    normalized_text = _normalize_whitespace(raw_text)
    if not normalized_text:
        return None

    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", normalized_text)
        if sentence.strip()
    ]
    question_sentences = [sentence for sentence in sentences if sentence.endswith("?")]
    if question_sentences:
        return question_sentences[-1]

    if not options:
        return None

    option_keys = {option.casefold() for option in options}
    fallback_sentences = []
    for sentence in sentences:
        normalized_sentence = _normalize_whitespace(sentence)
        if not normalized_sentence or normalized_sentence.endswith(":"):
            continue

        word_count = len(normalized_sentence.split())
        if word_count < 3 or word_count > 24:
            continue

        cleaned_sentence = _clean_option_text(normalized_sentence)
        if cleaned_sentence is not None and cleaned_sentence.casefold() in option_keys:
            continue

        fallback_sentences.append(normalized_sentence)

    if fallback_sentences:
        return fallback_sentences[-1]
    return None


def _prune_subject_echo_options(user_message: str | None, options: list[str]) -> list[str]:
    if not user_message or len(options) < 2:
        return options

    subject_match = re.search(
        r"\bwhich\s+category\b.*\b(?:for|by)\s+[\"'](?P<subject>[^\"'\n]{1,60})[\"']\s*\??$",
        user_message,
        flags=re.IGNORECASE,
    )
    if subject_match is None:
        return options

    subject = _normalize_whitespace(subject_match.group("subject"))
    if not subject:
        return options

    pruned_options = [
        option
        for option in options
        if option.casefold() != subject.casefold()
    ]
    return pruned_options or options


def normalize_clarification_response(raw_text: str) -> dict[str, Any] | None:
    clarification = parse_clarification_response(raw_text)
    if clarification is not None:
        return clarification

    embedded_clarification_json = _extract_embedded_clarification_json(raw_text)
    if embedded_clarification_json is not None:
        clarification = parse_clarification_response(embedded_clarification_json)
        if clarification is not None:
            return clarification

    jsonish_options = [
        option for option in (_clean_option_text(value) for value in _extract_jsonish_options(raw_text))
        if option is not None
    ]
    if jsonish_options:
        options = _dedupe_preserve_order(jsonish_options)[:10]
    else:
        options = _dedupe_preserve_order(
            [
                option
                for option in [
                    *(_clean_option_text(value) for value in _extract_quoted_options(raw_text)),
                    *(_clean_option_text(value) for value in _extract_line_options(raw_text)),
                ]
                if option is not None
            ]
        )[:10]

    user_message = _extract_user_message(raw_text, options)
    options = _prune_subject_echo_options(user_message, options)
    if user_message is None and not options:
        return None
    return build_clarification_response(user_message, options)


def _augment_clarification_user_message(user_message: str, options: list[Any]) -> str:
    normalized_message = _normalize_whitespace(user_message)
    if not normalized_message or not options:
        return normalized_message
    if _CLARIFICATION_REPLY_GUIDANCE.casefold() in normalized_message.casefold():
        return normalized_message

    suffix = "" if normalized_message.endswith((".", "?", "!")) else "."
    return f"{normalized_message}{suffix} {_CLARIFICATION_REPLY_GUIDANCE}"


def format_clarification_response(clarification: dict[str, Any]) -> str:
    options = clarification.get("options") or []
    user_message = _augment_clarification_user_message(
        str(clarification.get("user_message") or ""),
        options,
    )
    parts: list[str] = []
    if user_message:
        parts.append(user_message)
    if options:
        parts.append("\n".join(f"- {option}" for option in options))
    return "\n\n".join(parts).strip()

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

    output = (
        "Generated SQL:\n"
        f"```sql\n{sql}\n```\n\n"
        "Result:\n"
        f"{result_block}"
    )

    if tool_result.get("public_result_kind") == "detail_count_fallback":
        output += (
            "\n\nNote: Individual row-level data cannot be returned due to privacy guardrails. "
            "Only the number of matching records is shown."
        )

    return output
