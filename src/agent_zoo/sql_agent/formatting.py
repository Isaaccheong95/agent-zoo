"""Format SQL execution results into the agent's final user-facing response.

This module converts structured tool output into the deterministic response
format expected by the SQL agent. It is used by callbacks and is not meant to
be run as a standalone script.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
from typing import Any


_CLARIFICATION_METADATA_KEYS = {
    "clarification_kind",
    "clarification",
    "options",
    "response_type",
    "user_message",
}

CLARIFICATION_KIND_CATEGORICAL_VALUES = "categorical_values"
CLARIFICATION_KIND_GENERIC = "generic"
CLARIFICATION_KIND_INTERPRETATION = "interpretation"

_CLARIFICATION_REPLY_GUIDANCE = "Choose one or more options, or describe your own rule."
_CLARIFICATION_NUMBER_REPLY_GUIDANCE = "You can reply with option numbers like 2 or 2 and 3."
_INTERPRETATION_REPLY_GUIDANCE = (
    "You can reply with one field, multiple fields, no fields, "
    "or describe the field you mean in your own words."
)
_GROUNDED_FILTERS_HEADING = "Already matched from your request:"
_FALLBACK_CLARIFICATION_MESSAGE = (
    "I need clarification before I can run the query. "
    "Please specify the exact category, value, or rule you want me to use."
)
_GROUPED_SUPPRESSION_NOTE = (
    "Note: Some grouped results were omitted due to privacy guardrails."
)
_FILTER_COVERAGE_HEADING = "Cohort filter summary"
_SAFE_AGGREGATE_COLUMN_PATTERNS = (
    "avg",
    "average",
    "min",
    "minimum",
    "max",
    "maximum",
)


@dataclass(slots=True)
class SQLResultViewModel:
    """Deterministic contract for rendering final SQL result responses."""

    status: str
    sql: str
    result_payload: str
    query_summary_section: str
    public_result_kind: str | None
    matched_row_count: int | float | None
    query_summary_context: dict[str, Any] | None
    note: str | None = None


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


def _normalize_clarification_kind(value: Any) -> str | None:
    if not isinstance(value, str):
        return None

    normalized_value = _normalize_whitespace(value).lower()
    if normalized_value in {
        CLARIFICATION_KIND_CATEGORICAL_VALUES,
        CLARIFICATION_KIND_GENERIC,
        CLARIFICATION_KIND_INTERPRETATION,
    }:
        return normalized_value
    return None


def _clean_option_text(value: str, *, clarification_kind: str | None = None) -> str | None:
    candidate = re.sub(r"^\s*(?:[-*•]|\d+\s*[.)-])\s*", "", value).strip()
    candidate = candidate.strip("`\"'")
    candidate = _normalize_whitespace(candidate)
    if not candidate:
        return None
    if candidate.casefold() in _CLARIFICATION_METADATA_KEYS:
        return None

    if clarification_kind == CLARIFICATION_KIND_INTERPRETATION:
        if any(char in candidate for char in "{}[]"):
            return None
        if "`" in candidate or "|" in candidate:
            return None
        if re.search(r"\b(?:select|from|where|group\s+by|order\s+by|having|limit)\b", candidate, flags=re.IGNORECASE):
            return None
        if candidate.endswith(("?", ".", ":", ";")):
            return None
        if len(candidate.split()) > 12 or len(candidate) > 120:
            return None
        return candidate

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
    if "`" in candidate or "|" in candidate or ":" in candidate:
        return None
    if re.search(
        r"\b(?:select|from|where|group\s+by|order\s+by|having|limit|previous\s+query|latest\s+request|current\s+query|user\s+reply)\b",
        candidate,
        flags=re.IGNORECASE,
    ):
        return None
    if re.match(r"^(?:the|if|when|use|current|previous|latest)\b", candidate, flags=re.IGNORECASE):
        return None
    if candidate.endswith(("?", ".", ":", ";")):
        return None
    if len(candidate.split()) > 6 or len(candidate) > 60:
        return None
    return candidate


def _clean_authoritative_option_text(value: str) -> str | None:
    candidate = re.sub(r"^\s*(?:[-*•]|\d+\s*[.)-])\s*", "", value).strip()
    candidate = candidate.strip("`\"'")
    candidate = _normalize_whitespace(candidate)
    if not candidate:
        return None
    if candidate.casefold() in _CLARIFICATION_METADATA_KEYS:
        return None
    if any(char in candidate for char in "{}[]"):
        return None
    if len(candidate) > 160:
        return None
    return candidate


def _looks_like_interpretation_clarification(
    user_message: str | None,
    options: list[str] | None = None,
    raw_text: str | None = None,
) -> bool:
    normalized_message = _normalize_whitespace(user_message or "")
    if normalized_message and re.search(
        r"\bwhich\s+(?:column|field|interpretation|meaning|concept|measure|dimension)\b",
        normalized_message,
        flags=re.IGNORECASE,
    ):
        return True
    if normalized_message and re.search(r"\bgroup(?:\s+the\s+results)?\s+by\b", normalized_message, flags=re.IGNORECASE):
        return True
    if normalized_message and "broader" in normalized_message.casefold() and "specific" in normalized_message.casefold():
        return True

    raw_candidates = [
        value
        for value in (options or [])
        if isinstance(value, str) and value.strip()
    ]
    if any(re.search(r"\([A-Za-z_][A-Za-z0-9_]*\)", value) for value in raw_candidates):
        return True

    looks_like_value_prompt = bool(
        normalized_message
        and re.search(
            r"\bwhich\s+(?:category|value|option)\s+do\s+you\s+mean\b",
            normalized_message,
            flags=re.IGNORECASE,
        )
    )
    if _options_share_interpretation_stem(raw_candidates) and normalized_message and re.search(
        r"\b(?:do\s+you\s+mean|which\s+one\s+do\s+you\s+mean|are\s+you\s+asking)\b",
        normalized_message,
        flags=re.IGNORECASE,
    ) and not looks_like_value_prompt:
        return True

    normalized_raw_text = _normalize_whitespace(raw_text or "")
    if normalized_raw_text and re.search(r"\bgroup(?:\s+the\s+results)?\s+by\b", normalized_raw_text, flags=re.IGNORECASE):
        return True
    if normalized_raw_text and "broader" in normalized_raw_text.casefold() and "specific" in normalized_raw_text.casefold():
        return True

    return False


def _extract_option_stem_tokens(value: str) -> set[str]:
    normalized_value = _normalize_whitespace(value)
    if not normalized_value:
        return set()

    normalized_value = re.sub(r"\([A-Za-z_][A-Za-z0-9_]*\)", " ", normalized_value)
    return {
        token.casefold()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_]+", normalized_value)
        if len(token) >= 4
    }


def _options_share_interpretation_stem(options: list[str]) -> bool:
    token_counts: dict[str, int] = {}
    for option in options:
        for token in _extract_option_stem_tokens(option):
            token_counts[token] = token_counts.get(token, 0) + 1
    return any(count >= 2 for count in token_counts.values())


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
    *,
    clarification_kind: str | None = None,
    preserve_option_text: bool = False,
) -> dict[str, Any]:
    normalized_kind = _normalize_clarification_kind(clarification_kind)
    option_cleaner = (
        _clean_authoritative_option_text
        if preserve_option_text
        else lambda value: _clean_option_text(value, clarification_kind=normalized_kind)
    )
    cleaned_options = [
        option
        for option in (
            option_cleaner(value)
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
    if normalized_kind:
        response["clarification_kind"] = normalized_kind
    return response


def build_fallback_clarification_response() -> dict[str, Any]:
    return build_clarification_response(
        _FALLBACK_CLARIFICATION_MESSAGE,
        clarification_kind=CLARIFICATION_KIND_GENERIC,
    )


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

    inferred_kind = _normalize_clarification_kind(payload.get("clarification_kind"))
    if inferred_kind is None and _looks_like_interpretation_clarification(user_message, options, candidate):
        inferred_kind = CLARIFICATION_KIND_INTERPRETATION

    return build_clarification_response(
        user_message,
        options,
        clarification_kind=inferred_kind,
    )


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

        match = re.match(r"^(?:[-*•]|\d+\s*[.)-])\s*(.+)$", line)
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


def looks_like_clarification_attempt(raw_text: str) -> bool:
    normalized_text = _normalize_whitespace(raw_text)
    if not normalized_text:
        return False

    if re.search(r"\bclarif(?:y|ication)\b", normalized_text, flags=re.IGNORECASE):
        return True
    if re.search(r"\bplease\s+specify\b", normalized_text, flags=re.IGNORECASE):
        return True
    if re.search(r"\bwhat\s+do\s+you\s+mean\b", normalized_text, flags=re.IGNORECASE):
        return True
    if re.search(
        r"\bwhich\s+(?:category|categories|value|values|option|options|column|columns|group|groups)\b",
        normalized_text,
        flags=re.IGNORECASE,
    ):
        return True
    if re.search(r"\bchoose\s+(?:from|one|one\s+or\s+more)\b", normalized_text, flags=re.IGNORECASE):
        return True
    if "?" in normalized_text and re.search(
        r"\b(?:which|what|specify|choose|mean|should\s+i\s+use)\b",
        normalized_text,
        flags=re.IGNORECASE,
    ):
        return True
    if re.search(r"^(?:\s*(?:[-*•]|\d+\s*[.)-])\s*.+)$", raw_text, flags=re.MULTILINE) and re.search(
        r"\b(?:category|categories|option|options|value|values)\b",
        normalized_text,
        flags=re.IGNORECASE,
    ):
        return True
    return False


def clarification_requires_deterministic_fallback(clarification: dict[str, Any]) -> bool:
    user_message = _normalize_whitespace(str(clarification.get("user_message") or ""))
    if not user_message:
        return True

    if re.match(
        r"^(?:available\s+)?(?:categories|options|values)\s*:",
        user_message,
        flags=re.IGNORECASE,
    ):
        return True
    return False


def normalize_clarification_response(raw_text: str) -> dict[str, Any] | None:
    clarification = parse_clarification_response(raw_text)
    if clarification is not None:
        return clarification

    embedded_clarification_json = _extract_embedded_clarification_json(raw_text)
    if embedded_clarification_json is not None:
        clarification = parse_clarification_response(embedded_clarification_json)
        if clarification is not None:
            return clarification

    jsonish_option_candidates = _extract_jsonish_options(raw_text)
    jsonish_options = [
        option for option in (_clean_option_text(value) for value in jsonish_option_candidates)
        if option is not None
    ]
    if jsonish_options:
        options = _dedupe_preserve_order(jsonish_options)[:10]
        raw_option_candidates = jsonish_option_candidates
    else:
        raw_option_candidates = [
            *_extract_quoted_options(raw_text),
            *_extract_line_options(raw_text),
        ]
        options = _dedupe_preserve_order(
            [
                option
                for option in [
                    *(_clean_option_text(value) for value in raw_option_candidates),
                ]
                if option is not None
            ]
        )[:10]

    user_message = _extract_user_message(raw_text, options)
    options = _prune_subject_echo_options(user_message, options)
    if user_message is None and not options:
        return None

    inferred_kind = (
        CLARIFICATION_KIND_INTERPRETATION
        if _looks_like_interpretation_clarification(user_message, raw_option_candidates, raw_text)
        else None
    )
    return build_clarification_response(
        user_message,
        raw_option_candidates if inferred_kind == CLARIFICATION_KIND_INTERPRETATION else options,
        clarification_kind=inferred_kind,
    )


def _augment_clarification_user_message(
    user_message: str,
    options: list[Any],
    *,
    clarification_kind: str | None = None,
) -> str:
    normalized_message = _normalize_whitespace(user_message)
    if not normalized_message or not options:
        return normalized_message

    additions: list[str] = []
    normalized_casefold = normalized_message.casefold()
    if clarification_kind == CLARIFICATION_KIND_INTERPRETATION:
        if _INTERPRETATION_REPLY_GUIDANCE.casefold() not in normalized_casefold:
            additions.append(_INTERPRETATION_REPLY_GUIDANCE)
    else:
        if _CLARIFICATION_REPLY_GUIDANCE.casefold() not in normalized_casefold:
            additions.append(_CLARIFICATION_REPLY_GUIDANCE)
        if _CLARIFICATION_NUMBER_REPLY_GUIDANCE.casefold() not in normalized_casefold:
            additions.append(_CLARIFICATION_NUMBER_REPLY_GUIDANCE)
    if not additions:
        return normalized_message

    suffix = "" if normalized_message.endswith((".", "?", "!")) else "."
    return f"{normalized_message}{suffix} {' '.join(additions)}"


def _build_grounded_filter_lines(clarification: dict[str, Any]) -> list[str]:
    grounded_filters = clarification.get("grounded_filters")
    if not isinstance(grounded_filters, dict):
        return []

    lines: list[str] = []
    for column_name, selected_values in grounded_filters.items():
        normalized_column_name = str(column_name or "").strip()
        normalized_values = [
            value.strip()
            for value in selected_values or []
            if isinstance(value, str) and value.strip()
        ]
        if not normalized_column_name or not normalized_values:
            continue
        if len(normalized_values) == 1:
            lines.append(f"{normalized_column_name} = {normalized_values[0]}")
            continue
        lines.append(f"{normalized_column_name} in {', '.join(normalized_values)}")
    return lines


def format_clarification_response(clarification: dict[str, Any]) -> str:
    options = clarification.get("options") or []
    clarification_kind = _normalize_clarification_kind(clarification.get("clarification_kind"))
    user_message = _augment_clarification_user_message(
        str(clarification.get("user_message") or ""),
        options,
        clarification_kind=clarification_kind,
    )
    parts: list[str] = []
    if user_message:
        parts.append(user_message)
    if options:
        parts.append("\n".join(f"{index}. {option}" for index, option in enumerate(options, start=1)))
    grounded_filter_lines = _build_grounded_filter_lines(clarification)
    if grounded_filter_lines:
        parts.append(
            _GROUNDED_FILTERS_HEADING + "\n" + "\n".join(f"- {line}" for line in grounded_filter_lines)
        )
    return "\n\n".join(parts).strip()


def _is_numeric_display_value(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _normalize_result_column_name(value: str) -> str:
    return re.sub(r"\s+", "_", str(value).strip().lower())


def _is_count_result_column(column_name: str) -> bool:
    return "count" in _normalize_result_column_name(column_name)


def _is_safe_aggregate_result_column(column_name: str) -> bool:
    normalized = _normalize_result_column_name(column_name)
    if _is_count_result_column(normalized):
        return False
    return any(pattern in normalized for pattern in _SAFE_AGGREGATE_COLUMN_PATTERNS)


def _get_display_aggregate_columns(tool_result: dict[str, Any], rows: list[Any]) -> set[str]:
    aggregate_columns = {
        str(column).strip()
        for column in (tool_result.get("aggregate_columns") or [])
        if isinstance(column, str) and str(column).strip()
    }
    if aggregate_columns:
        return aggregate_columns

    columns = [
        str(column).strip()
        for column in (tool_result.get("columns") or [])
        if isinstance(column, str) and str(column).strip()
    ]
    if not columns or not rows:
        return set()

    detected_columns: set[str] = set()
    for column in columns:
        if not _is_safe_aggregate_result_column(column):
            continue
        if all(isinstance(row, dict) and _is_numeric_display_value(row.get(column)) for row in rows):
            detected_columns.add(column)
    return detected_columns


def _format_fixed_decimal(value: int | float) -> str:
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        return str(value)
    return f"{numeric_value:.2f}"


def _build_display_rows(tool_result: dict[str, Any], rows: list[Any]) -> list[Any]:
    aggregate_columns = _get_display_aggregate_columns(tool_result, rows)
    if not aggregate_columns:
        return rows

    display_rows: list[Any] = []
    for row in rows:
        if not isinstance(row, dict):
            return rows

        display_row: dict[str, Any] = {}
        for column, value in row.items():
            if column in aggregate_columns and _is_numeric_display_value(value):
                display_row[column] = _format_fixed_decimal(value)
                continue
            display_row[column] = value
        display_rows.append(display_row)
    return display_rows

def format_result_payload(tool_result: dict) -> str:
    if tool_result["status"] != "success":
        return tool_result.get("error") or "Execution failed."

    rows = tool_result.get("rows", [])
    if not rows:
        return "No rows returned."

    display_rows = _build_display_rows(tool_result, rows)

    if len(display_rows) == 1 and isinstance(display_rows[0], dict) and len(display_rows[0]) == 1:
        return str(next(iter(display_rows[0].values())))

    return json.dumps(display_rows, indent=2)


def _get_matching_row_count(tool_result: dict) -> int | float | None:
    matched_row_count = tool_result.get("matched_row_count")
    if isinstance(matched_row_count, (int, float)) and not isinstance(matched_row_count, bool):
        return matched_row_count
    return None


def _build_query_action_summary(tool_result: dict) -> str:
    if tool_result.get("status") != "success":
        return "Attempted a read-only query against the current filtered dataset."

    public_result_kind = tool_result.get("public_result_kind")
    grouped_result_suppressed = bool(tool_result.get("grouped_result_suppressed"))
    if public_result_kind == "detail_count_fallback":
        return "Matched rows in the current filtered dataset and returned only the safe count."
    if grouped_result_suppressed and public_result_kind == "safe_aggregate":
        return "Computed grouped aggregate values for the privacy-safe matched groups."
    if grouped_result_suppressed and public_result_kind == "count_aggregate":
        return "Counted rows for the privacy-safe matched groups in the current filtered dataset."
    if public_result_kind == "safe_aggregate":
        if tool_result.get("row_count", 0) > 1:
            return "Computed grouped aggregate values for the matched cohort."
        return "Computed aggregate values for the matched cohort."
    if public_result_kind == "count_aggregate":
        return "Counted matching rows in the current filtered dataset."

    rows = tool_result.get("rows") or []
    if len(rows) == 1 and isinstance(rows[0], dict) and len(rows[0]) == 1:
        return "Computed a scalar result from the current filtered dataset."
    return "Selected matching rows from the current filtered dataset."


def _format_row_count_label(value: int | float) -> str:
    normalized_value = int(value) if isinstance(value, float) and value.is_integer() else value
    return f"{normalized_value} row" if normalized_value == 1 else f"{normalized_value} rows"


def _build_categorical_filter_bullet(entry: dict[str, Any]) -> str | None:
    column_name = str(entry.get("column") or "").strip()
    selected_values = [
        value.strip()
        for value in entry.get("selected_values") or []
        if isinstance(value, str) and value.strip()
    ]
    available_values = [
        value.strip()
        for value in entry.get("available_values") or []
        if isinstance(value, str) and value.strip()
    ]
    if not column_name or not selected_values:
        return None

    bullet_lines = [
        f"{column_name}:",
        f"  Matched categories: {', '.join(selected_values)}",
    ]

    normalized_selected_values = {value.casefold() for value in selected_values}
    normalized_available_values = {value.casefold() for value in available_values}
    if available_values and normalized_available_values:
        if normalized_selected_values < normalized_available_values:
            other_values = [
                value for value in available_values if value.casefold() not in normalized_selected_values
            ]
            if other_values:
                bullet_lines.append(f"  other stored values: {', '.join(other_values)}")
        elif normalized_selected_values == normalized_available_values:
            bullet_lines.append("  coverage: all non-null stored values for this column")

    missing_or_blank_rows_excluded = entry.get("missing_or_blank_rows_excluded")
    if (
        isinstance(missing_or_blank_rows_excluded, (int, float))
        and not isinstance(missing_or_blank_rows_excluded, bool)
        and missing_or_blank_rows_excluded > 0
    ):
        bullet_lines.append(
            "  no. of missing / blank rows excluded: "
            + _format_row_count_label(missing_or_blank_rows_excluded)
        )

    return "\n".join(bullet_lines)


def _build_query_filter_bullets(tool_result: dict) -> list[str]:
    query_summary_context = tool_result.get("query_summary_context")
    if not isinstance(query_summary_context, dict):
        return []

    bullets: list[str] = []
    categorical_filters = [
        entry
        for entry in (query_summary_context.get("categorical_filters") or [])
        if isinstance(entry, dict)
    ]
    for entry in categorical_filters:
        bullet = _build_categorical_filter_bullet(entry)
        if bullet:
            bullets.append(bullet)

    comparison_filters = [
        entry
        for entry in (query_summary_context.get("comparison_filters") or [])
        if isinstance(entry, dict)
    ]
    for entry in comparison_filters:
        column_name = str(entry.get("column") or "").strip()
        operator = str(entry.get("operator") or "").strip()
        value = str(entry.get("value") or "").strip()
        if not column_name or not operator or not value:
            continue
        bullets.append(f"{column_name} {operator} {value}")

    return bullets


def _format_query_summary_section(tool_result: dict) -> str:
    bullets = _build_query_filter_bullets(tool_result)
    if not bullets:
        bullets = [_build_query_action_summary(tool_result)]

    return _FILTER_COVERAGE_HEADING + ":\n" + "\n".join(f"- {bullet}" for bullet in bullets)


def build_sql_result_view_model(tool_result: dict[str, Any]) -> SQLResultViewModel:
    display_sql = tool_result.get("display_sql")
    if not isinstance(display_sql, str) or not display_sql.strip():
        display_sql = str(tool_result.get("sql") or "Not executed")
    result_payload = format_result_payload(tool_result)
    query_summary_section = _format_query_summary_section(tool_result)
    public_result_kind = tool_result.get("public_result_kind")
    if not isinstance(public_result_kind, str):
        public_result_kind = None

    note = None
    if public_result_kind == "detail_count_fallback":
        note = (
            "Note: Individual row-level data cannot be returned due to privacy guardrails. "
            "Only the number of matching records is shown."
        )
    elif tool_result.get("grouped_result_suppressed"):
        note = _GROUPED_SUPPRESSION_NOTE

    return SQLResultViewModel(
        status=str(tool_result.get("status") or "error"),
        sql=display_sql,
        result_payload=result_payload,
        query_summary_section=query_summary_section,
        public_result_kind=public_result_kind,
        matched_row_count=_get_matching_row_count(tool_result),
        query_summary_context=(
            tool_result.get("query_summary_context")
            if isinstance(tool_result.get("query_summary_context"), dict)
            else None
        ),
        note=note,
    )


def render_sql_result_view_model(view_model: SQLResultViewModel) -> str:
    code_block = (
        "json"
        if view_model.result_payload.startswith("[") or view_model.result_payload.startswith("{")
        else ""
    )
    result_block = f"```{code_block}\n{view_model.result_payload}\n```".strip()

    output = "\n\n".join(
        [
            "Generated SQL:\n" f"```sql\n{view_model.sql}\n```",
            view_model.query_summary_section,
            "Result:\n" f"{result_block}",
        ]
    )
    if view_model.note:
        output += f"\n\n{view_model.note}"
    return output


def format_public_query_result(tool_result: dict[str, Any]) -> str:
    """Format a callback-owned public SQL result using the stable view model."""

    return render_sql_result_view_model(build_sql_result_view_model(tool_result))
