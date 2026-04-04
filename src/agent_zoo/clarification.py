"""Shared clarification response helpers for reusable agents."""

from __future__ import annotations

import json
import re
from typing import Any


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


def build_clarification_response(
    user_message: str | None,
    options: list[str] | None = None,
) -> dict[str, Any]:
    normalized_message = _normalize_whitespace(user_message or "")
    cleaned_options = [
        option
        for option in (_clean_option_text(value) for value in (options or []))
        if option is not None
    ]
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
    options: list[str] = []
    for line in raw_text.splitlines():
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


def normalize_clarification_response(raw_text: str) -> dict[str, Any] | None:
    clarification = parse_clarification_response(raw_text)
    if clarification is not None:
        return clarification

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
    if user_message is None and not options:
        return None
    return build_clarification_response(user_message, options)


def format_clarification_response(clarification: dict[str, Any]) -> str:
    user_message = _normalize_whitespace(str(clarification.get("user_message") or ""))
    options = clarification.get("options") or []
    parts: list[str] = []
    if user_message:
        parts.append(user_message)
    if options:
        parts.append("\n".join(f"- {option}" for option in options))
    return "\n\n".join(parts).strip()