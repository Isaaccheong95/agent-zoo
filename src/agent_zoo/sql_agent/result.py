"""Structured SQL-agent result objects for programmatic consumers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..tabular import TabularPayload


def _extract_error(result: Mapping[str, Any] | None) -> str | None:
    if result is None:
        return None
    error = result.get("error")
    if error in (None, ""):
        return None
    return str(error)


def _build_tabular_payload(
    result: Mapping[str, Any] | None,
    *,
    question: str,
    schema_text: str | None,
) -> TabularPayload | None:
    if result is None or result.get("status") != "success":
        return None
    return TabularPayload.from_sql_result(result, question=question, schema_text=schema_text)


@dataclass(slots=True)
class SQLAgentStructuredResult:
    question: str
    final_response: str
    public_data: TabularPayload | None = None
    internal_data: TabularPayload | None = None
    public_error: str | None = None
    internal_error: str | None = None
    public_result_kind: str | None = None

    def get_tabular_payload(self, *, prefer_internal: bool = False) -> TabularPayload | None:
        if prefer_internal and self.internal_data is not None:
            return self.internal_data
        return self.public_data


def build_structured_result(
    *,
    question: str,
    final_response: str,
    public_result: Mapping[str, Any] | None,
    internal_result: Mapping[str, Any] | None,
    schema_text: str | None = None,
) -> SQLAgentStructuredResult:
    return SQLAgentStructuredResult(
        question=question,
        final_response=final_response,
        public_data=_build_tabular_payload(public_result, question=question, schema_text=schema_text),
        internal_data=_build_tabular_payload(internal_result, question=question, schema_text=schema_text),
        public_error=_extract_error(public_result),
        internal_error=_extract_error(internal_result),
        public_result_kind=(
            str(public_result.get("public_result_kind"))
            if public_result is not None and public_result.get("public_result_kind") not in (None, "")
            else None
        ),
    )