"""Shared tabular payloads for passing structured results between agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


def _copy_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows]


def _infer_columns(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            normalized_key = str(key)
            if normalized_key in seen:
                continue
            seen.add(normalized_key)
            columns.append(normalized_key)
    return columns


@dataclass(slots=True)
class TabularPayload:
    columns: list[str]
    rows: list[dict[str, Any]]
    row_count: int
    preview_row_count: int
    truncated: bool = False
    question: str | None = None
    sql: str | None = None
    schema_text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_rows(
        cls,
        rows: Sequence[Mapping[str, Any]],
        *,
        columns: Sequence[str] | None = None,
        row_count: int | None = None,
        preview_row_count: int | None = None,
        truncated: bool = False,
        question: str | None = None,
        sql: str | None = None,
        schema_text: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> TabularPayload:
        normalized_rows = _copy_rows(rows)
        normalized_columns = [str(column) for column in columns] if columns is not None else _infer_columns(normalized_rows)
        resolved_preview_count = len(normalized_rows) if preview_row_count is None else max(0, int(preview_row_count))
        resolved_row_count = len(normalized_rows) if row_count is None else max(0, int(row_count))

        return cls(
            columns=normalized_columns,
            rows=normalized_rows,
            row_count=resolved_row_count,
            preview_row_count=resolved_preview_count,
            truncated=bool(truncated),
            question=question,
            sql=sql,
            schema_text=schema_text,
            metadata=dict(metadata or {}),
        )

    @classmethod
    def from_sql_result(
        cls,
        result: Mapping[str, Any],
        *,
        question: str | None = None,
        schema_text: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> TabularPayload:
        if result.get("status") != "success":
            raise ValueError(result.get("error") or "SQL result is not successful.")

        extra_metadata = {
            key: value
            for key, value in result.items()
            if key not in {
                "status",
                "db_path",
                "sql",
                "columns",
                "rows",
                "row_count",
                "preview_row_count",
                "truncated",
                "error",
            }
        }
        if metadata:
            extra_metadata.update(dict(metadata))

        return cls.from_rows(
            result.get("rows") or [],
            columns=result.get("columns"),
            row_count=result.get("row_count"),
            preview_row_count=result.get("preview_row_count"),
            truncated=bool(result.get("truncated", False)),
            question=question,
            sql=result.get("sql"),
            schema_text=schema_text,
            metadata=extra_metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "columns": list(self.columns),
            "rows": [dict(row) for row in self.rows],
            "row_count": self.row_count,
            "preview_row_count": self.preview_row_count,
            "truncated": self.truncated,
            "question": self.question,
            "sql": self.sql,
            "schema_text": self.schema_text,
            "metadata": dict(self.metadata),
        }