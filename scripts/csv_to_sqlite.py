from __future__ import annotations

import argparse
import csv
import re
import sqlite3
from pathlib import Path


DEFAULT_INPUT = Path("dataset/titantic/Titanic-Dataset.csv")
DEFAULT_OUTPUT = Path("dataset/titantic/titanic.sqlite")
DEFAULT_TABLE = "titanic_passengers"


def sanitize_identifier(name: str) -> str:
    sanitized = re.sub(r"\W+", "_", name.strip()).strip("_").lower()
    if not sanitized:
        sanitized = "column"
    if sanitized[0].isdigit():
        sanitized = f"col_{sanitized}"
    return sanitized


def infer_sqlite_type(values: list[str]) -> str:
    non_empty = [value.strip() for value in values if value is not None and value.strip() != ""]
    if not non_empty:
        return "TEXT"

    if all(re.fullmatch(r"[-+]?\d+", value) for value in non_empty):
        return "INTEGER"

    if all(re.fullmatch(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)", value) for value in non_empty):
        return "REAL"

    return "TEXT"


def convert_value(value: str, sqlite_type: str) -> int | float | str | None:
    if value is None:
        return None

    value = value.strip()
    if value == "":
        return None

    if sqlite_type == "INTEGER":
        return int(value)
    if sqlite_type == "REAL":
        return float(value)
    return value


def load_csv(csv_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"No headers found in {csv_path}")
        rows = list(reader)
        return reader.fieldnames, rows


def create_table(
    connection: sqlite3.Connection,
    table_name: str,
    original_headers: list[str],
    column_names: list[str],
    column_types: list[str],
) -> None:
    quoted_table = f'"{table_name}"'
    column_defs = [
        f'"{column_name}" {column_type}'
        for column_name, column_type in zip(column_names, column_types, strict=True)
    ]
    connection.execute(f"DROP TABLE IF EXISTS {quoted_table}")
    connection.execute(f"CREATE TABLE {quoted_table} ({', '.join(column_defs)})")

    # Preserve the original CSV headers for quick reference.
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS __column_mapping (
            table_name TEXT NOT NULL,
            csv_header TEXT NOT NULL,
            sqlite_column TEXT NOT NULL
        )
        """
    )
    connection.execute("DELETE FROM __column_mapping WHERE table_name = ?", (table_name,))
    connection.executemany(
        "INSERT INTO __column_mapping (table_name, csv_header, sqlite_column) VALUES (?, ?, ?)",
        [(table_name, header, column) for header, column in zip(original_headers, column_names, strict=True)],
    )


def insert_rows(
    connection: sqlite3.Connection,
    table_name: str,
    headers: list[str],
    column_names: list[str],
    column_types: list[str],
    rows: list[dict[str, str]],
) -> None:
    placeholders = ", ".join("?" for _ in column_names)
    columns_sql = ", ".join(f'"{column_name}"' for column_name in column_names)
    insert_sql = f'INSERT INTO "{table_name}" ({columns_sql}) VALUES ({placeholders})'

    records = []
    for row in rows:
        record = tuple(
            convert_value(row.get(header, ""), column_type)
            for header, column_type in zip(headers, column_types, strict=True)
        )
        records.append(record)

    connection.executemany(insert_sql, records)


def convert_csv_to_sqlite(csv_path: Path, sqlite_path: Path, table_name: str) -> None:
    headers, rows = load_csv(csv_path)
    column_names = [sanitize_identifier(header) for header in headers]
    column_types = [infer_sqlite_type([row.get(header, "") for row in rows]) for header in headers]

    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(sqlite_path) as connection:
        create_table(connection, table_name, headers, column_names, column_types)
        insert_rows(connection, table_name, headers, column_names, column_types, rows)
        connection.commit()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a CSV file into a SQLite database.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Path to the source CSV file. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Path to the destination SQLite file. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--table",
        default=DEFAULT_TABLE,
        help=f"SQLite table name. Default: {DEFAULT_TABLE}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_csv_to_sqlite(args.input, args.output, sanitize_identifier(args.table))
    print(f"Created {args.output} with table '{sanitize_identifier(args.table)}' from {args.input}")


if __name__ == "__main__":
    main()
