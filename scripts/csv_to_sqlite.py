from __future__ import annotations

"""Convert a CSV dataset into a simple SQLite database.

This script is intentionally lightweight and uses only the Python standard
library. It is aimed at small to medium CSV files where loading the full file
into memory is acceptable.

The conversion flow is:
1. Read the CSV headers and rows.
2. Sanitize the headers into SQLite-friendly identifiers.
3. Infer a coarse SQLite type for each column.
4. Create or replace the destination table.
5. Insert the CSV rows into SQLite.
6. Store a `__column_mapping` table so the original CSV headers can still be
   traced back after sanitization.

The type inference is intentionally conservative. Columns are inferred as
`INTEGER`, `REAL`, or `TEXT`; values such as dates and booleans are not given
special handling.
"""

import argparse
import csv
import re
import sqlite3
from pathlib import Path


DEFAULT_INPUT = Path("dataset/titantic/Titanic-Dataset.csv")
DEFAULT_OUTPUT = Path("dataset/titantic/titanic.sqlite")
DEFAULT_TABLE = "titanic_passengers"


def sanitize_identifier(name: str) -> str:
    """Convert an arbitrary string into a SQLite-friendly identifier.

    Rules:
    - replace runs of non-word characters with underscores
    - trim leading and trailing underscores
    - lowercase the result
    - fall back to `column` if nothing remains
    - prefix with `col_` if the identifier starts with a digit

    Args:
        name: A raw CSV header or user-provided table name.

    Returns:
        A normalized identifier suitable for use in the generated SQLite schema.
    """

    sanitized = re.sub(r"\W+", "_", name.strip()).strip("_").lower()
    if not sanitized:
        sanitized = "column"
    if sanitized[0].isdigit():
        sanitized = f"col_{sanitized}"
    return sanitized


def infer_sqlite_type(values: list[str]) -> str:
    """Infer a SQLite type for a single CSV column.

    Empty values are ignored during inference. If every non-empty value looks
    like an integer, the column becomes `INTEGER`. If every non-empty value is
    numeric but not necessarily integral, the column becomes `REAL`. Anything
    else falls back to `TEXT`.

    Args:
        values: All observed values for a single CSV column.

    Returns:
        One of `INTEGER`, `REAL`, or `TEXT`.
    """

    non_empty = [value.strip() for value in values if value is not None and value.strip() != ""]
    if not non_empty:
        return "TEXT"

    if all(re.fullmatch(r"[-+]?\d+", value) for value in non_empty):
        return "INTEGER"

    if all(re.fullmatch(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)", value) for value in non_empty):
        return "REAL"

    return "TEXT"


def convert_value(value: str, sqlite_type: str) -> int | float | str | None:
    """Convert a raw CSV cell into a SQLite-ready Python value.

    Empty strings are normalized to `None` so SQLite stores them as `NULL`.
    Numeric conversion is driven by the inferred destination type for the
    column.

    Args:
        value: The raw CSV cell value.
        sqlite_type: The inferred SQLite type for the destination column.

    Returns:
        An `int`, `float`, `str`, or `None` depending on the input and type.
    """

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
    """Load the CSV file into memory.

    The file is opened using `utf-8-sig` so UTF-8 byte-order marks are handled
    gracefully. Rows are returned through `csv.DictReader`, which means each row
    is keyed by the original CSV header names.

    Args:
        csv_path: Path to the input CSV file.

    Returns:
        A tuple containing:
        - the original CSV headers in order
        - a list of row dictionaries keyed by those headers

    Raises:
        ValueError: If the CSV does not contain a header row.
    """

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
    """Create the destination table and refresh header mapping metadata.

    This function replaces any existing table with the same name so the output
    database always reflects the current CSV contents.

    It also maintains a helper table called `__column_mapping`. That metadata
    table is useful because CSV headers may contain spaces or punctuation, while
    the generated SQLite columns are sanitized for easier querying.

    Args:
        connection: Open SQLite connection.
        table_name: Destination SQLite table name.
        original_headers: Raw headers exactly as they appeared in the CSV file.
        column_names: Sanitized SQLite column names.
        column_types: Inferred SQLite types aligned to `column_names`.
    """

    quoted_table = f'"{table_name}"'
    column_defs = [
        f'"{column_name}" {column_type}'
        for column_name, column_type in zip(column_names, column_types, strict=True)
    ]

    connection.execute(f"DROP TABLE IF EXISTS {quoted_table}")
    connection.execute(f"CREATE TABLE {quoted_table} ({', '.join(column_defs)})")

    # Keep a lookup table that maps the original CSV headers to the sanitized
    # SQLite column names. This makes the generated schema easier to understand
    # later when users compare SQL columns back to the original CSV file.
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
    """Insert all CSV rows into the destination SQLite table.

    Rows are converted column-by-column according to the inferred SQLite type.
    The function builds a parameterized `INSERT` statement to avoid manual value
    interpolation and to let `sqlite3` handle escaping safely.

    Args:
        connection: Open SQLite connection.
        table_name: Destination SQLite table name.
        headers: Original CSV headers in order.
        column_names: Sanitized SQLite column names aligned with `headers`.
        column_types: Inferred SQLite types aligned with `headers`.
        rows: CSV rows returned by `load_csv`.
    """

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
    """Convert a CSV file into a SQLite database file.

    This is the high-level orchestration function used by the CLI entrypoint.
    It loads the CSV, derives sanitized column names, infers column types,
    creates the destination table, inserts rows, and commits the transaction.

    Args:
        csv_path: Path to the source CSV file.
        sqlite_path: Path to the destination SQLite database file.
        table_name: Destination SQLite table name. This should already be
            sanitized before being passed in.
    """

    headers, rows = load_csv(csv_path)

    # Column names are derived from the CSV headers, while types are inferred by
    # scanning the full set of observed values for each original header.
    column_names = [sanitize_identifier(header) for header in headers]
    column_types = [infer_sqlite_type([row.get(header, "") for row in rows]) for header in headers]

    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(sqlite_path) as connection:
        create_table(connection, table_name, headers, column_names, column_types)
        insert_rows(connection, table_name, headers, column_names, column_types, rows)
        connection.commit()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the conversion script.

    Returns:
        Parsed `argparse.Namespace` containing the input path, output path, and
        destination table name.
    """

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
    """Run the CLI conversion flow and print a short success summary."""

    args = parse_args()
    table_name = sanitize_identifier(args.table)
    convert_csv_to_sqlite(args.input, args.output, table_name)
    print(f"Created {args.output} with table '{table_name}' from {args.input}")


if __name__ == "__main__":
    main()
