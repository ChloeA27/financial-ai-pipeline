"""
Read-only access to a pipeline database for the dashboard (``src/dashboard.py``).

Plain Python (no Streamlit), so it is unit-tested. Every call opens its own
connection: Streamlit reruns a page on a new thread, and a sqlite3 connection
may only be used by the thread that created it. ``mode=ro`` guarantees the
dashboard never writes to the database.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


def read_rows(db_path: Path, sql: str) -> tuple[list[str], list[tuple[Any, ...]]]:
    """(column names, rows) of one read-only query on a fresh connection."""
    conn = sqlite3.connect(f"file:{Path(db_path)}?mode=ro", uri=True)
    try:
        cur = conn.execute(sql)
        return [d[0] for d in cur.description or []], cur.fetchall()
    finally:
        conn.close()
