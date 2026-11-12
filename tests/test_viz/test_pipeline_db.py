"""The dashboard's DB reads work from any thread and never write."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest

from src.viz.pipeline_db import read_rows


@pytest.fixture
def db(tmp_path: Path) -> Path:
    path = tmp_path / "p.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE documents (source_hash TEXT, last_status TEXT)")
        conn.execute("INSERT INTO documents VALUES ('h', 'stored')")
    return path


def test_reads_from_a_thread_other_than_the_first_caller(db: Path) -> None:
    """Streamlit reruns a page on a new thread (the 2026-11-12 ProgrammingError)."""
    assert read_rows(db, "SELECT last_status FROM documents")[1] == [("stored",)]
    out: list = []
    t = threading.Thread(
        target=lambda: out.append(read_rows(db, "SELECT count(*) FROM documents"))
    )
    t.start()
    t.join()
    assert out == [(["count(*)"], [(1,)])]


def test_the_old_pattern_fails_across_threads(db: Path) -> None:
    """Why read_rows opens its own connection: a shared one breaks on another thread."""
    shared = sqlite3.connect(db)
    errors: list = []

    def use() -> None:
        try:
            shared.execute("SELECT 1")
        except sqlite3.ProgrammingError as exc:
            errors.append(exc)

    t = threading.Thread(target=use)
    t.start()
    t.join()
    shared.close()
    assert errors


def test_connection_is_read_only(db: Path) -> None:
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        read_rows(db, "DELETE FROM documents")
