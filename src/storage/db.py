"""
Database layer — aiosqlite connection management & schema initialisation.

Creates the tables + indices on startup via ``init_db()``.  They map onto a
medallion layout (bronze → silver → gold):

  bronze  documents           — registry of every source text seen; the text
                                itself is a write-once file in the raw store
                                (``src/storage/raw_store.py``), keyed by SHA-256.
  silver  step_runs           — one row per node execution (every attempt, every
                                outcome): raw LLM reply, parsed output, verdict,
                                prompt version, tokens, latency.
          correction_logs     — Self-correction audit trail (per loop: classify / extract).
          dead_letter_queue   — every document that left the pipeline without a
                                validated extraction, with the stage / code /
                                category that says where and why.
  gold    extractions         — validated results only, SCD Type 2 versioned.

``extraction_sources`` (a 500-character text snippet per result) is no longer
created: the full text lives in the raw store.  Databases created earlier keep
the table untouched; nothing writes to it any more.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import aiosqlite
from loguru import logger

from src.config import settings


_SCHEMA_SQL = """
-- ============================================================
-- Table 1: extractions — Core asset table (SCD Type 2 variant)
-- ============================================================
CREATE TABLE IF NOT EXISTS extractions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id              TEXT NOT NULL,
    file_path           TEXT NOT NULL,
    source_hash         TEXT NOT NULL,
    doc_type            TEXT NOT NULL,
    extracted_data      TEXT NOT NULL,
    version             INTEGER NOT NULL DEFAULT 1,
    is_current          INTEGER NOT NULL DEFAULT 1,
    pipeline_version    TEXT NOT NULL DEFAULT '1.0.0',
    model_name          TEXT NOT NULL,
    retry_count         INTEGER NOT NULL DEFAULT 0,
    created_at          TEXT NOT NULL,
    updated_at          TEXT NOT NULL,
    run_id              TEXT
);

-- Table 2: documents — bronze registry (the text itself is in the raw store)
CREATE TABLE IF NOT EXISTS documents (
    source_hash         TEXT PRIMARY KEY,
    raw_uri             TEXT NOT NULL,
    size_chars          INTEGER NOT NULL,
    first_file_path     TEXT NOT NULL,
    first_seen_at       TEXT NOT NULL,
    last_file_path      TEXT NOT NULL,
    last_seen_at        TEXT NOT NULL,
    last_run_id         TEXT,
    last_status         TEXT            -- 'stored' (gold) | 'dlq'
);

-- Table 5: step_runs — silver audit trail, one row per node execution
CREATE TABLE IF NOT EXISTS step_runs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id              TEXT NOT NULL,
    source_hash         TEXT NOT NULL,
    file_path           TEXT,
    stage               TEXT NOT NULL,  -- classify | classify_validate | extract | validate
    loop                TEXT NOT NULL,  -- classify | extract
    attempt             INTEGER NOT NULL,
    outcome             TEXT NOT NULL,  -- passed | rejected | fatal
    error_code          TEXT,
    error_category      TEXT,
    output              TEXT,           -- JSON
    model_name          TEXT,
    prompt_version      TEXT,
    raw_response        TEXT,
    prompt_tokens       INTEGER,
    completion_tokens   INTEGER,
    total_tokens        INTEGER,
    latency_ms          INTEGER,
    started_at          TEXT NOT NULL
);

-- Table 3: correction_logs — Self-correction audit trail
CREATE TABLE IF NOT EXISTS correction_logs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id              TEXT NOT NULL,
    cycle               INTEGER NOT NULL,
    error_summary       TEXT NOT NULL,
    llm_raw_response    TEXT,
    created_at          TEXT NOT NULL,
    loop                TEXT
);

-- Table 4: dead_letter_queue — Exhausted retry cemetery
CREATE TABLE IF NOT EXISTS dead_letter_queue (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path           TEXT NOT NULL,
    source_hash         TEXT NOT NULL,
    doc_type            TEXT,
    last_error          TEXT NOT NULL,
    retry_count         INTEGER NOT NULL,
    failed_at           TEXT NOT NULL,
    failed_stage        TEXT,
    error_code          TEXT,
    error_category      TEXT,
    failure_details     TEXT,
    classification      TEXT,
    correction_logs     TEXT,
    run_id              TEXT
);

-- Indices
CREATE INDEX IF NOT EXISTS idx_extractions_active_snapshot
    ON extractions(doc_id, is_current);

CREATE INDEX IF NOT EXISTS idx_extractions_hash
    ON extractions(source_hash);

CREATE INDEX IF NOT EXISTS idx_extractions_type
    ON extractions(doc_type);

CREATE INDEX IF NOT EXISTS idx_correction_logs_doc_id
    ON correction_logs(doc_id);

CREATE INDEX IF NOT EXISTS idx_step_runs_run
    ON step_runs(run_id);

CREATE INDEX IF NOT EXISTS idx_step_runs_doc
    ON step_runs(source_hash);

CREATE INDEX IF NOT EXISTS idx_step_runs_stage_outcome
    ON step_runs(stage, outcome);
"""


# Columns added after the first release.  ``CREATE TABLE IF NOT EXISTS`` does not
# touch an existing table, so databases created earlier get them via ALTER TABLE.
_ADDED_COLUMNS: list[tuple[str, str, str]] = [
    ("extractions", "run_id", "TEXT"),
    ("dead_letter_queue", "run_id", "TEXT"),
    ("correction_logs", "loop", "TEXT"),
    ("dead_letter_queue", "failed_stage", "TEXT"),
    ("dead_letter_queue", "error_code", "TEXT"),
    ("dead_letter_queue", "error_category", "TEXT"),
    ("dead_letter_queue", "failure_details", "TEXT"),
    ("dead_letter_queue", "classification", "TEXT"),
    ("dead_letter_queue", "correction_logs", "TEXT"),
]

_POST_MIGRATION_SQL = """
CREATE INDEX IF NOT EXISTS idx_dlq_stage_code
    ON dead_letter_queue(failed_stage, error_code);
"""


async def _migrate(conn: aiosqlite.Connection) -> None:
    """Add any column that an older database file is missing (idempotent)."""
    for table, column, ddl_type in _ADDED_COLUMNS:
        existing = {
            row["name"]
            for row in await conn.execute_fetchall(f"PRAGMA table_info({table})")
        }
        if column not in existing:
            await conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl_type}")
            logger.info("🗄️  Migrated: added {}.{}", table, column)


async def get_connection() -> aiosqlite.Connection:
    """
    Return a new aiosqlite Connection to the configured SQLite database.

    Uses WAL mode for better concurrent-read performance and enables
    foreign-key enforcement.
    """
    db_path = settings.sqlite_db_path
    conn = await aiosqlite.connect(db_path)
    conn.row_factory = sqlite3.Row
    await conn.execute("PRAGMA journal_mode=WAL;")
    await conn.execute("PRAGMA foreign_keys=ON;")
    return conn


async def init_db() -> None:
    """Initialise the database schema (idempotent — safe to call repeatedly)."""
    db_path = Path(settings.sqlite_db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = await get_connection()
    try:
        await conn.executescript(_SCHEMA_SQL)
        await _migrate(conn)
        await conn.executescript(_POST_MIGRATION_SQL)
        await conn.commit()
        logger.info("🗄️  Database initialised at '{}'", db_path)
    finally:
        await conn.close()
