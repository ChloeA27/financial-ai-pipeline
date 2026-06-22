"""
Database layer — aiosqlite connection management & schema initialisation.

Creates 4 tables + indices on startup via ``init_db()``:

  extractions         — Core asset table with SCD Type 2 versioning.
  extraction_sources  — Cold-storage for raw text (hot/cold separation).
  correction_logs     — Self-correction audit trail.
  dead_letter_queue   — DLQ for documents that failed all retry cycles.
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
    updated_at          TEXT NOT NULL
);

-- Table 2: extraction_sources — Cold storage for raw text
-- NOTE: No FK constraint on doc_id because extractions.doc_id is NOT
-- UNIQUE (SCD Type 2 allows multiple versions per doc_id).
CREATE TABLE IF NOT EXISTS extraction_sources (
    doc_id              TEXT NOT NULL,
    raw_text_snippet    TEXT NOT NULL,
    PRIMARY KEY (doc_id)
);

-- Table 3: correction_logs — Self-correction audit trail
CREATE TABLE IF NOT EXISTS correction_logs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id              TEXT NOT NULL,
    cycle               INTEGER NOT NULL,
    error_summary       TEXT NOT NULL,
    llm_raw_response    TEXT,
    created_at          TEXT NOT NULL
);

-- Table 4: dead_letter_queue — Exhausted retry cemetery
CREATE TABLE IF NOT EXISTS dead_letter_queue (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path           TEXT NOT NULL,
    source_hash         TEXT NOT NULL,
    doc_type            TEXT,
    last_error          TEXT NOT NULL,
    retry_count         INTEGER NOT NULL,
    failed_at           TEXT NOT NULL
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
"""


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
        await conn.commit()
        logger.info("🗄️  Database initialised at '{}'", db_path)
    finally:
        await conn.close()
