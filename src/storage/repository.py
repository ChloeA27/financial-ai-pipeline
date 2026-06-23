"""
Repository — the "estate manager" for all persistence operations.

Provides a single high-level method ``save_extraction(final_state)``
that coordinates dual-write (JSON + SQLite) with intelligent routing:

  ✅ Passed validation  → **Branch A**: Idempotent Upsert (SCD Type 2)
                          If ``source_hash`` changed, the old row is archived
                          (``is_current=0``) and a new version is inserted.
  ❌ Failed / exhausted → **Branch B**: Clean write to dead_letter_queue (DLQ).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from src.rag.repository import save_to_vector_store
from src.storage.db import get_connection
from src.storage.json_writer import write_json_output


def _now() -> str:
    """Return current UTC timestamp as ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _hash_content(raw_content: str) -> str:
    """Return SHA-256 hex digest of the raw source text."""
    return hashlib.sha256(raw_content.encode("utf-8")).hexdigest()


def _strip_raw_text(extracted: dict[str, Any]) -> dict[str, Any]:
    """Remove ``raw_text_snippet`` from extracted data for clean business storage."""
    return {k: v for k, v in extracted.items() if k != "raw_text_snippet"}


class ExtractionRepository:
    """
    Persistence repository — coordinates JSON dual-write & SQLite storage.

    Usage::

        repo = ExtractionRepository(output_dir="./output")
        await repo.save_extraction(final_state)
    """

    def __init__(self, output_dir: str = "") -> None:
        self._output_dir = output_dir

    # ────────────────────────────────────────────────────────────────
    #  Public API
    # ────────────────────────────────────────────────────────────────

    async def save_extraction(self, final_state: dict) -> None:
        """Persist a pipeline result — dual-write JSON + SQLite."""
        # ── Step 1: Always write the debug JSON file ──
        if self._output_dir:
            await write_json_output(final_state, self._output_dir)

        # ── Step 2: Route to the correct persistence branch ──
        validation_passed = final_state.get("validation_passed") is True
        has_error = bool(final_state.get("error"))

        if validation_passed and not has_error:
            await self._upsert_extraction(final_state)
        else:
            await self._insert_dead_letter(final_state)

    # ────────────────────────────────────────────────────────────────
    #  Branch A — Idempotent Upsert (SCD Type 2)
    # ────────────────────────────────────────────────────────────────

    async def _upsert_extraction(self, final_state: dict) -> None:
        """
        Insert or update extraction record with SCD Type 2 versioning.

        Idempotency Guard — the TRUE dedup key is **source_hash** (SHA256
        of the raw file content), NOT doc_id (which is an LLM-generated
        UUID that changes every run).

        Rules:
          1. If ANY current row (``is_current = 1``) with the same
             ``source_hash`` already exists → **SKIP** (idempotent).
             The content hasn't changed, so there's nothing to persist.
          2. If no row with this hash exists BUT another current row for
             the same ``file_path`` has a different hash → archive the old
             row (``is_current = 0``) and insert a new row with version + 1.
          3. Otherwise → insert version 1 fresh.
        """
        extracted = final_state.get("extracted_data") or {}
        doc_id = extracted.get("doc_id", "")
        file_path = final_state.get("file_path", "")
        raw_content = final_state.get("raw_content", "")
        source_hash = _hash_content(raw_content)
        doc_type = final_state.get("doc_type", "")
        retry_count = final_state.get("retry_count", 0)

        # Clean business data (strip raw_text_snippet)
        business_data = _strip_raw_text(extracted)

        conn = await get_connection()
        try:
            # ══════════════════════════════════════════════════════
            #  Idempotency Guard — check by source_hash
            # ══════════════════════════════════════════════════════
            hash_matches = await conn.execute_fetchall(
                "SELECT id, version FROM extractions "
                "WHERE source_hash = ? AND is_current = 1",
                (source_hash,),
            )

            if hash_matches:
                logger.info(
                    "⏭️  [Idempotency Guard] {} | hash={} | "
                    "static content detected → skipping write",
                    Path(file_path).name,
                    source_hash[:12],
                )
                return

            # ══════════════════════════════════════════════════════
            #  SCD Type 2 — check by file_path for versioning
            # ══════════════════════════════════════════════════════
            existing = await conn.execute_fetchall(
                "SELECT id, version, source_hash FROM extractions "
                "WHERE file_path = ? AND is_current = 1",
                (file_path,),
            )

            if existing:
                row = existing[0]
                # Hash is guaranteed to be different since we already
                # checked hash_matches above → archive old, insert new
                new_version = row["version"] + 1
                await conn.execute(
                    "UPDATE extractions SET is_current = 0, updated_at = ? "
                    "WHERE id = ?",
                    (_now(), row["id"]),
                )
                logger.info(
                    "📦 {} | archived v{} (hash changed: {}...)",
                    Path(file_path).name,
                    row["version"],
                    row["source_hash"][:12],
                )
            else:
                new_version = 1

            # ── Insert new version row ──
            now = _now()
            await conn.execute(
                """
                INSERT INTO extractions
                    (doc_id, file_path, source_hash, doc_type, extracted_data,
                     version, is_current, pipeline_version, model_name,
                     retry_count, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?)
                """,
                (
                    doc_id,
                    file_path,
                    source_hash,
                    doc_type,
                    json.dumps(business_data, ensure_ascii=False),
                    new_version,
                    (extracted.get("metadata") or {}).get("pipeline_version", "1.0.0"),
                    (extracted.get("metadata") or {}).get("model_name", ""),
                    retry_count,
                    now,
                    now,
                ),
            )

            # ── Write raw text snippet to extraction_sources ──
            snippet = extracted.get("raw_text_snippet", "")[:500]
            await conn.execute(
                "INSERT OR REPLACE INTO extraction_sources (doc_id, raw_text_snippet) "
                "VALUES (?, ?)",
                (doc_id, snippet),
            )

            # ── Write correction_logs if any ──
            correction_logs = final_state.get("correction_logs") or []
            for log in correction_logs:
                await conn.execute(
                    "INSERT INTO correction_logs "
                    "(doc_id, cycle, error_summary, llm_raw_response, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        doc_id,
                        log.get("cycle", 0),
                        log.get("error_summary", ""),
                        log.get("raw_feedback"),
                        now,
                    ),
                )

            await conn.commit()

            logger.success(
                "💾 {} | doc_id={} | v{} | {} | committed to DB",
                Path(file_path).name,
                doc_id,
                new_version,
                doc_type,
            )

            # ── Write validated extraction to vector store (RAG) ──
            await save_to_vector_store(extracted)

        finally:
            await conn.close()

    # ────────────────────────────────────────────────────────────────
    #  Branch B — Dead Letter Queue
    # ────────────────────────────────────────────────────────────────

    async def _insert_dead_letter(self, final_state: dict) -> None:
        """Write a failed document to the dead_letter_queue."""
        file_path = final_state.get("file_path", "")
        raw_content = final_state.get("raw_content", "")
        source_hash = _hash_content(raw_content)
        doc_type = final_state.get("doc_type")
        last_error = final_state.get("error", "Unknown error")
        retry_count = final_state.get("retry_count", 0)

        conn = await get_connection()
        try:
            await conn.execute(
                "INSERT INTO dead_letter_queue "
                "(file_path, source_hash, doc_type, last_error, retry_count, failed_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (file_path, source_hash, doc_type, last_error, retry_count, _now()),
            )
            await conn.commit()

            logger.warning(
                "☠️  {} → dead_letter_queue | error='{}' | retries={}",
                Path(file_path).name,
                last_error[:80],
                retry_count,
            )
        finally:
            await conn.close()
