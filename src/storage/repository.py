"""
Repository — the "estate manager" for all persistence operations.

Provides a single high-level method ``save_extraction(final_state)``
that persists one pipeline run, for passed AND failed documents alike:

  1. bronze — the source text is written once to the raw store
     (``<raw_store_path>/<sha256>.txt``) and registered in ``documents``;
  2. gold / DLQ routing:

  ✅ Passed validation  → **Branch A**: Idempotent Upsert (SCD Type 2)
                          If ``source_hash`` changed, the old row is archived
                          (``is_current=0``) and a new version is inserted.
  ❌ Failed / exhausted → **Branch B**: Clean write to dead_letter_queue (DLQ);
                          one row per (file_path, source_hash), refreshed on
                          repeated failures of the same content.

  3. silver — every ``step_runs`` record of the run (each LLM call and each
     validation, with its outcome) is appended to ``step_runs``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from src.config import settings
from src.rag.repository import delete_from_vector_store, save_to_vector_store
from src.state.failure import fallback_failure
from src.storage.db import get_connection
from src.storage.raw_store import store_raw
from src.utils.hashing import content_hash
from src.storage.json_writer import write_json_output


def _now() -> str:
    """Return current UTC timestamp as ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


_hash_content = content_hash  # kept for existing imports


def _source_hash(final_state: dict) -> str:
    """The run's content hash (computed by the Reader; recomputed if absent)."""
    return final_state.get("source_hash") or content_hash(
        final_state.get("raw_content", "")
    )


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
        """Persist one pipeline run: raw text, gold-or-DLQ row, step audit trail."""
        # ── Step 1: Always write the debug JSON file ──
        if self._output_dir:
            await write_json_output(final_state, self._output_dir)

        # ── Step 2: Bronze — keep the exact source text, passed or failed ──
        source_hash = _source_hash(final_state)
        raw_uri = await store_raw(source_hash, final_state.get("raw_content", ""))

        # ── Step 3: Route to the correct persistence branch ──
        validation_passed = final_state.get("validation_passed") is True
        has_error = bool(final_state.get("error"))

        if validation_passed and not has_error:
            await self._upsert_extraction(final_state)
            status = "stored"
        else:
            await self._insert_dead_letter(final_state)
            status = "dlq"

        # ── Step 4: Register the document and append the run's step records ──
        await self._record_run(final_state, source_hash, raw_uri, status)

    # ────────────────────────────────────────────────────────────────
    #  Bronze registry + silver step audit
    # ────────────────────────────────────────────────────────────────

    async def _record_run(
        self, final_state: dict, source_hash: str, raw_uri: str, status: str
    ) -> None:
        """Upsert ``documents`` and insert every ``step_runs`` record of the run."""
        file_path = final_state.get("file_path", "")
        run_id = final_state.get("run_id") or None
        now = _now()
        conn = await get_connection()
        try:
            await conn.execute(
                "INSERT INTO documents (source_hash, raw_uri, size_chars, "
                "first_file_path, first_seen_at, last_file_path, last_seen_at, "
                "last_run_id, last_status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(source_hash) DO UPDATE SET "
                "last_file_path = excluded.last_file_path, "
                "last_seen_at = excluded.last_seen_at, "
                "last_run_id = excluded.last_run_id, "
                "last_status = excluded.last_status",
                (
                    source_hash,
                    raw_uri,
                    len(final_state.get("raw_content", "")),
                    file_path,
                    now,
                    file_path,
                    now,
                    run_id,
                    status,
                ),
            )
            for step in final_state.get("step_runs") or []:
                await conn.execute(
                    "INSERT INTO step_runs (run_id, source_hash, file_path, stage, "
                    "loop, attempt, outcome, error_code, error_category, output, "
                    "model_name, prompt_version, raw_response, prompt_tokens, "
                    "completion_tokens, total_tokens, latency_ms, started_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        step.get("run_id") or run_id or "",
                        step.get("source_hash") or source_hash,
                        step.get("file_path") or file_path,
                        step["stage"],
                        step["loop"],
                        step["attempt"],
                        step["outcome"],
                        step.get("error_code"),
                        step.get("error_category"),
                        _to_json(step.get("output")),
                        step.get("model_name"),
                        step.get("prompt_version"),
                        step.get("raw_response"),
                        step.get("prompt_tokens"),
                        step.get("completion_tokens"),
                        step.get("total_tokens"),
                        step.get("latency_ms"),
                        step.get("started_at") or now,
                    ),
                )
            await conn.commit()
        finally:
            await conn.close()

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
             The archived version is also dropped from the vector store, so
             outdated extractions are never retrieved as RAG context.
          3. Otherwise → insert version 1 fresh.
        """
        extracted = final_state.get("extracted_data") or {}
        doc_id = extracted.get("doc_id", "")
        file_path = final_state.get("file_path", "")
        source_hash = _source_hash(final_state)
        doc_type = final_state.get("doc_type", "")
        retry_count = final_state.get("retry_count", 0)
        run_id = final_state.get("run_id") or None

        business_data = extracted

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
                "SELECT id, version, source_hash, doc_id, doc_type FROM extractions "
                "WHERE file_path = ? AND is_current = 1",
                (file_path,),
            )

            superseded: tuple[str, str] | None = None
            if existing:
                row = existing[0]
                superseded = (row["doc_id"], row["doc_type"])
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
                     retry_count, created_at, updated_at, run_id)
                VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?)
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
                    run_id,
                ),
            )

            # ── Write correction_logs if any ──
            correction_logs = final_state.get("correction_logs") or []
            for log in correction_logs:
                await conn.execute(
                    "INSERT INTO correction_logs "
                    "(doc_id, cycle, error_summary, llm_raw_response, created_at, "
                    "loop) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        doc_id,
                        log.get("cycle", 0),
                        log.get("error_summary", ""),
                        log.get("raw_feedback"),
                        now,
                        log.get("loop", "extract"),
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

            # ── Write validated extraction to vector store (RAG, if enabled) ──
            if settings.rag_enabled:
                await save_to_vector_store(extracted)
                if superseded is not None and superseded[0] != doc_id:
                    await delete_from_vector_store(*superseded)

        finally:
            await conn.close()

    # ────────────────────────────────────────────────────────────────
    #  Branch B — Dead Letter Queue
    # ────────────────────────────────────────────────────────────────

    async def _insert_dead_letter(self, final_state: dict) -> None:
        """
        Write a failed document to the dead_letter_queue.

        The row says *where* (``failed_stage``) and *why* (``error_code``,
        ``error_category``) the document stopped, and keeps the classifier
        output and the per-cycle correction history, which are otherwise lost
        for documents that never reach the ``extractions`` table.
        """
        file_path = final_state.get("file_path", "")
        source_hash = _source_hash(final_state)
        doc_type = final_state.get("doc_type")
        last_error = final_state.get("error") or "Unknown error"
        retry_count = final_state.get("retry_count", 0)

        failure = final_state.get("failure") or fallback_failure(last_error)
        failure_details = _to_json(failure.get("details"))
        classification = _to_json(final_state.get("classification"))
        correction_logs = _to_json(final_state.get("correction_logs") or [])

        conn = await get_connection()
        try:
            # Re-running an unchanged file that fails again must not pile up
            # duplicate rows: one DLQ entry per (file_path, content hash),
            # refreshed with the latest failure.
            existing = await conn.execute_fetchall(
                "SELECT id FROM dead_letter_queue "
                "WHERE file_path = ? AND source_hash = ?",
                (file_path, source_hash),
            )
            columns = (
                doc_type,
                last_error,
                retry_count,
                failure["stage"],
                failure["code"],
                failure["category"],
                failure_details,
                classification,
                correction_logs,
                final_state.get("run_id") or None,
            )
            if existing:
                await conn.execute(
                    "UPDATE dead_letter_queue "
                    "SET doc_type = ?, last_error = ?, retry_count = ?, "
                    "failed_stage = ?, error_code = ?, error_category = ?, "
                    "failure_details = ?, classification = ?, correction_logs = ?, "
                    "run_id = ?, failed_at = ? WHERE id = ?",
                    (*columns, _now(), existing[0]["id"]),
                )
            else:
                await conn.execute(
                    "INSERT INTO dead_letter_queue "
                    "(file_path, source_hash, doc_type, last_error, retry_count, "
                    "failed_stage, error_code, error_category, failure_details, "
                    "classification, correction_logs, run_id, failed_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (file_path, source_hash, *columns, _now()),
                )
            await conn.commit()

            logger.warning(
                "☠️  {} → dead_letter_queue | stage={} | code={} | error='{}' | retries={}",
                Path(file_path).name,
                failure["stage"],
                failure["code"],
                last_error[:80],
                retry_count,
            )
        finally:
            await conn.close()


def _to_json(value: Any) -> Optional[str]:
    """JSON-encode a value for a TEXT column (None stays NULL)."""
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, default=str)
