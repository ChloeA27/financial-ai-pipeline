"""
Pipeline entry point — CLI to run the full financial extraction pipeline.

Usage:
    python -m src.main --file raw_data/sample/ma_microsoft_activision.txt
    python -m src.main --dir raw_data/sample --output output
"""

from __future__ import annotations

import argparse
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

from loguru import logger

from src.workflows.graph import run_pipeline
from src.nodes.reader import read_directory
from src.schemas.failure import (
    CAT_UNKNOWN,
    PIPELINE_EXCEPTION,
    STAGE_PIPELINE,
    make_failure,
)
from src.storage import ExtractionRepository, init_db
from src.utils.hashing import content_hash


def _setup_logger(level: str = "INFO") -> None:
    """Configure loguru for structured console output."""
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:HH:mm:ss}</green> | <level>{level:7}</level> | "
            "<cyan>{function}</cyan> | {message}"
        ),
    )


def _summarise_state(file_name: str, state: dict) -> None:
    """Log a single summary line for a pipeline result."""
    doc_type = state.get("doc_type", "N/A")
    valid = state.get("validation_passed")
    retries = state.get("retry_count", 0)
    error = state.get("error")

    # Extract doc_id from extracted_data for traceability
    extracted = state.get("extracted_data") or {}
    doc_id = extracted.get("doc_id", "no-id")

    if valid:
        logger.success(
            "✅ {} | {} | doc_id={} | validated OK | retries={}",
            file_name,
            doc_type,
            doc_id,
            retries,
        )
    elif error:
        failure = state.get("failure") or {}
        logger.error(
            "❌ {} | {} | FAILED | stage={} code={} | retries={} | error={}",
            file_name,
            doc_type,
            failure.get("stage", "?"),
            failure.get("code", "?"),
            retries,
            error,
        )
    else:
        logger.warning(
            "⚠️  {} | {} | not validated | retries={}",
            file_name,
            doc_type,
            retries,
        )


async def _persist_result(state: dict, repo: ExtractionRepository) -> None:
    """
    Proxy/facade for persistence.

    Previously ``_write_output()`` did all JSON work in-place. Now it
    delegates to ``ExtractionRepository.save_extraction()``, which
    handles dual-write (JSON + SQLite) and branch routing.
    """
    await repo.save_extraction(state)


async def _process_single(file_path: str, repo: ExtractionRepository) -> dict:
    """Run the pipeline on a single file and persist the result."""
    logger.info("Processing: {}", file_path)
    start = time.perf_counter()

    final_state = await run_pipeline(file_path)

    elapsed = time.perf_counter() - start
    _summarise_state(Path(file_path).name, final_state)
    logger.debug("⏱  {:.2f}s for {}", elapsed, Path(file_path).name)

    await _persist_result(final_state, repo)
    return final_state


def _crashed_state(
    file_path: str,
    raw_content: str,
    exc: BaseException,
    read_state: Optional[dict] = None,
) -> dict:
    """
    State for a document whose pipeline raised, so it is routed to the DLQ.

    ``read_state`` (the Reader's state for this file, if any) supplies the
    bronze identity, so the DLQ row points at the stored original bytes.
    """
    message = f"pipeline raised {type(exc).__name__}: {exc}"
    read_state = read_state or {}
    return {
        "file_path": file_path,
        "raw_content": raw_content,
        "source_hash": read_state.get("source_hash") or content_hash(raw_content),
        "raw_uri": read_state.get("raw_uri"),
        "source_encoding": read_state.get("source_encoding"),
        "run_id": uuid.uuid4().hex,
        "step_runs": [],
        "doc_type": None,
        "extracted_data": None,
        "validation_passed": None,
        "validation_report": None,
        "correction_logs": [],
        "retry_count": 0,
        "error": message,
        "failure": make_failure(
            STAGE_PIPELINE, PIPELINE_EXCEPTION, CAT_UNKNOWN, message
        ),
    }


async def _process_batch_concurrent(
    states: list[dict], repo: ExtractionRepository, concurrency: int = 5
) -> list[dict]:
    """
    Process multiple files concurrently using asyncio.gather.

    This is the key performance optimization — instead of serial
    await calls, we launch N pipeline executions in parallel,
    each making independent LLM calls.
    """
    file_paths: list[str] = [s["file_path"] for s in states if s.get("file_path")]

    logger.info(
        "🚀 Launching {} pipeline(s) concurrently (max {} at a time)...",
        len(file_paths),
        concurrency,
    )
    start = time.perf_counter()

    # ── The magic: fire all pipelines in parallel ──
    results = await _gather_with_semaphore(file_paths, concurrency=concurrency)

    elapsed = time.perf_counter() - start

    # Summarise + persist
    state_by_path = {s["file_path"]: s for s in states}
    passed = 0
    failed = 0
    for file_path, res in zip(file_paths, results):
        if isinstance(res, Exception):
            logger.error("💥 {} raised exception: {}", Path(file_path).name, res)
            failed += 1
            # Don't lose the document: quarantine it so it shows up in the DLQ
            # (and on the dashboard) instead of only in this run's log.
            read_state = state_by_path.get(file_path, {})
            res = _crashed_state(
                file_path, read_state.get("raw_content", ""), res, read_state
            )
        else:
            _summarise_state(Path(file_path).name, res)
            if isinstance(res, dict) and res.get("validation_passed"):
                passed += 1
            elif isinstance(res, dict) and res.get("error"):
                failed += 1

        if isinstance(res, dict):
            try:
                await _persist_result(res, repo)
            except Exception as exc:  # one bad write must not abort the batch
                logger.error(
                    "💾 {} could not be persisted: {}", Path(file_path).name, exc
                )

    logger.info(
        "🏁 Batch complete: {}/{} passed, {} failed in {:.2f}s",
        passed,
        len(file_paths),
        failed,
        elapsed,
    )
    return results


async def _gather_with_semaphore(file_paths: list[str], concurrency: int = 5) -> list:
    """
    Run pipeline invocations concurrently with a semaphore to
    avoid overwhelming the LLM API with too many simultaneous requests.
    """
    import asyncio

    sem = asyncio.Semaphore(concurrency)

    async def _bounded_pipeline(fp: str) -> dict:
        async with sem:
            return await run_pipeline(fp)

    tasks = [_bounded_pipeline(fp) for fp in file_paths]
    return await asyncio.gather(*tasks, return_exceptions=True)


def _positive_int(value: str) -> int:
    """argparse type: a Semaphore(0) would deadlock the batch, so reject < 1."""
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {n}")
    return n


async def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Financial AI Pipeline — unstructured financial announcements "
            "→ structured extraction with self-correction"
        )
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--file",
        type=str,
        help="Path to a single raw text file to process.",
    )
    group.add_argument(
        "--dir",
        type=str,
        default=None,
        help="Directory containing raw .txt files to batch-process concurrently.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional: directory to write JSON output files.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--concurrency",
        type=_positive_int,
        default=5,
        help="Max concurrent pipeline executions (default: 5).",
    )

    args = parser.parse_args()

    # ── Setup ──
    _setup_logger(args.log_level)
    logger.info("🚀 Financial AI Pipeline v1.0.0")

    # ── Initialise DB + Repository (singleton-style, injected downstream) ──
    await init_db()
    repo = ExtractionRepository(output_dir=args.output or "")

    # ── Single file mode ──
    if args.file:
        await _process_single(args.file, repo=repo)

    # ── Batch directory mode (CONCURRENT) ──
    elif args.dir:
        states = await read_directory(args.dir)
        logger.info("📂 Found {} file(s) in '{}'", len(states), args.dir)

        if not states:
            logger.warning("No .txt files found in '{}'", args.dir)
            return

        await _process_batch_concurrent(states, repo=repo, concurrency=args.concurrency)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
