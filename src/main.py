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
from pathlib import Path

from loguru import logger

from src.graph.builder import run_pipeline
from src.nodes.reader import read_directory
from src.storage import ExtractionRepository, init_db


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
        logger.error(
            "❌ {} | {} | FAILED | retries={} | error={}",
            file_name,
            doc_type,
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


async def _process_batch_concurrent(
    states: list[dict], repo: ExtractionRepository
) -> list[dict]:
    """
    Process multiple files concurrently using asyncio.gather.

    This is the key performance optimization — instead of serial
    await calls, we launch N pipeline executions in parallel,
    each making independent LLM calls.
    """
    file_paths: list[str] = [
        s["file_path"] for s in states if s.get("file_path")
    ]

    logger.info("🚀 Launching {} pipeline(s) concurrently...", len(file_paths))
    start = time.perf_counter()

    # ── The magic: fire all pipelines in parallel ──
    results = await _gather_with_semaphore(file_paths, concurrency=5)

    elapsed = time.perf_counter() - start

    # Summarise
    passed = 0
    failed = 0
    for file_path, res in zip(file_paths, results):
        if isinstance(res, Exception):
            logger.error("💥 {} raised exception: {}", Path(file_path).name, res)
            failed += 1
            continue
        _summarise_state(Path(file_path).name, res)
        if isinstance(res, dict) and res.get("validation_passed"):
            passed += 1
        elif isinstance(res, dict) and res.get("error"):
            failed += 1

        if isinstance(res, dict):
            await _persist_result(res, repo)

    logger.info(
        "🏁 Batch complete: {}/{} passed, {} failed in {:.2f}s",
        passed,
        len(file_paths),
        failed,
        elapsed,
    )
    return results


async def _gather_with_semaphore(
    file_paths: list[str], concurrency: int = 5
) -> list:
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
        type=int,
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

        await _process_batch_concurrent(states, repo=repo)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
