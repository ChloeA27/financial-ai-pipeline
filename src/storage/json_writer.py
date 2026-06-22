"""
JSON writer — debug-friendly dual-write output.

Produces human-readable JSON files in ``output/`` so developers can
inspect individual extraction results without touching the database.

This was extracted from ``main.py._write_output()`` to keep the
repository layer responsible for all persistence coordination.
"""

from __future__ import annotations

import json
from pathlib import Path

import aiofiles
from loguru import logger


async def write_json_output(state: dict, output_dir: str) -> None:
    """
    Write a single pipeline result as a JSON file.

    The file is named ``{source_file_stem}_result.json`` and placed
    under ``output_dir/``.

    Args:
        state: The final PipelineState dict from a pipeline run.
        output_dir: Directory path to write the JSON file into.
    """
    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    file_stem = Path(state.get("file_path", "unknown")).stem
    output_file = out_path / f"{file_stem}_result.json"

    output_data = {
        "doc_id": (state.get("extracted_data") or {}).get("doc_id"),
        "file_path": state.get("file_path"),
        "doc_type": state.get("doc_type"),
        "validation_passed": state.get("validation_passed"),
        "retry_count": state.get("retry_count"),
        "correction_logs": state.get("correction_logs"),
        "extracted_data": state.get("extracted_data"),
        "error": state.get("error"),
    }

    async with aiofiles.open(output_file, "w", encoding="utf-8") as fh:
        await fh.write(json.dumps(output_data, indent=2, ensure_ascii=False))

    logger.info("📄 Wrote JSON output to {}", output_file)
