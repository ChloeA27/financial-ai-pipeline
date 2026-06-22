"""
Storage Layer — unified persistence for the Financial AI Pipeline.

Exports the high-level interface used by main.py:

    from src.storage import ExtractionRepository, init_db

    await init_db()
    repo = ExtractionRepository(output_dir="./output")
    await repo.save_extraction(final_state)
"""

from src.storage.db import init_db
from src.storage.repository import ExtractionRepository

__all__ = ["init_db", "ExtractionRepository"]
