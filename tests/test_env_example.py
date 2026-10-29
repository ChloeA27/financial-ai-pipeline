"""The documented .env template must be copy-paste safe."""

from __future__ import annotations

import re
from pathlib import Path

from dotenv import dotenv_values

ENV_EXAMPLE = Path(__file__).resolve().parents[1] / ".env.example"


def test_no_whitespace_after_equals() -> None:
    bad = [
        line
        for line in ENV_EXAMPLE.read_text().splitlines()
        if re.match(r"^[A-Z_][A-Z0-9_]*=\s", line)
    ]
    assert bad == []


def test_every_setting_it_documents_exists_in_config() -> None:
    from src.config import Settings

    known = {name.upper() for name in Settings.model_fields}
    assert set(dotenv_values(ENV_EXAMPLE)) <= known
