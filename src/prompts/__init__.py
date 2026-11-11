"""Prompt assets: templates/*.md plus a loader (see registry.py)."""

from src.prompts.registry import available_prompts, load_prompt

__all__ = ["available_prompts", "load_prompt"]
