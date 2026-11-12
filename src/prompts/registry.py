"""
Prompt registry — prompts are assets in ``templates/``, not strings in code.

Each prompt is one Markdown file, ``templates/<name>.md``. ``load_prompt``
returns its text with the file's final newline removed, so moving a prompt
from Python into a file does not change a single byte that is sent.

Rendering: a template that takes parameters uses ``string.Template``
placeholders (``$doc_type``) and is rendered with ``substitute``, so a missing
parameter is an error instead of a silent leftover. A template is rendered only
when parameters are passed. Templates loaded without parameters are returned
verbatim, so a literal ``$`` in them is safe.

Versioning is by content, not by a hand-kept number: ``LLMClient`` records
``prompt_version = sha256(<system message actually sent>)[:12]`` on every call
(see ``src/llm/client.py``). Any edit to a template, or to the output schema,
changes that hash in ``step_runs`` and in E1 results.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from string import Template

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


def available_prompts() -> list[str]:
    """Names of all prompt templates (file names without ``.md``)."""
    return sorted(p.stem for p in TEMPLATES_DIR.glob("*.md"))


@lru_cache(maxsize=None)
def _read(name: str) -> str:
    path = TEMPLATES_DIR / f"{name}.md"
    if not path.is_file():
        raise KeyError(
            f"unknown prompt {name!r}; available: {', '.join(available_prompts())}"
        )
    text = path.read_text(encoding="utf-8")
    return text[:-1] if text.endswith("\n") else text


def load_prompt(name: str, **params: str) -> str:
    """
    Text of prompt ``name``, rendered with ``params`` if any are given.

    Raises:
        KeyError: no template with that name, or a placeholder without a value.
    """
    text = _read(name)
    if not params:
        return text
    return Template(text).substitute(params)
