"""
Classifier input selection — which part of a long filing the model gets to see.

The classifier has a fixed budget; filings are usually longer (dev median
~11.6k characters) and start with ~2.3k characters of SEC cover-page
boilerplate, while the event is often only described in an EX-99 press
release further down.  Three strategies, compared offline in E1:

  head        V0 — the first N characters (the original behaviour).
  structured  V1 — drop the cover page, then spend the budget on the start of
              every section: each 8-K Item and each attached document
              (EX-99.x press release), before extending any of them.
  keyword     V2 — drop the cover page, keep the start of the text, then add
              paragraphs that mention M&A / dividend vocabulary, in order.

All strategies are plain code (no LLM) and return the exact text sent plus
statistics, so "was the evidence even in the window?" can be measured
without calling a model.  Inputs that carry ``=== <TYPE> <file> ===`` markers
(main document + exhibits) are split into documents; plain text is one
document.

Budgets are in characters; ~4 characters per token for English filings is
the working approximation (the real count is recorded per call from the API).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

STRATEGIES = ("head", "structured", "keyword")
HEAD_CHARS = 4000  # V0: the original classifier window
CHARS_PER_TOKEN = 4
DEFAULT_BUDGET_TOKENS = 3500

_DOC_MARKER = re.compile(r"(?m)^=== (\S+) (\S+) ===[ \t]*$")
_ITEM_HEADER = re.compile(r"(?im)^[ \t]*item[ \t]+(\d\.\d\d)\b\.?[^\n]*")
_EGC = re.compile(r"(?i)emerging growth company")
_SECTION_HEAD_CHARS = 1500  # first pass: this much from every section
_KEYWORDS = re.compile(
    r"\b(merger|merge[sd]?|acqui(?:re|res|red|ring|sition|sitions)|plan of merger|"
    r"definitive agreement|tender offer|business combination|divest\w*|"
    r"disposition|spin-?off|dividends?|distribution|per share|record date|"
    r"payable)\b",
    re.IGNORECASE,
)
_CUT = "\n[…]\n"


@dataclass
class InputWindow:
    """The text actually sent to the classifier, and how it was chosen."""

    text: str
    strategy: str
    budget_chars: int
    chars_total: int
    sections: list[str] = field(default_factory=list)
    """Labels of the sections included, e.g. ``["8-K Item 8.01", "EX-99.1"]``."""
    truncated: bool = False
    """True when some source text did not fit in the window."""

    @property
    def chars_sent(self) -> int:
        return len(self.text)

    def stats(self) -> dict:
        return {
            "strategy": self.strategy,
            "budget_chars": self.budget_chars,
            "chars_total": self.chars_total,
            "chars_sent": self.chars_sent,
            "sections": self.sections,
            "truncated": self.truncated,
        }


@dataclass
class _Section:
    label: str
    text: str


def split_documents(text: str) -> list[tuple[str, str]]:
    """``[(doc_type, body), ...]`` using the ``=== TYPE file ===`` markers."""
    marks = list(_DOC_MARKER.finditer(text))
    if not marks:
        return [("DOC", text)]
    docs = []
    if text[: marks[0].start()].strip():
        docs.append(("PREAMBLE", text[: marks[0].start()]))
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        docs.append((m.group(1), text[m.end() : end]))
    return docs


def _cut_after_line(body: str, pos: int) -> str:
    nl = body.find("\n", pos)
    return body[nl + 1 :] if nl != -1 else ""


def strip_cover_page(body: str) -> str:
    """
    Drop SEC cover-page boilerplate from a main filing document.

    The cover page ends with fixed check-box lines, so the cut is made there —
    NOT at the first ``Item`` header, because some filings put an
    "Explanatory / Introductory Note" describing the event between the cover
    and Item 1.01 (cutting at the Item header loses it).

    1. 8-K: after the line of the last "emerging growth company" check box
       within the first 12,000 characters;
    2. 6-K: after the "Form 20-F / Form 40-F" check-box line;
    3. otherwise: at the first ``Item x.xx`` header if it is within the first
       8,000 characters; else the body is returned unchanged.
    """
    head = body[:12000]
    egc = [m.end() for m in _EGC.finditer(head)]
    if egc:
        return _cut_after_line(body, egc[-1])
    j = body[:6000].rfind("40-F")
    if j != -1:
        return _cut_after_line(body, j)
    m = _ITEM_HEADER.search(body)
    if m and m.start() < 8000:
        return body[m.start() :]
    return body


def _sections(text: str) -> list[_Section]:
    """Main document split into Items (cover page dropped), then each exhibit."""
    out: list[_Section] = []
    for i, (doc_type, body) in enumerate(split_documents(text)):
        is_main = i == 0 or doc_type in ("8-K", "8-K/A", "6-K", "DOC")
        if is_main and not doc_type.startswith("EX-"):
            body = strip_cover_page(body)
            items = list(_ITEM_HEADER.finditer(body))
            if not items:
                if body.strip():
                    out.append(_Section(doc_type, body))
                continue
            intro = body[: items[0].start()]
            if intro.strip():  # explanatory note, forward-looking statements, ...
                out.append(_Section(f"{doc_type} intro", intro))
            for k, m in enumerate(items):
                end = items[k + 1].start() if k + 1 < len(items) else len(body)
                part = body[m.start() : end]
                if m.group(1) == "9.01":  # exhibit list: header line is enough
                    part = m.group(0) + "\n"
                out.append(_Section(f"{doc_type} Item {m.group(1)}", part))
        elif body.strip():
            out.append(_Section(doc_type, body))
    return out


def _fill(sections: list[_Section], budget: int) -> tuple[str, list[str], bool]:
    """
    Two passes: first the head of every section (in order), then extend
    sections in order with what is left.  Returns (text, labels, truncated).
    """
    taken = [0] * len(sections)
    remaining = budget
    for pass_size in (_SECTION_HEAD_CHARS, None):
        for i, sec in enumerate(sections):
            if remaining <= 0:
                break
            want = len(sec.text) - taken[i]
            if pass_size is not None:
                want = min(want, pass_size - taken[i])
            grant = max(0, min(want, remaining))
            taken[i] += grant
            remaining -= grant
    parts, labels = [], []
    for sec, n in zip(sections, taken):
        if n <= 0:
            continue
        chunk = sec.text[:n]
        parts.append(f"[{sec.label}]\n{chunk.strip()}")
        if n < len(sec.text):
            parts[-1] += _CUT
        labels.append(sec.label)
    truncated = any(n < len(s.text) for s, n in zip(sections, taken))
    return "\n\n".join(parts), labels, truncated


def _paragraphs(text: str) -> list[str]:
    return [p for p in re.split(r"\n\s*\n", text) if p.strip()]


def build_input(
    raw_content: str,
    strategy: str = "head",
    budget_tokens: int = DEFAULT_BUDGET_TOKENS,
) -> InputWindow:
    """Select the classifier input for one document (see module docstring)."""
    if strategy not in STRATEGIES:
        raise ValueError(
            f"unknown input strategy {strategy!r}; use one of {STRATEGIES}"
        )
    total = len(raw_content)

    if strategy == "head":
        text = raw_content[:HEAD_CHARS]
        return InputWindow(
            text=text,
            strategy=strategy,
            budget_chars=HEAD_CHARS,
            chars_total=total,
            sections=["head"],
            truncated=total > HEAD_CHARS,
        )

    budget = budget_tokens * CHARS_PER_TOKEN
    sections = _sections(raw_content)

    if strategy == "structured":
        text, labels, truncated = _fill(sections, budget)
        return InputWindow(text, strategy, budget, total, labels, truncated)

    # keyword: start of the (cover-stripped) text, then matching paragraphs
    body = "\n\n".join(s.text for s in sections)
    lead_budget = budget // 3
    lead = body[:lead_budget]
    used = len(lead)
    picked: list[str] = []
    for para in _paragraphs(body[lead_budget:]):
        if not _KEYWORDS.search(para):
            continue
        para = para.strip()
        if len(para) > 1200:
            para = para[:1200] + " […]"
        if used + len(para) + len(_CUT) > budget:
            break
        picked.append(para)
        used += len(para) + len(_CUT)
    text = lead.strip()
    if picked:
        text += _CUT + _CUT.join(picked)
    truncated = len(body) > len(lead)
    labels = ["lead"] + (["keyword paragraphs"] if picked else [])
    return InputWindow(text, strategy, budget, total, labels, truncated)


def evidence_in_window(
    quotes: list[str], window_text: str, normalise
) -> Optional[bool]:
    """
    Input coverage for one document: are ALL gold evidence quotes inside the
    window (after the same normalisation the validator uses)?  ``None`` when
    there is no evidence to look for (e.g. a ``none`` document).
    """
    if not quotes:
        return None
    win = normalise(window_text)
    return all(normalise(q) in win for q in quotes)
