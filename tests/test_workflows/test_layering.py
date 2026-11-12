"""
Layering rules: nodes do work, contracts live in src/schemas.

A node may import shared helpers (input windows, retry bookkeeping, step
records) but never another node: a contract one node needs from another
(labels, output schemas, state types) belongs in ``src/schemas``.
"""

from __future__ import annotations

import ast
from pathlib import Path

NODES_DIR = Path(__file__).resolve().parents[2] / "src" / "nodes"
NODE_MODULES = {
    "reader",
    "classifier",
    "classification_validator",
    "extractor",
    "extraction_validator",
}


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
        elif isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
    return found


def test_every_node_module_exists() -> None:
    assert {p.stem for p in NODES_DIR.glob("*.py")} >= NODE_MODULES


def test_no_node_imports_another_node() -> None:
    offenders = {
        name: sorted(
            m
            for m in _imports(NODES_DIR / f"{name}.py")
            if m.startswith("src.nodes.") and m.split(".")[-1] in NODE_MODULES
        )
        for name in NODE_MODULES
    }
    assert {k: v for k, v in offenders.items() if v} == {}


def test_schemas_do_not_depend_on_nodes() -> None:
    schemas = NODES_DIR.parent / "schemas"
    bad = {
        str(p.relative_to(schemas)): sorted(
            m for m in _imports(p) if m.startswith(("src.nodes", "src.workflows"))
        )
        for p in schemas.rglob("*.py")
    }
    assert {k: v for k, v in bad.items() if v} == {}


def test_edges_only_route() -> None:
    # Routers read state and name the next node; they never call node code.
    edges = NODES_DIR.parent / "workflows" / "edges.py"
    assert not [m for m in _imports(edges) if m.startswith("src.nodes")]


def test_nodes_do_not_import_the_workflow() -> None:
    # A node returns a state update; it never decides or wires what runs next.
    bad = {
        p.stem: sorted(m for m in _imports(p) if m.startswith("src.workflows"))
        for p in NODES_DIR.glob("*.py")
    }
    assert {k: v for k, v in bad.items() if v} == {}


def test_production_code_does_not_depend_on_evals() -> None:
    # Evaluation code may use the pipeline; the pipeline never imports evals/.
    src = NODES_DIR.parent
    bad = {
        str(p.relative_to(src)): sorted(m for m in _imports(p) if m.startswith("evals"))
        for p in src.rglob("*.py")
    }
    assert {k: v for k, v in bad.items() if v} == {}
