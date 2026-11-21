"""Architectural boundary, import-cycle, function-size ratchet, and mixin-freeze tests."""

from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MAX_FUNCTION_LINES = 80
FORBIDDEN_LIBRARY_IMPORTS = ("PySide6", "PyQt5", "PyQt6", "shiboken6", "sessionprepgui", "rich")

# Ratchet allowlist for legacy functions exceeding MAX_FUNCTION_LINES (empty: all decomposed).
LEGACY_LONG_FUNCTIONS: dict[tuple[str, str], int] = {}

# Freeze ratchet on legacy SessionPrepWindow mixins (empty: all mixins retired).
LEGACY_MIXIN_METHOD_LIMITS: dict[tuple[str, str], int] = {}


def _production_files() -> list[Path]:
    return sorted(
        [
            *(REPO_ROOT / "sessionpreplib").rglob("*.py"),
            *(REPO_ROOT / "sessionprepgui").rglob("*.py"),
            REPO_ROOT / "sessionprep.py",
            REPO_ROOT / "sessionprep-gui.py",
        ]
    )


def _iter_functions(tree: ast.AST, prefix: str = ""):
    for child in ast.iter_child_nodes(tree):
        if isinstance(child, ast.ClassDef):
            yield from _iter_functions(child, f"{prefix}{child.name}.")
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            length = (child.end_lineno or child.lineno) - child.lineno + 1
            yield f"{prefix}{child.name}", length


def test_sessionpreplib_has_no_gui_or_cli_imports():
    """sessionpreplib must stay strictly decoupled from Qt, sessionprepgui, and rich."""
    violations: list[str] = []
    for path in sorted((REPO_ROOT / "sessionpreplib").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        rel = path.relative_to(REPO_ROOT).as_posix()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            else:
                continue
            for name in names:
                top = name.split(".", 1)[0]
                if top in FORBIDDEN_LIBRARY_IMPORTS:
                    violations.append(f"{rel}:{node.lineno} imports {name}")
    assert not violations, "Library boundary violations:\n" + "\n".join(violations)


def test_sessionpreplib_import_graph_is_acyclic():
    """Internal module imports inside sessionpreplib must form a DAG (no circular imports)."""
    lib_dir = REPO_ROOT / "sessionpreplib"
    mods: dict[str, Path] = {}
    for path in sorted(lib_dir.rglob("*.py")):
        parts = list(path.relative_to(REPO_ROOT).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        mods[".".join(parts)] = path

    graph: dict[str, set[str]] = defaultdict(set)
    for mod, path in mods.items():
        pkg = mod if path.name == "__init__.py" else mod.rsplit(".", 1)[0]
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in mods and alias.name != mod:
                        graph[mod].add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.level > 0:
                    base = pkg.split(".")
                    if node.level > 1:
                        base = base[: -(node.level - 1)]
                    target = ".".join(base + ([node.module] if node.module else []))
                else:
                    target = node.module or ""
                if node.module and target in mods and target != mod:
                    graph[mod].add(target)
                for alias in node.names:
                    sub = f"{target}.{alias.name}" if target else alias.name
                    if sub in mods and sub != mod:
                        graph[mod].add(sub)

    index: dict[str, int] = {}
    low: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    cycles: list[list[str]] = []
    counter = 0

    def strongconnect(node: str) -> None:
        nonlocal counter
        index[node] = low[node] = counter
        counter += 1
        stack.append(node)
        on_stack.add(node)
        for neighbor in sorted(graph[node]):
            if neighbor not in index:
                strongconnect(neighbor)
                low[node] = min(low[node], low[neighbor])
            elif neighbor in on_stack:
                low[node] = min(low[node], index[neighbor])
        if low[node] == index[node]:
            component: list[str] = []
            while True:
                member = stack.pop()
                on_stack.remove(member)
                component.append(member)
                if member == node:
                    break
            if len(component) > 1:
                cycles.append(sorted(component))

    for mod in sorted(mods):
        if mod not in index:
            strongconnect(mod)

    assert not cycles, f"Circular imports detected in sessionpreplib: {cycles}"


def test_function_size_ratchet():
    """No production function/method may exceed 80 lines (ratchet on legacy exceptions)."""
    violations: list[str] = []
    seen_long: set[tuple[str, str]] = set()
    for path in _production_files():
        rel = path.relative_to(REPO_ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for qname, length in _iter_functions(tree):
            key = (rel, qname)
            if length > MAX_FUNCTION_LINES:
                seen_long.add(key)
                limit = LEGACY_LONG_FUNCTIONS.get(key)
                if limit is None:
                    violations.append(
                        f"{rel}::{qname} is {length} lines (exceeds {MAX_FUNCTION_LINES}-line ceiling)"
                    )
                elif length > limit:
                    violations.append(
                        f"{rel}::{qname} grew to {length} lines (ratchet ceiling is {limit})"
                    )
    stale = sorted(set(LEGACY_LONG_FUNCTIONS) - seen_long)
    assert not violations, "Function size limit violations:\n" + "\n".join(violations)
    assert not stale, (
        "Remove refactored functions from LEGACY_LONG_FUNCTIONS:\n"
        + "\n".join(f"  {p}::{q}" for p, q in stale)
    )


def test_legacy_gui_mixins_do_not_grow():
    """Legacy SessionPrepWindow mixins are frozen and must shrink to deletion."""
    violations: list[str] = []
    for (rel, cls_name), limit in LEGACY_MIXIN_METHOD_LIMITS.items():
        path = REPO_ROOT / rel
        if not path.exists():
            violations.append(f"Deleted mixin {rel}::{cls_name} should be removed from LEGACY_MIXIN_METHOD_LIMITS")
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        cls_node = next(
            (n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls_name),
            None,
        )
        assert cls_node is not None, f"Class {cls_name} not found in {rel}"
        methods = [n for n in cls_node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        if len(methods) > limit:
            violations.append(
                f"{rel}::{cls_name} grew to {len(methods)} methods (frozen ceiling is {limit})"
            )
    assert not violations, "Mixin freeze violations:\n" + "\n".join(violations)


def test_mainwindow_has_no_mixins():
    """SessionPrepWindow must compose controllers instead of inheriting *Mixin classes."""
    path = REPO_ROOT / "sessionprepgui" / "mainwindow.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    cls_node = next(
        (n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SessionPrepWindow"),
        None,
    )
    assert cls_node is not None, "SessionPrepWindow class not found"
    base_names = [
        b.id if isinstance(b, ast.Name) else ast.unparse(b)
        for b in cls_node.bases
    ]
    mixin_bases = [name for name in base_names if "Mixin" in name]
    assert not mixin_bases, f"SessionPrepWindow must not inherit mixins: {mixin_bases}"
    assert base_names == ["QMainWindow"], f"Unexpected SessionPrepWindow bases: {base_names}"


def test_no_temp_shims_remain():
    """No temporary Phase 05 shims may remain in sessionprepgui."""
    hits: list[str] = []
    for path in sorted((REPO_ROOT / "sessionprepgui").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), 1):
            if "TEMP_SHIM" in line:
                hits.append(f"{path.relative_to(REPO_ROOT).as_posix()}:{lineno}")
    assert not hits, "Temporary shims must be removed:\n" + "\n".join(hits)
