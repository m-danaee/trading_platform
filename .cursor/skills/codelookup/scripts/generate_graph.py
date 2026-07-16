#!/usr/bin/env python3
"""Build a static Python import graph for codelookup blast-radius analysis."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

GRAPH_DIR = ".codelookup"
GRAPH_FILE = "graph.json"
SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
    ".mypy_cache",
    "outputs",
    "long_2",
    "optimized_long",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def module_name_for(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    parts = list(rel.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) if parts else rel.stem


def resolve_import(
    module: str | None,
    level: int,
    current: str,
    package_parts: list[str],
) -> str | None:
    if level:
        base = package_parts[: max(0, len(package_parts) - level + 1)]
        if module:
            return ".".join(base + module.split("."))
        return ".".join(base) if base else None
    return module


def imports_in_file(path: Path, module: str) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, OSError):
        return set()

    package_parts = module.split(".")
    found: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name:
                    found.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            target = resolve_import(
                node.module, node.level, module, package_parts)
            if target:
                found.add(target)
    return found


def collect_py_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*.py"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        files.append(path)
    return sorted(files)


def build_graph(root: Path) -> dict:
    py_files = collect_py_files(root)
    modules = {path: module_name_for(path, root) for path in py_files}
    name_to_path = {name: str(path.relative_to(root))
                    for path, name in modules.items()}

    edges: dict[str, list[str]] = {}
    reverse: dict[str, list[str]] = {}

    for path, mod in modules.items():
        deps = sorted(imports_in_file(path, mod))
        edges[mod] = deps
        for dep in deps:
            reverse.setdefault(dep, []).append(mod)

    for mod in edges:
        reverse.setdefault(mod, [])

    return {
        "root": str(root),
        "modules": {mod: name_to_path.get(mod, "") for mod in sorted(edges)},
        "imports": edges,
        "imported_by": {k: sorted(set(v)) for k, v in sorted(reverse.items())},
    }


def main() -> int:
    root = repo_root()
    out_dir = root / GRAPH_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    graph = build_graph(root)
    out_path = out_dir / GRAPH_FILE
    out_path.write_text(json.dumps(graph, indent=2), encoding="utf-8")
    print(f"Wrote {out_path} ({len(graph['modules'])} modules)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
