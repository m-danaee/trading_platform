#!/usr/bin/env python3
"""Trace blast radius of git-changed Python files using .codelookup/graph.json."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

GRAPH_PATH = Path(".codelookup/graph.json")
SKIP_PREFIXES = (
    ".venv/",
    "venv/",
    "__pycache__/",
    "outputs/",
    "long_2/",
    "optimized_long/",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def run_git(args: list[str], root: Path) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def changed_py_files(root: Path) -> list[str]:
    tracked = run_git(["diff", "--name-only", "HEAD"], root)
    untracked = run_git(["ls-files", "--others", "--exclude-standard"], root)
    names = set()
    for block in (tracked, untracked):
        for line in block.splitlines():
            line = line.strip()
            if not line.endswith(".py"):
                continue
            if any(line.startswith(p) for p in SKIP_PREFIXES):
                continue
            names.add(line)
    return sorted(names)


def path_to_module(path: str, graph: dict) -> str | None:
    modules: dict[str, str] = graph.get("modules", {})
    for mod, rel in modules.items():
        if rel == path or rel.replace("/", ".") == path.replace("/", "."):
            return mod
    stem = Path(path).with_suffix("").as_posix().replace("/", ".")
    if stem in modules:
        return stem
    for mod, rel in modules.items():
        if rel == path:
            return mod
    return None


def blast_radius(changed_modules: list[str], graph: dict, depth: int = 2) -> dict[str, set[str]]:
    imported_by: dict[str, list[str]] = graph.get("imported_by", {})
    radius: dict[str, set[str]] = {m: set() for m in changed_modules}
    frontier = set(changed_modules)
    for _ in range(depth):
        next_frontier: set[str] = set()
        for mod in frontier:
            for parent in imported_by.get(mod, []):
                for origin in changed_modules:
                    radius[origin].add(parent)
                next_frontier.add(parent)
        frontier = next_frontier - set(changed_modules)
    return radius


def mermaid_diagram(changed: list[str], radius: dict[str, set[str]], graph: dict) -> str:
    modules: dict[str, str] = graph.get("modules", {})
    lines = ["flowchart TD"]
    node_ids: dict[str, str] = {}

    def nid(name: str) -> str:
        if name not in node_ids:
            safe = "n" + str(len(node_ids))
            node_ids[name] = safe
        return node_ids[name]

    for mod in changed:
        label = modules.get(mod, mod)
        lines.append(f'  {nid(mod)}["{mod}\\n({label})"]:::changed')

    seen_edges: set[tuple[str, str]] = set()
    for origin, dependents in radius.items():
        for dep in sorted(dependents):
            lines.append(f"  {nid(dep)} --> {nid(origin)}")
            seen_edges.add((dep, origin))
            if dep not in node_ids:
                label = modules.get(dep, dep)
                lines.append(f'  {nid(dep)}["{dep}\\n({label})"]')

    lines.append("  classDef changed fill:#f96,stroke:#333")
    return "\n".join(lines)


def main() -> int:
    root = repo_root()
    graph_file = root / GRAPH_PATH
    if not graph_file.is_file():
        print("Graph missing. Run: .venv/bin/python .cursor/skills/codelookup/scripts/generate_graph.py")
        return 1

    graph = json.loads(graph_file.read_text(encoding="utf-8"))
    changed_paths = changed_py_files(root)
    if not changed_paths:
        print("No changed Python files detected.")
        return 0

    changed_modules: list[str] = []
    unmapped: list[str] = []
    for path in changed_paths:
        mod = path_to_module(path, graph)
        if mod:
            changed_modules.append(mod)
        else:
            unmapped.append(path)

    print("## Changed files")
    for p in changed_paths:
        print(f"- {p}")

    if unmapped:
        print("\n## Unmapped (re-run generate_graph.py if new packages)")
        for p in unmapped:
            print(f"- {p}")

    if not changed_modules:
        return 0

    radius = blast_radius(changed_modules, graph)
    print("\n## Blast radius (importers, depth 2)")
    for mod in changed_modules:
        deps = sorted(radius.get(mod, set()))
        print(f"\n### {mod}")
        if deps:
            for d in deps:
                rel = graph["modules"].get(d, d)
                print(f"- {d} ({rel})")
        else:
            print("- (no importers in graph)")

    print("\n## Mermaid blast radius")
    print("```mermaid")
    print(mermaid_diagram(changed_modules, radius, graph))
    print("```")
    return 0


if __name__ == "__main__":
    sys.exit(main())
