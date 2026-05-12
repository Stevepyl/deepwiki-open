from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from adalflow.utils import get_adalflow_default_root_path

from api.knowledge_store import RepoKnowledgeStore, default_repo_id


IGNORED_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
    ".mypy_cache",
    ".pytest_cache",
    ".tox",
    ".eggs",
    ".idea",
    ".vscode",
    "venv",
    "env",
}


@dataclass
class DiffLine:
    kind: str
    value: str
    old_line: int | None = None
    new_line: int | None = None


@dataclass
class DiffHunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: list[DiffLine] = field(default_factory=list)

    @property
    def old_end(self) -> int:
        return self.old_start + max(self.old_count, 1) - 1

    @property
    def new_end(self) -> int:
        return self.new_start + max(self.new_count, 1) - 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "old_start": self.old_start,
            "old_end": self.old_end,
            "new_start": self.new_start,
            "new_end": self.new_end,
            "added_lines": [
                {"line": line.new_line, "value": line.value}
                for line in self.lines
                if line.kind == "added"
            ],
            "removed_lines": [
                {"line": line.old_line, "value": line.value}
                for line in self.lines
                if line.kind == "removed"
            ],
        }


@dataclass
class FileDiff:
    old_path: str
    new_path: str
    change_type: str = "modified"
    hunks: list[DiffHunk] = field(default_factory=list)

    @property
    def path(self) -> str:
        return self.new_path if self.new_path != "/dev/null" else self.old_path

    def changed_old_lines(self) -> set[int]:
        return {
            line.old_line
            for hunk in self.hunks
            for line in hunk.lines
            if line.kind == "removed" and line.old_line is not None
        }

    def changed_new_lines(self) -> set[int]:
        return {
            line.new_line
            for hunk in self.hunks
            for line in hunk.lines
            if line.kind == "added" and line.new_line is not None
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "old_path": self.old_path,
            "new_path": self.new_path,
            "change_type": self.change_type,
            "hunks": [hunk.to_dict() for hunk in self.hunks],
        }


@dataclass
class SymbolInfo:
    id: str
    name: str
    symbol_type: str
    file_path: str
    module: str
    start_line: int
    end_line: int
    code_hash: str
    signature: str = ""
    runtime_tags: list[str] = field(default_factory=list)


@dataclass
class ReferenceInfo:
    file_path: str
    source_scope: str
    ref_type: str
    target: str
    ref_name: str
    line: int
    snippet: str = ""


def _run(command: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def _sanitize_git_error(message: str, token: str | None = None) -> str:
    if token:
        message = message.replace(token, "***TOKEN***")
    return message


def run_git_diff(repo_path: Path, base: str, head: str) -> str:
    proc = _run(["git", "-C", str(repo_path), "diff", f"{base}..{head}", "--"])
    if proc.returncode != 0:
        raise ValueError(proc.stderr.strip() or "git diff failed")
    return proc.stdout


def read_git_file(repo_path: Path, rev: str, rel_path: str) -> str:
    if not rel_path or rel_path == "/dev/null":
        return ""
    proc = _run(["git", "-C", str(repo_path), "show", f"{rev}:{rel_path}"])
    if proc.returncode != 0:
        return ""
    return proc.stdout


def parse_unified_diff(patch_text: str) -> list[FileDiff]:
    files: list[FileDiff] = []
    current_file: FileDiff | None = None
    current_hunk: DiffHunk | None = None
    old_line = 0
    new_line = 0
    hunk_header = re.compile(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")

    for raw_line in patch_text.splitlines():
        if raw_line.startswith("diff --git "):
            parts = raw_line.split()
            old_path = parts[2][2:] if len(parts) > 2 and parts[2].startswith("a/") else ""
            new_path = parts[3][2:] if len(parts) > 3 and parts[3].startswith("b/") else old_path
            current_file = FileDiff(old_path=old_path, new_path=new_path)
            current_hunk = None
            files.append(current_file)
            continue

        if current_file is None:
            continue

        if raw_line.startswith("new file mode"):
            current_file.change_type = "added"
            continue
        if raw_line.startswith("deleted file mode"):
            current_file.change_type = "deleted"
            continue
        if raw_line.startswith("rename from "):
            current_file.change_type = "renamed"
            current_file.old_path = raw_line.removeprefix("rename from ").strip()
            continue
        if raw_line.startswith("rename to "):
            current_file.change_type = "renamed"
            current_file.new_path = raw_line.removeprefix("rename to ").strip()
            continue
        if raw_line.startswith("--- "):
            value = raw_line[4:]
            current_file.old_path = value[2:] if value.startswith("a/") else value
            continue
        if raw_line.startswith("+++ "):
            value = raw_line[4:]
            current_file.new_path = value[2:] if value.startswith("b/") else value
            continue
        if raw_line.startswith("@@ "):
            match = hunk_header.search(raw_line)
            if not match:
                continue
            old_start = int(match.group(1))
            old_count = int(match.group(2) or "1")
            new_start = int(match.group(3))
            new_count = int(match.group(4) or "1")
            current_hunk = DiffHunk(old_start, old_count, new_start, new_count)
            current_file.hunks.append(current_hunk)
            old_line = old_start
            new_line = new_start
            continue

        if current_hunk is None or raw_line.startswith("\\"):
            continue

        marker = raw_line[:1]
        value = raw_line[1:] if marker in {" ", "+", "-"} else raw_line
        if marker == "+":
            current_hunk.lines.append(DiffLine("added", value, new_line=new_line))
            new_line += 1
        elif marker == "-":
            current_hunk.lines.append(DiffLine("removed", value, old_line=old_line))
            old_line += 1
        else:
            current_hunk.lines.append(DiffLine("context", value, old_line=old_line, new_line=new_line))
            old_line += 1
            new_line += 1

    return files


def module_name_from_path(rel_path: str) -> str:
    normalized = rel_path.replace("\\", "/")
    if normalized.endswith("/__init__.py"):
        normalized = normalized[: -len("/__init__.py")]
    elif normalized.endswith(".py"):
        normalized = normalized[:-3]
    return normalized.replace("/", ".").strip(".")


def runtime_tags_for(file_path: str, symbol: str = "") -> list[str]:
    text = f"{file_path} {symbol}".lower()
    tags = set()
    checks = [
        ("settings", ["settings.py", ".settings", "config"]),
        ("task_execution", ["execution_time", "task_runner", "supervisor", "task"]),
        ("scheduler", ["scheduler", "schedule"]),
        ("db_connection", ["sql_alchemy", "sqlalchemy", "database", "db", "engine"]),
        ("orm", ["orm", "session"]),
        ("executor", ["executor"]),
        ("runtime_guard", ["block_orm_access", "__getattr__", "sys.modules", "monkey"]),
    ]
    for tag, needles in checks:
        if any(needle in text for needle in needles):
            tags.add(tag)
    return sorted(tags)


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _source_segment(source: str, node: ast.AST, lines: list[str]) -> str:
    try:
        return ast.get_source_segment(source, node) or ""
    except Exception:
        start = max(getattr(node, "lineno", 1) - 1, 0)
        end = getattr(node, "end_lineno", start + 1)
        return "\n".join(lines[start:end])


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    args = []
    for arg in node.args.args:
        ann = ""
        if arg.annotation is not None:
            try:
                ann = f": {ast.unparse(arg.annotation)}"
            except Exception:
                ann = ""
        args.append(f"{arg.arg}{ann}")
    ret = ""
    if node.returns is not None:
        try:
            ret = f" -> {ast.unparse(node.returns)}"
        except Exception:
            ret = ""
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    return f"{prefix} {node.name}({', '.join(args)}){ret}"


def _assignment_targets(node: ast.AST) -> list[str]:
    targets: list[ast.AST] = []
    if isinstance(node, ast.Assign):
        targets = list(node.targets)
    elif isinstance(node, ast.AnnAssign):
        targets = [node.target]
    elif isinstance(node, ast.AugAssign):
        targets = [node.target]
    names = []
    for target in targets:
        if isinstance(target, ast.Name):
            names.append(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            names.extend(item.id for item in target.elts if isinstance(item, ast.Name))
    return names


def parse_python_symbols(source: str, rel_path: str) -> list[SymbolInfo]:
    module = module_name_from_path(rel_path)
    lines = source.splitlines()
    symbols = [
        SymbolInfo(
            id=module,
            name=module.rsplit(".", 1)[-1] if module else rel_path,
            symbol_type="module",
            file_path=rel_path,
            module=module,
            start_line=1,
            end_line=max(len(lines), 1),
            code_hash=_hash_text(source),
            runtime_tags=runtime_tags_for(rel_path, module),
        )
    ]
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return symbols

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            for name in _assignment_targets(node):
                raw = _source_segment(source, node, lines)
                symbol_id = f"{module}.{name}" if module else name
                symbols.append(
                    SymbolInfo(
                        id=symbol_id,
                        name=name,
                        symbol_type="module_variable",
                        file_path=rel_path,
                        module=module,
                        start_line=getattr(node, "lineno", 1),
                        end_line=getattr(node, "end_lineno", getattr(node, "lineno", 1)),
                        code_hash=_hash_text(raw),
                        runtime_tags=runtime_tags_for(rel_path, symbol_id),
                    )
                )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            raw = _source_segment(source, node, lines)
            symbol_id = f"{module}.{node.name}" if module else node.name
            symbols.append(
                SymbolInfo(
                    id=symbol_id,
                    name=node.name,
                    symbol_type="function",
                    file_path=rel_path,
                    module=module,
                    start_line=node.lineno,
                    end_line=node.end_lineno or node.lineno,
                    code_hash=_hash_text(raw),
                    signature=_signature(node),
                    runtime_tags=runtime_tags_for(rel_path, symbol_id),
                )
            )
        elif isinstance(node, ast.ClassDef):
            raw = _source_segment(source, node, lines)
            class_id = f"{module}.{node.name}" if module else node.name
            symbols.append(
                SymbolInfo(
                    id=class_id,
                    name=node.name,
                    symbol_type="class",
                    file_path=rel_path,
                    module=module,
                    start_line=node.lineno,
                    end_line=node.end_lineno or node.lineno,
                    code_hash=_hash_text(raw),
                    runtime_tags=runtime_tags_for(rel_path, class_id),
                )
            )
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    raw = _source_segment(source, child, lines)
                    method_id = f"{class_id}.{child.name}"
                    symbols.append(
                        SymbolInfo(
                            id=method_id,
                            name=child.name,
                            symbol_type="method",
                            file_path=rel_path,
                            module=module,
                            start_line=child.lineno,
                            end_line=child.end_lineno or child.lineno,
                            code_hash=_hash_text(raw),
                            signature=_signature(child),
                            runtime_tags=runtime_tags_for(rel_path, method_id),
                        )
                    )
    return symbols


class ReferenceVisitor(ast.NodeVisitor):
    def __init__(self, source: str, rel_path: str):
        self.source = source
        self.lines = source.splitlines()
        self.rel_path = rel_path
        self.module = module_name_from_path(rel_path)
        self.import_aliases: dict[str, str] = {}
        self.scope_stack: list[str] = [self.module]
        self.references: list[ReferenceInfo] = []

    @property
    def current_scope(self) -> str:
        return self.scope_stack[-1]

    def snippet(self, line: int) -> str:
        if line <= 0 or line > len(self.lines):
            return ""
        return self.lines[line - 1].strip()[:300]

    def add_ref(self, ref_type: str, target: str, ref_name: str, line: int) -> None:
        if not target:
            return
        self.references.append(
            ReferenceInfo(
                file_path=self.rel_path,
                source_scope=self.current_scope,
                ref_type=ref_type,
                target=target,
                ref_name=ref_name,
                line=line,
                snippet=self.snippet(line),
            )
        )

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            local = alias.asname or alias.name.split(".")[0]
            self.import_aliases[local] = alias.name
            self.add_ref("import", alias.name, local, node.lineno)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = "." * node.level + (node.module or "")
        module = module.strip(".")
        for alias in node.names:
            local = alias.asname or alias.name
            target = f"{module}.{alias.name}" if module else alias.name
            self.import_aliases[local] = target
            self.add_ref("import", target, local, node.lineno)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._push_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._push_function(node)

    def _push_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        prefix = self.current_scope
        scope = f"{prefix}.{node.name}" if prefix else node.name
        self.scope_stack.append(scope)
        self.generic_visit(node)
        self.scope_stack.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        prefix = self.current_scope
        scope = f"{prefix}.{node.name}" if prefix else node.name
        self.scope_stack.append(scope)
        self.generic_visit(node)
        self.scope_stack.pop()

    def visit_Attribute(self, node: ast.Attribute) -> None:
        target = self.resolve_expr(node)
        if target:
            self.add_ref("attribute_access", target, self.expr_text(node), node.lineno)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        target = self.resolve_expr(node.func)
        if target:
            self.add_ref("call", target, self.expr_text(node.func), getattr(node, "lineno", 0))
            config_target = self.config_read_target(node, target)
            if config_target:
                self.add_ref("config_read", config_target, self.expr_text(node.func), node.lineno)
        self.generic_visit(node)

    def expr_text(self, node: ast.AST) -> str:
        try:
            return ast.unparse(node)
        except Exception:
            return ""

    def resolve_expr(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return self.import_aliases.get(node.id, f"{self.module}.{node.id}" if self.module else node.id)
        if isinstance(node, ast.Attribute):
            base = self.resolve_expr(node.value)
            if base:
                return f"{base}.{node.attr}"
        return ""

    def config_read_target(self, node: ast.Call, call_target: str) -> str:
        if not (
            call_target.endswith(".conf.get")
            or call_target.endswith(".conf.getboolean")
            or call_target.endswith(".conf.getint")
            or call_target.endswith(".conf.getfloat")
            or call_target.endswith(".os.environ.get")
            or call_target.endswith(".environ.get")
        ):
            return ""
        string_args = [arg.value for arg in node.args if isinstance(arg, ast.Constant) and isinstance(arg.value, str)]
        if call_target.endswith("environ.get") and string_args:
            return f"env.{string_args[0]}"
        if len(string_args) >= 2:
            return f"config.{string_args[0]}.{string_args[1]}"
        return ""


def parse_python_references(source: str, rel_path: str) -> list[ReferenceInfo]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    visitor = ReferenceVisitor(source, rel_path)
    visitor.visit(tree)
    return visitor.references


def discover_python_files(repo_path: Path) -> list[Path]:
    files: list[Path] = []
    for root, dirs, filenames in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRS and not d.startswith(".")]
        for filename in filenames:
            if filename.endswith(".py"):
                files.append(Path(root) / filename)
    return sorted(files)


def build_reference_index(repo_id: str, repo_path: Path, store: RepoKnowledgeStore) -> dict[str, int]:
    store.clear_code_graph(repo_id)
    all_symbols: list[SymbolInfo] = []
    all_refs: list[ReferenceInfo] = []
    for file_path in discover_python_files(repo_path):
        rel_path = file_path.relative_to(repo_path).as_posix()
        try:
            source = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        all_symbols.extend(parse_python_symbols(source, rel_path))
        all_refs.extend(parse_python_references(source, rel_path))
    store.save_symbols(repo_id, all_symbols)
    store.save_references(repo_id, all_refs)
    return {"symbol_count": len(all_symbols), "reference_count": len(all_refs)}


NOISY_REFERENCE_NEEDLES = {
    "__init__",
    "db",
    "env",
    "models",
    "reset",
    "test_db",
    "utils",
}
TARGETED_REFERENCE_FILE_LIMIT = 80


def _is_test_path(file_path: str) -> bool:
    normalized = file_path.replace("\\", "/")
    return "/tests/" in normalized or normalized.startswith("tests/")


def _is_noisy_reference_needle(needle: str) -> bool:
    if len(needle) < 3:
        return True
    if needle in NOISY_REFERENCE_NEEDLES:
        return True
    if needle.startswith("test_"):
        return True
    if needle.startswith("__") and needle.endswith("__") and needle != "__getattr__":
        return True
    return False


def _reference_needles_for_changed_symbols(changed_symbols: list[dict[str, Any]]) -> set[str]:
    needles: set[str] = set()
    for symbol in changed_symbols:
        symbol_id = str(symbol.get("id") or "")
        name = str(symbol.get("name") or "")
        file_path = str(symbol.get("file") or "")
        if _is_test_path(file_path):
            continue
        module = module_name_from_path(file_path) if file_path.endswith(".py") else ""
        for item in (symbol_id, name, module, module.rsplit(".", 1)[-1] if module else ""):
            if item and not _is_noisy_reference_needle(item):
                needles.add(item)
    return needles


def build_targeted_reference_index(
    repo_id: str,
    repo_path: Path,
    store: RepoKnowledgeStore,
    changed_symbols: list[dict[str, Any]],
) -> dict[str, int]:
    """Build a reverse-ref index only from files likely to mention changed symbols."""
    store.clear_code_graph(repo_id)
    needles = _reference_needles_for_changed_symbols(changed_symbols)
    if not needles:
        return {"symbol_count": 0, "reference_count": 0, "scanned_file_count": 0}

    all_symbols: list[SymbolInfo] = []
    all_refs: list[ReferenceInfo] = []
    candidates: list[tuple[int, str, Path, str]] = []
    changed_files = {
        str(symbol.get("file") or "").replace("\\", "/")
        for symbol in changed_symbols
        if symbol.get("file")
    }
    for file_path in discover_python_files(repo_path):
        rel_path = file_path.relative_to(repo_path).as_posix()
        try:
            source = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        matched = [needle for needle in needles if needle in source or needle in rel_path]
        if not matched:
            continue
        score = len(matched)
        if rel_path in changed_files:
            score += 100
        if not _is_test_path(rel_path):
            score += 10
        if "import airflow.settings" in source or "from airflow import settings" in source:
            score += 30
        if "SQL_ALCHEMY_CONN" in source or "block_orm_access" in source or "__getattr__" in source:
            score += 20
        candidates.append((score, rel_path, file_path, source))

    candidates.sort(key=lambda item: (-item[0], item[1]))
    selected = candidates[:TARGETED_REFERENCE_FILE_LIMIT]
    for _, rel_path, _, source in selected:
        all_symbols.extend(parse_python_symbols(source, rel_path))
        all_refs.extend(parse_python_references(source, rel_path))
    store.save_symbols(repo_id, all_symbols)
    store.save_references(repo_id, all_refs)
    return {
        "symbol_count": len(all_symbols),
        "reference_count": len(all_refs),
        "scanned_file_count": len(selected),
        "candidate_file_count": len(candidates),
        "targeted": 1,
    }


def references_to_edges(references: list[ReferenceInfo]) -> list[dict[str, str]]:
    return [
        {"source": ref.source_scope, "target": ref.target, "relation": ref.ref_type}
        for ref in references
        if ref.source_scope and ref.target
    ]


def build_chunk_graph_index(repo_id: str, repo_path: Path, store: RepoKnowledgeStore) -> dict[str, int]:
    all_symbols: list[SymbolInfo] = []
    all_refs: list[ReferenceInfo] = []
    for file_path in discover_python_files(repo_path):
        rel_path = file_path.relative_to(repo_path).as_posix()
        try:
            source = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        all_symbols.extend(parse_python_symbols(source, rel_path))
        all_refs.extend(parse_python_references(source, rel_path))
    store.clear_code_graph(repo_id)
    store.save_symbols(repo_id, all_symbols)
    store.save_references(repo_id, all_refs)
    store.replace_edges(repo_id, references_to_edges(all_refs))
    return {"symbol_count": len(all_symbols), "reference_count": len(all_refs)}


def changed_symbols_for_file(repo_path: Path, base: str, head: str, file_diff: FileDiff) -> list[dict[str, Any]]:
    rel_path = file_diff.path
    old_source = read_git_file(repo_path, base, file_diff.old_path)
    new_source = read_git_file(repo_path, head, file_diff.new_path)
    old_symbols = parse_python_symbols(old_source, file_diff.old_path) if old_source else []
    new_symbols = parse_python_symbols(new_source, file_diff.new_path) if new_source else []
    old_by_id = {symbol.id: symbol for symbol in old_symbols}
    new_by_id = {symbol.id: symbol for symbol in new_symbols}
    changed: list[dict[str, Any]] = []
    old_lines = file_diff.changed_old_lines()
    new_lines = file_diff.changed_new_lines()

    for symbol_id in sorted(set(old_by_id) | set(new_by_id)):
        old_sym = old_by_id.get(symbol_id)
        new_sym = new_by_id.get(symbol_id)
        sym = new_sym or old_sym
        if sym is None or sym.symbol_type == "module":
            continue

        change_type = ""
        if old_sym is None:
            change_type = "added"
        elif new_sym is None:
            change_type = "removed"
        elif old_sym.code_hash != new_sym.code_hash:
            change_type = "modified"
        else:
            old_touched = any(old_sym.start_line <= line <= old_sym.end_line for line in old_lines)
            new_touched = any(new_sym.start_line <= line <= new_sym.end_line for line in new_lines)
            if old_touched or new_touched:
                change_type = "touched"
        if not change_type:
            continue
        changed.append(
            {
                "id": symbol_id,
                "name": sym.name,
                "type": sym.symbol_type,
                "change_type": change_type,
                "file": rel_path,
                "old_lines": [old_sym.start_line, old_sym.end_line] if old_sym else None,
                "new_lines": [new_sym.start_line, new_sym.end_line] if new_sym else None,
                "runtime_tags": runtime_tags_for(rel_path, symbol_id),
            }
        )
    return changed


def expand_impacts(
    repo_id: str,
    changed_symbols: list[dict[str, Any]],
    store: RepoKnowledgeStore,
    max_depth: int = 3,
) -> list[dict[str, Any]]:
    paths: list[dict[str, Any]] = []
    seen_paths: set[tuple[str, str, str]] = set()

    for changed in changed_symbols:
        start_id = changed["id"]
        queue: list[tuple[str, list[dict[str, Any]], int]] = [
            (
                start_id,
                [
                    {
                        "id": start_id,
                        "role": "changed_symbol",
                        "type": changed["type"],
                        "file": changed["file"],
                        "runtime_tags": changed.get("runtime_tags", []),
                    }
                ],
                0,
            )
        ]
        while queue:
            current, nodes, depth = queue.pop(0)
            if depth >= max_depth:
                continue
            refs = store.find_references_to(repo_id, current, limit=40)
            if not refs and depth == 0:
                refs = store.find_references_by_prefix(repo_id, f"{current}.", limit=40)
            for ref in refs:
                next_symbol = ref["source_scope"]
                if not next_symbol or next_symbol == current:
                    continue
                key = (current, next_symbol, ref["ref_type"])
                if key in seen_paths:
                    continue
                seen_paths.add(key)
                tags = runtime_tags_for(ref["file_path"], next_symbol)
                node = {
                    "id": next_symbol,
                    "role": "direct_consumer" if depth == 0 else "upstream_consumer",
                    "relation": ref["ref_type"],
                    "target": ref["target"],
                    "file": ref["file_path"],
                    "line": ref["line"],
                    "snippet": ref["snippet"],
                    "runtime_tags": tags,
                }
                new_nodes = nodes + [node]
                paths.append(
                    {
                        "path_id": f"p{len(paths) + 1}",
                        "changed_symbol": start_id,
                        "nodes": new_nodes,
                        "score": score_path(changed, new_nodes),
                    }
                )
                queue.append((next_symbol, new_nodes, depth + 1))

    paths.sort(key=lambda item: item["score"], reverse=True)
    return paths[:10]


def score_path(changed: dict[str, Any], nodes: list[dict[str, Any]]) -> int:
    score = 0
    tag_set = {tag for node in nodes for tag in node.get("runtime_tags", [])}
    if tag_set & {"task_execution", "scheduler", "db_connection", "orm"}:
        score += 5
    if tag_set & {"settings", "runtime_guard", "executor"}:
        score += 4
    if any(node.get("relation") in {"call", "attribute_access", "config_read"} for node in nodes):
        score += 3
    if changed.get("change_type") in {"removed", "modified"}:
        score += 4
    if changed.get("type") == "module_variable":
        score += 3
    if any("/tests/" in f"/{node.get('file', '')}" or node.get("file", "").startswith("tests/") for node in nodes):
        score -= 3
    score -= max(len(nodes) - 2, 0)
    return score


def evaluate_risks(
    changed_symbols: list[dict[str, Any]],
    impact_paths: list[dict[str, Any]],
    file_diffs: list[FileDiff],
) -> list[dict[str, Any]]:
    risks: list[dict[str, Any]] = []
    changed_text = " ".join([s["id"] for s in changed_symbols] + [fd.path for fd in file_diffs]).lower()
    all_targets = " ".join(
        node.get("target", "") for path in impact_paths for node in path.get("nodes", [])
    ).lower()
    all_tags = {
        tag
        for path in impact_paths
        for node in path.get("nodes", [])
        for tag in node.get("runtime_tags", [])
    }

    if any(s["type"] == "module_variable" and s["change_type"] == "removed" for s in changed_symbols):
        risks.append(
            risk(
                "module_global_removed",
                "high",
                "A module-level variable was removed. Runtime attribute access from other modules may break.",
                impact_paths,
            )
        )
    if "settings.py" in changed_text and ("settings" in all_targets or impact_paths):
        risks.append(
            risk(
                "settings_attribute_compat",
                "high",
                "Settings/config module attribute exposure changed and may cause runtime AttributeError in consumers.",
                impact_paths,
            )
        )
    if "sql_alchemy_conn" in changed_text or "sql_alchemy_conn" in all_targets or {"db_connection", "orm"} & all_tags:
        risks.append(
            risk(
                "config_db_chain_changed",
                "high",
                "Database connection or ORM configuration chain changed; task initialization or execution may be affected.",
                impact_paths,
            )
        )
    if any(token in changed_text or token in all_targets for token in ["__getattr__", "sys.modules", "block_orm_access"]):
        risks.append(
            risk(
                "runtime_global_mutation",
                "high",
                "Runtime module proxying or global access control changed; task execution behavior may change.",
                impact_paths,
            )
        )
    return risks


def risk(rule: str, level: str, message: str, paths: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rule": rule,
        "level": level,
        "message": message,
        "evidence_refs": [path["path_id"] for path in paths[:3]],
    }


def generate_report(
    diff_summary: dict[str, Any],
    changed_symbols: list[dict[str, Any]],
    impact_paths: list[dict[str, Any]],
    risks: list[dict[str, Any]],
) -> str:
    high_risks = [item for item in risks if item["level"] == "high"]
    lines = [
        "## Conclusion",
        (
            f"This change hit {len(high_risks)} high-risk rule(s)."
            if high_risks
            else "No high-risk v1 rule was triggered, but key changed paths should still be tested."
        ),
    ]
    if high_risks:
        lines.append("Main risks concentrate around settings/config compatibility, database/ORM setup, or task execution paths.")

    lines.extend(["", "## Changed Symbols"])
    if changed_symbols:
        for symbol in changed_symbols[:20]:
            lines.append(f"- `{symbol['id']}` ({symbol['type']}) `{symbol['change_type']}` in `{symbol['file']}`")
    else:
        lines.append("- No Python symbol-level changes were identified.")

    lines.extend(["", "## Impact Paths"])
    if impact_paths:
        for path in impact_paths[:5]:
            readable = " -> ".join(f"`{node['id']}`" for node in path["nodes"])
            lines.append(f"- {path['path_id']} score={path['score']}: {readable}")
            for node in path["nodes"][1:3]:
                if node.get("file"):
                    lines.append(f"  - Evidence: `{node['file']}:L{node.get('line', 0)}` {node.get('snippet', '')}")
    else:
        lines.append("- No explicit impact path was expanded from the reverse reference index.")

    lines.extend(["", "## Risk Judgement"])
    if risks:
        for item in risks:
            lines.append(f"- [{item['level']}] `{item['rule']}`: {item['message']}")
    else:
        lines.append("- No built-in v1 risk rule matched.")

    lines.extend(["", "## Suggested Verification"])
    lines.append("- Cover config/settings loading and compatibility behavior for changed files.")
    lines.append("- Cover direct consumers in task execution, scheduler, database, ORM, or session initialization paths.")
    lines.append("- Add regression coverage for module attribute access if settings/config symbols changed.")

    lines.extend(["", "## Evidence Summary"])
    lines.append(f"- Diff files: {diff_summary.get('file_count', 0)}")
    lines.append(f"- Python files: {diff_summary.get('python_file_count', 0)}")
    lines.append(f"- Changed symbols: {len(changed_symbols)}")
    lines.append(f"- Impact paths: {len(impact_paths)}")
    return "\n".join(lines)


def answer_followup(question: str, session: dict[str, Any]) -> str:
    data = session.get("data", {})
    paths = data.get("impact_paths", [])
    risks = data.get("risks", [])
    changed = data.get("changed_symbols", [])
    q = question.lower()

    lines = ["## Answer"]
    if "why" in q or "impact" in q or "risk" in q or "为什么" in question or "影响" in question:
        lines.append("The analysis connects changed symbols to runtime consumers through the reverse reference index, then applies risk rules to the resulting paths.")
        for path in paths[:3]:
            readable = " -> ".join(f"`{node['id']}`" for node in path["nodes"])
            lines.append(f"- {path['path_id']}: {readable}")
    elif "trigger" in q or "condition" in q or "触发" in question:
        lines.append("Likely trigger conditions include:")
        lines.append("- Runtime access to changed settings/config module attributes.")
        lines.append("- Task execution, scheduler, database, or ORM initialization paths calling an impacted consumer.")
        lines.append("- Compatibility attributes no longer being exposed in the same way.")
    elif "test" in q or "verify" in q or "测试" in question or "验证" in question:
        lines.append("Recommended tests:")
        lines.append("- Settings/config attribute compatibility tests.")
        lines.append("- Direct consumer startup or call-path tests.")
        lines.append("- Database connection, ORM/session, and task execution regression tests when those tags appear.")
    else:
        lines.append("This session can answer based on saved changed symbols, impact paths, and triggered risk rules.")

    if risks:
        lines.extend(["", "## Matched Risks"])
        for item in risks:
            lines.append(f"- `{item['rule']}`: {item['message']}")
    if changed:
        lines.extend(["", "## Relevant Changes"])
        for symbol in changed[:8]:
            lines.append(f"- `{symbol['id']}` ({symbol['change_type']})")
    return "\n".join(lines)


def analyze_diff(repo_id: str, repo_path: Path, base: str, head: str, store: RepoKnowledgeStore) -> dict[str, Any]:
    patch_text = run_git_diff(repo_path, base, head)
    file_diffs = parse_unified_diff(patch_text)

    changed_symbols: list[dict[str, Any]] = []
    for file_diff in file_diffs:
        if file_diff.path.endswith(".py"):
            changed_symbols.extend(changed_symbols_for_file(repo_path, base, head, file_diff))

    build_stats = build_targeted_reference_index(repo_id, repo_path, store, changed_symbols)
    impact_paths = expand_impacts(repo_id, changed_symbols, store)
    risks = evaluate_risks(changed_symbols, impact_paths, file_diffs)
    diff_summary = {
        "base": base,
        "head": head,
        "file_count": len(file_diffs),
        "python_file_count": len([fd for fd in file_diffs if fd.path.endswith(".py")]),
        "changed_files": [fd.to_dict() for fd in file_diffs],
        "index_stats": build_stats,
    }
    report = generate_report(diff_summary, changed_symbols, impact_paths, risks)
    session_id = uuid.uuid4().hex[:16]
    data = {
        "diff_summary": diff_summary,
        "changed_symbols": changed_symbols,
        "impact_paths": impact_paths,
        "risks": risks,
        "report": report,
    }
    store.save_analysis_session(session_id, repo_id, base, head, data)
    return {"session_id": session_id, **data}


def parse_github_pr_url(pr_url: str) -> tuple[str, str, str]:
    parsed = urlparse(pr_url)
    if parsed.netloc.lower() != "github.com":
        raise ValueError("Only github.com PR URLs are supported in v1.")
    parts = [part for part in parsed.path.strip("/").split("/") if part]
    if len(parts) < 4 or parts[2] != "pull" or not parts[3].isdigit():
        raise ValueError("Invalid GitHub PR URL. Expected https://github.com/{owner}/{repo}/pull/{number}.")
    return parts[0], parts[1], parts[3]


def prepare_github_pr_repo(pr_url: str, token: str | None = None) -> tuple[Path, str, str, str]:
    owner, repo, pr_number = parse_github_pr_url(pr_url)
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    response = requests.get(
        f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}",
        headers=headers,
        timeout=30,
    )
    if response.status_code == 404:
        raise ValueError("GitHub PR was not found or the token cannot access it.")
    if response.status_code >= 400:
        raise ValueError(f"GitHub PR lookup failed: HTTP {response.status_code} {response.text[:200]}")
    pr_data = response.json()
    base_sha = pr_data.get("base", {}).get("sha")
    head_sha = pr_data.get("head", {}).get("sha")
    if not base_sha or not head_sha:
        raise ValueError("GitHub PR response did not include base/head SHAs.")

    root = Path(get_adalflow_default_root_path()) / "pr_repos"
    root.mkdir(parents=True, exist_ok=True)
    repo_dir = root / f"github_{owner}_{repo}_{pr_number}"
    clone_url = f"https://github.com/{owner}/{repo}.git"
    if token:
        clone_url = f"https://{token}@github.com/{owner}/{repo}.git"

    if repo_dir.exists() and not (repo_dir / ".git").exists():
        shutil.rmtree(repo_dir)
    if not repo_dir.exists():
        proc = _run(["git", "clone", "--no-checkout", clone_url, str(repo_dir)])
        if proc.returncode != 0:
            raise ValueError(_sanitize_git_error(proc.stderr.strip(), token) or "git clone failed")

    fetch = _run(
        [
            "git",
            "-C",
            str(repo_dir),
            "fetch",
            "origin",
            f"{head_sha}:refs/heads/pr-{pr_number}",
            f"{base_sha}:refs/heads/pr-{pr_number}-base",
        ]
    )
    if fetch.returncode != 0:
        raise ValueError(_sanitize_git_error(fetch.stderr.strip(), token) or "git fetch PR refs failed")

    checkout = _run(["git", "-C", str(repo_dir), "checkout", "-f", f"pr-{pr_number}"])
    if checkout.returncode != 0:
        raise ValueError(_sanitize_git_error(checkout.stderr.strip(), token) or "git checkout PR head failed")
    return repo_dir, f"pr-{pr_number}-base", f"pr-{pr_number}", default_repo_id(pr_url)


def analyze_github_pr(pr_url: str, token: str | None, store: RepoKnowledgeStore) -> dict[str, Any]:
    repo_path, base, head, repo_id = prepare_github_pr_repo(pr_url, token)
    return analyze_diff(repo_id, repo_path, base, head, store)
