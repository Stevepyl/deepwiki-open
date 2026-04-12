import ast
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class AstChunk:
    """
    Represents a logical chunk of Python code derived from the AST.

    This is a lightweight internal structure. The caller is expected to
    turn it into adalflow `Document` objects and attach it to meta_data.
    """

    text: str
    file_path: str
    module_path: str
    symbol_name: Optional[str]
    parent_symbol: Optional[str]
    symbol_full_name: Optional[str]
    ast_type: str
    start_line: int
    end_line: int
    token_count: int


def _infer_module_path(file_path: str) -> str:
    """
    Infer a dotted module path from a repo-relative file path like
    `api/data_pipeline.py` -> `api.data_pipeline`.
    """
    # Normalize separators and strip extension
    rel = file_path.replace(os.sep, "/")
    if rel.endswith(".py"):
        rel = rel[:-3]
    # Drop leading "./"
    if rel.startswith("./"):
        rel = rel[2:]
    # Convert to dotted path
    parts = [p for p in rel.split("/") if p not in ("", ".")]
    return ".".join(parts) if parts else rel


def _node_text_span(lines: List[str], node: ast.AST, default_end: int) -> str:
    """
    Extract source text for a node from the original file lines.
    """
    start = getattr(node, "lineno", None)
    end = getattr(node, "end_lineno", None)

    if start is None:
        return ""

    if end is None:
        end = default_end

    start_idx = max(start - 1, 0)
    end_idx = min(end, len(lines))

    if end_idx <= start_idx:
        return ""

    return "\n".join(lines[start_idx:end_idx])


def _walk_relevant_nodes(tree: ast.AST) -> Iterable[Dict[str, Any]]:
    """
    Walk the AST and yield function / async function / class nodes
    together with their parent symbol information.

    The minimal unit we care about:
    - Top-level functions
    - Top-level classes
    - Methods inside classes
    - Nested / inner functions
    """

    # For parent tracking we keep a stack of symbol names
    parent_stack: List[Optional[str]] = [None]

    def visit(node: ast.AST):
        nonlocal parent_stack

        is_function = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        is_class = isinstance(node, ast.ClassDef)

        if is_function or is_class:
            symbol_name = getattr(node, "name", None)
            parent_symbol = parent_stack[-1]

            yield {
                "node": node,
                "symbol_name": symbol_name,
                "parent_symbol": parent_symbol,
                "ast_type": type(node).__name__,
            }

            # Recurse with updated parent stack
            parent_stack.append(symbol_name)
            for child in node.body:  # type: ignore[attr-defined]
                yield from visit(child)
            parent_stack.pop()
        else:
            # Keep descending so we can capture inner functions
            for child in ast.iter_child_nodes(node):
                yield from visit(child)

    for top_level in tree.body:  # type: ignore[attr-defined]
        yield from visit(top_level)


def split_python_source_by_ast(
    source: str,
    file_path: str,
    count_tokens_func,
    max_embedding_tokens: int,
) -> List[AstChunk]:
    """
    Split Python source code into AST-level chunks.

    Args:
        source: Full text of the Python file.
        file_path: Repo-relative path (used for meta_data).
        count_tokens_func: Callable(text) -> int, uses project-specific tokenizer.
        max_embedding_tokens: Token upper bound for a single chunk. Very large
            functions beyond this *10 will be skipped entirely.
    """
    rel_path = file_path
    module_path = _infer_module_path(rel_path)

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        logger.warning(
            "AST parse failed, falling back to whole file: %s (%s)", rel_path, e
        )
        tok = count_tokens_func(source)
        if tok > max_embedding_tokens * 10:
            logger.warning(
                "File token=%s exceeds hard limit, skipping: %s", tok, rel_path
            )
            return []

        lines = source.splitlines()
        return [
            AstChunk(
                text=source,
                file_path=rel_path,
                module_path=module_path,
                symbol_name=None,
                parent_symbol=None,
                symbol_full_name=module_path or None,
                ast_type="Module",
                start_line=1,
                end_line=len(lines),
                token_count=tok,
            )
        ]

    lines = source.splitlines()
    chunks: List[AstChunk] = []

    for info in _walk_relevant_nodes(tree):
        node = info["node"]
        symbol_name = info["symbol_name"]
        parent_symbol = info["parent_symbol"]
        ast_type = info["ast_type"]

        # Robust end line handling
        if getattr(node, "end_lineno", None) is not None:
            next_end = int(node.end_lineno)  # type: ignore[attr-defined]
        else:
            next_end = len(lines)

        snippet = _node_text_span(lines, node, next_end).strip("\n")
        if not snippet:
            continue

        tok = count_tokens_func(snippet)
        if tok > max_embedding_tokens * 10:
            logger.warning(
                "Skipping overly large AST chunk token=%s: %s.%s",
                tok,
                module_path,
                symbol_name,
            )
            continue

        start_line = getattr(node, "lineno", 1)
        end_line = getattr(node, "end_lineno", next_end)

        # Build a readable full symbol name
        if module_path and symbol_name:
            if parent_symbol:
                symbol_full_name = f"{module_path}.{parent_symbol}.{symbol_name}"
            else:
                symbol_full_name = f"{module_path}.{symbol_name}"
        elif module_path:
            symbol_full_name = module_path
        else:
            symbol_full_name = symbol_name

        chunks.append(
            AstChunk(
                text=snippet,
                file_path=rel_path,
                module_path=module_path,
                symbol_name=symbol_name,
                parent_symbol=parent_symbol,
                symbol_full_name=symbol_full_name,
                ast_type=ast_type,
                start_line=int(start_line),
                end_line=int(end_line),
                token_count=tok,
            )
        )

    # Fallback: no relevant nodes found, treat whole file as one chunk
    if not chunks:
        tok = count_tokens_func(source)
        if tok > max_embedding_tokens * 10:
            logger.warning(
                "File token=%s exceeds hard limit, skipping: %s", tok, rel_path
            )
            return []

        chunks.append(
            AstChunk(
                text=source,
                file_path=rel_path,
                module_path=module_path,
                symbol_name=None,
                parent_symbol=None,
                symbol_full_name=module_path or None,
                ast_type="Module",
                start_line=1,
                end_line=len(lines),
                token_count=tok,
            )
        )

    return chunks


def split_python_file_by_ast(
    file_path: str,
    count_tokens_func,
    max_embedding_tokens: int,
) -> List[AstChunk]:
    """
    Convenience wrapper to read a file and split it into AST chunks.
    `file_path` is expected to be an absolute path; the returned `AstChunk`
    will contain a repo-relative path derived from it if the caller wishes.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Let the caller decide what "repo relative" means; here we just keep
    # the basename, and the caller can overwrite file_path if needed.
    rel = os.path.basename(file_path)
    return split_python_source_by_ast(
        source=content,
        file_path=rel,
        count_tokens_func=count_tokens_func,
        max_embedding_tokens=max_embedding_tokens,
    )

