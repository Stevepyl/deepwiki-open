#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import ast
import argparse
import logging
from typing import List, Dict, Any

# 粗略的 embedding token 上限（只用来做阈值演示）
MAX_EMBEDDING_TOKENS = 8192


def count_tokens(text: str) -> int:
    """
    粗略 token 计数；如果有 tiktoken 就用，没有的话用 len(text)//4 近似。
    """
    try:
        import tiktoken

        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        # 粗略：约 4 个字符一个 token
        return len(text) // 4


def split_python_file_by_ast(file_path: str) -> List[Dict[str, Any]]:
    """
    读取一个 .py 文件，用 AST 按函数 / 类级别拆分成多个块。
    返回每个块的信息：text, symbol_name, ast_type, start_line, end_line, token_count。
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    rel_path = os.path.basename(file_path)

    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        logging.warning("AST 解析失败，退回整文件: %s (%s)", rel_path, e)
        tok = count_tokens(content)
        if tok > MAX_EMBEDDING_TOKENS * 10:
            logging.warning(
                "整文件 token=%s 超过阈值，直接丢弃: %s", tok, rel_path
            )
            return []

        return [
            {
                "text": content,
                "file_path": rel_path,
                "symbol_name": None,
                "ast_type": "Module",
                "start_line": 1,
                "end_line": len(content.splitlines()),
                "token_count": tok,
            }
        ]

    lines = content.splitlines()
    chunks: List[Dict[str, Any]] = []

    def node_text_span(node, next_node_end: int) -> str:
        start = getattr(node, "lineno", None)
        end = getattr(node, "end_lineno", None)

        if start is None:
            return ""

        if end is None:
            end = next_node_end

        start_idx = max(start - 1, 0)
        end_idx = min(end, len(lines))

        if end_idx <= start_idx:
            return ""

        return "\n".join(lines[start_idx:end_idx])

    # 只看顶层函数 / 类定义
    top_level_nodes = [
        n
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]

    for idx, node in enumerate(top_level_nodes):
        # 估一个 end 行号（如果没有 end_lineno，用下一个节点或文件结尾）
        if getattr(node, "end_lineno", None) is not None:
            next_end = node.end_lineno
        else:
            if idx + 1 < len(top_level_nodes):
                next_node = top_level_nodes[idx + 1]
                next_end = getattr(next_node, "lineno", len(lines)) - 1
            else:
                next_end = len(lines)

        snippet = node_text_span(node, next_end).strip("\n")
        if not snippet:
            continue

        tok = count_tokens(snippet)
        if tok > MAX_EMBEDDING_TOKENS * 10:
            logging.warning(
                "丢弃过大函数/类块 token=%s: %s", tok, rel_path
            )
            continue

        symbol_name = getattr(node, "name", None)
        start_line = getattr(node, "lineno", None)
        end_line = getattr(node, "end_lineno", None)

        chunks.append(
            {
                "text": snippet,
                "file_path": rel_path,
                "symbol_name": symbol_name,
                "ast_type": type(node).__name__,
                "start_line": int(start_line) if start_line is not None else None,
                "end_line": int(end_line) if end_line is not None else None,
                "token_count": tok,
            }
        )

    # 如果什么都没拆出来，就退回整文件
    if not chunks:
        tok = count_tokens(content)
        if tok > MAX_EMBEDDING_TOKENS * 10:
            logging.warning(
                "整文件 token=%s 超过阈值，直接丢弃: %s", tok, rel_path
            )
            return []

        chunks.append(
            {
                "text": content,
                "file_path": rel_path,
                "symbol_name": None,
                "ast_type": "Module",
                "start_line": 1,
                "end_line": len(lines),
                "token_count": tok,
            }
        )

    return chunks


def main():
    parser = argparse.ArgumentParser(
        description="简单 AST 函数/类级别切分工具（单文件）"
    )
    parser.add_argument(
        "file",
        help="要分析的 Python 源文件路径，例如: api/data_pipeline.py",
    )
    parser.add_argument(
        "--show-snippet",
        action="store_true",
        help="是否打印每个 chunk 的源码片段",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    file_path = os.path.abspath(args.file)
    if not os.path.isfile(file_path):
        raise SystemExit(f"文件不存在: {file_path}")

    chunks = split_python_file_by_ast(file_path)

    logging.info("文件: %s", file_path)
    logging.info("共拆分出 %d 个 AST 块", len(chunks))

    for i, c in enumerate(chunks, start=1):
        title = (
            f"{c['file_path']}::{c['symbol_name']}"
            if c["symbol_name"]
            else c["file_path"]
        )
        logging.info(
            "[%02d] %s (%s, 行 %s-%s, tokens=%s)",
            i,
            title,
            c["ast_type"],
            c["start_line"],
            c["end_line"],
            c["token_count"],
        )
        if args.show_snippet:
            print("    --- snippet start ---")
            print(c["text"])
            print("    --- snippet end ---")


if __name__ == "__main__":
    main()
