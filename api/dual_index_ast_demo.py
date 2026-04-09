#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
用 AST 切出来的代码块，验证“双索引”（语义 + 符号）检索的一个小 demo。

使用方式（在仓库根目录）：

    uv run python -m api.dual_index_ast_demo api/data_pipeline.py \
        -q "怎么统计文本 token 数？" \
        -q "如何从 Git 仓库下载代码？" \
        --k 5 --alpha 0.6

语义路：按代码片段文本内容做 bag-of-words 余弦相似度；
符号路：按函数名 / 类名与 query 中 token 的匹配程度打分；
最终得分： final = alpha * semantic + (1-alpha) * symbol。
"""

import argparse
import logging
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

from api.ast_chunk_poc import split_python_file_by_ast


@dataclass
class CodeItem:
    """
    由 AST 切出来的一个代码单元（函数 / 类 / 模块块）。
    """

    id: int
    file_path: str
    symbol: str
    text: str


def tokenize(text: str) -> List[str]:
    """
    非常简单的 token 化：仅保留标识符风格的单词，并统一为小写。
    这里只是为了做一个粗粒度的“语义路”验证，后续可以替换成真正 embedding。
    """
    return [t.lower() for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text)]


def bow_vector(tokens: List[str]) -> Dict[str, int]:
    """构建一个 bag-of-words 计数向量。"""
    vec: Dict[str, int] = {}
    for t in tokens:
        vec[t] = vec.get(t, 0) + 1
    return vec


def cosine_sim(v1: Dict[str, int], v2: Dict[str, int]) -> float:
    """对稀疏 bag-of-words 做余弦相似度。"""
    if not v1 or not v2:
        return 0.0

    dot = 0.0
    for k, w in v1.items():
        if k in v2:
            dot += w * v2[k]

    norm1 = math.sqrt(sum(w * w for w in v1.values()))
    norm2 = math.sqrt(sum(w * w for w in v2.values()))

    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0

    return dot / (norm1 * norm2)


class DualIndex:
    """
    简单双索引：

    - 语义索引：对 CodeItem.text 做 bag-of-words 余弦相似度
    - 符号索引：对 CodeItem.symbol 与 query token 的匹配程度打分
    """

    def __init__(self, items: List[CodeItem]) -> None:
        self.items = items

        # 预计算语义向量
        self.semantic_vectors: List[Dict[str, int]] = [
            bow_vector(tokenize(item.text)) for item in self.items
        ]

    def _symbol_scores(self, query_tokens: List[str]) -> List[float]:
        """
        “符号路”打分规则：
        - 若 symbol 与 query 中某个 token 完全相等：1.0 分
        - 若 query token 作为子串出现在 symbol 中：0.5 分
        - 否则：0 分
        """
        scores = [0.0] * len(self.items)
        qset = set(query_tokens)

        for idx, item in enumerate(self.items):
            name = (item.symbol or "").lower()
            if not name:
                continue

            if name in qset:
                scores[idx] = 1.0
                continue

            for qt in qset:
                if qt and qt in name:
                    scores[idx] = max(scores[idx], 0.5)

        return scores

    def retrieve(
        self, query: str, k: int = 5, alpha: float = 0.6
    ) -> List[Tuple[CodeItem, float, float, float]]:
        """
        执行一次双索引检索。

        Args:
            query: 自然语言查询
            k: 返回的 Top-K 个结果
            alpha: 语义路权重（0~1），1 表示只看语义，0 表示只看符号

        Returns:
            列表，每个元素为 (CodeItem, final_score, semantic_score, symbol_score)
        """
        q_tokens = tokenize(query)
        q_vec = bow_vector(q_tokens)

        # 语义路得分
        sem_scores = [cosine_sim(q_vec, v) for v in self.semantic_vectors]

        # 符号路得分
        sym_scores = self._symbol_scores(q_tokens)

        # 融合
        final_scores: List[float] = [
            alpha * s_sem + (1.0 - alpha) * s_sym
            for s_sem, s_sym in zip(sem_scores, sym_scores)
        ]

        # 选 Top-K
        indices = sorted(
            range(len(self.items)),
            key=lambda i: final_scores[i],
            reverse=True,
        )[:k]

        return [
            (self.items[i], final_scores[i], sem_scores[i], sym_scores[i])
            for i in indices
        ]


def build_items_from_ast(file_path: str) -> List[CodeItem]:
    """
    使用现有的 split_python_file_by_ast，把一个 .py 文件切成多个 CodeItem。
    """
    abs_path = os.path.abspath(file_path)
    chunks = split_python_file_by_ast(abs_path)

    items: List[CodeItem] = []
    for idx, c in enumerate(chunks):
        items.append(
            CodeItem(
                id=idx,
                file_path=c.get("file_path") or os.path.basename(abs_path),
                symbol=c.get("symbol_name") or "",
                text=c.get("text") or "",
            )
        )

    return items


def main() -> None:
    parser = argparse.ArgumentParser(
        description="基于 AST 代码块的“双索引”（语义 + 符号）检索 demo"
    )
    parser.add_argument(
        "file",
        help="要分析的 Python 源码文件路径，例如: api/data_pipeline.py",
    )
    parser.add_argument(
        "-q",
        "--query",
        action="append",
        required=True,
        help="要测试的查询语句（可重复多次）",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="每个查询返回的 Top-K 结果数量（默认 5）",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.6,
        help="语义路权重 alpha（0~1，默认 0.6）",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    items = build_items_from_ast(args.file)
    if not items:
        logging.error("没有从文件 %s 中切出任何 AST 代码块", args.file)
        return

    logging.info("文件 %s 共切出了 %d 个 AST 代码块", args.file, len(items))

    index = DualIndex(items)

    for q in args.query:
        print("\n==== Query:", q, "====")
        results = index.retrieve(q, k=args.k, alpha=args.alpha)
        for item, final_s, sem_s, sym_s in results:
            first_line = item.text.strip().splitlines()[0] if item.text.strip() else ""
            print(
                f"- {item.symbol or '<module>'} ({item.file_path}) "
                f"final={final_s:.3f}, sem={sem_s:.3f}, sym={sym_s:.3f}"
            )
            if first_line:
                print("  ", first_line[:80])


if __name__ == "__main__":
    main()

