# Splitter 改进方案：基于代码结构（AST）的切分

## 背景

当前项目使用 `TextSplitter`（来自 `adalflow`）按词（word）进行滑动窗口切分：

- 配置：`chunk_size=350`，`chunk_overlap=100`，`split_by="word"`
- 问题：词级别切分不感知代码语义边界，函数/类可能被截断在两个 chunk 中，导致 RAG 检索到不完整的代码单元

## 可行性结论

**完全可行，改动仅 1 行。**

`api/data_pipeline.py:405` 的 `prepare_data_pipeline` 函数中：

```python
# 当前
splitter = TextSplitter(**configs["text_splitter"])

# 替换为
splitter = AstCodeSplitter(...)
```

`adal.Sequential` 对 splitter 的类型无感知，只要满足接口契约即可。Embedding 管道、`LocalDB`、`db.transform()` 全部不需要修改。

---

## 接口契约

自定义 splitter 的输出 `Document` 必须包含以下字段（与 `TextSplitter.call()` 保持一致）：

```python
Document(
    text=chunk_text,           # 切分后的代码文本
    meta_data=meta_data,       # 从父文档 deepcopy，保留文件路径等元信息
    parent_doc_id=f"{doc.id}", # 指向原文件，RAG 检索时用于溯源
    order=i,                   # chunk 在原文件中的顺序
    vector=[],                 # 空列表，等待后续 embedder 填充
)
```

---

## 方案对比

### 方案 A：继承 `DataComponent`（推荐）

```python
from adalflow.core.component import DataComponent
from adalflow.core.types import Document
from typing import List

class AstCodeSplitter(DataComponent):
    def call(self, documents: List[Document]) -> List[Document]:
        split_docs = []
        for doc in documents:
            chunks = self._parse_ast(doc.text, doc.meta_data)
            split_docs.extend(chunks)
        return split_docs

    def _parse_ast(self, text: str, meta_data: dict) -> List[Document]:
        # 按语言选择 AST parser，提取函数/类作为 chunk
        ...
```

**优点：**
- 与 AdalFlow 生态完全兼容（支持序列化、日志、`_extra_repr` 等）
- 可持有状态（如 `max_chunk_size`、`fallback_splitter` 等配置参数）
- 类型清晰，易于测试

**缺点：**
- 需要继承 AdalFlow 内部类，有一定耦合

---

### 方案 B：`FuncDataComponent` 包装（轻量）

```python
from adalflow.core.component import FuncDataComponent

def ast_split(documents: List[Document]) -> List[Document]:
    ...

splitter = FuncDataComponent(fun=ast_split)
```

**优点：**
- 零继承，实现与框架完全解耦
- 适合快速原型验证

**缺点：**
- 无法持有配置状态（参数需通过闭包传入）
- 不支持序列化（`FuncDataComponent` 用 `EntityMapping` 注册函数名）

---

### 方案 C：混合策略（AST + 词级回退）

对可解析语言（Python、JS/TS）用 AST 切分；对无法解析的文件（配置文件、Markdown、未知语言）回退到现有 `TextSplitter`。

```python
class HybridCodeSplitter(DataComponent):
    def __init__(self, fallback_splitter: TextSplitter):
        super().__init__()
        self.fallback_splitter = fallback_splitter

    def call(self, documents: List[Document]) -> List[Document]:
        ast_docs, fallback_docs = self._partition_by_language(documents)
        result = self._split_with_ast(ast_docs)
        result += self.fallback_splitter.call(fallback_docs)
        return result
```

**优点：**
- 渐进式迁移，不破坏对非代码文件的处理
- 风险最低

**缺点：**
- 实现复杂度最高
- 两种 chunk 的粒度不一致，可能影响检索质量

---

## AST 实现参考

| 语言 | 推荐工具 | 说明 |
|---|---|---|
| Python | 内置 `ast` 模块 | 零依赖，提取 `FunctionDef`、`ClassDef` |
| JavaScript / TypeScript | `tree-sitter`（`py-tree-sitter`） | 支持 100+ 语言，统一接口 |
| Go / Rust / Java | `tree-sitter` | 同上 |
| 其他 / 无法解析 | 回退 `TextSplitter` | 兜底策略 |

语言识别可复用 `api/config/repo.json` 中已有的文件扩展名过滤配置，或读取 `doc.meta_data["title"]` 的文件后缀。

---

## 预期收益

- RAG 检索到的 chunk 更可能是完整的函数/类，减少 LLM 生成 wiki 时因截断导致的幻觉
- chunk 数量可能减少（大函数不会被切为多个词级 chunk），降低嵌入 API 调用次数
- `parent_doc_id` + `order` 天然保留调用栈溯源能力

## 风险点

- 超大函数（>350词）仍需在 AST 级别二次切分，需要处理嵌套结构
- `tree-sitter` 引入新依赖，需更新 `pyproject.toml`
- 不同语言的 AST 节点类型命名不一致，需要 per-language 映射表
