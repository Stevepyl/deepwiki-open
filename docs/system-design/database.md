# 数据库设计

## 概述

DeepWiki 不使用传统关系型数据库，而是采用 **FAISS 向量存储 + 本地文件系统**的组合，以每个仓库一个 pickle 文件的形式持久化。底层由 `adalflow` 的 `LocalDB` 类管理。

### 存储目录结构

```
~/.adalflow/
├── repos/{owner}_{repo}/         # 克隆的仓库源码
├── databases/{owner}_{repo}.pkl  # 序列化的 LocalDB（含向量索引）
└── wikicache/                    # 生成的 Wiki 缓存（前端使用）
```

---

## LocalDB

`LocalDB`（`adalflow.core.db.LocalDB`）是 adalflow 提供的**带数据变换管道的内存存储容器**，附带 pickle 序列化持久化能力。它本身不是向量数据库，也不做相似度搜索——它只负责存储和组织数据，FAISS 的搜索能力由上层的 `FAISSRetriever` 提供。

### 数据结构

```python
@dataclass
class LocalDB(Component):
    name: str                                     # 数据库标识符
    items: List[Document]                         # 原始表 — 每个文件对应一条记录
    transformed_items: Dict[str, List[Document]]  # 派生表 — key 为变换名称
    transformer_setups: Dict[str, Component]      # 已注册的变换管道
    mapper_setups: Dict[str, Callable]            # 可选的前处理映射函数
    index_path: str                               # pkl 文件自身的保存路径
```

`transformed_items` 是 `Dict[str, List]`，支持同一份原始数据注册**多套变换管道**并行保存结果。DeepWiki 目前只注册了一个 key `"split_and_embed"`，因此实际存在两张有效的表：

| 属性 | 角色 | 内容 |
|------|------|------|
| `items` | 原始表 | 每个源码文件对应一个 `Document`，不含向量 |
| `transformed_items["split_and_embed"]` | 向量表 | 分块后的 `Document`，含 256 维浮点向量 |

### 三个核心职责

**1. 数据变换管道**

先注册变换器，后触发执行（`adalflow.core.db:155-223`）：

```python
# data_pipeline.py 中的实际用法
db = LocalDB()
db.register_transformer(transformer=data_transformer, key="split_and_embed")
db.load(documents)               # → 存入 items[]
db.transform(key="split_and_embed")
# items[] → TextSplitter → ToEmbeddings → transformed_items["split_and_embed"]
```

**2. pickle 序列化持久化**

```python
db.save_state(filepath)          # pickle.dump(self, file)
LocalDB.load_state(filepath)     # pickle.load(file)
```

`transformer_setups` 中的管道对象无法直接 pickle，LocalDB 专门通过 `__getstate__` / `__setstate__` 处理：保存时将 transformer 序列化为 dict（`to_dict()`），加载时从 dict 重建（`from_dict()`）。

**3. 基础 CRUD 操作**

| 方法 | 说明 |
|------|------|
| `load(items)` | 替换全部原始数据 |
| `extend(items)` | 追加并自动触发变换 |
| `add(item)` | 追加单条并自动触发变换 |
| `delete(index)` | 按索引删除，同步删除变换数据 |
| `get_transformed_data(key, filter_fn)` | 按条件过滤变换数据 |

### LocalDB 与 FAISS 的分工

LocalDB **不包含** FAISS 索引本身，两者职责分离：

```
LocalDB (.pkl)                              FAISSRetriever（内存中，每次启动重建）
──────────────────────────────              ──────────────────────────────────────
items[]                ── 持久化 ──►        （不使用）
transformed_items["split_and_embed"]
  └─ doc.vector  ─────────────────────────► 构建 FAISS 平面索引
  └─ doc.text    ─────────────────────────► 检索命中后返回原文
```

`api/rag.py:385` 中，FAISS 索引在每次进程启动时从向量数据重建，不落盘：

```python
self.transformed_docs = db.get_transformed_data(key="split_and_embed")
self.retriever = FAISSRetriever(
    documents=self.transformed_docs,
    document_map_func=lambda doc: doc.vector,  # 只取 vector 字段建索引
)
```

`index_path` 字段虽然命名为索引路径，实际只记录 `.pkl` 文件自身的路径——FAISS 索引的落盘能力被预留但未使用。这意味着每次冷启动都要重建索引（100K 向量约耗时 1-2 秒），是一个可优化的点。

### `.pkl` 文件大小估算

`.pkl` 同时包含原始文件全文（`items`）和 chunk 文本（`transformed_items`），存在文本冗余：

| 组成部分 | 内容 | 估算大小（10K 文件） |
|---------|------|------------------|
| `items` 原始文本 | 10K 文件完整内容 | ~50 MB |
| `transformed_items` chunk 文本 | 20K × 350 词 | ~40 MB |
| `transformed_items` 向量数据 | 20K × 256 × 4 bytes | ~20 MB |
| metadata + id 等字段 | 每条约 300 bytes | ~6 MB |
| **合计** | | **~116 MB** |

---

## Document Schema（行定义）

两张表均存储 `Document` 对象（`adalflow.core.types.Document`）：

```python
@dataclass
class Document:
    # 核心字段
    text:                 str            # 文件内容（原始表）或代码块文本（向量表）
    vector:               List[float]    # 嵌入向量（256 维）；原始表中为空列表

    # 主键与关联关系
    id:                   str            # UUID — 行唯一标识
    parent_doc_id:        str            # 外键 → items[].id（分块后填充）
    order:                int            # 该块在原始文档中的顺序编号

    # 业务元数据
    meta_data: {
        "file_path":         str         # 相对路径，如 "src/api/rag.py"
        "type":              str         # 文件扩展名，如 "py"、"md"
        "is_code":           bool        # 是否为源代码文件
        "is_implementation": bool        # 是否为实现文件（非测试文件）
        "title":             str         # 同 file_path
        "token_count":       int         # 原始文件的 token 数量
    }

    # 检索时填充
    score:                float          # FAISS 相似度得分（查询时写入）
    estimated_num_tokens: int            # 当前块的 token 数（自动估算）
```

---

## 两张表的关系

```
items（原始表）                      transformed_items["split_and_embed"]（向量表）
──────────────────────────          ──────────────────────────────────────────────────
Document                            Document  order=0  ──┐
  id: "abc-123"                     Document  order=1  ──┤── parent_doc_id = "abc-123"
  text: <完整文件内容>                Document  order=2  ──┘
  vector: []
  meta_data: { file_path: "..." }
```

每个原始 `Document`（一个文件一条）经过 `TextSplitter` 切分为多个块，每块成为向量表中的一条新记录，通过 `parent_doc_id` 关联回原始文件，并由 `order` 字段标记顺序。

---

## 分块参数

配置于 `api/config/embedder.json`：

| 参数 | 值 | 说明 |
|------|----|------|
| `split_by` | `"word"` | 按单词边界分割 |
| `chunk_size` | `350` | 平衡上下文窗口使用与检索粒度 |
| `chunk_overlap` | `100` | 避免语义在块边界断裂（代价是产生冗余嵌入） |

---

## 嵌入配置

向量维度因提供商而异，默认使用 OpenAI `text-embedding-3-small`，维度为 256。

| 提供商 | 模型 | 向量维度 | 批处理大小 |
|--------|------|----------|------------|
| OpenAI（默认） | `text-embedding-3-small` | 256 | 500 |
| Google | `gemini-embedding-001` | 不固定 | 100 |
| AWS Bedrock | `amazon.titan-embed-text-v2:0` | 256 | 100 |
| Ollama | `nomic-embed-text` | 不固定 | 1（不支持批处理） |

---

## FAISS 检索

查询时，`api/rag.py` 加载向量表并基于 `vector` 列构建 FAISS 平面索引：

```python
# api/rag.py:385
self.retriever = FAISSRetriever(
    top_k=20,
    embedder=retrieve_embedder,
    documents=self.transformed_docs,           # 向量表的全部行
    document_map_func=lambda doc: doc.vector,  # 提取 vector 列
)
```

查询文本经同一提供商嵌入后，FAISS 通过余弦/L2 距离返回 `top_k=20` 个最相似的行，`meta_data["file_path"]` 字段用于标识每条结果对应的源文件。

---

## 嵌入维度验证机制

切换嵌入提供商会导致向量维度变化，若直接加载旧 `.pkl` 文件会造成 FAISS 索引错误。`api/rag.py` 在构建索引前执行防御性验证：

1. 扫描所有文档，统计各维度的出现次数。
2. 将出现次数最多的维度认定为目标维度。
3. 过滤掉维度不匹配的文档。

相关实现：`api/rag.py` — `_validate_and_filter_embeddings()`

---

## 文件大小限制

| 文件类型 | Token 上限 | 超限处理 |
|----------|------------|---------|
| 代码文件 | `8192 × 10 = 81,920` | 跳过并记录警告 |
| 文档文件 | `8192` | 跳过并记录警告 |

---

## 关键实现文件

| 文件 | 职责 |
|------|------|
| `api/data_pipeline.py` | 仓库克隆、文档读取、分块、嵌入、数据库持久化 |
| `api/rag.py` | 数据库加载、维度验证、FAISS 检索器构建、查询执行 |
| `api/config/embedder.json` | 分块参数、嵌入提供商配置、检索器 `top_k` |
| `api/config/repo.json` | 文件包含/排除规则（122 种排除模式） |
