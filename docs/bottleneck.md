# FAISS 性能容量分析

## 背景

DeepWiki 使用 FAISS 平面索引（Flat Index）作为向量检索后端，每个仓库对应一个 `.pkl` 文件持久化到 `~/.adalflow/databases/`。本文档分析在保证检索性能的前提下，系统能够支持的最大代码仓库规模。

---

## 核心参数（来自 `api/config/embedder.json`）

| 参数 | 值 |
|------|----|
| 向量维度 | 256（OpenAI `text-embedding-3-small`） |
| 分块大小 | 350 词 |
| 分块重叠 | 100 词 |
| 检索 top_k | 20 |
| 嵌入批处理大小 | 500（OpenAI）/ 100（Google/Bedrock）/ 1（Ollama） |

---

## 分块数量推导

每步移动词数：

```
step = chunk_size - chunk_overlap = 350 - 100 = 250 词/步
chunks_per_file ≈ file_words / 250
```

典型代码文件约 400-600 词，平均约 **2 chunks/文件**。

---

## 内存占用（决定性约束）

`api/rag.py:385` 中，构建 FAISS 索引时 `transformed_docs` 全量加载进 RAM：

```python
self.retriever = FAISSRetriever(
    documents=self.transformed_docs,  # 全量加载
    document_map_func=lambda doc: doc.vector,
)
```

每个 chunk 的内存开销：

| 组成 | 大小 |
|------|------|
| `vector`（256 dim × float32） | 1 KB |
| `text`（~350 词 ≈ 2000 字符） | ~2 KB |
| `meta_data` + `id` 等字段 | ~0.3 KB |
| **单 chunk 合计** | **~3.3 KB** |

不同仓库规模的内存占用估算：

| 源码文件数 | chunk 数 | 内存占用 | 典型项目 |
|-----------|---------|---------|---------|
| 1,000 | ~2,000 | ~6 MB | 中型库（Flask、FastAPI） |
| 10,000 | ~20,000 | ~66 MB | 大型项目（Django、React） |
| 50,000 | ~100,000 | ~330 MB | 超大仓库（VS Code、Kubernetes） |
| 150,000 | ~300,000 | ~990 MB | Linux 内核级别 |

---

## FAISS 检索延迟

FAISS Flat Index 是精确搜索，检索质量不随向量数量增多而下降，但搜索时间线性增长（O(n×d)）。FAISS 利用 BLAS/SIMD 指令，256 维单次查询在 CPU 上的量级：

| chunk 数 | 单次查询延迟 | 评估 |
|---------|------------|------|
| 10,000 | < 1 ms | 无感知 |
| 100,000 | ~5 ms | 可接受 |
| 500,000 | ~25 ms | 可接受 |
| 1,000,000 | ~100 ms | 边缘 |

**FAISS 检索延迟本身不是瓶颈**，在 100K chunk 量级下几乎不可察觉。

---

## 真正的瓶颈

### 瓶颈一：摄取时间（一次性，但影响首次使用体验）

以 OpenAI 为例（batch_size=500，每次 API 调用约 200ms）：

| chunk 数 | API 调用次数 | 摄取时间估算 |
|---------|------------|------------|
| 20,000 | 40 次 | ~8 秒 |
| 100,000 | 200 次 | ~40 秒 |
| 300,000 | 600 次 | ~2 分钟 |

Ollama 无批处理支持（逐 chunk，1-2s/次），100K chunk 需约 27 小时，实际上不可用于大型仓库。

相关代码（`api/data_pipeline.py:411-418`）：

```python
if embedder_type == "ollama":
    embedder_transformer = OllamaDocumentProcessor(embedder=embedder)
else:
    batch_size = embedder_config.get("batch_size", 500)
    embedder_transformer = ToEmbeddings(embedder=embedder, batch_size=batch_size)
```

### 瓶颈二：单文件 token 上限导致静默跳过

```python
# data_pipeline.py:326
if token_count > MAX_EMBEDDING_TOKENS * 10:  # 81,920 tokens
    logger.warning("Skipping large file...")
    continue
```

| 文件类型 | Token 上限 | 约等于词数 |
|----------|------------|---------|
| 代码文件 | 81,920 | ~65,000 词 |
| 文档文件 | 8,192 | ~6,500 词 |

自动生成代码、打包产物、大型 JSON 等文件会被静默跳过，导致部分代码缺失覆盖。

### 瓶颈三：`.pkl` 反序列化开销

整个 `LocalDB` 含原始文件内容 + 所有向量，以 pickle 序列化。每次查询初始化（`LocalDB.load_state()`）需完整反序列化：

- 300K chunk 的 `.pkl` 约 1-2 GB
- 冷启动时反序列化时间显著，热路径下已缓存在内存中

---

## 综合容量结论

| 仓库规模 | 文件数量 | chunk 数 | 体验 |
|--------|---------|---------|-----|
| 小型 | < 2,000 | < 4K | 流畅，秒级摄取 |
| 中型 | 2,000–20,000 | 4K–40K | **最佳区间**，检索 < 5ms，摄取 < 2 分钟 |
| 大型 | 20,000–80,000 | 40K–160K | 可用，RAM 需 ~500 MB，摄取约 5-10 分钟 |
| 超大型 | > 80,000 | > 160K | 摄取慢、RAM 压力大，建议使用 `included_dirs` 限制范围 |

---

## 应对大型仓库的现有机制

系统已内置 `included_dirs` / `included_files` 包含模式（`api/data_pipeline.py` 排除/包含双模式），可将索引范围限制在核心子目录：

```python
# 仅索引 net/ 和 fs/ 子目录，而非全量内核
included_dirs=["net", "fs"]
```

这是处理 Linux 内核、Chromium 等超大型仓库的推荐做法。

---

## 缺少 Re-ranking 对 RAG 效果的影响

### 问题描述

`RAG.call()`（`api/rag.py:427`）在 FAISS 返回结果后直接交给 LLM，没有精排步骤：

```python
retrieved_documents = self.retriever(query)   # FAISS 余弦距离排序
# 无任何重排序或过滤
return retrieved_documents                     # 原样返回
```

### 具体影响

| 现象 | 根因 |
|------|------|
| top-20 结果集中在同一文件的相邻 chunk | 余弦距离无多样性控制，相邻块向量天然相近 |
| 能回答问题的关键 chunk 排名靠后 | 向量相似度衡量的是表面词汇分布，而非"是否能回答问题" |
| LLM 收到冗余上下文，消耗 token 预算 | top-20 中可能有 10 个来自同一函数的不同分块，内容高度重叠 |
| 跨文件调用链的关键依赖被截断 | 与查询词汇相似度低的依赖文件，即使对回答至关重要也可能排不进 top-20 |

### 在代码 RAG 场景中尤为突出

文本 RAG 中查询与答案通常在词汇层面接近，余弦距离足够可靠。代码 RAG 中用户用自然语言提问（"认证逻辑在哪里"），而代码块包含的是 `check_token()`、`verify_credentials()` 等符号，两者在嵌入空间中天然存在 gap，单靠向量相似度的召回准确率有限。

---

## 多仓库联合查询场景下的架构瓶颈

当前架构为**每个仓库独立一个 FAISS 索引**（一个 `.pkl` 文件），查询时只检索当前仓库。若扩展为单数据库多仓库联合查询，代码量将成为硬性瓶颈。

### 瓶颈转变

| 维度 | 单仓库（现状） | 多仓库联合查询 |
|------|-------------|-------------|
| FAISS 搜索复杂度 | O(n_repo × d) | O(Σn_all_repos × d) |
| 内存占用 | 单仓库向量 | **所有仓库向量同时加载** |
| 搜索延迟 | 随单仓库 chunk 数线性增长 | 随**总 chunk 数**线性增长 |
| 检索质量 | 结果全来自同一仓库 | 不同仓库的 chunk 在向量空间中无隔离，跨仓库竞争 top-k |

### 多仓库场景下的内存与延迟估算

| 仓库数 × 平均规模 | 总 chunk 数 | RAM 需求 | FAISS 搜索延迟 |
|----------------|-----------|---------|--------------|
| 5 × 10K 文件 | ~100K | ~330 MB | ~5 ms |
| 20 × 10K 文件 | ~400K | ~1.3 GB | ~20 ms |
| 50 × 10K 文件 | ~1M | ~3.3 GB | ~100 ms |
| 100 × 10K 文件 | ~2M | ~6.6 GB | ~200 ms+ |

单仓库场景下总 chunk 数有自然上限（受单个仓库代码量约束），多仓库场景下这个上限消失，**总代码量直接映射为内存占用和检索延迟**。

### 额外引入的检索质量问题

合并索引后，来自不同仓库的 chunk 在向量空间中没有命名空间隔离。查询"如何处理认证"可能在 top-20 中混入多个不相关仓库的认证代码，稀释真正有用的上下文。

### 架构层面的根本原因

FAISS 是纯向量搜索库，不具备 metadata 过滤能力，因此在多仓库场景下只有两种实现路径，均有明显代价：

| 路径 | 做法 | 代价 |
|------|------|------|
| 联合全量搜索 | 所有仓库向量合并为单一索引 | 性能随仓库数线性劣化；无命名空间隔离 |
| 应用层多实例 | 每个仓库维护独立 FAISS 实例，查询时并发检索后合并结果 | 实现复杂；并发内存占用高 |

生产级多租户向量数据库（Pinecone、Qdrant、Weaviate）通过内置的 **namespace / collection 过滤**解决此问题——先按 namespace 缩小候选集，再做相似度搜索。这是当前 FAISS 架构在多仓库方向扩展时的核心架构债。

---

## 潜在优化方向

| 优化点 | 当前状态 | 改进方向 |
|--------|---------|---------|
| Ollama 摄取速度 | 逐 chunk 串行 | 实现批量嵌入 API 或并发请求 |
| 大仓库摄取 | 单进程串行 | 多进程并行嵌入（`multiprocessing`） |
| FAISS 索引类型 | Flat（精确） | 超大规模可切换 IVF 近似索引 |
| 检索后精排 | 无，直接按余弦距离使用 | 引入 cross-encoder re-ranking（如 `flashrank`），粗检索 top-100 后精排到 top-20 |
| `.pkl` 冷启动 | 全量反序列化 | 分片存储或 mmap 映射 |
| 仓库更新 | 手动删除 `.pkl` 重建 | 增量更新机制（按文件哈希差分） |
| 多仓库扩展 | 每仓库独立 FAISS，不支持联合查询 | 迁移至支持 namespace 过滤的向量数据库（Qdrant、Weaviate） |

---

## 相关文件

| 文件 | 职责 |
|------|------|
| `api/data_pipeline.py` | 文件读取、分块、嵌入、数据库持久化 |
| `api/rag.py` | FAISS 检索器构建、嵌入维度验证 |
| `api/config/embedder.json` | 分块参数、嵌入提供商配置 |
| `api/config/repo.json` | 文件包含/排除规则 |
