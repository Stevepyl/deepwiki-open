# DeepWiki RAG 系统设计文档

## 概述

DeepWiki 的 RAG（Retrieval-Augmented Generation）系统基于 [AdalFlow](https://github.com/SylphAI-Inc/AdalFlow) 框架构建，用于回答用户关于代码仓库的问题。核心特点是 Embedder 与 Generator 完全解耦，支持跨提供商混用（如 Google 嵌入 + OpenAI 生成）。

---

## 完整数据流

```
GitHub/GitLab/Bitbucket URL
    ↓
git clone --depth=1 --single-branch
    ↓
read_all_documents()  →  [Document(text, meta)]
    ↓
TextSplitter (350词块 / 100词重叠)
    ↓
Embedder (Google / OpenAI / Ollama / Bedrock)  →  [Document(text, meta, vector)]
    ↓
FAISS 平面索引  →  LocalDB.pkl 缓存到 ~/.adalflow/databases/
    ↓
用户查询 → embed(query) → FAISS top-k 相似度搜索 → 检索 chunks
    ↓
RAG_TEMPLATE 注入 contexts + conversation history
    ↓
LLM Generator (流式输出 RAGAnswer{rationale, answer})
    ↓
前端渲染 Markdown
```

---

## Document 对象：流水线的核心载体

`Document` 是整个 RAG 流水线中数据流转的统一数据类型，在不同阶段承载不同信息。

### 生命周期

| 阶段 | `text` | `meta_data` | `vector` |
|------|--------|-------------|---------|
| 读取后 | 完整文件内容 | file_path, type, is_code... | 无 |
| TextSplitter 后 | 约 350 词的 chunk | 继承，含来源文件路径 | 无 |
| Embedder 后 | 同上 | 同上 | `[0.012, -0.834, ...]` |

### 分块示意

一个大文件会被拆分成多个 `Document`，相邻块之间有 100 词重叠：

```
原始 Document (api/rag.py, 2000词)
    ↓ TextSplitter
Document[0]  词 0-350    file_path: api/rag.py
Document[1]  词 250-600  file_path: api/rag.py  ← 与前一块重叠 100 词
Document[2]  词 500-850  file_path: api/rag.py
...
```

### 为什么用 Document 而不是裸字符串

1. **元数据与内容绑定**：`meta_data["file_path"]` 让 LLM 回答能引用具体文件位置，而不是匿名代码片段。这在 Prompt 中直接体现：
   ```
   File Path: {{context.meta_data.get('file_path', 'unknown')}}
   Content: {{context.text}}
   ```

2. **向量与原文共生**：`doc.vector` 用于 FAISS 相似度搜索，`doc.text` 用于注入 Prompt，两者必须一一对应。同一对象封装避免索引错位。

3. **AdalFlow 管道契约**：`TextSplitter → ToEmbeddings` 这条 `adal.Sequential` 流水线的输入输出协议统一为 `List[Document]`，是框架内组件间通信的标准格式。

---

## 阶段一：数据摄取（Ingestion）

**文件**: `api/data_pipeline.py`

### 仓库克隆

```
DatabaseManager._create_repo()
  ├─ 支持 GitHub / GitLab / Bitbucket（含私有仓库 access_token）
  ├─ git clone --depth=1 --single-branch（浅克隆，节省空间）
  ├─ 存储到 ~/.adalflow/repos/{owner}_{repo}/
  └─ 若目录已存在则跳过克隆（幂等）
```

### 文档读取与过滤

`read_all_documents()` 按优先级处理两类文件：
    
| 类型 | 扩展名 | Token 上限 |
|------|--------|-----------|
| 代码文件 | `.py .js .ts .java .cpp .go .rs .jsx .tsx .html .css .php .swift .cs` | 81,920 |
| 文档文件 | `.md .txt .rst .json .yaml .yml` | 8,192 |

代码文件的 token 上限是文档的 10 倍，因为代码文件通常大得多。

每个文件生成一个 `Document` 对象，携带 `meta_data`：
```python
{
    "file_path": "api/rag.py",
    "type": "py",
    "is_code": True,
    "is_implementation": True,  # 非 test_/app_ 前缀
    "title": "api/rag.py",
    "token_count": 1234,
}
```

过滤规则支持两种模式（互斥）：
- **排除模式**（默认）：基于 `DEFAULT_EXCLUDED_DIRS` + `api/config/repo.json`
- **包含模式**：当指定 `included_dirs` 或 `included_files` 时激活，只处理指定范围

### 文档变换与存储

```python
# data_pipeline.py:421-423
data_transformer = adal.Sequential(
    splitter,          # TextSplitter: 350词 / 100词重叠
    embedder_transformer   # ToEmbeddings(batch_size=500) 或 OllamaDocumentProcessor
)
db.transform(key="split_and_embed")
db.save_state(filepath=db_path)  # → ~/.adalflow/databases/{repo}.pkl
```

**Ollama 特殊处理**：Ollama 不支持批量嵌入，使用 `OllamaDocumentProcessor` 逐文档处理。

---

## 阶段二：检索器初始化（Retriever Setup）

**文件**: `api/rag.py:345`

### 缓存策略

```python
# data_pipeline.py:869-892
if os.path.exists(save_db_file):
    db = LocalDB.load_state(save_db_file)  # 直接加载，跳过重新计算
    if non_empty == 0:
        # 全部嵌入为空时才重建（如切换了 embedder）
        rebuild()
    else:
        return documents
```

### 嵌入维度校验（关键防御逻辑）

切换 embedder 提供商时，旧 `.pkl` 中的向量维度与新 embedder 不一致会导致 FAISS 崩溃。

```python
# rag.py:251 - _validate_and_filter_embeddings()
# 策略：以"多数派维度"为基准，过滤掉维度异常的文档
target_size = max(embedding_sizes.keys(), key=lambda k: embedding_sizes[k])
valid_documents = [doc for doc in documents if len(doc.vector) == target_size]
```

### FAISS 索引构建

```python
self.retriever = FAISSRetriever(
    **configs["retriever"],         # top-k 等超参数
    embedder=self.query_embedder,   # Ollama 用单字符串包装器，其他用批量 embedder
    documents=self.transformed_docs,
    document_map_func=lambda doc: doc.vector,  # 复用预计算向量
)
```

使用平面索引（Flat Index），无近似聚类，保证召回精度。

---

## 阶段三：查询时检索（Retrieval）

**文件**: `api/rag.py:416`

```python
def call(self, query: str, language: str = "en") -> Tuple[List]:
    retrieved_documents = self.retriever(query)
    # retriever 内部：embed(query) → FAISS.search(top-k)

    # 将索引映射回实际文档对象
    retrieved_documents[0].documents = [
        self.transformed_docs[doc_index]
        for doc_index in retrieved_documents[0].doc_indices
    ]
    return retrieved_documents
```

**重要**：`RAG.call()` 只负责**检索**，不调用生成器。生成由上层 API 路由控制，将 contexts 注入 prompt 后调用 `self.generator`。

---

## 阶段四：Prompt 构建与流式生成

### Prompt 模板结构（`api/prompts.py:31`）

```
<SYS_PROMPT>
  [system_prompt] 检测用户语言，以相同语言回答；使用 Markdown 格式
  [output_format_str] RAGAnswer schema（强制结构化输出）
</SYS_PROMPT>

<CONVERSATION_HISTORY>（可选）
  1. User: ... / You: ...
  2. ...
</CONVERSATION_HISTORY>

<CONTEXT>（可选，来自 FAISS 检索）
  1. File Path: api/rag.py
     Content: ...chunk text...
</CONTEXT>

<USER_PROMPT>
  {input_str}
</USER_PROMPT>
```

### 结构化输出（Chain-of-Thought）

```python
# rag.py:147
@dataclass
class RAGAnswer(adal.DataClass):
    rationale: str  # Chain-of-Thought 思维链（内部使用）
    answer: str     # 最终答案（展示给用户，Markdown 格式）
```

`rationale` 字段引导 LLM 先推理再回答，提升回答质量；`answer` 字段即为用户看到的内容。

### 对话记忆（Memory）

```python
# rag.py:51
class Memory(adal.core.component.DataComponent):
    current_conversation: CustomConversation  # 包含 dialog_turns 列表

    def call(self) -> Dict[str, DialogTurn]:
        # 返回 OrderedDict，注入到 RAG_TEMPLATE 的 conversation_history
```

**无状态设计**：每次 `call()` 重新读取全部历史，不维护流式状态。历史轮数越多，Prompt 越长，需注意上下文窗口预算。

---

## 关键设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 向量存储 | FAISS 平面索引 | 精度优先；大型仓库可考虑 IVF 近似索引 |
| 分块策略 | 350词 + 100词重叠 | 平衡语义连贯性与检索粒度 |
| 嵌入缓存 | LocalDB.pkl | 避免重复 API 调用；切换 embedder 需清除缓存 |
| Embedder/Generator 解耦 | 独立配置 | 可混用不同提供商，灵活性高 |
| 生成器输出格式 | DataClassParser + RAGAnswer | 强制结构化，减少格式错误 |
| 对话历史 | 全量注入 Prompt | 简单可靠，但长对话会消耗大量 token |

---

## 已知性能问题

- **Ollama 嵌入慢**：约 1-2s/请求，无批量支持
- **大型仓库摄取慢**：文件遍历和嵌入计算是串行的，可考虑多进程并行
- **FAISS 索引无过期机制**：仓库更新后需手动删除 `.pkl` 重建
- **长对话 token 膨胀**：对话历史全量注入，未做截断或摘要处理

---

## 相关文件索引

| 文件 | 职责 |
|------|------|
| `api/rag.py` | RAG 核心：Memory、检索、Generator 初始化 |
| `api/data_pipeline.py` | 仓库克隆、文档读取、分块、嵌入、数据库管理 |
| `api/prompts.py` | 所有 Prompt 模板（RAG、SimpleChat、DeepResearch） |
| `api/config.py` | 配置加载、Provider 客户端映射 |
| `api/config/generator.json` | LLM 提供商和模型定义 |
| `api/config/embedder.json` | 嵌入模型配置 |
| `api/config/repo.json` | 文件过滤规则 |
| `api/tools/embedder.py` | Embedder 工厂函数 |
| `api/ollama_patch.py` | Ollama 单文档处理器适配 |
