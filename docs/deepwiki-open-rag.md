# DeepWiki-Open RAG 系统总览

## 1. 整体架构

DeepWiki 的 RAG 系统基于 [AdalFlow](https://github.com/SylphAI-Inc/AdalFlow) 框架构建，核心特点是 **Embedder 与 Generator 完全解耦**，支持跨提供商混用（如 Google 嵌入 + OpenAI 生成）。

### 组件关系图

```
用户请求 (WebSocket / HTTP)
        │
        ▼
┌─────────────────────────────┐
│  websocket_wiki.py          │  请求协调层：路由、流式响应
│  simple_chat.py             │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  RAG  (api/rag.py)          │  核心大脑
│  ├── Memory                 │  对话历史管理
│  ├── Embedder               │  查询向量化
│  ├── FAISSRetriever         │  向量相似度检索
│  └── Generator (adal)       │  LLM 封装与流式生成
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  DatabaseManager            │  数据层
│  (api/data_pipeline.py)     │
│  ├── 仓库克隆                │
│  ├── TextSplitter           │  350词块 / 100词重叠
│  └── FAISS 索引持久化         │  ~/.adalflow/databases/
└─────────────────────────────┘
```

### 完整数据流

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
LLM Generator（流式输出 RAGAnswer{rationale, answer}）
    ↓
前端渲染 Markdown
```

---

## 2. 核心数据结构

### 2.1 Document 对象

`Document`（`adalflow.core.types.Document`）是整个 RAG 流水线中数据流转的统一类型，在不同阶段承载不同信息。

```python
@dataclass
class Document:
    text:                 str            # 文件完整内容（原始表）或代码块文本（向量表）
    vector:               List[float]    # 嵌入向量（维度取决于 embedder：OpenAI/Bedrock 为 256 维，Google/Ollama 由模型决定）；原始表中为空列表

    id:                   str            # UUID — 行唯一标识
    parent_doc_id:        str            # 外键 → 原始文件 Document.id
    order:                int            # 该块在原始文档中的顺序编号

    meta_data: {
        "file_path":         str         # 相对路径，如 "api/rag.py"
        "type":              str         # 文件扩展名，如 "py"、"md"
        "is_code":           bool        # 是否为源代码文件
        "is_implementation": bool        # 是否为实现文件（非测试文件）
        "title":             str         # 同 file_path
        "token_count":       int         # 原始文件的 token 数
    }

    score:                float          # FAISS 相似度得分（查询时写入）
    estimated_num_tokens: int            # 当前块的 token 数（自动估算）
```

**Document 在各阶段的生命周期：**

| 阶段 | `text` | `meta_data` | `vector` |
|------|--------|-------------|---------|
| 文件读取后 | 完整文件内容 | file_path, type, is_code... | 无 |
| TextSplitter 后 | 约 350 词的 chunk | 继承，含来源文件路径 | 无 |
| Embedder 后 | 同上 | 同上 | `[0.012, -0.834, ...]` |

**分块示意（一个大文件被拆分为多个 Document）：**

```
原始 Document (api/rag.py, 2000词)
    ↓ TextSplitter（step=250词）
Document[0]  词 0-350    order=0
Document[1]  词 250-600  order=1  <- 与前一块重叠 100 词
Document[2]  词 500-850  order=2  <- 与前一块重叠 100 词
...
```

### 2.2 LocalDB（持久化层）

`LocalDB`（`adalflow.core.db.LocalDB`）是 AdalFlow 提供的**带数据变换管道的内存存储容器**，附带 pickle 序列化能力。它本身不是向量数据库，FAISS 的搜索能力由上层的 `FAISSRetriever` 提供。

```python
@dataclass
class LocalDB(Component):
    name: str
    items: List[Document]                         # 原始表 — 每个文件对应一条记录（无向量）
    transformed_items: Dict[str, List[Document]]  # 派生表 — key 为变换名称
    transformer_setups: Dict[str, Component]      # 已注册的变换管道
```

DeepWiki 目前只注册了一个变换 key `"split_and_embed"`，实际维护两张表：

| 属性 | 角色 | 内容 |
|------|------|------|
| `items` | 原始表 | 每个源码文件对应一个 `Document`，不含向量 |
| `transformed_items["split_and_embed"]` | 向量表 | 分块后的 `Document`，含浮点向量（维度取决于 embedder 提供商） |

**LocalDB 与 FAISS 的分工：**

```
LocalDB (.pkl)                              FAISSRetriever（内存中，每次启动重建）
──────────────────────────────              ──────────────────────────────────────
transformed_items["split_and_embed"]
  └─ doc.vector  ─────────────────────────► 构建 FAISS 平面索引（余弦/L2 距离）
  └─ doc.text    ─────────────────────────► 检索命中后返回原文
```

**存储目录结构：**

```
~/.adalflow/
├── repos/{owner}_{repo}/         # 克隆的仓库源码
├── databases/{owner}_{repo}.pkl  # 序列化的 LocalDB（含原文 + 向量）
└── wikicache/                    # 生成的 Wiki 缓存
```

---

## 3. 阶段一：数据摄取（Ingestion）

**文件**: `api/data_pipeline.py`

### 3.1 仓库克隆

```
DatabaseManager._create_repo()
  ├─ 支持 GitHub / GitLab / Bitbucket（含私有仓库 access_token）
  ├─ git clone --depth=1 --single-branch（浅克隆，节省空间）
  ├─ 存储到 ~/.adalflow/repos/{owner}_{repo}/
  └─ 若目录已存在则跳过克隆（幂等）
```

### 3.2 文档读取与过滤

`read_all_documents()` 处理两类文件，各有 token 上限：

| 类型 | 扩展名 | Token 上限 | 说明 |
|------|--------|-----------|------|
| 代码文件 | `.py .js .ts .java .cpp .go .rs .jsx .tsx .html .css .php .swift .cs .c, .h, .hpp` | 81,920 | 约等于 65,000 词 |
| 文档文件 | `.md .txt .rst .json .yaml .yml` | 8,192 | 约等于 6,500 词 |

超限文件会被**静默跳过**并记录警告（`data_pipeline.py:326-328, 360-361`），是覆盖率的隐患。

过滤规则支持两种模式（互斥），由内部函数 `should_process_file()` 执行（`data_pipeline.py:235-302`）：
- **排除模式**（默认）：基于 `DEFAULT_EXCLUDED_DIRS` + `api/config/repo.json`（含 122 种排除模式）
- **包含模式**：指定 `included_dirs` 或 `included_files` 时激活，只处理指定范围（推荐用于超大型仓库）

路径判断时会做规范化处理（`data_pipeline.py:290`）：先用 `strip("./").rstrip("/")` 去掉前缀，再检查规范化后的目录名是否出现在文件路径的路径分量列表中，避免误匹配子字符串（如不会把 `src` 误匹配到 `src_backup`）。

### 3.3 分块策略（TextSplitter）

分块参数配置于 `api/config/embedder.json`：

| 参数 | 值 | 说明 |
|------|----|------|
| `split_by` | `"word"` | 按单词边界分割（非 AST 感知） |
| `chunk_size` | `350` | 平衡上下文窗口使用与检索粒度 |
| `chunk_overlap` | `100` | 避免语义在块边界断裂（代价是产生冗余嵌入） |

每步移动词数：`step = 350 - 100 = 250`，典型代码文件（400-600 词）约产生 **2 个 chunk/文件**。

### 3.4 嵌入生成

实际调用链（`data_pipeline.py:426-449`）：

```python
# transform_documents_and_save_to_db() 内部
data_transformer = prepare_data_pipeline(embedder_type, is_ollama_embedder)
# prepare_data_pipeline() 返回：
#   adal.Sequential(
#       TextSplitter(split_by="word", chunk_size=350, chunk_overlap=100),
#       OllamaDocumentProcessor(embedder)  # Ollama 路径
#       ToEmbeddings(embedder, batch_size=500)  # 其他提供商路径
#   )

db = LocalDB()
db.register_transformer(transformer=data_transformer, key="split_and_embed")
db.load(documents)            # items[] 写入
db.transform(key="split_and_embed")  # TextSplitter → Embedder，结果写入 transformed_items["split_and_embed"]
db.save_state(filepath=db_path)  # → ~/.adalflow/databases/{repo}.pkl
```

**嵌入提供商配置：**

| 提供商 | 模型 | 向量维度 | 批处理大小 |
|--------|------|----------|------------|
| OpenAI（默认） | `text-embedding-3-small` | 256 | 500 |
| Google | `gemini-embedding-001` | 不固定 | 100 |
| AWS Bedrock | `amazon.titan-embed-text-v2:0` | 256 | 100 |
| Ollama | `nomic-embed-text` | 不固定 | 1（不支持批处理） |

**Ollama 特殊处理**：Ollama 不支持批量嵌入，使用 `OllamaDocumentProcessor` 逐文档处理（约 1-2s/请求），大型仓库实际不可用。

---

## 4. 阶段二：检索器初始化

**文件**: `api/rag.py`

### 4.1 缓存策略

```python
# data_pipeline.py:869-892
if os.path.exists(save_db_file):
    db = LocalDB.load_state(save_db_file)
    documents = db.get_transformed_data(key="split_and_embed")
    lengths = [_embedding_vector_length(doc) for doc in documents]
    non_empty = sum(1 for n in lengths if n > 0)

    if non_empty == 0:
        # 全部嵌入为空（如首次嵌入失败），重建
        rebuild()
    else:
        return documents  # 缓存命中，跳过克隆和嵌入
```

缓存命中需同时满足三个条件：① `.pkl` 文件存在；② 文件能成功 `pickle.load()`；③ 至少有一个非空向量（`non_empty > 0`）。

**切换 embedder 提供商的隐患**：旧 `.pkl` 中的向量非空但来自旧维度空间，`non_empty > 0` 仍会命中缓存并直接返回。缓存中的旧维度向量在下一步 `_validate_and_filter_embeddings()` 中才会被过滤——如果旧向量占多数，"多数派"逻辑反而会把新维度的向量过滤掉。因此切换 embedder 后应手动删除对应 `.pkl` 文件。

### 4.2 嵌入维度校验

切换 embedder 提供商时，旧 `.pkl` 中的向量维度与新 embedder 不一致会导致 FAISS 崩溃。`_validate_and_filter_embeddings()` 在构建索引前执行防御性校验，具体流程为：

1. 扫描所有文档，统计各向量维度的出现次数
2. 以**多数派维度**为目标维度
3. 过滤掉维度不匹配的文档

```python
# rag.py:300
target_size = max(embedding_sizes.keys(), key=lambda k: embedding_sizes[k])
valid_documents = [doc for doc in documents if len(doc.vector) == target_size]
```

### 4.3 FAISS 索引构建

```python
# rag.py:384-390
# Ollama 使用单字符串包装器（query_embedder），其他使用批量 embedder
retrieve_embedder = self.query_embedder if self.is_ollama_embedder else self.embedder
self.retriever = FAISSRetriever(
    **configs["retriever"],           # top_k=20 等超参数
    embedder=retrieve_embedder,       # 查询时用于嵌入输入 query
    documents=self.transformed_docs,
    document_map_func=lambda doc: doc.vector,  # 复用预计算向量，不重新嵌入
)
```

**Ollama 的 `query_embedder`** 是一个闭包（`rag.py:195-203`），接受单字符串或长度为 1 的列表并调用 `self.embedder`——这是对 FAISSRetriever 期望接收 list 输入但 Ollama 只能处理单字符串的适配层。非 Ollama 提供商直接使用支持批量输入的 `self.embedder`。

使用**平面索引（Flat Index）**，无近似聚类，保证召回精度。FAISS 索引在每次进程启动时从向量数据**内存重建**，不写入磁盘（冷启动 100K 向量约耗时 1-2 秒）。

---

## 5. 阶段三：查询时检索

**文件**: `api/rag.py:416`，实际由 `api/websocket_wiki.py` 调用

在调用 `RAG.call()` 前，`websocket_wiki.py` 会对请求的最后一条消息做 token 数检查（`websocket_wiki.py:80-84`）：

```python
tokens = count_tokens(last_message.content, request.provider == "ollama")
if tokens > 8000:
    input_too_large = True  # 跳过 RAG 检索，直接生成（无上下文）
```

`input_too_large = True` 时直接跳过 `RAG.call()`，以空上下文调用 LLM，并在 prompt 中注入 `<note>Answering without retrieval augmentation.</note>`。`rag.py:49` 定义的 `MAX_INPUT_TOKENS = 7500` 与此逻辑无关，目前**未被任何代码使用**。

```python
def call(self, query: str, language: str = "en") -> Tuple[List]:
    retrieved_documents = self.retriever(query)
    # retriever 内部：embed(query) → FAISS.search(top_k=20)

    retrieved_documents[0].documents = [
        self.transformed_docs[doc_index]
        for doc_index in retrieved_documents[0].doc_indices
    ]
    return retrieved_documents
```

值得注意的是，`RAG.call()` 只负责**检索**，不调用生成器。生成由上层 API 路由（`api.py` 或 `websocket_wiki.py`）控制，将 contexts 注入 prompt 后再调用 `self.generator`。这是一个架构缺陷：检索与生成的胶水代码散落在调用方。

这里存在一个**返回类型不一致**的问题：正常路径返回 `List[RetrieverOutput]`，但异常路径（`rag.py:441`）返回 `(RAGAnswer, [])`——一个 Tuple。调用方需同时处理两种返回类型，这是接口设计上的隐患。同时，检索后**缺少 re-ranking 或多样性控制**，直接按 FAISS 余弦距离原样使用。

---

## 6. 阶段四：Prompt 构建与流式生成

**文件**: `api/prompts.py`（RAG、SimpleChat 模板）, `api/rag.py`（Generator 初始化）, `api/websocket_wiki.py`（Deep Research 模板，内联定义）

### 6.1 Prompt 模板结构（RAG_TEMPLATE）

`RAG_TEMPLATE`（`prompts.py:31-57`）是 AdalFlow 的 Jinja2 模板，用于**常规 Chat 路径**（非 Deep Research）。完整内容：

```jinja2
<START_OF_SYS_PROMPT>
{system_prompt}
{output_format_str}
<END_OF_SYS_PROMPT>
{# 以 UUID 为 key 的 dict，遍历 DialogTurn #}
{% if conversation_history %}
<START_OF_CONVERSATION_HISTORY>
{% for key, dialog_turn in conversation_history.items() %}
{{key}}.
User: {{dialog_turn.user_query.query_str}}
You: {{dialog_turn.assistant_response.response_str}}
{% endfor %}
<END_OF_CONVERSATION_HISTORY>
{% endif %}
{% if contexts %}
<START_OF_CONTEXT>
{% for context in contexts %}
{{loop.index}}.
File Path: {{context.meta_data.get('file_path', 'unknown')}}
Content: {{context.text}}
{% endfor %}
<END_OF_CONTEXT>
{% endif %}
<START_OF_USER_PROMPT>
{{input_str}}
<END_OF_USER_PROMPT>
```

`{% if contexts %}` 块条件渲染——当 `RAG.call()` 未被调用（如 `input_too_large = True`）时，`contexts` 为 `None`，该块直接跳过。检索到的每个 chunk 单独列出，并附带 `File Path`，让 LLM 回答能引用具体文件位置。

### 6.2 结构化输出（Chain-of-Thought）

```python
# rag.py:147
@dataclass
class RAGAnswer(adal.DataClass):
    rationale: str  # Chain-of-Thought 思维链（内部使用，引导 LLM 先推理）
    answer: str     # 最终答案（展示给用户，Markdown 格式）
```

`DataClassParser` 强制 LLM 输出符合 schema，减少格式错误；`rationale` 字段引导先推理再回答，提升质量。

### 6.3 对话记忆（Memory）

```python
# rag.py:51
class Memory(adal.core.component.DataComponent):
    current_conversation: CustomConversation  # 自定义实现，替代 AdalFlow 原生 Conversation（修复 list assignment index out of range）

    def call(self) -> Dict[str, DialogTurn]:
        # 返回普通 Dict（非 OrderedDict），注入到 RAG_TEMPLATE 的 conversation_history
```

**无状态设计**：服务端不持久化会话，历史记录由前端随每次请求携带。每次 `call()` 重新读取全部历史注入 Prompt——历史轮数越多，Prompt 越长，无截断机制。

> `CustomConversation`（`rag.py:28`）是 DeepWiki 自定义的对话容器，用于替代 AdalFlow 原生 `Conversation` 类。原生实现在追加对话轮次时存在 `list assignment index out of range` 异常，自定义版本通过安全的 `append` 操作规避了该问题。

---

## 7. Deep Research 模式

**文件**: `api/websocket_wiki.py`

> `api/prompts.py` 中定义了 `DEEP_RESEARCH_FIRST_ITERATION_PROMPT`、`DEEP_RESEARCH_INTERMEDIATE_ITERATION_PROMPT`、`DEEP_RESEARCH_FINAL_ITERATION_PROMPT`，但 `websocket_wiki.py` **并未导入或使用**这些常量，而是在 `handle_websocket_chat()` 内以 f-string 形式内联了等效模板。两者的核心差异在于：内联版本在运行时插入了 `{repo_type}`、`{repo_url}`、`{repo_name}`、`{language_name}`、`{research_iteration}` 等上下文变量，而 `prompts.py` 中的静态字符串无法直接携带这些信息。`prompts.py` 中的三个模板是**功能重叠但实际未被调用的冗余代码**。

### 7.1 设计思路

Deep Research 是一个**伪 Agent loop**：LLM 不自主决策下一步，而是由前端负责计轮次、后端切换 Prompt 模板。每端保持简单，复杂度分散。

### 7.2 触发机制与轮次计数

前端通过在消息内容中嵌入 `[DEEP RESEARCH]` 标签触发；后端通过统计历史中 `assistant` 消息数量计算当前迭代编号（无需服务端 session 状态）：

```python
research_iteration = sum(1 for msg in request.messages if msg.role == "assistant") + 1
```

### 7.3 Prompt 切换策略

| 轮次 | Prompt 模板 | 结构约定 |
|------|-------------|---------|
| 1 | `FIRST_ITERATION_PROMPT` | `## Research Plan` 制定计划 + 初步发现 |
| 2-4 | `INTERMEDIATE_ITERATION_PROMPT` | `## Research Update N` 深入新角度，不重复已覆盖内容 |
| 5+ | `FINAL_ITERATION_PROMPT` | `## Final Conclusion` 综合所有轮次输出结论 |

### 7.4 每轮 Prompt 完整结构与上下文组装

**上下文组装**（`websocket_wiki.py:215-234`）：检索到的 chunks 先**按文件路径分组**，同一文件的多个 chunk 合并为一个块，文件间以 10 个短横线分隔：

```python
# 按文件分组
docs_by_file = {}
for doc in documents:
    file_path = doc.meta_data.get("file_path", "unknown")
    docs_by_file[file_path].append(doc)

# 格式化
context_parts = []
for file_path, docs in docs_by_file.items():
    header = f"## File Path: {file_path}\n\n"
    content = "\n\n".join([doc.text for doc in docs])
    context_parts.append(f"{header}{content}")

context_text = "\n\n" + "----------\n\n".join(context_parts)
```

**完整 prompt 拼接顺序**（`websocket_wiki.py:418-438`）：

```
/no_think {system_prompt}

<conversation_history>
  <turn><user>...</user><assistant>...</assistant></turn>
</conversation_history>

（可选）<currentFileContent path="src/xxx.py">
...文件内容...
</currentFileContent>

<START_OF_CONTEXT>

## File Path: api/rag.py

...chunk text A...

...chunk text B...（同文件多个 chunk 直接拼接）

----------

## File Path: api/data_pipeline.py

...chunk text...
<END_OF_CONTEXT>

<query>
{当前研究问题}
</query>
```
prompt 末尾追加 `\n\nAssistant: `（`websocket_wiki.py:438`），引导 LLM 直接开始回答。若使用 Ollama provider，还会再追加 ` /no_think`（`websocket_wiki.py:443`）。
每轮使用**相同的原始用户问题**进行 FAISS 检索（`rag_query = query`），不随迭代演化，意味着每轮可能检索到相同代码块。

---

## 8. LLM 提供商抽象

通过 `provider` 参数在运行时切换，所有提供者支持流式输出接口：

| 提供者 | 客户端文件 |
|--------|-----------|
| google | `google_embedder_client.py` |
| openai | OpenAI Client (adalflow) |
| openrouter | `openrouter_client.py` |
| ollama | `ollama_patch.py` |
| bedrock | `bedrock_client.py` |
| azure | `azureai_client.py` |
| dashscope | `dashscope_client.py` |

Embedder（嵌入模型）和 Generator（生成模型）各自独立配置，可以混用不同提供商。

---

## 9. 关键设计汇总

| 设计 | 选择 | 理由 |
|------|------|------|
| 向量存储 | FAISS 平面索引 | 精度优先；大型仓库可考虑 IVF 近似索引 |
| 分块策略 | 350词 + 100词重叠 | 平衡语义连贯性与检索粒度 |
| 嵌入缓存 | LocalDB.pkl | 避免重复 API 调用；切换 embedder 需清除缓存 |
| Embedder/Generator 解耦 | 独立配置 | 可混用不同提供商 |
| 生成器输出格式 | DataClassParser + RAGAnswer | 强制结构化，减少格式错误 |
| 对话历史 | 全量注入 Prompt | 简单可靠，但长对话会消耗大量 token |
| FAISS 索引 | 不落盘，每次重建 | 实现简单；100K 向量重建约 1-2 秒 |
| Deep Research | 前端计轮次 + 后端切模板 | 每端保持简单，无需服务端状态 |

---

## 10. 相关文件索引

| 文件 | 职责 |
|------|------|
| `api/rag.py` | RAG 核心：Memory、检索、Generator 初始化 |
| `api/data_pipeline.py` | 仓库克隆、文档读取、分块、嵌入、数据库管理 |
| `api/websocket_wiki.py` | WebSocket 请求协调、Deep Research 流程控制 |
| `api/prompts.py` | RAG 和 SimpleChat 的 Prompt 模板（其中的 Deep Research prompt 模板为冗余代码，实际运行时的 prompt 由 `websocket_wiki.py` 以 f-string 内联定义，含运行时变量） |
| `api/config.py` | 配置加载、Provider 客户端映射 |
| `api/config/generator.json` | LLM 提供商和模型定义 |
| `api/config/embedder.json` | 嵌入模型配置、分块参数、top_k |
| `api/config/repo.json` | 文件包含/排除规则 |
| `api/tools/embedder.py` | Embedder 工厂函数 |
| `api/ollama_patch.py` | Ollama 单文档处理器适配 |

