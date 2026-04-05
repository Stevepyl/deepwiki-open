# DeepWiki RAG 系统问题清单

---

## 1. 超限文件被静默丢弃

**位置**: `data_pipeline.py:326-328`, `data_pipeline.py:360-362`

代码文件超过 81,920 tokens 或文档文件超过 8,192 tokens 时，仅记录 `logger.warning` 后 `continue` 跳过，不会通知上层调用者或最终用户。

```python
# data_pipeline.py:326
if token_count > MAX_EMBEDDING_TOKENS * 10:
    logger.warning(f"Skipping large file {relative_path}: Token count ({token_count}) exceeds limit")
    continue
```

**影响**: 大型核心文件（如单体配置文件、生成代码）可能完全不参与 RAG 检索，导致回答覆盖盲区。警告仅写入服务端日志（`logger.warning`），不反映在 API 响应或 WebSocket 消息中，前端和用户完全无感知哪些文件被排除。

---

## 2. 缓存无 Embedder 溯源，切换提供商需手动干预

**位置**: `data_pipeline.py:869-892`

`.pkl` 缓存文件中不记录生成嵌入时使用的提供商和模型信息。加载时仅检查向量是否非空（`non_empty == 0`），不验证向量来源。

```python
# data_pipeline.py:887-892
if non_empty == 0:
    logger.warning("Existing database contains no usable embeddings. Rebuilding embeddings...")
else:
    return documents  # 直接返回，不管向量由哪个 embedder 生成
```

**影响**: 切换 embedder 提供商后，旧向量仍非空，系统不会自动重建，用户必须手动删除 `~/.adalflow/databases/{repo}.pkl` 才能触发重新构建嵌入。若新旧 embedder 维度不同（如从 OpenAI 256 维切换到 Google 可变维度），后续的 `_validate_and_filter_embeddings` 可以捕获维度不匹配并过滤；若维度相同，则过滤也不会触发，问题完全透明通过（见问题 3）。

---

## 3. 嵌入维度校验存在盲区

**位置**: `rag.py:300`

`_validate_and_filter_embeddings()` 以多数派维度作为目标维度，过滤掉维度不一致的文档。但当新旧 embedder 输出**相同维度**（如两个都输出 256 维）但语义空间不同时，校验无法检测到问题。

```python
# rag.py:300
target_size = max(embedding_sizes.keys(), key=lambda k: embedding_sizes[k])
```

**具体触发场景**: `api/config/embedder.json` 中，OpenAI（`text-embedding-3-small`）和 Bedrock（`amazon.titan-embed-text-v2:0`）均配置为 **256 维**。用户从 OpenAI 切换到 Bedrock（或反之）后：

1. 旧 `.pkl` 非空 → `data_pipeline.py:892` 直接 `return documents`，跳过重建
2. `_validate_and_filter_embeddings` 扫描到所有向量均为 256 维 → 无维度不匹配 → 全部通过
3. 查询时，FAISS 用 Bedrock 生成的查询向量去搜索 OpenAI 生成的文档向量 → 不同语义空间，余弦距离无意义
4. 检索无报错，但 top-20 结果与查询语义不相关 → **系统"正常运行"，但回答质量已退化**

**影响**: 此场景无任何错误日志产生，是系统中最难被发现的 silent failure。用户只能通过回答质量主观感知，而非错误信息定位。

---

## 4. `RAG.call()` 返回类型不一致

**位置**: `rag.py:416-445`

正常路径返回 `List[RetrieverOutput]`，异常路径返回 `(RAGAnswer, [])`（Tuple），两种返回类型不统一。

```python
# 正常路径 — rag.py:435
return retrieved_documents  # List[RetrieverOutput]

# 异常路径 — rag.py:441-445
return error_response, []   # Tuple[RAGAnswer, list]
```

**精确崩溃路径**:

```python
# rag.py:437-445 — 异常路径
except Exception as e:
    error_response = RAGAnswer(rationale="...", answer="...")
    return error_response, []          # 返回 Tuple[RAGAnswer, list]

# websocket_wiki.py:208-210 — 调用方
retrieved_documents = request_rag(rag_query, ...)
# retrieved_documents = (RAGAnswer_obj, [])

if retrieved_documents and retrieved_documents[0].documents:
#                          ↑ RAGAnswer_obj（只有 rationale: str 和 answer: str）
#                                          ↑ AttributeError: 'RAGAnswer' object has no attribute 'documents'
```

`RAGAnswer` 是仅含 `rationale: str` 和 `answer: str` 的 `@dataclass`（`rag.py:1-50`），没有 `documents` 属性。`AttributeError` 会被调用方的外层 `except Exception` 捕获，触发 fallback（不含 RAG context 的重试），但原始错误信息丢失，难以诊断根因。

---

## 5. 检索后无 Re-ranking 或多样性控制

**位置**: `rag.py:427-433`, `websocket_wiki.py:210-231`

FAISS 返回 top-k 结果后，直接按原始余弦距离排序使用，未经过任何 re-ranking、MMR（Maximal Marginal Relevance）或多样性过滤。

经 Grep 搜索确认：整个 `api/` 目录中不存在 `rerank`、`re-rank`、`diversity`、`mmr` 相关代码。

**影响**: 检索结果可能高度相似（特别是相邻 chunk 因 100 词重叠而语义接近），占用有限的 context 窗口却不提供增量信息。

---

## 6. 对话历史全量注入，无截断机制

**位置**: `websocket_wiki.py:412-421`, `simple_chat.py:302-311`

每次请求将全部历史对话轮次拼接注入 Prompt，没有基于 token 数量的截断或滑动窗口。

```python
# websocket_wiki.py:412-415
conversation_history = ""
for turn_id, turn in request_rag.memory().items():
    if not isinstance(turn_id, int) and hasattr(turn, 'user_query') and hasattr(turn, 'assistant_response'):
        conversation_history += f"<turn>\n<user>{turn.user_query.query_str}</user>\n<assistant>{turn.assistant_response.response_str}</assistant>\n</turn>\n"
```

**已有 fallback，但属被动触发**: `websocket_wiki.py` 和 `simple_chat.py` 都有 token limit fallback，但逻辑是：

1. 发送含全量历史 + RAG context 的完整 prompt → 等待 LLM 处理（一次完整 API 往返）
2. LLM 返回 token limit 错误（关键词匹配：`"maximum context length"` / `"token limit"` / `"too many tokens"`）
3. 重建 `simplified_prompt`：**去掉 RAG context，但全量历史仍全部保留**
4. 重发

```python
# websocket_wiki.py:724-727 — fallback 路径
simplified_prompt = f"/no_think {system_prompt}\n\n"
if conversation_history:
    simplified_prompt += f"<conversation_history>\n{conversation_history}</conversation_history>\n\n"
# ↑ 历史仍全量注入，只去掉了 RAG context
```

**影响**: fallback 的保护上限是"无 RAG context 的回答"，不能防止历史本身过长导致的失败。且 fallback 触发前已有一次失败的 LLM 请求产生延迟，用户感知到明显卡顿。

---

## 7. Deep Research 每轮使用相同查询进行 FAISS 检索

**位置**: `websocket_wiki.py:175-186`, `websocket_wiki.py:190`

Deep Research 的后续迭代使用 `"continue research"` 类消息触发，后端将其替换为第一轮的原始用户问题后进行 FAISS 检索。

```python
# websocket_wiki.py:175-186
if "continue" in last_message.content.lower() and "research" in last_message.content.lower():
    original_topic = None
    for msg in request.messages:
        if msg.role == "user" and "continue" not in msg.content.lower():
            original_topic = msg.content.replace("[DEEP RESEARCH]", "").strip()
            break
    if original_topic:
        last_message.content = original_topic
```

**影响**: 每轮检索到的代码块高度重复。虽然 Prompt 模板指示 LLM "不重复已覆盖内容"，但 LLM 收到的上下文每次都相同，深入研究依赖 LLM 自身从相同上下文中提取不同角度，而非通过检索发现新信息。

---

## 8. 使用 Ollama 部署的 embedding 模型不支持批处理

**位置**: `ollama_patch.py:62-101`

`OllamaDocumentProcessor` 逐文档调用 embedder，每次只处理一个文档，受限于 AdalFlow Ollama Client 不支持批量嵌入。

```python
# ollama_patch.py:78-81
for i, doc in enumerate(tqdm(output, desc="Processing documents for Ollama embeddings")):
    result = self.embedder(input=doc.text)
```

**影响**: 每个文档的嵌入请求约 1-2 秒（含网络开销），一个 1000 文件、平均 2 chunk/文件 的仓库需要约 30-60 分钟完成嵌入。对于中大型仓库，Ollama 嵌入实际不可用。

---

## 9. 检索与生成逻辑分离，胶水代码分散

**位置**: `rag.py:416-435`（仅检索）, `websocket_wiki.py:411-436`（Prompt 构建与生成）, `simple_chat.py:302-311`（同样的 Prompt 构建逻辑）

`RAG.call()` 只负责 FAISS 检索并返回文档，不调用 Generator。Prompt 的上下文注入和 LLM 调用由 `websocket_wiki.py` 和 `simple_chat.py` 各自实现，存在重复的上下文格式化逻辑。

**重复程度**: 经代码对比确认，两个文件的 prompt 构建代码**字符级完全相同**（非"类似"），连变量名、注释、常量命名均一致：

```python
# websocket_wiki.py:412-445  ≡  simple_chat.py:302-335（逐字相同）
conversation_history = ""
for turn_id, turn in request_rag.memory().items():
    if not isinstance(turn_id, int) and hasattr(turn, "user_query") and hasattr(turn, "assistant_response"):
        conversation_history += f"<turn>\n<user>...</user>\n<assistant>...</assistant>\n</turn>\n"

prompt = f"/no_think {system_prompt}\n\n"
if conversation_history:
    prompt += f"<conversation_history>\n{conversation_history}</conversation_history>\n\n"
if file_content:
    prompt += f"<currentFileContent path=\"{request.filePath}\">\n{file_content}\n</currentFileContent>\n\n"
CONTEXT_START = "<START_OF_CONTEXT>"
CONTEXT_END = "<END_OF_CONTEXT>"
if context_text.strip():
    prompt += f"{CONTEXT_START}\n{context_text}\n{CONTEXT_END}\n\n"
else:
    prompt += "<note>Answering without retrieval augmentation.</note>\n\n"
prompt += f"<query>\n{query}\n</query>\n\nAssistant: "
```

token limit fallback 中的 `simplified_prompt` 构建逻辑（`websocket_wiki.py:724-737` 与 `simple_chat.py:568-581`）同样完全重复。

**影响**: 两个调用方独立维护上下文格式化代码，修改 prompt 格式时必须同步更新两处，且容易出现不同步导致行为不一致。`RAG` 类持有 `self.generator` 但不在 `call()` 中使用，语义上具有误导性。

