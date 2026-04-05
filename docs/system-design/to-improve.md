# DeepWiki 改进方向

本文档记录在分析 RAG 流程、数据管道和系统架构后发现的可改进点，按优先级排列。

---

## 一、RAG 检索质量

### 1.1 代码感知分块

**现状**：`TextSplitter` 以 350 词为单位做滑动窗口分块，不感知代码结构。一个函数可能被切成两个 chunk，语义被人为割裂。

**问题**：
- 检索到的 chunk 可能缺失关键的类定义或导入语句
- 跨 chunk 边界的函数在 embedding 空间中语义不完整

**可能的改进方向**：引入 AST 感知分块器，以函数/类为最小单元，再按 token 预算合并相邻节点。可参考：
- [tree-sitter](https://github.com/tree-sitter/tree-sitter)：支持 Python/JS/TS/Go 等主流语言的 AST 解析
- LlamaIndex 的 `CodeSplitter`：已实现基于 tree-sitter 的代码感知分块

```python
# 期望的分块粒度（当前 vs 目标）
# 当前：word-based sliding window (350 words)
# 目标：function/class boundary → 再按 token 上限合并
```

### 1.2 元数据过滤支持

**现状**：FAISS 检索阶段只看向量相似度，`file_path`、`is_code` 等元数据在检索后才被使用，无法在搜索阶段过滤（如只检索 `src/` 目录）。

**可能的改进方向**：在 FAISS 返回结果后增加元数据后过滤层，或迁移到支持混合检索的向量数据库（如 Qdrant、Weaviate），直接在搜索时施加 `where` 过滤条件。

### 1.3 引入 Re-ranking（检索后精排）

**现状**：`RAG.call()` 在 FAISS 返回 top_k=20 个结果后，直接按余弦距离排序原样交给 LLM，没有任何后处理：

```python
# api/rag.py:427-435
retrieved_documents = self.retriever(query)
retrieved_documents[0].documents = [
    self.transformed_docs[doc_index]
    for doc_index in retrieved_documents[0].doc_indices
]
return retrieved_documents  # 直接返回，无精排
```

**问题**：
- 向量相似度高 ≠ 对回答有用，FAISS 分数反映的是嵌入空间距离而非"是否能回答问题"
- top_k=20 的结果可能集中在同一文件的相邻 chunk，存在高冗余
- 无多样性控制，关键的跨文件依赖 chunk 可能被排在末尾甚至截断

**可能的改进方向**：引入两阶段检索：
1. FAISS 粗检索（召回 top-100）
2. Cross-encoder 精排（输出最终 top-20）

Cross-encoder 对"query + chunk"拼接直接打相关性分数，精度显著高于纯向量距离。可参考：
- `ms-marco-MiniLM-L-6-v2`：轻量级 cross-encoder，延迟约 20-50ms
- `flashrank`：专为 RAG 设计的轻量重排库，无需 GPU

### 1.4 对话历史截断

**现状**：`Memory.call()` 将全部历史 `dialog_turns` 注入 Prompt，轮数越多 token 消耗线性增长，无截断或摘要机制。

**可能的改进方向**：
- 滑动窗口：只保留最近 N 轮
- 摘要压缩：超过阈值后将旧轮次摘要为一段文字
- Token 预算控制：检测 context 长度后动态裁剪历史

---

## 二、数据摄取性能

### 2.1 嵌入计算并行化

**现状**：`transform_documents_and_save_to_db()` 串行处理所有 chunk，大型仓库（如 > 5000 文件）摄取慢。

**可能的改进方向**：对 `ToEmbeddings` 的 batch 调用使用 `asyncio` 或 `multiprocessing.Pool` 并行化，嵌入 API 本身是 I/O 密集型操作，并发收益明显。

### 2.2 FAISS 索引增量更新

**现状**：仓库内容更新后，必须删除**整个 `.pkl` 文件**重新摄取，无增量更新机制。

**可能的改进方向**：记录文件的 `mtime` 或 git commit hash，只对变更文件重新嵌入并更新 FAISS 索引中对应的向量条目。

### 2.3 Ollama 嵌入吞吐

**现状**：`OllamaDocumentProcessor` 逐文档调用 Ollama，约 1-2s/请求，无批量支持。

**可能的改进方向**：调研 Ollama 是否支持批量嵌入 API（`/api/embed` 端点的 `input` 字段已支持数组）；若支持，修改 `OllamaDocumentProcessor` 以批量调用。

---

## 三、架构清晰度

### 3.1 RAG 生成逻辑分散

**现状**：`RAG.call()` 只做检索，`self.generator` 在 `__init__` 中初始化但实际调用在 `api.py` 的路由层，contexts 注入逻辑散落在调用方，增加了理解和测试的难度。

**可能的改进方向**：将"检索 + contexts 注入 + 生成"封装为一个完整的 `RAG.query(input_str)` 方法，让 `RAG` 类成为真正自包含的组件，`api.py` 只负责 HTTP/WebSocket 层。

### 3.2 `handle_websocket_chat` 是一个 916 行的上帝函数

**现状**：`api/websocket_wiki.py` 中 `handle_websocket_chat` 函数长达 916 行，包含：
- 7 个 provider 的 `if-elif` 处理链（正常路径）
- 相同的 7 个 provider 处理链再重复一次（fallback 路径）
- 请求解析、Memory 重建、Deep Research 检测、Prompt 拼装、LLM 调用全混在一起

添加新 provider 时需要在两处同步修改，极易遗漏。

**可能的改进方向**：

1. 将每个 provider 的调用封装为独立函数或策略类：

```python
# 目标结构
class LLMProvider(Protocol):
    async def stream(self, prompt: str, model_config: dict) -> AsyncGenerator[str, None]: ...

PROVIDERS: dict[str, LLMProvider] = {
    "google": GoogleProvider(),
    "openai": OpenAIProvider(),
    "ollama": OllamaProvider(),
    # ...
}
```

2. fallback 路径复用同一 provider 实例，只替换 prompt，消除重复逻辑。

### 3.3 Prompt 模板在两处定义（死代码）

**现状**：`api/prompts.py` 中定义了 `DEEP_RESEARCH_FIRST_ITERATION_PROMPT`、`DEEP_RESEARCH_FINAL_ITERATION_PROMPT`、`DEEP_RESEARCH_INTERMEDIATE_ITERATION_PROMPT`，但 `websocket_wiki.py` 中 `handle_websocket_chat` 并未 import 这些常量，而是用内联 f-string 重新定义了几乎相同的内容（`websocket_wiki.py:266-357`）。

`prompts.py` 中的三个 Deep Research 模板是**实际未被使用的死代码**。

**可能的改进方向**：删除 `prompts.py` 中的重复定义，在 `websocket_wiki.py` 中改为 import 并使用 `prompts.py` 的常量，或反过来将 inline 逻辑迁移到 `prompts.py`，统一维护。

### 3.4 `RAG` 对象在 `input_too_large` 时仍被完整初始化

**现状**：`websocket_wiki.py:88-109` 中，`RAG(...)` 实例化和 `prepare_retriever()`（含仓库克隆、FAISS 索引加载）在判断 `input_too_large` 之前执行。即使输入过大、最终跳过 RAG 检索，这些昂贵操作依然发生。

```python
# 当前顺序（有问题）
request_rag = RAG(...)          # 昂贵
request_rag.prepare_retriever() # 更昂贵（可能触发克隆）
# ...
if input_too_large:
    # 跳过检索，但上面的工作已做完
```

**可能的改进方向**：将 `input_too_large` 检测提前到 RAG 初始化之前；或将 `prepare_retriever()` 改为懒加载，仅在实际调用 `retriever` 时触发。

### 3.5 Deep Research `[DEEP RESEARCH]` 协议耦合在消息内容中

**现状**：Deep Research 模式通过扫描消息 content 中的字符串 `[DEEP RESEARCH]` 来触发，这将传输协议耦合进业务内容。副作用是历史消息中的 tag 需要刻意保留（除最后一条外不清除），增加了理解难度。

**可能的改进方向**：在 `ChatCompletionRequest` 中增加一个显式字段：

```python
class ChatCompletionRequest(BaseModel):
    # ...
    deep_research: bool = Field(False, description="Enable Deep Research mode")
```

前端直接传 `deep_research: true`，无需在消息内容中嵌入 tag，后端也不需要扫描历史消息。

### 3.6 "continue research" 检测逻辑脆弱

**现状**：`websocket_wiki.py:175` 用简单字符串匹配判断是否为续研究请求：

```python
if "continue" in last_message.content.lower() and "research" in last_message.content.lower():
```

用户真实提问如 "I want to continue researching this topic in depth" 会误触发，导致原始问题被替换为上一个 topic。

**可能的改进方向**：通过前端显式发送特定指令（如专用 `action` 字段）替代模糊的关键词匹配，或用更严格的模式（如精确匹配 "continue the research"）降低误判率。

### 3.7 Google provider 在 async 上下文中使用同步 API

**现状**：`websocket_wiki.py:709-712` 中 Google provider 调用同步方法：

```python
response = model.generate_content(prompt, stream=True)  # 同步调用
for chunk in response:                                   # 同步迭代
    await websocket.send_text(chunk.text)
```

在 async WebSocket handler 中执行同步阻塞 I/O 会阻塞整个事件循环，影响并发性能。其他 provider 均使用 `await model.acall()`。

**可能的改进方向**：使用 `asyncio.to_thread()` 包装同步调用，或调研 `google.generativeai` 是否提供原生 async API（`generate_content_async`）。

---

## 四、Deep Research 专项

### 4.1 各迭代使用相同 RAG 查询（未自适应）

**现状**：Deep Research 的每次迭代都用原始用户问题进行 FAISS 检索（`rag_query = query`）。随着迭代深入，LLM 已经发现了新线索，但检索查询从未更新，可能导致每轮检索到相同的代码块。

**可能的改进方向**：在中间迭代中，基于上一轮 LLM 输出提取"下一步调查方向"，用这个方向构造新的检索查询（类似 HyDE 或 query rewriting 技术）。

### 4.2 最终轮次编号硬编码

**现状**：`websocket_wiki.py:263` 中 `is_final_iteration = research_iteration >= 5` 是一个魔法数字，研究深度不可配置。

**可能的改进方向**：将最大迭代次数提取为常量或 `ChatCompletionRequest` 中的可选参数 `max_research_iterations`，允许用户/前端控制研究深度。

---

## 五、可观测性

### 5.1 检索结果质量追踪

**现状**：检索到的 chunk 直接注入 Prompt，没有记录哪些 chunk 被检索到、相似度分数是多少。

**可能的改进方向**：在 DEBUG 日志中记录 top-k 结果的 `file_path` 和相似度分数，便于调优 `top_k`、chunk size 等超参数。





## Insights
possible: 类似于graphRAG的方法做一下，让大模型直接决策去拆分代码