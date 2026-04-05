# DeepWiki Agent 设计文档

## 1. 整体架构

DeepWiki 没有使用显式的 Agent 类或通用 Agent 框架，而是通过**组件化 RAG 架构**实现 Agent 行为。核心设计原则是最小抽象、垂直优化，专为"理解代码仓库"场景服务。

### 组件关系图

```
用户请求 (WebSocket/HTTP)
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
│  └── Generator (adal)       │  LLM 封装
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  DatabaseManager            │  数据层
│  (api/data_pipeline.py)     │
│  ├── 仓库克隆               │
│  ├── TextSplitter           │  350词块 / 100词重叠
│  └── FAISS 索引持久化       │
└─────────────────────────────┘
```

### 请求完整数据流

```
① 用户发送 GitHub URL + 问题
② DatabaseManager 克隆仓库 → 读取文件 → 分块 → 生成嵌入 → 存入 FAISS
③ RAG.call(query) → 查询向量化 → FAISS 相似度检索 → 返回相关代码块
④ 拼装 Prompt：系统提示 + 对话历史(Memory) + 检索上下文 + 当前问题
⑤ 调用 LLM 提供者流式生成 → WebSocket 推送到前端
```

---

## 2. 核心组件

### 2.1 RAG 类 (`api/rag.py`)

整个 Agent 的大脑，协调所有检索和生成逻辑。

```python
class RAG(adal.Component):
    def __init__(self, provider="google", model=None, use_s3: bool = False):
        self.memory = Memory()             # 对话历史
        self.embedder = get_embedder(...)  # 嵌入模型
        self.db_manager = DatabaseManager()
        self.retriever = FAISSRetriever(...)
```

### 2.2 Memory 类 (`api/rag.py`)

管理多轮对话上下文。

- 每次请求重建，历史记录由前端随请求携带（服务端无状态）
- 每轮对话注入 Prompt 的格式：

```xml
<conversation_history>
  <turn>
    <user>上一个问题</user>
    <assistant>上一个回答</assistant>
  </turn>
</conversation_history>
```

### 2.3 DatabaseManager 类 (`api/data_pipeline.py`)

仓库克隆、文件处理、向量索引管理。

处理流程：
1. 克隆仓库（支持 GitHub / GitLab / Bitbucket，支持 access token）
2. 按规则过滤文件（`api/config/repo.json`）
3. TextSplitter 分块（350词块，100词重叠）
4. 生成嵌入（支持多提供者）
5. 序列化 FAISS 索引到 `~/.adalflow/databases/`

**缓存机制：** 若 pickle 文件已存在且嵌入有效，直接复用，跳过重新克隆和嵌入。

### 2.4 多 LLM 提供者抽象

通过 `provider` 参数在运行时切换，所有提供者暴露统一流式接口：

| 提供者 | 客户端 |
|--------|--------|
| google | GoogleGenAI Client |
| openai | OpenAI Client (adalflow) |
| openrouter | OpenRouter Client |
| ollama | Ollama Client (adalflow) |
| bedrock | AWS Bedrock Client |
| azure | Azure AI Client |
| dashscope | Dashscope Client |

---

## 3. Deep Research 模式

### 3.1 设计思路

Deep Research 是一个**伪 Agent loop**：不是 LLM 自主决策下一步，而是前端负责计数轮次、后端切换 Prompt 模板的协作设计。复杂度分散到两端，每端保持简单。

### 3.2 触发机制

前端在用户点击"Deep Research"时，将 `[DEEP RESEARCH]` 标签拼接到消息内容中：

```python
# api/websocket_wiki.py
for msg in request.messages:
    if "[DEEP RESEARCH]" in msg.content:
        is_deep_research = True
        if msg == request.messages[-1]:
            msg.content = msg.content.replace("[DEEP RESEARCH]", "").strip()
```

### 3.3 迭代计数

直接统计历史消息中 `assistant` 条数，无需服务端 session 状态：

```python
research_iteration = sum(1 for msg in request.messages if msg.role == 'assistant') + 1
```

| 消息历史 | research_iteration |
|----------|--------------------|
| [user] | 1 |
| [user, assistant, user] | 2 |
| [user, assistant, user, assistant, user] | 3 |

### 3.4 Prompt 模板切换逻辑

```
iteration == 1    → FIRST_ITERATION_PROMPT
2 <= iteration <= 4 → INTERMEDIATE_ITERATION_PROMPT（含 iteration 编号）
iteration >= 5    → FINAL_ITERATION_PROMPT
```

各轮次的结构约定：

```
轮次 1  → "## Research Plan"       制定研究计划 + 初步发现
轮次 2  → "## Research Update 2"   深入新角度，不重复轮次 1
轮次 3  → "## Research Update 3"   继续深挖，提示下轮收尾
轮次 4  → "## Research Update 4"   填补空白
轮次 5+ → "## Final Conclusion"    综合所有轮次，输出结论
```

所有中间轮次的 Prompt 均包含约束：**不重复已覆盖内容，专注于新发现**。

### 3.5 "continue research" 的特殊处理

用户发送 "continue" 类消息时，后端从历史中找回原始问题，防止 LLM 把 "continue" 当新问题处理：

```python
if "continue" in last_message.content.lower() and "research" in last_message.content.lower():
    for msg in request.messages:
        if msg.role == "user" and "continue" not in msg.content.lower():
            original_topic = msg.content.replace("[DEEP RESEARCH]", "").strip()
            break
    if original_topic:
        last_message.content = original_topic
```

### 3.6 每轮 Prompt 完整结构

```
/no_think {system_prompt}

<conversation_history>
  <turn>
    <user>...</user>
    <assistant>...</assistant>
  </turn>
</conversation_history>

<currentFileContent path="...">   （可选，用户指定文件）
  ...
</currentFileContent>

<START_OF_CONTEXT>                （RAG 检索到的相关代码块，按文件分组）
  ## File Path: src/xxx.py
  ...
<END_OF_CONTEXT>

<query>
  {当前研究问题}
</query>