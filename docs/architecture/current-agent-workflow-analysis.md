---
number: DOC-010
name: Current Agent Workflow Analysis
description: 分析当前 DeepWiki-Open 自定义 Agent 的工作模式、入口流程、工具链、事件协议和风险边界。
update_at: 2026-05-08
category: architecture
language: zh-CN
audience: developers-and-agents
---

# 当前 Agent 工作模式与流程分析

## 1. 结论

当前 DeepWiki-Open 的 Agent 不是 OpenAI Agents SDK，也不是旧版 `api/rag.py` 里的单次 RAG 生成器，而是一套仓库内自实现的 ReAct 工具循环：

```
用户问题
  -> AgentConfig 选择 agent 类型、工具集、step 上限、system prompt
  -> UnifiedProvider 流式调用 LLM
  -> LLM 输出 text_delta 或 tool_call_start
  -> Agent Loop 并发执行工具
  -> 工具结果作为 tool message 回灌给 LLM
  -> 重复直到无工具调用、出错或达到 step 上限
```

它的核心特征是：

- **证据驱动**：系统 prompt 要求先用 `rag_search`、`grep`、`read`、`glob`、`ls` 等工具验证，再回答或写 wiki。
- **请求级无状态**：前端每次提交完整 `messages`；后端不保存长期会话状态，只在本次请求中维护 loop conversation。
- **工具受 agent 配置控制**：不同 agent 的工具集、step 上限和 prompt 不同。
- **流式事件协议**：对前端输出结构化 `text_delta`、`tool_call_start`、`tool_call_end`、`finish`、`error`。
- **RAG 已变成工具**：`rag_search` 只负责语义检索，不负责生成；生成仍由 Agent Loop 和 LLM 完成。

## 2. 当前有三条相关路径

| 路径 | 入口 | 当前角色 | 是否结构化 Agent |
|---|---|---|---|
| Ask Agent Chat | 前端 `src/app/[owner]/[repo]/ask/page.tsx` -> `/ws/agent-chat`，失败后走 `/api/chat/agent-stream` -> `/chat/agent-stream` | 面向用户的仓库问答 | 是 |
| Agent Wiki | `/ws/agent-wiki` -> `api/agent/wiki_generator.py` | 两阶段生成 wiki 结构和页面内容 | 是 |
| Legacy RAG Chat | `/ws/chat`、`/chat/completions/stream` | 旧的 RAG 文本流，仍供部分 wiki/workshop/slides 生成路径使用 | 否 |

注意：旧文档 `docs/architecture/agent-design.md` 主要描述 RAG 组件如何形成类 Agent 行为；当前真正的结构化工具 Agent 以 `api/agent/` 为实现中心。

## 3. Agent 类型

Agent 配置集中在 `api/agent/config.py`。当前注册 5 个内置 agent：

| Agent | 对外入口 | 工具模式 | Step 上限 | 作用 |
|---|---|---|---:|---|
| `explore` | Ask 默认 agent，可由 `task` 调用 | 只读：`grep`、`glob`、`ls`、`rag_search`、`read` | 15 | 快速仓库探索和普通问答 |
| `wiki` | Ask 可选 agent | 全工具：含 `bash`、`task`、`todowrite` | 25 | 更强的通用仓库问答 |
| `deep-research` | Ask 的 Deep Research 模式 | 全工具，强调计划、交叉验证和多步追踪 | 40 | 深入研究型问题 |
| `wiki-planner` | 仅 `/ws/agent-wiki` 内部使用 | 只读工具 | 20 | 探索仓库后输出严格 wiki JSON 结构 |
| `wiki-writer` | 仅 `/ws/agent-wiki` 内部使用 | `grep`、`glob`、`ls`、`rag_search`、`read`、`bash` | 25 | 按单页主题执行 explore-then-write 写作 |

`/ws/agent-chat` 和 `/chat/agent-stream` 只允许 `wiki`、`explore`、`deep-research`。`wiki-planner` 和 `wiki-writer` 依赖特定 prompt 形状，不能作为通用聊天 agent 暴露。

## 4. Ask Agent Chat 流程

Ask 页的实际流程：

1. 前端根据 URL、设置面板和聊天历史构造 `AgentChatRequest`。
2. 默认 `agent_name` 是 `explore`；打开 Deep Research 后改为 `deep-research`；用户也可以在 composer 里切换 agent。
3. 前端优先建立 `ws://<backend>/ws/agent-chat`。
4. WebSocket 失败时，前端清空当前 assistant 流式内容，降级为 `/api/chat/agent-stream`，再由 Next.js 代理到后端 `/chat/agent-stream`。
5. 后端 `AgentChatRequest` 校验 `agent_name`、`messages`、最后一条消息必须是 `user`。
6. 后端将 legacy `{role, content}` 消息转换成内部 `AgentMessage`。
7. 后端用 `download_repo()` 把仓库同步到 `~/.adalflow/repos/{owner}_{repo}`。
8. 后端解析 include/exclude 过滤规则，用 `FilteredToolWrapper` 包装工具。
9. 后端创建 `UnifiedProvider(provider, model)`，再调用 `run_agent_loop()`。
10. 前端根据事件更新 UI：`text_delta` 拼接正文，`tool_call_start`/`tool_call_end` 更新工具卡片，`finish` 结束流式状态。

简化时序：

```mermaid
sequenceDiagram
    participant UI as Ask Page
    participant Proxy as Next Proxy
    participant API as FastAPI
    participant Loop as Agent Loop
    participant Tool as Tools
    participant LLM as UnifiedProvider

    UI->>API: WebSocket /ws/agent-chat
    UI-->>Proxy: fallback POST /api/chat/agent-stream
    Proxy-->>API: POST /chat/agent-stream
    API->>API: validate request, clone repo, wrap filters
    API->>Loop: run_agent_loop(config, messages, provider, tools)
    Loop->>LLM: stream_chat(messages, tool schemas)
    LLM-->>UI: text_delta / tool_call_start
    Loop->>Tool: execute tool calls in parallel
    Tool-->>Loop: ToolResult
    Loop-->>UI: tool_call_end
    Loop->>LLM: append tool results and continue
    Loop-->>UI: finish or error
```

## 5. Agent Loop 工作模式

`api/agent/loop.py` 是当前 Agent 的执行核心。每轮循环做以下事情：

1. **注入 system prompt**：按当前 `AgentConfig.system_prompt_template` 填入 `repo_type`、`repo_url`、`repo_name`、`language_name`。
2. **注入 TaskTool executor**：如果当前 agent 有 `task` 工具，loop 会绑定子 agent 调度器；子 agent 当前主要是 `explore`。
3. **构建工具 schema**：将工具转换为 OpenAI function calling 兼容 schema；无工具时进入文本模式。
4. **调用模型**：`UnifiedProvider.stream_chat()` 负责适配不同 provider 的流式输出和工具调用格式。
5. **转发流式文本和工具开始事件**：`TextDelta` 和 `ToolCallStart` 立即发送给调用方。
6. **无工具调用即结束**：如果本轮只有文本，输出 `FinishEvent(finish_reason="stop")`。
7. **重复调用检测**：同一工具和同一参数重复达到阈值时，注入提醒消息，要求模型换策略或基于已有信息回答。
8. **并发执行工具**：同一轮里的多个工具调用通过 `asyncio.gather()` 并发执行。
9. **回灌工具结果**：工具输出变成 `role="tool"` 消息，进入下一轮 LLM 调用。
10. **达到 step 上限时收尾**：最后一步禁用工具，并追加总结请求，让模型基于已收集信息回答。

这个 loop 的边界很明确：它不直接理解仓库，也不直接生成 wiki schema；它只负责把配置、消息、provider、工具和事件串起来。

## 6. Agent Wiki 两阶段流程

`/ws/agent-wiki` 是结构化 wiki 生成路径，位于 `api/agent/wiki_generator.py`。它不是一个单 agent 长任务，而是两个 agent loop 顺序执行：

### Phase 0：准备阶段

1. 接收 `AgentWikiRequest`。
2. `download_repo()` 克隆或更新仓库。
3. 解析 include/exclude 过滤规则。
4. 调用 `get_or_build_retriever()` 预热语义检索，保证后续 `rag_search` 可用。
5. 构造 `file_tree` 和 `readme` 提示，优先使用前端传入的 hint。

### Phase 1：wiki-planner

1. 使用 `wiki-planner` 配置，工具集为只读工具。
2. 用户 prompt 包含 file tree hint、README 和 comprehensive/concise 指令。
3. planner 必须探索仓库并输出单个 JSON 对象。
4. 后端收集 planner 文本输出，执行 JSON 提取和轻量 schema 校验。
5. 成功后发送 `wiki_structure_ready`；失败则发送 `wiki_structure_error` 并终止连接。

### Phase 2：wiki-writer

1. 后端按 section 顺序展开 pages。
2. 每一页单独运行一次 `wiki-writer` loop。
3. writer 必须先 `rag_search`，再验证 planner 给出的 filePaths hint，然后读文件、搜索调用链，最后写 Markdown。
4. 每页成功发送 `wiki_page_done`；单页失败发送 `wiki_page_error`，继续下一页。
5. 所有页结束后发送 `finish`。

```mermaid
flowchart TD
    A[/ws/agent-wiki request] --> B[clone repo and parse filters]
    B --> C[warm CodeRetriever for rag_search]
    C --> D[build file tree and README hints]
    D --> E[wiki-planner loop]
    E --> F{valid wiki JSON?}
    F -- no --> G[wiki_structure_error]
    F -- yes --> H[wiki_structure_ready]
    H --> I[flatten pages in section order]
    I --> J[wiki-writer loop for page N]
    J --> K[wiki_page_done or wiki_page_error]
    K --> L{more pages?}
    L -- yes --> J
    L -- no --> M[finish]
```

## 7. 工具系统

工具注册表在 `api/tools/__init__.py`，当前工具包括：

| 工具 | 作用 |
|---|---|
| `rag_search` | 基于已有 FAISS / pickle cache 做语义检索，返回 ranked code chunks 和文件行号 |
| `grep` | 精确字符串或正则搜索 |
| `glob` | 按文件模式发现路径 |
| `ls` | 查看目录结构 |
| `read` | 读取指定文件 |
| `bash` | 在仓库目录内执行 shell 命令 |
| `task` | 调度子 agent，当前可用子 agent 主要是 `explore` |
| `todowrite` | 供复杂 agent 任务记录步骤 |

`FilteredToolWrapper` 会把用户的 include/exclude 规则应用到工具层：

- `read`、`ls`、`grep`：执行前检查路径。
- `glob`：执行后过滤路径列表。
- `rag_search`：执行后过滤 markdown chunk。
- `bash`：当前只校验 workdir 在 repo 内，命令字符串不做白名单过滤。

## 8. RAG 工具模式

`rag_search` 的实现分成两层：

1. `api/retriever.py` 的 `CodeRetriever`：只负责 embedder、数据库准备、embedding 校验、FAISS 检索。
2. `api/tools/rag.py` 的 `RagTool`：把 `CodeRetriever.retrieve()` 结果格式化成 agent 可读的 markdown chunk。

关键边界：

- `rag_search` 复用 `DatabaseManager.prepare_database()` 和 `~/.adalflow/databases/*.pkl`，避免重新实现索引流程。
- 进程内还有 LRU cache，最多保留 4 个 `CodeRetriever`，cache key 包含 repo 输入和 embedder 类型。
- `top_k` 限制在 1 到 20，默认 10。
- 它返回的是检索片段，不是最终答案；agent prompt 要求用 `read` 或 `grep` 继续验证上下文。
- 工具描述明确提示：索引来自 pickle cache，可能不包含 cache 构建后的文件变更。

## 9. Provider 适配模式

`UnifiedProvider` 把多 provider 统一成 `stream_chat(messages, tools)`：

- OpenAI、Azure、Dashscope、Google 走原生 function calling 或等价工具调用能力。
- OpenRouter、Ollama、Bedrock 走工具 schema 注入 prompt，再解析 `<tool_call>{...}</tool_call>` 的 fallback 模式。
- Provider 输出统一转换成 `TextDelta`、`ToolCallStart`、`FinishEvent`、`ErrorEvent`，Agent Loop 不关心底层 SDK 差异。

因此当前 Agent 的可移植性来自 provider 适配层，而不是来自外部 agent 框架。

## 10. 事件协议

### Chat Agent 事件

Ask UI 只处理 5 类事件：

| 事件 | 说明 |
|---|---|
| `text_delta` | LLM 增量文本 |
| `tool_call_start` | LLM 请求调用工具，携带完整工具参数 |
| `tool_call_end` | 工具执行结束，携带摘要、耗时、错误状态和 metadata |
| `finish` | 本次会话结束 |
| `error` | 请求或 provider 出错 |

WebSocket 是每个事件一个 JSON message；HTTP fallback 是 NDJSON，每行一个事件。两种传输的事件形状一致。

### Wiki Agent 事件

`/ws/agent-wiki` 复用基础事件，并增加 wiki 专用事件：

- `wiki_structure_ready`
- `wiki_structure_error`
- `wiki_page_done`
- `wiki_page_error`

planner/writer 的普通事件还会带 `phase="planning"` 或 `phase="writing"`，writer 事件额外带 `page_index` 和 `page_id`。

## 11. 当前边界与风险

1. **Agent Chat 和 Legacy RAG 是并行路径**：Ask 页已经走结构化 agent-chat；部分 wiki/workshop/slides 仍可能使用旧 `/ws/chat` 文本流。
2. **`wiki-planner` / `wiki-writer` 不对聊天开放**：它们需要特定 prompt 和输出协议，通用聊天只开放 `wiki`、`explore`、`deep-research`。
3. **服务端不保存长期会话**：历史由前端带入，后端只在一次请求内维护 conversation。
4. **`bash` 仍是主要风险点**：当前只限制工作目录，不解析命令主体；风险已记录在 `handbooks/risks/RISK-002-bash-agent-sandbox-gap.md`。
5. **RAG cache 可能滞后**：`rag_search` 使用已有索引，cache 构建后的未索引改动可能不可见。
6. **Provider fallback 质量不等价**：非原生 function calling provider 依赖 prompt 注入和 XML 解析，工具调用稳定性弱于原生模式。
7. **Agent Wiki 写作是顺序执行**：每页一个 writer loop，准确性优先于并发速度。

## 12. 代码依据

| 文件 | 作用 |
|---|---|
| `api/agent/config.py` | Agent 注册表、工具集、step 上限 |
| `api/agent/loop.py` | ReAct 主循环、并发工具执行、doom loop 检测、step 上限收尾 |
| `api/agent/chat_handler.py` | `/ws/agent-chat`、`/chat/agent-stream`、`/agent/info` 的核心处理 |
| `api/agent/wiki_generator.py` | `/ws/agent-wiki` 两阶段 planner/writer 工作流 |
| `api/agent/provider.py` | 多 provider 流式和工具调用适配 |
| `api/agent/message.py` | 内部结构化消息模型和 legacy chat message 转换 |
| `api/agent/stream_events.py` | Agent 与前端之间的事件协议 |
| `api/tools/__init__.py` | 工具注册表 |
| `api/tools/rag.py` | `rag_search` 工具适配层 |
| `api/retriever.py` | `CodeRetriever` 与 retriever cache |
| `api/agent/filtered_tools.py` | 用户 filter 到工具执行的约束层 |
| `src/app/[owner]/[repo]/ask/page.tsx` | Ask 页构造 `AgentChatRequest` 并消费事件 |
| `src/utils/websocketClient.ts` | Agent Chat 和 Agent Wiki WebSocket 客户端 |
| `src/utils/agentChatStream.ts` | Agent Chat HTTP NDJSON fallback |
