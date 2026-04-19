# DeepWiki Agent CLI Demo

> 对应代码：`cli/__init__.py`（空包标记）、`cli/deepwiki_cli.py`（单文件，~490 行）
>
> 目标：不修改任何现有端点，在终端端到端验证 agent 框架的多轮问答 + 两阶段 wiki 生成能力，为后续正式集成提供可运行基准。

---

## 1. 背景与目标

后端已具备两套 agent 路径：

- **通用 Q&A**：`wiki` agent config + `run_agent_loop()`（`api/agent/loop.py`）
- **两阶段 Wiki 生成**：`wiki-planner` + `wiki-writer`（`api/agent/wiki_generator.py`，通过 `/ws/agent-wiki` WebSocket 端点暴露）

但两者都只能通过 WebSocket 访问，缺少一个本地、不依赖前端、可在终端直接跑的入口来做冒烟测试。CLI demo 的目标是：

1. 绕过 WebSocket，直接调用 `run_agent_loop()` 异步生成器
2. 在同一个 REPL 里支持多轮 Q&A 和 `/wiki` 生成命令
3. 和 WebSocket 路径复用完全相同的 prompt 格式、JSON 解析、页面排序逻辑（通过 import 而非复制粘贴）

---

## 2. 文件结构

```
cli/
├── __init__.py          # 空包标记（支持 python -m cli.deepwiki_cli 调用）
└── deepwiki_cli.py      # 单文件，~490 行，分 6 个内部 section
```

`deepwiki_cli.py` 内部结构：

| Section | 职责 |
|---------|------|
| Section 1: Bootstrap | argparse、provider/model 解析 |
| Section 2: Repo metadata | `compute_repo_meta()`、`slugify()` |
| Section 3: Event printing | `print_tool_call()`、`print_tool_end()`、`print_text_delta()` |
| Section 4: Prompt vars | `qa_prompt_vars()`、`planner_prompt_vars()` |
| Section 5: Async handlers | `run_qa_turn()`、`run_planner()`、`run_writer_for_page()`、`run_wiki_command()` |
| Section 6: REPL + main | `repl()`、`main()` |

段间依赖单向，无循环。每个 section 只有纯函数或简单 async 函数，无类继承、无装饰器框架。

---

## 3. 快速开始

### 前置要求

```bash
# 安装 api 依赖（已有可跳过）
uv sync --directory api

# 激活 venv（每次在新 shell 里都要执行）
source api/.venv/bin/activate
```

### 启动 CLI

```bash
# 最简：google + gemini-2.5-flash + en + concise wiki
python -m cli.deepwiki_cli https://github.com/tiangolo/fastapi

# OpenAI + 中文 + comprehensive (8-12 页)
python -m cli.deepwiki_cli https://github.com/tiangolo/fastapi \
    --provider openai --model gpt-4o-mini --language zh --comprehensive

# 私有仓库（需 PAT）
python -m cli.deepwiki_cli https://github.com/me/private-repo --token ghp_xxx

# 本地仓库（相对路径或绝对路径均可）
python -m cli.deepwiki_cli . --type local
python -m cli.deepwiki_cli ~/code/myrepo --type local
```

### 启动时输出示例

```
Provider : openai/gpt-4o-mini
Repo     : ~/.adalflow/repos/tiangolo_fastapi
Language : Chinese
Wiki mode: concise (4-6 pages)

Building FAISS index (first run may take a while)...

DeepWiki CLI ready. Repo: https://github.com/tiangolo/fastapi
Commands:  /wiki   /clear   /exit   (Ctrl+C also exits)

>
```

### REPL 命令

| 输入 | 行为 |
|------|------|
| 任意文本 | 运行一轮 Q&A，流式输出，历史自动积累 |
| `/wiki` | 触发两阶段 wiki 生成（见 §5） |
| `/clear` | 清空 Q&A 对话历史（不重启进程） |
| `/exit` 或 `/quit` | 退出 REPL |
| Ctrl+C | 任意时刻退出（`asyncio.to_thread(input)` 在线程中，不阻塞 event loop） |

---

## 4. CLI 参数参考

```
usage: deepwiki-cli [-h] [--type {github,gitlab,bitbucket,local}]
                    [--token PAT] [--provider PROVIDER] [--model MODEL]
                    [--language LANG] [--comprehensive]
                    repo_url
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `repo_url` | 位置参数 | — | GitHub/GitLab/Bitbucket URL 或本地路径 |
| `--type` | 选项 | `github` | `github` / `gitlab` / `bitbucket` / `local` |
| `--token` | 选项 | `None` | 私有仓库 PAT |
| `--provider` | 选项 | generator.json 默认 | 覆盖 LLM provider |
| `--model` | 选项 | provider 默认模型 | 覆盖模型名 |
| `--language` | 选项 | `en` | `ar/de/en/es/fr/ja/ko/pt/ru/zh` |
| `--comprehensive` | flag | `False`（concise） | 加上后生成 8-12 页 + sections 树 |

**Provider 解析顺序**（`resolve_provider_model()`）：
1. `--provider` CLI 参数
2. `api/config/generator.json` 中的 `default_provider`（通常 `"google"`）
3. 不在 `configs["providers"]` 中 → `sys.exit()`

**Model 解析顺序**：
1. `--model` CLI 参数
2. `configs["providers"][provider]["default_model"]`
3. 为空 → `sys.exit()`

---

## 5. 启动阶段逻辑

`main()` 同步执行四步，完成后进入 async REPL：

```
parse_args()
    └─ resolve_provider_model()   → (provider_name, model)
    └─ compute_repo_meta()        → (repo_name, repo_path)

print 启动摘要

DatabaseManager().prepare_retriever(repo_url, repo_type, token)
    └─ 幂等：若 FAISS 已存在则跳过重新嵌入
    └─ 副作用：git clone → 读文件 → 嵌入 → 持久化 FAISS
    └─ 目的：为未来 rag_search 工具铺路（当前 agent 只用文件系统工具）

UnifiedProvider(provider_name, model)

asyncio.run(repl(...))
```

**本地 repo 的路径计算**（`compute_repo_meta()`）：

```python
if repo_type == "local":
    repo_path = os.path.abspath(os.path.expanduser(repo_url))
    repo_name = os.path.basename(repo_path.rstrip(os.sep)) or "local-repo"
```

Remote 模式（github/gitlab/bitbucket）：
- `repo_name = f"{owner}_{repo}"` → `repo_path = ~/.adalflow/repos/{repo_name}`
- 与 `wiki_generator._compute_repo_path()` 逻辑一致

Local 模式**有意不 mirror** `wiki_generator._compute_repo_path()`：
因为 `DatabaseManager._create_repo()` 对 local 仓库不拷贝到 `~/.adalflow/repos/`，
若用相同逻辑拼路径，agent 工具会指向不存在的目录（Codex #1 已修复）。

---

## 6. 多轮 Q&A 机制

### 关键契约：`run_agent_loop` 不回填 messages

`api/agent/loop.py` 的 `run_agent_loop()` 在内部维护一份独立的 `conversation` 列表（`[system_msg, *messages]`），**不修改调用方传入的 `messages`**。因此 CLI 必须在每轮结束后手动 append 双方消息：

```python
async def run_qa_turn(query, history, ...):
    history.append(AgentMessage.user(query))   # ← 必须手动 append
    chunks: list[str] = []
    async for evt in run_agent_loop(agent_config, history, ...):
        if isinstance(evt, TextDelta):
            chunks.append(evt.content)
            print_text_delta(evt.content)
        ...
    history.append(AgentMessage.assistant_text("".join(chunks)))  # ← 必须手动 append
```

**优势**：`/clear` 只需 `history.clear()`，历史管理完全在调用方。

### Q&A REPL 状态

`repl()` 内只有 3 个不变量（REPL 期间不变）：
- `qa_cfg = get_agent_config("wiki")`
- `qa_tools = get_tools_for_agent(qa_cfg, repo_path)`
- `qa_vars = qa_prompt_vars(...)` — 4 键 dict: `repo_type / repo_url / repo_name / language_name`

和 1 个可变量：
- `history: list[AgentMessage]`（每轮积累，`/clear` 清零）

### `wiki` agent 的工具集

`wiki` agent 持有全部 7 种工具：`bash / grep / glob / ls / read / task / todowrite`。

`run_agent_loop()` 自动处理 `task` tool 的 executor 绑定，CLI 无需额外操作。

---

## 7. 两阶段 Wiki 生成（`/wiki` 命令）

### 整体流程

```
/wiki
  │
  ▼
run_wiki_command()
  │
  ├─ output_dir = ~/.adalflow/wiki-output/{repo_name}/
  │
  ├─ run_planner()                         ← Phase 1
  │   ├─ build_file_tree(repo_path)
  │   ├─ read_repo_readme(repo_path)
  │   ├─ _format_planner_user_prompt(...)
  │   ├─ run_agent_loop("wiki-planner", ...)
  │   │   └─ TextDelta 静默收集（JSON，不流到 stdout）
  │   │   └─ ToolCallStart/End 打印工具调用
  │   ├─ parse_wiki_structure(raw)         ← 4 层解析（fence→{}块→json.loads→schema校验）
  │   └─ 打印页面目录
  │
  ├─ 清理 output_dir/*.md                  ← planner 成功后清理旧产物
  │   （planner 失败则跳过，保留上次好的输出）
  │
  └─ for page in _flatten_pages_in_section_order(structure):
        run_writer_for_page()              ← Phase 2（顺序，每页独立 agent loop）
          ├─ _format_writer_user_prompt(page, language)
          ├─ run_agent_loop("wiki-writer", ...)
          │   └─ TextDelta 流到 stdout
          │   └─ ToolCallStart/End 缩进打印（"  "）
          └─ 写入 {i:02d}-{slug}.md
```

### 与 WebSocket 路径的关键区别

| 维度 | WebSocket 路径（`/ws/agent-wiki`） | CLI 路径（`/wiki`） |
|------|-----------------------------------|---------------------|
| 超时 | `asyncio.wait_for(..., timeout=300)` | 无超时（demo 用） |
| 过滤 | `wrap_tools_with_filters(ParsedFilters)` | 无（demo 用） |
| 事件封装 | `_send_tagged_event(ws, evt, phase=..., page_index=...)` | `print_*` 函数 |
| 失败处理 | 发送 `WikiStructureError` / `WikiPageError` 事件 | 打印到 stderr，REPL 不退出 |
| 并发 | 顺序（与 WS 路径一致） | 顺序 |

### 复用的 wiki_generator 符号

CLI 直接 import（不复制）以下符号，保证与 WebSocket 路径的行为完全对齐：

| 符号 | 说明 |
|------|------|
| `parse_wiki_structure(raw)` | 4 层 JSON 解析，返回 dict 或 None |
| `_flatten_pages_in_section_order(structure)` | 按 sections 树视觉顺序展开页面 |
| `_format_planner_user_prompt(file_tree, readme, comprehensive, language)` | planner 用户消息模板 |
| `_format_writer_user_prompt(page, language)` | writer 用户消息模板 |
| `_LANGUAGE_NAMES` | 语言代码 → 显示名称映射 |
| `_COMPREHENSIVE_INSTRUCTION` / `_CONCISE_INSTRUCTION` | planner 系统 prompt 的 `{comprehensive_instruction}` 值 |
| `_ADALFLOW_ROOT` | `~/.adalflow` 根目录（一致的存储位置） |

不复用（WebSocket 绑定，无法直接调用）：`_run_planner_phase()`、`_run_writer_phase()`，CLI 自己用 stdout print 替代。

### 页面文件命名

每页保存为：`~/.adalflow/wiki-output/{repo_name}/{i:02d}-{slug}.md`

- `i` 从 1 开始，左填 0（如 `01`、`02`）
- `slug` = 标题小写后将非字母数字字符替换为 `-`，截取 60 字符

示例：第 3 页标题 "API Routing" → `03-api-routing.md`

### Stale 页面清理策略

```python
# run_wiki_command 中，planner 成功后：
for stale in output_dir.glob("*.md"):
    stale.unlink()
```

只清 `*.md`，不递归，保留目录中的其他文件。

**选择在 planner 成功之后清理**（而非 planner 之前）的原因：若 planner 失败，用户可以重试 `/wiki`，此时上一次成功生成的页面仍然可用，不会丢失。

---

## 8. 事件处理

三个 print 函数组成完整的事件显示层：

```python
# TextDelta: 实时流式，不换行
print_text_delta(content: str)
# → print(content, end="", flush=True)

# ToolCallStart: 工具名 + 关键参数（每个参数截取 80 字符）
print_tool_call(name: str, args_dict: dict, indent: str = "")
# → \n[{name}({k=v, ...})]

# ToolCallEnd: 状态 + 耗时
print_tool_end(end: ToolCallEnd, indent: str = "")
# →   └─ OK (124ms) 或   └─ ERR (45ms)
```

Q&A 轮次和 wiki Phase 2 (writer) 都会流式输出 TextDelta；Phase 1 (planner) 的 TextDelta 静默收集（因为输出是 JSON，流到 stdout 无意义）。

writer 的 print_tool_call/end 用 `indent="  "` 缩进，视觉上区分于 planner 的工具调用。

### 处理的事件类型

| 事件 | Q&A | Planner | Writer |
|------|-----|---------|--------|
| `TextDelta` | 流到 stdout | 静默收集 | 流到 stdout |
| `ToolCallStart` | print（无缩进） | print（无缩进） | print（2 空格缩进） |
| `ToolCallEnd` | print | print | print（2 空格缩进） |
| `FinishEvent` | 换行 | 不处理 | 不处理 |
| `ErrorEvent` | print → stderr | print → stderr | print → stderr |

---

## 9. 重要设计决策

### 为什么不修改 `api/` 任何代码

CLI 绕开 WebSocket 直接调用 `run_agent_loop()`，证明了这个接口对"非 WebSocket 使用者"是可用的，同时不破坏任何现有端点。若后续要把 CLI 升级为 production 工具，只需在 CLI 层加参数，不需要改后端。

### 为什么 FAISS 索引在启动时一次性建好

```python
DatabaseManager().prepare_retriever(args.repo_url, args.repo_type, args.token)
```

当前 agent 工具（grep/ls/read/bash/glob）不需要 FAISS。建索引纯粹是"为未来 `rag_search` 工具铺路"——第一次跑会慢，后续重启命中缓存（准确来说是 FAISS 已持久化 → `prepare_retriever` 检测到有效的 pickle 文件后跳过重新嵌入）。

### 为什么用 `asyncio.to_thread(input)` 而不是 `input()`

REPL 运行在 `asyncio.run()` 的 event loop 里。`input()` 是阻塞调用，直接调用会阻塞 event loop，导致 `async for evt in run_agent_loop(...)` 无法并发处理其他 task。`asyncio.to_thread(input, ...)` 在线程池中运行，event loop 保持非阻塞。

### 为什么 /wiki 不支持在 REPL 内传参数（如 `/wiki zh`）

启动 flag（`--language`、`--comprehensive`）已经覆盖配置场景，整个 REPL 会话使用固定配置。REPL 内解析 `/wiki <args>` 的字符串分词逻辑对 demo 是过度设计，且与"会话配置不变"的直觉相悖。

---

## 10. 已知限制

### BashTool 无命令白名单

`wiki-writer` 持有 `bash` 工具。与 WebSocket 路径相同的已知缺口——见 `handbooks/bash-agent-sandbox-gap.md`。

### 无超时

CLI 未实现每轮 `asyncio.wait_for()` 超时。对于响应缓慢的 LLM 或大型仓库，`/wiki` 可能运行数十分钟。Ctrl+C 是唯一中止方式。

### 过滤规则未集成

`ParsedFilters` / `wrap_tools_with_filters()` 未应用到 CLI。如需过滤敏感目录，WebSocket 路径（`/ws/agent-wiki`）提供了完整的 `excluded_dirs`/`excluded_files` 支持。

### REPL 历史不持久化

进程退出后历史丢失。这是有意的 non-goal——单进程单仓库，`/clear` 加 Ctrl+C 重启即可。

---

## 11. 关键文件索引

| 文件 | 用途 |
|------|------|
| `cli/deepwiki_cli.py` | 单文件 CLI 入口（~490 行） |
| `api/agent/loop.py` | `run_agent_loop()` 异步生成器 |
| `api/agent/config.py` | `wiki` / `wiki-planner` / `wiki-writer` agent 配置 |
| `api/agent/wiki_generator.py` | CLI 复用的 prompt 格式化函数 + 解析函数 |
| `api/agent/message.py` | `AgentMessage.user()` / `.assistant_text()` 工厂方法 |
| `api/agent/provider.py` | `UnifiedProvider` |
| `api/data_pipeline.py` | `DatabaseManager().prepare_retriever()` |
| `api/utils/repo_tree.py` | `build_file_tree()` / `read_repo_readme()` |
| `api/config.py` | `configs["providers"]` — provider/model 默认值来源 |
