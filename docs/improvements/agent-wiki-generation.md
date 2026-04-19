# Agent Wiki 生成后端（路径 B）

> 对应代码变更：`git diff HEAD` 中尚未 commit 的全部文件。
> 核心实现集中在 `api/agent/wiki_generator.py`、`api/utils/filters.py`、`api/agent/filtered_tools.py`。

---

## 1. 背景与动机

### 旧路径的问题

旧版 wiki 生成是"前端编排 + 后端通用 RAG 代理"模式：

```
前端 page.tsx
  ├── 调 GitHub/GitLab API 拿 fileTree + README（远程拉取）
  ├── 拼 XML prompt → /ws/chat → DOMParser 解析 <wiki_structure>（Phase 1）
  └── 循环每页：拼 prompt → /ws/chat（Phase 2）
```

三个核心问题：
1. **文件路径幻觉**：planner LLM 只看到远程 API 返回的 fileTree 字符串，无工具验证能力，`<relevant_files>` 字段经常幻觉不存在的路径。
2. **Writer 无法主动探索**：每页内容仅依赖 RAG 检索碎片，引用 `Sources:` 块行号经常错误。
3. **前端承担后端职责**：wiki schema、prompt 模板、并发队列全在前端 TypeScript 里。

### 新路径（路径 B）的目标

新增 `/ws/agent-wiki` 端点，用两个独立的 agent loop 驱动：

- **wiki-planner**：通过 `glob`/`ls`/`read` 工具真正探索本地 clone，输出经过工具验证的 JSON wiki 结构。
- **wiki-writer**：每页独立运行 agent loop，用 `grep`/`read`/`bash git log` 查证相关代码后再撰写，"先探索后撰写"。

---

## 2. 整体架构

```
客户端 WebSocket → /ws/agent-wiki
                        │
                        ▼
              handle_agent_wiki_websocket()
                        │
             ┌──────────┴──────────────┐
             │ Step 1: download_repo() │  ← 幂等 git clone
             │ Step 2: build filters   │  ← ParsedFilters
             │ Step 3: build file_tree │  ← 过滤后的文件树 hint
             └──────────┬──────────────┘
                        │
              ┌─────────▼──────────────────┐
              │  Phase 1: wiki-planner     │  max_steps=20, 只读工具
              │  run_agent_loop(...)        │  输出: WikiStructureReady
              │  parse_wiki_structure()     │
              └─────────┬──────────────────┘
                        │ 结构 JSON
              ┌─────────▼──────────────────────────────────┐
              │  Phase 2: wiki-writer (per page, 顺序)      │  max_steps=25
              │  for page in flatten_pages_in_section_order│  带过滤的工具
              │      run_agent_loop(...)                    │
              │      → WikiPageDone (完整 markdown)        │
              └─────────────────────────────────────────────┘
                        │
              FinishEvent(finish_reason="stop")
```

---

## 3. WebSocket 端点调用方式

### 连接

```
ws://localhost:8001/ws/agent-wiki
```

### 请求格式（JSON，发送一次）

```json
{
  "repo_url": "https://github.com/owner/repo",
  "type": "github",
  "token": "ghp_xxx",
  "provider": "openai",
  "model": "gpt-4o",
  "language": "zh",
  "comprehensive": true,
  "file_tree_hint": "src/...\napi/...",
  "readme_hint": "# My Project\n...",
  "excluded_dirs": "node_modules\n.git\nsecrets",
  "excluded_files": "package-lock.json",
  "included_dirs": null,
  "included_files": null
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `repo_url` | `str` | ✅ | 仓库 URL（支持 github/gitlab/bitbucket/local） |
| `type` | `str` | 否 | `"github"` / `"gitlab"` / `"bitbucket"` / `"local"`，默认 `"github"` |
| `token` | `str` | 否 | 私有仓库 PAT |
| `provider` | `str` | 否 | LLM provider，默认 `"google"` |
| `model` | `str` | 否 | 模型名称，默认 provider 的默认模型 |
| `language` | `str` | 否 | 输出语言代码（`en`/`zh`/`ja`/`ko`/`es`/`fr`/`de`/`pt`/`ru`/`ar`），默认 `"en"` |
| `comprehensive` | `bool` | 否 | `true` = 8-12 页 + sections 树；`false` = 4-6 页扁平，默认 `true` |
| `file_tree_hint` | `str` | 否 | 前端已 fetch 的 fileTree 字符串，传入可避免后端重复 walk |
| `readme_hint` | `str` | 否 | 前端已 fetch 的 README 内容 |
| `excluded_dirs` | `str` | 否 | `\n` 分隔的排除目录列表（如 `"node_modules\n.git"`） |
| `excluded_files` | `str` | 否 | `\n` 分隔的排除文件名列表（精确匹配） |
| `included_dirs` | `str` | 否 | `\n` 分隔的包含目录列表（非空时切换为 inclusion mode，忽略 excluded_*） |
| `included_files` | `str` | 否 | `\n` 分隔的包含文件名/后缀列表 |

### wscat 快速测试

```bash
npx wscat -c ws://localhost:8001/ws/agent-wiki
# 连接后粘贴：
{"repo_url":"https://github.com/tiangolo/fastapi","provider":"openai","model":"gpt-4o-mini","language":"en","comprehensive":false}
```

---

## 4. WebSocket 事件协议（服务端 → 客户端）

所有事件均为 JSON 对象，用 `type` 字段区分。

### Phase 1（规划阶段）

规划阶段的事件附带 `"phase": "planning"` 信封字段：

| type | 说明 |
|------|------|
| `text_delta` | planner LLM 的文字输出增量 |
| `tool_call_start` | planner 调用工具（glob/ls/read/grep） |
| `tool_call_end` | 工具执行结束 |
| `wiki_structure_ready` | **Phase 1 成功**，包含完整 wiki 结构 JSON |
| `wiki_structure_error` | **Phase 1 失败**，连接随后关闭 |

```json
// wiki_structure_ready 示例
{
  "type": "wiki_structure_ready",
  "structure": {
    "id": "wiki-root",
    "title": "FastAPI Documentation",
    "description": "...",
    "pages": [
      {
        "id": "page-1",
        "title": "Architecture Overview",
        "content": "",
        "filePaths": ["fastapi/applications.py", "fastapi/routing.py"],
        "importance": "high",
        "relatedPages": ["page-2"]
      }
    ],
    "sections": [{"id": "s-1", "title": "Core", "pages": ["page-1"], "subsections": []}],
    "rootSections": ["s-1"]
  }
}

// wiki_structure_error 示例
{
  "type": "wiki_structure_error",
  "code": "json_parse_error",
  "message": "Planner output could not be parsed as a valid wiki structure"
}
```

**`wiki_structure_error` 的 code 枚举**：

| code | 含义 |
|------|------|
| `json_parse_error` | LLM 输出无法解析为 JSON |
| `schema_validation_error` | JSON 结构不符合 wiki schema |
| `planner_timeout` | 超过 5 分钟上限 |
| `clone_failed` | git clone / download_repo 失败 |
| `internal_error` | 服务端未预期异常 |

### Phase 2（写作阶段）

写作阶段的事件附带 `"phase": "writing"`、`"page_index": N`、`"page_id": "page-N"` 信封字段：

| type | 说明 |
|------|------|
| `text_delta` | writer LLM 的 markdown 输出增量 |
| `tool_call_start` | writer 调用工具（grep/glob/ls/read/bash） |
| `tool_call_end` | 工具执行结束 |
| `wiki_page_done` | **单页完成**，包含完整 markdown 内容 |
| `wiki_page_error` | **单页失败**（非致命），继续下一页 |

```json
// wiki_page_done 示例
{
  "type": "wiki_page_done",
  "page_id": "page-1",
  "page_title": "Architecture Overview",
  "page_index": 0,
  "total_pages": 6,
  "content": "# Architecture Overview\n\n<details>..."
}

// wiki_page_error 示例
{
  "type": "wiki_page_error",
  "page_id": "page-3",
  "page_index": 2,
  "code": "writer_timeout",
  "message": "Writer exceeded 300s for page 'API Routing'"
}
```

### 收尾事件

```json
{"type": "finish", "finish_reason": "stop"}
```

### 完整事件序列示意

```
wiki_structure_ready
↓
text_delta (phase=writing, page_index=0) × N
tool_call_start (phase=writing, page_index=0)
tool_call_end   (phase=writing, page_index=0)
...
wiki_page_done  (page_index=0, total_pages=6)
↓
text_delta (phase=writing, page_index=1) × N
...
wiki_page_done  (page_index=1, total_pages=6)
↓
...（重复直到所有页面完成）
↓
finish
```

---

## 5. 两个 Agent 的配置

### wiki-planner（`api/agent/config.py`）

| 属性 | 值 |
|------|-----|
| 工具 | `grep`, `glob`, `ls`, `read`（只读，无 bash） |
| max_steps | 20 |
| 超时 | 300s |
| 系统 prompt | `WIKI_PLANNER_SYSTEM_PROMPT`（`api/prompts.py`） |
| 模式 | primary |

**行为**：先用 `glob`/`ls` 扫描目录结构，用 `read` 确认关键文件存在，再根据 `comprehensive` flag 输出严格 JSON（含 sections 树或扁平列表）。JSON 输出中 `content` 字段始终为 `""`（内容由 Phase 2 填充）。

### wiki-writer（`api/agent/config.py`）

| 属性 | 值 |
|------|-----|
| 工具 | `grep`, `glob`, `ls`, `read`, `bash`（只读 + bash） |
| max_steps | 25 |
| 超时 | 300s（per page） |
| 系统 prompt | `WIKI_WRITER_SYSTEM_PROMPT`（`api/prompts.py`） |
| 模式 | primary |

**行为**：接收 planner 给出的页面标题、描述、文件路径 hint（标记为"hint 非事实，需工具验证"），按"先探索后撰写"工作流输出完整 markdown 页面（包含 `<details>` 引用块、mermaid 图、`Sources: [file:line]()` 引用）。

---

## 6. 过滤系统

### 设计动机

用户在 UI 中设置的 `excluded_dirs`/`excluded_files` 字段，旧版在 RAG 路径会传给 `prepare_retriever()` 影响向量检索范围。新版 agent 路径必须同样尊重这些过滤规则，否则 agent 工具可以直接访问用户明确排除的目录（如 `secrets/`、`.env.backup`）。

### 核心类：`ParsedFilters`（`api/utils/filters.py`）

```python
from api.utils.filters import ParsedFilters, should_exclude_path

# 从请求字段解析
filters = ParsedFilters.from_strings(
    excluded_dirs="node_modules\nsecrets",
    excluded_files="package-lock.json",
)

# 判断一个相对路径是否应被排除
should_exclude_path("secrets/api_key.txt", filters)   # → True
should_exclude_path("src/main.py", filters)            # → False
```

**两种模式**（与 `data_pipeline.py:should_process_file()` 语义对齐）：

| 模式 | 触发条件 | 语义 |
|------|----------|------|
| Exclusion mode（默认） | `included_dirs` 和 `included_files` 均为空 | 排除匹配任一 `excluded_*` 的路径 |
| Inclusion mode | `included_dirs` 或 `included_files` 任一非空 | 只保留匹配 `included_*` 的路径，`excluded_*` 被完全忽略 |

**匹配规则**（继承 legacy 语义）：
- `excluded_dirs`：路径任一**段**等于过滤值（如 `secrets` 匹配 `api/secrets/key.py`）
- `excluded_files`：文件名**精确匹配**（非 glob，继承 legacy 行为）
- `included_files`：文件名相等或文件名**以 pattern 结尾**（如 `.py` 匹配所有 Python 文件）

### 工具层强制执行：`FilteredToolWrapper`（`api/agent/filtered_tools.py`）

Wrapper 模式，不修改现有 Tool 类，在 wiki_generator 调用层包裹：

```python
from api.agent.filtered_tools import wrap_tools_with_filters

tools = get_tools_for_agent(config, repo_path)
tools = wrap_tools_with_filters(tools, filters, repo_path)
# 若 filters.is_empty，直接返回原 tools 字典（零开销）
```

| 工具 | 过滤策略 |
|------|----------|
| `read` | **前置路径校验**：`file_path` 被排除时，直接返回 "Blocked" 消息，不读文件 |
| `ls` | **前置目录校验**：请求 path 被排除时，直接返回 "Blocked" 消息 |
| `grep` | **前置搜索根校验**：搜索目录被排除时返回 "Blocked" |
| `glob` | **后置输出过滤**：结果列表（一行一路径）中删除被排除的路径 |
| `bash` | **Pass-through（已知缺口）** — 见 `handbooks/bash-agent-sandbox-gap.md` |

### 文件树 hint 过滤

`build_file_tree()` 也接受 `ParsedFilters`，确保 planner 的初始文件树 hint 不包含被排除的路径：

```python
file_tree = build_file_tree(repo_path, filters=filters)
```

---

## 7. JSON 结构解析

Planner 输出的 JSON 经过 `parse_wiki_structure()` 的四层解析（`api/agent/wiki_generator.py:266`）：

```
Layer 1: 剥除 ```json ... ``` markdown fence
Layer 2: 找最外层 { ... } 块（忽略前后解释性文字）
Layer 3: json.loads
Layer 4: 轻量 schema 校验（检查必填字段，自动补全可选字段默认值）
```

任一层失败返回 `None`，handler 发送 `WikiStructureError` 后关闭 WebSocket。

---

## 8. 页面顺序展开

`_flatten_pages_in_section_order()` 按 sections 树的视觉顺序展开页面，确保 writer 按照用户预期的目录顺序逐页生成：

```
rootSections: ["s-1", "s-2"]
s-1.pages: ["p1", "p2"]
s-1.subsections: ["s-1-1"]
s-1-1.pages: ["p3"]
s-2.pages: ["p4"]
孤页（未归入任何 section）: ["p5"]

→ 最终顺序: [p1, p2, p3, p4, p5]
```

---

## 9. 关键文件索引

| 文件 | 用途 |
|------|------|
| `api/agent/wiki_generator.py` | WebSocket handler 主文件（580 行） |
| `api/agent/filtered_tools.py` | FilteredToolWrapper + wrap_tools_with_filters |
| `api/utils/filters.py` | ParsedFilters + should_exclude_path |
| `api/utils/repo_tree.py` | build_file_tree() + read_repo_readme() |
| `api/agent/config.py` | wiki-planner / wiki-writer 注册（文件末尾） |
| `api/agent/stream_events.py` | WikiStructureReady / WikiPageDone / WikiStructureError / WikiPageError |
| `api/agent/__init__.py` | 公开导出新事件 |
| `api/prompts.py` | WIKI_PLANNER_SYSTEM_PROMPT / WIKI_WRITER_SYSTEM_PROMPT |
| `api/api.py` | 路由注册（`/ws/agent-wiki`） |
| `tests/unit/test_wiki_generator.py` | wiki_generator 单元测试（31 个） |
| `tests/unit/test_filters.py` | 过滤系统单元测试（38 个） |

---

## 10. 已知限制与安全说明

### BashTool 无命令白名单

`wiki-writer` 持有 `bash` 工具，但 BashTool 目前只约束 CWD（不过滤命令字符串本身）。当 agent 处理包含 prompt injection 的仓库内容时，理论上可以执行网络请求或读取环境变量。详细风险分析和缓解路线图见：

```
handbooks/bash-agent-sandbox-gap.md
```

### 缓存键冲突（先存在问题）

`~/.adalflow/repos/{owner}_{repo}/` 的命名不包含 host 信息，不同 Git 托管平台上同名仓库会共用同一本地目录。详细分析和建议重新设计方案见：

```
handbooks/repo-cache-key-collision-risk.md
```

### 过滤规则的 bash 盲区

`FilteredToolWrapper` 对 `bash` 是 pass-through：用户设置的 `excluded_dirs`/`excluded_files` 不阻止 writer 通过 bash 命令访问被排除的路径。在 BashTool 加入命令白名单前，这是一个已知但可接受的权衡（Phase 1 见上方 handbook）。

---

## 11. 旧路径向后兼容

`/ws/chat` 路由完全不受影响，前端旧版 wiki 生成流程正常工作。路径 B 是通过新增独立路由和新增代码实现的，**不修改任何已有文件的业务逻辑**（`websocket_wiki.py`、`rag.py`、`data_pipeline.py`、各 provider client、所有现有 Tool 类均未改动）。

前端适配（子任务 13）需要新增 `createWikiAgentWebSocket()` 连接 `/ws/agent-wiki` 并消费上述事件流，但这不在本次实现范围内。
