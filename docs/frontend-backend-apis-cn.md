# 前端 ↔ 后端 API 参考

本文档列出 DeepWiki-Open 中 Next.js 前端与 FastAPI 后端之间的 API 边界。内容涵盖浏览器可见路由、它们对应的 Python 后端路由、请求与响应结构、调用方，以及运行层面的注意事项。

## 1. 通信模型

DeepWiki-Open 使用三种通信路径：

```text
浏览器客户端
  │
  ├─ 同源 REST 调用
  │    /api/wiki_cache, /export/wiki, /api/lang/config, ...
  │
  ├─ Next.js route-handler 代理
  │    /api/chat/stream, /api/models/config, /api/wiki/projects, ...
  │
  └─ 直接 WebSocket 调用
       ws://<backend>/ws/chat
```

前端是位于 `src/` 中的 Next.js 应用。后端是位于 `api/api.py` 中的 FastAPI 应用。

### 后端基础 URL

| 设置项 | 使用方 | 默认值 | 说明 |
|---|---|---:|---|
| `SERVER_BASE_URL` | `next.config.ts`、大多数 Next 路由代理、若干客户端组件 | `http://localhost:8001` | 主后端 URL。 |
| `PYTHON_BACKEND_HOST` | 仅 `src/app/api/wiki/projects/route.ts` | `http://localhost:8001` | 为 processed-project 代理单独设置的环境变量。 |

### 代理策略

前端大多调用诸如 `/api/wiki_cache` 这样的同源 URL。Next.js 要么将这些 URL 重写到后端，要么通过 route handler 使用 `fetch()` 调用后端。

源文件：

- `next.config.ts` 定义了缓存、导出、认证、语言配置以及本地仓库结构的 rewrites。
- `src/app/api/*/route.ts` 包含显式的 Next route-handler 代理。
- `api/api.py` 注册了 FastAPI 的 REST 和 WebSocket 端点。

## 2. API 清单

### 前端可见 API

| 前端 URL | 方法 | 后端目标 | 传输方式 | 主要用途 |
|---|---:|---|---|---|
| `/api/lang/config` | `GET` | `/lang/config` | Next rewrite | 加载支持的 UI/内容语言。 |
| `/api/auth/status` | `GET` | `/auth/status` | Next route proxy and rewrite | 检查生成 wiki 是否需要认证码。 |
| `/api/auth/validate` | `POST` | `/auth/validate` | Next route proxy and rewrite | 校验用户输入的认证码。 |
| `/api/models/config` | `GET` | `/models/config` | Next route proxy | 加载可用 provider 与 model。 |
| `/api/wiki/projects` | `GET` | `/api/processed_projects` | Next route proxy | 列出已缓存/已生成的项目。 |
| `/api/wiki/projects` | `DELETE` | `/api/wiki_cache` | Next route proxy | 在 processed-projects UI 中删除项目缓存。 |
| `/api/wiki_cache` | `GET` | `/api/wiki_cache` | Next rewrite | 加载缓存的 wiki 数据。 |
| `/api/wiki_cache` | `POST` | `/api/wiki_cache` | Next rewrite | 存储生成后的 wiki 数据。 |
| `/api/wiki_cache` | `DELETE` | `/api/wiki_cache` | Next rewrite | 在重新生成前清除服务端 wiki 缓存。 |
| `/export/wiki` | `POST` | `/export/wiki` | Next rewrite | 将生成的 wiki 下载为 Markdown 或 JSON。 |
| `/local_repo/structure` | `GET` | `/local_repo/structure` | Next rewrite | 读取本地仓库的文件树和 README。 |
| `ws://<backend>/ws/chat` | WebSocket | `/ws/chat` | 浏览器直连 WebSocket | 流式传输 wiki 生成、Ask 回答、slides 和 workshop 内容。 |
| `/api/chat/stream` | `POST` | `/chat/completions/stream` | Next route proxy | `/ws/chat` 的 HTTP 流式回退方案。 |

### 后端已注册但当前前端未调用的 API

| 后端 URL | 方法 | 用途 |
|---|---:|---|
| `/ws/agent-wiki` | WebSocket | 基于 Agent 的两阶段 wiki 生成器。已在 FastAPI 中注册，但目前未发现前端调用方。 |
| `/health` | `GET` | 提供给 Docker/监控使用的后端健康检查。 |
| `/` | `GET` | 动态列出已注册路由的根端点。 |

### 前端直接调用的外部 API

这些并非前端↔后端 API，但它们对完整数据流同样重要：

- GitHub REST API，用于获取仓库文件树和 README。
- GitLab REST API，用于获取项目元数据、仓库文件树和 README。
- Bitbucket REST API，用于获取仓库元数据、源码树和 README。

本地仓库场景通过后端的 `/local_repo/structure` 处理；托管仓库的元数据则由浏览器直接拉取。

## 3. 共享数据模型

### `ChatMessage`

由 `/ws/chat` 和 `/api/chat/stream` 使用。

```ts
{
  role: "user" | "assistant" | "system";
  content: string;
}
```

后端校验要求最后一条消息的 `role` 必须为 `user`。

### `ChatCompletionRequest`

由 `/ws/chat` 和 `/api/chat/stream` 使用。

```ts
{
  repo_url: string;
  messages: ChatMessage[];
  filePath?: string;
  token?: string;
  type?: "github" | "gitlab" | "bitbucket" | "local" | string;
  provider?: "google" | "openai" | "openrouter" | "ollama" | "bedrock" | "azure" | "dashscope" | string;
  model?: string | null;
  language?: string;
  excluded_dirs?: string;
  excluded_files?: string;
  included_dirs?: string;
  included_files?: string;
}
```

说明：

- `excluded_*` 和 `included_*` 都是字符串。后端会按换行符拆分它们，尽管后端模型中的注释写的是“逗号分隔”。
- 某些前端生成流程会发送 `custom_model`，但后端 `ChatCompletionRequest` 模型并未定义该字段。除非所选自定义模型被复制到 `model` 中，否则后端会忽略 `custom_model`。
- 后端在生成响应前会准备或复用仓库 embeddings。

### `RepoInfo`

用于 wiki 缓存载荷和 URL 状态重建。

```ts
{
  owner: string;
  repo: string;
  type: string;
  token: string | null;
  localPath: string | null;
  repoUrl: string | null;
}
```

### `WikiPage`

用于 wiki 缓存、导出载荷和前端渲染。

```ts
{
  id: string;
  title: string;
  content: string;
  filePaths: string[];
  importance: "high" | "medium" | "low";
  relatedPages: string[];
  parentId?: string;
  isSection?: boolean;
  children?: string[];
}
```

后端 Pydantic 模型要求提供 `id`、`title`、`content`、`filePaths`、`importance` 和 `relatedPages`。额外字段不会被后端响应模型使用。

### `WikiSection`

用于前端 wiki 树，并在存在时持久化到缓存中。

```ts
{
  id: string;
  title: string;
  pages: string[];
  subsections?: string[];
}
```

### `WikiStructure`

用于缓存载荷和前端渲染。

```ts
{
  id: string;
  title: string;
  description: string;
  pages: WikiPage[];
  sections?: WikiSection[];
  rootSections?: string[];
}
```

### `WikiCacheData`

由 `GET /api/wiki_cache` 返回，并由 `POST /api/wiki_cache` 写入。

```ts
{
  wiki_structure: WikiStructure;
  generated_pages: Record<string, WikiPage>;
  repo_url?: string | null;
  repo?: RepoInfo | null;
  provider?: string | null;
  model?: string | null;
}
```

前端在保存缓存时还会发送 `comprehensive`，但后端缓存请求模型并未定义该字段。

## 4. REST API 契约

### 4.1 `GET /api/lang/config`

前端 URL：

```http
GET /api/lang/config
```

后端目标：

```http
GET /lang/config
```

用途：

- 加载可用语言代码以及默认语言。
- 由 `LanguageContext` 使用。

响应：

```json
{
  "supported_languages": {
    "en": "English",
    "ja": "Japanese (日本語)",
    "zh": "Mandarin Chinese (中文)"
  },
  "default": "en"
}
```

实现说明：

- 后端返回 `api/config/lang.json` 中的 `configs["lang_config"]`。
- 无请求体，也无查询参数。

### 4.2 `GET /api/auth/status`

前端 URL：

```http
GET /api/auth/status
```

后端目标：

```http
GET /auth/status
```

用途：

- 在开始生成 wiki 或刷新缓存前，检查是否启用了认证码门禁。

响应：

```json
{
  "auth_required": true
}
```

实现说明：

- `auth_required` 通过后端配置从 `DEEPWIKI_AUTH_MODE` 推导而来。
- 该路径同时存在 Next route handler 和 rewrite。route handler 会转发到同一个后端端点。

### 4.3 `POST /api/auth/validate`

前端 URL：

```http
POST /api/auth/validate
Content-Type: application/json
```

后端目标：

```http
POST /auth/validate
```

请求：

```json
{
  "code": "user-entered-code"
}
```

响应：

```json
{
  "success": true
}
```

实现说明：

- 后端将 `code` 与 `DEEPWIKI_AUTH_CODE` 进行比对。
- 该路径同时存在 Next route handler 和 rewrite。route handler 会转发到同一个后端端点。
- 当认证码无效时，该端点返回 `200` 且 `success: false`，而不是 `401`。

### 4.4 `GET /api/models/config`

前端 URL：

```http
GET /api/models/config
```

后端目标：

```http
GET /models/config
```

用途：

- 为模型选择器和 Ask 页面回退逻辑加载 provider/model 选项。

响应：

```json
{
  "providers": [
    {
      "id": "google",
      "name": "Google",
      "supportsCustomModel": true,
      "models": [
        {
          "id": "gemini-2.5-flash",
          "name": "gemini-2.5-flash"
        }
      ]
    }
  ],
  "defaultProvider": "google"
}
```

实现说明：

- Next route handler 会转发到 `/models/config`。
- 后端根据 generator provider 配置构建响应。
- 后端报错时，FastAPI 会返回一个回退的 Google provider 配置。

### 4.5 `GET /api/wiki/projects`

前端 URL：

```http
GET /api/wiki/projects
```

后端目标：

```http
GET /api/processed_projects
```

用途：

- 为 processed-projects UI 列出已生成/已缓存的 wiki 项目。

响应：

```json
[
  {
    "id": "deepwiki_cache_github_owner_repo_en.json",
    "owner": "owner",
    "repo": "repo",
    "name": "owner/repo",
    "repo_type": "github",
    "submittedAt": 1710000000000,
    "language": "en"
  }
]
```

实现说明：

- 后端会扫描 `~/.adalflow/wikicache`。
- 缓存文件名按 `deepwiki_cache_{repo_type}_{owner}_{repo}_{language}.json` 解析。
- 结果会按修改时间排序，最新的排在最前面。
- 该 Next 路由使用的是 `PYTHON_BACKEND_HOST`，而不是 `SERVER_BASE_URL`。

### 4.6 `DELETE /api/wiki/projects`

前端 URL：

```http
DELETE /api/wiki/projects
Content-Type: application/json
```

后端目标：

```http
DELETE /api/wiki_cache?owner=...&repo=...&repo_type=...&language=...
```

请求：

```json
{
  "owner": "owner",
  "repo": "repo",
  "repo_type": "github",
  "language": "en"
}
```

响应：

```json
{
  "message": "Project deleted successfully"
}
```

实现说明：

- Next 路由在转发前会校验这四个字段。
- 该代理不会转发 `authorization_code`。
- 如果后端启用了认证模式，删除可能会因 `/api/wiki_cache` 期望从查询参数读取 `authorization_code` 而返回 `401`。

### 4.7 `GET /api/wiki_cache`

前端 URL：

```http
GET /api/wiki_cache?owner=owner&repo=repo&repo_type=github&language=en
```

后端目标：

```http
GET /api/wiki_cache
```

查询参数：

| 名称 | 必填 | 说明 |
|---|---:|---|
| `owner` | 是 | 仓库 owner 或命名空间。 |
| `repo` | 是 | 仓库名称。 |
| `repo_type` | 是 | 仓库 provider/type，例如 `github`、`gitlab`、`bitbucket` 或 `local`。 |
| `language` | 是 | 内容语言。不支持的语言会回退为配置中的默认语言。 |

找到时的响应：

```json
{
  "wiki_structure": {
    "id": "wiki",
    "title": "Project Wiki",
    "description": "Generated wiki",
    "pages": [],
    "sections": [],
    "rootSections": []
  },
  "generated_pages": {},
  "repo": {
    "owner": "owner",
    "repo": "repo",
    "type": "github",
    "token": null,
    "localPath": null,
    "repoUrl": "https://github.com/owner/repo"
  },
  "provider": "google",
  "model": "gemini-2.5-flash"
}
```

未找到时的响应：

```json
null
```

实现说明：

- 若缓存不存在，后端返回 `200` 和 `null`。
- 前端在读取或删除缓存时有时会附带 `comprehensive`、`provider`、`model`、`custom_model` 以及过滤参数。该端点会忽略未知查询参数。
- 后端缓存键不包含 `comprehensive`、provider、model 或过滤条件，只使用 `repo_type`、`owner`、`repo` 和 `language`。

### 4.8 `POST /api/wiki_cache`

前端 URL：

```http
POST /api/wiki_cache
Content-Type: application/json
```

后端目标：

```http
POST /api/wiki_cache
```

请求：

```json
{
  "repo": {
    "owner": "owner",
    "repo": "repo",
    "type": "github",
    "token": null,
    "localPath": null,
    "repoUrl": "https://github.com/owner/repo"
  },
  "language": "en",
  "wiki_structure": {
    "id": "wiki",
    "title": "Project Wiki",
    "description": "Generated wiki",
    "pages": [],
    "sections": [],
    "rootSections": []
  },
  "generated_pages": {},
  "provider": "google",
  "model": "gemini-2.5-flash"
}
```

响应：

```json
{
  "message": "Wiki cache saved successfully"
}
```

实现说明：

- 后端会写入 `~/.adalflow/wikicache/deepwiki_cache_{repo_type}_{owner}_{repo}_{language}.json`。
- 不支持的语言会被替换为配置中的默认语言。
- 前端会发送 `comprehensive`，但后端请求模型并未定义该字段。

### 4.9 `DELETE /api/wiki_cache`

前端 URL：

```http
DELETE /api/wiki_cache?owner=owner&repo=repo&repo_type=github&language=en&authorization_code=code
```

后端目标：

```http
DELETE /api/wiki_cache
```

查询参数：

| 名称 | 必填 | 说明 |
|---|---:|---|
| `owner` | 是 | 仓库 owner 或命名空间。 |
| `repo` | 是 | 仓库名称。 |
| `repo_type` | 是 | 仓库 provider/type。 |
| `language` | 是 | 内容语言，必须受支持。 |
| `authorization_code` | 仅在启用认证模式时必填 | 与后端配置进行比对的认证码。 |

成功响应：

```json
{
  "message": "Wiki cache for owner/repo (en) deleted successfully"
}
```

错误响应：

| 状态码 | 原因 |
|---:|---|
| `400` | 不支持的语言。 |
| `401` | 已启用认证模式，且 `authorization_code` 缺失或无效。 |
| `404` | 缓存文件不存在。 |
| `500` | 删除文件失败。 |

实现说明：

- 仓库 wiki 页面在强制重新生成前会调用该端点。
- processed-projects 代理同样通过这个后端端点删除，但不会传 `authorization_code`。

### 4.10 `POST /export/wiki`

前端 URL：

```http
POST /export/wiki
Content-Type: application/json
```

后端目标：

```http
POST /export/wiki
```

请求：

```json
{
  "repo_url": "https://github.com/owner/repo",
  "pages": [
    {
      "id": "overview",
      "title": "Overview",
      "content": "# Overview\n...",
      "filePaths": ["README.md"],
      "importance": "high",
      "relatedPages": []
    }
  ],
  "format": "markdown"
}
```

响应：

- 当 `format: "markdown"` 时，返回可下载的 `text/markdown` 响应。
- 当 `format: "json"` 时，返回可下载的 `application/json` 响应。
- 后端会设置 `Content-Disposition: attachment; filename=<repo>_wiki_<timestamp>.<md|json>`。

实现说明：

- 前端请求体中包含 `type`，但后端 `WikiExportRequest` 并未定义该字段。
- 后端根据 `repo_url` 的最后一个路径段推导输出文件名。

### 4.11 `GET /local_repo/structure`

前端 URL：

```http
GET /local_repo/structure?path=/absolute/path/to/repo
```

后端目标：

```http
GET /local_repo/structure
```

查询参数：

| 名称 | 必填 | 说明 |
|---|---:|---|
| `path` | 是 | 仓库目录的绝对路径或后端主机本地路径。 |

成功响应：

```json
{
  "file_tree": "README.md\nsrc/index.ts\n...",
  "readme": "# Project\n..."
}
```

错误响应：

| 状态码 | 原因 |
|---:|---|
| `400` | 缺少 `path`。 |
| `404` | `path` 在后端主机上不是一个目录。 |
| `500` | 遍历仓库或读取 README 失败。 |

实现说明：

- 会跳过隐藏文件/目录、`__pycache__`、`node_modules`、`.venv`、`__init__.py` 和 `.DS_Store`。
- 遍历过程中找到的第一个大小写不敏感匹配 `README.md` 的文件会被返回。
- 该路由仅用于 `type=local` 的仓库。

## 5. 流式 API

### 5.1 `WebSocket /ws/chat`

前端 URL：

```text
ws://localhost:8001/ws/chat
```

或者在 HTTPS 部署中：

```text
wss://<backend-host>/ws/chat
```

后端端点：

```text
/ws/chat
```

用途：

- 仓库 wiki 生成的主流式端点。
- Ask/chat 的主流式端点。
- 也用于 slides 和 workshop 生成。

请求：

客户端打开 socket，等待 `onopen`，然后发送一个 `ChatCompletionRequest` JSON 对象：

```json
{
  "repo_url": "https://github.com/owner/repo",
  "type": "github",
  "messages": [
    {
      "role": "user",
      "content": "Generate a wiki structure..."
    }
  ],
  "provider": "google",
  "model": "gemini-2.5-flash",
  "language": "en",
  "token": "optional-private-repo-token",
  "excluded_dirs": "node_modules\n.next",
  "excluded_files": "*.lock",
  "included_dirs": "src\napi",
  "included_files": "*.ts\n*.py"
}
```

响应：

- 服务端通过 `send_text` 发送原始文本分片。
- 客户端将这些分片拼接成一个字符串。
- 完成后服务端会关闭 WebSocket。
- 错误情况同样以原始文本发送，通常带有 `Error:` 前缀，随后关闭 socket。

校验与错误：

| 条件 | 行为 |
|---|---|
| `messages` 为空 | 发送 `Error: No messages provided`，然后关闭。 |
| 最后一条消息的 `role` 不是 `user` | 发送 `Error: Last message must be from the user`，然后关闭。 |
| Retriever 准备失败 | 发送错误文本分片，然后关闭。 |
| Embedding 维度不一致 | 发送 embedding 相关错误文本分片，然后关闭。 |
| Provider 调用失败 | 如果在 provider 分支中被捕获，则发送 provider 相关文本。 |

前端调用方：

- `src/app/[owner]/[repo]/page.tsx` 中的 wiki 结构生成。
- `src/app/[owner]/[repo]/page.tsx` 中的 wiki 页面内容生成。
- `src/app/[owner]/[repo]/ask/page.tsx` 中的 Ask 页面。
- `src/app/[owner]/[repo]/slides/page.tsx` 中的 Slides 页面。
- `src/app/[owner]/[repo]/workshop/page.tsx` 中的 Workshop 页面。
- `src/utils/websocketClient.ts` 中的共享辅助函数。

实现说明：

- 该 WebSocket 为浏览器直连后端，不经过 `next.config.ts` 代理。
- 某些客户端组件会直接读取 `process.env.SERVER_BASE_URL`。在浏览器 bundle 中，非 `NEXT_PUBLIC_` 环境变量可能无法按预期获取，因此本地开发时回退到 `http://localhost:8001` 很重要。
- 代码会将 `http` 转为 `ws`，将 `https` 转为 `wss`。

### 5.2 `POST /api/chat/stream`

前端 URL：

```http
POST /api/chat/stream
Content-Type: application/json
```

后端目标：

```http
POST /chat/completions/stream
```

用途：

- 当 `/ws/chat` 失败或超时时的 HTTP 回退方案。
- 也被 Ask 页面回退逻辑以及 wiki/slides/workshop 回退逻辑直接使用。

请求：

与 `/ws/chat` 的 `ChatCompletionRequest` 结构相同。

响应：

- Next route handler 返回一个 `ReadableStream`。
- 后端返回 `StreamingResponse`。
- 分片是原始文本，而不是结构化 JSON 事件。
- Next 代理会设置 `Cache-Control: no-cache, no-transform`，并在存在时转发后端的 `Content-Type`。

后端校验：

| 条件 | 状态码 | 详情 |
|---|---:|---|
| 没有 messages | `400` | `No messages provided` |
| 最后一条消息不是 user | `400` | `Last message must be from the user` |
| Retriever 准备报错 | `500` | 错误详情文本 |
| Embedding 不一致 | `500` | embedding 相关详情 |

实现说明：

- Next 代理会发送 `Accept: text/event-stream`，但后端返回的是原始文本分片。不要期待 Server-Sent Event 的帧格式。
- HTTP 实现和 WebSocket 实现共享同一个请求模型，生成逻辑也几乎一致。

### 5.3 `WebSocket /ws/agent-wiki`

后端端点：

```text
/ws/agent-wiki
```

前端状态：

- 已在 FastAPI 中注册。
- 在 `src/` 中未发现当前前端调用方。

用途：

- 使用两阶段协议进行基于 Agent 的 wiki 生成：
  1. Planner 阶段生成 wiki 结构。
  2. Writer 阶段逐页编写内容。

请求：

```json
{
  "repo_url": "https://github.com/owner/repo",
  "type": "github",
  "token": "optional-token",
  "provider": "google",
  "model": "gemini-2.5-flash",
  "language": "en",
  "comprehensive": true,
  "file_tree_hint": "optional file tree",
  "readme_hint": "optional README",
  "excluded_dirs": "node_modules\n.next",
  "excluded_files": "*.lock",
  "included_dirs": "src\napi",
  "included_files": "*.ts\n*.py"
}
```

服务端事件：

| 事件类型 | 方向 | 含义 |
|---|---|---|
| `text_delta` | 服务端 → 客户端 | 增量模型文本。 |
| `tool_call_start` | 服务端 → 客户端 | Agent 请求执行一次工具调用。 |
| `tool_call_end` | 服务端 → 客户端 | 工具执行完成。 |
| `wiki_structure_ready` | 服务端 → 客户端 | Planner 产出了有效的 wiki 结构。 |
| `wiki_structure_error` | 服务端 → 客户端 | Planner 或初始化失败。 |
| `wiki_page_done` | 服务端 → 客户端 | 一个 wiki 页面完成。 |
| `wiki_page_error` | 服务端 → 客户端 | 某一页失败；会话可能继续。 |
| `finish` | 服务端 → 客户端 | 整个 agent 流完成。 |
| `error` | 服务端 → 客户端 | provider 或 agent-loop 错误。 |

实现说明：

- 事件是带有判别字段 `type` 的 JSON 对象。
- Planner 和 Writer 阶段会附带诸如 `phase`、`page_index` 和 `page_id` 等元数据。
- 该端点会在规划前先 clone/download 仓库。

## 6. 端到端流程

### 6.1 首页启动

```text
首页
  ├─ GET /api/auth/status
  └─ 用户打开模型选择器
       └─ GET /api/models/config
```

认证状态决定模型选择弹窗是否需要在生成前要求用户输入认证码。

### 6.2 Wiki 页面生成

```text
仓库 wiki 页面
  │
  ├─ GET /api/auth/status
  ├─ GET /api/wiki_cache?owner&repo&repo_type&language
  │    ├─ cache hit  → 渲染缓存的 wiki
  │    └─ cache miss → 获取仓库结构
  │
  ├─ 托管仓库：浏览器直接调用 GitHub/GitLab/Bitbucket API
  ├─ 本地仓库：GET /local_repo/structure?path=...
  │
  ├─ WebSocket /ws/chat
  │    └─ 生成 wiki 结构
  │
  ├─ 每个页面一个 WebSocket /ws/chat
  │    └─ 生成页面内容
  │
  └─ POST /api/wiki_cache
       └─ 持久化生成后的 wiki
```

如果 WebSocket 建连或流式传输失败，页面会回退到 `POST /api/chat/stream`。

### 6.3 Ask 页面流程

```text
Ask 页面
  ├─ 若 URL 状态缺少 provider/model，则 GET /api/models/config
  ├─ WebSocket /ws/chat
  │    └─ 流式返回回答分片
  └─ POST /api/chat/stream
       └─ WebSocket 出错时回退
```

Ask 页面会把历史轮次作为 `messages` 一并发送，以便后端在回答前重建对话记忆。

### 6.4 Slides 与 Workshop 流程

```text
Slides / Workshop 页面
  ├─ GET /api/wiki_cache?owner&repo&repo_type&language
  ├─ WebSocket /ws/chat
  │    └─ 生成计划/内容
  └─ POST /api/chat/stream
       └─ WebSocket 失败时回退
```

Slides 会先生成一个计划，再将每一页 slide 生成为 HTML。Workshop 则生成一个单体的长篇 workshop 文档。

### 6.5 缓存刷新流程

```text
用户请求刷新
  ├─ DELETE /api/wiki_cache?owner&repo&repo_type&language&authorization_code=...
  ├─ 清空本地组件状态
  ├─ 重新生成结构和页面
  └─ POST /api/wiki_cache
```

如果后端启用了认证模式，则刷新流程需要有效的认证码。

## 7. 运行层面的注意事项

### 7.1 重复的认证代理定义

`/api/auth/status` 和 `/api/auth/validate` 同时存在于：

- `src/app/api/auth/*/route.ts` 中的 Next route handler。
- `next.config.ts` 中的 Next rewrites。

实际运行中，是 route handler 在转发到后端。除非移除这些 route handler，否则 rewrites 是冗余的。

### 7.2 不同的后端 URL 环境变量

大多数代理使用 `SERVER_BASE_URL`，但 `/api/wiki/projects` 使用 `PYTHON_BACKEND_HOST`。如果两者不一致，processed-project 列表/删除可能会指向与应用其余部分不同的后端。

### 7.3 直接 WebSocket 不受 Next Rewrites 覆盖

REST 路由可以通过 Next 以同源方式访问，但 `/ws/chat` 是直连后端主机。生产部署必须将 FastAPI WebSocket 端点暴露给浏览器，或额外添加一个 WebSocket 代理。

### 7.4 缓存键不包含生成模式

前端会在本地跟踪 `comprehensive`，并在一些请求中发送它，但后端缓存文件名只包含：

```text
repo_type + owner + repo + language
```

这意味着同一仓库/语言下的 concise 和 comprehensive 生成结果可能会在服务端互相覆盖。

### 7.5 自定义模型字段不匹配

若干前端生成路径会发送 `custom_model`，但后端聊天 schema 只接受 `model`。如果没有把自定义模型赋给 `model`，后端就会忽略 `custom_model`。

Ask 页面在这点上处理正确：启用自定义模式时，会把 `model` 设为自定义模型。Wiki/slides/workshop 辅助逻辑则分别设置了 `model` 和 `custom_model`。

### 7.6 Processed-Project 删除不会转发认证码

`DELETE /api/wiki/projects` 只会转发 `owner`、`repo`、`repo_type` 和 `language`。如果后端启用了认证模式，目标 `DELETE /api/wiki_cache` 可能会以 `401` 拒绝请求。

### 7.7 HTTP 聊天回退是原始文本流

代理声明的是 `Accept: text/event-stream`，但后端输出的是原始文本分片。客户端应直接读取流正文并拼接分片，而不是按 SSE 帧解析。

### 7.8 WebSocket 错误格式是纯文本

`/ws/chat` 将正常模型输出和错误都作为纯文本发送。客户端无法区分结构化错误与普通输出，只能依赖诸如 `Error:` 这样的字符串约定。

### 7.9 后端 CORS 较为宽松

FastAPI 允许所有 origin、method 和 header。这有利于本地开发以及直接访问 WebSocket/REST，但生产部署应考虑收紧 CORS。

## 8. 源码映射

| 区域 | 源文件 |
|---|---|
| Next rewrites | `next.config.ts` |
| Auth status proxy | `src/app/api/auth/status/route.ts` |
| Auth validate proxy | `src/app/api/auth/validate/route.ts` |
| Chat HTTP fallback proxy | `src/app/api/chat/stream/route.ts` |
| Model config proxy | `src/app/api/models/config/route.ts` |
| Processed projects proxy | `src/app/api/wiki/projects/route.ts` |
| Shared WebSocket helper | `src/utils/websocketClient.ts` |
| Main wiki page callers | `src/app/[owner]/[repo]/page.tsx` |
| Ask page callers | `src/app/[owner]/[repo]/ask/page.tsx` |
| Slides page callers | `src/app/[owner]/[repo]/slides/page.tsx` |
| Workshop page callers | `src/app/[owner]/[repo]/workshop/page.tsx` |
| FastAPI app and REST routes | `api/api.py` |
| HTTP chat streaming backend | `api/simple_chat.py` |
| WebSocket chat backend | `api/websocket_wiki.py` |
| Agent wiki WebSocket backend | `api/agent/wiki_generator.py` |
| Agent event models | `api/agent/stream_events.py` |
