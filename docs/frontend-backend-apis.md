# Frontend ↔ Backend API Reference

This document lists the API boundary between the Next.js frontend and the FastAPI backend in DeepWiki-Open. It covers the browser-visible routes, the Python backend routes they reach, request and response shapes, callers, and operational caveats.

## 1. Communication Model

DeepWiki-Open uses three communication paths:

```text
Browser client
  │
  ├─ Same-origin REST calls
  │    /api/wiki_cache, /export/wiki, /api/lang/config, ...
  │
  ├─ Next.js route-handler proxies
  │    /api/chat/stream, /api/models/config, /api/wiki/projects, ...
  │
  └─ Direct WebSocket calls
       ws://<backend>/ws/chat
```

The frontend is a Next.js application in `src/`. The backend is a FastAPI app in `api/api.py`.

### Backend Base URLs

| Setting | Used by | Default | Notes |
|---|---|---:|---|
| `SERVER_BASE_URL` | `next.config.ts`, most Next route proxies, several client components | `http://localhost:8001` | Main backend URL. |
| `PYTHON_BACKEND_HOST` | `src/app/api/wiki/projects/route.ts` only | `http://localhost:8001` | Separate env var for processed-project proxy. |

### Proxy Strategy

The frontend mostly calls same-origin URLs such as `/api/wiki_cache`. Next.js either rewrites those URLs to the backend or handles them through route handlers that call the backend with `fetch()`.

Source files:

- `next.config.ts` defines rewrites for cache, export, auth, language config, and local repo structure.
- `src/app/api/*/route.ts` contains explicit Next route-handler proxies.
- `api/api.py` registers the FastAPI REST and WebSocket endpoints.

## 2. API Inventory

### Frontend-Visible APIs

| Frontend URL | Method | Backend target | Transport | Primary purpose |
|---|---:|---|---|---|
| `/api/lang/config` | `GET` | `/lang/config` | Next rewrite | Load supported UI/content languages. |
| `/api/auth/status` | `GET` | `/auth/status` | Next route proxy and rewrite | Check whether wiki generation requires an auth code. |
| `/api/auth/validate` | `POST` | `/auth/validate` | Next route proxy and rewrite | Validate user-entered auth code. |
| `/api/models/config` | `GET` | `/models/config` | Next route proxy | Load available providers and models. |
| `/api/wiki/projects` | `GET` | `/api/processed_projects` | Next route proxy | List cached/generated projects. |
| `/api/wiki/projects` | `DELETE` | `/api/wiki_cache` | Next route proxy | Delete a project cache from the processed-projects UI. |
| `/api/wiki_cache` | `GET` | `/api/wiki_cache` | Next rewrite | Load cached wiki data. |
| `/api/wiki_cache` | `POST` | `/api/wiki_cache` | Next rewrite | Store generated wiki data. |
| `/api/wiki_cache` | `DELETE` | `/api/wiki_cache` | Next rewrite | Clear server-side wiki cache before regeneration. |
| `/export/wiki` | `POST` | `/export/wiki` | Next rewrite | Download generated wiki as Markdown or JSON. |
| `/local_repo/structure` | `GET` | `/local_repo/structure` | Next rewrite | Read a local repository file tree and README. |
| `ws://<backend>/ws/chat` | WebSocket | `/ws/chat` | Direct browser WebSocket | Stream wiki generation, Ask answers, slides, and workshop content. |
| `/api/chat/stream` | `POST` | `/chat/completions/stream` | Next route proxy | HTTP streaming fallback for `/ws/chat`. |

### Backend-Registered APIs Not Currently Called by the Frontend

| Backend URL | Method | Purpose |
|---|---:|---|
| `/ws/agent-wiki` | WebSocket | Agent-based two-phase wiki generator. Registered in FastAPI, but no current frontend caller was found. |
| `/health` | `GET` | Backend health check for Docker/monitoring. |
| `/` | `GET` | Root endpoint that lists registered routes dynamically. |

### External APIs Called Directly by the Frontend

These are not frontend↔backend APIs, but they matter for the full data flow:

- GitHub REST API for repository tree and README.
- GitLab REST API for project metadata, repository tree, and README.
- Bitbucket REST API for repository metadata, source tree, and README.

The local repository case goes through the backend via `/local_repo/structure`; hosted repository metadata is fetched directly from the browser.

## 3. Shared Data Models

### `ChatMessage`

Used by `/ws/chat` and `/api/chat/stream`.

```ts
{
  role: "user" | "assistant" | "system";
  content: string;
}
```

Backend validation expects the final message to have `role: "user"`.

### `ChatCompletionRequest`

Used by `/ws/chat` and `/api/chat/stream`.

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

Notes:

- `excluded_*` and `included_*` are strings. The backend splits them by newline, despite comments in the backend model saying "comma-separated".
- `custom_model` is sent by some frontend generators, but the backend `ChatCompletionRequest` model does not define it. Unless the selected custom model is copied into `model`, the backend ignores `custom_model`.
- The backend prepares or reuses repository embeddings before producing a response.

### `RepoInfo`

Used in wiki cache payloads and URL-state reconstruction.

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

Used in wiki cache, export payloads, and frontend rendering.

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

The backend Pydantic model requires `id`, `title`, `content`, `filePaths`, `importance`, and `relatedPages`. Extra fields are not used by the backend response model.

### `WikiSection`

Used by the frontend wiki tree and persisted in cache when present.

```ts
{
  id: string;
  title: string;
  pages: string[];
  subsections?: string[];
}
```

### `WikiStructure`

Used in cache payloads and frontend rendering.

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

Returned by `GET /api/wiki_cache` and written by `POST /api/wiki_cache`.

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

The frontend also sends `comprehensive` when saving cache, but the backend cache request model does not define it.

## 4. REST API Contracts

### 4.1 `GET /api/lang/config`

Frontend URL:

```http
GET /api/lang/config
```

Backend target:

```http
GET /lang/config
```

Purpose:

- Loads available language codes and the default language.
- Used by `LanguageContext`.

Response:

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

Implementation notes:

- The backend returns `configs["lang_config"]` from `api/config/lang.json`.
- No request body or query parameters.

### 4.2 `GET /api/auth/status`

Frontend URL:

```http
GET /api/auth/status
```

Backend target:

```http
GET /auth/status
```

Purpose:

- Checks whether auth-code gating is enabled before wiki generation or cache refresh.

Response:

```json
{
  "auth_required": true
}
```

Implementation notes:

- `auth_required` is derived from `DEEPWIKI_AUTH_MODE` through backend config.
- There is both a Next route handler and a rewrite for this path. The route handler forwards to the same backend endpoint.

### 4.3 `POST /api/auth/validate`

Frontend URL:

```http
POST /api/auth/validate
Content-Type: application/json
```

Backend target:

```http
POST /auth/validate
```

Request:

```json
{
  "code": "user-entered-code"
}
```

Response:

```json
{
  "success": true
}
```

Implementation notes:

- The backend compares `code` against `DEEPWIKI_AUTH_CODE`.
- There is both a Next route handler and a rewrite for this path. The route handler forwards to the same backend endpoint.
- The endpoint returns `200` with `success: false` for an invalid code, rather than a `401`.

### 4.4 `GET /api/models/config`

Frontend URL:

```http
GET /api/models/config
```

Backend target:

```http
GET /models/config
```

Purpose:

- Loads provider/model choices for model selectors and Ask page fallbacks.

Response:

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

Implementation notes:

- Next route handler forwards to `/models/config`.
- Backend builds the response from generator provider config.
- On backend errors, FastAPI returns a fallback Google provider config.

### 4.5 `GET /api/wiki/projects`

Frontend URL:

```http
GET /api/wiki/projects
```

Backend target:

```http
GET /api/processed_projects
```

Purpose:

- Lists generated/cached wiki projects for the processed-projects UI.

Response:

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

Implementation notes:

- The backend scans `~/.adalflow/wikicache`.
- Cache filenames are parsed as `deepwiki_cache_{repo_type}_{owner}_{repo}_{language}.json`.
- Results are sorted by modified time, newest first.
- The Next route uses `PYTHON_BACKEND_HOST`, not `SERVER_BASE_URL`.

### 4.6 `DELETE /api/wiki/projects`

Frontend URL:

```http
DELETE /api/wiki/projects
Content-Type: application/json
```

Backend target:

```http
DELETE /api/wiki_cache?owner=...&repo=...&repo_type=...&language=...
```

Request:

```json
{
  "owner": "owner",
  "repo": "repo",
  "repo_type": "github",
  "language": "en"
}
```

Response:

```json
{
  "message": "Project deleted successfully"
}
```

Implementation notes:

- The Next route validates all four fields before forwarding.
- The proxy does not forward `authorization_code`.
- If backend auth mode is enabled, backend deletion can fail with `401` because `/api/wiki_cache` expects `authorization_code` as a query parameter.

### 4.7 `GET /api/wiki_cache`

Frontend URL:

```http
GET /api/wiki_cache?owner=owner&repo=repo&repo_type=github&language=en
```

Backend target:

```http
GET /api/wiki_cache
```

Query parameters:

| Name | Required | Description |
|---|---:|---|
| `owner` | Yes | Repository owner or namespace. |
| `repo` | Yes | Repository name. |
| `repo_type` | Yes | Repository provider/type, such as `github`, `gitlab`, `bitbucket`, or `local`. |
| `language` | Yes | Content language. Unsupported languages fall back to the configured default. |

Response when found:

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

Response when not found:

```json
null
```

Implementation notes:

- The backend returns `200` with `null` if no cache exists.
- The frontend sometimes passes `comprehensive`, `provider`, `model`, `custom_model`, and filter parameters while reading or deleting cache. The backend ignores unknown query parameters for this endpoint.
- The cache key on the backend does not include `comprehensive`, provider, model, or filters. It only uses `repo_type`, `owner`, `repo`, and `language`.

### 4.8 `POST /api/wiki_cache`

Frontend URL:

```http
POST /api/wiki_cache
Content-Type: application/json
```

Backend target:

```http
POST /api/wiki_cache
```

Request:

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

Response:

```json
{
  "message": "Wiki cache saved successfully"
}
```

Implementation notes:

- The backend writes to `~/.adalflow/wikicache/deepwiki_cache_{repo_type}_{owner}_{repo}_{language}.json`.
- Unsupported languages are replaced with the configured default language.
- The frontend sends `comprehensive`, but the backend request model does not define it.

### 4.9 `DELETE /api/wiki_cache`

Frontend URL:

```http
DELETE /api/wiki_cache?owner=owner&repo=repo&repo_type=github&language=en&authorization_code=code
```

Backend target:

```http
DELETE /api/wiki_cache
```

Query parameters:

| Name | Required | Description |
|---|---:|---|
| `owner` | Yes | Repository owner or namespace. |
| `repo` | Yes | Repository name. |
| `repo_type` | Yes | Repository provider/type. |
| `language` | Yes | Content language. Must be supported. |
| `authorization_code` | Required only when auth mode is enabled | Auth code checked against backend config. |

Success response:

```json
{
  "message": "Wiki cache for owner/repo (en) deleted successfully"
}
```

Error responses:

| Status | Cause |
|---:|---|
| `400` | Unsupported language. |
| `401` | Auth mode is enabled and `authorization_code` is missing or invalid. |
| `404` | Cache file does not exist. |
| `500` | File deletion failed. |

Implementation notes:

- The repository wiki page uses this endpoint before forced regeneration.
- The processed-projects proxy also deletes through this backend endpoint, but does not pass `authorization_code`.

### 4.10 `POST /export/wiki`

Frontend URL:

```http
POST /export/wiki
Content-Type: application/json
```

Backend target:

```http
POST /export/wiki
```

Request:

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

Response:

- `format: "markdown"` returns a downloadable `text/markdown` response.
- `format: "json"` returns a downloadable `application/json` response.
- The backend sets `Content-Disposition: attachment; filename=<repo>_wiki_<timestamp>.<md|json>`.

Implementation notes:

- The frontend includes `type` in the request body, but the backend `WikiExportRequest` does not define it.
- The backend derives the output filename from the last segment of `repo_url`.

### 4.11 `GET /local_repo/structure`

Frontend URL:

```http
GET /local_repo/structure?path=/absolute/path/to/repo
```

Backend target:

```http
GET /local_repo/structure
```

Query parameters:

| Name | Required | Description |
|---|---:|---|
| `path` | Yes | Absolute or backend-local path to the repository directory. |

Success response:

```json
{
  "file_tree": "README.md\nsrc/index.ts\n...",
  "readme": "# Project\n..."
}
```

Error responses:

| Status | Cause |
|---:|---|
| `400` | Missing `path`. |
| `404` | `path` is not a directory on the backend host. |
| `500` | Repository walk or README read failed. |

Implementation notes:

- Hidden files/directories, `__pycache__`, `node_modules`, `.venv`, `__init__.py`, and `.DS_Store` are skipped.
- The first case-insensitive `README.md` found during traversal is returned.
- This route is only used for repositories with `type=local`.

## 5. Streaming APIs

### 5.1 `WebSocket /ws/chat`

Frontend URL:

```text
ws://localhost:8001/ws/chat
```

or, for HTTPS deployments:

```text
wss://<backend-host>/ws/chat
```

Backend endpoint:

```text
/ws/chat
```

Purpose:

- Main streaming endpoint for repository wiki generation.
- Main streaming endpoint for Ask/chat.
- Also used for slides and workshop generation.

Request:

The client opens the socket, waits for `onopen`, then sends one `ChatCompletionRequest` JSON object:

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

Response:

- Server sends raw text chunks with `send_text`.
- The client concatenates chunks into one string.
- The server closes the WebSocket after completion.
- Error cases are also sent as raw text, typically prefixed with `Error:`, then the socket closes.

Validation and errors:

| Condition | Behavior |
|---|---|
| Empty `messages` | Sends `Error: No messages provided`, then closes. |
| Last message is not `role: "user"` | Sends `Error: Last message must be from the user`, then closes. |
| Retriever preparation fails | Sends an error text chunk, then closes. |
| Embedding dimensions are inconsistent | Sends an embedding-specific error text chunk, then closes. |
| Provider call fails | Sends provider-specific text if caught by provider branch. |

Frontend callers:

- Wiki structure generation in `src/app/[owner]/[repo]/page.tsx`.
- Wiki page content generation in `src/app/[owner]/[repo]/page.tsx`.
- Ask page in `src/app/[owner]/[repo]/ask/page.tsx`.
- Slides page in `src/app/[owner]/[repo]/slides/page.tsx`.
- Workshop page in `src/app/[owner]/[repo]/workshop/page.tsx`.
- Shared helper in `src/utils/websocketClient.ts`.

Implementation notes:

- This WebSocket is direct browser-to-backend. It is not proxied by `next.config.ts`.
- Some client components read `process.env.SERVER_BASE_URL` directly. In browser bundles, non-`NEXT_PUBLIC_` env vars may not be available as expected, so the fallback `http://localhost:8001` is important in local development.
- The code converts `http` to `ws` and `https` to `wss`.

### 5.2 `POST /api/chat/stream`

Frontend URL:

```http
POST /api/chat/stream
Content-Type: application/json
```

Backend target:

```http
POST /chat/completions/stream
```

Purpose:

- HTTP fallback when `/ws/chat` fails or times out.
- Used directly by Ask page fallback and by wiki/slides/workshop fallback logic.

Request:

Same shape as `ChatCompletionRequest` for `/ws/chat`.

Response:

- Next route handler returns a `ReadableStream`.
- Backend returns `StreamingResponse`.
- Chunks are raw text, not structured JSON events.
- The Next proxy sets `Cache-Control: no-cache, no-transform` and forwards the backend `Content-Type` when present.

Backend validation:

| Condition | Status | Detail |
|---|---:|---|
| No messages | `400` | `No messages provided` |
| Last message is not user | `400` | `Last message must be from the user` |
| Retriever prep error | `500` | Error detail text |
| Embedding inconsistency | `500` | Embedding-specific detail |

Implementation notes:

- The Next proxy sends `Accept: text/event-stream`, but the backend streams raw text chunks. Do not expect Server-Sent Event framing.
- The HTTP implementation and the WebSocket implementation share the same request model and near-identical generation logic.

### 5.3 `WebSocket /ws/agent-wiki`

Backend endpoint:

```text
/ws/agent-wiki
```

Frontend status:

- Registered in FastAPI.
- No current frontend caller was found in `src/`.

Purpose:

- Agent-based wiki generation using a two-phase protocol:
  1. Planner phase produces the wiki structure.
  2. Writer phase writes pages one at a time.

Request:

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

Server events:

| Event type | Direction | Meaning |
|---|---|---|
| `text_delta` | Server → client | Incremental model text. |
| `tool_call_start` | Server → client | Agent requested a tool call. |
| `tool_call_end` | Server → client | Tool execution completed. |
| `wiki_structure_ready` | Server → client | Planner produced a valid wiki structure. |
| `wiki_structure_error` | Server → client | Planner or setup failed. |
| `wiki_page_done` | Server → client | One wiki page completed. |
| `wiki_page_error` | Server → client | One page failed; session may continue. |
| `finish` | Server → client | Whole agent stream finished. |
| `error` | Server → client | Provider or agent-loop error. |

Implementation notes:

- Events are JSON objects with a discriminating `type` field.
- Planner and writer phases add metadata such as `phase`, `page_index`, and `page_id`.
- The endpoint clones/downloads the repository before planning.

## 6. End-to-End Flows

### 6.1 Home Page Startup

```text
Home page
  ├─ GET /api/auth/status
  └─ User opens model selector
       └─ GET /api/models/config
```

Auth status determines whether the model selection modal asks for an authorization code before generation.

### 6.2 Wiki Page Generation

```text
Repo wiki page
  │
  ├─ GET /api/auth/status
  ├─ GET /api/wiki_cache?owner&repo&repo_type&language
  │    ├─ cache hit  → render cached wiki
  │    └─ cache miss → fetch repository structure
  │
  ├─ Hosted repo: browser calls GitHub/GitLab/Bitbucket APIs directly
  ├─ Local repo:  GET /local_repo/structure?path=...
  │
  ├─ WebSocket /ws/chat
  │    └─ generate wiki structure
  │
  ├─ WebSocket /ws/chat per page
  │    └─ generate page content
  │
  └─ POST /api/wiki_cache
       └─ persist generated wiki
```

If WebSocket setup or streaming fails, the page falls back to `POST /api/chat/stream`.

### 6.3 Ask Page Flow

```text
Ask page
  ├─ GET /api/models/config if URL state lacks provider/model
  ├─ WebSocket /ws/chat
  │    └─ stream answer chunks
  └─ POST /api/chat/stream
       └─ fallback if WebSocket errors
```

The Ask page sends previous turns as `messages`, so the backend can reconstruct conversation memory before answering.

### 6.4 Slides and Workshop Flows

```text
Slides / Workshop pages
  ├─ GET /api/wiki_cache?owner&repo&repo_type&language
  ├─ WebSocket /ws/chat
  │    └─ generate plan/content
  └─ POST /api/chat/stream
       └─ fallback if WebSocket fails
```

Slides generate a plan first, then generate each slide as HTML. Workshop generates a single long-form workshop document.

### 6.5 Cache Refresh Flow

```text
User requests refresh
  ├─ DELETE /api/wiki_cache?owner&repo&repo_type&language&authorization_code=...
  ├─ clear local component state
  ├─ regenerate structure and pages
  └─ POST /api/wiki_cache
```

If backend auth mode is enabled, the refresh flow requires a valid authorization code.

## 7. Operational Caveats

### 7.1 Duplicate Auth Proxy Definitions

`/api/auth/status` and `/api/auth/validate` exist as both:

- Next route handlers in `src/app/api/auth/*/route.ts`.
- Next rewrites in `next.config.ts`.

In practice, the route handlers forward to the backend. The rewrites are redundant unless the route handlers are removed.

### 7.2 Different Backend URL Env Vars

Most proxies use `SERVER_BASE_URL`, but `/api/wiki/projects` uses `PYTHON_BACKEND_HOST`. If these diverge, processed-project listing/deletion can point to a different backend than the rest of the app.

### 7.3 Direct WebSocket Is Not Covered by Next Rewrites

REST routes can be same-origin through Next. `/ws/chat` is direct to the backend host. Production deployments must expose the FastAPI WebSocket endpoint to the browser or add a separate WebSocket proxy.

### 7.4 Cache Key Does Not Include Generation Mode

The frontend tracks `comprehensive` locally and sends it in some requests, but the backend cache filename only includes:

```text
repo_type + owner + repo + language
```

This means concise and comprehensive generations for the same repository/language can overwrite each other on the server.

### 7.5 Custom Model Field Mismatch

Several frontend generation paths send `custom_model`, but the backend chat schema only accepts `model`. If a custom model is not assigned to `model`, the backend ignores `custom_model`.

The Ask page handles this correctly by setting `model` to the custom model when custom mode is active. Wiki/slides/workshop helpers set `model` and `custom_model` separately.

### 7.6 Processed-Project Delete Does Not Forward Auth Code

`DELETE /api/wiki/projects` forwards only `owner`, `repo`, `repo_type`, and `language`. If backend auth mode is enabled, the target `DELETE /api/wiki_cache` can reject the request with `401`.

### 7.7 HTTP Chat Fallback Is Raw Text Streaming

The proxy advertises `Accept: text/event-stream`, but the backend yields raw text chunks. Clients should read the stream body directly and concatenate chunks, not parse SSE frames.

### 7.8 WebSocket Error Shape Is Plain Text

`/ws/chat` sends normal model output and errors as plain text. The client cannot distinguish structured errors from ordinary output except by string conventions such as `Error:`.

### 7.9 Backend CORS Is Permissive

FastAPI allows all origins, methods, and headers. This helps local development and direct WebSocket/REST access, but production deployments should consider narrowing CORS.

## 8. Source Map

| Area | Source |
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

