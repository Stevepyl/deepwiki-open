# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DeepWiki-Open is a full-stack application that generates AI-powered interactive wikis from GitHub/GitLab/Bitbucket repositories. It analyzes code, creates vector embeddings, streams wiki generation via WebSocket, and provides a RAG-powered chat interface.

## Development Commands

### Backend (Python/FastAPI)

```bash
# Install dependencies (run from project root)
uv sync --directory api # install all deps including dev if in the root directory of the proj

# Activating virtual environment
source api/.venv/bin/activate

# Start API server (port 8001, auto-reload enabled)
uv run -m api.main
```

### Frontend (Next.js)

```bash
bun install
bun dev       # Development with Turbopack hot reload (port 3000)
bun run build     # Production build
bun run lint      # ESLint
```

### Docker

```bash
docker-compose up           # Build and run both services
PORT=9000 docker-compose up # Custom port
```

### Testing

```bash
pytest tests/              # All tests
pytest tests/unit/         # Unit tests only
pytest tests/integration/  # Integration tests
pytest tests/api/          # API endpoint tests
pytest tests/ -v -k unit   # Verbose with marker filter
```

## Environment Setup

Copy `.env.example` to `.env` in project root. Key variables:

| Variable | Purpose |
|---|---|
| `GOOGLE_API_KEY` | Google Gemini + embeddings |
| `OPENAI_API_KEY` | OpenAI models + embeddings |
| `OPENROUTER_API_KEY` | Multi-model access |
| `OLLAMA_HOST` | Local self-hosted models |
| `SERVER_BASE_URL` | Frontend-accessible API URL (default: `http://localhost:8001`) |
| `DEEPWIKI_AUTH_MODE` | Enable auth code gate |
| `DEEPWIKI_CONFIG_DIR` | Override config file directory |
| `PORT` | API server port (default: 8001) |

## Architecture

### Request Flow

```
User (GitHub URL) → Frontend (Next.js WebSocket)
    → Backend data_pipeline.py (clone repo, chunk code, create FAISS embeddings)
    → websocket_wiki.py (LLM generation, streaming)
    → Frontend (real-time render: Mermaid diagrams, Markdown, WikiTreeView)

Ask/Chat → rag.py (FAISS vector search → context assembly → LLM → streaming response)
```

### Backend (`api/`)

- **`api.py`** — FastAPI app, all routes, WebSocket endpoint, CORS
- **`data_pipeline.py`** — Repo cloning, file traversal, code chunking (350-word chunks, 100-word overlap), FAISS embedding storage
- **`rag.py`** — RAG implementation: vector similarity search + prompt assembly + streaming
- **`websocket_wiki.py`** — Wiki generation orchestration via WebSocket
- **`simple_chat.py`** — LLM chat completion abstraction
- **`config.py`** — Loads `api/config/*.json` files with env var overrides
- **`prompts.py`** — All LLM prompt templates

Provider clients: `google_embedder_client.py`, `openai_client.py`, `openrouter_client.py`, `azureai_client.py`, `bedrock_client.py`, `dashscope_client.py`, `ollama_patch.py`

### Frontend (`src/`)

- **`app/[owner]/[repo]/page.tsx`** — Main wiki display page
- **`app/page.tsx`** — Home page (repo URL input, model selection)
- **`components/Ask.tsx`** — RAG chat interface
- **`components/Mermaid.tsx`** — Diagram renderer
- **`components/Markdown.tsx`** — Markdown + syntax highlighting

API proxying: `next.config.ts` rewrites route `/api/wiki_cache/*`, `/export/wiki/*`, and `/api/auth/*` to the Python backend at `SERVER_BASE_URL`.

### Configuration Files (`api/config/`)

- **`generator.json`** — LLM provider and model definitions
- **`embedder.json`** — Embedding model settings
- **`repo.json`** — File filters and repository processing rules
- **`lang.json`** — Language configuration

### Persistent Storage (defaults to `~/.adalflow/`)

- `repos/` — Cloned repositories
- `databases/` — FAISS vector indexes
- `wikicache/` — Generated wiki cache

## Key Patterns

- Wiki generation streams over WebSocket; REST endpoints are used for cache reads, auth, and chat
- All LLM providers implement a common interface; the active provider is selected via config/env vars
- Next.js API routes in `src/app/api/` are thin proxies that call the Python backend; direct backend rewrites handle the rest
- `pytest.ini` marks tests as `unit`, `integration`, or `api` — use `-k` flag to filter

## Verification

Done-conditions for common task types:

| Task | Verification Command |
|------|----------------------|
| Backend API changes | `pytest tests/api/ -q --tb=short 2>&1 \| tail -50` and manual test in browser |
| Backend logic (RAG, data pipeline) | `pytest tests/unit/ -q --tb=short -k unit 2>&1 \| tail -50` |
| Frontend component changes | `bun run lint 2>&1 \| tail -30` (no errors), manual test at `http://localhost:3000` |
| Full-stack changes | Run all above + verify WebSocket connection in Network tab |
| Configuration changes | Verify `.env.example` documents new vars; manual test with fresh `.env` |

**Rule:** Before marking work complete, run the verification command for your change type. Do not skip verification even if the code looks correct.

## Code Standards

See `~/.claude/rules/coding-style.md` for:
- Immutability patterns (CRITICAL: always use spread/destructuring, never mutate)
- Naming conventions (camelCase JS/TS, snake_case Python)
- Code organization (200-400 lines per file, high cohesion)
- Error handling and input validation requirements

