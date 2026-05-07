---
number: REC-003
name: Vector DB Cache and Generation Retry Bugfix
description: Records the fix for redundant embedding rebuilds from legacy Python metadata checks and frontend HTTP retry timeouts during wiki generation.
update_at: 2026-05-07
category: implementation-record
language: en
status: recorded
---

# Vector DB Cache and Generation Retry Bugfix

## 1. Problem

When a repository URL was submitted from the frontend, the wiki generation route could rebuild embeddings even when a usable vector database already existed under `~/.adalflow/databases/`.

Observed symptoms:

- Backend logs showed `Loaded ... documents from existing database`, followed by `Existing database is missing Python chunk metadata ... Rebuilding embeddings`.
- During long wiki generation, the frontend could fall back to `POST /api/chat/stream` and produce a Next.js `UND_ERR_HEADERS_TIMEOUT`.
- Refreshing `/{owner}/{repo}?status=generating` remounted the generation screen and restarted the client-side generation path. With the old backend cache check, this could immediately re-enter embedding for an existing DB.

Expected behavior:

- If the vector DB exists and contains non-empty embeddings, reuse it and skip embedding.
- Wiki generation should not retry the same non-idempotent generation request through a second HTTP transport after the WebSocket path is already in flight.

## 2. Root Cause

### Backend cache invalidation was too strict

`DatabaseManager.prepare_db_index()` already loaded `LocalDB` and counted non-empty vectors. However, it also called `_needs_python_metadata_refresh()` and forced a full rebuild if cached Python chunks were missing newer AST metadata fields:

- `start_line`
- `end_line`
- `ast_chunk_index`
- `ast_chunk_count`

Those fields improve neighbor expansion and ordering, but they are not required to reuse existing embedding vectors. Treating their absence as a cache miss made old but valid vector DBs expensive and non-idempotent.

### Frontend retried generation through a second transport

`src/utils/wikiGeneration.ts` used raw-text `/ws/chat` for wiki generation, but on WebSocket failure or a 180 second client timeout it fell back to `POST /api/chat/stream`.

That fallback is unsafe for wiki generation because it sends the same request again. The backend prepares RAG again for each request, so the fallback could trigger another clone/load/embed/generate path. While the backend was busy before sending response headers, the Next.js route handler could hit `UND_ERR_HEADERS_TIMEOUT`.

## 3. Fix

Backend change in `api/data_pipeline.py`:

- Keep the existing vector-length check.
- Rebuild only when all cached documents have empty or missing vectors.
- Return cached documents whenever at least one usable embedding exists.
- Remove the Python metadata refresh gate from DB reuse.

Frontend change in `src/utils/wikiGeneration.ts`:

- Remove the HTTP fallback to `/api/chat/stream` from wiki generation.
- Replace the 180 second WebSocket timeout with a 45 minute timeout guard.
- Keep wiki generation on the single raw WebSocket transport for the lifetime of the request.

Regression test in `tests/unit/test_retriever.py`:

- Added `test_database_manager_reuses_legacy_python_pickle_without_reembedding`.
- The test constructs a cached Python document with a non-empty vector but no AST metadata.
- It asserts `prepare_db_index()` returns cached docs and does not call `transform_documents_and_save_to_db()`.

## 4. Verification

Commands run after the fix:

```bash
api/.venv/bin/python -m pytest tests/unit/test_retriever.py -q --tb=short
bun run lint
git diff --check
```

Results:

- Focused backend regression: `8 passed`.
- Frontend lint: no warnings or errors.
- Whitespace check: passed.

Broader unit verification was also run:

```bash
api/.venv/bin/python -m pytest tests/unit/ -q --tb=short -k unit
```

Result:

- `250 passed`, `7 failed`.
- The failures were existing unrelated issues in embedder config/live API assumptions and multi-hop metadata expectations, not introduced by this bugfix.

## 5. Manual Test Notes

For a repository whose vector DB was deleted:

1. First generation should build the vector DB once.
2. Subsequent generation attempts for the same repo should log that the existing DB was loaded and should skip embedding.
3. Long wiki generation should stay on `/ws/chat`; it should not produce a secondary `POST /api/chat/stream` timeout from the wiki generation path.
4. Wiki generation should fail only if the single WebSocket request exceeds the 45 minute timeout guard.

Known boundary:

- `?status=generating` is still a client-side generation route. If the vector DB does not exist yet and the page is refreshed before the first build completes, there is no backend job lock that can reuse an unfinished build. This record covers the fixed existing-DB reuse bug and the removed frontend HTTP retry. A persistent server-side generation job would be a separate architecture change.

## 6. Rollback Notes

If this fix must be rolled back:

- Restoring the Python metadata gate will reintroduce full embedding rebuilds for legacy caches.
- Restoring the HTTP fallback can reintroduce duplicate generation requests and Next.js header timeouts during long wiki generation.
- Shortening the WebSocket timeout too aggressively can fail valid large-repo generation before the backend finishes streaming.
