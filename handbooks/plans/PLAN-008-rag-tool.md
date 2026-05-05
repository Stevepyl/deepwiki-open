---
number: PLAN-008
name: RAG Retrieval as an Agent Tool
description: Extract the retrieval half of api/rag.py into a small CodeRetriever and expose it to the agent as a rag_search tool, reusing the existing data pipeline cache so repeated calls are fast.
status: proposed
update_at: 2026-05-06
category: plan
language: en
audience: developers-and-agents
---

# PLAN-008 — RAG Retrieval as an Agent Tool

## Context

The agent in `api/agent/` currently has no semantic-search tool. It can only navigate the repo through filesystem tools (`grep`, `glob`, `ls`, `read`, `bash`) — fast for keyword lookups, but blind to semantically related code that does not share lexical tokens with the query.

Meanwhile `api/rag.py` already has a working FAISS retriever, with a robust disk cache at `~/.adalflow/databases/{owner}_{repo_name}.pkl` and a multi-stage validation path that survives partial/empty embeddings. The problem is that `RAG.__init__` (`api/rag.py:175-261`) constructs a `Memory`, an `Embedder`, an `adal.Generator` (with provider/model/system prompt/output parser) — all of which the agent does not need. For the agent we want only embedder + transformed_docs + FAISSRetriever.

A second pain point is cost: `prepare_retriever()` (`api/rag.py:696-766`) delegates to `DatabaseManager.prepare_database()` which can clone the repo, walk the tree, chunk, and re-embed. On a cold start that takes minutes. The agent will call this tool many times per session, so a per-process retriever cache is essential — but the disk cache at `~/.adalflow/databases/*.pkl` already exists, so the in-memory layer only has to avoid re-loading the pickle and re-building the FAISS index on every tool invocation.

The intended outcome: a `rag_search` tool that the agent can invoke alongside `grep`/`read`, returning ranked code chunks with file paths and line ranges. RAG-as-tool is the goal; the legacy `RAG` class stays as-is for the existing `/chat/completions/stream` and `/ws/chat` paths. We are explicitly **not** migrating to the OpenAI Agents SDK — this is just one more tool in `api/tools/`.

## Design overview

Two-layer split:

1. **`api/retriever.py` (NEW)** — `CodeRetriever`, a slim component that owns: an embedder, a `DatabaseManager`, transformed documents, and a `FAISSRetriever`. No Memory, no Generator, no provider/model. Plus a module-level `get_or_build_retriever()` that wraps a bounded LRU cache and an `asyncio.Lock` per cache key so two concurrent agents asking for the same repo do not double-build.
2. **`api/tools/rag.py` (NEW)** — `RagTool`, a thin adapter conforming to the existing `Tool` ABC and the single-arg `__init__(repo_path)` factory contract used by `_TOOL_CLASSES`. It calls `get_or_build_retriever(repo_path, "local")` and shapes the documents into a stable text output for the LLM.

`api/rag.py` is **not** touched in this plan. A follow-up can refactor `RAG` to compose `CodeRetriever`, but doing both in one change widens the blast radius and risks the wiki/chat paths.

### Why this split rather than other options

- **Subclass / parameter to skip Memory+Generator** — leaves the heavyweight constructor signature in place, every caller still pays for the import surface, and the boundary "what does the agent actually need from RAG?" stays implicit.
- **Inline the retrieval code into `RagTool`** — works, but bypasses `DatabaseManager.prepare_database()`'s disk-cache + validation logic and makes the tool the only place that knows how to build a retriever. Re-implementing both is bug-prone.
- **`CodeRetriever` reusing `DatabaseManager` directly** (chosen) — the retrieval contract becomes a small named class; the disk cache is reused unchanged; any callsite (tool today, refactored RAG tomorrow) shares one path.

## Files

### NEW — `api/retriever.py` (~150 lines)

```python
import asyncio
import logging
import os
from typing import Optional

from adalflow.components.retriever.faiss_retriever import FAISSRetriever
from adalflow.core.types import Document

from api.config import configs, get_embedder_type
from api.data_pipeline import DatabaseManager
from api.tools.embedder import get_embedder

logger = logging.getLogger(__name__)

_CACHE_MAX = 4  # tuned for typical server memory; one repo's vectors can be ~150MB

class CodeRetriever:
    """Pure FAISS-backed retriever for one repository.

    No Memory, no Generator, no provider. Reuses DatabaseManager so the
    on-disk pickle cache at ~/.adalflow/databases/*.pkl is shared with RAG.
    """

    def __init__(self, embedder_type: Optional[str] = None) -> None:
        self.embedder_type = embedder_type or get_embedder_type()
        self.embedder = get_embedder(embedder_type=self.embedder_type)
        self.db_manager = DatabaseManager()
        self.transformed_docs: list[Document] = []
        self.retriever: Optional[FAISSRetriever] = None

    def prepare(self, repo_url_or_path: str, repo_type: str = "local",
                access_token: Optional[str] = None,
                excluded_dirs=None, excluded_files=None,
                included_dirs=None, included_files=None) -> None:
        # Defensive: trailing slashes break os.path.basename in DatabaseManager._create_repo.
        repo_url_or_path = os.path.normpath(repo_url_or_path)
        docs = self.db_manager.prepare_database(
            repo_url_or_path, repo_type, access_token,
            embedder_type=self.embedder_type,
            excluded_dirs=excluded_dirs, excluded_files=excluded_files,
            included_dirs=included_dirs, included_files=included_files,
        )
        # Mirror RAG._validate_and_filter_embeddings inline (kept small and local;
        # if it grows, lift the helper out of api/rag.py into api/retriever.py).
        docs = [d for d in docs if getattr(d, "vector", None) is not None and len(d.vector) > 0]
        if not docs:
            raise ValueError(f"No usable embeddings for {repo_url_or_path}")
        self.transformed_docs = docs

        cfg = dict(configs.get("retriever", {}) or {})
        cfg.pop("symbol_alpha", None)
        cfg.pop("multi_hop", None)
        self.retriever = FAISSRetriever(
            **cfg,
            embedder=self.embedder,
            documents=docs,
            document_map_func=lambda doc: doc.vector,
        )

    def retrieve(self, query: str, top_k: Optional[int] = None) -> list[Document]:
        if self.retriever is None:
            raise RuntimeError("CodeRetriever.prepare() must be called before retrieve()")
        result = self.retriever(query, top_k=top_k) if top_k else self.retriever(query)
        return [self.transformed_docs[i] for i in result[0].doc_indices]


# --- module-level cache ----------------------------------------------------

_CACHE: "OrderedDict[tuple, CodeRetriever]" = OrderedDict()
_LOCKS: dict[tuple, asyncio.Lock] = {}
_CACHE_GUARD = asyncio.Lock()

def _key(repo_path: str, embedder_type: str) -> tuple:
    return (os.path.normpath(repo_path), embedder_type)

async def get_or_build_retriever(repo_url_or_path: str, repo_type: str = "local",
                                 access_token: Optional[str] = None) -> CodeRetriever:
    embedder_type = get_embedder_type()
    k = _key(repo_url_or_path, embedder_type)

    # Fast path: already cached.
    if k in _CACHE:
        _CACHE.move_to_end(k)
        return _CACHE[k]

    # Slow path: serialise builds for this key.
    async with _CACHE_GUARD:
        lock = _LOCKS.setdefault(k, asyncio.Lock())
    async with lock:
        if k in _CACHE:
            _CACHE.move_to_end(k)
            return _CACHE[k]

        retriever = CodeRetriever(embedder_type=embedder_type)
        await asyncio.to_thread(retriever.prepare, repo_url_or_path, repo_type, access_token)

        _CACHE[k] = retriever
        while len(_CACHE) > _CACHE_MAX:
            evicted_key, _ = _CACHE.popitem(last=False)
            logger.info("CodeRetriever cache evicted %s", evicted_key)
        return retriever
```

Notes:
- Cache key includes `embedder_type` so a config flip (Ollama 768d ↔ OpenAI 1536d) does not return a stale retriever.
- `prepare()` runs in a thread because `DatabaseManager.prepare_database` is sync and can be slow.
- Bound = 4. Larger repos (~150MB of vectors each) make 8 too memory-hungry on a small server.
- We do **not** thread `excluded_dirs/files/...` from the agent request into `prepare()` for v1: filters are an indexing-time concept and threading them would invalidate the shared `.pkl` cache. Filters are handled at tool-output time instead (see B3 below).

### NEW — `api/tools/rag.py` (~80 lines)

```python
import asyncio
import logging
import os
from typing import Any

from api.retriever import get_or_build_retriever
from api.tools.tool import Tool, ToolResult, load_description, truncate_output

logger = logging.getLogger(__name__)

_DESCRIPTION_PATH = os.path.join(os.path.dirname(__file__), "rag.txt")
_DEFAULT_TOP_K = 10
_MAX_TOP_K = 20

class RagTool(Tool):
    name = "rag_search"
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Natural-language description of the code/concept to find."},
            "top_k": {"type": "integer", "description": f"Number of chunks to return (1-{_MAX_TOP_K}). Defaults to {_DEFAULT_TOP_K}.", "minimum": 1, "maximum": _MAX_TOP_K},
        },
        "required": ["query"],
    }

    def __init__(self, repo_path: str) -> None:
        self.repo_path = repo_path
        self.description = load_description(_DESCRIPTION_PATH)

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        query = (params.get("query") or "").strip()
        if not query:
            return ToolResult(title="rag_search", output="Error: query is required.", metadata={"error": "missing_query"})
        top_k = min(max(int(params.get("top_k") or _DEFAULT_TOP_K), 1), _MAX_TOP_K)

        try:
            retriever = await get_or_build_retriever(self.repo_path, repo_type="local")
            docs = await asyncio.to_thread(retriever.retrieve, query, top_k)
        except Exception as exc:
            logger.exception("rag_search failed")
            return ToolResult(title="rag_search", output=f"Error: {exc}", metadata={"error": "retrieval_failed"})

        return ToolResult(
            title=f"rag_search: {query}",
            output=truncate_output(_format(docs, self.repo_path)),
            metadata={"matches": len(docs), "query": query},
        )

def _format(docs, repo_path: str) -> str:
    if not docs:
        return "No matches."
    lines: list[str] = []
    for i, doc in enumerate(docs, 1):
        meta = getattr(doc, "meta_data", {}) or {}
        path = meta.get("file_path", "<unknown>")
        start = meta.get("start_line"); end = meta.get("end_line")
        loc = f"{path}:{start}-{end}" if start and end else path
        snippet = (doc.text or "").strip()
        lines.append(f"### {i}. {loc}\n{snippet}\n")
    return "\n".join(lines)
```

### NEW — `api/tools/rag.txt`

LLM-facing description. Tells the agent: this is **semantic** search (returns chunks similar in meaning, not just lexical match); prefer it for "how does X work?" / "where is Y handled?" questions; results are ranked code chunks with file paths and line ranges; follow up with `read` for full context. Mention it does NOT see uncommitted changes (cache is per-pickle).

### MODIFIED — `api/tools/__init__.py`

Register `"rag_search": RagTool`. Mirror existing imports/`__all__` block. Single-line change in `_TOOL_CLASSES` (`api/tools/__init__.py:46-54`).

### MODIFIED — `api/agent/config.py`

Add `"rag_search"` to `_ALL_TOOLS` (`api/agent/config.py:40`) and `_READ_ONLY_TOOLS` (`api/agent/config.py:41`). It is read-only — no shell, no writes. The `wiki`, `explore`, `deep-research`, and `wiki-planner`/`wiki-writer` agents all gain it automatically through their existing `_ALL_TOOLS`/`_READ_ONLY_TOOLS` references.

### MODIFIED — `api/agent/filtered_tools.py`

Add a post-filter case for `rag_search` that drops results whose `meta_data.file_path` matches `should_exclude_path(...)`. Without this, `rag_search` would leak chunks from paths the user excluded — a regression of the `wrap_tools_with_filters` contract. Implementation: extend `_POST_FILTER_TOOLS` semantics, or add a small `_post_filter_rag(result, filters)` branch parallel to `_filter_path_list_output`. Output already groups chunks under `### N. path:start-end` headers, so per-chunk filtering is straightforward via re-parsing those headers — or, cleaner, pass the structured matches through `metadata` and let the wrapper inspect them.

### NEW — `tests/unit/test_retriever.py`

Cases:
- `CodeRetriever.prepare` builds a retriever from a fixture repo and reuses an existing `.pkl` instead of rebuilding (assert by mocking `transform_documents_and_save_to_db` and verifying it is **not** called when the pickle exists).
- `get_or_build_retriever` returns the same instance across calls with the same key.
- `get_or_build_retriever` builds independent instances when `embedder_type` differs (monkeypatch `get_embedder_type`).
- LRU eviction: build 5 retrievers with `_CACHE_MAX=4`, assert oldest is evicted.
- Concurrent build with `asyncio.gather` invokes `prepare` exactly once.

### NEW — `tests/unit/test_rag_tool.py`

Cases:
- Empty query returns `error == "missing_query"`.
- `top_k` clamped to `[1, _MAX_TOP_K]`.
- Output format includes `path:start-end` headers and snippet.
- Retriever exception surfaces as `error == "retrieval_failed"` (not raised).
- Tool registers via `get_tool("rag_search", repo_path)` from `api.tools`.

### NEW — `tests/api/test_agent_chat_rag_tool.py`

Smoke test: drive `_run_agent_chat` with a stubbed `run_agent_loop` that emits a `tool_call_start` for `rag_search` and verify the wrapped tool is in the dict passed to the loop. Optional: assert the `FilteredToolWrapper` post-filter drops excluded paths from a mock `rag_search` result.

## Critical files to read before implementation

- `api/rag.py:171-261, 696-797` — RAG init/prepare/call to mirror exactly the retrieval half.
- `api/data_pipeline.py:712-953` — `DatabaseManager` API surface and the disk-cache logic at lines 890-917.
- `api/tools/embedder.py` — shared `get_embedder()` factory.
- `api/tools/tool.py` — `Tool` ABC, `ToolResult`, `truncate_output`, `load_description`.
- `api/tools/read.py:1-90` — concrete tool example (description loading, error metadata).
- `api/tools/__init__.py:46-54` — registry shape.
- `api/agent/config.py:40-41, 111-148` — tool name sets and the `_TOOL_CLASSES[name](repo_path)` factory contract.
- `api/agent/filtered_tools.py:42-50, 88-145` — wrapper pattern; the only place that knows about post-filtering.
- `api/agent/chat_handler.py:84-124` — where tools are handed to `run_agent_loop`.

## Implementation steps

1. **`api/retriever.py`**. Implement `CodeRetriever` and `get_or_build_retriever`. Unit-test against a small fixture repo. Confirm: cold call builds the `.pkl`, second cold call (process restart) loads from `.pkl` without re-embedding, in-process second call returns the cached instance.
2. **`api/tools/rag.py` + `api/tools/rag.txt`**. Implement `RagTool`. Unit-test in isolation.
3. **Register** in `api/tools/__init__.py` and `api/agent/config.py`. Confirm `get_all_tools(repo_path)` includes `rag_search` and that all five agents enumerate it under `allowed_tools`.
4. **`FilteredToolWrapper`** post-filter for `rag_search`. Unit-test path exclusion.
5. **End-to-end smoke**: start the dev server, hit `/chat/agent-stream` with a question that benefits from semantic search ("where is repository cloning handled?"), confirm the agent issues a `rag_search` tool call and the `tool_call_end` carries match results.
6. **Update handbooks index**: `handbooks/index.md`, `handbooks/index.json`, `handbooks/plans/index.md`, `handbooks/plans/index.json` to list PLAN-008.

Steps 1-4 are independent of `api/rag.py`. The legacy RAG class is **untouched**; the wiki and existing chat paths cannot regress from this change.

## Verification

| Stage | Command |
|---|---|
| Retriever unit tests (step 1) | `pytest tests/unit/test_retriever.py -q --tb=short 2>&1 \| tail -50` |
| Tool unit tests (step 2) | `pytest tests/unit/test_rag_tool.py -q --tb=short 2>&1 \| tail -50` |
| Filter regression (step 4) | `pytest tests/api/ -q --tb=short -k filter 2>&1 \| tail -50` |
| Full backend regression | `pytest tests/ -q --tb=short 2>&1 \| tail -80` |
| Manual cache reuse | Two consecutive `rag_search` calls in one process — second must complete in well under one second. After process restart, third call must skip embedding (logs `Loading existing database...`). |
| Manual end-to-end | `curl -N -X POST http://localhost:8001/chat/agent-stream -H 'Content-Type: application/json' -d '{"repo_url":"https://github.com/AsyncFuncAI/deepwiki-open","agent_name":"explore","messages":[{"role":"user","content":"Where is repository cloning handled?"}]}'` — observe a `tool_call_start` with `tool_name:"rag_search"`. |

## Out of scope (explicit)

- **Refactoring `api/rag.py` to compose `CodeRetriever`.** Worth doing later; does not belong in this plan.
- **Multi-hop retrieval**. The dormant multi-hop methods in `api/rag.py:53-600` stay dormant. v1 of the tool uses vanilla FAISS — same as the current `RAG.call()`. Multi-hop can be a follow-up flag on `CodeRetriever.retrieve(...)` once we know the agent benefits from it.
- **Threading request-time filters into indexing**. v1 ignores `excluded_dirs/files/...` for the build-time scope and relies on the post-filter wrapper. Per-request indexing would invalidate the shared `.pkl` cache and is not justified for this plan.
- **Cross-process / Redis-backed cache**. Module-level dict is enough; the FastAPI server is single-process by default and the `.pkl` already gives cross-process persistence.
- **Migration to OpenAI Agents SDK**.

## Sub-tasks

1. Implement `api/retriever.py` (`CodeRetriever`, module-level cache with LRU + per-key lock).
2. Implement `api/tools/rag.py` and `api/tools/rag.txt`.
3. Register `rag_search` in `api/tools/__init__.py` and add to `_ALL_TOOLS` / `_READ_ONLY_TOOLS` in `api/agent/config.py`.
4. Extend `api/agent/filtered_tools.py` to post-filter `rag_search` results by path.
5. Add `tests/unit/test_retriever.py`, `tests/unit/test_rag_tool.py`, and `tests/api/test_agent_chat_rag_tool.py`.
6. Update `handbooks/plans/index.md`, `handbooks/plans/index.json`, and `handbooks/index.md` to list PLAN-008.
