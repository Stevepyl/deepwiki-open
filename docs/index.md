---
number: DOC-INDEX
name: Docs Index
description: Index of developer and agent documentation under docs.
update_at: 2026-05-04
category: index
language: en
audience: developers-and-agents
---

# Docs Index

Read this index first when looking for project documentation.

## Architecture

- [rag-system-overview](architecture/rag-system-overview.md) - Explains the end-to-end RAG architecture, data flow, provider boundaries, and operational behavior.
- [rag-system-design](architecture/rag-system-design.md) - Documents the RAG pipeline, Document lifecycle, embedding cache behavior, and retrieval implementation details.
- [agent-design](architecture/agent-design.md) - Describes how DeepWiki implements agent-like behavior through its RAG components and request flow.
- [database-design](architecture/database-design.md) - Explains DeepWiki's FAISS-backed local storage, LocalDB tables, and cache files.
- [adalflow-integration](architecture/adalflow-integration.md) - Placeholder for documenting how DeepWiki integrates AdalFlow components.

## API

- [frontend-backend-apis](api/frontend-backend-apis.md) - Maps browser-visible frontend routes to FastAPI backend endpoints, payloads, streaming behavior, and caveats.
- [frontend-backend-apis-cn](api/frontend-backend-apis-cn.md) - Chinese translation of the frontend-backend API boundary reference.

## Analysis

- [faiss-capacity-analysis](analysis/faiss-capacity-analysis.md) - Analyzes repository size limits and memory constraints for DeepWiki's FAISS retrieval backend.
