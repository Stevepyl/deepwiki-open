---
number: HBK-INDEX
name: Handbooks Index
description: Index of development plans, decisions, records, risks, backlog notes, and handbook guides under handbooks.
update_at: 2026-05-07
category: index
language: en
audience: developers-and-agents
---

# Handbooks Index

Read this index first when looking for plans, decisions, implementation records, risks, backlog notes, or handbook guides.

## Architecture Decisions

- [ADR-001-remove-wiki-type-toggle](adr/ADR-001-remove-wiki-type-toggle.md) - Records the proposed decision to remove the concise/comprehensive wiki generation mode toggle and default to comprehensive output.

## Plans

- [PLAN-INDEX](plans/index.md) - Tracks implementation status for plans and their sub-tasks under handbooks/plans.
- [PLAN-001-ast-code-splitter](plans/PLAN-001-ast-code-splitter.md) - Proposes replacing word-based splitting with an AST-aware code splitter while preserving the existing pipeline contract.
- [PLAN-002-frontend-refinement-overview](plans/PLAN-002-frontend-refinement-overview.md) - Cross-cutting plan that ties together the four sub-plans rebranding the frontend while preserving legacy chat, agent-chat, and wiki contracts.
- [PLAN-003-foundation](plans/PLAN-003-foundation.md) - Replaces the dark-mode token set with Paper and Ink tokens, rebuilds the shared app shell, and preserves shared chat/agent-chat connectors.
- [PLAN-004-welcome-and-projects](plans/PLAN-004-welcome-and-projects.md) - Rewrites the home route and the projects directory to match the OpsWiki welcome page and project library prototypes.
- [PLAN-005-chat-view](plans/PLAN-005-chat-view.md) - Rewrites the Ask route into a chat-stream UI using the structured agent-chat connectors from PLAN-007.
- [PLAN-006-wiki-family](plans/PLAN-006-wiki-family.md) - Implemented the wiki reading view, workshop view, slides presenter, and loading screen while keeping legacy `/ws/chat` generation separate from agent chat.
- [PLAN-007-agent-chat-api](plans/PLAN-007-agent-chat-api.md) - Exposes the existing agent infrastructure through WebSocket and HTTP chat APIs with frontend connector utilities.
- [PLAN-008-rag-tool](plans/PLAN-008-rag-tool.md) - Extracts RAG retrieval into a small CodeRetriever and exposes it to the agent as a rag_search tool, reusing the existing pickle cache.

## Implementation Records

- [REC-001-agent-wiki-generation-backend](records/REC-001-agent-wiki-generation-backend.md) - Records the Path B backend implementation for two-stage agent-driven wiki generation and its WebSocket protocol.
- [REC-002-agent-cli-demo](records/REC-002-agent-cli-demo.md) - Records the local CLI demo used to smoke-test multi-turn agent Q&A and two-stage wiki generation.
- [REC-003-vector-db-cache-and-generation-retry-bugfix](records/REC-003-vector-db-cache-and-generation-retry-bugfix.md) - Records the fix for redundant embedding rebuilds from legacy Python metadata checks and frontend HTTP retry timeouts during wiki generation.
- [REC-004-generation-loader-websocket-lifecycle-bugfix](records/REC-004-generation-loader-websocket-lifecycle-bugfix.md) - Records the fix for generation WebSockets being opened and immediately closed by the frontend loading screen lifecycle.

## Risks

- [RISK-001-agent-websocket-input-validation](risks/RISK-001-agent-websocket-input-validation.md) - Records the role-part invariant gap at the external agent WebSocket boundary and the recommended validation strategy.
- [RISK-002-bash-agent-sandbox-gap](risks/RISK-002-bash-agent-sandbox-gap.md) - Records the shell execution risks introduced by the wiki-writer bash tool and the staged mitigation roadmap.
- [RISK-003-repo-cache-key-collision](risks/RISK-003-repo-cache-key-collision.md) - Records the cache key collision risk caused by owner-repo naming across hosts, forks, and private repositories.
- [RISK-004-rag-system-issues](risks/RISK-004-rag-system-issues.md) - Records observed RAG system risks around skipped files, embedding cache provenance, validation gaps, and retrieval quality.

## Backlog

- [BACKLOG-001-deepwiki-improvement-directions](backlog/BACKLOG-001-deepwiki-improvement-directions.md) - Captures prioritized improvement opportunities across RAG quality, ingestion performance, architecture, deep research, and observability.

## Guides

- [HBK-001-rag-to-wiki-call-chain](guides/HBK-001-rag-to-wiki-call-chain.md) - Explains how repository code becomes structured wiki output through request handling, retrieval, prompting, and rendering.
