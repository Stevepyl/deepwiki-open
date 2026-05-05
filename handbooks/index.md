---
number: HBK-INDEX
name: Handbooks Index
description: Index of development plans, decisions, records, risks, backlog notes, and handbook guides under handbooks.
update_at: 2026-05-05
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

## Implementation Records

- [REC-001-agent-wiki-generation-backend](records/REC-001-agent-wiki-generation-backend.md) - Records the Path B backend implementation for two-stage agent-driven wiki generation and its WebSocket protocol.
- [REC-002-agent-cli-demo](records/REC-002-agent-cli-demo.md) - Records the local CLI demo used to smoke-test multi-turn agent Q&A and two-stage wiki generation.

## Risks

- [RISK-001-agent-websocket-input-validation](risks/RISK-001-agent-websocket-input-validation.md) - Records the role-part invariant gap at the external agent WebSocket boundary and the recommended validation strategy.
- [RISK-002-bash-agent-sandbox-gap](risks/RISK-002-bash-agent-sandbox-gap.md) - Records the shell execution risks introduced by the wiki-writer bash tool and the staged mitigation roadmap.
- [RISK-003-repo-cache-key-collision](risks/RISK-003-repo-cache-key-collision.md) - Records the cache key collision risk caused by owner-repo naming across hosts, forks, and private repositories.
- [RISK-004-rag-system-issues](risks/RISK-004-rag-system-issues.md) - Records observed RAG system risks around skipped files, embedding cache provenance, validation gaps, and retrieval quality.

## Backlog

- [BACKLOG-001-deepwiki-improvement-directions](backlog/BACKLOG-001-deepwiki-improvement-directions.md) - Captures prioritized improvement opportunities across RAG quality, ingestion performance, architecture, deep research, and observability.

## Guides

- [HBK-001-rag-to-wiki-call-chain](guides/HBK-001-rag-to-wiki-call-chain.md) - Explains how repository code becomes structured wiki output through request handling, retrieval, prompting, and rendering.
