---
number: PLAN-INDEX
name: Plans Index
description: Tracks implementation status for plans and their sub-tasks under handbooks/plans.
update_at: 2026-05-05
category: index
language: en
audience: developers-and-agents
---

# Plans Index

Read this index before opening individual plans when you need current implementation status.

## Status Legend

| Status | Meaning |
|---|---|
| `proposed` | Plan is documented, but implementation has not started or is not visible in current code. |
| `in-progress` | Implementation work has started, but the plan is not fully delivered. |
| `implemented` | Implementation appears complete against the plan's stated scope. |
| `blocked` | Implementation is paused on a known dependency or unresolved decision. |

## Plans

| Plan | Plan Status | Implementation Status | Evidence | Sub-task Status |
|---|---|---|---|---|
| [PLAN-001 - AST Code Splitter Improvement Plan](PLAN-001-ast-code-splitter.md) | `proposed` | `not-started` | Plan frontmatter is `status: proposed`; `api/data_pipeline.py` still imports `TextSplitter` and `prepare_data_pipeline` still constructs `TextSplitter(**configs["text_splitter"])`. | No explicit sub-tasks recorded. |

## Sub-tasks

| Plan | Sub-task | Status | Notes |
|---|---|---|---|
| PLAN-001 | None recorded | N/A | The plan is organized by solution options, implementation references, benefits, and risks rather than a task checklist. |
