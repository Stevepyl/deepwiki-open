# Handoff Guide

Use this template to hand off work between Claude Code sessions. Save it to memory when starting a new session if the previous session's work is incomplete.

## At End of Session (if work incomplete)

Copy-paste this and save to `~/.claude/projects/-Users-steveliu-School-06-proj-deepwiki-open/memory/MEMORY.md`:

```
## [Date] Handoff: [Feature/Bug Name]

**Status:** [In progress / Blocked / Needs review]

**What was done:**
- [Specific file changes made]
- [Tests run: pass/fail]
- [Any commits created: yes/no, with hashes if yes]

**What's next:**
- [Exact next step: e.g., "Implement error handling in rag.py", "Debug WebSocket connection"]
- [Any blockers or uncertainties]

**Files modified:**
- [path/to/file.py: description of change]
- [path/to/file.tsx: description of change]

**Context to remember:**
- [Architecture decisions made]
- [Why specific approach was chosen over alternatives]
- [Performance/reliability tradeoffs discovered]
```

## At Start of New Session

Before asking Claude to resume:
1. Check `/Users/steveliu/.claude/projects/-Users-steveliu-School-06-proj-deepwiki-open/memory/MEMORY.md`
2. If there's a handoff note, paste it into the chat
3. Claude will resume from the exact state: files modified, tests passing/failing, next steps clear

## Architecture Memory (Persistent)

Keep in `memory/MEMORY.md` across sessions:
- **Data pipeline**: Chunking strategy (350-word chunks, 100-word overlap), FAISS index structure
- **RAG**: Vector search → context assembly → prompt templates → streaming
- **WebSocket**: Wiki generation flow, real-time render (Mermaid → Markdown → WikiTreeView)
- **LLM providers**: How abstraction layer works, which provider is active
- **Known issues**: Performance bottlenecks (e.g., Ollama latency), reliability tradeoffs

---

**Goal:** No context loss across sessions; design decisions and progress preserved.
