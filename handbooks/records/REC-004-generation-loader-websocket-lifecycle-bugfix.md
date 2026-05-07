---
number: REC-004
name: Generation Loader WebSocket Lifecycle Bugfix
description: Records the fix for generation WebSockets being opened and immediately closed by the frontend loading screen lifecycle.
update_at: 2026-05-07
category: implementation-record
language: en
status: recorded
---

# Generation Loader WebSocket Lifecycle Bugfix

## 1. Problem

After submitting a GitHub repository URL from the home page, the backend could log:

```text
INFO:     127.0.0.1:53304 - "WebSocket /ws/chat" [accepted]
INFO:     connection closed
2026-05-07 23:30:51,216 - INFO - api.websocket_wiki - websocket_wiki.py:908 - WebSocket disconnected
```

The disconnect happened before `api.websocket_wiki.handle_websocket_chat()` logged request size or retriever setup, which means the browser closed the socket before the request payload was processed.

## 2. Root Cause

`src/components/generation/GenerationLoader.tsx` started wiki generation inside a React effect and used a `started` ref to suppress repeat runs.

That was unsafe because the same effect cleanup closes `socketRef.current`. In development lifecycle checks, and also on any early dependency refresh, React can run the cleanup and then run the effect again. One concrete early refresh is `SettingsProvider` hydrating saved provider/model/token values from localStorage after mount.

The cleanup closes the newly created `/ws/chat` socket, but `started.current` remains `true`, so the next effect run returns without starting generation again.

The backend symptom is therefore a client-side lifecycle close, not a backend generation failure.

## 3. Fix

Frontend change in `src/components/generation/GenerationLoader.tsx`:

- Remove the persistent `started` guard.
- Wait until `SettingsProvider` has hydrated local settings before generation can start.
- Schedule the generation start with a zero-delay timer.
- Clear that timer during cleanup so immediate development-mode cleanup can cancel the pending start before any WebSocket is opened.
- Keep cleanup closing the active socket for real navigation or cancel actions.

Frontend change in `src/contexts/SettingsContext.tsx`:

- Expose the existing `hydrated` state in `SettingsContextValue` so generation can wait for stable settings.

This keeps one active generation on the real mounted loader while avoiding the accepted-then-disconnected socket from the throwaway effect pass.

## 4. Verification

Commands to run after the fix:

```bash
bun run lint
git diff --check
```

Manual check:

1. Start the backend with `source api/.venv/bin/activate && uv run -m api.main`.
2. Start the frontend with `bun run dev`.
3. Submit a GitHub repo URL from `/`.
4. Confirm `/ws/chat` receives a request payload and proceeds to retriever setup instead of immediately logging only `WebSocket disconnected`.

## 5. Rollback Notes

Restoring the `started` ref reintroduces the bug because cleanup can close the socket while the next effect pass is blocked from restarting generation.
