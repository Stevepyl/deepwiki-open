"""
Stream event types for the Unified Provider.

These events are yielded by UnifiedProvider.stream_chat() and consumed by the
Agent Loop (subtask 4). They form the clean boundary between "how each provider
streams internally" and "what the loop sees."

5 event types, discriminated by the `type` field — the same pattern established
in message.py's MessagePart union.

Design notes:
- All events are frozen Pydantic BaseModel (immutable, JSON-serializable)
- ToolCallStart carries *complete* tool_args (a parsed dict, not a raw JSON string)
  — the provider accumulates streaming deltas internally and only emits this event
  after the full argument JSON has been assembled
- ToolCallDelta exists for forward compatibility (e.g., future UI "typing" effects)
  but the current provider implementation does not emit it — callers may safely
  ignore it for now
- FinishEvent carries an optional usage dict for token accounting by the loop
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

StreamEventType = Literal[
    "text_delta", "tool_call_start", "tool_call_delta", "tool_call_end", "finish", "error",
    "wiki_structure_ready", "wiki_page_done", "wiki_structure_error", "wiki_page_error",
]


# ---------------------------------------------------------------------------
# Event classes
# ---------------------------------------------------------------------------


class TextDelta(BaseModel):
    """Incremental text token produced by the LLM."""

    model_config = ConfigDict(frozen=True)

    type: Literal["text_delta"] = "text_delta"
    content: str


class ToolCallStart(BaseModel):
    """
    A complete tool invocation requested by the LLM.

    Emitted once per tool call *after* the provider has accumulated all
    argument fragments from the streaming response. The agent loop converts
    this into a ToolCallPart (pending state) and schedules execution.

    tool_args is a parsed dict, never a raw JSON string.
    """

    model_config = ConfigDict(frozen=True)

    type: Literal["tool_call_start"] = "tool_call_start"
    tool_call_id: str
    tool_name: str
    tool_args: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        """
        Replace tool_args with a defensive copy to prevent silent mutation.

        Pydantic's frozen=True blocks field reassignment but not in-place
        mutation of a dict that was passed in by the caller. Mirrors the
        same pattern used in message.py:ToolCallPart.model_post_init().
        """
        object.__setattr__(self, "tool_args", dict(self.tool_args))


class ToolCallDelta(BaseModel):
    """
    Incremental JSON argument fragment for a tool call in progress.

    Not currently emitted by any provider adapter — retained for future use
    by UI components that want to show arguments being "typed out" during
    streaming. Consumers may safely ignore this event type.
    """

    model_config = ConfigDict(frozen=True)

    type: Literal["tool_call_delta"] = "tool_call_delta"
    tool_call_id: str
    args_delta: str  # raw JSON string fragment


class ToolCallEnd(BaseModel):
    """
    Signals that a tool has finished executing.

    Emitted by the agent loop (not the provider) after tool.execute() returns.
    The frontend uses this to transition tool call cards from the "running"
    state to "completed" or "error".

    result_summary is a truncated preview (first 200 chars) of the tool output,
    suitable for display in collapsed tool cards. The full result is fed back
    to the LLM in a tool message but is not sent to the frontend separately.
    """

    model_config = ConfigDict(frozen=True)

    type: Literal["tool_call_end"] = "tool_call_end"
    tool_call_id: str
    tool_name: str
    result_summary: str
    is_error: bool = False
    duration_ms: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        """Defensive copy of mutable metadata dict."""
        object.__setattr__(self, "metadata", dict(self.metadata))


class FinishEvent(BaseModel):
    """
    Signals that the provider has finished streaming.

    finish_reason values:
        "stop"       -- LLM produced a normal text response with no tool calls
        "tool_calls" -- LLM requested one or more tool calls (ToolCallStart
                        events were emitted before this)
        "error"      -- The stream ended due to an error (ErrorEvent was
                        emitted before this)

    usage is present when the provider reports token counts. Not all
    providers expose this information in streaming mode.
    """

    model_config = ConfigDict(frozen=True)

    type: Literal["finish"] = "finish"
    finish_reason: str
    usage: Optional[dict[str, int]] = None  # e.g. {"prompt_tokens": N, "completion_tokens": N}


class ErrorEvent(BaseModel):
    """
    A provider-level error that terminated the stream.

    The agent loop should treat this as a failed LLM call and handle it
    according to the loop's error policy (retry, propagate, etc.).

    code values (non-exhaustive):
        "provider_error"     -- generic API / SDK error
        "api_key_missing"    -- no API key configured
        "unsupported_provider" -- provider name not recognised
        "import_error"       -- required package not installed
    """

    model_config = ConfigDict(frozen=True)

    type: Literal["error"] = "error"
    error: str
    code: Optional[str] = None


# ---------------------------------------------------------------------------
# Wiki generation events (subtask 12)
# ---------------------------------------------------------------------------


class WikiStructureReady(BaseModel):
    """
    Signals Phase 1 (wiki-planner) completed successfully.

    `structure` is a WikiStructureModel.model_dump() dict, carried as
    dict[str, Any] to avoid a circular import with api.api where
    WikiStructureModel is defined. The frontend validates the shape.
    """

    model_config = ConfigDict(frozen=True)

    type: Literal["wiki_structure_ready"] = "wiki_structure_ready"
    structure: dict[str, Any]


class WikiPageDone(BaseModel):
    """Signals that a single wiki page has been fully generated by the writer agent."""

    model_config = ConfigDict(frozen=True)

    type: Literal["wiki_page_done"] = "wiki_page_done"
    page_id: str
    page_title: str
    page_index: int
    total_pages: int
    content: str


class WikiStructureError(BaseModel):
    """
    Fatal error during Phase 1 (wiki-planner).

    After this event the WebSocket is closed. The frontend should offer
    the user a retry or fall back to the legacy /ws/chat flow.

    code values:
        "json_parse_error"        -- planner output could not be decoded
        "schema_validation_error" -- JSON decoded but Pydantic validation failed
        "planner_timeout"         -- planner exceeded its 5-minute ceiling
        "clone_failed"            -- git clone / download_repo raised an error
        "internal_error"          -- unexpected server-side exception
    """

    model_config = ConfigDict(frozen=True)

    type: Literal["wiki_structure_error"] = "wiki_structure_error"
    code: Literal[
        "json_parse_error",
        "schema_validation_error",
        "planner_timeout",
        "clone_failed",
        "internal_error",
    ]
    message: str


class WikiPageError(BaseModel):
    """
    Non-fatal error during Phase 2 for a single page.

    The session continues with the next page after this event is sent.
    """

    model_config = ConfigDict(frozen=True)

    type: Literal["wiki_page_error"] = "wiki_page_error"
    page_id: str
    page_index: int
    code: Literal["writer_timeout", "writer_failed"]
    message: str


# ---------------------------------------------------------------------------
# Discriminated union
# ---------------------------------------------------------------------------

# Pydantic reads the `type` field first for O(1) dispatch — same approach
# as MessagePart in message.py.
StreamEvent = Annotated[
    Union[
        TextDelta,
        ToolCallStart,
        ToolCallDelta,
        ToolCallEnd,
        FinishEvent,
        ErrorEvent,
        WikiStructureReady,
        WikiPageDone,
        WikiStructureError,
        WikiPageError,
    ],
    Field(discriminator="type"),
]
