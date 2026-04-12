"""
Structured message model for the Agent Loop.

Defines the message types used within the agent loop, replacing the simple
ChatMessage(role, content) model with a rich part-based structure that
supports text, tool calls, tool results, and errors in a single message.

Design mirrors OpenCode's message-v2.ts but simplified for Python/Pydantic:
- 4 part types instead of 12 (no reasoning, file, patch, snapshot parts)
- Pydantic BaseModel instead of Zod for JSON validation and serialization
- Immutable state transitions via model_copy() instead of Effect mutations
- float timestamps instead of datetime (simpler JSON serialization)

Usage:
    # Creating messages
    user_msg = AgentMessage.user("find the main function")
    sys_msg = AgentMessage.system("You are a helpful code assistant.")

    # Building assistant messages with tool calls
    tc = ToolCallPart(tool_name="grep", tool_args={"pattern": "def main"})
    asst_msg = AgentMessage.assistant_tool_calls([tc], text="Let me search...")

    # State machine transitions (immutable)
    running = tc.start()
    completed = running.complete(ToolResult(title="grep", output="...", metadata={}))

    # Converting to OpenAI API format
    openai_msgs = messages_to_openai_format([user_msg, asst_msg])
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Annotated, Any, Literal, Optional, Union, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator

from api.tools.tool import ToolResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

PartType = Literal["text", "tool_call", "tool_result", "error"]
ToolCallStatus = Literal["pending", "running", "completed", "error"]
Role = Literal["user", "assistant", "system", "tool"]


# ---------------------------------------------------------------------------
# Part types
# ---------------------------------------------------------------------------


class TextPart(BaseModel):
    """Plain text produced by the LLM."""

    model_config = ConfigDict(frozen=True)

    type: Literal["text"] = "text"
    content: str


class ToolCallPart(BaseModel):
    """
    A tool invocation requested by the LLM, with full lifecycle tracking.

    State machine (immutable transitions):
        pending  ->  running  ->  completed
                              ->  error

    Use .start(), .complete(), and .fail() to advance the state.
    Each method returns a new instance; the original is never mutated.
    """

    model_config = ConfigDict(frozen=True)

    type: Literal["tool_call"] = "tool_call"
    tool_call_id: str = Field(
        default_factory=lambda: f"call_{uuid.uuid4().hex[:24]}"
    )
    tool_name: str
    tool_args: dict[str, Any] = Field(default_factory=dict)
    state: ToolCallStatus = "pending"
    result: Optional[str] = None
    error: Optional[str] = None
    title: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    def model_post_init(self, __context: Any) -> None:
        """
        Ensure initial construction produces independent dict copies.

        Pydantic's frozen=True prevents field reassignment but not mutation of
        dict contents. On direct construction ToolCallPart(tool_args={...}),
        the caller's dict could be mutated later. We replace both dict fields
        with fresh copies via object.__setattr__ (bypassing frozen).

        Note: model_copy() does NOT call model_post_init, so state transition
        methods (start, complete, fail) must explicitly include dict copies in
        their model_copy(update={...}) calls.
        """
        object.__setattr__(self, "tool_args", dict(self.tool_args))
        object.__setattr__(self, "metadata", dict(self.metadata))

    # --- State machine transitions ---

    def start(self) -> ToolCallPart:
        """Transition pending -> running. Returns new instance."""
        if self.state != "pending":
            raise ValueError(
                f"start() called on tool call in state '{self.state}' "
                f"(tool_call_id={self.tool_call_id}). Expected 'pending'."
            )
        # model_copy() does not call model_post_init, so we must include
        # explicit dict copies here to maintain per-instance independence.
        return self.model_copy(update={
            "state": "running",
            "started_at": time.time(),
            "tool_args": dict(self.tool_args),
            "metadata": dict(self.metadata),
        })

    def complete(self, tool_result: ToolResult) -> ToolCallPart:
        """Transition running -> completed. Returns new instance."""
        if self.state != "running":
            raise ValueError(
                f"complete() called on tool call in state '{self.state}' "
                f"(tool_call_id={self.tool_call_id}). Expected 'running'."
            )
        return self.model_copy(update={
            "state": "completed",
            "result": tool_result.output,
            "title": tool_result.title,
            "tool_args": dict(self.tool_args),
            "metadata": dict(tool_result.metadata),
            "completed_at": time.time(),
        })

    def fail(self, error_message: str) -> ToolCallPart:
        """Transition running -> error. Returns new instance."""
        if self.state != "running":
            raise ValueError(
                f"fail() called on tool call in state '{self.state}' "
                f"(tool_call_id={self.tool_call_id}). Expected 'running'."
            )
        return self.model_copy(update={
            "state": "error",
            "error": error_message,
            "tool_args": dict(self.tool_args),
            "metadata": dict(self.metadata),
            "completed_at": time.time(),
        })


class ToolResultPart(BaseModel):
    """
    Tool execution result for the LLM conversation format.

    Lives in role='tool' messages, providing the output back to the LLM.
    Distinct from ToolCallPart.result (which is for UI display convenience):
    this part is what gets serialized into the 'tool' role message sent to
    the provider.
    """

    model_config = ConfigDict(frozen=True)

    type: Literal["tool_result"] = "tool_result"
    tool_call_id: str
    tool_name: str
    content: str
    is_error: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_tool_result(
        cls,
        tool_call_id: str,
        tool_name: str,
        tool_result: ToolResult,
        is_error: bool = False,
    ) -> ToolResultPart:
        """Bridge from the ToolResult dataclass (api/tools/tool.py) into this model."""
        return cls(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            content=tool_result.output,
            is_error=is_error,
            # dict() creates a shallow copy to avoid sharing mutable state
            metadata=dict(tool_result.metadata),
        )


class ErrorPart(BaseModel):
    """
    Agent-level error (not a tool execution error).

    Used for errors such as max_steps_exceeded, doom_loop detection,
    or provider failures that the frontend should display distinctly.
    """

    model_config = ConfigDict(frozen=True)

    type: Literal["error"] = "error"
    content: str
    code: Optional[str] = None


# Discriminated union — Pydantic reads the 'type' field first for O(1) dispatch.
MessagePart = Annotated[
    Union[TextPart, ToolCallPart, ToolResultPart, ErrorPart],
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# AgentMessage
# ---------------------------------------------------------------------------


class AgentMessage(BaseModel):
    """
    A single message in the agent loop conversation.

    Unlike the legacy ChatMessage model (which stores content as a plain
    string), AgentMessage uses a parts-based structure that can represent
    interleaved text and tool calls within a single LLM response.

    Roles:
        user      -- human input or injected system messages during the loop
        assistant -- LLM response (may contain TextParts + ToolCallParts)
        system    -- system prompt prepended to the conversation
        tool      -- tool execution results (ToolResultParts), sent back to LLM

    Immutability:
        The model is frozen. Use with_updated_part() and with_appended_part()
        to build modified copies.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: Role
    parts: tuple[MessagePart, ...] = ()
    created_at: float = Field(default_factory=time.time)
    model: Optional[str] = None
    provider: Optional[str] = None

    @field_validator("parts", mode="before")
    @classmethod
    def _coerce_parts_to_tuple(cls, value: Any) -> Any:
        """Accept list[...] from JSON deserialization, convert to tuple."""
        if isinstance(value, list):
            return tuple(value)
        return value

    # --- Convenience constructors ---

    @classmethod
    def user(cls, content: str) -> AgentMessage:
        """Create a user message with a single TextPart."""
        return cls(role="user", parts=(TextPart(content=content),))

    @classmethod
    def system(cls, content: str) -> AgentMessage:
        """Create a system message with a single TextPart."""
        return cls(role="system", parts=(TextPart(content=content),))

    @classmethod
    def assistant_text(
        cls,
        content: str,
        model: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> AgentMessage:
        """Create an assistant message containing only text."""
        return cls(
            role="assistant",
            parts=(TextPart(content=content),),
            model=model,
            provider=provider,
        )

    @classmethod
    def assistant_tool_calls(
        cls,
        tool_calls: list[ToolCallPart],
        text: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> AgentMessage:
        """
        Create an assistant message with tool calls and optional leading text.

        Matches the OpenAI pattern where an assistant response can contain
        both content text and a tool_calls array simultaneously.
        """
        parts: list[MessagePart] = []
        if text is not None:
            parts.append(TextPart(content=text))
        parts.extend(tool_calls)
        return cls(
            role="assistant",
            parts=tuple(parts),
            model=model,
            provider=provider,
        )

    @classmethod
    def tool_result(
        cls,
        tool_call_id: str,
        tool_name: str,
        result: ToolResult,
        is_error: bool = False,
    ) -> AgentMessage:
        """Create a role='tool' message from a ToolResult dataclass."""
        part = ToolResultPart.from_tool_result(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            tool_result=result,
            is_error=is_error,
        )
        return cls(role="tool", parts=(part,))

    @classmethod
    def from_chat_messages(cls, messages: list[dict[str, Any]]) -> list[AgentMessage]:
        """
        Convert legacy ChatMessage dicts to AgentMessage instances.

        Bridges the existing ChatMessage format (role + content string) used
        by websocket_wiki.py and simple_chat.py into the new structured model.
        The returned list preserves message order.

        Note: role='tool' messages cannot be converted from the legacy format
        (they require a tool_call_id that plain text messages do not carry)
        and are skipped with a warning.
        """
        valid_roles: frozenset[str] = frozenset({"user", "assistant", "system"})
        result: list[AgentMessage] = []
        for raw in messages:
            role = raw.get("role", "")
            content = str(raw.get("content", ""))
            if role == "tool":
                logger.warning(
                    "from_chat_messages: 'tool' role messages cannot be "
                    "converted from legacy format (no tool_call_id). Skipping."
                )
                continue
            if role not in valid_roles:
                logger.warning(
                    "from_chat_messages: skipping message with unknown role '%s'",
                    role,
                )
                continue
            result.append(cls(role=cast(Role, role), parts=(TextPart(content=content),)))
        return result

    # --- Part accessor properties ---

    @property
    def text(self) -> Optional[str]:
        """
        Concatenated content of all TextParts, or None if no text parts exist.

        Returns None (not empty string) to distinguish "no text" from
        "empty text", which matters when building OpenAI tool_calls messages
        where content=None is valid but content="" may confuse some models.
        """
        text_parts = [p.content for p in self.parts if isinstance(p, TextPart)]
        if not text_parts:
            return None
        return "\n".join(text_parts)

    @property
    def tool_calls(self) -> tuple[ToolCallPart, ...]:
        """All ToolCallParts in this message, in order."""
        return tuple(p for p in self.parts if isinstance(p, ToolCallPart))

    @property
    def tool_results(self) -> tuple[ToolResultPart, ...]:
        """All ToolResultParts in this message, in order."""
        return tuple(p for p in self.parts if isinstance(p, ToolResultPart))

    @property
    def has_tool_calls(self) -> bool:
        """True if this message contains at least one ToolCallPart."""
        return any(isinstance(p, ToolCallPart) for p in self.parts)

    # --- Immutable update helpers ---

    def with_updated_part(self, index: int, new_part: MessagePart) -> AgentMessage:
        """
        Return a new AgentMessage with the part at index replaced.

        Used by the agent loop to advance a ToolCallPart's state while
        keeping the message structure intact.
        """
        if index < 0 or index >= len(self.parts):
            raise IndexError(
                f"Part index {index} out of range for message with "
                f"{len(self.parts)} parts."
            )
        new_parts = (
            *self.parts[:index],
            new_part,
            *self.parts[index + 1:],
        )
        return self.model_copy(update={"parts": new_parts})

    def with_appended_part(self, part: MessagePart) -> AgentMessage:
        """Return a new AgentMessage with the given part added at the end."""
        return self.model_copy(update={"parts": self.parts + (part,)})


# ---------------------------------------------------------------------------
# OpenAI API format conversion
# ---------------------------------------------------------------------------


def message_to_openai_format(
    message: AgentMessage,
) -> dict[str, Any] | list[dict[str, Any]]:
    """
    Convert a single AgentMessage to OpenAI Chat Completions API format.

    Returns a dict for user/system/assistant messages, or a list of dicts
    for tool messages (one dict per ToolResultPart, since the OpenAI API
    requires separate messages for each tool_call_id).

    Callers should use messages_to_openai_format() to convert a full
    conversation, which handles the flattening automatically.
    """
    role = message.role

    if role in ("user", "system"):
        return {"role": role, "content": message.text or ""}

    if role == "assistant":
        if not message.has_tool_calls:
            return {"role": "assistant", "content": message.text or ""}

        tool_calls_payload = [
            {
                "id": tc.tool_call_id,
                "type": "function",
                "function": {
                    "name": tc.tool_name,
                    # OpenAI expects a JSON string, not a dict
                    "arguments": json.dumps(tc.tool_args),
                },
            }
            for tc in message.tool_calls
        ]
        return {
            "role": "assistant",
            # content may be None when tool_calls is present — OpenAI allows this
            "content": message.text,
            "tool_calls": tool_calls_payload,
        }

    if role == "tool":
        results = [
            {
                "role": "tool",
                "tool_call_id": tr.tool_call_id,
                "content": tr.content,
            }
            for tr in message.tool_results
        ]
        if not results:
            logger.warning(
                "message_to_openai_format: tool message (id=%s) has no "
                "ToolResultParts — it will be dropped from the conversation.",
                message.id,
            )
        return results

    logger.warning("message_to_openai_format: unhandled role '%s'", role)
    return {"role": role, "content": message.text or ""}


def messages_to_openai_format(
    messages: list[AgentMessage],
) -> list[dict[str, Any]]:
    """
    Convert a full conversation history to OpenAI Chat Completions API format.

    Handles flattening of tool messages, which produce multiple dicts
    (one per ToolResultPart) from a single AgentMessage.

    This is the function called by the Agent Loop before every LLM invocation.
    """
    result: list[dict[str, Any]] = []
    for message in messages:
        converted = message_to_openai_format(message)
        if isinstance(converted, list):
            result.extend(converted)
        else:
            result.append(converted)
    return result


def tool_call_part_from_openai(tool_call_dict: dict[str, Any]) -> ToolCallPart:
    """
    Parse an OpenAI streaming response tool_call object into a ToolCallPart.

    The returned part is in 'pending' state. The agent loop advances it to
    'running' via .start() just before executing the tool.

    Handles malformed arguments JSON gracefully (logs a warning and returns
    empty tool_args) so that a single bad tool call does not crash the loop.
    """
    args_raw = tool_call_dict.get("function", {}).get("arguments", "{}")
    try:
        parsed_args: dict[str, Any] = json.loads(args_raw)
        if not isinstance(parsed_args, dict):
            logger.warning(
                "tool_call_part_from_openai: arguments parsed to non-dict type "
                "(%s), falling back to empty args. tool_call_id=%s",
                type(parsed_args).__name__,
                tool_call_dict.get("id"),
            )
            parsed_args = {}
    except (json.JSONDecodeError, TypeError):
        logger.warning(
            "tool_call_part_from_openai: failed to parse arguments JSON: %r "
            "(tool_call_id=%s)",
            args_raw,
            tool_call_dict.get("id"),
        )
        parsed_args = {}

    func_dict = tool_call_dict.get("function", {})
    tool_name = func_dict.get("name") or ""
    if not tool_name:
        logger.warning(
            "tool_call_part_from_openai: missing or empty function name "
            "(tool_call_id=%s). Falling back to 'unknown'.",
            tool_call_dict.get("id"),
        )
        tool_name = "unknown"

    return ToolCallPart(
        tool_call_id=tool_call_dict.get("id") or f"call_{uuid.uuid4().hex[:24]}",
        tool_name=tool_name,
        tool_args=parsed_args,
        state="pending",
    )
