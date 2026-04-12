"""
Agent package for DeepWiki-Open.

Provides the Agent Loop infrastructure: message model, unified provider,
agent configuration, and the core loop that lets the LLM iteratively call
tools to explore codebases before producing a final answer.

Public surface for Subtask 1 (Message Model) and Subtask 2 (Unified Provider):
"""

from api.agent.message import (
    AgentMessage,
    ErrorPart,
    MessagePart,
    PartType,
    Role,
    TextPart,
    ToolCallPart,
    ToolCallStatus,
    ToolResultPart,
    message_to_openai_format,
    messages_to_openai_format,
    tool_call_part_from_openai,
)
from api.agent.provider import UnifiedProvider
from api.agent.stream_events import (
    ErrorEvent,
    FinishEvent,
    StreamEvent,
    StreamEventType,
    TextDelta,
    ToolCallDelta,
    ToolCallStart,
)

__all__ = [
    # Subtask 1 — Message Model
    "AgentMessage",
    "ErrorPart",
    "MessagePart",
    "PartType",
    "Role",
    "TextPart",
    "ToolCallPart",
    "ToolCallStatus",
    "ToolResultPart",
    "message_to_openai_format",
    "messages_to_openai_format",
    "tool_call_part_from_openai",
    # Subtask 2 — Unified Provider
    "UnifiedProvider",
    "ErrorEvent",
    "FinishEvent",
    "StreamEvent",
    "StreamEventType",
    "TextDelta",
    "ToolCallDelta",
    "ToolCallStart",
]
