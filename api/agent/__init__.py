"""
Agent package for DeepWiki-Open.

Provides the Agent Loop infrastructure: message model, unified provider,
agent configuration, and the core loop that lets the LLM iteratively call
tools to explore codebases before producing a final answer.

Public surface for Subtask 1 (Message Model):
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

__all__ = [
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
]
