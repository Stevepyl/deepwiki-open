"""Helpers for request-local chat history formatting."""

from __future__ import annotations

from typing import Protocol, Sequence


class ChatMessageLike(Protocol):
    role: str
    content: str


def format_conversation_history(messages: Sequence[ChatMessageLike]) -> str:
    conversation_history = ""
    for i in range(0, len(messages) - 1, 2):
        if i + 1 >= len(messages):
            continue
        user_msg = messages[i]
        assistant_msg = messages[i + 1]
        if user_msg.role == "user" and assistant_msg.role == "assistant":
            conversation_history += (
                "<turn>\n"
                f"<user>{user_msg.content}</user>\n"
                f"<assistant>{assistant_msg.content}</assistant>\n"
                "</turn>\n"
            )
    return conversation_history
