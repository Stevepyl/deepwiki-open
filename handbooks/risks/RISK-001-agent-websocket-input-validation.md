---
number: RISK-001
name: Agent WebSocket Input Validation
description: Records the role-part invariant gap at the external agent WebSocket boundary and the recommended validation strategy.
update_at: 2026-05-04
category: risk-record
language: zh-CN
status: open
---

# Agent WebSocket 输入验证：角色-Part 不变式

## 背景

在实现 Subtask 1（Message Model）时，Codex 对抗性审查发现了一个设计缺口：

`AgentMessage` 的数据模型层**没有**在构造期强制执行角色-Part 对应关系。`message_to_openai_format()` 已经隐式假设了这些不变式（`role=tool` 只序列化 `ToolResultPart`，其余丢弃），但模型本身接受任意 part 组合。这两者的不一致被刻意保留在数据层，因为 Agent Loop（Subtask 4）内部只通过 convenience constructors 创建消息，不会产生违规组合。

**然而**，Subtask 6（WebSocket handler）是第一个从外部接收 JSON 并通过 `AgentMessage.model_validate()` 反序列化的代码路径。这里是真正的系统边界，必须在此执行角色-Part 不变式校验。

## 不变式规则

| `role` | 允许的 Part 类型 | 禁止的 Part 类型 |
|--------|----------------|----------------|
| `user` | `TextPart` | `ToolCallPart`, `ToolResultPart` |
| `system` | `TextPart` | `ToolCallPart`, `ToolResultPart`, `ErrorPart` |
| `assistant` | `TextPart`, `ToolCallPart`, `ErrorPart` | `ToolResultPart` |
| `tool` | `ToolResultPart` | `TextPart`, `ToolCallPart`, `ErrorPart` |

**为什么 `user` 允许 `ErrorPart`？** 不允许。`ErrorPart` 是 agent 内部用于 doom loop / 步数超限等情况的标记，不应出现在任何来自外部的消息中。

## 静默丢弃的风险场景

若没有输入验证，以下格式错误的外部 JSON 会产生无声的数据丢失：

```json
{
  "role": "tool",
  "parts": [
    {"type": "text", "content": "工具结果"},
    {"type": "error", "content": "执行出错"}
  ]
}
```

`message_to_openai_format()` 对 `role=tool` 的消息只提取 `ToolResultPart`，该消息会返回空列表 `[]`，在 `messages_to_openai_format()` 中被 `extend` 后**无声消失**。下一轮 LLM 调用将缺失这个工具结果，可能导致模型重复发出已执行的工具调用（副作用重放）。

## 在 Subtask 6 中的实施位置

验证应在 WebSocket handler 将外部请求的 `messages` 字段转换为 `list[AgentMessage]` 之后、传入 `run_agent_loop()` 之前执行。

```
WebSocket 收到 JSON
    -> 解析为 ChatCompletionRequest (Pydantic 验证字段类型和必填项)
    -> AgentMessage.from_chat_messages(request.messages)  <- 已有的 legacy 转换
          或
       [AgentMessage.model_validate(m) for m in raw_messages]  <- agent 模式下的转换
    -> [HERE] 执行角色-Part 不变式验证  <- 需要在此添加
    -> run_agent_loop(agent_config, messages, ...)
```

## 推荐实现

在 `api/agent/websocket_handler.py` 中添加以下验证函数（或放入 `api/agent/message.py` 作为模块级工具函数）：

```python
_ROLE_ALLOWED_PARTS: dict[str, frozenset[str]] = {
    "user":      frozenset({"text"}),
    "system":    frozenset({"text"}),
    "assistant": frozenset({"text", "tool_call", "error"}),
    "tool":      frozenset({"tool_result"}),
}


def validate_message_role_parts(messages: list[AgentMessage]) -> None:
    """
    Enforce role/part invariants at the system boundary.

    Raises ValueError with a descriptive message if any AgentMessage
    contains a part type not allowed for its role.

    Call this immediately after deserializing external JSON into AgentMessage
    instances, before passing the list to run_agent_loop().
    """
    for i, msg in enumerate(messages):
        allowed = _ROLE_ALLOWED_PARTS.get(msg.role, frozenset())
        for part in msg.parts:
            if part.type not in allowed:
                raise ValueError(
                    f"messages[{i}]: role='{msg.role}' does not allow "
                    f"part type '{part.type}'. "
                    f"Allowed types for this role: {sorted(allowed)}. "
                    f"message_id={msg.id}"
                )
```

## WebSocket 错误响应格式

验证失败时，WebSocket handler 应发送结构化错误事件（而非直接关闭连接），让前端能正确显示错误原因：

```python
await websocket.send_json({
    "type": "error",
    "content": f"Invalid message format: {e}",
    "code": "invalid_message_role_parts",
})
await websocket.close()
```

## 与 Subtask 4（Agent Loop）的关系

Agent Loop 内部**不需要**再次执行此验证。它只通过以下方式创建消息：

- `AgentMessage.user(...)` — TextPart only ✓
- `AgentMessage.system(...)` — TextPart only ✓
- `AgentMessage.assistant_tool_calls(...)` — TextPart + ToolCallPart ✓
- `AgentMessage.tool_result(...)` — ToolResultPart only ✓

Doom loop 注入和步数超限注入也只产生合法组合。验证逻辑只属于系统边界（Subtask 6/7），不属于内部流转（Subtask 4）。

## 相关文件

- `api/agent/message.py` — `AgentMessage`, `message_to_openai_format()`（序列化的隐式假设在此）
- `api/agent/websocket_handler.py` — 需要添加验证的目标文件（Subtask 6 产出）
- `api/agent/rest_handler.py` — 同样需要相同验证（Subtask 7 产出）
