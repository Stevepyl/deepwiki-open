"""
Agent Loop -- core iterative tool-calling engine.

Implements the ReAct pattern (Reason-Act-Observe) as an async generator:

    Query -> [LLM call -> tool_calls? -> execute -> feed back]* -> Final response

Each iteration:
    1. Convert conversation to OpenAI format and call the provider
    2. Collect text deltas and tool call requests from the stream
    3. If no tool calls: yield FinishEvent and return (done)
    4. Execute tool calls in parallel, yield ToolCallEnd events
    5. Append tool results to conversation and loop back to step 1

Key mechanisms:
    - Doom loop detection: repeating the same tool call 3+ times triggers a
      user-role warning injected into the conversation
    - Step limit: on the final allowed step, tools are disabled and the LLM
      is prompted to summarize what it has gathered
    - TaskTool injection: if the "task" tool is present, this loop injects
      itself as the executor so sub-agents can be dispatched recursively

References:
    - OpenCode session/processor.ts -- doom loop detection, max-step handling
    - OpenCode session/prompt.ts -- outer loop, step counter, completion check
    - UPDATE-PLAN.md subtask 4
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, AsyncGenerator

from api.agent.config import (
    AgentConfig,
    get_agent_config,
    get_all_agent_infos,
    get_tools_for_agent,
)
from api.agent.message import AgentMessage, ToolCallPart, messages_to_openai_format
from api.agent.provider import UnifiedProvider
from api.agent.stream_events import (
    ErrorEvent,
    FinishEvent,
    StreamEvent,
    TextDelta,
    ToolCallEnd,
    ToolCallStart,
)
from api.tools.task import TaskTool
from api.tools.tool import Tool, ToolResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DOOM_LOOP_THRESHOLD = 3
MAX_RESULT_SUMMARY_LEN = 200

MAX_STEPS_MESSAGE = (
    "You have reached the maximum number of tool-calling steps for this request. "
    "Tools are now disabled. Please provide your best answer based on the information "
    "you have gathered so far. Summarize your findings and respond to the user's "
    "original question."
)

DOOM_LOOP_MESSAGE = (
    "You are repeating the same tool call with identical arguments, which is not "
    "making progress. Try a different approach: use different arguments, try a "
    "different tool, or provide your answer based on what you have found so far."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_doom_loop(
    recent_calls: list[tuple[str, str]],
    current_tool_calls: list[ToolCallPart],
) -> bool:
    """
    Detect if the LLM is stuck repeating the same tool call.

    Checks whether the last DOOM_LOOP_THRESHOLD entries in recent_calls are
    all identical AND match the current batch of tool calls exactly.

    Args:
        recent_calls: History of (tool_name, json_args) tuples from prior steps.
                      json_args is produced by json.dumps(args, sort_keys=True).
        current_tool_calls: Tool calls from the current LLM response.

    Returns:
        True if doom loop is detected, False otherwise.
    """
    if len(recent_calls) < DOOM_LOOP_THRESHOLD or not current_tool_calls:
        return False

    # Build fingerprint set for current batch
    current_fps: set[tuple[str, str]] = set()
    for tc in current_tool_calls:
        fp = (tc.tool_name, json.dumps(tc.tool_args, sort_keys=True))
        current_fps.add(fp)

    # All calls in current batch must be identical (single unique fingerprint)
    if len(current_fps) != 1:
        return False

    # Last DOOM_LOOP_THRESHOLD entries must all match that fingerprint
    tail = recent_calls[-DOOM_LOOP_THRESHOLD:]
    tail_set = set(tail)
    return len(tail_set) == 1 and tail_set == current_fps


async def _execute_tools_parallel(
    tool_calls: list[ToolCallPart],
    tools: dict[str, Tool],
) -> list[tuple[ToolResult, bool, int]]:
    """
    Execute tool calls concurrently. Returns one result tuple per call, in order.

    Each result is (ToolResult, is_error, duration_ms). Individual failures are
    caught and returned as error ToolResult instances -- they never propagate.

    Args:
        tool_calls: Ordered list of tool calls to execute.
        tools:      Name -> Tool mapping for the current agent session.

    Returns:
        List of (ToolResult, is_error, duration_ms) in the same order as tool_calls.
    """

    async def _run_one(tc: ToolCallPart) -> tuple[ToolResult, bool, int]:
        start = time.monotonic()
        tool = tools.get(tc.tool_name)

        if tool is None:
            duration_ms = int((time.monotonic() - start) * 1000)
            available = ", ".join(sorted(tools.keys()))
            return (
                ToolResult(
                    title=tc.tool_name,
                    output=(
                        f"Error: Unknown tool '{tc.tool_name}'. "
                        f"Available tools: {available or '(none)'}"
                    ),
                    metadata={"error": "unknown_tool"},
                ),
                True,
                duration_ms,
            )

        try:
            result = await tool.execute(tc.tool_args)
            duration_ms = int((time.monotonic() - start) * 1000)
            is_error = bool(result.metadata.get("error"))
            return (result, is_error, duration_ms)
        except Exception as exc:
            duration_ms = int((time.monotonic() - start) * 1000)
            logger.error(
                "Tool '%s' raised an exception (call_id=%s): %s",
                tc.tool_name,
                tc.tool_call_id,
                exc,
            )
            return (
                ToolResult(
                    title=tc.tool_name,
                    output=f"Error executing '{tc.tool_name}': {exc}",
                    metadata={"error": "execution_error"},
                ),
                True,
                duration_ms,
            )

    results = await asyncio.gather(*(_run_one(tc) for tc in tool_calls))
    return list(results)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def run_agent_loop(
    agent_config: AgentConfig,
    messages: list[AgentMessage],
    provider: UnifiedProvider,
    tools: dict[str, Tool],
    repo_path: str,
    system_prompt_vars: dict[str, str] | None = None,
) -> AsyncGenerator[StreamEvent, None]:
    """
    Run an iterative agent loop, yielding stream events as they occur.

    The caller receives real-time TextDelta and ToolCallStart events during
    each LLM call, followed by ToolCallEnd events after tool execution. The
    loop ends with a single FinishEvent (or ErrorEvent + FinishEvent on failure).

    The loop prepends a system prompt (formatted from agent_config.system_prompt_template
    and system_prompt_vars) to the conversation automatically. Callers should NOT
    include a system message in the messages argument.

    Args:
        agent_config:       Configuration for this agent (tools, steps, prompt).
        messages:           Initial conversation -- user messages and prior turns.
                            Do NOT include a system message; the loop adds it.
        provider:           Constructed UnifiedProvider (provider + model already chosen).
        tools:              Tool name -> Tool instance map. Obtained from
                            get_tools_for_agent(agent_config, repo_path).
                            If "task" is present, the loop injects itself as executor.
        repo_path:          Absolute path to the cloned repository.
        system_prompt_vars: Optional dict with keys repo_type, repo_url, repo_name,
                            language_name for formatting the system prompt template.
                            Falls back to the raw template if a key is missing.

    Yields:
        TextDelta           -- incremental text tokens (forwarded from provider)
        ToolCallStart       -- LLM requested a tool (forwarded from provider)
        ToolCallEnd         -- tool finished executing (emitted by this loop)
        FinishEvent         -- loop completed (text-only response or max steps)
        ErrorEvent          -- provider error (followed immediately by FinishEvent)
    """
    # ------------------------------------------------------------------
    # Setup: system prompt, TaskTool injection, tool schemas
    # ------------------------------------------------------------------

    # 1. Format system prompt
    system_text = agent_config.system_prompt_template
    if system_prompt_vars:
        try:
            system_text = system_text.format(**system_prompt_vars)
        except KeyError as exc:
            logger.warning(
                "Agent '%s': system prompt template has unresolved placeholder %s. "
                "Using raw template.",
                agent_config.name,
                exc,
            )
    system_msg = AgentMessage.system(system_text)

    # 2. TaskTool executor closure -- defined before the while loop so it
    #    can close over provider and system_prompt_vars.
    async def _task_executor(
        prompt: str,
        agent_name: str,
        task_id: str | None,
        executor_repo_path: str,
    ) -> ToolResult:
        """Recursive sub-agent dispatch for the TaskTool."""
        try:
            sub_config = get_agent_config(agent_name)
        except KeyError as exc:
            return ToolResult(
                title=f"task:{agent_name}",
                output=f"Error: {exc}",
                metadata={"error": "unknown_agent"},
            )

        # Get tools for the sub-agent. No TaskTool executor is injected,
        # which prevents infinite nesting (sub-agents cannot call task).
        sub_tools = get_tools_for_agent(sub_config, executor_repo_path)

        # Reuse parent's system_prompt_vars when available, otherwise build
        # minimal fallback vars.
        if system_prompt_vars:
            sub_vars = dict(system_prompt_vars)
        else:
            repo_name = (
                executor_repo_path.rsplit("/", 1)[-1]
                if "/" in executor_repo_path
                else executor_repo_path
            )
            sub_vars = {
                "repo_type": "repository",
                "repo_url": executor_repo_path,
                "repo_name": repo_name,
                "language_name": "English",
            }

        sub_messages = [AgentMessage.user(prompt)]
        collected_text: list[str] = []
        subagent_error: str | None = None

        try:
            async for event in run_agent_loop(
                agent_config=sub_config,
                messages=sub_messages,
                provider=provider,  # shared -- safe: no per-call mutable state
                tools=sub_tools,
                repo_path=executor_repo_path,
                system_prompt_vars=sub_vars,
            ):
                if isinstance(event, TextDelta):
                    collected_text.append(event.content)
                elif isinstance(event, ErrorEvent):
                    subagent_error = event.error
                # Other events (ToolCallStart/End, FinishEvent) are discarded;
                # the parent agent only sees the sub-agent's final text output.
        except Exception as exc:
            logger.error("Sub-agent '%s' failed: %s", agent_name, exc)
            return ToolResult(
                title=f"task:{agent_name}",
                output=f"Sub-agent error: {exc}",
                metadata={"error": "subagent_error"},
            )

        if subagent_error is not None and not collected_text:
            return ToolResult(
                title=f"task:{agent_name}",
                output=f"Sub-agent error: {subagent_error}",
                metadata={"error": "subagent_error", "agent_name": agent_name},
            )

        output = "".join(collected_text) or "(Sub-agent produced no output)"
        return ToolResult(
            title=f"task:{agent_name}",
            output=output,
            metadata={"agent_name": agent_name, "task_id": task_id or ""},
        )

    # 3. Inject TaskTool dependencies
    if "task" in tools and isinstance(tools["task"], TaskTool):
        agent_infos = get_all_agent_infos()
        tools = {
            **tools,
            "task": tools["task"].with_agents(agent_infos).with_executor(_task_executor),
        }

    # 4. Build tool schemas (None -> text-only mode)
    tool_schemas: list[dict[str, Any]] | None = (
        [t.to_function_schema() for t in tools.values()] or None
    )

    # 5. Initialize loop state
    conversation: list[AgentMessage] = [system_msg, *messages]
    step = 0
    recent_calls: list[tuple[str, str]] = []  # (tool_name, json_args) for doom detection
    total_usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}

    # ------------------------------------------------------------------
    # Main agent loop
    # ------------------------------------------------------------------

    while step < agent_config.max_steps:
        step += 1
        is_final_step = (step == agent_config.max_steps)

        # A. Final step: disable tools, inject summary request as assistant
        #    message so the LLM treats it as its own prior realization.
        if is_final_step:
            logger.info(
                "Agent '%s': step %d is the final step (max=%d). Disabling tools.",
                agent_config.name,
                step,
                agent_config.max_steps,
            )
            conversation.append(AgentMessage.assistant_text(MAX_STEPS_MESSAGE))
            current_tool_schemas = None
        else:
            current_tool_schemas = tool_schemas

        # B. Convert conversation history to OpenAI format
        openai_messages = messages_to_openai_format(conversation)

        # C. Stream a single LLM call
        collected_text: list[str] = []
        collected_tool_calls: list[ToolCallPart] = []
        step_usage: dict[str, int] | None = None
        had_error = False

        try:
            async for event in provider.stream_chat(openai_messages, current_tool_schemas):
                if isinstance(event, TextDelta):
                    collected_text.append(event.content)
                    yield event  # forward to caller in real time

                elif isinstance(event, ToolCallStart):
                    tc_part = ToolCallPart(
                        tool_call_id=event.tool_call_id,
                        tool_name=event.tool_name,
                        tool_args=event.tool_args,
                        state="pending",
                    )
                    collected_tool_calls.append(tc_part)
                    yield event  # forward to caller in real time

                elif isinstance(event, FinishEvent):
                    step_usage = event.usage
                    # Do not yield FinishEvent here; the loop emits its own
                    # FinishEvent when all iterations are complete.

                elif isinstance(event, ErrorEvent):
                    yield event
                    had_error = True
                    break

        except Exception as exc:
            logger.error(
                "Agent '%s': provider stream failed at step %d: %s",
                agent_config.name,
                step,
                exc,
            )
            yield ErrorEvent(error=f"Provider error: {exc}", code="provider_error")
            yield FinishEvent(finish_reason="error", usage=total_usage or None)
            return

        if had_error:
            yield FinishEvent(finish_reason="error", usage=total_usage or None)
            return

        # D. Accumulate token usage across steps
        if step_usage:
            total_usage["prompt_tokens"] += step_usage.get("prompt_tokens", 0)
            total_usage["completion_tokens"] += step_usage.get("completion_tokens", 0)

        # E. Build the assistant message from collected content
        text_content = "".join(collected_text) or None

        if collected_tool_calls:
            assistant_msg = AgentMessage.assistant_tool_calls(
                tool_calls=collected_tool_calls,
                text=text_content,
                model=provider.model,
                provider=provider.provider,
            )
        elif text_content:
            assistant_msg = AgentMessage.assistant_text(
                content=text_content,
                model=provider.model,
                provider=provider.provider,
            )
        else:
            # Empty response (no text, no tool calls) -- treat as done
            logger.warning(
                "Agent '%s': step %d produced an empty response. Stopping.",
                agent_config.name,
                step,
            )
            yield FinishEvent(finish_reason="stop", usage=total_usage or None)
            return

        conversation.append(assistant_msg)

        # F. No tool calls -> LLM is done; yield FinishEvent and return
        if not collected_tool_calls:
            logger.info(
                "Agent '%s': step %d produced text only. Loop complete.",
                agent_config.name,
                step,
            )
            yield FinishEvent(finish_reason="stop", usage=total_usage or None)
            return

        # G. Doom loop detection: inject a warning if LLM is stuck
        if _check_doom_loop(recent_calls, collected_tool_calls):
            logger.warning(
                "Agent '%s': doom loop detected at step %d (tool='%s'). "
                "Injecting intervention message.",
                agent_config.name,
                step,
                collected_tool_calls[0].tool_name,
            )
            conversation.append(AgentMessage.user(DOOM_LOOP_MESSAGE))

        # Track this step's calls for future doom detection
        for tc_part in collected_tool_calls:
            recent_calls.append((
                tc_part.tool_name,
                json.dumps(tc_part.tool_args, sort_keys=True),
            ))

        # H. Execute all tool calls concurrently
        logger.info(
            "Agent '%s': step %d executing %d tool call(s): %s",
            agent_config.name,
            step,
            len(collected_tool_calls),
            [tc.tool_name for tc in collected_tool_calls],
        )
        results = await _execute_tools_parallel(collected_tool_calls, tools)

        # I + J. Yield ToolCallEnd events and build tool result messages
        for tc_part, (tool_result, is_error, duration_ms) in zip(
            collected_tool_calls, results
        ):
            summary = tool_result.output[:MAX_RESULT_SUMMARY_LEN]
            if len(tool_result.output) > MAX_RESULT_SUMMARY_LEN:
                summary += "..."

            yield ToolCallEnd(
                tool_call_id=tc_part.tool_call_id,
                tool_name=tc_part.tool_name,
                result_summary=summary,
                is_error=is_error,
                duration_ms=duration_ms,
                metadata=dict(tool_result.metadata),
            )

            conversation.append(
                AgentMessage.tool_result(
                    tool_call_id=tc_part.tool_call_id,
                    tool_name=tc_part.tool_name,
                    result=tool_result,
                    is_error=is_error,
                )
            )

        # K. Continue to the next iteration

    # L. Safety net -- reached here only if the while condition was exhausted
    #    without a return inside the loop (should not happen in practice because
    #    the final-step handling disables tools and the LLM produces text-only).
    logger.warning(
        "Agent '%s': safety net reached after %d steps.",
        agent_config.name,
        agent_config.max_steps,
    )
    yield FinishEvent(finish_reason="stop", usage=total_usage or None)
