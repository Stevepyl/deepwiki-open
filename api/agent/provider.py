"""
Unified Provider — wraps all LLM provider clients behind a single streaming interface.

Problem solved: api/websocket_wiki.py and api/simple_chat.py each contain
~300 lines of near-identical if/elif chains for provider selection and streaming,
repeated 3× each (init, streaming, fallback) = 6 copies total. The agent loop
(subtask 4) cannot use this pattern — it needs to call LLMs in a tight loop
without knowing provider details.

Design (from UPDATE-PLAN.md, subtask 2):
- "包装而非替换" — delegates to existing AdalFlow ModelClient subclasses for
  client initialisation and API key handling
- Single public method: stream_chat(messages, tools) -> AsyncGenerator[StreamEvent]
- Native function calling: OpenAI / Azure / Dashscope / Google genai
- Tools-in-prompt fallback: OpenRouter / Ollama / Bedrock (inject tool schemas
  into system prompt, parse <tool_call> XML tags from text response)

Provider streaming quirks handled here so callers never need to:
- OpenAI/Azure: raw streaming via async_client (bypasses AdalFlow acall wrapper)
- Dashscope: same bypass; adds enable_thinking=False + workspace_id header
- OpenRouter: uses acall() which internally forces stream=False and yields text
- Ollama: needs options:{} re-nesting + /no_think suffix + 4-level chunk fallback
- Bedrock: synchronous, non-streaming; concatenates messages into flat prompt
- Google genai: native SDK, synchronous streaming wrapped in run_in_executor
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from typing import Any, AsyncGenerator, Iterator, Optional, TypeVar

from adalflow.core.types import ModelType

from api.agent.stream_events import (
    ErrorEvent,
    FinishEvent,
    StreamEvent,
    TextDelta,
    ToolCallStart,
)
from api.config import get_model_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

_OPENAI_COMPAT_PROVIDERS = frozenset({"openai", "azure", "dashscope"})
_NATIVE_FC_PROVIDERS = frozenset({"openai", "azure", "dashscope", "google"})

# Regex for parsing <tool_call>{...}</tool_call> blocks from LLM text output.
# XML tags chosen over ``` fences because they are less likely to appear in
# code-related LLM output, reducing false-positive matches.
_TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL,
)

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Async utility
# ---------------------------------------------------------------------------


async def _iter_sync_generator(gen: Iterator[T]) -> AsyncGenerator[T, None]:
    """
    Wrap a synchronous iterator as an async generator.

    Each next() call is dispatched to the default executor (thread pool) so
    that network-blocking calls (e.g. Google genai's synchronous streaming)
    do not block the event loop between chunks.
    """
    loop = asyncio.get_running_loop()
    sentinel = object()
    while True:
        item = await loop.run_in_executor(
            None, lambda: next(gen, sentinel)  # noqa: B023
        )
        if item is sentinel:
            break
        yield item  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Google genai format converters (module-level — stateless, independently testable)
# ---------------------------------------------------------------------------


def _openai_tools_to_google_tools(
    tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Convert OpenAI function-calling schemas to Google genai format.

    OpenAI: [{"type": "function", "function": {"name": ..., "description": ..., "parameters": {...}}}]
    Google: [{"function_declarations": [{"name": ..., "description": ..., "parameters": {...}}]}]
    """
    declarations = []
    for tool in tools:
        func = tool.get("function", {})
        declarations.append({
            "name": func.get("name", ""),
            "description": func.get("description", ""),
            "parameters": func.get("parameters", {}),
        })
    return [{"function_declarations": declarations}]


def _messages_to_google_contents(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Convert OpenAI-format messages to Google genai content format.

    Google genai Content format: [{"role": "user"|"model", "parts": [{"text": "..."}]}]

    Mapping:
        system    -> merged into the first user message's text
        user      -> role "user"
        assistant -> role "model"; tool_calls converted to function_call parts
        tool      -> role "user" with function_response part

    The role="tool" → function_response conversion requires the function name,
    which is not present in the OpenAI tool message itself. We build a lookup
    from tool_call_id → function_name as we encounter assistant messages with
    tool_calls, so that subsequent role="tool" messages can be resolved.
    """
    system_parts: list[str] = []
    contents: list[dict[str, Any]] = []
    # Built as we scan assistant messages; used to resolve role="tool" names.
    _tc_id_to_name: dict[str, str] = {}

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "") or ""

        if role == "system":
            system_parts.append(content)

        elif role == "user":
            text = content
            # Prepend accumulated system instructions to the first user message
            if system_parts:
                text = "\n\n".join(system_parts) + "\n\n" + text
                system_parts = []
            contents.append({"role": "user", "parts": [{"text": text}]})

        elif role == "assistant":
            parts: list[dict[str, Any]] = []
            if content:
                parts.append({"text": content})
            for tc in (msg.get("tool_calls") or []):
                func = tc.get("function", {})
                name = func.get("name", "")
                args_raw = func.get("arguments", "{}")
                try:
                    args: dict[str, Any] = (
                        json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                    )
                except json.JSONDecodeError:
                    args = {}
                parts.append({"function_call": {"name": name, "args": args}})
                tc_id = tc.get("id", "")
                if tc_id and name:
                    _tc_id_to_name[tc_id] = name
            if parts:
                contents.append({"role": "model", "parts": parts})

        elif role == "tool":
            tc_id = msg.get("tool_call_id", "")
            func_name = _tc_id_to_name.get(tc_id, tc_id or "unknown")
            contents.append({
                "role": "user",
                "parts": [{
                    "function_response": {
                        "name": func_name,
                        "response": {"result": content},
                    }
                }],
            })

    # If only system messages were provided (edge case), emit one user turn
    if system_parts and not contents:
        contents.append({"role": "user", "parts": [{"text": "\n\n".join(system_parts)}]})

    return contents


# ---------------------------------------------------------------------------
# Ollama helpers
# ---------------------------------------------------------------------------


def _extract_ollama_text(chunk: Any) -> Optional[str]:
    """
    Extract text from an Ollama streaming chunk using a 4-level fallback.

    Replicates the defensive logic in websocket_wiki.py:580-601 which handles
    the fact that OllamaClient chunks can be dicts, dataclasses, or plain
    objects depending on the Ollama server version and model.
    """
    text = None

    # Level 1: dict path ({"message": {"content": "..."}})
    if isinstance(chunk, dict):
        msg = chunk.get("message")
        text = msg.get("content") if isinstance(msg, dict) else msg
    else:
        # Level 2: attribute path (chunk.message.content)
        message = getattr(chunk, "message", None)
        if message is not None:
            if isinstance(message, dict):
                text = message.get("content")
            else:
                text = getattr(message, "content", None)

    # Level 3: alternative top-level attributes
    if not text:
        text = getattr(chunk, "response", None) or getattr(chunk, "text", None)

    # Level 4: __dict__ fallback
    if not text and hasattr(chunk, "__dict__"):
        msg = chunk.__dict__.get("message")
        if isinstance(msg, dict):
            text = msg.get("content")

    # Filter spurious metadata strings and strip thinking tags
    if isinstance(text, str) and text:
        if text.startswith("model=") or text.startswith("created_at="):
            return None
        return text.replace("<think>", "").replace("</think>", "")

    return None


def _inject_ollama_no_think(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Append ' /no_think' to the last user message to suppress chain-of-thought
    output from Qwen and similar models hosted on Ollama.

    Returns a new list; does not mutate input.
    Mirrors the existing pattern in websocket_wiki.py:443.
    """
    if not messages:
        return messages

    result = list(messages)
    # Find the last user message (usually the last item, but iterate in reverse
    # to be safe in multi-turn conversations)
    for i in range(len(result) - 1, -1, -1):
        if result[i].get("role") == "user":
            original = result[i]
            result[i] = {
                **original,
                "content": (original.get("content") or "") + " /no_think",
            }
            break

    return result


# ---------------------------------------------------------------------------
# Bedrock helpers
# ---------------------------------------------------------------------------


def _messages_to_bedrock_prompt(messages: list[dict[str, Any]]) -> str:
    """
    Concatenate structured messages into a flat Anthropic Human/Assistant prompt.

    Bedrock's BedrockClient._format_prompt_for_provider() expects a single
    prompt string. Multi-turn context is preserved via the Human/Assistant
    alternating format that Anthropic models understand.
    """
    parts: list[str] = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "") or ""

        if role == "system":
            # System instructions prepended without a role label
            parts.append(content)
        elif role == "user":
            parts.append(f"\n\nHuman: {content}")
        elif role == "assistant":
            parts.append(f"\n\nAssistant: {content}")
        # tool messages are skipped (Bedrock has no tool calling support here)

    # Always end with the Assistant turn marker so the model continues
    parts.append("\n\nAssistant: ")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Tools-in-prompt helpers (module-level — stateless, independently testable)
# ---------------------------------------------------------------------------


def _build_tools_in_prompt(tools: list[dict[str, Any]]) -> str:
    """
    Format OpenAI tool schemas into a natural-language prompt block.

    Uses <tool_call> XML tags rather than ``` fences to avoid ambiguity with
    code blocks in LLM output about code.
    """
    lines: list[str] = [
        "## Available Tools",
        "",
        "To call a tool, respond with a JSON block wrapped in <tool_call> tags:",
        "",
        "<tool_call>",
        '{"name": "tool_name", "arguments": {"param1": "value1"}}',
        "</tool_call>",
        "",
        "You may use multiple <tool_call> blocks in a single response.",
        "Only use tools when they help answer the question. "
        "If no tools are needed, respond normally.",
        "",
        "### Tool Definitions",
        "",
    ]

    for tool in tools:
        func = tool.get("function", {})
        name = func.get("name", "")
        description = func.get("description", "")
        parameters = func.get("parameters", {})
        properties = parameters.get("properties", {})
        required = set(parameters.get("required", []))

        lines.append(f"**{name}** — {description}")

        if properties:
            lines.append("Parameters:")
            for param_name, param_schema in properties.items():
                param_type = param_schema.get("type", "any")
                param_desc = param_schema.get("description", "")
                req_marker = ", required" if param_name in required else ""
                desc_part = f": {param_desc}" if param_desc else ""
                lines.append(f"- {param_name} ({param_type}{req_marker}){desc_part}")

        lines.append("")

    return "\n".join(lines)


def _inject_tools_prompt(
    messages: list[dict[str, Any]],
    tools_prompt: str,
) -> list[dict[str, Any]]:
    """
    Inject the tools-in-prompt block into the conversation's system message.

    - If a system message exists: appends to its content (creating a new dict).
    - If no system message exists: prepends a new system message.
    - Does NOT mutate the input list or any message dicts.
    """
    result = list(messages)

    # Find the first system message
    for i, msg in enumerate(result):
        if msg.get("role") == "system":
            result[i] = {
                **msg,
                "content": (msg.get("content") or "") + "\n\n" + tools_prompt,
            }
            return result

    # No system message found — prepend one
    result.insert(0, {"role": "system", "content": tools_prompt})
    return result


def _parse_tool_calls_from_text(
    text: str,
) -> tuple[str, list[ToolCallStart]]:
    """
    Extract <tool_call> JSON blocks from LLM text output.

    Returns:
        (clean_text, tool_calls) where clean_text has the <tool_call> blocks
        removed and tool_calls is a list of ToolCallStart events ready for
        the agent loop.

    Malformed JSON blocks are logged and skipped (no exception raised).
    """
    tool_calls: list[ToolCallStart] = []

    for match in _TOOL_CALL_PATTERN.finditer(text):
        raw_json = match.group(1)
        try:
            raw = json.loads(raw_json)
        except (json.JSONDecodeError, TypeError):
            logger.warning(
                "_parse_tool_calls_from_text: failed to parse block: %.200s",
                match.group(0),
            )
            continue

        if not isinstance(raw, dict):
            logger.warning(
                "_parse_tool_calls_from_text: parsed non-dict JSON: %s",
                type(raw).__name__,
            )
            continue

        tool_name = raw.get("name") or "unknown"
        tool_args = raw.get("arguments", {})
        if not isinstance(tool_args, dict):
            tool_args = {}

        tool_calls.append(
            ToolCallStart(
                tool_call_id=f"call_{uuid.uuid4().hex[:24]}",
                tool_name=tool_name,
                tool_args=tool_args,
            )
        )

    clean_text = _TOOL_CALL_PATTERN.sub("", text).strip()
    return clean_text, tool_calls


# ---------------------------------------------------------------------------
# UnifiedProvider
# ---------------------------------------------------------------------------


class UnifiedProvider:
    """
    Wraps all LLM provider clients behind a single streaming interface.

    Usage:
        provider = UnifiedProvider("openai", "gpt-5-nano")
        msgs = [{"role": "user", "content": "Find the main function"}]
        async for event in provider.stream_chat(msgs, tools=tool_schemas):
            if isinstance(event, TextDelta):
                print(event.content, end="", flush=True)
            elif isinstance(event, ToolCallStart):
                result = await execute_tool(event.tool_name, event.tool_args)

    Thread safety: UnifiedProvider instances are not thread-safe. Create one
    per agent loop session.
    """

    def __init__(self, provider: str, model: str) -> None:
        """
        Args:
            provider: Provider ID ("openai", "google", "openrouter", "ollama",
                      "bedrock", "azure", "dashscope").
            model: Model name (validated against generator.json config).

        Raises:
            ValueError: If the provider or model is not recognised by
                        get_model_config().
        """
        self.provider = provider
        self.model = model
        # Eager config load: validates provider/model existence at construction
        # time rather than at first call. Raises ValueError on unknown provider.
        self._config = get_model_config(provider, model)
        self._model_kwargs: dict[str, Any] = self._config["model_kwargs"]
        self._client: Any = None
        self._client_initialized: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """
        Stream a chat completion, yielding structured StreamEvent instances.

        Args:
            messages: Conversation history in OpenAI Chat Completions format.
                      Each dict must have "role" and "content" keys.
            tools: OpenAI-format function schemas, or None for text-only mode.
                   For providers without native function calling, tool schemas
                   are injected into the system prompt (tools-in-prompt mode).

        Yields:
            TextDelta        -- incremental text tokens
            ToolCallStart    -- complete tool invocation (args fully assembled)
            FinishEvent      -- stream complete
            ErrorEvent       -- irrecoverable error; no further events follow
        """
        self._ensure_client()

        use_native_tools = bool(tools) and self._supports_native_tools()
        use_prompt_tools = bool(tools) and not use_native_tools

        effective_messages = messages
        effective_tools = tools if use_native_tools else None

        if use_prompt_tools:
            tools_prompt = _build_tools_in_prompt(tools)  # type: ignore[arg-type]
            effective_messages = _inject_tools_prompt(messages, tools_prompt)

        if use_prompt_tools:
            # Buffer mode: non-native-FC providers cannot stream tool calls.
            # Collect all text, then parse <tool_call> blocks at the end.
            buffered: list[str] = []
            async for event in self._dispatch(effective_messages, effective_tools):
                if isinstance(event, TextDelta):
                    buffered.append(event.content)
                elif isinstance(event, ErrorEvent):
                    yield event
                    return
                # FinishEvent from inner dispatch is discarded — we emit our own below
            full_text = "".join(buffered)
            clean_text, parsed_tool_calls = _parse_tool_calls_from_text(full_text)
            if clean_text:
                yield TextDelta(content=clean_text)
            for tc in parsed_tool_calls:
                yield tc
            yield FinishEvent(
                finish_reason="tool_calls" if parsed_tool_calls else "stop"
            )
        else:
            # Stream mode: pass events through directly (native FC or no tools)
            async for event in self._dispatch(effective_messages, effective_tools):
                yield event

    # ------------------------------------------------------------------
    # Internals — provider routing
    # ------------------------------------------------------------------

    def _ensure_client(self) -> None:
        """Lazily initialise the AdalFlow ModelClient for this provider."""
        if self._client_initialized:
            return
        if self.provider != "google":
            client_class = self._config["model_client"]
            self._client = client_class()
        # Google genai creates a GenerativeModel per stream_chat call; no
        # persistent client needed.
        self._client_initialized = True

    def _supports_native_tools(self) -> bool:
        """True if this provider supports native function calling."""
        return self.provider in _NATIVE_FC_PROVIDERS

    def _get_generation_params(self) -> dict[str, Any]:
        """
        Extract generation hyperparameters (temperature, top_p, etc.) from
        the loaded model_kwargs, excluding the model name itself.
        """
        params: dict[str, Any] = {}
        for key in ("temperature", "top_p", "top_k", "max_tokens"):
            if key in self._model_kwargs:
                params[key] = self._model_kwargs[key]
        return params

    async def _dispatch(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """
        Route to the provider-specific streaming method.

        All uncaught exceptions from the provider methods bubble up here and
        are converted to ErrorEvent so the caller always sees a clean stream.
        """
        try:
            if self.provider in _OPENAI_COMPAT_PROVIDERS:
                async for event in self._stream_openai_compat(messages, tools):
                    yield event
            elif self.provider == "google":
                async for event in self._stream_google_genai(messages, tools):
                    yield event
            elif self.provider == "openrouter":
                async for event in self._stream_openrouter(messages):
                    yield event
            elif self.provider == "ollama":
                async for event in self._stream_ollama(messages):
                    yield event
            elif self.provider == "bedrock":
                async for event in self._stream_bedrock(messages):
                    yield event
            else:
                yield ErrorEvent(
                    error=f"Unsupported provider: {self.provider!r}",
                    code="unsupported_provider",
                )
        except Exception as exc:
            logger.error(
                "UnifiedProvider._dispatch: provider=%s error=%s",
                self.provider,
                exc,
                exc_info=True,
            )
            yield ErrorEvent(error=str(exc), code="provider_error")

    # ------------------------------------------------------------------
    # Provider-specific streaming methods
    # ------------------------------------------------------------------

    async def _stream_openai_compat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """
        Stream from OpenAI, Azure, or Dashscope.

        Why we bypass AdalFlow acall():
          - DashscopeClient.acall() streaming wrapper extracts only delta.content
            (dashscope_client.py:519-528), discarding delta.tool_calls entirely.
          - OpenAIClient.acall() forces streaming even for non-streaming calls and
            reconstructs a synthetic ChatCompletion (openai_client.py:419-460).
          Accessing async_client directly gives us raw chunks with full tool_calls.

        Tool calls are accumulated by index across stream chunks and emitted as
        complete ToolCallStart events after the stream ends.
        """
        # Ensure the async SDK client is ready.
        # init_async_client() only returns the client; it does NOT set self.async_client
        # internally (confirmed: openai_client.py:198-204, dashscope_client.py:186).
        # We bypass acall() so we must do the assignment ourselves.
        if self._client.async_client is None:
            self._client.async_client = self._client.init_async_client()

        gen_params = self._get_generation_params()
        api_kwargs: dict[str, Any] = {
            "messages": messages,
            "model": self.model,
            "stream": True,
            **gen_params,
        }

        # Dashscope-specific: workspace ID header + enable_thinking flag.
        # enable_thinking=False prevents Qwen models from returning reasoning
        # tokens in the streamed response (dashscope_client.py:506-510 shows
        # this is only needed for non-streaming, but injecting it for streaming
        # is harmless and future-proofs against model updates).
        if self.provider == "dashscope":
            api_kwargs["extra_body"] = {"enable_thinking": False}
            workspace_id = getattr(self._client.async_client, "_workspace_id", None)
            if workspace_id:
                api_kwargs["extra_headers"] = {
                    "X-DashScope-WorkSpace": workspace_id
                }

        if tools:
            api_kwargs["tools"] = tools

        response = await self._client.async_client.chat.completions.create(
            **api_kwargs
        )

        # Index-based accumulator for tool call argument fragments.
        # OpenAI sends: chunk 1 has id+name+args="", subsequent chunks have args fragments.
        tool_calls_acc: dict[int, dict[str, str]] = {}  # idx -> {id, name, args}
        usage: Optional[dict[str, int]] = None

        async for chunk in response:
            if not chunk.choices:
                # Some providers send a final usage-only chunk with no choices
                if getattr(chunk, "usage", None):
                    usage = {
                        "prompt_tokens": chunk.usage.prompt_tokens,
                        "completion_tokens": chunk.usage.completion_tokens,
                    }
                continue

            choice = chunk.choices[0]
            delta = choice.delta

            # Text content
            if delta and delta.content:
                yield TextDelta(content=delta.content)

            # Tool call fragments — accumulate by index
            if delta and getattr(delta, "tool_calls", None):
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_calls_acc:
                        tool_calls_acc[idx] = {"id": "", "name": "", "args": ""}
                    if tc_delta.id:
                        tool_calls_acc[idx]["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            tool_calls_acc[idx]["name"] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            tool_calls_acc[idx]["args"] += tc_delta.function.arguments

            # Capture usage if available on this chunk
            if getattr(choice, "finish_reason", None) and getattr(chunk, "usage", None):
                usage = {
                    "prompt_tokens": chunk.usage.prompt_tokens,
                    "completion_tokens": chunk.usage.completion_tokens,
                }

        # Emit complete ToolCallStart events in index order
        for idx in sorted(tool_calls_acc):
            tc = tool_calls_acc[idx]
            args_str = tc["args"]
            try:
                parsed_args: dict[str, Any] = json.loads(args_str) if args_str else {}
                if not isinstance(parsed_args, dict):
                    parsed_args = {}
            except json.JSONDecodeError:
                logger.warning(
                    "_stream_openai_compat: failed to parse tool args: %.200s",
                    args_str,
                )
                parsed_args = {}

            yield ToolCallStart(
                tool_call_id=tc["id"] or f"call_{uuid.uuid4().hex[:24]}",
                tool_name=tc["name"] or "unknown",
                tool_args=parsed_args,
            )

        finish_reason = "tool_calls" if tool_calls_acc else "stop"
        yield FinishEvent(finish_reason=finish_reason, usage=usage)

    async def _stream_google_genai(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """
        Stream from Google genai (Gemini).

        Google genai is the default/fallback provider in the existing codebase
        and does NOT go through AdalFlow. A fresh GenerativeModel is created
        per call (cheap — no network connection at construction time).

        The SDK's generate_content(stream=True) returns a synchronous iterator.
        We wrap it with _iter_sync_generator() to avoid blocking the event loop
        on each chunk fetch.
        """
        try:
            import google.generativeai as genai  # type: ignore[import]
        except ImportError:
            yield ErrorEvent(
                error="google-generativeai is not installed. "
                      "Run: pip install google-generativeai",
                code="import_error",
            )
            return

        gen_params = self._get_generation_params()
        generation_config = {
            "temperature": gen_params.get("temperature", 0.7),
            "top_p": gen_params.get("top_p", 0.9),
            "top_k": int(gen_params.get("top_k", 40)),
        }

        model_kwargs: dict[str, Any] = {}
        if tools:
            model_kwargs["tools"] = _openai_tools_to_google_tools(tools)

        # Use model name from config (Google uses config model, not request.model)
        model_name = self._model_kwargs.get("model", self.model)
        genai_model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config,
            **model_kwargs,
        )

        contents = _messages_to_google_contents(messages)
        response = genai_model.generate_content(contents, stream=True)

        has_tool_calls = False
        async for chunk in _iter_sync_generator(iter(response)):
            # Text content
            text = getattr(chunk, "text", None)
            if text:
                yield TextDelta(content=text)

            # Function calls (native tool calling)
            parts = getattr(chunk, "parts", None) or []
            for part in parts:
                fc = getattr(part, "function_call", None)
                if fc:
                    # Google genai args may be a MapComposite — convert to dict
                    try:
                        args = dict(fc.args) if fc.args else {}
                    except (TypeError, AttributeError):
                        args = {}
                    has_tool_calls = True
                    yield ToolCallStart(
                        tool_call_id=f"call_{uuid.uuid4().hex[:24]}",
                        tool_name=fc.name or "unknown",
                        tool_args=args,
                    )

        yield FinishEvent(finish_reason="tool_calls" if has_tool_calls else "stop")

    async def _stream_openrouter(
        self,
        messages: list[dict[str, Any]],
    ) -> AsyncGenerator[StreamEvent, None]:
        """
        Stream from OpenRouter.

        OpenRouterClient.acall() internally forces stream=False (openrouter_client.py:139)
        and returns an async generator of pre-extracted text strings. No further
        chunk processing is required.

        OpenRouterClient.convert_inputs_to_api_kwargs() accepts list[dict] as
        input (openrouter_client.py:84) so we pass messages directly.
        """
        gen_params = self._get_generation_params()
        model_kwargs = {
            "model": self.model,
            **gen_params,
        }
        api_kwargs = self._client.convert_inputs_to_api_kwargs(
            input=messages,
            model_kwargs=model_kwargs,
            model_type=ModelType.LLM,
        )

        response = await self._client.acall(
            api_kwargs=api_kwargs, model_type=ModelType.LLM
        )

        async for chunk in response:
            if chunk:
                yield TextDelta(content=chunk)

        yield FinishEvent(finish_reason="stop")

    async def _stream_ollama(
        self,
        messages: list[dict[str, Any]],
    ) -> AsyncGenerator[StreamEvent, None]:
        """
        Stream from Ollama.

        Quirks handled:
          - Appends ' /no_think' to suppress chain-of-thought tokens (Qwen models)
          - Re-nests flat config params back into options:{} (websocket_wiki.py:449-453)
          - Applies 4-level defensive chunk extraction (_extract_ollama_text)
            to handle varied chunk shapes across Ollama server versions

        Note on messages format: OllamaClient.convert_inputs_to_api_kwargs expects
        a string input. We concatenate message contents as a flat prompt, which
        loses role structure but is consistent with the existing implementation.
        If AdalFlow's OllamaClient gains messages support in future, this can be
        upgraded without changing the external interface.
        """
        adjusted = _inject_ollama_no_think(messages)

        # Flatten multi-turn messages to a single prompt string
        prompt_parts = []
        for msg in adjusted:
            role = msg.get("role", "user")
            content = msg.get("content", "") or ""
            if role == "system":
                prompt_parts.append(content)
            elif role == "user":
                prompt_parts.append(f"\n\nHuman: {content}")
            elif role == "assistant":
                prompt_parts.append(f"\n\nAssistant: {content}")
        prompt = "".join(prompt_parts)

        # Re-nest Ollama-specific params into options: {} structure
        options: dict[str, Any] = {}
        for key in ("temperature", "top_p", "num_ctx"):
            if key in self._model_kwargs:
                options[key] = self._model_kwargs[key]

        model_kwargs: dict[str, Any] = {
            "model": self._model_kwargs.get("model", self.model),
            "stream": True,
        }
        if options:
            model_kwargs["options"] = options

        api_kwargs = self._client.convert_inputs_to_api_kwargs(
            input=prompt,
            model_kwargs=model_kwargs,
            model_type=ModelType.LLM,
        )

        response = await self._client.acall(
            api_kwargs=api_kwargs, model_type=ModelType.LLM
        )

        async for chunk in response:
            text = _extract_ollama_text(chunk)
            if text:
                yield TextDelta(content=text)

        yield FinishEvent(finish_reason="stop")

    async def _stream_bedrock(
        self,
        messages: list[dict[str, Any]],
    ) -> AsyncGenerator[StreamEvent, None]:
        """
        Emit a single response from AWS Bedrock (non-streaming).

        BedrockClient.acall() delegates to the synchronous call() which uses
        invoke_model (not invoke_model_with_response_stream). The complete text
        response is wrapped in a single TextDelta event followed by FinishEvent.
        """
        prompt = _messages_to_bedrock_prompt(messages)

        model_kwargs: dict[str, Any] = {"model": self.model}
        for key in ("temperature", "top_p"):
            if key in self._model_kwargs:
                model_kwargs[key] = self._model_kwargs[key]

        api_kwargs = self._client.convert_inputs_to_api_kwargs(
            input=prompt,
            model_kwargs=model_kwargs,
            model_type=ModelType.LLM,
        )

        response = await self._client.acall(
            api_kwargs=api_kwargs, model_type=ModelType.LLM
        )

        text = response if isinstance(response, str) else str(response)
        if text:
            yield TextDelta(content=text)

        yield FinishEvent(finish_reason="stop")
