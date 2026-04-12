"""
Agent configuration for the DeepWiki agent loop.

Defines AgentConfig (allowed tools, model preferences, system prompt, step limit)
and a simple module-level registry of built-in agent types.

Three built-in agents:
    wiki          -- general wiki Q&A, all tools, 25 steps (primary)
    explore       -- read-only code exploration, 15 steps (subagent)
    deep-research -- thorough multi-step research, all tools, 40 steps (primary)

Usage:
    from api.agent.config import get_agent_config, get_tools_for_agent

    config = get_agent_config("wiki")
    tools = get_tools_for_agent(config, repo_path="/path/to/repo")
    provider = UnifiedProvider(
        config.model_override.provider or request_provider,
        config.model_override.model or request_model,
    )
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from api.tools.task import AgentInfo, AgentMode
from api.tools.tool import Tool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool name sets for built-in agents.
# Hardcoded rather than derived from _TOOL_CLASSES.keys() to avoid import-time
# coupling. Mismatches are caught at runtime by get_tools_for_agent() (warning).
# ---------------------------------------------------------------------------
_ALL_TOOLS: tuple[str, ...] = ("bash", "grep", "glob", "ls", "read", "task", "todowrite")
_READ_ONLY_TOOLS: tuple[str, ...] = ("grep", "glob", "ls", "read")


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class ModelOverride(BaseModel):
    """Optional per-agent model/provider override.

    Both fields are optional so callers can override only the model while
    keeping the request-level provider, or vice versa.
    Resolution logic lives in the agent loop (subtask 4), not here.
    """

    model_config = ConfigDict(frozen=True)

    provider: Optional[str] = None
    model: Optional[str] = None


class AgentConfig(BaseModel):
    """Configuration for a single agent type.

    Fields:
        name                  Unique identifier (e.g., "wiki", "explore").
        description           Human-readable summary used in TaskTool descriptions.
        mode                  "primary" | "subagent" | "all". Primary agents are
                              excluded from the TaskTool subagent list.
        system_prompt_template  Format string with {repo_type}, {repo_url},
                              {repo_name}, {language_name} placeholders.
        allowed_tools         Tool names the agent may call. Filtered from
                              _TOOL_CLASSES at runtime by get_tools_for_agent().
        max_steps             Maximum agent loop iterations before forced summary.
        model_override        Optional provider/model override for this agent.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    mode: AgentMode
    system_prompt_template: str
    allowed_tools: tuple[str, ...] = Field(default=())
    max_steps: int = 25
    model_override: Optional[ModelOverride] = None

    @field_validator("allowed_tools", mode="before")
    @classmethod
    def _coerce_tools_to_tuple(cls, value: Any) -> tuple[str, ...]:
        if isinstance(value, (list, tuple)):
            return tuple(value)
        raise ValueError(f"allowed_tools must be a list or tuple, got {type(value).__name__}")

    @field_validator("max_steps")
    @classmethod
    def _validate_max_steps(cls, value: int) -> int:
        if not (1 <= value <= 200):
            raise ValueError(f"max_steps must be between 1 and 200, got {value}")
        return value

    def to_agent_info(self) -> AgentInfo:
        """Project to AgentInfo for use with TaskTool.with_agents()."""
        return AgentInfo(name=self.name, description=self.description, mode=self.mode)


# ---------------------------------------------------------------------------
# Tool instantiation
# ---------------------------------------------------------------------------

def get_tools_for_agent(agent_config: AgentConfig, repo_path: str) -> dict[str, Tool]:
    """Return tool instances the agent is permitted to use.

    Skips tool names not yet registered in _TOOL_CLASSES (logs a warning).
    This allows configs to reference future tools (e.g., "rag_search" before
    subtask 5 registers it) without breaking at config load time.

    Note: the "task" tool is returned as a bare TaskTool instance without
    agents or an executor bound. The agent loop (subtask 4) is responsible
    for calling TaskTool.with_agents(...).with_executor(...) before the tool
    becomes functional. Callers that bypass the agent loop will receive a
    task tool that returns metadata.error == "no_executor" on every call.

    Args:
        agent_config: The agent's configuration.
        repo_path:    Absolute path to the repository root.

    Returns:
        Mapping of tool name -> instantiated Tool object.
    """
    # Lazy import to avoid circular dependency:
    # api.tools.__init__ imports from api.tools.task; config.py already imports
    # from api.tools.task at module level, so a top-level import of api.tools
    # here would be safe today -- but the lazy form is more defensive.
    from api.tools import _TOOL_CLASSES  # type: ignore[attr-defined]

    result: dict[str, Tool] = {}
    for tool_name in agent_config.allowed_tools:
        if tool_name not in _TOOL_CLASSES:
            logger.warning(
                "Agent '%s' lists tool '%s' which is not registered in _TOOL_CLASSES. "
                "Skipping. (Will be available once the tool is registered.)",
                agent_config.name,
                tool_name,
            )
            continue
        result[tool_name] = _TOOL_CLASSES[tool_name](repo_path)
    return result


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_AGENT_CONFIGS: dict[str, AgentConfig] = {}


def register_agent(config: AgentConfig) -> None:
    """Register an agent configuration by name.

    Replaces any existing registration for the same name with a warning.
    """
    if config.name in _AGENT_CONFIGS:
        logger.warning(
            "Replacing existing agent config for '%s'.", config.name
        )
    _AGENT_CONFIGS[config.name] = config


def get_agent_config(name: str) -> AgentConfig:
    """Look up an agent by name.

    Raises:
        KeyError: if the name is not registered.
    """
    if name not in _AGENT_CONFIGS:
        raise KeyError(
            f"Unknown agent '{name}'. Registered agents: {list(_AGENT_CONFIGS)}"
        )
    return _AGENT_CONFIGS[name]


def get_all_agent_configs() -> dict[str, AgentConfig]:
    """Return a shallow copy of the full agent registry."""
    return dict(_AGENT_CONFIGS)


def get_all_agent_infos() -> list[AgentInfo]:
    """Return AgentInfo for all registered agents.

    Convenience wrapper for TaskTool.with_agents().
    """
    return [config.to_agent_info() for config in _AGENT_CONFIGS.values()]


# ---------------------------------------------------------------------------
# Built-in agent registration
# ---------------------------------------------------------------------------

def _register_defaults() -> None:
    """Register the three built-in agent configurations.

    Called once at module import time. Prompts are lazy-imported here to avoid
    a circular dependency if api.prompts ever imports from api.agent.
    """
    from api.prompts import AGENT_SYSTEM_PROMPT, DEEP_RESEARCH_AGENT_SYSTEM_PROMPT, EXPLORE_AGENT_SYSTEM_PROMPT  # noqa: PLC0415

    register_agent(AgentConfig(
        name="wiki",
        description="General wiki Q&A: answers questions about the codebase using all available tools.",
        mode="primary",
        system_prompt_template=AGENT_SYSTEM_PROMPT,
        allowed_tools=_ALL_TOOLS,
        max_steps=25,
    ))

    register_agent(AgentConfig(
        name="explore",
        description="Read-only code exploration: searches and reads files without executing commands.",
        mode="subagent",
        system_prompt_template=EXPLORE_AGENT_SYSTEM_PROMPT,
        allowed_tools=_READ_ONLY_TOOLS,
        max_steps=15,
    ))

    register_agent(AgentConfig(
        name="deep-research",
        description="Deep research: thorough multi-step investigation using all tools, with a higher step budget.",
        mode="primary",
        system_prompt_template=DEEP_RESEARCH_AGENT_SYSTEM_PROMPT,
        allowed_tools=_ALL_TOOLS,
        max_steps=40,
    ))


_register_defaults()
