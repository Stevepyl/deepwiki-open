"""
Unit tests for api/agent/config.py.

Tests cover:
- ModelOverride fields and immutability
- AgentConfig construction, validation, and coercion
- AgentConfig.to_agent_info()
- get_tools_for_agent() filtering and unknown-tool handling
- Registry functions (register_agent, get_agent_config, get_all_agent_configs, get_all_agent_infos)
- Built-in agent properties (wiki, explore, deep-research)
"""

import logging

import pytest
from pydantic import ValidationError

from api.agent.config import (
    AgentConfig,
    ModelOverride,
    _AGENT_CONFIGS,
    _ALL_TOOLS,
    _READ_ONLY_TOOLS,
    _register_defaults,
    get_agent_config,
    get_all_agent_configs,
    get_all_agent_infos,
    get_tools_for_agent,
    register_agent,
)
from api.tools.task import AgentInfo


# ---------------------------------------------------------------------------
# Fixture: reset registry around every test for isolation
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_registry():
    """Clear and re-register defaults before and after each test."""
    _AGENT_CONFIGS.clear()
    _register_defaults()
    yield
    _AGENT_CONFIGS.clear()
    _register_defaults()


# ---------------------------------------------------------------------------
# ModelOverride
# ---------------------------------------------------------------------------

class TestModelOverride:
    def test_defaults_are_none(self):
        mo = ModelOverride()
        assert mo.provider is None
        assert mo.model is None

    def test_partial_override_provider_only(self):
        mo = ModelOverride(provider="openai")
        assert mo.provider == "openai"
        assert mo.model is None

    def test_full_override(self):
        mo = ModelOverride(provider="openai", model="gpt-4o")
        assert mo.provider == "openai"
        assert mo.model == "gpt-4o"

    def test_is_frozen(self):
        mo = ModelOverride(provider="openai")
        with pytest.raises((ValidationError, TypeError)):
            mo.provider = "google"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# AgentConfig construction
# ---------------------------------------------------------------------------

class TestAgentConfigConstruction:
    def test_required_fields_and_defaults(self):
        config = AgentConfig(
            name="test",
            description="A test agent",
            mode="subagent",
            system_prompt_template="You are {repo_name}.",
        )
        assert config.name == "test"
        assert config.description == "A test agent"
        assert config.mode == "subagent"
        assert config.allowed_tools == ()
        assert config.max_steps == 25
        assert config.model_override is None

    def test_list_coerced_to_tuple(self):
        config = AgentConfig(
            name="test",
            description="desc",
            mode="subagent",
            system_prompt_template="prompt",
            allowed_tools=["grep", "read"],
        )
        assert isinstance(config.allowed_tools, tuple)
        assert config.allowed_tools == ("grep", "read")

    def test_tuple_stays_tuple(self):
        config = AgentConfig(
            name="test",
            description="desc",
            mode="subagent",
            system_prompt_template="prompt",
            allowed_tools=("grep",),
        )
        assert config.allowed_tools == ("grep",)

    def test_max_steps_zero_raises(self):
        with pytest.raises(ValidationError):
            AgentConfig(
                name="test",
                description="desc",
                mode="subagent",
                system_prompt_template="prompt",
                max_steps=0,
            )

    def test_max_steps_over_limit_raises(self):
        with pytest.raises(ValidationError):
            AgentConfig(
                name="test",
                description="desc",
                mode="subagent",
                system_prompt_template="prompt",
                max_steps=201,
            )

    def test_max_steps_boundary_values_accepted(self):
        low = AgentConfig(
            name="t1",
            description="d",
            mode="subagent",
            system_prompt_template="p",
            max_steps=1,
        )
        high = AgentConfig(
            name="t2",
            description="d",
            mode="subagent",
            system_prompt_template="p",
            max_steps=200,
        )
        assert low.max_steps == 1
        assert high.max_steps == 200

    def test_is_frozen(self):
        config = AgentConfig(
            name="test",
            description="desc",
            mode="subagent",
            system_prompt_template="prompt",
        )
        with pytest.raises((ValidationError, TypeError)):
            config.name = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# to_agent_info()
# ---------------------------------------------------------------------------

class TestToAgentInfo:
    def test_fields_mapped_correctly(self):
        config = AgentConfig(
            name="explorer",
            description="Explores code",
            mode="subagent",
            system_prompt_template="prompt",
        )
        info = config.to_agent_info()
        assert info.name == "explorer"
        assert info.description == "Explores code"
        assert info.mode == "subagent"

    def test_returns_agent_info_type(self):
        config = AgentConfig(
            name="explorer",
            description="desc",
            mode="subagent",
            system_prompt_template="prompt",
        )
        assert isinstance(config.to_agent_info(), AgentInfo)

    def test_primary_mode_preserved(self):
        config = AgentConfig(
            name="main",
            description="Primary agent",
            mode="primary",
            system_prompt_template="prompt",
        )
        assert config.to_agent_info().mode == "primary"


# ---------------------------------------------------------------------------
# get_tools_for_agent()
# ---------------------------------------------------------------------------

FAKE_REPO = "/fake/repo"


class TestGetToolsForAgent:
    def test_read_only_config_returns_four_tools(self):
        config = AgentConfig(
            name="explore",
            description="read only",
            mode="subagent",
            system_prompt_template="prompt",
            allowed_tools=_READ_ONLY_TOOLS,
        )
        tools = get_tools_for_agent(config, FAKE_REPO)
        assert set(tools.keys()) == {"grep", "glob", "ls", "read"}

    def test_all_tools_config_returns_seven_tools(self):
        config = AgentConfig(
            name="wiki",
            description="all tools",
            mode="primary",
            system_prompt_template="prompt",
            allowed_tools=_ALL_TOOLS,
        )
        tools = get_tools_for_agent(config, FAKE_REPO)
        assert set(tools.keys()) == set(_ALL_TOOLS)
        assert len(tools) == 7

    def test_empty_allowed_tools_returns_empty_dict(self):
        config = AgentConfig(
            name="no_tools",
            description="no tools",
            mode="subagent",
            system_prompt_template="prompt",
            allowed_tools=(),
        )
        tools = get_tools_for_agent(config, FAKE_REPO)
        assert tools == {}

    def test_unknown_tool_skipped_with_warning(self, caplog):
        config = AgentConfig(
            name="future",
            description="future tools",
            mode="subagent",
            system_prompt_template="prompt",
            allowed_tools=("grep", "rag_search"),  # rag_search not yet registered
        )
        with caplog.at_level(logging.WARNING, logger="api.agent.config"):
            tools = get_tools_for_agent(config, FAKE_REPO)

        assert "rag_search" in tools or "rag_search" not in tools  # doesn't raise
        assert "rag_search" not in tools
        assert "grep" in tools
        assert any("rag_search" in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# Registry functions
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_default_agents_registered(self):
        assert "wiki" in _AGENT_CONFIGS
        assert "explore" in _AGENT_CONFIGS
        assert "deep-research" in _AGENT_CONFIGS

    def test_get_agent_config_returns_correct_config(self):
        config = get_agent_config("wiki")
        assert config.name == "wiki"
        assert isinstance(config, AgentConfig)

    def test_get_agent_config_unknown_raises_key_error(self):
        with pytest.raises(KeyError, match="unknown-agent"):
            get_agent_config("unknown-agent")

    def test_register_agent_adds_custom(self):
        custom = AgentConfig(
            name="custom",
            description="Custom agent",
            mode="subagent",
            system_prompt_template="prompt",
        )
        register_agent(custom)
        assert get_agent_config("custom") is custom

    def test_register_agent_replaces_existing_with_warning(self, caplog):
        replacement = AgentConfig(
            name="wiki",
            description="Replacement wiki",
            mode="primary",
            system_prompt_template="new prompt",
        )
        with caplog.at_level(logging.WARNING, logger="api.agent.config"):
            register_agent(replacement)

        assert get_agent_config("wiki").description == "Replacement wiki"
        assert any("wiki" in record.message for record in caplog.records)

    def test_get_all_agent_configs_returns_copy(self):
        all_configs = get_all_agent_configs()
        all_configs["injected"] = None  # type: ignore[assignment]
        assert "injected" not in _AGENT_CONFIGS

    def test_get_all_agent_infos_returns_agent_info_list(self):
        infos = get_all_agent_infos()
        assert isinstance(infos, list)
        assert all(isinstance(i, AgentInfo) for i in infos)
        names = {i.name for i in infos}
        assert {"wiki", "explore", "deep-research"}.issubset(names)


# ---------------------------------------------------------------------------
# Built-in agent properties
# ---------------------------------------------------------------------------

class TestBuiltInAgents:
    def test_wiki_agent(self):
        config = get_agent_config("wiki")
        assert config.mode == "primary"
        assert config.max_steps == 25
        assert set(config.allowed_tools) == set(_ALL_TOOLS)

    def test_explore_agent(self):
        config = get_agent_config("explore")
        assert config.mode == "subagent"
        assert config.max_steps == 15
        assert set(config.allowed_tools) == set(_READ_ONLY_TOOLS)
        # Explore must not include write/exec tools
        assert "bash" not in config.allowed_tools
        assert "task" not in config.allowed_tools
        assert "todowrite" not in config.allowed_tools

    def test_deep_research_agent(self):
        config = get_agent_config("deep-research")
        assert config.mode == "primary"
        assert config.max_steps == 40
        assert set(config.allowed_tools) == set(_ALL_TOOLS)

    def test_deep_research_has_higher_steps_than_wiki(self):
        assert get_agent_config("deep-research").max_steps > get_agent_config("wiki").max_steps

    def test_explore_has_lower_steps_than_wiki(self):
        assert get_agent_config("explore").max_steps < get_agent_config("wiki").max_steps


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_agent_config_roundtrip(self):
        config = get_agent_config("explore")
        data = config.model_dump()
        restored = AgentConfig.model_validate(data)
        assert restored == config

    def test_agent_config_with_model_override_roundtrip(self):
        config = AgentConfig(
            name="custom",
            description="custom",
            mode="subagent",
            system_prompt_template="prompt",
            model_override=ModelOverride(provider="openai", model="gpt-4o"),
        )
        data = config.model_dump()
        restored = AgentConfig.model_validate(data)
        assert restored.model_override is not None
        assert restored.model_override.provider == "openai"
        assert restored.model_override.model == "gpt-4o"
