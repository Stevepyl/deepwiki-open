from api import config as config_module
from api.agent.chat_handler import AgentChatRequest
from api.simple_chat import ChatCompletionRequest as HttpChatRequest
from api.websocket_wiki import ChatCompletionRequest as WebSocketChatRequest


def test_generator_env_overrides_default_openai_chat_model(monkeypatch):
    monkeypatch.setenv("CHAT_MODEL", "qwen3.6-35b-a3b")
    generator_config = {
        "default_provider": "google",
        "providers": {
            "google": {
                "default_model": "gemini-2.5-flash",
                "models": {"gemini-2.5-flash": {"temperature": 1.0}},
            },
            "openai": {
                "default_model": "gpt-5-nano",
                "models": {"gpt-5-nano": {"temperature": 1.0}},
            },
        },
    }

    updated = config_module.apply_generator_env_overrides(generator_config)

    assert updated["default_provider"] == "openai"
    openai_config = updated["providers"]["openai"]
    assert openai_config["default_model"] == "qwen3.6-35b-a3b"
    assert openai_config["models"]["qwen3.6-35b-a3b"] == {"temperature": 1.0}


def test_embedder_env_overrides_openai_embedding_model(monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-v4")
    embedder_config = {
        "embedder": {
            "client_class": "OpenAIClient",
            "model_kwargs": {
                "model": "text-embedding-3-small",
                "dimensions": 256,
            },
        }
    }

    updated = config_module.apply_embedder_env_overrides(embedder_config)

    assert updated["embedder"]["model_kwargs"]["model"] == "text-embedding-v4"
    assert updated["embedder"]["model_kwargs"]["dimensions"] == 256


def test_streaming_request_defaults_use_openai_provider():
    assert HttpChatRequest.model_fields["provider"].default == "openai"
    assert WebSocketChatRequest.model_fields["provider"].default == "openai"
    assert AgentChatRequest.model_fields["provider"].default == "openai"
