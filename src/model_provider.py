from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProviderConfig:
    """Configuration shared by every supported chat-model provider."""

    provider: str
    model_name: str
    temperature: float = 0.0
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    """Return a canonical provider name and accept common aliases/typos."""

    normalized = value.strip().lower().replace("-", "_")
    aliases = {
        "anthorpic": "anthropic",
        "claude": "anthropic",
        "google": "gemini",
        "google_genai": "gemini",
        "open_router": "openrouter",
        "openai_compatible": "custom",
    }
    normalized = aliases.get(normalized, normalized)
    supported = {"openai", "custom", "gemini", "anthropic", "ollama", "openrouter"}
    if normalized not in supported:
        raise ValueError(f"Unsupported provider: {value!r}. Choose one of {sorted(supported)}")
    return normalized


def build_chat_model(config: ProviderConfig):
    """Build a provider model while keeping optional SDK imports lazy."""

    provider = normalize_provider(config.provider)
    common = {"model": config.model_name, "temperature": config.temperature}

    if provider in {"openai", "custom"}:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:  # pragma: no cover - live mode only
            raise RuntimeError("Install langchain-openai to use this provider") from exc
        if config.api_key:
            common["api_key"] = config.api_key
        if provider == "custom":
            if not config.base_url:
                raise ValueError("CUSTOM_BASE_URL is required for the custom provider")
            common["base_url"] = config.base_url
        return ChatOpenAI(**common)

    if provider == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install langchain-google-genai to use Gemini") from exc
        if config.api_key:
            common["google_api_key"] = config.api_key
        return ChatGoogleGenerativeAI(**common)

    if provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install langchain-anthropic to use Anthropic") from exc
        if config.api_key:
            common["api_key"] = config.api_key
        return ChatAnthropic(**common)

    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install langchain-ollama to use Ollama") from exc
        if config.base_url:
            common["base_url"] = config.base_url
        return ChatOllama(**common)

    try:
        from langchain_openrouter import ChatOpenRouter
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install langchain-openrouter to use OpenRouter") from exc
    if config.api_key:
        common["api_key"] = config.api_key
    return ChatOpenRouter(**common)
