from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from model_provider import ProviderConfig, normalize_provider


@dataclass
class LabConfig:
    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


def _provider_config(prefix: str = "LLM") -> ProviderConfig:
    provider = normalize_provider(os.getenv(f"{prefix}_PROVIDER", "openai"))
    default_models = {
        "openai": "gpt-4o-mini",
        "custom": "gpt-4o-mini",
        "gemini": "gemini-2.0-flash",
        "anthropic": "claude-3-5-haiku-latest",
        "ollama": "llama3.2",
        "openrouter": "openai/gpt-4o-mini",
    }
    key_by_provider = {
        "openai": "OPENAI_API_KEY",
        "custom": "CUSTOM_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }
    base_by_provider = {"custom": "CUSTOM_BASE_URL", "ollama": "OLLAMA_BASE_URL"}
    key_name = key_by_provider.get(provider)
    base_name = base_by_provider.get(provider)
    return ProviderConfig(
        provider=provider,
        model_name=os.getenv(f"{prefix}_MODEL", default_models[provider]),
        temperature=float(os.getenv(f"{prefix}_TEMPERATURE", "0")),
        api_key=os.getenv(key_name) if key_name else None,
        base_url=os.getenv(base_name) if base_name else None,
    )


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load optional local environment values and create runtime directories."""

    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()
    try:
        from dotenv import load_dotenv

        load_dotenv(root / ".env")
    except ImportError:
        pass

    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    model = _provider_config("LLM")
    judge = _provider_config("JUDGE") if os.getenv("JUDGE_PROVIDER") else model
    return LabConfig(
        base_dir=root,
        data_dir=root / "data",
        state_dir=state_dir,
        compact_threshold_tokens=max(100, int(os.getenv("COMPACT_THRESHOLD_TOKENS", "1200"))),
        compact_keep_messages=max(2, int(os.getenv("COMPACT_KEEP_MESSAGES", "4"))),
        model=model,
        judge_model=judge,
    )
