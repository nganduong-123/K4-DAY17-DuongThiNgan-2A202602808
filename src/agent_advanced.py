from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import CompactMemoryManager, UserProfileStore, estimate_tokens, extract_profile_updates
from model_provider import build_chat_model, normalize_provider


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Agent B: persistent structured profile plus compact per-thread memory."""

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}
        self.langchain_agent = None if force_offline else self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        for key, value in extract_profile_updates(message).items():
            self.profile_store.upsert_fact(user_id, key, value)

        self.compact_memory.append(thread_id, "user", message)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        response = self._offline_response(user_id, thread_id, message)
        response_tokens = estimate_tokens(response)
        self.compact_memory.append(thread_id, "assistant", response)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + response_tokens
        self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens
        return {
            "response": response,
            "agent_tokens": response_tokens,
            "prompt_tokens": prompt_tokens,
            "memory_path": str(self.profile_store.path_for(user_id)),
        }

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        context = self.compact_memory.context(thread_id)
        messages = context["messages"]
        assert isinstance(messages, list)
        parts = [self.profile_store.read_text(user_id), str(context["summary"])]
        parts.extend(str(item["content"]) for item in messages)
        return sum(estimate_tokens(part) for part in parts)

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        facts = self.profile_store.facts(user_id)
        lower = message.casefold()
        recall_markers = (
            "?",
            "nhắc lại",
            "tóm tắt",
            "tên mình",
            "style trả lời",
            "đồ uống",
            "món ăn yêu thích",
            "nghề hiện tại",
            "nơi ở hiện tại",
        )
        if not any(marker in lower for marker in recall_markers):
            return "Mình đã cập nhật các thông tin ổn định và vẫn giữ ngữ cảnh gần nhất."
        if not facts:
            return "Mình chưa có thông tin bền vững để trả lời câu hỏi này."

        labels = {
            "name": "Tên",
            "profession": "nghề hiện tại",
            "location": "nơi ở hiện tại",
            "favorite_drink": "đồ uống yêu thích",
            "favorite_food": "món ăn yêu thích",
            "pet": "thú cưng",
            "response_style": "style trả lời",
            "interests": "mối quan tâm kỹ thuật",
        }
        ordered = [
            "name",
            "profession",
            "location",
            "favorite_drink",
            "favorite_food",
            "pet",
            "response_style",
            "interests",
        ]
        # Returning the full small profile makes cross-session recall robust,
        # while the schema keeps this bounded and auditable.
        return "; ".join(f"{labels[key]}: {facts[key]}" for key in ordered if key in facts) + "."

    def _maybe_build_langchain_agent(self):
        provider = normalize_provider(self.config.model.provider)
        if provider not in {"ollama", "custom"} and not self.config.model.api_key:
            return None
        try:
            return build_chat_model(self.config.model)
        except (RuntimeError, ValueError):
            return None
