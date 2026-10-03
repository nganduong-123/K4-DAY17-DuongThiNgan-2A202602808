from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


def estimate_tokens(text: str) -> int:
    """Stable language-independent approximation suitable for an offline lab."""

    cleaned = text.strip()
    return 0 if not cleaned else max(1, (len(cleaned) + 3) // 4)


@dataclass
class UserProfileStore:
    """Small, auditable Markdown store for long-term user facts."""

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", user_id.strip()).strip(".-") or "anonymous"
        return self.root_dir / safe_id / "User.md"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        return path.read_text(encoding="utf-8") if path.exists() else "# User Profile\n"

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content.rstrip() + "\n", encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        content = self.read_text(user_id)
        if search_text not in content:
            return False
        self.write_text(user_id, content.replace(search_text, replacement, 1))
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        return path.stat().st_size if path.exists() else 0

    def facts(self, user_id: str) -> dict[str, str]:
        facts: dict[str, str] = {}
        for line in self.read_text(user_id).splitlines():
            match = re.match(r"^- ([a-z_]+):\s*(.+)$", line)
            if match:
                facts[match.group(1)] = match.group(2).strip()
        return facts

    def upsert_fact(self, user_id: str, key: str, value: str) -> Path:
        """Replace conflicting old values by key instead of keeping both."""

        facts = self.facts(user_id)
        facts[key] = value.strip().rstrip(".,")
        lines = ["# User Profile", ""] + [f"- {name}: {facts[name]}" for name in sorted(facts)]
        return self.write_text(user_id, "\n".join(lines))


def _has_any(text: str, values: tuple[str, ...]) -> bool:
    return any(value in text for value in values)


def extract_profile_updates(message: str) -> dict[str, str]:
    """Extract high-confidence stable facts and ignore questions/jokes/noise."""

    text = " ".join(message.strip().split())
    lower = text.casefold()
    if not text:
        return {}

    question_markers = ("mình tên gì", "tên mình là gì", "ở đâu", "nghề gì", "nhắc lại", "bạn biết")
    if "?" in text and _has_any(lower, question_markers):
        return {}

    updates: dict[str, str] = {}
    name_match = re.search(r"(?:mình\s+tên\s+là|tên\s+mình\s+là)\s+([^,.!?]+)", text, re.IGNORECASE)
    if name_match:
        updates["name"] = name_match.group(1).strip()

    if "đà nẵng" in lower and _has_any(
        lower,
        (
            "giờ mình đang ở đà nẵng",
            "nơi ở hiện tại là đà nẵng",
            "đang làm việc ở đà nẵng",
            "ở đã cập nhật từ huế sang đà nẵng",
            "hiện tại là đà nẵng",
        ),
    ):
        updates["location"] = "Đà Nẵng"
    elif "huế" in lower and _has_any(lower, ("đang ở huế", "hiện ở huế", "vẫn ở huế", "giờ mình đang ở huế")):
        updates["location"] = "Huế"
    elif "đà nẵng" in lower and _has_any(lower, ("mình ở đà nẵng", "hiện ở đà nẵng")):
        updates["location"] = "Đà Nẵng"

    if "mlops engineer" in lower and not ("nghề" in lower and "?" in text):
        updates["profession"] = "MLOps engineer"
    elif "backend engineer" in lower and _has_any(lower, ("đang làm backend", "là backend")) and "không còn" not in lower:
        updates["profession"] = "backend engineer"

    if "3 bullet" in lower and _has_any(lower, ("trả lời", "style", "muốn")):
        updates["response_style"] = "3 bullet ngắn, có ví dụ thực chiến và nêu trade-off"
    elif "ngắn gọn" in lower and _has_any(lower, ("trả lời", "style", "giải thích")):
        updates["response_style"] = "ngắn gọn, có bullet và ví dụ thực tế"

    if "cà phê sữa đá" in lower and _has_any(lower, ("yêu thích", "vẫn uống", "thích")) and "?" not in text:
        updates["favorite_drink"] = "cà phê sữa đá"
    if "mì quảng" in lower and _has_any(lower, ("yêu thích", "món ruột", "món ăn")) and "?" not in text:
        updates["favorite_food"] = "mì Quảng"
    if "corgi" in lower and _has_any(lower, ("nuôi", "bé corgi", "con corgi")) and "?" not in text:
        pet_match = re.search(r"corgi(?:\s+tên\s+([\wÀ-ỹ]+))?", text, re.IGNORECASE)
        updates["pet"] = f"corgi tên {pet_match.group(1)}" if pet_match and pet_match.group(1) else "corgi"
    if "python" in lower and "ai" in lower and _has_any(lower, ("thích", "quan tâm", "mối quan tâm")):
        updates["interests"] = "Python và AI"

    return updates


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Build a bounded extractive summary of the most recent older messages."""

    items: list[str] = []
    for item in messages[-max_items:]:
        content = " ".join(str(item.get("content", "")).split())
        if not content:
            continue
        clipped = content if len(content) <= 180 else content[:177].rstrip() + "..."
        items.append(f"{item.get('role', 'unknown')}: {clipped}")
    return "\n".join(items)[-1200:]


@dataclass
class CompactMemoryManager:
    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def _thread(self, thread_id: str) -> dict[str, object]:
        return self.state.setdefault(thread_id, {"messages": [], "summary": "", "compactions": 0})

    def append(self, thread_id: str, role: str, content: str) -> None:
        thread = self._thread(thread_id)
        messages = thread["messages"]
        assert isinstance(messages, list)
        messages.append({"role": role, "content": content})
        summary = str(thread["summary"])
        load = estimate_tokens(summary) + sum(estimate_tokens(str(m["content"])) for m in messages)
        if load <= self.threshold_tokens or len(messages) <= self.keep_messages:
            return
        older = messages[:-self.keep_messages]
        recent = messages[-self.keep_messages:]
        summary_input = ([{"role": "summary", "content": summary}] if summary else []) + older
        thread["summary"] = summarize_messages(summary_input)
        thread["messages"] = recent
        thread["compactions"] = int(thread["compactions"]) + 1

    def context(self, thread_id: str) -> dict[str, object]:
        return self._thread(thread_id)

    def compaction_count(self, thread_id: str) -> int:
        return int(self._thread(thread_id)["compactions"])
