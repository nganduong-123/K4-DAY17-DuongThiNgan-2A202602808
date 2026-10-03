from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config
from memory_store import UserProfileStore


def make_config(tmp_path: Path):
    config = load_config(Path(__file__).resolve().parent.parent)
    return replace(config, state_dir=tmp_path / "state", compact_threshold_tokens=80, compact_keep_messages=2)


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")
    path = store.write_text("user/unsafe", "# User Profile\n\n- name: Ngân")
    assert path.exists()
    assert "Ngân" in store.read_text("user/unsafe")
    assert store.edit_text("user/unsafe", "Ngân", "Dương Thị Ngân")
    assert "Dương Thị Ngân" in store.read_text("user/unsafe")
    assert not store.edit_text("user/unsafe", "không tồn tại", "x")
    store.upsert_fact("user/unsafe", "location", "Đà Nẵng")
    store.upsert_fact("user/unsafe", "location", "Huế")
    assert store.facts("user/unsafe")["location"] == "Huế"
    assert "Đà Nẵng" not in store.read_text("user/unsafe")


def test_compact_trigger(tmp_path: Path) -> None:
    agent = AdvancedAgent(make_config(tmp_path), force_offline=True)
    for index in range(8):
        agent.reply("ngan", "long-thread", f"Lượt {index}: " + "nội dung dài " * 30)
    context = agent.compact_memory.context("long-thread")
    assert agent.compaction_count("long-thread") > 0
    assert context["summary"]
    assert len(context["messages"]) <= 2


def test_cross_session_recall(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    advanced = AdvancedAgent(config, force_offline=True)
    baseline = BaselineAgent(config, force_offline=True)
    fact = "Mình tên là Ngân và hiện tại mình đang ở Huế."
    advanced.reply("ngan", "thread-a", fact)
    baseline.reply("ngan", "thread-a", fact)
    advanced_answer = advanced.reply("ngan", "thread-b", "Mình tên gì và hiện đang ở đâu?")["response"]
    baseline_answer = baseline.reply("ngan", "thread-b", "Mình tên gì và hiện đang ở đâu?")["response"]
    assert "Ngân" in advanced_answer and "Huế" in advanced_answer
    assert "Ngân" not in baseline_answer and "Huế" not in baseline_answer


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)
    for index in range(12):
        message = f"Lượt {index}: " + "đây là ngữ cảnh stress benchmark rất dài " * 35
        baseline.reply("ngan", "stress", message)
        advanced.reply("ngan", "stress", message)
    assert advanced.compaction_count("stress") > 0
    assert advanced.prompt_token_usage("stress") < baseline.prompt_token_usage("stress")
