from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}")
    return data


def recall_points(answer: str, expected: list[str]) -> float:
    if not expected:
        return 1.0
    folded = answer.casefold()
    hits = sum(value.casefold() in folded for value in expected)
    if hits == 0:
        return 0.0
    return 1.0 if hits == len(expected) else 0.5


def heuristic_quality(answer: str, expected: list[str]) -> float:
    recall = recall_points(answer, expected)
    if not answer.strip():
        return 0.0
    concise_bonus = 0.1 if len(answer) <= 500 else 0.0
    return min(1.0, round(recall * 0.9 + concise_bonus, 3))


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config) -> BenchmarkRow:
    thread_ids: set[str] = set()
    user_ids: set[str] = set()
    initial_sizes: dict[str, int] = {}
    scores: list[float] = []
    qualities: list[float] = []
    initial_size = 0

    for conversation in conversations:
        user_id = str(conversation["user_id"])
        thread_id = str(conversation["id"])
        user_ids.add(user_id)
        thread_ids.add(thread_id)
        if hasattr(agent, "memory_file_size") and user_id not in initial_sizes:
            initial_sizes[user_id] = agent.memory_file_size(user_id)
        for turn in conversation["turns"]:
            agent.reply(user_id, thread_id, str(turn))

        for index, recall in enumerate(conversation.get("recall_questions", []), start=1):
            recall_thread = f"{thread_id}-recall-{index}"
            thread_ids.add(recall_thread)
            result = agent.reply(user_id, recall_thread, str(recall["question"]))
            answer = str(result["response"])
            expected = [str(value) for value in recall["expected_contains"]]
            scores.append(recall_points(answer, expected))
            qualities.append(heuristic_quality(answer, expected))

    initial_size = sum(initial_sizes.values())
    final_size = sum(agent.memory_file_size(user_id) for user_id in user_ids) if hasattr(agent, "memory_file_size") else 0
    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=sum(agent.token_usage(thread_id) for thread_id in thread_ids),
        prompt_tokens_processed=sum(agent.prompt_token_usage(thread_id) for thread_id in thread_ids),
        recall_score=round(sum(scores) / len(scores), 3) if scores else 0.0,
        response_quality=round(sum(qualities) / len(qualities), 3) if qualities else 0.0,
        memory_growth_bytes=max(0, final_size - initial_size),
        compactions=sum(agent.compaction_count(thread_id) for thread_id in thread_ids),
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    headers = [
        "Agent",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions",
    ]
    values = [
        [
            row.agent_name,
            row.agent_tokens_only,
            row.prompt_tokens_processed,
            f"{row.recall_score:.3f}",
            f"{row.response_quality:.3f}",
            row.memory_growth_bytes,
            row.compactions,
        ]
        for row in rows
    ]
    try:
        from tabulate import tabulate

        return tabulate(values, headers=headers, tablefmt="github")
    except ImportError:
        widths = [max(len(str(row[i])) for row in [headers, *values]) for i in range(len(headers))]
        render = lambda row: "| " + " | ".join(str(value).ljust(widths[i]) for i, value in enumerate(row)) + " |"
        return "\n".join([render(headers), render(["-" * width for width in widths]), *(render(row) for row in values)])


def _run_suite(title: str, conversations: list[dict[str, Any]], config) -> None:
    suite_slug = title.lower().replace(" ", "-")
    rows: list[BenchmarkRow] = []
    for name, agent_type in (("Baseline", BaselineAgent), ("Advanced", AdvancedAgent)):
        state_dir = config.state_dir / suite_slug / name.lower()
        shutil.rmtree(state_dir, ignore_errors=True)
        agent_config = replace(config, state_dir=state_dir)
        agent = agent_type(agent_config, force_offline=True)
        rows.append(run_agent_benchmark(name, agent, conversations, agent_config))
    print(f"\n## {title}\n")
    print(format_rows(rows))


def main() -> None:
    config = load_config(Path(__file__).resolve().parent.parent)
    standard = load_conversations(config.data_dir / "conversations.json")
    stress = load_conversations(config.data_dir / "advanced_long_context.json")
    _run_suite("Standard Benchmark", standard, config)
    _run_suite("Long-Context Stress Benchmark", stress, config)


if __name__ == "__main__":
    main()
