from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from llm_chat_archive.source_roots import normalize_source_root_platform
from llm_chat_archive.sources.codex_cli import CodexCliCollector, parse_rollout_file

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "codex_cli"
CURRENT_PLATFORM = normalize_source_root_platform().value


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "llm_chat_archive", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_parse_rollout_file_filters_noise_and_keeps_minimal_provenance() -> None:
    rollout_path = (
        FIXTURE_ROOT
        / "sessions"
        / "2026"
        / "03"
        / "14"
        / "rollout-20260314T090000-session-active.jsonl"
    )

    conversation = parse_rollout_file(rollout_path, collected_at="2026-03-19T00:00:00Z")

    assert conversation is not None
    payload = conversation.to_dict()
    assert payload["source"] == "codex_cli"
    assert payload["source_session_id"] == "session-active"
    assert payload["messages"] == [
        {
            "role": "developer",
            "text": "Follow the repository guardrails.",
            "source_message_id": "msg-dev",
        },
        {
            "role": "user",
            "text": "Implement the codex collector.",
            "source_message_id": "msg-user",
        },
        {
            "role": "assistant",
            "text": "I will inspect the repo and add the collector.",
            "source_message_id": "msg-assistant",
        },
    ]
    assert payload["provenance"] == {
        "session_started_at": "2026-03-14T09:00:00Z",
        "source": "cli",
        "originator": "codex_cli_rs",
        "cwd": "/Users/chenjing/dev/chat-collector",
        "cli_version": "0.32.0",
        "archived": False,
    }
    serialized = json.dumps(payload)
    assert "function_call" not in serialized
    assert "function_call_output" not in serialized
    assert "custom_tool_call" not in serialized
    assert "tool_search_call" not in serialized
    assert "web_search_call" not in serialized
    assert "turn_context" not in serialized
    assert "dynamic_tools" not in serialized


def test_codex_cli_collect_writes_normalized_jsonl(tmp_path: Path) -> None:
    collector = CodexCliCollector()

    result = collector.collect(tmp_path, input_roots=(FIXTURE_ROOT,))

    assert result.source == "codex_cli"
    assert result.scanned_artifact_count == 2
    assert result.conversation_count == 2
    assert result.message_count == 5
    rows = read_jsonl(result.output_path)
    assert [row["source_session_id"] for row in rows] == [
        "session-archived",
        "session-active",
    ]
    assert rows[0]["messages"] == [
        {
            "role": "user",
            "text": "Summarize the last change.",
            "source_message_id": "arch-user",
        },
        {
            "role": "assistant",
            "text": "The last change added the collector.\n\nIt also added tests.",
            "source_message_id": "arch-assistant",
        },
    ]
    assert rows[0]["provenance"]["archived"] is True
    assert rows[1]["provenance"]["archived"] is False


def test_cli_collect_codex_cli_plan_and_execute(tmp_path: Path) -> None:
    plan_result = run_cli("collect", "codex_cli", "--archive-root", str(tmp_path))

    assert plan_result.returncode == 0
    plan_payload = json.loads(plan_result.stdout)
    assert plan_payload["source"] == "codex_cli"
    assert plan_payload["implemented"] is True
    assert plan_payload["root_resolution"]["platform"] == CURRENT_PLATFORM
    assert plan_payload["root_resolution"]["resolution_source"] == "descriptor"
    assert any(
        root["path"].endswith("/.codex")
        for root in plan_payload["root_resolution"]["roots"]
    )

    execute_result = run_cli(
        "collect",
        "codex_cli",
        "--archive-root",
        str(tmp_path),
        "--input-root",
        str(FIXTURE_ROOT),
        "--execute",
    )

    assert execute_result.returncode == 0
    execute_payload = json.loads(execute_result.stdout)
    assert execute_payload["source"] == "codex_cli"
    assert execute_payload["scanned_artifact_count"] == 2
    assert execute_payload["conversation_count"] == 2
    assert execute_payload["root_resolution"] == {
        "platform": CURRENT_PLATFORM,
        "resolution_source": "cli_input_root",
        "resolved_roots": [str(FIXTURE_ROOT.resolve(strict=False))],
        "roots": [
            {
                "declared_path": str(FIXTURE_ROOT.resolve(strict=False)),
                "resolution_source": "cli_input_root",
                "path": str(FIXTURE_ROOT.resolve(strict=False)),
            }
        ],
    }
    output_path = Path(execute_payload["output_path"])
    rows = read_jsonl(output_path)
    assert len(rows) == 2
