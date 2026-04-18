from __future__ import annotations

import io
import json
import sqlite3
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

from llm_chat_archive import cli
from llm_chat_archive.registry import CollectorRegistry
from llm_chat_archive.runner import run_collection_batch
from llm_chat_archive.source_selection import build_source_selection_policy
from llm_chat_archive.sources.codex_cli import (
    CODEX_CLI_DESCRIPTOR,
    CodexCliCollector,
)
from tests.fixture_cases import FIXTURES_ROOT, unique_source_fixture_cases


def run_cli(*args: str) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        exit_code = cli.main(list(args))
    return exit_code, stdout.getvalue(), stderr.getvalue()


def write_item_table(state_db_path: Path, values: dict[str, object]) -> None:
    state_db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(state_db_path) as connection:
        connection.execute("CREATE TABLE ItemTable (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        connection.executemany(
            "INSERT INTO ItemTable (key, value) VALUES (?, ?)",
            [
                (key, json.dumps(value, ensure_ascii=False))
                for key, value in values.items()
            ],
        )
        connection.commit()


def write_codex_drift_rollout(root: Path) -> Path:
    rollout_path = (
        root
        / "sessions"
        / "2026"
        / "03"
        / "20"
        / "rollout-20260320T010000-session-drift.jsonl"
    )
    rollout_path.parent.mkdir(parents=True, exist_ok=True)
    rollout_path.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "type": "session_meta",
                        "payload": {
                            "id": "session-drift",
                            "timestamp": "2026-03-20T01:00:00Z",
                            "source": "cli",
                            "originator": "codex_cli_rs",
                            "cli_version": "0.32.0",
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "response_item",
                        "payload": {
                            "id": "msg-user",
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "image", "url": "ignored"}],
                        },
                    }
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return rollout_path


@pytest.mark.parametrize(
    "case",
    tuple(
        case
        for case in unique_source_fixture_cases()
        if case.source != "antigravity_editor_view"
    ),
    ids=lambda case: case.case_id,
)
def test_doctor_does_not_report_drift_for_known_fixtures(case) -> None:
    exit_code, stdout, stderr = run_cli(
        "doctor",
        case.source,
        "--input-root",
        str(case.fixture_root),
    )

    assert exit_code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["parser_assumption"]["status"] != "drift_suspected"


def test_doctor_reports_drift_for_antigravity_unknown_variant_fixture() -> None:
    exit_code, stdout, stderr = run_cli(
        "doctor",
        "antigravity_editor_view",
        "--input-root",
        str(FIXTURES_ROOT / "antigravity_editor_view"),
    )

    assert exit_code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["status"] == "partial-ready"
    assert payload["status_reason"].startswith("drift suspected:")
    assert payload["parser_assumption"]["status"] == "drift_suspected"
    assert any(
        "variant_unknown" in evidence
        for evidence in payload["parser_assumption"]["evidence"]
    )


def test_doctor_reports_drift_for_codex_rollout_shape_change(tmp_path: Path) -> None:
    write_codex_drift_rollout(tmp_path)

    exit_code, stdout, stderr = run_cli(
        "doctor",
        "codex_cli",
        "--input-root",
        str(tmp_path),
    )

    assert exit_code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["status"] == "partial-ready"
    assert payload["status_reason"].startswith("drift suspected:")
    assert payload["parser_assumption"]["status"] == "drift_suspected"


def test_doctor_reports_drift_for_claude_ide_marker_without_parseable_messages(
    tmp_path: Path,
) -> None:
    session_id = "11111111-1111-4111-8111-111111111111"
    history_path = tmp_path / "history.jsonl"
    transcript_path = tmp_path / "projects" / "project-alpha" / f"{session_id}.jsonl"
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(
        json.dumps({"display": "/ide open", "sessionId": session_id}) + "\n",
        encoding="utf-8",
    )
    transcript_path.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "type": "system",
                        "subtype": "local_command",
                        "message": {"content": ["<command-name>/ide</command-name>"]},
                    }
                ),
                json.dumps(
                    {
                        "type": "assistant",
                        "timestamp": "2026-03-20T02:00:00Z",
                        "uuid": "assistant-1",
                        "message": {
                            "id": "assistant-1",
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "tool_use",
                                    "id": "toolu_1",
                                    "name": "Read",
                                    "input": {"path": "README.md"},
                                }
                            ],
                        },
                    }
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    exit_code, stdout, stderr = run_cli(
        "doctor",
        "claude_code_ide",
        "--input-root",
        str(tmp_path),
    )

    assert exit_code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["status_reason"].startswith("drift suspected:")
    assert payload["parser_assumption"]["status"] == "drift_suspected"


def test_doctor_reports_drift_for_gemini_code_assist_provider_without_transcript(
    tmp_path: Path,
) -> None:
    workspace_root = (
        tmp_path
        / "Library"
        / "Application Support"
        / "Code"
        / "User"
        / "workspaceStorage"
        / "workspace-alpha"
    )
    write_item_table(
        workspace_root / "state.vscdb",
        {
            "chat.ChatSessionStore.index": {
                "version": 1,
                "entries": [
                    {
                        "sessionId": "gemini-drift",
                        "lastMessageDate": "2026-03-20T03:00:00Z",
                    }
                ],
            },
            "workbench.view.extension.geminiChat.state": {
                "cloudcode.gemini.chatView": {"collapsed": False}
            },
        },
    )
    (workspace_root / "workspace.json").write_text(
        json.dumps({"folder": "file:///tmp/gemini-drift"}),
        encoding="utf-8",
    )
    chat_session_path = workspace_root / "chatSessions" / "gemini-drift.json"
    chat_session_path.parent.mkdir(parents=True, exist_ok=True)
    chat_session_path.write_text(
        json.dumps(
            {
                "sessionId": "gemini-drift",
                "provider": "google.geminicodeassist",
                "requests": [
                    {
                        "message": {"unsupported": True},
                        "response": {"unsupported": True},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    exit_code, stdout, stderr = run_cli(
        "doctor",
        "gemini_code_assist_ide",
        "--input-root",
        str(tmp_path),
    )

    assert exit_code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["status_reason"].startswith("drift suspected:")
    assert payload["parser_assumption"]["status"] == "drift_suspected"


def test_validate_reports_drift_for_codex_cli_run(tmp_path: Path) -> None:
    fixture_root = tmp_path / "fixture-root"
    write_codex_drift_rollout(fixture_root)
    archive_root = tmp_path / "archive-root"
    registry = CollectorRegistry()
    registry.register(CodexCliCollector(descriptor=CODEX_CLI_DESCRIPTOR))
    run_result = run_collection_batch(
        registry,
        archive_root,
        input_roots=(fixture_root,),
        selection_policy=build_source_selection_policy(
            include_sources=("codex_cli",),
        ),
    )

    exit_code, stdout, stderr = run_cli(
        "validate",
        "--archive-root",
        str(archive_root),
        "--run",
        run_result.run_id,
    )

    assert exit_code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["status"] == "warning"
    assert any(
        finding["code"] == "drift_suspected" and finding["source"] == "codex_cli"
        for finding in payload["findings"]
    )
    assert payload["sources"][0]["drift_suspected"] is True
    assert payload["sources"][0]["parser_assumption_summary"].startswith("drift suspected:")
