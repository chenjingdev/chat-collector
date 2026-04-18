from __future__ import annotations

import json
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

from llm_chat_archive.models import AppShellProvenance
from llm_chat_archive.sources.codex_app import (
    CodexAppCollector,
    build_codex_app_metadata_index,
    discover_app_shell_provenance,
    parse_rollout_file,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "codex_app"


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "llm_chat_archive", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def copy_fixture_root(tmp_path: Path) -> Path:
    copied_root = tmp_path / "codex_app"
    shutil.copytree(FIXTURE_ROOT, copied_root)
    return copied_root


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    return path


def expected_app_shell(root: Path) -> dict[str, object]:
    return {
        "application_support_roots": [
            str(root / "Library" / "Application Support" / "Codex")
        ],
        "log_roots": [str(root / "Library" / "Logs" / "com.openai.codex")],
        "state_db_paths": [str((root / "state_5.sqlite").resolve())],
        "preference_paths": [
            str(root / "Library" / "Preferences" / "com.openai.codex.plist")
        ],
        "cache_roots": [str(root / "Library" / "Caches" / "com.openai.codex")],
        "auxiliary_paths": [
            str((root / "automations" / "auto-nightly-collect" / "automation.toml").resolve()),
            str((root / "sqlite" / "codex-dev.db").resolve()),
        ],
    }


def expected_automation_provenance(
    root: Path,
    *,
    status: str,
    thread_title: str,
    thread_record_title: str | None,
    inbox_title: str,
    inbox_summary: str,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "automation_id": "auto-nightly-collect",
        "automation_name": "Nightly Collector Sweep",
        "status": status,
        "schedule": "RRULE:FREQ=DAILY;BYHOUR=7;BYMINUTE=15",
        "source_cwd": "/Users/chenjing/dev/chat-collector",
        "model": "gpt-5.4",
        "reasoning_effort": "medium",
        "definition_path": str(
            (root / "automations" / "auto-nightly-collect" / "automation.toml").resolve()
        ),
        "thread_title": thread_title,
        "inbox_title": inbox_title,
        "inbox_summary": inbox_summary,
        "resolved_title": thread_title,
        "resolved_title_source": "automation_runs.thread_title",
        "resolved_summary": inbox_summary,
        "resolved_summary_source": "automation_runs.inbox_summary",
    }
    if thread_record_title is not None:
        payload["thread_record_title"] = thread_record_title
    return payload


def archived_rollout_rows(
    *,
    session_id: str | None = "desktop-automation-archived",
    cwd: str | None = None,
    messages: list[dict[str, object]],
) -> list[dict[str, object]]:
    session_meta: dict[str, object] = {
        "timestamp": "2026-03-05T07:15:00Z",
        "source": "exec",
        "originator": "Codex Desktop",
    }
    if session_id is not None:
        session_meta["id"] = session_id
    if cwd is not None:
        session_meta["cwd"] = cwd
    return [{"type": "session_meta", "payload": session_meta}, *messages]


def test_parse_rollout_file_requires_codex_desktop_originator() -> None:
    rollout_path = (
        FIXTURE_ROOT
        / "sessions"
        / "2026"
        / "03"
        / "14"
        / "rollout-20260314T113000-cli-ignored.jsonl"
    )

    assert parse_rollout_file(rollout_path, collected_at="2026-03-19T00:00:00Z") is None


def test_parse_rollout_file_keeps_messages_and_attaches_app_shell_provenance() -> None:
    rollout_path = (
        FIXTURE_ROOT
        / "sessions"
        / "2026"
        / "03"
        / "14"
        / "rollout-20260314T110000-desktop-active.jsonl"
    )
    app_shell = AppShellProvenance(
        application_support_roots=("/tmp/mock/Application Support/Codex",),
        log_roots=("/tmp/mock/Logs/com.openai.codex",),
        preference_paths=("/tmp/mock/Preferences/com.openai.codex.plist",),
        cache_roots=("/tmp/mock/Caches/com.openai.codex",),
    )

    conversation = parse_rollout_file(
        rollout_path,
        collected_at="2026-03-19T00:00:00Z",
        app_shell=app_shell,
    )

    assert conversation is not None
    payload = conversation.to_dict()
    assert payload["source"] == "codex_app"
    assert payload["execution_context"] == "standalone_app"
    assert payload["source_session_id"] == "desktop-active"
    assert payload["messages"] == [
        {
            "role": "developer",
            "text": "Prefer the shared rollout transcript.",
            "source_message_id": "app-dev",
        },
        {
            "role": "user",
            "text": "Collect the Desktop app sessions only.",
            "source_message_id": "app-user",
        },
        {
            "role": "assistant",
            "text": "I will filter on Codex Desktop and keep shell artifacts as provenance only.",
            "source_message_id": "app-assistant",
        },
    ]
    assert payload["provenance"] == {
        "session_started_at": "2026-03-14T11:00:00Z",
        "source": "exec",
        "originator": "Codex Desktop",
        "cwd": "/Users/chenjing/dev/chat-collector",
        "cli_version": "0.32.0",
        "archived": False,
        "conversation_origin": "interactive",
        "app_shell": {
            "application_support_roots": ["/tmp/mock/Application Support/Codex"],
            "log_roots": ["/tmp/mock/Logs/com.openai.codex"],
            "preference_paths": ["/tmp/mock/Preferences/com.openai.codex.plist"],
            "cache_roots": ["/tmp/mock/Caches/com.openai.codex"],
        },
    }
    serialized = json.dumps(payload)
    assert "function_call" not in serialized
    assert "function_call_output" not in serialized
    assert "web_search_call" not in serialized
    assert "Local storage prompt should never become a transcript." not in serialized


def test_parse_rollout_file_reconstructs_archived_automation_messages() -> None:
    rollout_path = (
        FIXTURE_ROOT
        / "archived_sessions"
        / "rollout-20260305T071500-desktop-automation-archived.jsonl"
    )

    conversation = parse_rollout_file(
        rollout_path,
        collected_at="2026-03-19T00:00:00Z",
        app_shell=discover_app_shell_provenance((FIXTURE_ROOT,)),
        metadata_index=build_codex_app_metadata_index((FIXTURE_ROOT,)),
    )

    assert conversation is not None
    payload = conversation.to_dict()
    assert payload["source_session_id"] == "desktop-automation-archived"
    assert payload["transcript_completeness"] == "partial"
    assert payload["limitations"] == [
        "automation_origin_user_message_reconstructed_from_archived_snapshot",
        "automation_origin_assistant_message_reconstructed_from_archived_snapshot",
    ]
    assert payload["messages"] == [
        {
            "role": "user",
            "text": "Collect the archived automation threads too.",
            "timestamp": "2026-03-05T07:15:00Z",
            "source_message_id": "automation-archived-user",
            "provenance": {
                "body_source": "automation_runs.archived_user_message",
                "fallback": True,
            },
        },
        {
            "role": "assistant",
            "text": "The archived automation run is reconstructed from the inbox snapshot.",
            "source_message_id": "automation-archived-assistant",
            "provenance": {
                "body_source": "automation_runs.archived_assistant_message",
                "fallback": True,
            },
        },
    ]
    assert payload["provenance"] == {
        "session_started_at": "2026-03-05T07:15:00Z",
        "source": "exec",
        "originator": "Codex Desktop",
        "cwd": "/Users/chenjing/dev/chat-collector",
        "cli_version": "0.31.0",
        "archived": True,
        "archived_reason": "accepted_after_review",
        "conversation_origin": "automation",
        "automation": expected_automation_provenance(
            FIXTURE_ROOT,
            status="ARCHIVED",
            thread_title="Nightly collector archive",
            thread_record_title="Nightly collector archive",
            inbox_title="Nightly collector archive",
            inbox_summary="Archived automation inbox summary",
        ),
        "app_shell": expected_app_shell(FIXTURE_ROOT),
    }


def test_parse_rollout_file_keeps_archived_automation_rollout_messages_canonical(
    tmp_path: Path,
) -> None:
    copied_root = copy_fixture_root(tmp_path)
    rollout_path = write_jsonl(
        copied_root
        / "archived_sessions"
        / "rollout-20260305T071500-desktop-automation-complete.jsonl",
        archived_rollout_rows(
            messages=[
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "id": "archived-rollout-user",
                        "role": "user",
                        "timestamp": "2026-03-05T07:15:00Z",
                        "content": [
                            {
                                "type": "input_text",
                                "text": "Collect the archived automation threads too.",
                            }
                        ],
                    },
                },
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "id": "archived-rollout-assistant",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "The rollout transcript already preserved the archived automation reply.",
                            }
                        ],
                    },
                },
            ]
        ),
    )

    conversation = parse_rollout_file(
        rollout_path,
        collected_at="2026-03-19T00:00:00Z",
        metadata_index=build_codex_app_metadata_index((copied_root,)),
    )

    assert conversation is not None
    payload = conversation.to_dict()
    assert "transcript_completeness" not in payload
    assert "limitations" not in payload
    assert payload["messages"] == [
        {
            "role": "user",
            "text": "Collect the archived automation threads too.",
            "timestamp": "2026-03-05T07:15:00Z",
            "source_message_id": "archived-rollout-user",
            "provenance": {"body_source": "rollout.message"},
        },
        {
            "role": "assistant",
            "text": "The rollout transcript already preserved the archived automation reply.",
            "source_message_id": "archived-rollout-assistant",
            "provenance": {"body_source": "rollout.message"},
        },
    ]
    assert payload["provenance"]["automation"] == expected_automation_provenance(
        copied_root,
        status="ARCHIVED",
        thread_title="Nightly collector archive",
        thread_record_title="Nightly collector archive",
        inbox_title="Nightly collector archive",
        inbox_summary="Archived automation inbox summary",
    )


def test_parse_rollout_file_reconstructs_only_missing_archived_automation_user(
    tmp_path: Path,
) -> None:
    copied_root = copy_fixture_root(tmp_path)
    rollout_path = write_jsonl(
        copied_root
        / "archived_sessions"
        / "rollout-20260305T071500-desktop-automation-user-gap.jsonl",
        archived_rollout_rows(
            messages=[
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "id": "archived-rollout-assistant",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "The archived automation run is reconstructed from the inbox snapshot.",
                            }
                        ],
                    },
                }
            ]
        ),
    )

    conversation = parse_rollout_file(
        rollout_path,
        collected_at="2026-03-19T00:00:00Z",
        metadata_index=build_codex_app_metadata_index((copied_root,)),
    )

    assert conversation is not None
    payload = conversation.to_dict()
    assert payload["transcript_completeness"] == "partial"
    assert payload["limitations"] == [
        "automation_origin_user_message_reconstructed_from_archived_snapshot"
    ]
    assert payload["messages"] == [
        {
            "role": "user",
            "text": "Collect the archived automation threads too.",
            "timestamp": "2026-03-05T07:15:00Z",
            "source_message_id": "automation-archived-user",
            "provenance": {
                "body_source": "automation_runs.archived_user_message",
                "fallback": True,
            },
        },
        {
            "role": "assistant",
            "text": "The archived automation run is reconstructed from the inbox snapshot.",
            "source_message_id": "archived-rollout-assistant",
            "provenance": {"body_source": "rollout.message"},
        },
    ]


def test_parse_rollout_file_reconstructs_only_missing_archived_automation_assistant(
    tmp_path: Path,
) -> None:
    copied_root = copy_fixture_root(tmp_path)
    rollout_path = write_jsonl(
        copied_root
        / "archived_sessions"
        / "rollout-20260305T071500-desktop-automation-assistant-gap.jsonl",
        archived_rollout_rows(
            messages=[
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "id": "archived-rollout-user",
                        "role": "user",
                        "timestamp": "2026-03-05T07:15:00Z",
                        "content": [
                            {
                                "type": "input_text",
                                "text": "Collect the archived automation threads too.",
                            }
                        ],
                    },
                }
            ]
        ),
    )

    conversation = parse_rollout_file(
        rollout_path,
        collected_at="2026-03-19T00:00:00Z",
        metadata_index=build_codex_app_metadata_index((copied_root,)),
    )

    assert conversation is not None
    payload = conversation.to_dict()
    assert payload["transcript_completeness"] == "partial"
    assert payload["limitations"] == [
        "automation_origin_assistant_message_reconstructed_from_archived_snapshot"
    ]
    assert payload["messages"] == [
        {
            "role": "user",
            "text": "Collect the archived automation threads too.",
            "timestamp": "2026-03-05T07:15:00Z",
            "source_message_id": "archived-rollout-user",
            "provenance": {"body_source": "rollout.message"},
        },
        {
            "role": "assistant",
            "text": "The archived automation run is reconstructed from the inbox snapshot.",
            "source_message_id": "automation-archived-assistant",
            "provenance": {
                "body_source": "automation_runs.archived_assistant_message",
                "fallback": True,
            },
        },
    ]


def test_parse_rollout_file_attributes_archived_automation_via_thread_metadata_fallback(
    tmp_path: Path,
) -> None:
    copied_root = copy_fixture_root(tmp_path)
    rollout_path = copied_root / "archived_sessions" / "rollout-20260305T071500-detached.jsonl"
    write_jsonl(
        rollout_path,
        archived_rollout_rows(
            session_id=None,
            messages=[
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "id": "detached-user",
                        "role": "user",
                        "timestamp": "2026-03-05T07:15:00Z",
                        "content": [
                            {
                                "type": "input_text",
                                "text": "Collect the archived automation threads too.",
                            }
                        ],
                    },
                }
            ],
        ),
    )
    with sqlite3.connect(copied_root / "state_5.sqlite") as connection:
        connection.execute(
            "INSERT INTO threads (id, rollout_path, cwd, title, first_user_message, archived, cli_version) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                "detached-archived-thread",
                str(rollout_path.resolve()),
                "/Users/chenjing/dev/chat-collector",
                "Nightly collector archive",
                "Collect the archived automation threads too.",
                1,
                "0.31.0",
            ),
        )
        connection.commit()

    conversation = parse_rollout_file(
        rollout_path,
        collected_at="2026-03-19T00:00:00Z",
        metadata_index=build_codex_app_metadata_index((copied_root,)),
    )

    assert conversation is not None
    payload = conversation.to_dict()
    assert payload["source_session_id"] == "detached-archived-thread"
    assert payload["provenance"]["conversation_origin"] == "automation"
    assert payload["provenance"]["automation"] == expected_automation_provenance(
        copied_root,
        status="ARCHIVED",
        thread_title="Nightly collector archive",
        thread_record_title="Nightly collector archive",
        inbox_title="Nightly collector archive",
        inbox_summary="Archived automation inbox summary",
    )
    assert payload["messages"][-1] == {
        "role": "assistant",
        "text": "The archived automation run is reconstructed from the inbox snapshot.",
        "source_message_id": "automation-archived-assistant",
        "provenance": {
            "body_source": "automation_runs.archived_assistant_message",
            "fallback": True,
        },
    }


def test_codex_app_collect_selects_only_desktop_sessions(tmp_path: Path) -> None:
    collector = CodexAppCollector()

    result = collector.collect(tmp_path, input_roots=(FIXTURE_ROOT,))

    assert result.source == "codex_app"
    assert result.scanned_artifact_count == 5
    assert result.conversation_count == 4
    assert result.message_count == 9
    rows = read_jsonl(result.output_path)
    assert [row["source_session_id"] for row in rows] == [
        "desktop-archived",
        "desktop-automation-archived",
        "desktop-active",
        "desktop-automation-active",
    ]
    assert {row["provenance"]["originator"] for row in rows} == {"Codex Desktop"}
    rows_by_session_id = {row["source_session_id"]: row for row in rows}
    assert rows_by_session_id["desktop-archived"]["provenance"]["archived"] is True
    assert rows_by_session_id["desktop-active"]["provenance"]["archived"] is False
    assert rows_by_session_id["desktop-active"]["provenance"]["conversation_origin"] == (
        "interactive"
    )
    assert rows_by_session_id["desktop-automation-active"]["provenance"] == {
        "session_started_at": "2026-03-15T09:00:00Z",
        "source": "exec",
        "originator": "Codex Desktop",
        "cwd": "/Users/chenjing/dev/chat-collector",
        "cli_version": "0.32.0",
        "archived": False,
        "conversation_origin": "automation",
        "automation": expected_automation_provenance(
            FIXTURE_ROOT,
            status="ACCEPTED",
            thread_title="Nightly collector sweep",
            thread_record_title="Nightly collector sweep",
            inbox_title="Nightly collector sweep",
            inbox_summary="Active automation inbox summary",
        ),
        "app_shell": expected_app_shell(FIXTURE_ROOT),
    }
    assert rows_by_session_id["desktop-automation-archived"][
        "transcript_completeness"
    ] == "partial"
    assert rows_by_session_id["desktop-automation-archived"]["limitations"] == [
        "automation_origin_user_message_reconstructed_from_archived_snapshot",
        "automation_origin_assistant_message_reconstructed_from_archived_snapshot",
    ]
    assert rows_by_session_id["desktop-automation-archived"]["messages"] == [
        {
            "role": "user",
            "text": "Collect the archived automation threads too.",
            "timestamp": "2026-03-05T07:15:00Z",
            "source_message_id": "automation-archived-user",
            "provenance": {
                "body_source": "automation_runs.archived_user_message",
                "fallback": True,
            },
        },
        {
            "role": "assistant",
            "text": "The archived automation run is reconstructed from the inbox snapshot.",
            "source_message_id": "automation-archived-assistant",
            "provenance": {
                "body_source": "automation_runs.archived_assistant_message",
                "fallback": True,
            },
        },
    ]
    assert rows_by_session_id["desktop-archived"]["provenance"]["app_shell"] == expected_app_shell(
        FIXTURE_ROOT
    )
    serialized = json.dumps(rows, ensure_ascii=False)
    assert "This CLI session should be ignored." not in serialized
    assert "Local storage prompt should never become a transcript." not in serialized
    assert "Desktop log line must stay out of transcript output." not in serialized
    assert "Preference value that should never become transcript text." not in serialized
    assert "Cache payload that should never become transcript text." not in serialized


def test_cli_collect_codex_app_plan_and_execute(tmp_path: Path) -> None:
    plan_result = run_cli("collect", "codex_app", "--archive-root", str(tmp_path))

    assert plan_result.returncode == 0
    plan_payload = json.loads(plan_result.stdout)
    assert plan_payload["source"] == "codex_app"
    assert plan_payload["implemented"] is True

    execute_result = run_cli(
        "collect",
        "codex_app",
        "--archive-root",
        str(tmp_path),
        "--input-root",
        str(FIXTURE_ROOT),
        "--execute",
    )

    assert execute_result.returncode == 0
    execute_payload = json.loads(execute_result.stdout)
    assert execute_payload["source"] == "codex_app"
    assert execute_payload["scanned_artifact_count"] == 5
    assert execute_payload["conversation_count"] == 4
    output_path = Path(execute_payload["output_path"])
    rows = read_jsonl(output_path)
    assert len(rows) == 4
