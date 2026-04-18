from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from llm_chat_archive.sources.cursor_editor import (
    CursorEditorCollector,
    discover_cursor_auxiliary_artifacts,
    parse_workspace_state,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "cursor_editor"


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "llm_chat_archive", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_parse_workspace_state_reconstructs_complete_transcript_from_cursor_disk_kv() -> None:
    state_path = (
        FIXTURE_ROOT
        / "Library"
        / "Application Support"
        / "Cursor"
        / "User"
        / "workspaceStorage"
        / "workspace-alpha"
        / "state.vscdb"
    )
    auxiliary = discover_cursor_auxiliary_artifacts((FIXTURE_ROOT,))

    conversation = parse_workspace_state(
        state_path,
        collected_at="2026-03-19T00:00:00Z",
        auxiliary=auxiliary,
    )

    assert conversation is not None
    payload = conversation.to_dict()
    assert payload["source"] == "cursor_editor"
    assert payload["execution_context"] == "ide_native"
    assert "transcript_completeness" not in payload
    assert "limitations" not in payload
    assert payload["source_session_id"] == "composer-alpha"
    assert payload["messages"] == [
        {
            "source_message_id": "alpha-user-1",
            "provenance": {"body_source": "cursor_disk_kv.text"},
            "role": "user",
            "text": "Implement the Cursor editor collector.",
        },
        {
            "source_message_id": "alpha-assistant-1",
            "provenance": {"body_source": "cursor_disk_kv.text"},
            "role": "assistant",
            "text": "Use workspace state.vscdb for composer metadata and global cursorDiskKV bubble rows for explicit transcript recovery.",
            "timestamp": "2026-03-14T10:00:02Z",
        },
        {
            "source_message_id": "alpha-user-2",
            "provenance": {"body_source": "cursor_disk_kv.text"},
            "role": "user",
            "text": "Do not classify Codex extension state as Cursor-native transcript.",
        },
        {
            "source_message_id": "alpha-assistant-2",
            "provenance": {"body_source": "cursor_disk_kv.text"},
            "role": "assistant",
            "text": "Codex extension state stays provenance only, so the normalized transcript keeps only user and assistant message bodies.",
            "timestamp": "2026-03-14T10:01:05Z",
        },
    ]
    assert payload["session_metadata"] == {
        "workspace_id": "workspace-alpha",
        "workspace_folder": "/Users/chenjing/dev/chat-collector",
        "composer_id": "composer-alpha",
        "reconstructed_message_count": 4,
        "composer_name": "Cursor editor collector",
        "composer_subtitle": "Native metadata only",
        "mode": "agent",
        "created_at": "2026-03-14T10:00:00Z",
        "last_updated_at": "2026-03-14T10:02:00Z",
        "context_usage_percent": 37,
        "is_archived": False,
        "is_worktree": False,
        "is_spec": False,
        "prompt_count": 2,
        "generation_count": 2,
        "transcript_source": "cursor_disk_kv",
        "transcript_header_count": 5,
        "skipped_tool_bubble_count": 1,
        "selected": True,
        "last_focused": True,
        "memory_enabled": True,
        "pending_memories_count": 0,
        "hook_session_end_count": 1,
        "skipped_tool_bubble_ids": [
            "alpha-tool-1",
        ],
        "ignored_state_keys": [
            "memento/webviewView.chatgpt.sidebarView",
            "openai.chatgpt",
        ],
    }
    assert payload["provenance"] == {
        "session_started_at": "2026-03-14T10:00:00Z",
        "source": "cursor",
        "originator": "cursor_editor",
        "cwd": "/Users/chenjing/dev/chat-collector",
        "archived": False,
        "app_shell": {
            "application_support_roots": [
                str(FIXTURE_ROOT / "Library" / "Application Support" / "Cursor"),
            ],
            "state_db_paths": [
                str(
                    FIXTURE_ROOT
                    / "Library"
                    / "Application Support"
                    / "Cursor"
                    / "User"
                    / "globalStorage"
                    / "state.vscdb"
                ),
                str(state_path),
            ],
            "log_paths": [
                str(
                    FIXTURE_ROOT
                    / "Library"
                    / "Application Support"
                    / "Cursor"
                    / "logs"
                    / "20260314T100500"
                    / "window1"
                    / "output_20260314T100550"
                    / "cursor.hooks.log"
                ),
            ],
            "cache_roots": [
                str(FIXTURE_ROOT / ".cursor"),
            ],
            "auxiliary_paths": [
                str(FIXTURE_ROOT / ".cursor" / "ai-tracking" / "ai-code-tracking.db"),
            ],
        },
    }
    serialized = json.dumps(payload, ensure_ascii=False)
    assert "This generation description must stay metadata-only." not in serialized
    assert "Assistant reply body is still unverified here." not in serialized
    assert "Workspace ChatGPT extension text must stay out of Cursor native transcript." not in serialized
    assert "This global ChatGPT extension text must stay out of Cursor native transcript." not in serialized
    assert "Hook log text should stay provenance only." not in serialized
    assert "Tracking summary must stay out of messages." not in serialized
    assert "run_terminal_command" not in serialized


def test_parse_workspace_state_recovers_nested_explicit_bubble_body_keys() -> None:
    state_path = (
        FIXTURE_ROOT
        / "Library"
        / "Application Support"
        / "Cursor"
        / "User"
        / "workspaceStorage"
        / "workspace-delta"
        / "state.vscdb"
    )
    auxiliary = discover_cursor_auxiliary_artifacts((FIXTURE_ROOT,))

    conversation = parse_workspace_state(
        state_path,
        collected_at="2026-03-19T00:00:00Z",
        auxiliary=auxiliary,
    )

    assert conversation is not None
    payload = conversation.to_dict()
    assert payload["source_session_id"] == "composer-delta"
    assert "transcript_completeness" not in payload
    assert "limitations" not in payload
    assert payload["messages"] == [
        {
            "source_message_id": "delta-user-1",
            "provenance": {"body_source": "cursor_disk_kv.body.text"},
            "role": "user",
            "text": "Recover explicit body keys beyond text.",
        },
        {
            "source_message_id": "delta-assistant-1",
            "provenance": {"body_source": "cursor_disk_kv.markdown"},
            "role": "assistant",
            "text": "Nested markdown body is explicit enough for recovery.",
            "timestamp": "2026-03-14T12:30:04Z",
        },
    ]
    assert payload["session_metadata"] == {
        "workspace_id": "workspace-delta",
        "workspace_folder": "/Users/chenjing/dev/chat-collector/CHE-150",
        "composer_id": "composer-delta",
        "reconstructed_message_count": 2,
        "composer_name": "Cursor editor nested bodies",
        "composer_subtitle": "Nested cursorDiskKV body keys",
        "mode": "agent",
        "created_at": "2026-03-14T12:30:00Z",
        "last_updated_at": "2026-03-14T12:31:00Z",
        "context_usage_percent": 21,
        "is_archived": False,
        "is_worktree": False,
        "is_spec": False,
        "prompt_count": 1,
        "generation_count": 1,
        "transcript_source": "cursor_disk_kv",
        "transcript_header_count": 2,
        "selected": True,
        "last_focused": True,
        "memory_enabled": True,
        "pending_memories_count": 0,
        "hook_session_end_count": 1,
        "ignored_state_keys": [
            "openai.chatgpt",
        ],
    }
    serialized = json.dumps(payload, ensure_ascii=False)
    assert "Nested body metadata should stay out of transcript text." not in serialized


def test_parse_workspace_state_marks_tool_only_bubbles_without_promoting_them() -> None:
    state_path = (
        FIXTURE_ROOT
        / "Library"
        / "Application Support"
        / "Cursor"
        / "User"
        / "workspaceStorage"
        / "workspace-epsilon"
        / "state.vscdb"
    )
    auxiliary = discover_cursor_auxiliary_artifacts((FIXTURE_ROOT,))

    conversation = parse_workspace_state(
        state_path,
        collected_at="2026-03-19T00:00:00Z",
        auxiliary=auxiliary,
    )

    assert conversation is not None
    payload = conversation.to_dict()
    assert payload["source_session_id"] == "composer-epsilon"
    assert payload["transcript_completeness"] == "partial"
    assert payload["limitations"] == [
        "assistant_body_unverified",
    ]
    assert payload["messages"] == [
        {
            "source_message_id": "epsilon-user-1",
            "provenance": {"body_source": "cursor_disk_kv.text"},
            "role": "user",
            "text": "Keep tool-only assistant bubbles out of transcript text.",
        },
    ]
    assert payload["session_metadata"] == {
        "workspace_id": "workspace-epsilon",
        "workspace_folder": "/Users/chenjing/dev/chat-history",
        "composer_id": "composer-epsilon",
        "reconstructed_message_count": 1,
        "composer_name": "Cursor editor tool-only assistant",
        "composer_subtitle": "Tool-only bubble remains partial",
        "mode": "agent",
        "created_at": "2026-03-14T13:00:00Z",
        "last_updated_at": "2026-03-14T13:00:30Z",
        "context_usage_percent": 9,
        "is_archived": False,
        "is_worktree": False,
        "is_spec": False,
        "prompt_count": 1,
        "generation_count": 1,
        "transcript_source": "cursor_disk_kv",
        "transcript_header_count": 2,
        "selected": True,
        "last_focused": True,
        "memory_enabled": True,
        "pending_memories_count": 0,
        "hook_session_end_count": 1,
        "skipped_tool_bubble_count": 1,
        "skipped_tool_bubble_ids": [
            "epsilon-tool-1",
        ],
        "ignored_state_keys": [
            "openai.chatgpt",
        ],
    }
    serialized = json.dumps(payload, ensure_ascii=False)
    assert "Tool-only assistant metadata should stay out of transcript text." not in serialized
    assert "grep_workspace" not in serialized


def test_parse_workspace_state_keeps_partial_when_assistant_body_is_missing() -> None:
    state_path = (
        FIXTURE_ROOT
        / "Library"
        / "Application Support"
        / "Cursor"
        / "User"
        / "workspaceStorage"
        / "workspace-gamma"
        / "state.vscdb"
    )
    auxiliary = discover_cursor_auxiliary_artifacts((FIXTURE_ROOT,))

    conversation = parse_workspace_state(
        state_path,
        collected_at="2026-03-19T00:00:00Z",
        auxiliary=auxiliary,
    )

    assert conversation is not None
    payload = conversation.to_dict()
    assert payload["source_session_id"] == "composer-gamma"
    assert payload["transcript_completeness"] == "partial"
    assert payload["limitations"] == [
        "assistant_body_missing_from_cursor_disk_kv",
    ]
    assert payload["messages"] == [
        {
            "source_message_id": "gamma-user-1",
            "provenance": {"body_source": "cursor_disk_kv.text"},
            "role": "user",
            "text": "Keep this session partial when assistant body is missing.",
        },
    ]
    assert payload["session_metadata"] == {
        "workspace_id": "workspace-gamma",
        "workspace_folder": "/Users/chenjing/dev/chat-history",
        "composer_id": "composer-gamma",
        "reconstructed_message_count": 1,
        "composer_name": "Cursor editor degraded transcript",
        "composer_subtitle": "Assistant body missing",
        "mode": "agent",
        "created_at": "2026-03-14T12:00:00Z",
        "last_updated_at": "2026-03-14T12:01:00Z",
        "context_usage_percent": 12,
        "is_archived": False,
        "is_worktree": False,
        "is_spec": False,
        "prompt_count": 1,
        "generation_count": 1,
        "transcript_source": "cursor_disk_kv",
        "transcript_header_count": 2,
        "selected": True,
        "last_focused": True,
        "memory_enabled": True,
        "pending_memories_count": 0,
        "hook_session_end_count": 1,
        "missing_assistant_bubble_ids": [
            "gamma-assistant-1",
        ],
        "ignored_state_keys": [
            "openai.chatgpt",
        ],
    }
    assert payload["provenance"] == {
        "session_started_at": "2026-03-14T12:00:00Z",
        "source": "cursor",
        "originator": "cursor_editor",
        "cwd": "/Users/chenjing/dev/chat-history",
        "archived": False,
        "app_shell": {
            "application_support_roots": [
                str(FIXTURE_ROOT / "Library" / "Application Support" / "Cursor"),
            ],
            "state_db_paths": [
                str(
                    FIXTURE_ROOT
                    / "Library"
                    / "Application Support"
                    / "Cursor"
                    / "User"
                    / "globalStorage"
                    / "state.vscdb"
                ),
                str(state_path),
            ],
            "log_paths": [
                str(
                    FIXTURE_ROOT
                    / "Library"
                    / "Application Support"
                    / "Cursor"
                    / "logs"
                    / "20260314T100500"
                    / "window1"
                    / "output_20260314T100550"
                    / "cursor.hooks.log"
                ),
            ],
            "cache_roots": [
                str(FIXTURE_ROOT / ".cursor"),
            ],
            "auxiliary_paths": [
                str(FIXTURE_ROOT / ".cursor" / "ai-tracking" / "ai-code-tracking.db"),
            ],
        },
    }
    serialized = json.dumps(payload, ensure_ascii=False)
    assert "Missing assistant body should remain degraded." not in serialized


def test_parse_workspace_state_requires_resolvable_composer_session_index() -> None:
    state_path = (
        FIXTURE_ROOT
        / "Library"
        / "Application Support"
        / "Cursor"
        / "User"
        / "workspaceStorage"
        / "workspace-beta"
        / "state.vscdb"
    )

    assert (
        parse_workspace_state(
            state_path,
            collected_at="2026-03-19T00:00:00Z",
            auxiliary=discover_cursor_auxiliary_artifacts((FIXTURE_ROOT,)),
        )
        is None
    )


def test_cursor_editor_collect_writes_complete_and_partial_archive(tmp_path: Path) -> None:
    collector = CursorEditorCollector()

    result = collector.collect(tmp_path, input_roots=(FIXTURE_ROOT,))

    assert result.source == "cursor_editor"
    assert result.scanned_artifact_count == 5
    assert result.conversation_count == 4
    assert result.message_count == 8
    rows = read_jsonl(result.output_path)
    assert len(rows) == 4
    by_session_id = {row["source_session_id"]: row for row in rows}
    assert set(by_session_id) == {
        "composer-alpha",
        "composer-delta",
        "composer-epsilon",
        "composer-gamma",
    }
    assert "transcript_completeness" not in by_session_id["composer-alpha"]
    assert "transcript_completeness" not in by_session_id["composer-delta"]
    assert by_session_id["composer-epsilon"]["transcript_completeness"] == "partial"
    assert by_session_id["composer-epsilon"]["limitations"] == [
        "assistant_body_unverified"
    ]
    assert by_session_id["composer-gamma"]["transcript_completeness"] == "partial"
    assert by_session_id["composer-gamma"]["limitations"] == [
        "assistant_body_missing_from_cursor_disk_kv"
    ]
    serialized = json.dumps(rows, ensure_ascii=False)
    assert "This ambiguous workspace prompt must not be collected." not in serialized
    assert "Ambiguous generation description should stay metadata-only." not in serialized
    assert "Workspace ChatGPT extension text must stay out of Cursor native transcript." not in serialized
    assert "This global ChatGPT extension text must stay out of Cursor native transcript." not in serialized
    assert "Hook log text should stay provenance only." not in serialized
    assert "Tracking summary must stay out of messages." not in serialized
    assert "run_terminal_command" not in serialized


def test_cli_collect_cursor_editor_plan_and_execute(tmp_path: Path) -> None:
    plan_result = run_cli("collect", "cursor_editor", "--archive-root", str(tmp_path))

    assert plan_result.returncode == 0
    plan_payload = json.loads(plan_result.stdout)
    assert plan_payload["source"] == "cursor_editor"
    assert plan_payload["implemented"] is True
    assert plan_payload["support_level"] == "partial"

    execute_result = run_cli(
        "collect",
        "cursor_editor",
        "--archive-root",
        str(tmp_path),
        "--input-root",
        str(FIXTURE_ROOT),
        "--execute",
    )

    assert execute_result.returncode == 0
    execute_payload = json.loads(execute_result.stdout)
    assert execute_payload["source"] == "cursor_editor"
    assert execute_payload["scanned_artifact_count"] == 5
    assert execute_payload["conversation_count"] == 4
    rows = read_jsonl(Path(execute_payload["output_path"]))
    assert len(rows) == 4
