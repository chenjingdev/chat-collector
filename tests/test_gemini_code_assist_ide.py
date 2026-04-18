from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from llm_chat_archive.sources.gemini_code_assist_ide import (
    GeminiCodeAssistIdeCollector,
    attribute_chat_session,
    discover_gemini_code_assist_ide_artifacts,
    parse_chat_session_transcript,
    parse_global_state,
    parse_workspace_state,
    parse_workspace_state_rows,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "gemini_code_assist_ide"


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "llm_chat_archive", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_attribute_chat_session_detects_gemini_provider_without_exposing_body() -> None:
    chat_session_path = (
        FIXTURE_ROOT
        / "Library"
        / "Application Support"
        / "Code"
        / "User"
        / "workspaceStorage"
        / "workspace-alpha"
        / "chatSessions"
        / "gemini-candidate.json"
    )

    attribution = attribute_chat_session(chat_session_path)

    assert attribution is not None
    payload = attribution.to_dict()
    assert payload == {
        "session_id": "gemini-candidate",
        "ownership": "gemini",
        "provider": "Gemini Code Assist",
        "source_path": str(chat_session_path),
        "request_count": 3,
        "is_empty": False,
        "provider_candidates": [
            "Gemini Code Assist",
            "google.geminicodeassist",
        ],
        "ownership_reason": "explicit_gemini_provider_marker",
    }
    serialized = json.dumps(payload, ensure_ascii=False)
    assert "Summarize the current Gemini IDE collector." not in serialized
    assert "Those files remain provenance-only artifacts." not in serialized


def test_parse_chat_session_transcript_reconstructs_gemini_messages() -> None:
    chat_session_path = (
        FIXTURE_ROOT
        / "Library"
        / "Application Support"
        / "Code"
        / "User"
        / "workspaceStorage"
        / "workspace-alpha"
        / "chatSessions"
        / "gemini-candidate.json"
    )

    conversation = parse_chat_session_transcript(
        chat_session_path,
        collected_at="2026-03-19T00:00:00Z",
    )

    assert conversation is not None
    payload = conversation.to_dict()
    assert payload["source_session_id"] == "gemini-candidate"
    assert payload["messages"] == [
        {
            "role": "user",
            "text": "Summarize the current Gemini IDE collector.",
            "timestamp": "2026-03-14T10:00:00Z",
            "source_message_id": "request-1",
        },
        {
            "role": "assistant",
            "text": "The collector only emitted metadata-only rows before this upgrade.",
            "timestamp": "2026-03-14T10:00:04Z",
            "source_message_id": "response-1",
        },
        {
            "role": "user",
            "text": "Keep auth files and install residue out of transcript messages.",
            "timestamp": "2026-03-14T10:01:00Z",
            "source_message_id": "request-2",
        },
        {
            "role": "assistant",
            "text": "Confirmed.\n\nThose files remain provenance-only artifacts.",
            "timestamp": "2026-03-14T10:01:06Z",
            "source_message_id": "request-2:response-2",
        },
        {
            "role": "user",
            "text": "Recover the segmented Gemini body shape too.",
            "timestamp": "2026-03-14T10:02:00Z",
            "source_message_id": "request-3",
        },
        {
            "role": "assistant",
            "text": "Recovered from the alternate segmented Gemini response body.",
            "timestamp": "2026-03-14T10:02:08Z",
            "source_message_id": "request-3:response-3",
        },
    ]
    assert "transcript_completeness" not in payload
    assert "limitations" not in payload
    assert payload["session_metadata"] == {
        "scope": "chat_session",
        "chat_session_id": "gemini-candidate",
        "chat_session_ownership": "gemini",
        "chat_session_provider": "Gemini Code Assist",
        "chat_session_request_count": 3,
        "chat_session_is_empty": False,
        "user_message_count": 3,
        "assistant_message_count": 3,
    }
    assert payload["provenance"] == {
        "session_started_at": "2026-03-14T10:00:00Z",
        "source": "vscode",
        "originator": "google.geminicodeassist",
    }


def test_parse_chat_session_transcript_returns_partial_when_response_body_is_missing() -> None:
    chat_session_path = (
        FIXTURE_ROOT
        / "Library"
        / "Application Support"
        / "Code"
        / "User"
        / "workspaceStorage"
        / "workspace-alpha"
        / "chatSessions"
        / "gemini-degraded.json"
    )

    conversation = parse_chat_session_transcript(
        chat_session_path,
        collected_at="2026-03-19T00:00:00Z",
    )

    assert conversation is not None
    payload = conversation.to_dict()
    assert payload["source_session_id"] == "gemini-degraded"
    assert payload["messages"] == [
        {
            "role": "user",
            "text": "Explain why provider ownership must be explicit.",
            "timestamp": "2026-03-14T12:00:00Z",
            "source_message_id": "request-1",
        }
    ]
    assert payload["transcript_completeness"] == "partial"
    assert payload["limitations"] == [
        "assistant_message_missing_from_gemini_chat_session"
    ]
    assert payload["session_metadata"] == {
        "scope": "chat_session",
        "chat_session_id": "gemini-degraded",
        "chat_session_ownership": "gemini",
        "chat_session_provider": "Gemini Code Assist",
        "chat_session_request_count": 1,
        "chat_session_is_empty": False,
        "user_message_count": 1,
        "assistant_message_count": 0,
    }


def test_parse_chat_session_transcript_keeps_provider_explicit_body_missing_as_residue() -> None:
    chat_session_path = (
        FIXTURE_ROOT
        / "Library"
        / "Application Support"
        / "Code"
        / "User"
        / "workspaceStorage"
        / "workspace-alpha"
        / "chatSessions"
        / "gemini-body-missing.json"
    )

    assert parse_chat_session_transcript(chat_session_path) is None


def test_parse_chat_session_transcript_rejects_foreign_provider() -> None:
    chat_session_path = (
        FIXTURE_ROOT
        / "Library"
        / "Application Support"
        / "Code"
        / "User"
        / "workspaceStorage"
        / "workspace-alpha"
        / "chatSessions"
        / "copilot-session-1.json"
    )

    attribution = attribute_chat_session(chat_session_path)

    assert attribution is not None
    assert attribution.to_dict() == {
        "session_id": "copilot-session-1",
        "ownership": "foreign",
        "provider": "GitHub Copilot",
        "source_path": str(chat_session_path),
        "request_count": 1,
        "is_empty": True,
        "provider_candidates": [
            "GitHub Copilot",
            "GitHub.copilot-chat",
        ],
        "ownership_reason": "explicit_foreign_provider_marker",
    }
    assert parse_chat_session_transcript(chat_session_path) is None


def test_parse_global_state_returns_metadata_only_unsupported_row() -> None:
    state_path = (
        FIXTURE_ROOT
        / "Library"
        / "Application Support"
        / "Code"
        / "User"
        / "globalStorage"
        / "state.vscdb"
    )
    artifacts = discover_gemini_code_assist_ide_artifacts((FIXTURE_ROOT,))

    conversation = parse_global_state(
        state_path,
        collected_at="2026-03-19T00:00:00Z",
        artifacts=artifacts,
    )

    assert conversation is not None
    payload = conversation.to_dict()
    assert payload["source"] == "gemini_code_assist_ide"
    assert payload["execution_context"] == "ide_extension"
    assert payload["messages"] == []
    assert payload["transcript_completeness"] == "unsupported"
    assert payload["limitations"] == [
        "no_confirmed_gemini_code_assist_ide_transcript_store",
        "metadata_only_ide_state",
        "chat_session_provider_attribution_required",
    ]
    assert payload["source_session_id"] == "vscode:global"
    assert payload["session_metadata"] == {
        "scope": "global_state",
        "has_run_once": True,
        "last_opened_version": "2.73.0",
        "new_chat_is_agent": True,
        "last_chat_mode_was_agent": True,
        "show_agent_tips_card": False,
        "onboarding_tooltip_invoked_once": True,
        "cloudcode_session_index_count": 2,
        "cloudcode_hats_index_count": 1,
        "chat_view_hidden": False,
        "outline_view_hidden": False,
        "credential_artifacts_present": True,
        "credential_artifact_count": 2,
        "install_artifacts_present": True,
        "install_artifact_count": 1,
    }
    assert payload["provenance"] == {
        "source": "vscode",
        "originator": "google.geminicodeassist",
        "conversation_origin": "global_state_residue",
        "app_shell": {
            "application_support_roots": [
                str(
                    FIXTURE_ROOT
                    / "Library"
                    / "Application Support"
                    / "cloud-code"
                ),
                str(
                    FIXTURE_ROOT
                    / "Library"
                    / "Application Support"
                    / "google-vscode-extension"
                ),
            ],
            "state_db_paths": [
                str(state_path),
            ],
        },
    }

    serialized = json.dumps(payload, ensure_ascii=False)
    assert "ya29.mock-access-token" not in serialized
    assert "mock-refresh-token" not in serialized
    assert "super-secret-client-secret" not in serialized
    assert "state-secret-should-not-appear" not in serialized
    assert "install-12345" not in serialized


def test_parse_workspace_state_promotes_gemini_owned_chat_session_to_transcript() -> None:
    state_path = (
        FIXTURE_ROOT
        / "Library"
        / "Application Support"
        / "Code"
        / "User"
        / "workspaceStorage"
        / "workspace-alpha"
        / "state.vscdb"
    )
    artifacts = discover_gemini_code_assist_ide_artifacts((FIXTURE_ROOT,))

    conversation = parse_workspace_state(
        state_path,
        collected_at="2026-03-19T00:00:00Z",
        artifacts=artifacts,
    )

    assert conversation is not None
    payload = conversation.to_dict()
    assert payload["source"] == "gemini_code_assist_ide"
    assert payload["source_session_id"] == "vscode:workspace-alpha:gemini-candidate"
    assert payload["messages"] == [
        {
            "role": "user",
            "text": "Summarize the current Gemini IDE collector.",
            "timestamp": "2026-03-14T10:00:00Z",
            "source_message_id": "request-1",
        },
        {
            "role": "assistant",
            "text": "The collector only emitted metadata-only rows before this upgrade.",
            "timestamp": "2026-03-14T10:00:04Z",
            "source_message_id": "response-1",
        },
        {
            "role": "user",
            "text": "Keep auth files and install residue out of transcript messages.",
            "timestamp": "2026-03-14T10:01:00Z",
            "source_message_id": "request-2",
        },
        {
            "role": "assistant",
            "text": "Confirmed.\n\nThose files remain provenance-only artifacts.",
            "timestamp": "2026-03-14T10:01:06Z",
            "source_message_id": "request-2:response-2",
        },
        {
            "role": "user",
            "text": "Recover the segmented Gemini body shape too.",
            "timestamp": "2026-03-14T10:02:00Z",
            "source_message_id": "request-3",
        },
        {
            "role": "assistant",
            "text": "Recovered from the alternate segmented Gemini response body.",
            "timestamp": "2026-03-14T10:02:08Z",
            "source_message_id": "request-3:response-3",
        },
    ]
    assert "transcript_completeness" not in payload
    assert payload["session_metadata"] == {
        "scope": "chat_session",
        "workspace_id": "workspace-alpha",
        "workspace_folder": "/Users/chenjing/dev/chat-collector",
        "chat_view_state": {
            "collapsed": False,
            "is_hidden": True,
            "size": 972,
        },
        "outline_view_state": {
            "collapsed": False,
            "is_hidden": False,
            "size": 256,
        },
        "chat_view_memento_keys": ["debug", "selectedTab"],
        "number_of_visible_chat_views": 1,
        "chat_session_index_version": 1,
        "indexed_session_count": 3,
        "empty_indexed_session_count": 1,
        "latest_indexed_message_at": "2026-03-14T10:05:00Z",
        "chat_session_attribution": [
            {
                "session_id": "gemini-candidate",
                "ownership": "gemini",
                "provider": "Gemini Code Assist",
                "source_path": str(
                    FIXTURE_ROOT
                    / "Library"
                    / "Application Support"
                    / "Code"
                    / "User"
                    / "workspaceStorage"
                    / "workspace-alpha"
                    / "chatSessions"
                    / "gemini-candidate.json"
                ),
                "request_count": 3,
                "is_empty": False,
                "provider_candidates": [
                    "Gemini Code Assist",
                    "google.geminicodeassist",
                ],
                "ownership_reason": "explicit_gemini_provider_marker",
            },
            {
                "session_id": "gemini-body-missing",
                "ownership": "gemini",
                "provider": "Gemini Code Assist",
                "source_path": str(
                    FIXTURE_ROOT
                    / "Library"
                    / "Application Support"
                    / "Code"
                    / "User"
                    / "workspaceStorage"
                    / "workspace-alpha"
                    / "chatSessions"
                    / "gemini-body-missing.json"
                ),
                "request_count": 1,
                "is_empty": False,
                "provider_candidates": [
                    "Gemini Code Assist",
                    "google.geminicodeassist",
                ],
                "ownership_reason": "explicit_gemini_provider_marker",
            },
            {
                "session_id": "copilot-session-1",
                "ownership": "foreign",
                "provider": "GitHub Copilot",
                "source_path": str(
                    FIXTURE_ROOT
                    / "Library"
                    / "Application Support"
                    / "Code"
                    / "User"
                    / "workspaceStorage"
                    / "workspace-alpha"
                    / "chatSessions"
                    / "copilot-session-1.json"
                ),
                "request_count": 1,
                "is_empty": True,
                "provider_candidates": [
                    "GitHub Copilot",
                    "GitHub.copilot-chat",
                ],
                "ownership_reason": "explicit_foreign_provider_marker",
            }
        ],
        "gemini_owned_chat_session_count": 2,
        "foreign_chat_session_count": 1,
        "unknown_chat_session_count": 0,
        "chat_session_id": "gemini-candidate",
        "chat_session_ownership": "gemini",
        "chat_session_provider": "Gemini Code Assist",
        "chat_session_request_count": 3,
        "chat_session_is_empty": False,
        "user_message_count": 3,
        "assistant_message_count": 3,
        "credential_artifacts_present": True,
        "install_artifacts_present": True,
    }
    assert payload["provenance"] == {
        "session_started_at": "2026-03-14T10:00:00Z",
        "source": "vscode",
        "originator": "google.geminicodeassist",
        "cwd": "/Users/chenjing/dev/chat-collector",
        "app_shell": {
            "application_support_roots": [
                str(
                    FIXTURE_ROOT
                    / "Library"
                    / "Application Support"
                    / "cloud-code"
                ),
                str(
                    FIXTURE_ROOT
                    / "Library"
                    / "Application Support"
                    / "google-vscode-extension"
                ),
            ],
            "state_db_paths": [
                str(
                    FIXTURE_ROOT
                    / "Library"
                    / "Application Support"
                    / "Code"
                    / "User"
                    / "globalStorage"
                    / "state.vscdb"
                ),
                str(state_path),
            ],
        },
    }

    serialized = json.dumps(payload, ensure_ascii=False)
    assert "Copilot empty shell request text must stay out." not in serialized
    assert "ya29.mock-access-token" not in serialized


def test_parse_workspace_state_rows_emits_residue_row_for_body_missing_and_foreign_sessions() -> None:
    state_path = (
        FIXTURE_ROOT
        / "Library"
        / "Application Support"
        / "Code"
        / "User"
        / "workspaceStorage"
        / "workspace-alpha"
        / "state.vscdb"
    )
    artifacts = discover_gemini_code_assist_ide_artifacts((FIXTURE_ROOT,))

    rows = parse_workspace_state_rows(
        state_path,
        collected_at="2026-03-19T00:00:00Z",
        artifacts=artifacts,
    )

    assert [row.source_session_id for row in rows] == [
        "vscode:workspace-alpha:gemini-candidate",
        "vscode:workspace-alpha:residue",
    ]
    residue_payload = rows[1].to_dict()
    assert residue_payload["messages"] == []
    assert residue_payload["transcript_completeness"] == "unsupported"
    assert residue_payload["limitations"] == [
        "metadata_only_ide_state",
        "gemini_owned_chat_session_body_missing",
        "foreign_chat_session_rejected",
    ]
    assert residue_payload["session_metadata"] == {
        "scope": "workspace_state_residue",
        "workspace_id": "workspace-alpha",
        "workspace_folder": "/Users/chenjing/dev/chat-collector",
        "chat_view_state": {
            "collapsed": False,
            "is_hidden": True,
            "size": 972,
        },
        "outline_view_state": {
            "collapsed": False,
            "is_hidden": False,
            "size": 256,
        },
        "chat_view_memento_keys": ["debug", "selectedTab"],
        "number_of_visible_chat_views": 1,
        "chat_session_index_version": 1,
        "indexed_session_count": 3,
        "empty_indexed_session_count": 1,
        "latest_indexed_message_at": "2026-03-14T10:05:00Z",
        "chat_session_attribution": [
            {
                "session_id": "gemini-candidate",
                "ownership": "gemini",
                "provider": "Gemini Code Assist",
                "source_path": str(
                    FIXTURE_ROOT
                    / "Library"
                    / "Application Support"
                    / "Code"
                    / "User"
                    / "workspaceStorage"
                    / "workspace-alpha"
                    / "chatSessions"
                    / "gemini-candidate.json"
                ),
                "request_count": 3,
                "is_empty": False,
                "provider_candidates": [
                    "Gemini Code Assist",
                    "google.geminicodeassist",
                ],
                "ownership_reason": "explicit_gemini_provider_marker",
            },
            {
                "session_id": "gemini-body-missing",
                "ownership": "gemini",
                "provider": "Gemini Code Assist",
                "source_path": str(
                    FIXTURE_ROOT
                    / "Library"
                    / "Application Support"
                    / "Code"
                    / "User"
                    / "workspaceStorage"
                    / "workspace-alpha"
                    / "chatSessions"
                    / "gemini-body-missing.json"
                ),
                "request_count": 1,
                "is_empty": False,
                "provider_candidates": [
                    "Gemini Code Assist",
                    "google.geminicodeassist",
                ],
                "ownership_reason": "explicit_gemini_provider_marker",
            },
            {
                "session_id": "copilot-session-1",
                "ownership": "foreign",
                "provider": "GitHub Copilot",
                "source_path": str(
                    FIXTURE_ROOT
                    / "Library"
                    / "Application Support"
                    / "Code"
                    / "User"
                    / "workspaceStorage"
                    / "workspace-alpha"
                    / "chatSessions"
                    / "copilot-session-1.json"
                ),
                "request_count": 1,
                "is_empty": True,
                "provider_candidates": [
                    "GitHub Copilot",
                    "GitHub.copilot-chat",
                ],
                "ownership_reason": "explicit_foreign_provider_marker",
            },
        ],
        "gemini_owned_chat_session_count": 2,
        "foreign_chat_session_count": 1,
        "unknown_chat_session_count": 0,
        "credential_artifacts_present": True,
        "install_artifacts_present": True,
        "transcript_chat_session_count": 1,
        "metadata_only_chat_session_residue": [
            {
                "session_id": "gemini-body-missing",
                "ownership": "gemini",
                "provider": "Gemini Code Assist",
                "source_path": str(
                    FIXTURE_ROOT
                    / "Library"
                    / "Application Support"
                    / "Code"
                    / "User"
                    / "workspaceStorage"
                    / "workspace-alpha"
                    / "chatSessions"
                    / "gemini-body-missing.json"
                ),
                "request_count": 1,
                "is_empty": False,
                "provider_candidates": [
                    "Gemini Code Assist",
                    "google.geminicodeassist",
                ],
                "ownership_reason": "explicit_gemini_provider_marker",
                "residue_kind": "gemini_provider_explicit_but_body_missing",
                "limitations": [
                    "gemini_owned_chat_session_body_missing",
                ],
            },
            {
                "session_id": "copilot-session-1",
                "ownership": "foreign",
                "provider": "GitHub Copilot",
                "source_path": str(
                    FIXTURE_ROOT
                    / "Library"
                    / "Application Support"
                    / "Code"
                    / "User"
                    / "workspaceStorage"
                    / "workspace-alpha"
                    / "chatSessions"
                    / "copilot-session-1.json"
                ),
                "request_count": 1,
                "is_empty": True,
                "provider_candidates": [
                    "GitHub Copilot",
                    "GitHub.copilot-chat",
                ],
                "ownership_reason": "explicit_foreign_provider_marker",
                "residue_kind": "foreign_provider_rejected",
                "limitations": [
                    "foreign_chat_session_rejected",
                ],
            },
        ],
        "metadata_only_chat_session_residue_count": 2,
    }
    assert residue_payload["provenance"] == {
        "source": "vscode",
        "originator": "google.geminicodeassist",
        "cwd": "/Users/chenjing/dev/chat-collector",
        "conversation_origin": "workspace_chat_session_residue",
        "app_shell": {
            "application_support_roots": [
                str(
                    FIXTURE_ROOT
                    / "Library"
                    / "Application Support"
                    / "cloud-code"
                ),
                str(
                    FIXTURE_ROOT
                    / "Library"
                    / "Application Support"
                    / "google-vscode-extension"
                ),
            ],
            "state_db_paths": [
                str(
                    FIXTURE_ROOT
                    / "Library"
                    / "Application Support"
                    / "Code"
                    / "User"
                    / "globalStorage"
                    / "state.vscdb"
                ),
                str(state_path),
            ],
        },
    }


def test_parse_workspace_state_requires_gemini_specific_key_space() -> None:
    state_path = (
        FIXTURE_ROOT
        / "Library"
        / "Application Support"
        / "Code"
        / "User"
        / "workspaceStorage"
        / "workspace-beta"
        / "state.vscdb"
    )

    assert (
        parse_workspace_state(
            state_path,
            collected_at="2026-03-19T00:00:00Z",
            artifacts=discover_gemini_code_assist_ide_artifacts((FIXTURE_ROOT,)),
        )
        is None
    )


def test_gemini_code_assist_ide_collect_writes_global_and_workspace_rows(
    tmp_path: Path,
) -> None:
    collector = GeminiCodeAssistIdeCollector()

    result = collector.collect(tmp_path, input_roots=(FIXTURE_ROOT,))

    assert result.source == "gemini_code_assist_ide"
    assert result.scanned_artifact_count == 3
    assert result.conversation_count == 3
    assert result.message_count == 6
    rows = read_jsonl(result.output_path)
    assert [row["source_session_id"] for row in rows] == [
        "vscode:global",
        "vscode:workspace-alpha:gemini-candidate",
        "vscode:workspace-alpha:residue",
    ]
    serialized = json.dumps(rows, ensure_ascii=False)
    assert "Copilot empty shell request text must stay out." not in serialized
    assert "mock-refresh-token" not in serialized
    assert "install-12345" not in serialized


def test_cli_collect_gemini_code_assist_ide_plan_and_execute(tmp_path: Path) -> None:
    plan_result = run_cli("collect", "gemini_code_assist_ide", "--archive-root", str(tmp_path))

    assert plan_result.returncode == 0
    plan_payload = json.loads(plan_result.stdout)
    assert plan_payload["source"] == "gemini_code_assist_ide"
    assert plan_payload["implemented"] is True
    assert plan_payload["support_level"] == "partial"

    execute_result = run_cli(
        "collect",
        "gemini_code_assist_ide",
        "--archive-root",
        str(tmp_path),
        "--input-root",
        str(FIXTURE_ROOT),
        "--execute",
    )

    assert execute_result.returncode == 0
    execute_payload = json.loads(execute_result.stdout)
    assert execute_payload["source"] == "gemini_code_assist_ide"
    assert execute_payload["scanned_artifact_count"] == 3
    assert execute_payload["conversation_count"] == 3
    assert execute_payload["message_count"] == 6
    rows = read_jsonl(Path(execute_payload["output_path"]))
    assert len(rows) == 3
