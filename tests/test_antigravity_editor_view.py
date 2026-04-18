from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from llm_chat_archive.sources.antigravity_editor_view import (
    AntigravityEditorViewCollector,
    discover_antigravity_editor_view_artifacts,
    parse_conversation_blob,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "antigravity_editor_view"
SESSION_ALPHA = "11111111-1111-4111-8111-111111111111"
SESSION_BETA = "22222222-2222-4222-8222-222222222222"
SESSION_ORPHAN = "33333333-3333-4333-8333-333333333333"
UNKNOWN_PROTOBUF_TEXT = "Unknown protobuf entry that must stay out of the normalized transcript."
ALPHA_MESSAGES = (
    {
        "id": "alpha-user-1",
        "role": 1,
        "text": "Inspect the Antigravity collector entry points.",
        "timestamp": "2026-03-14T10:00:00Z",
    },
    {
        "id": "alpha-assistant-1",
        "role": 2,
        "text": "I will reconstruct the confirmed conversation protobuf transcript.",
        "timestamp": "2026-03-14T10:00:01Z",
    },
    {
        "id": "alpha-user-2",
        "role": 1,
        "text": "Keep brain artifacts and browser recordings out of the transcript body.",
        "timestamp": "2026-03-14T10:00:02Z",
    },
    {
        "id": "alpha-assistant-2",
        "role": 2,
        "text": "Only verified user and assistant turns will be normalized.",
        "timestamp": "2026-03-14T10:00:03Z",
    },
)
BETA_MESSAGES = (
    {
        "id": "beta-user-1",
        "role": 1,
        "text": "Show the degraded fallback behavior.",
        "timestamp": "2026-03-14T11:30:00Z",
    },
    {
        "id": "beta-assistant-1",
        "role": 2,
        "text": "I kept the confirmed messages and dropped the unconfirmed protobuf entry.",
        "timestamp": "2026-03-14T11:30:02Z",
    },
)


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "llm_chat_archive", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def conversation_path(root: Path, session_id: str) -> Path:
    return (
        root
        / ".gemini"
        / "antigravity"
        / "conversations"
        / f"{session_id}.pb"
    )


def prepare_decodable_fixture_root(tmp_path: Path) -> Path:
    target = tmp_path / "antigravity_editor_view"
    shutil.copytree(FIXTURE_ROOT, target)
    conversation_path(target, SESSION_ALPHA).write_bytes(
        encode_confirmed_conversation(SESSION_ALPHA, ALPHA_MESSAGES)
    )
    conversation_path(target, SESSION_BETA).write_bytes(
        encode_confirmed_conversation(
            SESSION_BETA,
            BETA_MESSAGES,
            extra_message_payloads=(
                encode_unconfirmed_message(
                    UNKNOWN_PROTOBUF_TEXT,
                    "2026-03-14T11:30:01Z",
                ),
            ),
            insert_unknown_after=1,
        )
    )
    return target


def encode_confirmed_conversation(
    session_id: str,
    messages: tuple[dict[str, object], ...],
    *,
    extra_message_payloads: tuple[bytes, ...] = (),
    insert_unknown_after: int | None = None,
) -> bytes:
    parts = [encode_string_field(1, session_id)]
    for index, message in enumerate(messages, start=1):
        parts.append(encode_length_delimited_field(2, encode_confirmed_message(message)))
        if insert_unknown_after == index:
            parts.extend(encode_length_delimited_field(2, payload) for payload in extra_message_payloads)
    if insert_unknown_after is None:
        parts.extend(encode_length_delimited_field(2, payload) for payload in extra_message_payloads)
    return b"".join(parts)


def encode_confirmed_message(message: dict[str, object]) -> bytes:
    return b"".join(
        (
            encode_string_field(1, str(message["id"])),
            encode_varint_field(2, int(message["role"])),
            encode_string_field(3, str(message["text"])),
            encode_string_field(4, str(message["timestamp"])),
        )
    )


def encode_unconfirmed_message(text: str, timestamp: str) -> bytes:
    return b"".join(
        (
            encode_string_field(8, text),
            encode_string_field(9, timestamp),
        )
    )


def encode_length_delimited_field(field_number: int, payload: bytes) -> bytes:
    return encode_varint((field_number << 3) | 2) + encode_varint(len(payload)) + payload


def encode_string_field(field_number: int, value: str) -> bytes:
    return encode_length_delimited_field(field_number, value.encode("utf-8"))


def encode_varint_field(field_number: int, value: int) -> bytes:
    return encode_varint((field_number << 3) | 0) + encode_varint(value)


def encode_varint(value: int) -> bytes:
    if value < 0:
        raise ValueError("protobuf varints must be non-negative in fixtures")
    encoded = bytearray()
    remaining = value
    while True:
        next_byte = remaining & 0x7F
        remaining >>= 7
        if remaining:
            encoded.append(next_byte | 0x80)
            continue
        encoded.append(next_byte)
        return bytes(encoded)


def test_discover_antigravity_editor_view_artifacts_indexes_session_sidecars() -> None:
    artifacts = discover_antigravity_editor_view_artifacts((FIXTURE_ROOT,))

    assert artifacts.conversation_paths == (
        str(conversation_path(FIXTURE_ROOT, SESSION_ALPHA)),
        str(conversation_path(FIXTURE_ROOT, SESSION_BETA)),
    )
    assert artifacts.brain_dirs == (
        str(FIXTURE_ROOT / ".gemini" / "antigravity" / "brain" / SESSION_ALPHA),
        str(FIXTURE_ROOT / ".gemini" / "antigravity" / "brain" / SESSION_ORPHAN),
    )
    assert artifacts.annotation_paths == (
        str(
            FIXTURE_ROOT
            / ".gemini"
            / "antigravity"
            / "annotations"
            / f"{SESSION_ALPHA}.pbtxt"
        ),
    )
    assert artifacts.browser_recording_dirs == (
        str(
            FIXTURE_ROOT
            / ".gemini"
            / "antigravity"
            / "browser_recordings"
            / SESSION_ALPHA
        ),
        str(
            FIXTURE_ROOT
            / ".gemini"
            / "antigravity"
            / "browser_recordings"
            / SESSION_ORPHAN
        ),
    )
    assert artifacts.global_state_paths == (
        str(
            FIXTURE_ROOT
            / "Library"
            / "Application Support"
            / "Antigravity"
            / "User"
            / "globalStorage"
            / "state.vscdb"
        ),
    )
    assert artifacts.workspace_state_paths == (
        str(
            FIXTURE_ROOT
            / "Library"
            / "Application Support"
            / "Antigravity"
            / "User"
            / "workspaceStorage"
            / "workspace-alpha"
            / "state.vscdb"
        ),
    )


def test_parse_conversation_blob_reconstructs_confirmed_transcript(tmp_path: Path) -> None:
    fixture_root = prepare_decodable_fixture_root(tmp_path)
    resolved_conversation_path = conversation_path(fixture_root, SESSION_ALPHA)
    artifacts = discover_antigravity_editor_view_artifacts((fixture_root,))

    conversation = parse_conversation_blob(
        resolved_conversation_path,
        collected_at="2026-03-19T00:00:00Z",
        artifacts=artifacts,
    )

    assert conversation is not None
    payload = conversation.to_dict()
    assert payload["source"] == "antigravity_editor_view"
    assert payload["execution_context"] == "ide_native"
    assert "transcript_completeness" not in payload
    assert "limitations" not in payload
    assert payload["source_session_id"] == SESSION_ALPHA
    assert payload["messages"] == [
        {
            "role": "user",
            "text": "Inspect the Antigravity collector entry points.",
            "timestamp": "2026-03-14T10:00:00Z",
            "source_message_id": "alpha-user-1",
        },
        {
            "role": "assistant",
            "text": "I will reconstruct the confirmed conversation protobuf transcript.",
            "timestamp": "2026-03-14T10:00:01Z",
            "source_message_id": "alpha-assistant-1",
        },
        {
            "role": "user",
            "text": "Keep brain artifacts and browser recordings out of the transcript body.",
            "timestamp": "2026-03-14T10:00:02Z",
            "source_message_id": "alpha-user-2",
        },
        {
            "role": "assistant",
            "text": "Only verified user and assistant turns will be normalized.",
            "timestamp": "2026-03-14T10:00:03Z",
            "source_message_id": "alpha-assistant-2",
        },
    ]
    assert payload["session_metadata"]["conversation_blob"] == {
        "path": str(resolved_conversation_path),
        "size_bytes": resolved_conversation_path.stat().st_size,
        "decode_status": "decoded",
        "protobuf_session_id": SESSION_ALPHA,
        "confirmed_field_mapping": {
            "session_id_field": 1,
            "message_field": 2,
            "message_id_field": 1,
            "message_role_field": 2,
            "message_text_field": 3,
            "message_timestamp_field": 4,
        },
        "recovered_message_count": 4,
        "user_message_count": 2,
        "assistant_message_count": 2,
    }
    assert payload["provenance"]["session_started_at"] == "2026-03-14T10:00:00Z"
    serialized = json.dumps(payload, ensure_ascii=False)
    assert "Task markdown body must stay out of transcript." not in serialized
    assert "Implementation plan resolved body must stay out of transcript." not in serialized
    assert "Browser frame text must stay out of transcript." not in serialized
    assert "Daemon log body must stay out of transcript." not in serialized
    assert "HTML artifact body must stay out of transcript." not in serialized
    assert UNKNOWN_PROTOBUF_TEXT not in serialized
    assert "opaque-shared-state-blob" not in serialized


def test_parse_conversation_blob_marks_unconfirmed_entries_partial(tmp_path: Path) -> None:
    fixture_root = prepare_decodable_fixture_root(tmp_path)
    resolved_conversation_path = conversation_path(fixture_root, SESSION_BETA)
    artifacts = discover_antigravity_editor_view_artifacts((fixture_root,))

    conversation = parse_conversation_blob(
        resolved_conversation_path,
        collected_at="2026-03-19T00:00:00Z",
        artifacts=artifacts,
    )

    assert conversation is not None
    payload = conversation.to_dict()
    assert payload["messages"] == [
        {
            "role": "user",
            "text": "Show the degraded fallback behavior.",
            "timestamp": "2026-03-14T11:30:00Z",
            "source_message_id": "beta-user-1",
        },
        {
            "role": "assistant",
            "text": "I kept the confirmed messages and dropped the unconfirmed protobuf entry.",
            "timestamp": "2026-03-14T11:30:02Z",
            "source_message_id": "beta-assistant-1",
        },
    ]
    assert payload["transcript_completeness"] == "partial"
    assert payload["limitations"] == ["variant_unknown"]
    assert payload["session_metadata"]["conversation_blob"] == {
        "path": str(resolved_conversation_path),
        "size_bytes": resolved_conversation_path.stat().st_size,
        "decode_status": "partially_decoded",
        "diagnostic_reason": "variant_unknown",
        "diagnostic_details": {
            "reason": "message_field_mapping_unconfirmed",
            "scope": "message_entry",
            "skipped_message_count": 1,
        },
        "protobuf_session_id": SESSION_BETA,
        "confirmed_field_mapping": {
            "session_id_field": 1,
            "message_field": 2,
            "message_id_field": 1,
            "message_role_field": 2,
            "message_text_field": 3,
            "message_timestamp_field": 4,
        },
        "recovered_message_count": 2,
        "user_message_count": 1,
        "assistant_message_count": 1,
        "skipped_message_count": 1,
    }
    serialized = json.dumps(payload, ensure_ascii=False)
    assert UNKNOWN_PROTOBUF_TEXT not in serialized
    assert "Task markdown body must stay out of transcript." not in serialized


def test_parse_conversation_blob_returns_unsupported_metadata_only_row_for_decode_failures() -> None:
    resolved_conversation_path = conversation_path(FIXTURE_ROOT, SESSION_ALPHA)
    global_state_path = (
        FIXTURE_ROOT
        / "Library"
        / "Application Support"
        / "Antigravity"
        / "User"
        / "globalStorage"
        / "state.vscdb"
    )
    workspace_state_path = (
        FIXTURE_ROOT
        / "Library"
        / "Application Support"
        / "Antigravity"
        / "User"
        / "workspaceStorage"
        / "workspace-alpha"
        / "state.vscdb"
    )
    artifacts = discover_antigravity_editor_view_artifacts((FIXTURE_ROOT,))

    conversation = parse_conversation_blob(
        resolved_conversation_path,
        collected_at="2026-03-19T00:00:00Z",
        artifacts=artifacts,
    )

    assert conversation is not None
    payload = conversation.to_dict()
    assert payload["source"] == "antigravity_editor_view"
    assert payload["execution_context"] == "ide_native"
    assert payload["messages"] == []
    assert payload["transcript_completeness"] == "unsupported"
    assert payload["limitations"] == [
        "variant_unknown",
        "decode_failed",
        "metadata_only_session_family",
    ]
    assert payload["source_session_id"] == SESSION_ALPHA
    assert payload["session_metadata"]["conversation_blob"]["path"] == str(
        resolved_conversation_path
    )
    assert payload["session_metadata"]["conversation_blob"]["size_bytes"] == (
        resolved_conversation_path.stat().st_size
    )
    assert payload["session_metadata"]["conversation_blob"]["decode_status"] == "decode_failed"
    assert payload["session_metadata"]["conversation_blob"]["diagnostic_reason"] == (
        "variant_unknown"
    )
    assert payload["session_metadata"]["conversation_blob"]["diagnostic_details"] == {
        "failure_offset": 0,
        "failure_stage": "field",
        "field_number": 9,
        "wire_type": 7,
        "reason": "unknown_top_level_wire_type",
    }
    assert payload["session_metadata"]["conversation_blob"]["decode_error"].startswith(
        "variant_unknown: top-level wire type 7"
    )
    assert payload["session_metadata"]["noise_separation"] == {
        "excluded_from_messages": [
            "browser_recordings",
            "html_artifacts",
            "daemon_logs",
            "unified_state_sync_blobs",
        ],
        "html_artifact_root_count": 1,
        "daemon_artifact_count": 1,
    }
    assert payload["session_metadata"]["brain"] == {
        "path": str(FIXTURE_ROOT / ".gemini" / "antigravity" / "brain" / SESSION_ALPHA),
        "artifact_names": [
            "implementation_plan.md",
            "task.md",
            "walkthrough.md",
        ],
        "artifact_summaries": [
            {
                "name": "implementation_plan.md",
                "artifact_type": "ARTIFACT_TYPE_IMPLEMENTATION_PLAN",
                "updated_at": "2026-03-14T10:06:00Z",
                "version": 2,
                "has_summary": True,
            },
            {
                "name": "task.md",
                "artifact_type": "ARTIFACT_TYPE_TASK",
                "updated_at": "2026-03-14T10:05:00Z",
                "version": 1,
                "has_summary": True,
            },
            {
                "name": "walkthrough.md",
                "artifact_type": "ARTIFACT_TYPE_WALKTHROUGH",
                "updated_at": "2026-03-14T10:07:00Z",
                "version": 3,
            },
        ],
        "resolved_artifact_count": 2,
        "image_artifact_count": 1,
    }
    assert payload["session_metadata"]["annotation"] == {
        "path": str(
            FIXTURE_ROOT
            / ".gemini"
            / "antigravity"
            / "annotations"
            / f"{SESSION_ALPHA}.pbtxt"
        ),
        "fields": {
            "last_user_view_time": "2026-03-14T10:08:00Z",
            "annotation_version": 4,
        },
    }
    assert payload["session_metadata"]["browser_recording"] == {
        "path": str(
            FIXTURE_ROOT
            / ".gemini"
            / "antigravity"
            / "browser_recordings"
            / SESSION_ALPHA
        ),
        "frame_count": 2,
    }
    assert payload["session_metadata"]["shared_state"] == {
        "global_state": [
            {
                "state_db_path": str(global_state_path),
                "matched_keys": [
                    "antigravityUnifiedStateSync.artifactReview",
                    "antigravityUnifiedStateSync.trajectorySummaries",
                ],
            }
        ],
        "workspace_state": [
            {
                "state_db_path": str(workspace_state_path),
                "workspace_id": "workspace-alpha",
                "matched_keys": [
                    "memento/antigravity.jetskiArtifactsEditor",
                ],
                "workspace_folder": "/Users/chenjing/dev/chat-collector",
                "chat_session_store_index_version": 1,
                "chat_session_store_entry_count": 0,
            }
        ],
    }
    assert payload["provenance"] == {
        "source": "antigravity",
        "originator": "antigravity_editor_view",
        "cwd": "/Users/chenjing/dev/chat-collector",
        "app_shell": {
            "application_support_roots": [
                str(FIXTURE_ROOT / "Library" / "Application Support" / "Antigravity"),
            ],
            "log_roots": [
                str(
                    FIXTURE_ROOT
                    / "Library"
                    / "Application Support"
                    / "Antigravity"
                    / "logs"
                ),
            ],
            "state_db_paths": [
                str(global_state_path),
                str(workspace_state_path),
            ],
            "log_paths": [
                str(
                    FIXTURE_ROOT
                    / ".gemini"
                    / "antigravity"
                    / "daemon"
                    / "ls_c318d4f90fc5aacc.log"
                ),
            ],
            "auxiliary_paths": [
                str(FIXTURE_ROOT / ".gemini" / "antigravity" / "html_artifacts"),
            ],
        },
    }

    serialized = json.dumps(payload, ensure_ascii=False)
    assert "Opaque transcript text that must stay undecoded." not in serialized
    assert "Task markdown body must stay out of transcript." not in serialized
    assert "Implementation plan resolved body must stay out of transcript." not in serialized
    assert "Browser frame text must stay out of transcript." not in serialized
    assert "Daemon log body must stay out of transcript." not in serialized
    assert "HTML artifact body must stay out of transcript." not in serialized
    assert "opaque-shared-state-blob" not in serialized


def test_antigravity_editor_view_collect_writes_one_row_per_conversation_blob(
    tmp_path: Path,
) -> None:
    fixture_root = prepare_decodable_fixture_root(tmp_path / "fixture-data")
    collector = AntigravityEditorViewCollector()

    result = collector.collect(tmp_path, input_roots=(fixture_root,))

    assert result.source == "antigravity_editor_view"
    assert result.scanned_artifact_count == 2
    assert result.conversation_count == 2
    assert result.message_count == 6
    rows = read_jsonl(result.output_path)
    assert [row["source_session_id"] for row in rows] == [SESSION_ALPHA, SESSION_BETA]
    assert SESSION_ORPHAN not in {row["source_session_id"] for row in rows}
    alpha_row = rows[0]
    beta_row = rows[1]
    assert len(alpha_row["messages"]) == 4
    assert "transcript_completeness" not in alpha_row
    assert len(beta_row["messages"]) == 2
    assert beta_row["transcript_completeness"] == "partial"
    assert beta_row["limitations"] == ["variant_unknown"]
    serialized = json.dumps(rows, ensure_ascii=False)
    assert "Task markdown body must stay out of transcript." not in serialized
    assert "Browser frame text must stay out of transcript." not in serialized
    assert UNKNOWN_PROTOBUF_TEXT not in serialized
    assert SESSION_ORPHAN not in serialized


def test_cli_collect_antigravity_editor_view_plan_and_execute(tmp_path: Path) -> None:
    fixture_root = prepare_decodable_fixture_root(tmp_path / "fixture-data")
    plan_result = run_cli("collect", "antigravity_editor_view", "--archive-root", str(tmp_path))

    assert plan_result.returncode == 0
    plan_payload = json.loads(plan_result.stdout)
    assert plan_payload["source"] == "antigravity_editor_view"
    assert plan_payload["implemented"] is True
    assert plan_payload["support_level"] == "partial"

    execute_result = run_cli(
        "collect",
        "antigravity_editor_view",
        "--archive-root",
        str(tmp_path),
        "--input-root",
        str(fixture_root),
        "--execute",
    )

    assert execute_result.returncode == 0
    execute_payload = json.loads(execute_result.stdout)
    assert execute_payload["source"] == "antigravity_editor_view"
    assert execute_payload["scanned_artifact_count"] == 2
    assert execute_payload["conversation_count"] == 2
    assert execute_payload["message_count"] == 6
    rows = read_jsonl(Path(execute_payload["output_path"]))
    assert len(rows) == 2
    assert len(rows[0]["messages"]) == 4
    assert rows[1]["transcript_completeness"] == "partial"
