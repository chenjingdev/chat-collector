from __future__ import annotations

import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from llm_chat_archive import cli


def run_cli(*args: str) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        exit_code = cli.main(list(args))
    return exit_code, stdout.getvalue(), stderr.getvalue()


def write_archive_output(
    archive_root: Path,
    *,
    source: str,
    filename: str,
    rows: tuple[dict[str, object], ...],
) -> Path:
    output_path = archive_root / source / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    return output_path


def make_conversation(
    source: str,
    *,
    session: str | None,
    collected_at: str,
    messages: list[dict[str, object]],
    artifact_path: str | None = None,
    transcript_completeness: str | None = None,
    limitations: list[str] | None = None,
    session_metadata: dict[str, object] | None = None,
    provenance: dict[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "source": source,
        "execution_context": "cli",
        "collected_at": collected_at,
        "messages": messages,
        "contract": {"schema_version": "2026-03-19"},
    }
    if session is not None:
        payload["source_session_id"] = session
    if artifact_path is not None:
        payload["source_artifact_path"] = artifact_path
    if transcript_completeness is not None:
        payload["transcript_completeness"] = transcript_completeness
    if limitations is not None:
        payload["limitations"] = limitations
    if session_metadata is not None:
        payload["session_metadata"] = session_metadata
    if provenance is not None:
        payload["provenance"] = provenance
    return payload


def seed_archive_root(archive_root: Path) -> None:
    write_archive_output(
        archive_root,
        source="codex_cli",
        filename="memory_chat_v1-20260319T060000-codex_cli.jsonl",
        rows=(
            make_conversation(
                "codex_cli",
                session="alpha",
                artifact_path="/tmp/codex-alpha.jsonl",
                collected_at="2026-03-19T06:00:00Z",
                transcript_completeness="partial",
                limitations=["missing assistant reply"],
                messages=[
                    {"role": "user", "text": "Need deploy checklist"},
                ],
            ),
            make_conversation(
                "codex_cli",
                session="alpha",
                collected_at="2026-03-19T06:10:00Z",
                messages=[
                    {"role": "user", "text": "Need deploy checklist"},
                    {"role": "assistant", "text": "Start with release notes"},
                ],
            ),
        ),
    )
    write_archive_output(
        archive_root,
        source="codex_cli",
        filename="memory_chat_v1-20260319T062000-codex_cli.jsonl",
        rows=(
            make_conversation(
                "codex_cli",
                session="beta",
                artifact_path="/tmp/codex-alpha.jsonl",
                collected_at="2026-03-19T06:20:00Z",
                session_metadata={"cwd": "/tmp/chat-collector"},
                provenance={"originator": "codex"},
                messages=[
                    {"role": "user", "text": "Need deploy checklist"},
                    {"role": "assistant", "text": "Start with release notes"},
                    {"role": "assistant", "text": "Track the rollback owner too"},
                ],
            ),
        ),
    )
    write_archive_output(
        archive_root,
        source="cursor_editor",
        filename="memory_chat_v1-20260319T070000-cursor_editor.jsonl",
        rows=(
            make_conversation(
                "cursor_editor",
                session="cursor-session-1",
                artifact_path="/tmp/cursor-session-1.json",
                collected_at="2026-03-19T07:00:00Z",
                messages=[
                    {"role": "user", "text": "Need onboarding outline"},
                    {"role": "assistant", "text": "Summarize the docs"},
                ],
            ),
        ),
    )


def test_archive_audit_identities_reports_source_grouped_collisions(
    tmp_path: Path,
) -> None:
    seed_archive_root(tmp_path)

    exit_code, stdout, stderr = run_cli(
        "archive",
        "audit-identities",
        "--archive-root",
        str(tmp_path),
    )

    assert exit_code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["archive_root"] == str(tmp_path)
    assert payload["source_filter"] is None
    assert payload["source_count"] == 2
    assert payload["conversation_count"] == 4
    assert payload["collision_source_count"] == 1
    assert payload["collision_group_count"] == 1
    assert payload["collision_row_count"] == 3
    assert payload["richer_collision_group_count"] == 1

    codex_payload = payload["sources"]["codex_cli"]
    assert codex_payload["file_count"] == 2
    assert codex_payload["conversation_count"] == 3
    assert codex_payload["collision_group_count"] == 1
    assert codex_payload["collision_row_count"] == 3
    assert codex_payload["richer_collision_group_count"] == 1
    assert codex_payload["mixed_identity_shape_group_count"] == 1

    collision_group = codex_payload["collision_groups"][0]
    assert collision_group["source"] == "codex_cli"
    assert collision_group["row_count"] == 3
    assert collision_group["source_session_id_count"] == 2
    assert collision_group["source_session_ids"] == ["alpha", "beta"]
    assert collision_group["source_artifact_path_count"] == 1
    assert collision_group["source_artifact_paths"] == ["/tmp/codex-alpha.jsonl"]
    assert collision_group["identity_shapes"] == {
        "session_and_artifact": 2,
        "session_only": 1,
    }
    assert collision_group["distinct_message_fingerprint_count"] == 3
    assert collision_group["has_richer_conversation"] is True
    assert {
        reason["code"] for reason in collision_group["reasons"]
    } == {
        "duplicate_source_session_id",
        "duplicate_source_artifact_path",
        "conflicting_source_session_ids",
        "mixed_identity_shapes",
        "message_variants",
        "richer_transcript_available",
    }

    preferred_conversation = collision_group["preferred_conversation"]
    assert preferred_conversation["source_session_id"] == "beta"
    assert preferred_conversation["source_artifact_path"] == "/tmp/codex-alpha.jsonl"
    assert preferred_conversation["transcript_completeness"] == "complete"
    assert preferred_conversation["message_count"] == 3
    assert preferred_conversation["text_message_count"] == 3
    assert preferred_conversation["has_session_metadata"] is True
    assert preferred_conversation["has_provenance"] is True
    assert preferred_conversation["preferred"] is True

    conversations = collision_group["conversations"]
    assert [conversation["source_session_id"] for conversation in conversations] == [
        "beta",
        "alpha",
        "alpha",
    ]
    assert [conversation["preferred"] for conversation in conversations] == [
        True,
        False,
        False,
    ]
    assert conversations[-1]["transcript_completeness"] == "partial"

    cursor_payload = payload["sources"]["cursor_editor"]
    assert cursor_payload == {
        "file_count": 1,
        "conversation_count": 1,
        "collision_group_count": 0,
        "collision_row_count": 0,
        "richer_collision_group_count": 0,
        "mixed_identity_shape_group_count": 0,
        "collision_groups": [],
    }


def test_archive_audit_identities_filters_to_single_source(tmp_path: Path) -> None:
    seed_archive_root(tmp_path)

    exit_code, stdout, stderr = run_cli(
        "archive",
        "audit-identities",
        "--archive-root",
        str(tmp_path),
        "--source",
        "cursor_editor",
    )

    assert exit_code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["source_filter"] == "cursor_editor"
    assert payload["source_count"] == 1
    assert payload["conversation_count"] == 1
    assert payload["collision_source_count"] == 0
    assert payload["collision_group_count"] == 0
    assert payload["collision_row_count"] == 0
    assert payload["richer_collision_group_count"] == 0
    assert list(payload["sources"]) == ["cursor_editor"]
