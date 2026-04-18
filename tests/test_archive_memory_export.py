from __future__ import annotations

import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from llm_chat_archive import cli
from llm_chat_archive.runner import MANIFEST_FILENAME, RUNS_DIRECTORY


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


def write_run_manifest(
    archive_root: Path,
    *,
    run_id: str,
    sources: tuple[dict[str, object], ...],
) -> Path:
    run_dir = archive_root / RUNS_DIRECTORY / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / MANIFEST_FILENAME
    manifest_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "archive_root": str(archive_root),
                "run_dir": str(run_dir),
                "sources": list(sources),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return manifest_path


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def make_conversation(
    source: str,
    *,
    session: str,
    collected_at: str,
    messages: list[dict[str, object]],
    transcript_completeness: str | None = None,
    limitations: list[str] | None = None,
    provenance: dict[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "source": source,
        "execution_context": "cli" if source == "codex_cli" else "ide_native",
        "collected_at": collected_at,
        "source_session_id": session,
        "source_artifact_path": f"/tmp/{source}/{session}.jsonl",
        "messages": messages,
        "contract": {"schema_version": "2026-03-19"},
    }
    if transcript_completeness is not None:
        payload["transcript_completeness"] = transcript_completeness
    if limitations is not None:
        payload["limitations"] = limitations
    if provenance is not None:
        payload["provenance"] = provenance
    return payload


def seed_archive_root(archive_root: Path) -> dict[str, Path]:
    codex_old_path = write_archive_output(
        archive_root,
        source="codex_cli",
        filename="memory_chat_v1-20260318T080000Z.jsonl",
        rows=(
            make_conversation(
                "codex_cli",
                session="session-old",
                collected_at="2026-03-18T08:00:00Z",
                messages=[
                    {"role": "user", "text": "Older transcript"},
                    {"role": "assistant", "text": "Older answer"},
                ],
                provenance={
                    "source": "cli",
                    "originator": "codex_cli_rs",
                    "session_started_at": "2026-03-18T08:00:00Z",
                },
            ),
        ),
    )
    codex_run_path = write_archive_output(
        archive_root,
        source="codex_cli",
        filename="memory_chat_v1-20260319T090000Z.jsonl",
        rows=(
            make_conversation(
                "codex_cli",
                session="session-redaction",
                collected_at="2026-03-19T09:00:00Z",
                messages=[
                    {
                        "role": "developer",
                        "source_message_id": "msg-dev-redaction",
                        "text": "Keep Authorization: Bearer [REDACTED] out of the archive.",
                    },
                    {
                        "role": "user",
                        "source_message_id": "msg-user-redaction",
                        "text": "Keys to mask: [REDACTED_API_KEY] and [REDACTED_API_KEY].",
                    },
                    {
                        "role": "assistant",
                        "source_message_id": "msg-assistant-redaction",
                        "text": "Credential JSON: {\"access_token\":\"[REDACTED]\"}",
                        "timestamp": "2026-03-19T09:00:03Z",
                    },
                ],
                provenance={
                    "archived": False,
                    "cli_version": "0.32.0",
                    "cwd": "/Users/chenjing/dev/chat-collector",
                    "originator": "codex_cli_rs",
                    "session_started_at": "2026-03-19T09:00:00Z",
                    "source": "cli",
                },
            ),
        ),
    )
    cursor_partial_path = write_archive_output(
        archive_root,
        source="cursor_editor",
        filename="memory_chat_v1-20260319T070000Z.jsonl",
        rows=(
            make_conversation(
                "cursor_editor",
                session="cursor-session-1",
                collected_at="2026-03-19T07:00:00Z",
                transcript_completeness="partial",
                limitations=["missing deleted draft messages"],
                messages=[
                    {"role": "user", "text": "Need deploy checklist follow-up"},
                    {
                        "role": "assistant",
                        "text": "Checklist updated with rollout notes",
                        "timestamp": "2026-03-19T07:00:05Z",
                    },
                ],
                provenance={
                    "originator": "cursor_editor",
                    "session_started_at": "2026-03-19T07:00:00Z",
                    "source": "cursor",
                },
            ),
        ),
    )
    write_run_manifest(
        archive_root,
        run_id="20260319T090000Z",
        sources=(
            {
                "source": "codex_cli",
                "support_level": "complete",
                "status": "complete",
                "output_path": str(codex_run_path),
                "scanned_artifact_count": 1,
                "conversation_count": 1,
                "message_count": 3,
                "skipped_conversation_count": 0,
                "written_conversation_count": 1,
                "upgraded_conversation_count": 0,
                "failed": False,
                "partial": False,
                "unsupported": False,
                "redaction_event_count": 4,
            },
        ),
    )
    return {
        "codex_old_path": codex_old_path,
        "codex_run_path": codex_run_path,
        "cursor_partial_path": cursor_partial_path,
    }


def test_archive_export_memory_dry_run_reports_watermark_filtered_counts(
    tmp_path: Path,
) -> None:
    seed_archive_root(tmp_path)
    output_dir = tmp_path.parent / "memory-export-dry-run"

    exit_code, stdout, stderr = run_cli(
        "archive",
        "export-memory",
        "--archive-root",
        str(tmp_path),
        "--output-dir",
        str(output_dir),
        "--after-collected-at",
        "2026-03-19T08:30:00Z",
    )

    assert exit_code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["conversation_count"] == 1
    assert payload["record_count"] == 1
    assert payload["message_count"] == 3
    assert payload["source_count"] == 1
    assert payload["earliest_collected_at"] == "2026-03-19T09:00:00Z"
    assert payload["latest_collected_at"] == "2026-03-19T09:00:00Z"
    assert payload["filters"] == {
        "after_collected_at": "2026-03-19T08:30:00Z",
        "run_id": None,
        "session": None,
        "source": None,
        "text": None,
        "transcript_completeness": None,
    }
    assert payload["contract"]["record_kind"] == "memory_ingestion_conversation_v1"
    assert payload["records_path"] == str(output_dir / "memory-records.jsonl")
    assert payload["manifest_path"] == str(output_dir / "memory-export-manifest.json")
    assert not (output_dir / "memory-records.jsonl").exists()
    assert not (output_dir / "memory-export-manifest.json").exists()


def test_archive_export_memory_execute_keeps_stable_ids_across_run_and_watermark_exports(
    tmp_path: Path,
) -> None:
    seed_archive_root(tmp_path)
    run_output_dir = tmp_path.parent / "memory-export-run"
    watermark_output_dir = tmp_path.parent / "memory-export-watermark"

    run_exit_code, run_stdout, run_stderr = run_cli(
        "archive",
        "export-memory",
        "--archive-root",
        str(tmp_path),
        "--output-dir",
        str(run_output_dir),
        "--run",
        "20260319T090000Z",
        "--execute",
    )

    assert run_exit_code == 0
    assert run_stderr == ""
    run_manifest = json.loads(run_stdout)
    assert run_manifest["filters"]["run_id"] == "20260319T090000Z"
    assert run_manifest["conversation_count"] == 1

    run_records = read_jsonl(run_output_dir / "memory-records.jsonl")
    assert len(run_records) == 1
    run_record = run_records[0]
    assert run_record["id"].startswith("sha256:")
    assert run_record["record_type"] == "conversation"
    assert run_record["redaction"] == {"status": "redacted", "marker_count": 4}
    assert run_record["source"] == "codex_cli"
    assert run_record["source_session_id"] == "session-redaction"
    assert run_record["source_provenance"] == {
        "archived": False,
        "cli_version": "0.32.0",
        "cwd": "/Users/chenjing/dev/chat-collector",
        "originator": "codex_cli_rs",
        "session_started_at": "2026-03-19T09:00:00Z",
        "source": "cli",
    }
    assert run_record["export_provenance"] == {
        "archive_output_path": str(tmp_path / "codex_cli" / "memory_chat_v1-20260319T090000Z.jsonl"),
        "archive_row_number": 1,
        "run_id": "20260319T090000Z",
    }
    assert run_record["transcript_text"] == (
        "developer: Keep Authorization: Bearer [REDACTED] out of the archive.\n\n"
        "user: Keys to mask: [REDACTED_API_KEY] and [REDACTED_API_KEY].\n\n"
        "assistant: Credential JSON: {\"access_token\":\"[REDACTED]\"}"
    )
    assert all(
        message["id"].startswith("sha256:") for message in run_record["messages"]
    )
    assert len({message["id"] for message in run_record["messages"]}) == 3
    assert [message["redaction"] for message in run_record["messages"]] == [
        {"status": "redacted", "marker_count": 1},
        {"status": "redacted", "marker_count": 2},
        {"status": "redacted", "marker_count": 1},
    ]

    watermark_exit_code, watermark_stdout, watermark_stderr = run_cli(
        "archive",
        "export-memory",
        "--archive-root",
        str(tmp_path),
        "--output-dir",
        str(watermark_output_dir),
        "--after-collected-at",
        "2026-03-19T08:30:00Z",
        "--execute",
    )

    assert watermark_exit_code == 0
    assert watermark_stderr == ""
    watermark_manifest = json.loads(watermark_stdout)
    assert watermark_manifest["filters"]["after_collected_at"] == "2026-03-19T08:30:00Z"

    watermark_record = read_jsonl(watermark_output_dir / "memory-records.jsonl")[0]
    assert watermark_record["id"] == run_record["id"]
    assert [message["id"] for message in watermark_record["messages"]] == [
        message["id"] for message in run_record["messages"]
    ]


def test_archive_export_memory_preserves_message_provenance(tmp_path: Path) -> None:
    codex_app_path = write_archive_output(
        tmp_path,
        source="codex_app",
        filename="memory_chat_v1-20260319T091500Z.jsonl",
        rows=(
            make_conversation(
                "codex_app",
                session="desktop-automation-archived",
                collected_at="2026-03-19T09:15:00Z",
                transcript_completeness="partial",
                limitations=[
                    "automation_origin_user_message_reconstructed_from_archived_snapshot",
                    "automation_origin_assistant_message_reconstructed_from_archived_snapshot",
                ],
                messages=[
                    {
                        "role": "user",
                        "source_message_id": "automation-archived-user",
                        "text": "Collect the archived automation threads too.",
                        "provenance": {
                            "body_source": "automation_runs.archived_user_message",
                            "fallback": True,
                        },
                    },
                    {
                        "role": "assistant",
                        "source_message_id": "automation-archived-assistant",
                        "text": "The archived automation run is reconstructed from the inbox snapshot.",
                        "provenance": {
                            "body_source": "automation_runs.archived_assistant_message",
                            "fallback": True,
                        },
                    },
                ],
                provenance={
                    "archived": True,
                    "conversation_origin": "automation",
                    "source": "exec",
                },
            ),
        ),
    )
    write_run_manifest(
        tmp_path,
        run_id="20260319T091500Z",
        sources=(
            {
                "source": "codex_app",
                "support_level": "complete",
                "status": "partial",
                "output_path": str(codex_app_path),
                "scanned_artifact_count": 1,
                "conversation_count": 1,
                "message_count": 2,
                "skipped_conversation_count": 0,
                "written_conversation_count": 1,
                "upgraded_conversation_count": 0,
                "failed": False,
                "partial": True,
                "unsupported": False,
                "redaction_event_count": 0,
            },
        ),
    )

    output_dir = tmp_path.parent / "memory-export-provenance"
    exit_code, stdout, stderr = run_cli(
        "archive",
        "export-memory",
        "--archive-root",
        str(tmp_path),
        "--output-dir",
        str(output_dir),
        "--run",
        "20260319T091500Z",
        "--execute",
    )

    assert exit_code == 0
    assert stderr == ""
    manifest = json.loads(stdout)
    assert manifest["conversation_count"] == 1
    record = read_jsonl(output_dir / "memory-records.jsonl")[0]
    assert [message["provenance"] for message in record["messages"]] == [
        {
            "body_source": "automation_runs.archived_user_message",
            "fallback": True,
        },
        {
            "body_source": "automation_runs.archived_assistant_message",
            "fallback": True,
        },
    ]
