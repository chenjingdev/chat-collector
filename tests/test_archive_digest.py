from __future__ import annotations

import io
import json
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

from llm_chat_archive import archive_digest, cli
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


def make_conversation(
    source: str,
    *,
    session: str,
    collected_at: str,
    messages: list[dict[str, object]],
    transcript_completeness: str | None = None,
    limitations: list[str] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "source": source,
        "execution_context": "cli",
        "collected_at": collected_at,
        "source_session_id": session,
        "messages": messages,
        "contract": {"schema_version": "2026-03-19"},
    }
    if transcript_completeness is not None:
        payload["transcript_completeness"] = transcript_completeness
    if limitations is not None:
        payload["limitations"] = limitations
    return payload


def write_run_manifest(
    archive_root: Path,
    *,
    run_id: str,
    started_at: str,
    completed_at: str,
    sources: tuple[dict[str, object], ...],
) -> Path:
    run_dir = archive_root / RUNS_DIRECTORY / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / MANIFEST_FILENAME
    payload = {
        "run_id": run_id,
        "archive_root": str(archive_root),
        "run_dir": str(run_dir),
        "manifest_path": str(manifest_path),
        "started_at": started_at,
        "completed_at": completed_at,
        "source_count": len(sources),
        "failed_source_count": sum(1 for source in sources if source["failed"]),
        "scanned_artifact_count": sum(
            int(source["scanned_artifact_count"]) for source in sources
        ),
        "conversation_count": sum(int(source["conversation_count"]) for source in sources),
        "skipped_conversation_count": sum(
            int(source["skipped_conversation_count"]) for source in sources
        ),
        "written_conversation_count": sum(
            int(source["written_conversation_count"]) for source in sources
        ),
        "message_count": sum(int(source["message_count"]) for source in sources),
        "sources": list(sources),
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def make_source_entry(
    *,
    archive_root: Path,
    source: str,
    support_level: str,
    status: str,
    scanned_artifact_count: int,
    conversation_count: int,
    message_count: int,
    output_path: Path | None,
    partial: bool = False,
    unsupported: bool = False,
    failed: bool = False,
) -> dict[str, object]:
    return {
        "source": source,
        "support_level": support_level,
        "status": status,
        "archive_root": str(archive_root),
        "output_path": None if output_path is None else str(output_path),
        "input_roots": [],
        "scanned_artifact_count": scanned_artifact_count,
        "conversation_count": conversation_count,
        "skipped_conversation_count": 0,
        "written_conversation_count": conversation_count if output_path is not None else 0,
        "message_count": message_count,
        "partial": partial,
        "unsupported": unsupported,
        "failed": failed,
    }


def test_archive_digest_emits_operator_summary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        archive_digest,
        "_utcnow",
        lambda: datetime(2026, 3, 19, 10, 15, tzinfo=timezone.utc),
    )

    codex_output = write_archive_output(
        tmp_path,
        source="codex_cli",
        filename="memory_chat_v1-20260319T060000-codex_cli.jsonl",
        rows=(
            make_conversation(
                "codex_cli",
                session="codex-session-1",
                collected_at="2026-03-19T06:00:00Z",
                messages=[
                    {"role": "user", "text": "Need deploy checklist"},
                    {"role": "assistant", "text": "Start with release notes"},
                ],
            ),
            make_conversation(
                "codex_cli",
                session="codex-session-2",
                collected_at="2026-03-19T06:30:00Z",
                messages=[
                    {"role": "user", "text": "Need rollback notes"},
                    {"role": "assistant", "text": "Rollback notes added"},
                ],
            ),
        ),
    )
    cursor_linked_output = write_archive_output(
        tmp_path,
        source="cursor_editor",
        filename="memory_chat_v1-20260319T070000-cursor_editor.jsonl",
        rows=(
            make_conversation(
                "cursor_editor",
                session="cursor-session-1",
                collected_at="2026-03-19T07:00:00Z",
                transcript_completeness="partial",
                limitations=[
                    "deleted draft messages unavailable",
                    "local cache omitted edits",
                ],
                messages=[
                    {"role": "user", "text": "Need deploy checklist follow-up"},
                    {"role": "assistant", "text": "Checklist updated"},
                ],
            ),
        ),
    )
    write_archive_output(
        tmp_path,
        source="cursor_editor",
        filename="memory_chat_v1-20260319T080000-cursor_editor.jsonl",
        rows=(
            make_conversation(
                "cursor_editor",
                session="cursor-session-2",
                collected_at="2026-03-19T08:00:00Z",
                transcript_completeness="unsupported",
                limitations=["deleted draft messages unavailable"],
                messages=[
                    {"role": "assistant", "text": "Recovered metadata only"},
                ],
            ),
        ),
    )
    write_run_manifest(
        tmp_path,
        run_id="20260319T090000Z",
        started_at="2026-03-19T09:00:00Z",
        completed_at="2026-03-19T09:00:12Z",
        sources=(
            make_source_entry(
                archive_root=tmp_path,
                source="codex_cli",
                support_level="complete",
                status="complete",
                scanned_artifact_count=2,
                conversation_count=2,
                message_count=4,
                output_path=codex_output,
            ),
            make_source_entry(
                archive_root=tmp_path,
                source="cursor_editor",
                support_level="partial",
                status="partial",
                scanned_artifact_count=1,
                conversation_count=1,
                message_count=2,
                output_path=cursor_linked_output,
                partial=True,
            ),
            make_source_entry(
                archive_root=tmp_path,
                source="gemini_cli",
                support_level="scaffold",
                status="unsupported",
                scanned_artifact_count=0,
                conversation_count=0,
                message_count=0,
                output_path=None,
                unsupported=True,
            ),
        ),
    )

    exit_code, stdout, stderr = run_cli(
        "archive",
        "digest",
        "--archive-root",
        str(tmp_path),
    )

    assert exit_code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["archive_root"] == str(tmp_path)
    assert payload["aggregated_at"] == "2026-03-19T10:15:00Z"
    assert payload["status"] == "warning"
    assert payload["latest_run_id"] == "20260319T090000Z"
    assert payload["latest_run"] == {
        "completed_at": "2026-03-19T09:00:12Z",
        "degraded_source_count": 2,
        "degraded_sources": ["cursor_editor", "gemini_cli"],
        "failed_source_count": 0,
        "failed_sources": [],
        "partial_source_count": 1,
        "run_id": "20260319T090000Z",
        "source_count": 3,
        "started_at": "2026-03-19T09:00:00Z",
        "unsupported_source_count": 1,
    }

    assert payload["overview"] == {
        "conversation_count": 4,
        "conversation_with_limitations_count": 2,
        "error_count": 0,
        "has_orphans": True,
        "message_count": 7,
        "orphan_file_count": 1,
        "source_count": 3,
        "sources_with_orphans_count": 1,
        "suspicious_conversation_count": 2,
        "suspicious_source_count": 1,
        "transcript_completeness": {
            "complete": {"count": 2, "ratio": 0.5},
            "partial": {"count": 1, "ratio": 0.25},
            "unsupported": {"count": 1, "ratio": 0.25},
        },
        "warning_count": 3,
    }
    assert payload["top_limitations"] == [
        {
            "count": 2,
            "limitation": "deleted draft messages unavailable",
        },
        {
            "count": 1,
            "limitation": "local cache omitted edits",
        },
    ]

    sources = payload["sources"]
    assert sources["codex_cli"] == {
        "attention_required": False,
        "conversation_count": 2,
        "conversation_with_limitations_count": 0,
        "error_count": 0,
        "failed": False,
        "file_count": 1,
        "has_orphans": False,
        "latest_collected_at": "2026-03-19T06:30:00Z",
        "latest_run_selected": True,
        "latest_run_status": "complete",
        "message_count": 4,
        "orphan_file_count": 0,
        "run_degraded": False,
        "source_reasons": [],
        "suspicious": False,
        "suspicious_conversation_count": 0,
        "support_level": "complete",
        "top_limitations": [],
        "transcript_completeness": {
            "complete": {"count": 2, "ratio": 1.0},
            "partial": {"count": 0, "ratio": 0.0},
            "unsupported": {"count": 0, "ratio": 0.0},
        },
        "verify_status": "success",
        "warning_count": 0,
    }
    assert sources["cursor_editor"] == {
        "attention_required": True,
        "conversation_count": 2,
        "conversation_with_limitations_count": 2,
        "error_count": 0,
        "failed": False,
        "file_count": 2,
        "has_orphans": True,
        "latest_collected_at": "2026-03-19T08:00:00Z",
        "latest_run_selected": True,
        "latest_run_status": "partial",
        "message_count": 3,
        "orphan_file_count": 1,
        "run_degraded": True,
        "source_reasons": [],
        "suspicious": True,
        "suspicious_conversation_count": 2,
        "support_level": "partial",
        "top_limitations": [
            {
                "count": 2,
                "limitation": "deleted draft messages unavailable",
            },
            {
                "count": 1,
                "limitation": "local cache omitted edits",
            },
        ],
        "transcript_completeness": {
            "complete": {"count": 0, "ratio": 0.0},
            "partial": {"count": 1, "ratio": 0.5},
            "unsupported": {"count": 1, "ratio": 0.5},
        },
        "verify_status": "warning",
        "warning_count": 3,
    }
    assert sources["gemini_cli"] == {
        "attention_required": True,
        "conversation_count": 0,
        "conversation_with_limitations_count": 0,
        "error_count": 0,
        "failed": False,
        "file_count": 0,
        "has_orphans": False,
        "latest_collected_at": None,
        "latest_run_selected": True,
        "latest_run_status": "unsupported",
        "message_count": 0,
        "orphan_file_count": 0,
        "run_degraded": True,
        "source_reasons": [],
        "suspicious": False,
        "suspicious_conversation_count": 0,
        "support_level": "scaffold",
        "top_limitations": [],
        "transcript_completeness": {
            "complete": {"count": 0, "ratio": 0.0},
            "partial": {"count": 0, "ratio": 0.0},
            "unsupported": {"count": 0, "ratio": 0.0},
        },
        "verify_status": None,
        "warning_count": 0,
    }


def test_archive_digest_handles_archive_without_recorded_runs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        archive_digest,
        "_utcnow",
        lambda: datetime(2026, 3, 19, 11, 0, tzinfo=timezone.utc),
    )
    write_archive_output(
        tmp_path,
        source="codex_cli",
        filename="memory_chat_v1-codex_cli.jsonl",
        rows=(
            make_conversation(
                "codex_cli",
                session="codex-session-1",
                collected_at="2026-03-19T10:30:00Z",
                messages=[
                    {"role": "user", "text": "Need summary"},
                    {"role": "assistant", "text": "Summary ready"},
                ],
            ),
        ),
    )

    exit_code, stdout, stderr = run_cli(
        "archive",
        "digest",
        "--archive-root",
        str(tmp_path),
    )

    assert exit_code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["aggregated_at"] == "2026-03-19T11:00:00Z"
    assert payload["latest_run_id"] is None
    assert payload["latest_run"] is None
    assert "latest_run_error" not in payload
    assert payload["status"] == "success"
    assert payload["overview"] == {
        "conversation_count": 1,
        "conversation_with_limitations_count": 0,
        "error_count": 0,
        "has_orphans": False,
        "message_count": 2,
        "orphan_file_count": 0,
        "source_count": 1,
        "sources_with_orphans_count": 0,
        "suspicious_conversation_count": 0,
        "suspicious_source_count": 0,
        "transcript_completeness": {
            "complete": {"count": 1, "ratio": 1.0},
            "partial": {"count": 0, "ratio": 0.0},
            "unsupported": {"count": 0, "ratio": 0.0},
        },
        "warning_count": 0,
    }
    assert payload["top_limitations"] == []
    assert payload["sources"] == {
        "codex_cli": {
            "attention_required": False,
            "conversation_count": 1,
            "conversation_with_limitations_count": 0,
            "error_count": 0,
            "failed": False,
            "file_count": 1,
            "has_orphans": False,
            "latest_collected_at": "2026-03-19T10:30:00Z",
            "latest_run_selected": False,
            "latest_run_status": None,
            "message_count": 2,
            "orphan_file_count": 0,
            "run_degraded": False,
            "source_reasons": [],
            "suspicious": False,
            "suspicious_conversation_count": 0,
            "support_level": None,
            "top_limitations": [],
            "transcript_completeness": {
                "complete": {"count": 1, "ratio": 1.0},
                "partial": {"count": 0, "ratio": 0.0},
                "unsupported": {"count": 0, "ratio": 0.0},
            },
            "verify_status": "success",
            "warning_count": 0,
        }
    }
