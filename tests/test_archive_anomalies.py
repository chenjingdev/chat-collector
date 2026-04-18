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


def seed_archive_root(archive_root: Path) -> None:
    write_archive_output(
        archive_root,
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
                    {"role": "assistant", "text": "Noted"},
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
                collected_at="2026-03-19T07:00:00Z",
                transcript_completeness="unsupported",
                limitations=[
                    "editor session transcript unavailable",
                    "message bodies missing from local cache",
                ],
                messages=[
                    {"role": "assistant", "text": "Recovered session metadata only"},
                ],
            ),
        ),
    )
    write_archive_output(
        archive_root,
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
                    {"role": "assistant", "text": "Recovered session metadata only"},
                ],
            ),
            make_conversation(
                "cursor_editor",
                session="cursor-session-3",
                collected_at="2026-03-19T09:00:00Z",
                transcript_completeness="partial",
                messages=[
                    {"role": "user", "text": "Need follow-up checklist"},
                    {"role": "assistant", "text": "Checklist updated"},
                ],
            ),
        ),
    )


def test_archive_anomalies_reports_conversation_and_source_reasons(
    tmp_path: Path,
) -> None:
    seed_archive_root(tmp_path)

    exit_code, stdout, stderr = run_cli(
        "archive",
        "anomalies",
        "--archive-root",
        str(tmp_path),
    )

    assert exit_code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["archive_root"] == str(tmp_path)
    assert payload["source_filter"] is None
    assert payload["thresholds"] == {
        "low_message_count": 1,
        "limitations_count": 2,
        "unsupported_count": 2,
        "unsupported_ratio": 0.5,
    }
    assert payload["source_count"] == 2
    assert payload["conversation_count"] == 5
    assert payload["message_count"] == 7
    assert payload["suspicious_source_count"] == 2
    assert payload["source_with_aggregate_reasons_count"] == 1
    assert payload["suspicious_conversation_count"] == 3

    codex_payload = payload["sources"]["codex_cli"]
    assert codex_payload == {
        "conversation_count": 2,
        "excessive_limitations_conversation_count": 0,
        "file_count": 1,
        "low_message_count_conversation_count": 1,
        "message_count": 3,
        "source_reasons": [],
        "suspicious": True,
        "suspicious_conversation_count": 1,
        "suspicious_conversations": [
            {
                "collected_at": "2026-03-19T06:30:00Z",
                "execution_context": "cli",
                "has_provenance": False,
                "limitations": [],
                "message_count": 1,
                "output_path": str(
                    tmp_path / "codex_cli" / "memory_chat_v1-20260319T060000-codex_cli.jsonl"
                ),
                "reasons": [
                    {
                        "code": "low_message_count",
                        "details": {"message_count": 1, "threshold": 1},
                        "message": "message_count 1 is at or below threshold 1",
                    }
                ],
                "row_number": 2,
                "source": "codex_cli",
                "source_session_id": "codex-session-2",
                "transcript_completeness": "complete",
            }
        ],
        "unsupported_conversation_count": 0,
        "unsupported_conversation_ratio": 0.0,
    }

    cursor_payload = payload["sources"]["cursor_editor"]
    assert cursor_payload["file_count"] == 2
    assert cursor_payload["conversation_count"] == 3
    assert cursor_payload["message_count"] == 4
    assert cursor_payload["suspicious"] is True
    assert cursor_payload["suspicious_conversation_count"] == 2
    assert cursor_payload["low_message_count_conversation_count"] == 2
    assert cursor_payload["excessive_limitations_conversation_count"] == 1
    assert cursor_payload["unsupported_conversation_count"] == 2
    assert abs(cursor_payload["unsupported_conversation_ratio"] - (2 / 3)) < 1e-9
    assert {reason["code"] for reason in cursor_payload["source_reasons"]} == {
        "high_unsupported_ratio"
    }

    suspicious_conversations = {
        conversation["source_session_id"]: conversation
        for conversation in cursor_payload["suspicious_conversations"]
    }
    assert set(suspicious_conversations) == {"cursor-session-1", "cursor-session-2"}
    assert {
        reason["code"]
        for reason in suspicious_conversations["cursor-session-1"]["reasons"]
    } == {
        "low_message_count",
        "excessive_limitations",
        "unsupported_transcript",
    }
    assert {
        reason["code"]
        for reason in suspicious_conversations["cursor-session-2"]["reasons"]
    } == {
        "low_message_count",
        "unsupported_transcript",
    }


def test_archive_anomalies_filters_source_and_allows_threshold_overrides(
    tmp_path: Path,
) -> None:
    seed_archive_root(tmp_path)

    exit_code, stdout, stderr = run_cli(
        "archive",
        "anomalies",
        "--archive-root",
        str(tmp_path),
        "--source",
        "cursor_editor",
        "--low-message-count",
        "0",
        "--limitations-count",
        "3",
        "--unsupported-count",
        "3",
        "--unsupported-ratio",
        "0.8",
    )

    assert exit_code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["source_filter"] == "cursor_editor"
    assert payload["source_count"] == 1
    assert payload["conversation_count"] == 3
    assert payload["message_count"] == 4
    assert payload["suspicious_source_count"] == 1
    assert payload["source_with_aggregate_reasons_count"] == 0
    assert payload["suspicious_conversation_count"] == 2
    assert payload["thresholds"] == {
        "low_message_count": 0,
        "limitations_count": 3,
        "unsupported_count": 3,
        "unsupported_ratio": 0.8,
    }

    cursor_payload = payload["sources"]["cursor_editor"]
    assert cursor_payload["source_reasons"] == []
    assert cursor_payload["low_message_count_conversation_count"] == 0
    assert cursor_payload["excessive_limitations_conversation_count"] == 0
    suspicious_conversations = cursor_payload["suspicious_conversations"]
    assert len(suspicious_conversations) == 2
    for conversation in suspicious_conversations:
        assert {reason["code"] for reason in conversation["reasons"]} == {
            "unsupported_transcript"
        }


def test_archive_anomalies_reports_empty_archive_root(tmp_path: Path) -> None:
    exit_code, stdout, stderr = run_cli(
        "archive",
        "anomalies",
        "--archive-root",
        str(tmp_path),
    )

    assert exit_code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["source_count"] == 0
    assert payload["conversation_count"] == 0
    assert payload["message_count"] == 0
    assert payload["suspicious_source_count"] == 0
    assert payload["source_with_aggregate_reasons_count"] == 0
    assert payload["suspicious_conversation_count"] == 0
    assert payload["sources"] == {}
