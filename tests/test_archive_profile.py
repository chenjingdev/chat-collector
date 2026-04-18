from __future__ import annotations

import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

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
    rows: tuple[dict[str, object], ...],
    filename: str,
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
                    {"role": "developer", "text": "Follow the repository guardrails."},
                    {"role": "user", "text": "Need deploy checklist"},
                    {"role": "assistant", "text": "Start with release notes"},
                ],
            ),
            make_conversation(
                "codex_cli",
                session="codex-session-2",
                collected_at="2026-03-19T06:30:00Z",
                transcript_completeness="partial",
                limitations=["missing hidden review thread export"],
                messages=[
                    {"role": "system", "text": "Session resumed from local cache."},
                    {"role": "user", "text": "Add rollback plan"},
                    {"role": "assistant", "text": "Rollback steps drafted"},
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
                    "missing deleted draft messages",
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
                transcript_completeness="partial",
                limitations=["missing deleted draft messages"],
                messages=[
                    {"role": "user", "text": "Need follow-up checklist"},
                    {"role": "assistant", "text": "Checklist updated"},
                ],
            ),
        ),
    )


def test_archive_profile_emits_role_completeness_and_limitations_summary(
    tmp_path: Path,
) -> None:
    seed_archive_root(tmp_path)

    exit_code, stdout, stderr = run_cli(
        "archive",
        "profile",
        "--archive-root",
        str(tmp_path),
    )

    assert exit_code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["archive_root"] == str(tmp_path)
    assert payload["source_filter"] is None
    assert payload["source_count"] == 2
    assert payload["file_count"] == 3
    assert payload["conversation_count"] == 4
    assert payload["message_count"] == 9
    assert payload["conversation_with_limitations_count"] == 3
    assert payload["conversation_with_limitations_ratio"] == pytest.approx(0.75)
    assert payload["message_roles"] == {
        "assistant": {"count": 4, "ratio": pytest.approx(4 / 9)},
        "developer": {"count": 1, "ratio": pytest.approx(1 / 9)},
        "system": {"count": 1, "ratio": pytest.approx(1 / 9)},
        "user": {"count": 3, "ratio": pytest.approx(1 / 3)},
    }
    assert payload["transcript_completeness"] == {
        "complete": {"count": 1, "ratio": pytest.approx(0.25)},
        "partial": {"count": 2, "ratio": pytest.approx(0.5)},
        "unsupported": {"count": 1, "ratio": pytest.approx(0.25)},
    }
    assert payload["limitations"] == {
        "missing deleted draft messages": {
            "count": 2,
            "ratio": pytest.approx(0.5),
        },
        "editor session transcript unavailable": {
            "count": 1,
            "ratio": pytest.approx(0.25),
        },
        "missing hidden review thread export": {
            "count": 1,
            "ratio": pytest.approx(0.25),
        },
    }

    assert payload["sources"] == {
        "codex_cli": {
            "conversation_count": 2,
            "conversation_with_limitations_count": 1,
            "conversation_with_limitations_ratio": pytest.approx(0.5),
            "file_count": 1,
            "limitations": {
                "missing hidden review thread export": {
                    "count": 1,
                    "ratio": pytest.approx(0.5),
                }
            },
            "message_count": 6,
            "message_roles": {
                "assistant": {"count": 2, "ratio": pytest.approx(1 / 3)},
                "developer": {"count": 1, "ratio": pytest.approx(1 / 6)},
                "system": {"count": 1, "ratio": pytest.approx(1 / 6)},
                "user": {"count": 2, "ratio": pytest.approx(1 / 3)},
            },
            "transcript_completeness": {
                "complete": {"count": 1, "ratio": pytest.approx(0.5)},
                "partial": {"count": 1, "ratio": pytest.approx(0.5)},
                "unsupported": {"count": 0, "ratio": pytest.approx(0.0)},
            },
        },
        "cursor_editor": {
            "conversation_count": 2,
            "conversation_with_limitations_count": 2,
            "conversation_with_limitations_ratio": pytest.approx(1.0),
            "file_count": 2,
            "limitations": {
                "missing deleted draft messages": {
                    "count": 2,
                    "ratio": pytest.approx(1.0),
                },
                "editor session transcript unavailable": {
                    "count": 1,
                    "ratio": pytest.approx(0.5),
                },
            },
            "message_count": 3,
            "message_roles": {
                "assistant": {"count": 2, "ratio": pytest.approx(2 / 3)},
                "developer": {"count": 0, "ratio": pytest.approx(0.0)},
                "system": {"count": 0, "ratio": pytest.approx(0.0)},
                "user": {"count": 1, "ratio": pytest.approx(1 / 3)},
            },
            "transcript_completeness": {
                "complete": {"count": 0, "ratio": pytest.approx(0.0)},
                "partial": {"count": 1, "ratio": pytest.approx(0.5)},
                "unsupported": {"count": 1, "ratio": pytest.approx(0.5)},
            },
        },
    }


def test_archive_profile_filters_to_single_source(tmp_path: Path) -> None:
    seed_archive_root(tmp_path)

    exit_code, stdout, stderr = run_cli(
        "archive",
        "profile",
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
    assert payload["file_count"] == 2
    assert payload["conversation_count"] == 2
    assert payload["message_count"] == 3
    assert payload["conversation_with_limitations_count"] == 2
    assert payload["conversation_with_limitations_ratio"] == pytest.approx(1.0)
    assert payload["message_roles"] == {
        "assistant": {"count": 2, "ratio": pytest.approx(2 / 3)},
        "developer": {"count": 0, "ratio": pytest.approx(0.0)},
        "system": {"count": 0, "ratio": pytest.approx(0.0)},
        "user": {"count": 1, "ratio": pytest.approx(1 / 3)},
    }
    assert payload["limitations"] == {
        "missing deleted draft messages": {
            "count": 2,
            "ratio": pytest.approx(1.0),
        },
        "editor session transcript unavailable": {
            "count": 1,
            "ratio": pytest.approx(0.5),
        },
    }
    assert set(payload["sources"]) == {"cursor_editor"}
    assert payload["sources"]["cursor_editor"]["file_count"] == 2
    assert payload["sources"]["cursor_editor"]["conversation_count"] == 2


def test_archive_profile_reports_empty_archive_root(tmp_path: Path) -> None:
    exit_code, stdout, stderr = run_cli(
        "archive",
        "profile",
        "--archive-root",
        str(tmp_path),
    )

    assert exit_code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["source_count"] == 0
    assert payload["file_count"] == 0
    assert payload["conversation_count"] == 0
    assert payload["message_count"] == 0
    assert payload["conversation_with_limitations_count"] == 0
    assert payload["conversation_with_limitations_ratio"] == 0.0
    assert payload["message_roles"] == {
        "assistant": {"count": 0, "ratio": 0.0},
        "developer": {"count": 0, "ratio": 0.0},
        "system": {"count": 0, "ratio": 0.0},
        "user": {"count": 0, "ratio": 0.0},
    }
    assert payload["transcript_completeness"] == {
        "complete": {"count": 0, "ratio": 0.0},
        "partial": {"count": 0, "ratio": 0.0},
        "unsupported": {"count": 0, "ratio": 0.0},
    }
    assert payload["limitations"] == {}
    assert payload["sources"] == {}
