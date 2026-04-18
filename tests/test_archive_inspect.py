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
    rows: tuple[dict[str, object], ...],
    filename: str | None = None,
) -> Path:
    output_path = archive_root / source / (
        filename if filename is not None else f"memory_chat_v1-{source}.jsonl"
    )
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
    provenance: dict[str, object] | None = None,
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
    if provenance is not None:
        payload["provenance"] = provenance
    return payload


def seed_archive_root(archive_root: Path) -> None:
    write_archive_output(
        archive_root,
        source="codex_cli",
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
        ),
    )
    write_archive_output(
        archive_root,
        source="cursor_editor",
        rows=(
            make_conversation(
                "cursor_editor",
                session="cursor-session-1",
                collected_at="2026-03-19T07:00:00Z",
                transcript_completeness="partial",
                limitations=["missing deleted draft messages"],
                provenance={"originator": "cursor"},
                messages=[
                    {"role": "user", "text": "Need deploy checklist follow-up"},
                    {"role": "assistant", "text": "Checklist updated with rollout notes"},
                ],
            ),
            make_conversation(
                "cursor_editor",
                session="cursor-session-2",
                collected_at="2026-03-19T08:00:00Z",
                messages=[
                    {"role": "user", "text": "Discuss onboarding"},
                    {"role": "assistant", "text": "Summarize the docs"},
                ],
            ),
        ),
    )


def test_archive_list_emits_conversation_summaries(tmp_path: Path) -> None:
    seed_archive_root(tmp_path)

    exit_code, stdout, stderr = run_cli(
        "archive",
        "list",
        "--archive-root",
        str(tmp_path),
    )

    assert exit_code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["conversation_count"] == 3

    conversations = payload["conversations"]
    assert [conversation["source_session_id"] for conversation in conversations] == [
        "cursor-session-2",
        "cursor-session-1",
        "codex-session-1",
    ]

    partial_conversation = conversations[1]
    assert partial_conversation["source"] == "cursor_editor"
    assert partial_conversation["transcript_completeness"] == "partial"
    assert partial_conversation["message_count"] == 2
    assert partial_conversation["limitations"] == ["missing deleted draft messages"]
    assert partial_conversation["has_provenance"] is True

    complete_conversation = conversations[2]
    assert complete_conversation["source"] == "codex_cli"
    assert complete_conversation["transcript_completeness"] == "complete"
    assert complete_conversation["limitations"] == []
    assert complete_conversation["has_provenance"] is False


def test_archive_show_emits_normalized_messages_in_order(tmp_path: Path) -> None:
    seed_archive_root(tmp_path)

    exit_code, stdout, stderr = run_cli(
        "archive",
        "show",
        "--archive-root",
        str(tmp_path),
        "--source",
        "cursor_editor",
        "--session",
        "cursor-session-1",
    )

    assert exit_code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["summary"]["source"] == "cursor_editor"
    assert payload["summary"]["source_session_id"] == "cursor-session-1"
    assert payload["conversation"]["transcript_completeness"] == "partial"
    assert payload["conversation"]["message_count"] == 2
    assert payload["conversation"]["has_provenance"] is True
    assert payload["conversation"]["messages"] == [
        {"role": "user", "text": "Need deploy checklist follow-up"},
        {"role": "assistant", "text": "Checklist updated with rollout notes"},
    ]


def test_archive_find_filters_by_source_completeness_and_text(tmp_path: Path) -> None:
    seed_archive_root(tmp_path)

    exit_code, stdout, stderr = run_cli(
        "archive",
        "find",
        "--archive-root",
        str(tmp_path),
        "--source",
        "cursor_editor",
        "--transcript-completeness",
        "partial",
        "--text",
        "checklist",
    )

    assert exit_code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["conversation_count"] == 1
    assert payload["text"] == "checklist"
    assert payload["conversations"] == [
        {
            "collected_at": "2026-03-19T07:00:00Z",
            "execution_context": "cli",
            "has_provenance": True,
            "limitations": ["missing deleted draft messages"],
            "matched_message_count": 2,
            "message_count": 2,
            "output_path": str(
                tmp_path / "cursor_editor" / "memory_chat_v1-cursor_editor.jsonl"
            ),
            "preview": "Need deploy checklist follow-up",
            "row_number": 1,
            "source": "cursor_editor",
            "source_session_id": "cursor-session-1",
            "transcript_completeness": "partial",
        }
    ]
