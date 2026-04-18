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
        source="cursor_editor",
        rows=(
            make_conversation(
                "cursor_editor",
                session="cursor-partial-1",
                collected_at="2026-03-19T07:00:00Z",
                transcript_completeness="partial",
                messages=[
                    {"role": "user", "text": "Checklist alpha follow-up"},
                    {"role": "assistant", "text": "Alpha notes captured"},
                ],
            ),
            make_conversation(
                "cursor_editor",
                session="cursor-partial-2",
                collected_at="2026-03-19T07:30:00Z",
                transcript_completeness="partial",
                messages=[
                    {"role": "user", "text": "Checklist beta follow-up"},
                    {"role": "assistant", "text": "Beta notes captured"},
                ],
            ),
            make_conversation(
                "cursor_editor",
                session="cursor-partial-3",
                collected_at="2026-03-19T08:00:00Z",
                transcript_completeness="partial",
                messages=[
                    {"role": "user", "text": "Checklist gamma follow-up"},
                    {"role": "assistant", "text": "Gamma notes captured"},
                ],
            ),
            make_conversation(
                "cursor_editor",
                session="cursor-complete-1",
                collected_at="2026-03-19T08:30:00Z",
                messages=[
                    {"role": "user", "text": "Checklist complete conversation"},
                    {"role": "assistant", "text": "Complete notes captured"},
                ],
            ),
        ),
    )
    write_archive_output(
        archive_root,
        source="codex_cli",
        rows=(
            make_conversation(
                "codex_cli",
                session="codex-partial-1",
                collected_at="2026-03-19T09:00:00Z",
                transcript_completeness="partial",
                messages=[
                    {"role": "user", "text": "Checklist outside cursor"},
                    {"role": "assistant", "text": "Codex notes captured"},
                ],
            ),
        ),
    )


def _sample_session_ids(payload: dict[str, object]) -> list[str]:
    return [
        conversation["source_session_id"]
        for conversation in payload["conversations"]
    ]


def test_archive_sample_filters_subset_and_reuses_same_seed(tmp_path: Path) -> None:
    seed_archive_root(tmp_path)

    args = (
        "archive",
        "sample",
        "--archive-root",
        str(tmp_path),
        "--source",
        "cursor_editor",
        "--transcript-completeness",
        "partial",
        "--text",
        "checklist",
        "--count",
        "2",
        "--seed",
        "seed-123",
    )

    first_exit_code, first_stdout, first_stderr = run_cli(*args)
    second_exit_code, second_stdout, second_stderr = run_cli(*args)

    assert first_exit_code == 0
    assert second_exit_code == 0
    assert first_stderr == ""
    assert second_stderr == ""

    first_payload = json.loads(first_stdout)
    second_payload = json.loads(second_stdout)

    assert first_payload == second_payload
    assert first_payload["archive_root"] == str(tmp_path)
    assert first_payload["filters"] == {
        "source": "cursor_editor",
        "text": "checklist",
        "transcript_completeness": "partial",
    }
    assert first_payload["seed"] == "seed-123"
    assert first_payload["requested_count"] == 2
    assert first_payload["candidate_count"] == 3
    assert first_payload["conversation_count"] == 2
    assert first_payload["message_count"] == 4
    assert first_payload["source_count"] == 1
    assert set(_sample_session_ids(first_payload)) <= {
        "cursor-partial-1",
        "cursor-partial-2",
        "cursor-partial-3",
    }
    assert all(
        "checklist" in conversation["preview"].casefold()
        for conversation in first_payload["conversations"]
    )


def test_archive_sample_returns_generated_seed_that_can_be_reused(tmp_path: Path) -> None:
    seed_archive_root(tmp_path)

    exit_code, stdout, stderr = run_cli(
        "archive",
        "sample",
        "--archive-root",
        str(tmp_path),
        "--source",
        "cursor_editor",
        "--count",
        "2",
    )

    assert exit_code == 0
    assert stderr == ""

    first_payload = json.loads(stdout)
    assert isinstance(first_payload["seed"], str)
    assert first_payload["seed"]

    replay_exit_code, replay_stdout, replay_stderr = run_cli(
        "archive",
        "sample",
        "--archive-root",
        str(tmp_path),
        "--source",
        "cursor_editor",
        "--count",
        "2",
        "--seed",
        first_payload["seed"],
    )

    assert replay_exit_code == 0
    assert replay_stderr == ""
    replay_payload = json.loads(replay_stdout)
    assert replay_payload["seed"] == first_payload["seed"]
    assert replay_payload["conversations"] == first_payload["conversations"]


def test_archive_sample_rejects_non_positive_count(tmp_path: Path) -> None:
    seed_archive_root(tmp_path)

    exit_code, stdout, stderr = run_cli(
        "archive",
        "sample",
        "--archive-root",
        str(tmp_path),
        "--count",
        "0",
    )

    assert exit_code == 2
    assert stdout == ""
    assert "sample count must be greater than zero" in stderr
