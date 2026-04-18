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
    session_metadata: dict[str, object] | None = None,
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
    if session_metadata is not None:
        payload["session_metadata"] = session_metadata
    if provenance is not None:
        payload["provenance"] = provenance
    return payload


def seed_archive_root(archive_root: Path) -> tuple[Path, Path]:
    first_output = write_archive_output(
        archive_root,
        source="codex_cli",
        filename="memory_chat_v1-20260319T060000-codex_cli.jsonl",
        rows=(
            make_conversation(
                "codex_cli",
                session="alpha",
                collected_at="2026-03-19T06:00:00Z",
                transcript_completeness="partial",
                limitations=["missing assistant reply"],
                messages=[
                    {"role": "user", "text": "Need deploy checklist"},
                ],
            ),
            make_conversation(
                "codex_cli",
                session="beta",
                collected_at="2026-03-19T06:10:00Z",
                messages=[
                    {"role": "user", "text": "Create release notes"},
                    {"role": "assistant", "text": "Start with the changelog"},
                ],
            ),
        ),
    )
    second_output = write_archive_output(
        archive_root,
        source="codex_cli",
        filename="memory_chat_v1-20260319T062000-codex_cli.jsonl",
        rows=(
            make_conversation(
                "codex_cli",
                session="alpha",
                collected_at="2026-03-19T06:20:00Z",
                messages=[
                    {"role": "user", "text": "Need deploy checklist"},
                    {"role": "assistant", "text": "Start with release notes"},
                ],
            ),
            make_conversation(
                "codex_cli",
                session="beta",
                collected_at="2026-03-19T06:10:00Z",
                messages=[
                    {"role": "user", "text": "Create release notes"},
                    {"role": "assistant", "text": "Start with the changelog"},
                ],
            ),
            make_conversation(
                "codex_cli",
                session="gamma",
                collected_at="2026-03-19T06:30:00Z",
                transcript_completeness="complete",
                limitations=[],
                session_metadata={"cwd": "/tmp/demo"},
                provenance={},
                messages=[
                    {"role": "assistant", "text": "Track the follow-up items"},
                ],
            ),
        ),
    )
    return first_output, second_output


def test_archive_rewrite_dry_run_reports_compaction_without_writing(tmp_path: Path) -> None:
    first_output, second_output = seed_archive_root(tmp_path)

    exit_code, stdout, stderr = run_cli(
        "archive",
        "rewrite",
        "--archive-root",
        str(tmp_path),
    )

    assert exit_code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["write_mode"] == "dry_run"
    assert payload["source_count"] == 1
    assert payload["changed_source_count"] == 1
    assert payload["input_file_count"] == 2
    assert payload["output_file_count"] == 1
    assert payload["before_conversation_count"] == 5
    assert payload["after_conversation_count"] == 3
    assert payload["dropped_row_count"] == 2
    assert payload["upgraded_row_count"] == 2
    assert payload["untouched_row_count"] == 1
    assert payload["sources"]["codex_cli"] == {
        "after_conversation_count": 3,
        "before_conversation_count": 5,
        "changed": True,
        "dropped_row_count": 2,
        "input_file_count": 2,
        "output_file_count": 1,
        "output_path": str(tmp_path / "codex_cli" / "memory_chat_v1-codex_cli.jsonl"),
        "untouched_row_count": 1,
        "upgraded_row_count": 2,
    }

    assert first_output.exists()
    assert second_output.exists()
    assert not (tmp_path / "codex_cli" / "memory_chat_v1-codex_cli.jsonl").exists()


def test_archive_rewrite_execute_rewrites_source_to_canonical_file(tmp_path: Path) -> None:
    seed_archive_root(tmp_path)

    exit_code, stdout, stderr = run_cli(
        "archive",
        "rewrite",
        "--archive-root",
        str(tmp_path),
        "--execute",
    )

    assert exit_code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["write_mode"] == "in_place"

    source_dir = tmp_path / "codex_cli"
    output_paths = sorted(source_dir.glob("memory_chat_v1-*.jsonl"))
    assert output_paths == [source_dir / "memory_chat_v1-codex_cli.jsonl"]

    rows = read_jsonl(output_paths[0])
    assert [row["source_session_id"] for row in rows] == ["beta", "alpha", "gamma"]
    assert rows[0] == {
        "collected_at": "2026-03-19T06:10:00Z",
        "contract": {"schema_version": "2026-03-19"},
        "execution_context": "cli",
        "messages": [
            {"role": "user", "text": "Create release notes"},
            {"role": "assistant", "text": "Start with the changelog"},
        ],
        "source": "codex_cli",
        "source_session_id": "beta",
    }
    assert rows[1] == {
        "collected_at": "2026-03-19T06:20:00Z",
        "contract": {"schema_version": "2026-03-19"},
        "execution_context": "cli",
        "messages": [
            {"role": "user", "text": "Need deploy checklist"},
            {"role": "assistant", "text": "Start with release notes"},
        ],
        "source": "codex_cli",
        "source_session_id": "alpha",
    }
    assert rows[2] == {
        "collected_at": "2026-03-19T06:30:00Z",
        "contract": {"schema_version": "2026-03-19"},
        "execution_context": "cli",
        "messages": [
            {"role": "assistant", "text": "Track the follow-up items"},
        ],
        "session_metadata": {"cwd": "/tmp/demo"},
        "source": "codex_cli",
        "source_session_id": "gamma",
    }

    expected_text = "\n".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for row in rows
    ) + "\n"
    assert output_paths[0].read_text(encoding="utf-8") == expected_text


def test_archive_rewrite_execute_can_stage_output_in_separate_root(tmp_path: Path) -> None:
    source_root = tmp_path / "archive"
    stage_root = tmp_path / "staging"
    first_output, second_output = seed_archive_root(source_root)

    exit_code, stdout, stderr = run_cli(
        "archive",
        "rewrite",
        "--archive-root",
        str(source_root),
        "--output-root",
        str(stage_root),
        "--execute",
    )

    assert exit_code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["write_mode"] == "staging"
    assert payload["archive_root"] == str(source_root)
    assert payload["output_root"] == str(stage_root)

    assert first_output.exists()
    assert second_output.exists()
    staged_output = stage_root / "codex_cli" / "memory_chat_v1-codex_cli.jsonl"
    assert staged_output.exists()
    assert [row["source_session_id"] for row in read_jsonl(staged_output)] == [
        "beta",
        "alpha",
        "gamma",
    ]
