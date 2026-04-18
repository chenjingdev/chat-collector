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


def test_archive_export_dry_run_reports_filtered_counts_without_writing(
    tmp_path: Path,
) -> None:
    seed_archive_root(tmp_path)
    output_dir = tmp_path.parent / "export-bundle"

    exit_code, stdout, stderr = run_cli(
        "archive",
        "export",
        "--archive-root",
        str(tmp_path),
        "--output-dir",
        str(output_dir),
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
    assert payload == {
        "archive_root": str(tmp_path),
        "conversation_count": 1,
        "conversations_path": str(output_dir / "conversations.jsonl"),
        "filters": {
            "session": None,
            "source": "cursor_editor",
            "text": "checklist",
            "transcript_completeness": "partial",
        },
        "manifest_path": str(output_dir / "export-manifest.json"),
        "message_count": 2,
        "output_dir": str(output_dir),
        "source_count": 1,
        "write_mode": "dry_run",
    }
    assert not (output_dir / "conversations.jsonl").exists()
    assert not (output_dir / "export-manifest.json").exists()


def test_archive_export_execute_writes_portable_bundle_and_manifest(
    tmp_path: Path,
) -> None:
    seed_archive_root(tmp_path)
    output_dir = tmp_path.parent / "session-export"

    exit_code, stdout, stderr = run_cli(
        "archive",
        "export",
        "--archive-root",
        str(tmp_path),
        "--output-dir",
        str(output_dir),
        "--source",
        "cursor_editor",
        "--session",
        "cursor-session-2",
        "--execute",
    )

    assert exit_code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["write_mode"] == "write"
    assert payload["conversation_count"] == 1
    assert payload["message_count"] == 2
    assert payload["source_count"] == 1

    rows = read_jsonl(output_dir / "conversations.jsonl")
    assert rows == [
        {
            "collected_at": "2026-03-19T08:00:00Z",
            "contract": {"schema_version": "2026-03-19"},
            "execution_context": "cli",
            "messages": [
                {"role": "user", "text": "Discuss onboarding"},
                {"role": "assistant", "text": "Summarize the docs"},
            ],
            "source": "cursor_editor",
            "source_session_id": "cursor-session-2",
        }
    ]

    manifest = json.loads((output_dir / "export-manifest.json").read_text(encoding="utf-8"))
    assert manifest == payload
    assert manifest["archive_root"] == str(tmp_path)
    assert manifest["filters"] == {
        "source": "cursor_editor",
        "session": "cursor-session-2",
        "transcript_completeness": None,
        "text": None,
    }
