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


def write_export_bundle(
    bundle_dir: Path,
    *,
    rows: tuple[dict[str, object], ...],
    filters: dict[str, object],
) -> None:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    conversations_path = bundle_dir / "conversations.jsonl"
    manifest_path = bundle_dir / "export-manifest.json"
    conversations_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(
            {
                "archive_root": str(bundle_dir.parent / "source-archive"),
                "output_dir": str(bundle_dir),
                "write_mode": "write",
                "filters": filters,
                "conversation_count": len(rows),
                "message_count": sum(len(row["messages"]) for row in rows),
                "source_count": len({str(row["source"]) for row in rows}),
                "conversations_path": str(conversations_path),
                "manifest_path": str(manifest_path),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


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
        ),
    )
    second_output = write_archive_output(
        archive_root,
        source="codex_cli",
        filename="memory_chat_v1-20260319T061000-codex_cli.jsonl",
        rows=(
            make_conversation(
                "codex_cli",
                session="gamma",
                collected_at="2026-03-19T06:10:00Z",
                messages=[
                    {"role": "user", "text": "Keep the deployment notes concise"},
                    {"role": "assistant", "text": "I will summarize the rollout steps"},
                ],
            ),
        ),
    )
    return first_output, second_output


def make_bundle_rows() -> tuple[dict[str, object], ...]:
    return (
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
            collected_at="2026-03-19T06:30:00Z",
            messages=[
                {"role": "user", "text": "Create release notes"},
                {"role": "assistant", "text": "Start with the changelog"},
            ],
        ),
        make_conversation(
            "codex_cli",
            session="gamma",
            collected_at="2026-03-19T06:10:00Z",
            messages=[
                {"role": "user", "text": "Keep the deployment notes concise"},
                {"role": "assistant", "text": "I will summarize the rollout steps"},
            ],
        ),
    )


def test_archive_import_dry_run_reports_import_upgrade_and_skip_counts(
    tmp_path: Path,
) -> None:
    archive_root = tmp_path / "archive"
    first_output, second_output = seed_archive_root(archive_root)
    bundle_dir = tmp_path / "bundle"
    write_export_bundle(
        bundle_dir,
        rows=make_bundle_rows(),
        filters={
            "source": "codex_cli",
            "session": None,
            "text": None,
            "transcript_completeness": None,
        },
    )

    exit_code, stdout, stderr = run_cli(
        "archive",
        "import",
        "--archive-root",
        str(archive_root),
        "--bundle-dir",
        str(bundle_dir),
    )

    assert exit_code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["write_mode"] == "dry_run"
    assert payload["source_count"] == 1
    assert payload["conversation_count"] == 3
    assert payload["message_count"] == 6
    assert payload["before_conversation_count"] == 2
    assert payload["after_conversation_count"] == 3
    assert payload["imported_count"] == 1
    assert payload["skipped_count"] == 1
    assert payload["upgraded_count"] == 1
    assert payload["sources"]["codex_cli"] == {
        "after_conversation_count": 3,
        "before_conversation_count": 2,
        "changed": True,
        "imported_count": 1,
        "input_file_count": 2,
        "output_path": str(archive_root / "codex_cli" / "memory_chat_v1-codex_cli.jsonl"),
        "skipped_count": 1,
        "upgraded_count": 1,
    }
    assert first_output.exists()
    assert second_output.exists()
    assert not (archive_root / "codex_cli" / "memory_chat_v1-codex_cli.jsonl").exists()


def test_archive_import_execute_merges_bundle_into_canonical_output(
    tmp_path: Path,
) -> None:
    archive_root = tmp_path / "archive"
    seed_archive_root(archive_root)
    bundle_dir = tmp_path / "bundle"
    write_export_bundle(
        bundle_dir,
        rows=make_bundle_rows(),
        filters={
            "source": "codex_cli",
            "session": None,
            "text": None,
            "transcript_completeness": None,
        },
    )

    exit_code, stdout, stderr = run_cli(
        "archive",
        "import",
        "--archive-root",
        str(archive_root),
        "--bundle-dir",
        str(bundle_dir),
        "--execute",
    )

    assert exit_code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["write_mode"] == "write"
    assert payload["imported_count"] == 1
    assert payload["skipped_count"] == 1
    assert payload["upgraded_count"] == 1

    source_dir = archive_root / "codex_cli"
    output_paths = sorted(source_dir.glob("memory_chat_v1-*.jsonl"))
    assert output_paths == [source_dir / "memory_chat_v1-codex_cli.jsonl"]
    rows = read_jsonl(output_paths[0])
    assert [row["source_session_id"] for row in rows] == ["gamma", "alpha", "beta"]
    assert rows == [
        {
            "collected_at": "2026-03-19T06:10:00Z",
            "contract": {"schema_version": "2026-03-19"},
            "execution_context": "cli",
            "messages": [
                {"role": "user", "text": "Keep the deployment notes concise"},
                {"role": "assistant", "text": "I will summarize the rollout steps"},
            ],
            "source": "codex_cli",
            "source_session_id": "gamma",
        },
        {
            "collected_at": "2026-03-19T06:20:00Z",
            "contract": {"schema_version": "2026-03-19"},
            "execution_context": "cli",
            "messages": [
                {"role": "user", "text": "Need deploy checklist"},
                {"role": "assistant", "text": "Start with release notes"},
            ],
            "source": "codex_cli",
            "source_session_id": "alpha",
        },
        {
            "collected_at": "2026-03-19T06:30:00Z",
            "contract": {"schema_version": "2026-03-19"},
            "execution_context": "cli",
            "messages": [
                {"role": "user", "text": "Create release notes"},
                {"role": "assistant", "text": "Start with the changelog"},
            ],
            "source": "codex_cli",
            "source_session_id": "beta",
        },
    ]


def test_archive_import_rejects_bundle_when_manifest_filters_do_not_match_rows(
    tmp_path: Path,
) -> None:
    archive_root = tmp_path / "archive"
    bundle_dir = tmp_path / "bundle"
    write_export_bundle(
        bundle_dir,
        rows=make_bundle_rows()[:1],
        filters={
            "source": "cursor_editor",
            "session": None,
            "text": None,
            "transcript_completeness": None,
        },
    )

    exit_code, stdout, stderr = run_cli(
        "archive",
        "import",
        "--archive-root",
        str(archive_root),
        "--bundle-dir",
        str(bundle_dir),
    )

    assert exit_code == 1
    assert stdout == ""
    assert "does not satisfy export filter source='cursor_editor'" in stderr
