from __future__ import annotations

import io
import json
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

from llm_chat_archive import archive_prune, cli
from llm_chat_archive.runner import MANIFEST_FILENAME, RUNS_DIRECTORY


def run_cli(*args: str) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        exit_code = cli.main(list(args))
    return exit_code, stdout.getvalue(), stderr.getvalue()


def write_archive_output(archive_root: Path, *, source: str) -> Path:
    output_path = archive_root / source / f"memory_chat_v1-{source}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"source": source, "messages": [{"role": "user", "text": "keep"}]})
        + "\n",
        encoding="utf-8",
    )
    return output_path


def write_auxiliary_file(root: Path, relative_path: str, content: str) -> Path:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def make_source_entry(source: str) -> dict[str, object]:
    return {
        "source": source,
        "support_level": "complete",
        "status": "complete",
        "output_path": None,
        "scanned_artifact_count": 1,
        "conversation_count": 1,
        "skipped_conversation_count": 0,
        "written_conversation_count": 1,
        "message_count": 1,
        "partial": False,
        "unsupported": False,
        "failed": False,
    }


def write_run_manifest(
    archive_root: Path,
    *,
    run_id: str,
    started_at: str,
    completed_at: str,
    source: str = "codex_cli",
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
        "sources": [make_source_entry(source)],
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest_path


def test_archive_prune_dry_run_reports_targets_without_deleting_files(
    tmp_path: Path,
) -> None:
    canonical_output = write_archive_output(tmp_path, source="codex_cli")
    first_manifest = write_run_manifest(
        tmp_path,
        run_id="20260319T010000Z",
        started_at="2026-03-19T01:00:00Z",
        completed_at="2026-03-19T01:00:05Z",
    )
    second_manifest = write_run_manifest(
        tmp_path,
        run_id="20260319T020000Z",
        started_at="2026-03-19T02:00:00Z",
        completed_at="2026-03-19T02:00:05Z",
    )
    write_run_manifest(
        tmp_path,
        run_id="20260319T030000Z",
        started_at="2026-03-19T03:00:00Z",
        completed_at="2026-03-19T03:00:05Z",
    )
    staging_file = write_auxiliary_file(
        tmp_path,
        "rewrite-staging/codex_cli/staged.jsonl",
        "rewrite staging\n",
    )
    export_file = write_auxiliary_file(
        tmp_path,
        "exports/session-export/conversations.jsonl",
        "temporary export\n",
    )
    archive_index_file = write_auxiliary_file(
        tmp_path,
        "archive-index/conversations.sqlite3",
        "sqlite cache\n",
    )

    exit_code, stdout, stderr = run_cli(
        "archive",
        "prune",
        "--archive-root",
        str(tmp_path),
        "--keep-last-runs",
        "1",
        "--prune-auxiliary",
    )

    assert exit_code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["write_mode"] == "dry_run"
    assert payload["recorded_run_count"] == 3
    assert payload["kept_run_count"] == 1
    assert payload["deleted_run_count"] == 2
    assert payload["deleted_auxiliary_directory_count"] == 3
    assert payload["deleted_file_count"] == 5
    assert payload["reclaimed_bytes"] == (
        first_manifest.stat().st_size
        + second_manifest.stat().st_size
        + archive_index_file.stat().st_size
        + staging_file.stat().st_size
        + export_file.stat().st_size
    )
    assert payload["latest_kept_run_id"] == "20260319T030000Z"
    assert [run["run_id"] for run in payload["deleted_runs"]] == [
        "20260319T020000Z",
        "20260319T010000Z",
    ]
    assert {directory["directory"] for directory in payload["deleted_auxiliary_directories"]} == {
        "archive-index",
        "exports",
        "rewrite-staging",
    }

    assert canonical_output.exists()
    assert first_manifest.exists()
    assert second_manifest.exists()
    assert (tmp_path / "archive-index").exists()
    assert (tmp_path / "rewrite-staging").exists()
    assert (tmp_path / "exports").exists()


def test_archive_prune_execute_removes_old_runs_and_auxiliary_dirs_only(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        archive_prune,
        "_utc_now",
        lambda: datetime(2026, 3, 20, tzinfo=timezone.utc),
    )
    canonical_output = write_archive_output(tmp_path, source="cursor_editor")
    first_manifest = write_run_manifest(
        tmp_path,
        run_id="20260301T010000Z",
        started_at="2026-03-01T01:00:00Z",
        completed_at="2026-03-01T01:00:05Z",
        source="cursor_editor",
    )
    second_manifest = write_run_manifest(
        tmp_path,
        run_id="20260310T010000Z",
        started_at="2026-03-10T01:00:00Z",
        completed_at="2026-03-10T01:00:05Z",
        source="cursor_editor",
    )
    newest_manifest = write_run_manifest(
        tmp_path,
        run_id="20260318T010000Z",
        started_at="2026-03-18T01:00:00Z",
        completed_at="2026-03-18T01:00:05Z",
        source="cursor_editor",
    )
    backup_file = write_auxiliary_file(
        tmp_path,
        "rewrite-backups/cursor_editor/before.jsonl",
        "backup copy\n",
    )
    expected_reclaimed_bytes = (
        first_manifest.stat().st_size
        + second_manifest.stat().st_size
        + backup_file.stat().st_size
    )

    exit_code, stdout, stderr = run_cli(
        "archive",
        "prune",
        "--archive-root",
        str(tmp_path),
        "--older-than-days",
        "7",
        "--prune-auxiliary",
        "--execute",
    )

    assert exit_code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["write_mode"] == "prune"
    assert payload["deleted_run_count"] == 2
    assert payload["deleted_auxiliary_directory_count"] == 1
    assert payload["deleted_file_count"] == 3
    assert payload["reclaimed_bytes"] == expected_reclaimed_bytes
    assert payload["latest_kept_run_id"] == "20260318T010000Z"
    assert [run["run_id"] for run in payload["deleted_runs"]] == [
        "20260310T010000Z",
        "20260301T010000Z",
    ]

    assert canonical_output.exists()
    assert newest_manifest.exists()
    assert not first_manifest.exists()
    assert not second_manifest.exists()
    assert not (tmp_path / "rewrite-backups").exists()


def test_archive_prune_requires_at_least_one_prune_criterion(tmp_path: Path) -> None:
    exit_code, stdout, stderr = run_cli(
        "archive",
        "prune",
        "--archive-root",
        str(tmp_path),
    )

    assert exit_code == 2
    assert stdout == ""
    assert stderr.strip() == "archive prune requires at least one prune criterion"
