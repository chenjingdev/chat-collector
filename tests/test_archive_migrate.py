from __future__ import annotations

import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from llm_chat_archive import cli
from llm_chat_archive.models import SCHEMA_VERSION
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


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


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
    schema_version: str,
    messages: list[dict[str, object]],
    provenance: dict[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "source": source,
        "execution_context": "cli",
        "collected_at": collected_at,
        "source_session_id": session,
        "messages": messages,
        "contract": {"schema_version": schema_version},
    }
    if provenance is not None:
        payload["provenance"] = provenance
    return payload


def write_run_manifest(
    archive_root: Path,
    *,
    run_id: str,
    output_path: Path,
) -> Path:
    run_dir = archive_root / RUNS_DIRECTORY / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / MANIFEST_FILENAME
    payload = {
        "run_id": run_id,
        "archive_root": str(archive_root),
        "run_dir": str(run_dir),
        "manifest_path": str(manifest_path),
        "sources": [
            {
                "source": output_path.parent.name,
                "archive_root": str(archive_root),
                "output_path": str(output_path),
                "input_roots": [],
                "support_level": "complete",
                "status": "complete",
                "scanned_artifact_count": 1,
                "conversation_count": 2,
                "skipped_conversation_count": 0,
                "written_conversation_count": 2,
                "upgraded_conversation_count": 0,
                "message_count": 3,
                "redaction_event_count": 0,
                "partial": False,
                "unsupported": False,
                "failed": False,
            }
        ],
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def seed_legacy_archive(archive_root: Path) -> tuple[Path, Path]:
    output_path = write_archive_output(
        archive_root,
        source="codex_cli",
        filename="memory_chat_v1-20260319T060000-codex_cli.jsonl",
        rows=(
            make_conversation(
                "codex_cli",
                session="legacy-session",
                collected_at="2026-03-19T06:00:00Z",
                schema_version="2026-03-18",
                provenance={
                    "source": "codex_cli",
                    "session_path": "/tmp/codex/session.jsonl",
                },
                messages=[
                    {"role": "user", "text": "Need deploy checklist"},
                    {"role": "assistant", "text": "Start with release notes"},
                ],
            ),
            make_conversation(
                "codex_cli",
                session="current-session",
                collected_at="2026-03-19T06:05:00Z",
                schema_version=SCHEMA_VERSION,
                messages=[{"role": "user", "text": "Ship the fix"}],
            ),
        ),
    )
    manifest_path = write_run_manifest(
        archive_root,
        run_id="20260319T080000Z",
        output_path=output_path,
    )
    return output_path, manifest_path


def test_archive_migrate_dry_run_reports_schema_upgrades_without_writing(
    tmp_path: Path,
) -> None:
    output_path, manifest_path = seed_legacy_archive(tmp_path)

    exit_code, stdout, stderr = run_cli(
        "archive",
        "migrate",
        "--archive-root",
        str(tmp_path),
    )

    assert exit_code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["write_mode"] == "dry_run"
    assert payload["target_schema_version"] == SCHEMA_VERSION
    assert payload["manifest_count"] == 1
    assert payload["manifest_rewrite_count"] == 0
    assert payload["file_count"] == 1
    assert payload["row_count"] == 2
    assert payload["migrated_row_count"] == 1
    assert payload["unchanged_row_count"] == 1
    assert payload["required_fields_preserved"] is True
    assert payload["provenance_core_preserved"] is True
    assert payload["sources"]["codex_cli"]["files"] == [
        {
            "source": "codex_cli",
            "input_path": str(output_path),
            "output_path": str(output_path),
            "action": "migrate",
            "row_count": 2,
            "migrated_row_count": 1,
            "unchanged_row_count": 1,
            "backup_path": None,
            "schema_versions_before": {
                "2026-03-18": 1,
                SCHEMA_VERSION: 1,
            },
            "required_fields_preserved": True,
            "provenance_core_preserved": True,
        }
    ]
    assert payload["manifests"] == [
        {
            "input_path": str(manifest_path),
            "output_path": str(manifest_path),
            "action": "noop",
            "rewritten_path_count": 0,
        }
    ]

    rows = read_jsonl(output_path)
    assert rows[0]["contract"]["schema_version"] == "2026-03-18"


def test_archive_migrate_execute_rewrites_rows_in_place_and_keeps_backup(
    tmp_path: Path,
) -> None:
    output_path, _ = seed_legacy_archive(tmp_path)
    backup_dir = tmp_path.parent / "archive-migrate-backups"

    exit_code, stdout, stderr = run_cli(
        "archive",
        "migrate",
        "--archive-root",
        str(tmp_path),
        "--backup-dir",
        str(backup_dir),
        "--execute",
    )

    assert exit_code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["write_mode"] == "in_place"
    assert payload["post_verify"] == {
        "archive_root": str(tmp_path),
        "source_filter": None,
        "status": "success",
        "finding_count": 0,
        "warning_count": 0,
        "error_count": 0,
    }

    rows = read_jsonl(output_path)
    assert [row["contract"]["schema_version"] for row in rows] == [
        SCHEMA_VERSION,
        SCHEMA_VERSION,
    ]
    assert rows[0]["provenance"] == {
        "source": "codex_cli",
        "session_path": "/tmp/codex/session.jsonl",
    }

    backup_root = Path(payload["backup_root"])
    backup_rows = read_jsonl(backup_root / "codex_cli" / output_path.name)
    assert backup_rows[0]["contract"]["schema_version"] == "2026-03-18"

    verify_exit_code, verify_stdout, verify_stderr = run_cli(
        "archive",
        "verify",
        "--archive-root",
        str(tmp_path),
    )
    assert verify_exit_code == 0
    assert verify_stderr == ""
    verify_payload = json.loads(verify_stdout)
    assert verify_payload["error_count"] == 0


def test_archive_migrate_execute_can_stage_archive_and_rewrite_run_manifest(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "archive"
    stage_root = tmp_path / "staging"
    output_path, manifest_path = seed_legacy_archive(source_root)

    exit_code, stdout, stderr = run_cli(
        "archive",
        "migrate",
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
    assert payload["manifest_count"] == 1
    assert payload["manifest_rewrite_count"] == 1
    assert payload["post_verify"]["archive_root"] == str(stage_root)

    staged_rows = read_jsonl(stage_root / "codex_cli" / output_path.name)
    assert staged_rows[0]["contract"]["schema_version"] == SCHEMA_VERSION

    original_rows = read_jsonl(output_path)
    assert original_rows[0]["contract"]["schema_version"] == "2026-03-18"

    staged_manifest = read_json(stage_root / RUNS_DIRECTORY / "20260319T080000Z" / MANIFEST_FILENAME)
    assert staged_manifest["archive_root"] == str(stage_root)
    assert staged_manifest["run_dir"] == str(stage_root / RUNS_DIRECTORY / "20260319T080000Z")
    assert staged_manifest["manifest_path"] == str(
        stage_root / RUNS_DIRECTORY / "20260319T080000Z" / MANIFEST_FILENAME
    )
    assert staged_manifest["sources"][0]["archive_root"] == str(stage_root)
    assert staged_manifest["sources"][0]["output_path"] == str(
        stage_root / "codex_cli" / output_path.name
    )

    original_manifest = read_json(manifest_path)
    assert original_manifest["archive_root"] == str(source_root)


def test_archive_migrate_rejects_source_filtered_staging(tmp_path: Path) -> None:
    seed_legacy_archive(tmp_path)
    stage_root = tmp_path.parent / "archive-migrate-stage"

    exit_code, stdout, stderr = run_cli(
        "archive",
        "migrate",
        "--archive-root",
        str(tmp_path),
        "--source",
        "codex_cli",
        "--output-root",
        str(stage_root),
    )

    assert exit_code == 1
    assert stdout == ""
    assert (
        stderr.strip()
        == "archive migrate does not support combining --source with --output-root"
    )
