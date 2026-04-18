from __future__ import annotations

import io
import json
from contextlib import redirect_stderr, redirect_stdout
from datetime import UTC, datetime
from pathlib import Path

import llm_chat_archive.archive_quarantine_export as archive_quarantine_export
from llm_chat_archive import cli
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
    rows: tuple[dict[str, object] | str, ...],
) -> Path:
    output_path = archive_root / source / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(
            row if isinstance(row, str) else json.dumps(row, ensure_ascii=False)
            for row in rows
        )
        + ("\n" if rows else ""),
        encoding="utf-8",
    )
    return output_path


def write_run_manifest(
    archive_root: Path,
    *,
    run_id: str,
    sources: tuple[dict[str, object], ...],
) -> Path:
    run_dir = archive_root / RUNS_DIRECTORY / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / MANIFEST_FILENAME
    manifest_path.write_text(
        json.dumps({"run_id": run_id, "sources": list(sources)}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest_path


def make_manifest_source_entry(output_path: Path) -> dict[str, object]:
    return {
        "source": output_path.parent.name,
        "output_path": str(output_path),
    }


def make_conversation(
    source: str,
    *,
    session: str,
    collected_at: str,
    messages: list[dict[str, object]],
    transcript_completeness: str | None = None,
    contract: dict[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "source": source,
        "execution_context": "cli",
        "collected_at": collected_at,
        "source_session_id": session,
        "messages": messages,
        "contract": contract if contract is not None else {"schema_version": "2026-03-19"},
    }
    if transcript_completeness is not None:
        payload["transcript_completeness"] = transcript_completeness
    return payload


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def freeze_created_at(monkeypatch) -> None:
    monkeypatch.setattr(
        archive_quarantine_export,
        "_utcnow",
        lambda: datetime(2026, 3, 19, 9, 30, tzinfo=UTC),
    )


def seed_archive_root(archive_root: Path) -> None:
    codex_output = write_archive_output(
        archive_root,
        source="codex_cli",
        filename="memory_chat_v1-20260319T060000-codex_cli.jsonl",
        rows=(
            make_conversation(
                "codex_cli",
                session="good-row",
                collected_at="2026-03-19T06:00:00Z",
                messages=[{"role": "user", "text": "Need deploy checklist"}],
            ),
            make_conversation(
                "codex_cli",
                session="bad-schema",
                collected_at="2026-03-19T06:05:00Z",
                contract={"schema_version": "2026-03-18"},
                messages=[{"role": "assistant", "text": "Schema drift"}],
            ),
        ),
    )
    write_archive_output(
        archive_root,
        source="codex_cli",
        filename="memory_chat_v1-20260319T061500-codex_cli.jsonl",
        rows=(
            '{"source":"codex_cli"',
        ),
    )
    cursor_output = write_archive_output(
        archive_root,
        source="cursor_editor",
        filename="memory_chat_v1-20260319T070000-cursor_editor.jsonl",
        rows=(
            make_conversation(
                "cursor_editor",
                session="cursor-session-1",
                collected_at="2026-03-19T07:00:00Z",
                transcript_completeness="partial",
                messages=[{"role": "assistant", "text": "Checklist updated"}],
            ),
        ),
    )
    write_run_manifest(
        archive_root,
        run_id="20260319T080000Z",
        sources=(
            make_manifest_source_entry(codex_output),
            make_manifest_source_entry(cursor_output),
        ),
    )


def test_archive_quarantine_export_dry_run_reports_row_and_finding_counts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    freeze_created_at(monkeypatch)
    seed_archive_root(tmp_path)
    output_dir = tmp_path.parent / "quarantine-bundle"

    exit_code, stdout, stderr = run_cli(
        "archive",
        "quarantine-export",
        "--archive-root",
        str(tmp_path),
        "--output-dir",
        str(output_dir),
    )

    assert exit_code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload == {
        "archive_root": str(tmp_path),
        "created_at": "2026-03-19T09:30:00Z",
        "filters": {"source": None},
        "finding_code_counts": {
            "incomplete_transcript": 1,
            "invalid_contract": 1,
            "malformed_row": 1,
        },
        "finding_count": 3,
        "manifest_path": str(output_dir / "quarantine-manifest.json"),
        "output_dir": str(output_dir),
        "quarantine_path": str(output_dir / "quarantine.jsonl"),
        "row_count": 3,
        "source_count": 2,
        "source_row_counts": {
            "codex_cli": 2,
            "cursor_editor": 1,
        },
        "write_mode": "dry_run",
    }
    assert not (output_dir / "quarantine.jsonl").exists()
    assert not (output_dir / "quarantine-manifest.json").exists()


def test_archive_quarantine_export_execute_writes_bundle_and_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    freeze_created_at(monkeypatch)
    seed_archive_root(tmp_path)
    output_dir = tmp_path.parent / "quarantine-write"

    exit_code, stdout, stderr = run_cli(
        "archive",
        "quarantine-export",
        "--archive-root",
        str(tmp_path),
        "--output-dir",
        str(output_dir),
        "--source",
        "codex_cli",
        "--execute",
    )

    assert exit_code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload == {
        "archive_root": str(tmp_path),
        "created_at": "2026-03-19T09:30:00Z",
        "filters": {"source": "codex_cli"},
        "finding_code_counts": {
            "invalid_contract": 1,
            "malformed_row": 1,
        },
        "finding_count": 2,
        "manifest_path": str(output_dir / "quarantine-manifest.json"),
        "output_dir": str(output_dir),
        "quarantine_path": str(output_dir / "quarantine.jsonl"),
        "row_count": 2,
        "source_count": 1,
        "source_row_counts": {"codex_cli": 2},
        "write_mode": "write",
    }

    rows = read_jsonl(output_dir / "quarantine.jsonl")
    assert rows == [
        {
            "archive_path": "codex_cli/memory_chat_v1-20260319T060000-codex_cli.jsonl",
            "findings": [
                {
                    "code": "invalid_contract",
                    "level": "error",
                    "message": (
                        "source 'codex_cli' row 2 contract schema_version "
                        "'2026-03-18' does not match '2026-03-19'"
                    ),
                }
            ],
            "row": {
                "collected_at": "2026-03-19T06:05:00Z",
                "contract": {"schema_version": "2026-03-18"},
                "execution_context": "cli",
                "messages": [{"role": "assistant", "text": "Schema drift"}],
                "source": "codex_cli",
                "source_session_id": "bad-schema",
            },
            "row_number": 2,
            "source": "codex_cli",
        },
        {
            "archive_path": "codex_cli/memory_chat_v1-20260319T061500-codex_cli.jsonl",
            "findings": [
                {
                    "code": "malformed_row",
                    "level": "error",
                    "message": "source 'codex_cli' row 1 is not valid JSON",
                }
            ],
            "raw_line": '{"source":"codex_cli"',
            "row_number": 1,
            "source": "codex_cli",
        },
    ]

    manifest = json.loads((output_dir / "quarantine-manifest.json").read_text(encoding="utf-8"))
    assert manifest == payload


def test_archive_quarantine_export_filters_to_single_source(
    tmp_path: Path,
    monkeypatch,
) -> None:
    freeze_created_at(monkeypatch)
    seed_archive_root(tmp_path)
    output_dir = tmp_path.parent / "cursor-quarantine"

    exit_code, stdout, stderr = run_cli(
        "archive",
        "quarantine-export",
        "--archive-root",
        str(tmp_path),
        "--output-dir",
        str(output_dir),
        "--source",
        "cursor_editor",
    )

    assert exit_code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["filters"] == {"source": "cursor_editor"}
    assert payload["row_count"] == 1
    assert payload["source_count"] == 1
    assert payload["source_row_counts"] == {"cursor_editor": 1}
    assert payload["finding_code_counts"] == {"incomplete_transcript": 1}
