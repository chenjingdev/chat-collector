from __future__ import annotations

import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

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


def make_conversation(
    source: str,
    *,
    session: str,
    collected_at: str,
    messages: list[dict[str, object]],
    transcript_completeness: str | None = None,
    limitations: list[str] | None = None,
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
    if limitations is not None:
        payload["limitations"] = limitations
    return payload


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


def test_archive_verify_reports_archive_health_and_skips_canonical_outputs(tmp_path: Path) -> None:
    canonical_output = write_archive_output(
        tmp_path,
        source="codex_cli",
        filename="memory_chat_v1-codex_cli.jsonl",
        rows=(
            make_conversation(
                "codex_cli",
                session="codex-session-1",
                collected_at="2026-03-19T06:00:00Z",
                messages=[{"role": "user", "text": "Need deploy checklist"}],
            ),
        ),
    )
    manifest_linked_output = write_archive_output(
        tmp_path,
        source="cursor_editor",
        filename="memory_chat_v1-20260319T070000-cursor_editor.jsonl",
        rows=(
            make_conversation(
                "cursor_editor",
                session="cursor-session-1",
                collected_at="2026-03-19T07:00:00Z",
                transcript_completeness="partial",
                limitations=["variant_unknown"],
                messages=[{"role": "assistant", "text": "Checklist updated"}],
            ),
        ),
    )
    write_run_manifest(
        tmp_path,
        run_id="20260319T080000Z",
        sources=(make_manifest_source_entry(manifest_linked_output),),
    )

    exit_code, stdout, stderr = run_cli(
        "archive",
        "verify",
        "--archive-root",
        str(tmp_path),
    )

    assert exit_code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["status"] == "warning"
    assert payload["source_filter"] is None
    assert payload["manifest_count"] == 1
    assert payload["linked_output_file_count"] == 1
    assert payload["source_count"] == 2
    assert payload["file_count"] == 2
    assert payload["row_count"] == 2
    assert payload["verified_row_count"] == 2
    assert payload["bad_row_count"] == 0
    assert payload["orphan_file_count"] == 0
    assert payload["warning_count"] == 1
    assert payload["error_count"] == 0

    assert payload["sources"]["codex_cli"] == {
        "status": "success",
        "file_count": 1,
        "row_count": 1,
        "verified_row_count": 1,
        "bad_row_count": 0,
        "orphan_file_count": 0,
        "finding_count": 0,
        "warning_count": 0,
        "error_count": 0,
        "files": [
            {
                "path": str(canonical_output),
                "status": "success",
                "row_count": 1,
                "verified_row_count": 1,
                "bad_row_count": 0,
                "finding_count": 0,
                "warning_count": 0,
                "error_count": 0,
                "manifest_linked": False,
                "orphan": False,
            }
        ],
    }
    assert payload["sources"]["cursor_editor"]["status"] == "warning"
    assert payload["sources"]["cursor_editor"]["verified_row_count"] == 1
    assert payload["sources"]["cursor_editor"]["bad_row_count"] == 0
    assert payload["sources"]["cursor_editor"]["files"][0]["manifest_linked"] is True
    assert payload["sources"]["cursor_editor"]["files"][0]["orphan"] is False
    assert {finding["code"] for finding in payload["findings"]} == {"incomplete_transcript"}
    assert "limitations: variant_unknown" in payload["findings"][0]["message"]


def test_archive_verify_reports_bad_rows_invalid_contracts_and_orphans(tmp_path: Path) -> None:
    manifest_linked_output = write_archive_output(
        tmp_path,
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
    orphan_output = write_archive_output(
        tmp_path,
        source="codex_cli",
        filename="memory_chat_v1-20260319T061500-codex_cli.jsonl",
        rows=(
            '{"source":"codex_cli"',
        ),
    )
    write_run_manifest(
        tmp_path,
        run_id="20260319T080000Z",
        sources=(make_manifest_source_entry(manifest_linked_output),),
    )

    exit_code, stdout, stderr = run_cli(
        "archive",
        "verify",
        "--archive-root",
        str(tmp_path),
    )

    assert exit_code == 1
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["status"] == "error"
    assert payload["source_count"] == 1
    assert payload["file_count"] == 2
    assert payload["row_count"] == 3
    assert payload["verified_row_count"] == 1
    assert payload["bad_row_count"] == 2
    assert payload["orphan_file_count"] == 1
    assert payload["warning_count"] == 1
    assert payload["error_count"] == 2

    source_payload = payload["sources"]["codex_cli"]
    assert source_payload["status"] == "error"
    assert source_payload["file_count"] == 2
    assert source_payload["verified_row_count"] == 1
    assert source_payload["bad_row_count"] == 2
    assert source_payload["orphan_file_count"] == 1

    files_by_path = {entry["path"]: entry for entry in source_payload["files"]}
    assert files_by_path[str(manifest_linked_output)] == {
        "path": str(manifest_linked_output),
        "status": "error",
        "row_count": 2,
        "verified_row_count": 1,
        "bad_row_count": 1,
        "finding_count": 1,
        "warning_count": 0,
        "error_count": 1,
        "manifest_linked": True,
        "orphan": False,
    }
    assert files_by_path[str(orphan_output)] == {
        "path": str(orphan_output),
        "status": "error",
        "row_count": 1,
        "verified_row_count": 0,
        "bad_row_count": 1,
        "finding_count": 2,
        "warning_count": 1,
        "error_count": 1,
        "manifest_linked": False,
        "orphan": True,
    }
    assert {finding["code"] for finding in payload["findings"]} == {
        "invalid_contract",
        "malformed_row",
        "orphan_output_file",
    }


def test_archive_verify_filters_to_single_source(tmp_path: Path) -> None:
    write_archive_output(
        tmp_path,
        source="codex_cli",
        filename="memory_chat_v1-codex_cli.jsonl",
        rows=(
            make_conversation(
                "codex_cli",
                session="codex-session-1",
                collected_at="2026-03-19T06:00:00Z",
                messages=[{"role": "user", "text": "Need deploy checklist"}],
            ),
        ),
    )
    cursor_output = write_archive_output(
        tmp_path,
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
        tmp_path,
        run_id="20260319T080000Z",
        sources=(make_manifest_source_entry(cursor_output),),
    )

    exit_code, stdout, stderr = run_cli(
        "archive",
        "verify",
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
    assert payload["file_count"] == 1
    assert payload["row_count"] == 1
    assert payload["verified_row_count"] == 1
    assert payload["bad_row_count"] == 0
    assert set(payload["sources"]) == {"cursor_editor"}
    assert payload["sources"]["cursor_editor"]["status"] == "warning"
