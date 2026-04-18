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


def write_run_manifest(
    archive_root: Path,
    *,
    run_id: str,
    started_at: str,
    completed_at: str,
    sources: tuple[dict[str, object], ...],
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
        "source_count": len(sources),
        "failed_source_count": sum(1 for source in sources if source["failed"]),
        "scanned_artifact_count": sum(
            int(source["scanned_artifact_count"]) for source in sources
        ),
        "conversation_count": sum(int(source["conversation_count"]) for source in sources),
        "skipped_conversation_count": sum(
            int(source["skipped_conversation_count"]) for source in sources
        ),
        "written_conversation_count": sum(
            int(source["written_conversation_count"]) for source in sources
        ),
        "message_count": sum(int(source["message_count"]) for source in sources),
        "sources": list(sources),
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def make_source_entry(
    archive_root: Path,
    *,
    source: str,
    support_level: str,
    status: str,
    scanned_artifact_count: int,
    conversation_count: int,
    message_count: int,
    support_limitation_summary: str | None = None,
    support_limitations: tuple[str, ...] = (),
    skipped_conversation_count: int = 0,
    written_conversation_count: int | None = None,
    partial: bool = False,
    unsupported: bool = False,
    failed: bool = False,
    failure_reason: str | None = None,
    create_output: bool = True,
) -> dict[str, object]:
    if written_conversation_count is None:
        written_conversation_count = conversation_count

    output_path: Path | None = None
    if create_output:
        output_dir = archive_root / source
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"memory_chat_v1-{source}.jsonl"
        output_path.write_text(
            json.dumps({"source": source, "messages": []}) + "\n",
            encoding="utf-8",
        )

    payload: dict[str, object] = {
        "source": source,
        "support_level": support_level,
        "status": status,
        "archive_root": str(archive_root),
        "output_path": str(output_path) if output_path is not None else None,
        "input_roots": [],
        "scanned_artifact_count": scanned_artifact_count,
        "conversation_count": conversation_count,
        "skipped_conversation_count": skipped_conversation_count,
        "written_conversation_count": written_conversation_count,
        "message_count": message_count,
        "partial": partial,
        "unsupported": unsupported,
        "failed": failed,
    }
    if failure_reason is not None:
        payload["failure_reason"] = failure_reason
    if support_limitation_summary is not None:
        payload["support_limitation_summary"] = support_limitation_summary
    if support_limitations:
        payload["support_limitations"] = list(support_limitations)
    return payload


def test_runs_list_orders_manifests_latest_first(tmp_path: Path) -> None:
    write_run_manifest(
        tmp_path,
        run_id="20260319T010000Z",
        started_at="2026-03-19T01:00:00Z",
        completed_at="2026-03-19T01:00:05Z",
        sources=(
            make_source_entry(
                tmp_path,
                source="codex_cli",
                support_level="complete",
                status="complete",
                scanned_artifact_count=2,
                conversation_count=1,
                message_count=3,
            ),
        ),
    )
    latest_manifest = write_run_manifest(
        tmp_path,
        run_id="20260319T020000Z",
        started_at="2026-03-19T02:00:00Z",
        completed_at="2026-03-19T02:00:06Z",
        sources=(
            make_source_entry(
                tmp_path,
                source="claude_code_cli",
                support_level="partial",
                status="partial",
                scanned_artifact_count=4,
                conversation_count=2,
                message_count=5,
                partial=True,
            ),
        ),
    )

    exit_code, stdout, stderr = run_cli(
        "runs",
        "list",
        "--archive-root",
        str(tmp_path),
    )

    assert exit_code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["archive_root"] == str(tmp_path)
    assert payload["run_count"] == 2
    assert [run["run_id"] for run in payload["runs"]] == [
        "20260319T020000Z",
        "20260319T010000Z",
    ]
    assert payload["runs"][0]["manifest_path"] == str(latest_manifest)
    assert payload["runs"][0]["partial_source_count"] == 1
    assert payload["runs"][0]["written_conversation_count"] == 2


def test_runs_latest_reports_source_statuses_and_counts(tmp_path: Path) -> None:
    write_run_manifest(
        tmp_path,
        run_id="20260319T030000Z",
        started_at="2026-03-19T03:00:00Z",
        completed_at="2026-03-19T03:00:12Z",
        sources=(
            make_source_entry(
                tmp_path,
                source="codex_cli",
                support_level="complete",
                status="complete",
                scanned_artifact_count=3,
                conversation_count=2,
                message_count=7,
            ),
            make_source_entry(
                tmp_path,
                source="claude_code_cli",
                support_level="partial",
                status="partial",
                scanned_artifact_count=2,
                conversation_count=1,
                message_count=2,
                partial=True,
            ),
            make_source_entry(
                tmp_path,
                source="cursor_editor",
                support_level="partial",
                status="unsupported",
                scanned_artifact_count=0,
                conversation_count=0,
                message_count=0,
                support_limitation_summary=(
                    "Cursor editor recovery restores known explicit "
                    "cursorDiskKV bubble body variants, but sessions whose "
                    "headers resolve only to empty or tool-only rows remain "
                    "partial and opt-in for unattended batches."
                ),
                support_limitations=(
                    "Composer sessions whose cursorDiskKV headers resolve only to empty or tool-only bubble rows still degrade to partial output, with missing bubble ids retained in session metadata.",
                ),
                unsupported=True,
                create_output=False,
            ),
            make_source_entry(
                tmp_path,
                source="gemini",
                support_level="complete",
                status="failed",
                scanned_artifact_count=0,
                conversation_count=0,
                message_count=0,
                failed=True,
                failure_reason="RuntimeError: collector exploded",
                create_output=False,
            ),
        ),
    )

    exit_code, stdout, stderr = run_cli(
        "runs",
        "latest",
        "--archive-root",
        str(tmp_path),
    )

    assert exit_code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["run_id"] == "20260319T030000Z"
    assert payload["source_count"] == 4
    assert payload["failed_source_count"] == 1
    assert payload["partial_source_count"] == 1
    assert payload["unsupported_source_count"] == 1
    assert payload["scanned_artifact_count"] == 5
    assert payload["conversation_count"] == 3
    assert payload["skipped_conversation_count"] == 0
    assert payload["written_conversation_count"] == 3
    assert payload["message_count"] == 9

    sources = {entry["source"]: entry for entry in payload["sources"]}
    assert sources["codex_cli"]["success"] is True
    assert sources["codex_cli"]["output_path"].endswith("memory_chat_v1-codex_cli.jsonl")
    assert sources["claude_code_cli"]["partial"] is True
    assert sources["claude_code_cli"]["status"] == "partial"
    assert sources["cursor_editor"]["unsupported"] is True
    assert sources["cursor_editor"]["output_path"] is None
    assert sources["cursor_editor"]["support_limitation_summary"] == (
        "Cursor editor recovery restores known explicit cursorDiskKV bubble "
        "body variants, but sessions whose headers resolve only to empty or "
        "tool-only rows remain partial and opt-in for unattended batches."
    )
    assert sources["cursor_editor"]["support_limitations"] == [
        "Composer sessions whose cursorDiskKV headers resolve only to empty or tool-only bubble rows still degrade to partial output, with missing bubble ids retained in session metadata.",
    ]
    assert sources["gemini"]["failed"] is True
    assert sources["gemini"]["success"] is False
    assert sources["gemini"]["failure_reason"] == "RuntimeError: collector exploded"


def test_runs_show_returns_requested_run_even_when_not_latest(tmp_path: Path) -> None:
    write_run_manifest(
        tmp_path,
        run_id="20260319T040000Z",
        started_at="2026-03-19T04:00:00Z",
        completed_at="2026-03-19T04:00:04Z",
        sources=(
            make_source_entry(
                tmp_path,
                source="codex_cli",
                support_level="complete",
                status="complete",
                scanned_artifact_count=1,
                conversation_count=1,
                message_count=2,
            ),
        ),
    )
    write_run_manifest(
        tmp_path,
        run_id="20260319T050000Z",
        started_at="2026-03-19T05:00:00Z",
        completed_at="2026-03-19T05:00:04Z",
        sources=(
            make_source_entry(
                tmp_path,
                source="claude_code_cli",
                support_level="partial",
                status="partial",
                scanned_artifact_count=1,
                conversation_count=1,
                message_count=1,
                partial=True,
            ),
        ),
    )

    exit_code, stdout, stderr = run_cli(
        "runs",
        "show",
        "20260319T040000Z",
        "--archive-root",
        str(tmp_path),
    )

    assert exit_code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["run_id"] == "20260319T040000Z"
    assert payload["sources"][0]["source"] == "codex_cli"


def test_runs_latest_fails_when_source_output_is_missing(tmp_path: Path) -> None:
    missing_output = tmp_path / "claude_code_cli" / "memory_chat_v1-claude_code_cli.jsonl"
    write_run_manifest(
        tmp_path,
        run_id="20260319T060000Z",
        started_at="2026-03-19T06:00:00Z",
        completed_at="2026-03-19T06:00:04Z",
        sources=(
            {
                "source": "claude_code_cli",
                "support_level": "complete",
                "status": "complete",
                "archive_root": str(tmp_path),
                "output_path": str(missing_output),
                "input_roots": [],
                "scanned_artifact_count": 1,
                "conversation_count": 1,
                "skipped_conversation_count": 0,
                "written_conversation_count": 1,
                "message_count": 1,
                "partial": False,
                "unsupported": False,
                "failed": False,
            },
        ),
    )

    exit_code, stdout, stderr = run_cli(
        "runs",
        "latest",
        "--archive-root",
        str(tmp_path),
    )

    assert exit_code == 1
    assert stdout == ""
    assert "output is missing" in stderr
    assert "claude_code_cli" in stderr


def test_runs_list_fails_clearly_when_no_manifests_exist(tmp_path: Path) -> None:
    exit_code, stdout, stderr = run_cli(
        "runs",
        "list",
        "--archive-root",
        str(tmp_path),
    )

    assert exit_code == 1
    assert stdout == ""
    assert "run manifests directory does not exist" in stderr
