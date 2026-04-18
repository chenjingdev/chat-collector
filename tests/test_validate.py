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


def make_row(
    source: str,
    *,
    messages: list[dict[str, object]] | None = None,
    transcript_completeness: str | None = None,
    contract: dict[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "source": source,
        "execution_context": "cli",
        "collected_at": "2026-03-19T06:00:00Z",
        "messages": messages
        if messages is not None
        else [{"role": "user", "text": "hello"}],
        "contract": contract if contract is not None else {"schema_version": "2026-03-19"},
    }
    if transcript_completeness is not None:
        payload["transcript_completeness"] = transcript_completeness
    return payload


def make_source_entry(
    archive_root: Path,
    *,
    source: str,
    support_level: str,
    status: str,
    support_limitation_summary: str | None = None,
    support_limitations: tuple[str, ...] = (),
    rows: tuple[dict[str, object] | str, ...] = (),
    conversation_count: int | None = None,
    skipped_conversation_count: int = 0,
    written_conversation_count: int | None = None,
    message_count: int | None = None,
    partial: bool = False,
    unsupported: bool = False,
    failed: bool = False,
    create_output: bool = True,
    output_path: Path | None = None,
    archive_root_field: Path | None = None,
) -> dict[str, object]:
    if output_path is None and (create_output or rows):
        output_path = archive_root / source / f"memory_chat_v1-{source}.jsonl"

    if create_output and output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            "\n".join(
                row if isinstance(row, str) else json.dumps(row, ensure_ascii=False)
                for row in rows
            )
            + ("\n" if rows else ""),
            encoding="utf-8",
        )

    if conversation_count is None:
        conversation_count = len(rows)
    if written_conversation_count is None:
        written_conversation_count = len(rows)
    if message_count is None:
        message_count = sum(
            len(row.get("messages", []))
            for row in rows
            if isinstance(row, dict) and isinstance(row.get("messages"), list)
        )

    payload = {
        "source": source,
        "support_level": support_level,
        "status": status,
        "archive_root": str(archive_root_field or archive_root),
        "output_path": str(output_path) if output_path is not None else None,
        "input_roots": [],
        "scanned_artifact_count": 0,
        "conversation_count": conversation_count,
        "skipped_conversation_count": skipped_conversation_count,
        "written_conversation_count": written_conversation_count,
        "message_count": message_count,
        "partial": partial,
        "unsupported": unsupported,
        "failed": failed,
    }
    if support_limitation_summary is not None:
        payload["support_limitation_summary"] = support_limitation_summary
    if support_limitations:
        payload["support_limitations"] = list(support_limitations)
    return payload


def write_run_manifest(
    archive_root: Path,
    *,
    run_id: str,
    sources: tuple[dict[str, object], ...],
    archive_root_field: Path | None = None,
) -> Path:
    run_dir = archive_root / RUNS_DIRECTORY / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / MANIFEST_FILENAME
    payload = {
        "run_id": run_id,
        "archive_root": str(archive_root_field or archive_root),
        "run_dir": str(run_dir),
        "manifest_path": str(manifest_path),
        "source_count": len(sources),
        "failed_source_count": sum(1 for source in sources if source["failed"]),
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


def finding_codes(payload: dict[str, object]) -> set[str]:
    return {finding["code"] for finding in payload["findings"]}  # type: ignore[index]


def test_validate_reports_success_for_valid_run(tmp_path: Path) -> None:
    write_run_manifest(
        tmp_path,
        run_id="20260319T060000Z",
        sources=(
            make_source_entry(
                tmp_path,
                source="codex_cli",
                support_level="complete",
                status="complete",
                rows=(
                    make_row("codex_cli"),
                    make_row(
                        "codex_cli",
                        messages=[{"role": "assistant", "text": "done"}],
                    ),
                ),
                message_count=2,
            ),
        ),
    )

    exit_code, stdout, stderr = run_cli(
        "validate",
        "--archive-root",
        str(tmp_path),
        "--run",
        "20260319T060000Z",
    )

    assert exit_code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["status"] == "success"
    assert payload["summary"]["warning_count"] == 0
    assert payload["summary"]["error_count"] == 0
    assert payload["sources"] == [
        {
            "source": "codex_cli",
            "validation_status": "success",
            "support_level": "complete",
            "status": "complete",
            "failed": False,
            "output_path": str(
                tmp_path / "codex_cli" / "memory_chat_v1-codex_cli.jsonl"
            ),
            "row_count": 2,
            "actual_message_count": 2,
            "declared_conversation_count": 2,
            "declared_skipped_conversation_count": 0,
            "declared_written_conversation_count": 2,
            "declared_message_count": 2,
            "drift_suspected": False,
        }
    ]
    assert finding_codes(payload) >= {"manifest_loaded", "source_validated"}


def test_validate_reports_warnings_for_partial_but_valid_output(tmp_path: Path) -> None:
    write_run_manifest(
        tmp_path,
        run_id="20260319T061500Z",
        sources=(
            make_source_entry(
                tmp_path,
                source="cursor_editor",
                support_level="partial",
                status="partial",
                rows=(
                    make_row(
                        "cursor_editor",
                        transcript_completeness="partial",
                    ),
                ),
                partial=True,
                support_limitation_summary=(
                    "Cursor editor recovery restores known explicit "
                    "cursorDiskKV bubble body variants, but sessions whose "
                    "headers resolve only to empty or tool-only rows remain "
                    "partial and opt-in for unattended batches."
                ),
                support_limitations=(
                    "Composer sessions whose cursorDiskKV headers resolve only to empty or tool-only bubble rows still degrade to partial output, with missing bubble ids retained in session metadata.",
                ),
            ),
        ),
    )

    exit_code, stdout, stderr = run_cli(
        "validate",
        "--archive-root",
        str(tmp_path),
        "--run",
        "20260319T061500Z",
    )

    assert exit_code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["status"] == "warning"
    assert payload["summary"]["warning_count"] >= 3
    assert payload["summary"]["error_count"] == 0
    assert payload["sources"][0]["validation_status"] == "warning"
    assert payload["sources"][0]["support_limitation_summary"] == (
        "Cursor editor recovery restores known explicit cursorDiskKV bubble "
        "body variants, but sessions whose headers resolve only to empty or "
        "tool-only rows remain partial and opt-in for unattended batches."
    )
    assert payload["sources"][0]["support_limitations"] == [
        "Composer sessions whose cursorDiskKV headers resolve only to empty or tool-only bubble rows still degrade to partial output, with missing bubble ids retained in session metadata.",
    ]
    assert finding_codes(payload) >= {
        "degraded_support_level",
        "degraded_source_status",
        "incomplete_transcript",
    }
    assert any(
        finding["code"] == "degraded_support_level"
        and "tool-only rows" in finding["message"]
        for finding in payload["findings"]
    )


def test_validate_reports_errors_for_contract_violations(tmp_path: Path) -> None:
    missing_output_path = tmp_path / "claude_code_cli" / "memory_chat_v1-missing.jsonl"
    write_run_manifest(
        tmp_path,
        run_id="20260319T063000Z",
        sources=(
            make_source_entry(
                tmp_path,
                source="claude_code_cli",
                support_level="broken",
                status="complete",
                create_output=False,
                output_path=missing_output_path,
                conversation_count=1,
                written_conversation_count=1,
                message_count=1,
            ),
            make_source_entry(
                tmp_path,
                source="codex_cli",
                support_level="complete",
                status="complete",
                rows=(
                    make_row(
                        "codex_cli",
                        messages=[{"role": "wizard", "text": "bad role"}],
                        contract={},
                    ),
                    "{not json",
                ),
                conversation_count=2,
                written_conversation_count=1,
                message_count=1,
            ),
        ),
    )

    exit_code, stdout, stderr = run_cli(
        "validate",
        "--archive-root",
        str(tmp_path),
        "--run",
        "20260319T063000Z",
    )

    assert exit_code == 1
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["status"] == "error"
    assert payload["summary"]["error_count"] >= 5
    assert finding_codes(payload) >= {
        "invalid_enum",
        "missing_file",
        "missing_required_field",
        "malformed_row",
        "count_mismatch",
    }


def test_validate_reports_repo_inside_archive_root_violation(tmp_path: Path) -> None:
    repo_root = cli.repository_root()
    write_run_manifest(
        tmp_path,
        run_id="20260319T064500Z",
        archive_root_field=repo_root,
        sources=(
            make_source_entry(
                tmp_path,
                source="gemini_cli",
                support_level="complete",
                status="complete",
                rows=(make_row("gemini_cli"),),
            ),
        ),
    )

    exit_code, stdout, stderr = run_cli(
        "validate",
        "--archive-root",
        str(tmp_path),
        "--run",
        "20260319T064500Z",
    )

    assert exit_code == 1
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["status"] == "error"
    assert any(
        finding["code"] == "external_archive_only_violation"
        and finding.get("path") == str(repo_root)
        for finding in payload["findings"]
    )
