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


def write_run_manifest(
    archive_root: Path,
    *,
    run_id: str,
    sources: tuple[dict[str, object], ...],
    started_at: str | None = None,
    completed_at: str | None = None,
) -> Path:
    run_dir = archive_root / RUNS_DIRECTORY / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / MANIFEST_FILENAME
    payload: dict[str, object] = {
        "run_id": run_id,
        "archive_root": str(archive_root),
        "run_dir": str(run_dir),
        "manifest_path": str(manifest_path),
        "source_count": len(sources),
        "failed_source_count": sum(1 for source in sources if source.get("failed", False)),
        "conversation_count": sum(int(source.get("conversation_count", 0)) for source in sources),
        "skipped_conversation_count": sum(
            int(source.get("skipped_conversation_count", 0)) for source in sources
        ),
        "written_conversation_count": sum(
            int(source.get("written_conversation_count", 0)) for source in sources
        ),
        "message_count": sum(int(source.get("message_count", 0)) for source in sources),
        "sources": list(sources),
    }
    if started_at is not None:
        payload["started_at"] = started_at
    if completed_at is not None:
        payload["completed_at"] = completed_at
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


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


def make_manifest_source_entry(
    *,
    archive_root: Path,
    source: str,
    output_path: Path | None,
    support_level: str,
    status: str,
    partial: bool = False,
    unsupported: bool = False,
    failed: bool = False,
    conversation_count: int = 0,
    message_count: int = 0,
    support_limitation_summary: str | None = None,
    support_limitations: tuple[str, ...] = (),
) -> dict[str, object]:
    payload: dict[str, object] = {
        "source": source,
        "support_level": support_level,
        "status": status,
        "archive_root": str(archive_root),
        "output_path": None if output_path is None else str(output_path),
        "input_roots": [],
        "scanned_artifact_count": 0,
        "conversation_count": conversation_count,
        "skipped_conversation_count": 0,
        "written_conversation_count": 0 if output_path is None else conversation_count,
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


def write_baseline(archive_root: Path, *, entries: list[dict[str, object]]) -> Path:
    baseline_path = archive_root / "baseline-policy.json"
    baseline_path.write_text(
        json.dumps({"version": 1, "entries": entries}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return baseline_path


def test_validate_applies_baseline_without_hiding_raw_findings(tmp_path: Path) -> None:
    output_path = write_archive_output(
        tmp_path,
        source="cursor_editor",
        filename="memory_chat_v1-cursor_editor.jsonl",
        rows=(
            make_conversation(
                "cursor_editor",
                session="cursor-session-1",
                collected_at="2026-03-20T01:00:00Z",
                transcript_completeness="partial",
                messages=[{"role": "assistant", "text": "Recovered partial transcript"}],
            ),
        ),
    )
    write_run_manifest(
        tmp_path,
        run_id="20260320T010000Z",
        sources=(
            make_manifest_source_entry(
                archive_root=tmp_path,
                source="cursor_editor",
                output_path=output_path,
                support_level="partial",
                status="partial",
                partial=True,
                conversation_count=1,
                message_count=1,
                support_limitation_summary="Known Cursor editor cache gaps remain partial.",
                support_limitations=(
                    "Composer sessions whose cursorDiskKV headers resolve only to empty or tool-only bubble rows still degrade to partial output, with missing bubble ids retained in session metadata.",
                ),
            ),
        ),
    )
    write_baseline(
        tmp_path,
        entries=[
            {
                "kind": "degraded_source",
                "source": "cursor_editor",
                "support_level": "partial",
                "status": "partial",
                "reason": "Known Cursor editor degradation is expected during unattended runs.",
            },
            {
                "kind": "finding",
                "report": "validate",
                "source": "cursor_editor",
                "code": "incomplete_transcript",
                "reason": "Partial transcript rows are expected for this source.",
            },
        ],
    )

    exit_code, stdout, stderr = run_cli(
        "validate",
        "--archive-root",
        str(tmp_path),
        "--run",
        "20260320T010000Z",
    )

    assert exit_code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["status"] == "success"
    assert payload["raw_status"] == "warning"
    assert payload["summary"]["warning_count"] == 0
    assert payload["summary"]["raw_warning_count"] == 3
    assert payload["summary"]["suppressed_warning_count"] == 3
    assert payload["baseline"]["path"] == str(tmp_path / "baseline-policy.json")
    assert payload["sources"][0]["validation_status"] == "success"
    assert payload["sources"][0]["raw_validation_status"] == "warning"
    assert payload["sources"][0]["suppressed_warning_count"] == 3

    suppressed_codes = {
        finding["code"]
        for finding in payload["findings"]
        if finding.get("suppressed") is True
    }
    assert suppressed_codes == {
        "degraded_support_level",
        "degraded_source_status",
        "incomplete_transcript",
    }


def test_archive_verify_applies_baseline_to_known_findings(tmp_path: Path) -> None:
    output_path = write_archive_output(
        tmp_path,
        source="cursor_editor",
        filename="memory_chat_v1-20260320T020000-cursor_editor.jsonl",
        rows=(
            make_conversation(
                "cursor_editor",
                session="cursor-session-1",
                collected_at="2026-03-20T02:00:00Z",
                transcript_completeness="partial",
                messages=[{"role": "assistant", "text": "Recovered partial transcript"}],
            ),
        ),
    )
    write_run_manifest(
        tmp_path,
        run_id="20260320T020500Z",
        sources=(
            {
                "source": "cursor_editor",
                "output_path": str(output_path),
            },
        ),
    )
    write_baseline(
        tmp_path,
        entries=[
            {
                "kind": "finding",
                "report": "archive_verify",
                "source": "cursor_editor",
                "code": "incomplete_transcript",
                "reason": "Known partial rows should not page the operator.",
            }
        ],
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
    assert payload["status"] == "success"
    assert payload["raw_status"] == "warning"
    assert payload["warning_count"] == 0
    assert payload["raw_warning_count"] == 1
    assert payload["suppressed_warning_count"] == 1
    assert payload["sources"]["cursor_editor"]["status"] == "success"
    assert payload["sources"]["cursor_editor"]["raw_status"] == "warning"
    assert payload["sources"]["cursor_editor"]["suppressed_warning_count"] == 1
    assert payload["sources"]["cursor_editor"]["files"][0]["status"] == "success"
    assert payload["sources"]["cursor_editor"]["files"][0]["raw_status"] == "warning"
    assert payload["findings"][0]["suppressed"] is True


def test_archive_anomalies_and_digest_separate_suppressed_known_noise(
    tmp_path: Path,
) -> None:
    output_path = write_archive_output(
        tmp_path,
        source="cursor_editor",
        filename="memory_chat_v1-20260320T030000-cursor_editor.jsonl",
        rows=(
            make_conversation(
                "cursor_editor",
                session="cursor-session-1",
                collected_at="2026-03-20T03:00:00Z",
                transcript_completeness="unsupported",
                limitations=[
                    "deleted draft messages unavailable",
                    "local cache omitted edits",
                ],
                messages=[
                    {"role": "user", "text": "Need recovery notes"},
                    {"role": "assistant", "text": "Recovered metadata only"},
                ],
            ),
            make_conversation(
                "cursor_editor",
                session="cursor-session-2",
                collected_at="2026-03-20T03:05:00Z",
                transcript_completeness="unsupported",
                limitations=["deleted draft messages unavailable"],
                messages=[
                    {"role": "user", "text": "Need recovery notes"},
                    {"role": "assistant", "text": "Recovered metadata only"},
                ],
            ),
        ),
    )
    write_run_manifest(
        tmp_path,
        run_id="20260320T031000Z",
        started_at="2026-03-20T03:10:00Z",
        completed_at="2026-03-20T03:10:08Z",
        sources=(
            make_manifest_source_entry(
                archive_root=tmp_path,
                source="cursor_editor",
                output_path=output_path,
                support_level="partial",
                status="partial",
                partial=True,
                conversation_count=2,
                message_count=4,
            ),
        ),
    )
    write_baseline(
        tmp_path,
        entries=[
            {
                "kind": "degraded_source",
                "source": "cursor_editor",
                "support_level": "partial",
                "status": "partial",
                "reason": "Cursor editor degradation is expected until cache coverage improves.",
            },
            {
                "kind": "finding",
                "report": "archive_verify",
                "source": "cursor_editor",
                "code": "incomplete_transcript",
                "reason": "Known partial rows should not show as new verify warnings.",
            },
            {
                "kind": "limitation",
                "source": "cursor_editor",
                "limitation": "deleted draft messages unavailable",
                "reason": "Deleted drafts are not recoverable from the local cache.",
            },
            {
                "kind": "limitation",
                "source": "cursor_editor",
                "limitation": "local cache omitted edits",
                "reason": "Known editor cache omission.",
            },
        ],
    )

    anomalies_exit_code, anomalies_stdout, anomalies_stderr = run_cli(
        "archive",
        "anomalies",
        "--archive-root",
        str(tmp_path),
        "--low-message-count",
        "0",
    )

    assert anomalies_exit_code == 0
    assert anomalies_stderr == ""
    anomalies_payload = json.loads(anomalies_stdout)
    assert anomalies_payload["suspicious_source_count"] == 0
    assert anomalies_payload["raw_suspicious_source_count"] == 1
    assert anomalies_payload["suspicious_conversation_count"] == 0
    assert anomalies_payload["raw_suspicious_conversation_count"] == 2
    cursor_anomalies = anomalies_payload["sources"]["cursor_editor"]
    assert cursor_anomalies["suspicious"] is False
    assert cursor_anomalies["raw_suspicious_conversation_count"] == 2
    assert cursor_anomalies["suppressed_suspicious_conversation_count"] == 2
    assert cursor_anomalies["source_reasons"][0]["suppressed"] is True

    digest_exit_code, digest_stdout, digest_stderr = run_cli(
        "archive",
        "digest",
        "--archive-root",
        str(tmp_path),
    )

    assert digest_exit_code == 0
    assert digest_stderr == ""
    digest_payload = json.loads(digest_stdout)
    assert digest_payload["status"] == "success"
    assert digest_payload["raw_status"] == "warning"
    assert digest_payload["latest_run"]["degraded_source_count"] == 0
    assert digest_payload["latest_run"]["raw_degraded_source_count"] == 1
    assert digest_payload["latest_run"]["suppressed_degraded_sources"] == [
        "cursor_editor"
    ]
    assert digest_payload["overview"]["warning_count"] == 0
    assert digest_payload["overview"]["raw_warning_count"] == 2
    assert digest_payload["overview"]["suppressed_warning_count"] == 2
    assert digest_payload["overview"]["suspicious_source_count"] == 0
    assert digest_payload["overview"]["raw_suspicious_source_count"] == 1
    assert digest_payload["top_limitations"] == [
        {
            "count": 0,
            "limitation": "deleted draft messages unavailable",
            "raw_count": 2,
            "suppressed_count": 2,
        },
        {
            "count": 0,
            "limitation": "local cache omitted edits",
            "raw_count": 1,
            "suppressed_count": 1,
        },
    ]
    cursor_digest = digest_payload["sources"]["cursor_editor"]
    assert cursor_digest["run_degraded"] is False
    assert cursor_digest["raw_run_degraded"] is True
    assert cursor_digest["suppressed_run_degraded"] is True
    assert cursor_digest["warning_count"] == 0
    assert cursor_digest["suppressed_warning_count"] == 2
    assert cursor_digest["raw_verify_status"] == "warning"
    assert cursor_digest["suspicious"] is False
    assert cursor_digest["raw_suspicious_conversation_count"] == 2


def test_baseline_snapshot_merges_current_validate_state(tmp_path: Path) -> None:
    output_path = write_archive_output(
        tmp_path,
        source="cursor_editor",
        filename="memory_chat_v1-cursor_editor.jsonl",
        rows=(
            make_conversation(
                "cursor_editor",
                session="cursor-session-1",
                collected_at="2026-03-20T04:00:00Z",
                transcript_completeness="partial",
                messages=[{"role": "assistant", "text": "Recovered partial transcript"}],
            ),
        ),
    )
    write_run_manifest(
        tmp_path,
        run_id="20260320T040000Z",
        sources=(
            make_manifest_source_entry(
                archive_root=tmp_path,
                source="cursor_editor",
                output_path=output_path,
                support_level="partial",
                status="partial",
                partial=True,
                conversation_count=1,
                message_count=1,
            ),
        ),
    )

    exit_code, stdout, stderr = run_cli(
        "baseline",
        "snapshot",
        "--archive-root",
        str(tmp_path),
        "--from",
        "validate",
        "--run",
        "20260320T040000Z",
        "--reason",
        "Known Cursor editor degradation baseline.",
    )

    assert exit_code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["snapshot_entry_count"] == 2
    assert payload["added_entry_count"] == 2
    assert payload["entry_count"] == 2

    baseline_payload = json.loads(
        (tmp_path / "baseline-policy.json").read_text(encoding="utf-8")
    )
    assert baseline_payload["version"] == 1
    assert {entry["kind"] for entry in baseline_payload["entries"]} == {
        "degraded_source",
        "finding",
    }

    second_exit_code, second_stdout, second_stderr = run_cli(
        "baseline",
        "snapshot",
        "--archive-root",
        str(tmp_path),
        "--from",
        "validate",
        "--run",
        "20260320T040000Z",
        "--reason",
        "Known Cursor editor degradation baseline.",
    )

    assert second_exit_code == 0
    assert second_stderr == ""
    second_payload = json.loads(second_stdout)
    assert second_payload["snapshot_entry_count"] == 2
    assert second_payload["added_entry_count"] == 0
    assert second_payload["entry_count"] == 2
