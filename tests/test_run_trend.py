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
    rows: tuple[dict[str, object], ...],
) -> Path:
    output_path = archive_root / source / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    return output_path


def make_conversation(
    source: str,
    *,
    session: str,
    collected_at: str,
    message_count: int,
    transcript_completeness: str | None = None,
    limitations: list[str] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "source": source,
        "execution_context": "cli",
        "collected_at": collected_at,
        "source_session_id": session,
        "messages": [
            {"role": "assistant", "text": f"{source} message {index}"}
            for index in range(1, message_count + 1)
        ],
        "contract": {"schema_version": "2026-03-19"},
    }
    if transcript_completeness is not None:
        payload["transcript_completeness"] = transcript_completeness
    if limitations is not None:
        payload["limitations"] = limitations
    return payload


def make_source_entry(
    archive_root: Path,
    *,
    source: str,
    support_level: str,
    status: str,
    scanned_artifact_count: int,
    conversation_count: int,
    message_count: int,
    output_path: Path | None,
    skipped_conversation_count: int = 0,
    written_conversation_count: int | None = None,
    partial: bool = False,
    unsupported: bool = False,
    failed: bool = False,
    failure_reason: str | None = None,
) -> dict[str, object]:
    if written_conversation_count is None:
        written_conversation_count = conversation_count

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
        "redaction_event_count": 0,
        "partial": partial,
        "unsupported": unsupported,
        "failed": failed,
    }
    if failure_reason is not None:
        payload["failure_reason"] = failure_reason
    return payload


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
        "redaction_event_count": 0,
        "sources": list(sources),
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def seed_archive_root(archive_root: Path) -> None:
    codex_partial_output = write_archive_output(
        archive_root,
        source="codex_cli",
        filename="memory_chat_v1-20260319T010000-codex_cli.jsonl",
        rows=(
            make_conversation(
                "codex_cli",
                session="codex-partial-1",
                collected_at="2026-03-19T01:00:02Z",
                message_count=2,
                transcript_completeness="partial",
                limitations=["missing deleted messages"],
            ),
        ),
    )
    codex_complete_output = write_archive_output(
        archive_root,
        source="codex_cli",
        filename="memory_chat_v1-20260319T020000-codex_cli.jsonl",
        rows=(
            make_conversation(
                "codex_cli",
                session="codex-complete-1",
                collected_at="2026-03-19T02:00:02Z",
                message_count=2,
            ),
            make_conversation(
                "codex_cli",
                session="codex-complete-2",
                collected_at="2026-03-19T02:00:04Z",
                message_count=1,
            ),
        ),
    )
    claude_complete_run2 = write_archive_output(
        archive_root,
        source="claude_code_cli",
        filename="memory_chat_v1-20260319T020000-claude_code_cli.jsonl",
        rows=(
            make_conversation(
                "claude_code_cli",
                session="claude-complete-1",
                collected_at="2026-03-19T02:00:03Z",
                message_count=1,
            ),
        ),
    )
    claude_complete_run3 = write_archive_output(
        archive_root,
        source="claude_code_cli",
        filename="memory_chat_v1-20260319T030000-claude_code_cli.jsonl",
        rows=(
            make_conversation(
                "claude_code_cli",
                session="claude-complete-2",
                collected_at="2026-03-19T03:00:03Z",
                message_count=2,
            ),
        ),
    )

    write_run_manifest(
        archive_root,
        run_id="20260319T010000Z",
        started_at="2026-03-19T01:00:00Z",
        completed_at="2026-03-19T01:00:05Z",
        sources=(
            make_source_entry(
                archive_root,
                source="codex_cli",
                support_level="partial",
                status="partial",
                scanned_artifact_count=3,
                conversation_count=1,
                message_count=2,
                output_path=codex_partial_output,
                partial=True,
            ),
            make_source_entry(
                archive_root,
                source="claude_code_cli",
                support_level="scaffold",
                status="unsupported",
                scanned_artifact_count=0,
                conversation_count=0,
                message_count=0,
                output_path=None,
                unsupported=True,
            ),
        ),
    )
    write_run_manifest(
        archive_root,
        run_id="20260319T020000Z",
        started_at="2026-03-19T02:00:00Z",
        completed_at="2026-03-19T02:00:06Z",
        sources=(
            make_source_entry(
                archive_root,
                source="codex_cli",
                support_level="complete",
                status="complete",
                scanned_artifact_count=4,
                conversation_count=2,
                message_count=3,
                output_path=codex_complete_output,
            ),
            make_source_entry(
                archive_root,
                source="claude_code_cli",
                support_level="complete",
                status="complete",
                scanned_artifact_count=1,
                conversation_count=1,
                message_count=1,
                output_path=claude_complete_run2,
            ),
        ),
    )
    write_run_manifest(
        archive_root,
        run_id="20260319T030000Z",
        started_at="2026-03-19T03:00:00Z",
        completed_at="2026-03-19T03:00:07Z",
        sources=(
            make_source_entry(
                archive_root,
                source="codex_cli",
                support_level="complete",
                status="failed",
                scanned_artifact_count=0,
                conversation_count=0,
                message_count=0,
                output_path=None,
                failed=True,
                failure_reason="RuntimeError: collector exploded",
            ),
            make_source_entry(
                archive_root,
                source="claude_code_cli",
                support_level="complete",
                status="complete",
                scanned_artifact_count=2,
                conversation_count=1,
                message_count=2,
                output_path=claude_complete_run3,
            ),
        ),
    )


def test_runs_trend_reports_source_health_timeline_and_archive_stats(tmp_path: Path) -> None:
    seed_archive_root(tmp_path)

    exit_code, stdout, stderr = run_cli(
        "runs",
        "trend",
        "--archive-root",
        str(tmp_path),
    )

    assert exit_code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["archive_root"] == str(tmp_path)
    assert payload["source_filter"] is None
    assert payload["run_count"] == 3
    assert payload["source_count"] == 2
    assert payload["transition_count"] == 3
    assert payload["degraded_to_complete_count"] == 2
    assert [run["run_id"] for run in payload["runs"]] == [
        "20260319T010000Z",
        "20260319T020000Z",
        "20260319T030000Z",
    ]

    codex_trend = payload["sources"]["codex_cli"]
    assert codex_trend["run_count"] == 3
    assert codex_trend["first_run_id"] == "20260319T010000Z"
    assert codex_trend["latest_run_id"] == "20260319T030000Z"
    assert codex_trend["latest_status"] == "failed"
    assert codex_trend["latest_support_level"] == "complete"
    assert codex_trend["support_levels"] == ["partial", "complete"]
    assert codex_trend["status_counts"] == {
        "complete": 1,
        "failed": 1,
        "partial": 1,
        "unsupported": 0,
    }
    assert codex_trend["status_ratios"] == {
        "complete": 1 / 3,
        "degraded": 1 / 3,
        "failed": 1 / 3,
        "partial": 1 / 3,
        "unsupported": 0.0,
    }
    assert codex_trend["degraded_to_complete_count"] == 1
    assert codex_trend["transition_counts"] == {
        "complete_to_failed": 1,
        "partial_to_complete": 1,
    }
    assert [entry["label"] for entry in codex_trend["transitions"]] == [
        "partial_to_complete",
        "complete_to_failed",
    ]

    first_codex_point, second_codex_point, third_codex_point = codex_trend["timeline"]
    assert first_codex_point["degraded"] is True
    assert first_codex_point["manifest"]["status"] == "partial"
    assert first_codex_point["status_ratios"] == {
        "complete": 0.0,
        "degraded": 1.0,
        "failed": 0.0,
        "partial": 1.0,
        "unsupported": 0.0,
    }
    assert first_codex_point["archive_stats"] == {
        "conversation_count": 1,
        "conversation_with_limitations_count": 1,
        "conversation_with_limitations_ratio": 1.0,
        "earliest_collected_at": "2026-03-19T01:00:02Z",
        "file_count": 1,
        "latest_collected_at": "2026-03-19T01:00:02Z",
        "message_count": 2,
        "transcript_completeness": {
            "complete": {"count": 0, "ratio": 0.0},
            "partial": {"count": 1, "ratio": 1.0},
            "unsupported": {"count": 0, "ratio": 0.0},
        },
    }

    assert second_codex_point["transition_from_previous"] == {
        "category": "improved",
        "degraded_to_complete": True,
        "from_run_id": "20260319T010000Z",
        "from_status": "partial",
        "label": "partial_to_complete",
        "source": "codex_cli",
        "to_run_id": "20260319T020000Z",
        "to_status": "complete",
    }
    assert second_codex_point["status_ratios"] == {
        "complete": 0.5,
        "degraded": 0.5,
        "failed": 0.0,
        "partial": 0.5,
        "unsupported": 0.0,
    }
    assert second_codex_point["archive_stats"]["conversation_count"] == 2
    assert second_codex_point["archive_stats"]["message_count"] == 3
    assert second_codex_point["archive_stats"]["transcript_completeness"] == {
        "complete": {"count": 2, "ratio": 1.0},
        "partial": {"count": 0, "ratio": 0.0},
        "unsupported": {"count": 0, "ratio": 0.0},
    }

    assert third_codex_point["manifest"]["status"] == "failed"
    assert third_codex_point["manifest"]["failure_reason"] == "RuntimeError: collector exploded"
    assert third_codex_point["status_ratios"] == {
        "complete": 1 / 3,
        "degraded": 1 / 3,
        "failed": 1 / 3,
        "partial": 1 / 3,
        "unsupported": 0.0,
    }
    assert third_codex_point["archive_stats"]["file_count"] == 0
    assert third_codex_point["archive_stats"]["conversation_count"] == 0
    assert third_codex_point["transition_from_previous"]["label"] == "complete_to_failed"

    claude_trend = payload["sources"]["claude_code_cli"]
    assert claude_trend["support_levels"] == ["scaffold", "complete"]
    assert claude_trend["latest_status"] == "complete"
    assert claude_trend["degraded_to_complete_count"] == 1
    assert claude_trend["transition_counts"] == {"unsupported_to_complete": 1}
    assert claude_trend["timeline"][0]["manifest"]["status"] == "unsupported"
    assert claude_trend["timeline"][1]["transition_from_previous"]["label"] == (
        "unsupported_to_complete"
    )
    assert claude_trend["timeline"][2]["transition_from_previous"] is None


def test_runs_trend_filters_to_selected_sources(tmp_path: Path) -> None:
    seed_archive_root(tmp_path)

    exit_code, stdout, stderr = run_cli(
        "runs",
        "trend",
        "--archive-root",
        str(tmp_path),
        "--source",
        "codex_cli",
    )

    assert exit_code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["source_filter"] == ["codex_cli"]
    assert payload["source_count"] == 1
    assert set(payload["sources"]) == {"codex_cli"}
