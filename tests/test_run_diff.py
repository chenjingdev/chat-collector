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
    selection_policy: dict[str, object] | None = None,
    selected_sources: tuple[str, ...] = (),
    excluded_sources: tuple[dict[str, object], ...] = (),
) -> Path:
    run_dir = archive_root / RUNS_DIRECTORY / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / MANIFEST_FILENAME
    payload: dict[str, object] = {
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
        "redaction_event_count": sum(
            int(source.get("redaction_event_count", 0)) for source in sources
        ),
        "sources": list(sources),
    }
    if selection_policy is not None:
        payload["selection_policy"] = selection_policy
        payload["selected_sources"] = list(selected_sources)
        payload["excluded_sources"] = list(excluded_sources)
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
    skipped_conversation_count: int = 0,
    written_conversation_count: int | None = None,
    redaction_event_count: int = 0,
    partial: bool = False,
    unsupported: bool = False,
    failed: bool = False,
    failure_reason: str | None = None,
    create_output: bool = True,
    output_filename: str | None = None,
) -> dict[str, object]:
    if written_conversation_count is None:
        written_conversation_count = conversation_count

    output_path: Path | None = None
    if create_output:
        output_dir = archive_root / source
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = output_filename or f"memory_chat_v1-{source}.jsonl"
        output_path = output_dir / filename
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
        "redaction_event_count": redaction_event_count,
        "partial": partial,
        "unsupported": unsupported,
        "failed": failed,
    }
    if failure_reason is not None:
        payload["failure_reason"] = failure_reason
    return payload


def test_runs_diff_reports_status_deltas_and_selection_changes(tmp_path: Path) -> None:
    write_run_manifest(
        tmp_path,
        run_id="20260319T010000Z",
        started_at="2026-03-19T01:00:00Z",
        completed_at="2026-03-19T01:00:05Z",
        selection_policy={
            "profile": "default",
            "minimum_support_level": "complete",
            "include_sources": ["codex_cli", "claude_code_cli", "cursor_editor"],
            "exclude_sources": [],
        },
        selected_sources=("codex_cli", "claude_code_cli", "cursor_editor"),
        excluded_sources=(
            {
                "source": "gemini_cli",
                "support_level": "partial",
                "reason": "not included by --source allowlist",
            },
        ),
        sources=(
            make_source_entry(
                tmp_path,
                source="codex_cli",
                support_level="complete",
                status="failed",
                scanned_artifact_count=3,
                conversation_count=0,
                message_count=0,
                failed=True,
                failure_reason="RuntimeError: collector exploded",
                create_output=False,
            ),
            make_source_entry(
                tmp_path,
                source="claude_code_cli",
                support_level="partial",
                status="partial",
                scanned_artifact_count=4,
                conversation_count=2,
                message_count=6,
                partial=True,
                output_filename="memory_chat_v1-claude_code_cli-old.jsonl",
            ),
            make_source_entry(
                tmp_path,
                source="cursor_editor",
                support_level="scaffold",
                status="unsupported",
                scanned_artifact_count=0,
                conversation_count=0,
                message_count=0,
                unsupported=True,
                create_output=False,
            ),
        ),
    )
    write_run_manifest(
        tmp_path,
        run_id="20260319T020000Z",
        started_at="2026-03-19T02:00:00Z",
        completed_at="2026-03-19T02:00:07Z",
        selection_policy={
            "profile": "complete_only",
            "minimum_support_level": "complete",
            "include_sources": ["codex_cli", "claude_code_cli", "gemini_cli"],
            "exclude_sources": [],
        },
        selected_sources=("codex_cli", "claude_code_cli", "gemini_cli"),
        excluded_sources=(
            {
                "source": "cursor_editor",
                "support_level": "scaffold",
                "reason": "support level 'scaffold' is below minimum 'complete'",
            },
        ),
        sources=(
            make_source_entry(
                tmp_path,
                source="codex_cli",
                support_level="complete",
                status="complete",
                scanned_artifact_count=4,
                conversation_count=2,
                message_count=5,
                redaction_event_count=1,
            ),
            make_source_entry(
                tmp_path,
                source="claude_code_cli",
                support_level="complete",
                status="complete",
                scanned_artifact_count=5,
                conversation_count=3,
                message_count=9,
                skipped_conversation_count=1,
                written_conversation_count=2,
                output_filename="memory_chat_v1-claude_code_cli-new.jsonl",
            ),
            make_source_entry(
                tmp_path,
                source="gemini_cli",
                support_level="complete",
                status="complete",
                scanned_artifact_count=1,
                conversation_count=1,
                message_count=2,
            ),
        ),
    )

    exit_code, stdout, stderr = run_cli(
        "runs",
        "diff",
        "--archive-root",
        str(tmp_path),
        "--from",
        "20260319T010000Z",
        "--to",
        "20260319T020000Z",
    )

    assert exit_code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["comparison_mode"] == "explicit"
    assert payload["from_run"]["run_id"] == "20260319T010000Z"
    assert payload["to_run"]["run_id"] == "20260319T020000Z"
    assert payload["counts"]["conversation_count"] == {"from": 2, "to": 6, "delta": 4}
    assert payload["counts"]["message_count"] == {"from": 6, "to": 16, "delta": 10}
    assert payload["counts"]["written_conversation_count"] == {
        "from": 2,
        "to": 5,
        "delta": 3,
    }
    assert payload["counts"]["redaction_event_count"] == {"from": 0, "to": 1, "delta": 1}
    assert payload["new_sources"] == ["gemini_cli"]
    assert payload["removed_sources"] == ["cursor_editor"]

    selection_policy = payload["selection_policy"]
    assert selection_policy["changed"] is True
    assert selection_policy["selected_sources_added"] == ["gemini_cli"]
    assert selection_policy["selected_sources_removed"] == ["cursor_editor"]
    assert selection_policy["excluded_sources_added"][0]["source"] == "cursor_editor"
    assert selection_policy["excluded_sources_removed"][0]["source"] == "gemini_cli"

    sources = {entry["source"]: entry for entry in payload["sources"]}
    assert sources["codex_cli"]["status"] == {
        "from": "failed",
        "to": "complete",
        "changed": True,
    }
    assert sources["codex_cli"]["failure_reason"]["from"] == "RuntimeError: collector exploded"
    assert sources["codex_cli"]["failure_reason"]["to"] is None
    assert sources["codex_cli"]["counts"]["conversation_count"] == {
        "from": 0,
        "to": 2,
        "delta": 2,
    }
    assert sources["codex_cli"]["important_transition"] == {
        "source": "codex_cli",
        "from_status": "failed",
        "to_status": "complete",
        "label": "failed_to_complete",
        "category": "improved",
    }
    assert sources["claude_code_cli"]["status"]["from"] == "partial"
    assert sources["claude_code_cli"]["status"]["to"] == "complete"
    assert sources["claude_code_cli"]["output_path"]["changed"] is True
    assert sources["claude_code_cli"]["counts"]["skipped_conversation_count"] == {
        "from": 0,
        "to": 1,
        "delta": 1,
    }
    assert sources["cursor_editor"]["change_type"] == "removed"
    assert sources["gemini_cli"]["change_type"] == "added"

    transitions = {entry["label"] for entry in payload["important_transitions"]}
    assert transitions == {"failed_to_complete", "partial_to_complete"}


def test_runs_diff_defaults_to_latest_vs_previous(tmp_path: Path) -> None:
    for run_id, conversation_count in (
        ("20260319T010000Z", 1),
        ("20260319T020000Z", 2),
        ("20260319T030000Z", 3),
    ):
        write_run_manifest(
            tmp_path,
            run_id=run_id,
            started_at=f"2026-03-19T{run_id[9:11]}:00:00Z",
            completed_at=f"2026-03-19T{run_id[9:11]}:00:05Z",
            sources=(
                make_source_entry(
                    tmp_path,
                    source="codex_cli",
                    support_level="complete",
                    status="complete",
                    scanned_artifact_count=conversation_count,
                    conversation_count=conversation_count,
                    message_count=conversation_count * 2,
                ),
            ),
        )

    exit_code, stdout, stderr = run_cli(
        "runs",
        "diff",
        "--archive-root",
        str(tmp_path),
    )

    assert exit_code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["comparison_mode"] == "latest_vs_previous"
    assert payload["from_run"]["run_id"] == "20260319T020000Z"
    assert payload["to_run"]["run_id"] == "20260319T030000Z"
    assert payload["counts"]["conversation_count"] == {"from": 2, "to": 3, "delta": 1}


def test_runs_diff_requires_two_runs_for_latest_vs_previous(tmp_path: Path) -> None:
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
                scanned_artifact_count=1,
                conversation_count=1,
                message_count=2,
            ),
        ),
    )

    exit_code, stdout, stderr = run_cli(
        "runs",
        "diff",
        "--archive-root",
        str(tmp_path),
    )

    assert exit_code == 1
    assert stdout == ""
    assert "requires at least two run manifests" in stderr


def test_runs_diff_requires_both_from_and_to(tmp_path: Path) -> None:
    exit_code, stdout, stderr = run_cli(
        "runs",
        "diff",
        "--archive-root",
        str(tmp_path),
        "--from",
        "20260319T010000Z",
    )

    assert exit_code == 2
    assert stdout == ""
    assert "provide both --from and --to" in stderr
