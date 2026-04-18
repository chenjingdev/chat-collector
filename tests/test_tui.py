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
    *,
    archive_root: Path,
    source: str,
    support_level: str,
    status: str,
    scanned_artifact_count: int,
    conversation_count: int,
    message_count: int,
    output_path: Path | None,
    partial: bool = False,
    unsupported: bool = False,
    failed: bool = False,
) -> dict[str, object]:
    return {
        "source": source,
        "support_level": support_level,
        "status": status,
        "archive_root": str(archive_root),
        "output_path": None if output_path is None else str(output_path),
        "input_roots": [],
        "scanned_artifact_count": scanned_artifact_count,
        "conversation_count": conversation_count,
        "skipped_conversation_count": 0,
        "written_conversation_count": conversation_count if output_path is not None else 0,
        "message_count": message_count,
        "partial": partial,
        "unsupported": unsupported,
        "failed": failed,
    }


def seed_tui_archive(archive_root: Path) -> None:
    codex_output = write_archive_output(
        archive_root,
        source="codex_cli",
        filename="memory_chat_v1-20260319T060000-codex_cli.jsonl",
        rows=(
            make_conversation(
                "codex_cli",
                session="codex-session-1",
                collected_at="2026-03-19T06:00:00Z",
                messages=[
                    {"role": "user", "text": "Need deploy checklist"},
                    {"role": "assistant", "text": "Start with release notes"},
                ],
            ),
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
                limitations=["deleted draft messages unavailable"],
                messages=[
                    {"role": "user", "text": "Need deploy checklist follow-up"},
                    {"role": "assistant", "text": "Checklist updated"},
                ],
            ),
            make_conversation(
                "cursor_editor",
                session="cursor-session-2",
                collected_at="2026-03-19T08:00:00Z",
                messages=[
                    {"role": "user", "text": "Need archive index refresh"},
                    {"role": "assistant", "text": "Index refresh path validated"},
                ],
            ),
        ),
    )

    write_run_manifest(
        archive_root,
        run_id="20260319T080000Z",
        started_at="2026-03-19T08:00:00Z",
        completed_at="2026-03-19T08:00:04Z",
        sources=(
            make_source_entry(
                archive_root=archive_root,
                source="codex_cli",
                support_level="complete",
                status="complete",
                scanned_artifact_count=1,
                conversation_count=1,
                message_count=2,
                output_path=codex_output,
            ),
        ),
    )
    write_run_manifest(
        archive_root,
        run_id="20260319T090000Z",
        started_at="2026-03-19T09:00:00Z",
        completed_at="2026-03-19T09:00:12Z",
        sources=(
            make_source_entry(
                archive_root=archive_root,
                source="codex_cli",
                support_level="complete",
                status="complete",
                scanned_artifact_count=1,
                conversation_count=1,
                message_count=2,
                output_path=codex_output,
            ),
            make_source_entry(
                archive_root=archive_root,
                source="cursor_editor",
                support_level="partial",
                status="partial",
                scanned_artifact_count=2,
                conversation_count=2,
                message_count=4,
                output_path=cursor_output,
                partial=True,
            ),
            make_source_entry(
                archive_root=archive_root,
                source="gemini_cli",
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


def test_tui_overview_snapshot_renders_latest_run_and_source_health(
    tmp_path: Path,
) -> None:
    seed_tui_archive(tmp_path)

    exit_code, stdout, stderr = run_cli(
        "tui",
        "--archive-root",
        str(tmp_path),
        "--snapshot",
        "--view",
        "overview",
        "--width",
        "88",
    )

    assert exit_code == 0
    assert stderr == ""
    assert "llm-chat-archive operator triage [overview]" in stdout
    assert "Latest run: 20260319T090000Z" in stdout
    assert "Degraded sources: cursor_editor, gemini_cli" in stdout
    assert "Digest: status=warning" in stdout
    assert "> cursor_editor [degraded" in stdout
    assert "codex_cli [complete]" in stdout


def test_tui_samples_snapshot_renders_selected_conversation_drill_down(
    tmp_path: Path,
) -> None:
    seed_tui_archive(tmp_path)

    exit_code, stdout, stderr = run_cli(
        "tui",
        "--archive-root",
        str(tmp_path),
        "--snapshot",
        "--view",
        "samples",
        "--source",
        "cursor_editor",
        "--session",
        "cursor-session-1",
        "--width",
        "88",
    )

    assert exit_code == 0
    assert stderr == ""
    assert "llm-chat-archive operator triage [samples]" in stdout
    assert "Selection: source=cursor_editor seed=operator-triage count=5" in stdout
    assert "> cursor-session-1 partial" in stdout
    assert "Summary: session=cursor-session-1" in stdout
    assert "Need deploy checklist follow-up" in stdout
    assert "Checklist updated" in stdout
