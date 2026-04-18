from __future__ import annotations

import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from llm_chat_archive import archive_inspect, archive_profile, archive_stats, cli


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
    rows: tuple[dict[str, object], ...],
    filename: str | None = None,
) -> Path:
    output_path = archive_root / source / (
        filename if filename is not None else f"memory_chat_v1-{source}.jsonl"
    )
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
    provenance: dict[str, object] | None = None,
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
    if provenance is not None:
        payload["provenance"] = provenance
    return payload


def seed_archive_root(archive_root: Path) -> None:
    write_archive_output(
        archive_root,
        source="codex_cli",
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
    write_archive_output(
        archive_root,
        source="cursor_editor",
        rows=(
            make_conversation(
                "cursor_editor",
                session="cursor-session-1",
                collected_at="2026-03-19T07:00:00Z",
                transcript_completeness="partial",
                limitations=["missing deleted draft messages"],
                provenance={"originator": "cursor"},
                messages=[
                    {"role": "user", "text": "Need deploy checklist follow-up"},
                    {
                        "role": "assistant",
                        "text": "Checklist updated with rollout notes",
                    },
                ],
            ),
            make_conversation(
                "cursor_editor",
                session="cursor-session-2",
                collected_at="2026-03-19T08:00:00Z",
                messages=[
                    {"role": "user", "text": "Discuss onboarding"},
                    {"role": "assistant", "text": "Summarize the docs"},
                ],
            ),
        ),
    )


def test_archive_index_status_refresh_and_stale_detection(tmp_path: Path) -> None:
    seed_archive_root(tmp_path)

    exit_code, stdout, stderr = run_cli(
        "archive",
        "index",
        "status",
        "--archive-root",
        str(tmp_path),
    )

    assert exit_code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["state"] == "missing"
    assert payload["ready"] is False
    assert payload["stale"] is True
    assert payload["rebuild_required"] is True
    assert payload["file_count"] == 2
    assert payload["indexed_file_count"] == 0

    exit_code, stdout, stderr = run_cli(
        "archive",
        "index",
        "refresh",
        "--archive-root",
        str(tmp_path),
    )

    assert exit_code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["rebuilt"] is True
    assert payload["status"]["state"] == "ready"
    assert payload["status"]["conversation_count"] == 3
    assert payload["status"]["indexed_file_count"] == 2

    write_archive_output(
        tmp_path,
        source="codex_cli",
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
            make_conversation(
                "codex_cli",
                session="codex-session-2",
                collected_at="2026-03-19T09:30:00Z",
                messages=[
                    {"role": "user", "text": "Need archive index refresh"},
                    {"role": "assistant", "text": "Index refresh path validated"},
                ],
            ),
        ),
    )

    exit_code, stdout, stderr = run_cli(
        "archive",
        "index",
        "status",
        "--archive-root",
        str(tmp_path),
    )

    assert exit_code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["state"] == "stale"
    assert payload["ready"] is False
    assert payload["stale"] is True
    assert payload["rebuild_required"] is False
    assert payload["updated_file_count"] == 1
    assert payload["indexed_conversation_count"] == 3


def test_archive_queries_use_index_and_auto_refresh_when_stale(
    tmp_path: Path,
    monkeypatch,
) -> None:
    seed_archive_root(tmp_path)

    exit_code, stdout, stderr = run_cli(
        "archive",
        "index",
        "refresh",
        "--archive-root",
        str(tmp_path),
    )

    assert exit_code == 0
    assert stderr == ""
    assert json.loads(stdout)["status"]["ready"] is True

    def fail_iter_archive_records(*args, **kwargs):
        raise AssertionError("iter_archive_records should not be used for index-backed queries")

    monkeypatch.setattr(archive_inspect, "iter_archive_records", fail_iter_archive_records)
    monkeypatch.setattr(
        archive_stats,
        "iter_archive_records",
        fail_iter_archive_records,
        raising=False,
    )
    monkeypatch.setattr(
        archive_profile,
        "iter_archive_records",
        fail_iter_archive_records,
        raising=False,
    )

    write_archive_output(
        tmp_path,
        source="codex_cli",
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
            make_conversation(
                "codex_cli",
                session="codex-session-2",
                collected_at="2026-03-19T09:30:00Z",
                messages=[
                    {"role": "user", "text": "Need archive index refresh"},
                    {"role": "assistant", "text": "Index refresh path validated"},
                ],
            ),
        ),
    )

    exit_code, stdout, stderr = run_cli(
        "archive",
        "list",
        "--archive-root",
        str(tmp_path),
    )

    assert exit_code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["conversation_count"] == 4
    assert payload["conversations"][0]["source_session_id"] == "codex-session-2"

    exit_code, stdout, stderr = run_cli(
        "archive",
        "find",
        "--archive-root",
        str(tmp_path),
        "--text",
        "index refresh",
    )

    assert exit_code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["conversation_count"] == 1
    assert payload["conversations"][0]["source_session_id"] == "codex-session-2"

    exit_code, stdout, stderr = run_cli(
        "archive",
        "stats",
        "--archive-root",
        str(tmp_path),
    )

    assert exit_code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["conversation_count"] == 4
    assert payload["message_count"] == 8

    exit_code, stdout, stderr = run_cli(
        "archive",
        "profile",
        "--archive-root",
        str(tmp_path),
    )

    assert exit_code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["conversation_count"] == 4
    assert payload["message_roles"]["assistant"]["count"] == 4

    exit_code, stdout, stderr = run_cli(
        "archive",
        "show",
        "--archive-root",
        str(tmp_path),
        "--source",
        "codex_cli",
        "--session",
        "codex-session-2",
    )

    assert exit_code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["summary"]["source_session_id"] == "codex-session-2"
    assert payload["conversation"]["messages"] == [
        {"role": "user", "text": "Need archive index refresh"},
        {"role": "assistant", "text": "Index refresh path validated"},
    ]

    exit_code, stdout, stderr = run_cli(
        "archive",
        "index",
        "status",
        "--archive-root",
        str(tmp_path),
    )

    assert exit_code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["state"] == "ready"
    assert payload["ready"] is True
    assert payload["conversation_count"] == 4
