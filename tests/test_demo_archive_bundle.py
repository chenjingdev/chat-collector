from __future__ import annotations

import io
import json
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

from llm_chat_archive import archive_digest, cli

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_BUNDLE_DIR = REPO_ROOT / "examples" / "demo-archive-bundle"


def run_cli(*args: str) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        exit_code = cli.main(list(args))
    return exit_code, stdout.getvalue(), stderr.getvalue()


def test_demo_archive_bundle_supports_end_to_end_operator_walkthrough(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        archive_digest,
        "_utcnow",
        lambda: datetime(2026, 3, 19, 10, 45, tzinfo=timezone.utc),
    )

    exit_code, stdout, stderr = run_cli(
        "archive",
        "import",
        "--archive-root",
        str(tmp_path),
        "--bundle-dir",
        str(DEMO_BUNDLE_DIR),
    )

    assert exit_code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["write_mode"] == "dry_run"
    assert payload["source_count"] == 3
    assert payload["conversation_count"] == 3
    assert payload["message_count"] == 6
    assert payload["before_conversation_count"] == 0
    assert payload["after_conversation_count"] == 3
    assert payload["imported_count"] == 3
    assert payload["skipped_count"] == 0
    assert payload["upgraded_count"] == 0

    exit_code, stdout, stderr = run_cli(
        "archive",
        "import",
        "--archive-root",
        str(tmp_path),
        "--bundle-dir",
        str(DEMO_BUNDLE_DIR),
        "--execute",
    )

    assert exit_code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["write_mode"] == "write"
    assert payload["sources"]["codex_cli"]["output_path"] == str(
        tmp_path / "codex_cli" / "memory_chat_v1-codex_cli.jsonl"
    )
    assert payload["sources"]["cursor_editor"]["output_path"] == str(
        tmp_path / "cursor_editor" / "memory_chat_v1-cursor_editor.jsonl"
    )
    assert payload["sources"]["gemini_code_assist_ide"]["output_path"] == str(
        tmp_path
        / "gemini_code_assist_ide"
        / "memory_chat_v1-gemini_code_assist_ide.jsonl"
    )

    exit_code, stdout, stderr = run_cli(
        "archive",
        "export",
        "--archive-root",
        str(tmp_path),
        "--output-dir",
        str(DEMO_BUNDLE_DIR),
        "--source",
        "codex_cli",
    )

    assert exit_code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["write_mode"] == "dry_run"
    assert payload["output_dir"] == str(DEMO_BUNDLE_DIR)
    assert payload["source_count"] == 1

    exit_code, stdout, stderr = run_cli(
        "archive",
        "list",
        "--archive-root",
        str(tmp_path),
    )

    assert exit_code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["conversation_count"] == 3
    assert [
        conversation["source_session_id"] for conversation in payload["conversations"]
    ] == [
        "demo-gemini-unsupported",
        "demo-cursor-partial",
        "demo-codex-complete",
    ]
    assert [
        conversation["transcript_completeness"]
        for conversation in payload["conversations"]
    ] == ["unsupported", "partial", "complete"]

    exit_code, stdout, stderr = run_cli(
        "archive",
        "show",
        "--archive-root",
        str(tmp_path),
        "--source",
        "cursor_editor",
        "--session",
        "demo-cursor-partial",
    )

    assert exit_code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["summary"]["source"] == "cursor_editor"
    assert payload["summary"]["source_session_id"] == "demo-cursor-partial"
    assert payload["conversation"]["transcript_completeness"] == "partial"
    assert payload["conversation"]["limitations"] == [
        "deleted draft messages unavailable"
    ]
    assert payload["conversation"]["messages"] == [
        {
            "role": "user",
            "text": "Update the release checklist with rollback notes.",
        },
        {
            "role": "assistant",
            "text": "Rollback notes added; hidden drafts are still unavailable.",
        },
    ]

    exit_code, stdout, stderr = run_cli(
        "archive",
        "find",
        "--archive-root",
        str(tmp_path),
        "--text",
        "rollback",
    )

    assert exit_code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["conversation_count"] == 1
    assert payload["conversations"][0]["source"] == "cursor_editor"
    assert payload["conversations"][0]["source_session_id"] == "demo-cursor-partial"

    exit_code, stdout, stderr = run_cli(
        "archive",
        "stats",
        "--archive-root",
        str(tmp_path),
    )

    assert exit_code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["source_count"] == 3
    assert payload["conversation_count"] == 3
    assert payload["message_count"] == 6
    assert payload["conversation_with_limitations_count"] == 2
    assert payload["transcript_completeness"] == {
        "complete": {"count": 1, "ratio": 1 / 3},
        "partial": {"count": 1, "ratio": 1 / 3},
        "unsupported": {"count": 1, "ratio": 1 / 3},
    }

    exit_code, stdout, stderr = run_cli(
        "archive",
        "profile",
        "--archive-root",
        str(tmp_path),
    )

    assert exit_code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["source_count"] == 3
    assert payload["conversation_count"] == 3
    assert payload["message_count"] == 6
    assert payload["conversation_with_limitations_count"] == 2
    assert payload["message_roles"]["assistant"]["count"] == 3
    assert payload["message_roles"]["developer"]["count"] == 1
    assert payload["message_roles"]["user"]["count"] == 2
    assert payload["limitations"] == {
        "deleted draft messages unavailable": {
            "count": 1,
            "ratio": 1 / 3,
        },
        "editor session transcript unavailable": {
            "count": 1,
            "ratio": 1 / 3,
        },
        "metadata-only recovery": {
            "count": 1,
            "ratio": 1 / 3,
        },
    }

    exit_code, stdout, stderr = run_cli(
        "archive",
        "digest",
        "--archive-root",
        str(tmp_path),
    )

    assert exit_code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["aggregated_at"] == "2026-03-19T10:45:00Z"
    assert payload["status"] == "warning"
    assert payload["latest_run_id"] is None
    assert payload["latest_run"] is None
    assert payload["overview"]["source_count"] == 3
    assert payload["overview"]["conversation_count"] == 3
    assert payload["overview"]["message_count"] == 6
    assert payload["overview"]["warning_count"] == 2
    assert payload["overview"]["error_count"] == 0
    assert payload["sources"]["codex_cli"]["attention_required"] is False
    assert payload["sources"]["cursor_editor"]["attention_required"] is True
    assert payload["sources"]["gemini_code_assist_ide"]["attention_required"] is True

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
    assert payload["source_count"] == 3
    assert payload["file_count"] == 3
    assert payload["row_count"] == 3
    assert payload["verified_row_count"] == 3
    assert payload["bad_row_count"] == 0
    assert payload["orphan_file_count"] == 0
    assert payload["warning_count"] == 2
    assert payload["error_count"] == 0


def test_demo_bundle_docs_are_operator_facing_and_separate_from_test_fixtures() -> None:
    walkthrough = (REPO_ROOT / "docs" / "demo-archive-walkthrough.md").read_text(
        encoding="utf-8"
    )
    bundle_readme = (DEMO_BUNDLE_DIR / "README.md").read_text(encoding="utf-8")

    assert "archive import" in walkthrough
    assert "archive list" in walkthrough
    assert "archive show" in walkthrough
    assert "archive find" in walkthrough
    assert "archive stats" in walkthrough
    assert "archive profile" in walkthrough
    assert "archive digest" in walkthrough
    assert "archive verify" in walkthrough

    assert "tests/fixtures/" in bundle_readme
    assert "operator walkthroughs and local demos" in bundle_readme
