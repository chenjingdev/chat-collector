from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "llm_chat_archive", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def test_sources_json_emits_support_matrix_summary() -> None:
    result = run_cli("sources", "--format", "json")

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["default_batch_profile"] == "default"

    entries = {entry["source"]: entry for entry in payload["sources"]}
    assert entries["codex_cli"] == {
        "source": "codex_cli",
        "display_name": "Codex CLI",
        "product_label": "Codex",
        "host_surface": "CLI",
        "support_level": "complete",
        "expected_transcript_completeness": "complete",
        "limitation_summary": (
            "Filters event, reasoning, tool, and search noise out of the transcript."
        ),
        "included_in_default_batch_profile": True,
        "default_input_roots": ["~/.codex"],
        "notes": [
            "Scans ~/.codex/sessions/**/rollout-*.jsonl and ~/.codex/archived_sessions/rollout-*.jsonl.",
            "Keeps response_item message rows for developer, user, and assistant roles only.",
            "Excludes event, reasoning, tool, search, and turn-context noise from normalized output.",
        ],
    }
    assert entries["cursor"][
        "limitation_summary"
    ] == (
        "Cursor CLI still depends on shared editor transcript rows plus unique "
        "invocation attribution, so it remains partial and opt-in for unattended "
        "batches."
    )
    assert entries["cursor"]["included_in_default_batch_profile"] is False
    assert entries["cursor_editor"]["included_in_default_batch_profile"] is False
    assert entries["gemini_code_assist_ide"]["included_in_default_batch_profile"] is False
    assert entries["antigravity_editor_view"]["included_in_default_batch_profile"] is False
    assert entries["windsurf_editor"]["included_in_default_batch_profile"] is False
    assert entries["codex_app"]["limitation_summary"] == (
        "Desktop automation run state comes from shared SQLite metadata, and "
        "archived automation rollout gaps stay rollout-first while repaired "
        "bodies are tagged with explicit fallback provenance."
    )
    assert entries["codex_app"]["limitations"] == [
        "Automation-origin attribution depends on local sqlite/codex-dev.db automation_runs rows being present, with state_5.sqlite thread metadata used as a secondary join signal.",
        "Archived automation conversations keep rollout bodies canonical and tag any repaired user or assistant body with message-level provenance from archived snapshot or thread metadata.",
    ]
    assert (
        entries["gemini_code_assist_ide"]["expected_transcript_completeness"]
        == "partial"
    )
    assert entries["windsurf_editor"]["expected_transcript_completeness"] == "partial"
    assert entries["windsurf_editor"]["limitation_summary"] == (
        "Windsurf local memories and rules can be normalized, but no confirmed "
        "native editor session-history store is available yet, so the "
        "collector remains partial and opt-in."
    )
    assert entries["windsurf_editor"]["limitations"] == [
        "Memories and rules are captured as partial context rows because they are not original turn-by-turn Cascade transcripts.",
        "mcp_config.json, skills directories, and bare workspace metadata degrade to unsupported rows until a confirmed session-history store is observed.",
    ]
    assert entries["gemini_code_assist_ide"]["limitations"] == [
        "Only explicitly Gemini-owned chatSessions with recoverable request or response bodies are promoted to transcript rows.",
        "Foreign or unknown provider chatSessions, and Gemini-owned sessions without a confirmed body, remain metadata-only residue with explicit diagnostics.",
    ]


def test_sources_markdown_matches_committed_support_doc() -> None:
    result = run_cli("sources", "--format", "markdown")

    assert result.returncode == 0
    expected = (REPO_ROOT / "docs" / "source-support-matrix.md").read_text(
        encoding="utf-8"
    )
    assert result.stdout == expected
