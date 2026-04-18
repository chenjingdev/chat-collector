from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from llm_chat_archive.config import (
    render_collect_config_template,
    resolve_collect_config,
)
from llm_chat_archive.models import DEFAULT_ARCHIVE_ROOT

REPO_ROOT = Path(__file__).resolve().parents[1]


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "llm_chat_archive", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def test_module_help_succeeds() -> None:
    result = run_cli("--help")

    assert result.returncode == 0
    assert "Collect local coding-agent chats" in result.stdout


def test_collect_rejects_repo_internal_archive_root() -> None:
    result = run_cli(
        "collect",
        "codex_cli",
        "--archive-root",
        str(REPO_ROOT / "tests" / "fixtures" / "archives"),
    )

    assert result.returncode == 2
    assert "outside the repository" in result.stderr


def test_collect_emits_plan_for_external_root(tmp_path: Path) -> None:
    result = run_cli("collect", "gemini", "--archive-root", str(tmp_path))

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["source"] == "gemini"
    assert payload["archive_root"] == str(tmp_path)
    assert payload["implemented"] is True


def test_sample_config_matches_scaffold_template_and_parses() -> None:
    sample_config_path = (REPO_ROOT / "examples" / "collector.sample.toml").resolve(
        strict=False
    )
    assert sample_config_path.read_text(encoding="utf-8") == render_collect_config_template()

    effective_config = resolve_collect_config(
        config_path=sample_config_path,
        cli_archive_root=None,
        cli_profile=None,
        cli_include_sources=None,
        cli_exclude_sources=None,
        cli_incremental=None,
        cli_redaction=None,
        cli_validation=None,
    )

    assert effective_config.to_dict() == {
        "archive_root": str(DEFAULT_ARCHIVE_ROOT),
        "selection_policy": {
            "profile": "default",
            "minimum_support_level": "complete",
            "include_sources": [],
            "exclude_sources": [],
        },
        "execution_policy": {
            "incremental": True,
            "redaction": "on",
            "validation": "report",
        },
        "rerun": {
            "selection_preset": "failed_and_degraded",
            "selection_reason": "failed_or_degraded",
            "source": "config",
        },
        "config_source": "explicit",
        "config_path": str(sample_config_path),
    }


def test_readme_mentions_quickstart_and_config_rules() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert "uv sync" in readme
    assert "One-pass first-run smoke" in readme
    assert "examples/collector.sample.toml" in readme
    assert "uv run llm-chat-archive config init" in readme
    assert "uv run llm-chat-archive config init --archive-root /absolute/path/to/chat-history" in readme
    assert "uv run llm-chat-archive config init --print" in readme
    assert "~/.config/llm-chat-archive/collector.toml" in readme
    assert "/Users/chenjing/dev/chat-history" in readme
    assert "/absolute/path/to/chat-history" in readme
    assert "uv run llm-chat-archive sources" in readme
    assert "uv run llm-chat-archive sources --format markdown" in readme
    assert "docs/source-support-matrix.md" in readme
    assert "uv run llm-chat-archive doctor --all --profile default" in readme
    assert "uv run llm-chat-archive collect --all" in readme
    assert "uv run llm-chat-archive scheduled run" in readme
    assert "uv run llm-chat-archive acceptance ship" in readme
    assert "--force-unlock-stale" in readme
    assert "uv run llm-chat-archive runs latest" in readme
    assert "uv run llm-chat-archive runs show" in readme
    assert "uv run llm-chat-archive tui" in readme
    assert "docs/operator-terminal-tui.md" in readme
    assert "docs/ship-acceptance.md" in readme
    assert "uv run llm-chat-archive archive list" in readme
    assert "uv run llm-chat-archive archive show" in readme
    assert "uv run llm-chat-archive archive find" in readme
    assert "uv run llm-chat-archive archive export-memory" in readme
    assert "archive root must be an absolute path outside the repository" in readme
    assert "`redaction`" in readme
    assert "`validation`" in readme
    assert "`incremental`" in readme
    assert "`rerun`" in readme
    assert "`scheduled run`" in readme
