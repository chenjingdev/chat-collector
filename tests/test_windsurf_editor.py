from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from llm_chat_archive.sources.windsurf_editor import (
    WindsurfEditorCollector,
    discover_windsurf_editor_artifacts,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "windsurf_editor"


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "llm_chat_archive", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_discover_windsurf_editor_artifacts_indexes_memories_rules_and_workspace_roots() -> None:
    artifacts = discover_windsurf_editor_artifacts((FIXTURE_ROOT,))

    assert artifacts.global_root_paths == (
        str(FIXTURE_ROOT / ".codeium" / "windsurf"),
    )
    assert artifacts.application_support_roots == (
        str(FIXTURE_ROOT / "Library" / "Application Support" / "Windsurf"),
    )
    assert artifacts.global_memory_paths == (
        str(
            FIXTURE_ROOT
            / ".codeium"
            / "windsurf"
            / "memories"
            / "workspace-alpha"
            / "build-preferences.md"
        ),
    )
    assert artifacts.global_rule_paths == (
        str(FIXTURE_ROOT / ".codeium" / "windsurf" / "memories" / "global_rules.md"),
    )
    assert artifacts.system_rule_paths == (
        str(
            FIXTURE_ROOT
            / "Library"
            / "Application Support"
            / "Windsurf"
            / "rules"
            / "security.md"
        ),
    )
    assert artifacts.workspace_rule_paths == (
        str(FIXTURE_ROOT / "repo-alpha" / ".windsurf" / "rules" / "always-on.md"),
        str(FIXTURE_ROOT / "repo-alpha" / ".windsurf" / "rules" / "tests.md"),
    )
    assert artifacts.workspace_root_paths == (
        str(FIXTURE_ROOT / "repo-alpha"),
        str(FIXTURE_ROOT / "repo-beta"),
    )
    assert artifacts.global_skill_root_paths == (
        str(FIXTURE_ROOT / ".codeium" / "windsurf" / "skills"),
    )
    assert artifacts.workspace_skill_root_paths == (
        str(FIXTURE_ROOT / "repo-alpha" / ".windsurf" / "skills"),
        str(FIXTURE_ROOT / "repo-beta" / ".windsurf" / "skills"),
    )
    assert artifacts.mcp_config_paths == (
        str(FIXTURE_ROOT / ".codeium" / "windsurf" / "mcp_config.json"),
    )
    assert artifacts.mcp_server_names == ("filesystem",)


def test_windsurf_editor_collect_writes_partial_rows_and_workspace_metadata_fallback(
    tmp_path: Path,
) -> None:
    collector = WindsurfEditorCollector()

    result = collector.collect(tmp_path, input_roots=(FIXTURE_ROOT,))

    assert result.source == "windsurf_editor"
    assert result.scanned_artifact_count == 6
    assert result.conversation_count == 6
    assert result.message_count == 5

    rows = read_jsonl(result.output_path)
    assert len(rows) == 6
    by_session_id = {row["source_session_id"]: row for row in rows}
    assert set(by_session_id) == {
        "global:rule:global_rules",
        "memory:workspace-alpha/build-preferences",
        "system:rule:security",
        "workspace:repo-alpha:rule:always-on",
        "workspace:repo-alpha:rule:tests",
        "workspace:repo-beta:metadata",
    }

    memory_row = by_session_id["memory:workspace-alpha/build-preferences"]
    assert memory_row["messages"] == [
        {
            "role": "system",
            "text": (
                "Prefer `uv run` over direct `python`.\n\nCollected archives must stay "
                "under `/Users/chenjing/dev/chat-history`."
            ),
        }
    ]
    assert memory_row["transcript_completeness"] == "partial"
    assert memory_row["limitations"] == [
        "memory_entry_not_original_conversation_transcript",
        "no_confirmed_windsurf_editor_session_history",
    ]
    assert memory_row["session_metadata"] == {
        "scope": "memory",
        "memory_scope": "workspace",
        "memory_key": "workspace-alpha/build-preferences",
        "relative_path": "workspace-alpha/build-preferences.md",
        "content_format": "text",
        "workspace_key": "workspace-alpha",
        "mcp_server_count": 1,
        "mcp_server_names": ["filesystem"],
        "global_skill_file_count": 1,
    }

    global_rule_row = by_session_id["global:rule:global_rules"]
    assert global_rule_row["messages"] == [
        {
            "role": "developer",
            "text": (
                "# Global Windsurf Rules\n\n- Use `uv run` for Python commands.\n"
                "- Keep real archive output outside the repository checkout."
            ),
        }
    ]
    assert global_rule_row["transcript_completeness"] == "partial"
    assert global_rule_row["session_metadata"] == {
        "scope": "rule",
        "rule_scope": "global",
        "rule_name": "global_rules",
        "relative_path": "memories/global_rules.md",
        "activation_mode": "always_on",
        "mcp_server_count": 1,
        "mcp_server_names": ["filesystem"],
        "global_skill_file_count": 1,
    }

    workspace_rule_row = by_session_id["workspace:repo-alpha:rule:tests"]
    assert workspace_rule_row["messages"] == [
        {
            "role": "developer",
            "text": (
                "Test files should prefer exact archive-shape assertions over loose "
                "substring checks."
            ),
        }
    ]
    assert workspace_rule_row["transcript_completeness"] == "partial"
    assert workspace_rule_row["session_metadata"] == {
        "scope": "rule",
        "rule_scope": "workspace",
        "rule_name": "tests",
        "workspace_root": str(FIXTURE_ROOT / "repo-alpha"),
        "workspace_label": "repo-alpha",
        "relative_path": ".windsurf/rules/tests.md",
        "activation_mode": "glob",
        "workspace_skill_dir_count": 1,
        "workspace_skill_file_count": 1,
        "frontmatter": {
            "trigger": "glob",
            "globs": ["tests/**/*.py"],
            "description": "Apply stricter assertions in test files.",
        },
    }
    assert workspace_rule_row["provenance"]["cwd"] == str(FIXTURE_ROOT / "repo-alpha")

    system_rule_row = by_session_id["system:rule:security"]
    assert system_rule_row["messages"] == [
        {
            "role": "system",
            "text": (
                "# System Security Rules\n\n- Never write real collected archives "
                "inside the repository.\n- Treat secrets as redaction targets before "
                "export."
            ),
        }
    ]
    assert system_rule_row["transcript_completeness"] == "partial"
    assert system_rule_row["session_metadata"] == {
        "scope": "rule",
        "rule_scope": "system",
        "rule_name": "security",
        "relative_path": "rules/security.md",
        "activation_mode": "always_on",
    }

    workspace_metadata_row = by_session_id["workspace:repo-beta:metadata"]
    assert workspace_metadata_row["messages"] == []
    assert workspace_metadata_row["transcript_completeness"] == "unsupported"
    assert workspace_metadata_row["limitations"] == [
        "workspace_metadata_only",
        "no_confirmed_windsurf_editor_session_history",
    ]
    assert workspace_metadata_row["session_metadata"] == {
        "scope": "workspace_metadata",
        "workspace_root": str(FIXTURE_ROOT / "repo-beta"),
        "workspace_label": "repo-beta",
        "workspace_rule_count": 0,
        "workspace_skill_dir_count": 1,
        "workspace_skill_file_count": 1,
    }
    assert workspace_metadata_row["provenance"]["cwd"] == str(FIXTURE_ROOT / "repo-beta")

    serialized = json.dumps(rows, ensure_ascii=False)
    assert "server.js" not in serialized
    assert "global:metadata" not in serialized


def test_windsurf_editor_collect_emits_global_metadata_only_row_for_config_only_root(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / ".codeium" / "windsurf"
    input_root.mkdir(parents=True)
    (input_root / "skills" / "triage").mkdir(parents=True)
    (input_root / "skills" / "triage" / "README.md").write_text(
        "Summarize recurring failures.",
        encoding="utf-8",
    )
    (input_root / "mcp_config.json").write_text(
        json.dumps({"mcpServers": {"github": {"command": "github-mcp"}}}),
        encoding="utf-8",
    )

    collector = WindsurfEditorCollector()
    result = collector.collect(tmp_path / "archive", input_roots=(input_root,))

    assert result.scanned_artifact_count == 1
    assert result.conversation_count == 1
    assert result.message_count == 0
    rows = read_jsonl(result.output_path)
    assert rows == [
        {
            "source": "windsurf_editor",
            "execution_context": "ide_native",
            "collected_at": rows[0]["collected_at"],
            "messages": [],
            "contract": rows[0]["contract"],
            "transcript_completeness": "unsupported",
            "limitations": [
                "global_metadata_only",
                "no_confirmed_windsurf_editor_session_history",
            ],
            "source_session_id": "global:metadata",
            "source_artifact_path": str(input_root / "mcp_config.json"),
            "session_metadata": {
                "scope": "global_metadata",
                "mcp_server_count": 1,
                "mcp_server_names": ["github"],
                "global_skill_dir_count": 1,
                "global_skill_file_count": 1,
                "global_memory_file_count": 0,
                "global_rule_count": 0,
            },
            "provenance": {
                "source": "windsurf",
                "originator": "windsurf_editor",
                "app_shell": {
                    "auxiliary_paths": [
                        str(input_root),
                        str(input_root / "mcp_config.json"),
                        str(input_root / "skills"),
                    ]
                },
            },
        }
    ]


def test_cli_collect_windsurf_editor_plan_and_execute(tmp_path: Path) -> None:
    plan_result = run_cli("collect", "windsurf_editor", "--archive-root", str(tmp_path))

    assert plan_result.returncode == 0
    plan_payload = json.loads(plan_result.stdout)
    assert plan_payload["source"] == "windsurf_editor"
    assert plan_payload["implemented"] is True
    assert plan_payload["support_level"] == "partial"

    execute_result = run_cli(
        "collect",
        "windsurf_editor",
        "--archive-root",
        str(tmp_path),
        "--input-root",
        str(FIXTURE_ROOT),
        "--execute",
    )

    assert execute_result.returncode == 0
    execute_payload = json.loads(execute_result.stdout)
    assert execute_payload["source"] == "windsurf_editor"
    assert execute_payload["scanned_artifact_count"] == 6
    assert execute_payload["conversation_count"] == 6
    rows = read_jsonl(Path(execute_payload["output_path"]))
    assert len(rows) == 6
