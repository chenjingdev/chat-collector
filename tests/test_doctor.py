from __future__ import annotations

import io
import json
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

from llm_chat_archive import cli
from llm_chat_archive.models import CollectionPlan, SourceDescriptor, SupportLevel
from llm_chat_archive.registry import CollectorRegistry
from llm_chat_archive.source_roots import normalize_source_root_platform

REPO_ROOT = Path(__file__).resolve().parents[1]
CURSOR_EDITOR_FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "cursor_editor"
WINDSURF_EDITOR_FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "windsurf_editor"
CURRENT_PLATFORM = normalize_source_root_platform().value


def run_cli(*args: str, registry: CollectorRegistry | None = None) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        if registry is None:
            exit_code = cli.main(args)
        else:
            with patch("llm_chat_archive.cli.build_registry", return_value=registry):
                exit_code = cli.main(args)
    return exit_code, stdout.getvalue(), stderr.getvalue()


def write_claude_transcript(root: Path, session_id: str = "11111111-1111-4111-8111-111111111111") -> Path:
    transcript_path = root / "projects" / "project-alpha" / f"{session_id}.jsonl"
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    rows = (
        {
            "type": "user",
            "timestamp": "2026-03-19T00:00:01Z",
            "uuid": "user-1",
            "message": {
                "id": "user-1",
                "role": "user",
                "content": "doctor readiness check",
            },
        },
        {
            "type": "assistant",
            "timestamp": "2026-03-19T00:00:02Z",
            "uuid": "assistant-1",
            "message": {
                "id": "assistant-1",
                "role": "assistant",
                "content": [{"type": "text", "text": "ready"}],
            },
        },
    )
    transcript_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    return transcript_path


@dataclass(frozen=True, slots=True)
class StubCollector:
    descriptor: SourceDescriptor

    def build_plan(self, archive_root: Path) -> CollectionPlan:
        return CollectionPlan(
            source=self.descriptor.key,
            display_name=self.descriptor.display_name,
            archive_root=archive_root,
            execution_context=self.descriptor.execution_context,
            support_level=self.descriptor.support_level,
            default_input_roots=self.descriptor.default_input_roots,
            implemented=self.descriptor.support_level != SupportLevel.SCAFFOLD,
        )


def make_stub_registry() -> CollectorRegistry:
    registry = CollectorRegistry()
    registry.register(
        StubCollector(
            SourceDescriptor(
                key="complete_source",
                display_name="Complete Source",
                execution_context="test",
                support_level=SupportLevel.COMPLETE,
                default_input_roots=(),
            )
        )
    )
    registry.register(
        StubCollector(
            SourceDescriptor(
                key="partial_source",
                display_name="Partial Source",
                execution_context="test",
                support_level=SupportLevel.PARTIAL,
                default_input_roots=(),
            )
        )
    )
    registry.register(
        StubCollector(
            SourceDescriptor(
                key="scaffold_source",
                display_name="Scaffold Source",
                execution_context="test",
                support_level=SupportLevel.SCAFFOLD,
                default_input_roots=(),
            )
        )
    )
    return registry


def test_doctor_reports_missing_root_for_unavailable_source(tmp_path: Path) -> None:
    missing_root = tmp_path / "does-not-exist"

    exit_code, stdout, stderr = run_cli(
        "doctor",
        "claude",
        "--input-root",
        str(missing_root),
    )

    assert exit_code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["source"] == "claude"
    assert payload["status"] == "missing"
    assert payload["status_reason"] == "no readable input roots"
    assert payload["candidate_artifact_count"] == 0
    assert payload["root_resolution"] == {
        "platform": CURRENT_PLATFORM,
        "resolution_source": "cli_input_root",
        "resolved_roots": [str(missing_root.resolve(strict=False))],
        "roots": [
            {
                "declared_path": str(missing_root.resolve(strict=False)),
                "resolution_source": "cli_input_root",
                "path": str(missing_root.resolve(strict=False)),
            }
        ],
    }
    assert payload["roots"] == [
        {
            "declared_path": str(missing_root.resolve(strict=False)),
            "path": str(missing_root.resolve(strict=False)),
            "kind": "missing",
            "exists": False,
            "readable": False,
            "miss_reason": "path does not exist",
        }
    ]


def test_doctor_distinguishes_empty_root_from_missing_root(tmp_path: Path) -> None:
    empty_root = tmp_path / "empty-claude-root"
    empty_root.mkdir()

    exit_code, stdout, stderr = run_cli(
        "doctor",
        "claude",
        "--input-root",
        str(empty_root),
    )

    assert exit_code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["status"] == "partial-ready"
    assert payload["status_reason"] == "readable roots found but no candidate artifacts"
    assert payload["candidate_artifact_count"] == 0
    assert payload["root_resolution"] == {
        "platform": CURRENT_PLATFORM,
        "resolution_source": "cli_input_root",
        "resolved_roots": [str(empty_root.resolve(strict=False))],
        "roots": [
            {
                "declared_path": str(empty_root.resolve(strict=False)),
                "resolution_source": "cli_input_root",
                "path": str(empty_root.resolve(strict=False)),
            }
        ],
    }
    assert payload["roots"] == [
        {
            "declared_path": str(empty_root.resolve(strict=False)),
            "path": str(empty_root.resolve(strict=False)),
            "kind": "directory",
            "exists": True,
            "readable": True,
        }
    ]


def test_doctor_reports_ready_when_candidate_artifacts_exist(tmp_path: Path) -> None:
    write_claude_transcript(tmp_path)

    exit_code, stdout, stderr = run_cli(
        "doctor",
        "claude",
        "--input-root",
        str(tmp_path),
    )

    assert exit_code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["status"] == "ready"
    assert payload["status_reason"] == "candidate artifacts found in readable roots"
    assert payload["candidate_artifact_count"] == 1
    assert payload["support_level"] == "complete"


def test_doctor_reports_support_limitations_for_partial_source() -> None:
    exit_code, stdout, stderr = run_cli(
        "doctor",
        "cursor_editor",
        "--input-root",
        str(CURSOR_EDITOR_FIXTURE_ROOT),
    )

    assert exit_code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["source"] == "cursor_editor"
    assert payload["status"] == "partial-ready"
    assert payload["support_level"] == "partial"
    assert payload["support_limitation_summary"] == (
        "Cursor editor recovery restores known explicit cursorDiskKV bubble "
        "body variants, but sessions whose headers resolve only to empty or "
        "tool-only rows remain partial and opt-in for unattended batches."
    )
    assert payload["support_limitations"] == [
        "Composer sessions whose cursorDiskKV headers resolve only to empty or tool-only bubble rows still degrade to partial output, with missing bubble ids retained in session metadata.",
        "Cursor host logs, memory flags, and third-party extension state never promote a session without confirmed transcript rows.",
    ]
    assert "opt-in for unattended batches" in payload["status_reason"]


def test_doctor_reports_support_limitations_for_windsurf_editor() -> None:
    exit_code, stdout, stderr = run_cli(
        "doctor",
        "windsurf_editor",
        "--input-root",
        str(WINDSURF_EDITOR_FIXTURE_ROOT),
    )

    assert exit_code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["source"] == "windsurf_editor"
    assert payload["status"] == "partial-ready"
    assert payload["support_level"] == "partial"
    assert payload["candidate_artifact_count"] == 6
    assert payload["support_limitation_summary"] == (
        "Windsurf local memories and rules can be normalized, but no confirmed "
        "native editor session-history store is available yet, so the collector "
        "remains partial and opt-in."
    )
    assert payload["support_limitations"] == [
        "Memories and rules are captured as partial context rows because they are not original turn-by-turn Cascade transcripts.",
        "mcp_config.json, skills directories, and bare workspace metadata degrade to unsupported rows until a confirmed session-history store is observed.",
    ]
    assert "opt-in" in payload["status_reason"]


def test_doctor_all_applies_selection_policy() -> None:
    exit_code, stdout, stderr = run_cli(
        "doctor",
        "--all",
        "--profile",
        "complete_only",
        registry=make_stub_registry(),
    )

    assert exit_code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["selection_policy"] == {
        "profile": "complete_only",
        "minimum_support_level": "complete",
        "include_sources": [],
        "exclude_sources": [],
    }
    assert payload["selected_sources"] == ["complete_source"]
    assert [entry["source"] for entry in payload["excluded_sources"]] == [
        "partial_source",
        "scaffold_source",
    ]
    assert [entry["source"] for entry in payload["sources"]] == ["complete_source"]


def test_doctor_does_not_emit_credential_contents(tmp_path: Path) -> None:
    credential_path = (
        tmp_path
        / "Library"
        / "Application Support"
        / "google-vscode-extension"
        / "auth"
        / "credentials.json"
    )
    credential_path.parent.mkdir(parents=True, exist_ok=True)
    credential_path.write_text('{"token":"secret-token"}', encoding="utf-8")

    exit_code, stdout, stderr = run_cli(
        "doctor",
        "gemini_code_assist_ide",
        "--input-root",
        str(tmp_path),
    )

    assert exit_code == 0
    assert stderr == ""
    assert "secret-token" not in stdout
