from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_ROOT = REPO_ROOT / "tests" / "fixtures"


@dataclass(frozen=True, slots=True)
class FixtureMetadata:
    app: str
    version_markers: tuple[str, ...] = ()
    build_markers: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {"app": self.app}
        if self.version_markers:
            payload["version_markers"] = list(self.version_markers)
        if self.build_markers:
            payload["build_markers"] = list(self.build_markers)
        return payload


@dataclass(frozen=True, slots=True)
class FixtureCase:
    case_id: str
    source: str
    fixture_subpath: str
    fixture_metadata: FixtureMetadata | None = None

    @property
    def fixture_root(self) -> Path:
        return FIXTURES_ROOT / self.fixture_subpath


FIXTURE_CASES = (
    FixtureCase(
        case_id="antigravity_editor_view",
        source="antigravity_editor_view",
        fixture_subpath="antigravity_editor_view",
        fixture_metadata=FixtureMetadata(
            app="Antigravity",
            build_markers=("annotation_version=4", "chat_session_store_index_version=1"),
        ),
    ),
    FixtureCase(
        case_id="claude",
        source="claude",
        fixture_subpath="claude_code_cli",
        fixture_metadata=FixtureMetadata(app="Claude Code"),
    ),
    FixtureCase(
        case_id="claude_code_ide",
        source="claude_code_ide",
        fixture_subpath="claude_code_ide",
        fixture_metadata=FixtureMetadata(app="Claude Code IDE Bridge"),
    ),
    FixtureCase(
        case_id="codex_app",
        source="codex_app",
        fixture_subpath="codex_app",
        fixture_metadata=FixtureMetadata(
            app="Codex Desktop",
            version_markers=("0.31.0", "0.32.0"),
        ),
    ),
    FixtureCase(
        case_id="codex_cli",
        source="codex_cli",
        fixture_subpath="codex_cli",
        fixture_metadata=FixtureMetadata(
            app="Codex CLI",
            version_markers=("0.31.1", "0.32.0"),
        ),
    ),
    FixtureCase(
        case_id="codex_cli_redaction",
        source="codex_cli",
        fixture_subpath="redaction/codex_cli",
        fixture_metadata=FixtureMetadata(
            app="Codex CLI",
            version_markers=("0.32.0",),
        ),
    ),
    FixtureCase(
        case_id="codex_ide_extension",
        source="codex_ide_extension",
        fixture_subpath="codex_ide_extension",
        fixture_metadata=FixtureMetadata(
            app="Codex IDE Extension",
            version_markers=("0.31.1", "0.32.0"),
        ),
    ),
    FixtureCase(
        case_id="cursor",
        source="cursor",
        fixture_subpath="cursor_cli",
        fixture_metadata=FixtureMetadata(
            app="Cursor CLI",
            version_markers=("0.48.2",),
        ),
    ),
    FixtureCase(
        case_id="cursor_editor",
        source="cursor_editor",
        fixture_subpath="cursor_editor",
        fixture_metadata=FixtureMetadata(app="Cursor Editor"),
    ),
    FixtureCase(
        case_id="gemini",
        source="gemini",
        fixture_subpath="gemini_cli",
        fixture_metadata=FixtureMetadata(app="Gemini CLI"),
    ),
    FixtureCase(
        case_id="gemini_code_assist_ide",
        source="gemini_code_assist_ide",
        fixture_subpath="gemini_code_assist_ide",
        fixture_metadata=FixtureMetadata(
            app="Gemini Code Assist",
            version_markers=("2.73.0",),
            build_markers=("chat_session_index_version=1",),
        ),
    ),
    FixtureCase(
        case_id="windsurf_editor",
        source="windsurf_editor",
        fixture_subpath="windsurf_editor",
        fixture_metadata=FixtureMetadata(
            app="Windsurf",
            build_markers=("mcp_server_count=1", "workspace_rule_trigger=glob"),
        ),
    ),
)


def unique_source_fixture_cases() -> tuple[FixtureCase, ...]:
    seen: set[str] = set()
    cases: list[FixtureCase] = []
    for case in FIXTURE_CASES:
        if case.source in seen:
            continue
        seen.add(case.source)
        cases.append(case)
    return tuple(cases)


__all__ = [
    "FIXTURE_CASES",
    "FIXTURES_ROOT",
    "FixtureCase",
    "FixtureMetadata",
    "REPO_ROOT",
    "unique_source_fixture_cases",
]
