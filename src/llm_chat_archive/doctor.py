from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from .models import ExcludedSource, SourceSelectionPolicy, SupportLevel
from .parser_drift import ParserAssumptionReport, inspect_parser_assumptions
from .registry import Collector, CollectorRegistry
from .source_roots import SourceRootResolution, resolve_source_roots
from .source_selection import build_source_selection_policy, select_collectors
from .sources.antigravity_editor_view import discover_antigravity_editor_view_artifacts
from .sources.claude_code_cli import (
    iter_transcript_paths as iter_claude_transcript_paths,
)
from .sources.claude_code_ide import (
    discover_ide_bridge_provenance as discover_claude_ide_bridge_provenance,
    parse_transcript_file as parse_claude_ide_transcript_file,
)
from .sources.codex_app import parse_rollout_file as parse_codex_app_rollout_file
from .sources.codex_cli import parse_rollout_file as parse_codex_cli_rollout_file
from .sources.codex_ide_extension import (
    parse_rollout_file as parse_codex_ide_extension_rollout_file,
)
from .sources.codex_rollout import iter_rollout_paths
from .sources.cursor_cli import iter_cli_log_paths
from .sources.cursor_editor import iter_workspace_state_paths as iter_cursor_workspace_state_paths
from .sources.gemini_cli import discover_project_sessions
from .sources.gemini_code_assist_ide import discover_gemini_code_assist_ide_artifacts
from .sources.windsurf_editor import (
    build_windsurf_conversations,
    discover_windsurf_editor_artifacts,
)


class DoctorStatus(StrEnum):
    READY = "ready"
    MISSING = "missing"
    PARTIAL_READY = "partial-ready"


@dataclass(frozen=True, slots=True)
class DoctorRootReport:
    declared_path: str
    path: str | None
    kind: str
    exists: bool
    readable: bool
    miss_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "declared_path": self.declared_path,
            "path": self.path,
            "kind": self.kind,
            "exists": self.exists,
            "readable": self.readable,
        }
        if self.miss_reason is not None:
            payload["miss_reason"] = self.miss_reason
        return payload


@dataclass(frozen=True, slots=True)
class DoctorSourceReport:
    source: str
    display_name: str
    execution_context: str
    support_level: SupportLevel
    support_limitation_summary: str | None
    support_limitations: tuple[str, ...]
    status: DoctorStatus
    status_reason: str
    candidate_artifact_count: int
    root_resolution: SourceRootResolution
    roots: tuple[DoctorRootReport, ...]
    parser_assumption_report: ParserAssumptionReport | None = None
    inspection_error: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "source": self.source,
            "display_name": self.display_name,
            "execution_context": self.execution_context,
            "support_level": self.support_level.value,
            "status": self.status.value,
            "status_reason": self.status_reason,
            "candidate_artifact_count": self.candidate_artifact_count,
            "root_resolution": self.root_resolution.to_dict(),
            "roots": [root.to_dict() for root in self.roots],
        }
        if self.parser_assumption_report is not None:
            payload["parser_assumption"] = self.parser_assumption_report.to_dict()
        if self.support_limitation_summary is not None:
            payload["support_limitation_summary"] = self.support_limitation_summary
        if self.support_limitations:
            payload["support_limitations"] = list(self.support_limitations)
        if self.inspection_error is not None:
            payload["inspection_error"] = self.inspection_error
        return payload


@dataclass(frozen=True, slots=True)
class DoctorBatchReport:
    selection_policy: SourceSelectionPolicy
    selected_sources: tuple[str, ...]
    excluded_sources: tuple[ExcludedSource, ...]
    sources: tuple[DoctorSourceReport, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "selection_policy": self.selection_policy.to_dict(),
            "selected_sources": list(self.selected_sources),
            "excluded_sources": [
                excluded_source.to_dict() for excluded_source in self.excluded_sources
            ],
            "sources": [source.to_dict() for source in self.sources],
        }


def inspect_source_readiness(
    collector: Collector,
    *,
    input_roots: tuple[Path, ...] | None = None,
) -> DoctorSourceReport:
    root_resolution = resolve_source_roots(
        collector.descriptor,
        input_roots=input_roots,
    )
    resolved_input_roots = root_resolution.resolved_paths
    root_reports = tuple(_inspect_root(root) for root in root_resolution.roots)

    inspection_error: str | None = None
    parser_assumption_report: ParserAssumptionReport | None = None
    support_metadata = collector.descriptor.support_metadata
    try:
        candidate_artifact_count = _count_candidate_artifacts(
            collector,
            resolved_input_roots=resolved_input_roots,
        )
    except Exception as exc:
        candidate_artifact_count = 0
        inspection_error = f"{type(exc).__name__}: {exc}"

    try:
        parser_assumption_report = inspect_parser_assumptions(
            collector.descriptor.key,
            input_roots=resolved_input_roots,
            repo_path=Path(getattr(collector, "repo_path", Path.cwd())),
        )
    except Exception as exc:
        parser_assumption_report = None
        if inspection_error is None:
            inspection_error = f"{type(exc).__name__}: {exc}"

    status, status_reason = _determine_status(
        support_level=collector.descriptor.support_level,
        support_limitation_summary=(
            None if support_metadata is None else support_metadata.limitation_summary
        ),
        root_resolution=root_resolution,
        root_reports=root_reports,
        candidate_artifact_count=candidate_artifact_count,
        parser_assumption_report=parser_assumption_report,
        inspection_error=inspection_error,
    )
    return DoctorSourceReport(
        source=collector.descriptor.key,
        display_name=collector.descriptor.display_name,
        execution_context=collector.descriptor.execution_context,
        support_level=collector.descriptor.support_level,
        support_limitation_summary=(
            None if support_metadata is None else support_metadata.limitation_summary
        ),
        support_limitations=(() if support_metadata is None else support_metadata.limitations),
        status=status,
        status_reason=status_reason,
        candidate_artifact_count=candidate_artifact_count,
        root_resolution=root_resolution,
        roots=root_reports,
        parser_assumption_report=parser_assumption_report,
        inspection_error=inspection_error,
    )


def inspect_registry_readiness(
    registry: CollectorRegistry,
    *,
    input_roots: tuple[Path, ...] | None = None,
    selection_policy: SourceSelectionPolicy | None = None,
) -> DoctorBatchReport:
    selected_collectors = select_collectors(
        registry,
        policy=selection_policy or build_source_selection_policy(),
    )
    return DoctorBatchReport(
        selection_policy=selected_collectors.policy,
        selected_sources=selected_collectors.selected_sources,
        excluded_sources=selected_collectors.excluded_sources,
        sources=tuple(
            inspect_source_readiness(collector, input_roots=input_roots)
            for collector in selected_collectors.collectors
        ),
    )
def _inspect_root(root: ResolvedSourceRoot) -> DoctorRootReport:
    if root.path is None:
        return DoctorRootReport(
            declared_path=root.declared_path,
            path=None,
            kind="unresolved",
            exists=False,
            readable=False,
            miss_reason=root.miss_reason,
        )

    resolved_root = Path(root.path).expanduser().resolve(strict=False)
    exists = resolved_root.exists()
    if not exists:
        return DoctorRootReport(
            declared_path=root.declared_path,
            path=str(resolved_root),
            kind="missing",
            exists=False,
            readable=False,
            miss_reason="path does not exist",
        )

    if resolved_root.is_dir():
        readable = os.access(resolved_root, os.R_OK | os.X_OK)
        kind = "directory"
    elif resolved_root.is_file():
        readable = os.access(resolved_root, os.R_OK)
        kind = "file"
    else:
        readable = os.access(resolved_root, os.R_OK)
        kind = "other"

    return DoctorRootReport(
        declared_path=root.declared_path,
        path=str(resolved_root),
        kind=kind,
        exists=True,
        readable=readable,
        miss_reason=(None if readable else "path is not readable"),
    )


def _count_candidate_artifacts(
    collector: Collector,
    *,
    resolved_input_roots: tuple[Path, ...],
) -> int:
    source = collector.descriptor.key
    if source == "codex_cli":
        return sum(
            1
            for rollout_path in iter_rollout_paths(resolved_input_roots)
            if parse_codex_cli_rollout_file(rollout_path) is not None
        )
    if source == "codex_app":
        return sum(
            1
            for rollout_path in iter_rollout_paths(resolved_input_roots)
            if parse_codex_app_rollout_file(rollout_path) is not None
        )
    if source == "codex_ide_extension":
        return sum(
            1
            for rollout_path in iter_rollout_paths(resolved_input_roots)
            if parse_codex_ide_extension_rollout_file(rollout_path) is not None
        )
    if source == "claude":
        return sum(1 for _ in iter_claude_transcript_paths(resolved_input_roots))
    if source == "claude_code_ide":
        discovery = discover_claude_ide_bridge_provenance(resolved_input_roots)
        return sum(
            1
            for transcript_path in iter_claude_transcript_paths(resolved_input_roots)
            if parse_claude_ide_transcript_file(
                transcript_path,
                discovery=discovery,
            )
            is not None
        )
    if source == "gemini":
        repo_path = Path(getattr(collector, "repo_path", Path.cwd()))
        discovery = discover_project_sessions(repo_path, resolved_input_roots)
        return len(discovery.session_paths)
    if source == "gemini_code_assist_ide":
        artifacts = discover_gemini_code_assist_ide_artifacts(resolved_input_roots)
        return len(artifacts.global_state_paths) + len(artifacts.workspace_state_paths)
    if source == "cursor":
        return sum(1 for _ in iter_cli_log_paths(resolved_input_roots))
    if source == "cursor_editor":
        return sum(1 for _ in iter_cursor_workspace_state_paths(resolved_input_roots))
    if source == "antigravity_editor_view":
        artifacts = discover_antigravity_editor_view_artifacts(resolved_input_roots)
        return len(artifacts.conversation_paths)
    if source == "windsurf_editor":
        artifacts = discover_windsurf_editor_artifacts(resolved_input_roots)
        return len(build_windsurf_conversations(artifacts))
    return 0


def _determine_status(
    *,
    support_level: SupportLevel,
    support_limitation_summary: str | None,
    root_resolution: SourceRootResolution,
    root_reports: tuple[DoctorRootReport, ...],
    candidate_artifact_count: int,
    parser_assumption_report: ParserAssumptionReport | None,
    inspection_error: str | None,
) -> tuple[DoctorStatus, str]:
    readable_root_count = sum(1 for root in root_reports if root.readable)

    if inspection_error is not None:
        return DoctorStatus.PARTIAL_READY, "artifact inspection failed"
    if readable_root_count == 0:
        if root_resolution.miss_reasons:
            return DoctorStatus.MISSING, root_resolution.miss_reasons[0]
        return DoctorStatus.MISSING, "no readable input roots"
    if (
        parser_assumption_report is not None
        and parser_assumption_report.drift_suspected
    ):
        return DoctorStatus.PARTIAL_READY, parser_assumption_report.summary
    if candidate_artifact_count == 0:
        return DoctorStatus.PARTIAL_READY, "readable roots found but no candidate artifacts"
    if support_level != SupportLevel.COMPLETE:
        reason = f"candidate artifacts found but support level is {support_level.value}"
        if support_limitation_summary is not None:
            reason = f"{reason}: {support_limitation_summary}"
        return (
            DoctorStatus.PARTIAL_READY,
            reason,
        )
    return DoctorStatus.READY, "candidate artifacts found in readable roots"


__all__ = [
    "DoctorBatchReport",
    "DoctorRootReport",
    "DoctorSourceReport",
    "DoctorStatus",
    "inspect_registry_readiness",
    "inspect_source_readiness",
]
