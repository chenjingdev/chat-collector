from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .archive_digest import ArchiveDigestReport, summarize_archive_digest
from .archive_export import ArchiveExportReport, export_archive_subset
from .archive_memory_export import (
    ArchiveMemoryExportReport,
    export_archive_memory_records,
)
from .archive_verify import ArchiveVerifyReport, verify_archive
from .models import (
    CollectionExecutionPolicy,
    CollectionRunResult,
    EffectiveCollectConfig,
    SourceRunStatus,
    SourceSelectionProfile,
    ValidationMode,
)
from .registry import CollectorRegistry
from .runner import run_collection_batch
from .source_selection import build_source_selection_policy
from .validate import ValidationReport, validate_run

SHIP_ACCEPTANCE_SCHEMA_VERSION = "2026-03-20"
SHIP_ACCEPTANCE_PROFILE_NAME = "ship_acceptance"
SHIP_ACCEPTANCE_SOURCES = (
    "antigravity_editor_view",
    "claude",
    "codex_app",
    "codex_cli",
    "codex_ide_extension",
    "cursor",
    "cursor_editor",
)
SHIP_ACCEPTANCE_ALLOWED_DEGRADED_SOURCES = (
    "antigravity_editor_view",
    "cursor",
    "cursor_editor",
)
SHIP_ACCEPTANCE_RELEASE_BLOCKING_SOURCES = tuple(
    source
    for source in SHIP_ACCEPTANCE_SOURCES
    if source not in SHIP_ACCEPTANCE_ALLOWED_DEGRADED_SOURCES
)
SHIP_ACCEPTANCE_ARTIFACTS_DIRNAME = "acceptance"
SHIP_ACCEPTANCE_ARCHIVE_EXPORT_DIRNAME = "archive-export"
SHIP_ACCEPTANCE_MEMORY_EXPORT_DIRNAME = "memory-export"
_IGNORED_EMPTY_ROOT_ENTRIES = frozenset({".DS_Store"})


@dataclass(frozen=True, slots=True)
class ShipAcceptanceFinding:
    code: str
    message: str
    source: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "code": self.code,
            "message": self.message,
        }
        if self.source is not None:
            payload["source"] = self.source
        return payload


@dataclass(frozen=True, slots=True)
class ShipAcceptanceSourceOutcome:
    source: str
    support_level: str
    expectation: str
    observed_status: str
    scanned_artifact_count: int
    conversation_count: int
    written_conversation_count: int
    message_count: int
    passes_expectation: bool
    attention_required: bool
    verify_status: str | None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "source": self.source,
            "support_level": self.support_level,
            "expectation": self.expectation,
            "observed_status": self.observed_status,
            "scanned_artifact_count": self.scanned_artifact_count,
            "conversation_count": self.conversation_count,
            "written_conversation_count": self.written_conversation_count,
            "message_count": self.message_count,
            "passes_expectation": self.passes_expectation,
            "attention_required": self.attention_required,
            "verify_status": self.verify_status,
        }
        return payload


@dataclass(frozen=True, slots=True)
class ShipAcceptanceArtifacts:
    archive_export_dir: Path
    memory_export_dir: Path
    snapshot_path: Path | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "archive_export_dir": str(self.archive_export_dir),
            "memory_export_dir": str(self.memory_export_dir),
        }
        if self.snapshot_path is not None:
            payload["snapshot_path"] = str(self.snapshot_path)
        return payload


@dataclass(frozen=True, slots=True)
class ShipAcceptanceReport:
    archive_root: Path
    artifacts: ShipAcceptanceArtifacts
    run: CollectionRunResult
    validation: ValidationReport
    archive_verify: ArchiveVerifyReport
    archive_digest: ArchiveDigestReport
    archive_export: ArchiveExportReport
    memory_export: ArchiveMemoryExportReport
    sources: tuple[ShipAcceptanceSourceOutcome, ...]
    blocking_findings: tuple[ShipAcceptanceFinding, ...]

    @property
    def status(self) -> str:
        return "pass" if not self.blocking_findings else "fail"

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": SHIP_ACCEPTANCE_SCHEMA_VERSION,
            "profile": {
                "name": SHIP_ACCEPTANCE_PROFILE_NAME,
                "selected_sources": list(SHIP_ACCEPTANCE_SOURCES),
                "release_blocking_sources": list(
                    SHIP_ACCEPTANCE_RELEASE_BLOCKING_SOURCES
                ),
                "allowed_degraded_sources": list(
                    SHIP_ACCEPTANCE_ALLOWED_DEGRADED_SOURCES
                ),
            },
            "status": self.status,
            "archive_root": str(self.archive_root),
            "artifacts": self.artifacts.to_dict(),
            "run": {
                "run_id": self.run.run_id,
                "run_dir": str(self.run.run_dir),
                "manifest_path": str(self.run.manifest_path),
                "started_at": self.run.started_at,
                "completed_at": self.run.completed_at,
                "source_count": len(self.run.sources),
                "scanned_artifact_count": sum(
                    source.scanned_artifact_count for source in self.run.sources
                ),
                "conversation_count": sum(
                    source.conversation_count for source in self.run.sources
                ),
                "message_count": sum(source.message_count for source in self.run.sources),
                "written_conversation_count": sum(
                    source.written_conversation_count for source in self.run.sources
                ),
                "redaction_event_count": self.run.redaction_event_count,
            },
            "validation": {
                "status": self.validation.status.value,
                "success_count": self.validation.success_count,
                "warning_count": self.validation.warning_count,
                "error_count": self.validation.error_count,
            },
            "archive_verify": {
                "status": self.archive_verify.status.value,
                "warning_count": self.archive_verify.warning_count,
                "error_count": self.archive_verify.error_count,
                "bad_row_count": self.archive_verify.bad_row_count,
                "orphan_file_count": self.archive_verify.orphan_file_count,
            },
            "archive_digest": {
                "status": self.archive_digest.status,
                "warning_count": self.archive_digest.warning_count,
                "error_count": self.archive_digest.error_count,
                "suspicious_source_count": self.archive_digest.suspicious_source_count,
                "suspicious_conversation_count": (
                    self.archive_digest.suspicious_conversation_count
                ),
            },
            "archive_export": {
                "output_dir": str(self.archive_export.output_dir),
                "conversation_count": self.archive_export.conversation_count,
                "message_count": self.archive_export.message_count,
                "source_count": self.archive_export.source_count,
            },
            "memory_export": {
                "output_dir": str(self.memory_export.output_dir),
                "record_count": self.memory_export.record_count,
                "conversation_count": self.memory_export.conversation_count,
                "message_count": self.memory_export.message_count,
                "source_count": self.memory_export.source_count,
            },
            "sources": [source.to_dict() for source in self.sources],
            "blocking_findings": [
                finding.to_dict() for finding in self.blocking_findings
            ],
        }


def ensure_clean_acceptance_archive_root(archive_root: Path) -> Path:
    resolved_root = archive_root.expanduser().resolve(strict=False)
    if not resolved_root.exists():
        return resolved_root
    if not resolved_root.is_dir():
        raise ValueError(
            f"ship acceptance archive root is not a directory: {resolved_root}"
        )

    unexpected_entries = sorted(
        entry.name
        for entry in resolved_root.iterdir()
        if entry.name not in _IGNORED_EMPTY_ROOT_ENTRIES
    )
    if unexpected_entries:
        preview = ", ".join(unexpected_entries[:5])
        suffix = "" if len(unexpected_entries) <= 5 else ", ..."
        raise ValueError(
            "ship acceptance archive root must be empty before collection: "
            f"{resolved_root} ({preview}{suffix})"
        )
    return resolved_root


def ship_acceptance_artifacts_dir(archive_root: Path) -> Path:
    return archive_root / SHIP_ACCEPTANCE_ARTIFACTS_DIRNAME


def ship_acceptance_archive_export_dir(archive_root: Path) -> Path:
    return (
        ship_acceptance_artifacts_dir(archive_root)
        / SHIP_ACCEPTANCE_ARCHIVE_EXPORT_DIRNAME
    )


def ship_acceptance_memory_export_dir(archive_root: Path) -> Path:
    return (
        ship_acceptance_artifacts_dir(archive_root)
        / SHIP_ACCEPTANCE_MEMORY_EXPORT_DIRNAME
    )


def run_ship_acceptance(
    registry: CollectorRegistry,
    *,
    archive_root: Path,
    repo_root: Path,
    snapshot_path: Path | None = None,
) -> ShipAcceptanceReport:
    resolved_archive_root = ensure_clean_acceptance_archive_root(archive_root)
    selection_policy = build_source_selection_policy(
        profile=SourceSelectionProfile.ALL,
        include_sources=SHIP_ACCEPTANCE_SOURCES,
    )
    execution_policy = CollectionExecutionPolicy(
        incremental=True,
        validation=ValidationMode.OFF,
    )
    effective_config = EffectiveCollectConfig(
        archive_root=resolved_archive_root,
        selection_policy=selection_policy,
        execution_policy=execution_policy,
        config_source=SHIP_ACCEPTANCE_PROFILE_NAME,
    )
    run = run_collection_batch(
        registry,
        resolved_archive_root,
        selection_policy=selection_policy,
        execution_policy=execution_policy,
        effective_config=effective_config,
    )
    validation = validate_run(
        resolved_archive_root,
        run_id=run.run_id,
        repo_root=repo_root,
    )
    archive_verify = verify_archive(resolved_archive_root)
    archive_digest = summarize_archive_digest(resolved_archive_root)
    archive_export = export_archive_subset(
        resolved_archive_root,
        output_dir=ship_acceptance_archive_export_dir(resolved_archive_root),
        execute=True,
    )
    memory_export = export_archive_memory_records(
        resolved_archive_root,
        output_dir=ship_acceptance_memory_export_dir(resolved_archive_root),
        execute=True,
    )
    artifacts = ShipAcceptanceArtifacts(
        archive_export_dir=archive_export.output_dir,
        memory_export_dir=memory_export.output_dir,
        snapshot_path=snapshot_path,
    )
    report = build_ship_acceptance_report(
        archive_root=resolved_archive_root,
        artifacts=artifacts,
        run=run,
        validation=validation,
        archive_verify=archive_verify,
        archive_digest=archive_digest,
        archive_export=archive_export,
        memory_export=memory_export,
    )
    if snapshot_path is not None:
        write_ship_acceptance_snapshot(report, snapshot_path)
    return report


def build_ship_acceptance_report(
    *,
    archive_root: Path,
    artifacts: ShipAcceptanceArtifacts,
    run: CollectionRunResult,
    validation: ValidationReport,
    archive_verify: ArchiveVerifyReport,
    archive_digest: ArchiveDigestReport,
    archive_export: ArchiveExportReport,
    memory_export: ArchiveMemoryExportReport,
) -> ShipAcceptanceReport:
    digest_sources = {
        source_report.source: source_report for source_report in archive_digest.sources
    }
    verify_sources = {
        source_report.source: source_report for source_report in archive_verify.sources
    }

    blocking_findings: list[ShipAcceptanceFinding] = []
    source_outcomes: list[ShipAcceptanceSourceOutcome] = []
    for source_result in run.sources:
        digest_source = digest_sources.get(source_result.source)
        verify_source = verify_sources.get(source_result.source)
        attention_required = (
            False if digest_source is None else digest_source.attention_required
        )
        expectation = (
            "allowed_degraded"
            if source_result.source in SHIP_ACCEPTANCE_ALLOWED_DEGRADED_SOURCES
            else "release_blocking"
        )
        passes_expectation = _passes_source_expectation(
            source_result=source_result,
        )
        source_outcomes.append(
            ShipAcceptanceSourceOutcome(
                source=source_result.source,
                support_level=source_result.support_level.value,
                expectation=expectation,
                observed_status=source_result.status.value,
                scanned_artifact_count=source_result.scanned_artifact_count,
                conversation_count=source_result.conversation_count,
                written_conversation_count=source_result.written_conversation_count,
                message_count=source_result.message_count,
                passes_expectation=passes_expectation,
                attention_required=attention_required,
                verify_status=(
                    None if verify_source is None else verify_source.status.value
                ),
            )
        )
        blocking_findings.extend(
            _source_blocking_findings(
                source_result=source_result,
            )
        )

    if validation.error_count:
        blocking_findings.append(
            ShipAcceptanceFinding(
                code="validation_errors",
                message=(
                    f"validate reported {validation.error_count} error(s) for run "
                    f"{run.run_id}"
                ),
            )
        )
    if archive_verify.error_count:
        blocking_findings.append(
            ShipAcceptanceFinding(
                code="archive_verify_errors",
                message=(
                    f"archive verify reported {archive_verify.error_count} error(s)"
                ),
            )
        )
    if archive_verify.bad_row_count:
        blocking_findings.append(
            ShipAcceptanceFinding(
                code="archive_bad_rows",
                message=(
                    f"archive verify found {archive_verify.bad_row_count} bad row(s)"
                ),
            )
        )
    if archive_verify.orphan_file_count:
        blocking_findings.append(
            ShipAcceptanceFinding(
                code="archive_orphans",
                message=(
                    "archive verify found "
                    f"{archive_verify.orphan_file_count} orphan file(s)"
                ),
            )
        )
    if archive_digest.error_count:
        blocking_findings.append(
            ShipAcceptanceFinding(
                code="archive_digest_errors",
                message=(
                    f"archive digest reported {archive_digest.error_count} error(s)"
                ),
            )
        )

    written_conversation_count = sum(
        source.written_conversation_count for source in run.sources
    )
    if archive_export.conversation_count != written_conversation_count:
        blocking_findings.append(
            ShipAcceptanceFinding(
                code="archive_export_mismatch",
                message=(
                    "archive export conversation count does not match the collection run: "
                    f"{archive_export.conversation_count} != "
                    f"{written_conversation_count}"
                ),
            )
        )
    if memory_export.conversation_count != written_conversation_count:
        blocking_findings.append(
            ShipAcceptanceFinding(
                code="memory_export_mismatch",
                message=(
                    "memory export conversation count does not match the collection run: "
                    f"{memory_export.conversation_count} != "
                    f"{written_conversation_count}"
                ),
            )
        )
    if memory_export.record_count != written_conversation_count:
        blocking_findings.append(
            ShipAcceptanceFinding(
                code="memory_record_mismatch",
                message=(
                    "memory export record count does not match the collection run: "
                    f"{memory_export.record_count} != {written_conversation_count}"
                ),
            )
        )

    return ShipAcceptanceReport(
        archive_root=archive_root,
        artifacts=artifacts,
        run=run,
        validation=validation,
        archive_verify=archive_verify,
        archive_digest=archive_digest,
        archive_export=archive_export,
        memory_export=memory_export,
        sources=tuple(source_outcomes),
        blocking_findings=tuple(blocking_findings),
    )


def write_ship_acceptance_snapshot(
    report: ShipAcceptanceReport,
    snapshot_path: Path,
) -> None:
    resolved_path = snapshot_path.expanduser().resolve(strict=False)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _passes_source_expectation(
    *,
    source_result,
) -> bool:
    if source_result.failed:
        return False
    if source_result.scanned_artifact_count <= 0:
        return False
    if source_result.written_conversation_count <= 0:
        return False
    if source_result.source in SHIP_ACCEPTANCE_ALLOWED_DEGRADED_SOURCES:
        return True
    return source_result.status == SourceRunStatus.COMPLETE


def _source_blocking_findings(
    *,
    source_result,
) -> tuple[ShipAcceptanceFinding, ...]:
    findings: list[ShipAcceptanceFinding] = []
    if source_result.failed:
        findings.append(
            ShipAcceptanceFinding(
                code="source_failed",
                message=(
                    f"source '{source_result.source}' failed: "
                    f"{source_result.failure_reason or 'unknown failure'}"
                ),
                source=source_result.source,
            )
        )
        return tuple(findings)
    if source_result.scanned_artifact_count <= 0:
        findings.append(
            ShipAcceptanceFinding(
                code="source_missing_artifacts",
                message=(
                    f"source '{source_result.source}' did not find any candidate artifacts"
                ),
                source=source_result.source,
            )
        )
    if source_result.written_conversation_count <= 0:
        findings.append(
            ShipAcceptanceFinding(
                code="source_no_written_conversations",
                message=(
                    f"source '{source_result.source}' did not write any archive rows"
                ),
                source=source_result.source,
            )
        )
    if source_result.source in SHIP_ACCEPTANCE_ALLOWED_DEGRADED_SOURCES:
        return tuple(findings)
    if source_result.status != SourceRunStatus.COMPLETE:
        findings.append(
            ShipAcceptanceFinding(
                code="source_unexpected_degraded",
                message=(
                    "release-blocking source "
                    f"'{source_result.source}' finished with status "
                    f"'{source_result.status.value}'"
                ),
                source=source_result.source,
            )
        )
    return tuple(findings)


__all__ = [
    "SHIP_ACCEPTANCE_ALLOWED_DEGRADED_SOURCES",
    "SHIP_ACCEPTANCE_PROFILE_NAME",
    "SHIP_ACCEPTANCE_RELEASE_BLOCKING_SOURCES",
    "SHIP_ACCEPTANCE_SOURCES",
    "ShipAcceptanceArtifacts",
    "ShipAcceptanceFinding",
    "ShipAcceptanceReport",
    "ShipAcceptanceSourceOutcome",
    "build_ship_acceptance_report",
    "ensure_clean_acceptance_archive_root",
    "run_ship_acceptance",
    "ship_acceptance_archive_export_dir",
    "ship_acceptance_artifacts_dir",
    "ship_acceptance_memory_export_dir",
    "write_ship_acceptance_snapshot",
]
