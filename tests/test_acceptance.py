from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from llm_chat_archive.acceptance import (
    SHIP_ACCEPTANCE_ALLOWED_DEGRADED_SOURCES,
    SHIP_ACCEPTANCE_RELEASE_BLOCKING_SOURCES,
    SHIP_ACCEPTANCE_SOURCES,
    ShipAcceptanceArtifacts,
    build_ship_acceptance_report,
    ensure_clean_acceptance_archive_root,
)
from llm_chat_archive.archive_digest import ArchiveDigestReport, ArchiveDigestSourceReport
from llm_chat_archive.archive_export import ArchiveExportFilter, ArchiveExportReport
from llm_chat_archive.archive_memory_export import (
    ArchiveMemoryExportContract,
    ArchiveMemoryExportFilter,
    ArchiveMemoryExportReport,
)
from llm_chat_archive.archive_verify import ArchiveVerifyReport, ArchiveVerifySourceReport
from llm_chat_archive.models import (
    CollectionExecutionPolicy,
    CollectionRunResult,
    EffectiveCollectConfig,
    SourceRunResult,
    SourceRunStatus,
    SourceSelectionPolicy,
    SourceSelectionProfile,
    SupportLevel,
)
from llm_chat_archive.validate import ValidationLevel, ValidationReport

REPO_ROOT = Path(__file__).resolve().parents[1]


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "llm_chat_archive", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def test_ship_acceptance_constants_pin_operator_source_set() -> None:
    assert SHIP_ACCEPTANCE_SOURCES == (
        "antigravity_editor_view",
        "claude",
        "codex_app",
        "codex_cli",
        "codex_ide_extension",
        "cursor",
        "cursor_editor",
    )
    assert SHIP_ACCEPTANCE_ALLOWED_DEGRADED_SOURCES == (
        "antigravity_editor_view",
        "cursor",
        "cursor_editor",
    )
    assert SHIP_ACCEPTANCE_RELEASE_BLOCKING_SOURCES == (
        "claude",
        "codex_app",
        "codex_cli",
        "codex_ide_extension",
    )


def test_acceptance_cli_rejects_non_clean_archive_root(tmp_path: Path) -> None:
    (tmp_path / "existing.txt").write_text("occupied\n", encoding="utf-8")

    result = run_cli(
        "acceptance",
        "ship",
        "--archive-root",
        str(tmp_path),
    )

    assert result.returncode == 2
    assert "must be empty before collection" in result.stderr


def test_ensure_clean_acceptance_archive_root_allows_missing_or_empty_dir(
    tmp_path: Path,
) -> None:
    missing_root = tmp_path / "missing"
    assert ensure_clean_acceptance_archive_root(missing_root) == missing_root

    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    assert ensure_clean_acceptance_archive_root(empty_root) == empty_root


def test_build_ship_acceptance_report_allows_pinned_degraded_sources(
    tmp_path: Path,
) -> None:
    archive_root = tmp_path / "archive"
    run = _build_run_result(
        archive_root,
        [
            _source_result("claude", SupportLevel.COMPLETE, SourceRunStatus.COMPLETE),
            _source_result("codex_app", SupportLevel.COMPLETE, SourceRunStatus.COMPLETE),
            _source_result("codex_cli", SupportLevel.COMPLETE, SourceRunStatus.COMPLETE),
            _source_result(
                "codex_ide_extension",
                SupportLevel.COMPLETE,
                SourceRunStatus.COMPLETE,
            ),
            _source_result(
                "antigravity_editor_view",
                SupportLevel.PARTIAL,
                SourceRunStatus.UNSUPPORTED,
            ),
            _source_result("cursor", SupportLevel.PARTIAL, SourceRunStatus.PARTIAL),
            _source_result(
                "cursor_editor",
                SupportLevel.PARTIAL,
                SourceRunStatus.PARTIAL,
            ),
        ],
    )
    report = build_ship_acceptance_report(
        archive_root=archive_root,
        artifacts=_artifacts(archive_root),
        run=run,
        validation=_validation_report(archive_root, run.run_id),
        archive_verify=_archive_verify_report(archive_root, [source.source for source in run.sources]),
        archive_digest=_archive_digest_report(
            archive_root,
            run_id=run.run_id,
            source_names=[source.source for source in run.sources],
        ),
        archive_export=_archive_export_report(
            archive_root,
            conversation_count=len(run.sources),
            source_count=len(run.sources),
            message_count=sum(source.message_count for source in run.sources),
        ),
        memory_export=_memory_export_report(
            archive_root,
            record_count=len(run.sources),
            conversation_count=len(run.sources),
            source_count=len(run.sources),
            message_count=sum(source.message_count for source in run.sources),
        ),
    )

    assert report.status == "pass"
    assert report.blocking_findings == ()
    outcomes = {outcome.source: outcome for outcome in report.sources}
    assert outcomes["antigravity_editor_view"].passes_expectation is True
    assert outcomes["antigravity_editor_view"].expectation == "allowed_degraded"
    assert outcomes["cursor"].passes_expectation is True
    assert outcomes["cursor_editor"].passes_expectation is True
    assert outcomes["claude"].written_conversation_count == 1


def test_build_ship_acceptance_report_blocks_release_blocking_regressions(
    tmp_path: Path,
) -> None:
    archive_root = tmp_path / "archive"
    run = _build_run_result(
        archive_root,
        [
            _source_result("claude", SupportLevel.COMPLETE, SourceRunStatus.PARTIAL),
            _source_result("codex_app", SupportLevel.COMPLETE, SourceRunStatus.COMPLETE),
            _source_result("codex_cli", SupportLevel.COMPLETE, SourceRunStatus.COMPLETE),
            _source_result(
                "codex_ide_extension",
                SupportLevel.COMPLETE,
                SourceRunStatus.COMPLETE,
            ),
            _source_result(
                "antigravity_editor_view",
                SupportLevel.PARTIAL,
                SourceRunStatus.PARTIAL,
            ),
            _source_result("cursor", SupportLevel.PARTIAL, SourceRunStatus.PARTIAL),
            _source_result(
                "cursor_editor",
                SupportLevel.PARTIAL,
                SourceRunStatus.PARTIAL,
            ),
        ],
    )
    report = build_ship_acceptance_report(
        archive_root=archive_root,
        artifacts=_artifacts(archive_root),
        run=run,
        validation=_validation_report(archive_root, run.run_id),
        archive_verify=_archive_verify_report(archive_root, [source.source for source in run.sources]),
        archive_digest=_archive_digest_report(
            archive_root,
            run_id=run.run_id,
            source_names=[source.source for source in run.sources],
        ),
        archive_export=_archive_export_report(
            archive_root,
            conversation_count=len(run.sources),
            source_count=len(run.sources),
            message_count=sum(source.message_count for source in run.sources),
        ),
        memory_export=_memory_export_report(
            archive_root,
            record_count=len(run.sources),
            conversation_count=len(run.sources),
            source_count=len(run.sources),
            message_count=sum(source.message_count for source in run.sources),
        ),
    )

    payload = report.to_dict()

    assert report.status == "fail"
    assert {finding.code for finding in report.blocking_findings} == {
        "source_unexpected_degraded",
    }
    claude_outcome = {
        outcome.source: outcome for outcome in report.sources
    }["claude"]
    assert claude_outcome.passes_expectation is False
    assert payload["status"] == "fail"
    assert payload["profile"]["release_blocking_sources"] == list(
        SHIP_ACCEPTANCE_RELEASE_BLOCKING_SOURCES
    )


def _source_result(
    source: str,
    support_level: SupportLevel,
    status: SourceRunStatus,
) -> SourceRunResult:
    return SourceRunResult(
        source=source,
        support_level=support_level,
        status=status,
        archive_root=Path("/tmp/archive"),
        output_path=Path(f"/tmp/archive/{source}/memory_chat_v1-{source}.jsonl"),
        input_roots=(Path("/tmp/input"),),
        scanned_artifact_count=1,
        conversation_count=1,
        message_count=2,
        written_conversation_count=1,
        partial=status == SourceRunStatus.PARTIAL,
        unsupported=status == SourceRunStatus.UNSUPPORTED,
    )


def _build_run_result(
    archive_root: Path,
    sources: list[SourceRunResult],
) -> CollectionRunResult:
    selection_policy = SourceSelectionPolicy(
        profile=SourceSelectionProfile.ALL,
        minimum_support_level=SupportLevel.SCAFFOLD,
        include_sources=tuple(source.source for source in sources),
    )
    return CollectionRunResult(
        run_id="20260320T120000Z",
        archive_root=archive_root,
        run_dir=archive_root / "runs" / "20260320T120000Z",
        manifest_path=archive_root / "runs" / "20260320T120000Z" / "manifest.json",
        started_at="2026-03-20T12:00:00Z",
        completed_at="2026-03-20T12:00:30Z",
        selection_policy=selection_policy,
        effective_config=EffectiveCollectConfig(
            archive_root=archive_root,
            selection_policy=selection_policy,
            execution_policy=CollectionExecutionPolicy(),
            config_source="ship_acceptance",
        ),
        selected_sources=tuple(source.source for source in sources),
        excluded_sources=(),
        sources=tuple(sources),
    )


def _validation_report(archive_root: Path, run_id: str) -> ValidationReport:
    return ValidationReport(
        run_id=run_id,
        archive_root=archive_root,
        manifest_path=archive_root / "runs" / run_id / "manifest.json",
        sources=(),
        findings=(),
    )


def _archive_verify_report(
    archive_root: Path,
    source_names: list[str],
) -> ArchiveVerifyReport:
    return ArchiveVerifyReport(
        archive_root=archive_root,
        source_filter=None,
        manifest_count=1,
        linked_output_file_count=len(source_names),
        sources=tuple(
            ArchiveVerifySourceReport(
                source=source_name,
                status=ValidationLevel.SUCCESS,
                file_count=1,
                row_count=1,
                verified_row_count=1,
                bad_row_count=0,
                orphan_file_count=0,
                finding_count=0,
                warning_count=0,
                error_count=0,
                files=(),
            )
            for source_name in source_names
        ),
        findings=(),
    )


def _archive_digest_report(
    archive_root: Path,
    *,
    run_id: str,
    source_names: list[str],
    attention_required_sources: set[str] | None = None,
) -> ArchiveDigestReport:
    attention_required_sources = attention_required_sources or set()
    transcript_completeness = {
        "complete": {"count": len(source_names), "ratio": 1.0},
        "partial": {"count": 0, "ratio": 0.0},
        "unsupported": {"count": 0, "ratio": 0.0},
    }
    return ArchiveDigestReport(
        archive_root=archive_root,
        aggregated_at="2026-03-20T12:05:00Z",
        status="success",
        raw_status=None,
        raw_warning_count=None,
        raw_suspicious_source_count=None,
        latest_run_id=run_id,
        latest_run=None,
        latest_run_error=None,
        source_count=len(source_names),
        conversation_count=len(source_names),
        message_count=len(source_names) * 2,
        conversation_with_limitations_count=0,
        suspicious_source_count=0,
        suspicious_conversation_count=0,
        sources_with_orphans_count=0,
        orphan_file_count=0,
        warning_count=0,
        error_count=0,
        transcript_completeness=transcript_completeness,
        top_limitations=(),
        sources=tuple(
            ArchiveDigestSourceReport(
                source=source_name,
                latest_run_selected=True,
                latest_run_status="complete",
                support_level="complete",
                failed=False,
                run_degraded=False,
                attention_required=source_name in attention_required_sources,
                file_count=1,
                conversation_count=1,
                message_count=2,
                latest_collected_at="2026-03-20T12:00:00Z",
                conversation_with_limitations_count=0,
                transcript_completeness=transcript_completeness,
                top_limitations=(),
                verify_status="success",
                warning_count=0,
                error_count=0,
                orphan_file_count=0,
                has_orphans=False,
                suspicious=False,
                suspicious_conversation_count=0,
                source_reasons=(),
            )
            for source_name in source_names
        ),
    )


def _archive_export_report(
    archive_root: Path,
    *,
    conversation_count: int,
    source_count: int,
    message_count: int,
) -> ArchiveExportReport:
    output_dir = archive_root / "acceptance" / "archive-export"
    return ArchiveExportReport(
        archive_root=archive_root,
        output_dir=output_dir,
        write_mode="write",
        filters=ArchiveExportFilter(),
        conversation_count=conversation_count,
        message_count=message_count,
        source_count=source_count,
        conversations_path=output_dir / "conversations.jsonl",
        manifest_path=output_dir / "export-manifest.json",
    )


def _memory_export_report(
    archive_root: Path,
    *,
    record_count: int,
    conversation_count: int,
    source_count: int,
    message_count: int,
) -> ArchiveMemoryExportReport:
    output_dir = archive_root / "acceptance" / "memory-export"
    return ArchiveMemoryExportReport(
        archive_root=archive_root,
        output_dir=output_dir,
        write_mode="write",
        contract=ArchiveMemoryExportContract(),
        filters=ArchiveMemoryExportFilter(),
        record_count=record_count,
        conversation_count=conversation_count,
        message_count=message_count,
        source_count=source_count,
        earliest_collected_at="2026-03-20T12:00:00Z",
        latest_collected_at="2026-03-20T12:00:00Z",
        records_path=output_dir / "memory-records.jsonl",
        manifest_path=output_dir / "memory-export-manifest.json",
    )


def _artifacts(archive_root: Path) -> ShipAcceptanceArtifacts:
    return ShipAcceptanceArtifacts(
        archive_export_dir=archive_root / "acceptance" / "archive-export",
        memory_export_dir=archive_root / "acceptance" / "memory-export",
        snapshot_path=archive_root / "snapshot.json",
    )
