from __future__ import annotations

import json
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path

from .baseline_policy import BaselinePolicy, BaselineReport
from .models import (
    ArchiveTargetPolicy,
    MessageRole,
    SourceRunStatus,
    SupportLevel,
    TranscriptCompleteness,
)
from .parser_drift import inspect_parser_assumptions
from .runner import MANIFEST_FILENAME, RUNS_DIRECTORY


class ValidationLevel(StrEnum):
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class ValidationFinding:
    level: ValidationLevel
    code: str
    message: str
    source: str | None = None
    path: Path | None = None
    row_number: int | None = None
    suppressed: bool = False
    suppression_entry_id: str | None = None
    suppression_kind: str | None = None
    suppression_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "level": self.level.value,
            "code": self.code,
            "message": self.message,
        }
        if self.source is not None:
            payload["source"] = self.source
        if self.path is not None:
            payload["path"] = str(self.path)
        if self.row_number is not None:
            payload["row_number"] = self.row_number
        if self.suppressed:
            payload["suppressed"] = True
            payload["suppressed_by"] = {
                "entry_id": self.suppression_entry_id,
                "kind": self.suppression_kind,
                "reason": self.suppression_reason,
            }
        return payload


@dataclass(frozen=True, slots=True)
class ValidatedSource:
    source: str
    support_level: str | None
    status: str | None
    failed: bool | None
    output_path: Path | None
    support_limitation_summary: str | None
    support_limitations: tuple[str, ...]
    row_count: int
    actual_message_count: int
    declared_conversation_count: int | None
    declared_skipped_conversation_count: int | None
    declared_written_conversation_count: int | None
    declared_message_count: int | None
    validation_status: ValidationLevel
    raw_validation_status: ValidationLevel | None = None
    drift_suspected: bool = False
    parser_assumption_summary: str | None = None
    suppressed_warning_count: int = 0

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "source": self.source,
            "validation_status": self.validation_status.value,
            "support_level": self.support_level,
            "status": self.status,
            "failed": self.failed,
            "output_path": str(self.output_path) if self.output_path is not None else None,
            "row_count": self.row_count,
            "actual_message_count": self.actual_message_count,
            "declared_conversation_count": self.declared_conversation_count,
            "declared_skipped_conversation_count": self.declared_skipped_conversation_count,
            "declared_written_conversation_count": self.declared_written_conversation_count,
            "declared_message_count": self.declared_message_count,
            "drift_suspected": self.drift_suspected,
        }
        raw_validation_status = self.raw_validation_status or self.validation_status
        if raw_validation_status != self.validation_status:
            payload["raw_validation_status"] = raw_validation_status.value
        if self.suppressed_warning_count > 0:
            payload["suppressed_warning_count"] = self.suppressed_warning_count
        if self.support_limitation_summary is not None:
            payload["support_limitation_summary"] = self.support_limitation_summary
        if self.support_limitations:
            payload["support_limitations"] = list(self.support_limitations)
        if self.parser_assumption_summary is not None:
            payload["parser_assumption_summary"] = self.parser_assumption_summary
        return payload


@dataclass(frozen=True, slots=True)
class ValidationReport:
    run_id: str
    archive_root: Path
    manifest_path: Path
    sources: tuple[ValidatedSource, ...]
    findings: tuple[ValidationFinding, ...]
    baseline_path: Path | None = None
    baseline_entry_count: int = 0

    @property
    def status(self) -> ValidationLevel:
        return _worst_level(self.findings)

    @property
    def raw_status(self) -> ValidationLevel:
        return _worst_level(self.findings, include_suppressed=True)

    @property
    def success_count(self) -> int:
        return sum(1 for finding in self.findings if finding.level == ValidationLevel.SUCCESS)

    @property
    def warning_count(self) -> int:
        return sum(
            1
            for finding in self.findings
            if finding.level == ValidationLevel.WARNING and not finding.suppressed
        )

    @property
    def raw_warning_count(self) -> int:
        return sum(1 for finding in self.findings if finding.level == ValidationLevel.WARNING)

    @property
    def suppressed_warning_count(self) -> int:
        return self.raw_warning_count - self.warning_count

    @property
    def error_count(self) -> int:
        return sum(1 for finding in self.findings if finding.level == ValidationLevel.ERROR)

    def to_dict(self) -> dict[str, object]:
        payload = {
            "run_id": self.run_id,
            "archive_root": str(self.archive_root),
            "manifest_path": str(self.manifest_path),
            "status": self.status.value,
            "summary": {
                "success_count": self.success_count,
                "warning_count": self.warning_count,
                "error_count": self.error_count,
            },
            "sources": [source.to_dict() for source in self.sources],
            "findings": [finding.to_dict() for finding in self.findings],
        }
        if self.suppressed_warning_count > 0:
            payload["raw_status"] = self.raw_status.value
            payload["summary"]["raw_warning_count"] = self.raw_warning_count
            payload["summary"]["suppressed_warning_count"] = (
                self.suppressed_warning_count
            )
        if self.baseline_path is not None:
            payload["baseline"] = {
                "path": str(self.baseline_path),
                "entry_count": self.baseline_entry_count,
            }
        return payload


def validate_run(
    archive_root: Path,
    *,
    run_id: str,
    repo_root: Path,
    baseline_policy: BaselinePolicy | None = None,
) -> ValidationReport:
    resolved_archive_root = archive_root.expanduser().resolve(strict=False)
    resolved_repo_root = repo_root.expanduser().resolve(strict=False)
    manifest_path = resolved_archive_root / RUNS_DIRECTORY / run_id / MANIFEST_FILENAME
    findings: list[ValidationFinding] = []
    validated_sources: list[ValidatedSource] = []

    payload = _load_json_object(manifest_path, findings=findings, run_id=run_id)
    if payload is None:
        report = ValidationReport(
            run_id=run_id,
            archive_root=resolved_archive_root,
            manifest_path=manifest_path,
            sources=(),
            findings=tuple(findings),
        )
        if baseline_policy is not None:
            return _apply_baseline_policy(report, baseline_policy=baseline_policy)
        return report

    _add_finding(
        findings,
        level=ValidationLevel.SUCCESS,
        code="manifest_loaded",
        message=f"loaded run manifest {manifest_path}",
        path=manifest_path,
    )

    manifest_run_id = _required_string(
        payload,
        "run_id",
        findings=findings,
        context="run manifest",
        code="missing_required_field",
    )
    if manifest_run_id is not None and manifest_run_id != run_id:
        _add_finding(
            findings,
            level=ValidationLevel.ERROR,
            code="invalid_field",
            message=(
                f"run manifest run_id '{manifest_run_id}' does not match requested "
                f"run '{run_id}'"
            ),
            path=manifest_path,
        )

    manifest_archive_root = _required_absolute_path(
        payload,
        "archive_root",
        findings=findings,
        context="run manifest",
        code="missing_required_field",
    )
    if manifest_archive_root is not None:
        _validate_external_path(
            manifest_archive_root,
            repo_root=resolved_repo_root,
            findings=findings,
            label="manifest archive_root",
        )
        if manifest_archive_root != resolved_archive_root:
            _add_finding(
                findings,
                level=ValidationLevel.ERROR,
                code="invalid_field",
                message=(
                    f"run manifest archive_root '{manifest_archive_root}' does not match "
                    f"command archive_root '{resolved_archive_root}'"
                ),
                path=manifest_archive_root,
            )
    else:
        manifest_archive_root = resolved_archive_root

    run_dir = _required_absolute_path(
        payload,
        "run_dir",
        findings=findings,
        context="run manifest",
        code="missing_required_field",
    )
    if run_dir is not None:
        _validate_external_path(
            run_dir,
            repo_root=resolved_repo_root,
            findings=findings,
            label="manifest run_dir",
        )

    manifest_path_field = _required_absolute_path(
        payload,
        "manifest_path",
        findings=findings,
        context="run manifest",
        code="missing_required_field",
    )
    if manifest_path_field is not None:
        _validate_external_path(
            manifest_path_field,
            repo_root=resolved_repo_root,
            findings=findings,
            label="manifest manifest_path",
        )
        if manifest_path_field != manifest_path:
            _add_finding(
                findings,
                level=ValidationLevel.ERROR,
                code="invalid_field",
                message=(
                    f"run manifest manifest_path '{manifest_path_field}' does not match "
                    f"actual manifest path '{manifest_path}'"
                ),
                path=manifest_path_field,
            )

    if run_dir is not None and manifest_path != run_dir / MANIFEST_FILENAME:
        _add_finding(
            findings,
            level=ValidationLevel.ERROR,
            code="invalid_field",
            message=(
                f"run manifest path '{manifest_path}' is not located under run_dir '{run_dir}'"
            ),
            path=manifest_path,
        )

    source_count = _required_int(
        payload,
        "source_count",
        findings=findings,
        context="run manifest",
        code="missing_required_field",
    )
    failed_source_count = _required_int(
        payload,
        "failed_source_count",
        findings=findings,
        context="run manifest",
        code="missing_required_field",
    )
    conversation_count = _required_int(
        payload,
        "conversation_count",
        findings=findings,
        context="run manifest",
        code="missing_required_field",
    )
    skipped_conversation_count = _required_int(
        payload,
        "skipped_conversation_count",
        findings=findings,
        context="run manifest",
        code="missing_required_field",
    )
    written_conversation_count = _required_int(
        payload,
        "written_conversation_count",
        findings=findings,
        context="run manifest",
        code="missing_required_field",
    )
    message_count = _required_int(
        payload,
        "message_count",
        findings=findings,
        context="run manifest",
        code="missing_required_field",
    )

    sources_payload = payload.get("sources")
    if not isinstance(sources_payload, list):
        _add_finding(
            findings,
            level=ValidationLevel.ERROR,
            code="invalid_field",
            message=f"run manifest field 'sources' must be an array: {manifest_path}",
            path=manifest_path,
        )
        return ValidationReport(
            run_id=run_id,
            archive_root=resolved_archive_root,
            manifest_path=manifest_path,
            sources=(),
            findings=tuple(findings),
        )

    for index, source_payload in enumerate(sources_payload, start=1):
        validated_sources.append(
            _validate_source_payload(
                source_payload,
                index=index,
                manifest_archive_root=manifest_archive_root,
                repo_root=resolved_repo_root,
                findings=findings,
            )
        )

    if source_count is not None and source_count != len(sources_payload):
        _add_finding(
            findings,
            level=ValidationLevel.ERROR,
            code="count_mismatch",
            message=(
                f"run manifest source_count is {source_count} but contains "
                f"{len(sources_payload)} source entries"
            ),
            path=manifest_path,
        )

    if failed_source_count is not None:
        actual_failed_source_count = sum(1 for source in validated_sources if source.failed is True)
        if failed_source_count != actual_failed_source_count:
            _add_finding(
                findings,
                level=ValidationLevel.ERROR,
                code="count_mismatch",
                message=(
                    f"run manifest failed_source_count is {failed_source_count} but "
                    f"sources report {actual_failed_source_count} failures"
                ),
                path=manifest_path,
            )

    _compare_manifest_total(
        findings=findings,
        manifest_path=manifest_path,
        field_name="conversation_count",
        declared=conversation_count,
        actual=sum(
            source.declared_conversation_count or 0 for source in validated_sources
        ),
    )
    _compare_manifest_total(
        findings=findings,
        manifest_path=manifest_path,
        field_name="skipped_conversation_count",
        declared=skipped_conversation_count,
        actual=sum(
            source.declared_skipped_conversation_count or 0
            for source in validated_sources
        ),
    )
    _compare_manifest_total(
        findings=findings,
        manifest_path=manifest_path,
        field_name="written_conversation_count",
        declared=written_conversation_count,
        actual=sum(
            source.declared_written_conversation_count or 0
            for source in validated_sources
        ),
    )
    _compare_manifest_total(
        findings=findings,
        manifest_path=manifest_path,
        field_name="message_count",
        declared=message_count,
        actual=sum(source.declared_message_count or 0 for source in validated_sources),
    )

    report = ValidationReport(
        run_id=run_id,
        archive_root=resolved_archive_root,
        manifest_path=manifest_path,
        sources=tuple(validated_sources),
        findings=tuple(findings),
    )
    if baseline_policy is not None:
        return _apply_baseline_policy(report, baseline_policy=baseline_policy)
    return report


def _validate_source_payload(
    source_payload: object,
    *,
    index: int,
    manifest_archive_root: Path,
    repo_root: Path,
    findings: list[ValidationFinding],
) -> ValidatedSource:
    if not isinstance(source_payload, dict):
        placeholder = f"<source #{index}>"
        _add_finding(
            findings,
            level=ValidationLevel.ERROR,
            code="invalid_field",
            message=f"run manifest sources[{index - 1}] must be an object",
            source=placeholder,
        )
        return ValidatedSource(
            source=placeholder,
            support_level=None,
            status=None,
            failed=None,
            output_path=None,
            support_limitation_summary=None,
            support_limitations=(),
            row_count=0,
            actual_message_count=0,
            declared_conversation_count=None,
            declared_skipped_conversation_count=None,
            declared_written_conversation_count=None,
            declared_message_count=None,
            drift_suspected=False,
            parser_assumption_summary=None,
            validation_status=ValidationLevel.ERROR,
        )

    source = _required_string(
        source_payload,
        "source",
        findings=findings,
        context=f"run manifest source #{index}",
        code="missing_required_field",
    ) or f"<source #{index}>"
    support_limitation_summary = _optional_string(source_payload, "support_limitation_summary")
    support_limitations = _optional_string_sequence(source_payload, "support_limitations")
    input_roots = _optional_absolute_path_sequence(
        source_payload,
        "input_roots",
        findings=findings,
        context=f"run manifest source '{source}'",
        source=source,
    )

    start_index = len(findings)

    support_level = _required_string(
        source_payload,
        "support_level",
        findings=findings,
        context=f"run manifest source '{source}'",
        code="missing_required_field",
        source=source,
    )
    if support_level is not None:
        _validate_enum(
            support_level,
            SupportLevel,
            findings=findings,
            field_name="support_level",
            context=f"run manifest source '{source}'",
            source=source,
        )
        if support_level != SupportLevel.COMPLETE.value:
            message = f"source '{source}' support level is {support_level}"
            if support_limitation_summary is not None:
                message = f"{message}: {support_limitation_summary}"
            _add_finding(
                findings,
                level=ValidationLevel.WARNING,
                code="degraded_support_level",
                message=message,
                source=source,
            )

    status = _required_string(
        source_payload,
        "status",
        findings=findings,
        context=f"run manifest source '{source}'",
        code="missing_required_field",
        source=source,
    )
    if status is not None:
        _validate_enum(
            status,
            SourceRunStatus,
            findings=findings,
            field_name="status",
            context=f"run manifest source '{source}'",
            source=source,
        )
        if status != SourceRunStatus.COMPLETE.value:
            _add_finding(
                findings,
                level=ValidationLevel.WARNING,
                code="degraded_source_status",
                message=f"source '{source}' status is {status}",
                source=source,
            )

    source_archive_root = _required_absolute_path(
        source_payload,
        "archive_root",
        findings=findings,
        context=f"run manifest source '{source}'",
        code="missing_required_field",
        source=source,
    )
    if source_archive_root is not None:
        _validate_external_path(
            source_archive_root,
            repo_root=repo_root,
            findings=findings,
            label=f"source '{source}' archive_root",
            source=source,
        )
        if source_archive_root != manifest_archive_root:
            _add_finding(
                findings,
                level=ValidationLevel.ERROR,
                code="invalid_field",
                message=(
                    f"source '{source}' archive_root '{source_archive_root}' does not "
                    f"match manifest archive_root '{manifest_archive_root}'"
                ),
                source=source,
                path=source_archive_root,
            )

    output_path = _required_nullable_absolute_path(
        source_payload,
        "output_path",
        findings=findings,
        context=f"run manifest source '{source}'",
        code="missing_required_field",
        source=source,
    )

    conversation_count = _required_int(
        source_payload,
        "conversation_count",
        findings=findings,
        context=f"run manifest source '{source}'",
        code="missing_required_field",
        source=source,
    )
    skipped_conversation_count = _required_int(
        source_payload,
        "skipped_conversation_count",
        findings=findings,
        context=f"run manifest source '{source}'",
        code="missing_required_field",
        source=source,
    )
    written_conversation_count = _required_int(
        source_payload,
        "written_conversation_count",
        findings=findings,
        context=f"run manifest source '{source}'",
        code="missing_required_field",
        source=source,
    )
    declared_message_count = _required_int(
        source_payload,
        "message_count",
        findings=findings,
        context=f"run manifest source '{source}'",
        code="missing_required_field",
        source=source,
    )

    failed = _required_bool(
        source_payload,
        "failed",
        findings=findings,
        context=f"run manifest source '{source}'",
        code="missing_required_field",
        source=source,
    )
    partial = _required_bool(
        source_payload,
        "partial",
        findings=findings,
        context=f"run manifest source '{source}'",
        code="missing_required_field",
        source=source,
    )
    unsupported = _required_bool(
        source_payload,
        "unsupported",
        findings=findings,
        context=f"run manifest source '{source}'",
        code="missing_required_field",
        source=source,
    )

    if (
        status is not None
        and failed is not None
        and partial is not None
        and unsupported is not None
    ):
        expected_status = _expected_source_status(
            failed=failed,
            partial=partial,
            unsupported=unsupported,
        )
        if status != expected_status:
            _add_finding(
                findings,
                level=ValidationLevel.ERROR,
                code="status_mismatch",
                message=(
                    f"source '{source}' status is {status} but flags imply "
                    f"{expected_status}"
                ),
                source=source,
            )

    row_count = 0
    actual_message_count = 0
    if output_path is None:
        if status in (SourceRunStatus.COMPLETE.value, SourceRunStatus.PARTIAL.value):
            _add_finding(
                findings,
                level=ValidationLevel.ERROR,
                code="missing_required_field",
                message=f"source '{source}' must declare output_path for status {status}",
                source=source,
            )
        if (written_conversation_count or 0) != 0:
            _add_finding(
                findings,
                level=ValidationLevel.ERROR,
                code="count_mismatch",
                message=(
                    f"source '{source}' declares written_conversation_count "
                    f"{written_conversation_count} but output_path is null"
                ),
                source=source,
            )
    else:
        _validate_external_path(
            output_path,
            repo_root=repo_root,
            findings=findings,
            label=f"source '{source}' output_path",
            source=source,
        )
        if not _is_within(output_path, manifest_archive_root):
            _add_finding(
                findings,
                level=ValidationLevel.ERROR,
                code="invalid_field",
                message=(
                    f"source '{source}' output_path '{output_path}' is not inside "
                    f"archive_root '{manifest_archive_root}'"
                ),
                source=source,
                path=output_path,
            )
        if not output_path.exists():
            _add_finding(
                findings,
                level=ValidationLevel.ERROR,
                code="missing_file",
                message=f"source '{source}' output_path does not exist: {output_path}",
                source=source,
                path=output_path,
            )
        elif not output_path.is_file():
            _add_finding(
                findings,
                level=ValidationLevel.ERROR,
                code="invalid_field",
                message=f"source '{source}' output_path is not a file: {output_path}",
                source=source,
                path=output_path,
            )
        else:
            row_count, actual_message_count = _validate_output_rows(
                output_path,
                source=source,
                findings=findings,
            )

    if (
        conversation_count is not None
        and skipped_conversation_count is not None
        and written_conversation_count is not None
        and conversation_count != skipped_conversation_count + written_conversation_count
    ):
        _add_finding(
            findings,
            level=ValidationLevel.ERROR,
            code="count_mismatch",
            message=(
                f"source '{source}' conversation_count is {conversation_count} but "
                f"skipped + written equals "
                f"{skipped_conversation_count + written_conversation_count}"
            ),
            source=source,
        )

    if written_conversation_count is not None and written_conversation_count != row_count:
        _add_finding(
            findings,
            level=ValidationLevel.ERROR,
            code="count_mismatch",
            message=(
                f"source '{source}' written_conversation_count is "
                f"{written_conversation_count} but output has {row_count} rows"
            ),
            source=source,
            path=output_path,
        )

    if declared_message_count is not None and declared_message_count != actual_message_count:
        _add_finding(
            findings,
            level=ValidationLevel.ERROR,
            code="count_mismatch",
            message=(
                f"source '{source}' message_count is {declared_message_count} but "
                f"output contains {actual_message_count} messages"
            ),
            source=source,
            path=output_path,
        )

    parser_assumption_summary: str | None = None
    drift_suspected = False
    parser_assumption_report = inspect_parser_assumptions(
        source,
        input_roots=input_roots,
        repo_path=repo_root,
    )
    if parser_assumption_report.drift_suspected:
        drift_suspected = True
        parser_assumption_summary = parser_assumption_report.summary
        _add_finding(
            findings,
            level=ValidationLevel.WARNING,
            code="drift_suspected",
            message=f"source '{source}' {parser_assumption_report.summary}",
            source=source,
            path=output_path,
        )

    source_level = _worst_level(findings[start_index:])
    if source_level == ValidationLevel.SUCCESS:
        _add_finding(
            findings,
            level=ValidationLevel.SUCCESS,
            code="source_validated",
            message=f"source '{source}' output and manifest entry validated successfully",
            source=source,
            path=output_path,
        )

    return ValidatedSource(
        source=source,
        support_level=support_level,
        status=status,
        failed=failed,
        output_path=output_path,
        support_limitation_summary=support_limitation_summary,
        support_limitations=support_limitations,
        row_count=row_count,
        actual_message_count=actual_message_count,
        declared_conversation_count=conversation_count,
        declared_skipped_conversation_count=skipped_conversation_count,
        declared_written_conversation_count=written_conversation_count,
        declared_message_count=declared_message_count,
        drift_suspected=drift_suspected,
        parser_assumption_summary=parser_assumption_summary,
        validation_status=source_level,
    )


def _validate_output_rows(
    output_path: Path,
    *,
    source: str,
    findings: list[ValidationFinding],
) -> tuple[int, int]:
    row_count = 0
    message_count = 0

    try:
        lines = output_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        _add_finding(
            findings,
            level=ValidationLevel.ERROR,
            code="missing_file",
            message=f"failed to read source '{source}' output_path: {exc}",
            source=source,
            path=output_path,
        )
        return row_count, message_count

    for row_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        row_count += 1
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            _add_finding(
                findings,
                level=ValidationLevel.ERROR,
                code="malformed_row",
                message=f"source '{source}' row {row_number} is not valid JSON",
                source=source,
                path=output_path,
                row_number=row_number,
            )
            continue

        if not isinstance(payload, dict):
            _add_finding(
                findings,
                level=ValidationLevel.ERROR,
                code="malformed_row",
                message=f"source '{source}' row {row_number} must be an object",
                source=source,
                path=output_path,
                row_number=row_number,
            )
            continue

        row_source = _required_string(
            payload,
            "source",
            findings=findings,
            context=f"source '{source}' row {row_number}",
            code="missing_required_field",
            source=source,
            path=output_path,
            row_number=row_number,
        )
        if row_source is not None and row_source != source:
            _add_finding(
                findings,
                level=ValidationLevel.ERROR,
                code="invalid_field",
                message=(
                    f"source '{source}' row {row_number} declares source "
                    f"'{row_source}'"
                ),
                source=source,
                path=output_path,
                row_number=row_number,
            )

        _required_string(
            payload,
            "execution_context",
            findings=findings,
            context=f"source '{source}' row {row_number}",
            code="missing_required_field",
            source=source,
            path=output_path,
            row_number=row_number,
        )
        _required_string(
            payload,
            "collected_at",
            findings=findings,
            context=f"source '{source}' row {row_number}",
            code="missing_required_field",
            source=source,
            path=output_path,
            row_number=row_number,
        )

        contract = _required_object(
            payload,
            "contract",
            findings=findings,
            context=f"source '{source}' row {row_number}",
            code="missing_required_field",
            source=source,
            path=output_path,
            row_number=row_number,
        )
        if contract is not None:
            _required_string(
                contract,
                "schema_version",
                findings=findings,
                context=f"source '{source}' row {row_number} contract",
                code="missing_required_field",
                source=source,
                path=output_path,
                row_number=row_number,
            )

        transcript_completeness = payload.get(
            "transcript_completeness",
            TranscriptCompleteness.COMPLETE.value,
        )
        if isinstance(transcript_completeness, str):
            _validate_enum(
                transcript_completeness,
                TranscriptCompleteness,
                findings=findings,
                field_name="transcript_completeness",
                context=f"source '{source}' row {row_number}",
                source=source,
                path=output_path,
                row_number=row_number,
            )
            if transcript_completeness != TranscriptCompleteness.COMPLETE.value:
                _add_finding(
                    findings,
                    level=ValidationLevel.WARNING,
                    code="incomplete_transcript",
                    message=(
                        f"source '{source}' row {row_number} transcript completeness is "
                        f"{transcript_completeness}"
                    ),
                    source=source,
                    path=output_path,
                    row_number=row_number,
                )
        else:
            _add_finding(
                findings,
                level=ValidationLevel.ERROR,
                code="invalid_field",
                message=(
                    f"source '{source}' row {row_number} field "
                    f"'transcript_completeness' must be a string"
                ),
                source=source,
                path=output_path,
                row_number=row_number,
            )

        messages = _required_list(
            payload,
            "messages",
            findings=findings,
            context=f"source '{source}' row {row_number}",
            code="missing_required_field",
            source=source,
            path=output_path,
            row_number=row_number,
        )
        if messages is None:
            continue

        message_count += len(messages)
        for message_index, message in enumerate(messages, start=1):
            if not isinstance(message, dict):
                _add_finding(
                    findings,
                    level=ValidationLevel.ERROR,
                    code="malformed_row",
                    message=(
                        f"source '{source}' row {row_number} message {message_index} "
                        "must be an object"
                    ),
                    source=source,
                    path=output_path,
                    row_number=row_number,
                )
                continue

            role = _required_string(
                message,
                "role",
                findings=findings,
                context=(
                    f"source '{source}' row {row_number} message {message_index}"
                ),
                code="missing_required_field",
                source=source,
                path=output_path,
                row_number=row_number,
            )
            if role is not None:
                _validate_enum(
                    role,
                    MessageRole,
                    findings=findings,
                    field_name="role",
                    context=(
                        f"source '{source}' row {row_number} message {message_index}"
                    ),
                    source=source,
                    path=output_path,
                    row_number=row_number,
                )

            has_text = "text" in message
            has_images = "images" in message
            if not has_text and not has_images:
                _add_finding(
                    findings,
                    level=ValidationLevel.ERROR,
                    code="missing_required_field",
                    message=(
                        f"source '{source}' row {row_number} message {message_index} "
                        "must include text or images"
                    ),
                    source=source,
                    path=output_path,
                    row_number=row_number,
                )
            if has_text:
                text = message.get("text")
                if text is not None and not isinstance(text, str):
                    _add_finding(
                        findings,
                        level=ValidationLevel.ERROR,
                        code="invalid_field",
                        message=(
                            f"source '{source}' row {row_number} message "
                            f"{message_index} field 'text' must be a string or null"
                        ),
                        source=source,
                        path=output_path,
                        row_number=row_number,
                    )
            if has_images:
                images = message.get("images")
                if not isinstance(images, list):
                    _add_finding(
                        findings,
                        level=ValidationLevel.ERROR,
                        code="invalid_field",
                        message=(
                            f"source '{source}' row {row_number} message "
                            f"{message_index} field 'images' must be an array"
                        ),
                        source=source,
                        path=output_path,
                        row_number=row_number,
                    )
                else:
                    for image_index, image in enumerate(images, start=1):
                        if not isinstance(image, dict):
                            _add_finding(
                                findings,
                                level=ValidationLevel.ERROR,
                                code="invalid_field",
                                message=(
                                    f"source '{source}' row {row_number} message "
                                    f"{message_index} image {image_index} must be an object"
                                ),
                                source=source,
                                path=output_path,
                                row_number=row_number,
                            )
                            continue
                        _required_string(
                            image,
                            "source",
                            findings=findings,
                            context=(
                                f"source '{source}' row {row_number} message "
                                f"{message_index} image {image_index}"
                            ),
                            code="missing_required_field",
                            source=source,
                            path=output_path,
                            row_number=row_number,
                        )

    return row_count, message_count


def _load_json_object(
    manifest_path: Path,
    *,
    findings: list[ValidationFinding],
    run_id: str,
) -> dict[str, object] | None:
    if not manifest_path.exists():
        _add_finding(
            findings,
            level=ValidationLevel.ERROR,
            code="missing_file",
            message=f"run manifest does not exist for run '{run_id}': {manifest_path}",
            path=manifest_path,
        )
        return None

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        _add_finding(
            findings,
            level=ValidationLevel.ERROR,
            code="invalid_field",
            message=f"run manifest is not valid JSON: {manifest_path}",
            path=manifest_path,
        )
        return None
    except OSError as exc:
        _add_finding(
            findings,
            level=ValidationLevel.ERROR,
            code="missing_file",
            message=f"failed to read run manifest {manifest_path}: {exc}",
            path=manifest_path,
        )
        return None

    if not isinstance(payload, dict):
        _add_finding(
            findings,
            level=ValidationLevel.ERROR,
            code="invalid_field",
            message=f"run manifest root must be an object: {manifest_path}",
            path=manifest_path,
        )
        return None
    return payload


def _compare_manifest_total(
    *,
    findings: list[ValidationFinding],
    manifest_path: Path,
    field_name: str,
    declared: int | None,
    actual: int,
) -> None:
    if declared is None:
        return
    if declared != actual:
        _add_finding(
            findings,
            level=ValidationLevel.ERROR,
            code="count_mismatch",
            message=f"run manifest {field_name} is {declared} but sources sum to {actual}",
            path=manifest_path,
        )


def _validate_external_path(
    path: Path,
    *,
    repo_root: Path,
    findings: list[ValidationFinding],
    label: str,
    source: str | None = None,
) -> None:
    try:
        ArchiveTargetPolicy(repo_root=repo_root).validate(path)
    except ValueError as exc:
        _add_finding(
            findings,
            level=ValidationLevel.ERROR,
            code="external_archive_only_violation",
            message=f"{label} violates external-only archive policy: {exc}",
            source=source,
            path=path,
        )


def _required_string(
    payload: dict[str, object],
    key: str,
    *,
    findings: list[ValidationFinding],
    context: str,
    code: str,
    source: str | None = None,
    path: Path | None = None,
    row_number: int | None = None,
) -> str | None:
    if key not in payload:
        _add_finding(
            findings,
            level=ValidationLevel.ERROR,
            code=code,
            message=f"{context} is missing required field '{key}'",
            source=source,
            path=path,
            row_number=row_number,
        )
        return None
    value = payload.get(key)
    if isinstance(value, str) and value:
        return value
    _add_finding(
        findings,
        level=ValidationLevel.ERROR,
        code="invalid_field",
        message=f"{context} field '{key}' must be a non-empty string",
        source=source,
        path=path,
        row_number=row_number,
    )
    return None


def _optional_string(payload: dict[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, str) and value:
        return value
    return None


def _optional_string_sequence(
    payload: dict[str, object],
    key: str,
) -> tuple[str, ...]:
    value = payload.get(key)
    if value is None:
        return ()
    if not isinstance(value, list):
        return ()
    normalized: list[str] = []
    for entry in value:
        if isinstance(entry, str) and entry:
            normalized.append(entry)
    return tuple(normalized)


def _optional_absolute_path_sequence(
    payload: dict[str, object],
    key: str,
    *,
    findings: list[ValidationFinding],
    context: str,
    source: str | None = None,
) -> tuple[Path, ...]:
    value = payload.get(key)
    if value is None:
        return ()
    if not isinstance(value, list):
        _add_finding(
            findings,
            level=ValidationLevel.ERROR,
            code="invalid_field",
            message=f"{context} field '{key}' must be an array of absolute paths",
            source=source,
        )
        return ()

    paths: list[Path] = []
    for index, entry in enumerate(value, start=1):
        if not isinstance(entry, str) or not entry:
            _add_finding(
                findings,
                level=ValidationLevel.ERROR,
                code="invalid_field",
                message=(
                    f"{context} field '{key}' entry {index} must be a non-empty absolute path string"
                ),
                source=source,
            )
            continue
        candidate = Path(entry)
        if not candidate.is_absolute():
            _add_finding(
                findings,
                level=ValidationLevel.ERROR,
                code="invalid_field",
                message=(
                    f"{context} field '{key}' entry {index} must be an absolute path"
                ),
                source=source,
            )
            continue
        paths.append(candidate.resolve(strict=False))
    return tuple(paths)


def _required_int(
    payload: dict[str, object],
    key: str,
    *,
    findings: list[ValidationFinding],
    context: str,
    code: str,
    source: str | None = None,
) -> int | None:
    if key not in payload:
        _add_finding(
            findings,
            level=ValidationLevel.ERROR,
            code=code,
            message=f"{context} is missing required field '{key}'",
            source=source,
        )
        return None
    value = payload.get(key)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    _add_finding(
        findings,
        level=ValidationLevel.ERROR,
        code="invalid_field",
        message=f"{context} field '{key}' must be an integer",
        source=source,
    )
    return None


def _required_bool(
    payload: dict[str, object],
    key: str,
    *,
    findings: list[ValidationFinding],
    context: str,
    code: str,
    source: str | None = None,
) -> bool | None:
    if key not in payload:
        _add_finding(
            findings,
            level=ValidationLevel.ERROR,
            code=code,
            message=f"{context} is missing required field '{key}'",
            source=source,
        )
        return None
    value = payload.get(key)
    if isinstance(value, bool):
        return value
    _add_finding(
        findings,
        level=ValidationLevel.ERROR,
        code="invalid_field",
        message=f"{context} field '{key}' must be a boolean",
        source=source,
    )
    return None


def _required_object(
    payload: dict[str, object],
    key: str,
    *,
    findings: list[ValidationFinding],
    context: str,
    code: str,
    source: str | None = None,
    path: Path | None = None,
    row_number: int | None = None,
) -> dict[str, object] | None:
    if key not in payload:
        _add_finding(
            findings,
            level=ValidationLevel.ERROR,
            code=code,
            message=f"{context} is missing required field '{key}'",
            source=source,
            path=path,
            row_number=row_number,
        )
        return None
    value = payload.get(key)
    if isinstance(value, dict):
        return value
    _add_finding(
        findings,
        level=ValidationLevel.ERROR,
        code="invalid_field",
        message=f"{context} field '{key}' must be an object",
        source=source,
        path=path,
        row_number=row_number,
    )
    return None


def _required_list(
    payload: dict[str, object],
    key: str,
    *,
    findings: list[ValidationFinding],
    context: str,
    code: str,
    source: str | None = None,
    path: Path | None = None,
    row_number: int | None = None,
) -> list[object] | None:
    if key not in payload:
        _add_finding(
            findings,
            level=ValidationLevel.ERROR,
            code=code,
            message=f"{context} is missing required field '{key}'",
            source=source,
            path=path,
            row_number=row_number,
        )
        return None
    value = payload.get(key)
    if isinstance(value, list):
        return value
    _add_finding(
        findings,
        level=ValidationLevel.ERROR,
        code="invalid_field",
        message=f"{context} field '{key}' must be an array",
        source=source,
        path=path,
        row_number=row_number,
    )
    return None


def _required_absolute_path(
    payload: dict[str, object],
    key: str,
    *,
    findings: list[ValidationFinding],
    context: str,
    code: str,
    source: str | None = None,
) -> Path | None:
    if key not in payload:
        _add_finding(
            findings,
            level=ValidationLevel.ERROR,
            code=code,
            message=f"{context} is missing required field '{key}'",
            source=source,
        )
        return None
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        _add_finding(
            findings,
            level=ValidationLevel.ERROR,
            code="invalid_field",
            message=f"{context} field '{key}' must be a non-empty absolute path string",
            source=source,
        )
        return None
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        _add_finding(
            findings,
            level=ValidationLevel.ERROR,
            code="invalid_field",
            message=f"{context} field '{key}' must be an absolute path",
            source=source,
        )
        return None
    return candidate.resolve(strict=False)


def _required_nullable_absolute_path(
    payload: dict[str, object],
    key: str,
    *,
    findings: list[ValidationFinding],
    context: str,
    code: str,
    source: str | None = None,
) -> Path | None:
    if key not in payload:
        _add_finding(
            findings,
            level=ValidationLevel.ERROR,
            code=code,
            message=f"{context} is missing required field '{key}'",
            source=source,
        )
        return None
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        _add_finding(
            findings,
            level=ValidationLevel.ERROR,
            code="invalid_field",
            message=f"{context} field '{key}' must be null or an absolute path string",
            source=source,
        )
        return None
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        _add_finding(
            findings,
            level=ValidationLevel.ERROR,
            code="invalid_field",
            message=f"{context} field '{key}' must be an absolute path",
            source=source,
        )
        return None
    return candidate.resolve(strict=False)


def _validate_enum(
    value: str,
    enum_type: type[StrEnum],
    *,
    findings: list[ValidationFinding],
    field_name: str,
    context: str,
    source: str | None = None,
    path: Path | None = None,
    row_number: int | None = None,
) -> None:
    allowed = {member.value for member in enum_type}
    if value in allowed:
        return
    _add_finding(
        findings,
        level=ValidationLevel.ERROR,
        code="invalid_enum",
        message=(
            f"{context} field '{field_name}' has invalid value '{value}'; expected one "
            f"of {sorted(allowed)}"
        ),
        source=source,
        path=path,
        row_number=row_number,
    )


def _expected_source_status(
    *,
    failed: bool,
    partial: bool,
    unsupported: bool,
) -> str:
    if failed:
        return SourceRunStatus.FAILED.value
    if unsupported:
        return SourceRunStatus.UNSUPPORTED.value
    if partial:
        return SourceRunStatus.PARTIAL.value
    return SourceRunStatus.COMPLETE.value


def _is_within(candidate: Path, root: Path) -> bool:
    return candidate == root or root in candidate.parents


def _worst_level(
    findings: list[ValidationFinding] | tuple[ValidationFinding, ...],
    *,
    include_suppressed: bool = False,
) -> ValidationLevel:
    considered_findings = (
        findings
        if include_suppressed
        else tuple(finding for finding in findings if not finding.suppressed)
    )
    if any(finding.level == ValidationLevel.ERROR for finding in considered_findings):
        return ValidationLevel.ERROR
    if any(finding.level == ValidationLevel.WARNING for finding in considered_findings):
        return ValidationLevel.WARNING
    return ValidationLevel.SUCCESS


def _apply_baseline_policy(
    report: ValidationReport,
    *,
    baseline_policy: BaselinePolicy,
) -> ValidationReport:
    source_metadata = {
        source.source: (source.support_level, source.status) for source in report.sources
    }
    findings = tuple(
        _apply_baseline_to_finding(
            finding,
            baseline_policy=baseline_policy,
            source_metadata=source_metadata,
        )
        for finding in report.findings
    )
    findings_by_source: dict[str, list[ValidationFinding]] = {}
    for finding in findings:
        if finding.source is None:
            continue
        findings_by_source.setdefault(finding.source, []).append(finding)

    sources = tuple(
        _apply_baseline_to_source(
            source,
            findings=tuple(findings_by_source.get(source.source, ())),
        )
        for source in report.sources
    )
    return replace(
        report,
        sources=sources,
        findings=findings,
        baseline_path=baseline_policy.path,
        baseline_entry_count=baseline_policy.entry_count,
    )


def _apply_baseline_to_source(
    source: ValidatedSource,
    *,
    findings: tuple[ValidationFinding, ...],
) -> ValidatedSource:
    raw_status = source.validation_status
    active_status = _worst_level(findings)
    suppressed_warning_count = sum(
        1
        for finding in findings
        if finding.level == ValidationLevel.WARNING and finding.suppressed
    )
    return replace(
        source,
        validation_status=active_status,
        raw_validation_status=raw_status,
        suppressed_warning_count=suppressed_warning_count,
    )


def _apply_baseline_to_finding(
    finding: ValidationFinding,
    *,
    baseline_policy: BaselinePolicy,
    source_metadata: dict[str, tuple[str | None, str | None]],
) -> ValidationFinding:
    matched_entry = None
    if (
        finding.level == ValidationLevel.WARNING
        and finding.source is not None
        and finding.code in {"degraded_support_level", "degraded_source_status"}
    ):
        support_level, status = source_metadata.get(finding.source, (None, None))
        matched_entry = baseline_policy.match_degraded_source(
            source=finding.source,
            support_level=support_level,
            status=status,
        )
    elif finding.level == ValidationLevel.WARNING:
        matched_entry = baseline_policy.match_finding(
            report=BaselineReport.VALIDATE,
            source=finding.source,
            code=finding.code,
            level=finding.level.value,
        )

    if matched_entry is None:
        return finding
    return replace(
        finding,
        suppressed=True,
        suppression_entry_id=matched_entry.id,
        suppression_kind=matched_entry.kind.value,
        suppression_reason=matched_entry.reason,
    )


def _add_finding(
    findings: list[ValidationFinding],
    *,
    level: ValidationLevel,
    code: str,
    message: str,
    source: str | None = None,
    path: Path | None = None,
    row_number: int | None = None,
) -> None:
    findings.append(
        ValidationFinding(
            level=level,
            code=code,
            message=message,
            source=source,
            path=path,
            row_number=row_number,
        )
    )


__all__ = [
    "ValidatedSource",
    "ValidationFinding",
    "ValidationLevel",
    "ValidationReport",
    "validate_run",
]
