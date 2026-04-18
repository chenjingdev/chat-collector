from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

from .archive_inspect import ARCHIVE_OUTPUT_GLOB
from .archive_rewrite import CANONICAL_OUTPUT_TEMPLATE
from .baseline_policy import BaselinePolicy, BaselineReport
from .models import MessageRole, SCHEMA_VERSION, TranscriptCompleteness
from .runner import MANIFEST_FILENAME, RUNS_DIRECTORY
from .validate import (
    ValidationFinding,
    ValidationLevel,
    _add_finding,
    _required_list,
    _required_object,
    _required_string,
    _validate_enum,
    _worst_level,
)


class ArchiveVerifyError(ValueError):
    """Raised when archive verify cannot scan the requested archive root."""


@dataclass(frozen=True, slots=True)
class ArchiveVerifiedFile:
    path: Path
    status: ValidationLevel
    row_count: int
    verified_row_count: int
    bad_row_count: int
    finding_count: int
    warning_count: int
    error_count: int
    manifest_linked: bool
    orphan: bool
    raw_status: ValidationLevel | None = None
    suppressed_warning_count: int = 0

    def to_dict(self) -> dict[str, object]:
        payload = {
            "path": str(self.path),
            "status": self.status.value,
            "row_count": self.row_count,
            "verified_row_count": self.verified_row_count,
            "bad_row_count": self.bad_row_count,
            "finding_count": self.finding_count,
            "warning_count": self.warning_count,
            "error_count": self.error_count,
            "manifest_linked": self.manifest_linked,
            "orphan": self.orphan,
        }
        raw_status = self.raw_status or self.status
        if raw_status != self.status:
            payload["raw_status"] = raw_status.value
        if self.suppressed_warning_count > 0:
            payload["suppressed_warning_count"] = self.suppressed_warning_count
        return payload


@dataclass(frozen=True, slots=True)
class ArchiveVerifySourceReport:
    source: str
    status: ValidationLevel
    file_count: int
    row_count: int
    verified_row_count: int
    bad_row_count: int
    orphan_file_count: int
    finding_count: int
    warning_count: int
    error_count: int
    files: tuple[ArchiveVerifiedFile, ...]
    raw_status: ValidationLevel | None = None
    suppressed_warning_count: int = 0

    def to_dict(self) -> dict[str, object]:
        payload = {
            "status": self.status.value,
            "file_count": self.file_count,
            "row_count": self.row_count,
            "verified_row_count": self.verified_row_count,
            "bad_row_count": self.bad_row_count,
            "orphan_file_count": self.orphan_file_count,
            "finding_count": self.finding_count,
            "warning_count": self.warning_count,
            "error_count": self.error_count,
            "files": [file_report.to_dict() for file_report in self.files],
        }
        raw_status = self.raw_status or self.status
        if raw_status != self.status:
            payload["raw_status"] = raw_status.value
        if self.suppressed_warning_count > 0:
            payload["suppressed_warning_count"] = self.suppressed_warning_count
        return payload


@dataclass(frozen=True, slots=True)
class ArchiveVerifyReport:
    archive_root: Path
    source_filter: str | None
    manifest_count: int
    linked_output_file_count: int
    sources: tuple[ArchiveVerifySourceReport, ...]
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
    def file_count(self) -> int:
        return sum(source.file_count for source in self.sources)

    @property
    def row_count(self) -> int:
        return sum(source.row_count for source in self.sources)

    @property
    def verified_row_count(self) -> int:
        return sum(source.verified_row_count for source in self.sources)

    @property
    def bad_row_count(self) -> int:
        return sum(source.bad_row_count for source in self.sources)

    @property
    def orphan_file_count(self) -> int:
        return sum(source.orphan_file_count for source in self.sources)

    @property
    def finding_count(self) -> int:
        return len(self.findings)

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
            "archive_root": str(self.archive_root),
            "source_filter": self.source_filter,
            "status": self.status.value,
            "manifest_count": self.manifest_count,
            "linked_output_file_count": self.linked_output_file_count,
            "source_count": len(self.sources),
            "file_count": self.file_count,
            "row_count": self.row_count,
            "verified_row_count": self.verified_row_count,
            "bad_row_count": self.bad_row_count,
            "orphan_file_count": self.orphan_file_count,
            "finding_count": self.finding_count,
            "warning_count": self.warning_count,
            "error_count": self.error_count,
            "sources": {
                source_report.source: source_report.to_dict()
                for source_report in self.sources
            },
            "findings": [finding.to_dict() for finding in self.findings],
        }
        if self.suppressed_warning_count > 0:
            payload["raw_status"] = self.raw_status.value
            payload["raw_warning_count"] = self.raw_warning_count
            payload["suppressed_warning_count"] = self.suppressed_warning_count
        if self.baseline_path is not None:
            payload["baseline"] = {
                "path": str(self.baseline_path),
                "entry_count": self.baseline_entry_count,
            }
        return payload


def verify_archive(
    archive_root: Path,
    *,
    source: str | None = None,
    baseline_policy: BaselinePolicy | None = None,
) -> ArchiveVerifyReport:
    resolved_archive_root = archive_root.expanduser().resolve(strict=False)
    _validate_archive_root_directory(resolved_archive_root)

    findings: list[ValidationFinding] = []
    manifest_count, linked_output_paths = _load_manifest_output_paths(
        resolved_archive_root,
        findings=findings,
    )
    source_reports = tuple(
        _verify_source(
            resolved_archive_root,
            source_name=source_name,
            linked_output_paths=linked_output_paths,
            findings=findings,
        )
        for source_name in _select_sources(resolved_archive_root, source=source)
    )
    report = ArchiveVerifyReport(
        archive_root=resolved_archive_root,
        source_filter=source,
        manifest_count=manifest_count,
        linked_output_file_count=len(linked_output_paths),
        sources=source_reports,
        findings=tuple(findings),
    )
    if baseline_policy is not None:
        return _apply_baseline_policy(report, baseline_policy=baseline_policy)
    return report


def _validate_archive_root_directory(archive_root: Path) -> None:
    if not archive_root.exists():
        raise ArchiveVerifyError(f"archive root does not exist: {archive_root}")
    if not archive_root.is_dir():
        raise ArchiveVerifyError(f"archive root is not a directory: {archive_root}")

    runs_dir = archive_root / RUNS_DIRECTORY
    if runs_dir.exists() and not runs_dir.is_dir():
        raise ArchiveVerifyError(f"archive runs path is not a directory: {runs_dir}")


def _load_manifest_output_paths(
    archive_root: Path,
    *,
    findings: list[ValidationFinding],
) -> tuple[int, frozenset[Path]]:
    runs_dir = archive_root / RUNS_DIRECTORY
    if not runs_dir.is_dir():
        return 0, frozenset()

    manifest_paths = tuple(sorted(runs_dir.glob(f"*/{MANIFEST_FILENAME}")))
    linked_output_paths: set[Path] = set()
    for manifest_path in manifest_paths:
        payload = _load_manifest_payload(manifest_path, findings=findings)
        if payload is None:
            continue
        sources_payload = payload.get("sources")
        if not isinstance(sources_payload, list):
            _add_finding(
                findings,
                level=ValidationLevel.WARNING,
                code="invalid_manifest_reference",
                message=(
                    f"run manifest field 'sources' must be an array for orphan checks: "
                    f"{manifest_path}"
                ),
                path=manifest_path,
            )
            continue

        for index, source_payload in enumerate(sources_payload, start=1):
            if not isinstance(source_payload, dict):
                _add_finding(
                    findings,
                    level=ValidationLevel.WARNING,
                    code="invalid_manifest_reference",
                    message=(
                        f"run manifest source entry {index} must be an object for orphan "
                        f"checks: {manifest_path}"
                    ),
                    path=manifest_path,
                )
                continue
            output_path = source_payload.get("output_path")
            if output_path is None:
                continue
            if not isinstance(output_path, str) or not output_path:
                _add_finding(
                    findings,
                    level=ValidationLevel.WARNING,
                    code="invalid_manifest_reference",
                    message=(
                        f"run manifest source entry {index} output_path must be a non-empty "
                        f"absolute path string: {manifest_path}"
                    ),
                    path=manifest_path,
                )
                continue
            candidate = Path(output_path).expanduser()
            if not candidate.is_absolute():
                _add_finding(
                    findings,
                    level=ValidationLevel.WARNING,
                    code="invalid_manifest_reference",
                    message=(
                        f"run manifest source entry {index} output_path must be absolute: "
                        f"{manifest_path}"
                    ),
                    path=manifest_path,
                )
                continue
            resolved_output_path = candidate.resolve(strict=False)
            if not _is_within(resolved_output_path, archive_root):
                _add_finding(
                    findings,
                    level=ValidationLevel.WARNING,
                    code="invalid_manifest_reference",
                    message=(
                        f"run manifest source entry {index} output_path is outside the "
                        f"archive root: {resolved_output_path}"
                    ),
                    path=manifest_path,
                )
                continue
            linked_output_paths.add(resolved_output_path)

    return len(manifest_paths), frozenset(linked_output_paths)


def _load_manifest_payload(
    manifest_path: Path,
    *,
    findings: list[ValidationFinding],
) -> dict[str, object] | None:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        _add_finding(
            findings,
            level=ValidationLevel.WARNING,
            code="invalid_manifest_reference",
            message=f"run manifest is not valid JSON for orphan checks: {manifest_path}",
            path=manifest_path,
        )
        return None
    except OSError as exc:
        _add_finding(
            findings,
            level=ValidationLevel.WARNING,
            code="invalid_manifest_reference",
            message=f"failed to read run manifest for orphan checks: {exc}",
            path=manifest_path,
        )
        return None

    if not isinstance(payload, dict):
        _add_finding(
            findings,
            level=ValidationLevel.WARNING,
            code="invalid_manifest_reference",
            message=f"run manifest root must be an object for orphan checks: {manifest_path}",
            path=manifest_path,
        )
        return None
    return payload


def _select_sources(archive_root: Path, *, source: str | None) -> tuple[str, ...]:
    if source is not None:
        source_dir = archive_root / source
        if not source_dir.is_dir():
            return ()
        return (source,)
    return tuple(
        path.name
        for path in sorted(archive_root.iterdir())
        if path.is_dir()
        and path.name != RUNS_DIRECTORY
        and any(path.glob(ARCHIVE_OUTPUT_GLOB))
    )


def _verify_source(
    archive_root: Path,
    *,
    source_name: str,
    linked_output_paths: frozenset[Path],
    findings: list[ValidationFinding],
) -> ArchiveVerifySourceReport:
    source_dir = archive_root / source_name
    output_paths = tuple(sorted(source_dir.glob(ARCHIVE_OUTPUT_GLOB)))
    source_start_index = len(findings)
    file_reports = tuple(
        _verify_output_file(
            output_path,
            source_name=source_name,
            linked_output_paths=linked_output_paths,
            findings=findings,
        )
        for output_path in output_paths
    )
    source_findings = findings[source_start_index:]
    warning_count = sum(
        1 for finding in source_findings if finding.level == ValidationLevel.WARNING
    )
    error_count = sum(
        1 for finding in source_findings if finding.level == ValidationLevel.ERROR
    )
    return ArchiveVerifySourceReport(
        source=source_name,
        status=_worst_level(source_findings),
        file_count=len(file_reports),
        row_count=sum(file_report.row_count for file_report in file_reports),
        verified_row_count=sum(file_report.verified_row_count for file_report in file_reports),
        bad_row_count=sum(file_report.bad_row_count for file_report in file_reports),
        orphan_file_count=sum(1 for file_report in file_reports if file_report.orphan),
        finding_count=len(source_findings),
        warning_count=warning_count,
        error_count=error_count,
        files=file_reports,
    )


def _verify_output_file(
    output_path: Path,
    *,
    source_name: str,
    linked_output_paths: frozenset[Path],
    findings: list[ValidationFinding],
) -> ArchiveVerifiedFile:
    file_start_index = len(findings)
    manifest_linked = output_path in linked_output_paths
    orphan = not manifest_linked and not _is_canonical_output_path(output_path, source=source_name)
    if orphan:
        _add_finding(
            findings,
            level=ValidationLevel.WARNING,
            code="orphan_output_file",
            message=(
                f"source '{source_name}' output file is not referenced by any run manifest: "
                f"{output_path}"
            ),
            source=source_name,
            path=output_path,
        )

    row_count = 0
    verified_row_count = 0
    bad_row_count = 0

    try:
        lines = output_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        _add_finding(
            findings,
            level=ValidationLevel.ERROR,
            code="missing_file",
            message=f"failed to read source '{source_name}' output file: {exc}",
            source=source_name,
            path=output_path,
        )
        return _build_file_report(
            output_path,
            manifest_linked=manifest_linked,
            orphan=orphan,
            row_count=row_count,
            verified_row_count=verified_row_count,
            bad_row_count=bad_row_count,
            findings=findings[file_start_index:],
        )

    for row_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        row_count += 1
        row_start_index = len(findings)
        _verify_row(
            line,
            source_name=source_name,
            output_path=output_path,
            row_number=row_number,
            findings=findings,
        )
        row_level = _worst_level(findings[row_start_index:])
        if row_level == ValidationLevel.ERROR:
            bad_row_count += 1
        else:
            verified_row_count += 1

    return _build_file_report(
        output_path,
        manifest_linked=manifest_linked,
        orphan=orphan,
        row_count=row_count,
        verified_row_count=verified_row_count,
        bad_row_count=bad_row_count,
        findings=findings[file_start_index:],
    )


def _build_file_report(
    output_path: Path,
    *,
    manifest_linked: bool,
    orphan: bool,
    row_count: int,
    verified_row_count: int,
    bad_row_count: int,
    findings: list[ValidationFinding],
) -> ArchiveVerifiedFile:
    warning_count = sum(
        1 for finding in findings if finding.level == ValidationLevel.WARNING
    )
    error_count = sum(1 for finding in findings if finding.level == ValidationLevel.ERROR)
    return ArchiveVerifiedFile(
        path=output_path,
        status=_worst_level(findings),
        row_count=row_count,
        verified_row_count=verified_row_count,
        bad_row_count=bad_row_count,
        finding_count=len(findings),
        warning_count=warning_count,
        error_count=error_count,
        manifest_linked=manifest_linked,
        orphan=orphan,
    )


def _verify_row(
    line: str,
    *,
    source_name: str,
    output_path: Path,
    row_number: int,
    findings: list[ValidationFinding],
) -> None:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        _add_finding(
            findings,
            level=ValidationLevel.ERROR,
            code="malformed_row",
            message=f"source '{source_name}' row {row_number} is not valid JSON",
            source=source_name,
            path=output_path,
            row_number=row_number,
        )
        return

    if not isinstance(payload, dict):
        _add_finding(
            findings,
            level=ValidationLevel.ERROR,
            code="malformed_row",
            message=f"source '{source_name}' row {row_number} must be an object",
            source=source_name,
            path=output_path,
            row_number=row_number,
        )
        return

    row_source = _required_string(
        payload,
        "source",
        findings=findings,
        context=f"source '{source_name}' row {row_number}",
        code="missing_required_field",
        source=source_name,
        path=output_path,
        row_number=row_number,
    )
    if row_source is not None and row_source != source_name:
        _add_finding(
            findings,
            level=ValidationLevel.ERROR,
            code="invalid_field",
            message=(
                f"source '{source_name}' row {row_number} declares source '{row_source}'"
            ),
            source=source_name,
            path=output_path,
            row_number=row_number,
        )

    _required_string(
        payload,
        "execution_context",
        findings=findings,
        context=f"source '{source_name}' row {row_number}",
        code="missing_required_field",
        source=source_name,
        path=output_path,
        row_number=row_number,
    )
    _required_string(
        payload,
        "collected_at",
        findings=findings,
        context=f"source '{source_name}' row {row_number}",
        code="missing_required_field",
        source=source_name,
        path=output_path,
        row_number=row_number,
    )

    contract = _required_object(
        payload,
        "contract",
        findings=findings,
        context=f"source '{source_name}' row {row_number}",
        code="missing_required_field",
        source=source_name,
        path=output_path,
        row_number=row_number,
    )
    if contract is not None:
        schema_version = _required_string(
            contract,
            "schema_version",
            findings=findings,
            context=f"source '{source_name}' row {row_number} contract",
            code="missing_required_field",
            source=source_name,
            path=output_path,
            row_number=row_number,
        )
        if schema_version is not None and schema_version != SCHEMA_VERSION:
            _add_finding(
                findings,
                level=ValidationLevel.ERROR,
                code="invalid_contract",
                message=(
                    f"source '{source_name}' row {row_number} contract schema_version "
                    f"'{schema_version}' does not match '{SCHEMA_VERSION}'"
                ),
                source=source_name,
                path=output_path,
                row_number=row_number,
            )

    transcript_completeness = payload.get(
        "transcript_completeness",
        TranscriptCompleteness.COMPLETE.value,
    )
    limitations_payload = payload.get("limitations", [])
    limitations: list[str] = []
    if isinstance(limitations_payload, list):
        limitations = [
            limitation
            for limitation in limitations_payload
            if isinstance(limitation, str) and limitation
        ]
    if isinstance(transcript_completeness, str):
        _validate_enum(
            transcript_completeness,
            TranscriptCompleteness,
            findings=findings,
            field_name="transcript_completeness",
            context=f"source '{source_name}' row {row_number}",
            source=source_name,
            path=output_path,
            row_number=row_number,
        )
        if transcript_completeness != TranscriptCompleteness.COMPLETE.value:
            limitation_suffix = ""
            if limitations:
                limitation_suffix = (
                    " (limitations: " + ", ".join(limitations) + ")"
                )
            _add_finding(
                findings,
                level=ValidationLevel.WARNING,
                code="incomplete_transcript",
                message=(
                    f"source '{source_name}' row {row_number} transcript completeness is "
                    f"{transcript_completeness}{limitation_suffix}"
                ),
                source=source_name,
                path=output_path,
                row_number=row_number,
            )
    else:
        _add_finding(
            findings,
            level=ValidationLevel.ERROR,
            code="invalid_field",
            message=(
                f"source '{source_name}' row {row_number} field "
                f"'transcript_completeness' must be a string"
            ),
            source=source_name,
            path=output_path,
            row_number=row_number,
        )

    messages = _required_list(
        payload,
        "messages",
        findings=findings,
        context=f"source '{source_name}' row {row_number}",
        code="missing_required_field",
        source=source_name,
        path=output_path,
        row_number=row_number,
    )
    if messages is None:
        return

    for message_index, message in enumerate(messages, start=1):
        if not isinstance(message, dict):
            _add_finding(
                findings,
                level=ValidationLevel.ERROR,
                code="malformed_row",
                message=(
                    f"source '{source_name}' row {row_number} message {message_index} "
                    "must be an object"
                ),
                source=source_name,
                path=output_path,
                row_number=row_number,
            )
            continue

        role = _required_string(
            message,
            "role",
            findings=findings,
            context=f"source '{source_name}' row {row_number} message {message_index}",
            code="missing_required_field",
            source=source_name,
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
                    f"source '{source_name}' row {row_number} message {message_index}"
                ),
                source=source_name,
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
                    f"source '{source_name}' row {row_number} message {message_index} "
                    "must include text or images"
                ),
                source=source_name,
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
                        f"source '{source_name}' row {row_number} message {message_index} "
                        "field 'text' must be a string or null"
                    ),
                    source=source_name,
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
                        f"source '{source_name}' row {row_number} message {message_index} "
                        "field 'images' must be an array"
                    ),
                    source=source_name,
                    path=output_path,
                    row_number=row_number,
                )
                continue
            for image_index, image in enumerate(images, start=1):
                if not isinstance(image, dict):
                    _add_finding(
                        findings,
                        level=ValidationLevel.ERROR,
                        code="invalid_field",
                        message=(
                            f"source '{source_name}' row {row_number} message "
                            f"{message_index} image {image_index} must be an object"
                        ),
                        source=source_name,
                        path=output_path,
                        row_number=row_number,
                    )
                    continue
                _required_string(
                    image,
                    "source",
                    findings=findings,
                    context=(
                        f"source '{source_name}' row {row_number} message "
                        f"{message_index} image {image_index}"
                    ),
                    code="missing_required_field",
                    source=source_name,
                    path=output_path,
                    row_number=row_number,
                )


def _is_canonical_output_path(output_path: Path, *, source: str) -> bool:
    return output_path.name == CANONICAL_OUTPUT_TEMPLATE.format(source=source)


def _is_within(candidate: Path, root: Path) -> bool:
    return candidate == root or root in candidate.parents


def _apply_baseline_policy(
    report: ArchiveVerifyReport,
    *,
    baseline_policy: BaselinePolicy,
) -> ArchiveVerifyReport:
    findings = tuple(
        _apply_baseline_to_finding(finding, baseline_policy=baseline_policy)
        for finding in report.findings
    )
    sources = tuple(
        _apply_baseline_to_source_report(source_report, findings=findings)
        for source_report in report.sources
    )
    return replace(
        report,
        sources=sources,
        findings=findings,
        baseline_path=baseline_policy.path,
        baseline_entry_count=baseline_policy.entry_count,
    )


def _apply_baseline_to_source_report(
    source_report: ArchiveVerifySourceReport,
    *,
    findings: tuple[ValidationFinding, ...],
) -> ArchiveVerifySourceReport:
    source_findings = tuple(
        finding for finding in findings if finding.source == source_report.source
    )
    files = tuple(
        _apply_baseline_to_file_report(
            file_report,
            findings=tuple(
                finding
                for finding in source_findings
                if finding.path == file_report.path
            ),
        )
        for file_report in source_report.files
    )
    suppressed_warning_count = sum(
        1
        for finding in source_findings
        if finding.level == ValidationLevel.WARNING and finding.suppressed
    )
    warning_count = sum(
        1
        for finding in source_findings
        if finding.level == ValidationLevel.WARNING and not finding.suppressed
    )
    error_count = sum(
        1 for finding in source_findings if finding.level == ValidationLevel.ERROR
    )
    return replace(
        source_report,
        status=_worst_level(source_findings),
        raw_status=source_report.status,
        warning_count=warning_count,
        error_count=error_count,
        files=files,
        suppressed_warning_count=suppressed_warning_count,
    )


def _apply_baseline_to_file_report(
    file_report: ArchiveVerifiedFile,
    *,
    findings: tuple[ValidationFinding, ...],
) -> ArchiveVerifiedFile:
    suppressed_warning_count = sum(
        1
        for finding in findings
        if finding.level == ValidationLevel.WARNING and finding.suppressed
    )
    warning_count = sum(
        1
        for finding in findings
        if finding.level == ValidationLevel.WARNING and not finding.suppressed
    )
    error_count = sum(1 for finding in findings if finding.level == ValidationLevel.ERROR)
    return replace(
        file_report,
        status=_worst_level(findings),
        raw_status=file_report.status,
        warning_count=warning_count,
        error_count=error_count,
        suppressed_warning_count=suppressed_warning_count,
    )


def _apply_baseline_to_finding(
    finding: ValidationFinding,
    *,
    baseline_policy: BaselinePolicy,
) -> ValidationFinding:
    matched_entry = baseline_policy.match_finding(
        report=BaselineReport.ARCHIVE_VERIFY,
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


__all__ = [
    "ArchiveVerifyError",
    "ArchiveVerifiedFile",
    "ArchiveVerifyReport",
    "ArchiveVerifySourceReport",
    "verify_archive",
]
