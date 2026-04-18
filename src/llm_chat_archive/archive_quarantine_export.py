from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .archive_verify import verify_archive
from .validate import ValidationFinding

QUARANTINE_OUTPUT_FILENAME = "quarantine.jsonl"
QUARANTINE_MANIFEST_FILENAME = "quarantine-manifest.json"


class ArchiveQuarantineExportError(ValueError):
    """Raised when archive quarantine export cannot read the requested rows."""


@dataclass(frozen=True, slots=True)
class ArchiveQuarantineExportFilter:
    source: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class ArchiveQuarantineFinding:
    code: str
    level: str
    message: str

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "level": self.level,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class ArchiveQuarantineBundleRow:
    source: str
    archive_path: str
    row_number: int
    findings: tuple[ArchiveQuarantineFinding, ...]
    row_present: bool = False
    row: object | None = None
    raw_line: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "source": self.source,
            "archive_path": self.archive_path,
            "row_number": self.row_number,
            "findings": [finding.to_dict() for finding in self.findings],
        }
        if self.row_present:
            payload["row"] = self.row
        if self.raw_line is not None:
            payload["raw_line"] = self.raw_line
        return payload


@dataclass(frozen=True, slots=True)
class ArchiveQuarantineExportReport:
    archive_root: Path
    output_dir: Path
    write_mode: str
    created_at: str
    filters: ArchiveQuarantineExportFilter
    row_count: int
    finding_count: int
    source_row_counts: dict[str, int]
    finding_code_counts: dict[str, int]
    quarantine_path: Path
    manifest_path: Path

    def to_dict(self) -> dict[str, object]:
        return {
            "archive_root": str(self.archive_root),
            "output_dir": str(self.output_dir),
            "write_mode": self.write_mode,
            "created_at": self.created_at,
            "filters": self.filters.to_dict(),
            "row_count": self.row_count,
            "source_count": len(self.source_row_counts),
            "finding_count": self.finding_count,
            "source_row_counts": dict(sorted(self.source_row_counts.items())),
            "finding_code_counts": dict(sorted(self.finding_code_counts.items())),
            "quarantine_path": str(self.quarantine_path),
            "manifest_path": str(self.manifest_path),
        }


@dataclass(frozen=True, slots=True)
class _ExportableFinding:
    source: str
    path: Path
    row_number: int
    finding: ValidationFinding


def export_archive_quarantine(
    archive_root: Path,
    *,
    output_dir: Path,
    source: str | None = None,
    execute: bool = False,
) -> ArchiveQuarantineExportReport:
    resolved_archive_root = archive_root.expanduser().resolve(strict=False)
    resolved_output_dir = output_dir.expanduser().resolve(strict=False)
    filters = ArchiveQuarantineExportFilter(source=source)
    findings = _collect_exportable_findings(
        resolved_archive_root,
        source=source,
    )
    bundle_rows = _build_bundle_rows(
        findings,
        archive_root=resolved_archive_root,
    )
    source_row_counts = Counter(row.source for row in bundle_rows)
    finding_code_counts = Counter(finding.finding.code for finding in findings)
    report = ArchiveQuarantineExportReport(
        archive_root=resolved_archive_root,
        output_dir=resolved_output_dir,
        write_mode="write" if execute else "dry_run",
        created_at=_format_timestamp(_utcnow()),
        filters=filters,
        row_count=len(bundle_rows),
        finding_count=sum(finding_code_counts.values()),
        source_row_counts=dict(source_row_counts),
        finding_code_counts=dict(finding_code_counts),
        quarantine_path=resolved_output_dir / QUARANTINE_OUTPUT_FILENAME,
        manifest_path=resolved_output_dir / QUARANTINE_MANIFEST_FILENAME,
    )
    if execute:
        _write_bundle(report, bundle_rows)
    return report


def _collect_exportable_findings(
    archive_root: Path,
    *,
    source: str | None,
) -> tuple[_ExportableFinding, ...]:
    report = verify_archive(archive_root, source=source)
    exportable_findings: list[_ExportableFinding] = []
    for finding in report.findings:
        if finding.source is None or finding.path is None or finding.row_number is None:
            continue
        resolved_path = finding.path.expanduser().resolve(strict=False)
        if not _is_within(resolved_path, archive_root):
            continue
        exportable_findings.append(
            _ExportableFinding(
                source=finding.source,
                path=resolved_path,
                row_number=finding.row_number,
                finding=finding,
            )
        )
    return tuple(exportable_findings)


def _build_bundle_rows(
    findings: tuple[_ExportableFinding, ...],
    *,
    archive_root: Path,
) -> tuple[ArchiveQuarantineBundleRow, ...]:
    grouped_findings: dict[tuple[str, Path, int], list[_ExportableFinding]] = {}
    for finding in findings:
        grouped_findings.setdefault(
            (finding.source, finding.path, finding.row_number),
            [],
        ).append(finding)

    line_cache: dict[Path, tuple[str, ...]] = {}
    bundle_rows: list[ArchiveQuarantineBundleRow] = []
    for source, path, row_number in sorted(
        grouped_findings,
        key=lambda item: (item[0], str(item[1]), item[2]),
    ):
        raw_line = _load_line(path, row_number=row_number, line_cache=line_cache)
        row_present, row_payload, raw_line_payload = _decode_quarantine_row(raw_line)
        bundle_rows.append(
            ArchiveQuarantineBundleRow(
                source=source,
                archive_path=str(path.relative_to(archive_root)),
                row_number=row_number,
                findings=tuple(
                    ArchiveQuarantineFinding(
                        code=exportable.finding.code,
                        level=exportable.finding.level.value,
                        message=exportable.finding.message,
                    )
                    for exportable in grouped_findings[(source, path, row_number)]
                ),
                row_present=row_present,
                row=row_payload,
                raw_line=raw_line_payload,
            )
        )
    return tuple(bundle_rows)


def _load_line(
    path: Path,
    *,
    row_number: int,
    line_cache: dict[Path, tuple[str, ...]],
) -> str:
    if path not in line_cache:
        try:
            line_cache[path] = tuple(path.read_text(encoding="utf-8").splitlines())
        except OSError as exc:
            raise ArchiveQuarantineExportError(
                f"failed to read quarantine source row from {path}: {exc}"
            ) from exc

    lines = line_cache[path]
    if row_number < 1 or row_number > len(lines):
        raise ArchiveQuarantineExportError(
            f"quarantine source row {row_number} is out of range for {path}"
        )
    return lines[row_number - 1]


def _decode_quarantine_row(raw_line: str) -> tuple[bool, object | None, str | None]:
    try:
        return True, json.loads(raw_line.strip()), None
    except json.JSONDecodeError:
        return False, None, raw_line


def _write_bundle(
    report: ArchiveQuarantineExportReport,
    bundle_rows: tuple[ArchiveQuarantineBundleRow, ...],
) -> None:
    report.output_dir.mkdir(parents=True, exist_ok=True)
    report.quarantine_path.write_text(
        "".join(_serialize_jsonl_line(row.to_dict()) for row in bundle_rows),
        encoding="utf-8",
    )
    report.manifest_path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _serialize_jsonl_line(payload: dict[str, object]) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n"


def _is_within(candidate: Path, root: Path) -> bool:
    return candidate == root or root in candidate.parents


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _format_timestamp(value: datetime) -> str:
    return value.replace(microsecond=0).isoformat().replace("+00:00", "Z")


__all__ = [
    "ArchiveQuarantineBundleRow",
    "ArchiveQuarantineExportError",
    "ArchiveQuarantineExportFilter",
    "ArchiveQuarantineExportReport",
    "QUARANTINE_MANIFEST_FILENAME",
    "QUARANTINE_OUTPUT_FILENAME",
    "export_archive_quarantine",
]
