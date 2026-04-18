from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .archive_export import (
    ArchiveExportFilter,
    EXPORT_MANIFEST_FILENAME,
    EXPORT_OUTPUT_FILENAME,
)
from .archive_inspect import (
    ARCHIVE_OUTPUT_GLOB,
    ArchiveConversationRecord,
    build_archive_record,
    iter_archive_records,
    load_archive_json_line,
)
from .archive_merge import (
    ArchiveMergeCandidate,
    CANONICAL_OUTPUT_TEMPLATE,
    archive_candidate_group_key,
    archive_candidate_sort_key,
    build_archive_merge_candidate,
    compact_archive_candidates,
    select_archive_group_winner,
)


class ArchiveImportError(Exception):
    """Raised when an export bundle cannot be validated or merged."""


@dataclass(frozen=True, slots=True)
class ArchiveImportSourceReport:
    source: str
    changed: bool
    input_file_count: int
    before_conversation_count: int
    after_conversation_count: int
    imported_count: int
    skipped_count: int
    upgraded_count: int
    output_path: Path | None

    def to_dict(self) -> dict[str, object]:
        return {
            "changed": self.changed,
            "input_file_count": self.input_file_count,
            "before_conversation_count": self.before_conversation_count,
            "after_conversation_count": self.after_conversation_count,
            "imported_count": self.imported_count,
            "skipped_count": self.skipped_count,
            "upgraded_count": self.upgraded_count,
            "output_path": str(self.output_path) if self.output_path is not None else None,
        }


@dataclass(frozen=True, slots=True)
class ArchiveImportReport:
    archive_root: Path
    bundle_dir: Path
    write_mode: str
    manifest_path: Path
    conversations_path: Path
    filters: ArchiveExportFilter
    source_count: int
    conversation_count: int
    message_count: int
    before_conversation_count: int
    after_conversation_count: int
    imported_count: int
    skipped_count: int
    upgraded_count: int
    sources: tuple[ArchiveImportSourceReport, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "archive_root": str(self.archive_root),
            "bundle_dir": str(self.bundle_dir),
            "write_mode": self.write_mode,
            "manifest_path": str(self.manifest_path),
            "conversations_path": str(self.conversations_path),
            "filters": self.filters.to_dict(),
            "source_count": self.source_count,
            "changed_source_count": sum(1 for source in self.sources if source.changed),
            "conversation_count": self.conversation_count,
            "message_count": self.message_count,
            "before_conversation_count": self.before_conversation_count,
            "after_conversation_count": self.after_conversation_count,
            "imported_count": self.imported_count,
            "skipped_count": self.skipped_count,
            "upgraded_count": self.upgraded_count,
            "sources": {
                source.source: source.to_dict() for source in self.sources
            },
        }


@dataclass(frozen=True, slots=True)
class _ExportBundleManifest:
    filters: ArchiveExportFilter
    conversation_count: int
    message_count: int
    source_count: int


@dataclass(frozen=True, slots=True)
class _SourceImportPlan:
    report: ArchiveImportSourceReport
    serialized_rows: tuple[str, ...]


def import_archive_bundle(
    archive_root: Path,
    *,
    bundle_dir: Path,
    execute: bool = False,
) -> ArchiveImportReport:
    resolved_archive_root = archive_root.expanduser().resolve(strict=False)
    resolved_bundle_dir = bundle_dir.expanduser().resolve(strict=False)
    manifest_path = resolved_bundle_dir / EXPORT_MANIFEST_FILENAME
    conversations_path = resolved_bundle_dir / EXPORT_OUTPUT_FILENAME

    manifest = _load_export_bundle_manifest(manifest_path)
    bundle_records = _load_bundle_records(conversations_path)
    _validate_bundle_records(bundle_records, manifest=manifest)

    source_plans: list[_SourceImportPlan] = []
    for source_name in sorted({record.summary.source for record in bundle_records}):
        plan = _plan_source_import(
            resolved_archive_root,
            source=source_name,
            bundle_records=tuple(
                record for record in bundle_records if record.summary.source == source_name
            ),
        )
        source_plans.append(plan)
        if execute:
            _write_source_plan(plan, archive_root=resolved_archive_root)

    return ArchiveImportReport(
        archive_root=resolved_archive_root,
        bundle_dir=resolved_bundle_dir,
        write_mode="write" if execute else "dry_run",
        manifest_path=manifest_path,
        conversations_path=conversations_path,
        filters=manifest.filters,
        source_count=manifest.source_count,
        conversation_count=manifest.conversation_count,
        message_count=manifest.message_count,
        before_conversation_count=sum(
            plan.report.before_conversation_count for plan in source_plans
        ),
        after_conversation_count=sum(
            plan.report.after_conversation_count for plan in source_plans
        ),
        imported_count=sum(plan.report.imported_count for plan in source_plans),
        skipped_count=sum(plan.report.skipped_count for plan in source_plans),
        upgraded_count=sum(plan.report.upgraded_count for plan in source_plans),
        sources=tuple(plan.report for plan in source_plans),
    )


def _load_export_bundle_manifest(manifest_path: Path) -> _ExportBundleManifest:
    if not manifest_path.exists():
        raise ArchiveImportError(f"export bundle manifest does not exist: {manifest_path}")
    if not manifest_path.is_file():
        raise ArchiveImportError(f"export bundle manifest is not a file: {manifest_path}")

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ArchiveImportError(
            f"export bundle manifest is not valid JSON: {manifest_path}"
        ) from exc
    except OSError as exc:
        raise ArchiveImportError(
            f"failed to read export bundle manifest: {exc}"
        ) from exc

    if not isinstance(payload, dict):
        raise ArchiveImportError(
            f"export bundle manifest root must be an object: {manifest_path}"
        )

    write_mode = _required_string(payload, "write_mode", manifest_path)
    if write_mode != "write":
        raise ArchiveImportError(
            f"export bundle manifest write_mode must be 'write': {manifest_path}"
        )

    filters_payload = payload.get("filters")
    if not isinstance(filters_payload, dict):
        raise ArchiveImportError(
            f"export bundle manifest field 'filters' must be an object: {manifest_path}"
        )

    return _ExportBundleManifest(
        filters=ArchiveExportFilter(
            source=_optional_string(filters_payload, "source", manifest_path),
            session=_optional_string(filters_payload, "session", manifest_path),
            transcript_completeness=_optional_string(
                filters_payload,
                "transcript_completeness",
                manifest_path,
            ),
            text=_optional_string(filters_payload, "text", manifest_path),
        ),
        conversation_count=_required_int(payload, "conversation_count", manifest_path),
        message_count=_required_int(payload, "message_count", manifest_path),
        source_count=_required_int(payload, "source_count", manifest_path),
    )


def _load_bundle_records(conversations_path: Path) -> tuple[ArchiveConversationRecord, ...]:
    if not conversations_path.exists():
        raise ArchiveImportError(
            f"export bundle conversations file does not exist: {conversations_path}"
        )
    if not conversations_path.is_file():
        raise ArchiveImportError(
            f"export bundle conversations path is not a file: {conversations_path}"
        )

    records: list[ArchiveConversationRecord] = []
    with conversations_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            payload = load_archive_json_line(
                raw_line,
                output_path=conversations_path,
                line_number=line_number,
            )
            if payload is None:
                continue
            records.append(
                build_archive_record(
                    payload,
                    output_path=conversations_path,
                    line_number=line_number,
                )
            )
    return tuple(records)


def _validate_bundle_records(
    records: tuple[ArchiveConversationRecord, ...],
    *,
    manifest: _ExportBundleManifest,
) -> None:
    actual_conversation_count = len(records)
    if actual_conversation_count != manifest.conversation_count:
        raise ArchiveImportError(
            "export bundle conversation_count mismatch: "
            f"manifest={manifest.conversation_count} actual={actual_conversation_count}"
        )

    actual_message_count = sum(record.summary.message_count for record in records)
    if actual_message_count != manifest.message_count:
        raise ArchiveImportError(
            "export bundle message_count mismatch: "
            f"manifest={manifest.message_count} actual={actual_message_count}"
        )

    actual_source_count = len({record.summary.source for record in records})
    if actual_source_count != manifest.source_count:
        raise ArchiveImportError(
            "export bundle source_count mismatch: "
            f"manifest={manifest.source_count} actual={actual_source_count}"
        )

    for record in records:
        _validate_record_against_filters(record, filters=manifest.filters)


def _validate_record_against_filters(
    record: ArchiveConversationRecord,
    *,
    filters: ArchiveExportFilter,
) -> None:
    output_path = record.summary.output_path
    row_number = record.summary.row_number
    location = f"{output_path}:{row_number}"

    if filters.source is not None and record.summary.source != filters.source:
        raise ArchiveImportError(
            f"bundle row does not satisfy export filter source={filters.source!r}: "
            f"{location}"
        )
    if (
        filters.session is not None
        and record.summary.source_session_id != filters.session
    ):
        raise ArchiveImportError(
            f"bundle row does not satisfy export filter session={filters.session!r}: "
            f"{location}"
        )
    if (
        filters.transcript_completeness is not None
        and record.summary.transcript_completeness
        != filters.transcript_completeness
    ):
        raise ArchiveImportError(
            "bundle row does not satisfy export filter "
            f"transcript_completeness={filters.transcript_completeness!r}: {location}"
        )
    if filters.text is not None and not any(
        isinstance(message.get("text"), str)
        and filters.text.casefold() in message["text"].casefold()
        for message in record.messages
    ):
        raise ArchiveImportError(
            f"bundle row does not satisfy export filter text={filters.text!r}: "
            f"{location}"
        )


def _plan_source_import(
    archive_root: Path,
    *,
    source: str,
    bundle_records: tuple[ArchiveConversationRecord, ...],
) -> _SourceImportPlan:
    existing_records = _load_existing_records(archive_root, source=source)
    existing_candidates = tuple(
        build_archive_merge_candidate(record) for record in existing_records
    )
    bundle_candidates = tuple(
        build_archive_merge_candidate(record) for record in bundle_records
    )

    before_selected, _, _, _ = compact_archive_candidates(existing_candidates)
    merged_selected, _, _, _ = compact_archive_candidates(
        existing_candidates + bundle_candidates
    )
    imported_count, skipped_count, upgraded_count = _count_bundle_changes(
        existing_candidates=existing_candidates,
        bundle_candidates=bundle_candidates,
    )

    input_paths = tuple(
        sorted({record.summary.output_path for record in existing_records})
    )
    serialized_rows = tuple(
        candidate.serialized_payload
        for candidate in sorted(merged_selected, key=archive_candidate_sort_key)
    )
    output_path = (
        archive_root / source / CANONICAL_OUTPUT_TEMPLATE.format(source=source)
        if serialized_rows
        else None
    )
    expected_paths = () if output_path is None else (output_path,)

    report = ArchiveImportSourceReport(
        source=source,
        changed=(
            imported_count > 0
            or upgraded_count > 0
            or input_paths != expected_paths
        ),
        input_file_count=len(input_paths),
        before_conversation_count=len(before_selected),
        after_conversation_count=len(merged_selected),
        imported_count=imported_count,
        skipped_count=skipped_count,
        upgraded_count=upgraded_count,
        output_path=output_path,
    )
    return _SourceImportPlan(
        report=report,
        serialized_rows=serialized_rows,
    )


def _load_existing_records(
    archive_root: Path,
    *,
    source: str,
) -> tuple[ArchiveConversationRecord, ...]:
    if not archive_root.exists():
        return ()
    if not archive_root.is_dir():
        raise ArchiveImportError(f"archive root is not a directory: {archive_root}")
    return tuple(iter_archive_records(archive_root, source=source))


def _count_bundle_changes(
    *,
    existing_candidates: tuple[ArchiveMergeCandidate, ...],
    bundle_candidates: tuple[ArchiveMergeCandidate, ...],
) -> tuple[int, int, int]:
    existing_groups: dict[tuple[str, ...], list[ArchiveMergeCandidate]] = {}
    for candidate in existing_candidates:
        existing_groups.setdefault(archive_candidate_group_key(candidate), []).append(
            candidate
        )

    bundle_groups: dict[tuple[str, ...], list[ArchiveMergeCandidate]] = {}
    for candidate in bundle_candidates:
        bundle_groups.setdefault(archive_candidate_group_key(candidate), []).append(
            candidate
        )

    imported_count = 0
    skipped_count = 0
    upgraded_count = 0

    for group_key in sorted(bundle_groups):
        bundle_group = tuple(bundle_groups[group_key])
        existing_group = tuple(existing_groups.get(group_key, ()))
        if not existing_group:
            imported_count += 1
            skipped_count += len(bundle_group) - 1
            continue

        existing_winner, _, _ = select_archive_group_winner(existing_group)
        merged_winner, _, _ = select_archive_group_winner(existing_group + bundle_group)
        if merged_winner.serialized_payload == existing_winner.serialized_payload:
            skipped_count += len(bundle_group)
            continue

        upgraded_count += 1
        skipped_count += len(bundle_group) - 1

    return imported_count, skipped_count, upgraded_count


def _write_source_plan(
    plan: _SourceImportPlan,
    *,
    archive_root: Path,
) -> None:
    output_path = plan.report.output_path
    if output_path is None or not plan.report.changed:
        return

    source_dir = archive_root / plan.report.source
    source_dir.mkdir(parents=True, exist_ok=True)
    existing_paths = tuple(sorted(source_dir.glob(ARCHIVE_OUTPUT_GLOB)))
    temporary_path = source_dir / f".{output_path.name}.tmp"
    try:
        temporary_path.write_text(
            "".join(f"{row}\n" for row in plan.serialized_rows),
            encoding="utf-8",
        )
        for existing_path in existing_paths:
            existing_path.unlink()
        temporary_path.replace(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _required_string(
    payload: dict[str, object],
    key: str,
    manifest_path: Path,
) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ArchiveImportError(
            f"export bundle manifest field '{key}' must be a non-empty string: "
            f"{manifest_path}"
        )
    return value


def _required_int(
    payload: dict[str, object],
    key: str,
    manifest_path: Path,
) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ArchiveImportError(
            f"export bundle manifest field '{key}' must be an integer: {manifest_path}"
        )
    return value


def _optional_string(
    payload: dict[str, object],
    key: str,
    manifest_path: Path,
) -> str | None:
    if key not in payload:
        raise ArchiveImportError(
            f"export bundle manifest field '{key}' must be present: {manifest_path}"
        )
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ArchiveImportError(
            f"export bundle manifest field '{key}' must be a string or null: "
            f"{manifest_path}"
        )
    return value


__all__ = [
    "ArchiveImportError",
    "ArchiveImportReport",
    "ArchiveImportSourceReport",
    "import_archive_bundle",
]
