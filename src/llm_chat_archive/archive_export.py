from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .archive_inspect import (
    ArchiveConversationFilter,
    ArchiveConversationRecord,
    ArchiveInspectError,
    iter_archive_records,
)

EXPORT_OUTPUT_FILENAME = "conversations.jsonl"
EXPORT_MANIFEST_FILENAME = "export-manifest.json"


@dataclass(frozen=True, slots=True)
class ArchiveExportFilter:
    source: str | None = None
    session: str | None = None
    transcript_completeness: str | None = None
    text: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "session": self.session,
            "transcript_completeness": self.transcript_completeness,
            "text": self.text,
        }


@dataclass(frozen=True, slots=True)
class ArchiveExportReport:
    archive_root: Path
    output_dir: Path
    write_mode: str
    filters: ArchiveExportFilter
    conversation_count: int
    message_count: int
    source_count: int
    conversations_path: Path
    manifest_path: Path

    def to_dict(self) -> dict[str, object]:
        return {
            "archive_root": str(self.archive_root),
            "output_dir": str(self.output_dir),
            "write_mode": self.write_mode,
            "filters": self.filters.to_dict(),
            "conversation_count": self.conversation_count,
            "message_count": self.message_count,
            "source_count": self.source_count,
            "conversations_path": str(self.conversations_path),
            "manifest_path": str(self.manifest_path),
        }


def export_archive_subset(
    archive_root: Path,
    *,
    output_dir: Path,
    source: str | None = None,
    session: str | None = None,
    transcript_completeness: str | None = None,
    text: str | None = None,
    execute: bool = False,
) -> ArchiveExportReport:
    resolved_archive_root = archive_root.expanduser().resolve(strict=False)
    resolved_output_dir = output_dir.expanduser().resolve(strict=False)
    filters = ArchiveExportFilter(
        source=source,
        session=session,
        transcript_completeness=transcript_completeness,
        text=_normalize_text_query(text),
    )
    records = _collect_records(
        resolved_archive_root,
        filters=filters,
    )
    conversations_path = resolved_output_dir / EXPORT_OUTPUT_FILENAME
    manifest_path = resolved_output_dir / EXPORT_MANIFEST_FILENAME
    report = ArchiveExportReport(
        archive_root=resolved_archive_root,
        output_dir=resolved_output_dir,
        write_mode="write" if execute else "dry_run",
        filters=filters,
        conversation_count=len(records),
        message_count=sum(record.summary.message_count for record in records),
        source_count=len({record.summary.source for record in records}),
        conversations_path=conversations_path,
        manifest_path=manifest_path,
    )
    if execute:
        _write_bundle(report, records)
    return report


def _collect_records(
    archive_root: Path,
    *,
    filters: ArchiveExportFilter,
) -> tuple[ArchiveConversationRecord, ...]:
    conversation_filter = ArchiveConversationFilter(
        source=filters.source,
        session=filters.session,
        transcript_completeness=filters.transcript_completeness,
    )
    records = [
        record
        for record in iter_archive_records(archive_root, source=filters.source)
        if conversation_filter.matches(record.summary)
        and _matches_text_filter(record, filters.text)
    ]
    return tuple(sorted(records, key=_record_sort_key, reverse=True))


def _normalize_text_query(text: str | None) -> str | None:
    if text is None:
        return None
    query = text.strip()
    if not query:
        raise ValueError("text query must not be empty")
    return query


def _matches_text_filter(
    record: ArchiveConversationRecord,
    text: str | None,
) -> bool:
    if text is None:
        return True
    normalized_query = text.casefold()
    return any(
        isinstance(message.get("text"), str)
        and normalized_query in message["text"].casefold()
        for message in record.messages
    )


def _record_sort_key(record: ArchiveConversationRecord) -> tuple[object, ...]:
    summary = record.summary
    return (
        summary.collected_at,
        summary.source,
        summary.source_session_id or "",
        str(summary.output_path),
        summary.row_number,
    )


def _write_bundle(
    report: ArchiveExportReport,
    records: tuple[ArchiveConversationRecord, ...],
) -> None:
    report.output_dir.mkdir(parents=True, exist_ok=True)
    report.conversations_path.write_text(
        "".join(_serialize_jsonl_line(record.payload) for record in records),
        encoding="utf-8",
    )
    report.manifest_path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _serialize_jsonl_line(payload: dict[str, object]) -> str:
    try:
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ) + "\n"
    except TypeError as exc:
        raise ArchiveInspectError(
            "normalized archive row could not be serialized for export"
        ) from exc


__all__ = [
    "ArchiveExportFilter",
    "ArchiveExportReport",
    "EXPORT_MANIFEST_FILENAME",
    "EXPORT_OUTPUT_FILENAME",
    "export_archive_subset",
]
