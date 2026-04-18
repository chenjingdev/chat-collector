from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from .archive_inspect import (
    ArchiveConversationFilter,
    ArchiveConversationRecord,
    ArchiveInspectError,
    build_archive_record,
    is_superseded_archive_payload,
    iter_archive_records,
    load_archive_json_line,
)
from .incremental import build_message_fingerprint
from .redaction import REDACTED_API_KEY, REDACTED_VALUE
from .reporting import load_run_summary

MEMORY_EXPORT_SCHEMA_VERSION = "2026-03-20"
MEMORY_EXPORT_OUTPUT_FILENAME = "memory-records.jsonl"
MEMORY_EXPORT_MANIFEST_FILENAME = "memory-export-manifest.json"
_REDACTION_STATUS_REDACTED = "redacted"
_REDACTION_STATUS_NONE = "none_detected"


@dataclass(frozen=True, slots=True)
class ArchiveMemoryExportContract:
    schema_version: str = MEMORY_EXPORT_SCHEMA_VERSION
    record_kind: str = "memory_ingestion_conversation_v1"
    record_unit: str = "conversation"
    transcript_text_format: str = "role_prefixed_blocks"
    watermark_field: str = "collected_at"

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "record_kind": self.record_kind,
            "record_unit": self.record_unit,
            "transcript_text_format": self.transcript_text_format,
            "watermark_field": self.watermark_field,
            "stable_ids": {
                "conversation_id": [
                    "source",
                    "source_session_id | source_artifact_path | content_fingerprint",
                ],
                "message_id": [
                    "conversation_id",
                    "source_message_id | message_fingerprint_occurrence",
                ],
            },
            "redaction_status_values": [
                _REDACTION_STATUS_NONE,
                _REDACTION_STATUS_REDACTED,
            ],
        }


@dataclass(frozen=True, slots=True)
class ArchiveMemoryExportFilter:
    source: str | None = None
    session: str | None = None
    transcript_completeness: str | None = None
    text: str | None = None
    run_id: str | None = None
    after_collected_at: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "session": self.session,
            "transcript_completeness": self.transcript_completeness,
            "text": self.text,
            "run_id": self.run_id,
            "after_collected_at": self.after_collected_at,
        }


@dataclass(frozen=True, slots=True)
class ArchiveMemoryExportReport:
    archive_root: Path
    output_dir: Path
    write_mode: str
    contract: ArchiveMemoryExportContract
    filters: ArchiveMemoryExportFilter
    record_count: int
    conversation_count: int
    message_count: int
    source_count: int
    earliest_collected_at: str | None
    latest_collected_at: str | None
    records_path: Path
    manifest_path: Path

    def to_dict(self) -> dict[str, object]:
        return {
            "archive_root": str(self.archive_root),
            "output_dir": str(self.output_dir),
            "write_mode": self.write_mode,
            "contract": self.contract.to_dict(),
            "filters": self.filters.to_dict(),
            "record_count": self.record_count,
            "conversation_count": self.conversation_count,
            "message_count": self.message_count,
            "source_count": self.source_count,
            "earliest_collected_at": self.earliest_collected_at,
            "latest_collected_at": self.latest_collected_at,
            "records_path": str(self.records_path),
            "manifest_path": str(self.manifest_path),
        }


@dataclass(frozen=True, slots=True)
class _RedactionSummary:
    status: str
    marker_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "marker_count": self.marker_count,
        }


def export_archive_memory_records(
    archive_root: Path,
    *,
    output_dir: Path,
    source: str | None = None,
    session: str | None = None,
    transcript_completeness: str | None = None,
    text: str | None = None,
    run_id: str | None = None,
    after_collected_at: str | None = None,
    execute: bool = False,
) -> ArchiveMemoryExportReport:
    resolved_archive_root = archive_root.expanduser().resolve(strict=False)
    resolved_output_dir = output_dir.expanduser().resolve(strict=False)
    contract = ArchiveMemoryExportContract()
    filters = ArchiveMemoryExportFilter(
        source=source,
        session=session,
        transcript_completeness=transcript_completeness,
        text=_normalize_text_query(text),
        run_id=_normalize_optional_value(run_id, label="run id"),
        after_collected_at=_normalize_timestamp_filter(
            after_collected_at,
            label="after-collected-at",
        ),
    )
    records = _collect_records(resolved_archive_root, filters=filters)
    memory_records = tuple(
        _build_memory_export_record(record, contract=contract, run_id=filters.run_id)
        for record in records
    )
    records_path = resolved_output_dir / MEMORY_EXPORT_OUTPUT_FILENAME
    manifest_path = resolved_output_dir / MEMORY_EXPORT_MANIFEST_FILENAME
    report = ArchiveMemoryExportReport(
        archive_root=resolved_archive_root,
        output_dir=resolved_output_dir,
        write_mode="write" if execute else "dry_run",
        contract=contract,
        filters=filters,
        record_count=len(memory_records),
        conversation_count=len(records),
        message_count=sum(record.summary.message_count for record in records),
        source_count=len({record.summary.source for record in records}),
        earliest_collected_at=(
            None if not records else min(record.summary.collected_at for record in records)
        ),
        latest_collected_at=(
            None if not records else max(record.summary.collected_at for record in records)
        ),
        records_path=records_path,
        manifest_path=manifest_path,
    )
    if execute:
        _write_bundle(report, memory_records)
    return report


def _collect_records(
    archive_root: Path,
    *,
    filters: ArchiveMemoryExportFilter,
) -> tuple[ArchiveConversationRecord, ...]:
    conversation_filter = ArchiveConversationFilter(
        source=filters.source,
        session=filters.session,
        transcript_completeness=filters.transcript_completeness,
    )
    if filters.run_id is not None:
        candidates = _iter_run_records(
            archive_root,
            run_id=filters.run_id,
            source=filters.source,
        )
    else:
        candidates = iter_archive_records(archive_root, source=filters.source)

    records = [
        record
        for record in candidates
        if conversation_filter.matches(record.summary)
        and _matches_text_filter(record, filters.text)
        and _matches_after_collected_at(record, filters.after_collected_at)
    ]
    return tuple(sorted(records, key=_record_sort_key, reverse=True))


def _iter_run_records(
    archive_root: Path,
    *,
    run_id: str,
    source: str | None,
):
    run_summary = load_run_summary(archive_root, run_id)
    for source_summary in run_summary.sources:
        if source is not None and source_summary.source != source:
            continue
        if source_summary.output_path is None:
            continue
        yield from _iter_output_path_records(
            source_summary.output_path,
            source=source_summary.source,
        )


def _iter_output_path_records(output_path: Path, *, source: str):
    with output_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            payload = load_archive_json_line(
                raw_line,
                output_path=output_path,
                line_number=line_number,
            )
            if payload is None or is_superseded_archive_payload(payload):
                continue
            record = build_archive_record(
                payload,
                output_path=output_path,
                line_number=line_number,
            )
            if record.summary.source != source:
                continue
            yield record


def _build_memory_export_record(
    record: ArchiveConversationRecord,
    *,
    contract: ArchiveMemoryExportContract,
    run_id: str | None,
) -> dict[str, object]:
    conversation_id = _build_conversation_id(record)
    messages = _build_memory_messages(record, conversation_id=conversation_id)
    return {
        "contract": contract.to_dict(),
        "id": conversation_id,
        "record_type": "conversation",
        "source": record.summary.source,
        "execution_context": record.summary.execution_context,
        "collected_at": record.summary.collected_at,
        "source_session_id": record.summary.source_session_id,
        "source_artifact_path": record.summary.source_artifact_path,
        "transcript_completeness": record.summary.transcript_completeness,
        "limitations": list(record.summary.limitations),
        "redaction": _build_redaction_summary(record.payload).to_dict(),
        "message_count": record.summary.message_count,
        "transcript_text": _build_transcript_text(messages),
        "messages": messages,
        "source_provenance": _optional_object(record.payload.get("provenance")),
        "export_provenance": {
            "archive_output_path": str(record.summary.output_path),
            "archive_row_number": record.summary.row_number,
            "run_id": run_id,
        },
    }


def _build_memory_messages(
    record: ArchiveConversationRecord,
    *,
    conversation_id: str,
) -> list[dict[str, object]]:
    messages: list[dict[str, object]] = []
    fingerprint_counts: dict[str, int] = {}
    for index, message in enumerate(record.messages, start=1):
        role = _required_message_role(record, message)
        source_message_id = _optional_string(message.get("source_message_id"))
        message_fingerprint = _hash_payload(message)
        fingerprint_counts[message_fingerprint] = (
            fingerprint_counts.get(message_fingerprint, 0) + 1
        )
        identity_payload: dict[str, object] = {"conversation_id": conversation_id}
        if source_message_id is not None:
            identity_payload["source_message_id"] = source_message_id
        else:
            identity_payload["message_fingerprint"] = message_fingerprint
            identity_payload["message_occurrence"] = fingerprint_counts[message_fingerprint]

        images = message.get("images")
        image_count = len(images) if isinstance(images, list) else 0
        payload = {
            "id": _stable_hash_id(identity_payload),
            "index": index,
            "role": role,
            "timestamp": _optional_string(message.get("timestamp")),
            "source_message_id": source_message_id,
            "text": _optional_string(message.get("text")),
            "image_count": image_count,
            "has_images": image_count > 0,
            "redaction": _build_redaction_summary(message).to_dict(),
        }
        message_provenance = _optional_object(message.get("provenance"))
        if message_provenance is not None:
            payload["provenance"] = message_provenance
        messages.append(payload)
    return messages


def _build_conversation_id(record: ArchiveConversationRecord) -> str:
    identity_payload: dict[str, object] = {"source": record.summary.source}
    if record.summary.source_session_id is not None:
        identity_payload["identity_basis"] = "source_session_id"
        identity_payload["source_session_id"] = record.summary.source_session_id
    elif record.summary.source_artifact_path is not None:
        identity_payload["identity_basis"] = "source_artifact_path"
        identity_payload["source_artifact_path"] = record.summary.source_artifact_path
    else:
        identity_payload["identity_basis"] = "content_fingerprint"
        identity_payload["collected_at"] = record.summary.collected_at
        identity_payload["message_fingerprint"] = build_message_fingerprint(
            list(record.messages)
        )
    return _stable_hash_id(identity_payload)


def _build_transcript_text(messages: list[dict[str, object]]) -> str:
    return "\n\n".join(
        f"{message['role']}: {_message_content_text(message)}" for message in messages
    )


def _message_content_text(message: dict[str, object]) -> str:
    text = _optional_string(message.get("text"))
    image_count = message.get("image_count")
    image_suffix = ""
    if isinstance(image_count, int) and image_count > 0:
        if image_count == 1:
            image_suffix = "[image]"
        else:
            image_suffix = f"[images:{image_count}]"

    if text and image_suffix:
        return f"{text} {image_suffix}"
    if text:
        return text
    if image_suffix:
        return image_suffix
    return "[empty]"


def _build_redaction_summary(value: object) -> _RedactionSummary:
    marker_count = _count_redaction_markers(value)
    return _RedactionSummary(
        status=(
            _REDACTION_STATUS_REDACTED
            if marker_count > 0
            else _REDACTION_STATUS_NONE
        ),
        marker_count=marker_count,
    )


def _count_redaction_markers(value: object) -> int:
    if isinstance(value, str):
        return value.count(REDACTED_VALUE) + value.count(REDACTED_API_KEY)
    if isinstance(value, dict):
        return sum(_count_redaction_markers(item) for item in value.values())
    if isinstance(value, list):
        return sum(_count_redaction_markers(item) for item in value)
    return 0


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


def _matches_after_collected_at(
    record: ArchiveConversationRecord,
    after_collected_at: str | None,
) -> bool:
    if after_collected_at is None:
        return True
    return _parse_timestamp(record.summary.collected_at) > _parse_timestamp(
        after_collected_at
    )


def _normalize_text_query(text: str | None) -> str | None:
    if text is None:
        return None
    query = text.strip()
    if not query:
        raise ValueError("text query must not be empty")
    return query


def _normalize_optional_value(value: str | None, *, label: str) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{label} must not be empty")
    return normalized


def _normalize_timestamp_filter(value: str | None, *, label: str) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{label} must not be empty")
    return _format_timestamp(_parse_timestamp(normalized))


def _parse_timestamp(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"invalid ISO timestamp: {value}") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"timestamp must include timezone: {value}")
    return parsed.astimezone(timezone.utc)


def _format_timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00",
        "Z",
    )


def _required_message_role(
    record: ArchiveConversationRecord,
    message: dict[str, object],
) -> str:
    role = message.get("role")
    if isinstance(role, str) and role:
        return role
    raise ArchiveInspectError(
        "normalized message requires role at "
        f"{record.summary.output_path}:{record.summary.row_number}"
    )


def _optional_object(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return dict(value)
    return {}


def _optional_string(value: object) -> str | None:
    if isinstance(value, str):
        return value
    return None


def _record_sort_key(record: ArchiveConversationRecord) -> tuple[object, ...]:
    summary = record.summary
    return (
        summary.collected_at,
        summary.source,
        summary.source_session_id or "",
        str(summary.output_path),
        summary.row_number,
    )


def _stable_hash_id(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _hash_payload(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_bundle(
    report: ArchiveMemoryExportReport,
    records: tuple[dict[str, object], ...],
) -> None:
    report.output_dir.mkdir(parents=True, exist_ok=True)
    report.records_path.write_text(
        "".join(_serialize_jsonl_line(record) for record in records),
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
            "memory export row could not be serialized"
        ) from exc


__all__ = [
    "ArchiveMemoryExportContract",
    "ArchiveMemoryExportFilter",
    "ArchiveMemoryExportReport",
    "MEMORY_EXPORT_MANIFEST_FILENAME",
    "MEMORY_EXPORT_OUTPUT_FILENAME",
    "MEMORY_EXPORT_SCHEMA_VERSION",
    "export_archive_memory_records",
]
