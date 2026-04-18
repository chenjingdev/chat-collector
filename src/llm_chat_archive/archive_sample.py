from __future__ import annotations

import hashlib
import json
import secrets
from dataclasses import dataclass
from pathlib import Path

from .archive_inspect import (
    ArchiveConversationFilter,
    ArchiveConversationRecord,
    ArchiveConversationSummary,
    iter_archive_records,
)


@dataclass(frozen=True, slots=True)
class ArchiveSampleFilter:
    source: str | None = None
    transcript_completeness: str | None = None
    text: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "transcript_completeness": self.transcript_completeness,
            "text": self.text,
        }


@dataclass(frozen=True, slots=True)
class ArchiveConversationSample:
    conversation: ArchiveConversationSummary
    preview: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload = self.conversation.to_dict()
        if self.preview is not None:
            payload["preview"] = self.preview
        return payload


@dataclass(frozen=True, slots=True)
class ArchiveSampleReport:
    archive_root: Path
    filters: ArchiveSampleFilter
    seed: str
    requested_count: int
    candidate_count: int
    conversation_count: int
    message_count: int
    source_count: int
    conversations: tuple[ArchiveConversationSample, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "archive_root": str(self.archive_root),
            "filters": self.filters.to_dict(),
            "seed": self.seed,
            "requested_count": self.requested_count,
            "candidate_count": self.candidate_count,
            "conversation_count": self.conversation_count,
            "message_count": self.message_count,
            "source_count": self.source_count,
            "conversations": [conversation.to_dict() for conversation in self.conversations],
        }


def sample_archive_subset(
    archive_root: Path,
    *,
    count: int,
    source: str | None = None,
    transcript_completeness: str | None = None,
    text: str | None = None,
    seed: str | None = None,
) -> ArchiveSampleReport:
    if count <= 0:
        raise ValueError("sample count must be greater than zero")

    resolved_archive_root = archive_root.expanduser().resolve(strict=False)
    filters = ArchiveSampleFilter(
        source=source,
        transcript_completeness=transcript_completeness,
        text=_normalize_text_query(text),
    )
    effective_seed = _resolve_seed(seed)
    records = _collect_records(
        resolved_archive_root,
        filters=filters,
    )
    sampled_records = _sample_records(
        records,
        count=count,
        seed=effective_seed,
    )
    conversations = tuple(
        ArchiveConversationSample(
            conversation=record.summary,
            preview=_build_preview(record, filters.text),
        )
        for record in sampled_records
    )
    return ArchiveSampleReport(
        archive_root=resolved_archive_root,
        filters=filters,
        seed=effective_seed,
        requested_count=count,
        candidate_count=len(records),
        conversation_count=len(sampled_records),
        message_count=sum(record.summary.message_count for record in sampled_records),
        source_count=len({record.summary.source for record in sampled_records}),
        conversations=conversations,
    )


def _collect_records(
    archive_root: Path,
    *,
    filters: ArchiveSampleFilter,
) -> tuple[ArchiveConversationRecord, ...]:
    conversation_filter = ArchiveConversationFilter(
        source=filters.source,
        transcript_completeness=filters.transcript_completeness,
    )
    records = [
        record
        for record in iter_archive_records(archive_root, source=filters.source)
        if conversation_filter.matches(record.summary)
        and _matches_text_filter(record, filters.text)
    ]
    return tuple(sorted(records, key=_record_sort_key, reverse=True))


def _sample_records(
    records: tuple[ArchiveConversationRecord, ...],
    *,
    count: int,
    seed: str,
) -> tuple[ArchiveConversationRecord, ...]:
    ranked_records = sorted(
        records,
        key=lambda record: (_seeded_rank(record, seed=seed), _record_sort_key(record)),
    )
    return tuple(ranked_records[:count])


def _seeded_rank(record: ArchiveConversationRecord, *, seed: str) -> str:
    canonical_payload = json.dumps(
        record.payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(seed.encode("utf-8") + b"\0" + canonical_payload).hexdigest()


def _resolve_seed(seed: str | None) -> str:
    if seed is None:
        return secrets.token_hex(8)
    normalized_seed = seed.strip()
    if not normalized_seed:
        raise ValueError("seed must not be empty")
    return normalized_seed


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


def _build_preview(
    record: ArchiveConversationRecord,
    text: str | None,
) -> str | None:
    normalized_query = text.casefold() if text is not None else None
    for message in record.messages:
        message_text = message.get("text")
        if not isinstance(message_text, str):
            continue
        if normalized_query is None or normalized_query in message_text.casefold():
            return _preview_text(message_text)
    return None


def _preview_text(text: str, *, max_length: int = 120) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_length:
        return compact
    return compact[: max_length - 3] + "..."


def _record_sort_key(record: ArchiveConversationRecord) -> tuple[object, ...]:
    summary = record.summary
    return (
        summary.collected_at,
        summary.source,
        summary.source_session_id or "",
        str(summary.output_path),
        summary.row_number,
    )


__all__ = [
    "ArchiveConversationSample",
    "ArchiveSampleFilter",
    "ArchiveSampleReport",
    "sample_archive_subset",
]
