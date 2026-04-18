from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .archive_inspect import ArchiveConversationRecord, build_archive_record
from .models import TranscriptCompleteness

CANONICAL_OUTPUT_TEMPLATE = "memory_chat_v1-{source}.jsonl"

_TRANSCRIPT_COMPLETENESS_RANK = {
    TranscriptCompleteness.UNSUPPORTED.value: 0,
    TranscriptCompleteness.PARTIAL.value: 1,
    TranscriptCompleteness.COMPLETE.value: 2,
}


@dataclass(frozen=True, slots=True)
class ArchiveMergeCandidate:
    source: str
    raw_payload: dict[str, object]
    canonical_payload: dict[str, object]
    serialized_payload: str
    collected_at: str
    source_session_id: str | None
    source_artifact_path: str | None
    transcript_completeness: str
    message_count: int
    text_message_count: int
    text_size: int
    image_count: int
    limitation_count: int
    session_metadata_score: int
    provenance_score: int
    output_path: Path
    row_number: int

    @property
    def identity_key(self) -> tuple[str, str, str] | None:
        if self.source_session_id is not None:
            return ("source_session_id", self.source, self.source_session_id)
        if self.source_artifact_path is not None:
            return ("source_artifact_path", self.source, self.source_artifact_path)
        return None

    @property
    def is_canonical(self) -> bool:
        return self.raw_payload == self.canonical_payload

    @property
    def selection_key(self) -> tuple[object, ...]:
        return (
            _TRANSCRIPT_COMPLETENESS_RANK[self.transcript_completeness],
            self.text_message_count,
            self.message_count,
            self.image_count,
            self.text_size,
            self.session_metadata_score + self.provenance_score,
            -self.limitation_count,
            self.collected_at,
            str(self.output_path),
            self.row_number,
        )


def build_archive_merge_candidate_from_payload(
    payload: Mapping[str, object],
    *,
    output_path: Path,
    row_number: int,
) -> ArchiveMergeCandidate:
    record = build_archive_record(
        dict(payload),
        output_path=output_path,
        line_number=row_number,
    )
    return build_archive_merge_candidate(record)


def build_archive_merge_candidate(
    record: ArchiveConversationRecord,
) -> ArchiveMergeCandidate:
    payload = dict(record.payload)
    canonical_payload = canonicalize_archive_record(record)
    text_message_count = 0
    text_size = 0
    image_count = 0

    for message in record.messages:
        text = message.get("text")
        if isinstance(text, str) and text.strip():
            text_message_count += 1
            text_size += len(text)
        images = message.get("images")
        if isinstance(images, list):
            image_count += len(images)

    session_metadata = payload.get("session_metadata")
    provenance = payload.get("provenance")

    return ArchiveMergeCandidate(
        source=record.summary.source,
        raw_payload=payload,
        canonical_payload=canonical_payload,
        serialized_payload=serialize_archive_payload(canonical_payload),
        collected_at=record.summary.collected_at,
        source_session_id=record.summary.source_session_id,
        source_artifact_path=record.summary.source_artifact_path,
        transcript_completeness=record.summary.transcript_completeness,
        message_count=record.summary.message_count,
        text_message_count=text_message_count,
        text_size=text_size,
        image_count=image_count,
        limitation_count=len(record.summary.limitations),
        session_metadata_score=1 if session_metadata not in (None, {}, []) else 0,
        provenance_score=1 if provenance not in (None, {}) else 0,
        output_path=record.summary.output_path,
        row_number=record.summary.row_number,
    )


def archive_candidate_richness_key(
    candidate: ArchiveMergeCandidate,
) -> tuple[object, ...]:
    return (
        _TRANSCRIPT_COMPLETENESS_RANK[candidate.transcript_completeness],
        candidate.text_message_count,
        candidate.message_count,
        candidate.image_count,
        candidate.text_size,
        candidate.session_metadata_score + candidate.provenance_score,
        -candidate.limitation_count,
    )


def archive_candidate_group_key(
    candidate: ArchiveMergeCandidate,
) -> tuple[str, ...]:
    identity_key = candidate.identity_key
    if identity_key is not None:
        return ("identity", *identity_key)
    return ("exact", candidate.serialized_payload)


def compact_archive_candidates(
    candidates: tuple[ArchiveMergeCandidate, ...],
) -> tuple[tuple[ArchiveMergeCandidate, ...], int, int, int]:
    groups: dict[tuple[str, ...], list[ArchiveMergeCandidate]] = {}
    for candidate in candidates:
        groups.setdefault(archive_candidate_group_key(candidate), []).append(candidate)

    selected: list[ArchiveMergeCandidate] = []
    dropped_row_count = 0
    upgraded_row_count = 0
    untouched_row_count = 0

    for group_key in sorted(groups):
        winner, dropped_count, upgraded = select_archive_group_winner(
            tuple(groups[group_key])
        )
        selected.append(winner)
        dropped_row_count += dropped_count
        if upgraded:
            upgraded_row_count += 1
        else:
            untouched_row_count += 1

    return tuple(selected), dropped_row_count, upgraded_row_count, untouched_row_count


def select_archive_group_winner(
    candidates: tuple[ArchiveMergeCandidate, ...],
) -> tuple[ArchiveMergeCandidate, int, bool]:
    winner = max(candidates, key=lambda candidate: candidate.selection_key)
    dropped_count = len(candidates) - 1
    has_superseded_variant = any(
        candidate.serialized_payload != winner.serialized_payload
        for candidate in candidates
        if candidate is not winner
    )
    has_existing_canonical_equivalent = any(
        candidate.is_canonical
        and candidate.serialized_payload == winner.serialized_payload
        for candidate in candidates
    )
    upgraded = has_superseded_variant or not has_existing_canonical_equivalent
    return winner, dropped_count, upgraded


def archive_candidate_sort_key(
    candidate: ArchiveMergeCandidate,
) -> tuple[object, ...]:
    return (
        candidate.collected_at,
        candidate.source_session_id or "",
        candidate.source_artifact_path or "",
        candidate.serialized_payload,
    )


def canonicalize_archive_record(record: ArchiveConversationRecord) -> dict[str, object]:
    payload = record.payload
    canonical: dict[str, object] = {
        "collected_at": record.summary.collected_at,
        "messages": [_canonicalize_json_value(message) for message in record.messages],
        "source": record.summary.source,
    }

    contract = payload.get("contract")
    if isinstance(contract, Mapping):
        canonical["contract"] = _canonicalize_json_value(dict(contract))

    execution_context = payload.get("execution_context")
    if isinstance(execution_context, str):
        canonical["execution_context"] = execution_context

    if record.summary.limitations:
        canonical["limitations"] = list(record.summary.limitations)

    provenance = payload.get("provenance")
    if isinstance(provenance, Mapping) and provenance:
        canonical["provenance"] = _canonicalize_json_value(dict(provenance))

    session_metadata = payload.get("session_metadata")
    if session_metadata is not None:
        canonical["session_metadata"] = _canonicalize_json_value(session_metadata)

    if record.summary.source_artifact_path is not None:
        canonical["source_artifact_path"] = record.summary.source_artifact_path
    if record.summary.source_session_id is not None:
        canonical["source_session_id"] = record.summary.source_session_id
    if (
        record.summary.transcript_completeness
        != TranscriptCompleteness.COMPLETE.value
    ):
        canonical["transcript_completeness"] = (
            record.summary.transcript_completeness
        )

    for key in sorted(payload):
        if key in canonical or key in {
            "contract",
            "execution_context",
            "limitations",
            "messages",
            "provenance",
            "session_metadata",
            "source_artifact_path",
            "source_session_id",
            "transcript_completeness",
        }:
            continue
        canonical[key] = _canonicalize_json_value(payload[key])

    return canonical


def serialize_archive_payload(payload: dict[str, object]) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _canonicalize_json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _canonicalize_json_value(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, list):
        return [_canonicalize_json_value(item) for item in value]
    return value


__all__ = [
    "ArchiveMergeCandidate",
    "CANONICAL_OUTPUT_TEMPLATE",
    "archive_candidate_richness_key",
    "archive_candidate_group_key",
    "archive_candidate_sort_key",
    "build_archive_merge_candidate",
    "build_archive_merge_candidate_from_payload",
    "canonicalize_archive_record",
    "compact_archive_candidates",
    "select_archive_group_winner",
    "serialize_archive_payload",
]
