from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from .execution_policy import get_collection_execution_policy
from .models import (
    CollectionResult,
    NormalizedConversation,
    RedactionMode,
    TranscriptCompleteness,
)
from .redaction import RedactionResult, redact_archive_payload

OUTPUT_GLOB = "*.jsonl"
SUPERSEDED_AT_FIELD = "superseded_at"

_TRANSCRIPT_COMPLETENESS_RANK = {
    TranscriptCompleteness.UNSUPPORTED.value: 0,
    TranscriptCompleteness.PARTIAL.value: 1,
    TranscriptCompleteness.COMPLETE.value: 2,
}


@dataclass(slots=True)
class _ArchiveFileLineState:
    raw_line: str
    payload: dict[str, object] | None
    updated_payload: dict[str, object] | None = None


@dataclass(frozen=True, slots=True)
class _ExistingArchiveRow:
    candidate: "_ConversationCandidate"
    output_path: Path
    line_index: int


@dataclass(frozen=True, slots=True)
class _PreparedIncomingConversation:
    payload: dict[str, object]
    candidate: "_ConversationCandidate"
    message_count: int
    redaction_event_count: int
    order: int


@dataclass(frozen=True, slots=True)
class _MergeParticipant:
    candidate: "_ConversationCandidate"
    incoming: _PreparedIncomingConversation | None = None
    existing: _ExistingArchiveRow | None = None


@dataclass(frozen=True, slots=True)
class _ConversationCandidate:
    raw_payload: dict[str, object]
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
    def selection_key(self) -> tuple[object, ...]:
        return (
            *_candidate_richness_key(self),
            self.collected_at,
            str(self.output_path),
            self.row_number,
        )


def build_conversation_dedupe_components(
    conversation: NormalizedConversation,
) -> dict[str, str | None]:
    return build_payload_dedupe_components(conversation.to_dict())


def build_conversation_dedupe_key(conversation: NormalizedConversation) -> str:
    return build_payload_dedupe_key(conversation.to_dict())


def build_payload_dedupe_components(
    payload: Mapping[str, object],
) -> dict[str, str | None]:
    source = _string_value(payload.get("source"))
    if source is None:
        raise ValueError("normalized conversation payload requires source")

    messages = payload.get("messages")
    if not isinstance(messages, list):
        raise ValueError("normalized conversation payload requires messages")

    return {
        "source": source,
        "source_session_id": _string_value(payload.get("source_session_id")),
        "source_artifact_path": _string_value(payload.get("source_artifact_path")),
        "message_fingerprint": build_message_fingerprint(messages),
    }


def build_payload_dedupe_key(payload: Mapping[str, object]) -> str:
    components = build_payload_dedupe_components(payload)
    encoded = json.dumps(
        components,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def build_message_fingerprint(messages: list[object]) -> str:
    encoded = json.dumps(
        messages,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_existing_dedupe_keys(archive_root: Path, source: str) -> set[str]:
    source_dir = archive_root / source
    if not source_dir.is_dir():
        return set()

    dedupe_keys: set[str] = set()
    for output_path in sorted(source_dir.glob(OUTPUT_GLOB)):
        dedupe_keys.update(_load_output_dedupe_keys(output_path, source=source))
    return dedupe_keys


def write_incremental_collection(
    *,
    source: str,
    archive_root: Path,
    input_roots: tuple[Path, ...],
    scanned_artifact_count: int,
    collected_at: str,
    conversations: Iterable[NormalizedConversation | None],
    incremental: bool | None = None,
    redaction: RedactionMode | None = None,
) -> CollectionResult:
    execution_policy = get_collection_execution_policy()
    resolved_incremental = (
        execution_policy.incremental if incremental is None else incremental
    )
    resolved_redaction = execution_policy.redaction if redaction is None else redaction
    output_dir = archive_root / source
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"memory_chat_v1-{timestamp_slug(collected_at)}.jsonl"

    if not resolved_incremental:
        return _write_non_incremental_collection(
            source=source,
            archive_root=archive_root,
            input_roots=input_roots,
            scanned_artifact_count=scanned_artifact_count,
            output_path=output_path,
            conversations=conversations,
            redaction=resolved_redaction,
        )

    prepared_conversations = _prepare_incoming_conversations(
        source=source,
        output_path=output_path,
        conversations=conversations,
        redaction=resolved_redaction,
    )
    existing_rows, file_states = _load_existing_archive_rows(archive_root, source=source)
    selected_incoming, superseded_rows, skipped_count, upgraded_count = (
        _merge_incremental_candidates(
            existing_rows=existing_rows,
            incoming_rows=prepared_conversations,
        )
    )

    _mark_rows_superseded(
        file_states=file_states,
        rows=superseded_rows,
        superseded_at=collected_at,
    )
    selected_incoming = tuple(
        sorted(selected_incoming, key=lambda conversation: conversation.order)
    )
    with output_path.open("w", encoding="utf-8") as handle:
        for incoming in selected_incoming:
            handle.write(json.dumps(incoming.payload, ensure_ascii=False))
            handle.write("\n")

    return CollectionResult(
        source=source,
        archive_root=archive_root,
        output_path=output_path,
        input_roots=input_roots,
        scanned_artifact_count=scanned_artifact_count,
        conversation_count=len(prepared_conversations),
        skipped_conversation_count=skipped_count,
        written_conversation_count=len(selected_incoming),
        upgraded_conversation_count=upgraded_count,
        message_count=sum(incoming.message_count for incoming in selected_incoming),
        redaction_event_count=sum(
            incoming.redaction_event_count for incoming in selected_incoming
        ),
    )


def _write_non_incremental_collection(
    *,
    source: str,
    archive_root: Path,
    input_roots: tuple[Path, ...],
    scanned_artifact_count: int,
    output_path: Path,
    conversations: Iterable[NormalizedConversation | None],
    redaction: RedactionMode,
) -> CollectionResult:
    conversation_count = 0
    message_count = 0
    written_conversation_count = 0
    redaction_event_count = 0

    with output_path.open("w", encoding="utf-8") as handle:
        for conversation in conversations:
            if conversation is None:
                continue
            if conversation.source != source:
                raise ValueError(
                    f"conversation source mismatch: expected {source}, got {conversation.source}"
                )

            conversation_count += 1
            prepared_output = _prepare_output_payload(
                conversation.to_dict(),
                redaction=redaction,
            )
            handle.write(json.dumps(prepared_output.payload, ensure_ascii=False))
            handle.write("\n")
            written_conversation_count += 1
            message_count += len(conversation.messages)
            redaction_event_count += prepared_output.event_count

    return CollectionResult(
        source=source,
        archive_root=archive_root,
        output_path=output_path,
        input_roots=input_roots,
        scanned_artifact_count=scanned_artifact_count,
        conversation_count=conversation_count,
        skipped_conversation_count=0,
        written_conversation_count=written_conversation_count,
        upgraded_conversation_count=0,
        message_count=message_count,
        redaction_event_count=redaction_event_count,
    )


def _prepare_incoming_conversations(
    *,
    source: str,
    output_path: Path,
    conversations: Iterable[NormalizedConversation | None],
    redaction: RedactionMode,
) -> tuple[_PreparedIncomingConversation, ...]:
    prepared_conversations: list[_PreparedIncomingConversation] = []
    for order, conversation in enumerate(conversations, start=1):
        if conversation is None:
            continue
        if conversation.source != source:
            raise ValueError(
                f"conversation source mismatch: expected {source}, got {conversation.source}"
            )

        prepared_output = _prepare_output_payload(
            conversation.to_dict(),
            redaction=redaction,
        )
        prepared_conversations.append(
            _PreparedIncomingConversation(
                payload=dict(prepared_output.payload),
                candidate=_build_candidate_from_payload(
                    prepared_output.payload,
                    output_path=output_path,
                    row_number=order,
                ),
                message_count=len(conversation.messages),
                redaction_event_count=prepared_output.event_count,
                order=order,
            )
        )
    return tuple(prepared_conversations)


def _load_existing_archive_rows(
    archive_root: Path,
    *,
    source: str,
) -> tuple[tuple[_ExistingArchiveRow, ...], dict[Path, list[_ArchiveFileLineState]]]:
    source_dir = archive_root / source
    if not source_dir.is_dir():
        return (), {}

    existing_rows: list[_ExistingArchiveRow] = []
    file_states: dict[Path, list[_ArchiveFileLineState]] = {}
    for output_path in sorted(source_dir.glob(OUTPUT_GLOB)):
        try:
            raw_lines = output_path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            continue

        line_states: list[_ArchiveFileLineState] = []
        for line_index, raw_line in enumerate(raw_lines):
            payload = _load_json_line(raw_line)
            line_states.append(_ArchiveFileLineState(raw_line=raw_line, payload=payload))
            if (
                payload is None
                or payload.get("source") != source
                or _is_superseded_payload(payload)
            ):
                continue
            try:
                candidate = _build_candidate_from_payload(
                    payload,
                    output_path=output_path,
                    row_number=line_index + 1,
                )
            except ValueError:
                continue
            existing_rows.append(
                _ExistingArchiveRow(
                    candidate=candidate,
                    output_path=output_path,
                    line_index=line_index,
                )
            )
        file_states[output_path] = line_states

    return tuple(existing_rows), file_states


def _merge_incremental_candidates(
    *,
    existing_rows: tuple[_ExistingArchiveRow, ...],
    incoming_rows: tuple[_PreparedIncomingConversation, ...],
) -> tuple[
    tuple[_PreparedIncomingConversation, ...],
    tuple[_ExistingArchiveRow, ...],
    int,
    int,
]:
    participants = tuple(_iter_merge_participants(existing_rows, incoming_rows))
    selected_incoming: list[_PreparedIncomingConversation] = []
    superseded_rows: list[_ExistingArchiveRow] = []
    skipped_count = 0
    upgraded_count = 0

    for group in _build_merge_groups(participants):
        group_participants = tuple(participants[index] for index in group)
        incoming_participants = tuple(
            participant
            for participant in group_participants
            if participant.incoming is not None
        )
        if not incoming_participants:
            continue

        existing_participants = tuple(
            participant
            for participant in group_participants
            if participant.existing is not None
        )
        incoming_winner = max(
            incoming_participants,
            key=lambda participant: (
                _candidate_richness_key(participant.candidate),
                participant.candidate.selection_key,
            ),
        )
        skipped_count += len(incoming_participants) - 1

        if not existing_participants:
            selected_incoming.append(incoming_winner.incoming)
            continue

        best_existing_richness = max(
            _candidate_richness_key(participant.candidate)
            for participant in existing_participants
        )
        if _candidate_richness_key(incoming_winner.candidate) <= best_existing_richness:
            skipped_count += 1
            continue

        selected_incoming.append(incoming_winner.incoming)
        superseded_rows.extend(
            participant.existing
            for participant in existing_participants
            if participant.existing is not None
        )
        upgraded_count += 1

    return (
        tuple(selected_incoming),
        tuple(superseded_rows),
        skipped_count,
        upgraded_count,
    )


def _iter_merge_participants(
    existing_rows: tuple[_ExistingArchiveRow, ...],
    incoming_rows: tuple[_PreparedIncomingConversation, ...],
) -> Iterable[_MergeParticipant]:
    for existing in existing_rows:
        yield _MergeParticipant(candidate=existing.candidate, existing=existing)
    for incoming in incoming_rows:
        yield _MergeParticipant(candidate=incoming.candidate, incoming=incoming)


def _build_merge_groups(
    participants: tuple[_MergeParticipant, ...],
) -> tuple[tuple[int, ...], ...]:
    if not participants:
        return ()

    parents = list(range(len(participants)))
    session_index: dict[str, list[int]] = {}
    artifact_index: dict[str, list[int]] = {}
    exact_index: dict[str, list[int]] = {}

    for index, participant in enumerate(participants):
        session_id = _normalized_identity_value(participant.candidate.source_session_id)
        artifact_path = _normalized_identity_value(
            participant.candidate.source_artifact_path
        )
        if session_id is not None:
            session_index.setdefault(session_id, []).append(index)
        if artifact_path is not None:
            artifact_index.setdefault(artifact_path, []).append(index)
        if session_id is None and artifact_path is None:
            exact_index.setdefault(
                build_payload_dedupe_key(participant.candidate.raw_payload),
                [],
            ).append(index)

    for indexes in session_index.values():
        _union_component_indexes(parents, indexes)
    for indexes in artifact_index.values():
        _union_component_indexes(parents, indexes)
    for indexes in exact_index.values():
        _union_component_indexes(parents, indexes)

    components: dict[int, list[int]] = {}
    for index in range(len(participants)):
        root = _find_parent(parents, index)
        components.setdefault(root, []).append(index)

    return tuple(
        sorted(
            (tuple(sorted(indexes)) for indexes in components.values()),
            key=lambda indexes: indexes[0],
        )
    )


def _union_component_indexes(parents: list[int], indexes: list[int]) -> None:
    if len(indexes) <= 1:
        return
    first = indexes[0]
    for index in indexes[1:]:
        _union_parents(parents, first, index)


def _find_parent(parents: list[int], index: int) -> int:
    parent = parents[index]
    if parent != index:
        parents[index] = _find_parent(parents, parent)
    return parents[index]


def _union_parents(parents: list[int], left: int, right: int) -> None:
    left_root = _find_parent(parents, left)
    right_root = _find_parent(parents, right)
    if left_root != right_root:
        parents[right_root] = left_root


def _mark_rows_superseded(
    *,
    file_states: dict[Path, list[_ArchiveFileLineState]],
    rows: tuple[_ExistingArchiveRow, ...],
    superseded_at: str,
) -> None:
    modified_paths: set[Path] = set()
    for row in rows:
        line_states = file_states.get(row.output_path)
        if line_states is None or row.line_index >= len(line_states):
            continue
        line_state = line_states[row.line_index]
        if line_state.payload is None:
            continue
        updated_payload = dict(line_state.payload)
        updated_payload[SUPERSEDED_AT_FIELD] = superseded_at
        line_state.updated_payload = updated_payload
        modified_paths.add(row.output_path)

    for output_path in sorted(modified_paths):
        line_states = file_states[output_path]
        serialized_lines = [
            json.dumps(line_state.updated_payload, ensure_ascii=False)
            if line_state.updated_payload is not None
            else line_state.raw_line
            for line_state in line_states
        ]
        payload = "\n".join(serialized_lines)
        if serialized_lines:
            payload += "\n"
        output_path.write_text(payload, encoding="utf-8")


def _build_candidate_from_payload(
    payload: Mapping[str, object],
    *,
    output_path: Path,
    row_number: int,
) -> _ConversationCandidate:
    source = _string_value(payload.get("source"))
    if source is None:
        raise ValueError("normalized conversation payload requires source")
    collected_at = _string_value(payload.get("collected_at"))
    if collected_at is None:
        raise ValueError("normalized conversation payload requires collected_at")
    messages = payload.get("messages")
    if not isinstance(messages, list):
        raise ValueError("normalized conversation payload requires messages")

    text_message_count = 0
    text_size = 0
    image_count = 0
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("normalized conversation payload requires message objects")
        text = message.get("text")
        if isinstance(text, str) and text.strip():
            text_message_count += 1
            text_size += len(text)
        images = message.get("images")
        if isinstance(images, list):
            image_count += len(images)

    limitations = payload.get("limitations", [])
    if not isinstance(limitations, list):
        raise ValueError("normalized conversation payload requires limitations list")
    transcript_completeness = _string_value(
        payload.get(
            "transcript_completeness",
            TranscriptCompleteness.COMPLETE.value,
        )
    )
    if transcript_completeness not in _TRANSCRIPT_COMPLETENESS_RANK:
        raise ValueError("invalid transcript_completeness value")

    session_metadata = payload.get("session_metadata")
    provenance = payload.get("provenance")
    return _ConversationCandidate(
        raw_payload=dict(payload),
        collected_at=collected_at,
        source_session_id=_normalized_identity_value(payload.get("source_session_id")),
        source_artifact_path=_normalized_identity_value(
            payload.get("source_artifact_path")
        ),
        transcript_completeness=transcript_completeness,
        message_count=len(messages),
        text_message_count=text_message_count,
        text_size=text_size,
        image_count=image_count,
        limitation_count=len(limitations),
        session_metadata_score=1 if session_metadata not in (None, {}, []) else 0,
        provenance_score=1 if provenance not in (None, {}) else 0,
        output_path=output_path,
        row_number=row_number,
    )


def _candidate_richness_key(candidate: _ConversationCandidate) -> tuple[object, ...]:
    return (
        _TRANSCRIPT_COMPLETENESS_RANK[candidate.transcript_completeness],
        candidate.text_message_count,
        candidate.message_count,
        candidate.image_count,
        candidate.text_size,
        candidate.session_metadata_score + candidate.provenance_score,
        -candidate.limitation_count,
    )


def _prepare_output_payload(
    payload: Mapping[str, object],
    *,
    redaction: RedactionMode,
) -> RedactionResult:
    if redaction == RedactionMode.OFF:
        return RedactionResult(payload=dict(payload), event_count=0)
    return redact_archive_payload(payload)


def _load_output_dedupe_keys(output_path: Path, *, source: str) -> set[str]:
    try:
        lines = output_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return set()

    dedupe_keys: set[str] = set()
    for raw_line in lines:
        payload = _load_json_line(raw_line)
        if (
            payload is None
            or payload.get("source") != source
            or _is_superseded_payload(payload)
        ):
            continue
        try:
            dedupe_keys.add(build_payload_dedupe_key(payload))
        except ValueError:
            continue
    return dedupe_keys


def _load_json_line(raw_line: str) -> dict[str, object] | None:
    line = raw_line.strip()
    if not line:
        return None
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _normalized_identity_value(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized:
        return None
    return normalized


def _is_superseded_payload(payload: Mapping[str, object]) -> bool:
    value = payload.get(SUPERSEDED_AT_FIELD)
    return isinstance(value, str) and bool(value.strip())


def _string_value(value: object) -> str | None:
    if isinstance(value, str):
        return value
    return None


def timestamp_slug(timestamp: str) -> str:
    return timestamp.replace("-", "").replace(":", "").replace("+00:00", "Z")


__all__ = [
    "build_conversation_dedupe_components",
    "build_conversation_dedupe_key",
    "build_message_fingerprint",
    "build_payload_dedupe_components",
    "build_payload_dedupe_key",
    "load_existing_dedupe_keys",
    "write_incremental_collection",
]
