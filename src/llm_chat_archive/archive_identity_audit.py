from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .archive_inspect import ArchiveConversationSummary, iter_archive_records
from .archive_merge import ArchiveMergeCandidate, build_archive_merge_candidate
from .incremental import build_message_fingerprint
from .models import TranscriptCompleteness

_TRANSCRIPT_COMPLETENESS_RANK = {
    TranscriptCompleteness.UNSUPPORTED.value: 0,
    TranscriptCompleteness.PARTIAL.value: 1,
    TranscriptCompleteness.COMPLETE.value: 2,
}


@dataclass(frozen=True, slots=True)
class ArchiveIdentityAuditReason:
    code: str
    message: str
    details: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "details": dict(self.details),
        }


@dataclass(frozen=True, slots=True)
class ArchiveIdentityCollisionConversation:
    conversation: ArchiveConversationSummary
    identity_shape: str
    message_fingerprint: str
    text_message_count: int
    text_size: int
    image_count: int
    limitation_count: int
    has_session_metadata: bool
    preferred: bool

    def to_dict(self) -> dict[str, object]:
        payload = self.conversation.to_dict()
        payload.update(
            {
                "identity_shape": self.identity_shape,
                "message_fingerprint": self.message_fingerprint,
                "text_message_count": self.text_message_count,
                "text_size": self.text_size,
                "image_count": self.image_count,
                "limitation_count": self.limitation_count,
                "has_session_metadata": self.has_session_metadata,
                "preferred": self.preferred,
            }
        )
        return payload


@dataclass(frozen=True, slots=True)
class ArchiveIdentityCollisionGroup:
    source: str
    row_count: int
    source_session_ids: tuple[str, ...]
    source_artifact_paths: tuple[str, ...]
    identity_shapes: tuple[tuple[str, int], ...]
    distinct_message_fingerprint_count: int
    has_richer_conversation: bool
    reasons: tuple[ArchiveIdentityAuditReason, ...]
    preferred_conversation: ArchiveIdentityCollisionConversation
    conversations: tuple[ArchiveIdentityCollisionConversation, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "row_count": self.row_count,
            "source_session_id_count": len(self.source_session_ids),
            "source_artifact_path_count": len(self.source_artifact_paths),
            "source_session_ids": list(self.source_session_ids),
            "source_artifact_paths": list(self.source_artifact_paths),
            "identity_shapes": {
                identity_shape: count
                for identity_shape, count in self.identity_shapes
            },
            "distinct_message_fingerprint_count": (
                self.distinct_message_fingerprint_count
            ),
            "has_richer_conversation": self.has_richer_conversation,
            "reasons": [reason.to_dict() for reason in self.reasons],
            "preferred_conversation": self.preferred_conversation.to_dict(),
            "conversations": [
                conversation.to_dict() for conversation in self.conversations
            ],
        }


@dataclass(frozen=True, slots=True)
class ArchiveIdentitySourceAudit:
    source: str
    file_count: int
    conversation_count: int
    collision_groups: tuple[ArchiveIdentityCollisionGroup, ...]

    @property
    def collision_group_count(self) -> int:
        return len(self.collision_groups)

    @property
    def collision_row_count(self) -> int:
        return sum(group.row_count for group in self.collision_groups)

    @property
    def richer_collision_group_count(self) -> int:
        return sum(1 for group in self.collision_groups if group.has_richer_conversation)

    @property
    def mixed_identity_shape_group_count(self) -> int:
        return sum(
            1
            for group in self.collision_groups
            if len(group.identity_shapes) > 1
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "file_count": self.file_count,
            "conversation_count": self.conversation_count,
            "collision_group_count": self.collision_group_count,
            "collision_row_count": self.collision_row_count,
            "richer_collision_group_count": self.richer_collision_group_count,
            "mixed_identity_shape_group_count": (
                self.mixed_identity_shape_group_count
            ),
            "collision_groups": [group.to_dict() for group in self.collision_groups],
        }


@dataclass(frozen=True, slots=True)
class ArchiveIdentityAuditReport:
    archive_root: Path
    source_filter: str | None
    sources: tuple[ArchiveIdentitySourceAudit, ...]

    @property
    def source_count(self) -> int:
        return len(self.sources)

    @property
    def conversation_count(self) -> int:
        return sum(source.conversation_count for source in self.sources)

    @property
    def collision_source_count(self) -> int:
        return sum(1 for source in self.sources if source.collision_group_count)

    @property
    def collision_group_count(self) -> int:
        return sum(source.collision_group_count for source in self.sources)

    @property
    def collision_row_count(self) -> int:
        return sum(source.collision_row_count for source in self.sources)

    @property
    def richer_collision_group_count(self) -> int:
        return sum(source.richer_collision_group_count for source in self.sources)

    def to_dict(self) -> dict[str, object]:
        return {
            "archive_root": str(self.archive_root),
            "source_filter": self.source_filter,
            "source_count": self.source_count,
            "conversation_count": self.conversation_count,
            "collision_source_count": self.collision_source_count,
            "collision_group_count": self.collision_group_count,
            "collision_row_count": self.collision_row_count,
            "richer_collision_group_count": self.richer_collision_group_count,
            "sources": {
                source_report.source: source_report.to_dict()
                for source_report in self.sources
            },
        }


@dataclass(frozen=True, slots=True)
class _IndexedCandidate:
    candidate: ArchiveMergeCandidate
    summary: ArchiveConversationSummary
    message_fingerprint: str
    identity_shape: str


def audit_archive_identities(
    archive_root: Path,
    *,
    source: str | None = None,
) -> ArchiveIdentityAuditReport:
    resolved_root = archive_root.expanduser().resolve(strict=False)
    source_records: dict[str, list[_IndexedCandidate]] = {}
    source_output_paths: dict[str, set[Path]] = {}

    for record in iter_archive_records(resolved_root, source=source):
        indexed_candidate = _IndexedCandidate(
            candidate=build_archive_merge_candidate(record),
            summary=record.summary,
            message_fingerprint=build_message_fingerprint(list(record.messages)),
            identity_shape=_identity_shape(
                session_id=record.summary.source_session_id,
                artifact_path=record.summary.source_artifact_path,
            ),
        )
        source_records.setdefault(record.summary.source, []).append(indexed_candidate)
        source_output_paths.setdefault(record.summary.source, set()).add(
            record.summary.output_path
        )

    source_reports = tuple(
        _build_source_report(
            source_name=source_name,
            indexed_candidates=tuple(indexed_candidates),
            file_count=len(source_output_paths[source_name]),
        )
        for source_name, indexed_candidates in sorted(source_records.items())
    )
    return ArchiveIdentityAuditReport(
        archive_root=resolved_root,
        source_filter=source,
        sources=source_reports,
    )


def _build_source_report(
    *,
    source_name: str,
    indexed_candidates: tuple[_IndexedCandidate, ...],
    file_count: int,
) -> ArchiveIdentitySourceAudit:
    collision_groups = tuple(
        sorted(
            (
                _build_collision_group(
                    source_name=source_name,
                    indexed_candidates=tuple(
                        indexed_candidates[index] for index in component
                    ),
                )
                for component in _find_collision_components(indexed_candidates)
            ),
            key=_collision_group_sort_key,
            reverse=True,
        )
    )
    return ArchiveIdentitySourceAudit(
        source=source_name,
        file_count=file_count,
        conversation_count=len(indexed_candidates),
        collision_groups=collision_groups,
    )


def _find_collision_components(
    indexed_candidates: tuple[_IndexedCandidate, ...],
) -> tuple[tuple[int, ...], ...]:
    if not indexed_candidates:
        return ()

    parents = list(range(len(indexed_candidates)))
    session_index: dict[str, list[int]] = {}
    artifact_index: dict[str, list[int]] = {}

    for index, indexed_candidate in enumerate(indexed_candidates):
        session_id = _normalized_identity_value(
            indexed_candidate.candidate.source_session_id
        )
        artifact_path = _normalized_identity_value(
            indexed_candidate.candidate.source_artifact_path
        )
        if session_id is not None:
            session_index.setdefault(session_id, []).append(index)
        if artifact_path is not None:
            artifact_index.setdefault(artifact_path, []).append(index)

    for indexes in session_index.values():
        _union_component_indexes(parents, indexes)
    for indexes in artifact_index.values():
        _union_component_indexes(parents, indexes)

    components: dict[int, list[int]] = {}
    for index in range(len(indexed_candidates)):
        root = _find_parent(parents, index)
        components.setdefault(root, []).append(index)

    collision_components = [
        tuple(sorted(indexes))
        for indexes in components.values()
        if len(indexes) > 1
    ]
    return tuple(sorted(collision_components))


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


def _build_collision_group(
    *,
    source_name: str,
    indexed_candidates: tuple[_IndexedCandidate, ...],
) -> ArchiveIdentityCollisionGroup:
    preferred_indexed_candidate = max(
        indexed_candidates,
        key=lambda indexed_candidate: indexed_candidate.candidate.selection_key,
    )
    preferred_richness = _richness_key(preferred_indexed_candidate.candidate)
    message_fingerprints = {
        indexed_candidate.message_fingerprint for indexed_candidate in indexed_candidates
    }
    identity_shapes = Counter(
        indexed_candidate.identity_shape for indexed_candidate in indexed_candidates
    )
    source_session_ids = tuple(
        sorted(
            {
                session_id
                for indexed_candidate in indexed_candidates
                if (
                    session_id := _normalized_identity_value(
                        indexed_candidate.candidate.source_session_id
                    )
                )
                is not None
            }
        )
    )
    source_artifact_paths = tuple(
        sorted(
            {
                artifact_path
                for indexed_candidate in indexed_candidates
                if (
                    artifact_path := _normalized_identity_value(
                        indexed_candidate.candidate.source_artifact_path
                    )
                )
                is not None
            }
        )
    )
    conversations = tuple(
        _build_collision_conversation(
            indexed_candidate=indexed_candidate,
            preferred=indexed_candidate is preferred_indexed_candidate,
        )
        for indexed_candidate in sorted(
            indexed_candidates,
            key=lambda indexed_candidate: indexed_candidate.candidate.selection_key,
            reverse=True,
        )
    )
    has_richer_conversation = len(
        {
            _richness_key(indexed_candidate.candidate)
            for indexed_candidate in indexed_candidates
        }
    ) > 1
    reasons = _build_collision_reasons(
        indexed_candidates=indexed_candidates,
        preferred_indexed_candidate=preferred_indexed_candidate,
        preferred_richness=preferred_richness,
        identity_shapes=identity_shapes,
        source_session_ids=source_session_ids,
        source_artifact_paths=source_artifact_paths,
        message_fingerprints=message_fingerprints,
        has_richer_conversation=has_richer_conversation,
    )
    return ArchiveIdentityCollisionGroup(
        source=source_name,
        row_count=len(indexed_candidates),
        source_session_ids=source_session_ids,
        source_artifact_paths=source_artifact_paths,
        identity_shapes=tuple(sorted(identity_shapes.items())),
        distinct_message_fingerprint_count=len(message_fingerprints),
        has_richer_conversation=has_richer_conversation,
        reasons=reasons,
        preferred_conversation=conversations[0],
        conversations=conversations,
    )


def _build_collision_conversation(
    *,
    indexed_candidate: _IndexedCandidate,
    preferred: bool,
) -> ArchiveIdentityCollisionConversation:
    candidate = indexed_candidate.candidate
    return ArchiveIdentityCollisionConversation(
        conversation=indexed_candidate.summary,
        identity_shape=indexed_candidate.identity_shape,
        message_fingerprint=indexed_candidate.message_fingerprint,
        text_message_count=candidate.text_message_count,
        text_size=candidate.text_size,
        image_count=candidate.image_count,
        limitation_count=candidate.limitation_count,
        has_session_metadata=bool(candidate.session_metadata_score),
        preferred=preferred,
    )


def _build_collision_reasons(
    *,
    indexed_candidates: tuple[_IndexedCandidate, ...],
    preferred_indexed_candidate: _IndexedCandidate,
    preferred_richness: tuple[object, ...],
    identity_shapes: Counter[str],
    source_session_ids: tuple[str, ...],
    source_artifact_paths: tuple[str, ...],
    message_fingerprints: set[str],
    has_richer_conversation: bool,
) -> tuple[ArchiveIdentityAuditReason, ...]:
    reasons: list[ArchiveIdentityAuditReason] = []

    repeated_session_ids = {
        session_id: count
        for session_id, count in Counter(
            _normalized_identity_value(indexed_candidate.candidate.source_session_id)
            for indexed_candidate in indexed_candidates
        ).items()
        if session_id is not None and count > 1
    }
    if repeated_session_ids:
        reasons.append(
            ArchiveIdentityAuditReason(
                code="duplicate_source_session_id",
                message="source_session_id가 같은 row가 여러 개 존재합니다.",
                details={"source_session_ids": repeated_session_ids},
            )
        )

    repeated_artifact_paths = {
        artifact_path: count
        for artifact_path, count in Counter(
            _normalized_identity_value(indexed_candidate.candidate.source_artifact_path)
            for indexed_candidate in indexed_candidates
        ).items()
        if artifact_path is not None and count > 1
    }
    if repeated_artifact_paths:
        reasons.append(
            ArchiveIdentityAuditReason(
                code="duplicate_source_artifact_path",
                message="source_artifact_path가 같은 row가 여러 개 존재합니다.",
                details={"source_artifact_paths": repeated_artifact_paths},
            )
        )

    if len(source_session_ids) > 1:
        reasons.append(
            ArchiveIdentityAuditReason(
                code="conflicting_source_session_ids",
                message="같은 충돌군 안에 서로 다른 source_session_id가 연결되어 있습니다.",
                details={"source_session_ids": list(source_session_ids)},
            )
        )

    if len(source_artifact_paths) > 1:
        reasons.append(
            ArchiveIdentityAuditReason(
                code="conflicting_source_artifact_paths",
                message="같은 충돌군 안에 서로 다른 source_artifact_path가 연결되어 있습니다.",
                details={"source_artifact_paths": list(source_artifact_paths)},
            )
        )

    if len(identity_shapes) > 1:
        reasons.append(
            ArchiveIdentityAuditReason(
                code="mixed_identity_shapes",
                message="같은 충돌군이 서로 다른 identity shape으로 저장되어 있습니다.",
                details={"identity_shapes": dict(sorted(identity_shapes.items()))},
            )
        )

    if len(message_fingerprints) > 1:
        reasons.append(
            ArchiveIdentityAuditReason(
                code="message_variants",
                message="같은 충돌군 안에 서로 다른 message payload variant가 있습니다.",
                details={
                    "distinct_message_fingerprint_count": len(message_fingerprints)
                },
            )
        )

    if has_richer_conversation:
        preferred_summary = preferred_indexed_candidate.summary
        reasons.append(
            ArchiveIdentityAuditReason(
                code="richer_transcript_available",
                message="같은 충돌군 안에서 더 풍부한 transcript row가 확인되었습니다.",
                details={
                    "preferred_source_session_id": preferred_summary.source_session_id,
                    "preferred_source_artifact_path": (
                        preferred_summary.source_artifact_path
                    ),
                    "preferred_output_path": str(preferred_summary.output_path),
                    "preferred_row_number": preferred_summary.row_number,
                    "preferred_richness": {
                        "transcript_completeness": (
                            preferred_indexed_candidate.candidate.transcript_completeness
                        ),
                        "message_count": preferred_summary.message_count,
                        "text_message_count": (
                            preferred_indexed_candidate.candidate.text_message_count
                        ),
                        "image_count": preferred_indexed_candidate.candidate.image_count,
                        "text_size": preferred_indexed_candidate.candidate.text_size,
                        "limitation_count": (
                            preferred_indexed_candidate.candidate.limitation_count
                        ),
                        "has_session_metadata": bool(
                            preferred_indexed_candidate.candidate.session_metadata_score
                        ),
                        "has_provenance": bool(
                            preferred_indexed_candidate.candidate.provenance_score
                        ),
                    },
                    "richness_key": [*preferred_richness],
                },
            )
        )

    return tuple(reasons)


def _identity_shape(*, session_id: str | None, artifact_path: str | None) -> str:
    normalized_session_id = _normalized_identity_value(session_id)
    normalized_artifact_path = _normalized_identity_value(artifact_path)
    if normalized_session_id is not None and normalized_artifact_path is not None:
        return "session_and_artifact"
    if normalized_session_id is not None:
        return "session_only"
    if normalized_artifact_path is not None:
        return "artifact_only"
    return "unidentified"


def _richness_key(candidate: ArchiveMergeCandidate) -> tuple[object, ...]:
    return (
        _TRANSCRIPT_COMPLETENESS_RANK[candidate.transcript_completeness],
        candidate.text_message_count,
        candidate.message_count,
        candidate.image_count,
        candidate.text_size,
        candidate.session_metadata_score + candidate.provenance_score,
        -candidate.limitation_count,
    )


def _normalized_identity_value(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        return None
    return normalized


def _collision_group_sort_key(
    group: ArchiveIdentityCollisionGroup,
) -> tuple[object, ...]:
    preferred = group.preferred_conversation.conversation
    return (
        preferred.collected_at,
        group.source,
        preferred.source_session_id or "",
        preferred.source_artifact_path or "",
        str(preferred.output_path),
        preferred.row_number,
    )


__all__ = [
    "ArchiveIdentityAuditReason",
    "ArchiveIdentityAuditReport",
    "ArchiveIdentityCollisionConversation",
    "ArchiveIdentityCollisionGroup",
    "ArchiveIdentitySourceAudit",
    "audit_archive_identities",
]
