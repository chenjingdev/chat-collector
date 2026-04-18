from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .archive_inspect import (
    ArchiveConversationRecord,
    ArchiveInspectError,
)
from .models import MessageRole, TranscriptCompleteness

MESSAGE_ROLE_ORDER = tuple(role.value for role in MessageRole)
VALID_MESSAGE_ROLES = frozenset(MESSAGE_ROLE_ORDER)
TRANSCRIPT_COMPLETENESS_ORDER = tuple(
    completeness.value for completeness in TranscriptCompleteness
)


@dataclass(frozen=True, slots=True)
class ArchiveSourceProfile:
    source: str
    file_count: int
    conversation_count: int
    message_count: int
    message_role_counts: tuple[tuple[str, int], ...]
    transcript_completeness_counts: tuple[tuple[str, int], ...]
    limitation_counts: tuple[tuple[str, int], ...]
    conversation_with_limitations_count: int

    def to_dict(self) -> dict[str, object]:
        return _build_profile_payload(
            file_count=self.file_count,
            conversation_count=self.conversation_count,
            message_count=self.message_count,
            message_role_counts=self.message_role_counts,
            transcript_completeness_counts=self.transcript_completeness_counts,
            limitation_counts=self.limitation_counts,
            conversation_with_limitations_count=(
                self.conversation_with_limitations_count
            ),
        )


@dataclass(frozen=True, slots=True)
class ArchiveProfileReport:
    archive_root: Path
    source_filter: str | None
    file_count: int
    conversation_count: int
    message_count: int
    message_role_counts: tuple[tuple[str, int], ...]
    transcript_completeness_counts: tuple[tuple[str, int], ...]
    limitation_counts: tuple[tuple[str, int], ...]
    conversation_with_limitations_count: int
    sources: tuple[ArchiveSourceProfile, ...]

    def to_dict(self) -> dict[str, object]:
        payload = _build_profile_payload(
            file_count=self.file_count,
            conversation_count=self.conversation_count,
            message_count=self.message_count,
            message_role_counts=self.message_role_counts,
            transcript_completeness_counts=self.transcript_completeness_counts,
            limitation_counts=self.limitation_counts,
            conversation_with_limitations_count=(
                self.conversation_with_limitations_count
            ),
        )
        payload.update(
            {
                "archive_root": str(self.archive_root),
                "source_count": len(self.sources),
                "source_filter": self.source_filter,
                "sources": {
                    source_profile.source: source_profile.to_dict()
                    for source_profile in self.sources
                },
            }
        )
        return payload


@dataclass(slots=True)
class _ProfileAccumulator:
    output_paths: set[Path] = field(default_factory=set)
    conversation_count: int = 0
    message_count: int = 0
    message_role_counts: dict[str, int] = field(
        default_factory=lambda: {role: 0 for role in MESSAGE_ROLE_ORDER}
    )
    transcript_completeness_counts: dict[str, int] = field(
        default_factory=lambda: {
            completeness: 0 for completeness in TRANSCRIPT_COMPLETENESS_ORDER
        }
    )
    conversation_with_limitations_count: int = 0
    limitation_counts: dict[str, int] = field(default_factory=dict)

    def add(self, record: ArchiveConversationRecord) -> None:
        summary = record.summary
        self.output_paths.add(summary.output_path)
        self.conversation_count += 1
        self.message_count += summary.message_count
        self.transcript_completeness_counts[summary.transcript_completeness] += 1

        if summary.limitations:
            self.conversation_with_limitations_count += 1
        for limitation in summary.limitations:
            self.limitation_counts[limitation] = (
                self.limitation_counts.get(limitation, 0) + 1
            )

        for message_index, message in enumerate(record.messages, start=1):
            role = message.get("role")
            if not isinstance(role, str) or role not in VALID_MESSAGE_ROLES:
                raise ArchiveInspectError(
                    "invalid message role at "
                    f"{summary.output_path}:{summary.row_number} message={message_index}"
                )
            self.message_role_counts[role] += 1

    def finalized_message_role_counts(self) -> tuple[tuple[str, int], ...]:
        return tuple(
            (role, self.message_role_counts[role]) for role in MESSAGE_ROLE_ORDER
        )

    def finalized_transcript_completeness_counts(self) -> tuple[tuple[str, int], ...]:
        return tuple(
            (
                completeness,
                self.transcript_completeness_counts[completeness],
            )
            for completeness in TRANSCRIPT_COMPLETENESS_ORDER
        )

    def finalized_limitation_counts(self) -> tuple[tuple[str, int], ...]:
        return tuple(
            sorted(
                self.limitation_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )
        )


def summarize_archive_profile(
    archive_root: Path,
    *,
    source: str | None = None,
) -> ArchiveProfileReport:
    from .archive_index import summarize_indexed_archive_profile

    return summarize_indexed_archive_profile(
        archive_root,
        source=source,
    )


def _build_profile_payload(
    *,
    file_count: int,
    conversation_count: int,
    message_count: int,
    message_role_counts: tuple[tuple[str, int], ...],
    transcript_completeness_counts: tuple[tuple[str, int], ...],
    limitation_counts: tuple[tuple[str, int], ...],
    conversation_with_limitations_count: int,
) -> dict[str, object]:
    return {
        "file_count": file_count,
        "conversation_count": conversation_count,
        "message_count": message_count,
        "message_roles": {
            role: {
                "count": count,
                "ratio": _ratio(count, message_count),
            }
            for role, count in message_role_counts
        },
        "transcript_completeness": {
            completeness: {
                "count": count,
                "ratio": _ratio(count, conversation_count),
            }
            for completeness, count in transcript_completeness_counts
        },
        "conversation_with_limitations_count": conversation_with_limitations_count,
        "conversation_with_limitations_ratio": _ratio(
            conversation_with_limitations_count,
            conversation_count,
        ),
        "limitations": {
            limitation: {
                "count": count,
                "ratio": _ratio(count, conversation_count),
            }
            for limitation, count in limitation_counts
        },
    }


def _ratio(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return count / total
