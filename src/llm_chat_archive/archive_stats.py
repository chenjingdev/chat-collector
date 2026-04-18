from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from .archive_inspect import (
    ArchiveConversationSummary,
    ArchiveInspectError,
    build_archive_record,
    load_archive_json_line,
)
from .models import TranscriptCompleteness

TRANSCRIPT_COMPLETENESS_ORDER = tuple(
    completeness.value for completeness in TranscriptCompleteness
)


@dataclass(frozen=True, slots=True)
class ArchiveSourceStats:
    source: str
    file_count: int
    conversation_count: int
    message_count: int
    transcript_completeness_counts: tuple[tuple[str, int], ...]
    earliest_collected_at: str | None
    latest_collected_at: str | None
    conversation_with_limitations_count: int

    def to_dict(self) -> dict[str, object]:
        return _build_stats_payload(
            file_count=self.file_count,
            conversation_count=self.conversation_count,
            message_count=self.message_count,
            transcript_completeness_counts=self.transcript_completeness_counts,
            earliest_collected_at=self.earliest_collected_at,
            latest_collected_at=self.latest_collected_at,
            conversation_with_limitations_count=self.conversation_with_limitations_count,
        )


@dataclass(frozen=True, slots=True)
class ArchiveStatsSnapshot:
    file_count: int
    conversation_count: int
    message_count: int
    transcript_completeness_counts: tuple[tuple[str, int], ...]
    earliest_collected_at: str | None
    latest_collected_at: str | None
    conversation_with_limitations_count: int

    def to_dict(self) -> dict[str, object]:
        return _build_stats_payload(
            file_count=self.file_count,
            conversation_count=self.conversation_count,
            message_count=self.message_count,
            transcript_completeness_counts=self.transcript_completeness_counts,
            earliest_collected_at=self.earliest_collected_at,
            latest_collected_at=self.latest_collected_at,
            conversation_with_limitations_count=self.conversation_with_limitations_count,
        )


@dataclass(frozen=True, slots=True)
class ArchiveStatsReport:
    archive_root: Path
    source_filter: str | None
    file_count: int
    conversation_count: int
    message_count: int
    transcript_completeness_counts: tuple[tuple[str, int], ...]
    earliest_collected_at: str | None
    latest_collected_at: str | None
    conversation_with_limitations_count: int
    sources: tuple[ArchiveSourceStats, ...]

    def to_dict(self) -> dict[str, object]:
        payload = _build_stats_payload(
            file_count=self.file_count,
            conversation_count=self.conversation_count,
            message_count=self.message_count,
            transcript_completeness_counts=self.transcript_completeness_counts,
            earliest_collected_at=self.earliest_collected_at,
            latest_collected_at=self.latest_collected_at,
            conversation_with_limitations_count=self.conversation_with_limitations_count,
        )
        payload.update(
            {
                "archive_root": str(self.archive_root),
                "source_count": len(self.sources),
                "source_filter": self.source_filter,
                "sources": {
                    source_stats.source: source_stats.to_dict()
                    for source_stats in self.sources
                },
            }
        )
        return payload


@dataclass(slots=True)
class _StatsAccumulator:
    output_paths: set[Path] = field(default_factory=set)
    conversation_count: int = 0
    message_count: int = 0
    transcript_completeness_counts: dict[str, int] = field(
        default_factory=lambda: {
            completeness: 0 for completeness in TRANSCRIPT_COMPLETENESS_ORDER
        }
    )
    earliest_collected_at: str | None = None
    latest_collected_at: str | None = None
    conversation_with_limitations_count: int = 0

    def add(self, summary: ArchiveConversationSummary) -> None:
        self.output_paths.add(summary.output_path)
        self.conversation_count += 1
        self.message_count += summary.message_count
        self.transcript_completeness_counts[summary.transcript_completeness] += 1
        if (
            self.earliest_collected_at is None
            or summary.collected_at < self.earliest_collected_at
        ):
            self.earliest_collected_at = summary.collected_at
        if self.latest_collected_at is None or summary.collected_at > self.latest_collected_at:
            self.latest_collected_at = summary.collected_at
        if summary.limitations:
            self.conversation_with_limitations_count += 1

    def finalized_counts(self) -> tuple[tuple[str, int], ...]:
        return tuple(
            (
                completeness,
                self.transcript_completeness_counts[completeness],
            )
            for completeness in TRANSCRIPT_COMPLETENESS_ORDER
        )


def summarize_archive_stats(
    archive_root: Path,
    *,
    source: str | None = None,
) -> ArchiveStatsReport:
    from .archive_index import summarize_indexed_archive_stats

    return summarize_indexed_archive_stats(
        archive_root,
        source=source,
    )


def summarize_archive_output_paths(
    output_paths: Sequence[Path],
    *,
    source: str | None = None,
) -> ArchiveStatsSnapshot:
    accumulator = _StatsAccumulator()

    for output_path in output_paths:
        resolved_output_path = output_path.expanduser().resolve(strict=False)
        if not resolved_output_path.exists():
            raise ArchiveInspectError(
                f"archive output does not exist: {resolved_output_path}"
            )
        if not resolved_output_path.is_file():
            raise ArchiveInspectError(
                f"archive output is not a file: {resolved_output_path}"
            )

        with resolved_output_path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                payload = load_archive_json_line(
                    raw_line,
                    output_path=resolved_output_path,
                    line_number=line_number,
                )
                if payload is None:
                    continue
                record = build_archive_record(
                    payload,
                    output_path=resolved_output_path,
                    line_number=line_number,
                )
                if source is not None and record.summary.source != source:
                    continue
                accumulator.add(record.summary)

    return _snapshot_from_accumulator(accumulator)


def _build_stats_payload(
    *,
    file_count: int,
    conversation_count: int,
    message_count: int,
    transcript_completeness_counts: tuple[tuple[str, int], ...],
    earliest_collected_at: str | None,
    latest_collected_at: str | None,
    conversation_with_limitations_count: int,
) -> dict[str, object]:
    return {
        "file_count": file_count,
        "conversation_count": conversation_count,
        "message_count": message_count,
        "transcript_completeness": {
            completeness: {
                "count": count,
                "ratio": _ratio(count, conversation_count),
            }
            for completeness, count in transcript_completeness_counts
        },
        "earliest_collected_at": earliest_collected_at,
        "latest_collected_at": latest_collected_at,
        "conversation_with_limitations_count": conversation_with_limitations_count,
        "conversation_with_limitations_ratio": _ratio(
            conversation_with_limitations_count,
            conversation_count,
        ),
    }


def _ratio(count: int, total: int) -> float:
    if total == 0:
        return 0.0
    return count / total


def _snapshot_from_accumulator(accumulator: _StatsAccumulator) -> ArchiveStatsSnapshot:
    return ArchiveStatsSnapshot(
        file_count=len(accumulator.output_paths),
        conversation_count=accumulator.conversation_count,
        message_count=accumulator.message_count,
        transcript_completeness_counts=accumulator.finalized_counts(),
        earliest_collected_at=accumulator.earliest_collected_at,
        latest_collected_at=accumulator.latest_collected_at,
        conversation_with_limitations_count=(
            accumulator.conversation_with_limitations_count
        ),
    )


__all__ = [
    "ArchiveStatsSnapshot",
    "ArchiveSourceStats",
    "ArchiveStatsReport",
    "summarize_archive_stats",
    "summarize_archive_output_paths",
]
