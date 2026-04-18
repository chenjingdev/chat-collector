from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

from .archive_inspect import ArchiveConversationSummary, iter_archive_records
from .baseline_policy import BaselinePolicy
from .models import TranscriptCompleteness

DEFAULT_LOW_MESSAGE_COUNT_THRESHOLD = 1
DEFAULT_LIMITATIONS_COUNT_THRESHOLD = 2
DEFAULT_UNSUPPORTED_COUNT_THRESHOLD = 2
DEFAULT_UNSUPPORTED_RATIO_THRESHOLD = 0.5


@dataclass(frozen=True, slots=True)
class ArchiveAnomalyThresholds:
    low_message_count: int = DEFAULT_LOW_MESSAGE_COUNT_THRESHOLD
    limitations_count: int = DEFAULT_LIMITATIONS_COUNT_THRESHOLD
    unsupported_count: int = DEFAULT_UNSUPPORTED_COUNT_THRESHOLD
    unsupported_ratio: float = DEFAULT_UNSUPPORTED_RATIO_THRESHOLD

    def validate(self) -> None:
        if self.low_message_count < 0:
            raise ValueError("low_message_count threshold must be non-negative")
        if self.limitations_count < 0:
            raise ValueError("limitations_count threshold must be non-negative")
        if self.unsupported_count < 0:
            raise ValueError("unsupported_count threshold must be non-negative")
        if not 0.0 <= self.unsupported_ratio <= 1.0:
            raise ValueError("unsupported_ratio threshold must be between 0.0 and 1.0")

    def to_dict(self) -> dict[str, object]:
        return {
            "low_message_count": self.low_message_count,
            "limitations_count": self.limitations_count,
            "unsupported_count": self.unsupported_count,
            "unsupported_ratio": self.unsupported_ratio,
        }


@dataclass(frozen=True, slots=True)
class ArchiveAnomalyReason:
    code: str
    message: str
    details: dict[str, object]
    suppressed: bool = False
    suppression_entry_id: str | None = None
    suppression_kind: str | None = None
    suppression_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload = {
            "code": self.code,
            "message": self.message,
            "details": dict(self.details),
        }
        if self.suppressed:
            payload["suppressed"] = True
            payload["suppressed_by"] = {
                "entry_id": self.suppression_entry_id,
                "kind": self.suppression_kind,
                "reason": self.suppression_reason,
            }
        return payload


@dataclass(frozen=True, slots=True)
class ArchiveConversationAnomaly:
    conversation: ArchiveConversationSummary
    reasons: tuple[ArchiveAnomalyReason, ...]

    @property
    def active_reason_count(self) -> int:
        return sum(1 for reason in self.reasons if not reason.suppressed)

    @property
    def suppressed_reason_count(self) -> int:
        return len(self.reasons) - self.active_reason_count

    @property
    def suspicious(self) -> bool:
        return self.active_reason_count > 0

    def to_dict(self) -> dict[str, object]:
        payload = self.conversation.to_dict()
        payload["reasons"] = [reason.to_dict() for reason in self.reasons]
        if self.suppressed_reason_count > 0:
            payload["suspicious"] = self.suspicious
            payload["suppressed_reason_count"] = self.suppressed_reason_count
        return payload


@dataclass(frozen=True, slots=True)
class ArchiveSourceAnomalies:
    source: str
    file_count: int
    conversation_count: int
    message_count: int
    suspicious_conversations: tuple[ArchiveConversationAnomaly, ...]
    low_message_count_conversation_count: int
    excessive_limitations_conversation_count: int
    unsupported_conversation_count: int
    source_reasons: tuple[ArchiveAnomalyReason, ...]

    @property
    def suspicious_conversation_count(self) -> int:
        return sum(
            1 for conversation in self.suspicious_conversations if conversation.suspicious
        )

    @property
    def raw_suspicious_conversation_count(self) -> int:
        return len(self.suspicious_conversations)

    @property
    def suppressed_suspicious_conversation_count(self) -> int:
        return (
            self.raw_suspicious_conversation_count - self.suspicious_conversation_count
        )

    @property
    def source_reason_count(self) -> int:
        return sum(1 for reason in self.source_reasons if not reason.suppressed)

    @property
    def suppressed_source_reason_count(self) -> int:
        return len(self.source_reasons) - self.source_reason_count

    @property
    def unsupported_conversation_ratio(self) -> float:
        return _ratio(self.unsupported_conversation_count, self.conversation_count)

    @property
    def suspicious(self) -> bool:
        return bool(self.source_reason_count or self.suspicious_conversation_count)

    def to_dict(self) -> dict[str, object]:
        payload = {
            "file_count": self.file_count,
            "conversation_count": self.conversation_count,
            "message_count": self.message_count,
            "suspicious": self.suspicious,
            "suspicious_conversation_count": self.suspicious_conversation_count,
            "low_message_count_conversation_count": (
                self.low_message_count_conversation_count
            ),
            "excessive_limitations_conversation_count": (
                self.excessive_limitations_conversation_count
            ),
            "unsupported_conversation_count": self.unsupported_conversation_count,
            "unsupported_conversation_ratio": self.unsupported_conversation_ratio,
            "source_reasons": [reason.to_dict() for reason in self.source_reasons],
            "suspicious_conversations": [
                conversation.to_dict()
                for conversation in self.suspicious_conversations
            ],
        }
        if self.suppressed_source_reason_count > 0:
            payload["suppressed_source_reason_count"] = (
                self.suppressed_source_reason_count
            )
        if self.suppressed_suspicious_conversation_count > 0:
            payload["raw_suspicious_conversation_count"] = (
                self.raw_suspicious_conversation_count
            )
            payload["suppressed_suspicious_conversation_count"] = (
                self.suppressed_suspicious_conversation_count
            )
        return payload


@dataclass(frozen=True, slots=True)
class ArchiveAnomaliesReport:
    archive_root: Path
    source_filter: str | None
    thresholds: ArchiveAnomalyThresholds
    sources: tuple[ArchiveSourceAnomalies, ...]
    baseline_path: Path | None = None
    baseline_entry_count: int = 0

    @property
    def source_count(self) -> int:
        return len(self.sources)

    @property
    def conversation_count(self) -> int:
        return sum(source.conversation_count for source in self.sources)

    @property
    def message_count(self) -> int:
        return sum(source.message_count for source in self.sources)

    @property
    def suspicious_source_count(self) -> int:
        return sum(1 for source in self.sources if source.suspicious)

    @property
    def raw_suspicious_source_count(self) -> int:
        return sum(
            1
            for source in self.sources
            if source.source_reasons or source.suspicious_conversations
        )

    @property
    def source_with_aggregate_reasons_count(self) -> int:
        return sum(1 for source in self.sources if source.source_reason_count > 0)

    @property
    def raw_source_with_aggregate_reasons_count(self) -> int:
        return sum(1 for source in self.sources if source.source_reasons)

    @property
    def suspicious_conversation_count(self) -> int:
        return sum(source.suspicious_conversation_count for source in self.sources)

    @property
    def raw_suspicious_conversation_count(self) -> int:
        return sum(source.raw_suspicious_conversation_count for source in self.sources)

    def to_dict(self) -> dict[str, object]:
        payload = {
            "archive_root": str(self.archive_root),
            "source_filter": self.source_filter,
            "thresholds": self.thresholds.to_dict(),
            "source_count": self.source_count,
            "conversation_count": self.conversation_count,
            "message_count": self.message_count,
            "suspicious_source_count": self.suspicious_source_count,
            "source_with_aggregate_reasons_count": (
                self.source_with_aggregate_reasons_count
            ),
            "suspicious_conversation_count": self.suspicious_conversation_count,
            "sources": {
                source_report.source: source_report.to_dict()
                for source_report in self.sources
            },
        }
        if self.raw_suspicious_source_count != self.suspicious_source_count:
            payload["raw_suspicious_source_count"] = self.raw_suspicious_source_count
        if (
            self.raw_source_with_aggregate_reasons_count
            != self.source_with_aggregate_reasons_count
        ):
            payload["raw_source_with_aggregate_reasons_count"] = (
                self.raw_source_with_aggregate_reasons_count
            )
        if self.raw_suspicious_conversation_count != self.suspicious_conversation_count:
            payload["raw_suspicious_conversation_count"] = (
                self.raw_suspicious_conversation_count
            )
        if self.baseline_path is not None:
            payload["baseline"] = {
                "path": str(self.baseline_path),
                "entry_count": self.baseline_entry_count,
            }
        return payload


@dataclass(slots=True)
class _SourceAccumulator:
    output_paths: set[Path] = field(default_factory=set)
    conversation_count: int = 0
    message_count: int = 0
    low_message_count_conversation_count: int = 0
    excessive_limitations_conversation_count: int = 0
    unsupported_conversation_count: int = 0
    suspicious_conversations: list[ArchiveConversationAnomaly] = field(
        default_factory=list
    )

    def add(
        self,
        summary: ArchiveConversationSummary,
        *,
        thresholds: ArchiveAnomalyThresholds,
    ) -> None:
        self.output_paths.add(summary.output_path)
        self.conversation_count += 1
        self.message_count += summary.message_count

        reasons: list[ArchiveAnomalyReason] = []
        if summary.message_count <= thresholds.low_message_count:
            self.low_message_count_conversation_count += 1
            reasons.append(
                ArchiveAnomalyReason(
                    code="low_message_count",
                    message=(
                        f"message_count {summary.message_count} is at or below "
                        f"threshold {thresholds.low_message_count}"
                    ),
                    details={
                        "message_count": summary.message_count,
                        "threshold": thresholds.low_message_count,
                    },
                )
            )

        limitations_count = len(summary.limitations)
        if limitations_count > 0 and limitations_count >= thresholds.limitations_count:
            self.excessive_limitations_conversation_count += 1
            reasons.append(
                ArchiveAnomalyReason(
                    code="excessive_limitations",
                    message=(
                        f"limitations_count {limitations_count} meets or exceeds "
                        f"threshold {thresholds.limitations_count}"
                    ),
                    details={
                        "limitations_count": limitations_count,
                        "threshold": thresholds.limitations_count,
                        "limitations": list(summary.limitations),
                    },
                )
            )

        if (
            summary.transcript_completeness
            == TranscriptCompleteness.UNSUPPORTED.value
        ):
            self.unsupported_conversation_count += 1
            reasons.append(
                ArchiveAnomalyReason(
                    code="unsupported_transcript",
                    message="transcript_completeness is unsupported",
                    details={
                        "transcript_completeness": summary.transcript_completeness,
                    },
                )
            )

        if reasons:
            self.suspicious_conversations.append(
                ArchiveConversationAnomaly(
                    conversation=summary,
                    reasons=tuple(reasons),
                )
            )

    def finalize(
        self,
        *,
        source: str,
        thresholds: ArchiveAnomalyThresholds,
    ) -> ArchiveSourceAnomalies:
        unsupported_ratio = _ratio(
            self.unsupported_conversation_count,
            self.conversation_count,
        )
        source_reasons: list[ArchiveAnomalyReason] = []
        if (
            self.unsupported_conversation_count >= thresholds.unsupported_count
            and unsupported_ratio >= thresholds.unsupported_ratio
        ):
            source_reasons.append(
                ArchiveAnomalyReason(
                    code="high_unsupported_ratio",
                    message=(
                        f"unsupported_conversation_ratio {unsupported_ratio:.3f} "
                        f"meets or exceeds threshold {thresholds.unsupported_ratio:.3f}"
                    ),
                    details={
                        "unsupported_conversation_count": (
                            self.unsupported_conversation_count
                        ),
                        "conversation_count": self.conversation_count,
                        "unsupported_conversation_ratio": unsupported_ratio,
                        "count_threshold": thresholds.unsupported_count,
                        "ratio_threshold": thresholds.unsupported_ratio,
                    },
                )
            )

        suspicious_conversations = tuple(
            sorted(
                self.suspicious_conversations,
                key=_conversation_sort_key,
                reverse=True,
            )
        )
        return ArchiveSourceAnomalies(
            source=source,
            file_count=len(self.output_paths),
            conversation_count=self.conversation_count,
            message_count=self.message_count,
            suspicious_conversations=suspicious_conversations,
            low_message_count_conversation_count=(
                self.low_message_count_conversation_count
            ),
            excessive_limitations_conversation_count=(
                self.excessive_limitations_conversation_count
            ),
            unsupported_conversation_count=self.unsupported_conversation_count,
            source_reasons=tuple(source_reasons),
        )


def summarize_archive_anomalies(
    archive_root: Path,
    *,
    source: str | None = None,
    thresholds: ArchiveAnomalyThresholds | None = None,
    baseline_policy: BaselinePolicy | None = None,
) -> ArchiveAnomaliesReport:
    resolved_root = archive_root.expanduser().resolve(strict=False)
    effective_thresholds = thresholds or ArchiveAnomalyThresholds()
    effective_thresholds.validate()

    accumulators: dict[str, _SourceAccumulator] = {}
    for record in iter_archive_records(resolved_root, source=source):
        summary = record.summary
        accumulator = accumulators.setdefault(summary.source, _SourceAccumulator())
        accumulator.add(summary, thresholds=effective_thresholds)

    source_reports = tuple(
        accumulator.finalize(source=source_name, thresholds=effective_thresholds)
        for source_name, accumulator in sorted(accumulators.items())
    )
    report = ArchiveAnomaliesReport(
        archive_root=resolved_root,
        source_filter=source,
        thresholds=effective_thresholds,
        sources=source_reports,
    )
    if baseline_policy is not None:
        return _apply_baseline_policy(report, baseline_policy=baseline_policy)
    return report


def _ratio(count: int, total: int) -> float:
    if total == 0:
        return 0.0
    return count / total


def _conversation_sort_key(
    anomaly: ArchiveConversationAnomaly,
) -> tuple[object, ...]:
    summary = anomaly.conversation
    return (
        summary.collected_at,
        summary.source_session_id or "",
        str(summary.output_path),
        summary.row_number,
    )


def _apply_baseline_policy(
    report: ArchiveAnomaliesReport,
    *,
    baseline_policy: BaselinePolicy,
) -> ArchiveAnomaliesReport:
    sources = tuple(
        _apply_baseline_to_source(source_report, baseline_policy=baseline_policy)
        for source_report in report.sources
    )
    return replace(
        report,
        sources=sources,
        baseline_path=baseline_policy.path,
        baseline_entry_count=baseline_policy.entry_count,
    )


def _apply_baseline_to_source(
    source_report: ArchiveSourceAnomalies,
    *,
    baseline_policy: BaselinePolicy,
) -> ArchiveSourceAnomalies:
    source_reasons = tuple(
        _apply_baseline_to_reason(
            source_report.source,
            reason,
            limitations=(),
            baseline_policy=baseline_policy,
        )
        for reason in source_report.source_reasons
    )
    suspicious_conversations = tuple(
        _apply_baseline_to_conversation(
            source_report.source,
            conversation,
            baseline_policy=baseline_policy,
        )
        for conversation in source_report.suspicious_conversations
    )
    return replace(
        source_report,
        source_reasons=source_reasons,
        suspicious_conversations=suspicious_conversations,
    )


def _apply_baseline_to_conversation(
    source: str,
    conversation: ArchiveConversationAnomaly,
    *,
    baseline_policy: BaselinePolicy,
) -> ArchiveConversationAnomaly:
    return replace(
        conversation,
        reasons=tuple(
            _apply_baseline_to_reason(
                source,
                reason,
                limitations=conversation.conversation.limitations,
                baseline_policy=baseline_policy,
            )
            for reason in conversation.reasons
        ),
    )


def _apply_baseline_to_reason(
    source: str,
    reason: ArchiveAnomalyReason,
    *,
    limitations: tuple[str, ...],
    baseline_policy: BaselinePolicy,
) -> ArchiveAnomalyReason:
    matched_entry = None
    if reason.code in {"unsupported_transcript", "high_unsupported_ratio"}:
        matched_entry = baseline_policy.match_degraded_source(
            source=source,
            support_level=None,
            status=None,
            ignore_state=True,
        )
    elif reason.code == "excessive_limitations" and limitations:
        matching_entries = [
            baseline_policy.match_limitation(source=source, limitation=limitation)
            for limitation in limitations
        ]
        if all(entry is not None for entry in matching_entries):
            matched_entry = matching_entries[0]

    if matched_entry is None:
        return reason
    return replace(
        reason,
        suppressed=True,
        suppression_entry_id=matched_entry.id,
        suppression_kind=matched_entry.kind.value,
        suppression_reason=matched_entry.reason,
    )


__all__ = [
    "ArchiveAnomaliesReport",
    "ArchiveAnomalyReason",
    "ArchiveAnomalyThresholds",
    "ArchiveConversationAnomaly",
    "ArchiveSourceAnomalies",
    "DEFAULT_LIMITATIONS_COUNT_THRESHOLD",
    "DEFAULT_LOW_MESSAGE_COUNT_THRESHOLD",
    "DEFAULT_UNSUPPORTED_COUNT_THRESHOLD",
    "DEFAULT_UNSUPPORTED_RATIO_THRESHOLD",
    "summarize_archive_anomalies",
]
