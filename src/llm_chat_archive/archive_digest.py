from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .archive_anomalies import ArchiveAnomalyThresholds, summarize_archive_anomalies
from .archive_stats import summarize_archive_stats
from .archive_verify import verify_archive
from .baseline_policy import BaselinePolicy
from .models import TranscriptCompleteness
from .reporting import RunReportingError, load_latest_run_summary
from .validate import ValidationLevel

DEFAULT_TOP_LIMITATIONS = 5
_MISSING_RUN_HISTORY_PREFIXES = (
    "run manifests directory does not exist:",
    "no run manifests found under:",
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _format_utc_timestamp(value: datetime) -> str:
    return (
        value.astimezone(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


@dataclass(frozen=True, slots=True)
class ArchiveDigestLimitationItem:
    limitation: str
    count: int
    raw_count: int | None = None
    suppressed_count: int = 0

    def to_dict(self) -> dict[str, object]:
        payload = {
            "limitation": self.limitation,
            "count": self.count,
        }
        if self.raw_count is not None and self.raw_count != self.count:
            payload["raw_count"] = self.raw_count
        if self.suppressed_count > 0:
            payload["suppressed_count"] = self.suppressed_count
        return payload


@dataclass(frozen=True, slots=True)
class ArchiveDigestReason:
    code: str
    message: str
    suppressed: bool = False
    suppression_entry_id: str | None = None
    suppression_kind: str | None = None
    suppression_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, str | bool | dict[str, str | None]] = {
            "code": self.code,
            "message": self.message,
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
class ArchiveDigestLatestRun:
    run_id: str
    started_at: str | None
    completed_at: str | None
    source_count: int
    failed_source_count: int
    partial_source_count: int
    unsupported_source_count: int
    degraded_source_count: int
    failed_sources: tuple[str, ...]
    degraded_sources: tuple[str, ...]
    raw_degraded_source_count: int | None = None
    suppressed_degraded_source_count: int = 0
    suppressed_degraded_sources: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        payload = {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "source_count": self.source_count,
            "failed_source_count": self.failed_source_count,
            "partial_source_count": self.partial_source_count,
            "unsupported_source_count": self.unsupported_source_count,
            "degraded_source_count": self.degraded_source_count,
            "failed_sources": list(self.failed_sources),
            "degraded_sources": list(self.degraded_sources),
        }
        if (
            self.raw_degraded_source_count is not None
            and self.raw_degraded_source_count != self.degraded_source_count
        ):
            payload["raw_degraded_source_count"] = self.raw_degraded_source_count
        if self.suppressed_degraded_source_count > 0:
            payload["suppressed_degraded_source_count"] = (
                self.suppressed_degraded_source_count
            )
            payload["suppressed_degraded_sources"] = list(
                self.suppressed_degraded_sources
            )
        return payload


@dataclass(frozen=True, slots=True)
class ArchiveDigestSourceReport:
    source: str
    latest_run_selected: bool
    latest_run_status: str | None
    support_level: str | None
    failed: bool
    run_degraded: bool
    attention_required: bool
    file_count: int
    conversation_count: int
    message_count: int
    latest_collected_at: str | None
    conversation_with_limitations_count: int
    transcript_completeness: dict[str, dict[str, float | int]]
    top_limitations: tuple[ArchiveDigestLimitationItem, ...]
    verify_status: str | None
    warning_count: int
    error_count: int
    orphan_file_count: int
    has_orphans: bool
    suspicious: bool
    suspicious_conversation_count: int
    source_reasons: tuple[ArchiveDigestReason, ...]
    raw_run_degraded: bool | None = None
    suppressed_run_degraded: bool = False
    raw_verify_status: str | None = None
    suppressed_warning_count: int = 0
    raw_suspicious_conversation_count: int | None = None

    def to_dict(self) -> dict[str, object]:
        payload = {
            "latest_run_selected": self.latest_run_selected,
            "latest_run_status": self.latest_run_status,
            "support_level": self.support_level,
            "failed": self.failed,
            "run_degraded": self.run_degraded,
            "attention_required": self.attention_required,
            "file_count": self.file_count,
            "conversation_count": self.conversation_count,
            "message_count": self.message_count,
            "latest_collected_at": self.latest_collected_at,
            "conversation_with_limitations_count": (
                self.conversation_with_limitations_count
            ),
            "transcript_completeness": self.transcript_completeness,
            "top_limitations": [
                limitation.to_dict() for limitation in self.top_limitations
            ],
            "verify_status": self.verify_status,
            "warning_count": self.warning_count,
            "error_count": self.error_count,
            "orphan_file_count": self.orphan_file_count,
            "has_orphans": self.has_orphans,
            "suspicious": self.suspicious,
            "suspicious_conversation_count": self.suspicious_conversation_count,
            "source_reasons": [reason.to_dict() for reason in self.source_reasons],
        }
        if self.raw_run_degraded is not None and self.raw_run_degraded != self.run_degraded:
            payload["raw_run_degraded"] = self.raw_run_degraded
            payload["suppressed_run_degraded"] = self.suppressed_run_degraded
        if self.suppressed_warning_count > 0:
            payload["suppressed_warning_count"] = self.suppressed_warning_count
        if self.raw_verify_status is not None and self.raw_verify_status != self.verify_status:
            payload["raw_verify_status"] = self.raw_verify_status
        if (
            self.raw_suspicious_conversation_count is not None
            and self.raw_suspicious_conversation_count
            != self.suspicious_conversation_count
        ):
            payload["raw_suspicious_conversation_count"] = (
                self.raw_suspicious_conversation_count
            )
        return payload


@dataclass(frozen=True, slots=True)
class ArchiveDigestReport:
    archive_root: Path
    aggregated_at: str
    status: str
    raw_status: str | None
    raw_warning_count: int | None
    raw_suspicious_source_count: int | None
    latest_run_id: str | None
    latest_run: ArchiveDigestLatestRun | None
    latest_run_error: str | None
    source_count: int
    conversation_count: int
    message_count: int
    conversation_with_limitations_count: int
    suspicious_source_count: int
    suspicious_conversation_count: int
    sources_with_orphans_count: int
    orphan_file_count: int
    warning_count: int
    error_count: int
    transcript_completeness: dict[str, dict[str, float | int]]
    top_limitations: tuple[ArchiveDigestLimitationItem, ...]
    sources: tuple[ArchiveDigestSourceReport, ...]
    baseline_path: Path | None = None
    baseline_entry_count: int = 0

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "archive_root": str(self.archive_root),
            "aggregated_at": self.aggregated_at,
            "status": self.status,
            "latest_run_id": self.latest_run_id,
            "latest_run": (
                None if self.latest_run is None else self.latest_run.to_dict()
            ),
            "overview": {
                "source_count": self.source_count,
                "conversation_count": self.conversation_count,
                "message_count": self.message_count,
                "conversation_with_limitations_count": (
                    self.conversation_with_limitations_count
                ),
                "suspicious_source_count": self.suspicious_source_count,
                "suspicious_conversation_count": self.suspicious_conversation_count,
                "sources_with_orphans_count": self.sources_with_orphans_count,
                "orphan_file_count": self.orphan_file_count,
                "has_orphans": self.orphan_file_count > 0,
                "warning_count": self.warning_count,
                "error_count": self.error_count,
                "transcript_completeness": self.transcript_completeness,
            },
            "top_limitations": [
                limitation.to_dict() for limitation in self.top_limitations
            ],
            "sources": {
                source_report.source: source_report.to_dict()
                for source_report in self.sources
            },
        }
        if self.raw_status is not None and self.raw_status != self.status:
            payload["raw_status"] = self.raw_status
        if self.raw_warning_count is not None and self.raw_warning_count != self.warning_count:
            payload["overview"]["raw_warning_count"] = self.raw_warning_count
            payload["overview"]["suppressed_warning_count"] = (
                self.raw_warning_count - self.warning_count
            )
        if (
            self.raw_suspicious_source_count is not None
            and self.raw_suspicious_source_count != self.suspicious_source_count
        ):
            payload["overview"]["raw_suspicious_source_count"] = (
                self.raw_suspicious_source_count
            )
            payload["overview"]["suppressed_suspicious_source_count"] = (
                self.raw_suspicious_source_count - self.suspicious_source_count
            )
        if self.latest_run_error is not None:
            payload["latest_run_error"] = self.latest_run_error
        if self.baseline_path is not None:
            payload["baseline"] = {
                "path": str(self.baseline_path),
                "entry_count": self.baseline_entry_count,
            }
        return payload


def summarize_archive_digest(
    archive_root: Path,
    *,
    baseline_policy: BaselinePolicy | None = None,
) -> ArchiveDigestReport:
    resolved_root = archive_root.expanduser().resolve(strict=False)
    stats_report = summarize_archive_stats(resolved_root)
    verify_report = verify_archive(resolved_root, baseline_policy=baseline_policy)
    anomalies_report = summarize_archive_anomalies(
        resolved_root,
        thresholds=ArchiveAnomalyThresholds(),
        baseline_policy=baseline_policy,
    )
    latest_run_summary, latest_run_error = _load_latest_run_summary_optional(
        resolved_root
    )

    stats_payload = stats_report.to_dict()
    source_stats_payloads = {
        source_stats.source: source_stats.to_dict()
        for source_stats in stats_report.sources
    }
    (
        limitation_counts,
        raw_limitation_counts,
        source_limitation_counts,
        source_raw_limitation_counts,
    ) = _collect_limitation_counts(
        resolved_root,
        baseline_policy=baseline_policy,
    )
    verify_sources = {
        source_report.source: source_report for source_report in verify_report.sources
    }
    anomaly_sources = {
        source_report.source: source_report for source_report in anomalies_report.sources
    }
    latest_run_sources = (
        {}
        if latest_run_summary is None
        else {
            source_summary.source: source_summary
            for source_summary in latest_run_summary.sources
        }
    )

    source_names = sorted(
        {
            *source_stats_payloads,
            *verify_sources,
            *anomaly_sources,
            *latest_run_sources,
        }
    )
    source_reports = tuple(
        _build_source_report(
            source_name,
            stats_payload=source_stats_payloads.get(source_name, _empty_stats_payload()),
            verify_source=verify_sources.get(source_name),
            anomaly_source=anomaly_sources.get(source_name),
            latest_run_source=latest_run_sources.get(source_name),
            limitation_counts=source_limitation_counts.get(source_name, Counter()),
            raw_limitation_counts=source_raw_limitation_counts.get(source_name, Counter()),
            baseline_policy=baseline_policy,
        )
        for source_name in source_names
    )

    latest_run = (
        None
        if latest_run_summary is None
        else _build_latest_run_digest(
            latest_run_summary,
            baseline_policy=baseline_policy,
        )
    )

    return ArchiveDigestReport(
        archive_root=resolved_root,
        aggregated_at=_format_utc_timestamp(_utcnow()),
        status=_resolve_digest_status(
            latest_run=latest_run,
            latest_run_error=latest_run_error,
            verify_warning_count=verify_report.warning_count,
            verify_error_count=verify_report.error_count,
            suspicious_source_count=anomalies_report.suspicious_source_count,
        ),
        raw_status=(
            None
            if baseline_policy is None
            else _resolve_digest_status(
                latest_run=latest_run,
                latest_run_error=latest_run_error,
                verify_warning_count=verify_report.raw_warning_count,
                verify_error_count=verify_report.error_count,
                suspicious_source_count=anomalies_report.raw_suspicious_source_count,
                degraded_source_count=(
                    0
                    if latest_run is None
                    else (
                        latest_run.raw_degraded_source_count
                        or latest_run.degraded_source_count
                    )
                ),
            )
        ),
        raw_warning_count=(
            None if baseline_policy is None else verify_report.raw_warning_count
        ),
        raw_suspicious_source_count=(
            None
            if baseline_policy is None
            else anomalies_report.raw_suspicious_source_count
        ),
        latest_run_id=None if latest_run is None else latest_run.run_id,
        latest_run=latest_run,
        latest_run_error=latest_run_error,
        source_count=len(source_reports),
        conversation_count=stats_report.conversation_count,
        message_count=stats_report.message_count,
        conversation_with_limitations_count=(
            stats_report.conversation_with_limitations_count
        ),
        suspicious_source_count=anomalies_report.suspicious_source_count,
        suspicious_conversation_count=anomalies_report.suspicious_conversation_count,
        sources_with_orphans_count=sum(
            1 for source_report in source_reports if source_report.has_orphans
        ),
        orphan_file_count=verify_report.orphan_file_count,
        warning_count=verify_report.warning_count,
        error_count=verify_report.error_count,
        transcript_completeness=stats_payload["transcript_completeness"],
        top_limitations=_build_limitation_items(
            limitation_counts,
            raw_limitation_counts=raw_limitation_counts,
        ),
        sources=source_reports,
        baseline_path=None if baseline_policy is None else baseline_policy.path,
        baseline_entry_count=0 if baseline_policy is None else baseline_policy.entry_count,
    )


def _load_latest_run_summary_optional(
    archive_root: Path,
):
    try:
        return load_latest_run_summary(
            archive_root,
            verify_output_paths=False,
        ), None
    except RunReportingError as exc:
        message = str(exc)
        if message.startswith(_MISSING_RUN_HISTORY_PREFIXES):
            return None, None
        return None, message


def _build_latest_run_digest(
    latest_run_summary,
    *,
    baseline_policy: BaselinePolicy | None,
) -> ArchiveDigestLatestRun:
    failed_sources = tuple(
        sorted(source.source for source in latest_run_summary.sources if source.failed)
    )
    raw_degraded_sources = tuple(
        sorted(
            source.source
            for source in latest_run_summary.sources
            if source.partial or source.unsupported
        )
    )
    degraded_sources: list[str] = []
    suppressed_degraded_sources: list[str] = []
    for source in latest_run_summary.sources:
        if not (source.partial or source.unsupported):
            continue
        matched_entry = (
            None
            if baseline_policy is None
            else baseline_policy.match_degraded_source(
                source=source.source,
                support_level=source.support_level,
                status=source.status,
            )
        )
        if matched_entry is None:
            degraded_sources.append(source.source)
            continue
        suppressed_degraded_sources.append(source.source)
    return ArchiveDigestLatestRun(
        run_id=latest_run_summary.run_id,
        started_at=latest_run_summary.started_at,
        completed_at=latest_run_summary.completed_at,
        source_count=latest_run_summary.source_count,
        failed_source_count=latest_run_summary.failed_source_count,
        partial_source_count=latest_run_summary.partial_source_count,
        unsupported_source_count=latest_run_summary.unsupported_source_count,
        degraded_source_count=len(degraded_sources),
        failed_sources=failed_sources,
        degraded_sources=tuple(sorted(degraded_sources)),
        raw_degraded_source_count=len(raw_degraded_sources),
        suppressed_degraded_source_count=len(suppressed_degraded_sources),
        suppressed_degraded_sources=tuple(sorted(suppressed_degraded_sources)),
    )


def _build_source_report(
    source: str,
    *,
    stats_payload: dict[str, object],
    verify_source,
    anomaly_source,
    latest_run_source,
    limitation_counts: Counter[str],
    raw_limitation_counts: Counter[str],
    baseline_policy: BaselinePolicy | None,
) -> ArchiveDigestSourceReport:
    failed = bool(latest_run_source.failed) if latest_run_source is not None else False
    raw_run_degraded = (
        bool(latest_run_source.partial or latest_run_source.unsupported)
        if latest_run_source is not None
        else False
    )
    suppressed_run_degraded = False
    if raw_run_degraded and latest_run_source is not None and baseline_policy is not None:
        suppressed_run_degraded = (
            baseline_policy.match_degraded_source(
                source=source,
                support_level=latest_run_source.support_level,
                status=latest_run_source.status,
            )
            is not None
        )
    run_degraded = raw_run_degraded and not suppressed_run_degraded
    verify_status = None if verify_source is None else verify_source.status.value
    raw_verify_status = (
        None
        if verify_source is None
        else (verify_source.raw_status or verify_source.status).value
    )
    suspicious = False if anomaly_source is None else anomaly_source.suspicious
    has_orphans = (
        False if verify_source is None else verify_source.orphan_file_count > 0
    )
    attention_required = (
        failed
        or run_degraded
        or (
            verify_source is not None
            and verify_source.status != ValidationLevel.SUCCESS
        )
        or suspicious
    )
    source_reasons = (
        ()
        if anomaly_source is None
        else tuple(
            ArchiveDigestReason(
                code=reason.code,
                message=reason.message,
                suppressed=reason.suppressed,
                suppression_entry_id=reason.suppression_entry_id,
                suppression_kind=reason.suppression_kind,
                suppression_reason=reason.suppression_reason,
            )
            for reason in anomaly_source.source_reasons
        )
    )
    transcript_completeness = dict(stats_payload["transcript_completeness"])

    return ArchiveDigestSourceReport(
        source=source,
        latest_run_selected=latest_run_source is not None,
        latest_run_status=(
            None if latest_run_source is None else latest_run_source.status
        ),
        support_level=(
            None if latest_run_source is None else latest_run_source.support_level
        ),
        failed=failed,
        run_degraded=run_degraded,
        raw_run_degraded=raw_run_degraded,
        suppressed_run_degraded=suppressed_run_degraded,
        attention_required=attention_required,
        file_count=int(stats_payload["file_count"]),
        conversation_count=int(stats_payload["conversation_count"]),
        message_count=int(stats_payload["message_count"]),
        latest_collected_at=stats_payload["latest_collected_at"],
        conversation_with_limitations_count=int(
            stats_payload["conversation_with_limitations_count"]
        ),
        transcript_completeness=transcript_completeness,
        top_limitations=_build_limitation_items(
            limitation_counts,
            raw_limitation_counts=raw_limitation_counts,
        ),
        verify_status=verify_status,
        raw_verify_status=raw_verify_status,
        warning_count=0 if verify_source is None else verify_source.warning_count,
        suppressed_warning_count=(
            0 if verify_source is None else verify_source.suppressed_warning_count
        ),
        error_count=0 if verify_source is None else verify_source.error_count,
        orphan_file_count=0 if verify_source is None else verify_source.orphan_file_count,
        has_orphans=has_orphans,
        suspicious=suspicious,
        suspicious_conversation_count=(
            0
            if anomaly_source is None
            else anomaly_source.suspicious_conversation_count
        ),
        raw_suspicious_conversation_count=(
            None
            if anomaly_source is None
            else anomaly_source.raw_suspicious_conversation_count
        ),
        source_reasons=source_reasons,
    )


def _collect_limitation_counts(
    archive_root: Path,
    *,
    baseline_policy: BaselinePolicy | None,
):
    from .archive_index import collect_indexed_limitation_counts

    return collect_indexed_limitation_counts(
        archive_root,
        baseline_policy=baseline_policy,
    )


def _build_limitation_items(
    limitation_counts: Counter[str],
    *,
    raw_limitation_counts: Counter[str],
) -> tuple[ArchiveDigestLimitationItem, ...]:
    return tuple(
        ArchiveDigestLimitationItem(
            limitation=limitation,
            count=limitation_counts.get(limitation, 0),
            raw_count=raw_limitation_counts.get(limitation, 0),
            suppressed_count=(
                raw_limitation_counts.get(limitation, 0) - limitation_counts.get(limitation, 0)
            ),
        )
        for limitation in sorted(
            raw_limitation_counts,
            key=lambda item: (-raw_limitation_counts[item], item),
        )[:DEFAULT_TOP_LIMITATIONS]
    )


def _empty_stats_payload() -> dict[str, object]:
    return {
        "file_count": 0,
        "conversation_count": 0,
        "message_count": 0,
        "transcript_completeness": {
            completeness.value: {
                "count": 0,
                "ratio": 0.0,
            }
            for completeness in TranscriptCompleteness
        },
        "earliest_collected_at": None,
        "latest_collected_at": None,
        "conversation_with_limitations_count": 0,
        "conversation_with_limitations_ratio": 0.0,
    }


def _resolve_digest_status(
    *,
    latest_run: ArchiveDigestLatestRun | None,
    latest_run_error: str | None,
    verify_warning_count: int,
    verify_error_count: int,
    suspicious_source_count: int,
    degraded_source_count: int | None = None,
) -> str:
    effective_degraded_source_count = (
        degraded_source_count
        if degraded_source_count is not None
        else (
            0
            if latest_run is None
            else latest_run.degraded_source_count
        )
    )
    if (
        verify_error_count > 0
        or latest_run_error is not None
        or (latest_run is not None and latest_run.failed_source_count > 0)
    ):
        return ValidationLevel.ERROR.value
    if (
        verify_warning_count > 0
        or suspicious_source_count > 0
        or effective_degraded_source_count > 0
    ):
        return ValidationLevel.WARNING.value
    return ValidationLevel.SUCCESS.value
