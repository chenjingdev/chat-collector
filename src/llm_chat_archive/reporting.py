from __future__ import annotations

from collections import Counter
import json
from dataclasses import dataclass
from pathlib import Path

from .archive_inspect import ArchiveInspectError
from .archive_stats import ArchiveStatsSnapshot, summarize_archive_output_paths
from .models import SourceRunStatus
from .runner import MANIFEST_FILENAME, RUNS_DIRECTORY


class RunReportingError(ValueError):
    """Raised when a recorded run manifest cannot be loaded or summarized."""


_STATUS_ORDER = {
    "failed": 0,
    "unsupported": 1,
    "partial": 2,
    "complete": 3,
}
_STATUS_VALUES = tuple(status.value for status in SourceRunStatus)


@dataclass(frozen=True, slots=True)
class ReportedSourceSummary:
    source: str
    support_level: str
    status: str
    output_path: Path | None
    scanned_artifact_count: int
    conversation_count: int
    message_count: int
    skipped_conversation_count: int
    written_conversation_count: int
    upgraded_conversation_count: int
    partial: bool
    unsupported: bool
    failed: bool
    failure_reason: str | None = None
    support_limitation_summary: str | None = None
    support_limitations: tuple[str, ...] = ()
    redaction_event_count: int = 0

    @property
    def success(self) -> bool:
        return not self.failed

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "source": self.source,
            "support_level": self.support_level,
            "status": self.status,
            "output_path": str(self.output_path) if self.output_path is not None else None,
            "scanned_artifact_count": self.scanned_artifact_count,
            "conversation_count": self.conversation_count,
            "message_count": self.message_count,
            "skipped_conversation_count": self.skipped_conversation_count,
            "written_conversation_count": self.written_conversation_count,
            "upgraded_conversation_count": self.upgraded_conversation_count,
            "success": self.success,
            "failed": self.failed,
            "partial": self.partial,
            "unsupported": self.unsupported,
            "redaction_event_count": self.redaction_event_count,
        }
        if self.failure_reason is not None:
            payload["failure_reason"] = self.failure_reason
        if self.support_limitation_summary is not None:
            payload["support_limitation_summary"] = self.support_limitation_summary
        if self.support_limitations:
            payload["support_limitations"] = list(self.support_limitations)
        return payload


@dataclass(frozen=True, slots=True)
class RunSummary:
    run_id: str
    started_at: str | None
    completed_at: str | None
    archive_root: Path
    run_dir: Path
    manifest_path: Path
    selection_policy: dict[str, object] | None
    effective_config: dict[str, object] | None
    rerun: dict[str, object] | None
    scheduled: dict[str, object] | None
    selected_sources: tuple[str, ...]
    excluded_sources: tuple[dict[str, object], ...]
    sources: tuple[ReportedSourceSummary, ...]

    @property
    def source_count(self) -> int:
        return len(self.sources)

    @property
    def failed_source_count(self) -> int:
        return sum(1 for source in self.sources if source.failed)

    @property
    def partial_source_count(self) -> int:
        return sum(1 for source in self.sources if source.partial)

    @property
    def unsupported_source_count(self) -> int:
        return sum(1 for source in self.sources if source.unsupported)

    @property
    def scanned_artifact_count(self) -> int:
        return sum(source.scanned_artifact_count for source in self.sources)

    @property
    def conversation_count(self) -> int:
        return sum(source.conversation_count for source in self.sources)

    @property
    def message_count(self) -> int:
        return sum(source.message_count for source in self.sources)

    @property
    def skipped_conversation_count(self) -> int:
        return sum(source.skipped_conversation_count for source in self.sources)

    @property
    def written_conversation_count(self) -> int:
        return sum(source.written_conversation_count for source in self.sources)

    @property
    def upgraded_conversation_count(self) -> int:
        return sum(source.upgraded_conversation_count for source in self.sources)

    @property
    def redaction_event_count(self) -> int:
        return sum(source.redaction_event_count for source in self.sources)

    def to_overview_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "archive_root": str(self.archive_root),
            "run_dir": str(self.run_dir),
            "manifest_path": str(self.manifest_path),
            "source_count": self.source_count,
            "failed_source_count": self.failed_source_count,
            "partial_source_count": self.partial_source_count,
            "unsupported_source_count": self.unsupported_source_count,
            "scanned_artifact_count": self.scanned_artifact_count,
            "conversation_count": self.conversation_count,
            "skipped_conversation_count": self.skipped_conversation_count,
            "written_conversation_count": self.written_conversation_count,
            "upgraded_conversation_count": self.upgraded_conversation_count,
            "message_count": self.message_count,
            "redaction_event_count": self.redaction_event_count,
        }
        if self.scheduled is not None:
            payload["scheduled"] = self.scheduled
        return payload

    def to_dict(self) -> dict[str, object]:
        payload = self.to_overview_dict()
        if self.selection_policy is not None:
            payload["selection_policy"] = self.selection_policy
            payload["selected_sources"] = list(self.selected_sources)
            payload["excluded_sources"] = list(self.excluded_sources)
        if self.effective_config is not None:
            payload["effective_config"] = self.effective_config
        if self.rerun is not None:
            payload["rerun"] = self.rerun
        if self.scheduled is not None:
            payload["scheduled"] = self.scheduled
        payload["sources"] = [source.to_dict() for source in self.sources]
        return payload


@dataclass(frozen=True, slots=True)
class CountDelta:
    from_value: int
    to_value: int

    @property
    def delta(self) -> int:
        return self.to_value - self.from_value

    def to_dict(self) -> dict[str, int]:
        return {
            "from": self.from_value,
            "to": self.to_value,
            "delta": self.delta,
        }


@dataclass(frozen=True, slots=True)
class StatusTransition:
    source: str
    from_status: str
    to_status: str
    label: str
    category: str

    def to_dict(self) -> dict[str, str]:
        return {
            "source": self.source,
            "from_status": self.from_status,
            "to_status": self.to_status,
            "label": self.label,
            "category": self.category,
        }


@dataclass(frozen=True, slots=True)
class ExcludedSourceDiff:
    source: str
    from_entry: dict[str, object]
    to_entry: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "from": self.from_entry,
            "to": self.to_entry,
            "changed": self.from_entry != self.to_entry,
        }


@dataclass(frozen=True, slots=True)
class SelectionPolicyDiff:
    from_policy: dict[str, object] | None
    to_policy: dict[str, object] | None
    added_selected_sources: tuple[str, ...] = ()
    removed_selected_sources: tuple[str, ...] = ()
    added_excluded_sources: tuple[dict[str, object], ...] = ()
    removed_excluded_sources: tuple[dict[str, object], ...] = ()
    changed_excluded_sources: tuple[ExcludedSourceDiff, ...] = ()

    @property
    def changed(self) -> bool:
        return (
            self.from_policy != self.to_policy
            or bool(self.added_selected_sources)
            or bool(self.removed_selected_sources)
            or bool(self.added_excluded_sources)
            or bool(self.removed_excluded_sources)
            or bool(self.changed_excluded_sources)
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "changed": self.changed,
            "from": self.from_policy,
            "to": self.to_policy,
            "selected_sources_added": list(self.added_selected_sources),
            "selected_sources_removed": list(self.removed_selected_sources),
            "excluded_sources_added": list(self.added_excluded_sources),
            "excluded_sources_removed": list(self.removed_excluded_sources),
            "excluded_sources_changed": [
                entry.to_dict() for entry in self.changed_excluded_sources
            ],
        }


@dataclass(frozen=True, slots=True)
class SourceDiff:
    source: str
    from_source: ReportedSourceSummary | None
    to_source: ReportedSourceSummary | None
    scanned_artifact_count: CountDelta
    conversation_count: CountDelta
    message_count: CountDelta
    skipped_conversation_count: CountDelta
    written_conversation_count: CountDelta
    redaction_event_count: CountDelta

    @property
    def change_type(self) -> str:
        if self.from_source is None:
            return "added"
        if self.to_source is None:
            return "removed"
        if self.changed:
            return "changed"
        return "unchanged"

    @property
    def changed(self) -> bool:
        return (
            self._support_level() != self._support_level(to=False)
            or self._status() != self._status(to=False)
            or self._output_path() != self._output_path(to=False)
            or self._failure_reason() != self._failure_reason(to=False)
            or self.scanned_artifact_count.delta != 0
            or self.conversation_count.delta != 0
            or self.message_count.delta != 0
            or self.skipped_conversation_count.delta != 0
            or self.written_conversation_count.delta != 0
            or self.redaction_event_count.delta != 0
        )

    @property
    def status_transition(self) -> StatusTransition | None:
        from_status = self._status(to=False)
        to_status = self._status()
        if from_status is None or to_status is None or from_status == to_status:
            return None
        return StatusTransition(
            source=self.source,
            from_status=from_status,
            to_status=to_status,
            label=f"{from_status}_to_{to_status}",
            category=_classify_status_transition(from_status, to_status),
        )

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "source": self.source,
            "change_type": self.change_type,
            "changed": self.changed,
            "support_level": _value_change(
                self._support_level(to=False),
                self._support_level(),
            ),
            "status": _value_change(
                self._status(to=False),
                self._status(),
            ),
            "output_path": _value_change(
                self._output_path(to=False),
                self._output_path(),
            ),
            "failure_reason": _value_change(
                self._failure_reason(to=False),
                self._failure_reason(),
            ),
            "counts": {
                "scanned_artifact_count": self.scanned_artifact_count.to_dict(),
                "conversation_count": self.conversation_count.to_dict(),
                "message_count": self.message_count.to_dict(),
                "skipped_conversation_count": self.skipped_conversation_count.to_dict(),
                "written_conversation_count": self.written_conversation_count.to_dict(),
                "redaction_event_count": self.redaction_event_count.to_dict(),
            },
        }
        if self.status_transition is not None:
            payload["important_transition"] = self.status_transition.to_dict()
        return payload

    def _support_level(self, *, to: bool = True) -> str | None:
        source = self.to_source if to else self.from_source
        return source.support_level if source is not None else None

    def _status(self, *, to: bool = True) -> str | None:
        source = self.to_source if to else self.from_source
        return source.status if source is not None else None

    def _output_path(self, *, to: bool = True) -> str | None:
        source = self.to_source if to else self.from_source
        if source is None or source.output_path is None:
            return None
        return str(source.output_path)

    def _failure_reason(self, *, to: bool = True) -> str | None:
        source = self.to_source if to else self.from_source
        return source.failure_reason if source is not None else None


@dataclass(frozen=True, slots=True)
class RunDiff:
    archive_root: Path
    comparison_mode: str
    from_run: RunSummary
    to_run: RunSummary
    selection_policy: SelectionPolicyDiff
    sources: tuple[SourceDiff, ...]

    @property
    def new_sources(self) -> tuple[str, ...]:
        return tuple(
            diff.source for diff in self.sources if diff.change_type == "added"
        )

    @property
    def removed_sources(self) -> tuple[str, ...]:
        return tuple(
            diff.source for diff in self.sources if diff.change_type == "removed"
        )

    @property
    def important_transitions(self) -> tuple[StatusTransition, ...]:
        transitions: list[StatusTransition] = []
        for source in self.sources:
            transition = source.status_transition
            if transition is not None:
                transitions.append(transition)
        return tuple(transitions)

    def to_dict(self) -> dict[str, object]:
        return {
            "archive_root": str(self.archive_root),
            "comparison_mode": self.comparison_mode,
            "from_run": self.from_run.to_overview_dict(),
            "to_run": self.to_run.to_overview_dict(),
            "counts": {
                "source_count": _count_delta(
                    self.from_run.source_count,
                    self.to_run.source_count,
                ).to_dict(),
                "failed_source_count": _count_delta(
                    self.from_run.failed_source_count,
                    self.to_run.failed_source_count,
                ).to_dict(),
                "partial_source_count": _count_delta(
                    self.from_run.partial_source_count,
                    self.to_run.partial_source_count,
                ).to_dict(),
                "unsupported_source_count": _count_delta(
                    self.from_run.unsupported_source_count,
                    self.to_run.unsupported_source_count,
                ).to_dict(),
                "scanned_artifact_count": _count_delta(
                    self.from_run.scanned_artifact_count,
                    self.to_run.scanned_artifact_count,
                ).to_dict(),
                "conversation_count": _count_delta(
                    self.from_run.conversation_count,
                    self.to_run.conversation_count,
                ).to_dict(),
                "message_count": _count_delta(
                    self.from_run.message_count,
                    self.to_run.message_count,
                ).to_dict(),
                "skipped_conversation_count": _count_delta(
                    self.from_run.skipped_conversation_count,
                    self.to_run.skipped_conversation_count,
                ).to_dict(),
                "written_conversation_count": _count_delta(
                    self.from_run.written_conversation_count,
                    self.to_run.written_conversation_count,
                ).to_dict(),
                "redaction_event_count": _count_delta(
                    self.from_run.redaction_event_count,
                    self.to_run.redaction_event_count,
                ).to_dict(),
            },
            "selection_policy": self.selection_policy.to_dict(),
            "new_sources": list(self.new_sources),
            "removed_sources": list(self.removed_sources),
            "important_transitions": [
                transition.to_dict() for transition in self.important_transitions
            ],
            "sources": [source.to_dict() for source in self.sources],
        }


@dataclass(frozen=True, slots=True)
class RunTrendTransition:
    source: str
    from_run_id: str
    to_run_id: str
    from_status: str
    to_status: str
    label: str
    category: str

    @property
    def degraded_to_complete(self) -> bool:
        return (
            self.from_status in {"partial", "unsupported"}
            and self.to_status == "complete"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "from_run_id": self.from_run_id,
            "to_run_id": self.to_run_id,
            "from_status": self.from_status,
            "to_status": self.to_status,
            "label": self.label,
            "category": self.category,
            "degraded_to_complete": self.degraded_to_complete,
        }


@dataclass(frozen=True, slots=True)
class SourceHealthTrendPoint:
    run_id: str
    started_at: str | None
    completed_at: str | None
    manifest: ReportedSourceSummary
    archive_stats: ArchiveStatsSnapshot
    status_ratios: dict[str, float]
    transition_from_previous: RunTrendTransition | None = None
    archive_stats_error: str | None = None

    @property
    def degraded(self) -> bool:
        return self.manifest.partial or self.manifest.unsupported

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "degraded": self.degraded,
            "manifest": self.manifest.to_dict(),
            "status_ratios": self.status_ratios,
            "archive_stats": self.archive_stats.to_dict(),
            "transition_from_previous": (
                None
                if self.transition_from_previous is None
                else self.transition_from_previous.to_dict()
            ),
        }
        if self.archive_stats_error is not None:
            payload["archive_stats_error"] = self.archive_stats_error
        return payload


@dataclass(frozen=True, slots=True)
class SourceHealthTrend:
    source: str
    first_run_id: str
    latest_run_id: str
    latest_status: str
    latest_support_level: str
    support_levels: tuple[str, ...]
    status_counts: dict[str, int]
    transitions: tuple[RunTrendTransition, ...]
    timeline: tuple[SourceHealthTrendPoint, ...]

    @property
    def run_count(self) -> int:
        return len(self.timeline)

    @property
    def status_ratios(self) -> dict[str, float]:
        return _build_status_ratios(self.status_counts, total=self.run_count)

    @property
    def degraded_to_complete_count(self) -> int:
        return sum(1 for transition in self.transitions if transition.degraded_to_complete)

    @property
    def transition_counts(self) -> dict[str, int]:
        counts = Counter(transition.label for transition in self.transitions)
        return {label: counts[label] for label in sorted(counts)}

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "run_count": self.run_count,
            "first_run_id": self.first_run_id,
            "latest_run_id": self.latest_run_id,
            "latest_status": self.latest_status,
            "latest_support_level": self.latest_support_level,
            "support_levels": list(self.support_levels),
            "status_counts": self.status_counts,
            "status_ratios": self.status_ratios,
            "transition_count": len(self.transitions),
            "degraded_to_complete_count": self.degraded_to_complete_count,
            "transition_counts": self.transition_counts,
            "transitions": [transition.to_dict() for transition in self.transitions],
            "timeline": [point.to_dict() for point in self.timeline],
        }


@dataclass(frozen=True, slots=True)
class RunTrendReport:
    archive_root: Path
    source_filter: tuple[str, ...]
    runs: tuple[RunSummary, ...]
    sources: tuple[SourceHealthTrend, ...]

    @property
    def transition_count(self) -> int:
        return sum(len(source.transitions) for source in self.sources)

    @property
    def degraded_to_complete_count(self) -> int:
        return sum(source.degraded_to_complete_count for source in self.sources)

    def to_dict(self) -> dict[str, object]:
        return {
            "archive_root": str(self.archive_root),
            "source_filter": list(self.source_filter) if self.source_filter else None,
            "run_count": len(self.runs),
            "source_count": len(self.sources),
            "transition_count": self.transition_count,
            "degraded_to_complete_count": self.degraded_to_complete_count,
            "runs": [run.to_overview_dict() for run in self.runs],
            "sources": {
                source.source: source.to_dict()
                for source in self.sources
            },
        }


def list_run_summaries(archive_root: Path) -> tuple[RunSummary, ...]:
    resolved_root = archive_root.expanduser().resolve(strict=False)
    manifest_paths = _list_manifest_paths(resolved_root)
    return tuple(
        _load_run_summary_from_manifest(
            manifest_path,
            archive_root=resolved_root,
            verify_output_paths=False,
        )
        for manifest_path in manifest_paths
    )


def load_latest_run_summary(
    archive_root: Path,
    *,
    verify_output_paths: bool = True,
) -> RunSummary:
    resolved_root = archive_root.expanduser().resolve(strict=False)
    manifest_path = _list_manifest_paths(resolved_root)[0]
    return _load_run_summary_from_manifest(
        manifest_path,
        archive_root=resolved_root,
        verify_output_paths=verify_output_paths,
    )


def load_run_summary(
    archive_root: Path,
    run_id: str,
    *,
    verify_output_paths: bool = True,
) -> RunSummary:
    resolved_root = archive_root.expanduser().resolve(strict=False)
    manifest_path = resolved_root / RUNS_DIRECTORY / run_id / MANIFEST_FILENAME
    return _load_run_summary_from_manifest(
        manifest_path,
        archive_root=resolved_root,
        verify_output_paths=verify_output_paths,
    )


def load_run_diff(
    archive_root: Path,
    *,
    from_run_id: str | None = None,
    to_run_id: str | None = None,
) -> RunDiff:
    resolved_root = archive_root.expanduser().resolve(strict=False)
    if from_run_id is None and to_run_id is None:
        manifest_paths = _list_manifest_paths(resolved_root)
        if len(manifest_paths) < 2:
            raise RunReportingError(
                f"run diff requires at least two run manifests under: "
                f"{resolved_root / RUNS_DIRECTORY}"
            )
        from_summary = _load_run_summary_from_manifest(
            manifest_paths[1],
            archive_root=resolved_root,
            verify_output_paths=False,
        )
        to_summary = _load_run_summary_from_manifest(
            manifest_paths[0],
            archive_root=resolved_root,
            verify_output_paths=False,
        )
        comparison_mode = "latest_vs_previous"
    else:
        if from_run_id is None or to_run_id is None:
            raise ValueError(
                "provide both --from and --to, or neither to compare latest vs previous"
            )
        if from_run_id == to_run_id:
            raise ValueError("run diff requires different --from and --to run ids")
        from_summary = _load_run_summary_from_manifest(
            resolved_root / RUNS_DIRECTORY / from_run_id / MANIFEST_FILENAME,
            archive_root=resolved_root,
            verify_output_paths=False,
        )
        to_summary = _load_run_summary_from_manifest(
            resolved_root / RUNS_DIRECTORY / to_run_id / MANIFEST_FILENAME,
            archive_root=resolved_root,
            verify_output_paths=False,
        )
        comparison_mode = "explicit"

    source_names = sorted(
        {
            *(
                source.source
                for source in from_summary.sources
            ),
            *(
                source.source
                for source in to_summary.sources
            ),
        }
    )
    from_sources = {source.source: source for source in from_summary.sources}
    to_sources = {source.source: source for source in to_summary.sources}

    return RunDiff(
        archive_root=resolved_root,
        comparison_mode=comparison_mode,
        from_run=from_summary,
        to_run=to_summary,
        selection_policy=_build_selection_policy_diff(from_summary, to_summary),
        sources=tuple(
            _build_source_diff(
                source_name,
                from_source=from_sources.get(source_name),
                to_source=to_sources.get(source_name),
            )
            for source_name in source_names
        ),
    )


def load_run_trend(
    archive_root: Path,
    *,
    sources: tuple[str, ...] = (),
) -> RunTrendReport:
    resolved_root = archive_root.expanduser().resolve(strict=False)
    requested_sources = tuple(dict.fromkeys(source for source in sources if source))
    run_summaries = tuple(
        reversed(
            tuple(
                _load_run_summary_from_manifest(
                    manifest_path,
                    archive_root=resolved_root,
                    verify_output_paths=False,
                )
                for manifest_path in _list_manifest_paths(resolved_root)
            )
        )
    )

    grouped_sources: dict[str, list[tuple[RunSummary, ReportedSourceSummary]]] = {}
    for run_summary in run_summaries:
        for source_summary in run_summary.sources:
            if requested_sources and source_summary.source not in requested_sources:
                continue
            grouped_sources.setdefault(source_summary.source, []).append(
                (run_summary, source_summary)
            )

    source_trends = tuple(
        _build_source_health_trend(source_name, grouped_sources[source_name])
        for source_name in sorted(grouped_sources)
    )
    return RunTrendReport(
        archive_root=resolved_root,
        source_filter=requested_sources,
        runs=run_summaries,
        sources=source_trends,
    )


def _list_manifest_paths(archive_root: Path) -> tuple[Path, ...]:
    runs_dir = archive_root / RUNS_DIRECTORY
    if not runs_dir.exists():
        raise RunReportingError(f"run manifests directory does not exist: {runs_dir}")

    manifest_paths = tuple(
        sorted(
            runs_dir.glob(f"*/{MANIFEST_FILENAME}"),
            key=lambda path: path.parent.name,
            reverse=True,
        )
    )
    if not manifest_paths:
        raise RunReportingError(f"no run manifests found under: {runs_dir}")
    return manifest_paths


def _load_run_summary_from_manifest(
    manifest_path: Path,
    *,
    archive_root: Path,
    verify_output_paths: bool,
) -> RunSummary:
    payload = _load_manifest_payload(manifest_path)

    run_id = _optional_string(payload, "run_id") or manifest_path.parent.name
    started_at = _optional_string(payload, "started_at")
    completed_at = _optional_string(payload, "completed_at")
    manifest_archive_root = _optional_path(payload, "archive_root") or archive_root
    run_dir = _optional_path(payload, "run_dir") or manifest_path.parent

    sources_payload = payload.get("sources")
    if not isinstance(sources_payload, list):
        raise RunReportingError(
            f"run manifest has invalid sources payload: {manifest_path}"
        )

    sources = tuple(
        _parse_source_summary(
            source_payload,
            manifest_path=manifest_path,
            run_id=run_id,
            verify_output_paths=verify_output_paths,
        )
        for source_payload in sources_payload
    )

    return RunSummary(
        run_id=run_id,
        started_at=started_at,
        completed_at=completed_at,
        archive_root=manifest_archive_root,
        run_dir=run_dir,
        manifest_path=manifest_path,
        selection_policy=_optional_object(payload, "selection_policy"),
        effective_config=_optional_object(payload, "effective_config"),
        rerun=_optional_object(payload, "rerun"),
        scheduled=_optional_object(payload, "scheduled"),
        selected_sources=_optional_string_sequence(payload, "selected_sources"),
        excluded_sources=_optional_object_sequence(payload, "excluded_sources"),
        sources=sources,
    )


def _load_manifest_payload(manifest_path: Path) -> dict[str, object]:
    if not manifest_path.exists():
        raise RunReportingError(f"run manifest does not exist: {manifest_path}")

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RunReportingError(
            f"run manifest is not valid JSON: {manifest_path}"
        ) from exc

    if not isinstance(payload, dict):
        raise RunReportingError(f"run manifest root must be an object: {manifest_path}")
    return payload


def _parse_source_summary(
    source_payload: object,
    *,
    manifest_path: Path,
    run_id: str,
    verify_output_paths: bool,
) -> ReportedSourceSummary:
    if not isinstance(source_payload, dict):
        raise RunReportingError(
            f"run manifest has a non-object source entry: {manifest_path}"
        )

    source = _required_string(source_payload, "source", manifest_path)
    output_path = _optional_path(source_payload, "output_path")
    failed = _required_bool(source_payload, "failed", manifest_path)
    partial = _required_bool(source_payload, "partial", manifest_path)
    unsupported = _required_bool(source_payload, "unsupported", manifest_path)
    scanned_artifact_count = _required_int(
        source_payload, "scanned_artifact_count", manifest_path
    )
    conversation_count = _required_int(source_payload, "conversation_count", manifest_path)

    if verify_output_paths and output_path is not None and not output_path.exists():
        raise RunReportingError(
            f"run '{run_id}' source '{source}' output is missing: {output_path}"
        )

    return ReportedSourceSummary(
        source=source,
        support_level=_required_string(source_payload, "support_level", manifest_path),
        status=_required_string(source_payload, "status", manifest_path),
        output_path=output_path,
        scanned_artifact_count=scanned_artifact_count,
        conversation_count=conversation_count,
        message_count=_required_int(source_payload, "message_count", manifest_path),
        skipped_conversation_count=_optional_int(
            source_payload,
            "skipped_conversation_count",
            default=0,
        ),
        written_conversation_count=_optional_int(
            source_payload,
            "written_conversation_count",
            default=conversation_count,
        ),
        upgraded_conversation_count=_optional_int(
            source_payload,
            "upgraded_conversation_count",
            default=0,
        ),
        partial=partial,
        unsupported=unsupported,
        failed=failed,
        failure_reason=_optional_string(source_payload, "failure_reason"),
        support_limitation_summary=_optional_string(
            source_payload,
            "support_limitation_summary",
        ),
        support_limitations=_optional_string_sequence(
            source_payload,
            "support_limitations",
        ),
        redaction_event_count=_optional_int(
            source_payload,
            "redaction_event_count",
            default=0,
        ),
    )


def _required_string(
    payload: dict[str, object], key: str, manifest_path: Path
) -> str:
    value = payload.get(key)
    if isinstance(value, str) and value:
        return value
    raise RunReportingError(f"run manifest is missing '{key}': {manifest_path}")


def _optional_string(payload: dict[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, str):
        return value
    raise RunReportingError(f"run manifest field '{key}' must be a string")


def _required_bool(payload: dict[str, object], key: str, manifest_path: Path) -> bool:
    value = payload.get(key)
    if isinstance(value, bool):
        return value
    raise RunReportingError(f"run manifest is missing '{key}': {manifest_path}")


def _required_int(payload: dict[str, object], key: str, manifest_path: Path) -> int:
    value = payload.get(key)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise RunReportingError(f"run manifest is missing '{key}': {manifest_path}")


def _optional_int(payload: dict[str, object], key: str, *, default: int) -> int:
    value = payload.get(key)
    if value is None:
        return default
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise RunReportingError(f"run manifest field '{key}' must be an integer")


def _optional_object(payload: dict[str, object], key: str) -> dict[str, object] | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    raise RunReportingError(f"run manifest field '{key}' must be an object")


def _optional_string_sequence(
    payload: dict[str, object],
    key: str,
) -> tuple[str, ...]:
    value = payload.get(key)
    if value is None:
        return ()
    if not isinstance(value, list):
        raise RunReportingError(f"run manifest field '{key}' must be an array")
    normalized: list[str] = []
    for entry in value:
        if not isinstance(entry, str) or not entry:
            raise RunReportingError(
                f"run manifest field '{key}' must contain strings"
            )
        normalized.append(entry)
    return tuple(normalized)


def _optional_object_sequence(
    payload: dict[str, object],
    key: str,
) -> tuple[dict[str, object], ...]:
    value = payload.get(key)
    if value is None:
        return ()
    if not isinstance(value, list):
        raise RunReportingError(f"run manifest field '{key}' must be an array")
    normalized: list[dict[str, object]] = []
    for entry in value:
        if not isinstance(entry, dict):
            raise RunReportingError(
                f"run manifest field '{key}' must contain objects"
            )
        normalized.append(entry)
    return tuple(normalized)


def _optional_path(payload: dict[str, object], key: str) -> Path | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise RunReportingError(f"run manifest field '{key}' must be a string path")
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve(strict=False)
    return candidate.resolve(strict=False)


def _build_selection_policy_diff(
    from_run: RunSummary,
    to_run: RunSummary,
) -> SelectionPolicyDiff:
    from_excluded = _map_excluded_sources(from_run.excluded_sources)
    to_excluded = _map_excluded_sources(to_run.excluded_sources)
    shared_excluded_sources = sorted(set(from_excluded) & set(to_excluded))
    changed_excluded_sources = tuple(
        ExcludedSourceDiff(
            source=source_name,
            from_entry=from_excluded[source_name],
            to_entry=to_excluded[source_name],
        )
        for source_name in shared_excluded_sources
        if from_excluded[source_name] != to_excluded[source_name]
    )

    return SelectionPolicyDiff(
        from_policy=from_run.selection_policy,
        to_policy=to_run.selection_policy,
        added_selected_sources=tuple(
            sorted(set(to_run.selected_sources) - set(from_run.selected_sources))
        ),
        removed_selected_sources=tuple(
            sorted(set(from_run.selected_sources) - set(to_run.selected_sources))
        ),
        added_excluded_sources=tuple(
            to_excluded[source_name]
            for source_name in sorted(set(to_excluded) - set(from_excluded))
        ),
        removed_excluded_sources=tuple(
            from_excluded[source_name]
            for source_name in sorted(set(from_excluded) - set(to_excluded))
        ),
        changed_excluded_sources=changed_excluded_sources,
    )


def _map_excluded_sources(
    entries: tuple[dict[str, object], ...],
) -> dict[str, dict[str, object]]:
    mapped: dict[str, dict[str, object]] = {}
    for entry in entries:
        source_name = entry.get("source")
        if isinstance(source_name, str) and source_name:
            mapped[source_name] = entry
    return mapped


def _build_source_diff(
    source_name: str,
    *,
    from_source: ReportedSourceSummary | None,
    to_source: ReportedSourceSummary | None,
) -> SourceDiff:
    return SourceDiff(
        source=source_name,
        from_source=from_source,
        to_source=to_source,
        scanned_artifact_count=_count_delta(
            _source_count(from_source, "scanned_artifact_count"),
            _source_count(to_source, "scanned_artifact_count"),
        ),
        conversation_count=_count_delta(
            _source_count(from_source, "conversation_count"),
            _source_count(to_source, "conversation_count"),
        ),
        message_count=_count_delta(
            _source_count(from_source, "message_count"),
            _source_count(to_source, "message_count"),
        ),
        skipped_conversation_count=_count_delta(
            _source_count(from_source, "skipped_conversation_count"),
            _source_count(to_source, "skipped_conversation_count"),
        ),
        written_conversation_count=_count_delta(
            _source_count(from_source, "written_conversation_count"),
            _source_count(to_source, "written_conversation_count"),
        ),
        redaction_event_count=_count_delta(
            _source_count(from_source, "redaction_event_count"),
            _source_count(to_source, "redaction_event_count"),
        ),
    )


def _source_count(
    source: ReportedSourceSummary | None,
    field_name: str,
) -> int:
    if source is None:
        return 0
    return int(getattr(source, field_name))


def _count_delta(from_value: int, to_value: int) -> CountDelta:
    return CountDelta(from_value=from_value, to_value=to_value)


def _value_change(
    from_value: str | None,
    to_value: str | None,
) -> dict[str, object]:
    return {
        "from": from_value,
        "to": to_value,
        "changed": from_value != to_value,
    }


def _classify_status_transition(from_status: str, to_status: str) -> str:
    from_order = _STATUS_ORDER.get(from_status)
    to_order = _STATUS_ORDER.get(to_status)
    if from_order is None or to_order is None:
        return "changed"
    if to_order > from_order:
        return "improved"
    if to_order < from_order:
        return "regressed"
    return "changed"


def _build_source_health_trend(
    source_name: str,
    source_runs: list[tuple[RunSummary, ReportedSourceSummary]],
) -> SourceHealthTrend:
    status_counts = {status: 0 for status in _STATUS_VALUES}
    support_levels: list[str] = []
    transitions: list[RunTrendTransition] = []
    timeline: list[SourceHealthTrendPoint] = []
    previous_run: RunSummary | None = None
    previous_source: ReportedSourceSummary | None = None

    for run_summary, source_summary in source_runs:
        if source_summary.support_level not in support_levels:
            support_levels.append(source_summary.support_level)

        transition = _build_run_trend_transition(
            source_name,
            previous_run=previous_run,
            previous_source=previous_source,
            current_run=run_summary,
            current_source=source_summary,
        )
        if transition is not None:
            transitions.append(transition)

        status_counts[source_summary.status] = status_counts.get(source_summary.status, 0) + 1
        archive_stats, archive_stats_error = _load_archive_stats_for_source(source_summary)
        timeline.append(
            SourceHealthTrendPoint(
                run_id=run_summary.run_id,
                started_at=run_summary.started_at,
                completed_at=run_summary.completed_at,
                manifest=source_summary,
                archive_stats=archive_stats,
                status_ratios=_build_status_ratios(
                    status_counts,
                    total=len(timeline) + 1,
                ),
                transition_from_previous=transition,
                archive_stats_error=archive_stats_error,
            )
        )
        previous_run = run_summary
        previous_source = source_summary

    latest_point = timeline[-1]
    return SourceHealthTrend(
        source=source_name,
        first_run_id=timeline[0].run_id,
        latest_run_id=latest_point.run_id,
        latest_status=latest_point.manifest.status,
        latest_support_level=latest_point.manifest.support_level,
        support_levels=tuple(support_levels),
        status_counts=status_counts,
        transitions=tuple(transitions),
        timeline=tuple(timeline),
    )


def _build_run_trend_transition(
    source_name: str,
    *,
    previous_run: RunSummary | None,
    previous_source: ReportedSourceSummary | None,
    current_run: RunSummary,
    current_source: ReportedSourceSummary,
) -> RunTrendTransition | None:
    if previous_run is None or previous_source is None:
        return None
    if previous_source.status == current_source.status:
        return None
    return RunTrendTransition(
        source=source_name,
        from_run_id=previous_run.run_id,
        to_run_id=current_run.run_id,
        from_status=previous_source.status,
        to_status=current_source.status,
        label=f"{previous_source.status}_to_{current_source.status}",
        category=_classify_status_transition(
            previous_source.status,
            current_source.status,
        ),
    )


def _load_archive_stats_for_source(
    source_summary: ReportedSourceSummary,
) -> tuple[ArchiveStatsSnapshot, str | None]:
    if source_summary.output_path is None:
        return _empty_archive_stats_snapshot(), None
    try:
        return (
            summarize_archive_output_paths(
                (source_summary.output_path,),
                source=source_summary.source,
            ),
            None,
        )
    except ArchiveInspectError as exc:
        return _empty_archive_stats_snapshot(), str(exc)


def _empty_archive_stats_snapshot() -> ArchiveStatsSnapshot:
    return ArchiveStatsSnapshot(
        file_count=0,
        conversation_count=0,
        message_count=0,
        transcript_completeness_counts=tuple(
            (status, 0) for status in ("complete", "partial", "unsupported")
        ),
        earliest_collected_at=None,
        latest_collected_at=None,
        conversation_with_limitations_count=0,
    )


def _build_status_ratios(
    status_counts: dict[str, int],
    *,
    total: int,
) -> dict[str, float]:
    normalized_counts = {status: status_counts.get(status, 0) for status in _STATUS_VALUES}
    normalized_counts["degraded"] = (
        normalized_counts.get("partial", 0) + normalized_counts.get("unsupported", 0)
    )
    return {
        status: _ratio(count, total)
        for status, count in normalized_counts.items()
    }


def _ratio(count: int, total: int) -> float:
    if total == 0:
        return 0.0
    return count / total


__all__ = [
    "CountDelta",
    "ExcludedSourceDiff",
    "ReportedSourceSummary",
    "RunDiff",
    "RunReportingError",
    "RunTrendReport",
    "RunSummary",
    "RunTrendTransition",
    "SelectionPolicyDiff",
    "SourceHealthTrend",
    "SourceHealthTrendPoint",
    "SourceDiff",
    "StatusTransition",
    "list_run_summaries",
    "load_run_diff",
    "load_latest_run_summary",
    "load_run_summary",
    "load_run_trend",
]
