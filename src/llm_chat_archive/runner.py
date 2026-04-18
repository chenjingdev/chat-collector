from __future__ import annotations

import json
from pathlib import Path

from .execution_policy import collection_execution_policy_context
from .models import (
    CollectionExecutionPolicy,
    CollectionRunResult,
    EffectiveCollectConfig,
    RerunMetadata,
    ScheduledRunMetadata,
    SourceSelectionPolicy,
    SourceRunResult,
    SourceRunStatus,
    SupportLevel,
    TranscriptCompleteness,
)
from .registry import Collector, CollectorRegistry, ExecutableCollector
from .source_selection import build_source_selection_policy, select_collectors
from .sources.codex_rollout import timestamp_slug, utc_timestamp

MANIFEST_FILENAME = "manifest.json"
RUNS_DIRECTORY = "runs"


def run_collection_batch(
    registry: CollectorRegistry,
    archive_root: Path,
    *,
    input_roots: tuple[Path, ...] | None = None,
    selection_policy: SourceSelectionPolicy | None = None,
    execution_policy: CollectionExecutionPolicy | None = None,
    effective_config: EffectiveCollectConfig | None = None,
    rerun: RerunMetadata | None = None,
    scheduled: ScheduledRunMetadata | None = None,
) -> CollectionRunResult:
    started_at = utc_timestamp()
    run_id = timestamp_slug(started_at)
    run_dir = archive_root / RUNS_DIRECTORY / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved_selection_policy = (
        effective_config.selection_policy
        if effective_config is not None
        else selection_policy or build_source_selection_policy()
    )
    resolved_execution_policy = (
        effective_config.execution_policy
        if effective_config is not None
        else execution_policy or CollectionExecutionPolicy()
    )
    resolved_effective_config = effective_config or EffectiveCollectConfig(
        archive_root=archive_root,
        selection_policy=resolved_selection_policy,
        execution_policy=resolved_execution_policy,
    )
    selected_collectors = select_collectors(
        registry,
        policy=resolved_selection_policy,
    )
    source_results = tuple(
        _run_single_collector(
            collector,
            archive_root=archive_root,
            input_roots=input_roots,
            execution_policy=resolved_execution_policy,
        )
        for collector in selected_collectors.collectors
    )
    manifest_path = run_dir / MANIFEST_FILENAME
    result = CollectionRunResult(
        run_id=run_id,
        archive_root=archive_root,
        run_dir=run_dir,
        manifest_path=manifest_path,
        started_at=started_at,
        completed_at=utc_timestamp(),
        selection_policy=selected_collectors.policy,
        effective_config=resolved_effective_config,
        selected_sources=selected_collectors.selected_sources,
        excluded_sources=selected_collectors.excluded_sources,
        sources=source_results,
        rerun=rerun,
        scheduled=scheduled,
    )
    manifest_path.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return result


def _run_single_collector(
    collector: Collector,
    *,
    archive_root: Path,
    input_roots: tuple[Path, ...] | None,
    execution_policy: CollectionExecutionPolicy,
) -> SourceRunResult:
    descriptor = collector.descriptor
    support_metadata = descriptor.support_metadata
    resolved_input_roots = tuple(input_roots or ())

    if not isinstance(collector, ExecutableCollector):
        return SourceRunResult(
            source=descriptor.key,
            support_level=descriptor.support_level,
            status=SourceRunStatus.UNSUPPORTED,
            archive_root=archive_root,
            output_path=None,
            input_roots=resolved_input_roots,
            scanned_artifact_count=0,
            conversation_count=0,
            message_count=0,
            skipped_conversation_count=0,
            written_conversation_count=0,
            upgraded_conversation_count=0,
            unsupported=True,
            support_limitation_summary=(
                None if support_metadata is None else support_metadata.limitation_summary
            ),
            support_limitations=(
                () if support_metadata is None else support_metadata.limitations
            ),
        )

    try:
        with collection_execution_policy_context(execution_policy):
            result = collector.collect(archive_root, input_roots=input_roots)
    except Exception as exc:
        return SourceRunResult(
            source=descriptor.key,
            support_level=descriptor.support_level,
            status=SourceRunStatus.FAILED,
            archive_root=archive_root,
            output_path=None,
            input_roots=resolved_input_roots,
            scanned_artifact_count=0,
            conversation_count=0,
            message_count=0,
            skipped_conversation_count=0,
            written_conversation_count=0,
            upgraded_conversation_count=0,
            failed=True,
            failure_reason=f"{type(exc).__name__}: {exc}",
            support_limitation_summary=(
                None if support_metadata is None else support_metadata.limitation_summary
            ),
            support_limitations=(
                () if support_metadata is None else support_metadata.limitations
            ),
        )

    partial, unsupported = summarize_output_status(
        result.output_path,
        support_level=descriptor.support_level,
    )
    return SourceRunResult(
        source=result.source,
        support_level=descriptor.support_level,
        status=_status_for_result(partial=partial, unsupported=unsupported),
        archive_root=result.archive_root,
        output_path=result.output_path,
        input_roots=result.input_roots,
        scanned_artifact_count=result.scanned_artifact_count,
        conversation_count=result.conversation_count,
        message_count=result.message_count,
        skipped_conversation_count=result.skipped_conversation_count,
        written_conversation_count=result.written_conversation_count,
        upgraded_conversation_count=result.upgraded_conversation_count,
        partial=partial,
        unsupported=unsupported,
        support_limitation_summary=(
            None if support_metadata is None else support_metadata.limitation_summary
        ),
        support_limitations=(
            () if support_metadata is None else support_metadata.limitations
        ),
        redaction_event_count=result.redaction_event_count,
    )


def summarize_output_status(
    output_path: Path,
    *,
    support_level: SupportLevel,
) -> tuple[bool, bool]:
    partial_found = False
    unsupported_found = False

    try:
        lines = output_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        lines = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue

        transcript_completeness = payload.get("transcript_completeness")
        if transcript_completeness == TranscriptCompleteness.UNSUPPORTED.value:
            unsupported_found = True
        elif transcript_completeness == TranscriptCompleteness.PARTIAL.value:
            partial_found = True

    if partial_found or unsupported_found:
        return partial_found, unsupported_found
    if support_level == SupportLevel.SCAFFOLD:
        return False, True
    if support_level == SupportLevel.PARTIAL:
        return True, False
    return False, False


def _status_for_result(*, partial: bool, unsupported: bool) -> SourceRunStatus:
    if unsupported:
        return SourceRunStatus.UNSUPPORTED
    if partial:
        return SourceRunStatus.PARTIAL
    return SourceRunStatus.COMPLETE


__all__ = [
    "MANIFEST_FILENAME",
    "RUNS_DIRECTORY",
    "run_collection_batch",
    "summarize_output_status",
]
