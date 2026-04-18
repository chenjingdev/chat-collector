from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .models import (
    EffectiveRerunConfig,
    RerunMetadata,
    RerunSelectionPreset,
    RerunSelectionReason,
    SourceSelectionPolicy,
    SourceSelectionProfile,
)
from .reporting import RunSummary
from .source_selection import build_source_selection_policy


class RerunSelectionError(ValueError):
    """Raised when a rerun target set cannot be derived from a prior run."""


@dataclass(frozen=True, slots=True)
class PlannedRerun:
    selection_policy: SourceSelectionPolicy
    metadata: RerunMetadata


def resolve_rerun_config(
    *,
    cli_reason: str | RerunSelectionReason | None,
    configured_rerun: EffectiveRerunConfig | None,
) -> EffectiveRerunConfig:
    if cli_reason is not None:
        resolved_reason = RerunSelectionReason(cli_reason)
        return EffectiveRerunConfig(
            selection_preset=RerunSelectionPreset.from_selection_reason(resolved_reason),
            selection_reason=resolved_reason,
            source="cli",
        )
    if configured_rerun is not None:
        return configured_rerun
    raise ValueError(
        "rerun requires --reason or collector config [rerun] selection_preset"
    )


def plan_rerun(
    summary: RunSummary,
    *,
    selection_reason: str | RerunSelectionReason,
    include_sources: Iterable[str] = (),
    exclude_sources: Iterable[str] = (),
) -> PlannedRerun:
    resolved_reason = RerunSelectionReason(selection_reason)
    matched_sources = tuple(
        source.source
        for source in summary.sources
        if _matches_selection_reason(source, selection_reason=resolved_reason)
    )
    manual_include_sources = _normalize_sources(include_sources)
    manual_exclude_sources = _normalize_sources(exclude_sources)
    include_allowlist = _merge_sources(matched_sources, manual_include_sources)

    if not include_allowlist:
        raise RerunSelectionError(
            f"rerun selection '{resolved_reason.value}' matched no sources in run "
            f"'{summary.run_id}'"
        )

    return PlannedRerun(
        selection_policy=build_source_selection_policy(
            profile=SourceSelectionProfile.ALL,
            include_sources=include_allowlist,
            exclude_sources=manual_exclude_sources,
        ),
        metadata=RerunMetadata(
            origin_run_id=summary.run_id,
            selection_reason=resolved_reason,
            matched_sources=matched_sources,
            manual_include_sources=manual_include_sources,
            manual_exclude_sources=manual_exclude_sources,
        ),
    )


def _matches_selection_reason(
    source: object,
    *,
    selection_reason: RerunSelectionReason,
) -> bool:
    failed = bool(getattr(source, "failed"))
    degraded = bool(getattr(source, "partial")) or bool(getattr(source, "unsupported"))

    if selection_reason == RerunSelectionReason.FAILED:
        return failed
    if selection_reason == RerunSelectionReason.DEGRADED:
        return degraded
    return failed or degraded


def _normalize_sources(source_names: Iterable[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for source_name in source_names:
        if source_name in seen:
            continue
        seen.add(source_name)
        normalized.append(source_name)
    return tuple(normalized)


def _merge_sources(*source_groups: Iterable[str]) -> tuple[str, ...]:
    merged: list[str] = []
    seen: set[str] = set()
    for source_group in source_groups:
        for source_name in source_group:
            if source_name in seen:
                continue
            seen.add(source_name)
            merged.append(source_name)
    return tuple(merged)


__all__ = [
    "PlannedRerun",
    "RerunSelectionError",
    "plan_rerun",
    "resolve_rerun_config",
]
