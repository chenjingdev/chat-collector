from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .models import (
    ExcludedSource,
    SourceDescriptor,
    SourceSelectionPolicy,
    SourceSelectionProfile,
    SupportLevel,
)
from .registry import Collector, CollectorRegistry

_PROFILE_MINIMUM_SUPPORT_LEVEL = {
    SourceSelectionProfile.ALL: SupportLevel.SCAFFOLD,
    SourceSelectionProfile.DEFAULT: SupportLevel.COMPLETE,
    SourceSelectionProfile.COMPLETE_ONLY: SupportLevel.COMPLETE,
}

_SUPPORT_LEVEL_ORDER = {
    SupportLevel.SCAFFOLD: 0,
    SupportLevel.PARTIAL: 1,
    SupportLevel.COMPLETE: 2,
}


@dataclass(frozen=True, slots=True)
class SelectedCollectors:
    policy: SourceSelectionPolicy
    collectors: tuple[Collector, ...]
    excluded_sources: tuple[ExcludedSource, ...]

    @property
    def selected_sources(self) -> tuple[str, ...]:
        return tuple(collector.descriptor.key for collector in self.collectors)


def build_source_selection_policy(
    *,
    profile: str | SourceSelectionProfile | None = None,
    include_sources: Iterable[str] = (),
    exclude_sources: Iterable[str] = (),
) -> SourceSelectionPolicy:
    resolved_profile = _coerce_profile(profile)
    return SourceSelectionPolicy(
        profile=resolved_profile,
        minimum_support_level=_PROFILE_MINIMUM_SUPPORT_LEVEL[resolved_profile],
        include_sources=_normalize_source_names(include_sources),
        exclude_sources=_normalize_source_names(exclude_sources),
    )


def select_collectors(
    registry: CollectorRegistry,
    *,
    policy: SourceSelectionPolicy,
) -> SelectedCollectors:
    selected_collectors: list[Collector] = []
    excluded_sources: list[ExcludedSource] = []
    allowlist = set(policy.include_sources) if policy.include_sources else None
    denylist = set(policy.exclude_sources)

    for collector in registry.list():
        descriptor = collector.descriptor
        reason = _determine_exclusion_reason(
            descriptor,
            policy=policy,
            allowlist=allowlist,
            denylist=denylist,
        )
        if reason is None:
            selected_collectors.append(collector)
            continue
        excluded_sources.append(
            ExcludedSource(
                source=descriptor.key,
                support_level=descriptor.support_level,
                reason=reason,
            )
        )

    return SelectedCollectors(
        policy=policy,
        collectors=tuple(selected_collectors),
        excluded_sources=tuple(excluded_sources),
    )


def _coerce_profile(
    profile: str | SourceSelectionProfile | None,
) -> SourceSelectionProfile:
    if profile is None:
        return SourceSelectionProfile.ALL
    if isinstance(profile, SourceSelectionProfile):
        return profile
    return SourceSelectionProfile(profile)


def _normalize_source_names(source_names: Iterable[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for source_name in source_names:
        if source_name in seen:
            continue
        seen.add(source_name)
        normalized.append(source_name)
    return tuple(normalized)


def _determine_exclusion_reason(
    descriptor: SourceDescriptor,
    *,
    policy: SourceSelectionPolicy,
    allowlist: set[str] | None,
    denylist: set[str],
) -> str | None:
    if descriptor.key in denylist:
        return "explicitly excluded by --exclude-source"
    if allowlist is not None and descriptor.key not in allowlist:
        return "not included by --source allowlist"
    if not _meets_minimum_support_level(
        descriptor.support_level,
        minimum_support_level=policy.minimum_support_level,
    ):
        return (
            f"support level '{descriptor.support_level.value}' is below minimum "
            f"'{policy.minimum_support_level.value}'"
        )
    return None


def _meets_minimum_support_level(
    support_level: SupportLevel,
    *,
    minimum_support_level: SupportLevel,
) -> bool:
    return _SUPPORT_LEVEL_ORDER[support_level] >= _SUPPORT_LEVEL_ORDER[
        minimum_support_level
    ]


__all__ = [
    "SelectedCollectors",
    "build_source_selection_policy",
    "select_collectors",
]
