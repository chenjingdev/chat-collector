from __future__ import annotations

from dataclasses import dataclass

from .models import SourceDescriptor, SourceSelectionProfile
from .registry import CollectorRegistry
from .source_selection import build_source_selection_policy, select_collectors


@dataclass(frozen=True, slots=True)
class SourceSupportMatrixEntry:
    source: str
    display_name: str
    product_label: str
    host_surface: str
    support_level: str
    expected_transcript_completeness: str
    limitation_summary: str | None
    limitations: tuple[str, ...]
    included_in_default_batch_profile: bool
    default_input_roots: tuple[str, ...]
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "source": self.source,
            "display_name": self.display_name,
            "product_label": self.product_label,
            "host_surface": self.host_surface,
            "support_level": self.support_level,
            "expected_transcript_completeness": self.expected_transcript_completeness,
            "limitation_summary": self.limitation_summary,
            "included_in_default_batch_profile": (
                self.included_in_default_batch_profile
            ),
            "default_input_roots": list(self.default_input_roots),
            "notes": list(self.notes),
        }
        if self.limitations:
            payload["limitations"] = list(self.limitations)
        return payload


def build_source_support_matrix(
    registry: CollectorRegistry,
) -> tuple[SourceSupportMatrixEntry, ...]:
    default_selection = select_collectors(
        registry,
        policy=build_source_selection_policy(profile=SourceSelectionProfile.DEFAULT),
    )
    included_in_default_batch = set(default_selection.selected_sources)
    return tuple(
        _build_entry(
            collector.descriptor,
            included_in_default_batch_profile=(
                collector.descriptor.key in included_in_default_batch
            ),
        )
        for collector in registry.list()
    )


def render_source_support_matrix_markdown(
    entries: tuple[SourceSupportMatrixEntry, ...],
) -> str:
    lines = [
        "# Source Support Matrix",
        "",
        "This file is generated from registry metadata via "
        "`uv run llm-chat-archive sources --format markdown`.",
        "",
        "`Included in default batch` reflects the `default` selection profile.",
        "",
        "| Source key | Product | Host surface | Support level | Expected transcript completeness | Major limitation | Included in default batch |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for entry in entries:
        limitation_summary = entry.limitation_summary or "-"
        included_in_default_batch = (
            "yes" if entry.included_in_default_batch_profile else "no"
        )
        lines.append(
            "| "
            f"`{entry.source}` | "
            f"{entry.product_label} | "
            f"{entry.host_surface} | "
            f"`{entry.support_level}` | "
            f"`{entry.expected_transcript_completeness}` | "
            f"{limitation_summary} | "
            f"{included_in_default_batch} |"
        )
    lines.append("")
    return "\n".join(lines)


def _build_entry(
    descriptor: SourceDescriptor,
    *,
    included_in_default_batch_profile: bool,
) -> SourceSupportMatrixEntry:
    support_metadata = descriptor.support_metadata
    if support_metadata is None:
        raise ValueError(
            f"source descriptor is missing support metadata: {descriptor.key}"
        )

    return SourceSupportMatrixEntry(
        source=descriptor.key,
        display_name=descriptor.display_name,
        product_label=support_metadata.product_label,
        host_surface=support_metadata.host_surface,
        support_level=descriptor.support_level.value,
        expected_transcript_completeness=(
            support_metadata.expected_transcript_completeness.value
        ),
        limitation_summary=support_metadata.limitation_summary,
        limitations=support_metadata.limitations,
        included_in_default_batch_profile=included_in_default_batch_profile,
        default_input_roots=descriptor.default_input_roots,
        notes=descriptor.notes,
    )


__all__ = [
    "SourceSupportMatrixEntry",
    "build_source_support_matrix",
    "render_source_support_matrix_markdown",
]
