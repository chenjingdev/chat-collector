from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..incremental import write_incremental_collection
from ..models import (
    CollectionPlan,
    CollectionResult,
    SourceDescriptor,
    SourceSupportMetadata,
    SupportLevel,
    TranscriptCompleteness,
)
from ..source_roots import all_platform_root, default_descriptor_input_roots
from .codex_rollout import (
    iter_rollout_paths,
    parse_codex_rollout_file,
    resolve_input_roots,
    utc_timestamp,
)

CODEX_CLI_DESCRIPTOR = SourceDescriptor(
    key="codex_cli",
    display_name="Codex CLI",
    execution_context="cli",
    support_level=SupportLevel.COMPLETE,
    default_input_roots=("~/.codex",),
    artifact_root_candidates=(all_platform_root("$HOME/.codex"),),
    notes=(
        "Scans ~/.codex/sessions/**/rollout-*.jsonl and ~/.codex/archived_sessions/rollout-*.jsonl.",
        "Keeps response_item message rows for developer, user, and assistant roles only.",
        "Excludes event, reasoning, tool, search, and turn-context noise from normalized output.",
    ),
    support_metadata=SourceSupportMetadata(
        product_label="Codex",
        host_surface="CLI",
        expected_transcript_completeness=TranscriptCompleteness.COMPLETE,
        limitation_summary="Filters event, reasoning, tool, and search noise out of the transcript.",
    ),
)


@dataclass(frozen=True, slots=True)
class CodexCliCollector:
    descriptor: SourceDescriptor = CODEX_CLI_DESCRIPTOR

    def build_plan(self, archive_root: Path) -> CollectionPlan:
        return CollectionPlan(
            source=self.descriptor.key,
            display_name=self.descriptor.display_name,
            archive_root=archive_root,
            execution_context=self.descriptor.execution_context,
            support_level=self.descriptor.support_level,
            default_input_roots=self.descriptor.default_input_roots,
            implemented=True,
            notes=self.descriptor.notes,
        )

    def collect(
        self, archive_root: Path, input_roots: tuple[Path, ...] | None = None
    ) -> CollectionResult:
        resolved_input_roots = resolve_input_roots(input_roots or self._default_input_roots())
        rollout_paths = tuple(iter_rollout_paths(resolved_input_roots))
        collected_at = utc_timestamp()
        conversations = (
            parse_rollout_file(rollout_path, collected_at=collected_at)
            for rollout_path in rollout_paths
        )
        return write_incremental_collection(
            source=self.descriptor.key,
            archive_root=archive_root,
            input_roots=resolved_input_roots,
            scanned_artifact_count=len(rollout_paths),
            collected_at=collected_at,
            conversations=conversations,
        )

    def _default_input_roots(self) -> tuple[Path, ...]:
        return default_descriptor_input_roots(self.descriptor)


def parse_rollout_file(
    rollout_path: Path, *, collected_at: str | None = None
):
    return parse_codex_rollout_file(
        rollout_path,
        descriptor=CODEX_CLI_DESCRIPTOR,
        collected_at=collected_at,
    )


__all__ = [
    "CODEX_CLI_DESCRIPTOR",
    "CodexCliCollector",
    "parse_rollout_file",
]
