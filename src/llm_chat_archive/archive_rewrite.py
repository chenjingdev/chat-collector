from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .archive_merge import (
    CANONICAL_OUTPUT_TEMPLATE,
    archive_candidate_sort_key,
    build_archive_merge_candidate,
    compact_archive_candidates,
)
from .archive_inspect import (
    ARCHIVE_OUTPUT_GLOB,
    ArchiveInspectError,
    iter_archive_records,
)
from .runner import RUNS_DIRECTORY


@dataclass(frozen=True, slots=True)
class ArchiveRewriteSourceReport:
    source: str
    changed: bool
    input_file_count: int
    output_file_count: int
    before_conversation_count: int
    after_conversation_count: int
    dropped_row_count: int
    upgraded_row_count: int
    untouched_row_count: int
    output_path: Path | None

    def to_dict(self) -> dict[str, object]:
        return {
            "changed": self.changed,
            "input_file_count": self.input_file_count,
            "output_file_count": self.output_file_count,
            "before_conversation_count": self.before_conversation_count,
            "after_conversation_count": self.after_conversation_count,
            "dropped_row_count": self.dropped_row_count,
            "upgraded_row_count": self.upgraded_row_count,
            "untouched_row_count": self.untouched_row_count,
            "output_path": str(self.output_path) if self.output_path is not None else None,
        }


@dataclass(frozen=True, slots=True)
class ArchiveRewriteReport:
    archive_root: Path
    output_root: Path
    write_mode: str
    source_filter: str | None
    input_file_count: int
    output_file_count: int
    before_conversation_count: int
    after_conversation_count: int
    dropped_row_count: int
    upgraded_row_count: int
    untouched_row_count: int
    sources: tuple[ArchiveRewriteSourceReport, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "archive_root": str(self.archive_root),
            "output_root": str(self.output_root),
            "write_mode": self.write_mode,
            "source_filter": self.source_filter,
            "source_count": len(self.sources),
            "changed_source_count": sum(1 for source in self.sources if source.changed),
            "input_file_count": self.input_file_count,
            "output_file_count": self.output_file_count,
            "before_conversation_count": self.before_conversation_count,
            "after_conversation_count": self.after_conversation_count,
            "dropped_row_count": self.dropped_row_count,
            "upgraded_row_count": self.upgraded_row_count,
            "untouched_row_count": self.untouched_row_count,
            "sources": {
                source.source: source.to_dict() for source in self.sources
            },
        }


@dataclass(frozen=True, slots=True)
class _SourceRewritePlan:
    report: ArchiveRewriteSourceReport
    input_paths: tuple[Path, ...]
    serialized_rows: tuple[str, ...]


def rewrite_archive(
    archive_root: Path,
    *,
    output_root: Path | None = None,
    source: str | None = None,
    execute: bool = False,
) -> ArchiveRewriteReport:
    resolved_archive_root = archive_root.expanduser().resolve(strict=False)
    _validate_archive_directory(resolved_archive_root)
    resolved_output_root = (
        resolved_archive_root
        if output_root is None
        else output_root.expanduser().resolve(strict=False)
    )
    write_mode = _resolve_write_mode(
        execute=execute,
        archive_root=resolved_archive_root,
        output_root=resolved_output_root,
    )

    selected_sources = _select_sources(resolved_archive_root, source=source)
    source_plans: list[_SourceRewritePlan] = []
    for source_name in selected_sources:
        plan = _plan_source_rewrite(
            resolved_archive_root,
            output_root=resolved_output_root,
            source=source_name,
        )
        source_plans.append(plan)
        if execute:
            _write_source_plan(
                plan,
                archive_root=resolved_archive_root,
                output_root=resolved_output_root,
            )

    report = ArchiveRewriteReport(
        archive_root=resolved_archive_root,
        output_root=resolved_output_root,
        write_mode=write_mode,
        source_filter=source,
        input_file_count=sum(plan.report.input_file_count for plan in source_plans),
        output_file_count=sum(plan.report.output_file_count for plan in source_plans),
        before_conversation_count=sum(
            plan.report.before_conversation_count for plan in source_plans
        ),
        after_conversation_count=sum(
            plan.report.after_conversation_count for plan in source_plans
        ),
        dropped_row_count=sum(plan.report.dropped_row_count for plan in source_plans),
        upgraded_row_count=sum(plan.report.upgraded_row_count for plan in source_plans),
        untouched_row_count=sum(plan.report.untouched_row_count for plan in source_plans),
        sources=tuple(plan.report for plan in source_plans),
    )
    return report


def _validate_archive_directory(path: Path) -> None:
    if not path.exists():
        raise ArchiveInspectError(f"archive root does not exist: {path}")
    if not path.is_dir():
        raise ArchiveInspectError(f"archive root is not a directory: {path}")


def _resolve_write_mode(
    *,
    execute: bool,
    archive_root: Path,
    output_root: Path,
) -> str:
    if not execute:
        return "dry_run"
    if output_root == archive_root:
        return "in_place"
    return "staging"


def _select_sources(archive_root: Path, *, source: str | None) -> tuple[str, ...]:
    if source is not None:
        source_dir = archive_root / source
        if not source_dir.is_dir():
            return ()
        return (source,)
    return tuple(
        path.name
        for path in sorted(archive_root.iterdir())
        if path.is_dir() and path.name != RUNS_DIRECTORY
    )


def _plan_source_rewrite(
    archive_root: Path,
    *,
    output_root: Path,
    source: str,
) -> _SourceRewritePlan:
    records = tuple(iter_archive_records(archive_root, source=source))
    input_paths = tuple(sorted({record.summary.output_path for record in records}))
    candidates = tuple(build_archive_merge_candidate(record) for record in records)
    selected_candidates, dropped_row_count, upgraded_row_count, untouched_row_count = (
        compact_archive_candidates(candidates)
    )
    sorted_rows = tuple(
        candidate.serialized_payload
        for candidate in sorted(selected_candidates, key=archive_candidate_sort_key)
    )
    output_path = (
        output_root / source / CANONICAL_OUTPUT_TEMPLATE.format(source=source)
        if sorted_rows
        else None
    )
    expected_paths = () if output_path is None else (output_path,)
    report = ArchiveRewriteSourceReport(
        source=source,
        changed=(
            dropped_row_count > 0
            or upgraded_row_count > 0
            or input_paths != expected_paths
        ),
        input_file_count=len(input_paths),
        output_file_count=1 if output_path is not None else 0,
        before_conversation_count=len(candidates),
        after_conversation_count=len(sorted_rows),
        dropped_row_count=dropped_row_count,
        upgraded_row_count=upgraded_row_count,
        untouched_row_count=untouched_row_count,
        output_path=output_path,
    )
    return _SourceRewritePlan(
        report=report,
        input_paths=input_paths,
        serialized_rows=sorted_rows,
    )


def _write_source_plan(
    plan: _SourceRewritePlan,
    *,
    archive_root: Path,
    output_root: Path,
) -> None:
    source = plan.report.source
    output_path = plan.report.output_path
    if output_path is None:
        return

    in_place = output_root == archive_root
    if in_place and not plan.report.changed:
        return

    source_dir = output_root / source
    source_dir.mkdir(parents=True, exist_ok=True)
    existing_paths = tuple(sorted(source_dir.glob(ARCHIVE_OUTPUT_GLOB)))
    temporary_path = source_dir / f".{output_path.name}.tmp"
    try:
        temporary_path.write_text(
            "".join(f"{row}\n" for row in plan.serialized_rows),
            encoding="utf-8",
        )
        for existing_path in existing_paths:
            existing_path.unlink()
        temporary_path.replace(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


__all__ = [
    "ArchiveRewriteReport",
    "ArchiveRewriteSourceReport",
    "CANONICAL_OUTPUT_TEMPLATE",
    "rewrite_archive",
]
