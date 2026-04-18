from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .reporting import RunReportingError, RunSummary, load_run_summary
from .runner import MANIFEST_FILENAME, RUNS_DIRECTORY

DEFAULT_AUXILIARY_DIRECTORIES = (
    "archive-index",
    "rewrite-staging",
    "rewrite-backup",
    "rewrite-backups",
    "exports",
)


class ArchivePruneError(ValueError):
    """Raised when archive prune criteria or targets are invalid."""


@dataclass(frozen=True, slots=True)
class ArchivePruneRunTarget:
    run_id: str
    run_dir: Path
    manifest_path: Path
    started_at: str | None
    completed_at: str | None
    prune_reasons: tuple[str, ...]
    file_count: int
    reclaimed_bytes: int

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "manifest_path": str(self.manifest_path),
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "prune_reasons": list(self.prune_reasons),
            "file_count": self.file_count,
            "reclaimed_bytes": self.reclaimed_bytes,
        }


@dataclass(frozen=True, slots=True)
class ArchivePruneAuxiliaryTarget:
    directory: str
    path: Path
    file_count: int
    reclaimed_bytes: int

    def to_dict(self) -> dict[str, object]:
        return {
            "directory": self.directory,
            "path": str(self.path),
            "file_count": self.file_count,
            "reclaimed_bytes": self.reclaimed_bytes,
        }


@dataclass(frozen=True, slots=True)
class ArchivePruneReport:
    archive_root: Path
    write_mode: str
    keep_last_runs: int | None
    older_than_days: int | None
    auxiliary_directories: tuple[str, ...]
    recorded_run_count: int
    kept_run_count: int
    deleted_run_count: int
    deleted_auxiliary_directory_count: int
    deleted_file_count: int
    reclaimed_bytes: int
    latest_kept_run_id: str | None
    deleted_runs: tuple[ArchivePruneRunTarget, ...]
    deleted_auxiliary_directories: tuple[ArchivePruneAuxiliaryTarget, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "archive_root": str(self.archive_root),
            "write_mode": self.write_mode,
            "keep_last_runs": self.keep_last_runs,
            "older_than_days": self.older_than_days,
            "auxiliary_directories": list(self.auxiliary_directories),
            "recorded_run_count": self.recorded_run_count,
            "kept_run_count": self.kept_run_count,
            "deleted_run_count": self.deleted_run_count,
            "deleted_auxiliary_directory_count": self.deleted_auxiliary_directory_count,
            "deleted_file_count": self.deleted_file_count,
            "reclaimed_bytes": self.reclaimed_bytes,
            "latest_kept_run_id": self.latest_kept_run_id,
            "deleted_runs": [run.to_dict() for run in self.deleted_runs],
            "deleted_auxiliary_directories": [
                directory.to_dict() for directory in self.deleted_auxiliary_directories
            ],
        }


@dataclass(frozen=True, slots=True)
class _RecordedRun:
    summary: RunSummary
    timestamp: datetime | None
    file_count: int
    reclaimed_bytes: int


def prune_archive(
    archive_root: Path,
    *,
    keep_last_runs: int | None = None,
    older_than_days: int | None = None,
    prune_auxiliary: bool = False,
    auxiliary_directories: tuple[str, ...] = (),
    execute: bool = False,
) -> ArchivePruneReport:
    if keep_last_runs is not None and keep_last_runs < 0:
        raise ArchivePruneError("keep_last_runs must be greater than or equal to zero")
    if older_than_days is not None and older_than_days < 0:
        raise ArchivePruneError("older_than_days must be greater than or equal to zero")

    selected_auxiliary_directories = _resolve_auxiliary_directories(
        prune_auxiliary=prune_auxiliary,
        auxiliary_directories=auxiliary_directories,
    )
    if (
        keep_last_runs is None
        and older_than_days is None
        and not selected_auxiliary_directories
    ):
        raise ArchivePruneError("archive prune requires at least one prune criterion")

    resolved_archive_root = archive_root.expanduser().resolve(strict=False)
    recorded_runs = _load_recorded_runs(resolved_archive_root)
    run_targets = _select_run_targets(
        recorded_runs,
        keep_last_runs=keep_last_runs,
        older_than_days=older_than_days,
        now=_utc_now(),
    )
    auxiliary_targets = _select_auxiliary_targets(
        resolved_archive_root,
        selected_auxiliary_directories,
    )

    if execute:
        for target in run_targets:
            if target.run_dir.exists():
                shutil.rmtree(target.run_dir)
        for target in auxiliary_targets:
            if target.path.exists():
                shutil.rmtree(target.path)

    deleted_run_ids = {target.run_id for target in run_targets}
    kept_runs = [
        recorded_run
        for recorded_run in recorded_runs
        if recorded_run.summary.run_id not in deleted_run_ids
    ]
    report = ArchivePruneReport(
        archive_root=resolved_archive_root,
        write_mode="prune" if execute else "dry_run",
        keep_last_runs=keep_last_runs,
        older_than_days=older_than_days,
        auxiliary_directories=selected_auxiliary_directories,
        recorded_run_count=len(recorded_runs),
        kept_run_count=len(kept_runs),
        deleted_run_count=len(run_targets),
        deleted_auxiliary_directory_count=len(auxiliary_targets),
        deleted_file_count=sum(target.file_count for target in run_targets)
        + sum(target.file_count for target in auxiliary_targets),
        reclaimed_bytes=sum(target.reclaimed_bytes for target in run_targets)
        + sum(target.reclaimed_bytes for target in auxiliary_targets),
        latest_kept_run_id=kept_runs[0].summary.run_id if kept_runs else None,
        deleted_runs=run_targets,
        deleted_auxiliary_directories=auxiliary_targets,
    )
    return report


def _load_recorded_runs(archive_root: Path) -> tuple[_RecordedRun, ...]:
    runs_dir = archive_root / RUNS_DIRECTORY
    if not runs_dir.exists():
        return ()

    manifest_paths = tuple(
        sorted(
            runs_dir.glob(f"*/{MANIFEST_FILENAME}"),
            key=lambda path: path.parent.name,
            reverse=True,
        )
    )
    recorded_runs: list[_RecordedRun] = []
    for manifest_path in manifest_paths:
        run_id = manifest_path.parent.name
        try:
            summary = load_run_summary(
                archive_root,
                run_id,
                verify_output_paths=False,
            )
        except RunReportingError as exc:
            raise ArchivePruneError(str(exc)) from exc
        file_count, reclaimed_bytes = _summarize_tree(summary.run_dir)
        recorded_runs.append(
            _RecordedRun(
                summary=summary,
                timestamp=_resolve_run_timestamp(summary),
                file_count=file_count,
                reclaimed_bytes=reclaimed_bytes,
            )
        )
    return tuple(recorded_runs)


def _resolve_auxiliary_directories(
    *,
    prune_auxiliary: bool,
    auxiliary_directories: tuple[str, ...],
) -> tuple[str, ...]:
    selected: list[str] = []
    seen: set[str] = set()

    for directory in (
        DEFAULT_AUXILIARY_DIRECTORIES if prune_auxiliary else ()
    ) + auxiliary_directories:
        normalized = _normalize_auxiliary_directory(directory)
        if normalized in seen:
            continue
        seen.add(normalized)
        selected.append(normalized)
    return tuple(selected)


def _normalize_auxiliary_directory(directory: str) -> str:
    normalized = directory.strip()
    if not normalized:
        raise ArchivePruneError("auxiliary directory names must not be empty")
    path = Path(normalized)
    if path.name != normalized or normalized in {".", ".."}:
        raise ArchivePruneError(
            f"auxiliary directory must be a direct archive-root child: {directory}"
        )
    if normalized == RUNS_DIRECTORY:
        raise ArchivePruneError("auxiliary directory must not target runs/")
    return normalized


def _select_run_targets(
    recorded_runs: tuple[_RecordedRun, ...],
    *,
    keep_last_runs: int | None,
    older_than_days: int | None,
    now: datetime,
) -> tuple[ArchivePruneRunTarget, ...]:
    targets: list[ArchivePruneRunTarget] = []
    for index, recorded_run in enumerate(recorded_runs):
        exceeds_keep_last = (
            keep_last_runs is not None and index >= keep_last_runs
        )
        older_than_cutoff = (
            older_than_days is not None
            and recorded_run.timestamp is not None
            and recorded_run.timestamp < now - timedelta(days=older_than_days)
        )

        if keep_last_runs is not None and older_than_days is not None:
            should_delete = exceeds_keep_last and older_than_cutoff
        elif keep_last_runs is not None:
            should_delete = exceeds_keep_last
        else:
            should_delete = older_than_cutoff

        if not should_delete:
            continue

        prune_reasons = tuple(
            reason
            for reason, enabled in (
                ("exceeds_keep_last_runs", exceeds_keep_last),
                ("older_than_days", older_than_cutoff),
            )
            if enabled
        )
        targets.append(
            ArchivePruneRunTarget(
                run_id=recorded_run.summary.run_id,
                run_dir=recorded_run.summary.run_dir,
                manifest_path=recorded_run.summary.manifest_path,
                started_at=recorded_run.summary.started_at,
                completed_at=recorded_run.summary.completed_at,
                prune_reasons=prune_reasons,
                file_count=recorded_run.file_count,
                reclaimed_bytes=recorded_run.reclaimed_bytes,
            )
        )
    return tuple(targets)


def _select_auxiliary_targets(
    archive_root: Path,
    auxiliary_directories: tuple[str, ...],
) -> tuple[ArchivePruneAuxiliaryTarget, ...]:
    targets: list[ArchivePruneAuxiliaryTarget] = []
    for directory in auxiliary_directories:
        path = archive_root / directory
        if not path.is_dir():
            continue
        file_count, reclaimed_bytes = _summarize_tree(path)
        targets.append(
            ArchivePruneAuxiliaryTarget(
                directory=directory,
                path=path,
                file_count=file_count,
                reclaimed_bytes=reclaimed_bytes,
            )
        )
    return tuple(targets)


def _summarize_tree(path: Path) -> tuple[int, int]:
    file_count = 0
    reclaimed_bytes = 0
    for child in path.rglob("*"):
        if not child.is_file():
            continue
        file_count += 1
        reclaimed_bytes += child.stat().st_size
    if path.is_file():
        return 1, path.stat().st_size
    return file_count, reclaimed_bytes


def _resolve_run_timestamp(summary: RunSummary) -> datetime | None:
    for value in (summary.completed_at, summary.started_at, summary.run_id):
        if value is None:
            continue
        parsed = _parse_timestamp(value)
        if parsed is not None:
            return parsed
    return None


def _parse_timestamp(value: str) -> datetime | None:
    normalized = value.strip()
    if not normalized:
        return None

    iso_value = normalized[:-1] + "+00:00" if normalized.endswith("Z") else normalized
    try:
        parsed = datetime.fromisoformat(iso_value)
    except ValueError:
        parsed = None

    if parsed is None:
        try:
            parsed = datetime.strptime(normalized, "%Y%m%dT%H%M%SZ").replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            return None

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


__all__ = [
    "ArchivePruneAuxiliaryTarget",
    "ArchivePruneError",
    "ArchivePruneReport",
    "ArchivePruneRunTarget",
    "DEFAULT_AUXILIARY_DIRECTORIES",
    "prune_archive",
]
