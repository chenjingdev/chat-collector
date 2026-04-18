from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from .archive_inspect import ARCHIVE_OUTPUT_GLOB, ArchiveInspectError
from .archive_verify import _verify_row, verify_archive
from .models import SCHEMA_VERSION
from .runner import MANIFEST_FILENAME, RUNS_DIRECTORY
from .validate import ValidationFinding, ValidationLevel


class ArchiveMigrateError(ValueError):
    """Raised when archive migration cannot plan or execute safely."""


@dataclass(frozen=True, slots=True)
class ArchiveMigrateFileReport:
    source: str
    input_path: Path
    output_path: Path
    action: str
    row_count: int
    migrated_row_count: int
    unchanged_row_count: int
    backup_path: Path | None
    schema_versions_before: dict[str, int]
    required_fields_preserved: bool
    provenance_core_preserved: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "input_path": str(self.input_path),
            "output_path": str(self.output_path),
            "action": self.action,
            "row_count": self.row_count,
            "migrated_row_count": self.migrated_row_count,
            "unchanged_row_count": self.unchanged_row_count,
            "backup_path": str(self.backup_path) if self.backup_path is not None else None,
            "schema_versions_before": dict(sorted(self.schema_versions_before.items())),
            "required_fields_preserved": self.required_fields_preserved,
            "provenance_core_preserved": self.provenance_core_preserved,
        }


@dataclass(frozen=True, slots=True)
class ArchiveMigrateSourceReport:
    source: str
    changed: bool
    file_count: int
    row_count: int
    migrated_row_count: int
    unchanged_row_count: int
    required_fields_preserved: bool
    provenance_core_preserved: bool
    files: tuple[ArchiveMigrateFileReport, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "changed": self.changed,
            "file_count": self.file_count,
            "row_count": self.row_count,
            "migrated_row_count": self.migrated_row_count,
            "unchanged_row_count": self.unchanged_row_count,
            "required_fields_preserved": self.required_fields_preserved,
            "provenance_core_preserved": self.provenance_core_preserved,
            "files": [file_report.to_dict() for file_report in self.files],
        }


@dataclass(frozen=True, slots=True)
class ArchiveMigrateManifestReport:
    input_path: Path
    output_path: Path
    action: str
    rewritten_path_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "input_path": str(self.input_path),
            "output_path": str(self.output_path),
            "action": self.action,
            "rewritten_path_count": self.rewritten_path_count,
        }


@dataclass(frozen=True, slots=True)
class ArchiveMigrateVerificationReport:
    archive_root: Path
    source_filter: str | None
    status: str
    finding_count: int
    warning_count: int
    error_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "archive_root": str(self.archive_root),
            "source_filter": self.source_filter,
            "status": self.status,
            "finding_count": self.finding_count,
            "warning_count": self.warning_count,
            "error_count": self.error_count,
        }


@dataclass(frozen=True, slots=True)
class ArchiveMigrateReport:
    migration_id: str
    archive_root: Path
    output_root: Path
    write_mode: str
    source_filter: str | None
    target_schema_version: str
    backup_root: Path | None
    manifest_count: int
    manifest_rewrite_count: int
    file_count: int
    row_count: int
    migrated_row_count: int
    unchanged_row_count: int
    required_fields_preserved: bool
    provenance_core_preserved: bool
    sources: tuple[ArchiveMigrateSourceReport, ...]
    manifests: tuple[ArchiveMigrateManifestReport, ...]
    post_verify: ArchiveMigrateVerificationReport | None

    def to_dict(self) -> dict[str, object]:
        return {
            "migration_id": self.migration_id,
            "archive_root": str(self.archive_root),
            "output_root": str(self.output_root),
            "write_mode": self.write_mode,
            "source_filter": self.source_filter,
            "target_schema_version": self.target_schema_version,
            "backup_root": str(self.backup_root) if self.backup_root is not None else None,
            "manifest_count": self.manifest_count,
            "manifest_rewrite_count": self.manifest_rewrite_count,
            "source_count": len(self.sources),
            "file_count": self.file_count,
            "row_count": self.row_count,
            "migrated_row_count": self.migrated_row_count,
            "unchanged_row_count": self.unchanged_row_count,
            "required_fields_preserved": self.required_fields_preserved,
            "provenance_core_preserved": self.provenance_core_preserved,
            "sources": {
                source_report.source: source_report.to_dict()
                for source_report in self.sources
            },
            "manifests": [manifest_report.to_dict() for manifest_report in self.manifests],
            "post_verify": (
                None if self.post_verify is None else self.post_verify.to_dict()
            ),
        }


@dataclass(frozen=True, slots=True)
class _ArchiveMigrateFilePlan:
    report: ArchiveMigrateFileReport
    serialized_rows: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _ArchiveMigrateManifestPlan:
    report: ArchiveMigrateManifestReport
    payload: dict[str, object]


def migrate_archive(
    archive_root: Path,
    *,
    output_root: Path | None = None,
    source: str | None = None,
    backup_dir: Path | None = None,
    execute: bool = False,
) -> ArchiveMigrateReport:
    resolved_archive_root = archive_root.expanduser().resolve(strict=False)
    _validate_archive_directory(resolved_archive_root)
    resolved_output_root = (
        resolved_archive_root
        if output_root is None
        else output_root.expanduser().resolve(strict=False)
    )
    _validate_output_root(
        archive_root=resolved_archive_root,
        output_root=resolved_output_root,
        source=source,
    )
    migration_id = _migration_id()
    backup_root = _resolve_backup_root(
        archive_root=resolved_archive_root,
        output_root=resolved_output_root,
        backup_dir=backup_dir,
        migration_id=migration_id,
    )

    write_mode = _resolve_write_mode(
        execute=execute,
        archive_root=resolved_archive_root,
        output_root=resolved_output_root,
    )
    selected_sources = _select_sources(resolved_archive_root, source=source)
    source_plans = tuple(
        _plan_source_migration(
            resolved_archive_root,
            output_root=resolved_output_root,
            source_name=source_name,
            backup_root=backup_root,
        )
        for source_name in selected_sources
    )
    manifest_plans = _plan_manifest_migrations(
        resolved_archive_root,
        output_root=resolved_output_root,
    )

    post_verify = None
    if execute:
        if resolved_output_root != resolved_archive_root:
            resolved_output_root.mkdir(parents=True, exist_ok=True)
        if backup_root is not None:
            backup_root.mkdir(parents=True, exist_ok=False)
        for source_plan in source_plans:
            _write_source_plan(source_plan)
        for manifest_plan in manifest_plans:
            _write_manifest_plan(manifest_plan)
        verify_root = resolved_output_root
        verify_report = verify_archive(verify_root, source=source)
        post_verify = ArchiveMigrateVerificationReport(
            archive_root=verify_root,
            source_filter=source,
            status=verify_report.status.value,
            finding_count=verify_report.finding_count,
            warning_count=verify_report.warning_count,
            error_count=verify_report.error_count,
        )
        if verify_report.error_count:
            raise ArchiveMigrateError(
                f"post-migration verify reported {verify_report.error_count} errors"
            )

    return ArchiveMigrateReport(
        migration_id=migration_id,
        archive_root=resolved_archive_root,
        output_root=resolved_output_root,
        write_mode=write_mode,
        source_filter=source,
        target_schema_version=SCHEMA_VERSION,
        backup_root=backup_root,
        manifest_count=len(manifest_plans),
        manifest_rewrite_count=sum(
            1 for plan in manifest_plans if plan.report.action != "noop"
        ),
        file_count=sum(len(plan.report.files) for plan in source_plans),
        row_count=sum(plan.report.row_count for plan in source_plans),
        migrated_row_count=sum(plan.report.migrated_row_count for plan in source_plans),
        unchanged_row_count=sum(plan.report.unchanged_row_count for plan in source_plans),
        required_fields_preserved=all(
            plan.report.required_fields_preserved for plan in source_plans
        ),
        provenance_core_preserved=all(
            plan.report.provenance_core_preserved for plan in source_plans
        ),
        sources=tuple(plan.report for plan in source_plans),
        manifests=tuple(plan.report for plan in manifest_plans),
        post_verify=post_verify,
    )


def _validate_archive_directory(path: Path) -> None:
    if not path.exists():
        raise ArchiveInspectError(f"archive root does not exist: {path}")
    if not path.is_dir():
        raise ArchiveInspectError(f"archive root is not a directory: {path}")


def _validate_output_root(
    *,
    archive_root: Path,
    output_root: Path,
    source: str | None,
) -> None:
    if output_root == archive_root:
        return
    if source is not None:
        raise ArchiveMigrateError(
            "archive migrate does not support combining --source with --output-root"
        )
    if _is_within(output_root, archive_root) or _is_within(archive_root, output_root):
        raise ArchiveMigrateError(
            "archive migrate output_root must not be nested inside archive_root"
        )


def _resolve_backup_root(
    *,
    archive_root: Path,
    output_root: Path,
    backup_dir: Path | None,
    migration_id: str,
) -> Path | None:
    if backup_dir is None:
        return None
    if output_root != archive_root:
        raise ArchiveMigrateError(
            "archive migrate does not support combining --backup-dir with --output-root"
        )
    resolved_backup_dir = backup_dir.expanduser().resolve(strict=False)
    backup_root = resolved_backup_dir / f"archive-migrate-{migration_id}"
    if _is_within(backup_root, archive_root) or _is_within(archive_root, backup_root):
        raise ArchiveMigrateError(
            "archive migrate backup_dir must not overlap archive_root"
        )
    return backup_root


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


def _plan_source_migration(
    archive_root: Path,
    *,
    output_root: Path,
    source_name: str,
    backup_root: Path | None,
) -> _ArchiveMigrateSourcePlan:
    output_paths = tuple(sorted((archive_root / source_name).glob(ARCHIVE_OUTPUT_GLOB)))
    file_plans = tuple(
        _plan_file_migration(
            archive_root,
            output_root=output_root,
            source_name=source_name,
            input_path=input_path,
            backup_root=backup_root,
        )
        for input_path in output_paths
    )
    report = ArchiveMigrateSourceReport(
        source=source_name,
        changed=any(
            file_plan.report.action in {"migrate", "copy"} for file_plan in file_plans
        ),
        file_count=len(file_plans),
        row_count=sum(file_plan.report.row_count for file_plan in file_plans),
        migrated_row_count=sum(
            file_plan.report.migrated_row_count for file_plan in file_plans
        ),
        unchanged_row_count=sum(
            file_plan.report.unchanged_row_count for file_plan in file_plans
        ),
        required_fields_preserved=all(
            file_plan.report.required_fields_preserved for file_plan in file_plans
        ),
        provenance_core_preserved=all(
            file_plan.report.provenance_core_preserved for file_plan in file_plans
        ),
        files=tuple(file_plan.report for file_plan in file_plans),
    )
    return _ArchiveMigrateSourcePlan(report=report, files=file_plans)


@dataclass(frozen=True, slots=True)
class _ArchiveMigrateSourcePlan:
    report: ArchiveMigrateSourceReport
    files: tuple[_ArchiveMigrateFilePlan, ...]


def _plan_file_migration(
    archive_root: Path,
    *,
    output_root: Path,
    source_name: str,
    input_path: Path,
    backup_root: Path | None,
) -> _ArchiveMigrateFilePlan:
    raw_lines = input_path.read_text(encoding="utf-8").splitlines()
    serialized_rows: list[str] = []
    row_count = 0
    migrated_row_count = 0
    unchanged_row_count = 0
    schema_versions_before: dict[str, int] = {}
    required_fields_preserved = True
    provenance_core_preserved = True

    for row_number, raw_line in enumerate(raw_lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        row_count += 1
        payload = _load_payload(line, input_path=input_path, row_number=row_number)
        schema_version = _schema_version(payload, input_path=input_path, row_number=row_number)
        schema_versions_before[schema_version] = schema_versions_before.get(schema_version, 0) + 1
        _validate_migratable_row(
            payload,
            source_name=source_name,
            input_path=input_path,
            row_number=row_number,
        )
        migrated_payload, changed = _migrate_payload(
            payload,
            input_path=input_path,
            row_number=row_number,
        )
        _validate_migrated_row(
            migrated_payload,
            source_name=source_name,
            output_path=output_root / input_path.relative_to(archive_root),
            row_number=row_number,
        )
        if _provenance_core_signature(payload) != _provenance_core_signature(
            migrated_payload
        ):
            provenance_core_preserved = False
            raise ArchiveMigrateError(
                f"source '{source_name}' row {row_number} provenance core changed"
            )
        if not _required_fields_present(migrated_payload):
            required_fields_preserved = False
            raise ArchiveMigrateError(
                f"source '{source_name}' row {row_number} lost required fields"
            )
        if changed:
            migrated_row_count += 1
        else:
            unchanged_row_count += 1
        serialized_rows.append(
            line if not changed else json.dumps(migrated_payload, ensure_ascii=False)
        )

    output_path = output_root / input_path.relative_to(archive_root)
    backup_path = None
    if backup_root is not None and migrated_row_count > 0 and output_path == input_path:
        backup_path = backup_root / input_path.relative_to(archive_root)
    action = _file_action(
        input_path=input_path,
        output_path=output_path,
        migrated_row_count=migrated_row_count,
    )
    report = ArchiveMigrateFileReport(
        source=source_name,
        input_path=input_path,
        output_path=output_path,
        action=action,
        row_count=row_count,
        migrated_row_count=migrated_row_count,
        unchanged_row_count=unchanged_row_count,
        backup_path=backup_path,
        schema_versions_before=schema_versions_before,
        required_fields_preserved=required_fields_preserved,
        provenance_core_preserved=provenance_core_preserved,
    )
    return _ArchiveMigrateFilePlan(report=report, serialized_rows=tuple(serialized_rows))


def _plan_manifest_migrations(
    archive_root: Path,
    *,
    output_root: Path,
) -> tuple[_ArchiveMigrateManifestPlan, ...]:
    runs_dir = archive_root / RUNS_DIRECTORY
    manifest_paths = tuple(sorted(runs_dir.glob(f"*/{MANIFEST_FILENAME}")))
    if output_root == archive_root:
        return tuple(
            _ArchiveMigrateManifestPlan(
                report=ArchiveMigrateManifestReport(
                    input_path=manifest_path,
                    output_path=manifest_path,
                    action="noop",
                    rewritten_path_count=0,
                ),
                payload={},
            )
            for manifest_path in manifest_paths
        )

    plans: list[_ArchiveMigrateManifestPlan] = []
    for manifest_path in manifest_paths:
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ArchiveMigrateError(
                f"run manifest is not valid JSON: {manifest_path}: {exc.msg}"
            ) from exc
        if not isinstance(payload, dict):
            raise ArchiveMigrateError(f"run manifest root must be an object: {manifest_path}")
        rewritten_payload, rewritten_path_count = _rewrite_archive_paths(
            payload,
            archive_root=archive_root,
            output_root=output_root,
        )
        output_path = output_root / manifest_path.relative_to(archive_root)
        plans.append(
            _ArchiveMigrateManifestPlan(
                report=ArchiveMigrateManifestReport(
                    input_path=manifest_path,
                    output_path=output_path,
                    action="rewrite",
                    rewritten_path_count=rewritten_path_count,
                ),
                payload=rewritten_payload,
            )
        )
    return tuple(plans)


def _rewrite_archive_paths(
    payload: dict[str, object],
    *,
    archive_root: Path,
    output_root: Path,
) -> tuple[dict[str, object], int]:
    rewritten_count = 0
    rewritten: dict[str, object] = {}
    for key, value in payload.items():
        updated_value, updated_count = _rewrite_archive_value(
            value,
            archive_root=archive_root,
            output_root=output_root,
        )
        rewritten[key] = updated_value
        rewritten_count += updated_count
    return rewritten, rewritten_count


def _rewrite_archive_value(
    value: object,
    *,
    archive_root: Path,
    output_root: Path,
) -> tuple[object, int]:
    if isinstance(value, dict):
        rewritten, count = _rewrite_archive_paths(
            value,
            archive_root=archive_root,
            output_root=output_root,
        )
        return rewritten, count
    if isinstance(value, list):
        rewritten_items: list[object] = []
        rewritten_count = 0
        for item in value:
            rewritten_item, item_count = _rewrite_archive_value(
                item,
                archive_root=archive_root,
                output_root=output_root,
            )
            rewritten_items.append(rewritten_item)
            rewritten_count += item_count
        return rewritten_items, rewritten_count
    if isinstance(value, str):
        candidate = Path(value).expanduser()
        if candidate.is_absolute():
            resolved_candidate = candidate.resolve(strict=False)
            if _is_within(resolved_candidate, archive_root):
                return str(output_root / resolved_candidate.relative_to(archive_root)), 1
    return value, 0


def _write_source_plan(plan: _ArchiveMigrateSourcePlan) -> None:
    for file_plan in plan.files:
        report = file_plan.report
        if report.backup_path is not None:
            report.backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(report.input_path, report.backup_path)
        if report.action == "noop":
            continue
        _write_output_file(report.output_path, file_plan.serialized_rows)


def _write_manifest_plan(plan: _ArchiveMigrateManifestPlan) -> None:
    if plan.report.action == "noop":
        return
    plan.report.output_path.parent.mkdir(parents=True, exist_ok=True)
    plan.report.output_path.write_text(
        json.dumps(plan.payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _write_output_file(output_path: Path, serialized_rows: tuple[str, ...]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.parent / f".{output_path.name}.tmp"
    try:
        temporary_path.write_text(
            "".join(f"{row}\n" for row in serialized_rows),
            encoding="utf-8",
        )
        temporary_path.replace(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _load_payload(line: str, *, input_path: Path, row_number: int) -> dict[str, object]:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ArchiveMigrateError(
            f"source archive row is not valid JSON at {input_path}:{row_number}: {exc.msg}"
        ) from exc
    if not isinstance(payload, dict):
        raise ArchiveMigrateError(
            f"source archive row must decode to an object at {input_path}:{row_number}"
        )
    return payload


def _schema_version(
    payload: Mapping[str, object],
    *,
    input_path: Path,
    row_number: int,
) -> str:
    contract = payload.get("contract")
    if not isinstance(contract, Mapping):
        raise ArchiveMigrateError(
            f"source archive row is missing contract object at {input_path}:{row_number}"
        )
    schema_version = contract.get("schema_version")
    if not isinstance(schema_version, str) or not schema_version:
        raise ArchiveMigrateError(
            f"source archive row is missing contract schema_version at {input_path}:{row_number}"
        )
    return schema_version


def _validate_migratable_row(
    payload: dict[str, object],
    *,
    source_name: str,
    input_path: Path,
    row_number: int,
) -> None:
    findings = _collect_row_findings(
        payload,
        source_name=source_name,
        output_path=input_path,
        row_number=row_number,
    )
    blocking_findings = [
        finding
        for finding in findings
        if finding.level == ValidationLevel.ERROR and finding.code != "invalid_contract"
    ]
    if blocking_findings:
        message = "; ".join(finding.message for finding in blocking_findings)
        raise ArchiveMigrateError(message)


def _validate_migrated_row(
    payload: dict[str, object],
    *,
    source_name: str,
    output_path: Path,
    row_number: int,
) -> None:
    findings = _collect_row_findings(
        payload,
        source_name=source_name,
        output_path=output_path,
        row_number=row_number,
    )
    blocking_findings = [
        finding for finding in findings if finding.level == ValidationLevel.ERROR
    ]
    if blocking_findings:
        message = "; ".join(finding.message for finding in blocking_findings)
        raise ArchiveMigrateError(message)


def _collect_row_findings(
    payload: dict[str, object],
    *,
    source_name: str,
    output_path: Path,
    row_number: int,
) -> list[ValidationFinding]:
    findings: list[ValidationFinding] = []
    _verify_row(
        json.dumps(payload, ensure_ascii=False),
        source_name=source_name,
        output_path=output_path,
        row_number=row_number,
        findings=findings,
    )
    return findings


def _migrate_payload(
    payload: dict[str, object],
    *,
    input_path: Path,
    row_number: int,
) -> tuple[dict[str, object], bool]:
    schema_version = _schema_version(payload, input_path=input_path, row_number=row_number)
    if schema_version == SCHEMA_VERSION:
        return payload, False
    if schema_version != "2026-03-18":
        raise ArchiveMigrateError(
            "archive migrate does not support schema_version "
            f"{schema_version!r} at {input_path}:{row_number}"
        )
    migrated_payload = _clone_json_object(payload)
    contract = migrated_payload.get("contract")
    if not isinstance(contract, dict):
        raise ArchiveMigrateError(
            f"source archive row is missing contract object at {input_path}:{row_number}"
        )
    migrated_contract = _clone_json_object(contract)
    migrated_contract["schema_version"] = SCHEMA_VERSION
    migrated_payload["contract"] = migrated_contract
    return migrated_payload, True


def _required_fields_present(payload: Mapping[str, object]) -> bool:
    source = payload.get("source")
    execution_context = payload.get("execution_context")
    collected_at = payload.get("collected_at")
    messages = payload.get("messages")
    contract = payload.get("contract")
    if not isinstance(source, str) or not source:
        return False
    if not isinstance(execution_context, str) or not execution_context:
        return False
    if not isinstance(collected_at, str) or not collected_at:
        return False
    if not isinstance(contract, Mapping):
        return False
    schema_version = contract.get("schema_version")
    if not isinstance(schema_version, str) or not schema_version:
        return False
    if not isinstance(messages, list):
        return False
    return True


def _provenance_core_signature(payload: Mapping[str, object]) -> str:
    return json.dumps(
        {
            "provenance": payload.get("provenance"),
            "source_artifact_path": payload.get("source_artifact_path"),
            "source_session_id": payload.get("source_session_id"),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _file_action(
    *,
    input_path: Path,
    output_path: Path,
    migrated_row_count: int,
) -> str:
    if migrated_row_count > 0:
        return "migrate"
    if input_path != output_path:
        return "copy"
    return "noop"


def _clone_json_object(payload: Mapping[str, object]) -> dict[str, object]:
    cloned: dict[str, object] = {}
    for key, value in payload.items():
        cloned[key] = _clone_json_value(value)
    return cloned


def _clone_json_value(value: object) -> object:
    if isinstance(value, dict):
        return _clone_json_object(value)
    if isinstance(value, list):
        return [_clone_json_value(item) for item in value]
    return value


def _is_within(candidate: Path, root: Path) -> bool:
    return candidate == root or root in candidate.parents


def _migration_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


__all__ = [
    "ArchiveMigrateError",
    "ArchiveMigrateFileReport",
    "ArchiveMigrateManifestReport",
    "ArchiveMigrateReport",
    "ArchiveMigrateSourceReport",
    "ArchiveMigrateVerificationReport",
    "migrate_archive",
]
