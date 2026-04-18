from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Iterable

BASELINE_POLICY_FILENAME = "baseline-policy.json"
BASELINE_POLICY_VERSION = 1
_NON_SUPPRESSIBLE_FINDING_CODES = frozenset({"drift_suspected"})


class BaselineEntryKind(StrEnum):
    DEGRADED_SOURCE = "degraded_source"
    FINDING = "finding"
    LIMITATION = "limitation"


class BaselineReport(StrEnum):
    VALIDATE = "validate"
    ARCHIVE_VERIFY = "archive_verify"


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
class BaselineEntry:
    id: str
    kind: BaselineEntryKind
    source: str
    reason: str
    report: BaselineReport | None = None
    code: str | None = None
    limitation: str | None = None
    support_level: str | None = None
    status: str | None = None
    added_at: str | None = None

    @property
    def key(self) -> tuple[str, ...]:
        if self.kind == BaselineEntryKind.DEGRADED_SOURCE:
            return (
                self.kind.value,
                self.source,
                self.support_level or "*",
                self.status or "*",
            )
        if self.kind == BaselineEntryKind.FINDING:
            return (
                self.kind.value,
                self.report.value if self.report is not None else "*",
                self.source,
                self.code or "*",
            )
        return (
            self.kind.value,
            self.source,
            self.limitation or "*",
        )

    def matches_degraded_source(
        self,
        *,
        source: str,
        support_level: str | None,
        status: str | None,
        ignore_state: bool = False,
    ) -> bool:
        if self.kind != BaselineEntryKind.DEGRADED_SOURCE or self.source != source:
            return False
        if ignore_state:
            return True
        if self.support_level is not None and self.support_level != support_level:
            return False
        if self.status is not None and self.status != status:
            return False
        return True

    def matches_finding(
        self,
        *,
        report: BaselineReport,
        source: str | None,
        code: str,
        level: str,
    ) -> bool:
        if level != "warning" or source is None or code in _NON_SUPPRESSIBLE_FINDING_CODES:
            return False
        return (
            self.kind == BaselineEntryKind.FINDING
            and self.report == report
            and self.source == source
            and self.code == code
        )

    def matches_limitation(
        self,
        *,
        source: str,
        limitation: str,
    ) -> bool:
        return (
            self.kind == BaselineEntryKind.LIMITATION
            and self.source == source
            and self.limitation == limitation
        )

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "id": self.id,
            "kind": self.kind.value,
            "source": self.source,
            "reason": self.reason,
        }
        if self.report is not None:
            payload["report"] = self.report.value
        if self.code is not None:
            payload["code"] = self.code
        if self.limitation is not None:
            payload["limitation"] = self.limitation
        if self.support_level is not None:
            payload["support_level"] = self.support_level
        if self.status is not None:
            payload["status"] = self.status
        if self.added_at is not None:
            payload["added_at"] = self.added_at
        return payload


@dataclass(frozen=True, slots=True)
class BaselinePolicy:
    path: Path
    entries: tuple[BaselineEntry, ...]

    @property
    def entry_count(self) -> int:
        return len(self.entries)

    def to_dict(self) -> dict[str, object]:
        return {
            "version": BASELINE_POLICY_VERSION,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    def match_degraded_source(
        self,
        *,
        source: str,
        support_level: str | None,
        status: str | None,
        ignore_state: bool = False,
    ) -> BaselineEntry | None:
        matches = [
            entry
            for entry in self.entries
            if entry.matches_degraded_source(
                source=source,
                support_level=support_level,
                status=status,
                ignore_state=ignore_state,
            )
        ]
        if not matches:
            return None
        return sorted(
            matches,
            key=lambda entry: (
                entry.support_level is not None,
                entry.status is not None,
            ),
            reverse=True,
        )[0]

    def match_finding(
        self,
        *,
        report: BaselineReport,
        source: str | None,
        code: str,
        level: str,
    ) -> BaselineEntry | None:
        for entry in self.entries:
            if entry.matches_finding(
                report=report,
                source=source,
                code=code,
                level=level,
            ):
                return entry
        return None

    def match_limitation(
        self,
        *,
        source: str,
        limitation: str,
    ) -> BaselineEntry | None:
        for entry in self.entries:
            if entry.matches_limitation(source=source, limitation=limitation):
                return entry
        return None


def baseline_policy_path(
    archive_root: Path,
    *,
    baseline_path: Path | None = None,
) -> Path:
    if baseline_path is not None:
        return baseline_path.expanduser().resolve(strict=False)
    return (archive_root / BASELINE_POLICY_FILENAME).resolve(strict=False)


def load_baseline_policy(
    path: Path,
    *,
    allow_missing: bool,
) -> BaselinePolicy | None:
    resolved_path = path.expanduser().resolve(strict=False)
    if not resolved_path.exists():
        if allow_missing:
            return None
        raise ValueError(f"baseline policy does not exist: {resolved_path}")
    if not resolved_path.is_file():
        raise ValueError(f"baseline policy is not a file: {resolved_path}")

    try:
        payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"baseline policy is not valid JSON: {resolved_path}") from exc

    if not isinstance(payload, dict):
        raise ValueError("baseline policy root must be an object")

    version = payload.get("version")
    if version != BASELINE_POLICY_VERSION:
        raise ValueError(
            f"baseline policy version must be {BASELINE_POLICY_VERSION}: {resolved_path}"
        )

    entries_payload = payload.get("entries")
    if not isinstance(entries_payload, list):
        raise ValueError("baseline policy field 'entries' must be an array")

    entries = tuple(
        _entry_from_payload(entry_payload, index=index)
        for index, entry_payload in enumerate(entries_payload, start=1)
    )
    return BaselinePolicy(
        path=resolved_path,
        entries=tuple(sorted(entries, key=lambda entry: entry.key)),
    )


def save_baseline_policy(policy: BaselinePolicy) -> None:
    policy.path.parent.mkdir(parents=True, exist_ok=True)
    policy.path.write_text(
        json.dumps(policy.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def merge_baseline_entries(
    policy: BaselinePolicy | None,
    *,
    path: Path,
    entries: Iterable[BaselineEntry],
) -> tuple[BaselinePolicy, int]:
    existing_entries = {} if policy is None else {entry.key: entry for entry in policy.entries}
    added_count = 0
    for entry in entries:
        if entry.key in existing_entries:
            continue
        existing_entries[entry.key] = entry
        added_count += 1
    merged_policy = BaselinePolicy(
        path=path,
        entries=tuple(sorted(existing_entries.values(), key=lambda entry: entry.key)),
    )
    return merged_policy, added_count


def build_degraded_source_entry(
    *,
    source: str,
    reason: str,
    support_level: str | None,
    status: str | None,
    added_at: str | None = None,
) -> BaselineEntry:
    return BaselineEntry(
        id=_entry_id(
            kind=BaselineEntryKind.DEGRADED_SOURCE,
            source=source,
            support_level=support_level,
            status=status,
        ),
        kind=BaselineEntryKind.DEGRADED_SOURCE,
        source=source,
        reason=reason,
        support_level=support_level,
        status=status,
        added_at=added_at,
    )


def build_finding_entry(
    *,
    report: BaselineReport,
    source: str,
    code: str,
    reason: str,
    added_at: str | None = None,
) -> BaselineEntry:
    if code in _NON_SUPPRESSIBLE_FINDING_CODES:
        raise ValueError(f"baseline policy cannot suppress finding code '{code}'")
    return BaselineEntry(
        id=_entry_id(
            kind=BaselineEntryKind.FINDING,
            report=report,
            source=source,
            code=code,
        ),
        kind=BaselineEntryKind.FINDING,
        report=report,
        source=source,
        code=code,
        reason=reason,
        added_at=added_at,
    )


def build_limitation_entry(
    *,
    source: str,
    limitation: str,
    reason: str,
    added_at: str | None = None,
) -> BaselineEntry:
    return BaselineEntry(
        id=_entry_id(
            kind=BaselineEntryKind.LIMITATION,
            source=source,
            limitation=limitation,
        ),
        kind=BaselineEntryKind.LIMITATION,
        source=source,
        limitation=limitation,
        reason=reason,
        added_at=added_at,
    )


def snapshot_entries_from_validate(
    report,
    *,
    reason: str,
) -> tuple[BaselineEntry, ...]:
    added_at = _format_utc_timestamp(_utcnow())
    entries_by_key: dict[tuple[str, ...], BaselineEntry] = {}

    for source_report in report.sources:
        if (
            source_report.support_level not in {None, "complete"}
            or source_report.status not in {None, "complete"}
        ):
            entry = build_degraded_source_entry(
                source=source_report.source,
                reason=reason,
                support_level=(
                    None
                    if source_report.support_level in {None, "complete"}
                    else source_report.support_level
                ),
                status=(
                    None
                    if source_report.status in {None, "complete"}
                    else source_report.status
                ),
                added_at=added_at,
            )
            entries_by_key[entry.key] = entry

    for finding in report.findings:
        if (
            finding.level.value != "warning"
            or finding.source is None
            or finding.code in {"degraded_support_level", "degraded_source_status"}
            or finding.code in _NON_SUPPRESSIBLE_FINDING_CODES
        ):
            continue
        entry = build_finding_entry(
            report=BaselineReport.VALIDATE,
            source=finding.source,
            code=finding.code,
            reason=reason,
            added_at=added_at,
        )
        entries_by_key[entry.key] = entry

    return tuple(sorted(entries_by_key.values(), key=lambda entry: entry.key))


def snapshot_entries_from_archive_verify(
    report,
    *,
    reason: str,
) -> tuple[BaselineEntry, ...]:
    added_at = _format_utc_timestamp(_utcnow())
    entries_by_key: dict[tuple[str, ...], BaselineEntry] = {}
    for finding in report.findings:
        if (
            finding.level.value != "warning"
            or finding.source is None
            or finding.code in _NON_SUPPRESSIBLE_FINDING_CODES
        ):
            continue
        entry = build_finding_entry(
            report=BaselineReport.ARCHIVE_VERIFY,
            source=finding.source,
            code=finding.code,
            reason=reason,
            added_at=added_at,
        )
        entries_by_key[entry.key] = entry
    return tuple(sorted(entries_by_key.values(), key=lambda entry: entry.key))


def snapshot_entries_from_archive_anomalies(
    report,
    *,
    reason: str,
) -> tuple[BaselineEntry, ...]:
    added_at = _format_utc_timestamp(_utcnow())
    entries_by_key: dict[tuple[str, ...], BaselineEntry] = {}
    for source_report in report.sources:
        for conversation in source_report.suspicious_conversations:
            for limitation in conversation.conversation.limitations:
                entry = build_limitation_entry(
                    source=source_report.source,
                    limitation=limitation,
                    reason=reason,
                    added_at=added_at,
                )
                entries_by_key[entry.key] = entry
    return tuple(sorted(entries_by_key.values(), key=lambda entry: entry.key))


def _entry_from_payload(payload: object, *, index: int) -> BaselineEntry:
    if not isinstance(payload, dict):
        raise ValueError(f"baseline policy entry #{index} must be an object")

    raw_kind = _required_string(payload, "kind", index=index)
    try:
        kind = BaselineEntryKind(raw_kind)
    except ValueError as exc:
        raise ValueError(
            f"baseline policy entry #{index} kind must be one of: "
            f"{', '.join(kind.value for kind in BaselineEntryKind)}"
        ) from exc

    source = _required_string(payload, "source", index=index)
    reason = _required_string(payload, "reason", index=index)
    report = _optional_string(payload, "report")
    code = _optional_string(payload, "code")
    limitation = _optional_string(payload, "limitation")
    support_level = _optional_string(payload, "support_level")
    status = _optional_string(payload, "status")
    added_at = _optional_string(payload, "added_at")

    if kind == BaselineEntryKind.FINDING:
        if report is None:
            raise ValueError(f"baseline policy entry #{index} requires 'report'")
        if code is None:
            raise ValueError(f"baseline policy entry #{index} requires 'code'")
        if code in _NON_SUPPRESSIBLE_FINDING_CODES:
            raise ValueError(f"baseline policy cannot suppress finding code '{code}'")
        try:
            parsed_report = BaselineReport(report)
        except ValueError as exc:
            raise ValueError(
                f"baseline policy entry #{index} report must be one of: "
                f"{', '.join(report.value for report in BaselineReport)}"
            ) from exc
        entry = BaselineEntry(
            id=_optional_string(payload, "id")
            or _entry_id(kind=kind, report=parsed_report, source=source, code=code),
            kind=kind,
            report=parsed_report,
            source=source,
            code=code,
            reason=reason,
            added_at=added_at,
        )
    elif kind == BaselineEntryKind.LIMITATION:
        if limitation is None:
            raise ValueError(f"baseline policy entry #{index} requires 'limitation'")
        entry = BaselineEntry(
            id=_optional_string(payload, "id")
            or _entry_id(kind=kind, source=source, limitation=limitation),
            kind=kind,
            source=source,
            limitation=limitation,
            reason=reason,
            added_at=added_at,
        )
    else:
        entry = BaselineEntry(
            id=_optional_string(payload, "id")
            or _entry_id(
                kind=kind,
                source=source,
                support_level=support_level,
                status=status,
            ),
            kind=kind,
            source=source,
            reason=reason,
            support_level=support_level,
            status=status,
            added_at=added_at,
        )
    return entry


def _required_string(
    payload: dict[str, object],
    field_name: str,
    *,
    index: int,
) -> str:
    value = _optional_string(payload, field_name)
    if value is None:
        raise ValueError(f"baseline policy entry #{index} requires '{field_name}'")
    return value


def _optional_string(
    payload: dict[str, object],
    field_name: str,
) -> str | None:
    value = payload.get(field_name)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"baseline policy field '{field_name}' must be a non-empty string")
    return value


def _entry_id(
    *,
    kind: BaselineEntryKind,
    source: str,
    report: BaselineReport | None = None,
    code: str | None = None,
    limitation: str | None = None,
    support_level: str | None = None,
    status: str | None = None,
) -> str:
    if kind == BaselineEntryKind.FINDING:
        return f"{kind.value}:{report.value}:{source}:{code}"
    if kind == BaselineEntryKind.LIMITATION:
        return f"{kind.value}:{source}:{limitation}"
    return f"{kind.value}:{source}:{support_level or '*'}:{status or '*'}"


__all__ = [
    "BASELINE_POLICY_FILENAME",
    "BASELINE_POLICY_VERSION",
    "BaselineEntry",
    "BaselineEntryKind",
    "BaselinePolicy",
    "BaselineReport",
    "baseline_policy_path",
    "build_degraded_source_entry",
    "build_finding_entry",
    "build_limitation_entry",
    "load_baseline_policy",
    "merge_baseline_entries",
    "save_baseline_policy",
    "snapshot_entries_from_archive_anomalies",
    "snapshot_entries_from_archive_verify",
    "snapshot_entries_from_validate",
]
