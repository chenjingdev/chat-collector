from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

SCHEMA_VERSION = "2026-03-19"
DEFAULT_ARCHIVE_ROOT = Path("/Users/chenjing/dev/chat-history")
EXCLUDED_ARTIFACTS = (
    "tool_calls",
    "mcp_invocation_noise",
    "internal_reasoning",
    "execution_artifacts",
)


class SupportLevel(StrEnum):
    SCAFFOLD = "scaffold"
    PARTIAL = "partial"
    COMPLETE = "complete"


class SourceSelectionProfile(StrEnum):
    ALL = "all"
    DEFAULT = "default"
    COMPLETE_ONLY = "complete_only"


class TranscriptCompleteness(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    UNSUPPORTED = "unsupported"


class SourceRootPlatform(StrEnum):
    DARWIN = "darwin"
    LINUX = "linux"
    WINDOWS = "windows"


class SourceRunStatus(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"


class RerunSelectionReason(StrEnum):
    FAILED = "failed"
    DEGRADED = "degraded"
    FAILED_OR_DEGRADED = "failed_or_degraded"


class RerunSelectionPreset(StrEnum):
    FAILED_ONLY = "failed_only"
    DEGRADED_ONLY = "degraded_only"
    FAILED_AND_DEGRADED = "failed_and_degraded"

    @property
    def selection_reason(self) -> RerunSelectionReason:
        if self == RerunSelectionPreset.FAILED_ONLY:
            return RerunSelectionReason.FAILED
        if self == RerunSelectionPreset.DEGRADED_ONLY:
            return RerunSelectionReason.DEGRADED
        return RerunSelectionReason.FAILED_OR_DEGRADED

    @classmethod
    def from_selection_reason(
        cls,
        selection_reason: str | RerunSelectionReason,
    ) -> RerunSelectionPreset:
        resolved_reason = RerunSelectionReason(selection_reason)
        if resolved_reason == RerunSelectionReason.FAILED:
            return cls.FAILED_ONLY
        if resolved_reason == RerunSelectionReason.DEGRADED:
            return cls.DEGRADED_ONLY
        return cls.FAILED_AND_DEGRADED


class RedactionMode(StrEnum):
    ON = "on"
    OFF = "off"


class ValidationMode(StrEnum):
    OFF = "off"
    REPORT = "report"
    STRICT = "strict"


class ScheduledRunMode(StrEnum):
    COLLECT = "collect"
    RERUN = "rerun"


class MessageRole(StrEnum):
    SYSTEM = "system"
    DEVELOPER = "developer"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class ArchiveTargetPolicy:
    repo_root: Path
    default_archive_root: Path = DEFAULT_ARCHIVE_ROOT
    mode: str = "external_only"
    fixtures_only_inside_repo: bool = True

    def validate(self, archive_root: Path) -> Path:
        candidate = archive_root.expanduser()
        if not candidate.is_absolute():
            raise ValueError("archive root must be an absolute path outside the repository")

        resolved_root = candidate.resolve(strict=False)
        resolved_repo = self.repo_root.expanduser().resolve(strict=False)
        if resolved_root == resolved_repo or resolved_repo in resolved_root.parents:
            raise ValueError(
                f"archive root must stay outside the repository: {resolved_root}"
            )
        return resolved_root

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "default_archive_root": str(self.default_archive_root),
            "fixtures_only_inside_repo": self.fixtures_only_inside_repo,
            "repo_root": str(self.repo_root.resolve(strict=False)),
        }


@dataclass(frozen=True, slots=True)
class NormalizationContract:
    schema_version: str = SCHEMA_VERSION
    archive_kind: str = "memory_chat_v1"
    focus: str = "memory_usefulness"
    allowed_roles: tuple[MessageRole, ...] = (
        MessageRole.SYSTEM,
        MessageRole.DEVELOPER,
        MessageRole.USER,
        MessageRole.ASSISTANT,
    )
    excluded_artifacts: tuple[str, ...] = EXCLUDED_ARTIFACTS

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "archive_kind": self.archive_kind,
            "focus": self.focus,
            "allowed_roles": [role.value for role in self.allowed_roles],
            "excluded_artifacts": list(self.excluded_artifacts),
        }


@dataclass(frozen=True, slots=True)
class NormalizedImage:
    source: str
    mime_type: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {"source": self.source}
        if self.mime_type is not None:
            payload["mime_type"] = self.mime_type
        return payload


@dataclass(frozen=True, slots=True)
class MessageProvenance:
    body_source: str
    fallback: bool = False

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {"body_source": self.body_source}
        if self.fallback:
            payload["fallback"] = True
        return payload


@dataclass(frozen=True, slots=True)
class NormalizedMessage:
    role: MessageRole
    text: str | None = None
    images: tuple[NormalizedImage, ...] = ()
    timestamp: str | None = None
    source_message_id: str | None = None
    provenance: MessageProvenance | None = None

    def __post_init__(self) -> None:
        if self.text is None and not self.images:
            raise ValueError("normalized message requires text or images")

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {"role": self.role.value}
        if self.text is not None:
            payload["text"] = self.text
        if self.images:
            payload["images"] = [image.to_dict() for image in self.images]
        if self.timestamp is not None:
            payload["timestamp"] = self.timestamp
        if self.source_message_id is not None:
            payload["source_message_id"] = self.source_message_id
        if self.provenance is not None:
            payload["provenance"] = self.provenance.to_dict()
        return payload


@dataclass(frozen=True, slots=True)
class IdeBridgeProvenance:
    hosts: tuple[str, ...] = ()
    state_db_paths: tuple[str, ...] = ()
    config_paths: tuple[str, ...] = ()
    history_paths: tuple[str, ...] = ()
    keybinding_paths: tuple[str, ...] = ()
    log_paths: tuple[str, ...] = ()
    recent_file_paths: tuple[str, ...] = ()
    bridge_payload_paths: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {}
        if self.hosts:
            payload["hosts"] = list(self.hosts)
        if self.state_db_paths:
            payload["state_db_paths"] = list(self.state_db_paths)
        if self.config_paths:
            payload["config_paths"] = list(self.config_paths)
        if self.history_paths:
            payload["history_paths"] = list(self.history_paths)
        if self.keybinding_paths:
            payload["keybinding_paths"] = list(self.keybinding_paths)
        if self.log_paths:
            payload["log_paths"] = list(self.log_paths)
        if self.recent_file_paths:
            payload["recent_file_paths"] = list(self.recent_file_paths)
        if self.bridge_payload_paths:
            payload["bridge_payload_paths"] = list(self.bridge_payload_paths)
        return payload


@dataclass(frozen=True, slots=True)
class AppShellProvenance:
    application_support_roots: tuple[str, ...] = ()
    log_roots: tuple[str, ...] = ()
    state_db_paths: tuple[str, ...] = ()
    log_paths: tuple[str, ...] = ()
    preference_paths: tuple[str, ...] = ()
    cache_roots: tuple[str, ...] = ()
    auxiliary_paths: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {}
        if self.application_support_roots:
            payload["application_support_roots"] = list(self.application_support_roots)
        if self.log_roots:
            payload["log_roots"] = list(self.log_roots)
        if self.state_db_paths:
            payload["state_db_paths"] = list(self.state_db_paths)
        if self.log_paths:
            payload["log_paths"] = list(self.log_paths)
        if self.preference_paths:
            payload["preference_paths"] = list(self.preference_paths)
        if self.cache_roots:
            payload["cache_roots"] = list(self.cache_roots)
        if self.auxiliary_paths:
            payload["auxiliary_paths"] = list(self.auxiliary_paths)
        return payload


@dataclass(frozen=True, slots=True)
class AutomationRunProvenance:
    automation_id: str
    automation_name: str | None = None
    status: str | None = None
    schedule: str | None = None
    source_cwd: str | None = None
    model: str | None = None
    reasoning_effort: str | None = None
    definition_path: str | None = None
    thread_title: str | None = None
    thread_record_title: str | None = None
    inbox_title: str | None = None
    inbox_summary: str | None = None
    resolved_title: str | None = None
    resolved_title_source: str | None = None
    resolved_summary: str | None = None
    resolved_summary_source: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "automation_id": self.automation_id,
        }
        if self.automation_name is not None:
            payload["automation_name"] = self.automation_name
        if self.status is not None:
            payload["status"] = self.status
        if self.schedule is not None:
            payload["schedule"] = self.schedule
        if self.source_cwd is not None:
            payload["source_cwd"] = self.source_cwd
        if self.model is not None:
            payload["model"] = self.model
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort
        if self.definition_path is not None:
            payload["definition_path"] = self.definition_path
        if self.thread_title is not None:
            payload["thread_title"] = self.thread_title
        if self.thread_record_title is not None:
            payload["thread_record_title"] = self.thread_record_title
        if self.inbox_title is not None:
            payload["inbox_title"] = self.inbox_title
        if self.inbox_summary is not None:
            payload["inbox_summary"] = self.inbox_summary
        if self.resolved_title is not None:
            payload["resolved_title"] = self.resolved_title
        if self.resolved_title_source is not None:
            payload["resolved_title_source"] = self.resolved_title_source
        if self.resolved_summary is not None:
            payload["resolved_summary"] = self.resolved_summary
        if self.resolved_summary_source is not None:
            payload["resolved_summary_source"] = self.resolved_summary_source
        return payload


@dataclass(frozen=True, slots=True)
class ConversationProvenance:
    session_started_at: str | None = None
    source: str | None = None
    originator: str | None = None
    cwd: str | None = None
    cli_version: str | None = None
    archived: bool | None = None
    archived_reason: str | None = None
    conversation_origin: str | None = None
    automation: AutomationRunProvenance | None = None
    ide_bridge: IdeBridgeProvenance | None = None
    app_shell: AppShellProvenance | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {}
        if self.session_started_at is not None:
            payload["session_started_at"] = self.session_started_at
        if self.source is not None:
            payload["source"] = self.source
        if self.originator is not None:
            payload["originator"] = self.originator
        if self.cwd is not None:
            payload["cwd"] = self.cwd
        if self.cli_version is not None:
            payload["cli_version"] = self.cli_version
        if self.archived is not None:
            payload["archived"] = self.archived
        if self.archived_reason is not None:
            payload["archived_reason"] = self.archived_reason
        if self.conversation_origin is not None:
            payload["conversation_origin"] = self.conversation_origin
        if self.automation is not None:
            automation_payload = self.automation.to_dict()
            if automation_payload:
                payload["automation"] = automation_payload
        if self.ide_bridge is not None:
            ide_bridge_payload = self.ide_bridge.to_dict()
            if ide_bridge_payload:
                payload["ide_bridge"] = ide_bridge_payload
        if self.app_shell is not None:
            app_shell_payload = self.app_shell.to_dict()
            if app_shell_payload:
                payload["app_shell"] = app_shell_payload
        return payload


@dataclass(frozen=True, slots=True)
class NormalizedConversation:
    source: str
    execution_context: str
    collected_at: str
    messages: tuple[NormalizedMessage, ...]
    contract: NormalizationContract = field(default_factory=NormalizationContract)
    transcript_completeness: TranscriptCompleteness = TranscriptCompleteness.COMPLETE
    limitations: tuple[str, ...] = ()
    source_session_id: str | None = None
    source_artifact_path: str | None = None
    session_metadata: dict[str, object] | None = None
    provenance: ConversationProvenance | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "source": self.source,
            "execution_context": self.execution_context,
            "collected_at": self.collected_at,
            "messages": [message.to_dict() for message in self.messages],
            "contract": self.contract.to_dict(),
        }
        if self.transcript_completeness != TranscriptCompleteness.COMPLETE:
            payload["transcript_completeness"] = self.transcript_completeness.value
        if self.limitations:
            payload["limitations"] = list(self.limitations)
        if self.source_session_id is not None:
            payload["source_session_id"] = self.source_session_id
        if self.source_artifact_path is not None:
            payload["source_artifact_path"] = self.source_artifact_path
        if self.session_metadata is not None:
            payload["session_metadata"] = self.session_metadata
        if self.provenance is not None:
            payload["provenance"] = self.provenance.to_dict()
        return payload


@dataclass(frozen=True, slots=True)
class SourceSupportMetadata:
    product_label: str
    host_surface: str
    expected_transcript_completeness: TranscriptCompleteness
    limitation_summary: str | None = None
    limitations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "product_label": self.product_label,
            "host_surface": self.host_surface,
            "expected_transcript_completeness": (
                self.expected_transcript_completeness.value
            ),
        }
        if self.limitation_summary is not None:
            payload["limitation_summary"] = self.limitation_summary
        if self.limitations:
            payload["limitations"] = list(self.limitations)
        return payload


@dataclass(frozen=True, slots=True)
class SourceRootCandidate:
    path: str
    platforms: tuple[SourceRootPlatform, ...] = (
        SourceRootPlatform.DARWIN,
        SourceRootPlatform.LINUX,
        SourceRootPlatform.WINDOWS,
    )

    def applies_to(self, platform: SourceRootPlatform) -> bool:
        return platform in self.platforms

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "platforms": [platform.value for platform in self.platforms],
        }


@dataclass(frozen=True, slots=True)
class SourceDescriptor:
    key: str
    display_name: str
    execution_context: str
    support_level: SupportLevel
    default_input_roots: tuple[str, ...]
    artifact_root_candidates: tuple[SourceRootCandidate, ...] = ()
    notes: tuple[str, ...] = ()
    support_metadata: SourceSupportMetadata | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "key": self.key,
            "display_name": self.display_name,
            "execution_context": self.execution_context,
            "support_level": self.support_level.value,
            "default_input_roots": list(self.default_input_roots),
            "notes": list(self.notes),
        }
        if self.artifact_root_candidates:
            payload["artifact_root_candidates"] = [
                candidate.to_dict() for candidate in self.artifact_root_candidates
            ]
        if self.support_metadata is not None:
            payload["support_metadata"] = self.support_metadata.to_dict()
        return payload


@dataclass(frozen=True, slots=True)
class SourceSelectionPolicy:
    profile: SourceSelectionProfile
    minimum_support_level: SupportLevel
    include_sources: tuple[str, ...] = ()
    exclude_sources: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "profile": self.profile.value,
            "minimum_support_level": self.minimum_support_level.value,
            "include_sources": list(self.include_sources),
            "exclude_sources": list(self.exclude_sources),
        }


@dataclass(frozen=True, slots=True)
class CollectionExecutionPolicy:
    incremental: bool = True
    redaction: RedactionMode = RedactionMode.ON
    validation: ValidationMode = ValidationMode.OFF

    def to_dict(self) -> dict[str, object]:
        return {
            "incremental": self.incremental,
            "redaction": self.redaction.value,
            "validation": self.validation.value,
        }


@dataclass(frozen=True, slots=True)
class EffectiveCollectConfig:
    archive_root: Path
    selection_policy: SourceSelectionPolicy
    execution_policy: CollectionExecutionPolicy = field(
        default_factory=CollectionExecutionPolicy
    )
    rerun: EffectiveRerunConfig | None = None
    config_source: str = "defaults"
    config_path: Path | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "archive_root": str(self.archive_root),
            "selection_policy": self.selection_policy.to_dict(),
            "execution_policy": self.execution_policy.to_dict(),
            "config_source": self.config_source,
        }
        if self.rerun is not None:
            payload["rerun"] = self.rerun.to_dict()
        if self.config_path is not None:
            payload["config_path"] = str(self.config_path)
        return payload


@dataclass(frozen=True, slots=True)
class RerunMetadata:
    origin_run_id: str
    selection_reason: RerunSelectionReason
    matched_sources: tuple[str, ...] = ()
    manual_include_sources: tuple[str, ...] = ()
    manual_exclude_sources: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "origin_run_id": self.origin_run_id,
            "selection_reason": self.selection_reason.value,
            "matched_sources": list(self.matched_sources),
            "manual_include_sources": list(self.manual_include_sources),
            "manual_exclude_sources": list(self.manual_exclude_sources),
        }


@dataclass(frozen=True, slots=True)
class EffectiveRerunConfig:
    selection_preset: RerunSelectionPreset
    selection_reason: RerunSelectionReason
    source: str

    def to_dict(self) -> dict[str, object]:
        return {
            "selection_preset": self.selection_preset.value,
            "selection_reason": self.selection_reason.value,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class EffectiveScheduledConfig:
    archive_root: Path
    mode: ScheduledRunMode = ScheduledRunMode.COLLECT
    selection_policy: SourceSelectionPolicy = field(
        default_factory=lambda: SourceSelectionPolicy(
            profile=SourceSelectionProfile.DEFAULT,
            minimum_support_level=SupportLevel.COMPLETE,
        )
    )
    execution_policy: CollectionExecutionPolicy = field(
        default_factory=CollectionExecutionPolicy
    )
    rerun: EffectiveRerunConfig | None = None
    stale_after_seconds: int = 21600
    source: str = "defaults"
    config_source: str = "defaults"
    config_path: Path | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "archive_root": str(self.archive_root),
            "mode": self.mode.value,
            "selection_policy": self.selection_policy.to_dict(),
            "execution_policy": self.execution_policy.to_dict(),
            "stale_after_seconds": self.stale_after_seconds,
            "source": self.source,
            "config_source": self.config_source,
        }
        if self.rerun is not None:
            payload["rerun"] = self.rerun.to_dict()
        if self.config_path is not None:
            payload["config_path"] = str(self.config_path)
        return payload


@dataclass(frozen=True, slots=True)
class ScheduledLockRecord:
    path: Path
    acquired_at: str | None = None
    owner_pid: int | None = None
    owner_hostname: str | None = None
    mode: str | None = None
    age_seconds: int | None = None
    stale: bool = False

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "path": str(self.path),
            "stale": self.stale,
        }
        if self.acquired_at is not None:
            payload["acquired_at"] = self.acquired_at
        if self.owner_pid is not None:
            payload["owner_pid"] = self.owner_pid
        if self.owner_hostname is not None:
            payload["owner_hostname"] = self.owner_hostname
        if self.mode is not None:
            payload["mode"] = self.mode
        if self.age_seconds is not None:
            payload["age_seconds"] = self.age_seconds
        return payload


@dataclass(frozen=True, slots=True)
class ScheduledRunMetadata:
    mode: ScheduledRunMode
    lock: ScheduledLockRecord
    stale_after_seconds: int
    config_source: str
    origin_run_id: str | None = None
    force_unlocked_stale_lock: bool = False
    replaced_lock: ScheduledLockRecord | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "mode": self.mode.value,
            "lock": self.lock.to_dict(),
            "stale_after_seconds": self.stale_after_seconds,
            "config_source": self.config_source,
            "force_unlocked_stale_lock": self.force_unlocked_stale_lock,
        }
        if self.origin_run_id is not None:
            payload["origin_run_id"] = self.origin_run_id
        if self.replaced_lock is not None:
            payload["replaced_lock"] = self.replaced_lock.to_dict()
        return payload


@dataclass(frozen=True, slots=True)
class ExcludedSource:
    source: str
    support_level: SupportLevel
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "support_level": self.support_level.value,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class CollectionPlan:
    source: str
    display_name: str
    archive_root: Path
    execution_context: str
    support_level: SupportLevel
    default_input_roots: tuple[str, ...]
    contract: NormalizationContract = field(default_factory=NormalizationContract)
    write_mode: str = "dry_run"
    implemented: bool = False
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "display_name": self.display_name,
            "archive_root": str(self.archive_root),
            "execution_context": self.execution_context,
            "support_level": self.support_level.value,
            "default_input_roots": list(self.default_input_roots),
            "contract": self.contract.to_dict(),
            "write_mode": self.write_mode,
            "implemented": self.implemented,
            "notes": list(self.notes),
        }


@dataclass(frozen=True, slots=True)
class CollectionResult:
    source: str
    archive_root: Path
    output_path: Path
    input_roots: tuple[Path, ...]
    scanned_artifact_count: int
    conversation_count: int
    message_count: int
    skipped_conversation_count: int = 0
    written_conversation_count: int = 0
    upgraded_conversation_count: int = 0
    redaction_event_count: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "archive_root": str(self.archive_root),
            "output_path": str(self.output_path),
            "input_roots": [str(root) for root in self.input_roots],
            "scanned_artifact_count": self.scanned_artifact_count,
            "conversation_count": self.conversation_count,
            "skipped_conversation_count": self.skipped_conversation_count,
            "written_conversation_count": self.written_conversation_count,
            "upgraded_conversation_count": self.upgraded_conversation_count,
            "message_count": self.message_count,
            "redaction_event_count": self.redaction_event_count,
        }


@dataclass(frozen=True, slots=True)
class SourceRunResult:
    source: str
    support_level: SupportLevel
    status: SourceRunStatus
    archive_root: Path
    output_path: Path | None
    input_roots: tuple[Path, ...]
    scanned_artifact_count: int
    conversation_count: int
    message_count: int
    skipped_conversation_count: int = 0
    written_conversation_count: int = 0
    upgraded_conversation_count: int = 0
    partial: bool = False
    unsupported: bool = False
    failed: bool = False
    failure_reason: str | None = None
    support_limitation_summary: str | None = None
    support_limitations: tuple[str, ...] = ()
    redaction_event_count: int = 0

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "source": self.source,
            "support_level": self.support_level.value,
            "status": self.status.value,
            "archive_root": str(self.archive_root),
            "output_path": str(self.output_path) if self.output_path is not None else None,
            "input_roots": [str(root) for root in self.input_roots],
            "scanned_artifact_count": self.scanned_artifact_count,
            "conversation_count": self.conversation_count,
            "skipped_conversation_count": self.skipped_conversation_count,
            "written_conversation_count": self.written_conversation_count,
            "upgraded_conversation_count": self.upgraded_conversation_count,
            "message_count": self.message_count,
            "redaction_event_count": self.redaction_event_count,
            "partial": self.partial,
            "unsupported": self.unsupported,
            "failed": self.failed,
        }
        if self.failure_reason is not None:
            payload["failure_reason"] = self.failure_reason
        if self.support_limitation_summary is not None:
            payload["support_limitation_summary"] = self.support_limitation_summary
        if self.support_limitations:
            payload["support_limitations"] = list(self.support_limitations)
        return payload


@dataclass(frozen=True, slots=True)
class CollectionRunResult:
    run_id: str
    archive_root: Path
    run_dir: Path
    manifest_path: Path
    started_at: str
    completed_at: str
    selection_policy: SourceSelectionPolicy
    effective_config: EffectiveCollectConfig
    selected_sources: tuple[str, ...]
    excluded_sources: tuple[ExcludedSource, ...]
    sources: tuple[SourceRunResult, ...]
    rerun: RerunMetadata | None = None
    scheduled: ScheduledRunMetadata | None = None

    @property
    def redaction_event_count(self) -> int:
        return sum(source.redaction_event_count for source in self.sources)

    @property
    def upgraded_conversation_count(self) -> int:
        return sum(source.upgraded_conversation_count for source in self.sources)

    def to_dict(self) -> dict[str, object]:
        payload = {
            "run_id": self.run_id,
            "archive_root": str(self.archive_root),
            "run_dir": str(self.run_dir),
            "manifest_path": str(self.manifest_path),
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "selection_policy": self.selection_policy.to_dict(),
            "effective_config": self.effective_config.to_dict(),
            "selected_sources": list(self.selected_sources),
            "excluded_sources": [
                excluded_source.to_dict()
                for excluded_source in self.excluded_sources
            ],
            "source_count": len(self.sources),
            "failed_source_count": sum(1 for source in self.sources if source.failed),
            "scanned_artifact_count": sum(
                source.scanned_artifact_count for source in self.sources
            ),
            "conversation_count": sum(source.conversation_count for source in self.sources),
            "skipped_conversation_count": sum(
                source.skipped_conversation_count for source in self.sources
            ),
            "written_conversation_count": sum(
                source.written_conversation_count for source in self.sources
            ),
            "upgraded_conversation_count": self.upgraded_conversation_count,
            "message_count": sum(source.message_count for source in self.sources),
            "redaction_event_count": self.redaction_event_count,
            "sources": [source.to_dict() for source in self.sources],
        }
        if self.rerun is not None:
            payload["rerun"] = self.rerun.to_dict()
        if self.scheduled is not None:
            payload["scheduled"] = self.scheduled.to_dict()
        return payload
