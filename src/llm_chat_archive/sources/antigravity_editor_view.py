from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

from ..incremental import write_incremental_collection
from ..models import (
    AppShellProvenance,
    CollectionPlan,
    CollectionResult,
    ConversationProvenance,
    MessageRole,
    NormalizedConversation,
    NormalizedMessage,
    SourceDescriptor,
    SourceSupportMetadata,
    SupportLevel,
    TranscriptCompleteness,
)
from ..source_roots import (
    all_platform_root,
    darwin_root,
    default_descriptor_input_roots,
    linux_root,
    windows_root,
)
from .codex_rollout import resolve_input_roots, utc_timestamp

ANTIGRAVITY_EDITOR_VIEW_DESCRIPTOR = SourceDescriptor(
    key="antigravity_editor_view",
    display_name="Antigravity Editor View",
    execution_context="ide_native",
    support_level=SupportLevel.PARTIAL,
    default_input_roots=(
        "~/Library/Application Support/Antigravity",
        "~/.gemini/antigravity",
    ),
    artifact_root_candidates=(
        darwin_root("$HOME/Library/Application Support/Antigravity"),
        linux_root("$XDG_CONFIG_HOME/Antigravity"),
        windows_root("$APPDATA/Antigravity"),
        all_platform_root("$HOME/.gemini/antigravity"),
    ),
    notes=(
        "Uses ~/.gemini/antigravity/conversations/<uuid>.pb as the primary transcript source.",
        "Reconstructs user and assistant messages from the confirmed raw conversation protobuf variant when session and message fields match the verified mapping.",
        "Falls back to partial or unsupported rows with explicit variant_unknown or decode_failed diagnostics when conversation blob framing or message mapping differs from the confirmed variant.",
        "Treats brain, annotations, browser recordings, shared state, html artifacts, and daemon logs as provenance or noise rather than transcript body.",
    ),
    support_metadata=SourceSupportMetadata(
        product_label="Antigravity",
        host_surface="Editor view",
        expected_transcript_completeness=TranscriptCompleteness.PARTIAL,
        limitation_summary="Only the confirmed raw Antigravity conversation protobuf variant is promoted to transcript rows; current operator-local opaque blobs degrade to explicit variant_unknown or decode_failed diagnostics instead of false-complete output.",
        limitations=(
            "Only the verified raw conversation protobuf field mapping is promoted to complete transcript rows.",
            "Opaque or drifted conversation blobs surface explicit variant_unknown or decode_failed diagnostics and remain partial or unsupported until a new framing is confirmed.",
        ),
    ),
)

CONVERSATION_GLOBS = (
    "conversations/*.pb",
    ".gemini/antigravity/conversations/*.pb",
    "**/.gemini/antigravity/conversations/*.pb",
)
BRAIN_GLOBS = (
    "brain/*",
    ".gemini/antigravity/brain/*",
    "**/.gemini/antigravity/brain/*",
)
ANNOTATION_GLOBS = (
    "annotations/*.pbtxt",
    ".gemini/antigravity/annotations/*.pbtxt",
    "**/.gemini/antigravity/annotations/*.pbtxt",
)
BROWSER_RECORDING_GLOBS = (
    "browser_recordings/*",
    ".gemini/antigravity/browser_recordings/*",
    "**/.gemini/antigravity/browser_recordings/*",
)
DAEMON_GLOBS = (
    "daemon/*",
    ".gemini/antigravity/daemon/*",
    "**/.gemini/antigravity/daemon/*",
)
HTML_ARTIFACT_GLOBS = (
    "html_artifacts",
    ".gemini/antigravity/html_artifacts",
    "**/.gemini/antigravity/html_artifacts",
)
APPLICATION_SUPPORT_GLOBS = (
    "Antigravity",
    "Library/Application Support/Antigravity",
    "**/Library/Application Support/Antigravity",
)
LOG_ROOT_GLOBS = (
    "logs",
    "Library/Application Support/Antigravity/logs",
    "**/Library/Application Support/Antigravity/logs",
)
GLOBAL_STATE_GLOBS = (
    "User/globalStorage/state.vscdb",
    "Library/Application Support/Antigravity/User/globalStorage/state.vscdb",
    "**/Library/Application Support/Antigravity/User/globalStorage/state.vscdb",
)
WORKSPACE_STATE_GLOBS = (
    "User/workspaceStorage/*/state.vscdb",
    "Library/Application Support/Antigravity/User/workspaceStorage/*/state.vscdb",
    "**/Library/Application Support/Antigravity/User/workspaceStorage/*/state.vscdb",
)
GLOBAL_STATE_KEYS = (
    "google.antigravity",
    "chat.workspaceTransfer",
    "antigravityUnifiedStateSync.agentManagerWindow",
    "antigravityUnifiedStateSync.artifactReview",
    "antigravityUnifiedStateSync.browserPreferences",
    "antigravityUnifiedStateSync.sidebarWorkspaces",
    "antigravityUnifiedStateSync.trajectorySummaries",
)
WORKSPACE_STATE_KEYS = (
    "chat.ChatSessionStore.index",
    "history.entries",
    "antigravity.agentViewContainerId.state",
    "memento/antigravity.jetskiArtifactsEditor",
    "memento/antigravity.antigravityReviewChangesEditor",
)
CONVERSATION_SESSION_ID_FIELD = 1
CONVERSATION_MESSAGE_FIELD = 2
MESSAGE_ID_FIELD = 1
MESSAGE_ROLE_FIELD = 2
MESSAGE_TEXT_FIELD = 3
MESSAGE_TIMESTAMP_FIELD = 4
ROLE_ENUM_MAP = {
    1: MessageRole.USER,
    2: MessageRole.ASSISTANT,
}
VARIANT_UNKNOWN_LIMITATION = "variant_unknown"
DECODE_FAILED_LIMITATION = "decode_failed"
METADATA_ONLY_LIMITATION = "metadata_only_session_family"
SESSION_ID_MISMATCH_LIMITATION = "conversation_protobuf_session_id_mismatch"
MISSING_USER_LIMITATION = "user_message_missing_from_conversation_protobuf"
MISSING_ASSISTANT_LIMITATION = "assistant_message_missing_from_conversation_protobuf"
UNKNOWN_VARIANT_WIRE_TYPES = frozenset({3, 4, 6, 7})
NOISE_EXCLUSIONS = (
    "browser_recordings",
    "html_artifacts",
    "daemon_logs",
    "unified_state_sync_blobs",
)
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp"})
UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class AntigravityArtifacts:
    application_support_roots: tuple[str, ...] = ()
    antigravity_roots: tuple[str, ...] = ()
    conversation_paths: tuple[str, ...] = ()
    brain_dirs: tuple[str, ...] = ()
    annotation_paths: tuple[str, ...] = ()
    browser_recording_dirs: tuple[str, ...] = ()
    global_state_paths: tuple[str, ...] = ()
    workspace_state_paths: tuple[str, ...] = ()
    log_roots: tuple[str, ...] = ()
    html_artifact_roots: tuple[str, ...] = ()
    daemon_artifact_paths: tuple[str, ...] = ()

    def build_app_shell(self) -> AppShellProvenance | None:
        provenance = AppShellProvenance(
            application_support_roots=self.application_support_roots,
            log_roots=self.log_roots,
            state_db_paths=tuple(
                sorted({*self.global_state_paths, *self.workspace_state_paths})
            ),
            log_paths=self.daemon_artifact_paths,
            auxiliary_paths=self.html_artifact_roots,
        )
        if not provenance.to_dict():
            return None
        return provenance


@dataclass(frozen=True, slots=True)
class ProtobufField:
    number: int
    wire_type: int
    value: int | bytes


class ProtobufDecodeError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        offset: int,
        stage: str,
        field_number: int | None = None,
        wire_type: int | None = None,
    ) -> None:
        super().__init__(message)
        self.offset = offset
        self.stage = stage
        self.field_number = field_number
        self.wire_type = wire_type


@dataclass(frozen=True, slots=True)
class AntigravityTranscriptRecovery:
    messages: tuple[NormalizedMessage, ...]
    completeness: TranscriptCompleteness
    limitations: tuple[str, ...] = ()
    protobuf_session_id: str | None = None
    session_started_at: str | None = None
    decode_status: str = "unsupported"
    decode_error: str | None = None
    decoded_message_count: int = 0
    skipped_message_count: int = 0
    user_message_count: int = 0
    assistant_message_count: int = 0
    diagnostic_reason: str | None = None
    diagnostic_details: dict[str, object] | None = None


@dataclass(frozen=True, slots=True)
class AntigravityEditorViewCollector:
    descriptor: SourceDescriptor = ANTIGRAVITY_EDITOR_VIEW_DESCRIPTOR

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
        artifacts = discover_antigravity_editor_view_artifacts(resolved_input_roots)
        collected_at = utc_timestamp()
        conversations = (
            parse_conversation_blob(
                Path(conversation_path),
                collected_at=collected_at,
                artifacts=artifacts,
            )
            for conversation_path in artifacts.conversation_paths
        )
        return write_incremental_collection(
            source=self.descriptor.key,
            archive_root=archive_root,
            input_roots=resolved_input_roots,
            scanned_artifact_count=len(artifacts.conversation_paths),
            collected_at=collected_at,
            conversations=conversations,
        )

    def _default_input_roots(self) -> tuple[Path, ...]:
        return default_descriptor_input_roots(self.descriptor)


def discover_antigravity_editor_view_artifacts(
    input_roots: tuple[Path, ...] | None,
) -> AntigravityArtifacts:
    if not input_roots:
        return AntigravityArtifacts()

    return AntigravityArtifacts(
        application_support_roots=_discover_paths(
            input_roots,
            direct_match=_is_application_support_root,
            glob_patterns=APPLICATION_SUPPORT_GLOBS,
            expect_dir=True,
        ),
        antigravity_roots=_discover_paths(
            input_roots,
            direct_match=_is_antigravity_root,
            glob_patterns=("antigravity", ".gemini/antigravity", "**/.gemini/antigravity"),
            expect_dir=True,
        ),
        conversation_paths=_discover_paths(
            input_roots,
            direct_match=_is_conversation_blob,
            glob_patterns=CONVERSATION_GLOBS,
            expect_dir=False,
        ),
        brain_dirs=_discover_paths(
            input_roots,
            direct_match=_is_brain_dir,
            glob_patterns=BRAIN_GLOBS,
            expect_dir=True,
        ),
        annotation_paths=_discover_paths(
            input_roots,
            direct_match=_is_annotation_path,
            glob_patterns=ANNOTATION_GLOBS,
            expect_dir=False,
        ),
        browser_recording_dirs=_discover_paths(
            input_roots,
            direct_match=_is_browser_recording_dir,
            glob_patterns=BROWSER_RECORDING_GLOBS,
            expect_dir=True,
        ),
        global_state_paths=_discover_paths(
            input_roots,
            direct_match=_is_global_state_path,
            glob_patterns=GLOBAL_STATE_GLOBS,
            expect_dir=False,
        ),
        workspace_state_paths=_discover_paths(
            input_roots,
            direct_match=_is_workspace_state_path,
            glob_patterns=WORKSPACE_STATE_GLOBS,
            expect_dir=False,
        ),
        log_roots=_discover_paths(
            input_roots,
            direct_match=_is_log_root,
            glob_patterns=LOG_ROOT_GLOBS,
            expect_dir=True,
        ),
        html_artifact_roots=_discover_paths(
            input_roots,
            direct_match=_is_html_artifact_root,
            glob_patterns=HTML_ARTIFACT_GLOBS,
            expect_dir=True,
        ),
        daemon_artifact_paths=_discover_paths(
            input_roots,
            direct_match=_is_daemon_artifact_path,
            glob_patterns=DAEMON_GLOBS,
            expect_dir=False,
        ),
    )


def parse_conversation_blob(
    conversation_path: Path,
    *,
    collected_at: str | None = None,
    artifacts: AntigravityArtifacts | None = None,
) -> NormalizedConversation | None:
    resolved_path = conversation_path.expanduser().resolve(strict=False)
    if not resolved_path.is_file():
        return None

    session_id = _session_id_from_path(resolved_path)
    if session_id is None:
        return None

    artifact_view = artifacts or AntigravityArtifacts()
    transcript = _recover_conversation_transcript(resolved_path, session_id=session_id)
    brain_metadata = _brain_metadata(_match_session_path(artifact_view.brain_dirs, session_id))
    annotation_metadata = _annotation_metadata(
        _match_session_path(artifact_view.annotation_paths, session_id)
    )
    browser_recording_metadata = _browser_recording_metadata(
        _match_session_path(artifact_view.browser_recording_dirs, session_id)
    )
    shared_state_metadata, cwd = _shared_state_metadata(artifact_view, session_id)

    conversation_blob_metadata: dict[str, object] = {
        "path": str(resolved_path),
        "size_bytes": resolved_path.stat().st_size,
        "decode_status": transcript.decode_status,
    }
    if transcript.decode_error is not None:
        conversation_blob_metadata["decode_error"] = transcript.decode_error
    if transcript.diagnostic_reason is not None:
        conversation_blob_metadata["diagnostic_reason"] = transcript.diagnostic_reason
    if transcript.diagnostic_details is not None:
        conversation_blob_metadata["diagnostic_details"] = transcript.diagnostic_details
    if transcript.protobuf_session_id is not None:
        conversation_blob_metadata["protobuf_session_id"] = transcript.protobuf_session_id
    if transcript.messages:
        conversation_blob_metadata["confirmed_field_mapping"] = {
            "session_id_field": CONVERSATION_SESSION_ID_FIELD,
            "message_field": CONVERSATION_MESSAGE_FIELD,
            "message_id_field": MESSAGE_ID_FIELD,
            "message_role_field": MESSAGE_ROLE_FIELD,
            "message_text_field": MESSAGE_TEXT_FIELD,
            "message_timestamp_field": MESSAGE_TIMESTAMP_FIELD,
        }
        conversation_blob_metadata["recovered_message_count"] = transcript.decoded_message_count
        conversation_blob_metadata["user_message_count"] = transcript.user_message_count
        conversation_blob_metadata["assistant_message_count"] = (
            transcript.assistant_message_count
        )
    if transcript.skipped_message_count:
        conversation_blob_metadata["skipped_message_count"] = transcript.skipped_message_count

    session_metadata: dict[str, object] = {
        "conversation_blob": conversation_blob_metadata,
        "noise_separation": {
            "excluded_from_messages": list(NOISE_EXCLUSIONS),
            "html_artifact_root_count": len(artifact_view.html_artifact_roots),
            "daemon_artifact_count": len(artifact_view.daemon_artifact_paths),
        },
    }
    if brain_metadata is not None:
        session_metadata["brain"] = brain_metadata
    if annotation_metadata is not None:
        session_metadata["annotation"] = annotation_metadata
    if browser_recording_metadata is not None:
        session_metadata["browser_recording"] = browser_recording_metadata
    if shared_state_metadata is not None:
        session_metadata["shared_state"] = shared_state_metadata

    return NormalizedConversation(
        source=ANTIGRAVITY_EDITOR_VIEW_DESCRIPTOR.key,
        execution_context=ANTIGRAVITY_EDITOR_VIEW_DESCRIPTOR.execution_context,
        collected_at=collected_at or utc_timestamp(),
        messages=transcript.messages,
        transcript_completeness=transcript.completeness,
        limitations=transcript.limitations,
        source_session_id=session_id,
        source_artifact_path=str(resolved_path),
        session_metadata=session_metadata,
        provenance=ConversationProvenance(
            session_started_at=transcript.session_started_at,
            source="antigravity",
            originator="antigravity_editor_view",
            cwd=cwd,
            app_shell=artifact_view.build_app_shell(),
        ),
    )


def _recover_conversation_transcript(
    conversation_path: Path,
    *,
    session_id: str,
) -> AntigravityTranscriptRecovery:
    try:
        payload = conversation_path.read_bytes()
    except FileNotFoundError:
        return AntigravityTranscriptRecovery(
            messages=(),
            completeness=TranscriptCompleteness.UNSUPPORTED,
            limitations=(DECODE_FAILED_LIMITATION, METADATA_ONLY_LIMITATION),
            decode_status="decode_failed",
            decode_error="decode_failed: conversation blob disappeared before parsing",
            diagnostic_reason=DECODE_FAILED_LIMITATION,
            diagnostic_details={"reason": "file_missing"},
        )

    try:
        fields = _decode_protobuf_fields(payload)
    except ProtobufDecodeError as exc:
        diagnostic_reason, decode_error, diagnostic_details = (
            _diagnose_top_level_decode_failure(exc)
        )
        limitations = (
            (diagnostic_reason, DECODE_FAILED_LIMITATION, METADATA_ONLY_LIMITATION)
            if diagnostic_reason != DECODE_FAILED_LIMITATION
            else (DECODE_FAILED_LIMITATION, METADATA_ONLY_LIMITATION)
        )
        return AntigravityTranscriptRecovery(
            messages=(),
            completeness=TranscriptCompleteness.UNSUPPORTED,
            limitations=_unique_limitations(limitations),
            decode_status="decode_failed",
            decode_error=decode_error,
            diagnostic_reason=diagnostic_reason,
            diagnostic_details=diagnostic_details,
        )

    protobuf_session_id = _first_uuid_string(fields, CONVERSATION_SESSION_ID_FIELD)
    messages: list[NormalizedMessage] = []
    skipped_message_count = 0
    user_message_count = 0
    assistant_message_count = 0

    for field in fields:
        if field.number != CONVERSATION_MESSAGE_FIELD or not isinstance(field.value, bytes):
            continue
        decoded_message = _decode_confirmed_message(field.value)
        if decoded_message is None:
            skipped_message_count += 1
            continue
        messages.append(decoded_message)
        if decoded_message.role == MessageRole.USER:
            user_message_count += 1
        if decoded_message.role == MessageRole.ASSISTANT:
            assistant_message_count += 1

    limitations: list[str] = []
    if protobuf_session_id is not None and protobuf_session_id != session_id:
        limitations.append(SESSION_ID_MISMATCH_LIMITATION)

    if not messages:
        return AntigravityTranscriptRecovery(
            messages=(),
            completeness=TranscriptCompleteness.UNSUPPORTED,
            limitations=_unique_limitations(
                (
                    VARIANT_UNKNOWN_LIMITATION,
                    METADATA_ONLY_LIMITATION,
                    *limitations,
                )
            ),
            protobuf_session_id=protobuf_session_id,
            decode_status="variant_unknown",
            decode_error=(
                "variant_unknown: top-level protobuf decoded but the message field "
                "mapping did not match the confirmed transcript-bearing variant"
            ),
            skipped_message_count=skipped_message_count,
            diagnostic_reason=VARIANT_UNKNOWN_LIMITATION,
            diagnostic_details={
                "reason": "message_field_mapping_unconfirmed",
                "scope": "conversation",
                "skipped_message_count": skipped_message_count,
            },
        )

    diagnostic_reason: str | None = None
    diagnostic_details: dict[str, object] | None = None
    if skipped_message_count:
        limitations.append(VARIANT_UNKNOWN_LIMITATION)
        diagnostic_reason = VARIANT_UNKNOWN_LIMITATION
        diagnostic_details = {
            "reason": "message_field_mapping_unconfirmed",
            "scope": "message_entry",
            "skipped_message_count": skipped_message_count,
        }
    if user_message_count == 0:
        limitations.append(MISSING_USER_LIMITATION)
    if assistant_message_count == 0:
        limitations.append(MISSING_ASSISTANT_LIMITATION)

    completeness = (
        TranscriptCompleteness.COMPLETE
        if not limitations
        else TranscriptCompleteness.PARTIAL
    )
    return AntigravityTranscriptRecovery(
        messages=tuple(messages),
        completeness=completeness,
        limitations=_unique_limitations(tuple(limitations)),
        protobuf_session_id=protobuf_session_id,
        session_started_at=next(
            (message.timestamp for message in messages if message.timestamp is not None),
            None,
        ),
        decode_status=(
            "decoded" if completeness == TranscriptCompleteness.COMPLETE else "partially_decoded"
        ),
        decoded_message_count=len(messages),
        skipped_message_count=skipped_message_count,
        user_message_count=user_message_count,
        assistant_message_count=assistant_message_count,
        diagnostic_reason=diagnostic_reason,
        diagnostic_details=diagnostic_details,
    )


def _decode_confirmed_message(payload: bytes) -> NormalizedMessage | None:
    try:
        fields = _decode_protobuf_fields(payload)
    except ProtobufDecodeError:
        return None

    role = _decode_message_role(fields)
    text = _first_string(fields, MESSAGE_TEXT_FIELD, strip=True)
    if role is None or text is None:
        return None

    return NormalizedMessage(
        role=role,
        text=text,
        timestamp=_first_string(fields, MESSAGE_TIMESTAMP_FIELD),
        source_message_id=_first_string(fields, MESSAGE_ID_FIELD),
    )


def _decode_message_role(fields: tuple[ProtobufField, ...]) -> MessageRole | None:
    for field in fields:
        if field.number != MESSAGE_ROLE_FIELD:
            continue
        if isinstance(field.value, int):
            return ROLE_ENUM_MAP.get(field.value)
        if not isinstance(field.value, bytes):
            continue
        try:
            decoded = field.value.decode("utf-8").strip().lower()
        except UnicodeDecodeError:
            continue
        if decoded == "user":
            return MessageRole.USER
        if decoded == "assistant":
            return MessageRole.ASSISTANT
    return None


def _decode_protobuf_fields(payload: bytes) -> tuple[ProtobufField, ...]:
    fields: list[ProtobufField] = []
    offset = 0

    while offset < len(payload):
        tag_offset = offset
        tag, offset = _read_varint(payload, offset)
        if tag <= 0:
            raise ProtobufDecodeError(
                "invalid protobuf tag",
                offset=tag_offset,
                stage="tag",
            )
        field_number = tag >> 3
        wire_type = tag & 0x07

        if wire_type == 0:
            value, offset = _read_varint(payload, offset)
        elif wire_type == 1:
            end = offset + 8
            if end > len(payload):
                raise ProtobufDecodeError(
                    "truncated protobuf 64-bit field",
                    offset=offset,
                    stage="field",
                    field_number=field_number,
                    wire_type=wire_type,
                )
            value = payload[offset:end]
            offset = end
        elif wire_type == 2:
            size, offset = _read_varint(payload, offset)
            end = offset + size
            if end > len(payload):
                raise ProtobufDecodeError(
                    "truncated protobuf length-delimited field",
                    offset=offset,
                    stage="field",
                    field_number=field_number,
                    wire_type=wire_type,
                )
            value = payload[offset:end]
            offset = end
        elif wire_type == 5:
            end = offset + 4
            if end > len(payload):
                raise ProtobufDecodeError(
                    "truncated protobuf 32-bit field",
                    offset=offset,
                    stage="field",
                    field_number=field_number,
                    wire_type=wire_type,
                )
            value = payload[offset:end]
            offset = end
        else:
            raise ProtobufDecodeError(
                f"unsupported protobuf wire type {wire_type}",
                offset=tag_offset,
                stage="field",
                field_number=field_number,
                wire_type=wire_type,
            )

        fields.append(ProtobufField(number=field_number, wire_type=wire_type, value=value))

    return tuple(fields)


def _read_varint(payload: bytes, offset: int) -> tuple[int, int]:
    start_offset = offset
    result = 0
    shift = 0

    while True:
        if offset >= len(payload):
            raise ProtobufDecodeError(
                "truncated protobuf varint",
                offset=start_offset,
                stage="varint",
            )
        byte = payload[offset]
        offset += 1
        result |= (byte & 0x7F) << shift
        if byte < 0x80:
            return result, offset
        shift += 7
        if shift >= 64:
            raise ProtobufDecodeError(
                "protobuf varint exceeds 64 bits",
                offset=start_offset,
                stage="varint",
            )


def _diagnose_top_level_decode_failure(
    exc: ProtobufDecodeError,
) -> tuple[str, str, dict[str, object]]:
    diagnostic_details: dict[str, object] = {
        "failure_offset": exc.offset,
        "failure_stage": exc.stage,
    }
    if exc.field_number is not None:
        diagnostic_details["field_number"] = exc.field_number
    if exc.wire_type is not None:
        diagnostic_details["wire_type"] = exc.wire_type

    if exc.offset == 0 and exc.wire_type in UNKNOWN_VARIANT_WIRE_TYPES:
        diagnostic_details["reason"] = "unknown_top_level_wire_type"
        return (
            VARIANT_UNKNOWN_LIMITATION,
            (
                "variant_unknown: top-level wire type "
                f"{exc.wire_type} does not match the confirmed Antigravity raw "
                "protobuf conversation variant"
            ),
            diagnostic_details,
        )

    diagnostic_details["reason"] = "protobuf_decode_failed"
    return (
        DECODE_FAILED_LIMITATION,
        f"decode_failed: {exc}",
        diagnostic_details,
    )


def _first_string(
    fields: tuple[ProtobufField, ...],
    field_number: int,
    *,
    strip: bool = False,
) -> str | None:
    for field in fields:
        if field.number != field_number or not isinstance(field.value, bytes):
            continue
        try:
            value = field.value.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if strip:
            value = value.strip()
        if value:
            return value
    return None


def _first_uuid_string(
    fields: tuple[ProtobufField, ...],
    field_number: int,
) -> str | None:
    value = _first_string(fields, field_number)
    if value is None or not UUID_PATTERN.match(value):
        return None
    return value.lower()


def _unique_limitations(limitations: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(limitations))


def _brain_metadata(brain_dir_path: str | None) -> dict[str, object] | None:
    if brain_dir_path is None:
        return None

    brain_dir = Path(brain_dir_path)
    artifact_summaries: list[dict[str, object]] = []
    for metadata_path in sorted(brain_dir.glob("*.metadata.json")):
        payload = _load_json_file(metadata_path)
        summary: dict[str, object] = {
            "name": metadata_path.name.removesuffix(".metadata.json"),
        }
        artifact_type = _string_value(payload.get("artifactType")) if payload else None
        if artifact_type is not None:
            summary["artifact_type"] = artifact_type
        updated_at = _string_value(payload.get("updatedAt")) if payload else None
        if updated_at is not None:
            summary["updated_at"] = updated_at
        version = payload.get("version") if payload else None
        if isinstance(version, int):
            summary["version"] = version
        if payload and "summary" in payload:
            summary["has_summary"] = True
        artifact_summaries.append(summary)

    artifact_names = tuple(
        sorted(path.name for path in brain_dir.glob("*.md") if path.is_file())
    )
    resolved_artifact_count = sum(1 for path in brain_dir.glob("*.resolved") if path.is_file())
    image_artifact_count = sum(
        1
        for path in brain_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )

    payload: dict[str, object] = {"path": str(brain_dir)}
    if artifact_names:
        payload["artifact_names"] = list(artifact_names)
    if artifact_summaries:
        payload["artifact_summaries"] = artifact_summaries
    if resolved_artifact_count:
        payload["resolved_artifact_count"] = resolved_artifact_count
    if image_artifact_count:
        payload["image_artifact_count"] = image_artifact_count
    return payload


def _annotation_metadata(annotation_path: str | None) -> dict[str, object] | None:
    if annotation_path is None:
        return None

    fields = _parse_pbtxt_fields(Path(annotation_path))
    payload: dict[str, object] = {"path": annotation_path}
    if fields:
        payload["fields"] = fields
    return payload


def _browser_recording_metadata(browser_recording_dir: str | None) -> dict[str, object] | None:
    if browser_recording_dir is None:
        return None

    recording_dir = Path(browser_recording_dir)
    frame_count = sum(
        1
        for path in recording_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    return {
        "path": str(recording_dir),
        "frame_count": frame_count,
    }


def _shared_state_metadata(
    artifacts: AntigravityArtifacts,
    session_id: str,
) -> tuple[dict[str, object] | None, str | None]:
    if not artifacts.global_state_paths and not artifacts.workspace_state_paths:
        return None, None

    global_matches: list[dict[str, object]] = []
    for global_state_path in artifacts.global_state_paths:
        state_values = _read_state_values(Path(global_state_path), GLOBAL_STATE_KEYS)
        matched_keys = sorted(
            key
            for key, value in state_values.items()
            if _value_mentions_session(value, session_id)
        )
        global_matches.append(
            {
                "state_db_path": global_state_path,
                "matched_keys": matched_keys,
            }
        )

    workspace_entries: list[dict[str, object]] = []
    cwd: str | None = None
    for workspace_state_path in artifacts.workspace_state_paths:
        state_values = _read_state_values(Path(workspace_state_path), WORKSPACE_STATE_KEYS)
        workspace_folder = _read_workspace_folder(
            Path(workspace_state_path).parent / "workspace.json"
        )
        matched_keys = sorted(
            key
            for key, value in state_values.items()
            if key != "chat.ChatSessionStore.index"
            and _value_mentions_session(value, session_id)
        )
        if cwd is None and matched_keys and workspace_folder is not None:
            cwd = workspace_folder

        workspace_entry: dict[str, object] = {
            "state_db_path": workspace_state_path,
            "workspace_id": Path(workspace_state_path).parent.name,
            "matched_keys": matched_keys,
        }
        if workspace_folder is not None:
            workspace_entry["workspace_folder"] = workspace_folder

        index_version = _chat_session_store_index_version(
            state_values.get("chat.ChatSessionStore.index")
        )
        if index_version is not None:
            workspace_entry["chat_session_store_index_version"] = index_version

        entry_count = _chat_session_store_entry_count(
            state_values.get("chat.ChatSessionStore.index")
        )
        if entry_count is not None:
            workspace_entry["chat_session_store_entry_count"] = entry_count
        workspace_entries.append(workspace_entry)

    return (
        {
            "global_state": global_matches,
            "workspace_state": workspace_entries,
        },
        cwd,
    )


def _chat_session_store_index_version(payload: object) -> int | None:
    if not isinstance(payload, dict):
        return None
    version = payload.get("version")
    if isinstance(version, int):
        return version
    return None


def _chat_session_store_entry_count(payload: object) -> int | None:
    if not isinstance(payload, dict):
        return None
    entries = payload.get("entries")
    if not isinstance(entries, dict):
        return None
    return len(entries)


def _parse_pbtxt_fields(annotation_path: Path) -> dict[str, object]:
    try:
        lines = annotation_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return {}

    fields: dict[str, object] = {}
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        field_name = key.strip()
        field_value = raw_value.strip()
        if not field_name:
            continue
        if field_value.startswith('"') and field_value.endswith('"'):
            fields[field_name] = field_value[1:-1]
            continue
        if field_value in {"true", "false"}:
            fields[field_name] = field_value == "true"
            continue
        try:
            fields[field_name] = int(field_value)
        except ValueError:
            fields[field_name] = field_value
    return fields


def _discover_paths(
    input_roots: tuple[Path, ...],
    *,
    direct_match,
    glob_patterns: tuple[str, ...],
    expect_dir: bool,
) -> tuple[str, ...]:
    seen: set[Path] = set()
    candidates: list[str] = []

    for input_root in input_roots:
        matches: list[Path] = []
        if direct_match(input_root):
            matches.append(input_root)
        if input_root.is_dir():
            for pattern in glob_patterns:
                matches.extend(input_root.glob(pattern))

        for candidate in matches:
            if expect_dir and not candidate.is_dir():
                continue
            if not expect_dir and not candidate.is_file():
                continue

            resolved = candidate.resolve(strict=False)
            if resolved in seen:
                continue
            seen.add(resolved)
            candidates.append(str(resolved))

    return tuple(sorted(candidates))


def _read_state_values(
    state_db_path: Path,
    keys: tuple[str, ...],
) -> dict[str, object]:
    if not state_db_path.is_file():
        return {}

    try:
        with sqlite3.connect(str(state_db_path)) as connection:
            rows = connection.execute(
                "SELECT key, value FROM ItemTable WHERE key IN ({})".format(
                    ",".join("?" for _ in keys)
                ),
                keys,
            ).fetchall()
    except sqlite3.DatabaseError:
        return {}

    payload: dict[str, object] = {}
    for key, raw_value in rows:
        if not isinstance(key, str) or not isinstance(raw_value, str):
            continue
        try:
            payload[key] = json.loads(raw_value)
        except json.JSONDecodeError:
            payload[key] = raw_value
    return payload


def _read_workspace_folder(workspace_json_path: Path) -> str | None:
    payload = _load_json_file(workspace_json_path)
    if payload is None:
        return None

    folder_uri = _string_value(payload.get("folder"))
    if folder_uri is None:
        return None
    if folder_uri.startswith("file://"):
        parsed = urlparse(folder_uri)
        return unquote(parsed.path) or None
    return folder_uri


def _load_json_file(path: Path) -> dict[str, object] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _value_mentions_session(value: object, session_id: str) -> bool:
    if isinstance(value, str):
        return session_id in value
    try:
        serialized = json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return False
    return session_id in serialized


def _match_session_path(paths: tuple[str, ...], session_id: str) -> str | None:
    for raw_path in paths:
        candidate = Path(raw_path)
        candidate_session_id = _session_id_from_path(candidate)
        if candidate_session_id == session_id:
            return str(candidate)
    return None


def _session_id_from_path(path: Path) -> str | None:
    candidate = path.stem if path.is_file() else path.name
    if UUID_PATTERN.match(candidate):
        return candidate.lower()
    return None


def _is_antigravity_root(path: Path) -> bool:
    return path.name == "antigravity" and ".gemini" in path.parts


def _is_application_support_root(path: Path) -> bool:
    return path.name == "Antigravity" and "Application Support" in path.parts


def _is_log_root(path: Path) -> bool:
    return path.name == "logs" and "Antigravity" in path.parts


def _is_html_artifact_root(path: Path) -> bool:
    return path.name == "html_artifacts" and "antigravity" in path.parts


def _is_daemon_artifact_path(path: Path) -> bool:
    return path.parent.name == "daemon" and "antigravity" in path.parts


def _is_conversation_blob(path: Path) -> bool:
    return path.suffix == ".pb" and path.parent.name == "conversations"


def _is_brain_dir(path: Path) -> bool:
    return path.parent.name == "brain" and _session_id_from_path(path) is not None


def _is_annotation_path(path: Path) -> bool:
    return path.suffix == ".pbtxt" and path.parent.name == "annotations"


def _is_browser_recording_dir(path: Path) -> bool:
    return path.parent.name == "browser_recordings" and _session_id_from_path(path) is not None


def _is_global_state_path(path: Path) -> bool:
    return path.name == "state.vscdb" and path.parent.name == "globalStorage"


def _is_workspace_state_path(path: Path) -> bool:
    return path.name == "state.vscdb" and path.parent.parent.name == "workspaceStorage"


def _string_value(value: object) -> str | None:
    if isinstance(value, str):
        return value
    return None


__all__ = [
    "ANTIGRAVITY_EDITOR_VIEW_DESCRIPTOR",
    "AntigravityArtifacts",
    "AntigravityEditorViewCollector",
    "discover_antigravity_editor_view_artifacts",
    "parse_conversation_blob",
]
