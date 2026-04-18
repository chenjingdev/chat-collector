from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import chain
from pathlib import Path
from typing import Iterable
from urllib.parse import unquote, urlparse

from ..incremental import write_incremental_collection
from ..models import (
    AppShellProvenance,
    CollectionPlan,
    CollectionResult,
    ConversationProvenance,
    NormalizedConversation,
    NormalizedMessage,
    MessageRole,
    SourceDescriptor,
    SourceSupportMetadata,
    SupportLevel,
    TranscriptCompleteness,
)
from ..source_roots import (
    darwin_root,
    default_descriptor_input_roots,
    linux_root,
    windows_root,
)
from .codex_rollout import resolve_input_roots, utc_timestamp

GEMINI_CODE_ASSIST_IDE_DESCRIPTOR = SourceDescriptor(
    key="gemini_code_assist_ide",
    display_name="Gemini Code Assist IDE",
    execution_context="ide_extension",
    support_level=SupportLevel.PARTIAL,
    default_input_roots=(
        "~/Library/Application Support/google-vscode-extension",
        "~/Library/Application Support/cloud-code",
        "~/Library/Application Support/Code/User/globalStorage",
        "~/Library/Application Support/Code/User/workspaceStorage",
    ),
    artifact_root_candidates=(
        darwin_root("$HOME/Library/Application Support/google-vscode-extension"),
        linux_root("$XDG_CONFIG_HOME/google-vscode-extension"),
        windows_root("$APPDATA/google-vscode-extension"),
        darwin_root("$HOME/Library/Application Support/cloud-code"),
        linux_root("$XDG_CONFIG_HOME/cloud-code"),
        windows_root("$APPDATA/cloud-code"),
        darwin_root("$HOME/Library/Application Support/Code/User/globalStorage"),
        linux_root("$XDG_CONFIG_HOME/Code/User/globalStorage"),
        windows_root("$APPDATA/Code/User/globalStorage"),
        darwin_root("$HOME/Library/Application Support/Code/User/workspaceStorage"),
        linux_root("$XDG_CONFIG_HOME/Code/User/workspaceStorage"),
        windows_root("$APPDATA/Code/User/workspaceStorage"),
    ),
    notes=(
        "Reconstructs transcript messages from Gemini-owned VS Code chatSessions payloads when provider attribution is explicit and the body shape is recoverable.",
        "Treats google-vscode-extension auth files and cloud-code install residue as provenance only.",
        "Falls back to metadata-only IDE residue rows when Gemini transcript bodies cannot be confirmed from chatSessions payloads.",
        "Checks chatSessions provider attribution and emits explicit rejection diagnostics for non-Gemini transcript candidates.",
    ),
    support_metadata=SourceSupportMetadata(
        product_label="Gemini Code Assist",
        host_surface="IDE extension",
        expected_transcript_completeness=TranscriptCompleteness.PARTIAL,
        limitation_summary="Gemini-owned chatSessions with recoverable body shapes are promoted, but foreign providers and Gemini-owned body-missing sessions still degrade to explicit metadata-only residue, so the source stays partial and opt-in.",
        limitations=(
            "Only explicitly Gemini-owned chatSessions with recoverable request or response bodies are promoted to transcript rows.",
            "Foreign or unknown provider chatSessions, and Gemini-owned sessions without a confirmed body, remain metadata-only residue with explicit diagnostics.",
        ),
    ),
)

GOOGLE_VSCODE_EXTENSION_GLOB = "**/Application Support/google-vscode-extension"
CLOUD_CODE_GLOB = "**/Application Support/cloud-code"
GLOBAL_STATE_GLOB = "**/User/globalStorage/state.vscdb"
WORKSPACE_STATE_GLOB = "**/User/workspaceStorage/*/state.vscdb"
CREDENTIAL_FILE_NAMES = frozenset({"credentials.json", "application_default_credentials.json"})
GLOBAL_STATE_KEYS = (
    "google.geminicodeassist",
    "workbench.view.extension.geminiChat.state.hidden",
    "workbench.view.extension.geminiOutline.state.hidden",
)
WORKSPACE_STATE_KEYS = (
    "chat.ChatSessionStore.index",
    "workbench.view.extension.geminiChat.state",
    "workbench.view.extension.geminiOutline.state",
    "memento/webviewView.cloudcode.gemini.chatView",
    "workbench.view.extension.geminiChat.numberOfVisibleViews",
)
NO_CONFIRMED_TRANSCRIPT_LIMITATION = "no_confirmed_gemini_code_assist_ide_transcript_store"
METADATA_ONLY_IDE_STATE_LIMITATION = "metadata_only_ide_state"
CHAT_SESSION_PROVIDER_ATTRIBUTION_REQUIRED_LIMITATION = (
    "chat_session_provider_attribution_required"
)
FOREIGN_PROVIDER_REJECTED_LIMITATION = "foreign_chat_session_rejected"
UNKNOWN_PROVIDER_REJECTED_LIMITATION = "unknown_chat_session_rejected"
GEMINI_BODY_MISSING_LIMITATION = "gemini_owned_chat_session_body_missing"
UNSUPPORTED_LIMITATIONS = (
    NO_CONFIRMED_TRANSCRIPT_LIMITATION,
    METADATA_ONLY_IDE_STATE_LIMITATION,
    CHAT_SESSION_PROVIDER_ATTRIBUTION_REQUIRED_LIMITATION,
)
GLOBAL_RESIDUE_CONVERSATION_ORIGIN = "global_state_residue"
WORKSPACE_RESIDUE_CONVERSATION_ORIGIN = "workspace_chat_session_residue"
GEMINI_PROVIDER_MARKERS = (
    "gemini",
    "google.geminicodeassist",
    "cloudcode.gemini",
)
FOREIGN_PROVIDER_MARKERS = (
    "copilot",
    "claude",
    "cursor",
    "openai.chatgpt",
)
REQUEST_BODY_FIELDS = ("message", "text", "prompt", "query", "content", "parts")
REQUEST_NESTED_TEXT_KEYS = frozenset(
    {"message", "prompt", "query", "content", "parts", "segments", "blocks", "fragments"}
)
REQUEST_TIMESTAMP_PATHS = (
    ("timestamp",),
    ("requestTimestamp",),
    ("createdAt",),
    ("requestDate",),
    ("lastUpdatedAt",),
)
RESPONSE_BODY_FIELDS = (
    "response",
    "responseMessage",
    "responseText",
    "responseParts",
    "reply",
    "answer",
)
RESPONSE_NESTED_TEXT_KEYS = frozenset(
    {
        "message",
        "content",
        "parts",
        "response",
        "responseMessage",
        "responseText",
        "responseParts",
        "reply",
        "answer",
        "candidate",
        "candidates",
        "segments",
        "blocks",
        "fragments",
    }
)
RESPONSE_TIMESTAMP_PATHS = (
    ("response", "timestamp"),
    ("response", "createdAt"),
    ("response", "updatedAt"),
    ("responseTimestamp",),
    ("responseDate",),
    ("lastResponseDate",),
)
SESSION_STARTED_AT_PATHS = (
    ("createdAt",),
    ("creationDate",),
    ("timestamp",),
    ("lastMessageDate",),
)
MISSING_USER_MESSAGE_LIMITATION = "user_message_missing_from_gemini_chat_session"
MISSING_ASSISTANT_MESSAGE_LIMITATION = "assistant_message_missing_from_gemini_chat_session"


@dataclass(frozen=True, slots=True)
class GeminiChatSessionAttribution:
    session_id: str
    ownership: str
    provider: str | None = None
    source_path: str | None = None
    request_count: int | None = None
    is_empty: bool | None = None
    provider_candidates: tuple[str, ...] = ()
    ownership_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "session_id": self.session_id,
            "ownership": self.ownership,
        }
        if self.provider is not None:
            payload["provider"] = self.provider
        if self.source_path is not None:
            payload["source_path"] = self.source_path
        if self.request_count is not None:
            payload["request_count"] = self.request_count
        if self.is_empty is not None:
            payload["is_empty"] = self.is_empty
        if self.provider_candidates:
            payload["provider_candidates"] = list(self.provider_candidates)
        if self.ownership_reason is not None:
            payload["ownership_reason"] = self.ownership_reason
        return payload


@dataclass(frozen=True, slots=True)
class GeminiIdeArtifacts:
    application_support_roots: tuple[str, ...] = ()
    global_state_paths: tuple[str, ...] = ()
    workspace_state_paths: tuple[str, ...] = ()
    credential_artifact_count: int = 0
    install_artifact_count: int = 0

    def build_app_shell(self, *, state_db_path: str) -> AppShellProvenance | None:
        provenance = AppShellProvenance(
            application_support_roots=self.application_support_roots,
            state_db_paths=tuple(sorted({state_db_path, *self.global_state_paths})),
        )
        if not provenance.to_dict():
            return None
        return provenance


@dataclass(frozen=True, slots=True)
class GeminiChatSessionTranscript:
    messages: tuple[NormalizedMessage, ...]
    completeness: TranscriptCompleteness
    limitations: tuple[str, ...] = ()
    session_started_at: str | None = None
    user_message_count: int = 0
    assistant_message_count: int = 0


@dataclass(frozen=True, slots=True)
class GeminiCodeAssistIdeCollector:
    descriptor: SourceDescriptor = GEMINI_CODE_ASSIST_IDE_DESCRIPTOR

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
        artifacts = discover_gemini_code_assist_ide_artifacts(resolved_input_roots)
        collected_at = utc_timestamp()
        scanned_artifact_count = len(artifacts.global_state_paths) + len(
            artifacts.workspace_state_paths
        )
        conversations = chain(
            (
                parse_global_state(
                    Path(global_state_path),
                    collected_at=collected_at,
                    artifacts=artifacts,
                )
                for global_state_path in artifacts.global_state_paths
            ),
            chain.from_iterable(
                parse_workspace_state_rows(
                    Path(workspace_state_path),
                    collected_at=collected_at,
                    artifacts=artifacts,
                )
                for workspace_state_path in artifacts.workspace_state_paths
            ),
        )
        return write_incremental_collection(
            source=self.descriptor.key,
            archive_root=archive_root,
            input_roots=resolved_input_roots,
            scanned_artifact_count=scanned_artifact_count,
            collected_at=collected_at,
            conversations=conversations,
        )

    def _default_input_roots(self) -> tuple[Path, ...]:
        return default_descriptor_input_roots(self.descriptor)


def discover_gemini_code_assist_ide_artifacts(
    input_roots: tuple[Path, ...] | None,
) -> GeminiIdeArtifacts:
    if not input_roots:
        return GeminiIdeArtifacts()

    google_extension_roots = _discover_paths(
        input_roots,
        direct_match=_is_google_vscode_extension_root,
        glob_pattern=GOOGLE_VSCODE_EXTENSION_GLOB,
        expect_dir=True,
    )
    cloud_code_roots = _discover_paths(
        input_roots,
        direct_match=_is_cloud_code_root,
        glob_pattern=CLOUD_CODE_GLOB,
        expect_dir=True,
    )

    return GeminiIdeArtifacts(
        application_support_roots=tuple(sorted({*google_extension_roots, *cloud_code_roots})),
        global_state_paths=_discover_paths(
            input_roots,
            direct_match=_is_global_state_db,
            glob_pattern=GLOBAL_STATE_GLOB,
            expect_dir=False,
        ),
        workspace_state_paths=_discover_paths(
            input_roots,
            direct_match=_is_workspace_state_db,
            glob_pattern=WORKSPACE_STATE_GLOB,
            expect_dir=False,
        ),
        credential_artifact_count=_count_credential_artifacts(google_extension_roots),
        install_artifact_count=_count_install_artifacts(cloud_code_roots),
    )


def parse_global_state(
    state_db_path: Path,
    *,
    collected_at: str | None = None,
    artifacts: GeminiIdeArtifacts | None = None,
) -> NormalizedConversation | None:
    resolved_path = state_db_path.expanduser().resolve(strict=False)
    state_values = _read_state_values(resolved_path, GLOBAL_STATE_KEYS)
    gemini_state = state_values.get("google.geminicodeassist")
    chat_view_hidden = _hidden_view_state(
        state_values.get("workbench.view.extension.geminiChat.state.hidden")
    )
    outline_view_hidden = _hidden_view_state(
        state_values.get("workbench.view.extension.geminiOutline.state.hidden")
    )

    if not isinstance(gemini_state, dict) and chat_view_hidden is None and outline_view_hidden is None:
        return None

    return NormalizedConversation(
        source=GEMINI_CODE_ASSIST_IDE_DESCRIPTOR.key,
        execution_context=GEMINI_CODE_ASSIST_IDE_DESCRIPTOR.execution_context,
        collected_at=collected_at or utc_timestamp(),
        messages=(),
        transcript_completeness=TranscriptCompleteness.UNSUPPORTED,
        limitations=UNSUPPORTED_LIMITATIONS,
        source_session_id="vscode:global",
        source_artifact_path=str(resolved_path),
        session_metadata=_build_global_state_metadata(
            gemini_state=gemini_state if isinstance(gemini_state, dict) else None,
            chat_view_hidden=chat_view_hidden,
            outline_view_hidden=outline_view_hidden,
            artifacts=artifacts,
        ),
        provenance=ConversationProvenance(
            source="vscode",
            originator="google.geminicodeassist",
            conversation_origin=GLOBAL_RESIDUE_CONVERSATION_ORIGIN,
            app_shell=(
                artifacts.build_app_shell(state_db_path=str(resolved_path))
                if artifacts is not None
                else None
            ),
        ),
    )


def parse_workspace_state(
    state_db_path: Path,
    *,
    collected_at: str | None = None,
    artifacts: GeminiIdeArtifacts | None = None,
) -> NormalizedConversation | None:
    rows = parse_workspace_state_rows(
        state_db_path,
        collected_at=collected_at,
        artifacts=artifacts,
    )
    if not rows:
        return None
    return rows[0]


def parse_workspace_state_rows(
    state_db_path: Path,
    *,
    collected_at: str | None = None,
    artifacts: GeminiIdeArtifacts | None = None,
) -> tuple[NormalizedConversation, ...]:
    resolved_path = state_db_path.expanduser().resolve(strict=False)
    state_values = _read_state_values(resolved_path, WORKSPACE_STATE_KEYS)
    chat_view_state = _extract_view_state(
        state_values.get("workbench.view.extension.geminiChat.state"),
        "cloudcode.gemini.chatView",
    )
    outline_view_state = _extract_view_state(
        state_values.get("workbench.view.extension.geminiOutline.state"),
        "cloudcode.gemini.outlineView",
    )
    memento_payload = state_values.get("memento/webviewView.cloudcode.gemini.chatView")
    visible_views = _int_value(state_values.get("workbench.view.extension.geminiChat.numberOfVisibleViews"))
    chat_session_index = state_values.get("chat.ChatSessionStore.index")

    if (
        chat_view_state is None
        and outline_view_state is None
        and not isinstance(memento_payload, dict)
        and visible_views is None
    ):
        return ()

    workspace_id = resolved_path.parent.name
    workspace_folder = _read_workspace_folder(resolved_path.parent / "workspace.json")
    attributions = _attribute_indexed_chat_sessions(resolved_path.parent, chat_session_index)
    workspace_metadata = _build_workspace_state_metadata(
        workspace_id=workspace_id,
        workspace_folder=workspace_folder,
        chat_view_state=chat_view_state,
        outline_view_state=outline_view_state,
        memento_payload=memento_payload,
        visible_views=visible_views,
        chat_session_index=chat_session_index,
        attributions=attributions,
        artifacts=artifacts,
    )
    workspace_rows = _build_workspace_chat_session_conversations(
        workspace_dir=resolved_path.parent,
        workspace_id=workspace_id,
        workspace_folder=workspace_folder,
        state_db_path=resolved_path,
        chat_session_index=chat_session_index,
        attributions=attributions,
        workspace_metadata=workspace_metadata,
        collected_at=collected_at,
        artifacts=artifacts,
    )
    return workspace_rows


def attribute_chat_session(chat_session_path: Path) -> GeminiChatSessionAttribution | None:
    resolved_path = chat_session_path.expanduser().resolve(strict=False)
    payload = _read_chat_session_payload(resolved_path)
    if payload is None:
        return None

    return _attribute_chat_session_payload(resolved_path, payload)


def parse_chat_session_transcript(
    chat_session_path: Path,
    *,
    attribution: GeminiChatSessionAttribution | None = None,
    collected_at: str | None = None,
    workspace_id: str | None = None,
    workspace_folder: str | None = None,
    state_db_path: Path | None = None,
    artifacts: GeminiIdeArtifacts | None = None,
    workspace_metadata: dict[str, object] | None = None,
) -> NormalizedConversation | None:
    resolved_path = chat_session_path.expanduser().resolve(strict=False)
    payload = _read_chat_session_payload(resolved_path)
    if payload is None:
        return None

    resolved_attribution = attribution or _attribute_chat_session_payload(resolved_path, payload)
    if resolved_attribution is None or resolved_attribution.ownership != "gemini":
        return None

    transcript = _recover_chat_session_transcript(payload)
    if transcript is None or not transcript.messages:
        return None

    return NormalizedConversation(
        source=GEMINI_CODE_ASSIST_IDE_DESCRIPTOR.key,
        execution_context=GEMINI_CODE_ASSIST_IDE_DESCRIPTOR.execution_context,
        collected_at=collected_at or utc_timestamp(),
        messages=transcript.messages,
        transcript_completeness=transcript.completeness,
        limitations=transcript.limitations,
        source_session_id=_compose_chat_session_source_id(
            workspace_id=workspace_id,
            session_id=resolved_attribution.session_id,
        ),
        source_artifact_path=str(resolved_path),
        session_metadata=_build_chat_session_metadata(
            attribution=resolved_attribution,
            transcript=transcript,
            workspace_id=workspace_id,
            workspace_folder=workspace_folder,
            workspace_metadata=workspace_metadata,
            artifacts=artifacts,
        ),
        provenance=ConversationProvenance(
            session_started_at=transcript.session_started_at,
            source="vscode",
            originator="google.geminicodeassist",
            cwd=workspace_folder,
            app_shell=(
                artifacts.build_app_shell(state_db_path=str(state_db_path))
                if artifacts is not None and state_db_path is not None
                else None
            ),
        ),
    )


def _attribute_chat_session_payload(
    resolved_path: Path,
    payload: dict[str, object],
) -> GeminiChatSessionAttribution:

    session_id = _string_value(payload.get("sessionId")) or resolved_path.stem
    request_count = _list_length(payload.get("requests"))
    is_empty = _bool_value(payload.get("isEmpty"))
    provider_candidates = tuple(candidate for candidate in _provider_candidates(payload) if candidate)
    ownership, provider, ownership_reason = _classify_provider(provider_candidates)

    return GeminiChatSessionAttribution(
        session_id=session_id,
        ownership=ownership,
        provider=provider,
        source_path=str(resolved_path),
        request_count=request_count,
        is_empty=is_empty,
        provider_candidates=provider_candidates,
        ownership_reason=ownership_reason,
    )


def _build_workspace_chat_session_conversations(
    *,
    workspace_dir: Path,
    workspace_id: str,
    workspace_folder: str | None,
    state_db_path: Path,
    chat_session_index: object,
    attributions: tuple[GeminiChatSessionAttribution, ...],
    workspace_metadata: dict[str, object],
    collected_at: str | None,
    artifacts: GeminiIdeArtifacts | None,
) -> tuple[NormalizedConversation, ...]:
    indexed_session_ids = _indexed_session_ids(chat_session_index)
    if not indexed_session_ids:
        return ()

    chat_sessions_dir = workspace_dir / "chatSessions"
    if not chat_sessions_dir.is_dir():
        return ()

    session_paths = {
        path.stem: path.resolve(strict=False)
        for path in chat_sessions_dir.rglob("*.json")
        if path.is_file()
    }
    attributions_by_id = {row.session_id: row for row in attributions}

    conversations: list[NormalizedConversation] = []
    residue_payloads: list[dict[str, object]] = []
    for session_id in indexed_session_ids:
        attribution = attributions_by_id.get(session_id)
        if attribution is None:
            continue
        session_path = session_paths.get(session_id)
        if session_path is None:
            continue
        if attribution.ownership != "gemini":
            residue_payload = _build_chat_session_residue_payload(attribution=attribution)
            if residue_payload is not None:
                residue_payloads.append(residue_payload)
            continue

        conversation = parse_chat_session_transcript(
            session_path,
            attribution=attribution,
            collected_at=collected_at,
            workspace_id=workspace_id,
            workspace_folder=workspace_folder,
            state_db_path=state_db_path,
            artifacts=artifacts,
            workspace_metadata=workspace_metadata,
        )
        if conversation is not None:
            conversations.append(conversation)
            continue
        residue_payloads.append(
            _build_chat_session_residue_payload(
                attribution=attribution,
                residue_kind="gemini_provider_explicit_but_body_missing",
                limitations=(GEMINI_BODY_MISSING_LIMITATION,),
            )
        )
    residue_conversation = _build_workspace_residue_conversation(
        workspace_id=workspace_id,
        workspace_folder=workspace_folder,
        state_db_path=state_db_path,
        workspace_metadata=workspace_metadata,
        residue_payloads=tuple(residue_payloads),
        transcript_rows=tuple(conversations),
        collected_at=collected_at,
        artifacts=artifacts,
    )
    if residue_conversation is not None:
        conversations.append(residue_conversation)

    return tuple(conversations)


def _recover_chat_session_transcript(
    payload: dict[str, object],
) -> GeminiChatSessionTranscript | None:
    raw_requests = payload.get("requests")
    if not isinstance(raw_requests, list):
        return None

    messages: list[NormalizedMessage] = []
    user_message_count = 0
    assistant_message_count = 0
    missing_user_count = 0
    missing_assistant_count = 0

    for index, raw_request in enumerate(raw_requests, start=1):
        if not isinstance(raw_request, dict):
            continue

        request_id = _chat_session_request_id(raw_request, index=index)
        user_text = _extract_text_from_named_fields(
            raw_request,
            REQUEST_BODY_FIELDS,
            nested_keys=REQUEST_NESTED_TEXT_KEYS,
        )
        user_timestamp = _first_normalized_timestamp(raw_request, REQUEST_TIMESTAMP_PATHS)
        if user_text is not None:
            user_message_count += 1
            messages.append(
                NormalizedMessage(
                    role=MessageRole.USER,
                    text=user_text,
                    timestamp=user_timestamp,
                    source_message_id=request_id,
                )
            )
        elif _has_named_field(raw_request, REQUEST_BODY_FIELDS):
            missing_user_count += 1

        assistant_text = _extract_text_from_named_fields(
            raw_request,
            RESPONSE_BODY_FIELDS,
            nested_keys=RESPONSE_NESTED_TEXT_KEYS,
        )
        assistant_timestamp = _first_normalized_timestamp(raw_request, RESPONSE_TIMESTAMP_PATHS)
        if assistant_text is not None:
            assistant_message_count += 1
            messages.append(
                NormalizedMessage(
                    role=MessageRole.ASSISTANT,
                    text=assistant_text,
                    timestamp=assistant_timestamp,
                    source_message_id=_chat_session_response_id(
                        raw_request,
                        request_id=request_id,
                        index=index,
                    ),
                )
            )
        elif user_text is not None or _has_named_field(raw_request, RESPONSE_BODY_FIELDS):
            missing_assistant_count += 1

    if not messages:
        return None

    limitations: list[str] = []
    if missing_user_count:
        limitations.append(MISSING_USER_MESSAGE_LIMITATION)
    if missing_assistant_count:
        limitations.append(MISSING_ASSISTANT_MESSAGE_LIMITATION)

    completeness = (
        TranscriptCompleteness.COMPLETE
        if assistant_message_count > 0 and not limitations
        else TranscriptCompleteness.PARTIAL
    )
    session_started_at = next(
        (message.timestamp for message in messages if message.timestamp is not None),
        _first_normalized_timestamp(payload, SESSION_STARTED_AT_PATHS),
    )
    return GeminiChatSessionTranscript(
        messages=tuple(messages),
        completeness=completeness,
        limitations=tuple(limitations),
        session_started_at=session_started_at,
        user_message_count=user_message_count,
        assistant_message_count=assistant_message_count,
    )


def _build_chat_session_metadata(
    *,
    attribution: GeminiChatSessionAttribution,
    transcript: GeminiChatSessionTranscript,
    workspace_id: str | None,
    workspace_folder: str | None,
    workspace_metadata: dict[str, object] | None,
    artifacts: GeminiIdeArtifacts | None,
) -> dict[str, object]:
    payload = dict(workspace_metadata) if workspace_metadata is not None else {}
    payload["scope"] = "chat_session"
    if workspace_id is not None:
        payload["workspace_id"] = workspace_id
    if workspace_folder is not None:
        payload["workspace_folder"] = workspace_folder
    payload["chat_session_id"] = attribution.session_id
    payload["chat_session_ownership"] = attribution.ownership
    if attribution.provider is not None:
        payload["chat_session_provider"] = attribution.provider
    if attribution.request_count is not None:
        payload["chat_session_request_count"] = attribution.request_count
    if attribution.is_empty is not None:
        payload["chat_session_is_empty"] = attribution.is_empty
    payload["user_message_count"] = transcript.user_message_count
    payload["assistant_message_count"] = transcript.assistant_message_count
    if artifacts is not None:
        if artifacts.credential_artifact_count:
            payload["credential_artifacts_present"] = True
        if artifacts.install_artifact_count:
            payload["install_artifacts_present"] = True
    return payload


def _build_chat_session_residue_payload(
    *,
    attribution: GeminiChatSessionAttribution,
    residue_kind: str | None = None,
    limitations: tuple[str, ...] = (),
) -> dict[str, object] | None:
    if residue_kind is None:
        if attribution.ownership == "foreign":
            residue_kind = "foreign_provider_rejected"
            limitations = (FOREIGN_PROVIDER_REJECTED_LIMITATION,)
        elif attribution.ownership == "unknown":
            residue_kind = "unknown_provider_rejected"
            limitations = (UNKNOWN_PROVIDER_REJECTED_LIMITATION,)
        else:
            return None

    payload = attribution.to_dict()
    payload["residue_kind"] = residue_kind
    if limitations:
        payload["limitations"] = list(limitations)
    return payload


def _build_workspace_residue_conversation(
    *,
    workspace_id: str,
    workspace_folder: str | None,
    state_db_path: Path,
    workspace_metadata: dict[str, object],
    residue_payloads: tuple[dict[str, object], ...],
    transcript_rows: tuple[NormalizedConversation, ...],
    collected_at: str | None,
    artifacts: GeminiIdeArtifacts | None,
) -> NormalizedConversation | None:
    if not residue_payloads and transcript_rows:
        return None

    session_metadata = dict(workspace_metadata)
    session_metadata["scope"] = "workspace_state_residue"
    if transcript_rows:
        session_metadata["transcript_chat_session_count"] = len(transcript_rows)
    if residue_payloads:
        session_metadata["metadata_only_chat_session_residue"] = list(residue_payloads)
        session_metadata["metadata_only_chat_session_residue_count"] = len(residue_payloads)

    return NormalizedConversation(
        source=GEMINI_CODE_ASSIST_IDE_DESCRIPTOR.key,
        execution_context=GEMINI_CODE_ASSIST_IDE_DESCRIPTOR.execution_context,
        collected_at=collected_at or utc_timestamp(),
        messages=(),
        transcript_completeness=TranscriptCompleteness.UNSUPPORTED,
        limitations=_workspace_residue_limitations(
            residue_payloads=residue_payloads,
            transcript_rows=transcript_rows,
        ),
        source_session_id=f"vscode:{workspace_id}:residue",
        source_artifact_path=str(state_db_path),
        session_metadata=session_metadata,
        provenance=ConversationProvenance(
            source="vscode",
            originator="google.geminicodeassist",
            cwd=workspace_folder,
            conversation_origin=WORKSPACE_RESIDUE_CONVERSATION_ORIGIN,
            app_shell=(
                artifacts.build_app_shell(state_db_path=str(state_db_path))
                if artifacts is not None
                else None
            ),
        ),
    )


def _workspace_residue_limitations(
    *,
    residue_payloads: tuple[dict[str, object], ...],
    transcript_rows: tuple[NormalizedConversation, ...],
) -> tuple[str, ...]:
    limitations: list[str] = []
    if not transcript_rows:
        limitations.append(NO_CONFIRMED_TRANSCRIPT_LIMITATION)
    limitations.append(METADATA_ONLY_IDE_STATE_LIMITATION)

    if not residue_payloads and not transcript_rows:
        limitations.append(CHAT_SESSION_PROVIDER_ATTRIBUTION_REQUIRED_LIMITATION)

    for payload in residue_payloads:
        for limitation in payload.get("limitations", ()):
            if not isinstance(limitation, str) or limitation in limitations:
                continue
            limitations.append(limitation)

    return tuple(limitations)


def _build_global_state_metadata(
    *,
    gemini_state: dict[str, object] | None,
    chat_view_hidden: bool | None,
    outline_view_hidden: bool | None,
    artifacts: GeminiIdeArtifacts | None,
) -> dict[str, object]:
    payload: dict[str, object] = {"scope": "global_state"}
    if gemini_state is not None:
        field_map = {
            "geminicodeassist.hasRunOnce": "has_run_once",
            "geminicodeassist.lastOpenedVersion": "last_opened_version",
            "newChatIsAgent": "new_chat_is_agent",
            "lastChatModeWasAgent": "last_chat_mode_was_agent",
            "cloudcode.duetAI.showAgentTipsCard.agent": "show_agent_tips_card",
            "cloudcode.duetAI.onboardingTooltipInvokedOnce": "onboarding_tooltip_invoked_once",
        }
        for source_key, target_key in field_map.items():
            value = gemini_state.get(source_key)
            if isinstance(value, (bool, str, int, float)):
                payload[target_key] = value

        session_index = gemini_state.get("cloudcode.session-index")
        if isinstance(session_index, list):
            payload["cloudcode_session_index_count"] = len(session_index)

        hats_index = gemini_state.get("cloudcode.hats-index")
        if isinstance(hats_index, list):
            payload["cloudcode_hats_index_count"] = len(hats_index)

    if chat_view_hidden is not None:
        payload["chat_view_hidden"] = chat_view_hidden
    if outline_view_hidden is not None:
        payload["outline_view_hidden"] = outline_view_hidden
    if artifacts is not None:
        if artifacts.credential_artifact_count:
            payload["credential_artifacts_present"] = True
            payload["credential_artifact_count"] = artifacts.credential_artifact_count
        if artifacts.install_artifact_count:
            payload["install_artifacts_present"] = True
            payload["install_artifact_count"] = artifacts.install_artifact_count
    return payload


def _build_workspace_state_metadata(
    *,
    workspace_id: str,
    workspace_folder: str | None,
    chat_view_state: dict[str, object] | None,
    outline_view_state: dict[str, object] | None,
    memento_payload: object,
    visible_views: int | None,
    chat_session_index: object,
    attributions: tuple[GeminiChatSessionAttribution, ...],
    artifacts: GeminiIdeArtifacts | None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "scope": "workspace_state",
        "workspace_id": workspace_id,
    }
    if workspace_folder is not None:
        payload["workspace_folder"] = workspace_folder
    if chat_view_state is not None:
        payload["chat_view_state"] = chat_view_state
    if outline_view_state is not None:
        payload["outline_view_state"] = outline_view_state
    if isinstance(memento_payload, dict):
        payload["chat_view_memento_keys"] = sorted(memento_payload)
    if visible_views is not None:
        payload["number_of_visible_chat_views"] = visible_views

    index_summary = _summarize_chat_session_index(chat_session_index)
    payload.update(index_summary)

    if attributions:
        payload["chat_session_attribution"] = [row.to_dict() for row in attributions]
        payload["gemini_owned_chat_session_count"] = sum(
            1 for row in attributions if row.ownership == "gemini"
        )
        payload["foreign_chat_session_count"] = sum(
            1 for row in attributions if row.ownership == "foreign"
        )
        payload["unknown_chat_session_count"] = sum(
            1 for row in attributions if row.ownership == "unknown"
        )

    if artifacts is not None:
        if artifacts.credential_artifact_count:
            payload["credential_artifacts_present"] = True
        if artifacts.install_artifact_count:
            payload["install_artifacts_present"] = True
    return payload


def _summarize_chat_session_index(chat_session_index: object) -> dict[str, object]:
    if not isinstance(chat_session_index, dict):
        return {}

    payload: dict[str, object] = {}
    version = _int_value(chat_session_index.get("version"))
    if version is not None:
        payload["chat_session_index_version"] = version

    entries = chat_session_index.get("entries")
    if not isinstance(entries, list):
        return payload

    session_count = 0
    empty_count = 0
    external_count = 0
    pending_edits_count = 0
    latest_message_at: str | None = None

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if _string_value(entry.get("sessionId")) is None:
            continue
        session_count += 1

        if entry.get("isEmpty") is True:
            empty_count += 1
        if entry.get("isExternal") is True:
            external_count += 1
        if entry.get("hasPendingEdits") is True:
            pending_edits_count += 1

        candidate_timestamp = _normalize_timestamp(entry.get("lastMessageDate"))
        if candidate_timestamp is not None and (
            latest_message_at is None or candidate_timestamp > latest_message_at
        ):
            latest_message_at = candidate_timestamp

    payload["indexed_session_count"] = session_count
    if empty_count:
        payload["empty_indexed_session_count"] = empty_count
    if external_count:
        payload["external_indexed_session_count"] = external_count
    if pending_edits_count:
        payload["pending_edit_session_count"] = pending_edits_count
    if latest_message_at is not None:
        payload["latest_indexed_message_at"] = latest_message_at
    return payload


def _attribute_indexed_chat_sessions(
    workspace_dir: Path,
    chat_session_index: object,
) -> tuple[GeminiChatSessionAttribution, ...]:
    indexed_session_ids = _indexed_session_ids(chat_session_index)
    if not indexed_session_ids:
        return ()

    chat_sessions_dir = workspace_dir / "chatSessions"
    if not chat_sessions_dir.is_dir():
        return ()

    session_paths = {
        path.stem: path.resolve(strict=False)
        for path in chat_sessions_dir.rglob("*.json")
        if path.is_file()
    }

    attributions: list[GeminiChatSessionAttribution] = []
    for session_id in indexed_session_ids:
        session_path = session_paths.get(session_id)
        if session_path is None:
            continue
        attribution = attribute_chat_session(session_path)
        if attribution is None:
            continue
        attributions.append(attribution)

    return tuple(attributions)


def _indexed_session_ids(chat_session_index: object) -> tuple[str, ...]:
    if not isinstance(chat_session_index, dict):
        return ()

    entries = chat_session_index.get("entries")
    if not isinstance(entries, list):
        return ()

    session_ids: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        session_id = _string_value(entry.get("sessionId"))
        if session_id is None:
            continue
        session_ids.append(session_id)
    return tuple(session_ids)


def _provider_candidates(payload: dict[str, object]) -> tuple[str, ...]:
    candidates = [
        _string_value(payload.get("responderUsername")),
        _string_value(payload.get("provider")),
        _string_value(payload.get("providerId")),
        _nested_string(payload, "responder", "provider"),
        _nested_string(payload, "responder", "extensionId"),
        _nested_string(payload, "selectedModel", "metadata", "extension", "value"),
        _nested_string(payload, "selectedModel", "metadata", "extension", "id"),
        _nested_string(payload, "selectedModel", "metadata", "extensionId"),
        _nested_string(payload, "selectedModel", "metadata", "provider"),
        _nested_string(payload, "metadata", "extension", "value"),
        _nested_string(payload, "metadata", "extension", "id"),
        _nested_string(payload, "metadata", "extensionId"),
        _nested_string(payload, "metadata", "provider"),
    ]
    unique_candidates: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate is None or candidate in seen:
            continue
        seen.add(candidate)
        unique_candidates.append(candidate)
    return tuple(unique_candidates)


def _classify_provider(provider_candidates: tuple[str, ...]) -> tuple[str, str | None, str]:
    for candidate in provider_candidates:
        lowered = candidate.casefold()
        if any(marker in lowered for marker in GEMINI_PROVIDER_MARKERS):
            return "gemini", candidate, "explicit_gemini_provider_marker"

    for candidate in provider_candidates:
        lowered = candidate.casefold()
        if any(marker in lowered for marker in FOREIGN_PROVIDER_MARKERS):
            return "foreign", candidate, "explicit_foreign_provider_marker"

    if provider_candidates:
        return "unknown", provider_candidates[0], "provider_candidate_unrecognized"
    return "unknown", None, "provider_candidate_missing"


def _extract_view_state(payload: object, nested_key: str) -> dict[str, object] | None:
    if not isinstance(payload, dict):
        return None

    nested_payload = payload.get(nested_key)
    if isinstance(nested_payload, dict):
        payload = nested_payload

    extracted: dict[str, object] = {}
    collapsed = _bool_value(payload.get("collapsed"))
    if collapsed is not None:
        extracted["collapsed"] = collapsed
    is_hidden = _bool_value(payload.get("isHidden"))
    if is_hidden is not None:
        extracted["is_hidden"] = is_hidden
    size = _int_value(payload.get("size"))
    if size is not None:
        extracted["size"] = size

    if extracted:
        return extracted
    return None


def _hidden_view_state(payload: object) -> bool | None:
    if not isinstance(payload, list):
        return None

    hidden_values = [
        item.get("isHidden")
        for item in payload
        if isinstance(item, dict) and isinstance(item.get("isHidden"), bool)
    ]
    if not hidden_values:
        return None
    if len(hidden_values) == 1:
        return hidden_values[0]
    return all(hidden_values)


def _discover_paths(
    input_roots: tuple[Path, ...],
    *,
    direct_match,
    glob_pattern: str,
    expect_dir: bool,
) -> tuple[str, ...]:
    seen: set[Path] = set()
    candidates: list[str] = []

    for input_root in input_roots:
        matches: list[Path] = []
        if direct_match(input_root):
            matches.append(input_root)
        if input_root.is_dir():
            matches.extend(input_root.glob(glob_pattern))

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


def _count_credential_artifacts(google_extension_roots: tuple[str, ...]) -> int:
    count = 0
    for root in (Path(raw_root) for raw_root in google_extension_roots):
        auth_root = root / "auth"
        for credential_file_name in CREDENTIAL_FILE_NAMES:
            if (auth_root / credential_file_name).is_file():
                count += 1
    return count


def _count_install_artifacts(cloud_code_roots: tuple[str, ...]) -> int:
    count = 0
    for root in (Path(raw_root) for raw_root in cloud_code_roots):
        if (root / "install_id.txt").is_file():
            count += 1
    return count


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
    try:
        payload = json.loads(workspace_json_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None

    folder_uri = _string_value(payload.get("folder"))
    if folder_uri is None:
        return None
    if folder_uri.startswith("file://"):
        parsed = urlparse(folder_uri)
        return unquote(parsed.path) or None
    return folder_uri


def _read_chat_session_payload(chat_session_path: Path) -> dict[str, object] | None:
    try:
        payload = json.loads(chat_session_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _compose_chat_session_source_id(*, workspace_id: str | None, session_id: str) -> str:
    if workspace_id is None:
        return session_id
    return f"vscode:{workspace_id}:{session_id}"


def _chat_session_request_id(raw_request: dict[str, object], *, index: int) -> str:
    return (
        _string_value(raw_request.get("requestId"))
        or _string_value(raw_request.get("id"))
        or f"request-{index}"
    )


def _chat_session_response_id(
    raw_request: dict[str, object],
    *,
    request_id: str,
    index: int,
) -> str:
    response_payload = raw_request.get("response")
    if isinstance(response_payload, dict):
        response_id = (
            _string_value(response_payload.get("responseId"))
            or _string_value(response_payload.get("id"))
        )
        if response_id is not None:
            return response_id

    return (
        _string_value(raw_request.get("responseId"))
        or _string_value(raw_request.get("responseMessageId"))
        or f"{request_id}:response-{index}"
    )


def _extract_text_from_named_fields(
    payload: dict[str, object],
    field_names: tuple[str, ...],
    *,
    nested_keys: frozenset[str],
) -> str | None:
    fragments: list[str] = []
    for field_name in field_names:
        if field_name not in payload:
            continue
        _append_text_fragments(payload.get(field_name), fragments, nested_keys=nested_keys)

    if not fragments:
        return None

    unique_fragments: list[str] = []
    seen: set[str] = set()
    for fragment in fragments:
        if fragment in seen:
            continue
        seen.add(fragment)
        unique_fragments.append(fragment)

    if not unique_fragments:
        return None
    return "\n\n".join(unique_fragments)


def _append_text_fragments(
    content: object,
    fragments: list[str],
    *,
    nested_keys: frozenset[str],
) -> None:
    if isinstance(content, str):
        stripped = content.strip()
        if stripped:
            fragments.append(stripped)
        return

    if isinstance(content, list):
        for item in content:
            _append_text_fragments(item, fragments, nested_keys=nested_keys)
        return

    if not isinstance(content, dict):
        return

    for key in ("text", "markdown"):
        if key in content:
            _append_text_fragments(content.get(key), fragments, nested_keys=nested_keys)

    for key in nested_keys:
        if key in content:
            _append_text_fragments(content.get(key), fragments, nested_keys=nested_keys)


def _has_named_field(payload: dict[str, object], field_names: tuple[str, ...]) -> bool:
    for field_name in field_names:
        value = payload.get(field_name)
        if isinstance(value, str) and value.strip():
            return True
        if isinstance(value, (list, dict)) and value:
            return True
    return False


def _first_normalized_timestamp(
    payload: object,
    paths: tuple[tuple[str, ...], ...],
) -> str | None:
    for path in paths:
        timestamp = _normalize_timestamp(_nested_value(payload, *path))
        if timestamp is not None:
            return timestamp
    return None


def _nested_value(payload: object, *path: str) -> object:
    current = payload
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _nested_string(payload: object, *path: str) -> str | None:
    return _string_value(_nested_value(payload, *path))


def _normalize_timestamp(value: object) -> str | None:
    if isinstance(value, (int, float)):
        return _unix_ms_to_timestamp(value)
    if not isinstance(value, str):
        return None

    stripped = value.strip()
    if not stripped:
        return None
    if stripped.isdigit():
        return _unix_ms_to_timestamp(int(stripped))

    try:
        timestamp = datetime.fromisoformat(stripped.replace("Z", "+00:00"))
    except ValueError:
        return stripped
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _unix_ms_to_timestamp(value: int | float) -> str:
    timestamp = datetime.fromtimestamp(float(value) / 1000, tz=timezone.utc)
    return timestamp.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _string_value(value: object) -> str | None:
    if isinstance(value, str):
        return value
    return None


def _int_value(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _bool_value(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _list_length(value: object) -> int | None:
    if isinstance(value, list):
        return len(value)
    return None


def _is_google_vscode_extension_root(path: Path) -> bool:
    return path.name == "google-vscode-extension" and "Application Support" in path.parts


def _is_cloud_code_root(path: Path) -> bool:
    return path.name == "cloud-code" and "Application Support" in path.parts


def _is_global_state_db(path: Path) -> bool:
    return path.name == "state.vscdb" and path.parent.name == "globalStorage"


def _is_workspace_state_db(path: Path) -> bool:
    return path.name == "state.vscdb" and path.parent.parent.name == "workspaceStorage"


__all__ = [
    "GEMINI_CODE_ASSIST_IDE_DESCRIPTOR",
    "GeminiCodeAssistIdeCollector",
    "GeminiIdeArtifacts",
    "GeminiChatSessionAttribution",
    "attribute_chat_session",
    "discover_gemini_code_assist_ide_artifacts",
    "parse_chat_session_transcript",
    "parse_global_state",
    "parse_workspace_state",
    "parse_workspace_state_rows",
]
