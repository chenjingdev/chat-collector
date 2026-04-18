from __future__ import annotations

import json
import shutil
import sqlite3
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import unquote, urlparse

from ..incremental import write_incremental_collection
from ..models import (
    AppShellProvenance,
    CollectionPlan,
    CollectionResult,
    ConversationProvenance,
    MessageRole,
    MessageProvenance,
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

CURSOR_EDITOR_DESCRIPTOR = SourceDescriptor(
    key="cursor_editor",
    display_name="Cursor Editor",
    execution_context="ide_native",
    support_level=SupportLevel.PARTIAL,
    default_input_roots=(
        "~/.cursor",
        "~/Library/Application Support/Cursor",
    ),
    artifact_root_candidates=(
        all_platform_root("$HOME/.cursor"),
        darwin_root("$HOME/Library/Application Support/Cursor"),
        linux_root("$XDG_CONFIG_HOME/Cursor"),
        windows_root("$APPDATA/Cursor"),
    ),
    notes=(
        "Uses Cursor User/workspaceStorage/<workspace-id>/state.vscdb plus shared cursorDiskKV rows as the transcript-bearing composer path.",
        "Reconstructs ordered user and assistant messages when the selected composer has explicit cursorDiskKV bubble bodies in known body fields.",
        "Degrades to prompt-only evidence when a composer references missing, empty, or tool-only cursorDiskKV bubble rows.",
        "Treats cursor.hooks.log, .cursor/ai-tracking, memory flags, and third-party extension state as provenance or auxiliary metadata only.",
    ),
    support_metadata=SourceSupportMetadata(
        product_label="Cursor",
        host_surface="Editor",
        expected_transcript_completeness=TranscriptCompleteness.PARTIAL,
        limitation_summary="Cursor editor recovery restores known explicit cursorDiskKV bubble body variants, but sessions whose headers resolve only to empty or tool-only rows remain partial and opt-in for unattended batches.",
        limitations=(
            "Composer sessions whose cursorDiskKV headers resolve only to empty or tool-only bubble rows still degrade to partial output, with missing bubble ids retained in session metadata.",
            "Cursor host logs, memory flags, and third-party extension state never promote a session without confirmed transcript rows.",
        ),
    ),
)

WORKSPACE_STATE_GLOB = "**/User/workspaceStorage/*/state.vscdb"
APPLICATION_SUPPORT_GLOB = "**/Application Support/Cursor"
GLOBAL_STATE_GLOB = "**/User/globalStorage/state.vscdb"
HOOK_LOG_GLOB = "**/cursor.hooks.log"
TRACKING_DB_GLOB = "**/.cursor/ai-tracking/ai-code-tracking.db"
PROMPT_FALLBACK_LIMITATIONS = (
    "cursor_disk_kv_transcript_not_found",
    "assistant_body_unverified",
    "workspace_prompt_cache_not_explicitly_composer_scoped",
)
MISSING_ASSISTANT_LIMITATION = "assistant_body_missing_from_cursor_disk_kv"
MISSING_USER_LIMITATION = "user_body_missing_from_cursor_disk_kv"
THIRD_PARTY_STATE_KEYS = frozenset(
    {
        "memento/webviewView.chatgpt.sidebarView",
        "openai.chatgpt",
    }
)


@dataclass(frozen=True, slots=True)
class CursorAuxiliaryArtifacts:
    application_support_roots: tuple[str, ...] = ()
    global_state_paths: tuple[str, ...] = ()
    hook_log_paths: tuple[str, ...] = ()
    tracking_db_paths: tuple[str, ...] = ()
    ignored_state_keys: tuple[str, ...] = ()
    memory_enabled: bool | None = None
    pending_memories_count: int | None = None
    hook_session_end_count: int = 0

    def build_app_shell(self, *, state_db_path: str) -> AppShellProvenance | None:
        cache_roots = tuple(
            sorted(
                {
                    str(Path(path).resolve(strict=False).parent.parent)
                    for path in self.tracking_db_paths
                    if len(Path(path).parts) >= 2
                }
            )
        )
        state_db_paths = tuple(sorted({state_db_path, *self.global_state_paths}))

        provenance = AppShellProvenance(
            application_support_roots=self.application_support_roots,
            state_db_paths=state_db_paths,
            log_paths=self.hook_log_paths,
            cache_roots=cache_roots,
            auxiliary_paths=self.tracking_db_paths,
        )
        if not provenance.to_dict():
            return None
        return provenance


@dataclass(frozen=True, slots=True)
class CursorComposer:
    composer_id: str
    name: str | None = None
    subtitle: str | None = None
    created_at: str | None = None
    last_updated_at: str | None = None
    mode: str | None = None
    context_usage_percent: int | None = None
    archived: bool | None = None
    is_worktree: bool | None = None
    is_spec: bool | None = None


@dataclass(frozen=True, slots=True)
class CursorTranscriptRecovery:
    messages: tuple[NormalizedMessage, ...]
    completeness: TranscriptCompleteness
    limitations: tuple[str, ...]
    source_name: str
    source_path: str | None = None
    header_count: int = 0
    assistant_message_count: int = 0
    skipped_tool_bubble_count: int = 0
    missing_user_bubble_ids: tuple[str, ...] = ()
    missing_assistant_bubble_ids: tuple[str, ...] = ()
    skipped_tool_bubble_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CursorEditorCollector:
    descriptor: SourceDescriptor = CURSOR_EDITOR_DESCRIPTOR

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
        workspace_state_paths = tuple(iter_workspace_state_paths(resolved_input_roots))
        auxiliary = discover_cursor_auxiliary_artifacts(resolved_input_roots)
        collected_at = utc_timestamp()
        conversations = (
            parse_workspace_state(
                workspace_state_path,
                collected_at=collected_at,
                auxiliary=auxiliary,
            )
            for workspace_state_path in workspace_state_paths
        )
        return write_incremental_collection(
            source=self.descriptor.key,
            archive_root=archive_root,
            input_roots=resolved_input_roots,
            scanned_artifact_count=len(workspace_state_paths),
            collected_at=collected_at,
            conversations=conversations,
        )

    def _default_input_roots(self) -> tuple[Path, ...]:
        return default_descriptor_input_roots(self.descriptor)


def parse_workspace_state(
    state_db_path: Path,
    *,
    collected_at: str | None = None,
    auxiliary: CursorAuxiliaryArtifacts | None = None,
) -> NormalizedConversation | None:
    resolved_path = state_db_path.expanduser().resolve(strict=False)
    state_values = _read_state_values(
        resolved_path,
        (
            "composer.composerData",
            "aiService.prompts",
            "aiService.generations",
        ),
    )
    composer_payload = state_values.get("composer.composerData")
    prompts_payload = state_values.get("aiService.prompts")
    generations_payload = state_values.get("aiService.generations")

    if not isinstance(composer_payload, dict):
        return None
    resolved_prompts_payload = prompts_payload if isinstance(prompts_payload, list) else []

    selected_composer = _select_composer(composer_payload)
    if selected_composer is None:
        return None
    composer, selected, last_focused = selected_composer

    transcript = _recover_transcript(
        composer_id=composer.composer_id,
        prompts_payload=resolved_prompts_payload,
        generations_payload=generations_payload,
        auxiliary=auxiliary,
    )
    if transcript is None:
        return None

    workspace_id = resolved_path.parent.name
    workspace_folder = _read_workspace_folder(resolved_path.parent / "workspace.json")
    workspace_ignored_keys = _read_item_keys(resolved_path, THIRD_PARTY_STATE_KEYS)
    ignored_state_keys = tuple(
        sorted({*workspace_ignored_keys, *(auxiliary.ignored_state_keys if auxiliary else ())})
    )

    provenance = ConversationProvenance(
        session_started_at=composer.created_at or transcript.messages[0].timestamp,
        source="cursor",
        originator="cursor_editor",
        cwd=workspace_folder,
        archived=composer.archived,
        app_shell=(
            auxiliary.build_app_shell(state_db_path=str(resolved_path))
            if auxiliary is not None
            else None
        ),
    )

    return NormalizedConversation(
        source=CURSOR_EDITOR_DESCRIPTOR.key,
        execution_context=CURSOR_EDITOR_DESCRIPTOR.execution_context,
        collected_at=collected_at or utc_timestamp(),
        messages=transcript.messages,
        transcript_completeness=transcript.completeness,
        limitations=transcript.limitations,
        source_session_id=composer.composer_id,
        source_artifact_path=str(resolved_path),
        session_metadata=_build_session_metadata(
            workspace_id=workspace_id,
            workspace_folder=workspace_folder,
            composer=composer,
            selected=selected,
            last_focused=last_focused,
            prompt_count=len(resolved_prompts_payload),
            generation_count=_count_generations(generations_payload),
            transcript=transcript,
            auxiliary=auxiliary,
            ignored_state_keys=ignored_state_keys,
        ),
        provenance=provenance,
    )


def discover_cursor_auxiliary_artifacts(
    input_roots: tuple[Path, ...] | None,
) -> CursorAuxiliaryArtifacts:
    if not input_roots:
        return CursorAuxiliaryArtifacts()

    application_support_roots = _discover_paths(
        input_roots,
        direct_match=_is_cursor_application_support_root,
        glob_pattern=APPLICATION_SUPPORT_GLOB,
        expect_dir=True,
    )
    global_state_paths = _discover_paths(
        input_roots,
        direct_match=_is_cursor_global_state_db,
        glob_pattern=GLOBAL_STATE_GLOB,
        expect_dir=False,
    )
    hook_log_paths = _discover_paths(
        input_roots,
        direct_match=_is_cursor_hook_log,
        glob_pattern=HOOK_LOG_GLOB,
        expect_dir=False,
    )
    tracking_db_paths = _discover_paths(
        input_roots,
        direct_match=_is_cursor_tracking_db,
        glob_pattern=TRACKING_DB_GLOB,
        expect_dir=False,
    )

    memory_enabled: bool | None = None
    pending_memories_count: int | None = None
    ignored_state_keys: set[str] = set()

    if global_state_paths:
        global_state_path = Path(global_state_paths[0])
        global_values = _read_state_values(
            global_state_path,
            ("cursor/memoriesEnabled", "cursor/pendingMemories"),
        )
        memory_enabled = _bool_value(global_values.get("cursor/memoriesEnabled"))
        pending_memories = global_values.get("cursor/pendingMemories")
        if isinstance(pending_memories, list):
            pending_memories_count = len(pending_memories)
        ignored_state_keys.update(_read_item_keys(global_state_path, THIRD_PARTY_STATE_KEYS))

    hook_session_end_count = sum(_count_hook_session_end_records(Path(path)) for path in hook_log_paths)

    return CursorAuxiliaryArtifacts(
        application_support_roots=application_support_roots,
        global_state_paths=global_state_paths,
        hook_log_paths=hook_log_paths,
        tracking_db_paths=tracking_db_paths,
        ignored_state_keys=tuple(sorted(ignored_state_keys)),
        memory_enabled=memory_enabled,
        pending_memories_count=pending_memories_count,
        hook_session_end_count=hook_session_end_count,
    )


def iter_workspace_state_paths(input_roots: Iterable[Path]) -> Iterable[Path]:
    seen: set[Path] = set()
    candidates: list[Path] = []
    for input_root in input_roots:
        matches: list[Path] = []
        if _is_workspace_state_db(input_root):
            matches.append(input_root)
        if input_root.is_dir():
            matches.extend(input_root.glob(WORKSPACE_STATE_GLOB))

        for candidate in matches:
            if not candidate.is_file():
                continue
            resolved = candidate.resolve(strict=False)
            if resolved in seen:
                continue
            seen.add(resolved)
            candidates.append(resolved)
    yield from sorted(candidates)


def _build_prompt_messages(
    prompts_payload: list[object],
    generations_payload: object,
) -> list[NormalizedMessage]:
    generation_timestamps = _generation_timestamps(generations_payload)
    messages: list[NormalizedMessage] = []

    for index, prompt in enumerate(prompts_payload):
        if not isinstance(prompt, dict):
            continue

        prompt_text = _clean_prompt_text(prompt.get("text"))
        if prompt_text is None:
            continue

        timestamp = generation_timestamps[index] if index < len(generation_timestamps) else None
        messages.append(
            NormalizedMessage(
                role=MessageRole.USER,
                text=prompt_text,
                timestamp=timestamp,
            )
        )

    return messages


def _recover_transcript(
    *,
    composer_id: str,
    prompts_payload: list[object],
    generations_payload: object,
    auxiliary: CursorAuxiliaryArtifacts | None,
) -> CursorTranscriptRecovery | None:
    generation_timestamps = _generation_timestamps(generations_payload)
    transcript = _read_cursor_disk_kv_transcript(
        composer_id=composer_id,
        global_state_paths=auxiliary.global_state_paths if auxiliary is not None else (),
        generation_timestamps=generation_timestamps,
    )
    if transcript is not None and transcript.messages:
        return transcript

    prompt_messages = tuple(_build_prompt_messages(prompts_payload, generations_payload))
    if not prompt_messages:
        return None

    limitations = list(PROMPT_FALLBACK_LIMITATIONS)
    if transcript is not None:
        for limitation in transcript.limitations:
            if limitation not in limitations:
                limitations.append(limitation)

    return CursorTranscriptRecovery(
        messages=prompt_messages,
        completeness=TranscriptCompleteness.PARTIAL,
        limitations=tuple(limitations),
        source_name="workspace_prompt_cache",
        source_path=None if transcript is None else transcript.source_path,
        header_count=0 if transcript is None else transcript.header_count,
        assistant_message_count=(
            0 if transcript is None else transcript.assistant_message_count
        ),
        skipped_tool_bubble_count=(
            0 if transcript is None else transcript.skipped_tool_bubble_count
        ),
        missing_user_bubble_ids=(
            () if transcript is None else transcript.missing_user_bubble_ids
        ),
        missing_assistant_bubble_ids=(
            () if transcript is None else transcript.missing_assistant_bubble_ids
        ),
        skipped_tool_bubble_ids=(
            () if transcript is None else transcript.skipped_tool_bubble_ids
        ),
    )


def _read_cursor_disk_kv_transcript(
    *,
    composer_id: str,
    global_state_paths: tuple[str, ...],
    generation_timestamps: list[str | None],
) -> CursorTranscriptRecovery | None:
    if not global_state_paths:
        return None

    composer_key = _composer_data_key(composer_id)
    for raw_global_state_path in global_state_paths:
        global_state_path = Path(raw_global_state_path)
        composer_payload = _read_cursor_disk_kv_values(
            global_state_path,
            (composer_key,),
        ).get(composer_key)
        if not isinstance(composer_payload, dict):
            continue

        transcript = _build_cursor_disk_kv_transcript(
            composer_id=composer_id,
            composer_payload=composer_payload,
            global_state_path=global_state_path,
            generation_timestamps=generation_timestamps,
        )
        if transcript is not None:
            return transcript

    return None


def _build_cursor_disk_kv_transcript(
    *,
    composer_id: str,
    composer_payload: dict[str, object],
    global_state_path: Path,
    generation_timestamps: list[str | None],
) -> CursorTranscriptRecovery | None:
    headers = composer_payload.get("fullConversationHeadersOnly")
    if not isinstance(headers, list):
        return None

    ordered_bubbles: list[tuple[str, MessageRole]] = []
    for header in headers:
        if not isinstance(header, dict):
            continue
        bubble_id = _string_value(header.get("bubbleId"))
        bubble_type = _int_value(header.get("type"))
        if bubble_id is None or bubble_type not in (1, 2):
            continue
        ordered_bubbles.append(
            (
                bubble_id,
                MessageRole.USER if bubble_type == 1 else MessageRole.ASSISTANT,
            )
        )

    if not ordered_bubbles:
        return None

    bubble_values = _read_cursor_disk_kv_values(
        global_state_path,
        tuple(
            _bubble_data_key(composer_id, bubble_id)
            for bubble_id, _ in ordered_bubbles
        ),
    )

    messages: list[NormalizedMessage] = []
    assistant_message_count = 0
    missing_assistant_bubble_ids: list[str] = []
    missing_user_bubble_ids: list[str] = []
    skipped_tool_bubble_count = 0
    skipped_tool_bubble_ids: list[str] = []

    for bubble_id, role in ordered_bubbles:
        bubble_payload = bubble_values.get(_bubble_data_key(composer_id, bubble_id))
        if not isinstance(bubble_payload, dict):
            if role == MessageRole.ASSISTANT:
                missing_assistant_bubble_ids.append(bubble_id)
            else:
                missing_user_bubble_ids.append(bubble_id)
            continue

        body = _extract_bubble_body(bubble_payload)
        if body is None:
            if role == MessageRole.ASSISTANT and _is_tool_only_bubble(bubble_payload):
                skipped_tool_bubble_count += 1
                skipped_tool_bubble_ids.append(bubble_id)
                continue
            if role == MessageRole.ASSISTANT:
                missing_assistant_bubble_ids.append(bubble_id)
            else:
                missing_user_bubble_ids.append(bubble_id)
            continue
        text, body_source = body

        timestamp = None
        if role == MessageRole.ASSISTANT:
            if assistant_message_count < len(generation_timestamps):
                timestamp = generation_timestamps[assistant_message_count]
            assistant_message_count += 1

        messages.append(
            NormalizedMessage(
                role=role,
                text=text,
                timestamp=timestamp,
                source_message_id=bubble_id,
                provenance=MessageProvenance(body_source=body_source),
            )
        )

    limitations: list[str] = []
    if missing_user_bubble_ids:
        limitations.append(MISSING_USER_LIMITATION)
    if missing_assistant_bubble_ids:
        limitations.append(MISSING_ASSISTANT_LIMITATION)
    if assistant_message_count == 0 and not missing_assistant_bubble_ids:
        limitations.append("assistant_body_unverified")

    completeness = (
        TranscriptCompleteness.COMPLETE
        if assistant_message_count > 0 and not limitations
        else TranscriptCompleteness.PARTIAL
    )
    return CursorTranscriptRecovery(
        messages=tuple(messages),
        completeness=completeness,
        limitations=tuple(limitations),
        source_name="cursor_disk_kv",
        source_path=str(global_state_path),
        header_count=len(ordered_bubbles),
        assistant_message_count=assistant_message_count,
        skipped_tool_bubble_count=skipped_tool_bubble_count,
        missing_user_bubble_ids=tuple(missing_user_bubble_ids),
        missing_assistant_bubble_ids=tuple(missing_assistant_bubble_ids),
        skipped_tool_bubble_ids=tuple(skipped_tool_bubble_ids),
    )


def _is_tool_only_bubble(bubble_payload: dict[str, object]) -> bool:
    tool_former_data = bubble_payload.get("toolFormerData")
    if isinstance(tool_former_data, dict):
        return bool(tool_former_data)
    if isinstance(tool_former_data, list):
        return bool(tool_former_data)
    return False


def _extract_bubble_body(bubble_payload: dict[str, object]) -> tuple[str, str] | None:
    for key_path in (
        ("text",),
        ("markdown",),
        ("markdownText",),
        ("body", "text"),
        ("body", "markdown"),
        ("body", "content"),
        ("body", "value"),
        ("content", "text"),
        ("content", "markdown"),
        ("content", "content"),
        ("content", "value"),
        ("message", "text"),
        ("message", "markdown"),
        ("message", "content"),
        ("message", "value"),
        ("richText", "text"),
        ("richText", "markdown"),
        ("richText", "content"),
        ("body",),
        ("content",),
        ("message",),
    ):
        raw_value = _nested_value(bubble_payload, key_path)
        text = _clean_prompt_text(raw_value)
        if text is None:
            continue
        return text, "cursor_disk_kv." + ".".join(key_path)
    return None


def _nested_value(payload: object, key_path: tuple[str, ...]) -> object:
    current = payload
    for key in key_path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _select_composer(
    composer_payload: dict[str, object],
) -> tuple[CursorComposer, bool, bool] | None:
    raw_composers = composer_payload.get("allComposers")
    if not isinstance(raw_composers, list):
        return None

    composers: dict[str, CursorComposer] = {}
    for raw_composer in raw_composers:
        if not isinstance(raw_composer, dict):
            continue
        composer_id = _string_value(raw_composer.get("composerId"))
        if composer_id is None:
            continue
        composers[composer_id] = CursorComposer(
            composer_id=composer_id,
            name=_string_value(raw_composer.get("name")),
            subtitle=_string_value(raw_composer.get("subtitle")),
            created_at=_normalize_timestamp(raw_composer.get("createdAt")),
            last_updated_at=_normalize_timestamp(raw_composer.get("lastUpdatedAt")),
            mode=_string_value(raw_composer.get("forceMode"))
            or _string_value(raw_composer.get("unifiedMode")),
            context_usage_percent=_int_value(raw_composer.get("contextUsagePercent")),
            archived=_bool_value(raw_composer.get("isArchived")),
            is_worktree=_bool_value(raw_composer.get("isWorktree")),
            is_spec=_bool_value(raw_composer.get("isSpec")),
        )

    if not composers:
        return None

    selected_ids = _string_list(composer_payload.get("selectedComposerIds"))
    last_focused_ids = _string_list(composer_payload.get("lastFocusedComposerIds"))

    for composer_id in (*last_focused_ids, *selected_ids):
        composer = composers.get(composer_id)
        if composer is None:
            continue
        return composer, composer_id in selected_ids, composer_id in last_focused_ids

    if len(composers) != 1:
        return None

    composer = next(iter(composers.values()))
    return composer, False, False


def _build_session_metadata(
    *,
    workspace_id: str,
    workspace_folder: str | None,
    composer: CursorComposer,
    selected: bool,
    last_focused: bool,
    prompt_count: int,
    generation_count: int,
    transcript: CursorTranscriptRecovery,
    auxiliary: CursorAuxiliaryArtifacts | None,
    ignored_state_keys: tuple[str, ...],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "workspace_id": workspace_id,
        "composer_id": composer.composer_id,
        "prompt_count": prompt_count,
        "generation_count": generation_count,
        "reconstructed_message_count": len(transcript.messages),
        "transcript_source": transcript.source_name,
        "selected": selected,
        "last_focused": last_focused,
    }
    if transcript.header_count:
        payload["transcript_header_count"] = transcript.header_count
    if transcript.skipped_tool_bubble_count:
        payload["skipped_tool_bubble_count"] = transcript.skipped_tool_bubble_count
    if transcript.missing_user_bubble_ids:
        payload["missing_user_bubble_ids"] = list(transcript.missing_user_bubble_ids)
    if transcript.missing_assistant_bubble_ids:
        payload["missing_assistant_bubble_ids"] = list(
            transcript.missing_assistant_bubble_ids
        )
    if transcript.skipped_tool_bubble_ids:
        payload["skipped_tool_bubble_ids"] = list(transcript.skipped_tool_bubble_ids)
    if workspace_folder is not None:
        payload["workspace_folder"] = workspace_folder
    if composer.name is not None:
        payload["composer_name"] = composer.name
    if composer.subtitle is not None:
        payload["composer_subtitle"] = composer.subtitle
    if composer.mode is not None:
        payload["mode"] = composer.mode
    if composer.created_at is not None:
        payload["created_at"] = composer.created_at
    if composer.last_updated_at is not None:
        payload["last_updated_at"] = composer.last_updated_at
    if composer.context_usage_percent is not None:
        payload["context_usage_percent"] = composer.context_usage_percent
    if composer.archived is not None:
        payload["is_archived"] = composer.archived
    if composer.is_worktree is not None:
        payload["is_worktree"] = composer.is_worktree
    if composer.is_spec is not None:
        payload["is_spec"] = composer.is_spec
    if auxiliary is not None:
        if auxiliary.memory_enabled is not None:
            payload["memory_enabled"] = auxiliary.memory_enabled
        if auxiliary.pending_memories_count is not None:
            payload["pending_memories_count"] = auxiliary.pending_memories_count
        if auxiliary.hook_session_end_count:
            payload["hook_session_end_count"] = auxiliary.hook_session_end_count
    if ignored_state_keys:
        payload["ignored_state_keys"] = list(ignored_state_keys)
    return payload


def _generation_timestamps(generations_payload: object) -> list[str | None]:
    if not isinstance(generations_payload, list):
        return []

    timestamps: list[str | None] = []
    for generation in generations_payload:
        if not isinstance(generation, dict):
            continue
        generation_type = _string_value(generation.get("type"))
        if generation_type not in (None, "composer"):
            continue
        timestamps.append(_normalize_timestamp(generation.get("unixMs")))
    return timestamps


def _count_generations(generations_payload: object) -> int:
    return len(_generation_timestamps(generations_payload))


def _clean_prompt_text(value: object) -> str | None:
    text = _string_value(value)
    if text is None:
        return None
    stripped = text.strip()
    if not stripped:
        return None
    return stripped


def _composer_data_key(composer_id: str) -> str:
    return f"composerData:{composer_id}"


def _bubble_data_key(composer_id: str, bubble_id: str) -> str:
    return f"bubbleId:{composer_id}:{bubble_id}"


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


def _read_state_values(
    state_db_path: Path,
    keys: tuple[str, ...],
) -> dict[str, object]:
    return _read_table_values(state_db_path, table_name="ItemTable", keys=keys)


def _read_cursor_disk_kv_values(
    state_db_path: Path,
    keys: tuple[str, ...],
) -> dict[str, object]:
    return _read_table_values(state_db_path, table_name="cursorDiskKV", keys=keys)


def _read_table_values(
    state_db_path: Path,
    *,
    table_name: str,
    keys: tuple[str, ...],
) -> dict[str, object]:
    if not state_db_path.is_file() or not keys:
        return {}

    rows = _query_sqlite_rows(
        state_db_path,
        "SELECT key, value FROM {table} WHERE key IN ({placeholders})".format(
            table=table_name,
            placeholders=",".join("?" for _ in keys),
        ),
        keys,
    )

    payload: dict[str, object] = {}
    for key, raw_value in rows:
        if not isinstance(key, str):
            continue
        payload[key] = _parse_json_sql_value(raw_value)
    return payload


def _parse_json_sql_value(raw_value: object) -> object:
    if isinstance(raw_value, memoryview):
        raw_value = raw_value.tobytes()
    if isinstance(raw_value, bytes):
        try:
            raw_text = raw_value.decode("utf-8")
        except UnicodeDecodeError:
            return raw_value
    elif isinstance(raw_value, str):
        raw_text = raw_value
    else:
        return raw_value

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        return raw_text


def _query_sqlite_rows(
    state_db_path: Path,
    query: str,
    parameters: tuple[object, ...],
) -> list[tuple[object, ...]]:
    try:
        return _query_sqlite_rows_once(state_db_path, query, parameters)
    except sqlite3.DatabaseError as error:
        if "no such table" in str(error).lower():
            return []

        copied_path = _copy_sqlite_database(state_db_path)
        if copied_path is None:
            return []
        try:
            return _query_sqlite_rows_once(copied_path, query, parameters)
        except sqlite3.DatabaseError:
            return []
        finally:
            copied_path.unlink(missing_ok=True)


def _query_sqlite_rows_once(
    state_db_path: Path,
    query: str,
    parameters: tuple[object, ...],
) -> list[tuple[object, ...]]:
    if not state_db_path.is_file():
        return []

    with sqlite3.connect(str(state_db_path)) as connection:
        rows = connection.execute(query, parameters).fetchall()
    return [tuple(row) for row in rows]


def _copy_sqlite_database(state_db_path: Path) -> Path | None:
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f"{state_db_path.stem}-",
            suffix=state_db_path.suffix,
            delete=False,
        ) as handle:
            copied_path = Path(handle.name)
        shutil.copyfile(state_db_path, copied_path)
    except OSError:
        return None
    return copied_path


def _read_item_keys(state_db_path: Path, keys: frozenset[str]) -> tuple[str, ...]:
    if not state_db_path.is_file():
        return ()

    rows = _query_sqlite_rows(
        state_db_path,
        "SELECT key FROM ItemTable WHERE key IN ({})".format(
            ",".join("?" for _ in keys)
        ),
        tuple(sorted(keys)),
    )

    return tuple(sorted(key for (key,) in rows if isinstance(key, str)))


def _count_hook_session_end_records(hook_log_path: Path) -> int:
    try:
        lines = hook_log_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return 0

    count = 0
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        if payload.get("hook_event_name") == "sessionEnd":
            count += 1
    return count


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


def _string_list(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in (_string_value(entry) for entry in value) if item is not None)


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


def _is_workspace_state_db(path: Path) -> bool:
    return (
        path.name == "state.vscdb"
        and path.parent.parent.name == "workspaceStorage"
        and "Cursor" in path.parts
    )


def _is_cursor_application_support_root(path: Path) -> bool:
    return path.name == "Cursor" and "Application Support" in path.parts


def _is_cursor_global_state_db(path: Path) -> bool:
    return path.name == "state.vscdb" and path.parent.name == "globalStorage"


def _is_cursor_hook_log(path: Path) -> bool:
    return path.name == "cursor.hooks.log"


def _is_cursor_tracking_db(path: Path) -> bool:
    return path.name == "ai-code-tracking.db" and "ai-tracking" in path.parts


__all__ = [
    "CURSOR_EDITOR_DESCRIPTOR",
    "CursorEditorCollector",
    "discover_cursor_auxiliary_artifacts",
    "iter_workspace_state_paths",
    "parse_workspace_state",
]
