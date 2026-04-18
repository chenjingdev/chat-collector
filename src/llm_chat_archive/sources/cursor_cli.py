from __future__ import annotations

import json
import sqlite3
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

CURSOR_CLI_DESCRIPTOR = SourceDescriptor(
    key="cursor",
    display_name="Cursor CLI",
    execution_context="cli",
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
        "Anchors collection on ~/Library/Application Support/Cursor/logs/<timestamp>/cli.log invocation metadata.",
        "Reconstructs transcript messages only when a CLI invocation can be uniquely attributed to shared workspace prompt metadata plus explicit cursorDiskKV bubble rows.",
        "Keeps aiService.prompts and aiService.generations as metadata-only workspace evidence when transcript body rows cannot be confirmed for a specific invocation.",
        "Keeps mcp.json, bridge sidecars, cli-config.json, ide_state.json, and workspace state paths in provenance or session metadata only.",
    ),
    support_metadata=SourceSupportMetadata(
        product_label="Cursor",
        host_surface="CLI",
        expected_transcript_completeness=TranscriptCompleteness.PARTIAL,
        limitation_summary="Cursor CLI still depends on shared editor transcript rows plus unique invocation attribution, so it remains partial and opt-in for unattended batches.",
        limitations=(
            "CLI invocations are promoted only when shared cursorDiskKV transcript rows can be uniquely attributed to a specific invocation.",
            "cli.log and workspace prompt caches alone remain partial or unsupported evidence.",
        ),
    ),
)

APPLICATION_SUPPORT_GLOB = "**/Application Support/Cursor"
BRIDGE_INSTRUCTIONS_GLOB = "**/.cursor/projects/*/mcps/*/INSTRUCTIONS.md"
BRIDGE_METADATA_GLOB = "**/.cursor/projects/*/mcps/*/SERVER_METADATA.json"
CLI_CONFIG_GLOB = "**/cli-config.json"
CLI_LOG_GLOB = "**/cli.log"
CURSOR_ROOT_GLOB = "**/.cursor"
GLOBAL_STATE_GLOB = "**/User/globalStorage/state.vscdb"
IDE_STATE_GLOB = "**/ide_state.json"
MCP_CONFIG_GLOB = "**/mcp.json"
WORKSPACE_STATE_GLOB = "**/User/workspaceStorage/*/state.vscdb"
PROMPT_EVIDENCE_MATCH_WINDOW_SECONDS = 10 * 60
TRANSCRIPT_ATTRIBUTION_PADDING_SECONDS = 2 * 60
PARTIAL_LIMITATIONS = (
    "cursor_cli_transcript_not_confirmed",
    "workspace_prompt_cache_only",
)
UNSUPPORTED_LIMITATIONS = (
    "cursor_cli_transcript_not_confirmed",
    "metadata_only_cli_invocation",
)
MISSING_ASSISTANT_LIMITATION = "assistant_body_missing_from_cursor_disk_kv"
MISSING_USER_LIMITATION = "user_body_missing_from_cursor_disk_kv"
CLI_LOG_RELEVANT_KEYS = frozenset(
    {
        "argv",
        "headless",
        "list-extensions",
        "logsPath",
        "show-versions",
        "status",
        "trace",
        "verbose",
    }
)


@dataclass(frozen=True, slots=True)
class CursorBridgeServer:
    source_path: str
    server_identifier: str | None = None
    server_name: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {"source_path": self.source_path}
        if self.server_identifier is not None:
            payload["server_identifier"] = self.server_identifier
        if self.server_name is not None:
            payload["server_name"] = self.server_name
        return payload


@dataclass(frozen=True, slots=True)
class CursorCliTranscriptRecovery:
    messages: tuple[NormalizedMessage, ...]
    completeness: TranscriptCompleteness
    limitations: tuple[str, ...]
    source_name: str
    source_path: str | None = None
    header_count: int = 0
    assistant_message_count: int = 0
    skipped_tool_bubble_count: int = 0


@dataclass(frozen=True, slots=True)
class CursorCliWorkspaceSession:
    workspace_id: str
    composer_id: str
    prompt_count: int
    generation_count: int
    partial_prompt_texts: tuple[str, ...]
    workspace_state_path: str
    activity_at: str | None = None
    workspace_folder: str | None = None
    composer_name: str | None = None
    composer_subtitle: str | None = None
    created_at: str | None = None
    last_updated_at: str | None = None
    selected: bool = False
    last_focused: bool = False
    prompt_overlap_count: int = 0
    transcript: CursorCliTranscriptRecovery | None = None

    def to_prompt_evidence_dict(self, *, match_strategy: str) -> dict[str, object]:
        payload: dict[str, object] = {
            "match_strategy": match_strategy,
            "workspace_id": self.workspace_id,
            "composer_id": self.composer_id,
            "prompt_count": self.prompt_count,
            "generation_count": self.generation_count,
            "partial_prompt_texts": list(self.partial_prompt_texts),
            "source_artifact_path": self.workspace_state_path,
        }
        if self.workspace_folder is not None:
            payload["workspace_folder"] = self.workspace_folder
        if self.composer_name is not None:
            payload["composer_name"] = self.composer_name
        if self.composer_subtitle is not None:
            payload["composer_subtitle"] = self.composer_subtitle
        if self.created_at is not None:
            payload["created_at"] = self.created_at
        if self.last_updated_at is not None:
            payload["last_updated_at"] = self.last_updated_at
        if self.activity_at is not None:
            payload["activity_at"] = self.activity_at
        if self.selected:
            payload["selected"] = True
        if self.last_focused:
            payload["last_focused"] = True
        return payload

    def to_transcript_attribution_dict(self) -> dict[str, object]:
        payload = self.to_prompt_evidence_dict(
            match_strategy="prompt_overlap_plus_time_window"
        )
        payload["prompt_overlap_count"] = self.prompt_overlap_count
        if self.transcript is not None:
            payload["transcript_source"] = self.transcript.source_name
            payload["transcript_header_count"] = self.transcript.header_count
            payload["assistant_message_count"] = self.transcript.assistant_message_count
            if self.transcript.skipped_tool_bubble_count:
                payload["skipped_tool_bubble_count"] = (
                    self.transcript.skipped_tool_bubble_count
                )
            if self.transcript.source_path is not None:
                payload["transcript_artifact_path"] = self.transcript.source_path
        return payload


@dataclass(frozen=True, slots=True)
class CursorCliInvocation:
    invocation_id: str
    log_path: str
    invoked_at: str | None = None
    logs_path: str | None = None
    headless: bool | None = None
    verbose: bool | None = None
    status: bool | None = None
    trace: bool | None = None
    list_extensions: bool | None = None
    show_versions: bool | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "invocation_id": self.invocation_id,
            "log_path": self.log_path,
        }
        if self.invoked_at is not None:
            payload["invoked_at"] = self.invoked_at
        if self.logs_path is not None:
            payload["logs_path"] = self.logs_path
        for key, value in (
            ("headless", self.headless),
            ("verbose", self.verbose),
            ("status", self.status),
            ("trace", self.trace),
            ("list_extensions", self.list_extensions),
            ("show_versions", self.show_versions),
        ):
            if value is not None:
                payload[key] = value
        return payload


@dataclass(frozen=True, slots=True)
class CursorCliArtifacts:
    cursor_root_paths: tuple[str, ...] = ()
    application_support_roots: tuple[str, ...] = ()
    cli_config_paths: tuple[str, ...] = ()
    global_state_paths: tuple[str, ...] = ()
    ide_state_paths: tuple[str, ...] = ()
    mcp_config_paths: tuple[str, ...] = ()
    bridge_metadata_paths: tuple[str, ...] = ()
    bridge_instruction_paths: tuple[str, ...] = ()
    workspace_state_paths: tuple[str, ...] = ()
    cli_log_paths: tuple[str, ...] = ()
    cli_config: dict[str, object] | None = None
    recent_file_paths: tuple[str, ...] = ()
    bridge_servers: tuple[CursorBridgeServer, ...] = ()
    workspace_sessions: tuple[CursorCliWorkspaceSession, ...] = ()

    def build_app_shell(self, *, log_path: str) -> AppShellProvenance | None:
        log_roots = tuple(
            sorted(
                {
                    str(Path(path).resolve(strict=False).parent.parent)
                    for path in self.cli_log_paths
                    if len(Path(path).parts) >= 2
                }
            )
        )
        provenance = AppShellProvenance(
            application_support_roots=self.application_support_roots,
            log_roots=log_roots,
            state_db_paths=tuple(
                sorted({*self.global_state_paths, *self.workspace_state_paths})
            ),
            log_paths=(log_path,),
            preference_paths=tuple(
                sorted({*self.cli_config_paths, *self.ide_state_paths})
            ),
            cache_roots=self.cursor_root_paths,
            auxiliary_paths=tuple(
                sorted(
                    {
                        *self.mcp_config_paths,
                        *self.bridge_metadata_paths,
                        *self.bridge_instruction_paths,
                    }
                )
            ),
        )
        if not provenance.to_dict():
            return None
        return provenance


@dataclass(frozen=True, slots=True)
class CursorCliCollector:
    descriptor: SourceDescriptor = CURSOR_CLI_DESCRIPTOR

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
        artifacts = discover_cursor_cli_artifacts(resolved_input_roots)
        cli_log_paths = tuple(iter_cli_log_paths(resolved_input_roots))
        collected_at = utc_timestamp()
        conversations = (
            parse_cli_log(
                cli_log_path,
                collected_at=collected_at,
                artifacts=artifacts,
            )
            for cli_log_path in cli_log_paths
        )
        return write_incremental_collection(
            source=self.descriptor.key,
            archive_root=archive_root,
            input_roots=resolved_input_roots,
            scanned_artifact_count=len(cli_log_paths),
            collected_at=collected_at,
            conversations=conversations,
        )

    def _default_input_roots(self) -> tuple[Path, ...]:
        return default_descriptor_input_roots(self.descriptor)


def discover_cursor_cli_artifacts(
    input_roots: tuple[Path, ...] | None,
) -> CursorCliArtifacts:
    if not input_roots:
        return CursorCliArtifacts()

    cursor_root_paths = _discover_paths(
        input_roots,
        direct_match=_is_cursor_root,
        glob_pattern=CURSOR_ROOT_GLOB,
        expect_dir=True,
    )
    application_support_roots = _discover_paths(
        input_roots,
        direct_match=_is_cursor_application_support_root,
        glob_pattern=APPLICATION_SUPPORT_GLOB,
        expect_dir=True,
    )
    cli_config_paths = _discover_paths(
        input_roots,
        direct_match=_is_cli_config,
        glob_pattern=CLI_CONFIG_GLOB,
        expect_dir=False,
    )
    global_state_paths = _discover_paths(
        input_roots,
        direct_match=_is_cursor_global_state_db,
        glob_pattern=GLOBAL_STATE_GLOB,
        expect_dir=False,
    )
    ide_state_paths = _discover_paths(
        input_roots,
        direct_match=_is_ide_state,
        glob_pattern=IDE_STATE_GLOB,
        expect_dir=False,
    )
    mcp_config_paths = _discover_paths(
        input_roots,
        direct_match=_is_mcp_config,
        glob_pattern=MCP_CONFIG_GLOB,
        expect_dir=False,
    )
    bridge_metadata_paths = _discover_paths(
        input_roots,
        direct_match=_is_bridge_metadata,
        glob_pattern=BRIDGE_METADATA_GLOB,
        expect_dir=False,
    )
    bridge_instruction_paths = _discover_paths(
        input_roots,
        direct_match=_is_bridge_instruction,
        glob_pattern=BRIDGE_INSTRUCTIONS_GLOB,
        expect_dir=False,
    )
    workspace_state_paths = tuple(str(path) for path in iter_workspace_state_paths(input_roots))
    cli_log_paths = tuple(str(path) for path in iter_cli_log_paths(input_roots))

    return CursorCliArtifacts(
        cursor_root_paths=cursor_root_paths,
        application_support_roots=application_support_roots,
        cli_config_paths=cli_config_paths,
        global_state_paths=global_state_paths,
        ide_state_paths=ide_state_paths,
        mcp_config_paths=mcp_config_paths,
        bridge_metadata_paths=bridge_metadata_paths,
        bridge_instruction_paths=bridge_instruction_paths,
        workspace_state_paths=workspace_state_paths,
        cli_log_paths=cli_log_paths,
        cli_config=_read_cli_config(cli_config_paths),
        recent_file_paths=_read_recent_file_paths(ide_state_paths),
        bridge_servers=_read_bridge_servers(bridge_metadata_paths),
        workspace_sessions=_discover_workspace_sessions(
            workspace_state_paths,
            global_state_paths=global_state_paths,
        ),
    )


def parse_cli_log(
    cli_log_path: Path,
    *,
    collected_at: str | None = None,
    artifacts: CursorCliArtifacts | None = None,
) -> NormalizedConversation | None:
    resolved_path = cli_log_path.expanduser().resolve(strict=False)
    invocation = _read_cli_log_invocation(resolved_path)

    discovery = artifacts or discover_cursor_cli_artifacts((resolved_path.parent.parent.parent,))
    transcript_session = _match_transcript_session(
        invocation.invoked_at,
        discovery.workspace_sessions,
    )
    prompt_session = None
    messages: tuple[NormalizedMessage, ...] = ()
    transcript_completeness = TranscriptCompleteness.UNSUPPORTED
    limitations = UNSUPPORTED_LIMITATIONS
    cwd = None

    if transcript_session is not None and transcript_session.transcript is not None:
        messages = transcript_session.transcript.messages
        transcript_completeness = transcript_session.transcript.completeness
        limitations = transcript_session.transcript.limitations
        cwd = transcript_session.workspace_folder
    else:
        prompt_session = _match_prompt_only_session(
            invocation.invoked_at,
            discovery.workspace_sessions,
        )
        if prompt_session is not None:
            transcript_completeness = TranscriptCompleteness.PARTIAL
            limitations = PARTIAL_LIMITATIONS

    session_metadata: dict[str, object] = {
        "invocation": invocation.to_dict(),
        "workspace_state_count": len(discovery.workspace_state_paths),
    }
    if discovery.global_state_paths:
        session_metadata["global_state_count"] = len(discovery.global_state_paths)
    if discovery.cli_config is not None:
        session_metadata["cli_config"] = discovery.cli_config
    if discovery.recent_file_paths:
        session_metadata["recently_viewed_files"] = list(discovery.recent_file_paths)
    if discovery.bridge_servers:
        session_metadata["bridge_servers"] = [
            bridge_server.to_dict() for bridge_server in discovery.bridge_servers
        ]
    if discovery.mcp_config_paths:
        session_metadata["has_mcp_config"] = True
    if transcript_session is not None and transcript_session.transcript is not None:
        session_metadata["transcript_attribution"] = (
            transcript_session.to_transcript_attribution_dict()
        )
    elif prompt_session is not None:
        session_metadata["workspace_prompt_evidence"] = prompt_session.to_prompt_evidence_dict(
            match_strategy="time_proximity_metadata_only"
        )

    return NormalizedConversation(
        source=CURSOR_CLI_DESCRIPTOR.key,
        execution_context=CURSOR_CLI_DESCRIPTOR.execution_context,
        collected_at=collected_at or utc_timestamp(),
        messages=messages,
        transcript_completeness=transcript_completeness,
        limitations=limitations,
        source_session_id=invocation.invocation_id,
        source_artifact_path=str(resolved_path),
        session_metadata=session_metadata,
        provenance=ConversationProvenance(
            session_started_at=invocation.invoked_at,
            source="cli",
            originator="cursor_cli",
            cwd=cwd,
            app_shell=discovery.build_app_shell(log_path=str(resolved_path)),
        ),
    )


def iter_cli_log_paths(input_roots: Iterable[Path]) -> Iterable[Path]:
    seen: set[Path] = set()
    candidates: list[Path] = []
    for input_root in input_roots:
        matches: list[Path] = []
        if _is_cli_log(input_root):
            matches.append(input_root)
        if input_root.is_dir():
            matches.extend(input_root.glob(CLI_LOG_GLOB))

        for candidate in matches:
            if not candidate.is_file():
                continue
            resolved = candidate.resolve(strict=False)
            if resolved in seen:
                continue
            seen.add(resolved)
            candidates.append(resolved)
    yield from sorted(candidates)


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


def _discover_workspace_sessions(
    workspace_state_paths: tuple[str, ...],
    *,
    global_state_paths: tuple[str, ...],
) -> tuple[CursorCliWorkspaceSession, ...]:
    sessions: list[CursorCliWorkspaceSession] = []
    for raw_path in workspace_state_paths:
        state_db_path = Path(raw_path)
        state_values = _read_state_values(
            state_db_path,
            (
                "composer.composerData",
                "aiService.prompts",
                "aiService.generations",
            ),
        )
        composer_payload = state_values.get("composer.composerData")
        prompts_payload = state_values.get("aiService.prompts")
        generations_payload = state_values.get("aiService.generations")

        if not isinstance(composer_payload, dict) or not isinstance(prompts_payload, list):
            continue

        composer = _select_composer_summary(composer_payload)
        if composer is None:
            continue

        prompt_texts = tuple(
            prompt_text
            for prompt_text in (
                _clean_prompt_text(prompt.get("text")) if isinstance(prompt, dict) else None
                for prompt in prompts_payload
            )
            if prompt_text is not None
        )
        if not prompt_texts:
            continue

        generation_timestamps = _generation_timestamps(generations_payload)
        activity_at = next(
            (timestamp for timestamp in reversed(generation_timestamps) if timestamp is not None),
            None,
        )
        if activity_at is None:
            activity_at = composer["last_updated_at"] or composer["created_at"]

        transcript = _read_cursor_disk_kv_transcript(
            composer_id=composer["composer_id"],
            global_state_paths=global_state_paths,
            generation_timestamps=generation_timestamps,
        )
        prompt_overlap_count = _prompt_overlap_count(
            prompt_texts,
            transcript.messages if transcript is not None else (),
        )

        sessions.append(
            CursorCliWorkspaceSession(
                workspace_id=state_db_path.parent.name,
                composer_id=composer["composer_id"],
                prompt_count=len(prompt_texts),
                generation_count=len(generation_timestamps),
                partial_prompt_texts=prompt_texts,
                workspace_state_path=str(state_db_path.resolve(strict=False)),
                activity_at=activity_at,
                workspace_folder=_read_workspace_folder(state_db_path.parent / "workspace.json"),
                composer_name=composer["composer_name"],
                composer_subtitle=composer["composer_subtitle"],
                created_at=composer["created_at"],
                last_updated_at=composer["last_updated_at"],
                selected=bool(composer["selected"]),
                last_focused=bool(composer["last_focused"]),
                prompt_overlap_count=prompt_overlap_count,
                transcript=transcript,
            )
        )

    sessions.sort(key=lambda row: (row.activity_at or "", row.workspace_id, row.composer_id))
    return tuple(sessions)


def _match_transcript_session(
    invoked_at: str | None,
    workspace_sessions: tuple[CursorCliWorkspaceSession, ...],
) -> CursorCliWorkspaceSession | None:
    invoked_at_dt = _parse_iso_timestamp(invoked_at)
    if invoked_at_dt is None:
        return None

    candidates: list[tuple[tuple[int, float, float], CursorCliWorkspaceSession]] = []
    for session in workspace_sessions:
        transcript = session.transcript
        if transcript is None or not transcript.messages:
            continue
        if not _has_confirmed_prompt_overlap(session):
            continue
        score = _transcript_match_score(invoked_at_dt, session)
        if score is None:
            continue
        candidates.append((score, session))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], item[1].workspace_id, item[1].composer_id))
    if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
        return None
    return candidates[0][1]


def _match_prompt_only_session(
    invoked_at: str | None,
    workspace_sessions: tuple[CursorCliWorkspaceSession, ...],
) -> CursorCliWorkspaceSession | None:
    invoked_at_dt = _parse_iso_timestamp(invoked_at)
    if invoked_at_dt is None:
        return None

    candidates: list[tuple[float, CursorCliWorkspaceSession]] = []
    for session in workspace_sessions:
        if not session.partial_prompt_texts:
            continue
        activity_at_dt = _parse_iso_timestamp(session.activity_at)
        if activity_at_dt is None:
            continue

        delta_seconds = abs((activity_at_dt - invoked_at_dt).total_seconds())
        if delta_seconds > PROMPT_EVIDENCE_MATCH_WINDOW_SECONDS:
            continue
        candidates.append((delta_seconds, session))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], item[1].workspace_id, item[1].composer_id))
    if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
        return None
    return candidates[0][1]


def _read_cli_log_invocation(cli_log_path: Path) -> CursorCliInvocation:
    payload = _read_first_cli_log_payload(cli_log_path)
    logs_path = _string_value(payload.get("logsPath"))
    invocation_id = _invocation_id(cli_log_path, logs_path)
    invoked_at = _normalize_cli_log_timestamp(cli_log_path, logs_path)

    return CursorCliInvocation(
        invocation_id=invocation_id,
        log_path=str(cli_log_path),
        invoked_at=invoked_at,
        logs_path=logs_path,
        headless=_bool_value(payload.get("headless")),
        verbose=_bool_value(payload.get("verbose")),
        status=_bool_value(payload.get("status")),
        trace=_bool_value(payload.get("trace")),
        list_extensions=_bool_value(payload.get("list-extensions")),
        show_versions=_bool_value(payload.get("show-versions")),
    )


def _read_first_cli_log_payload(cli_log_path: Path) -> dict[str, object]:
    try:
        lines = cli_log_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return {}

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
        if any(key in payload for key in CLI_LOG_RELEVANT_KEYS):
            return payload
    return {}


def _read_cli_config(paths: tuple[str, ...]) -> dict[str, object] | None:
    for raw_path in paths:
        payload = _load_json_file(Path(raw_path))
        if not isinstance(payload, dict):
            continue

        result: dict[str, object] = {}
        version = _string_value(payload.get("version"))
        editor = _string_value(payload.get("editor"))
        has_changed_default_model = _bool_value(payload.get("hasChangedDefaultModel"))

        if version is not None:
            result["version"] = version
        if editor is not None:
            result["editor"] = editor
        if has_changed_default_model is not None:
            result["has_changed_default_model"] = has_changed_default_model
        for key in ("permissions", "network"):
            value = payload.get(key)
            if isinstance(value, (dict, list, str, int, float, bool)) or value is None:
                result[key] = value
        if result:
            return result
    return None


def _read_recent_file_paths(paths: tuple[str, ...]) -> tuple[str, ...]:
    recent_files: list[str] = []
    seen: set[str] = set()
    for raw_path in paths:
        payload = _load_json_file(Path(raw_path))
        if not isinstance(payload, dict):
            continue
        entries = payload.get("recentlyViewedFiles")
        if not isinstance(entries, list):
            continue

        for entry in entries:
            if not isinstance(entry, dict):
                continue
            file_path = _string_value(entry.get("absolutePath"))
            if file_path is None:
                file_path = _string_value(entry.get("relativePath"))
            if file_path is None or file_path in seen:
                continue
            seen.add(file_path)
            recent_files.append(file_path)

    return tuple(sorted(recent_files))


def _read_bridge_servers(paths: tuple[str, ...]) -> tuple[CursorBridgeServer, ...]:
    bridge_servers: list[CursorBridgeServer] = []
    for raw_path in paths:
        payload = _load_json_file(Path(raw_path))
        if not isinstance(payload, dict):
            continue
        bridge_servers.append(
            CursorBridgeServer(
                source_path=str(Path(raw_path).resolve(strict=False)),
                server_identifier=_string_value(payload.get("serverIdentifier")),
                server_name=_string_value(payload.get("serverName")),
            )
        )

    bridge_servers.sort(key=lambda bridge_server: bridge_server.source_path)
    return tuple(bridge_servers)


def _load_json_file(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


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

    try:
        with sqlite3.connect(str(state_db_path)) as connection:
            rows = connection.execute(
                "SELECT key, value FROM {table} WHERE key IN ({placeholders})".format(
                    table=table_name,
                    placeholders=",".join("?" for _ in keys),
                ),
                keys,
            ).fetchall()
    except sqlite3.DatabaseError:
        return {}

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


def _select_composer_summary(
    composer_payload: dict[str, object],
) -> dict[str, str | bool | None] | None:
    raw_composers = composer_payload.get("allComposers")
    if not isinstance(raw_composers, list):
        return None

    composers: dict[str, dict[str, str | bool | None]] = {}
    for raw_composer in raw_composers:
        if not isinstance(raw_composer, dict):
            continue
        composer_id = _string_value(raw_composer.get("composerId"))
        if composer_id is None:
            continue
        composers[composer_id] = {
            "composer_id": composer_id,
            "composer_name": _string_value(raw_composer.get("name")),
            "composer_subtitle": _string_value(raw_composer.get("subtitle")),
            "created_at": _normalize_timestamp(raw_composer.get("createdAt")),
            "last_updated_at": _normalize_timestamp(raw_composer.get("lastUpdatedAt")),
            "selected": False,
            "last_focused": False,
        }

    if not composers:
        return None

    selected_ids = _string_list(composer_payload.get("selectedComposerIds"))
    last_focused_ids = _string_list(composer_payload.get("lastFocusedComposerIds"))
    for composer_id in selected_ids:
        composer = composers.get(composer_id)
        if composer is not None:
            composer["selected"] = True
    for composer_id in last_focused_ids:
        composer = composers.get(composer_id)
        if composer is not None:
            composer["last_focused"] = True
    for composer_id in (*last_focused_ids, *selected_ids):
        composer = composers.get(composer_id)
        if composer is not None:
            return composer

    if len(composers) != 1:
        return None
    return next(iter(composers.values()))


def _read_cursor_disk_kv_transcript(
    *,
    composer_id: str,
    global_state_paths: tuple[str, ...],
    generation_timestamps: list[str | None],
) -> CursorCliTranscriptRecovery | None:
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
) -> CursorCliTranscriptRecovery | None:
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
    missing_assistant_count = 0
    missing_user_count = 0
    skipped_tool_bubble_count = 0

    for bubble_id, role in ordered_bubbles:
        bubble_payload = bubble_values.get(_bubble_data_key(composer_id, bubble_id))
        if not isinstance(bubble_payload, dict):
            if role == MessageRole.ASSISTANT:
                missing_assistant_count += 1
            else:
                missing_user_count += 1
            continue

        text = _clean_prompt_text(bubble_payload.get("text"))
        if text is None:
            if role == MessageRole.ASSISTANT and _is_tool_only_bubble(bubble_payload):
                skipped_tool_bubble_count += 1
                continue
            if role == MessageRole.ASSISTANT:
                missing_assistant_count += 1
            else:
                missing_user_count += 1
            continue

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
            )
        )

    limitations: list[str] = []
    if missing_user_count:
        limitations.append(MISSING_USER_LIMITATION)
    if missing_assistant_count:
        limitations.append(MISSING_ASSISTANT_LIMITATION)

    completeness = (
        TranscriptCompleteness.COMPLETE
        if assistant_message_count > 0 and not limitations
        else TranscriptCompleteness.PARTIAL
    )
    return CursorCliTranscriptRecovery(
        messages=tuple(messages),
        completeness=completeness,
        limitations=tuple(limitations),
        source_name="cursor_disk_kv",
        source_path=str(global_state_path),
        header_count=len(ordered_bubbles),
        assistant_message_count=assistant_message_count,
        skipped_tool_bubble_count=skipped_tool_bubble_count,
    )


def _prompt_overlap_count(
    prompt_texts: tuple[str, ...],
    messages: tuple[NormalizedMessage, ...],
) -> int:
    user_texts = {
        message.text.strip()
        for message in messages
        if message.role == MessageRole.USER and message.text.strip()
    }
    return sum(1 for prompt_text in prompt_texts if prompt_text in user_texts)


def _has_confirmed_prompt_overlap(session: CursorCliWorkspaceSession) -> bool:
    return session.prompt_count > 0 and session.prompt_overlap_count == session.prompt_count


def _transcript_match_score(
    invoked_at_dt: datetime,
    session: CursorCliWorkspaceSession,
) -> tuple[int, float, float] | None:
    start_dt = _parse_iso_timestamp(session.created_at)
    end_dt = _parse_iso_timestamp(session.activity_at) or _parse_iso_timestamp(
        session.last_updated_at
    )
    if start_dt is None and end_dt is None:
        return None

    if start_dt is None:
        start_dt = end_dt
    if end_dt is None:
        end_dt = start_dt
    if start_dt is None or end_dt is None:
        return None
    if end_dt < start_dt:
        end_dt = start_dt

    padded_start = start_dt.timestamp() - TRANSCRIPT_ATTRIBUTION_PADDING_SECONDS
    padded_end = end_dt.timestamp() + TRANSCRIPT_ATTRIBUTION_PADDING_SECONDS
    invoked_ts = invoked_at_dt.timestamp()
    if invoked_ts < padded_start or invoked_ts > padded_end:
        return None

    contains_invocation = 0 if start_dt <= invoked_at_dt <= end_dt else 1
    anchor_dt = _parse_iso_timestamp(session.activity_at) or end_dt
    return (
        contains_invocation,
        abs((invoked_at_dt - start_dt).total_seconds()),
        abs((anchor_dt - invoked_at_dt).total_seconds()),
    )


def _composer_data_key(composer_id: str) -> str:
    return f"composerData:{composer_id}"


def _bubble_data_key(composer_id: str, bubble_id: str) -> str:
    return f"bubbleId:{composer_id}:{bubble_id}"


def _is_tool_only_bubble(bubble_payload: dict[str, object]) -> bool:
    tool_former_data = bubble_payload.get("toolFormerData")
    if isinstance(tool_former_data, dict):
        return bool(tool_former_data)
    if isinstance(tool_former_data, list):
        return bool(tool_former_data)
    return False


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


def _read_workspace_folder(workspace_json_path: Path) -> str | None:
    payload = _load_json_file(workspace_json_path)
    if not isinstance(payload, dict):
        return None

    folder_uri = _string_value(payload.get("folder"))
    if folder_uri is None:
        return None
    if folder_uri.startswith("file://"):
        parsed = urlparse(folder_uri)
        return unquote(parsed.path) or None
    return folder_uri


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


def _invocation_id(cli_log_path: Path, logs_path: str | None) -> str:
    if logs_path is not None:
        candidate = Path(logs_path).name
        if candidate:
            return candidate
    return cli_log_path.parent.name or cli_log_path.stem


def _normalize_cli_log_timestamp(cli_log_path: Path, logs_path: str | None) -> str | None:
    for candidate in (
        Path(logs_path).name if logs_path is not None else None,
        cli_log_path.parent.name,
    ):
        if not candidate:
            continue
        parsed = _parse_logs_slug(candidate)
        if parsed is not None:
            return parsed
    return None


def _parse_logs_slug(value: str) -> str | None:
    try:
        timestamp = datetime.strptime(value, "%Y%m%dT%H%M%S")
    except ValueError:
        return None
    return timestamp.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_iso_timestamp(value: str | None) -> datetime | None:
    if value is None:
        return None
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


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


def _clean_prompt_text(value: object) -> str | None:
    text = _string_value(value)
    if text is None:
        return None
    stripped = text.strip()
    if not stripped:
        return None
    return stripped


def _string_list(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in (_string_value(entry) for entry in value) if item is not None)


def _string_value(value: object) -> str | None:
    if isinstance(value, str):
        return value
    return None


def _bool_value(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _int_value(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _is_cursor_root(path: Path) -> bool:
    return path.name == ".cursor"


def _is_cursor_application_support_root(path: Path) -> bool:
    return path.name == "Cursor" and "Application Support" in path.parts


def _is_cli_config(path: Path) -> bool:
    return path.name == "cli-config.json"


def _is_cursor_global_state_db(path: Path) -> bool:
    return (
        path.name == "state.vscdb"
        and path.parent.parent.name == "globalStorage"
        and "Cursor" in path.parts
    )


def _is_cli_log(path: Path) -> bool:
    return path.name == "cli.log"


def _is_ide_state(path: Path) -> bool:
    return path.name == "ide_state.json"


def _is_mcp_config(path: Path) -> bool:
    return path.name == "mcp.json"


def _is_bridge_metadata(path: Path) -> bool:
    return path.name == "SERVER_METADATA.json" and "mcps" in path.parts


def _is_bridge_instruction(path: Path) -> bool:
    return path.name == "INSTRUCTIONS.md" and "mcps" in path.parts


def _is_workspace_state_db(path: Path) -> bool:
    return (
        path.name == "state.vscdb"
        and path.parent.parent.name == "workspaceStorage"
        and "Cursor" in path.parts
    )


__all__ = [
    "CURSOR_CLI_DESCRIPTOR",
    "CursorCliCollector",
    "CursorCliArtifacts",
    "discover_cursor_cli_artifacts",
    "iter_cli_log_paths",
    "parse_cli_log",
]
