from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

from ..incremental import write_incremental_collection
from ..models import (
    CollectionPlan,
    CollectionResult,
    IdeBridgeProvenance,
    NormalizedConversation,
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
from .claude_code_cli import (
    MAIN_TRANSCRIPT_ROW_TYPES,
    ClaudeTranscriptMetadata,
    _is_subagent_path,
    _load_json_line,
    _normalize_message,
    _string_value,
    iter_transcript_paths,
)
from .codex_rollout import resolve_input_roots, utc_timestamp

IDE_COMMAND_MARKER = "<command-name>/ide</command-name>"
IDE_CONFIG_KEYS = frozenset(
    {
        "autoConnectIde",
        "officialMarketplaceAutoInstallAttempted",
        "officialMarketplaceAutoInstalled",
        "shiftEnterKeyBindingInstalled",
        "optionAsMetaKeyInstalled",
    }
)
LOCAL_COMMAND_WRAPPERS = (
    ("<local-command-stdout>", "</local-command-stdout>"),
    ("<local-command-caveat>", "</local-command-caveat>"),
)
LOG_MARKERS = ("keybindings.json", "marketplace", "auto-install")

CLAUDE_CODE_IDE_DESCRIPTOR = SourceDescriptor(
    key="claude_code_ide",
    display_name="Claude Code IDE Bridge",
    execution_context="ide_bridge",
    support_level=SupportLevel.COMPLETE,
    default_input_roots=(
        "~/.claude",
        "~/.claude.json",
        "~/Library/Application Support/Code/User/globalStorage",
        "~/Library/Application Support/Cursor/User/globalStorage",
    ),
    artifact_root_candidates=(
        all_platform_root("$HOME/.claude"),
        all_platform_root("$HOME/.claude.json"),
        darwin_root("$HOME/Library/Application Support/Code/User/globalStorage"),
        linux_root("$XDG_CONFIG_HOME/Code/User/globalStorage"),
        windows_root("$APPDATA/Code/User/globalStorage"),
        darwin_root("$HOME/Library/Application Support/Cursor/User/globalStorage"),
        linux_root("$XDG_CONFIG_HOME/Cursor/User/globalStorage"),
        windows_root("$APPDATA/Cursor/User/globalStorage"),
    ),
    notes=(
        "Uses shared ~/.claude/projects/<encoded-project-path>/<session-uuid>.jsonl as the canonical transcript source.",
        "Selects only IDE-attached sessions using /ide history rows or explicit /ide bridge markers in the shared session JSONL.",
        "Treats ~/.claude/history.jsonl, ~/.claude.json, ~/.claude/keybindings.json, IDE recent-file residue, and Claude debug logs as provenance only.",
    ),
    support_metadata=SourceSupportMetadata(
        product_label="Claude Code",
        host_surface="IDE bridge",
        expected_transcript_completeness=TranscriptCompleteness.COMPLETE,
        limitation_summary="IDE bridge metadata stays provenance-only; the shared Claude session JSONL remains canonical.",
    ),
)


@dataclass(frozen=True, slots=True)
class ClaudeIdeDiscovery:
    history_session_ids: frozenset[str] = frozenset()
    hosts: tuple[str, ...] = ()
    config_paths: tuple[str, ...] = ()
    history_paths: tuple[str, ...] = ()
    keybinding_paths: tuple[str, ...] = ()
    log_paths: tuple[str, ...] = ()
    recent_file_paths: tuple[str, ...] = ()

    def build_provenance(self) -> IdeBridgeProvenance | None:
        provenance = IdeBridgeProvenance(
            hosts=self.hosts,
            config_paths=self.config_paths,
            history_paths=self.history_paths,
            keybinding_paths=self.keybinding_paths,
            log_paths=self.log_paths,
            recent_file_paths=self.recent_file_paths,
        )
        if not provenance.to_dict():
            return None
        return provenance


@dataclass(frozen=True, slots=True)
class ClaudeCodeIdeCollector:
    descriptor: SourceDescriptor = CLAUDE_CODE_IDE_DESCRIPTOR

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
        transcript_paths = tuple(iter_transcript_paths(resolved_input_roots))
        discovery = discover_ide_bridge_provenance(resolved_input_roots)
        collected_at = utc_timestamp()
        conversations = (
            parse_transcript_file(
                transcript_path,
                collected_at=collected_at,
                discovery=discovery,
            )
            for transcript_path in transcript_paths
        )
        return write_incremental_collection(
            source=self.descriptor.key,
            archive_root=archive_root,
            input_roots=resolved_input_roots,
            scanned_artifact_count=len(transcript_paths),
            collected_at=collected_at,
            conversations=conversations,
        )

    def _default_input_roots(self) -> tuple[Path, ...]:
        return default_descriptor_input_roots(self.descriptor)


def parse_transcript_file(
    transcript_path: Path,
    *,
    collected_at: str | None = None,
    discovery: ClaudeIdeDiscovery | None = None,
) -> NormalizedConversation | None:
    resolved_path = transcript_path.expanduser().resolve(strict=False)
    metadata = ClaudeTranscriptMetadata(subagent=_is_subagent_path(resolved_path))
    messages = []
    has_ide_command = False

    try:
        lines = resolved_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return None

    for raw_line in lines:
        if IDE_COMMAND_MARKER in raw_line:
            has_ide_command = True

        record = _load_json_line(raw_line)
        if record is None:
            continue

        metadata.observe(record)

        record_type = _string_value(record.get("type"))
        if record_type == "system" and _is_ide_local_command(record):
            has_ide_command = True

        if record_type not in MAIN_TRANSCRIPT_ROW_TYPES:
            continue

        message = record.get("message")
        if not isinstance(message, dict):
            continue

        sanitized_message = _sanitize_message(message)
        normalized_message = _normalize_message(record_type, sanitized_message, record)
        if normalized_message is None:
            continue
        messages.append(normalized_message)

    if not messages:
        return None

    session_lookup_id = _session_lookup_id(resolved_path, metadata)
    history_session_ids = discovery.history_session_ids if discovery is not None else frozenset()
    if session_lookup_id not in history_session_ids and not has_ide_command:
        return None

    provenance = metadata.build_provenance()
    if discovery is not None:
        provenance = replace(provenance, ide_bridge=discovery.build_provenance())

    return NormalizedConversation(
        source=CLAUDE_CODE_IDE_DESCRIPTOR.key,
        execution_context=CLAUDE_CODE_IDE_DESCRIPTOR.execution_context,
        collected_at=collected_at or utc_timestamp(),
        messages=tuple(messages),
        source_session_id=metadata.build_source_session_id(resolved_path),
        source_artifact_path=str(resolved_path),
        provenance=provenance,
    )


def discover_ide_bridge_provenance(
    input_roots: tuple[Path, ...] | None,
) -> ClaudeIdeDiscovery:
    if not input_roots:
        return ClaudeIdeDiscovery()

    history_session_ids: set[str] = set()
    hosts: set[str] = set()
    config_paths: set[str] = set()
    history_paths: set[str] = set()
    keybinding_paths: set[str] = set()
    log_paths: set[str] = set()
    recent_file_paths: set[str] = set()

    for config_path in _iter_named_artifacts(input_roots, ".claude.json"):
        if _config_has_ide_flags(config_path):
            config_paths.add(str(config_path))

    for history_path in _iter_named_artifacts(input_roots, "history.jsonl"):
        matched_session_ids = _read_ide_history_session_ids(history_path)
        if not matched_session_ids:
            continue
        history_paths.add(str(history_path))
        history_session_ids.update(matched_session_ids)

    for keybindings_path in _iter_named_artifacts(input_roots, "keybindings.json"):
        keybinding_paths.add(str(keybindings_path))

    for log_path in _iter_named_artifacts(input_roots, "*.txt"):
        if "debug" not in log_path.parts or not _log_contains_ide_markers(log_path):
            continue
        log_paths.add(str(log_path))

    for storage_path in _iter_named_artifacts(input_roots, "storage.json"):
        residue_paths = _extract_recent_file_residue(storage_path)
        if not residue_paths:
            continue
        recent_file_paths.update(residue_paths)
        host = _infer_host(storage_path)
        if host is not None:
            hosts.add(host)

    return ClaudeIdeDiscovery(
        history_session_ids=frozenset(history_session_ids),
        hosts=tuple(sorted(hosts)),
        config_paths=tuple(sorted(config_paths)),
        history_paths=tuple(sorted(history_paths)),
        keybinding_paths=tuple(sorted(keybinding_paths)),
        log_paths=tuple(sorted(log_paths)),
        recent_file_paths=tuple(sorted(recent_file_paths)),
    )


def _iter_named_artifacts(input_roots: tuple[Path, ...], pattern: str) -> tuple[Path, ...]:
    seen: set[Path] = set()
    candidates: list[Path] = []
    for input_root in input_roots:
        if input_root.is_file():
            resolved = input_root.resolve(strict=False)
            if resolved in seen or not _matches_pattern(input_root, pattern):
                continue
            seen.add(resolved)
            candidates.append(resolved)
            continue

        if not input_root.is_dir():
            continue

        for candidate in input_root.rglob(pattern):
            if not candidate.is_file():
                continue
            resolved = candidate.resolve(strict=False)
            if resolved in seen:
                continue
            seen.add(resolved)
            candidates.append(resolved)
    return tuple(sorted(candidates))


def _matches_pattern(path: Path, pattern: str) -> bool:
    if "*" in pattern:
        return path.match(pattern)
    return path.name == pattern


def _config_has_ide_flags(config_path: Path) -> bool:
    payload = _load_json_file(config_path)
    if not isinstance(payload, dict):
        return False
    return any(payload.get(key) is True for key in IDE_CONFIG_KEYS)


def _read_ide_history_session_ids(history_path: Path) -> set[str]:
    session_ids: set[str] = set()
    try:
        lines = history_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return session_ids

    for raw_line in lines:
        record = _load_json_line(raw_line)
        if record is None:
            continue
        display = _string_value(record.get("display"))
        session_id = _string_value(record.get("sessionId"))
        if display is None or session_id is None:
            continue
        if display.lstrip().startswith("/ide"):
            session_ids.add(session_id)
    return session_ids


def _log_contains_ide_markers(log_path: Path) -> bool:
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except FileNotFoundError:
        return False
    lowered = text.lower()
    return any(marker in lowered for marker in LOG_MARKERS)


def _extract_recent_file_residue(storage_path: Path) -> tuple[str, ...]:
    payload = _load_json_file(storage_path)
    if payload is None:
        return ()

    residue_paths: set[str] = set()
    for value in _iter_string_values(payload):
        if ".claude/keybindings.json" in value:
            residue_paths.add(value)
            continue
        if "claude-prompt-" in value and value.endswith(".md"):
            residue_paths.add(value)

    return tuple(sorted(residue_paths))


def _iter_string_values(payload: object):
    if isinstance(payload, str):
        yield payload
        return
    if isinstance(payload, list):
        for item in payload:
            yield from _iter_string_values(item)
        return
    if isinstance(payload, dict):
        for value in payload.values():
            yield from _iter_string_values(value)


def _sanitize_message(message: dict[str, object]) -> dict[str, object]:
    sanitized = dict(message)
    sanitized["content"] = _sanitize_content(message.get("content"))
    return sanitized


def _sanitize_content(content: object) -> object:
    if isinstance(content, str):
        return None if _is_local_command_wrapper(content) else content

    if not isinstance(content, list):
        return content

    items: list[object] = []
    for item in content:
        if isinstance(item, str):
            if not _is_local_command_wrapper(item):
                items.append(item)
            continue
        if not isinstance(item, dict):
            items.append(item)
            continue

        item_text = _string_value(item.get("text"))
        if item_text is not None and _is_local_command_wrapper(item_text):
            continue
        items.append(item)
    return items


def _is_local_command_wrapper(text: str) -> bool:
    stripped = text.strip()
    return any(
        stripped.startswith(prefix) and stripped.endswith(suffix)
        for prefix, suffix in LOCAL_COMMAND_WRAPPERS
    )


def _is_ide_local_command(record: dict[str, object]) -> bool:
    if _string_value(record.get("subtype")) != "local_command":
        return False
    message = record.get("message")
    if not isinstance(message, dict):
        return False
    content = message.get("content")
    if isinstance(content, str):
        return IDE_COMMAND_MARKER in content
    if not isinstance(content, list):
        return False
    return any(isinstance(item, str) and IDE_COMMAND_MARKER in item for item in content)


def _session_lookup_id(
    transcript_path: Path, metadata: ClaudeTranscriptMetadata
) -> str | None:
    if metadata.subagent:
        return transcript_path.parents[1].name
    return metadata.session_id or transcript_path.stem


def _infer_host(path: Path) -> str | None:
    path_str = str(path)
    if "/Application Support/Cursor/" in path_str:
        return "cursor"
    if "/Application Support/Code/" in path_str:
        return "vscode"
    return None


def _load_json_file(path: Path) -> object | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


__all__ = [
    "CLAUDE_CODE_IDE_DESCRIPTOR",
    "ClaudeCodeIdeCollector",
    "ClaudeIdeDiscovery",
    "discover_ide_bridge_provenance",
    "parse_transcript_file",
]
