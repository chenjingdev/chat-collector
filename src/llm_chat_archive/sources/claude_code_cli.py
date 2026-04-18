from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..incremental import write_incremental_collection
from ..models import (
    CollectionPlan,
    CollectionResult,
    ConversationProvenance,
    MessageRole,
    NormalizedConversation,
    NormalizedImage,
    NormalizedMessage,
    SourceDescriptor,
    SourceSupportMetadata,
    SupportLevel,
    TranscriptCompleteness,
)
from ..source_roots import all_platform_root, default_descriptor_input_roots
from .codex_rollout import resolve_input_roots, utc_timestamp

CLAUDE_CODE_CLI_DESCRIPTOR = SourceDescriptor(
    key="claude",
    display_name="Claude Code CLI",
    execution_context="cli",
    support_level=SupportLevel.COMPLETE,
    default_input_roots=("~/.claude", "~/.claude.json"),
    artifact_root_candidates=(
        all_platform_root("$HOME/.claude"),
        all_platform_root("$HOME/.claude.json"),
    ),
    notes=(
        "Scans ~/.claude/projects/<encoded-project-path>/<session-uuid>.jsonl.",
        "Keeps subagents/*.jsonl as separate traces instead of merging them into parent transcripts.",
        "Retains only human-facing user text/image content and assistant text content.",
    ),
    support_metadata=SourceSupportMetadata(
        product_label="Claude Code",
        host_surface="CLI",
        expected_transcript_completeness=TranscriptCompleteness.COMPLETE,
        limitation_summary="Subagent traces remain separate JSONL sessions instead of being merged into the parent.",
    ),
)

CLAUDE_TRANSCRIPT_GLOBS = (
    "projects/*/*.jsonl",
    "projects/*/*/subagents/*.jsonl",
)
MAIN_TRANSCRIPT_ROW_TYPES = frozenset({"assistant", "user"})
ASSISTANT_TEXT_ITEM_TYPES = frozenset({"text"})
USER_TEXT_ITEM_TYPES = frozenset({"text"})
USER_IMAGE_ITEM_TYPES = frozenset({"image"})


@dataclass(slots=True)
class ClaudeTranscriptMetadata:
    session_id: str | None = None
    session_started_at: str | None = None
    cwd: str | None = None
    agent_id: str | None = None
    subagent: bool = False

    def observe(self, record: dict[str, object]) -> None:
        if self.session_id is None:
            self.session_id = _string_value(record.get("sessionId"))
        if self.session_started_at is None:
            self.session_started_at = _string_value(record.get("timestamp"))
        if self.cwd is None:
            self.cwd = _string_value(record.get("cwd"))
        if self.agent_id is None:
            self.agent_id = _string_value(record.get("agentId"))

    def build_source_session_id(self, transcript_path: Path) -> str:
        if not self.subagent:
            return self.session_id or transcript_path.stem

        parent_session_id = transcript_path.parents[1].name
        child_session_id = transcript_path.stem
        if self.session_id and self.session_id != parent_session_id:
            child_session_id = self.session_id
        return f"{parent_session_id}/subagents/{child_session_id}"

    def build_provenance(self) -> ConversationProvenance:
        return ConversationProvenance(
            session_started_at=self.session_started_at,
            source="cli",
            originator=(
                self.agent_id
                if self.agent_id is not None
                else "claude_code_cli_subagent"
                if self.subagent
                else "claude_code_cli"
            ),
            cwd=self.cwd,
        )


@dataclass(frozen=True, slots=True)
class ClaudeCodeCliCollector:
    descriptor: SourceDescriptor = CLAUDE_CODE_CLI_DESCRIPTOR

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
        collected_at = utc_timestamp()
        conversations = (
            parse_transcript_file(
                transcript_path,
                collected_at=collected_at,
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


def iter_transcript_paths(input_roots: Iterable[Path]) -> Iterable[Path]:
    seen: set[Path] = set()
    candidates: list[Path] = []
    for input_root in input_roots:
        for candidate in _iter_transcript_candidates(input_root):
            resolved = candidate.resolve(strict=False)
            if not candidate.is_file() or resolved in seen or not _is_transcript_path(candidate):
                continue
            seen.add(resolved)
            candidates.append(resolved)
    yield from sorted(candidates, key=_transcript_sort_key)


def parse_transcript_file(
    transcript_path: Path, *, collected_at: str | None = None
) -> NormalizedConversation | None:
    resolved_path = transcript_path.expanduser().resolve(strict=False)
    metadata = ClaudeTranscriptMetadata(subagent=_is_subagent_path(resolved_path))
    messages: list[NormalizedMessage] = []

    try:
        lines = resolved_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return None

    for raw_line in lines:
        record = _load_json_line(raw_line)
        if record is None:
            continue

        metadata.observe(record)

        record_type = _string_value(record.get("type"))
        if record_type not in MAIN_TRANSCRIPT_ROW_TYPES:
            continue

        message = record.get("message")
        if not isinstance(message, dict):
            continue

        normalized_message = _normalize_message(record_type, message, record)
        if normalized_message is None:
            continue
        messages.append(normalized_message)

    if not messages:
        return None

    return NormalizedConversation(
        source=CLAUDE_CODE_CLI_DESCRIPTOR.key,
        execution_context=CLAUDE_CODE_CLI_DESCRIPTOR.execution_context,
        collected_at=collected_at or utc_timestamp(),
        messages=tuple(messages),
        source_session_id=metadata.build_source_session_id(resolved_path),
        source_artifact_path=str(resolved_path),
        provenance=metadata.build_provenance(),
    )


def _iter_transcript_candidates(input_root: Path) -> Iterable[Path]:
    if input_root.is_file():
        yield input_root
        return

    if not input_root.is_dir():
        return

    patterns = list(CLAUDE_TRANSCRIPT_GLOBS)
    if input_root.name == "projects":
        patterns.extend(pattern.removeprefix("projects/") for pattern in CLAUDE_TRANSCRIPT_GLOBS)

    for pattern in patterns:
        yield from input_root.glob(pattern)


def _is_transcript_path(path: Path) -> bool:
    if path.suffix != ".jsonl":
        return False
    return _is_main_session_path(path) or _is_subagent_path(path)


def _is_main_session_path(path: Path) -> bool:
    return path.parent.parent.name == "projects"


def _is_subagent_path(path: Path) -> bool:
    return path.parent.name == "subagents"


def _transcript_sort_key(path: Path) -> tuple[str, str, int, str]:
    if _is_subagent_path(path):
        project_dir = path.parents[2]
        session_group = path.parents[1].name
        return (str(project_dir), session_group, 1, str(path))

    return (str(path.parent), path.stem, 0, str(path))


def _normalize_message(
    record_type: str,
    message: dict[str, object],
    record: dict[str, object],
) -> NormalizedMessage | None:
    if record_type == "assistant":
        text = _extract_text_content(
            message.get("content"),
            allowed_item_types=ASSISTANT_TEXT_ITEM_TYPES,
        )
        if text is None:
            return None
        return NormalizedMessage(
            role=MessageRole.ASSISTANT,
            text=text,
            timestamp=_string_value(record.get("timestamp")),
            source_message_id=_string_field(message, "id") or _string_value(record.get("uuid")),
        )

    if record_type != "user":
        return None

    text, images = _extract_user_content(message.get("content"))
    if text is None and not images:
        return None
    return NormalizedMessage(
        role=MessageRole.USER,
        text=text,
        images=images,
        timestamp=_string_value(record.get("timestamp")),
        source_message_id=_string_field(message, "id") or _string_value(record.get("uuid")),
    )


def _extract_text_content(
    content: object, *, allowed_item_types: frozenset[str]
) -> str | None:
    if isinstance(content, str):
        text = content.strip()
        return text or None

    if not isinstance(content, list):
        return None

    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            text = item.strip()
            if text:
                parts.append(text)
            continue
        if not isinstance(item, dict):
            continue
        if _string_field(item, "type") not in allowed_item_types:
            continue
        text = _string_field(item, "text")
        if text:
            parts.append(text.strip())

    if not parts:
        return None
    return "\n\n".join(part for part in parts if part)


def _extract_user_content(
    content: object,
) -> tuple[str | None, tuple[NormalizedImage, ...]]:
    if isinstance(content, str):
        text = content.strip()
        return (text or None, ())

    if not isinstance(content, list):
        return (None, ())

    text_parts: list[str] = []
    images: list[NormalizedImage] = []
    for item in content:
        if isinstance(item, str):
            text = item.strip()
            if text:
                text_parts.append(text)
            continue
        if not isinstance(item, dict):
            continue

        item_type = _string_field(item, "type")
        if item_type in USER_TEXT_ITEM_TYPES:
            text = _string_field(item, "text")
            if text:
                text_parts.append(text.strip())
            continue
        if item_type not in USER_IMAGE_ITEM_TYPES:
            continue

        image = _extract_image(item)
        if image is not None:
            images.append(image)

    text = "\n\n".join(part for part in text_parts if part) or None
    return (text, tuple(images))


def _extract_image(item: dict[str, object]) -> NormalizedImage | None:
    source = (
        _string_field(item, "source")
        or _string_field(item, "image_url")
        or _string_field(item, "url")
        or _string_field(item, "file_path")
        or _string_field(item, "path")
    )
    if source is None:
        return None

    return NormalizedImage(
        source=source,
        mime_type=_string_field(item, "mime_type") or _string_field(item, "media_type"),
    )


def _load_json_line(raw_line: str) -> dict[str, object] | None:
    line = raw_line.strip()
    if not line:
        return None
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _string_field(payload: dict[str, object], key: str) -> str | None:
    return _string_value(payload.get(key))


def _string_value(value: object) -> str | None:
    if isinstance(value, str):
        return value
    return None


__all__ = [
    "CLAUDE_CODE_CLI_DESCRIPTOR",
    "ClaudeCodeCliCollector",
    "iter_transcript_paths",
    "parse_transcript_file",
]
