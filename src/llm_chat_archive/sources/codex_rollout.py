from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

from ..models import (
    ConversationProvenance,
    MessageRole,
    NormalizedConversation,
    NormalizedMessage,
    SourceDescriptor,
)
from ..source_roots import resolve_explicit_input_roots

ROLE_MAP = {
    "assistant": MessageRole.ASSISTANT,
    "developer": MessageRole.DEVELOPER,
    "user": MessageRole.USER,
}
TEXT_ITEM_TYPES = frozenset({"input_text", "output_text", "text"})
ROLLOUT_GLOBS = (
    "sessions/**/rollout-*.jsonl",
    "archived_sessions/rollout-*.jsonl",
)

SessionFilter = Callable[["CodexSessionMetadata", Path], bool]
ProvenanceFactory = Callable[["CodexSessionMetadata", Path], ConversationProvenance]


@dataclass(frozen=True, slots=True)
class CodexSessionMetadata:
    session_id: str | None = None
    session_started_at: str | None = None
    source: str | None = None
    originator: str | None = None
    cwd: str | None = None
    cli_version: str | None = None
    archived: bool = False

    @classmethod
    def from_payload(cls, payload: dict[str, object], *, archived: bool) -> "CodexSessionMetadata":
        return cls(
            session_id=_string_field(payload, "id"),
            session_started_at=_string_field(payload, "timestamp"),
            source=_string_field(payload, "source"),
            originator=_string_field(payload, "originator"),
            cwd=_string_field(payload, "cwd"),
            cli_version=_string_field(payload, "cli_version"),
            archived=archived,
        )


def parse_codex_rollout_file(
    rollout_path: Path,
    *,
    descriptor: SourceDescriptor,
    collected_at: str | None = None,
    session_filter: SessionFilter | None = None,
    provenance_factory: ProvenanceFactory | None = None,
) -> NormalizedConversation | None:
    resolved_path = rollout_path.expanduser().resolve(strict=False)
    session_metadata = CodexSessionMetadata(archived=_is_archived_rollout(resolved_path))
    messages: list[NormalizedMessage] = []

    try:
        lines = resolved_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return None

    for raw_line in lines:
        record = _load_json_line(raw_line)
        if record is None:
            continue

        payload = record.get("payload")
        if not isinstance(payload, dict):
            continue

        if record.get("type") == "session_meta":
            session_metadata = CodexSessionMetadata.from_payload(
                payload,
                archived=_is_archived_rollout(resolved_path),
            )
            continue

        if record.get("type") != "response_item":
            continue

        if payload.get("type") != "message":
            continue

        role = ROLE_MAP.get(_string_value(payload.get("role")))
        if role is None:
            continue

        text = _extract_message_text(payload.get("content"))
        if not text:
            continue

        messages.append(
            NormalizedMessage(
                role=role,
                text=text,
                timestamp=_string_field(payload, "timestamp"),
                source_message_id=_string_field(payload, "id"),
            )
        )

    if session_filter is not None and not session_filter(session_metadata, resolved_path):
        return None

    if not messages:
        return None

    return NormalizedConversation(
        source=descriptor.key,
        execution_context=descriptor.execution_context,
        collected_at=collected_at or utc_timestamp(),
        messages=tuple(messages),
        source_session_id=session_metadata.session_id or resolved_path.stem,
        source_artifact_path=str(resolved_path),
        provenance=(
            provenance_factory(session_metadata, resolved_path)
            if provenance_factory is not None
            else build_conversation_provenance(session_metadata)
        ),
    )


def build_conversation_provenance(
    session_metadata: CodexSessionMetadata,
) -> ConversationProvenance:
    return ConversationProvenance(
        session_started_at=session_metadata.session_started_at,
        source=session_metadata.source,
        originator=session_metadata.originator,
        cwd=session_metadata.cwd,
        cli_version=session_metadata.cli_version,
        archived=session_metadata.archived,
    )


def resolve_input_roots(input_roots: Iterable[Path]) -> tuple[Path, ...]:
    return resolve_explicit_input_roots(input_roots)


def iter_rollout_paths(input_roots: Iterable[Path]) -> Iterable[Path]:
    seen: set[Path] = set()
    candidates: list[Path] = []
    for input_root in input_roots:
        for pattern in ROLLOUT_GLOBS:
            for candidate in input_root.glob(pattern):
                resolved = candidate.resolve(strict=False)
                if not candidate.is_file() or resolved in seen:
                    continue
                seen.add(resolved)
                candidates.append(resolved)
    yield from sorted(candidates)


def _is_archived_rollout(rollout_path: Path) -> bool:
    return "archived_sessions" in rollout_path.parts


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


def _extract_message_text(content: object) -> str | None:
    if isinstance(content, str):
        return content.strip() or None
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
        if item.get("type") not in TEXT_ITEM_TYPES:
            continue

        text = _string_field(item, "text")
        if text:
            parts.append(text.strip())

    if not parts:
        return None
    return "\n\n".join(part for part in parts if part)


def _string_field(payload: dict[str, object], key: str) -> str | None:
    return _string_value(payload.get(key))


def _string_value(value: object) -> str | None:
    if isinstance(value, str):
        return value
    return None


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def timestamp_slug(timestamp: str) -> str:
    return timestamp.replace("-", "").replace(":", "").replace("+00:00", "Z")


__all__ = [
    "CodexSessionMetadata",
    "build_conversation_provenance",
    "iter_rollout_paths",
    "parse_codex_rollout_file",
    "resolve_input_roots",
    "timestamp_slug",
    "utc_timestamp",
]
