from __future__ import annotations

import hashlib
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
    NormalizedMessage,
    SourceDescriptor,
    SourceSupportMetadata,
    SupportLevel,
    TranscriptCompleteness,
)
from ..source_roots import all_platform_root, default_descriptor_input_roots
from .codex_rollout import resolve_input_roots, utc_timestamp

GEMINI_CLI_DESCRIPTOR = SourceDescriptor(
    key="gemini",
    display_name="Gemini CLI",
    execution_context="cli",
    support_level=SupportLevel.COMPLETE,
    default_input_roots=("~/.gemini",),
    artifact_root_candidates=(all_platform_root("$HOME/.gemini"),),
    notes=(
        "Maps the current repository path to ~/.gemini/tmp/<sha256(repo-path)>/chats.",
        "Keeps user and gemini message content only.",
        "Excludes logs.json, history, thoughts, toolCalls, token counters, auth files, and MCP residue.",
    ),
    support_metadata=SourceSupportMetadata(
        product_label="Gemini",
        host_surface="CLI",
        expected_transcript_completeness=TranscriptCompleteness.COMPLETE,
        limitation_summary="Only user and gemini message content is retained; logs, auth, thoughts, and tool residue are excluded.",
    ),
)

GEMINI_SESSION_GLOB = "session-*.json"
MODULE_REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True, slots=True)
class GeminiProjectDiscovery:
    repo_path: Path
    project_hash: str
    input_roots: tuple[Path, ...]
    project_temp_dir: Path | None = None
    chat_root: Path | None = None
    session_paths: tuple[Path, ...] = ()
    negative_reason: str | None = None

    @property
    def matched(self) -> bool:
        return self.negative_reason is None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "repo_path": str(self.repo_path),
            "project_hash": self.project_hash,
            "input_roots": [str(root) for root in self.input_roots],
            "matched": self.matched,
            "session_paths": [str(path) for path in self.session_paths],
        }
        if self.project_temp_dir is not None:
            payload["project_temp_dir"] = str(self.project_temp_dir)
        if self.chat_root is not None:
            payload["chat_root"] = str(self.chat_root)
        if self.negative_reason is not None:
            payload["negative_reason"] = self.negative_reason
        return payload


@dataclass(slots=True)
class GeminiTranscriptMetadata:
    session_id: str | None = None
    session_started_at: str | None = None
    project_hash: str | None = None

    def observe(self, payload: dict[str, object]) -> None:
        if self.session_id is None:
            self.session_id = _string_value(payload.get("sessionId"))
        if self.session_started_at is None:
            self.session_started_at = _string_value(payload.get("startTime"))
        if self.project_hash is None:
            self.project_hash = _string_value(payload.get("projectHash"))


@dataclass(frozen=True, slots=True)
class GeminiCliCollector:
    descriptor: SourceDescriptor = GEMINI_CLI_DESCRIPTOR
    repo_path: Path = MODULE_REPO_ROOT

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
        discovery = discover_project_sessions(
            self.repo_path,
            resolved_input_roots,
        )
        collected_at = utc_timestamp()
        conversations = (
            parse_transcript_file(
                transcript_path,
                repo_path=discovery.repo_path,
                collected_at=collected_at,
            )
            for transcript_path in discovery.session_paths
        )
        return write_incremental_collection(
            source=self.descriptor.key,
            archive_root=archive_root,
            input_roots=resolved_input_roots,
            scanned_artifact_count=len(discovery.session_paths),
            collected_at=collected_at,
            conversations=conversations,
        )

    def _default_input_roots(self) -> tuple[Path, ...]:
        return default_descriptor_input_roots(self.descriptor)


def gemini_project_hash(repo_path: Path) -> str:
    resolved_repo_path = repo_path.expanduser().resolve(strict=False)
    return hashlib.sha256(str(resolved_repo_path).encode("utf-8")).hexdigest()


def discover_project_sessions(
    repo_path: Path,
    input_roots: Iterable[Path],
) -> GeminiProjectDiscovery:
    resolved_repo_path = repo_path.expanduser().resolve(strict=False)
    resolved_input_roots = resolve_input_roots(input_roots)
    project_hash = gemini_project_hash(resolved_repo_path)

    project_temp_dir: Path | None = None
    chat_root: Path | None = None
    seen_project_dirs: set[Path] = set()
    session_paths: list[Path] = []
    seen_sessions: set[Path] = set()

    for input_root in resolved_input_roots:
        for candidate in _iter_project_dir_candidates(input_root, project_hash):
            resolved_candidate = candidate.resolve(strict=False)
            if resolved_candidate in seen_project_dirs or not resolved_candidate.is_dir():
                continue

            seen_project_dirs.add(resolved_candidate)
            if project_temp_dir is None:
                project_temp_dir = resolved_candidate

            candidate_chat_root = resolved_candidate / "chats"
            if not candidate_chat_root.is_dir():
                continue

            if chat_root is None:
                chat_root = candidate_chat_root

            for session_path in sorted(candidate_chat_root.glob(GEMINI_SESSION_GLOB)):
                resolved_session = session_path.resolve(strict=False)
                if not session_path.is_file() or resolved_session in seen_sessions:
                    continue
                seen_sessions.add(resolved_session)
                session_paths.append(resolved_session)

    negative_reason: str | None = None
    if project_temp_dir is None:
        negative_reason = "missing_project_hash_dir"
    elif chat_root is None:
        negative_reason = "missing_chat_dir"
    elif not session_paths:
        negative_reason = "no_session_files"

    return GeminiProjectDiscovery(
        repo_path=resolved_repo_path,
        project_hash=project_hash,
        input_roots=resolved_input_roots,
        project_temp_dir=project_temp_dir,
        chat_root=chat_root,
        session_paths=tuple(sorted(session_paths)),
        negative_reason=negative_reason,
    )


def parse_transcript_file(
    transcript_path: Path,
    *,
    repo_path: Path | None = None,
    collected_at: str | None = None,
) -> NormalizedConversation | None:
    resolved_path = transcript_path.expanduser().resolve(strict=False)

    try:
        payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None

    metadata = GeminiTranscriptMetadata()
    metadata.observe(payload)

    raw_messages = payload.get("messages")
    if not isinstance(raw_messages, list):
        return None

    messages: list[NormalizedMessage] = []
    for record in raw_messages:
        if not isinstance(record, dict):
            continue

        normalized_message = _normalize_message(record)
        if normalized_message is not None:
            messages.append(normalized_message)

    if not messages:
        return None

    resolved_repo_path: Path | None = None
    if repo_path is not None:
        resolved_repo_path = repo_path.expanduser().resolve(strict=False)

    return NormalizedConversation(
        source=GEMINI_CLI_DESCRIPTOR.key,
        execution_context=GEMINI_CLI_DESCRIPTOR.execution_context,
        collected_at=collected_at or utc_timestamp(),
        messages=tuple(messages),
        source_session_id=metadata.session_id or resolved_path.stem,
        source_artifact_path=str(resolved_path),
        provenance=ConversationProvenance(
            session_started_at=metadata.session_started_at,
            source="cli",
            originator="gemini_cli",
            cwd=str(resolved_repo_path) if resolved_repo_path is not None else None,
        ),
    )


def _iter_project_dir_candidates(input_root: Path, project_hash: str) -> Iterable[Path]:
    if input_root.is_file():
        if _is_project_session_file(input_root, project_hash):
            yield input_root.parent.parent
        return

    if not input_root.is_dir():
        return

    yield input_root / "tmp" / project_hash
    yield input_root / project_hash

    if input_root.name == project_hash:
        yield input_root

    if input_root.name == "chats" and input_root.parent.name == project_hash:
        yield input_root.parent


def _is_project_session_file(path: Path, project_hash: str) -> bool:
    return (
        path.suffix == ".json"
        and path.name.startswith("session-")
        and path.parent.name == "chats"
        and path.parent.parent.name == project_hash
    )


def _normalize_message(record: dict[str, object]) -> NormalizedMessage | None:
    message_type = _string_value(record.get("type"))
    if message_type == "user":
        role = MessageRole.USER
    elif message_type == "gemini":
        role = MessageRole.ASSISTANT
    else:
        return None

    text = _extract_human_text(record.get("content"))
    if text is None:
        return None

    return NormalizedMessage(
        role=role,
        text=text,
        timestamp=_string_value(record.get("timestamp")),
        source_message_id=_string_value(record.get("id")),
    )


def _extract_human_text(content: object) -> str | None:
    parts: list[str] = []
    _append_text_parts(content, parts)
    if not parts:
        return None
    return "\n\n".join(parts)


def _append_text_parts(content: object, parts: list[str]) -> None:
    if isinstance(content, str):
        text = content.strip()
        if text:
            parts.append(text)
        return

    if isinstance(content, list):
        for item in content:
            _append_text_parts(item, parts)
        return

    if not isinstance(content, dict):
        return

    text = _string_value(content.get("text"))
    if text is not None:
        stripped = text.strip()
        if stripped:
            parts.append(stripped)

    nested_content = content.get("content")
    if nested_content is not None:
        _append_text_parts(nested_content, parts)

    nested_parts = content.get("parts")
    if isinstance(nested_parts, list):
        for item in nested_parts:
            _append_text_parts(item, parts)


def _string_value(value: object) -> str | None:
    if isinstance(value, str):
        return value
    return None


__all__ = [
    "GEMINI_CLI_DESCRIPTOR",
    "GeminiCliCollector",
    "GeminiProjectDiscovery",
    "discover_project_sessions",
    "gemini_project_hash",
    "parse_transcript_file",
]
