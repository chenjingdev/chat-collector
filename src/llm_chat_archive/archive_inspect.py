from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .models import TranscriptCompleteness
from .runner import RUNS_DIRECTORY

ARCHIVE_OUTPUT_GLOB = "memory_chat_v1-*.jsonl"
SUPERSEDED_AT_FIELD = "superseded_at"
TRANSCRIPT_COMPLETENESS_VALUES = frozenset(
    completeness.value for completeness in TranscriptCompleteness
)


class ArchiveInspectError(ValueError):
    """Raised when normalized archive inspection cannot load or resolve a conversation."""


@dataclass(frozen=True, slots=True)
class ArchiveConversationSummary:
    source: str
    source_session_id: str | None
    transcript_completeness: str
    collected_at: str
    message_count: int
    limitations: tuple[str, ...]
    has_provenance: bool
    output_path: Path
    row_number: int
    source_artifact_path: str | None = None
    execution_context: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "source": self.source,
            "source_session_id": self.source_session_id,
            "transcript_completeness": self.transcript_completeness,
            "collected_at": self.collected_at,
            "message_count": self.message_count,
            "limitations": list(self.limitations),
            "has_provenance": self.has_provenance,
            "output_path": str(self.output_path),
            "row_number": self.row_number,
        }
        if self.source_artifact_path is not None:
            payload["source_artifact_path"] = self.source_artifact_path
        if self.execution_context is not None:
            payload["execution_context"] = self.execution_context
        return payload


@dataclass(frozen=True, slots=True)
class ArchiveConversationRecord:
    summary: ArchiveConversationSummary
    payload: dict[str, object]
    messages: tuple[dict[str, object], ...]

    def to_dict(self) -> dict[str, object]:
        conversation = dict(self.payload)
        conversation["messages"] = [dict(message) for message in self.messages]
        conversation["transcript_completeness"] = self.summary.transcript_completeness
        conversation["limitations"] = list(self.summary.limitations)
        conversation["message_count"] = self.summary.message_count
        conversation["has_provenance"] = self.summary.has_provenance
        return conversation


@dataclass(frozen=True, slots=True)
class ArchiveConversationMatch:
    conversation: ArchiveConversationSummary
    matched_message_count: int
    preview: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload = self.conversation.to_dict()
        payload["matched_message_count"] = self.matched_message_count
        if self.preview is not None:
            payload["preview"] = self.preview
        return payload


@dataclass(frozen=True, slots=True)
class ArchiveConversationFilter:
    source: str | None = None
    session: str | None = None
    transcript_completeness: str | None = None

    def matches(self, summary: ArchiveConversationSummary) -> bool:
        if self.source is not None and summary.source != self.source:
            return False
        if self.session is not None and summary.source_session_id != self.session:
            return False
        if (
            self.transcript_completeness is not None
            and summary.transcript_completeness != self.transcript_completeness
        ):
            return False
        return True


def list_archive_conversations(
    archive_root: Path,
    *,
    source: str | None = None,
    session: str | None = None,
    transcript_completeness: str | None = None,
) -> tuple[ArchiveConversationSummary, ...]:
    from .archive_index import list_indexed_archive_conversations

    return list_indexed_archive_conversations(
        archive_root,
        source=source,
        session=session,
        transcript_completeness=transcript_completeness,
    )


def show_archive_conversation(
    archive_root: Path,
    *,
    source: str,
    session: str,
) -> ArchiveConversationRecord:
    from .archive_index import list_indexed_archive_conversations

    matches = list_indexed_archive_conversations(
        archive_root,
        source=source,
        session=session,
    )
    if not matches:
        raise ArchiveInspectError(
            f"conversation not found for source={source!r} session={session!r}"
        )
    if len(matches) > 1:
        locations = ", ".join(
            f"{match.output_path}:{match.row_number}" for match in matches
        )
        raise ArchiveInspectError(
            "multiple conversations matched "
            f"source={source!r} session={session!r}: {locations}"
        )
    match = matches[0]
    return load_archive_record_at(
        match.output_path,
        row_number=match.row_number,
    )


def find_archive_conversations(
    archive_root: Path,
    *,
    text: str,
    source: str | None = None,
    transcript_completeness: str | None = None,
) -> tuple[ArchiveConversationMatch, ...]:
    from .archive_index import find_indexed_archive_conversations

    return find_indexed_archive_conversations(
        archive_root,
        text=text,
        source=source,
        transcript_completeness=transcript_completeness,
    )


def iter_archive_records(
    archive_root: Path,
    *,
    source: str | None = None,
):
    resolved_root = archive_root.expanduser().resolve(strict=False)
    if not resolved_root.exists():
        raise ArchiveInspectError(f"archive root does not exist: {resolved_root}")
    if not resolved_root.is_dir():
        raise ArchiveInspectError(f"archive root is not a directory: {resolved_root}")

    for output_path in _iter_output_paths(resolved_root, source=source):
        with output_path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                payload = load_archive_json_line(
                    raw_line,
                    output_path=output_path,
                    line_number=line_number,
                )
                if payload is None or is_superseded_archive_payload(payload):
                    continue
                yield build_archive_record(
                    payload,
                    output_path=output_path,
                    line_number=line_number,
                )


def _iter_output_paths(
    archive_root: Path,
    *,
    source: str | None,
) -> tuple[Path, ...]:
    if source is not None:
        source_dirs = (archive_root / source,)
    else:
        source_dirs = tuple(
            path
            for path in sorted(archive_root.iterdir())
            if path.is_dir() and path.name != RUNS_DIRECTORY
        )

    output_paths: list[Path] = []
    for source_dir in source_dirs:
        if not source_dir.is_dir():
            continue
        output_paths.extend(sorted(source_dir.glob(ARCHIVE_OUTPUT_GLOB), reverse=True))
    return tuple(output_paths)


def load_archive_record_at(
    output_path: Path,
    *,
    row_number: int,
) -> ArchiveConversationRecord:
    resolved_output_path = output_path.expanduser().resolve(strict=False)
    if not resolved_output_path.exists():
        raise ArchiveInspectError(
            f"archive output does not exist: {resolved_output_path}"
        )
    if not resolved_output_path.is_file():
        raise ArchiveInspectError(
            f"archive output is not a file: {resolved_output_path}"
        )
    if row_number <= 0:
        raise ArchiveInspectError(
            f"archive row number must be positive: {resolved_output_path}:{row_number}"
        )

    with resolved_output_path.open("r", encoding="utf-8") as handle:
        for current_line_number, raw_line in enumerate(handle, start=1):
            if current_line_number != row_number:
                continue
            payload = load_archive_json_line(
                raw_line,
                output_path=resolved_output_path,
                line_number=current_line_number,
            )
            if payload is None or is_superseded_archive_payload(payload):
                raise ArchiveInspectError(
                    "archive row is empty or superseded at "
                    f"{resolved_output_path}:{current_line_number}"
                )
            return build_archive_record(
                payload,
                output_path=resolved_output_path,
                line_number=current_line_number,
            )

    raise ArchiveInspectError(
        f"archive row not found at {resolved_output_path}:{row_number}"
    )


def build_archive_record(
    payload: dict[str, object],
    *,
    output_path: Path,
    line_number: int,
) -> ArchiveConversationRecord:
    source = _required_string(
        payload,
        "source",
        output_path=output_path,
        line_number=line_number,
    )
    collected_at = _required_string(
        payload,
        "collected_at",
        output_path=output_path,
        line_number=line_number,
    )
    execution_context = _optional_validated_string(
        payload,
        "execution_context",
        output_path=output_path,
        line_number=line_number,
    )
    source_session_id = _optional_validated_string(
        payload,
        "source_session_id",
        output_path=output_path,
        line_number=line_number,
    )
    source_artifact_path = _optional_validated_string(
        payload,
        "source_artifact_path",
        output_path=output_path,
        line_number=line_number,
    )
    transcript_completeness = _load_transcript_completeness(
        payload,
        output_path=output_path,
        line_number=line_number,
    )
    limitations = _load_string_sequence(
        payload,
        "limitations",
        output_path=output_path,
        line_number=line_number,
    )
    provenance = payload.get("provenance")
    if provenance is not None and not isinstance(provenance, dict):
        raise ArchiveInspectError(
            f"invalid provenance payload at {output_path}:{line_number}"
        )
    messages = _load_messages(
        payload,
        output_path=output_path,
        line_number=line_number,
    )
    summary = ArchiveConversationSummary(
        source=source,
        source_session_id=source_session_id,
        transcript_completeness=transcript_completeness,
        collected_at=collected_at,
        message_count=len(messages),
        limitations=limitations,
        has_provenance=provenance is not None,
        output_path=output_path,
        row_number=line_number,
        source_artifact_path=source_artifact_path,
        execution_context=execution_context,
    )
    return ArchiveConversationRecord(
        summary=summary,
        payload=payload,
        messages=messages,
    )


def load_archive_json_line(
    raw_line: str,
    *,
    output_path: Path,
    line_number: int,
) -> dict[str, object] | None:
    line = raw_line.strip()
    if not line:
        return None
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ArchiveInspectError(
            f"invalid JSON in archive output {output_path}:{line_number}: {exc.msg}"
        ) from exc
    if not isinstance(payload, dict):
        raise ArchiveInspectError(
            f"archive row must decode to an object: {output_path}:{line_number}"
        )
    return payload


def is_superseded_archive_payload(payload: Mapping[str, object]) -> bool:
    value = payload.get(SUPERSEDED_AT_FIELD)
    return isinstance(value, str) and bool(value.strip())


def _load_messages(
    payload: dict[str, object],
    *,
    output_path: Path,
    line_number: int,
) -> tuple[dict[str, object], ...]:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        raise ArchiveInspectError(
            f"normalized conversation requires messages list: {output_path}:{line_number}"
        )

    normalized_messages: list[dict[str, object]] = []
    for message_index, message in enumerate(messages, start=1):
        if not isinstance(message, dict):
            raise ArchiveInspectError(
                "normalized message must be an object: "
                f"{output_path}:{line_number} message={message_index}"
            )
        normalized_messages.append(message)
    return tuple(normalized_messages)


def _load_transcript_completeness(
    payload: dict[str, object],
    *,
    output_path: Path,
    line_number: int,
) -> str:
    value = payload.get("transcript_completeness", TranscriptCompleteness.COMPLETE.value)
    if not isinstance(value, str) or value not in TRANSCRIPT_COMPLETENESS_VALUES:
        raise ArchiveInspectError(
            "invalid transcript_completeness value at "
            f"{output_path}:{line_number}"
        )
    return value


def _load_string_sequence(
    payload: dict[str, object],
    key: str,
    *,
    output_path: Path,
    line_number: int,
) -> tuple[str, ...]:
    value = payload.get(key, [])
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ArchiveInspectError(f"invalid {key} value at {output_path}:{line_number}")
    return tuple(value)


def _required_string(
    payload: dict[str, object],
    key: str,
    *,
    output_path: Path,
    line_number: int,
) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ArchiveInspectError(f"missing required {key} at {output_path}:{line_number}")
    return value


def _optional_validated_string(
    payload: dict[str, object],
    key: str,
    *,
    output_path: Path,
    line_number: int,
) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ArchiveInspectError(f"invalid {key} value at {output_path}:{line_number}")
    return value


def _optional_string(value: object) -> str | None:
    if isinstance(value, str):
        return value
    return None


def _preview_text(text: str, *, max_length: int = 120) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_length:
        return compact
    return compact[: max_length - 3] + "..."


def _summary_sort_key(summary: ArchiveConversationSummary) -> tuple[object, ...]:
    return (
        summary.collected_at,
        summary.source,
        summary.source_session_id or "",
        str(summary.output_path),
        summary.row_number,
    )


__all__ = [
    "ARCHIVE_OUTPUT_GLOB",
    "ArchiveConversationFilter",
    "ArchiveConversationMatch",
    "ArchiveConversationRecord",
    "ArchiveConversationSummary",
    "ArchiveInspectError",
    "SUPERSEDED_AT_FIELD",
    "build_archive_record",
    "find_archive_conversations",
    "is_superseded_archive_payload",
    "iter_archive_records",
    "list_archive_conversations",
    "load_archive_record_at",
    "load_archive_json_line",
    "show_archive_conversation",
]
