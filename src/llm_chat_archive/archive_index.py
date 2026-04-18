from __future__ import annotations

import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .archive_inspect import (
    ARCHIVE_OUTPUT_GLOB,
    ArchiveConversationMatch,
    ArchiveConversationSummary,
    ArchiveInspectError,
    build_archive_record,
    is_superseded_archive_payload,
    load_archive_json_line,
)
from .models import MessageRole, TranscriptCompleteness
from .runner import RUNS_DIRECTORY

ARCHIVE_INDEX_DIRECTORY = "archive-index"
ARCHIVE_INDEX_FILENAME = "conversations.sqlite3"
ARCHIVE_INDEX_SCHEMA_VERSION = "2026-03-20"

MESSAGE_ROLE_ORDER = tuple(role.value for role in MessageRole)
VALID_MESSAGE_ROLES = frozenset(MESSAGE_ROLE_ORDER)
TRANSCRIPT_COMPLETENESS_ORDER = tuple(
    completeness.value for completeness in TranscriptCompleteness
)
SUMMARY_ORDER_BY = (
    "c.collected_at DESC, "
    "c.source DESC, "
    "COALESCE(c.source_session_id, '') DESC, "
    "c.output_path DESC, "
    "c.row_number DESC"
)
INVALID_MESSAGE_ROLE_SENTINEL = "__invalid__"


@dataclass(frozen=True, slots=True)
class ArchiveIndexStatus:
    archive_root: Path
    index_path: Path
    state: str
    exists: bool
    ready: bool
    stale: bool
    rebuild_required: bool
    reason: str | None
    file_count: int
    indexed_file_count: int
    conversation_count: int
    indexed_conversation_count: int
    added_file_count: int
    updated_file_count: int
    removed_file_count: int
    last_refreshed_at: str | None

    def to_dict(self) -> dict[str, object]:
        payload = {
            "archive_root": str(self.archive_root),
            "index_path": str(self.index_path),
            "state": self.state,
            "exists": self.exists,
            "ready": self.ready,
            "stale": self.stale,
            "rebuild_required": self.rebuild_required,
            "file_count": self.file_count,
            "indexed_file_count": self.indexed_file_count,
            "conversation_count": self.conversation_count,
            "indexed_conversation_count": self.indexed_conversation_count,
            "added_file_count": self.added_file_count,
            "updated_file_count": self.updated_file_count,
            "removed_file_count": self.removed_file_count,
            "last_refreshed_at": self.last_refreshed_at,
        }
        if self.reason is not None:
            payload["reason"] = self.reason
        return payload


@dataclass(frozen=True, slots=True)
class ArchiveIndexRefreshReport:
    archive_root: Path
    index_path: Path
    force: bool
    rebuilt: bool
    refreshed: bool
    added_file_count: int
    updated_file_count: int
    removed_file_count: int
    status: ArchiveIndexStatus

    def to_dict(self) -> dict[str, object]:
        return {
            "archive_root": str(self.archive_root),
            "index_path": str(self.index_path),
            "force": self.force,
            "rebuilt": self.rebuilt,
            "refreshed": self.refreshed,
            "added_file_count": self.added_file_count,
            "updated_file_count": self.updated_file_count,
            "removed_file_count": self.removed_file_count,
            "status": self.status.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class _ArchiveOutputFile:
    output_path: Path
    size_bytes: int
    mtime_ns: int


@dataclass(frozen=True, slots=True)
class _IndexedFileRow:
    output_path: Path
    size_bytes: int
    mtime_ns: int


@dataclass(frozen=True, slots=True)
class _ArchiveIndexDiff:
    added_files: tuple[_ArchiveOutputFile, ...]
    updated_files: tuple[_ArchiveOutputFile, ...]
    removed_files: tuple[Path, ...]


def archive_index_path(archive_root: Path) -> Path:
    return archive_root / ARCHIVE_INDEX_DIRECTORY / ARCHIVE_INDEX_FILENAME


def inspect_archive_index(archive_root: Path) -> ArchiveIndexStatus:
    resolved_root = _validate_archive_root_directory(archive_root)
    discovered_files = _discover_archive_output_files(resolved_root)
    index_path = archive_index_path(resolved_root)
    if not index_path.exists():
        return _build_status(
            archive_root=resolved_root,
            index_path=index_path,
            discovered_files=discovered_files,
            indexed_files={},
            conversation_count=0,
            reason="archive index database does not exist",
            exists=False,
            rebuild_required=True,
            last_refreshed_at=None,
        )
    if not index_path.is_file():
        return _build_status(
            archive_root=resolved_root,
            index_path=index_path,
            discovered_files=discovered_files,
            indexed_files={},
            conversation_count=0,
            reason=f"archive index path is not a file: {index_path}",
            exists=True,
            rebuild_required=True,
            last_refreshed_at=None,
        )

    try:
        with _connect(index_path, read_only=True) as connection:
            _require_schema(connection)
            indexed_files = _load_indexed_files(connection)
            conversation_count = _scalar_int(
                connection,
                "SELECT COUNT(*) FROM conversations",
            )
            last_refreshed_at = _meta_value(connection, "last_refreshed_at")
    except sqlite3.DatabaseError as exc:
        return _build_status(
            archive_root=resolved_root,
            index_path=index_path,
            discovered_files=discovered_files,
            indexed_files={},
            conversation_count=0,
            reason=f"archive index database is unreadable: {exc}",
            exists=True,
            rebuild_required=True,
            last_refreshed_at=None,
        )

    return _build_status(
        archive_root=resolved_root,
        index_path=index_path,
        discovered_files=discovered_files,
        indexed_files=indexed_files,
        conversation_count=conversation_count,
        reason=None,
        exists=True,
        rebuild_required=False,
        last_refreshed_at=last_refreshed_at,
    )


def ensure_archive_index(archive_root: Path) -> ArchiveIndexStatus:
    status = inspect_archive_index(archive_root)
    if status.ready:
        return status
    refresh_archive_index(
        archive_root,
        force=status.rebuild_required,
    )
    return inspect_archive_index(archive_root)


def refresh_archive_index(
    archive_root: Path,
    *,
    force: bool = False,
) -> ArchiveIndexRefreshReport:
    resolved_root = _validate_archive_root_directory(archive_root)
    discovered_files = _discover_archive_output_files(resolved_root)
    index_path = archive_index_path(resolved_root)
    index_path.parent.mkdir(parents=True, exist_ok=True)

    rebuilt = force
    added_file_count = 0
    updated_file_count = 0
    removed_file_count = 0

    if not rebuilt and index_path.exists():
        if not index_path.is_file():
            raise ArchiveInspectError(f"archive index path is not a file: {index_path}")
        try:
            with _connect(index_path) as connection:
                _require_schema(connection)
                indexed_files = _load_indexed_files(connection)
                diff = _diff_files(discovered_files, indexed_files)
                added_file_count = len(diff.added_files)
                updated_file_count = len(diff.updated_files)
                removed_file_count = len(diff.removed_files)
                if diff.added_files or diff.updated_files or diff.removed_files:
                    with connection:
                        _apply_file_diff(
                            connection,
                            diff,
                            refreshed_at=_utcnow_string(),
                        )
        except sqlite3.DatabaseError:
            rebuilt = True
    else:
        rebuilt = True

    if rebuilt:
        _rebuild_archive_index(
            resolved_root,
            discovered_files=discovered_files,
            index_path=index_path,
        )
        added_file_count = len(discovered_files)
        updated_file_count = 0
        removed_file_count = 0

    status = inspect_archive_index(resolved_root)
    return ArchiveIndexRefreshReport(
        archive_root=resolved_root,
        index_path=index_path,
        force=force,
        rebuilt=rebuilt,
        refreshed=rebuilt or bool(
            added_file_count or updated_file_count or removed_file_count
        ),
        added_file_count=added_file_count,
        updated_file_count=updated_file_count,
        removed_file_count=removed_file_count,
        status=status,
    )


def list_indexed_archive_conversations(
    archive_root: Path,
    *,
    source: str | None = None,
    session: str | None = None,
    transcript_completeness: str | None = None,
) -> tuple[ArchiveConversationSummary, ...]:
    status = ensure_archive_index(archive_root)
    with _connect(status.index_path, read_only=True) as connection:
        where_clause, parameters = _conversation_filter_sql(
            source=source,
            session=session,
            transcript_completeness=transcript_completeness,
        )
        rows = connection.execute(
            f"""
            SELECT
                c.conversation_id,
                c.source,
                c.source_session_id,
                c.transcript_completeness,
                c.collected_at,
                c.message_count,
                c.has_provenance,
                c.output_path,
                c.row_number,
                c.source_artifact_path,
                c.execution_context,
                c.limitation_count
            FROM conversations AS c
            {where_clause}
            ORDER BY {SUMMARY_ORDER_BY}
            """,
            parameters,
        ).fetchall()
        limitations_by_conversation = _load_limitations_by_conversation(
            connection,
            rows,
        )
        return tuple(
            _summary_from_row(
                row,
                limitations=limitations_by_conversation.get(
                    int(row["conversation_id"]),
                    (),
                ),
            )
            for row in rows
        )


def find_indexed_archive_conversations(
    archive_root: Path,
    *,
    text: str,
    source: str | None = None,
    transcript_completeness: str | None = None,
) -> tuple[ArchiveConversationMatch, ...]:
    query = text.strip()
    if not query:
        raise ValueError("text query must not be empty")

    status = ensure_archive_index(archive_root)
    normalized_query = query.casefold()
    with _connect(status.index_path, read_only=True) as connection:
        where_clause, parameters = _conversation_filter_sql(
            source=source,
            session=None,
            transcript_completeness=transcript_completeness,
            additional_conditions=("instr(m.text_search, ?) > 0",),
            additional_parameters=(normalized_query,),
        )
        rows = connection.execute(
            f"""
            SELECT
                c.conversation_id,
                c.source,
                c.source_session_id,
                c.transcript_completeness,
                c.collected_at,
                c.message_count,
                c.has_provenance,
                c.output_path,
                c.row_number,
                c.source_artifact_path,
                c.execution_context,
                c.limitation_count,
                COUNT(*) AS matched_message_count,
                (
                    SELECT m2.text
                    FROM conversation_messages AS m2
                    WHERE m2.conversation_id = c.conversation_id
                    AND instr(m2.text_search, ?) > 0
                    ORDER BY m2.message_index ASC
                    LIMIT 1
                ) AS preview_text
            FROM conversations AS c
            INNER JOIN conversation_messages AS m
                ON m.conversation_id = c.conversation_id
            {where_clause}
            GROUP BY c.conversation_id
            ORDER BY {SUMMARY_ORDER_BY}
            """,
            (normalized_query, *parameters),
        ).fetchall()
        limitations_by_conversation = _load_limitations_by_conversation(
            connection,
            rows,
        )
        matches = [
            ArchiveConversationMatch(
                conversation=_summary_from_row(
                    row,
                    limitations=limitations_by_conversation.get(
                        int(row["conversation_id"]),
                        (),
                    ),
                ),
                matched_message_count=int(row["matched_message_count"]),
                preview=(
                    None
                    if row["preview_text"] is None
                    else _preview_text(str(row["preview_text"]))
                ),
            )
            for row in rows
        ]
        return tuple(matches)


def summarize_indexed_archive_stats(
    archive_root: Path,
    *,
    source: str | None = None,
):
    from .archive_stats import ArchiveSourceStats, ArchiveStatsReport

    status = ensure_archive_index(archive_root)
    with _connect(status.index_path, read_only=True) as connection:
        total_row = _fetch_stats_aggregate(connection, source=source)
        source_rows = connection.execute(
            """
            SELECT
                source,
                COUNT(*) AS conversation_count,
                COUNT(DISTINCT output_path) AS file_count,
                COALESCE(SUM(message_count), 0) AS message_count,
                MIN(collected_at) AS earliest_collected_at,
                MAX(collected_at) AS latest_collected_at,
                COALESCE(SUM(
                    CASE WHEN limitation_count > 0 THEN 1 ELSE 0 END
                ), 0) AS conversation_with_limitations_count
            FROM conversations
            WHERE (? IS NULL OR source = ?)
            GROUP BY source
            ORDER BY source ASC
            """,
            (source, source),
        ).fetchall()
        source_completeness = _fetch_transcript_completeness_counts(
            connection,
            source=source,
        )
        source_stats = tuple(
            ArchiveSourceStats(
                source=str(row["source"]),
                file_count=int(row["file_count"]),
                conversation_count=int(row["conversation_count"]),
                message_count=int(row["message_count"]),
                transcript_completeness_counts=_completeness_counts_for_source(
                    source_completeness,
                    str(row["source"]),
                ),
                earliest_collected_at=_optional_string(row["earliest_collected_at"]),
                latest_collected_at=_optional_string(row["latest_collected_at"]),
                conversation_with_limitations_count=int(
                    row["conversation_with_limitations_count"]
                ),
            )
            for row in source_rows
        )
        return ArchiveStatsReport(
            archive_root=status.archive_root,
            source_filter=source,
            file_count=int(total_row["file_count"]),
            conversation_count=int(total_row["conversation_count"]),
            message_count=int(total_row["message_count"]),
            transcript_completeness_counts=_completeness_counts_for_source(
                source_completeness,
                None,
            ),
            earliest_collected_at=_optional_string(total_row["earliest_collected_at"]),
            latest_collected_at=_optional_string(total_row["latest_collected_at"]),
            conversation_with_limitations_count=int(
                total_row["conversation_with_limitations_count"]
            ),
            sources=source_stats,
        )


def summarize_indexed_archive_profile(
    archive_root: Path,
    *,
    source: str | None = None,
):
    from .archive_profile import ArchiveProfileReport, ArchiveSourceProfile

    status = ensure_archive_index(archive_root)
    with _connect(status.index_path, read_only=True) as connection:
        invalid_message = connection.execute(
            """
            SELECT
                c.output_path,
                c.row_number,
                m.message_index,
                m.role
            FROM conversation_messages AS m
            INNER JOIN conversations AS c
                ON c.conversation_id = m.conversation_id
            WHERE (? IS NULL OR c.source = ?)
            AND (m.role = ? OR m.role NOT IN (?, ?, ?, ?))
            ORDER BY c.output_path ASC, c.row_number ASC, m.message_index ASC
            LIMIT 1
            """,
            (
                source,
                source,
                INVALID_MESSAGE_ROLE_SENTINEL,
                *MESSAGE_ROLE_ORDER,
            ),
        ).fetchone()
        if invalid_message is not None:
            raise ArchiveInspectError(
                "invalid message role at "
                f"{invalid_message['output_path']}:{invalid_message['row_number']} "
                f"message={invalid_message['message_index']}"
            )

        total_row = _fetch_stats_aggregate(connection, source=source)
        source_rows = connection.execute(
            """
            SELECT
                source,
                COUNT(*) AS conversation_count,
                COUNT(DISTINCT output_path) AS file_count,
                COALESCE(SUM(message_count), 0) AS message_count,
                COALESCE(SUM(
                    CASE WHEN limitation_count > 0 THEN 1 ELSE 0 END
                ), 0) AS conversation_with_limitations_count
            FROM conversations
            WHERE (? IS NULL OR source = ?)
            GROUP BY source
            ORDER BY source ASC
            """,
            (source, source),
        ).fetchall()
        source_completeness = _fetch_transcript_completeness_counts(
            connection,
            source=source,
        )
        source_roles = _fetch_message_role_counts(connection, source=source)
        source_limitations = _fetch_limitation_counts(connection, source=source)
        source_profiles = tuple(
            ArchiveSourceProfile(
                source=str(row["source"]),
                file_count=int(row["file_count"]),
                conversation_count=int(row["conversation_count"]),
                message_count=int(row["message_count"]),
                message_role_counts=_message_role_counts_for_source(
                    source_roles,
                    str(row["source"]),
                ),
                transcript_completeness_counts=_completeness_counts_for_source(
                    source_completeness,
                    str(row["source"]),
                ),
                limitation_counts=_limitation_counts_for_source(
                    source_limitations,
                    str(row["source"]),
                ),
                conversation_with_limitations_count=int(
                    row["conversation_with_limitations_count"]
                ),
            )
            for row in source_rows
        )
        return ArchiveProfileReport(
            archive_root=status.archive_root,
            source_filter=source,
            file_count=int(total_row["file_count"]),
            conversation_count=int(total_row["conversation_count"]),
            message_count=int(total_row["message_count"]),
            message_role_counts=_message_role_counts_for_source(source_roles, None),
            transcript_completeness_counts=_completeness_counts_for_source(
                source_completeness,
                None,
            ),
            limitation_counts=_limitation_counts_for_source(source_limitations, None),
            conversation_with_limitations_count=int(
                total_row["conversation_with_limitations_count"]
            ),
            sources=source_profiles,
        )


def collect_indexed_limitation_counts(
    archive_root: Path,
    *,
    baseline_policy,
):
    status = ensure_archive_index(archive_root)
    total_counts: Counter[str] = Counter()
    raw_total_counts: Counter[str] = Counter()
    source_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    raw_source_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)

    with _connect(status.index_path, read_only=True) as connection:
        rows = connection.execute(
            """
            SELECT c.source, l.limitation
            FROM conversation_limitations AS l
            INNER JOIN conversations AS c
                ON c.conversation_id = l.conversation_id
            ORDER BY c.source ASC, l.limitation ASC
            """
        ).fetchall()
        for row in rows:
            limitation = str(row["limitation"])
            source_name = str(row["source"])
            raw_total_counts[limitation] += 1
            raw_source_counts[source_name][limitation] += 1
            matched_entry = (
                None
                if baseline_policy is None
                else baseline_policy.match_limitation(
                    source=source_name,
                    limitation=limitation,
                )
            )
            if matched_entry is not None:
                continue
            total_counts[limitation] += 1
            source_counts[source_name][limitation] += 1

    return (
        total_counts,
        raw_total_counts,
        dict(source_counts),
        dict(raw_source_counts),
    )


def _rebuild_archive_index(
    archive_root: Path,
    *,
    discovered_files: tuple[_ArchiveOutputFile, ...],
    index_path: Path,
) -> None:
    if index_path.exists() and not index_path.is_file():
        raise ArchiveInspectError(f"archive index path is not a file: {index_path}")
    temp_path = index_path.with_suffix(index_path.suffix + ".tmp")
    if temp_path.exists():
        temp_path.unlink()
    with _connect(temp_path) as connection:
        _initialize_schema(connection)
        with connection:
            _index_output_files(
                connection,
                discovered_files,
                refreshed_at=_utcnow_string(),
            )
    temp_path.replace(index_path)


def _apply_file_diff(
    connection: sqlite3.Connection,
    diff: _ArchiveIndexDiff,
    *,
    refreshed_at: str,
) -> None:
    if diff.removed_files:
        connection.executemany(
            "DELETE FROM indexed_files WHERE output_path = ?",
            ((str(path),) for path in diff.removed_files),
        )
    files_to_reindex = diff.updated_files + diff.added_files
    for output_file in diff.updated_files:
        connection.execute(
            "DELETE FROM indexed_files WHERE output_path = ?",
            (str(output_file.output_path),),
        )
    _index_output_files(connection, files_to_reindex, refreshed_at=refreshed_at)
    _set_meta_value(connection, "last_refreshed_at", refreshed_at)


def _index_output_files(
    connection: sqlite3.Connection,
    output_files: tuple[_ArchiveOutputFile, ...],
    *,
    refreshed_at: str,
) -> None:
    for output_file in output_files:
        _index_output_file(connection, output_file, refreshed_at=refreshed_at)
    _set_meta_value(connection, "last_refreshed_at", refreshed_at)


def _index_output_file(
    connection: sqlite3.Connection,
    output_file: _ArchiveOutputFile,
    *,
    refreshed_at: str,
) -> None:
    connection.execute(
        """
        INSERT INTO indexed_files (
            output_path,
            size_bytes,
            mtime_ns,
            row_count,
            indexed_at
        ) VALUES (?, ?, ?, 0, ?)
        """,
        (
            str(output_file.output_path),
            output_file.size_bytes,
            output_file.mtime_ns,
            refreshed_at,
        ),
    )
    row_count = 0
    with output_file.output_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            payload = load_archive_json_line(
                raw_line,
                output_path=output_file.output_path,
                line_number=line_number,
            )
            if payload is None or is_superseded_archive_payload(payload):
                continue
            record = build_archive_record(
                payload,
                output_path=output_file.output_path,
                line_number=line_number,
            )
            cursor = connection.execute(
                """
                INSERT INTO conversations (
                    output_path,
                    row_number,
                    source,
                    source_session_id,
                    transcript_completeness,
                    collected_at,
                    message_count,
                    limitation_count,
                    has_provenance,
                    source_artifact_path,
                    execution_context
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(output_file.output_path),
                    line_number,
                    record.summary.source,
                    record.summary.source_session_id,
                    record.summary.transcript_completeness,
                    record.summary.collected_at,
                    record.summary.message_count,
                    len(record.summary.limitations),
                    1 if record.summary.has_provenance else 0,
                    record.summary.source_artifact_path,
                    record.summary.execution_context,
                ),
            )
            conversation_id = int(cursor.lastrowid)
            if record.summary.limitations:
                connection.executemany(
                    """
                    INSERT INTO conversation_limitations (
                        conversation_id,
                        limitation
                    ) VALUES (?, ?)
                    """,
                    (
                        (conversation_id, limitation)
                        for limitation in record.summary.limitations
                    ),
                )
            connection.executemany(
                """
                INSERT INTO conversation_messages (
                    conversation_id,
                    message_index,
                    role,
                    text,
                    text_search
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    (
                        conversation_id,
                        message_index,
                        _message_role_value(message.get("role")),
                        _optional_string(message.get("text")),
                        _searchable_text(message.get("text")),
                    )
                    for message_index, message in enumerate(record.messages, start=1)
                ),
            )
            row_count += 1
    connection.execute(
        """
        UPDATE indexed_files
        SET row_count = ?, indexed_at = ?
        WHERE output_path = ?
        """,
        (
            row_count,
            refreshed_at,
            str(output_file.output_path),
        ),
    )


def _summary_from_row(
    row: sqlite3.Row,
    *,
    limitations: tuple[str, ...],
) -> ArchiveConversationSummary:
    return ArchiveConversationSummary(
        source=str(row["source"]),
        source_session_id=_optional_string(row["source_session_id"]),
        transcript_completeness=str(row["transcript_completeness"]),
        collected_at=str(row["collected_at"]),
        message_count=int(row["message_count"]),
        limitations=limitations,
        has_provenance=bool(row["has_provenance"]),
        output_path=Path(str(row["output_path"])),
        row_number=int(row["row_number"]),
        source_artifact_path=_optional_string(row["source_artifact_path"]),
        execution_context=_optional_string(row["execution_context"]),
    )


def _load_limitations_by_conversation(
    connection: sqlite3.Connection,
    rows: list[sqlite3.Row],
) -> dict[int, tuple[str, ...]]:
    conversation_ids = [int(row["conversation_id"]) for row in rows]
    if not conversation_ids:
        return {}
    placeholder = ", ".join("?" for _ in conversation_ids)
    limitation_rows = connection.execute(
        f"""
        SELECT conversation_id, limitation
        FROM conversation_limitations
        WHERE conversation_id IN ({placeholder})
        ORDER BY conversation_id ASC, limitation ASC
        """,
        tuple(conversation_ids),
    ).fetchall()
    limitations_by_conversation: defaultdict[int, list[str]] = defaultdict(list)
    for limitation_row in limitation_rows:
        limitations_by_conversation[int(limitation_row["conversation_id"])].append(
            str(limitation_row["limitation"])
        )
    return {
        conversation_id: tuple(limitations)
        for conversation_id, limitations in limitations_by_conversation.items()
    }


def _fetch_stats_aggregate(
    connection: sqlite3.Connection,
    *,
    source: str | None,
) -> sqlite3.Row:
    return connection.execute(
        """
        SELECT
            COUNT(*) AS conversation_count,
            COUNT(DISTINCT output_path) AS file_count,
            COALESCE(SUM(message_count), 0) AS message_count,
            MIN(collected_at) AS earliest_collected_at,
            MAX(collected_at) AS latest_collected_at,
            COALESCE(SUM(
                CASE WHEN limitation_count > 0 THEN 1 ELSE 0 END
            ), 0) AS conversation_with_limitations_count
        FROM conversations
        WHERE (? IS NULL OR source = ?)
        """,
        (source, source),
    ).fetchone()


def _fetch_transcript_completeness_counts(
    connection: sqlite3.Connection,
    *,
    source: str | None,
) -> dict[str | None, dict[str, int]]:
    rows = connection.execute(
        """
        SELECT
            source,
            transcript_completeness,
            COUNT(*) AS count
        FROM conversations
        WHERE (? IS NULL OR source = ?)
        GROUP BY source, transcript_completeness
        """,
        (source, source),
    ).fetchall()
    counts: dict[str | None, dict[str, int]] = {
        None: {completeness: 0 for completeness in TRANSCRIPT_COMPLETENESS_ORDER}
    }
    for row in rows:
        source_name = str(row["source"])
        completeness = str(row["transcript_completeness"])
        count = int(row["count"])
        counts.setdefault(
            source_name,
            {value: 0 for value in TRANSCRIPT_COMPLETENESS_ORDER},
        )[completeness] = count
        counts[None][completeness] += count
    return counts


def _fetch_message_role_counts(
    connection: sqlite3.Connection,
    *,
    source: str | None,
) -> dict[str | None, dict[str, int]]:
    rows = connection.execute(
        """
        SELECT
            c.source,
            m.role,
            COUNT(*) AS count
        FROM conversation_messages AS m
        INNER JOIN conversations AS c
            ON c.conversation_id = m.conversation_id
        WHERE (? IS NULL OR c.source = ?)
        GROUP BY c.source, m.role
        """,
        (source, source),
    ).fetchall()
    counts: dict[str | None, dict[str, int]] = {
        None: {role: 0 for role in MESSAGE_ROLE_ORDER}
    }
    for row in rows:
        role = str(row["role"])
        if role == INVALID_MESSAGE_ROLE_SENTINEL:
            continue
        source_name = str(row["source"])
        count = int(row["count"])
        counts.setdefault(source_name, {value: 0 for value in MESSAGE_ROLE_ORDER})[
            role
        ] = count
        counts[None][role] += count
    return counts


def _fetch_limitation_counts(
    connection: sqlite3.Connection,
    *,
    source: str | None,
) -> dict[str | None, Counter[str]]:
    rows = connection.execute(
        """
        SELECT
            c.source,
            l.limitation,
            COUNT(*) AS count
        FROM conversation_limitations AS l
        INNER JOIN conversations AS c
            ON c.conversation_id = l.conversation_id
        WHERE (? IS NULL OR c.source = ?)
        GROUP BY c.source, l.limitation
        """,
        (source, source),
    ).fetchall()
    counts: dict[str | None, Counter[str]] = {None: Counter()}
    for row in rows:
        source_name = str(row["source"])
        limitation = str(row["limitation"])
        count = int(row["count"])
        counts.setdefault(source_name, Counter())[limitation] = count
        counts[None][limitation] += count
    return counts


def _message_role_counts_for_source(
    counts_by_source: dict[str | None, dict[str, int]],
    source: str | None,
) -> tuple[tuple[str, int], ...]:
    source_counts = counts_by_source.get(
        source,
        {role: 0 for role in MESSAGE_ROLE_ORDER},
    )
    return tuple((role, source_counts.get(role, 0)) for role in MESSAGE_ROLE_ORDER)


def _completeness_counts_for_source(
    counts_by_source: dict[str | None, dict[str, int]],
    source: str | None,
) -> tuple[tuple[str, int], ...]:
    source_counts = counts_by_source.get(
        source,
        {completeness: 0 for completeness in TRANSCRIPT_COMPLETENESS_ORDER},
    )
    return tuple(
        (completeness, source_counts.get(completeness, 0))
        for completeness in TRANSCRIPT_COMPLETENESS_ORDER
    )


def _limitation_counts_for_source(
    counts_by_source: dict[str | None, Counter[str]],
    source: str | None,
) -> tuple[tuple[str, int], ...]:
    source_counts = counts_by_source.get(source, Counter())
    return tuple(
        sorted(
            source_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
    )


def _conversation_filter_sql(
    *,
    source: str | None,
    session: str | None,
    transcript_completeness: str | None,
    additional_conditions: tuple[str, ...] = (),
    additional_parameters: tuple[object, ...] = (),
) -> tuple[str, tuple[object, ...]]:
    conditions: list[str] = []
    parameters: list[object] = []
    if source is not None:
        conditions.append("c.source = ?")
        parameters.append(source)
    if session is not None:
        conditions.append("c.source_session_id = ?")
        parameters.append(session)
    if transcript_completeness is not None:
        conditions.append("c.transcript_completeness = ?")
        parameters.append(transcript_completeness)
    conditions.extend(additional_conditions)
    parameters.extend(additional_parameters)
    if not conditions:
        return "", tuple(parameters)
    return f"WHERE {' AND '.join(conditions)}", tuple(parameters)


def _discover_archive_output_files(
    archive_root: Path,
) -> tuple[_ArchiveOutputFile, ...]:
    output_files: list[_ArchiveOutputFile] = []
    for source_dir in sorted(archive_root.iterdir()):
        if (
            not source_dir.is_dir()
            or source_dir.name in {RUNS_DIRECTORY, ARCHIVE_INDEX_DIRECTORY}
        ):
            continue
        for output_path in sorted(source_dir.glob(ARCHIVE_OUTPUT_GLOB), reverse=True):
            stat_result = output_path.stat()
            output_files.append(
                _ArchiveOutputFile(
                    output_path=output_path.expanduser().resolve(strict=False),
                    size_bytes=stat_result.st_size,
                    mtime_ns=stat_result.st_mtime_ns,
                )
            )
    return tuple(output_files)


def _diff_files(
    discovered_files: tuple[_ArchiveOutputFile, ...],
    indexed_files: dict[Path, _IndexedFileRow],
) -> _ArchiveIndexDiff:
    discovered_by_path = {file.output_path: file for file in discovered_files}
    added_files = tuple(
        file
        for path, file in discovered_by_path.items()
        if path not in indexed_files
    )
    updated_files = tuple(
        file
        for path, file in discovered_by_path.items()
        if path in indexed_files
        and (
            indexed_files[path].size_bytes != file.size_bytes
            or indexed_files[path].mtime_ns != file.mtime_ns
        )
    )
    removed_files = tuple(
        path for path in indexed_files if path not in discovered_by_path
    )
    return _ArchiveIndexDiff(
        added_files=added_files,
        updated_files=updated_files,
        removed_files=removed_files,
    )


def _load_indexed_files(connection: sqlite3.Connection) -> dict[Path, _IndexedFileRow]:
    rows = connection.execute(
        """
        SELECT output_path, size_bytes, mtime_ns
        FROM indexed_files
        """
    ).fetchall()
    return {
        Path(str(row["output_path"])): _IndexedFileRow(
            output_path=Path(str(row["output_path"])),
            size_bytes=int(row["size_bytes"]),
            mtime_ns=int(row["mtime_ns"]),
        )
        for row in rows
    }


def _build_status(
    *,
    archive_root: Path,
    index_path: Path,
    discovered_files: tuple[_ArchiveOutputFile, ...],
    indexed_files: dict[Path, _IndexedFileRow],
    conversation_count: int,
    reason: str | None,
    exists: bool,
    rebuild_required: bool,
    last_refreshed_at: str | None,
) -> ArchiveIndexStatus:
    diff = _diff_files(discovered_files, indexed_files)
    stale = rebuild_required or bool(
        diff.added_files or diff.updated_files or diff.removed_files
    )
    ready = exists and not stale
    if not exists:
        state = "missing"
    elif rebuild_required:
        state = "rebuild_required"
    elif stale:
        state = "stale"
        if reason is None:
            reason = "indexed archive outputs differ from files on disk"
    else:
        state = "ready"
    return ArchiveIndexStatus(
        archive_root=archive_root,
        index_path=index_path,
        state=state,
        exists=exists,
        ready=ready,
        stale=stale,
        rebuild_required=rebuild_required,
        reason=reason,
        file_count=len(discovered_files),
        indexed_file_count=len(indexed_files),
        conversation_count=conversation_count if exists else 0,
        indexed_conversation_count=conversation_count,
        added_file_count=len(diff.added_files),
        updated_file_count=len(diff.updated_files),
        removed_file_count=len(diff.removed_files),
        last_refreshed_at=last_refreshed_at,
    )


def _connect(
    path: Path,
    *,
    read_only: bool = False,
) -> sqlite3.Connection:
    if read_only:
        connection = sqlite3.connect(
            f"file:{path}?mode=ro",
            uri=True,
        )
    else:
        connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _initialize_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS indexed_files (
            output_path TEXT PRIMARY KEY,
            size_bytes INTEGER NOT NULL,
            mtime_ns INTEGER NOT NULL,
            row_count INTEGER NOT NULL,
            indexed_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS conversations (
            conversation_id INTEGER PRIMARY KEY,
            output_path TEXT NOT NULL REFERENCES indexed_files(output_path) ON DELETE CASCADE,
            row_number INTEGER NOT NULL,
            source TEXT NOT NULL,
            source_session_id TEXT,
            transcript_completeness TEXT NOT NULL,
            collected_at TEXT NOT NULL,
            message_count INTEGER NOT NULL,
            limitation_count INTEGER NOT NULL,
            has_provenance INTEGER NOT NULL CHECK (has_provenance IN (0, 1)),
            source_artifact_path TEXT,
            execution_context TEXT,
            UNIQUE(output_path, row_number)
        );
        CREATE TABLE IF NOT EXISTS conversation_messages (
            conversation_id INTEGER NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
            message_index INTEGER NOT NULL,
            role TEXT,
            text TEXT,
            text_search TEXT,
            PRIMARY KEY(conversation_id, message_index)
        );
        CREATE TABLE IF NOT EXISTS conversation_limitations (
            conversation_id INTEGER NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
            limitation TEXT NOT NULL,
            PRIMARY KEY(conversation_id, limitation)
        );
        CREATE INDEX IF NOT EXISTS idx_conversations_source_collected
            ON conversations (source, collected_at DESC, source_session_id DESC, output_path DESC, row_number DESC);
        CREATE INDEX IF NOT EXISTS idx_conversations_session
            ON conversations (source, source_session_id);
        CREATE INDEX IF NOT EXISTS idx_conversations_transcript
            ON conversations (transcript_completeness, collected_at DESC);
        CREATE INDEX IF NOT EXISTS idx_conversations_output_path
            ON conversations (output_path);
        CREATE INDEX IF NOT EXISTS idx_conversation_messages_role
            ON conversation_messages (role);
        CREATE INDEX IF NOT EXISTS idx_conversation_limitations_value
            ON conversation_limitations (limitation);
        """
    )
    _set_meta_value(connection, "schema_version", ARCHIVE_INDEX_SCHEMA_VERSION)


def _require_schema(connection: sqlite3.Connection) -> None:
    schema_version = _meta_value(connection, "schema_version")
    if schema_version != ARCHIVE_INDEX_SCHEMA_VERSION:
        raise sqlite3.DatabaseError(
            "archive index schema version mismatch: "
            f"expected {ARCHIVE_INDEX_SCHEMA_VERSION}, found {schema_version!r}"
        )


def _meta_value(connection: sqlite3.Connection, key: str) -> str | None:
    row = connection.execute(
        "SELECT value FROM meta WHERE key = ?",
        (key,),
    ).fetchone()
    if row is None:
        return None
    return str(row["value"])


def _set_meta_value(connection: sqlite3.Connection, key: str, value: str) -> None:
    connection.execute(
        """
        INSERT INTO meta (key, value)
        VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (key, value),
    )


def _scalar_int(connection: sqlite3.Connection, query: str) -> int:
    row = connection.execute(query).fetchone()
    if row is None:
        return 0
    return int(row[0])


def _message_role_value(value: object) -> str:
    if isinstance(value, str) and value in VALID_MESSAGE_ROLES:
        return value
    return INVALID_MESSAGE_ROLE_SENTINEL


def _searchable_text(value: object) -> str:
    if isinstance(value, str):
        return value.casefold()
    return ""


def _optional_string(value: object) -> str | None:
    if isinstance(value, str):
        return value
    return None


def _preview_text(text: str, *, max_length: int = 120) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_length:
        return compact
    return compact[: max_length - 3] + "..."


def _utcnow_string() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _validate_archive_root_directory(archive_root: Path) -> Path:
    resolved_root = archive_root.expanduser().resolve(strict=False)
    if not resolved_root.exists():
        raise ArchiveInspectError(f"archive root does not exist: {resolved_root}")
    if not resolved_root.is_dir():
        raise ArchiveInspectError(f"archive root is not a directory: {resolved_root}")
    return resolved_root


__all__ = [
    "ARCHIVE_INDEX_DIRECTORY",
    "ARCHIVE_INDEX_FILENAME",
    "ArchiveIndexRefreshReport",
    "ArchiveIndexStatus",
    "archive_index_path",
    "collect_indexed_limitation_counts",
    "ensure_archive_index",
    "find_indexed_archive_conversations",
    "inspect_archive_index",
    "list_indexed_archive_conversations",
    "refresh_archive_index",
    "summarize_indexed_archive_profile",
    "summarize_indexed_archive_stats",
]
