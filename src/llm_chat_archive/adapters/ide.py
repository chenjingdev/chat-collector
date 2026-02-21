from __future__ import annotations

import gzip
import hashlib
import json
import re
import sqlite3
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_chat_archive.models import Coverage, DisplayTurn, EventRecord, SessionRecord
from llm_chat_archive.token_usage import merge_usage_sum, normalize_usage_dict
from llm_chat_archive.utils import find_json_objects_in_text, parse_datetime


KEY_PATTERNS = (
    "openai.chatgpt%",
    "chat.%",
    "codex%",
    "workbench.panel.aichat.view.aichat.chatdata%",
    "memento/webview%",
    "workbench.view.extension.codex%",
    "workbench.view.extension.geminiChat%",
    "google.%",
    "antigravity%",
    "saoudrizwan.claude-dev%",
    "workbench.view.extension.claude-dev%",
    "externalCliAnalytics.lastTimestamp.%",
)


@dataclass(slots=True)
class StateRow:
    key: str
    value: Any
    source_path: Path


@dataclass(slots=True)
class LogSpec:
    pattern: str
    variant: str
    label: str


class _BaseIdeAdapter:
    source_name: str

    def __init__(self, app_support_root: Path | None = None) -> None:
        self.app_support_root = app_support_root

    def collect(self) -> list[SessionRecord]:
        sessions: list[SessionRecord] = []

        for db_path in self._state_db_paths():
            sessions.extend(self._parse_state_db(db_path))

        root = self._resolved_app_support_root()
        for spec in self._log_specs():
            for log_path in sorted(root.glob(spec.pattern)):
                session = self._parse_log(log_path, spec)
                if session is not None:
                    sessions.append(session)

        return sessions

    def _state_db_paths(self) -> list[Path]:
        root = self._resolved_app_support_root()
        global_db = root / "User" / "globalStorage" / "state.vscdb"
        workspace_glob = root / "User" / "workspaceStorage"
        db_paths: list[Path] = []

        if global_db.exists():
            db_paths.append(global_db)
        if workspace_glob.exists():
            db_paths.extend(sorted(workspace_glob.glob("*/state.vscdb")))

        return [path for path in db_paths if path.is_file()]

    def _resolved_app_support_root(self) -> Path:
        if self.app_support_root:
            return self.app_support_root
        return Path.home() / "Library" / "Application Support" / self.app_name()

    def app_name(self) -> str:
        raise NotImplementedError

    def _key_patterns(self) -> tuple[str, ...]:
        return KEY_PATTERNS

    def _log_specs(self) -> tuple[LogSpec, ...]:
        return (
            LogSpec(
                pattern="logs/**/exthost/openai.chatgpt/Codex.log",
                variant="codex-extension",
                label="Codex.log",
            ),
        )

    def _variant_from_state_key(self, key: str) -> str:
        lowered = key.lower()
        if lowered.startswith("workbench.panel.aichat.view.aichat.chatdata"):
            return "legacy-aichat"
        if "openai.chatgpt" in lowered or "codex" in lowered:
            return "codex-extension"
        if "gemini" in lowered or lowered.startswith("google."):
            return "gemini-extension"
        if "claude" in lowered:
            return "claude-extension"
        return "ide-state"

    def _parse_state_db(self, db_path: Path) -> list[SessionRecord]:
        rows, warnings = self._read_state_rows(db_path)
        if not rows and not warnings:
            return []

        grouped: dict[str, list[StateRow]] = {}
        for row in rows:
            variant = self._variant_from_state_key(row.key)
            grouped.setdefault(variant, []).append(row)

        if not grouped:
            grouped["ide-state"] = []

        sessions: list[SessionRecord] = []
        for variant in sorted(grouped.keys()):
            session = self._build_state_session(
                db_path=db_path,
                variant=variant,
                rows=grouped[variant],
                warnings=warnings,
            )
            sessions.append(session)
        return sessions

    def _build_state_session(
        self,
        *,
        db_path: Path,
        variant: str,
        rows: list[StateRow],
        warnings: list[str],
    ) -> SessionRecord:
        events: list[EventRecord] = []
        display_turns: list[DisplayTurn] = []
        session_warnings = list(dict.fromkeys(warnings))

        file_ts = datetime.fromtimestamp(db_path.stat().st_mtime, tz=timezone.utc)
        started_at = file_ts
        ended_at = file_ts

        for row in rows:
            row_ts = parse_datetime(_find_timestamp(row.value)) or file_ts
            if row_ts < started_at:
                started_at = row_ts
            if row_ts > ended_at:
                ended_at = row_ts

            events.append(
                EventRecord(
                    timestamp=row_ts,
                    role="system",
                    event_kind="state.kv",
                    content_text=f"key={row.key}",
                    raw_json={"key": row.key, "value": row.value},
                )
            )
            for turn in _extract_turns_from_any(row.value, row_ts):
                _append_turn(display_turns, turn)

        coverage = Coverage.PARTIAL if display_turns else Coverage.METADATA_ONLY

        digest_input = f"{db_path}|{variant}"
        digest = hashlib.sha1(digest_input.encode("utf-8")).hexdigest()[:16]
        session_id = f"{self.source_name}-state-{variant}-{digest}"

        if not display_turns:
            session_warnings.append(f"no full transcript reconstructed from state.vscdb ({variant})")

        if not events:
            session_warnings.append("no parseable state rows")

        return SessionRecord(
            source=self.source_name,
            source_session_id=session_id,
            source_variant=variant,
            project=db_path.parent.name,
            cwd=None,
            started_at=started_at,
            ended_at=ended_at,
            raw_events=events,
            display_turns=display_turns,
            coverage=coverage,
            evidence_paths=[str(db_path)],
            source_files=[db_path],
            warnings=list(dict.fromkeys(session_warnings)),
        )

    def _read_state_rows(self, db_path: Path) -> tuple[list[StateRow], list[str]]:
        rows: list[StateRow] = []
        warnings: list[str] = []

        try:
            conn = _open_sqlite_readonly(db_path)
            conn.row_factory = sqlite3.Row
        except sqlite3.Error as exc:
            return rows, [f"sqlite open failed: {exc}"]

        try:
            tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            table_names = [row[0] for row in tables if row and row[0]]

            patterns = self._key_patterns()
            where_clause = " OR ".join(["key LIKE ?" for _ in patterns])

            for table_name in table_names:
                columns = conn.execute(f"PRAGMA table_info('{table_name}')").fetchall()
                col_names = {col[1] for col in columns if len(col) > 1}
                if not {"key", "value"}.issubset(col_names):
                    continue

                query = f"SELECT key, value FROM '{table_name}' WHERE {where_clause}"
                for row in conn.execute(query, patterns):
                    key = row["key"]
                    value = row["value"]
                    if not isinstance(key, str):
                        continue
                    parsed_value = _parse_loose_json(value)
                    rows.append(StateRow(key=key, value=parsed_value, source_path=db_path))

        except sqlite3.Error as exc:
            warnings.append(f"sqlite query failed: {exc}")
        finally:
            conn.close()

        return rows, warnings

    def _parse_log(self, log_path: Path, spec: LogSpec) -> SessionRecord | None:
        events: list[EventRecord] = []
        display_turns: list[DisplayTurn] = []
        warnings: list[str] = []

        try:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:
            return SessionRecord(
                source=self.source_name,
                source_session_id=f"{self.source_name}-log-open-error-{spec.variant}",
                source_variant=spec.variant,
                project=None,
                cwd=None,
                started_at=None,
                ended_at=None,
                raw_events=[],
                display_turns=[],
                coverage=Coverage.METADATA_ONLY,
                evidence_paths=[str(log_path)],
                source_files=[log_path],
                warnings=[f"log read failed: {exc}"],
            )

        if not lines:
            return None

        started_at = None
        ended_at = None

        for line in lines:
            line_ts = _parse_log_timestamp(line)
            if line_ts:
                started_at = line_ts if started_at is None or line_ts < started_at else started_at
                ended_at = line_ts if ended_at is None or line_ts > ended_at else ended_at

            json_objects = []
            stripped = line.strip()
            try:
                maybe_obj = json.loads(stripped)
                if isinstance(maybe_obj, dict):
                    json_objects.append(maybe_obj)
            except json.JSONDecodeError:
                json_objects.extend(find_json_objects_in_text(line))

            events.append(
                EventRecord(
                    timestamp=line_ts,
                    role="system",
                    event_kind=f"log.{spec.variant}",
                    content_text=line,
                    raw_json={"line": line, "json_objects": json_objects},
                )
            )

            for obj in json_objects:
                for turn in _extract_turns_from_any(obj, line_ts):
                    _append_turn(display_turns, turn)

        if not started_at:
            file_ts = datetime.fromtimestamp(log_path.stat().st_mtime, tz=timezone.utc)
            started_at = file_ts
            ended_at = file_ts

        coverage = Coverage.PARTIAL if display_turns else Coverage.METADATA_ONLY
        if not display_turns:
            warnings.append(f"no full transcript reconstructed from {spec.label}")

        digest_input = f"{log_path}|{spec.variant}"
        digest = hashlib.sha1(digest_input.encode("utf-8")).hexdigest()[:16]
        session_id = f"{self.source_name}-log-{spec.variant}-{digest}"

        return SessionRecord(
            source=self.source_name,
            source_session_id=session_id,
            source_variant=spec.variant,
            project=log_path.parent.name,
            cwd=None,
            started_at=started_at,
            ended_at=ended_at,
            raw_events=events,
            display_turns=display_turns,
            coverage=coverage,
            evidence_paths=[str(log_path)],
            source_files=[log_path],
            warnings=warnings,
        )


class CursorAdapter(_BaseIdeAdapter):
    source_name = "cursor"

    def app_name(self) -> str:
        return "Cursor"

    def collect(self) -> list[SessionRecord]:
        composer_sessions = self._collect_composer_sessions()
        if composer_sessions:
            return composer_sessions
        return super().collect()

    def _collect_composer_sessions(self) -> list[SessionRecord]:
        root = self._resolved_app_support_root()
        global_db = root / "User" / "globalStorage" / "state.vscdb"
        workspace_root = root / "User" / "workspaceStorage"
        if not global_db.is_file() or not workspace_root.exists():
            return []

        sessions: list[SessionRecord] = []
        for workspace_db in sorted(workspace_root.glob("*/state.vscdb")):
            if not workspace_db.is_file():
                continue
            sessions.extend(self._parse_workspace_composer_sessions(workspace_db, global_db))
        return sessions

    def _parse_workspace_composer_sessions(self, workspace_db: Path, global_db: Path) -> list[SessionRecord]:
        sessions: list[SessionRecord] = []

        try:
            ws_conn = _open_sqlite_readonly(workspace_db)
            ws_conn.row_factory = sqlite3.Row
        except sqlite3.Error:
            return sessions

        try:
            gl_conn = _open_sqlite_readonly(global_db)
            gl_conn.row_factory = sqlite3.Row
        except sqlite3.Error:
            ws_conn.close()
            return sessions

        try:
            ws_item_table = _find_table_with_key_value(ws_conn, preferred="ItemTable")
            gl_kv_table = _find_table_with_key_value(gl_conn, preferred="cursorDiskKV")
            if ws_item_table is None or gl_kv_table is None:
                return sessions

            composer_index_raw = _read_kv_value(ws_conn, ws_item_table, "composer.composerData")
            composer_index = _parse_loose_json(composer_index_raw)
            if not isinstance(composer_index, dict):
                return sessions

            all_composers = composer_index.get("allComposers")
            if not isinstance(all_composers, list):
                return sessions

            for composer in all_composers:
                session = self._build_composer_session(
                    workspace_db=workspace_db,
                    global_db=global_db,
                    gl_conn=gl_conn,
                    gl_kv_table=gl_kv_table,
                    composer=composer,
                )
                if session is not None:
                    sessions.append(session)

        finally:
            gl_conn.close()
            ws_conn.close()

        return sessions

    def _build_composer_session(
        self,
        *,
        workspace_db: Path,
        global_db: Path,
        gl_conn: sqlite3.Connection,
        gl_kv_table: str,
        composer: Any,
    ) -> SessionRecord | None:
        if not isinstance(composer, dict):
            return None

        composer_id = composer.get("composerId")
        if not isinstance(composer_id, str) or not composer_id:
            return None

        warnings: list[str] = []
        events: list[EventRecord] = []
        display_turns: list[DisplayTurn] = []
        token_usage: dict[str, int] = {}

        started_at = _parse_datetime_any(composer.get("createdAt"), composer.get("timestamp"))
        ended_at = _parse_datetime_any(composer.get("lastUpdatedAt"), composer.get("updatedAt"))

        events.append(
            EventRecord(
                timestamp=started_at or ended_at,
                role="system",
                event_kind="composer.meta",
                content_text=f"name={composer.get('name') or ''}",
                raw_json=composer,
            )
        )

        composer_data_raw = _read_kv_value(gl_conn, gl_kv_table, f"composerData:{composer_id}")
        composer_data = _parse_loose_json(composer_data_raw)
        if not isinstance(composer_data, dict):
            warnings.append("composerData not found in cursorDiskKV")
            composer_data = {}
        else:
            meta_ts = _parse_datetime_any(
                composer_data.get("createdAt"),
                composer_data.get("lastUpdatedAt"),
                composer_data.get("updatedAt"),
                composer_data.get("timestamp"),
            )
            if meta_ts and (started_at is None or meta_ts < started_at):
                started_at = meta_ts
            if meta_ts and (ended_at is None or meta_ts > ended_at):
                ended_at = meta_ts

        events.append(
            EventRecord(
                timestamp=started_at or ended_at,
                role="system",
                event_kind="composer.data",
                raw_json=composer_data if composer_data else {"missing": True},
            )
        )

        bubble_map = _read_cursor_bubble_map(gl_conn, gl_kv_table, composer_id)
        ordered_bubble_ids = _ordered_cursor_bubble_ids(composer_data, bubble_map)
        if not ordered_bubble_ids:
            warnings.append("no bubble records found in cursorDiskKV")

        for bubble_id in ordered_bubble_ids:
            bubble = bubble_map.get(bubble_id)
            if bubble is None:
                warnings.append(f"bubble missing for id={bubble_id}")
                continue

            bubble_ts = _parse_datetime_any(
                bubble.get("timestamp"),
                bubble.get("createdAt"),
                bubble.get("lastUpdatedAt"),
                bubble.get("updatedAt"),
            )
            if bubble_ts and (started_at is None or bubble_ts < started_at):
                started_at = bubble_ts
            if bubble_ts and (ended_at is None or bubble_ts > ended_at):
                ended_at = bubble_ts

            role, display_role = _cursor_bubble_roles(bubble.get("type"))
            event_text = _cursor_bubble_text(bubble)
            display_text = _cursor_bubble_display_text(bubble)
            kind = "composer.bubble.system"
            if role == "user":
                kind = "composer.bubble.user"
            elif role == "assistant":
                kind = "composer.bubble.assistant"

            events.append(
                EventRecord(
                    timestamp=bubble_ts,
                    role=role,
                    event_kind=kind,
                    content_text=event_text or None,
                    raw_json=bubble,
                )
            )
            if display_role and display_text:
                _append_turn(display_turns, DisplayTurn(timestamp=bubble_ts, role=display_role, text=display_text))

            usage = _normalize_cursor_token_usage(bubble.get("tokenCount"))
            if usage:
                token_usage = merge_usage_sum(token_usage, usage)

        if not display_turns:
            warnings.append("no display turns reconstructed from composer bubbles")

        coverage = Coverage.FULL if display_turns and len(events) > 2 else Coverage.PARTIAL if events else Coverage.METADATA_ONLY

        return SessionRecord(
            source=self.source_name,
            source_session_id=composer_id,
            source_variant="composer",
            project=workspace_db.parent.name,
            cwd=None,
            started_at=started_at,
            ended_at=ended_at,
            raw_events=events,
            display_turns=display_turns,
            coverage=coverage,
            evidence_paths=[str(workspace_db), str(global_db)],
            source_files=[workspace_db, global_db],
            warnings=list(dict.fromkeys(warnings)),
            token_usage=token_usage,
        )

    def _variant_from_state_key(self, key: str) -> str:
        lowered = key.lower()
        if lowered.startswith("workbench.panel.aichat.view.aichat.chatdata"):
            return "legacy-aichat"
        if "openai.chatgpt" in lowered or "codex" in lowered:
            return "codex-extension"
        if "gemini" in lowered or "google.geminicodeassist" in lowered:
            return "gemini-extension"
        if "claude" in lowered:
            return "claude-extension"
        return "ide-state"


def _find_table_with_key_value(conn: sqlite3.Connection, preferred: str | None = None) -> str | None:
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    table_names = [row[0] for row in tables if row and row[0]]

    def has_key_value(table_name: str) -> bool:
        columns = conn.execute(f"PRAGMA table_info('{table_name}')").fetchall()
        col_names = {col[1] for col in columns if len(col) > 1}
        return {"key", "value"}.issubset(col_names)

    if preferred and preferred in table_names and has_key_value(preferred):
        return preferred

    for table_name in table_names:
        if has_key_value(table_name):
            return table_name
    return None


def _open_sqlite_readonly(db_path: Path) -> sqlite3.Connection:
    # Read-only URI mode lowers lock contention on actively written IDE DB files.
    uri = f"{db_path.resolve().as_uri()}?mode=ro"
    return sqlite3.connect(uri, uri=True, timeout=5.0)


def _read_kv_value(conn: sqlite3.Connection, table_name: str, key: str) -> Any | None:
    try:
        row = conn.execute(f"SELECT value FROM '{table_name}' WHERE key = ?", (key,)).fetchone()
    except sqlite3.Error:
        return None
    if row is None:
        return None
    return row[0]


def _parse_datetime_any(*values: Any) -> datetime | None:
    for value in values:
        dt = parse_datetime(value)
        if dt is not None:
            return dt

        if isinstance(value, str) and value.strip().isdigit():
            value = int(value.strip())
        if isinstance(value, (int, float)):
            raw = float(value)
            if raw > 1e11:
                raw = raw / 1000.0
            try:
                return datetime.fromtimestamp(raw, tz=timezone.utc)
            except (ValueError, OSError):
                continue

    return None


def _read_cursor_bubble_map(
    conn: sqlite3.Connection,
    table_name: str,
    composer_id: str,
) -> dict[str, dict[str, Any]]:
    bubble_map: dict[str, dict[str, Any]] = {}
    prefix = f"bubbleId:{composer_id}:"
    try:
        rows = conn.execute(
            f"SELECT key, value FROM '{table_name}' WHERE key LIKE ?",
            (f"{prefix}%",),
        ).fetchall()
    except sqlite3.Error:
        return bubble_map

    for row in rows:
        key = row[0]
        value = row[1]
        if isinstance(key, bytes):
            try:
                key = key.decode("utf-8")
            except UnicodeDecodeError:
                key = key.decode("utf-8", errors="replace")
        if not isinstance(key, str):
            continue
        if not key.startswith(prefix):
            continue

        bubble_id = key[len(prefix) :]
        parsed = _parse_loose_json(value)
        if isinstance(parsed, dict):
            bubble_map[bubble_id] = parsed
        else:
            bubble_map[bubble_id] = {"bubbleId": bubble_id, "raw": parsed}

    return bubble_map


def _ordered_cursor_bubble_ids(composer_data: dict[str, Any], bubble_map: dict[str, dict[str, Any]]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()

    headers = composer_data.get("fullConversationHeadersOnly")
    if isinstance(headers, list):
        for item in headers:
            if not isinstance(item, dict):
                continue
            bubble_id = item.get("bubbleId")
            if not isinstance(bubble_id, str) or not bubble_id:
                continue
            if bubble_id in seen:
                continue
            if bubble_id in bubble_map:
                ordered.append(bubble_id)
                seen.add(bubble_id)

    remaining = [bid for bid in bubble_map.keys() if bid not in seen]
    remaining.sort(key=lambda bid: _bubble_sort_key(bubble_map.get(bid) or {}))
    ordered.extend(remaining)
    return ordered


def _bubble_sort_key(bubble: dict[str, Any]) -> tuple[int, str]:
    ts = _parse_datetime_any(
        bubble.get("timestamp"),
        bubble.get("createdAt"),
        bubble.get("lastUpdatedAt"),
        bubble.get("updatedAt"),
    )
    ts_value = int(ts.timestamp()) if ts is not None else 0
    bubble_id = bubble.get("bubbleId")
    bubble_id = bubble_id if isinstance(bubble_id, str) else ""
    return ts_value, bubble_id


def _cursor_bubble_roles(bubble_type: Any) -> tuple[str, str | None]:
    if bubble_type in {1, "1", "user"}:
        return "user", "user"
    if bubble_type in {2, "2", "ai", "assistant"}:
        return "assistant", "assistant"
    return "system", None


def _cursor_bubble_text(bubble: dict[str, Any]) -> str:
    # Prefer natural-language text, then enrich with code blocks when present.
    primary_text = _cursor_primary_text(bubble)
    code_blocks_text = _cursor_code_blocks_text(bubble.get("codeBlocks"))
    if primary_text:
        if code_blocks_text and code_blocks_text != primary_text:
            return f"{primary_text}\n\n{code_blocks_text}".strip()
        return primary_text

    # If no direct text, fall back to code/reasoning/tool outputs in that order.
    if code_blocks_text:
        return code_blocks_text

    reasoning_text = _cursor_reasoning_text(bubble)
    if reasoning_text:
        return reasoning_text

    tool_text = _cursor_toolformer_text(bubble.get("toolFormerData"))
    if tool_text:
        return tool_text

    return ""


def _cursor_bubble_display_text(bubble: dict[str, Any]) -> str:
    primary_text = _cursor_primary_text(bubble)
    code_blocks_text = _cursor_code_blocks_text(bubble.get("codeBlocks"))
    if primary_text:
        if code_blocks_text and code_blocks_text != primary_text:
            return f"{primary_text}\n\n{code_blocks_text}".strip()
        return primary_text
    if code_blocks_text:
        return code_blocks_text
    return ""


def _cursor_primary_text(bubble: dict[str, Any]) -> str:
    direct = bubble.get("text")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    raw = bubble.get("rawText")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()

    delegate = bubble.get("delegate")
    if isinstance(delegate, dict):
        candidate = delegate.get("a")
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
        nested = _pick_text(delegate)
        if nested:
            return nested

    init_text = bubble.get("initText")
    if isinstance(init_text, str) and init_text.strip():
        parsed = _parse_loose_json(init_text)
        if isinstance(parsed, dict):
            parsed_text = _pick_text(parsed)
            if parsed_text:
                return parsed_text

    return ""


def _cursor_code_blocks_text(raw_code_blocks: Any) -> str:
    if not isinstance(raw_code_blocks, list):
        return ""

    blocks: list[str] = []
    for block in raw_code_blocks:
        if not isinstance(block, dict):
            continue

        content = _cursor_value_to_text(block.get("content"))
        if not content:
            content = _cursor_value_to_text(block.get("code"))
        if not content:
            content = _cursor_value_to_text(block.get("text"))
        if not content:
            content = _cursor_value_to_text(block.get("value"))
        if not content:
            continue

        lang = block.get("languageId")
        if not isinstance(lang, str) or not lang.strip():
            lang = block.get("language")

        if isinstance(lang, str) and lang.strip():
            blocks.append(f"```{lang.strip()}\n{content}\n```")
        else:
            blocks.append(content)

    return "\n\n".join(blocks).strip()


def _cursor_reasoning_text(bubble: dict[str, Any]) -> str:
    for key in ("thinking", "reasoning"):
        text = _cursor_value_to_text(bubble.get(key))
        if text:
            return text
    return ""


def _cursor_toolformer_text(raw_tool: Any) -> str:
    if not isinstance(raw_tool, dict):
        return ""

    name = raw_tool.get("name")
    name = name.strip() if isinstance(name, str) else ""
    result = _cursor_value_to_text(raw_tool.get("result"))
    if not result:
        result = _cursor_value_to_text(raw_tool.get("output"))

    if name and result:
        return f"[tool:{name}] {result}".strip()
    if result:
        return result
    if name:
        return f"[tool:{name}]"

    status = raw_tool.get("status")
    if isinstance(status, str) and status.strip():
        return f"[tool-status:{status.strip()}]"

    additional = raw_tool.get("additionalData")
    if isinstance(additional, dict):
        additional_status = additional.get("status")
        if isinstance(additional_status, str) and additional_status.strip():
            return f"[tool-status:{additional_status.strip()}]"

    return ""


def _cursor_value_to_text(value: Any) -> str:
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return ""
        parsed = _parse_loose_json(stripped)
        if parsed is value:
            return stripped
        return _cursor_value_to_text(parsed)

    if isinstance(value, dict):
        text = _pick_text(value)
        if text:
            return text
        return json.dumps(value, ensure_ascii=False, sort_keys=True)

    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            text = _cursor_value_to_text(item)
            if text:
                parts.append(text)
        return "\n".join(parts).strip()

    if value is None:
        return ""
    return str(value).strip()


def _normalize_cursor_token_usage(raw: Any) -> dict[str, int]:
    if not isinstance(raw, dict):
        return {}
    mapped = {
        "input_tokens": raw.get("inputTokens"),
        "output_tokens": raw.get("outputTokens"),
        "total_tokens": raw.get("totalTokens"),
    }
    return normalize_usage_dict(mapped)


def _parse_loose_json(value: Any) -> Any:
    if isinstance(value, bytes):
        value = _decode_blob_text(value)
    if isinstance(value, str):
        stripped = value.lstrip("\ufeff").strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return value
    return value


def _decode_blob_text(blob: bytes) -> str:
    candidates: list[bytes] = [blob]

    # gzip stream
    if len(blob) >= 2 and blob[:2] == b"\x1f\x8b":
        try:
            candidates.append(gzip.decompress(blob))
        except OSError:
            pass

    # zlib stream
    if len(blob) >= 2 and blob[0] == 0x78:
        try:
            candidates.append(zlib.decompress(blob))
        except zlib.error:
            pass

    for candidate in candidates:
        for encoding in ("utf-8", "utf-8-sig", "utf-16", "utf-16-le", "utf-16-be"):
            try:
                text = candidate.decode(encoding)
            except UnicodeDecodeError:
                continue
            if text:
                return text

    return blob.decode("utf-8", errors="replace")


def _find_timestamp(value: Any) -> Any:
    if isinstance(value, dict):
        for key in ("timestamp", "ts", "time", "createdAt", "updatedAt"):
            if key in value:
                return value[key]
        for nested in value.values():
            found = _find_timestamp(nested)
            if found is not None:
                return found
    if isinstance(value, list):
        for item in value:
            found = _find_timestamp(item)
            if found is not None:
                return found
    return None


def _extract_turns_from_any(value: Any, default_ts: datetime | None) -> list[DisplayTurn]:
    turns: list[DisplayTurn] = []

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            role = _pick_role(node)
            text = _pick_text(node)
            ts = parse_datetime(_find_timestamp(node)) or default_ts
            if role and text:
                turns.append(DisplayTurn(timestamp=ts, role=role, text=text))
            for nested in node.values():
                visit(nested)
            return

        if isinstance(node, list):
            for item in node:
                visit(item)

    visit(value)
    deduped: list[DisplayTurn] = []
    for turn in turns:
        _append_turn(deduped, turn)
    return deduped


def _pick_role(node: dict[str, Any]) -> str | None:
    role_candidates = [
        node.get("role"),
        node.get("author"),
        node.get("speaker"),
        node.get("sender"),
        node.get("type"),
    ]
    for candidate in role_candidates:
        if not isinstance(candidate, str):
            continue
        lowered = candidate.lower()
        if "assistant" in lowered or lowered == "ai" or lowered == "gemini":
            return "assistant"
        if "user" in lowered or lowered in {"human", "me"}:
            return "user"
        if "system" in lowered or lowered in {"info", "error"}:
            return "system_display"
    return None


def _pick_text(node: dict[str, Any]) -> str | None:
    candidates = [
        node.get("text"),
        node.get("rawText"),
        node.get("initText"),
        node.get("message"),
        node.get("prompt"),
        node.get("content"),
        node.get("value"),
    ]
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
        if isinstance(candidate, dict):
            nested_text = _pick_text(candidate)
            if nested_text:
                return nested_text
        if isinstance(candidate, list):
            pieces: list[str] = []
            for item in candidate:
                if isinstance(item, str) and item.strip():
                    pieces.append(item.strip())
                elif isinstance(item, dict):
                    nested = _pick_text(item)
                    if nested:
                        pieces.append(nested)
            if pieces:
                return "\n".join(pieces).strip()
    return None


def _append_turn(turns: list[DisplayTurn], turn: DisplayTurn) -> None:
    if not turn.text.strip():
        return
    if not turns:
        turns.append(turn)
        return
    last = turns[-1]
    if last.role == turn.role and last.text == turn.text and last.timestamp == turn.timestamp:
        return
    turns.append(turn)


def _parse_log_timestamp(line: str) -> datetime | None:
    iso_match = re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z", line)
    if iso_match:
        return parse_datetime(iso_match.group(0))

    bracket_match = re.search(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?)\]", line)
    if bracket_match:
        raw = bracket_match.group(1)
        dt = parse_datetime(raw.replace(" ", "T") + "+00:00")
        if dt:
            return dt

    prefix_match = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?)", line)
    if prefix_match:
        dt = parse_datetime(prefix_match.group(1).replace(" ", "T") + "+00:00")
        if dt:
            return dt

    return None
