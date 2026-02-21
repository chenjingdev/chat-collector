from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

from llm_chat_archive import __version__
from llm_chat_archive.adapters import AntigravityAdapter, ClaudeAdapter, CodexAdapter, CursorAdapter, GeminiAdapter
from llm_chat_archive.fingerprint import compute_fingerprint
from llm_chat_archive.index_store import SessionIndex
from llm_chat_archive.models import SessionRecord
from llm_chat_archive.render import write_session_markdowns

VALID_SOURCES = ("codex", "claude", "cursor", "antigravity", "gemini")


@dataclass(slots=True)
class CollectConfig:
    output_root: Path
    sources: list[str]
    full_rebuild: bool = False
    timezone: str | None = None
    since: date | None = None
    copy_source_jsonl: bool = False
    update_existing: bool = False
    codex_home: Path | None = None
    claude_home: Path | None = None
    cursor_support: Path | None = None
    antigravity_support: Path | None = None
    gemini_home: Path | None = None


@dataclass(slots=True)
class CollectResult:
    scanned: int
    written: int
    skipped_since: int
    skipped_empty: int
    skipped_existing: int
    sources_stats: dict[str, int]


def run_collect(config: CollectConfig) -> CollectResult:
    tzinfo = _resolve_tzinfo(config.timezone)
    output_root = config.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    preserved_summaries = _snapshot_summaries(output_root) if config.full_rebuild else {}

    if config.full_rebuild:
        _cleanup_generated_dirs(output_root)

    index = SessionIndex(output_root)
    if not config.full_rebuild:
        index.load()
        _migrate_index_dir_names(index, output_root)
    else:
        index.clear()

    sources_stats: dict[str, int] = {}
    collected_sessions = list(_collect_sessions(config.sources, config, sources_stats))
    scanned = len(collected_sessions)
    sessions = [session for session in collected_sessions if _has_display_chat(session)]
    skipped_empty = scanned - len(sessions)

    for session in sessions:
        session.fingerprint = compute_fingerprint(session)

    sessions.sort(key=lambda session: session.sort_key())

    written = 0
    skipped_since = 0
    skipped_existing = 0

    for session in sessions:
        anchor = _session_anchor(session, tzinfo)
        session_date = anchor.date()

        if config.since and session_date < config.since:
            skipped_since += 1
            continue

        date_key = session_date.isoformat()
        fingerprint = session.fingerprint or compute_fingerprint(session)

        existing = index.get(date_key, fingerprint)
        had_existing = existing is not None
        if existing is None:
            session_no = index.next_session_no(date_key)
            relative_dir = _format_relative_dir(anchor, session_no, session.source, session.source_variant)
            existing = index.put(
                date_key=date_key,
                fingerprint=fingerprint,
                session_no=session_no,
                relative_dir=relative_dir,
            )
        else:
            desired_relative_dir = _format_relative_dir(anchor, existing.session_no, session.source, session.source_variant)
            if existing.relative_dir != desired_relative_dir:
                _relocate_session_dir(output_root, existing.relative_dir, desired_relative_dir)
                existing.relative_dir = desired_relative_dir

        if had_existing and not config.update_existing:
            skipped_existing += 1
            continue

        session_dir = output_root / existing.relative_dir
        session_dir.mkdir(parents=True, exist_ok=True)

        write_session_markdowns(session_dir, session, preserve_summary=True)
        if config.full_rebuild and fingerprint in preserved_summaries:
            summary_path = session_dir / "chat요약.md"
            summary_path.write_text(preserved_summaries[fingerprint], encoding="utf-8")
        _write_session_meta(session_dir / "session.meta.json", session)

        if config.copy_source_jsonl:
            _copy_source_jsonl(session_dir, session)

        written += 1

    index.save()
    return CollectResult(
        scanned=scanned,
        written=written,
        skipped_since=skipped_since,
        skipped_empty=skipped_empty,
        skipped_existing=skipped_existing,
        sources_stats=sources_stats,
    )


def _collect_sessions(sources: Iterable[str], config: CollectConfig, sources_stats: dict[str, int]) -> Iterable[SessionRecord]:
    normalized = [source.strip().lower() for source in sources if source.strip()]
    unknown = sorted(set(normalized) - set(VALID_SOURCES))
    if unknown:
        raise ValueError(f"Unsupported sources: {', '.join(unknown)}")

    for source in normalized:
        if source == "codex":
            adapter = CodexAdapter(codex_home=config.codex_home)
        elif source == "claude":
            adapter = ClaudeAdapter(claude_home=config.claude_home)
        elif source == "cursor":
            adapter = CursorAdapter(app_support_root=config.cursor_support)
        elif source == "antigravity":
            adapter = AntigravityAdapter(app_support_root=config.antigravity_support)
        elif source == "gemini":
            adapter = GeminiAdapter(gemini_home=config.gemini_home)
        else:
            continue

        session_count = 0
        for session in adapter.collect():
            session_count += 1
            yield session
        sources_stats[source] = session_count


def _has_display_chat(session: SessionRecord) -> bool:
    if not session.display_turns:
        return False
    for turn in session.display_turns:
        if turn.text.strip():
            return True
    return False


def _resolve_tzinfo(timezone_name: str | None):
    if timezone_name:
        return ZoneInfo(timezone_name)
    return datetime.now().astimezone().tzinfo


def _session_anchor(session: SessionRecord, tzinfo) -> datetime:
    dt = session.started_at or session.ended_at
    if dt is None and session.raw_events:
        dt = session.raw_events[0].timestamp
    if dt is None:
        dt = datetime.now().astimezone()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.now().astimezone().tzinfo)
    return dt.astimezone(tzinfo)


def _format_relative_dir(anchor: datetime, session_no: int, source: str, source_variant: str | None) -> str:
    source_label = _build_source_label(source, source_variant)
    return f"{anchor.year:04d}년/{anchor.month:02d}월/{anchor.day}일/세션{session_no}({source_label})"


def _write_session_meta(path: Path, session: SessionRecord) -> None:
    now = datetime.now().astimezone().isoformat()
    first_seen_at = now

    if path.exists():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
            first_seen = previous.get("first_seen_at")
            if isinstance(first_seen, str) and first_seen:
                first_seen_at = first_seen
        except (json.JSONDecodeError, OSError):
            pass

    payload = {
        "fingerprint": session.fingerprint,
        "source": session.source,
        "source_variant": session.source_variant,
        "source_label": _build_source_label(session.source, session.source_variant),
        "source_session_id": session.source_session_id,
        "source_files": [str(path) for path in session.source_files],
        "evidence_paths": session.evidence_paths,
        "coverage": session.coverage.value,
        "parser_version": __version__,
        "warnings": session.warnings,
        "project": session.project,
        "cwd": session.cwd,
        "started_at": session.started_at.isoformat() if session.started_at else None,
        "ended_at": session.ended_at.isoformat() if session.ended_at else None,
        "event_count": len(session.raw_events),
        "display_turn_count": len(session.display_turns),
        "token_usage": session.token_usage,
        "first_seen_at": first_seen_at,
        "last_synced_at": now,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _copy_source_jsonl(session_dir: Path, session: SessionRecord) -> None:
    source_dir = session_dir / "source_files"
    source_dir.mkdir(parents=True, exist_ok=True)

    copied = set()
    for source_file in session.source_files:
        if source_file.suffix.lower() != ".jsonl":
            continue
        if not source_file.exists():
            continue

        destination = source_dir / source_file.name
        if destination.exists() and destination.read_bytes() == source_file.read_bytes():
            copied.add(destination.name)
            continue

        if destination.exists() and destination.name in copied:
            continue

        if destination.exists():
            destination = source_dir / f"{len(copied)+1:03d}_{source_file.name}"
        shutil.copy2(source_file, destination)
        copied.add(destination.name)


def _cleanup_generated_dirs(output_root: Path) -> None:
    for child in output_root.iterdir():
        if not child.is_dir():
            continue
        if child.name == ".index":
            shutil.rmtree(child)
            continue
        if len(child.name) == 5 and child.name.endswith("년") and child.name[:4].isdigit():
            shutil.rmtree(child)


def _snapshot_summaries(output_root: Path) -> dict[str, str]:
    snapshots: dict[str, str] = {}

    for meta_path in output_root.glob("**/session.meta.json"):
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        fingerprint = payload.get("fingerprint")
        if not isinstance(fingerprint, str) or not fingerprint:
            continue

        summary_path = meta_path.with_name("chat요약.md")
        if not summary_path.exists():
            continue

        try:
            snapshots[fingerprint] = summary_path.read_text(encoding="utf-8")
        except OSError:
            continue

    return snapshots


def _source_from_fingerprint(fingerprint: str) -> str:
    if ":" not in fingerprint:
        return "unknown"
    return fingerprint.split(":", 1)[0] or "unknown"


def _sanitize_source_label(source: str) -> str:
    normalized = re.sub(r"\s+", "-", source.strip().lower())
    normalized = re.sub(r"[^0-9a-zA-Z가-힣_-]", "", normalized)
    return normalized or "unknown"


def _build_source_label(source: str, source_variant: str | None) -> str:
    base = _sanitize_source_label(source)
    if source_variant:
        variant = _sanitize_source_label(source_variant)
        if variant and variant != base:
            return f"{base}-{variant}"
    return base


def _migrate_index_dir_names(index: SessionIndex, output_root: Path) -> None:
    for entry in index.entries.values():
        leaf = Path(entry.relative_dir).name
        if re.fullmatch(r"세션\d+\([^)]+\)", leaf):
            continue

        source = _source_from_fingerprint(entry.fingerprint)
        try:
            year_text, month_text, day_text = entry.date_key.split("-")
            year = int(year_text)
            month = int(month_text)
            day = int(day_text)
        except (ValueError, TypeError):
            continue

        desired_dir = f"{year:04d}년/{month:02d}월/{day}일/세션{entry.session_no}({_sanitize_source_label(source)})"
        if entry.relative_dir == desired_dir:
            continue

        _relocate_session_dir(output_root, entry.relative_dir, desired_dir)

        entry.relative_dir = desired_dir


def _relocate_session_dir(output_root: Path, old_relative_dir: str, new_relative_dir: str) -> None:
    old_dir = output_root / old_relative_dir
    new_dir = output_root / new_relative_dir
    if not old_dir.exists() or old_dir == new_dir:
        return
    new_dir.parent.mkdir(parents=True, exist_ok=True)
    if not new_dir.exists():
        old_dir.rename(new_dir)
    else:
        _merge_session_dir(old_dir, new_dir)
        _prune_empty_dir(old_dir)


def _merge_session_dir(old_dir: Path, new_dir: Path) -> None:
    for item in old_dir.iterdir():
        target = new_dir / item.name
        if target.exists():
            continue
        shutil.move(str(item), str(target))


def _prune_empty_dir(path: Path) -> None:
    current = path
    while True:
        try:
            current.rmdir()
        except OSError:
            break
        parent = current.parent
        if parent == current:
            break
        if parent.name.endswith("일") or parent.name.endswith("월") or parent.name.endswith("년"):
            current = parent
            continue
        break
