from __future__ import annotations

import json
import os
import socket
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from .models import ScheduledLockRecord, ScheduledRunMode
from .runner import RUNS_DIRECTORY
from .sources.codex_rollout import utc_timestamp

LOCK_FILENAME = ".scheduled.lock"


class ScheduledLockError(RuntimeError):
    def __init__(
        self,
        *,
        status: str,
        message: str,
        lock: ScheduledLockRecord,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.lock = lock


@dataclass(frozen=True, slots=True)
class ScheduledLockAcquisition:
    lock: ScheduledLockRecord
    force_unlocked_stale_lock: bool = False
    replaced_lock: ScheduledLockRecord | None = None


@contextmanager
def acquire_scheduled_lock(
    archive_root: Path,
    *,
    mode: ScheduledRunMode,
    stale_after_seconds: int,
    force_unlock_stale: bool,
) -> Iterator[ScheduledLockAcquisition]:
    lock_path = archive_root / RUNS_DIRECTORY / LOCK_FILENAME
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    force_unlocked_stale_lock = False
    replaced_lock: ScheduledLockRecord | None = None
    acquisition = _try_acquire_lock(lock_path, mode=mode)
    if acquisition is None:
        existing_lock = _read_lock_record(lock_path)
        evaluated_lock = _evaluate_existing_lock(
            existing_lock,
            lock_path=lock_path,
            stale_after_seconds=stale_after_seconds,
        )
        if evaluated_lock.stale and force_unlock_stale:
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass
            replaced_lock = evaluated_lock
            force_unlocked_stale_lock = True
            acquisition = _try_acquire_lock(lock_path, mode=mode)
        if acquisition is None:
            raise ScheduledLockError(
                status="stale" if evaluated_lock.stale else "held",
                message=_lock_error_message(
                    evaluated_lock,
                    stale_after_seconds=stale_after_seconds,
                ),
                lock=evaluated_lock,
            )

    try:
        yield ScheduledLockAcquisition(
            lock=acquisition,
            force_unlocked_stale_lock=force_unlocked_stale_lock,
            replaced_lock=replaced_lock,
        )
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _try_acquire_lock(
    lock_path: Path,
    *,
    mode: ScheduledRunMode,
) -> ScheduledLockRecord | None:
    acquired_at = utc_timestamp()
    payload = {
        "acquired_at": acquired_at,
        "owner_pid": os.getpid(),
        "owner_hostname": socket.gethostname(),
        "mode": mode.value,
    }
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        file_descriptor = os.open(lock_path, flags, 0o644)
    except FileExistsError:
        return None

    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
    except Exception:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
        raise

    return ScheduledLockRecord(
        path=lock_path,
        acquired_at=acquired_at,
        owner_pid=payload["owner_pid"],
        owner_hostname=payload["owner_hostname"],
        mode=payload["mode"],
        stale=False,
    )


def _read_lock_record(lock_path: Path) -> ScheduledLockRecord:
    acquired_at: str | None = None
    owner_pid: int | None = None
    owner_hostname: str | None = None
    mode: str | None = None
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        payload = {}
    if isinstance(payload, dict):
        value = payload.get("acquired_at")
        if isinstance(value, str):
            acquired_at = value
        value = payload.get("owner_pid")
        if isinstance(value, int) and not isinstance(value, bool):
            owner_pid = value
        value = payload.get("owner_hostname")
        if isinstance(value, str):
            owner_hostname = value
        value = payload.get("mode")
        if isinstance(value, str):
            mode = value
    return ScheduledLockRecord(
        path=lock_path,
        acquired_at=acquired_at,
        owner_pid=owner_pid,
        owner_hostname=owner_hostname,
        mode=mode,
    )


def _evaluate_existing_lock(
    lock: ScheduledLockRecord,
    *,
    lock_path: Path,
    stale_after_seconds: int,
) -> ScheduledLockRecord:
    reference_time = _parse_timestamp(lock.acquired_at)
    if reference_time is None:
        try:
            reference_time = datetime.fromtimestamp(
                lock_path.stat().st_mtime,
                tz=timezone.utc,
            )
        except FileNotFoundError:
            reference_time = datetime.now(timezone.utc)
    age_seconds = max(
        0,
        int((datetime.now(timezone.utc) - reference_time).total_seconds()),
    )
    return ScheduledLockRecord(
        path=lock.path,
        acquired_at=lock.acquired_at,
        owner_pid=lock.owner_pid,
        owner_hostname=lock.owner_hostname,
        mode=lock.mode,
        age_seconds=age_seconds,
        stale=age_seconds > stale_after_seconds,
    )


def _parse_timestamp(value: str | None) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _lock_error_message(
    lock: ScheduledLockRecord,
    *,
    stale_after_seconds: int,
) -> str:
    owner = []
    if lock.owner_pid is not None:
        owner.append(f"pid={lock.owner_pid}")
    if lock.owner_hostname is not None:
        owner.append(f"host={lock.owner_hostname}")
    owner_summary = ", ".join(owner) if owner else "unknown owner"
    age_summary = (
        f", age={lock.age_seconds}s"
        if lock.age_seconds is not None
        else ""
    )
    acquired_at_summary = (
        f", acquired_at={lock.acquired_at}"
        if lock.acquired_at is not None
        else ""
    )
    if lock.stale:
        return (
            f"scheduled lock is stale at {lock.path} "
            f"({owner_summary}{age_summary}{acquired_at_summary}); rerun with "
            f"--force-unlock-stale to replace it after the {stale_after_seconds}s limit"
        )
    return (
        f"scheduled lock is already held at {lock.path} "
        f"({owner_summary}{age_summary}{acquired_at_summary})"
    )


__all__ = [
    "LOCK_FILENAME",
    "ScheduledLockAcquisition",
    "ScheduledLockError",
    "acquire_scheduled_lock",
]
