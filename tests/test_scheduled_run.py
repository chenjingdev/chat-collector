from __future__ import annotations

import io
import json
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from llm_chat_archive import cli
from llm_chat_archive.incremental import write_incremental_collection
from llm_chat_archive.models import (
    CollectionPlan,
    CollectionResult,
    MessageRole,
    NormalizedConversation,
    NormalizedMessage,
    SourceDescriptor,
    SupportLevel,
)
from llm_chat_archive.registry import CollectorRegistry
from llm_chat_archive.reporting import load_run_summary
from llm_chat_archive.runner import MANIFEST_FILENAME, RUNS_DIRECTORY
from llm_chat_archive.scheduled import LOCK_FILENAME


def read_jsonl(path: Path) -> list[dict[str, object]]:
    content = path.read_text(encoding="utf-8")
    if not content.strip():
        return []
    return [json.loads(line) for line in content.splitlines()]


def run_cli(
    *args: str,
    registry: CollectorRegistry,
) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with (
        patch("llm_chat_archive.cli.build_registry", return_value=registry),
        redirect_stdout(stdout),
        redirect_stderr(stderr),
    ):
        exit_code = cli.main(["scheduled", "run", *args])
    return exit_code, stdout.getvalue(), stderr.getvalue()


def run_general_cli(*args: str) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        exit_code = cli.main(args)
    return exit_code, stdout.getvalue(), stderr.getvalue()


def write_config(
    path: Path,
    *,
    archive_root: Path,
    scheduled_mode: str = "collect",
    scheduled_sources: tuple[str, ...] = (),
    scheduled_profile: str = "all",
    scheduled_incremental: bool = True,
    scheduled_redaction: str = "on",
    scheduled_validation: str = "report",
    scheduled_stale_after_seconds: int = 60,
    scheduled_rerun_selection_preset: str = "failed_and_degraded",
    collect_profile: str = "all",
    collect_redaction: str = "off",
    collect_validation: str = "off",
) -> Path:
    scheduled_sources_payload = ", ".join(f'"{source}"' for source in scheduled_sources)
    path.write_text(
        "\n".join(
            (
                "[collect]",
                f'archive_root = "{archive_root}"',
                "incremental = true",
                f'redaction = "{collect_redaction}"',
                f'validation = "{collect_validation}"',
                "",
                "[collect.selection]",
                f'profile = "{collect_profile}"',
                "",
                "[rerun]",
                'selection_preset = "failed_and_degraded"',
                "",
                "[scheduled]",
                f'mode = "{scheduled_mode}"',
                f"incremental = {str(scheduled_incremental).lower()}",
                f'redaction = "{scheduled_redaction}"',
                f'validation = "{scheduled_validation}"',
                f"stale_after_seconds = {scheduled_stale_after_seconds}",
                "",
                "[scheduled.selection]",
                f'profile = "{scheduled_profile}"',
                f"sources = [{scheduled_sources_payload}]",
                "",
                "[scheduled.rerun]",
                f'selection_preset = "{scheduled_rerun_selection_preset}"',
                "",
            )
        ),
        encoding="utf-8",
    )
    return path


@dataclass(slots=True)
class StubExecutableCollector:
    descriptor: SourceDescriptor
    collected_at_values: list[str]
    secret_text: str = "Authorization: Bearer real-token-1234567890"
    _index: int = field(default=0, init=False, repr=False)

    def build_plan(self, archive_root: Path) -> CollectionPlan:
        return CollectionPlan(
            source=self.descriptor.key,
            display_name=self.descriptor.display_name,
            archive_root=archive_root,
            execution_context=self.descriptor.execution_context,
            support_level=self.descriptor.support_level,
            default_input_roots=self.descriptor.default_input_roots,
            implemented=True,
        )

    def collect(
        self,
        archive_root: Path,
        input_roots: tuple[Path, ...] | None = None,
    ) -> CollectionResult:
        collected_at = self.collected_at_values[self._index]
        self._index += 1
        conversation = NormalizedConversation(
            source=self.descriptor.key,
            execution_context=self.descriptor.execution_context,
            collected_at=collected_at,
            messages=(
                NormalizedMessage(
                    role=MessageRole.USER,
                    text=self.secret_text,
                ),
            ),
            source_session_id=f"{self.descriptor.key}-session-1",
            source_artifact_path="/tmp/source-artifact.jsonl",
        )
        return write_incremental_collection(
            source=self.descriptor.key,
            archive_root=archive_root,
            input_roots=tuple(input_roots or ()),
            scanned_artifact_count=1,
            collected_at=collected_at,
            conversations=(conversation,),
        )


def make_registry(*collectors: StubExecutableCollector) -> CollectorRegistry:
    registry = CollectorRegistry()
    for collector in collectors:
        registry.register(collector)
    return registry


def make_descriptor(
    key: str,
    support_level: SupportLevel = SupportLevel.COMPLETE,
) -> SourceDescriptor:
    return SourceDescriptor(
        key=key,
        display_name=key.replace("_", " ").title(),
        execution_context="test",
        support_level=support_level,
        default_input_roots=(),
    )


def write_lock(
    archive_root: Path,
    *,
    acquired_at: str,
    owner_pid: int = 1234,
    owner_hostname: str = "scheduler-host",
    mode: str = "collect",
) -> Path:
    lock_path = archive_root / RUNS_DIRECTORY / LOCK_FILENAME
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(
        json.dumps(
            {
                "acquired_at": acquired_at,
                "owner_pid": owner_pid,
                "owner_hostname": owner_hostname,
                "mode": mode,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return lock_path


def write_origin_manifest(
    archive_root: Path,
    *,
    run_id: str = "20260319T080000Z",
) -> None:
    sources = (
        {
            "source": "complete_source",
            "support_level": "complete",
            "status": "complete",
            "archive_root": str(archive_root),
            "output_path": str(archive_root / "complete_source" / "memory_chat_v1-complete_source.jsonl"),
            "input_roots": [],
            "scanned_artifact_count": 1,
            "conversation_count": 1,
            "skipped_conversation_count": 0,
            "written_conversation_count": 1,
            "message_count": 1,
            "partial": False,
            "unsupported": False,
            "failed": False,
        },
        {
            "source": "failed_source",
            "support_level": "complete",
            "status": "failed",
            "archive_root": str(archive_root),
            "output_path": None,
            "input_roots": [],
            "scanned_artifact_count": 0,
            "conversation_count": 0,
            "skipped_conversation_count": 0,
            "written_conversation_count": 0,
            "message_count": 0,
            "partial": False,
            "unsupported": False,
            "failed": True,
        },
        {
            "source": "partial_source",
            "support_level": "partial",
            "status": "partial",
            "archive_root": str(archive_root),
            "output_path": str(archive_root / "partial_source" / "memory_chat_v1-partial_source.jsonl"),
            "input_roots": [],
            "scanned_artifact_count": 1,
            "conversation_count": 1,
            "skipped_conversation_count": 0,
            "written_conversation_count": 1,
            "message_count": 1,
            "partial": True,
            "unsupported": False,
            "failed": False,
        },
    )
    for source in ("complete_source", "partial_source"):
        output_dir = archive_root / source
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"memory_chat_v1-{source}.jsonl"
        output_path.write_text(
            json.dumps({"source": source, "messages": []}) + "\n",
            encoding="utf-8",
        )
    run_dir = archive_root / RUNS_DIRECTORY / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / MANIFEST_FILENAME
    manifest_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "archive_root": str(archive_root),
                "run_dir": str(run_dir),
                "manifest_path": str(manifest_path),
                "selection_policy": {
                    "profile": "all",
                    "minimum_support_level": "scaffold",
                    "include_sources": [],
                    "exclude_sources": [],
                },
                "effective_config": {
                    "archive_root": str(archive_root),
                    "selection_policy": {
                        "profile": "all",
                        "minimum_support_level": "scaffold",
                        "include_sources": [],
                        "exclude_sources": [],
                    },
                    "execution_policy": {
                        "incremental": True,
                        "redaction": "on",
                        "validation": "off",
                    },
                    "config_source": "defaults",
                },
                "selected_sources": [source["source"] for source in sources],
                "excluded_sources": [],
                "source_count": len(sources),
                "failed_source_count": 1,
                "scanned_artifact_count": 2,
                "conversation_count": 2,
                "skipped_conversation_count": 0,
                "written_conversation_count": 2,
                "message_count": 2,
                "sources": list(sources),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def test_scheduled_run_uses_scheduled_presets_and_marks_run_metadata(
    tmp_path: Path,
) -> None:
    archive_root = tmp_path / "archive"
    config_path = write_config(
        tmp_path / "collector.toml",
        archive_root=archive_root,
        scheduled_sources=("alpha_source",),
        scheduled_redaction="on",
        scheduled_validation="report",
        collect_redaction="off",
        collect_validation="off",
    )
    registry = make_registry(
        StubExecutableCollector(
            descriptor=make_descriptor("alpha_source"),
            collected_at_values=["2026-03-20T00:00:00Z"],
        ),
        StubExecutableCollector(
            descriptor=make_descriptor("beta_source"),
            collected_at_values=["2026-03-20T00:00:00Z"],
        ),
    )

    exit_code, stdout, stderr = run_cli(
        "--config",
        str(config_path),
        registry=registry,
    )

    assert exit_code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["selected_sources"] == ["alpha_source"]
    assert payload["effective_config"]["execution_policy"] == {
        "incremental": True,
        "redaction": "on",
        "validation": "report",
    }
    assert payload["validation"]["mode"] == "report"
    assert payload["scheduled"]["mode"] == "collect"
    assert payload["scheduled"]["stale_after_seconds"] == 60
    assert payload["scheduled"]["force_unlocked_stale_lock"] is False
    assert payload["scheduled"]["lock"]["path"] == str(
        archive_root / RUNS_DIRECTORY / LOCK_FILENAME
    )

    written_rows = read_jsonl(Path(payload["sources"][0]["output_path"]))
    assert written_rows[0]["messages"][0]["text"] == "Authorization: Bearer [REDACTED]"

    summary = load_run_summary(
        archive_root,
        payload["run_id"],
        verify_output_paths=False,
    )
    assert summary.to_dict()["scheduled"]["mode"] == "collect"

    latest_exit_code, latest_stdout, latest_stderr = run_general_cli(
        "runs",
        "latest",
        "--archive-root",
        str(archive_root),
    )
    assert latest_exit_code == 0
    assert latest_stderr == ""
    latest_payload = json.loads(latest_stdout)
    assert latest_payload["scheduled"]["mode"] == "collect"


def test_scheduled_run_reports_active_lock_conflict(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    config_path = write_config(
        tmp_path / "collector.toml",
        archive_root=archive_root,
        scheduled_sources=("alpha_source",),
    )
    write_lock(
        archive_root,
        acquired_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
            "+00:00",
            "Z",
        ),
    )
    registry = make_registry(
        StubExecutableCollector(
            descriptor=make_descriptor("alpha_source"),
            collected_at_values=["2026-03-20T00:00:00Z"],
        )
    )

    exit_code, stdout, stderr = run_cli(
        "--config",
        str(config_path),
        registry=registry,
    )

    assert exit_code == 1
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["reason"] == "scheduled_lock_held"
    assert payload["lock"]["stale"] is False
    assert payload["force_unlock_stale_available"] is False


def test_scheduled_run_can_force_unlock_stale_lock(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    config_path = write_config(
        tmp_path / "collector.toml",
        archive_root=archive_root,
        scheduled_sources=("alpha_source",),
        scheduled_stale_after_seconds=60,
    )
    stale_timestamp = (
        datetime.now(timezone.utc) - timedelta(hours=2)
    ).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    write_lock(archive_root, acquired_at=stale_timestamp)
    registry = make_registry(
        StubExecutableCollector(
            descriptor=make_descriptor("alpha_source"),
            collected_at_values=["2026-03-20T00:00:00Z"],
        )
    )

    stale_exit_code, stale_stdout, stale_stderr = run_cli(
        "--config",
        str(config_path),
        registry=registry,
    )
    assert stale_exit_code == 1
    assert stale_stderr == ""
    stale_payload = json.loads(stale_stdout)
    assert stale_payload["reason"] == "scheduled_lock_stale"
    assert stale_payload["lock"]["stale"] is True
    assert stale_payload["force_unlock_stale_available"] is True

    forced_exit_code, forced_stdout, forced_stderr = run_cli(
        "--config",
        str(config_path),
        "--force-unlock-stale",
        registry=registry,
    )
    assert forced_exit_code == 0
    assert forced_stderr == ""
    forced_payload = json.loads(forced_stdout)
    assert forced_payload["scheduled"]["force_unlocked_stale_lock"] is True
    assert forced_payload["scheduled"]["replaced_lock"]["stale"] is True


def test_scheduled_run_reruns_latest_run_with_configured_preset(
    tmp_path: Path,
) -> None:
    write_origin_manifest(tmp_path)
    config_path = write_config(
        tmp_path / "collector.toml",
        archive_root=tmp_path,
        scheduled_mode="rerun",
        scheduled_validation="off",
        scheduled_rerun_selection_preset="failed_only",
    )
    registry = make_registry(
        StubExecutableCollector(
            descriptor=make_descriptor("failed_source"),
            collected_at_values=["2026-03-20T01:00:00Z"],
            secret_text="rerun result",
        )
    )

    exit_code, stdout, stderr = run_cli(
        "--config",
        str(config_path),
        registry=registry,
    )

    assert exit_code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["selected_sources"] == ["failed_source"]
    assert payload["rerun"]["origin_run_id"] == "20260319T080000Z"
    assert payload["rerun"]["selection_reason"] == "failed"
    assert payload["scheduled"]["mode"] == "rerun"
    assert payload["scheduled"]["origin_run_id"] == "20260319T080000Z"
