from __future__ import annotations

import io
import json
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

from llm_chat_archive import cli
from llm_chat_archive.models import (
    CollectionPlan,
    CollectionResult,
    SourceDescriptor,
    SupportLevel,
)
from llm_chat_archive.registry import CollectorRegistry
from llm_chat_archive.reporting import load_run_summary
from llm_chat_archive.runner import MANIFEST_FILENAME, RUNS_DIRECTORY


@dataclass(frozen=True, slots=True)
class StubExecutableCollector:
    descriptor: SourceDescriptor
    rows: tuple[dict[str, object], ...]

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
        self, archive_root: Path, input_roots: tuple[Path, ...] | None = None
    ) -> CollectionResult:
        output_dir = archive_root / self.descriptor.key
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"memory_chat_v1-{self.descriptor.key}.jsonl"
        with output_path.open("w", encoding="utf-8") as handle:
            for row in self.rows:
                handle.write(json.dumps(row))
                handle.write("\n")

        return CollectionResult(
            source=self.descriptor.key,
            archive_root=archive_root,
            output_path=output_path,
            input_roots=tuple(input_roots or ()),
            scanned_artifact_count=1,
            conversation_count=len(self.rows),
            message_count=sum(_message_count(row) for row in self.rows),
            written_conversation_count=len(self.rows),
        )


@dataclass(frozen=True, slots=True)
class StubPlanOnlyCollector:
    descriptor: SourceDescriptor

    def build_plan(self, archive_root: Path) -> CollectionPlan:
        return CollectionPlan(
            source=self.descriptor.key,
            display_name=self.descriptor.display_name,
            archive_root=archive_root,
            execution_context=self.descriptor.execution_context,
            support_level=self.descriptor.support_level,
            default_input_roots=self.descriptor.default_input_roots,
            implemented=False,
        )


def _message_count(row: dict[str, object]) -> int:
    messages = row.get("messages")
    if isinstance(messages, list):
        return len(messages)
    return 0


def _descriptor(key: str, support_level: SupportLevel) -> SourceDescriptor:
    return SourceDescriptor(
        key=key,
        display_name=key.replace("_", " ").title(),
        execution_context="test",
        support_level=support_level,
        default_input_roots=(),
    )


def _registry() -> CollectorRegistry:
    registry = CollectorRegistry()
    registry.register(
        StubExecutableCollector(
            descriptor=_descriptor("complete_source", SupportLevel.COMPLETE),
            rows=(
                {
                    "source": "complete_source",
                    "messages": [{"role": "assistant", "text": "complete"}],
                },
            ),
        )
    )
    registry.register(
        StubExecutableCollector(
            descriptor=_descriptor("failed_source", SupportLevel.COMPLETE),
            rows=(
                {
                    "source": "failed_source",
                    "messages": [{"role": "assistant", "text": "recovered"}],
                },
            ),
        )
    )
    registry.register(
        StubExecutableCollector(
            descriptor=_descriptor("partial_source", SupportLevel.PARTIAL),
            rows=(
                {
                    "source": "partial_source",
                    "messages": [{"role": "assistant", "text": "partial"}],
                    "transcript_completeness": "partial",
                },
            ),
        )
    )
    registry.register(
        StubPlanOnlyCollector(
            descriptor=_descriptor("unsupported_source", SupportLevel.SCAFFOLD)
        )
    )
    return registry


def _make_origin_source_entry(
    archive_root: Path,
    *,
    source: str,
    support_level: str,
    status: str,
    partial: bool = False,
    unsupported: bool = False,
    failed: bool = False,
    create_output: bool = True,
) -> dict[str, object]:
    output_path: Path | None = None
    if create_output:
        output_dir = archive_root / source
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"memory_chat_v1-{source}.jsonl"
        output_path.write_text(
            json.dumps({"source": source, "messages": []}) + "\n",
            encoding="utf-8",
        )

    return {
        "source": source,
        "support_level": support_level,
        "status": status,
        "archive_root": str(archive_root),
        "output_path": str(output_path) if output_path is not None else None,
        "input_roots": [],
        "scanned_artifact_count": 1 if create_output else 0,
        "conversation_count": 1 if create_output else 0,
        "skipped_conversation_count": 0,
        "written_conversation_count": 1 if create_output else 0,
        "message_count": 1 if create_output else 0,
        "partial": partial,
        "unsupported": unsupported,
        "failed": failed,
    }


def _write_origin_manifest(
    archive_root: Path,
    *,
    run_id: str = "20260319T080000Z",
) -> None:
    sources = (
        _make_origin_source_entry(
            archive_root,
            source="complete_source",
            support_level="complete",
            status="complete",
        ),
        _make_origin_source_entry(
            archive_root,
            source="failed_source",
            support_level="complete",
            status="failed",
            failed=True,
            create_output=False,
        ),
        _make_origin_source_entry(
            archive_root,
            source="partial_source",
            support_level="partial",
            status="partial",
            partial=True,
        ),
        _make_origin_source_entry(
            archive_root,
            source="unsupported_source",
            support_level="scaffold",
            status="unsupported",
            unsupported=True,
            create_output=False,
        ),
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


def write_rerun_config(
    path: Path,
    *,
    archive_root: Path,
    selection_preset: str,
) -> Path:
    path.write_text(
        "\n".join(
            (
                "[collect]",
                f'archive_root = "{archive_root}"',
                "",
                "[rerun]",
                f'selection_preset = "{selection_preset}"',
                "",
            )
        ),
        encoding="utf-8",
    )
    return path


def run_rerun_cli(*args: str) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with (
        patch("llm_chat_archive.cli.build_registry", return_value=_registry()),
        redirect_stdout(stdout),
        redirect_stderr(stderr),
    ):
        exit_code = cli.main(["rerun", *args])
    return exit_code, stdout.getvalue(), stderr.getvalue()


def test_rerun_uses_configured_selection_preset_when_reason_is_omitted(
    tmp_path: Path,
) -> None:
    _write_origin_manifest(tmp_path)
    config_path = write_rerun_config(
        tmp_path / "collector.toml",
        archive_root=tmp_path,
        selection_preset="failed_only",
    )

    exit_code, stdout, stderr = run_rerun_cli(
        "--run",
        "20260319T080000Z",
        "--config",
        str(config_path),
    )

    assert exit_code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["selected_sources"] == ["failed_source"]
    assert payload["rerun"]["selection_reason"] == "failed"
    assert payload["effective_config"]["rerun"] == {
        "selection_preset": "failed_only",
        "selection_reason": "failed",
        "source": "config",
    }

    manifest = json.loads(
        (
            tmp_path / RUNS_DIRECTORY / payload["run_id"] / MANIFEST_FILENAME
        ).read_text(encoding="utf-8")
    )
    assert manifest["effective_config"]["rerun"] == payload["effective_config"]["rerun"]

    summary = load_run_summary(
        tmp_path,
        payload["run_id"],
        verify_output_paths=False,
    )
    assert (
        summary.to_dict()["effective_config"]["rerun"]
        == payload["effective_config"]["rerun"]
    )


def test_rerun_reason_flag_overrides_configured_selection_preset(tmp_path: Path) -> None:
    _write_origin_manifest(tmp_path)
    config_path = write_rerun_config(
        tmp_path / "collector.toml",
        archive_root=tmp_path,
        selection_preset="failed_only",
    )

    exit_code, stdout, stderr = run_rerun_cli(
        "--run",
        "20260319T080000Z",
        "--config",
        str(config_path),
        "--reason",
        "degraded",
    )

    assert exit_code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["selected_sources"] == ["partial_source", "unsupported_source"]
    assert payload["rerun"]["selection_reason"] == "degraded"
    assert payload["effective_config"]["rerun"] == {
        "selection_preset": "degraded_only",
        "selection_reason": "degraded",
        "source": "cli",
    }

