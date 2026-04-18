from __future__ import annotations

import io
import json
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

from llm_chat_archive import cli
from llm_chat_archive.models import CollectionPlan, CollectionResult, SourceDescriptor, SupportLevel
from llm_chat_archive.registry import CollectorRegistry
from llm_chat_archive.runner import (
    MANIFEST_FILENAME,
    RUNS_DIRECTORY,
    run_collection_batch,
    summarize_output_status,
)


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True, slots=True)
class StubExecutableCollector:
    descriptor: SourceDescriptor
    rows: tuple[dict[str, object], ...]
    scanned_artifact_count: int = 1

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
        output_dir = archive_root / self.descriptor.key
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "memory_chat_v1-stub.jsonl"
        with output_path.open("w", encoding="utf-8") as handle:
            for row in self.rows:
                handle.write(json.dumps(row, ensure_ascii=False))
                handle.write("\n")

        return CollectionResult(
            source=self.descriptor.key,
            archive_root=archive_root,
            output_path=output_path,
            input_roots=tuple(input_roots or ()),
            scanned_artifact_count=self.scanned_artifact_count,
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


@dataclass(frozen=True, slots=True)
class StubFailingCollector:
    descriptor: SourceDescriptor
    message: str

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
        raise RuntimeError(self.message)


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


def test_run_collection_batch_writes_manifest_and_keeps_failures_isolated(
    tmp_path: Path,
) -> None:
    registry = CollectorRegistry()
    registry.register(
        StubExecutableCollector(
            descriptor=_descriptor("complete_source", SupportLevel.COMPLETE),
            rows=(
                {
                    "source": "complete_source",
                    "messages": [{"role": "user", "text": "hello"}],
                },
            ),
            scanned_artifact_count=2,
        )
    )
    registry.register(
        StubFailingCollector(
            descriptor=_descriptor("failing_source", SupportLevel.COMPLETE),
            message="collector exploded",
        )
    )
    registry.register(
        StubExecutableCollector(
            descriptor=_descriptor("partial_source", SupportLevel.PARTIAL),
            rows=(
                {
                    "source": "partial_source",
                    "messages": [],
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

    result = run_collection_batch(registry, tmp_path)

    assert result.manifest_path == tmp_path / RUNS_DIRECTORY / result.run_id / MANIFEST_FILENAME
    manifest = read_json(result.manifest_path)
    assert manifest["manifest_path"] == str(result.manifest_path)
    assert manifest["failed_source_count"] == 1
    assert manifest["conversation_count"] == 2
    assert manifest["skipped_conversation_count"] == 0
    assert manifest["written_conversation_count"] == 2
    assert manifest["upgraded_conversation_count"] == 0
    assert manifest["redaction_event_count"] == 0
    entries = {entry["source"]: entry for entry in manifest["sources"]}

    assert entries["complete_source"] == {
        "source": "complete_source",
        "support_level": "complete",
        "status": "complete",
        "archive_root": str(tmp_path),
        "output_path": str(tmp_path / "complete_source" / "memory_chat_v1-stub.jsonl"),
        "input_roots": [],
        "scanned_artifact_count": 2,
        "conversation_count": 1,
        "skipped_conversation_count": 0,
        "written_conversation_count": 1,
        "upgraded_conversation_count": 0,
        "message_count": 1,
        "redaction_event_count": 0,
        "partial": False,
        "unsupported": False,
        "failed": False,
    }
    assert entries["partial_source"] == {
        "source": "partial_source",
        "support_level": "partial",
        "status": "partial",
        "archive_root": str(tmp_path),
        "output_path": str(tmp_path / "partial_source" / "memory_chat_v1-stub.jsonl"),
        "input_roots": [],
        "scanned_artifact_count": 1,
        "conversation_count": 1,
        "skipped_conversation_count": 0,
        "written_conversation_count": 1,
        "upgraded_conversation_count": 0,
        "message_count": 0,
        "redaction_event_count": 0,
        "partial": True,
        "unsupported": False,
        "failed": False,
    }
    assert entries["unsupported_source"] == {
        "source": "unsupported_source",
        "support_level": "scaffold",
        "status": "unsupported",
        "archive_root": str(tmp_path),
        "output_path": None,
        "input_roots": [],
        "scanned_artifact_count": 0,
        "conversation_count": 0,
        "skipped_conversation_count": 0,
        "written_conversation_count": 0,
        "upgraded_conversation_count": 0,
        "message_count": 0,
        "redaction_event_count": 0,
        "partial": False,
        "unsupported": True,
        "failed": False,
    }
    assert entries["failing_source"] == {
        "source": "failing_source",
        "support_level": "complete",
        "status": "failed",
        "archive_root": str(tmp_path),
        "output_path": None,
        "input_roots": [],
        "scanned_artifact_count": 0,
        "conversation_count": 0,
        "skipped_conversation_count": 0,
        "written_conversation_count": 0,
        "upgraded_conversation_count": 0,
        "message_count": 0,
        "redaction_event_count": 0,
        "partial": False,
        "unsupported": False,
        "failed": True,
        "failure_reason": "RuntimeError: collector exploded",
    }


def test_cli_collect_all_executes_batch_without_execute_flag(tmp_path: Path) -> None:
    registry = CollectorRegistry()
    registry.register(
        StubExecutableCollector(
            descriptor=_descriptor("alpha_source", SupportLevel.COMPLETE),
            rows=(
                {
                    "source": "alpha_source",
                    "messages": [{"role": "assistant", "text": "done"}],
                },
            ),
        )
    )

    stdout = io.StringIO()
    stderr = io.StringIO()
    with (
        patch("llm_chat_archive.cli.build_registry", return_value=registry),
        redirect_stdout(stdout),
        redirect_stderr(stderr),
    ):
        exit_code = cli.main(["collect", "--all", "--archive-root", str(tmp_path)])

    assert exit_code == 0
    assert stderr.getvalue() == ""
    payload = json.loads(stdout.getvalue())
    assert payload["source_count"] == 1
    assert payload["failed_source_count"] == 0
    assert payload["skipped_conversation_count"] == 0
    assert payload["written_conversation_count"] == 1
    assert payload["upgraded_conversation_count"] == 0
    assert payload["manifest_path"] == str(
        tmp_path / RUNS_DIRECTORY / payload["run_id"] / MANIFEST_FILENAME
    )
    assert payload["sources"] == [
        {
            "source": "alpha_source",
            "support_level": "complete",
            "status": "complete",
            "archive_root": str(tmp_path),
            "output_path": str(tmp_path / "alpha_source" / "memory_chat_v1-stub.jsonl"),
            "input_roots": [],
            "scanned_artifact_count": 1,
            "conversation_count": 1,
            "skipped_conversation_count": 0,
            "written_conversation_count": 1,
            "upgraded_conversation_count": 0,
            "message_count": 1,
            "redaction_event_count": 0,
            "partial": False,
            "unsupported": False,
            "failed": False,
        }
    ]


def test_summarize_output_status_keeps_mixed_partial_and_unsupported_rows(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "mixed.jsonl"
    output_path.write_text(
        "\n".join(
            (
                json.dumps({"source": "mixed", "transcript_completeness": "partial"}),
                json.dumps({"source": "mixed", "transcript_completeness": "unsupported"}),
            )
        ),
        encoding="utf-8",
    )

    partial, unsupported = summarize_output_status(
        output_path,
        support_level=SupportLevel.PARTIAL,
    )

    assert partial is True
    assert unsupported is True
