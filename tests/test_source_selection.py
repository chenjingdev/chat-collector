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
from llm_chat_archive.runner import MANIFEST_FILENAME, RUNS_DIRECTORY, run_collection_batch
from llm_chat_archive.source_selection import (
    build_source_selection_policy,
    select_collectors,
)


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


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
        output_path = output_dir / "memory_chat_v1-stub.jsonl"
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
                    "messages": [{"role": "assistant", "text": "done"}],
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
                    "messages": [],
                    "transcript_completeness": "partial",
                },
            ),
        )
    )
    registry.register(
        StubPlanOnlyCollector(
            descriptor=_descriptor("scaffold_source", SupportLevel.SCAFFOLD)
        )
    )
    return registry


def test_select_collectors_applies_allowlist_denylist_and_profile() -> None:
    selection = select_collectors(
        _registry(),
        policy=build_source_selection_policy(
            profile="all",
            include_sources=("complete_source", "partial_source"),
            exclude_sources=("partial_source",),
        ),
    )

    assert selection.selected_sources == ("complete_source",)
    assert [entry.to_dict() for entry in selection.excluded_sources] == [
        {
            "source": "partial_source",
            "support_level": "partial",
            "reason": "explicitly excluded by --exclude-source",
        },
        {
            "source": "scaffold_source",
            "support_level": "scaffold",
            "reason": "not included by --source allowlist",
        },
    ]


def test_run_collection_batch_records_effective_selection_policy(tmp_path: Path) -> None:
    result = run_collection_batch(
        _registry(),
        tmp_path,
        selection_policy=build_source_selection_policy(profile="complete_only"),
    )

    assert result.manifest_path == tmp_path / RUNS_DIRECTORY / result.run_id / MANIFEST_FILENAME
    assert result.selected_sources == ("complete_source",)
    manifest = read_json(result.manifest_path)
    assert manifest["selection_policy"] == {
        "profile": "complete_only",
        "minimum_support_level": "complete",
        "include_sources": [],
        "exclude_sources": [],
    }
    assert manifest["selected_sources"] == ["complete_source"]
    assert manifest["excluded_sources"] == [
        {
            "source": "partial_source",
            "support_level": "partial",
            "reason": "support level 'partial' is below minimum 'complete'",
        },
        {
            "source": "scaffold_source",
            "support_level": "scaffold",
            "reason": "support level 'scaffold' is below minimum 'complete'",
        },
    ]
    assert [entry["source"] for entry in manifest["sources"]] == ["complete_source"]

    summary = load_run_summary(tmp_path, result.run_id)
    assert summary.to_dict()["selection_policy"] == manifest["selection_policy"]
    assert summary.to_dict()["excluded_sources"] == manifest["excluded_sources"]


def test_default_profile_excludes_partial_sources(tmp_path: Path) -> None:
    result = run_collection_batch(
        _registry(),
        tmp_path,
        selection_policy=build_source_selection_policy(profile="default"),
    )

    assert result.selected_sources == ("complete_source",)
    manifest = read_json(result.manifest_path)
    assert manifest["selection_policy"] == {
        "profile": "default",
        "minimum_support_level": "complete",
        "include_sources": [],
        "exclude_sources": [],
    }
    assert manifest["excluded_sources"] == [
        {
            "source": "partial_source",
            "support_level": "partial",
            "reason": "support level 'partial' is below minimum 'complete'",
        },
        {
            "source": "scaffold_source",
            "support_level": "scaffold",
            "reason": "support level 'scaffold' is below minimum 'complete'",
        },
    ]


def test_cli_collect_all_profile_complete_only_excludes_partial_sources(
    tmp_path: Path,
) -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with (
        patch("llm_chat_archive.cli.build_registry", return_value=_registry()),
        redirect_stdout(stdout),
        redirect_stderr(stderr),
    ):
        exit_code = cli.main(
            [
                "collect",
                "--all",
                "--profile",
                "complete_only",
                "--archive-root",
                str(tmp_path),
            ]
        )

    assert exit_code == 0
    assert stderr.getvalue() == ""
    payload = json.loads(stdout.getvalue())
    assert payload["selection_policy"]["profile"] == "complete_only"
    assert payload["selected_sources"] == ["complete_source"]
    assert [entry["source"] for entry in payload["sources"]] == ["complete_source"]
    assert payload["excluded_sources"][0]["source"] == "partial_source"


def test_cli_single_source_rejects_batch_selection_options(tmp_path: Path) -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with (
        patch("llm_chat_archive.cli.build_registry", return_value=_registry()),
        redirect_stdout(stdout),
        redirect_stderr(stderr),
    ):
        exit_code = cli.main(
            [
                "collect",
                "complete_source",
                "--profile",
                "complete_only",
                "--archive-root",
                str(tmp_path),
            ]
        )

    assert exit_code == 2
    assert stdout.getvalue() == ""
    assert "require --all" in stderr.getvalue()
