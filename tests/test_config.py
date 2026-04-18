from __future__ import annotations

import io
import json
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import patch

from llm_chat_archive import cli
from llm_chat_archive.config import render_collect_config_template
from llm_chat_archive.incremental import write_incremental_collection
from llm_chat_archive.models import (
    CollectionPlan,
    CollectionResult,
    DEFAULT_ARCHIVE_ROOT,
    MessageRole,
    NormalizedConversation,
    NormalizedMessage,
    SourceDescriptor,
    SupportLevel,
)
from llm_chat_archive.registry import CollectorRegistry


def read_jsonl(path: Path) -> list[dict[str, object]]:
    content = path.read_text(encoding="utf-8")
    if not content.strip():
        return []
    return [json.loads(line) for line in content.splitlines()]


def run_cli(
    *args: str,
    registry: CollectorRegistry | None = None,
) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        if registry is None:
            exit_code = cli.main(args)
        else:
            with patch("llm_chat_archive.cli.build_registry", return_value=registry):
                exit_code = cli.main(args)
    return exit_code, stdout.getvalue(), stderr.getvalue()


def write_config(
    path: Path,
    *,
    archive_root: Path,
    profile: str = "all",
    sources: tuple[str, ...] = (),
    exclude_sources: tuple[str, ...] = (),
    incremental: bool = True,
    redaction: str = "on",
    validation: str = "off",
) -> Path:
    sources_payload = ", ".join(f'"{source}"' for source in sources)
    exclude_payload = ", ".join(f'"{source}"' for source in exclude_sources)
    path.write_text(
        "\n".join(
            (
                "[collect]",
                f'archive_root = "{archive_root}"',
                f"incremental = {str(incremental).lower()}",
                f'redaction = "{redaction}"',
                f'validation = "{validation}"',
                "",
                "[collect.selection]",
                f'profile = "{profile}"',
                f"sources = [{sources_payload}]",
                f"exclude_sources = [{exclude_payload}]",
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
        self, archive_root: Path, input_roots: tuple[Path, ...] | None = None
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
            source_session_id="session-1",
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


def make_descriptor(key: str, support_level: SupportLevel = SupportLevel.COMPLETE) -> SourceDescriptor:
    return SourceDescriptor(
        key=key,
        display_name=key.replace("_", " ").title(),
        execution_context="test",
        support_level=support_level,
        default_input_roots=(),
    )


def test_collect_all_reads_config_and_runs_show_reports_effective_config(
    tmp_path: Path,
) -> None:
    archive_root = tmp_path / "archive"
    config_path = write_config(
        tmp_path / "collector.toml",
        archive_root=archive_root,
        profile="all",
        sources=("alpha_source",),
        incremental=False,
        redaction="off",
        validation="report",
    )
    registry = make_registry(
        StubExecutableCollector(
            descriptor=make_descriptor("alpha_source"),
            collected_at_values=[
                "2026-03-19T00:00:00Z",
                "2026-03-19T00:05:00Z",
            ],
        ),
        StubExecutableCollector(
            descriptor=make_descriptor("beta_source"),
            collected_at_values=["2026-03-19T00:00:00Z"],
        ),
    )

    exit_code, stdout, stderr = run_cli(
        "collect",
        "--all",
        "--config",
        str(config_path),
        registry=registry,
    )

    assert exit_code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["archive_root"] == str(archive_root)
    assert payload["selected_sources"] == ["alpha_source"]
    assert payload["effective_config"] == {
        "archive_root": str(archive_root),
        "selection_policy": {
            "profile": "all",
            "minimum_support_level": "scaffold",
            "include_sources": ["alpha_source"],
            "exclude_sources": [],
        },
        "execution_policy": {
            "incremental": False,
            "redaction": "off",
            "validation": "report",
        },
        "config_source": "explicit",
        "config_path": str(config_path),
    }
    assert payload["validation"]["mode"] == "report"

    written_rows = read_jsonl(Path(payload["sources"][0]["output_path"]))
    assert written_rows[0]["messages"][0]["text"] == "Authorization: Bearer real-token-1234567890"
    assert payload["sources"][0]["redaction_event_count"] == 0

    second_exit_code, second_stdout, second_stderr = run_cli(
        "collect",
        "--all",
        "--config",
        str(config_path),
        registry=registry,
    )
    assert second_exit_code == 0
    assert second_stderr == ""
    second_payload = json.loads(second_stdout)
    assert second_payload["sources"][0]["skipped_conversation_count"] == 0
    assert second_payload["sources"][0]["written_conversation_count"] == 1

    show_exit_code, show_stdout, show_stderr = run_cli(
        "runs",
        "show",
        second_payload["run_id"],
        "--archive-root",
        str(archive_root),
    )
    assert show_exit_code == 0
    assert show_stderr == ""
    show_payload = json.loads(show_stdout)
    assert show_payload["effective_config"] == payload["effective_config"]


def test_cli_flags_override_config_values(tmp_path: Path) -> None:
    config_path = write_config(
        tmp_path / "collector.toml",
        archive_root=tmp_path / "from-config",
        profile="complete_only",
        sources=("beta_source",),
        incremental=False,
        redaction="off",
        validation="report",
    )
    override_archive_root = tmp_path / "from-cli"
    registry = make_registry(
        StubExecutableCollector(
            descriptor=make_descriptor("alpha_source"),
            collected_at_values=[
                "2026-03-19T01:00:00Z",
                "2026-03-19T01:05:00Z",
            ],
        ),
        StubExecutableCollector(
            descriptor=make_descriptor("beta_source"),
            collected_at_values=["2026-03-19T01:00:00Z"],
        ),
    )

    exit_code, stdout, stderr = run_cli(
        "collect",
        "--all",
        "--config",
        str(config_path),
        "--archive-root",
        str(override_archive_root),
        "--profile",
        "all",
        "--source",
        "alpha_source",
        "--incremental",
        "--redaction",
        "on",
        "--validation",
        "strict",
        registry=registry,
    )

    assert exit_code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["archive_root"] == str(override_archive_root)
    assert payload["selected_sources"] == ["alpha_source"]
    assert payload["effective_config"]["archive_root"] == str(override_archive_root)
    assert payload["effective_config"]["selection_policy"] == {
        "profile": "all",
        "minimum_support_level": "scaffold",
        "include_sources": ["alpha_source"],
        "exclude_sources": [],
    }
    assert payload["effective_config"]["execution_policy"] == {
        "incremental": True,
        "redaction": "on",
        "validation": "strict",
    }
    assert payload["validation"]["mode"] == "strict"

    written_rows = read_jsonl(Path(payload["sources"][0]["output_path"]))
    assert written_rows[0]["messages"][0]["text"] == "Authorization: Bearer [REDACTED]"
    assert payload["sources"][0]["redaction_event_count"] == 1

    second_exit_code, second_stdout, second_stderr = run_cli(
        "collect",
        "--all",
        "--config",
        str(config_path),
        "--archive-root",
        str(override_archive_root),
        "--profile",
        "all",
        "--source",
        "alpha_source",
        "--incremental",
        "--redaction",
        "on",
        "--validation",
        "strict",
        registry=registry,
    )

    assert second_exit_code == 0
    assert second_stderr == ""
    second_payload = json.loads(second_stdout)
    assert second_payload["sources"][0]["skipped_conversation_count"] == 1
    assert second_payload["sources"][0]["written_conversation_count"] == 0


def test_collect_uses_default_config_path_when_present(tmp_path: Path) -> None:
    default_config_path = write_config(
        tmp_path / "default-collector.toml",
        archive_root=tmp_path / "archive",
        sources=("alpha_source",),
        incremental=True,
        redaction="on",
        validation="off",
    )
    registry = make_registry(
        StubExecutableCollector(
            descriptor=make_descriptor("alpha_source"),
            collected_at_values=["2026-03-19T02:00:00Z"],
        )
    )

    with patch(
        "llm_chat_archive.config.default_collect_config_path",
        return_value=default_config_path,
    ):
        exit_code, stdout, stderr = run_cli(
            "collect",
            "--all",
            registry=registry,
        )

    assert exit_code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["effective_config"]["config_source"] == "default_path"
    assert payload["effective_config"]["config_path"] == str(default_config_path)
    assert payload["archive_root"] == str(tmp_path / "archive")


def test_collect_fails_with_malformed_config(tmp_path: Path) -> None:
    config_path = tmp_path / "broken.toml"
    config_path.write_text("[collect\narchive_root = \"/tmp/archive\"\n", encoding="utf-8")

    exit_code, stdout, stderr = run_cli(
        "collect",
        "--all",
        "--config",
        str(config_path),
    )

    assert exit_code == 2
    assert stdout == ""
    assert "collector config is not valid TOML" in stderr


def test_config_init_writes_default_template(tmp_path: Path) -> None:
    output_path = tmp_path / "config" / "collector.toml"

    with patch(
        "llm_chat_archive.config.default_collect_config_path",
        return_value=output_path,
    ):
        exit_code, stdout, stderr = run_cli("config", "init")

    assert exit_code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload == {
        "archive_root": str(DEFAULT_ARCHIVE_ROOT),
        "output_path": str(output_path),
        "overwrote": False,
        "written": True,
    }
    assert output_path.read_text(encoding="utf-8") == render_collect_config_template()


def test_config_init_refuses_to_overwrite_existing_file_without_force(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "collector.toml"
    output_path.write_text("existing = true\n", encoding="utf-8")

    exit_code, stdout, stderr = run_cli(
        "config",
        "init",
        "--output",
        str(output_path),
    )

    assert exit_code == 2
    assert stdout == ""
    assert "collector config already exists" in stderr
    assert output_path.read_text(encoding="utf-8") == "existing = true\n"


def test_config_init_force_overwrites_existing_file(tmp_path: Path) -> None:
    output_path = tmp_path / "collector.toml"
    output_path.write_text("existing = true\n", encoding="utf-8")

    exit_code, stdout, stderr = run_cli(
        "config",
        "init",
        "--output",
        str(output_path),
        "--force",
    )

    assert exit_code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["output_path"] == str(output_path)
    assert payload["archive_root"] == str(DEFAULT_ARCHIVE_ROOT)
    assert payload["overwrote"] is True
    assert payload["written"] is True
    assert output_path.read_text(encoding="utf-8") == render_collect_config_template()


def test_config_init_supports_custom_archive_root(tmp_path: Path) -> None:
    output_path = tmp_path / "collector.toml"
    archive_root = tmp_path / "chat-history"

    exit_code, stdout, stderr = run_cli(
        "config",
        "init",
        "--output",
        str(output_path),
        "--archive-root",
        str(archive_root),
    )

    assert exit_code == 0
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload == {
        "archive_root": str(archive_root),
        "output_path": str(output_path),
        "overwrote": False,
        "written": True,
    }
    assert output_path.read_text(encoding="utf-8") == render_collect_config_template(
        archive_root=archive_root
    )


def test_config_init_prints_template_without_writing(tmp_path: Path) -> None:
    output_path = tmp_path / "collector.toml"

    with patch(
        "llm_chat_archive.config.default_collect_config_path",
        return_value=output_path,
    ):
        exit_code, stdout, stderr = run_cli("config", "init", "--print")

    assert exit_code == 0
    assert stderr == ""
    assert stdout == render_collect_config_template()
    assert not output_path.exists()
