from __future__ import annotations

import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from llm_chat_archive import cli
from llm_chat_archive.archive_inspect import list_archive_conversations
from llm_chat_archive.incremental import (
    build_conversation_dedupe_components,
    build_conversation_dedupe_key,
    build_message_fingerprint,
    write_incremental_collection,
)
from llm_chat_archive.models import (
    CollectionPlan,
    CollectionResult,
    MessageRole,
    NormalizedConversation,
    NormalizedMessage,
    SourceDescriptor,
    SupportLevel,
    TranscriptCompleteness,
)
from llm_chat_archive.registry import CollectorRegistry
from llm_chat_archive.runner import run_collection_batch
from llm_chat_archive.sources.codex_cli import CodexCliCollector, parse_rollout_file

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "codex_cli"
ACTIVE_ROLLOUT_PATH = (
    FIXTURE_ROOT
    / "sessions"
    / "2026"
    / "03"
    / "14"
    / "rollout-20260314T090000-session-active.jsonl"
)


def read_jsonl(path: Path) -> list[dict[str, object]]:
    content = path.read_text(encoding="utf-8")
    if not content.strip():
        return []
    return [json.loads(line) for line in content.splitlines()]


def run_cli_collect(registry: CollectorRegistry, archive_root: Path) -> tuple[int, dict[str, object], str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with (
        patch("llm_chat_archive.cli.build_registry", return_value=registry),
        redirect_stdout(stdout),
        redirect_stderr(stderr),
    ):
        exit_code = cli.main(
            [
                "collect",
                "codex_cli",
                "--archive-root",
                str(archive_root),
                "--input-root",
                str(FIXTURE_ROOT),
                "--execute",
            ]
        )
    return exit_code, json.loads(stdout.getvalue()), stderr.getvalue()


def make_conversation(
    *,
    source: str = "test_source",
    collected_at: str,
    session_id: str = "session-1",
    artifact_path: str = "/tmp/session-1.jsonl",
    messages: tuple[NormalizedMessage, ...],
    transcript_completeness: TranscriptCompleteness = TranscriptCompleteness.COMPLETE,
    limitations: tuple[str, ...] = (),
) -> NormalizedConversation:
    return NormalizedConversation(
        source=source,
        execution_context="cli",
        collected_at=collected_at,
        messages=messages,
        transcript_completeness=transcript_completeness,
        limitations=limitations,
        source_session_id=session_id,
        source_artifact_path=artifact_path,
    )


class UpgradeAwareStubCollector:
    def __init__(
        self,
        *,
        descriptor: SourceDescriptor,
        conversations_by_run: tuple[NormalizedConversation, ...],
    ) -> None:
        self.descriptor = descriptor
        self._conversations_by_run = conversations_by_run
        self._index = 0

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
        conversation = self._conversations_by_run[self._index]
        self._index += 1
        return write_incremental_collection(
            source=self.descriptor.key,
            archive_root=archive_root,
            input_roots=tuple(input_roots or ()),
            scanned_artifact_count=1,
            collected_at=conversation.collected_at,
            conversations=(conversation,),
        )


def test_incremental_dedupe_key_uses_source_identity_and_message_fingerprint() -> None:
    conversation = parse_rollout_file(
        ACTIVE_ROLLOUT_PATH,
        collected_at="2026-03-19T00:00:00Z",
    )

    assert conversation is not None
    components = build_conversation_dedupe_components(conversation)

    assert components == {
        "source": "codex_cli",
        "source_session_id": "session-active",
        "source_artifact_path": str(ACTIVE_ROLLOUT_PATH.resolve(strict=False)),
        "message_fingerprint": build_message_fingerprint(
            [message.to_dict() for message in conversation.messages]
        ),
    }
    assert build_conversation_dedupe_key(conversation).startswith("sha256:")


def test_batch_collect_skips_previously_archived_conversations(tmp_path: Path) -> None:
    registry = CollectorRegistry()
    registry.register(CodexCliCollector())

    with (
        patch(
            "llm_chat_archive.runner.utc_timestamp",
            side_effect=[
                "2026-03-19T00:00:00Z",
                "2026-03-19T00:00:05Z",
                "2026-03-19T00:01:00Z",
                "2026-03-19T00:01:05Z",
            ],
        ),
        patch(
            "llm_chat_archive.sources.codex_cli.utc_timestamp",
            side_effect=[
                "2026-03-19T00:00:01Z",
                "2026-03-19T00:01:01Z",
            ],
        ),
    ):
        first = run_collection_batch(registry, tmp_path, input_roots=(FIXTURE_ROOT,))
        second = run_collection_batch(registry, tmp_path, input_roots=(FIXTURE_ROOT,))

    first_source = first.sources[0]
    second_source = second.sources[0]

    assert first_source.conversation_count == 2
    assert first_source.skipped_conversation_count == 0
    assert first_source.written_conversation_count == 2
    assert read_jsonl(first_source.output_path) != []

    assert second_source.conversation_count == 2
    assert second_source.skipped_conversation_count == 2
    assert second_source.written_conversation_count == 0
    assert read_jsonl(second_source.output_path) == []

    manifest = json.loads(second.manifest_path.read_text(encoding="utf-8"))
    assert manifest["skipped_conversation_count"] == 2
    assert manifest["written_conversation_count"] == 0
    assert manifest["upgraded_conversation_count"] == 0
    assert manifest["sources"][0]["skipped_conversation_count"] == 2
    assert manifest["sources"][0]["written_conversation_count"] == 0
    assert manifest["sources"][0]["upgraded_conversation_count"] == 0


def test_single_source_collect_skips_previously_archived_conversations(tmp_path: Path) -> None:
    registry = CollectorRegistry()
    registry.register(CodexCliCollector())

    with patch(
        "llm_chat_archive.sources.codex_cli.utc_timestamp",
        side_effect=[
            "2026-03-19T02:00:00Z",
            "2026-03-19T02:01:00Z",
        ],
    ):
        first_exit_code, first_payload, first_stderr = run_cli_collect(registry, tmp_path)
        second_exit_code, second_payload, second_stderr = run_cli_collect(registry, tmp_path)

    assert first_exit_code == 0
    assert first_stderr == ""
    assert first_payload["conversation_count"] == 2
    assert first_payload["skipped_conversation_count"] == 0
    assert first_payload["written_conversation_count"] == 2
    assert read_jsonl(Path(first_payload["output_path"])) != []

    assert second_exit_code == 0
    assert second_stderr == ""
    assert second_payload["conversation_count"] == 2
    assert second_payload["skipped_conversation_count"] == 2
    assert second_payload["written_conversation_count"] == 0
    assert second_payload["upgraded_conversation_count"] == 0
    assert read_jsonl(Path(second_payload["output_path"])) == []


def test_incremental_collect_upgrades_richer_transcript_and_supersedes_old_row(
    tmp_path: Path,
) -> None:
    partial = make_conversation(
        collected_at="2026-03-19T03:00:00Z",
        messages=(
            NormalizedMessage(
                role=MessageRole.USER,
                text="Need a deploy checklist",
            ),
        ),
        transcript_completeness=TranscriptCompleteness.PARTIAL,
        limitations=("missing assistant reply",),
    )
    complete = make_conversation(
        collected_at="2026-03-19T03:05:00Z",
        messages=(
            NormalizedMessage(
                role=MessageRole.USER,
                text="Need a deploy checklist",
            ),
            NormalizedMessage(
                role=MessageRole.ASSISTANT,
                text="Start with the rollback plan and smoke test list.",
            ),
        ),
    )

    first = write_incremental_collection(
        source="test_source",
        archive_root=tmp_path,
        input_roots=(),
        scanned_artifact_count=1,
        collected_at=partial.collected_at,
        conversations=(partial,),
    )
    second = write_incremental_collection(
        source="test_source",
        archive_root=tmp_path,
        input_roots=(),
        scanned_artifact_count=1,
        collected_at=complete.collected_at,
        conversations=(complete,),
    )

    assert first.written_conversation_count == 1
    assert first.upgraded_conversation_count == 0
    assert second.skipped_conversation_count == 0
    assert second.written_conversation_count == 1
    assert second.upgraded_conversation_count == 1

    active_rows = list_archive_conversations(tmp_path, source="test_source")
    assert len(active_rows) == 1
    assert active_rows[0].source_session_id == "session-1"
    assert active_rows[0].transcript_completeness == "complete"
    assert active_rows[0].message_count == 2

    first_rows = read_jsonl(first.output_path)
    assert first_rows[0]["transcript_completeness"] == "partial"
    assert first_rows[0]["superseded_at"] == "2026-03-19T03:05:00Z"
    assert read_jsonl(second.output_path)[0]["messages"][-1]["role"] == "assistant"


def test_incremental_collect_skips_weaker_transcript_when_archive_already_has_richer_row(
    tmp_path: Path,
) -> None:
    complete = make_conversation(
        collected_at="2026-03-19T04:00:00Z",
        messages=(
            NormalizedMessage(
                role=MessageRole.USER,
                text="Summarize the release blockers.",
            ),
            NormalizedMessage(
                role=MessageRole.ASSISTANT,
                text="Blockers are test flakes and missing migration notes.",
            ),
        ),
    )
    degraded = make_conversation(
        collected_at="2026-03-19T04:05:00Z",
        messages=(
            NormalizedMessage(
                role=MessageRole.USER,
                text="Summarize the release blockers.",
            ),
        ),
        transcript_completeness=TranscriptCompleteness.PARTIAL,
        limitations=("missing assistant reply",),
    )

    first = write_incremental_collection(
        source="test_source",
        archive_root=tmp_path,
        input_roots=(),
        scanned_artifact_count=1,
        collected_at=complete.collected_at,
        conversations=(complete,),
    )
    second = write_incremental_collection(
        source="test_source",
        archive_root=tmp_path,
        input_roots=(),
        scanned_artifact_count=1,
        collected_at=degraded.collected_at,
        conversations=(degraded,),
    )

    assert first.written_conversation_count == 1
    assert second.skipped_conversation_count == 1
    assert second.written_conversation_count == 0
    assert second.upgraded_conversation_count == 0
    assert read_jsonl(second.output_path) == []

    active_rows = list_archive_conversations(tmp_path, source="test_source")
    assert len(active_rows) == 1
    assert active_rows[0].transcript_completeness == "complete"
    assert active_rows[0].message_count == 2
    assert "superseded_at" not in read_jsonl(first.output_path)[0]


def test_batch_collect_reports_upgraded_conversation_count_for_richer_transcript(
    tmp_path: Path,
) -> None:
    registry = CollectorRegistry()
    registry.register(
        UpgradeAwareStubCollector(
            descriptor=SourceDescriptor(
                key="upgrade_source",
                display_name="Upgrade Source",
                execution_context="test",
                support_level=SupportLevel.COMPLETE,
                default_input_roots=(),
            ),
            conversations_by_run=(
                make_conversation(
                    source="upgrade_source",
                    collected_at="2026-03-19T05:00:00Z",
                    messages=(
                        NormalizedMessage(
                            role=MessageRole.USER,
                            text="Need release notes.",
                        ),
                    ),
                    transcript_completeness=TranscriptCompleteness.PARTIAL,
                    limitations=("missing assistant reply",),
                ),
                make_conversation(
                    source="upgrade_source",
                    collected_at="2026-03-19T05:10:00Z",
                    messages=(
                        NormalizedMessage(
                            role=MessageRole.USER,
                            text="Need release notes.",
                        ),
                        NormalizedMessage(
                            role=MessageRole.ASSISTANT,
                            text="Start with the changelog and highlight migrations.",
                        ),
                    ),
                ),
            ),
        )
    )

    first = run_collection_batch(registry, tmp_path)
    second = run_collection_batch(registry, tmp_path)

    assert first.sources[0].upgraded_conversation_count == 0
    assert second.sources[0].written_conversation_count == 1
    assert second.sources[0].upgraded_conversation_count == 1

    manifest = json.loads(second.manifest_path.read_text(encoding="utf-8"))
    assert manifest["upgraded_conversation_count"] == 1
    assert manifest["sources"][0]["upgraded_conversation_count"] == 1
