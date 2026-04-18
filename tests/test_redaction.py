from __future__ import annotations

import json
from pathlib import Path

from llm_chat_archive.incremental import write_incremental_collection
from llm_chat_archive.models import (
    AppShellProvenance,
    ConversationProvenance,
    MessageRole,
    NormalizedConversation,
    NormalizedMessage,
)
from llm_chat_archive.registry import CollectorRegistry
from llm_chat_archive.runner import run_collection_batch
from llm_chat_archive.sources.codex_cli import CodexCliCollector
from llm_chat_archive.sources.gemini_cli import GeminiCliCollector

REPO_ROOT = Path(__file__).resolve().parents[1]
CODEX_REDACTION_FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "redaction" / "codex_cli"
GEMINI_FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "gemini_cli"


def read_jsonl(path: Path) -> list[dict[str, object]]:
    content = path.read_text(encoding="utf-8")
    if not content.strip():
        return []
    return [json.loads(line) for line in content.splitlines()]


def test_write_incremental_collection_redacts_messages_and_metadata_before_dedupe(
    tmp_path: Path,
) -> None:
    conversation = NormalizedConversation(
        source="test_source",
        execution_context="cli",
        collected_at="2026-03-19T08:00:00Z",
        messages=(
            NormalizedMessage(
                role=MessageRole.USER,
                text=(
                    "Authorization: Bearer ya29.a0AfH6SM-message-token-1234567890\n"
                    "OpenAI: sk-proj-1234567890abcdefghijklmnop\n"
                    "Anthropic: sk-ant-api03-1234567890abcdefghijklmnop\n"
                    "Google: AIzaSyA1234567890abcdefghijklmnopqrstu\n"
                    '{"refresh_token":"refresh-secret","access_token":"access-secret"}'
                ),
            ),
        ),
        session_metadata={
            "oauth": {
                "access_token": "metadata-access-token",
                "refresh_token": "metadata-refresh-token",
            },
            "notes": "client_secret=metadata-client-secret",
        },
        provenance=ConversationProvenance(
            source="cli",
            originator="test_runner",
            app_shell=AppShellProvenance(
                auxiliary_paths=(
                    "Authorization: Bearer metadata-bearer-token-1234567890",
                )
            ),
        ),
    )

    first_result = write_incremental_collection(
        source="test_source",
        archive_root=tmp_path,
        input_roots=(),
        scanned_artifact_count=1,
        collected_at="2026-03-19T08:00:00Z",
        conversations=(conversation,),
    )
    second_result = write_incremental_collection(
        source="test_source",
        archive_root=tmp_path,
        input_roots=(),
        scanned_artifact_count=1,
        collected_at="2026-03-19T08:05:00Z",
        conversations=(conversation,),
    )

    assert first_result.redaction_event_count == 10
    assert second_result.redaction_event_count == 0
    assert second_result.written_conversation_count == 0
    assert read_jsonl(second_result.output_path) == []

    row = read_jsonl(first_result.output_path)[0]
    message_text = row["messages"][0]["text"]
    assert message_text == (
        "Authorization: Bearer [REDACTED]\n"
        "OpenAI: [REDACTED_API_KEY]\n"
        "Anthropic: [REDACTED_API_KEY]\n"
        "Google: [REDACTED_API_KEY]\n"
        '{"refresh_token":"[REDACTED]","access_token":"[REDACTED]"}'
    )
    assert row["session_metadata"] == {
        "oauth": {
            "access_token": "[REDACTED]",
            "refresh_token": "[REDACTED]",
        },
        "notes": "client_secret=[REDACTED]",
    }
    assert row["provenance"]["app_shell"]["auxiliary_paths"] == [
        "Authorization: Bearer [REDACTED]"
    ]


def test_single_source_and_batch_collect_share_redaction_write_path(tmp_path: Path) -> None:
    collector = CodexCliCollector()
    single_archive_root = tmp_path / "single"
    batch_archive_root = tmp_path / "batch"

    single_result = collector.collect(
        single_archive_root,
        input_roots=(CODEX_REDACTION_FIXTURE_ROOT,),
    )

    single_rows = read_jsonl(single_result.output_path)
    assert single_result.redaction_event_count == 6
    assert single_rows[0]["messages"] == [
        {
            "role": "developer",
            "text": "Keep Authorization: Bearer [REDACTED] out of the archive.",
            "source_message_id": "msg-dev-redaction",
        },
        {
            "role": "user",
            "text": "Keys to mask: [REDACTED_API_KEY] and [REDACTED_API_KEY].",
            "source_message_id": "msg-user-redaction",
        },
        {
            "role": "assistant",
            "text": (
                'Credential JSON: {"access_token":"[REDACTED]",'
                '"refresh_token":"[REDACTED]","api_key":"[REDACTED]"}'
            ),
            "source_message_id": "msg-assistant-redaction",
        },
    ]

    registry = CollectorRegistry()
    registry.register(CodexCliCollector())
    batch_result = run_collection_batch(
        registry,
        batch_archive_root,
        input_roots=(CODEX_REDACTION_FIXTURE_ROOT,),
    )

    assert batch_result.redaction_event_count == 6
    assert batch_result.sources[0].redaction_event_count == 6

    manifest = json.loads(batch_result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["redaction_event_count"] == 6
    assert manifest["sources"][0]["redaction_event_count"] == 6

    batch_rows = read_jsonl(Path(batch_result.sources[0].output_path))
    assert batch_rows == single_rows


def test_credential_file_body_is_never_serialized_into_transcript_messages(
    tmp_path: Path,
) -> None:
    credential_body = (GEMINI_FIXTURE_ROOT / "oauth_creds.json").read_text(encoding="utf-8")
    collector = GeminiCliCollector(repo_path=REPO_ROOT)

    result = collector.collect(tmp_path, input_roots=(GEMINI_FIXTURE_ROOT,))
    rows = read_jsonl(result.output_path)
    message_texts = "\n".join(
        message.get("text", "")
        for row in rows
        for message in row["messages"]
        if isinstance(message, dict)
    )

    assert credential_body not in message_texts
    assert '"refresh_token": "secret"' not in message_texts
