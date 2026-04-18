from __future__ import annotations

import json
import shutil
import sqlite3
import tempfile
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path

from ..incremental import write_incremental_collection
from ..models import (
    AppShellProvenance,
    AutomationRunProvenance,
    CollectionPlan,
    CollectionResult,
    MessageProvenance,
    MessageRole,
    NormalizedConversation,
    NormalizedMessage,
    SourceDescriptor,
    SourceSupportMetadata,
    SupportLevel,
    TranscriptCompleteness,
)
from ..source_roots import (
    all_platform_root,
    darwin_root,
    default_descriptor_input_roots,
    linux_root,
    windows_root,
)
from .codex_rollout import (
    CodexSessionMetadata,
    build_conversation_provenance,
    iter_rollout_paths,
    resolve_input_roots,
    utc_timestamp,
)

ROLE_MAP = {
    "assistant": MessageRole.ASSISTANT,
    "developer": MessageRole.DEVELOPER,
    "user": MessageRole.USER,
}
TEXT_ITEM_TYPES = frozenset({"input_text", "output_text", "text"})
STATE_DB_GLOB = "**/state_5.sqlite"
AUTOMATION_DB_GLOB = "**/sqlite/codex-dev.db"
AUTOMATION_TOML_GLOB = "**/automations/*/automation.toml"
ARCHIVED_AUTOMATION_USER_SNAPSHOT_LIMITATION = (
    "automation_origin_user_message_reconstructed_from_archived_snapshot"
)
ARCHIVED_AUTOMATION_ASSISTANT_SNAPSHOT_LIMITATION = (
    "automation_origin_assistant_message_reconstructed_from_archived_snapshot"
)
THREAD_METADATA_USER_LIMITATION = "user_message_reconstructed_from_thread_metadata"
MISSING_AUTOMATION_USER_LIMITATION = "automation_origin_user_message_missing"
MISSING_AUTOMATION_ASSISTANT_LIMITATION = (
    "automation_origin_assistant_message_missing"
)

CODEX_APP_DESCRIPTOR = SourceDescriptor(
    key="codex_app",
    display_name="Codex Desktop App",
    execution_context="standalone_app",
    support_level=SupportLevel.COMPLETE,
    default_input_roots=(
        "~/.codex",
        "~/Library/Application Support/Codex",
        "~/Library/Logs/com.openai.codex",
        "~/Library/Preferences/com.openai.codex.plist",
        "~/Library/Caches/com.openai.codex",
    ),
    artifact_root_candidates=(
        all_platform_root("$HOME/.codex"),
        darwin_root("$HOME/Library/Application Support/Codex"),
        linux_root("$XDG_CONFIG_HOME/Codex"),
        windows_root("$APPDATA/Codex"),
        darwin_root("$HOME/Library/Logs/com.openai.codex"),
        linux_root("$XDG_STATE_HOME/Codex/logs"),
        windows_root("$LOCALAPPDATA/Codex/logs"),
        darwin_root("$HOME/Library/Preferences/com.openai.codex.plist"),
        darwin_root("$HOME/Library/Caches/com.openai.codex"),
        linux_root("$XDG_CACHE_HOME/Codex"),
        windows_root("$LOCALAPPDATA/Codex/Cache"),
    ),
    notes=(
        "Uses shared ~/.codex rollout JSONL as the primary transcript source and joins shared state_5.sqlite metadata.",
        'Selects only sessions whose session_meta payload originator is "Codex Desktop".',
        "Distinguishes interactive and automation-origin conversations via sqlite/codex-dev.db automation_runs metadata.",
        "Uses archived automation snapshot fields only to repair missing canonical user or assistant messages.",
        "Treats app shell support roots under ~/Library and automation definition files as provenance only.",
    ),
    support_metadata=SourceSupportMetadata(
        product_label="Codex",
        host_surface="Desktop app",
        expected_transcript_completeness=TranscriptCompleteness.COMPLETE,
        limitation_summary=(
            "Desktop automation run state comes from shared SQLite metadata, and archived "
            "automation rollout gaps stay rollout-first while repaired bodies are tagged "
            "with explicit fallback provenance."
        ),
        limitations=(
            "Automation-origin attribution depends on local sqlite/codex-dev.db automation_runs rows being present, with state_5.sqlite thread metadata used as a secondary join signal.",
            "Archived automation conversations keep rollout bodies canonical and tag any repaired user or assistant body with message-level provenance from archived snapshot or thread metadata.",
        ),
    ),
)


@dataclass(frozen=True, slots=True)
class CodexThreadRecord:
    thread_id: str
    rollout_path: str | None = None
    cwd: str | None = None
    title: str | None = None
    first_user_message: str | None = None
    archived: bool = False
    cli_version: str | None = None


@dataclass(frozen=True, slots=True)
class CodexAutomationDefinition:
    automation_id: str
    name: str | None = None
    prompt: str | None = None
    status: str | None = None
    schedule: str | None = None
    model: str | None = None
    reasoning_effort: str | None = None
    definition_path: str | None = None


@dataclass(frozen=True, slots=True)
class CodexAutomationRunRecord:
    thread_id: str
    automation_id: str
    status: str | None = None
    thread_title: str | None = None
    source_cwd: str | None = None
    inbox_title: str | None = None
    inbox_summary: str | None = None
    archived_user_message: str | None = None
    archived_assistant_message: str | None = None
    archived_reason: str | None = None


@dataclass(frozen=True, slots=True)
class CodexAppMetadataIndex:
    thread_records: dict[str, CodexThreadRecord]
    thread_records_by_rollout_path: dict[str, CodexThreadRecord]
    automation_runs: dict[str, CodexAutomationRunRecord]
    automation_runs_by_cwd_and_title: dict[tuple[str, str], CodexAutomationRunRecord]
    automation_runs_by_cwd_and_user_message: dict[tuple[str, str], CodexAutomationRunRecord]
    automation_definitions: dict[str, CodexAutomationDefinition]

    def lookup_thread(
        self,
        *,
        session_id: str | None,
        rollout_path: Path,
    ) -> CodexThreadRecord | None:
        if session_id is not None:
            record = self.thread_records.get(session_id)
            if record is not None:
                return record
        return self.thread_records_by_rollout_path.get(str(rollout_path))

    def lookup_automation_run(
        self,
        *,
        session_id: str | None,
        thread_record: CodexThreadRecord | None,
        session_metadata: CodexSessionMetadata,
        rollout_messages: tuple[NormalizedMessage, ...],
    ) -> CodexAutomationRunRecord | None:
        thread_id = session_id or (None if thread_record is None else thread_record.thread_id)
        if thread_id is not None:
            record = self.automation_runs.get(thread_id)
            if record is not None:
                return record

        resolved_cwd = (
            session_metadata.cwd or (None if thread_record is None else thread_record.cwd)
        )
        thread_title = None if thread_record is None else thread_record.title
        if resolved_cwd is not None and thread_title is not None:
            record = self.automation_runs_by_cwd_and_title.get((resolved_cwd, thread_title))
            if record is not None:
                return record

        first_user_text = _first_non_empty(
            _first_message_text(rollout_messages, MessageRole.USER),
            None if thread_record is None else thread_record.first_user_message,
        )
        if resolved_cwd is not None and first_user_text is not None:
            return self.automation_runs_by_cwd_and_user_message.get(
                (resolved_cwd, first_user_text)
            )
        return None

    def lookup_automation_definition(
        self,
        automation_run: CodexAutomationRunRecord | None,
    ) -> CodexAutomationDefinition | None:
        if automation_run is None:
            return None
        return self.automation_definitions.get(automation_run.automation_id)


@dataclass(frozen=True, slots=True)
class CodexAppRollout:
    session_metadata: CodexSessionMetadata
    messages: tuple[NormalizedMessage, ...]
    artifact_path: Path


@dataclass(frozen=True, slots=True)
class CodexAppCollector:
    descriptor: SourceDescriptor = CODEX_APP_DESCRIPTOR

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
        resolved_input_roots = resolve_input_roots(input_roots or self._default_input_roots())
        rollout_paths = tuple(iter_rollout_paths(resolved_input_roots))
        app_shell = discover_app_shell_provenance(resolved_input_roots)
        metadata_index = build_codex_app_metadata_index(resolved_input_roots)
        collected_at = utc_timestamp()
        conversations = (
            parse_rollout_file(
                rollout_path,
                collected_at=collected_at,
                app_shell=app_shell,
                metadata_index=metadata_index,
            )
            for rollout_path in rollout_paths
        )
        return write_incremental_collection(
            source=self.descriptor.key,
            archive_root=archive_root,
            input_roots=resolved_input_roots,
            scanned_artifact_count=len(rollout_paths),
            collected_at=collected_at,
            conversations=conversations,
        )

    def _default_input_roots(self) -> tuple[Path, ...]:
        return default_descriptor_input_roots(self.descriptor)


def parse_rollout_file(
    rollout_path: Path,
    *,
    collected_at: str | None = None,
    app_shell: AppShellProvenance | None = None,
    metadata_index: CodexAppMetadataIndex | None = None,
) -> NormalizedConversation | None:
    rollout = _parse_rollout(rollout_path)
    if rollout is None or not _is_codex_desktop_session(rollout.session_metadata, rollout.artifact_path):
        return None

    metadata_index = metadata_index or CodexAppMetadataIndex(
        thread_records={},
        thread_records_by_rollout_path={},
        automation_runs={},
        automation_runs_by_cwd_and_title={},
        automation_runs_by_cwd_and_user_message={},
        automation_definitions={},
    )
    thread_record = metadata_index.lookup_thread(
        session_id=rollout.session_metadata.session_id,
        rollout_path=rollout.artifact_path,
    )
    automation_run = metadata_index.lookup_automation_run(
        session_id=rollout.session_metadata.session_id,
        thread_record=thread_record,
        session_metadata=rollout.session_metadata,
        rollout_messages=rollout.messages,
    )
    automation_definition = metadata_index.lookup_automation_definition(automation_run)

    messages, completeness, limitations = _build_canonical_messages(
        rollout.messages,
        thread_record=thread_record,
        automation_run=automation_run,
        session_metadata=rollout.session_metadata,
    )
    if not messages:
        return None

    return NormalizedConversation(
        source=CODEX_APP_DESCRIPTOR.key,
        execution_context=CODEX_APP_DESCRIPTOR.execution_context,
        collected_at=collected_at or utc_timestamp(),
        messages=messages,
        transcript_completeness=completeness,
        limitations=limitations,
        source_session_id=(
            rollout.session_metadata.session_id
            or (None if thread_record is None else thread_record.thread_id)
            or rollout.artifact_path.stem
        ),
        source_artifact_path=str(rollout.artifact_path),
        provenance=_build_provenance(
            rollout.session_metadata,
            app_shell=app_shell,
            thread_record=thread_record,
            automation_run=automation_run,
            automation_definition=automation_definition,
        ),
    )


def build_codex_app_metadata_index(
    input_roots: tuple[Path, ...] | None,
) -> CodexAppMetadataIndex:
    if not input_roots:
        return CodexAppMetadataIndex(
            thread_records={},
            thread_records_by_rollout_path={},
            automation_runs={},
            automation_runs_by_cwd_and_title={},
            automation_runs_by_cwd_and_user_message={},
            automation_definitions={},
        )

    state_db_paths = _discover_paths(
        input_roots,
        direct_match=_is_codex_state_db_path,
        glob_pattern=STATE_DB_GLOB,
        expect_dir=False,
    )
    automation_db_paths = _discover_paths(
        input_roots,
        direct_match=_is_codex_automation_db_path,
        glob_pattern=AUTOMATION_DB_GLOB,
        expect_dir=False,
    )
    automation_toml_paths = _discover_paths(
        input_roots,
        direct_match=_is_codex_automation_toml_path,
        glob_pattern=AUTOMATION_TOML_GLOB,
        expect_dir=False,
    )

    thread_records: dict[str, CodexThreadRecord] = {}
    thread_records_by_rollout_path: dict[str, CodexThreadRecord] = {}
    for state_db_path in (Path(raw_path) for raw_path in state_db_paths):
        for record in _read_thread_records(state_db_path):
            thread_records.setdefault(record.thread_id, record)
            if record.rollout_path is not None:
                thread_records_by_rollout_path.setdefault(record.rollout_path, record)

    automation_definitions = _read_automation_definitions(
        automation_db_paths=tuple(Path(raw_path) for raw_path in automation_db_paths),
        automation_toml_paths=tuple(Path(raw_path) for raw_path in automation_toml_paths),
    )
    automation_runs: dict[str, CodexAutomationRunRecord] = {}
    automation_runs_by_cwd_and_title_entries: dict[
        tuple[str, str], list[CodexAutomationRunRecord]
    ] = {}
    automation_runs_by_cwd_and_user_message_entries: dict[
        tuple[str, str], list[CodexAutomationRunRecord]
    ] = {}
    for automation_db_path in (Path(raw_path) for raw_path in automation_db_paths):
        for record in _read_automation_run_records(automation_db_path):
            automation_runs.setdefault(record.thread_id, record)
            for title in (record.thread_title, record.inbox_title):
                if title is None or record.source_cwd is None:
                    continue
                automation_runs_by_cwd_and_title_entries.setdefault(
                    (record.source_cwd, title),
                    [],
                ).append(record)
            if record.archived_user_message is not None and record.source_cwd is not None:
                automation_runs_by_cwd_and_user_message_entries.setdefault(
                    (record.source_cwd, record.archived_user_message),
                    [],
                ).append(record)

    return CodexAppMetadataIndex(
        thread_records=thread_records,
        thread_records_by_rollout_path=thread_records_by_rollout_path,
        automation_runs=automation_runs,
        automation_runs_by_cwd_and_title=_resolve_unique_automation_run_index(
            automation_runs_by_cwd_and_title_entries
        ),
        automation_runs_by_cwd_and_user_message=_resolve_unique_automation_run_index(
            automation_runs_by_cwd_and_user_message_entries
        ),
        automation_definitions=automation_definitions,
    )


def discover_app_shell_provenance(
    input_roots: tuple[Path, ...] | None,
) -> AppShellProvenance | None:
    if not input_roots:
        return None

    application_support_roots = _discover_paths(
        input_roots,
        direct_match=_is_codex_application_support_root,
        glob_pattern="**/Application Support/Codex",
        expect_dir=True,
    )
    log_roots = _discover_paths(
        input_roots,
        direct_match=_is_codex_log_root,
        glob_pattern="**/Logs/com.openai.codex",
        expect_dir=True,
    )
    preference_paths = _discover_paths(
        input_roots,
        direct_match=_is_codex_preference_path,
        glob_pattern="**/Preferences/com.openai.codex.plist",
        expect_dir=False,
    )
    cache_roots = _discover_paths(
        input_roots,
        direct_match=_is_codex_cache_root,
        glob_pattern="**/Caches/com.openai.codex",
        expect_dir=True,
    )
    state_db_paths = _discover_paths(
        input_roots,
        direct_match=_is_codex_state_db_path,
        glob_pattern=STATE_DB_GLOB,
        expect_dir=False,
    )
    automation_db_paths = _discover_paths(
        input_roots,
        direct_match=_is_codex_automation_db_path,
        glob_pattern=AUTOMATION_DB_GLOB,
        expect_dir=False,
    )
    automation_definition_paths = _discover_paths(
        input_roots,
        direct_match=_is_codex_automation_toml_path,
        glob_pattern=AUTOMATION_TOML_GLOB,
        expect_dir=False,
    )

    auxiliary_paths = tuple(
        sorted({*automation_db_paths, *automation_definition_paths})
    )
    if (
        not application_support_roots
        and not log_roots
        and not preference_paths
        and not cache_roots
        and not state_db_paths
        and not auxiliary_paths
    ):
        return None

    return AppShellProvenance(
        application_support_roots=application_support_roots,
        log_roots=log_roots,
        state_db_paths=state_db_paths,
        preference_paths=preference_paths,
        cache_roots=cache_roots,
        auxiliary_paths=auxiliary_paths,
    )


def _build_provenance(
    session_metadata: CodexSessionMetadata,
    *,
    app_shell: AppShellProvenance | None,
    thread_record: CodexThreadRecord | None,
    automation_run: CodexAutomationRunRecord | None,
    automation_definition: CodexAutomationDefinition | None,
):
    resolved_cwd = (
        session_metadata.cwd
        or (None if automation_run is None else automation_run.source_cwd)
        or (None if thread_record is None else thread_record.cwd)
    )
    resolved_cli_version = session_metadata.cli_version or (
        None if thread_record is None else thread_record.cli_version
    )
    resolved_archived = session_metadata.archived or (
        False if thread_record is None else thread_record.archived
    )
    automation = None
    conversation_origin = "interactive"
    archived_reason = None
    if automation_run is not None:
        conversation_origin = "automation"
        archived_reason = automation_run.archived_reason
        resolved_title, resolved_title_source = _resolve_automation_title(
            thread_record=thread_record,
            automation_run=automation_run,
        )
        resolved_summary = automation_run.inbox_summary
        automation = AutomationRunProvenance(
            automation_id=automation_run.automation_id,
            automation_name=(
                None if automation_definition is None else automation_definition.name
            ),
            status=automation_run.status,
            schedule=(
                None if automation_definition is None else automation_definition.schedule
            ),
            source_cwd=automation_run.source_cwd,
            model=(
                None if automation_definition is None else automation_definition.model
            ),
            reasoning_effort=(
                None
                if automation_definition is None
                else automation_definition.reasoning_effort
            ),
            definition_path=(
                None
                if automation_definition is None
                else automation_definition.definition_path
            ),
            thread_title=automation_run.thread_title,
            thread_record_title=(
                None if thread_record is None else thread_record.title
            ),
            inbox_title=automation_run.inbox_title,
            inbox_summary=automation_run.inbox_summary,
            resolved_title=resolved_title,
            resolved_title_source=resolved_title_source,
            resolved_summary=resolved_summary,
            resolved_summary_source=(
                None if resolved_summary is None else "automation_runs.inbox_summary"
            ),
        )

    base_provenance = build_conversation_provenance(
        replace(
            session_metadata,
            cwd=resolved_cwd,
            cli_version=resolved_cli_version,
            archived=resolved_archived,
        )
    )
    return replace(
        base_provenance,
        cwd=resolved_cwd,
        cli_version=resolved_cli_version,
        archived=resolved_archived,
        archived_reason=archived_reason,
        conversation_origin=conversation_origin,
        automation=automation,
        app_shell=app_shell,
    )


def _build_canonical_messages(
    rollout_messages: tuple[NormalizedMessage, ...],
    *,
    thread_record: CodexThreadRecord | None,
    automation_run: CodexAutomationRunRecord | None,
    session_metadata: CodexSessionMetadata,
) -> tuple[tuple[NormalizedMessage, ...], TranscriptCompleteness, tuple[str, ...]]:
    archived_automation = automation_run is not None and (
        session_metadata.archived or (False if thread_record is None else thread_record.archived)
    )
    messages = _annotate_rollout_messages(
        rollout_messages,
        archived_automation=archived_automation,
    )
    limitations: list[str] = []

    if automation_run is not None:
        if not _has_role(messages, MessageRole.USER):
            user_text = _first_non_empty(
                automation_run.archived_user_message,
                None if thread_record is None else thread_record.first_user_message,
            )
            if user_text is not None:
                if _string_value(automation_run.archived_user_message) is not None:
                    limitations.append(ARCHIVED_AUTOMATION_USER_SNAPSHOT_LIMITATION)
                    source_message_id = "automation-archived-user"
                    message_provenance = MessageProvenance(
                        body_source="automation_runs.archived_user_message",
                        fallback=True,
                    )
                else:
                    limitations.append(THREAD_METADATA_USER_LIMITATION)
                    source_message_id = "thread-first-user-message"
                    message_provenance = MessageProvenance(
                        body_source="threads.first_user_message",
                        fallback=True,
                    )
                messages.insert(
                    _user_insertion_index(messages),
                    NormalizedMessage(
                        role=MessageRole.USER,
                        text=user_text,
                        timestamp=session_metadata.session_started_at,
                        source_message_id=source_message_id,
                        provenance=message_provenance,
                    ),
                )
        if not _has_role(messages, MessageRole.ASSISTANT):
            assistant_text = _string_value(automation_run.archived_assistant_message)
            if assistant_text is not None:
                limitations.append(ARCHIVED_AUTOMATION_ASSISTANT_SNAPSHOT_LIMITATION)
                messages.append(
                    NormalizedMessage(
                        role=MessageRole.ASSISTANT,
                        text=assistant_text,
                        source_message_id="automation-archived-assistant",
                        provenance=MessageProvenance(
                            body_source="automation_runs.archived_assistant_message",
                            fallback=True,
                        ),
                    )
                )

    limitations = _unique_limitations(limitations)
    completeness = TranscriptCompleteness.COMPLETE
    if automation_run is not None:
        if not _has_role(messages, MessageRole.USER):
            limitations.append(MISSING_AUTOMATION_USER_LIMITATION)
        if not _has_role(messages, MessageRole.ASSISTANT):
            limitations.append(MISSING_AUTOMATION_ASSISTANT_LIMITATION)
        limitations = _unique_limitations(limitations)
        if limitations:
            completeness = TranscriptCompleteness.PARTIAL

    return tuple(messages), completeness, tuple(limitations)


def _annotate_rollout_messages(
    messages: tuple[NormalizedMessage, ...],
    *,
    archived_automation: bool,
) -> list[NormalizedMessage]:
    if not archived_automation:
        return list(messages)
    return [
        replace(
            message,
            provenance=MessageProvenance(body_source="rollout.message"),
        )
        for message in messages
    ]


def _parse_rollout(rollout_path: Path) -> CodexAppRollout | None:
    resolved_path = rollout_path.expanduser().resolve(strict=False)
    session_metadata = CodexSessionMetadata(archived=_is_archived_rollout(resolved_path))
    messages: list[NormalizedMessage] = []

    try:
        lines = resolved_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return None

    saw_session_meta = False
    for raw_line in lines:
        record = _load_json_line(raw_line)
        if record is None:
            continue

        payload = record.get("payload")
        if not isinstance(payload, dict):
            continue

        if record.get("type") == "session_meta":
            session_metadata = CodexSessionMetadata.from_payload(
                payload,
                archived=_is_archived_rollout(resolved_path),
            )
            saw_session_meta = True
            continue

        if record.get("type") != "response_item":
            continue
        if payload.get("type") != "message":
            continue

        role = ROLE_MAP.get(_string_value(payload.get("role")))
        if role is None:
            continue

        text = _extract_message_text(payload.get("content"))
        if text is None:
            continue

        messages.append(
            NormalizedMessage(
                role=role,
                text=text,
                timestamp=_string_field(payload, "timestamp"),
                source_message_id=_string_field(payload, "id"),
            )
        )

    if not saw_session_meta and not messages:
        return None
    return CodexAppRollout(
        session_metadata=session_metadata,
        messages=tuple(messages),
        artifact_path=resolved_path,
    )


def _read_thread_records(state_db_path: Path) -> tuple[CodexThreadRecord, ...]:
    rows = _query_sqlite_rows(
        state_db_path,
        (
            "SELECT id, rollout_path, cwd, title, first_user_message, archived, cli_version "
            "FROM threads"
        ),
        (),
    )
    records: list[CodexThreadRecord] = []
    for row in rows:
        if len(row) != 7:
            continue
        thread_id, rollout_path, cwd, title, first_user_message, archived, cli_version = row
        if not isinstance(thread_id, str):
            continue
        records.append(
            CodexThreadRecord(
                thread_id=thread_id,
                rollout_path=_normalized_path_string(rollout_path),
                cwd=_string_value(cwd),
                title=_string_value(title),
                first_user_message=_string_value(first_user_message),
                archived=_bool_value(archived),
                cli_version=_string_value(cli_version),
            )
        )
    return tuple(records)


def _read_automation_definitions(
    *,
    automation_db_paths: tuple[Path, ...],
    automation_toml_paths: tuple[Path, ...],
) -> dict[str, CodexAutomationDefinition]:
    definitions: dict[str, CodexAutomationDefinition] = {}
    toml_by_id = {
        automation_toml_path.parent.name: automation_toml_path
        for automation_toml_path in automation_toml_paths
    }

    for automation_toml_path in automation_toml_paths:
        automation_id = automation_toml_path.parent.name
        payload = _read_toml_payload(automation_toml_path)
        definitions.setdefault(
            automation_id,
            CodexAutomationDefinition(
                automation_id=automation_id,
                name=_string_value(payload.get("name")),
                prompt=_string_value(payload.get("prompt")),
                status=_string_value(payload.get("status")),
                schedule=_string_value(payload.get("rrule")),
                model=_string_value(payload.get("model")),
                reasoning_effort=_string_value(payload.get("reasoning_effort")),
                definition_path=str(automation_toml_path.resolve(strict=False)),
            ),
        )

    for automation_db_path in automation_db_paths:
        rows = _query_sqlite_rows(
            automation_db_path,
            (
                "SELECT id, name, prompt, status, rrule, model, reasoning_effort "
                "FROM automations"
            ),
            (),
        )
        for row in rows:
            if len(row) != 7:
                continue
            automation_id, name, prompt, status, schedule, model, reasoning_effort = row
            if not isinstance(automation_id, str):
                continue
            existing = definitions.get(automation_id)
            definitions[automation_id] = CodexAutomationDefinition(
                automation_id=automation_id,
                name=_first_non_empty(_string_value(name), None if existing is None else existing.name),
                prompt=_first_non_empty(
                    _string_value(prompt),
                    None if existing is None else existing.prompt,
                ),
                status=_first_non_empty(
                    _string_value(status),
                    None if existing is None else existing.status,
                ),
                schedule=_first_non_empty(
                    _string_value(schedule),
                    None if existing is None else existing.schedule,
                ),
                model=_first_non_empty(
                    _string_value(model),
                    None if existing is None else existing.model,
                ),
                reasoning_effort=_first_non_empty(
                    _string_value(reasoning_effort),
                    None if existing is None else existing.reasoning_effort,
                ),
                definition_path=(
                    None
                    if automation_id not in toml_by_id
                    else str(toml_by_id[automation_id].resolve(strict=False))
                ),
            )
    return definitions


def _read_automation_run_records(
    automation_db_path: Path,
) -> tuple[CodexAutomationRunRecord, ...]:
    rows = _query_sqlite_rows(
        automation_db_path,
        (
            "SELECT thread_id, automation_id, status, thread_title, source_cwd, "
            "inbox_title, inbox_summary, archived_user_message, "
            "archived_assistant_message, archived_reason "
            "FROM automation_runs"
        ),
        (),
    )
    records: list[CodexAutomationRunRecord] = []
    for row in rows:
        if len(row) != 10:
            continue
        (
            thread_id,
            automation_id,
            status,
            thread_title,
            source_cwd,
            inbox_title,
            inbox_summary,
            archived_user_message,
            archived_assistant_message,
            archived_reason,
        ) = row
        if not isinstance(thread_id, str) or not isinstance(automation_id, str):
            continue
        records.append(
            CodexAutomationRunRecord(
                thread_id=thread_id,
                automation_id=automation_id,
                status=_string_value(status),
                thread_title=_string_value(thread_title),
                source_cwd=_string_value(source_cwd),
                inbox_title=_string_value(inbox_title),
                inbox_summary=_string_value(inbox_summary),
                archived_user_message=_string_value(archived_user_message),
                archived_assistant_message=_string_value(archived_assistant_message),
                archived_reason=_string_value(archived_reason),
            )
        )
    return tuple(records)


def _query_sqlite_rows(
    database_path: Path,
    query: str,
    parameters: tuple[object, ...],
) -> list[tuple[object, ...]]:
    try:
        return _query_sqlite_rows_once(database_path, query, parameters)
    except sqlite3.DatabaseError as error:
        if "no such table" in str(error).lower():
            return []

        copied_path = _copy_sqlite_database(database_path)
        if copied_path is None:
            return []
        try:
            return _query_sqlite_rows_once(copied_path, query, parameters)
        except sqlite3.DatabaseError:
            return []
        finally:
            copied_path.unlink(missing_ok=True)


def _query_sqlite_rows_once(
    database_path: Path,
    query: str,
    parameters: tuple[object, ...],
) -> list[tuple[object, ...]]:
    if not database_path.is_file():
        return []

    with sqlite3.connect(str(database_path)) as connection:
        rows = connection.execute(query, parameters).fetchall()
    return [tuple(row) for row in rows]


def _copy_sqlite_database(database_path: Path) -> Path | None:
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f"{database_path.stem}-",
            suffix=database_path.suffix,
            delete=False,
        ) as handle:
            copied_path = Path(handle.name)
        shutil.copyfile(database_path, copied_path)
    except OSError:
        return None
    return copied_path


def _read_toml_payload(path: Path) -> dict[str, object]:
    try:
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, tomllib.TOMLDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def _is_codex_desktop_session(
    session_metadata: CodexSessionMetadata, _rollout_path: Path
) -> bool:
    return session_metadata.originator == "Codex Desktop"


def _discover_paths(
    input_roots: tuple[Path, ...],
    *,
    direct_match,
    glob_pattern: str,
    expect_dir: bool,
) -> tuple[str, ...]:
    seen: set[Path] = set()
    candidates: list[str] = []

    for input_root in input_roots:
        matches: list[Path] = []
        if direct_match(input_root):
            matches.append(input_root)
        if input_root.is_dir():
            matches.extend(input_root.glob(glob_pattern))

        for candidate in matches:
            if expect_dir and not candidate.is_dir():
                continue
            if not expect_dir and not candidate.is_file():
                continue

            resolved = candidate.resolve(strict=False)
            if resolved in seen:
                continue
            seen.add(resolved)
            candidates.append(str(resolved))

    return tuple(sorted(candidates))


def _extract_message_text(content: object) -> str | None:
    if isinstance(content, str):
        text = content.strip()
        return text or None
    if not isinstance(content, list):
        return None

    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            text = item.strip()
            if text:
                parts.append(text)
            continue
        if not isinstance(item, dict):
            continue
        if item.get("type") not in TEXT_ITEM_TYPES:
            continue
        text = _string_field(item, "text")
        if text:
            parts.append(text.strip())

    if not parts:
        return None
    return "\n\n".join(part for part in parts if part)


def _load_json_line(raw_line: str) -> dict[str, object] | None:
    line = raw_line.strip()
    if not line:
        return None
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _resolve_unique_automation_run_index(
    candidates: dict[tuple[str, str], list[CodexAutomationRunRecord]],
) -> dict[tuple[str, str], CodexAutomationRunRecord]:
    resolved: dict[tuple[str, str], CodexAutomationRunRecord] = {}
    for key, matches in candidates.items():
        thread_ids = {match.thread_id for match in matches}
        if len(thread_ids) != 1:
            continue
        resolved[key] = matches[0]
    return resolved


def _has_role(messages: list[NormalizedMessage], role: MessageRole) -> bool:
    return any(message.role == role for message in messages)


def _first_message_text(
    messages: tuple[NormalizedMessage, ...] | list[NormalizedMessage],
    role: MessageRole,
) -> str | None:
    for message in messages:
        if message.role == role and message.text is not None:
            return message.text
    return None


def _user_insertion_index(messages: list[NormalizedMessage]) -> int:
    index = 0
    while index < len(messages) and messages[index].role == MessageRole.DEVELOPER:
        index += 1
    return index


def _unique_limitations(limitations: list[str]) -> list[str]:
    return list(dict.fromkeys(limitations))


def _resolve_automation_title(
    *,
    thread_record: CodexThreadRecord | None,
    automation_run: CodexAutomationRunRecord,
) -> tuple[str | None, str | None]:
    candidates = (
        ("automation_runs.thread_title", automation_run.thread_title),
        ("automation_runs.inbox_title", automation_run.inbox_title),
        (
            "state_5.sqlite threads.title",
            None if thread_record is None else thread_record.title,
        ),
    )
    for source, value in candidates:
        if value is not None:
            return value, source
    return None, None


def _normalized_path_string(value: object) -> str | None:
    raw_value = _string_value(value)
    if raw_value is None:
        return None
    return str(Path(raw_value).expanduser().resolve(strict=False))


def _string_field(payload: dict[str, object], key: str) -> str | None:
    return _string_value(payload.get(key))


def _string_value(value: object) -> str | None:
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return None


def _first_non_empty(*values: str | None) -> str | None:
    for value in values:
        if value is not None and value.strip():
            return value.strip()
    return None


def _bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    return False


def _is_archived_rollout(rollout_path: Path) -> bool:
    return "archived_sessions" in rollout_path.parts


def _is_codex_application_support_root(path: Path) -> bool:
    return path.name == "Codex" and "Application Support" in path.parts


def _is_codex_log_root(path: Path) -> bool:
    return path.name == "com.openai.codex" and "Logs" in path.parts


def _is_codex_preference_path(path: Path) -> bool:
    return path.name == "com.openai.codex.plist" and "Preferences" in path.parts


def _is_codex_cache_root(path: Path) -> bool:
    return path.name == "com.openai.codex" and "Caches" in path.parts


def _is_codex_state_db_path(path: Path) -> bool:
    return path.name == "state_5.sqlite"


def _is_codex_automation_db_path(path: Path) -> bool:
    return path.name == "codex-dev.db" and "sqlite" in path.parts


def _is_codex_automation_toml_path(path: Path) -> bool:
    return path.name == "automation.toml" and "automations" in path.parts


__all__ = [
    "CODEX_APP_DESCRIPTOR",
    "CodexAppCollector",
    "build_codex_app_metadata_index",
    "discover_app_shell_provenance",
    "parse_rollout_file",
]
