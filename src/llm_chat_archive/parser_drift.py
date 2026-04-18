from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from .sources.antigravity_editor_view import (
    discover_antigravity_editor_view_artifacts,
    parse_conversation_blob,
)
from .sources.claude_code_cli import iter_transcript_paths as iter_claude_transcript_paths
from .sources.claude_code_cli import parse_transcript_file as parse_claude_transcript_file
from .sources.claude_code_ide import (
    IDE_COMMAND_MARKER,
    discover_ide_bridge_provenance,
    parse_transcript_file as parse_claude_ide_transcript_file,
)
from .sources.codex_app import parse_rollout_file as parse_codex_app_rollout_file
from .sources.codex_cli import parse_rollout_file as parse_codex_cli_rollout_file
from .sources.codex_ide_extension import (
    parse_rollout_file as parse_codex_ide_extension_rollout_file,
)
from .sources.codex_rollout import iter_rollout_paths, resolve_input_roots
from .sources.cursor_cli import discover_cursor_cli_artifacts, parse_cli_log
from .sources.cursor_editor import (
    _read_state_values as read_cursor_editor_state_values,
)
from .sources.cursor_editor import _select_composer as select_cursor_editor_composer
from .sources.cursor_editor import iter_workspace_state_paths as iter_cursor_workspace_state_paths
from .sources.cursor_editor import parse_workspace_state as parse_cursor_workspace_state
from .sources.gemini_cli import discover_project_sessions
from .sources.gemini_cli import parse_transcript_file as parse_gemini_transcript_file
from .sources.gemini_code_assist_ide import (
    _attribute_indexed_chat_sessions as attribute_gemini_indexed_chat_sessions,
)
from .sources.gemini_code_assist_ide import (
    _read_state_values as read_gemini_state_values,
)
from .sources.gemini_code_assist_ide import WORKSPACE_STATE_KEYS as GEMINI_WORKSPACE_STATE_KEYS
from .sources.gemini_code_assist_ide import (
    discover_gemini_code_assist_ide_artifacts,
    parse_workspace_state_rows as parse_gemini_workspace_state_rows,
)
from .sources.windsurf_editor import (
    build_windsurf_conversations as build_windsurf_editor_conversations,
    discover_windsurf_editor_artifacts,
)

SOURCE_ASSUMPTIONS: dict[str, tuple[str, ...]] = {
    "antigravity_editor_view": (
        "conversation protobuf blobs must retain the confirmed session and message field mapping.",
        "Antigravity shared state markers stay provenance-only and must not replace conversation blobs.",
    ),
    "claude": (
        "Claude transcripts stay newline-delimited JSONL under projects/* with user or assistant rows.",
        "Known Claude message content shapes remain extractable into normalized text messages.",
    ),
    "claude_code_ide": (
        "IDE-attached Claude sessions keep /ide command markers or history.jsonl session IDs.",
        "Known Claude message content shapes remain extractable after IDE wrapper sanitization.",
    ),
    "codex_app": (
        "Codex Desktop sessions keep rollout-*.jsonl files with session_meta originator 'Codex Desktop'.",
        "response_item/message rows continue exposing text content under known Codex rollout item types.",
    ),
    "codex_cli": (
        "Codex rollout-*.jsonl files keep session_meta plus response_item/message rows.",
        "response_item/message content stays under the known Codex text item types.",
    ),
    "codex_ide_extension": (
        "Codex IDE sessions keep rollout-*.jsonl files with session_meta originator 'codex_vscode'.",
        "response_item/message rows continue exposing text content under the known Codex text item types.",
    ),
    "cursor": (
        "Cursor cli.log invocations stay paired with workspace state.vscdb recovery markers.",
        "Known Cursor transcript or prompt attribution signals remain recoverable from workspace state.",
    ),
    "cursor_editor": (
        "Cursor workspace state.vscdb keeps composer.composerData and aiService prompt or generation keys.",
        "Cursor transcript recovery continues to promote the selected composer into normalized messages.",
    ),
    "gemini": (
        "Gemini CLI project temp roots keep session-*.json files with a messages array.",
        "Known Gemini request or response shapes remain extractable into normalized text messages.",
    ),
    "gemini_code_assist_ide": (
        "Workspace state.vscdb keeps chat.ChatSessionStore.index and Gemini chat view keys.",
        "Gemini-owned chatSessions payloads keep provider attribution and recoverable request or response text.",
    ),
    "windsurf_editor": (
        "Windsurf local context remains rooted in ~/.codeium/windsurf plus repo-local .windsurf metadata.",
        "Confirmed memories and rule files stay parseable as partial context rows until a native session-history store is observed.",
    ),
}


class ParserAssumptionStatus(StrEnum):
    CLEAR = "clear"
    DRIFT_SUSPECTED = "drift_suspected"
    UNOBSERVED = "unobserved"


@dataclass(frozen=True, slots=True)
class ParserAssumptionReport:
    source: str
    status: ParserAssumptionStatus
    summary: str
    assumptions: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()

    @property
    def drift_suspected(self) -> bool:
        return self.status == ParserAssumptionStatus.DRIFT_SUSPECTED

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "status": self.status.value,
            "summary": self.summary,
        }
        if self.assumptions:
            payload["assumptions"] = list(self.assumptions)
        if self.evidence:
            payload["evidence"] = list(self.evidence)
        return payload


def inspect_parser_assumptions(
    source: str,
    *,
    input_roots: tuple[Path, ...],
    repo_path: Path | None = None,
) -> ParserAssumptionReport:
    resolved_input_roots = resolve_input_roots(input_roots)
    assumptions = SOURCE_ASSUMPTIONS.get(source, ())

    if source == "codex_cli":
        rollout_paths = tuple(iter_rollout_paths(resolved_input_roots))
        if not rollout_paths:
            return _unobserved(
                source,
                assumptions=assumptions,
                evidence=("no rollout-*.jsonl files were found under the inspected roots.",),
            )
        parseable_count = sum(
            1 for rollout_path in rollout_paths if parse_codex_cli_rollout_file(rollout_path) is not None
        )
        if parseable_count == 0:
            return _drift(
                source,
                assumptions=assumptions,
                evidence=(
                    f"observed {len(rollout_paths)} rollout file(s) but reconstructed 0 transcript conversations.",
                ),
            )
        return _clear(
            source,
            assumptions=assumptions,
            evidence=(
                f"observed {parseable_count} parseable rollout conversation(s) across {len(rollout_paths)} rollout file(s).",
            ),
        )

    if source in {"codex_app", "codex_ide_extension"}:
        originator = "Codex Desktop" if source == "codex_app" else "codex_vscode"
        parse_fn = (
            parse_codex_app_rollout_file
            if source == "codex_app"
            else parse_codex_ide_extension_rollout_file
        )
        rollout_paths = tuple(iter_rollout_paths(resolved_input_roots))
        relevant_paths = tuple(
            rollout_path
            for rollout_path in rollout_paths
            if _codex_rollout_originator(rollout_path) == originator
        )
        if not relevant_paths:
            return _unobserved(
                source,
                assumptions=assumptions,
                evidence=(
                    f"no rollout file advertised the expected originator marker {originator!r}.",
                ),
            )
        parseable_count = sum(
            1 for rollout_path in relevant_paths if parse_fn(rollout_path) is not None
        )
        if parseable_count == 0:
            return _drift(
                source,
                assumptions=assumptions,
                evidence=(
                    f"observed {len(relevant_paths)} rollout file(s) with originator {originator!r} but reconstructed 0 transcript conversations.",
                ),
            )
        return _clear(
            source,
            assumptions=assumptions,
            evidence=(
                f"observed {parseable_count} parseable rollout conversation(s) across {len(relevant_paths)} originator-matched rollout file(s).",
            ),
        )

    if source == "claude":
        transcript_paths = tuple(iter_claude_transcript_paths(resolved_input_roots))
        if not transcript_paths:
            return _unobserved(
                source,
                assumptions=assumptions,
                evidence=("no Claude transcript JSONL files were found under the inspected roots.",),
            )
        parseable_count = sum(
            1
            for transcript_path in transcript_paths
            if parse_claude_transcript_file(transcript_path) is not None
        )
        if parseable_count == 0:
            return _drift(
                source,
                assumptions=assumptions,
                evidence=(
                    f"observed {len(transcript_paths)} Claude transcript candidate file(s) but reconstructed 0 transcript conversations.",
                ),
            )
        return _clear(
            source,
            assumptions=assumptions,
            evidence=(
                f"observed {parseable_count} parseable Claude transcript conversation(s) across {len(transcript_paths)} candidate file(s).",
            ),
        )

    if source == "claude_code_ide":
        transcript_paths = tuple(iter_claude_transcript_paths(resolved_input_roots))
        discovery = discover_ide_bridge_provenance(resolved_input_roots)
        relevant_paths = tuple(
            transcript_path
            for transcript_path in transcript_paths
            if _is_claude_ide_candidate(
                transcript_path,
                history_session_ids=discovery.history_session_ids,
            )
        )
        if not relevant_paths:
            return _unobserved(
                source,
                assumptions=assumptions,
                evidence=(
                    "no Claude transcript candidate matched IDE command markers or IDE history session IDs.",
                ),
            )
        parseable_count = sum(
            1
            for transcript_path in relevant_paths
            if parse_claude_ide_transcript_file(
                transcript_path,
                discovery=discovery,
            )
            is not None
        )
        if parseable_count == 0:
            return _drift(
                source,
                assumptions=assumptions,
                evidence=(
                    f"observed {len(relevant_paths)} IDE-marked Claude transcript candidate(s) but reconstructed 0 IDE transcript conversations.",
                ),
            )
        return _clear(
            source,
            assumptions=assumptions,
            evidence=(
                f"observed {parseable_count} parseable Claude IDE conversation(s) across {len(relevant_paths)} IDE-marked transcript candidate(s).",
            ),
        )

    if source == "gemini":
        resolved_repo_path = (repo_path or Path.cwd()).expanduser().resolve(strict=False)
        discovery = discover_project_sessions(
            resolved_repo_path,
            resolved_input_roots,
        )
        if not discovery.session_paths:
            detail = (
                f"Gemini project discovery found no session files"
                if discovery.negative_reason is None
                else f"Gemini project discovery reported {discovery.negative_reason}"
            )
            return _unobserved(
                source,
                assumptions=assumptions,
                evidence=(detail,),
            )
        parseable_count = sum(
            1
            for transcript_path in discovery.session_paths
            if parse_gemini_transcript_file(
                transcript_path,
                repo_path=discovery.repo_path,
            )
            is not None
        )
        if parseable_count == 0:
            return _drift(
                source,
                assumptions=assumptions,
                evidence=(
                    f"observed {len(discovery.session_paths)} Gemini session file(s) but reconstructed 0 transcript conversations.",
                ),
            )
        return _clear(
            source,
            assumptions=assumptions,
            evidence=(
                f"observed {parseable_count} parseable Gemini session conversation(s) across {len(discovery.session_paths)} session file(s).",
            ),
        )

    if source == "gemini_code_assist_ide":
        artifacts = discover_gemini_code_assist_ide_artifacts(resolved_input_roots)
        if not artifacts.workspace_state_paths:
            return _unobserved(
                source,
                assumptions=assumptions,
                evidence=("no Gemini IDE workspace state.vscdb files were found under the inspected roots.",),
            )
        gemini_session_count = 0
        recovered_message_count = 0
        for raw_path in artifacts.workspace_state_paths:
            workspace_state_path = Path(raw_path)
            state_values = read_gemini_state_values(
                workspace_state_path,
                GEMINI_WORKSPACE_STATE_KEYS,
            )
            attributions = attribute_gemini_indexed_chat_sessions(
                workspace_state_path.parent,
                state_values.get("chat.ChatSessionStore.index"),
            )
            gemini_session_count += sum(
                1 for attribution in attributions if attribution.ownership == "gemini"
            )
            recovered_message_count += sum(
                len(conversation.messages)
                for conversation in parse_gemini_workspace_state_rows(
                    workspace_state_path,
                    artifacts=artifacts,
                )
            )
        if gemini_session_count == 0:
            return _unobserved(
                source,
                assumptions=assumptions,
                evidence=(
                    "no Gemini-owned indexed chatSessions were observed in the inspected workspace state.",
                ),
            )
        if recovered_message_count == 0:
            return _drift(
                source,
                assumptions=assumptions,
                evidence=(
                    f"observed {gemini_session_count} Gemini-owned indexed chatSession marker(s) but recovered 0 transcript messages.",
                ),
            )
        return _clear(
            source,
            assumptions=assumptions,
            evidence=(
                f"observed {gemini_session_count} Gemini-owned indexed chatSession marker(s) and recovered {recovered_message_count} transcript message(s).",
            ),
        )

    if source == "cursor_editor":
        workspace_state_paths = tuple(iter_cursor_workspace_state_paths(resolved_input_roots))
        if not workspace_state_paths:
            return _unobserved(
                source,
                assumptions=assumptions,
                evidence=("no Cursor workspace state.vscdb files were found under the inspected roots.",),
            )
        relevant_workspace_count = 0
        parseable_workspace_count = 0
        for workspace_state_path in workspace_state_paths:
            state_values = read_cursor_editor_state_values(
                workspace_state_path,
                (
                    "composer.composerData",
                    "aiService.prompts",
                    "aiService.generations",
                ),
            )
            composer_payload = state_values.get("composer.composerData")
            if not isinstance(composer_payload, dict):
                continue
            if select_cursor_editor_composer(composer_payload) is None:
                continue
            relevant_workspace_count += 1
            if parse_cursor_workspace_state(workspace_state_path) is not None:
                parseable_workspace_count += 1
        if relevant_workspace_count == 0:
            return _unobserved(
                source,
                assumptions=assumptions,
                evidence=(
                    "no selected Cursor composer session markers were observed in the inspected workspace state.",
                ),
            )
        if parseable_workspace_count == 0:
            return _drift(
                source,
                assumptions=assumptions,
                evidence=(
                    f"observed {relevant_workspace_count} Cursor composer session marker(s) but reconstructed 0 editor transcript conversations.",
                ),
            )
        return _clear(
            source,
            assumptions=assumptions,
            evidence=(
                f"observed {parseable_workspace_count} parseable Cursor editor conversation(s) across {relevant_workspace_count} selected composer session marker(s).",
            ),
        )

    if source == "cursor":
        artifacts = discover_cursor_cli_artifacts(resolved_input_roots)
        if not artifacts.cli_log_paths:
            return _unobserved(
                source,
                assumptions=assumptions,
                evidence=("no Cursor cli.log artifacts were found under the inspected roots.",),
            )
        relevant_log_count = 0
        recovery_signal_count = 0
        for raw_path in artifacts.cli_log_paths:
            conversation = parse_cli_log(Path(raw_path), artifacts=artifacts)
            session_metadata = conversation.session_metadata or {}
            invocation = session_metadata.get("invocation")
            if not isinstance(invocation, dict):
                continue
            if (
                invocation.get("status") is True
                or invocation.get("list_extensions") is True
                or invocation.get("show_versions") is True
            ):
                continue
            relevant_log_count += 1
            if (
                conversation.messages
                or "transcript_attribution" in session_metadata
                or "workspace_prompt_evidence" in session_metadata
            ):
                recovery_signal_count += 1
        if relevant_log_count == 0:
            return _unobserved(
                source,
                assumptions=assumptions,
                evidence=(
                    "no Cursor CLI agent-style invocation logs were observed after filtering status or version-only invocations.",
                ),
            )
        if recovery_signal_count == 0 and (
            artifacts.workspace_sessions or artifacts.workspace_state_paths
        ):
            return _drift(
                source,
                assumptions=assumptions,
                evidence=(
                    f"observed {relevant_log_count} Cursor CLI invocation log(s) plus workspace state artifacts but found no transcript or prompt recovery signals.",
                ),
            )
        if recovery_signal_count == 0:
            return _unobserved(
                source,
                assumptions=assumptions,
                evidence=(
                    f"observed {relevant_log_count} Cursor CLI invocation log(s) but no transcript-bearing workspace state markers were available.",
                ),
            )
        return _clear(
            source,
            assumptions=assumptions,
            evidence=(
                f"observed {recovery_signal_count} Cursor CLI invocation(s) with transcript or prompt recovery evidence across {relevant_log_count} relevant log(s).",
            ),
        )

    if source == "antigravity_editor_view":
        artifacts = discover_antigravity_editor_view_artifacts(resolved_input_roots)
        if not artifacts.conversation_paths:
            return _unobserved(
                source,
                assumptions=assumptions,
                evidence=("no Antigravity conversation protobuf blobs were found under the inspected roots.",),
            )
        recovered_conversation_count = 0
        decode_status_counts: Counter[str] = Counter()
        limitation_counts: Counter[str] = Counter()
        for raw_path in artifacts.conversation_paths:
            conversation = parse_conversation_blob(
                Path(raw_path),
                artifacts=artifacts,
            )
            if conversation is None:
                continue
            if conversation.messages:
                recovered_conversation_count += 1
            conversation_blob = (
                conversation.session_metadata.get("conversation_blob")
                if isinstance(conversation.session_metadata, dict)
                else None
            )
            if isinstance(conversation_blob, dict):
                decode_status = conversation_blob.get("decode_status")
                if isinstance(decode_status, str) and decode_status:
                    decode_status_counts[decode_status] += 1
            for limitation in conversation.limitations:
                limitation_counts[limitation] += 1
        if recovered_conversation_count == 0:
            evidence = [
                (
                    f"observed {len(artifacts.conversation_paths)} Antigravity "
                    "conversation blob(s) but reconstructed 0 transcript conversations."
                )
            ]
            decode_status_summary = _format_counter_summary(decode_status_counts)
            if decode_status_summary is not None:
                evidence.append(f"decode statuses: {decode_status_summary}.")
            limitation_summary = _format_counter_summary(limitation_counts)
            if limitation_summary is not None:
                evidence.append(f"explicit limitations: {limitation_summary}.")
            return _drift(
                source,
                assumptions=assumptions,
                evidence=tuple(evidence),
            )
        return _clear(
            source,
            assumptions=assumptions,
            evidence=(
                f"observed {recovered_conversation_count} parseable Antigravity conversation blob(s) across {len(artifacts.conversation_paths)} candidate blob(s).",
            ),
        )

    if source == "windsurf_editor":
        artifacts = discover_windsurf_editor_artifacts(resolved_input_roots)
        conversations = build_windsurf_editor_conversations(
            artifacts,
            collected_at="1970-01-01T00:00:00Z",
        )
        if not conversations:
            return _unobserved(
                source,
                assumptions=assumptions,
                evidence=(
                    "no Windsurf memories, rule files, or metadata-bearing roots were found under the inspected roots.",
                ),
            )
        partial_rows = sum(
            1
            for conversation in conversations
            if conversation.transcript_completeness == "partial"
        )
        unsupported_rows = sum(
            1
            for conversation in conversations
            if conversation.transcript_completeness == "unsupported"
        )
        return _clear(
            source,
            assumptions=assumptions,
            evidence=(
                f"observed {len(conversations)} Windsurf row(s), including {partial_rows} partial context row(s) and {unsupported_rows} metadata-only row(s).",
            ),
        )

    return _unobserved(
        source,
        assumptions=assumptions,
        evidence=("no parser assumption inspector is registered for this source.",),
    )


def _clear(
    source: str,
    *,
    assumptions: tuple[str, ...],
    evidence: tuple[str, ...],
) -> ParserAssumptionReport:
    return ParserAssumptionReport(
        source=source,
        status=ParserAssumptionStatus.CLEAR,
        summary="parser assumptions matched the observed transcript-bearing artifacts",
        assumptions=assumptions,
        evidence=evidence,
    )


def _drift(
    source: str,
    *,
    assumptions: tuple[str, ...],
    evidence: tuple[str, ...],
) -> ParserAssumptionReport:
    return ParserAssumptionReport(
        source=source,
        status=ParserAssumptionStatus.DRIFT_SUSPECTED,
        summary="drift suspected: transcript-bearing artifacts were observed but parser assumptions no longer reconstructed transcript messages",
        assumptions=assumptions,
        evidence=evidence,
    )


def _unobserved(
    source: str,
    *,
    assumptions: tuple[str, ...],
    evidence: tuple[str, ...],
) -> ParserAssumptionReport:
    return ParserAssumptionReport(
        source=source,
        status=ParserAssumptionStatus.UNOBSERVED,
        summary="no source-specific transcript-bearing parser assumptions were observed",
        assumptions=assumptions,
        evidence=evidence,
    )


def _codex_rollout_originator(rollout_path: Path) -> str | None:
    try:
        lines = rollout_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None

    for raw_line in lines:
        payload = _load_json_object_line(raw_line)
        if payload is None or payload.get("type") != "session_meta":
            continue
        session_payload = payload.get("payload")
        if isinstance(session_payload, dict):
            return _string_value(session_payload.get("originator"))
    return None


def _is_claude_ide_candidate(
    transcript_path: Path,
    *,
    history_session_ids: frozenset[str],
) -> bool:
    session_lookup_id = (
        transcript_path.parents[1].name if transcript_path.parent.name == "subagents" else transcript_path.stem
    )
    if session_lookup_id in history_session_ids:
        return True

    try:
        lines = transcript_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return False
    return any(IDE_COMMAND_MARKER in line for line in lines)


def _load_json_object_line(raw_line: str) -> dict[str, object] | None:
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


def _string_value(value: object) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _format_counter_summary(counter: Counter[str]) -> str | None:
    if not counter:
        return None
    return ", ".join(
        f"{key}={count}"
        for key, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    )


__all__ = [
    "ParserAssumptionReport",
    "ParserAssumptionStatus",
    "inspect_parser_assumptions",
]
