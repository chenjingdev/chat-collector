from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from ..incremental import write_incremental_collection
from ..models import (
    AppShellProvenance,
    CollectionPlan,
    CollectionResult,
    ConversationProvenance,
    MessageRole,
    NormalizedConversation,
    NormalizedMessage,
    SourceDescriptor,
    SourceSupportMetadata,
    SupportLevel,
    TranscriptCompleteness,
)
from ..source_roots import (
    darwin_root,
    default_descriptor_input_roots,
    linux_root,
    windows_root,
)
from .codex_rollout import resolve_input_roots, utc_timestamp

MODULE_REPO_ROOT = Path(__file__).resolve().parents[3]

WINDSURF_EDITOR_DESCRIPTOR = SourceDescriptor(
    key="windsurf_editor",
    display_name="Windsurf Editor",
    execution_context="ide_native",
    support_level=SupportLevel.PARTIAL,
    default_input_roots=(
        "~/.codeium/windsurf",
        "~/Library/Application Support/Windsurf",
        "/Library/Application Support/Windsurf",
    ),
    artifact_root_candidates=(
        darwin_root("$HOME/.codeium/windsurf"),
        linux_root("$HOME/.codeium/windsurf"),
        windows_root("$APPDATA/Codeium/windsurf"),
        darwin_root("$HOME/Library/Application Support/Windsurf"),
        darwin_root("/Library/Application Support/Windsurf"),
        linux_root("/etc/windsurf"),
        windows_root("$PROGRAMDATA/Windsurf"),
    ),
    notes=(
        "Collects auto-generated memories from ~/.codeium/windsurf/memories when text-bearing files are present.",
        "Collects global, workspace, and system Windsurf rules as partial context rows instead of fabricating turn-by-turn chat transcripts.",
        "Treats mcp_config.json, skills directories, and bare .windsurf roots as provenance or unsupported metadata when no message-bearing artifact is available.",
        "Also inspects the current repository's .windsurf root when no explicit input roots are provided.",
    ),
    support_metadata=SourceSupportMetadata(
        product_label="Windsurf",
        host_surface="Editor",
        expected_transcript_completeness=TranscriptCompleteness.PARTIAL,
        limitation_summary=(
            "Windsurf local memories and rules can be normalized, but no confirmed "
            "native editor session-history store is available yet, so the "
            "collector remains partial and opt-in."
        ),
        limitations=(
            "Memories and rules are captured as partial context rows because they are not original turn-by-turn Cascade transcripts.",
            "mcp_config.json, skills directories, and bare workspace metadata degrade to unsupported rows until a confirmed session-history store is observed.",
        ),
    ),
)

GLOBAL_ROOT_GLOB = "**/.codeium/windsurf"
APPLICATION_SUPPORT_GLOB = "**/Library/Application Support/Windsurf"
GLOBAL_MEMORY_GLOBS = (
    "memories/**/*",
    "**/.codeium/windsurf/memories/**/*",
)
GLOBAL_RULE_GLOBS = (
    "memories/global_rules.md",
    "global_rules.md",
    "**/.codeium/windsurf/memories/global_rules.md",
    "**/.codeium/windsurf/global_rules.md",
)
GLOBAL_SKILLS_GLOBS = (
    "skills",
    "**/.codeium/windsurf/skills",
)
WORKSPACE_ROOT_GLOB = "**/.windsurf"
WORKSPACE_RULE_GLOBS = (
    "rules/*.md",
    "**/.windsurf/rules/*.md",
)
WORKSPACE_SKILLS_GLOBS = (
    "skills",
    "**/.windsurf/skills",
)
SYSTEM_RULE_GLOBS = (
    "rules/*.md",
    "**/Library/Application Support/Windsurf/rules/*.md",
    "**/etc/windsurf/rules/*.md",
    "**/ProgramData/Windsurf/rules/*.md",
)
TEXT_MEMORY_SUFFIXES = frozenset({".md", ".markdown", ".txt"})
JSON_MEMORY_SUFFIXES = frozenset({".json", ".jsonl"})
MEMORY_LIMITATIONS = (
    "memory_entry_not_original_conversation_transcript",
    "no_confirmed_windsurf_editor_session_history",
)
RULE_LIMITATIONS = (
    "rule_file_not_original_conversation_transcript",
    "no_confirmed_windsurf_editor_session_history",
)
GLOBAL_METADATA_LIMITATIONS = (
    "global_metadata_only",
    "no_confirmed_windsurf_editor_session_history",
)
WORKSPACE_METADATA_LIMITATIONS = (
    "workspace_metadata_only",
    "no_confirmed_windsurf_editor_session_history",
)
RULE_FRONTMATTER_DELIMITER = "---"
RULE_FRONTMATTER_LIST_KEYS = frozenset({"globs"})
TEXT_VALUE_KEYS = (
    "title",
    "summary",
    "memory",
    "content",
    "text",
    "body",
    "message",
    "description",
    "value",
)


@dataclass(frozen=True, slots=True)
class WindsurfArtifacts:
    global_root_paths: tuple[str, ...] = ()
    application_support_roots: tuple[str, ...] = ()
    global_memory_paths: tuple[str, ...] = ()
    global_rule_paths: tuple[str, ...] = ()
    system_rule_paths: tuple[str, ...] = ()
    workspace_rule_paths: tuple[str, ...] = ()
    workspace_root_paths: tuple[str, ...] = ()
    global_skill_root_paths: tuple[str, ...] = ()
    workspace_skill_root_paths: tuple[str, ...] = ()
    mcp_config_paths: tuple[str, ...] = ()
    mcp_server_names: tuple[str, ...] = ()

    def build_app_shell(self, *, extra_paths: Iterable[str] = ()) -> AppShellProvenance | None:
        provenance = AppShellProvenance(
            application_support_roots=self.application_support_roots,
            auxiliary_paths=tuple(
                sorted(
                    {
                        *self.global_root_paths,
                        *self.global_skill_root_paths,
                        *self.workspace_skill_root_paths,
                        *self.mcp_config_paths,
                        *self.system_rule_paths,
                        *tuple(extra_paths),
                    }
                )
            ),
        )
        if not provenance.to_dict():
            return None
        return provenance


@dataclass(frozen=True, slots=True)
class WindsurfRuleDocument:
    frontmatter: dict[str, object]
    body: str


@dataclass(frozen=True, slots=True)
class WindsurfEditorCollector:
    descriptor: SourceDescriptor = WINDSURF_EDITOR_DESCRIPTOR
    repo_path: Path = MODULE_REPO_ROOT

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
        artifacts = discover_windsurf_editor_artifacts(resolved_input_roots)
        collected_at = utc_timestamp()
        conversations = build_windsurf_conversations(
            artifacts,
            collected_at=collected_at,
        )
        return write_incremental_collection(
            source=self.descriptor.key,
            archive_root=archive_root,
            input_roots=resolved_input_roots,
            scanned_artifact_count=len(conversations),
            collected_at=collected_at,
            conversations=conversations,
        )

    def _default_input_roots(self) -> tuple[Path, ...]:
        roots = list(default_descriptor_input_roots(self.descriptor))
        workspace_root = (self.repo_path / ".windsurf").expanduser().resolve(strict=False)
        if workspace_root not in roots:
            roots.append(workspace_root)
        return tuple(roots)


def discover_windsurf_editor_artifacts(
    input_roots: tuple[Path, ...] | None,
) -> WindsurfArtifacts:
    if not input_roots:
        return WindsurfArtifacts()

    global_root_paths = _discover_paths(
        input_roots,
        direct_match=_is_global_root,
        glob_patterns=(GLOBAL_ROOT_GLOB,),
        expect_dir=True,
    )
    application_support_roots = _discover_paths(
        input_roots,
        direct_match=_is_application_support_root,
        glob_patterns=(APPLICATION_SUPPORT_GLOB,),
        expect_dir=True,
    )
    global_rule_paths = _discover_paths(
        input_roots,
        direct_match=_is_global_rule_file,
        glob_patterns=GLOBAL_RULE_GLOBS,
        expect_dir=False,
    )
    global_memory_paths = tuple(
        path
        for path in _discover_paths(
            input_roots,
            direct_match=_is_memory_file,
            glob_patterns=GLOBAL_MEMORY_GLOBS,
            expect_dir=False,
        )
        if Path(path).name != "global_rules.md"
    )
    system_rule_paths = _discover_paths(
        input_roots,
        direct_match=_is_system_rule_file,
        glob_patterns=SYSTEM_RULE_GLOBS,
        expect_dir=False,
    )
    workspace_rule_paths = _discover_paths(
        input_roots,
        direct_match=_is_workspace_rule_file,
        glob_patterns=WORKSPACE_RULE_GLOBS,
        expect_dir=False,
    )
    workspace_root_paths = _discover_workspace_root_paths(
        input_roots,
        workspace_rule_paths=workspace_rule_paths,
    )
    global_skill_root_paths = _discover_paths(
        input_roots,
        direct_match=_is_global_skills_root,
        glob_patterns=GLOBAL_SKILLS_GLOBS,
        expect_dir=True,
    )
    workspace_skill_root_paths = _discover_paths(
        input_roots,
        direct_match=_is_workspace_skills_root,
        glob_patterns=WORKSPACE_SKILLS_GLOBS,
        expect_dir=True,
    )
    mcp_config_paths = _discover_paths(
        input_roots,
        direct_match=_is_mcp_config,
        glob_patterns=("mcp_config.json", "**/.codeium/windsurf/mcp_config.json"),
        expect_dir=False,
    )

    return WindsurfArtifacts(
        global_root_paths=global_root_paths,
        application_support_roots=application_support_roots,
        global_memory_paths=global_memory_paths,
        global_rule_paths=global_rule_paths,
        system_rule_paths=system_rule_paths,
        workspace_rule_paths=workspace_rule_paths,
        workspace_root_paths=workspace_root_paths,
        global_skill_root_paths=global_skill_root_paths,
        workspace_skill_root_paths=workspace_skill_root_paths,
        mcp_config_paths=mcp_config_paths,
        mcp_server_names=_load_mcp_server_names(mcp_config_paths),
    )


def build_windsurf_conversations(
    artifacts: WindsurfArtifacts,
    *,
    collected_at: str | None = None,
) -> tuple[NormalizedConversation, ...]:
    resolved_collected_at = collected_at or utc_timestamp()
    conversations: list[NormalizedConversation] = []

    for raw_path in artifacts.global_memory_paths:
        conversation = parse_memory_file(
            Path(raw_path),
            collected_at=resolved_collected_at,
            artifacts=artifacts,
        )
        if conversation is not None:
            conversations.append(conversation)

    for raw_path in artifacts.global_rule_paths:
        conversation = parse_rule_file(
            Path(raw_path),
            collected_at=resolved_collected_at,
            artifacts=artifacts,
            scope="global",
        )
        if conversation is not None:
            conversations.append(conversation)

    for raw_path in artifacts.workspace_rule_paths:
        conversation = parse_rule_file(
            Path(raw_path),
            collected_at=resolved_collected_at,
            artifacts=artifacts,
            scope="workspace",
        )
        if conversation is not None:
            conversations.append(conversation)

    for raw_path in artifacts.system_rule_paths:
        conversation = parse_rule_file(
            Path(raw_path),
            collected_at=resolved_collected_at,
            artifacts=artifacts,
            scope="system",
        )
        if conversation is not None:
            conversations.append(conversation)

    has_global_message_row = any(
        _is_global_message_row(conversation)
        for conversation in conversations
    )
    if (
        (artifacts.global_root_paths or artifacts.global_skill_root_paths or artifacts.mcp_config_paths)
        and not has_global_message_row
    ):
        global_metadata_row = build_global_metadata_row(
            artifacts,
            collected_at=resolved_collected_at,
        )
        if global_metadata_row is not None:
            conversations.append(global_metadata_row)

    workspace_roots_with_rows = {
        _workspace_root_from_row(conversation)
        for conversation in conversations
        if _workspace_root_from_row(conversation) is not None
    }
    for raw_workspace_root in artifacts.workspace_root_paths:
        if raw_workspace_root in workspace_roots_with_rows:
            continue
        workspace_metadata_row = build_workspace_metadata_row(
            Path(raw_workspace_root),
            artifacts=artifacts,
            collected_at=resolved_collected_at,
        )
        if workspace_metadata_row is not None:
            conversations.append(workspace_metadata_row)

    return tuple(sorted(conversations, key=_conversation_sort_key))


def parse_memory_file(
    memory_path: Path,
    *,
    collected_at: str | None = None,
    artifacts: WindsurfArtifacts | None = None,
) -> NormalizedConversation | None:
    resolved_path = memory_path.expanduser().resolve(strict=False)
    text, content_format = _read_memory_text(resolved_path)
    if text is None:
        return None

    relative_memory_path = _memory_relative_path(resolved_path)
    memory_parts = relative_memory_path.parts
    workspace_key = memory_parts[0] if len(memory_parts) > 1 else None

    session_metadata: dict[str, object] = {
        "scope": "memory",
        "memory_scope": "workspace" if workspace_key is not None else "global",
        "memory_key": relative_memory_path.with_suffix("").as_posix(),
        "relative_path": relative_memory_path.as_posix(),
        "content_format": content_format,
    }
    if workspace_key is not None:
        session_metadata["workspace_key"] = workspace_key
    if artifacts is not None:
        session_metadata["mcp_server_count"] = len(artifacts.mcp_server_names)
        if artifacts.mcp_server_names:
            session_metadata["mcp_server_names"] = list(artifacts.mcp_server_names)
        session_metadata["global_skill_file_count"] = _skill_file_count(
            artifacts.global_skill_root_paths
        )

    return NormalizedConversation(
        source=WINDSURF_EDITOR_DESCRIPTOR.key,
        execution_context=WINDSURF_EDITOR_DESCRIPTOR.execution_context,
        collected_at=collected_at or utc_timestamp(),
        messages=(
            NormalizedMessage(
                role=MessageRole.SYSTEM,
                text=text,
            ),
        ),
        transcript_completeness=TranscriptCompleteness.PARTIAL,
        limitations=MEMORY_LIMITATIONS,
        source_session_id=f"memory:{relative_memory_path.with_suffix('').as_posix()}",
        source_artifact_path=str(resolved_path),
        session_metadata=session_metadata,
        provenance=ConversationProvenance(
            source="windsurf",
            originator="windsurf_editor",
            app_shell=(
                None
                if artifacts is None
                else artifacts.build_app_shell(extra_paths=(str(resolved_path),))
            ),
        ),
    )


def parse_rule_file(
    rule_path: Path,
    *,
    scope: str,
    collected_at: str | None = None,
    artifacts: WindsurfArtifacts | None = None,
) -> NormalizedConversation | None:
    resolved_path = rule_path.expanduser().resolve(strict=False)
    text = _read_text_file(resolved_path)
    if text is None:
        return None

    document = _parse_rule_document(text)
    rule_text = document.body.strip()
    if not rule_text:
        description = document.frontmatter.get("description")
        if isinstance(description, str):
            rule_text = description.strip()
    if not rule_text:
        return None

    session_metadata: dict[str, object] = {
        "scope": "rule",
        "rule_scope": scope,
        "rule_name": resolved_path.stem,
    }
    activation_mode = _string_value(document.frontmatter.get("trigger"))
    if scope == "workspace":
        workspace_root = _workspace_root_for_rule(resolved_path)
        session_metadata["workspace_root"] = str(workspace_root)
        session_metadata["workspace_label"] = workspace_root.name
        session_metadata["relative_path"] = (
            resolved_path.relative_to(workspace_root).as_posix()
        )
        session_metadata["activation_mode"] = activation_mode or "always_on"
        if artifacts is not None:
            session_metadata["workspace_skill_dir_count"] = _workspace_skill_dir_count(
                workspace_root,
                artifacts.workspace_skill_root_paths,
            )
            session_metadata["workspace_skill_file_count"] = _workspace_skill_file_count(
                workspace_root,
                artifacts.workspace_skill_root_paths,
            )
    elif scope == "global":
        session_metadata["relative_path"] = _global_rule_relative_path(resolved_path).as_posix()
        session_metadata["activation_mode"] = "always_on"
        if artifacts is not None:
            session_metadata["mcp_server_count"] = len(artifacts.mcp_server_names)
            if artifacts.mcp_server_names:
                session_metadata["mcp_server_names"] = list(artifacts.mcp_server_names)
            session_metadata["global_skill_file_count"] = _skill_file_count(
                artifacts.global_skill_root_paths
            )
    else:
        session_metadata["relative_path"] = _system_rule_relative_path(resolved_path).as_posix()
        session_metadata["activation_mode"] = activation_mode or "always_on"

    if activation_mode is not None and scope != "workspace":
        session_metadata["frontmatter_trigger"] = activation_mode
    if document.frontmatter:
        session_metadata["frontmatter"] = _json_ready_frontmatter(document.frontmatter)

    workspace_root_string = (
        str(_workspace_root_for_rule(resolved_path)) if scope == "workspace" else None
    )
    role = MessageRole.SYSTEM if scope == "system" else MessageRole.DEVELOPER
    scope_prefix = "workspace" if scope == "workspace" else scope
    session_id = _rule_session_id(
        scope=scope_prefix,
        rule_path=resolved_path,
    )

    return NormalizedConversation(
        source=WINDSURF_EDITOR_DESCRIPTOR.key,
        execution_context=WINDSURF_EDITOR_DESCRIPTOR.execution_context,
        collected_at=collected_at or utc_timestamp(),
        messages=(
            NormalizedMessage(
                role=role,
                text=rule_text,
            ),
        ),
        transcript_completeness=TranscriptCompleteness.PARTIAL,
        limitations=RULE_LIMITATIONS,
        source_session_id=session_id,
        source_artifact_path=str(resolved_path),
        session_metadata=session_metadata,
        provenance=ConversationProvenance(
            source="windsurf",
            originator="windsurf_editor",
            cwd=workspace_root_string,
            app_shell=(
                None
                if artifacts is None
                else artifacts.build_app_shell(extra_paths=(str(resolved_path),))
            ),
        ),
    )


def build_global_metadata_row(
    artifacts: WindsurfArtifacts,
    *,
    collected_at: str | None = None,
) -> NormalizedConversation | None:
    primary_artifact_path = _first_path(
        artifacts.mcp_config_paths
        or artifacts.global_root_paths
        or artifacts.global_skill_root_paths
    )
    if primary_artifact_path is None:
        return None

    return NormalizedConversation(
        source=WINDSURF_EDITOR_DESCRIPTOR.key,
        execution_context=WINDSURF_EDITOR_DESCRIPTOR.execution_context,
        collected_at=collected_at or utc_timestamp(),
        messages=(),
        transcript_completeness=TranscriptCompleteness.UNSUPPORTED,
        limitations=GLOBAL_METADATA_LIMITATIONS,
        source_session_id="global:metadata",
        source_artifact_path=primary_artifact_path,
        session_metadata={
            "scope": "global_metadata",
            "mcp_server_count": len(artifacts.mcp_server_names),
            "mcp_server_names": list(artifacts.mcp_server_names),
            "global_skill_dir_count": _skill_dir_count(artifacts.global_skill_root_paths),
            "global_skill_file_count": _skill_file_count(artifacts.global_skill_root_paths),
            "global_memory_file_count": len(artifacts.global_memory_paths),
            "global_rule_count": len(artifacts.global_rule_paths),
        },
        provenance=ConversationProvenance(
            source="windsurf",
            originator="windsurf_editor",
            app_shell=artifacts.build_app_shell(extra_paths=(primary_artifact_path,)),
        ),
    )


def build_workspace_metadata_row(
    workspace_root: Path,
    *,
    artifacts: WindsurfArtifacts,
    collected_at: str | None = None,
) -> NormalizedConversation | None:
    resolved_workspace_root = workspace_root.expanduser().resolve(strict=False)
    windsurf_root = resolved_workspace_root / ".windsurf"
    primary_artifact_path = str(
        windsurf_root.resolve(strict=False)
        if windsurf_root.exists()
        else resolved_workspace_root.resolve(strict=False)
    )
    return NormalizedConversation(
        source=WINDSURF_EDITOR_DESCRIPTOR.key,
        execution_context=WINDSURF_EDITOR_DESCRIPTOR.execution_context,
        collected_at=collected_at or utc_timestamp(),
        messages=(),
        transcript_completeness=TranscriptCompleteness.UNSUPPORTED,
        limitations=WORKSPACE_METADATA_LIMITATIONS,
        source_session_id=f"workspace:{resolved_workspace_root.name}:metadata",
        source_artifact_path=primary_artifact_path,
        session_metadata={
            "scope": "workspace_metadata",
            "workspace_root": str(resolved_workspace_root),
            "workspace_label": resolved_workspace_root.name,
            "workspace_rule_count": _workspace_rule_count(
                resolved_workspace_root,
                artifacts.workspace_rule_paths,
            ),
            "workspace_skill_dir_count": _workspace_skill_dir_count(
                resolved_workspace_root,
                artifacts.workspace_skill_root_paths,
            ),
            "workspace_skill_file_count": _workspace_skill_file_count(
                resolved_workspace_root,
                artifacts.workspace_skill_root_paths,
            ),
        },
        provenance=ConversationProvenance(
            source="windsurf",
            originator="windsurf_editor",
            cwd=str(resolved_workspace_root),
            app_shell=artifacts.build_app_shell(extra_paths=(primary_artifact_path,)),
        ),
    )


def _discover_workspace_root_paths(
    input_roots: tuple[Path, ...],
    *,
    workspace_rule_paths: tuple[str, ...],
) -> tuple[str, ...]:
    discovered = set(
        str(_workspace_root_for_rule(Path(raw_path)))
        for raw_path in workspace_rule_paths
    )
    for raw_path in _discover_paths(
        input_roots,
        direct_match=_is_workspace_root,
        glob_patterns=(WORKSPACE_ROOT_GLOB,),
        expect_dir=True,
    ):
        discovered.add(str(_workspace_root_from_windsurf_root(Path(raw_path))))
    return tuple(sorted(discovered))


def _discover_paths(
    input_roots: tuple[Path, ...],
    *,
    direct_match: Callable[[Path], bool],
    glob_patterns: tuple[str, ...],
    expect_dir: bool,
) -> tuple[str, ...]:
    seen: set[Path] = set()
    candidates: list[str] = []
    for input_root in input_roots:
        for candidate in _iter_candidates(input_root, glob_patterns):
            resolved = candidate.expanduser().resolve(strict=False)
            if resolved in seen:
                continue
            if not direct_match(candidate):
                continue
            if expect_dir and not candidate.is_dir():
                continue
            if not expect_dir and not candidate.is_file():
                continue
            seen.add(resolved)
            candidates.append(str(resolved))
    return tuple(sorted(candidates))


def _iter_candidates(input_root: Path, glob_patterns: tuple[str, ...]) -> Iterable[Path]:
    yield input_root
    if not input_root.is_dir():
        return
    for pattern in glob_patterns:
        yield from input_root.glob(pattern)


def _is_global_root(path: Path) -> bool:
    return path.name == "windsurf" and path.parent.name in {".codeium", "Codeium"}


def _is_application_support_root(path: Path) -> bool:
    parts = path.parts
    if path.name != "Windsurf":
        return False
    return len(parts) >= 3 and parts[-3:] == ("Library", "Application Support", "Windsurf")


def _is_memory_file(path: Path) -> bool:
    if not path.is_file():
        return False
    if "memories" not in path.parts:
        return False
    if path.name == "global_rules.md":
        return False
    return True


def _is_global_rule_file(path: Path) -> bool:
    if not path.is_file() or path.name != "global_rules.md":
        return False
    parts = path.parts
    if len(parts) >= 3 and parts[-3:] == ("windsurf", "memories", "global_rules.md"):
        return True
    return len(parts) >= 2 and parts[-2:] == ("windsurf", "global_rules.md")


def _is_workspace_rule_file(path: Path) -> bool:
    return path.is_file() and path.suffix == ".md" and path.parent.name == "rules" and path.parent.parent.name == ".windsurf"


def _is_system_rule_file(path: Path) -> bool:
    if not path.is_file() or path.suffix != ".md":
        return False
    parts = path.parts
    if len(parts) >= 4 and parts[-4:] == ("Application Support", "Windsurf", "rules", path.name):
        return True
    return len(parts) >= 3 and parts[-3:] == ("windsurf", "rules", path.name)


def _is_workspace_root(path: Path) -> bool:
    return path.is_dir() and path.name == ".windsurf"


def _is_global_skills_root(path: Path) -> bool:
    return path.is_dir() and path.name == "skills" and _is_global_root(path.parent)


def _is_workspace_skills_root(path: Path) -> bool:
    return path.is_dir() and path.name == "skills" and path.parent.name == ".windsurf"


def _is_mcp_config(path: Path) -> bool:
    return path.is_file() and path.name == "mcp_config.json" and _is_global_root(path.parent)


def _load_mcp_server_names(mcp_config_paths: tuple[str, ...]) -> tuple[str, ...]:
    names: set[str] = set()
    for raw_path in mcp_config_paths:
        try:
            payload = json.loads(Path(raw_path).read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        server_map = payload.get("mcpServers")
        if not isinstance(server_map, dict):
            continue
        for key in server_map:
            if isinstance(key, str) and key:
                names.add(key)
    return tuple(sorted(names))


def _read_memory_text(path: Path) -> tuple[str | None, str]:
    if path.suffix.lower() in TEXT_MEMORY_SUFFIXES:
        text = _read_text_file(path)
        return (None if text is None else text.strip() or None, "text")

    if path.suffix.lower() in JSON_MEMORY_SUFFIXES:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return None, "json"
        parts = _extract_text_parts(payload)
        if not parts:
            return None, "json"
        return "\n\n".join(parts), "json"

    text = _read_text_file(path)
    if text is None:
        return None, "unknown"
    stripped = text.strip()
    if not stripped:
        return None, "unknown"
    return stripped, "unknown"


def _extract_text_parts(value: object) -> tuple[str, ...]:
    parts: list[str] = []

    def visit(node: object, *, allow_fallback: bool) -> None:
        if isinstance(node, str):
            text = node.strip()
            if text and text not in parts:
                parts.append(text)
            return
        if isinstance(node, list):
            for item in node:
                visit(item, allow_fallback=allow_fallback)
            return
        if not isinstance(node, dict):
            return

        matched_key = False
        for key in TEXT_VALUE_KEYS:
            if key not in node:
                continue
            matched_key = True
            visit(node[key], allow_fallback=False)
        if matched_key or not allow_fallback:
            return
        for nested_value in node.values():
            visit(nested_value, allow_fallback=False)

    visit(value, allow_fallback=True)
    return tuple(parts)


def _parse_rule_document(text: str) -> WindsurfRuleDocument:
    if not text.startswith(f"{RULE_FRONTMATTER_DELIMITER}\n"):
        return WindsurfRuleDocument(frontmatter={}, body=text.strip())

    delimiter = f"\n{RULE_FRONTMATTER_DELIMITER}\n"
    frontmatter_end = text.find(delimiter, len(RULE_FRONTMATTER_DELIMITER) + 1)
    if frontmatter_end == -1:
        return WindsurfRuleDocument(frontmatter={}, body=text.strip())

    raw_frontmatter = text[len(RULE_FRONTMATTER_DELIMITER) + 1 : frontmatter_end]
    body = text[frontmatter_end + len(delimiter) :].strip()
    return WindsurfRuleDocument(
        frontmatter=_parse_frontmatter(raw_frontmatter),
        body=body,
    )


def _parse_frontmatter(raw_frontmatter: str) -> dict[str, object]:
    parsed: dict[str, object] = {}
    current_list_key: str | None = None
    for raw_line in raw_frontmatter.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if current_list_key is not None and stripped.startswith("- "):
            existing = parsed.setdefault(current_list_key, [])
            if isinstance(existing, list):
                existing.append(stripped[2:].strip())
            continue
        current_list_key = None
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        normalized_key = key.strip()
        normalized_value = value.strip()
        if not normalized_key:
            continue
        if normalized_key in RULE_FRONTMATTER_LIST_KEYS and not normalized_value:
            parsed[normalized_key] = []
            current_list_key = normalized_key
            continue
        parsed[normalized_key] = normalized_value
    return parsed


def _read_text_file(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (FileNotFoundError, UnicodeDecodeError):
        return None


def _memory_relative_path(memory_path: Path) -> Path:
    for parent in memory_path.parents:
        if parent.name == "memories":
            return memory_path.relative_to(parent)
    return Path(memory_path.name)


def _global_rule_relative_path(rule_path: Path) -> Path:
    for parent in rule_path.parents:
        if parent.name == "windsurf":
            return rule_path.relative_to(parent)
    return Path(rule_path.name)


def _system_rule_relative_path(rule_path: Path) -> Path:
    if rule_path.parent.name == "rules":
        return Path("rules") / rule_path.name
    return Path(rule_path.name)


def _workspace_root_for_rule(rule_path: Path) -> Path:
    return rule_path.parents[2].resolve(strict=False)


def _workspace_root_from_windsurf_root(windsurf_root: Path) -> Path:
    return windsurf_root.parent.resolve(strict=False)


def _rule_session_id(*, scope: str, rule_path: Path) -> str:
    if scope == "workspace":
        workspace_root = _workspace_root_for_rule(rule_path)
        return f"workspace:{workspace_root.name}:rule:{rule_path.stem}"
    return f"{scope}:rule:{rule_path.stem}"


def _row_scope(source_session_id: str | None) -> str | None:
    if source_session_id is None:
        return None
    if source_session_id.startswith("global:"):
        return "global"
    if source_session_id.startswith("system:"):
        return "system"
    return None


def _is_global_message_row(conversation: NormalizedConversation) -> bool:
    if conversation.messages:
        if _row_scope(conversation.source_session_id) == "global":
            return True
        if not conversation.session_metadata:
            return False
        if conversation.session_metadata.get("scope") == "memory":
            return True
    return False


def _workspace_root_from_row(conversation: NormalizedConversation) -> str | None:
    if not conversation.session_metadata:
        return None
    workspace_root = conversation.session_metadata.get("workspace_root")
    if isinstance(workspace_root, str):
        return workspace_root
    return None


def _workspace_rule_count(workspace_root: Path, workspace_rule_paths: tuple[str, ...]) -> int:
    resolved_workspace_root = workspace_root.expanduser().resolve(strict=False)
    return sum(
        1
        for raw_path in workspace_rule_paths
        if _workspace_root_for_rule(Path(raw_path)) == resolved_workspace_root
    )


def _workspace_skill_root_paths_for_root(
    workspace_root: Path,
    workspace_skill_root_paths: tuple[str, ...],
) -> tuple[Path, ...]:
    resolved_workspace_root = workspace_root.expanduser().resolve(strict=False)
    return tuple(
        Path(raw_path)
        for raw_path in workspace_skill_root_paths
        if Path(raw_path).expanduser().resolve(strict=False).parent.parent
        == resolved_workspace_root
    )


def _workspace_skill_dir_count(
    workspace_root: Path,
    workspace_skill_root_paths: tuple[str, ...],
) -> int:
    return _skill_dir_count(
        tuple(
            str(path)
            for path in _workspace_skill_root_paths_for_root(
                workspace_root,
                workspace_skill_root_paths,
            )
        )
    )


def _workspace_skill_file_count(
    workspace_root: Path,
    workspace_skill_root_paths: tuple[str, ...],
) -> int:
    return _skill_file_count(
        tuple(
            str(path)
            for path in _workspace_skill_root_paths_for_root(
                workspace_root,
                workspace_skill_root_paths,
            )
        )
    )


def _skill_dir_count(skill_root_paths: tuple[str, ...]) -> int:
    count = 0
    for raw_path in skill_root_paths:
        root = Path(raw_path)
        if not root.is_dir():
            continue
        count += sum(1 for child in root.iterdir() if child.is_dir())
    return count


def _skill_file_count(skill_root_paths: tuple[str, ...]) -> int:
    count = 0
    for raw_path in skill_root_paths:
        root = Path(raw_path)
        if not root.is_dir():
            continue
        count += sum(1 for child in root.rglob("*") if child.is_file())
    return count


def _json_ready_frontmatter(frontmatter: dict[str, object]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in frontmatter.items():
        if isinstance(value, list):
            payload[key] = [str(item) for item in value]
            continue
        if isinstance(value, str):
            payload[key] = value
    return payload


def _conversation_sort_key(conversation: NormalizedConversation) -> tuple[str, str]:
    return (
        conversation.source_session_id or "",
        conversation.source_artifact_path or "",
    )


def _first_path(paths: tuple[str, ...]) -> str | None:
    return paths[0] if paths else None


def _string_value(value: object) -> str | None:
    if isinstance(value, str):
        return value
    return None


__all__ = [
    "WINDSURF_EDITOR_DESCRIPTOR",
    "WindsurfArtifacts",
    "WindsurfEditorCollector",
    "build_windsurf_conversations",
    "build_global_metadata_row",
    "build_workspace_metadata_row",
    "discover_windsurf_editor_artifacts",
    "parse_memory_file",
    "parse_rule_file",
]
