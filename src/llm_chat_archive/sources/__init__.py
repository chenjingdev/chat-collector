from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..models import (
    CollectionPlan,
    SourceDescriptor,
    SourceSupportMetadata,
    SupportLevel,
    TranscriptCompleteness,
)
from ..source_roots import (
    all_platform_root,
    darwin_root,
    linux_root,
    windows_root,
)
from ..registry import Collector, CollectorRegistry
from .antigravity_editor_view import (
    ANTIGRAVITY_EDITOR_VIEW_DESCRIPTOR,
    AntigravityEditorViewCollector,
)
from .claude_code_cli import CLAUDE_CODE_CLI_DESCRIPTOR, ClaudeCodeCliCollector
from .claude_code_ide import CLAUDE_CODE_IDE_DESCRIPTOR, ClaudeCodeIdeCollector
from .codex_app import CODEX_APP_DESCRIPTOR, CodexAppCollector
from .codex_cli import CODEX_CLI_DESCRIPTOR, CodexCliCollector
from .codex_ide_extension import (
    CODEX_IDE_EXTENSION_DESCRIPTOR,
    CodexIdeExtensionCollector,
)
from .cursor_cli import CURSOR_CLI_DESCRIPTOR, CursorCliCollector
from .cursor_editor import CURSOR_EDITOR_DESCRIPTOR, CursorEditorCollector
from .gemini_cli import GEMINI_CLI_DESCRIPTOR, GeminiCliCollector
from .gemini_code_assist_ide import (
    GEMINI_CODE_ASSIST_IDE_DESCRIPTOR,
    GeminiCodeAssistIdeCollector,
)
from .windsurf_editor import WINDSURF_EDITOR_DESCRIPTOR, WindsurfEditorCollector


@dataclass(frozen=True, slots=True)
class ScaffoldCollector(Collector):
    descriptor: SourceDescriptor

    def build_plan(self, archive_root: Path) -> CollectionPlan:
        return CollectionPlan(
            source=self.descriptor.key,
            display_name=self.descriptor.display_name,
            archive_root=archive_root,
            execution_context=self.descriptor.execution_context,
            support_level=self.descriptor.support_level,
            default_input_roots=self.descriptor.default_input_roots,
            notes=self.descriptor.notes,
        )


BUILTIN_SOURCE_DESCRIPTORS = (
    SourceDescriptor(
        key="cursor",
        display_name="Cursor CLI",
        execution_context="cli",
        support_level=SupportLevel.SCAFFOLD,
        default_input_roots=(
            "~/.cursor",
            "~/Library/Application Support/Cursor/User",
        ),
        artifact_root_candidates=(
            all_platform_root("$HOME/.cursor"),
            darwin_root("$HOME/Library/Application Support/Cursor/User"),
            linux_root("$XDG_CONFIG_HOME/Cursor/User"),
            windows_root("$APPDATA/Cursor/User"),
        ),
        notes=(
            "Current local evidence is metadata-first, not a clean transcript store.",
            "Collector implementation should start from workspace storage metadata, not logs.",
        ),
        support_metadata=SourceSupportMetadata(
            product_label="Cursor",
            host_surface="CLI",
            expected_transcript_completeness=TranscriptCompleteness.UNSUPPORTED,
            limitation_summary="Scaffold-only placeholder; no confirmed Cursor CLI transcript path is implemented here.",
        ),
    ),
)


def register_builtin_collectors(registry: CollectorRegistry) -> CollectorRegistry:
    registry.register(
        AntigravityEditorViewCollector(descriptor=ANTIGRAVITY_EDITOR_VIEW_DESCRIPTOR)
    )
    registry.register(ClaudeCodeCliCollector(descriptor=CLAUDE_CODE_CLI_DESCRIPTOR))
    registry.register(ClaudeCodeIdeCollector(descriptor=CLAUDE_CODE_IDE_DESCRIPTOR))
    registry.register(CodexAppCollector(descriptor=CODEX_APP_DESCRIPTOR))
    registry.register(CodexCliCollector(descriptor=CODEX_CLI_DESCRIPTOR))
    registry.register(CodexIdeExtensionCollector(descriptor=CODEX_IDE_EXTENSION_DESCRIPTOR))
    registry.register(CursorCliCollector(descriptor=CURSOR_CLI_DESCRIPTOR))
    registry.register(CursorEditorCollector(descriptor=CURSOR_EDITOR_DESCRIPTOR))
    registry.register(GeminiCliCollector(descriptor=GEMINI_CLI_DESCRIPTOR))
    registry.register(
        GeminiCodeAssistIdeCollector(descriptor=GEMINI_CODE_ASSIST_IDE_DESCRIPTOR)
    )
    registry.register(WindsurfEditorCollector(descriptor=WINDSURF_EDITOR_DESCRIPTOR))
    for descriptor in BUILTIN_SOURCE_DESCRIPTORS:
        if descriptor.key == CURSOR_CLI_DESCRIPTOR.key:
            continue
        registry.register(ScaffoldCollector(descriptor=descriptor))
    return registry


def build_registry() -> CollectorRegistry:
    return register_builtin_collectors(CollectorRegistry())


__all__ = [
    "ANTIGRAVITY_EDITOR_VIEW_DESCRIPTOR",
    "BUILTIN_SOURCE_DESCRIPTORS",
    "CLAUDE_CODE_CLI_DESCRIPTOR",
    "CLAUDE_CODE_IDE_DESCRIPTOR",
    "CODEX_APP_DESCRIPTOR",
    "CODEX_CLI_DESCRIPTOR",
    "CODEX_IDE_EXTENSION_DESCRIPTOR",
    "CURSOR_CLI_DESCRIPTOR",
    "CURSOR_EDITOR_DESCRIPTOR",
    "GEMINI_CLI_DESCRIPTOR",
    "GEMINI_CODE_ASSIST_IDE_DESCRIPTOR",
    "WINDSURF_EDITOR_DESCRIPTOR",
    "AntigravityEditorViewCollector",
    "ClaudeCodeCliCollector",
    "ClaudeCodeIdeCollector",
    "CodexAppCollector",
    "CodexCliCollector",
    "CodexIdeExtensionCollector",
    "CursorCliCollector",
    "CursorEditorCollector",
    "GeminiCliCollector",
    "GeminiCodeAssistIdeCollector",
    "ScaffoldCollector",
    "WindsurfEditorCollector",
    "build_registry",
    "register_builtin_collectors",
]
