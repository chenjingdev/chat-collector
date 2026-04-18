from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from ..incremental import write_incremental_collection
from ..models import (
    CollectionPlan,
    CollectionResult,
    IdeBridgeProvenance,
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
    parse_codex_rollout_file,
    resolve_input_roots,
    utc_timestamp,
)

IDE_BRIDGE_GLOBS = (
    "**/state.vscdb",
    "**/Codex.log",
    "**/app_pairing_extensions/*",
)

CODEX_IDE_EXTENSION_DESCRIPTOR = SourceDescriptor(
    key="codex_ide_extension",
    display_name="Codex IDE Extension",
    execution_context="ide_extension",
    support_level=SupportLevel.COMPLETE,
    default_input_roots=(
        "~/.codex",
        "~/Library/Application Support/Code/User/globalStorage",
        "~/Library/Application Support/Code/User/workspaceStorage",
        "~/Library/Application Support/Code/logs",
        "~/Library/Application Support/Cursor/User/workspaceStorage",
        "~/Library/Application Support/Cursor/logs",
        "~/Library/Application Support/com.openai.chat/app_pairing_extensions",
    ),
    artifact_root_candidates=(
        all_platform_root("$HOME/.codex"),
        darwin_root("$HOME/Library/Application Support/Code/User/globalStorage"),
        linux_root("$XDG_CONFIG_HOME/Code/User/globalStorage"),
        windows_root("$APPDATA/Code/User/globalStorage"),
        darwin_root("$HOME/Library/Application Support/Code/User/workspaceStorage"),
        linux_root("$XDG_CONFIG_HOME/Code/User/workspaceStorage"),
        windows_root("$APPDATA/Code/User/workspaceStorage"),
        darwin_root("$HOME/Library/Application Support/Code/logs"),
        linux_root("$XDG_CONFIG_HOME/Code/logs"),
        windows_root("$APPDATA/Code/logs"),
        darwin_root("$HOME/Library/Application Support/Cursor/User/workspaceStorage"),
        linux_root("$XDG_CONFIG_HOME/Cursor/User/workspaceStorage"),
        windows_root("$APPDATA/Cursor/User/workspaceStorage"),
        darwin_root("$HOME/Library/Application Support/Cursor/logs"),
        linux_root("$XDG_CONFIG_HOME/Cursor/logs"),
        windows_root("$APPDATA/Cursor/logs"),
        darwin_root("$HOME/Library/Application Support/com.openai.chat/app_pairing_extensions"),
        linux_root("$XDG_CONFIG_HOME/com.openai.chat/app_pairing_extensions"),
        windows_root("$APPDATA/com.openai.chat/app_pairing_extensions"),
    ),
    notes=(
        "Uses shared ~/.codex rollout JSONL as the canonical transcript source.",
        'Selects only sessions whose session_meta payload originator is "codex_vscode".',
        "Treats VS Code and Cursor state.vscdb, Codex.log, and bridge payload files as provenance only.",
    ),
    support_metadata=SourceSupportMetadata(
        product_label="Codex",
        host_surface="IDE extension",
        expected_transcript_completeness=TranscriptCompleteness.COMPLETE,
        limitation_summary="IDE bridge residue stays provenance-only; shared rollout JSONL remains canonical.",
    ),
)


@dataclass(frozen=True, slots=True)
class CodexIdeExtensionCollector:
    descriptor: SourceDescriptor = CODEX_IDE_EXTENSION_DESCRIPTOR

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
        ide_bridge = discover_ide_bridge_provenance(resolved_input_roots)
        collected_at = utc_timestamp()
        conversations = (
            parse_rollout_file(
                rollout_path,
                collected_at=collected_at,
                ide_bridge=ide_bridge,
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
    ide_bridge: IdeBridgeProvenance | None = None,
):
    return parse_codex_rollout_file(
        rollout_path,
        descriptor=CODEX_IDE_EXTENSION_DESCRIPTOR,
        collected_at=collected_at,
        session_filter=_is_codex_ide_extension_session,
        provenance_factory=lambda metadata, _path: _build_provenance(metadata, ide_bridge),
    )


def discover_ide_bridge_provenance(
    input_roots: tuple[Path, ...] | None,
) -> IdeBridgeProvenance | None:
    if not input_roots:
        return None

    hosts: set[str] = set()
    state_db_paths: list[str] = []
    log_paths: list[str] = []
    bridge_payload_paths: list[str] = []

    for artifact_path in _iter_ide_bridge_artifacts(input_roots):
        host = _infer_host(artifact_path)
        if host is not None:
            hosts.add(host)

        artifact_str = str(artifact_path)
        if artifact_path.name == "state.vscdb":
            state_db_paths.append(artifact_str)
            continue
        if artifact_path.name == "Codex.log":
            log_paths.append(artifact_str)
            continue
        bridge_payload_paths.append(artifact_str)

    if not state_db_paths and not log_paths and not bridge_payload_paths:
        return None

    return IdeBridgeProvenance(
        hosts=tuple(sorted(hosts)),
        state_db_paths=tuple(sorted(state_db_paths)),
        log_paths=tuple(sorted(log_paths)),
        bridge_payload_paths=tuple(sorted(bridge_payload_paths)),
    )


def _build_provenance(
    session_metadata: CodexSessionMetadata,
    ide_bridge: IdeBridgeProvenance | None,
):
    return replace(build_conversation_provenance(session_metadata), ide_bridge=ide_bridge)


def _is_codex_ide_extension_session(
    session_metadata: CodexSessionMetadata, _rollout_path: Path
) -> bool:
    return session_metadata.originator == "codex_vscode"


def _iter_ide_bridge_artifacts(input_roots: tuple[Path, ...]) -> tuple[Path, ...]:
    seen: set[Path] = set()
    candidates: list[Path] = []
    for input_root in input_roots:
        if input_root.is_file():
            resolved = input_root.resolve(strict=False)
            if resolved in seen or not input_root.exists():
                continue
            seen.add(resolved)
            candidates.append(resolved)
            continue

        for pattern in IDE_BRIDGE_GLOBS:
            for candidate in input_root.glob(pattern):
                if not candidate.is_file():
                    continue
                resolved = candidate.resolve(strict=False)
                if resolved in seen:
                    continue
                seen.add(resolved)
                candidates.append(resolved)
    return tuple(sorted(candidates))


def _infer_host(artifact_path: Path) -> str | None:
    path_str = str(artifact_path)
    if "Visual Studio Code-" in artifact_path.name:
        return "vscode"
    if artifact_path.name.startswith("Cursor-"):
        return "cursor"
    if "/Application Support/Code/" in path_str:
        return "vscode"
    if "/Application Support/Cursor/" in path_str:
        return "cursor"
    return None


__all__ = [
    "CODEX_IDE_EXTENSION_DESCRIPTOR",
    "CodexIdeExtensionCollector",
    "discover_ide_bridge_provenance",
    "parse_rollout_file",
]
