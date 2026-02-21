from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from llm_chat_archive.models import Coverage, DisplayTurn, EventRecord, SessionRecord
from llm_chat_archive.adapters.ide import _BaseIdeAdapter, LogSpec


class _ModernAntigravityIdeAdapter(_BaseIdeAdapter):
    """
    Parses genuine User/Assistant conversational turns from the Antigravity VSCode Fork
    by reading globalStorage and workspaceStorage `state.vscdb` SQLite databases.
    """
    source_name = "antigravity"

    def app_name(self) -> str:
        return "Antigravity"

    def _log_specs(self) -> tuple[LogSpec, ...]:
        return ()

    def _variant_from_state_key(self, key: str) -> str:
        lowered = key.lower()
        if (
            lowered.startswith("google.antigravity")
            or lowered.startswith("antigravity")
            or lowered.startswith("chat.chatsessionstore")
            or lowered.startswith("chat.participantnameregistry")
            or lowered.startswith("chat.workspace")
            or "geminichat" in lowered
            or lowered.startswith("google.geminicodeassist")
        ):
            return "ide-chat"
        if "openai.chatgpt" in lowered or "codex" in lowered:
            return "ide-chat"
        return "ide-chat"

    def _build_state_session(
        self, db_path: Path, variant: str, rows: list, warnings: list[str]
    ) -> SessionRecord:
        """
        Custom override: Antigravity stores its prompt arrays in 'openai.chatgpt' JSON structures.
        """
        session = super()._build_state_session(db_path=db_path, variant=variant, rows=rows, warnings=warnings)

        for event in session.raw_events:
            # We specifically target the openai.chatgpt record to extract prompt-history
            if event.raw_json and event.raw_json.get("key") == "openai.chatgpt":
                val = event.raw_json.get("value", {})
                if isinstance(val, dict):
                    atom_state = val.get("persisted-atom-state", {})
                    prompts = atom_state.get("prompt-history", [])
                    titles = val.get("thread-titles", {}).get("titles", {})

                    if prompts or titles:
                        text = "## Workspace Prompts History\n"
                        for p in prompts:
                            text += f"- **User**: {p}\n"
                        if titles:
                            text += "\n## Thread Titles\n"
                            for tk, tv in titles.items():
                                text += f"- {tv}\n"
                            
                        # Add these as display turns
                        session.display_turns.append(
                            DisplayTurn(
                                timestamp=event.timestamp,
                                role="user",
                                text=text,
                            )
                        )
                        session.coverage = Coverage.PARTIAL

        return session


class _LegacyAntigravityAdapter(_BaseIdeAdapter):
    """
    Parses legacy text logs (Antigravity.log, Codex.log) from older VSCode extensions.
    """
    source_name = "antigravity"

    def app_name(self) -> str:
        return "Antigravity"

    def _log_specs(self) -> tuple[LogSpec, ...]:
        return (
            LogSpec(
                pattern="logs/**/exthost/google.antigravity/Antigravity.log",
                variant="legacy-log",
                label="Antigravity.log",
            ),
            LogSpec(
                pattern="logs/**/exthost/openai.chatgpt/Codex.log",
                variant="codex-extension",
                label="Codex.log",
            ),
        )


class AntigravityAdapter:
    """
    Main entry point for Antigravity collection.
    It combines valid `state.vscdb` user conversations with background brain artifacts.
    """
    source_name = "antigravity"

    def __init__(self, app_support_root: Path | None = None) -> None:
        self.brain_root = Path.home() / ".gemini" / "antigravity" / "brain"
        # Extract conversational turns from actual DB states
        self.ide_adapter = _ModernAntigravityIdeAdapter(
            app_support_root=app_support_root or Path.home() / "Library" / "Application Support" / "Antigravity"
        )
        self.legacy_adapter = _LegacyAntigravityAdapter(
             app_support_root=app_support_root or Path.home() / "Library" / "Application Support" / "Antigravity"
        )

    def collect(self) -> list[SessionRecord]:
        sessions: list[SessionRecord] = []
        
        # 1. Collect genuine UI text conversations
        sessions.extend(self.ide_adapter.collect())

        # 2. Collect Legacy VSCode logs
        sessions.extend(self.legacy_adapter.collect())

        # 3. Collect Brain background tasks (as a separate 'brain' context)
        if self.brain_root.exists():
            for session_dir in sorted(self.brain_root.iterdir()):
                if not session_dir.is_dir() or session_dir.name.startswith("."):
                    continue

                session = self._parse_brain_dir(session_dir)
                if session:
                    sessions.append(session)

        return sessions

    def _parse_brain_dir(self, session_dir: Path) -> SessionRecord | None:
        events: list[EventRecord] = []
        display_turns: list[DisplayTurn] = []
        warnings: list[str] = []
        evidence_paths: list[str] = []
        source_files: list[Path] = []

        started_at = None
        ended_at = None

        def _update_times(path: Path):
            nonlocal started_at, ended_at
            if not path.exists():
                return
            try:
                stat = path.stat()
                ctime = datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc)
                mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
                for ts in (ctime, mtime):
                    if started_at is None or ts < started_at:
                        started_at = ts
                    if ended_at is None or ts > ended_at:
                        ended_at = ts
            except OSError:
                pass

        artifacts = ["task.md", "implementation_plan.md", "walkthrough.md"]
        combined_markdown = []

        for artifact in artifacts:
            path = session_dir / artifact
            if path.exists():
                _update_times(path)
                evidence_paths.append(str(path))
                source_files.append(path)
                try:
                    content = path.read_text(encoding="utf-8").strip()
                    if content:
                        header = artifact.replace(".md", "").replace("_", " ").title()
                        combined_markdown.append(f"## {header}\n\n{content}")
                        events.append(
                            EventRecord(
                                timestamp=ended_at,
                                role="assistant",
                                event_kind=f"artifact.{artifact}",
                                content_text=content,
                            )
                        )
                except OSError as e:
                    warnings.append(f"Failed to read {artifact}: {e}")

        if combined_markdown:
            text = "*(This session contains only background autonomous tasks and artifacts. User inputs may be logged in the IDE DB state).* \n\n" + "\n\n".join(combined_markdown)
            display_turns.append(
                DisplayTurn(
                    timestamp=ended_at,
                    role="assistant",
                    text=text,
                )
            )

        if not events:
            return None

        # Fallback to dir creation time if files lacked times
        if not started_at:
            _update_times(session_dir)

        return SessionRecord(
            source=self.source_name,
            source_session_id=session_dir.name,
            source_variant="brain-artifacts",
            project=None,
            cwd=None,
            started_at=started_at,
            ended_at=ended_at,
            raw_events=events,
            display_turns=display_turns,
            coverage=Coverage.PARTIAL, # Marked unsupported so users don't think it's broken text.
            evidence_paths=evidence_paths,
            source_files=source_files,
            warnings=warnings,
            token_usage={},
        )
