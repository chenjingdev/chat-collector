from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from llm_chat_archive.models import Coverage, DisplayTurn, EventRecord, SessionRecord
from llm_chat_archive.adapters.ide import _BaseIdeAdapter, LogSpec


class _LegacyAntigravityAdapter(_BaseIdeAdapter):
    source_name = "antigravity"

    def app_name(self) -> str:
        return "Antigravity"

    def _log_specs(self) -> tuple[LogSpec, ...]:
        return (
            LogSpec(
                pattern="logs/**/exthost/google.antigravity/Antigravity.log",
                variant="native",
                label="Antigravity.log",
            ),
            LogSpec(
                pattern="logs/**/exthost/openai.chatgpt/Codex.log",
                variant="codex-extension",
                label="Codex.log",
            ),
        )

    def _variant_from_state_key(self, key: str) -> str:
        lowered = key.lower()
        if lowered.startswith("workbench.panel.aichat.view.aichat.chatdata"):
            return "legacy-aichat"
        if (
            lowered.startswith("google.antigravity")
            or lowered.startswith("antigravity")
            or lowered.startswith("chat.chatsessionstore")
            or lowered.startswith("chat.participantnameregistry")
            or lowered.startswith("chat.workspace")
            or "geminichat" in lowered
            or lowered.startswith("google.geminicodeassist")
        ):
            return "native"
        if "openai.chatgpt" in lowered or "codex" in lowered:
            return "codex-extension"
        return "ide-state"


class AntigravityAdapter:
    source_name = "antigravity"

    def __init__(self, app_support_root: Path | None = None) -> None:
        self.app_support_root = app_support_root or Path.home() / ".gemini" / "antigravity" / "brain"
        # Legacy adapter looks in ~/Library/Application Support/Antigravity
        self.legacy_adapter = _LegacyAntigravityAdapter(
            app_support_root=app_support_root or Path.home() / "Library" / "Application Support" / "Antigravity"
        )

    def collect(self) -> list[SessionRecord]:
        sessions: list[SessionRecord] = []
        
        # 1. Collect from modern Brain Markdown format
        if self.app_support_root.exists():
            for session_dir in sorted(self.app_support_root.iterdir()):
                if not session_dir.is_dir() or session_dir.name.startswith("."):
                    continue

                session = self._parse_session_dir(session_dir)
                if session:
                    sessions.append(session)

        # 2. Collect from Legacy VSCode extension Logs and DBs
        sessions.extend(self.legacy_adapter.collect())

        return sessions

    def _parse_session_dir(self, session_dir: Path) -> SessionRecord | None:
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
            text = "\n\n".join(combined_markdown)
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

        coverage = Coverage.PARTIAL

        return SessionRecord(
            source=self.source_name,
            source_session_id=session_dir.name,
            source_variant="brain",
            project=None,
            cwd=None,
            started_at=started_at,
            ended_at=ended_at,
            raw_events=events,
            display_turns=display_turns,
            coverage=coverage,
            evidence_paths=evidence_paths,
            source_files=source_files,
            warnings=warnings,
            token_usage={},
        )
