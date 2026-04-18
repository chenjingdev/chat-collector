# Source Support Matrix

This file is generated from registry metadata via `uv run llm-chat-archive sources --format markdown`.

`Included in default batch` reflects the `default` selection profile.

| Source key | Product | Host surface | Support level | Expected transcript completeness | Major limitation | Included in default batch |
| --- | --- | --- | --- | --- | --- | --- |
| `antigravity_editor_view` | Antigravity | Editor view | `partial` | `partial` | Only the confirmed raw Antigravity conversation protobuf variant is promoted to transcript rows; current operator-local opaque blobs degrade to explicit variant_unknown or decode_failed diagnostics instead of false-complete output. | no |
| `claude` | Claude Code | CLI | `complete` | `complete` | Subagent traces remain separate JSONL sessions instead of being merged into the parent. | yes |
| `claude_code_ide` | Claude Code | IDE bridge | `complete` | `complete` | IDE bridge metadata stays provenance-only; the shared Claude session JSONL remains canonical. | yes |
| `codex_app` | Codex | Desktop app | `complete` | `complete` | Desktop automation run state comes from shared SQLite metadata, and archived automation rollout gaps stay rollout-first while repaired bodies are tagged with explicit fallback provenance. | yes |
| `codex_cli` | Codex | CLI | `complete` | `complete` | Filters event, reasoning, tool, and search noise out of the transcript. | yes |
| `codex_ide_extension` | Codex | IDE extension | `complete` | `complete` | IDE bridge residue stays provenance-only; shared rollout JSONL remains canonical. | yes |
| `cursor` | Cursor | CLI | `partial` | `partial` | Cursor CLI still depends on shared editor transcript rows plus unique invocation attribution, so it remains partial and opt-in for unattended batches. | no |
| `cursor_editor` | Cursor | Editor | `partial` | `partial` | Cursor editor recovery restores known explicit cursorDiskKV bubble body variants, but sessions whose headers resolve only to empty or tool-only rows remain partial and opt-in for unattended batches. | no |
| `gemini` | Gemini | CLI | `complete` | `complete` | Only user and gemini message content is retained; logs, auth, thoughts, and tool residue are excluded. | yes |
| `gemini_code_assist_ide` | Gemini Code Assist | IDE extension | `partial` | `partial` | Gemini-owned chatSessions with recoverable body shapes are promoted, but foreign providers and Gemini-owned body-missing sessions still degrade to explicit metadata-only residue, so the source stays partial and opt-in. | no |
| `windsurf_editor` | Windsurf | Editor | `partial` | `partial` | Windsurf local memories and rules can be normalized, but no confirmed native editor session-history store is available yet, so the collector remains partial and opt-in. | no |
