# Codex App Local Transcript And Worktree Artifacts

Updated: `2026-03-20`

Scope: local artifact reconnaissance for `Codex` in `standalone_app` execution context on macOS. This note stops at storage-path and format identification. It does not implement a collector or parser.

## High-Signal Summary

- Primary observed standalone-app support roots on macOS:
  - `~/Library/Application Support/Codex/`
  - `~/Library/Logs/com.openai.codex/`
  - `~/Library/Preferences/com.openai.codex.plist`
  - `~/Library/Caches/com.openai.codex/`
- Primary observed transcript and thread root for Desktop sessions: `~/.codex/`
- Strongest transcript candidate: `~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl`
- Strongest metadata candidate: `~/.codex/state_5.sqlite`
- Shared-storage conclusion: the standalone app writes shell-specific support files under `~/Library/...`, but the actual conversation and thread registry are shared with other Codex execution contexts under `~/.codex/`
- Negative result: no `IndexedDB` directory, transcript-like JSONL, or thread-specific SQLite database was observed under `~/Library/Application Support/Codex/`
- Negative result: no `*.patch`, `*.diff`, or `*.bundle` files were observed anywhere under `~/.codex/` on this machine at inspection time

## Observed macOS Roots

### Shared Codex-family Root: `~/.codex/`

- `~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl`
  - Observed `257` Desktop-originated rollout files where the first line carried `session_meta.payload.originator == "Codex Desktop"`
  - Desktop-originated `session_meta.payload.source` split observed on this machine:
    - `vscode`: `92`
    - `exec`: `29`
    - serialized subagent source objects: `136`
  - Sample `session_meta.payload` keys observed:
    - `base_instructions`
    - `cli_version`
    - `cwd`
    - `id`
    - `model_provider`
    - `originator`
    - `source`
    - `timestamp`
- `~/.codex/state_5.sqlite`
  - `threads` table columns observed:
    - `id`
    - `rollout_path`
    - `created_at`
    - `updated_at`
    - `source`
    - `model_provider`
    - `cwd`
    - `title`
    - `sandbox_policy`
    - `approval_mode`
    - `tokens_used`
    - `has_user_event`
    - `archived`
    - `archived_at`
    - `git_sha`
    - `git_branch`
    - `git_origin_url`
    - `cli_version`
    - `first_user_message`
    - `agent_nickname`
    - `agent_role`
    - `memory_mode`
  - For the `257` Desktop-originated thread IDs observed in rollout JSONL:
    - all `257` had `cwd`
    - `75` had `git_branch`
    - `74` had `git_sha`
    - `59` had `git_origin_url`
  - `threads.id` matched `session_meta.payload.id`, and `threads.rollout_path` pointed back to the JSONL rollout path
- `~/.codex/thread_dynamic_tools`
  - `thread_dynamic_tools` columns observed:
    - `thread_id`
    - `position`
    - `name`
    - `description`
    - `input_schema`
  - This is dynamic tool schema metadata, not transcript body
- `~/.codex/skills/`
  - Observed both user-installed and system skill packages under the shared root
  - Sample entries:
    - `~/.codex/skills/cmux-core-mini/`
    - `~/.codex/skills/pdf/`
    - `~/.codex/skills/cc-feature-implementer/`
    - `~/.codex/skills/.system/skill-installer/`
    - `~/.codex/skills/.system/skill-creator/`
- `~/.codex/sqlite/codex-dev.db`
  - Observed automation-related tables:
    - `automations`
    - `automation_runs`
    - `inbox_items`
  - Observed schema highlights:
    - `automations`: `name`, `prompt`, `status`, `cwds`, `rrule`, `model`, `reasoning_effort`
    - `automation_runs`: `thread_id`, `automation_id`, `status`, `thread_title`, `source_cwd`, `archived_user_message`, `archived_assistant_message`, `archived_reason`
    - `inbox_items`: `title`, `description`, `thread_id`, `read_at`
  - On this machine, all three tables were present but currently empty
- `~/.codex/automations/`
  - Directory exists
  - No child files were observed on this machine

### Standalone App Shell Roots

- `~/Library/Application Support/Codex/`
  - Observed Chromium or Electron-style support files:
    - `Local Storage/leveldb/`
    - `Session Storage/`
    - `Cookies`
    - `DIPS`
    - `SharedStorage`
    - `sentry/session.json`
    - `sentry/scope_v3.json`
    - `sentry/queue/queue-v2.json`
    - `Cache/`
    - `Code Cache/`
    - `GPUCache/`
  - `Preferences` is JSON, but only exposed two top-level keys on this machine:
    - `migrated_user_scripts_toggle`
    - `spellcheck`
  - `SharedStorage` is an SQLite database but had no tables at inspection time
  - `DIPS` is an SQLite database with generic browser interaction tables:
    - `bounces`
    - `config`
    - `meta`
    - `popups`
  - No transcript-like JSONL, no worktree table, and no Codex thread registry were observed under this app-support root
- `~/Library/Application Support/Codex/Local Storage/leveldb/`
  - Key-like strings observed included:
    - `react-resizable-panels:thread-vertical`
    - `codex_vsce_default_model_slug`
    - `statsig.session_id...`
    - model slug strings such as `gpt-5.3-codex`
  - Large feature-gate and telemetry payloads were also visible
  - This looks like UI state, feature flags, and runtime cache, not canonical transcript storage
- `~/Library/Logs/com.openai.codex/YYYY/MM/DD/codex-desktop-*.log`
  - Daily log files exist and are populated
  - Observed log methods included:
    - `app/list`
    - `skills/list`
    - `mcpServerStatus/list`
    - `turn/start`
    - `thread/metadata/update`
  - Observed warnings showed the app resolving git origins and worktrees for workspace paths and logging failures when paths were missing
  - These logs are useful for diagnostics and app behavior, but they are not clean transcript sources
- `~/Library/Preferences/com.openai.codex.plist`
  - Observed generic updater and macOS UI preference keys such as Sparkle update timestamps and open-panel defaults
  - No thread or message content was observed here
- `~/Library/Application Support/OpenAI/ChatGPT Atlas/`
  - Root exists, but only `NativeMessagingHosts/` was observed
  - No Codex transcript-like storage was identified there on this machine

## Desktop Session JSONL Structure

Across the `257` Desktop-originated rollout files observed on this machine, the dominant in-file line types were:

- `response_item`: `88,426`
- `event_msg`: `43,211`
- `turn_context`: `5,349`
- `compacted`: `46`

Observed `response_item.payload.type` categories:

- `function_call`: `25,791`
- `function_call_output`: `25,634`
- `reasoning`: `16,924`
- `message`: `12,924`
- `custom_tool_call`: `2,837`
- `custom_tool_call_output`: `2,837`
- `web_search_call`: `1,467`
- `tool_search_call`: `6`
- `tool_search_output`: `6`

Observed `message` roles:

- `assistant`: `8,747`
- `user`: `3,562`
- `developer`: `615`

Observed `message.content[].type` values:

- `output_text`: `8,747`
- `input_text`: `5,567`
- `input_image`: `247`

Observed `turn_context.payload` keys across Desktop sessions included:

- `cwd`
- `approval_policy`
- `sandbox_policy`
- `model`
- `personality`
- `collaboration_mode`
- `effort`
- `summary`
- `user_instructions`
- `truncation_policy`
- `developer_instructions`
- `turn_id`
- `current_date`
- `timezone`
- `realtime_active`

Observed `event_msg.payload.type` categories:

- `token_count`: `23,530`
- `agent_message`: `8,712`
- `agent_reasoning`: `4,191`
- `user_message`: `2,449`
- `task_started`: `2,172`
- `task_complete`: `1,970`
- `turn_aborted`: `103`
- `context_compacted`: `46`
- `item_completed`: `35`
- `thread_rolled_back`: `3`

## Collector Notes

- The safest transcript extraction rule remains: keep only `response_item` rows where `payload.type == "message"`.
- Within those message rows, `payload.role` can be at least `user`, `assistant`, and `developer`.
- `event_msg` rows are rich in lifecycle and accounting metadata, but they are not a clean transcript body.
- `function_call`, `function_call_output`, `custom_tool_call`, `custom_tool_call_output`, `web_search_call`, and `tool_search_*` rows are execution noise and should be excluded from memory-oriented output.
- `reasoning` rows should also be excluded from collector output.
- `turn_context` rows are useful provenance and environment metadata, especially for `cwd`, approvals, sandboxing, and model attribution, but they are not message content.
- App log files under `~/Library/Logs/com.openai.codex/` expose internal RPC methods and diagnostic failures, which should not be misclassified as conversation transcript.
- Collector implementation should join `~/.codex/state_5.sqlite threads` with `~/.codex/sqlite/codex-dev.db automation_runs` to separate interactive Desktop threads from automation-origin conversations.
- `~/.codex/automations/*/automation.toml` and `automations` table fields are useful provenance for automation name, schedule, model, and reasoning effort, but they should not replace canonical transcript text.
- Archived automation runs should treat rollout JSONL as canonical first and use `archived_user_message` or `archived_assistant_message` only when the rollout no longer contains the corresponding user or assistant body.

## Shared vs Independent Storage

This machine does not support an app-only storage conclusion.

Evidence points to a split model:

- Shared Codex-family state lives under `~/.codex/`
  - Desktop sessions are identifiable there by `session_meta.payload.originator == "Codex Desktop"`
  - The same shared root also contains `source` values such as `vscode`, `exec`, and serialized subagent spawn objects
  - Thread IDs, rollout paths, git metadata, skills, and automation tables are all present in this shared root
- Standalone-app shell state lives under `~/Library/Application Support/Codex/`, `~/Library/Logs/com.openai.codex/`, `~/Library/Preferences/com.openai.codex.plist`, and `~/Library/Caches/com.openai.codex/`
  - These roots look like Electron support state, browser storage, telemetry, and diagnostics
  - They do not appear to be the canonical transcript store on this machine

Working conclusion:

- `~/.codex/` is a shared Codex-family storage root used by the standalone app and other execution contexts
- The standalone app also maintains independent macOS shell-support roots under `~/Library/...`, but those roots supplement the shared transcript store rather than replace it

## Negative Results

- No `IndexedDB` directory was observed under `~/Library/Application Support/Codex/`
- No transcript-like JSONL files were observed under `~/Library/Application Support/Codex/`
- No Codex-specific thread registry database was observed under `~/Library/Application Support/Codex/`
- `~/Library/Application Support/OpenAI/ChatGPT Atlas/` did not expose Codex transcript artifacts on this machine
- No `*.patch`, `*.diff`, or `*.bundle` files were observed under `~/.codex/`

## Cross-Platform Note

Only macOS was directly inspected.

- Windows path translation guess, unverified:
  - app shell roots under `%APPDATA%\\Codex\\` and related `%LOCALAPPDATA%\\com.openai.codex\\...`
  - shared Codex-family root under `%USERPROFILE%\\.codex\\`
- Linux path translation guess, unverified:
  - app shell roots under `$XDG_CONFIG_HOME/Codex` or `~/.config/Codex`
  - shared Codex-family root under `$HOME/.codex/`

Do not treat those non-macOS paths as confirmed until they are inspected on real hosts.
