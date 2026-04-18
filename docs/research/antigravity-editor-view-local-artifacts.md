# Antigravity Editor View Local Artifacts

Updated: `2026-03-20`

Scope: local artifact reconnaissance for `Antigravity` in `ide_native` execution context on macOS. This note stops at storage-path and format identification. It does not implement a collector or parser.

## High-Signal Summary

- Installed macOS app observed at `/Applications/Antigravity.app`
- Observed bundle provenance: `com.google.antigravity`, version `1.20.5`, `Google LLC`, executable `Electron`
- Primary local roots:
  - `~/Library/Application Support/Antigravity`
  - `~/.gemini/antigravity`
- Strongest transcript candidate: `~/.gemini/antigravity/conversations/<uuid>.pb`
- Current operator-machine finding: all `42` sampled conversation blobs degraded to explicit `variant_unknown` plus `decode_failed` diagnostics under the confirmed raw protobuf parser
- Companion session-family artifacts:
  - `~/.gemini/antigravity/brain/<uuid>/...`
  - `~/.gemini/antigravity/annotations/<uuid>.pbtxt`
  - `~/.gemini/antigravity/browser_recordings/<uuid>/*.jpg`
- Shared-storage evidence exists: Editor View and Manager-related state co-reside in `~/Library/Application Support/Antigravity/User/globalStorage/state.vscdb` under `antigravityUnifiedStateSync.*`
- Negative result: `chat.ChatSessionStore.index` was empty across inspected `User/workspaceStorage/*/state.vscdb` databases, including the workspace for `/Users/chenjing/dev/chat-collector`
- Message-level fields such as `role`, `timestamp`, and `content` were not directly confirmed because the most likely conversation store is an opaque binary `*.pb` format
- The March 20 sample of real operator blobs did not expose any stable protobuf field framing at offset `0..64`; first-tag wire types clustered at unsupported values `3`, `4`, `6`, and `7`

## Provenance And Install Layout

`/Applications/Antigravity.app/Contents/Info.plist` showed:

- `CFBundleIdentifier = com.google.antigravity`
- `CFBundleDisplayName = Antigravity`
- `CFBundleShortVersionString = 1.20.5`
- `NSHumanReadableCopyright = Google LLC`
- `CFBundleExecutable = Electron`

This confirms an Electron desktop app and supports a VS Code fork-style storage layout, but that structure was verified from the local install rather than assumed from public docs alone.

The bundled extension tree at `/Applications/Antigravity.app/Contents/Resources/app/extensions/antigravity/` also exposed Antigravity-specific commands and APIs, including `antigravity.openConversationPicker` and `antigravityUnifiedStateSync`, which matches the state observed under `Application Support`.

## Observed macOS Roots

Primary roots with relevant state:

- `~/Library/Application Support/Antigravity`
- `~/.gemini/antigravity`

Secondary support roots also observed:

- `~/Library/Caches/com.google.antigravity`
- `~/Library/Caches/com.google.antigravity.ShipIt`
- `~/Library/Preferences/com.google.antigravity.plist`

The two primary roots serve different roles:

- `~/Library/Application Support/Antigravity` behaves like the Electron and VS Code-fork host state root
- `~/.gemini/antigravity` holds Antigravity-specific session artifacts, browser automation residue, and per-trajectory companion files

## Application Support Findings

Observed subtrees under `~/Library/Application Support/Antigravity` included:

- `User/globalStorage`
- `User/workspaceStorage`
- `User/History`
- `logs`
- `Local Storage/leveldb`
- `Session Storage`
- `WebStorage`
- `shared_proto_db`
- `Preferences`
- `Workspaces`

This is consistent with a VS Code-derived host layout.

### Logs

The `logs` tree contained rotating files such as:

- `antigravity-interactive-editor.log`
- `artifacts.log`
- `auth.log`
- `cloudcode.log`
- `editSessions.log`
- `main.log`
- `telemetry.log`
- `terminal.log`
- `window*/renderer.log`
- `window*/exthost/google.antigravity/Antigravity.log`

High-value observations from those logs:

- `antigravity-interactive-editor.log` recorded `Watching for HTML files in: file:///Users/chenjing/.gemini/antigravity/html_artifacts`
- `main.log` recorded a localhost onboarding server such as `http://localhost:50132`
- `~/.gemini/antigravity/daemon/ls_c318d4f90fc5aacc.log` recorded local language-server activity, `Entering local chrome mode`, and repeated `SendActionToChatPanel` errors

These log files are execution noise and diagnostics, not clean transcript material.

### globalStorage/state.vscdb

`~/Library/Application Support/Antigravity/User/globalStorage/state.vscdb` contained the following high-signal keys:

- `antigravityUnifiedStateSync.agentManagerWindow`
- `antigravityUnifiedStateSync.agentPreferences`
- `antigravityUnifiedStateSync.artifactReview`
- `antigravityUnifiedStateSync.browserPreferences`
- `antigravityUnifiedStateSync.editorPreferences`
- `antigravityUnifiedStateSync.enterprisePreferences`
- `antigravityUnifiedStateSync.modelCredits`
- `antigravityUnifiedStateSync.modelPreferences`
- `antigravityUnifiedStateSync.oauthToken`
- `antigravityUnifiedStateSync.onboarding`
- `antigravityUnifiedStateSync.overrideStore`
- `antigravityUnifiedStateSync.scratchWorkspaces`
- `antigravityUnifiedStateSync.seenNuxIds`
- `antigravityUnifiedStateSync.sidebarWorkspaces`
- `antigravityUnifiedStateSync.tabPreferences`
- `antigravityUnifiedStateSync.theme`
- `antigravityUnifiedStateSync.trajectorySummaries`
- `antigravityUnifiedStateSync.userStatus`
- `antigravityUnifiedStateSync.windowPreferences`
- `chat.ChatSessionStore.index`
- `chat.workspaceTransfer`
- `google.antigravity`
- `google.geminicodeassist`

Observed value-level clues:

- `google.antigravity` contained a stable installation identifier
- `chat.workspaceTransfer` was `[]`
- `antigravityUnifiedStateSync.agentManagerWindow` looked like base64 or protobuf-like state carrying window geometry fields such as `managerWidth`, `managerHeight`, `managerX`, and `managerWindowMode`
- `antigravityUnifiedStateSync.browserPreferences` contained allowlist-related strings and a browser sentinel key
- `antigravityUnifiedStateSync.sidebarWorkspaces` embedded real workspace URIs including `file:///Users/chenjing/dev/chat-collector`
- `antigravityUnifiedStateSync.artifactReview` embedded `~/.gemini/antigravity/brain/<uuid>/task.md`, `walkthrough.md`, and `implementation_plan.md` references

Working conclusion:

- Editor View does not appear to use a storage root isolated from Manager View
- At minimum, Editor View and Manager-related orchestration state share the same `Application Support/Antigravity` global store and point into the same `~/.gemini/antigravity` artifact family

### workspaceStorage

Observed workspace state databases lived under:

- `~/Library/Application Support/Antigravity/User/workspaceStorage/<workspace-id>/state.vscdb`

For the target repo workspace:

- Workspace ID: `9823a9f902ed79fa5ed582051518c14c`
- `workspace.json` pointed at `file:///Users/chenjing/dev/chat-collector`

Useful keys in inspected workspace databases:

- `antigravity.agentViewContainerId.numberOfVisibleViews`
- `antigravity.agentViewContainerId.state`
- `chat.ChatSessionStore.index`
- `chat.customModes`
- `history.entries`
- `memento/antigravity.jetskiArtifactsEditor`
- `memento/antigravity.antigravityReviewChangesEditor`
- `memento/multiDiffEditor`

Observed behavior:

- `chat.ChatSessionStore.index` was `{"version":1,"entries":{}}` in every inspected workspace database on this machine
- `antigravity.agentViewContainerId.state` stored panel visibility and sizing state for the side panel and panel
- `memento/antigravity.jetskiArtifactsEditor` referenced `~/.gemini/antigravity/brain/<uuid>/implementation_plan.md.resolved`, `task.md.resolved`, and `walkthrough.md.resolved`
- `history.entries` captured recently opened files and resource URIs, not a message transcript

Collector implication:

- The VS Code-style `chat.ChatSessionStore.index` should not be treated as the primary transcript source for Antigravity Editor View on this machine

## ~/.gemini/antigravity Findings

Observed top-level directories and files under `~/.gemini/antigravity` included:

- `annotations`
- `brain`
- `browserAllowlist.txt`
- `browserOnboardingStatus.txt`
- `browser_recordings`
- `code_tracker`
- `context_state`
- `conversations`
- `daemon`
- `global_skills`
- `html_artifacts`
- `implicit`
- `installation_id`
- `knowledge`
- `mcp_config.json`
- `playground`
- `scratch`
- `user_settings.pb`

### conversations

Observed `42` binary conversation files:

- Path pattern: `~/.gemini/antigravity/conversations/<uuid>.pb`

This is the strongest candidate for real session transcript storage. However:

- the files were binary and not directly readable as JSON, SQLite, or plain-text protobuf with trivial decoding
- ad hoc `strings`, `xxd`, and lightweight protobuf-wire inspection did not recover stable message fields
- all `42` sampled operator-machine blobs failed the confirmed raw protobuf parser before any stable top-level field shape was recovered
- decode failures clustered around unsupported first-tag wire types `7` (`12` blobs), `4` (`11` blobs), `6` (`9` blobs), and `3` (`8` blobs), with one oversize-varint case and one truncated-length case
- scanning candidate offsets `0..64` did not recover a second obvious protobuf wrapper with low-numbered field tags, which suggests opaque framing or encryption rather than a trivial fixed header shift

Working conclusion:

- `conversations/<uuid>.pb` is the most likely source for message-level transcript extraction
- the previously confirmed raw protobuf mapping remains valid for synthetic and fixture-backed variants, but the current operator-machine corpus should be treated as `variant_unknown` with explicit `decode_failed` diagnostics until a new framing or key path is confirmed

### brain

Observed `43` trajectory directories:

- Path pattern: `~/.gemini/antigravity/brain/<uuid>/...`

Typical files included:

- `task.md`
- `task.md.metadata.json`
- `task.md.resolved`
- `implementation_plan.md`
- `implementation_plan.md.metadata.json`
- `implementation_plan.md.resolved`
- `walkthrough.md`
- `walkthrough.md.metadata.json`
- `walkthrough.md.resolved`
- image artifacts such as `.png` and `.webp`

Observed metadata fields included:

- `artifactType`
- `summary`
- `updatedAt`
- `version`

Observed artifact types included:

- `ARTIFACT_TYPE_TASK`
- `ARTIFACT_TYPE_IMPLEMENTATION_PLAN`
- `ARTIFACT_TYPE_WALKTHROUGH`

These are trajectory artifacts and summaries, not a raw bidirectional chat log.

### annotations

Observed `41` files with this shape:

- Path pattern: `~/.gemini/antigravity/annotations/<uuid>.pbtxt`

Observed content was minimal, for example a `last_user_view_time` timestamp. This looks like per-trajectory annotation metadata rather than transcript text.

### browser_recordings

Observed `7` directories:

- Path pattern: `~/.gemini/antigravity/browser_recordings/<uuid>/*.jpg`

These held large numbers of timestamped image frames and appear to be browser-automation evidence. They are not memory-relevant transcript content and should be filtered out by default.

### code_tracker, daemon, html_artifacts

- `code_tracker/active/...` stored repo-linked file snapshots and code fragments, not conversational turns
- `daemon/*.log` and `daemon/*.json` stored local service discovery and runtime diagnostics
- `html_artifacts/` existed as a watched root, but it was empty at inspection time
- `context_state/` did not provide a confirmed transcript structure during this inspection

## Session Family Correlation

UUID overlap across observed roots:

- `conversations`: `42`
- `brain`: `43`
- `annotations`: `41`
- `browser_recordings`: `7`
- `implicit`: `0`

Observed overlap counts:

- `conversations` vs `brain`: `42`
- `conversations` vs `annotations`: `33`
- `conversations` vs `browser_recordings`: `4`
- `conversations` vs `implicit`: `0`
- `brain` vs `browser_recordings`: `4`

Collector implication:

- `uuid` is a strong candidate session-family key across transcript, artifact, annotation, and browser recording stores

## Noise Separation Notes

The following should not be mistaken for clean transcript messages:

- `logs/**/*.log`
- `daemon/*.log`
- `browser_recordings/**/*.jpg`
- `code_tracker/**`
- `Local Storage/leveldb`
- `Session Storage`
- `shared_proto_db`
- browser onboarding and allowlist files
- MCP, tool, shell, language-server, and browser-automation runtime output

Even if some of these contain text fragments, they are better treated as provenance or execution residue than as memory-safe chat turns.

## Shared vs Independent Storage

Current best conclusion for macOS:

- Antigravity Editor View is not using an entirely separate top-level transcript root
- Editor View host state and Manager-related orchestration state share `~/Library/Application Support/Antigravity/User/globalStorage/state.vscdb`
- The same shared state points into `~/.gemini/antigravity/brain/<uuid>/...`, which suggests a common session-family store across surfaces
- Storage separation, if any, is more likely encoded in per-surface keys, artifact types, or opaque conversation payloads than in distinct top-level directories

This should guide the later `antigravity_manager_view` ticket: treat Manager View as likely sharing roots with Editor View, not as guaranteed isolated storage.

## Negative Results

- No confirmed JSON or SQLite message store with directly readable `role`, `timestamp`, and `content` fields was identified
- No populated `chat.ChatSessionStore.index` entries were found in inspected workspace databases
- `html_artifacts/` existed but was empty
- `Local Storage/leveldb`, `Session Storage`, and `shared_proto_db` did not yield a confirmed transcript schema during this inspection

## Cross-Platform Note

Only macOS was directly inspected.

Unverified path translations worth checking later:

- Windows:
  - `%APPDATA%\\Antigravity`
  - `%USERPROFILE%\\.gemini\\antigravity`
- Linux:
  - `~/.config/Antigravity`
  - `~/.gemini/antigravity`

Do not treat those non-macOS paths as confirmed until they are verified on real hosts.
