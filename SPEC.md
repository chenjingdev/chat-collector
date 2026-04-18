# chat-collector v0.1 SPEC

Status: frozen operating model for v0.1 checkout-based product readiness

## Product Intent

- `chat-collector` exists to collect chats stored locally by coding agents and IDE assistants.
- The repository stores collector code, docs, tests, and fixtures.
- Real collected chat archives must live outside the repository at `/Users/chenjing/dev/chat-history`.
- Do not store real collected chat output in this repository unless the work is explicitly about fixtures or tests.
- Normalize archives for memory usefulness rather than raw forensic completeness.
- Exclude tool calls, MCP invocation noise, internal reasoning traces, and execution artifacts that do not help later memory inference.

## Source Scope

The authoritative per-source status lives in `docs/source-support-matrix.md` and the registry metadata that generates it.

Current v0.1 source families and support levels:

| Family | Source keys | Current support |
| --- | --- | --- |
| Codex | `codex_cli`, `codex_app`, `codex_ide_extension` | `complete` |
| Claude Code | `claude`, `claude_code_ide` | `complete` |
| Gemini | `gemini` | `complete` |
| Antigravity | `antigravity_editor_view` | `partial` |
| Cursor | `cursor`, `cursor_editor` | `partial` |
| Gemini Code Assist | `gemini_code_assist_ide` | `partial` |
| Windsurf | `windsurf_editor` | `partial` |

Selection profile meanings are fixed as follows:

- `default`
  - unattended-ready batch selection
  - includes only `complete` collectors
  - same concrete source set as `complete_only` today
- `complete_only`
  - explicit complete-only selection for tickets, docs, and release notes that need to spell out that boundary
  - currently resolves to the same concrete source set as `default`
- `all`
  - opt-in selection that lowers the minimum support bar to `scaffold`
  - includes partial collectors as well as complete collectors

`default` and `complete_only` are intentionally distinct names even while they resolve to the same set in v0.1. The semantic difference is operator intent, not current implementation spread.

## v0.1 Product-Ready Bar

v0.1 is considered checkout-based product-ready only when all three gates
below are satisfied without contradicting this spec:

1. `checkout-based operator flow`
   - `README.md` documents the local checkout install/config/init/collect/inspect/export path around `uv sync` and `uv run`
   - the flow keeps real archives on an external root and does not depend on wheel installs, release tags, GitHub releases, or browser verification
2. `one-pass smoke`
   - pass the install-to-first-memory-export operator path documented in `README.md`
   - the flow must be reproducible from a local checkout without extra oral context
3. `ship-acceptance`
   - pass the clean external-root flow defined in `docs/ship-acceptance.md`

The current checkout-based completion chain at the time this spec is frozen is:

- `CHE-145` for `one-pass smoke`
- `CHE-144` for `ship-acceptance`
- `CHE-161` for product-ready docs and completion-bar alignment

Workers may refine commands, docs, and acceptance evidence inside those
tickets, but they must not silently redefine the three gates above. If the
product-ready bar changes, update this file first.

## Publication Follow-up Outside the Product Bar

Versioned release verification, wheel-install smoke, release tags, GitHub
release creation, and browser-side release confirmation live in
`docs/releasing.md`.

Those publication operations are intentionally outside the v0.1 product-ready
bar and currently belong to `CHE-147` or its explicit replacement follow-up.
They may reuse the same one-pass smoke and ship-acceptance evidence, but they
must not become a prerequisite for declaring the checkout product-ready.

## Known Limitations Intentionally Kept in v0.1

- Partial sources remain opt-in and may stay degraded or metadata-heavy.
- The default unattended batch does not promise transcript completeness for every registered source.
- Memory-oriented normalization intentionally drops raw forensic detail that is not useful for later memory inference.
- Real archive output remains external-only; the repository keeps only fixtures, tests, and redacted examples.
- v0.1 does not promise new source collectors, unified cross-source stitching, or broader workflow expansion beyond the current operating model.

## Out of Scope for This Spec Version

- Implementing new source collectors just to expand the support matrix
- Storing real collected operator archives in the repository
- Expanding the tracker into a large speculative backlog
- Introducing PR/review/merge workflow stages for the default delivery path

## Default Decision Rules

- Treat this file as the frozen product boundary for v0.1 checkout-based product-ready.
- Treat `WORKFLOW.md` as the frozen orchestration boundary.
- Treat each Linear issue body as the immediate execution boundary.
- Treat `docs/releasing.md` as the publication follow-up workflow, not the default completion bar.
- If a ticket needs to change product intent, source scope, selection profile meaning, or product-ready bar, update `SPEC.md` before or with that ticket.
