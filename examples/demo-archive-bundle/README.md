# Demo Archive Bundle

This directory contains a redacted operator demo portable export bundle for
`archive import`.

It is intentionally separate from `tests/fixtures/`:

- `examples/demo-archive-bundle/` is for operator walkthroughs and local demos.
- `tests/fixtures/` is for automated source parser and unit test inputs.

The bundle contains three normalized conversations:

| Source | Session | Transcript completeness |
| --- | --- | --- |
| `codex_cli` | `demo-codex-complete` | `complete` |
| `cursor_editor` | `demo-cursor-partial` | `partial` |
| `gemini_code_assist_ide` | `demo-gemini-unsupported` | `unsupported` |

Use [docs/demo-archive-walkthrough.md](../../docs/demo-archive-walkthrough.md)
for the end-to-end operator walkthrough.
