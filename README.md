# llm-chat-archive

`llm-chat-archive` collects local coding-agent chats and writes normalized archives
that are optimized for memory usefulness rather than raw forensic completeness.

The archive intentionally keeps human-facing conversation content and excludes
noise such as tool calls, MCP invocation residue, internal reasoning traces, and
execution artifacts that do not help later memory inference.

Real collected archives must live outside this repository. The default external
archive root is `/Users/chenjing/dev/chat-history`.

## Checkout-based quickstart

This README is the v0.1 checkout-based operator path. The versioned wheel/tag
flow in [docs/releasing.md](docs/releasing.md) belongs to the publication
follow-up tracked by `CHE-147` and stays outside the v0.1 product-ready bar.

Assume `archive_root` is `/Users/chenjing/dev/chat-history`.

1. Install the checkout with `uv`:

```bash
uv sync
```

2. Scaffold the default operator config:

```bash
uv run llm-chat-archive config init
```

This writes `~/.config/llm-chat-archive/collector.toml` and pins
`collect.archive_root` to `/Users/chenjing/dev/chat-history`.

Useful config-init variants:

```bash
uv run llm-chat-archive config init --archive-root /absolute/path/to/chat-history
uv run llm-chat-archive config init --print
```

The same template also lives at `examples/collector.sample.toml`.

3. Run the first checkout-based collection flow:

```bash
uv run llm-chat-archive sources
uv run llm-chat-archive doctor --all --profile default
uv run llm-chat-archive collect --all
```

4. Inspect the latest run and normalized archive rows:

```bash
uv run llm-chat-archive runs latest --archive-root /Users/chenjing/dev/chat-history
uv run llm-chat-archive archive list --archive-root /Users/chenjing/dev/chat-history
uv run llm-chat-archive archive show --archive-root /Users/chenjing/dev/chat-history --source codex_cli --session <source-session-id>
uv run llm-chat-archive archive find --archive-root /Users/chenjing/dev/chat-history --text "refactor"
```

5. Export the first memory bundle from that run:

```bash
uv run llm-chat-archive archive export-memory \
  --archive-root /Users/chenjing/dev/chat-history \
  --run <run-id> \
  --output-dir /Users/chenjing/dev/chat-history/exports/first-memory \
  --execute
```

6. Use the fixed acceptance flow to back the current product-ready bar:

```bash
uv run llm-chat-archive acceptance ship \
  --archive-root /Users/chenjing/dev/chat-history/ship-acceptance-20260320 \
  --snapshot-path "$(pwd)/examples/ship-acceptance-golden/ship-acceptance.json"
```

The archive root must be an absolute path outside the repository. Keep real
collection output under an external location such as
`/Users/chenjing/dev/chat-history`.

## Current source support

The registry-backed support matrix lives in
[docs/source-support-matrix.md](docs/source-support-matrix.md).

- `uv run llm-chat-archive sources` keeps the compact source/support/root listing.
- `uv run llm-chat-archive sources --format json` emits the machine-readable support summary.
- `uv run llm-chat-archive sources --format markdown` regenerates the Markdown matrix document.

Use `--profile default` for unattended-ready complete collectors. `--profile
complete_only` keeps the same complete-only source set with explicit intent,
and `--profile all` opts into partial collectors as well.

## Install and configure

The quickstart above is the fixed checkout-based product-ready flow. The
release-wheel `uv tool install` path in [docs/releasing.md](docs/releasing.md)
is reserved for publication verification and remains outside the v0.1
product-ready bar.

Sync the environment with `uv`:

```bash
uv sync
```

Scaffold the default config directly into the standard location:

```bash
uv run llm-chat-archive config init
```

This writes `~/.config/llm-chat-archive/collector.toml`, creates parent
directories when needed, refuses to overwrite an existing file unless you pass
`--force`, and pins `collect.archive_root` to the default external root
`/Users/chenjing/dev/chat-history`.

To scaffold the same template with a different external archive root:

```bash
uv run llm-chat-archive config init --archive-root /absolute/path/to/chat-history
```

To inspect the scaffold without writing a file:

```bash
uv run llm-chat-archive config init --print
```

The same template also lives at `examples/collector.sample.toml`. It pins:

- `collect.archive_root`
- `collect.selection.profile`
- `collect.incremental`
- `collect.redaction`
- `collect.validation`
- `rerun.selection_preset`
- `scheduled.mode`
- `scheduled.selection.profile`
- `scheduled.rerun.selection_preset`
- `scheduled.stale_after_seconds`

`collect.archive_root` is scaffolded as `/Users/chenjing/dev/chat-history`.
If you want a different external root, pass
`config init --archive-root /absolute/path/to/chat-history` or override at
runtime with `--archive-root`.

If you do not want to use the default config path, either scaffold somewhere
else with `config init --output /absolute/path/to/collector.toml` or pass an
absolute path with `--config /absolute/path/to/collector.toml`. For example:

```bash
uv run llm-chat-archive collect --all --config "$(pwd)/examples/collector.sample.toml"
```

## Operator quickstart

Assume `archive_root` is `/Users/chenjing/dev/chat-history`.

### One-pass first-run smoke

This is the fixed install-to-first-memory-export path for a local checkout.

1. Sync the checkout:

```bash
uv sync
```

2. Scaffold the default operator config:

```bash
uv run llm-chat-archive config init
```

3. List registered collectors and their default artifact roots:

```bash
uv run llm-chat-archive sources
```

4. Run a no-write readiness check before the first batch:

```bash
uv run llm-chat-archive doctor --all --profile default
```

5. Run the first batch collection:

```bash
uv run llm-chat-archive collect --all
```

`collect --all` executes the batch immediately, writes normalized source output
to the archive root, and records a run manifest. For a single source, `collect
<source>` emits a plan only; add `--execute` to write output.

6. Inspect the latest recorded run and keep the emitted `run_id`:

```bash
uv run llm-chat-archive runs latest --archive-root /Users/chenjing/dev/chat-history
```

7. Inspect normalized archive rows:

```bash
uv run llm-chat-archive archive list --archive-root /Users/chenjing/dev/chat-history
uv run llm-chat-archive archive show --archive-root /Users/chenjing/dev/chat-history --source codex_cli --session <source-session-id>
uv run llm-chat-archive archive find --archive-root /Users/chenjing/dev/chat-history --text "refactor"
```

Use `archive list` or `archive find` to get the `source` and `source_session_id`
pair needed by `archive show`.

8. Export the first memory bundle from that run:

```bash
uv run llm-chat-archive archive export-memory \
  --archive-root /Users/chenjing/dev/chat-history \
  --run <run-id> \
  --output-dir /Users/chenjing/dev/chat-history/exports/first-memory \
  --execute
```

The command writes `memory-records.jsonl` and
`memory-export-manifest.json` into the chosen output directory.

### Ongoing operator commands

9. Run the scheduler-safe entrypoint that external cron/timer jobs can repeat:

```bash
uv run llm-chat-archive scheduled run
```

`scheduled run` uses the `[scheduled]` config preset, writes the same manifest
shape as `collect --all`, and acquires `<archive-root>/runs/.scheduled.lock`
before any collection work starts.

If the lock is active, the command exits `1` with JSON reason
`scheduled_lock_held`. If the lock age is older than
`scheduled.stale_after_seconds`, the command exits `1` with JSON reason
`scheduled_lock_stale`. Replace only stale locks with:

```bash
uv run llm-chat-archive scheduled run --force-unlock-stale
```

10. Run the fixed ship-acceptance flow on a clean archive root:

```bash
uv run llm-chat-archive acceptance ship \
  --archive-root /Users/chenjing/dev/chat-history/ship-acceptance-20260320 \
  --snapshot-path "$(pwd)/examples/ship-acceptance-golden/ship-acceptance.json"
```

This command pins the current operator acceptance source set, runs `collect`,
`validate`, `archive verify`, `archive digest`, `archive export`, and
`archive export-memory` end to end, and writes redacted export bundles under
`<archive-root>/acceptance/`. The optional snapshot path stores a repo-safe
golden summary without copying raw operator archive rows into the repository.

11. Inspect a specific recorded run:

```bash
uv run llm-chat-archive runs show --archive-root /Users/chenjing/dev/chat-history <run-id>
```

12. Inspect archive index state when list/find/profile output looks stale:

```bash
uv run llm-chat-archive archive index status --archive-root /Users/chenjing/dev/chat-history
uv run llm-chat-archive archive index refresh --archive-root /Users/chenjing/dev/chat-history
```

13. Open the operator triage TUI:

```bash
uv run llm-chat-archive tui --archive-root /Users/chenjing/dev/chat-history
```

Use `--view runs`, `--view sources`, or `--view samples` to start on a specific
screen. For a headless plain-text snapshot instead of curses, add
`--snapshot --view overview`.

`archive list`, `archive find`, `archive stats`, and `archive profile` refresh a
rebuildable SQLite query index under `<archive-root>/archive-index/` when the
index is missing or stale. Raw JSONL files remain the authoritative archive
storage.

For operator-side troubleshooting and output interpretation, see
[docs/operator-troubleshooting.md](docs/operator-troubleshooting.md) and
[docs/operator-terminal-tui.md](docs/operator-terminal-tui.md). For the fixed
ship-acceptance profile, pass/fail criteria, and golden snapshot workflow, see
[docs/ship-acceptance.md](docs/ship-acceptance.md).

## Demo archive walkthrough

A redacted operator demo bundle lives in
`examples/demo-archive-bundle/`. It is a portable export bundle for
`archive import` and is intentionally separate from `tests/fixtures/`.

Follow [docs/demo-archive-walkthrough.md](docs/demo-archive-walkthrough.md)
to import the bundle into a temporary external archive root and run
`archive list`, `archive show`, `archive find`, `archive stats`,
`archive profile`, `archive digest`, and `archive verify` end to end.

## Concepts

`redaction`
: Redacts credentials and similar secrets before archive rows are written.

`validation`
: Verifies the recorded batch manifest and source outputs after `collect --all`
or `rerun`. `off` skips validation, `report` includes the validation summary in
the JSON output, and `strict` also fails the command when validation reports
errors.

`incremental`
: Deduplicates against existing archive rows before new conversations are
written, so repeated unattended runs can append safely.

`rerun`
: Replays a source subset derived from a previous batch run. With the sample
config, `uv run llm-chat-archive rerun --run <run-id>` defaults to rerunning
failed and degraded sources. Use `--reason failed`, `--reason degraded`, or
`--reason failed_or_degraded` to override the preset from the CLI.

`scheduled run`
: Non-interactive runner for cron, launchd, or systemd timer style invocations.
  It reuses the `[scheduled]` config preset, records `scheduled` metadata in the
  run manifest, and prevents overlap with a lock file under
  `<archive-root>/runs/.scheduled.lock`.

`stale lock`
: A scheduled lock becomes stale only when its age exceeds
  `scheduled.stale_after_seconds`. The runner never breaks active locks. Use
  `scheduled run --force-unlock-stale` only after the CLI reports
  `scheduled_lock_stale`.

## Config lookup and override order

Collector config is used by `collect` and `rerun`.

1. CLI flags win over config values.
   Examples: `--archive-root`, `--profile`, `--source`, `--exclude-source`,
   `--incremental`, `--redaction`, `--validation`, and rerun `--reason`.
2. If `--config` is provided, it must be an absolute path and it becomes the
   config source of truth for that command.
3. If `--config` is omitted, the CLI looks for the default path:
   `~/.config/llm-chat-archive/collector.toml`
4. If neither config path is available, built-in defaults are used.

`runs` and `archive` commands do not read the collector config. They only use
`--archive-root`, which defaults to `/Users/chenjing/dev/chat-history`.

## Archive root rule

The archive root must be an absolute path outside the repository. Do not point
real collection output at any path inside this checkout. Keep fixtures and tests
inside the repo, and keep real collected chat history under an external location
such as `/Users/chenjing/dev/chat-history`.
