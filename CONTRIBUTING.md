# Contributing

## CI validation

The default CI workflow lives in `.github/workflows/ci.yml`.

- Pull requests and pushes to `main` run `uv sync --frozen` followed by `uv run pytest` on Linux.
- A separate fixture smoke matrix runs on Linux and macOS so source-specific failures stay visible in the job list.
- The smoke matrix covers these commands:
  - `uv run pytest tests/test_ci_fixture_smoke.py -k test_codex_cli_fixture_collect_smoke -q`
  - `uv run pytest tests/test_ci_fixture_smoke.py -k test_cursor_fixture_collect_smoke -q`
  - `uv run pytest tests/test_ci_fixture_smoke.py -k test_gemini_code_assist_ide_fixture_collect_smoke -q`
  - `uv run pytest tests/test_ci_fixture_smoke.py -k test_archive_commands_fixture_smoke -q`

The smoke tests use repository fixtures under `tests/fixtures/` and write their temporary archive output to pytest-managed temporary directories instead of the real archive root.

## Local reproduction

Use the same `uv` workflow locally before opening a PR:

```bash
uv sync
uv run pytest
uv run pytest tests/test_ci_fixture_smoke.py -q
```

To narrow failures to a single CI smoke path, run the matching selector locally:

```bash
uv run pytest tests/test_ci_fixture_smoke.py -k test_cursor_fixture_collect_smoke -q
```
