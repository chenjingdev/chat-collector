# Releasing `llm-chat-archive`

This project supports versioned releases as wheel and sdist artifacts built
from `pyproject.toml` with `uv` and `hatchling`.

This document covers publication work only. The checkout-based operator flow in
`README.md` defines the v0.1 product-ready bar, while the steps here stay
outside that bar as the follow-up publication track currently assigned to
`CHE-147`.

## Publication Install Path

External users are expected to install from a versioned release wheel.

```bash
uv tool install --python 3.12 \
  "https://github.com/chenjingdev/chat-collector/releases/download/vX.Y.Z/llm_chat_archive-X.Y.Z-py3-none-any.whl"
```

For a locally built artifact, use the exact wheel from `dist/`:

```bash
uv tool install --python 3.12 --force ./dist/llm_chat_archive-X.Y.Z-py3-none-any.whl
```

Both commands install the `llm-chat-archive` executable into uv's tool bin
directory. Any explicit `3.11+` interpreter works; `3.12` is the release
baseline used in CI and the GitHub release workflow.

The local checkout operator path remains standardized on `uv sync` plus
`uv run` as documented in `README.md`. Use the wheel install path in this file
to verify the release artifact itself, not to redefine the default checkout
workflow or the v0.1 product-ready bar.

## Source of truth

- The release version lives in `[project].version` in `pyproject.toml`.
- The release notes live in `CHANGELOG.md`.
- The release tag must be `vX.Y.Z` and must match `[project].version`.
- The automated release workflow lives in `.github/workflows/release.yml`.

## Version bump rule

1. Update `pyproject.toml` `project.version`.
2. Add or update the matching `CHANGELOG.md` section as
   `## [X.Y.Z] - YYYY-MM-DD`.
3. Commit the release prep on `main`.
4. Create an annotated tag `vX.Y.Z` from that commit.

`CHANGELOG.md` is the canonical human-written release note source. The GitHub
release body is rendered directly from the matching changelog section so the
tag, artifact version, and release notes cannot drift silently.

## Preflight checks

Run the fixed preflight path before any release tag is pushed:

```bash
./scripts/release_preflight.sh
```

The script does all of the following:

- `uv sync --frozen`
- `uv run pytest`
- `uv run pytest tests/test_ci_fixture_smoke.py -q`
- `uv build --sdist --wheel --out-dir dist --no-build-isolation`
- verifies the expected `dist/` artifact names for the current version
- inspects wheel and sdist contents
- installs the wheel with `uv tool install` in a temporary HOME and runs
  `llm-chat-archive sources`

## Publication Checklist

Every publication candidate must pass both checks below before the publication
ticket moves to `Done`. This checklist is downstream from the product-ready bar
in `SPEC.md`; it does not decide whether the checkout itself is ready to use.

### 1. Install smoke

- `./scripts/release_preflight.sh` passes on a clean environment.
- The temporary `uv tool install` step can launch `llm-chat-archive sources`
  from the built wheel without importing the checkout.

### 2. First-run smoke

Run the installed binary through the fixed first-run path against a clean
external archive root:

```bash
llm-chat-archive config init --archive-root /Users/chenjing/dev/chat-history --force
llm-chat-archive doctor --all --profile default
llm-chat-archive collect --all
llm-chat-archive runs latest --archive-root /Users/chenjing/dev/chat-history
llm-chat-archive archive list --archive-root /Users/chenjing/dev/chat-history
llm-chat-archive archive show --archive-root /Users/chenjing/dev/chat-history --source <source> --session <source-session-id>
llm-chat-archive archive find --archive-root /Users/chenjing/dev/chat-history --text "<known text>"
llm-chat-archive archive export-memory \
  --archive-root /Users/chenjing/dev/chat-history \
  --run <run-id> \
  --output-dir /Users/chenjing/dev/chat-history/exports/release-candidate-memory \
  --execute
```

The checklist is complete only when all commands above succeed and the final
memory export writes both `memory-records.jsonl` and
`memory-export-manifest.json`.

## Manual artifact inspection

If a release candidate needs manual inspection, use these commands after
`./scripts/release_preflight.sh`:

```bash
uv run python -m zipfile -l dist/llm_chat_archive-X.Y.Z-py3-none-any.whl
tar -tzf dist/llm_chat_archive-X.Y.Z.tar.gz
```

The wheel should contain `llm_chat_archive/__init__.py`, and the sdist should
contain `pyproject.toml`, `CHANGELOG.md`, `docs/releasing.md`, and the package
source tree.

## Tag Workflow

Push the release commit and tag after preflight passes:

```bash
git push origin main
git tag -a vX.Y.Z -m "vX.Y.Z"
git push origin vX.Y.Z
```

When the tag reaches GitHub, `.github/workflows/release.yml`:

1. reruns `./scripts/release_preflight.sh` on a clean runner
2. renders release notes from `CHANGELOG.md`
3. creates or updates the GitHub release for that tag
4. uploads the wheel and sdist from `dist/`

If the workflow fails, do not move the publication ticket to `Done` until the
failure is understood and the tag is replaced with a corrected version.
