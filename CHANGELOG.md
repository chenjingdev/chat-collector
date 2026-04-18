# Changelog

All notable changes to this project should be recorded in this file.

Each tagged release must have a matching `## [X.Y.Z] - YYYY-MM-DD` section.
`scripts/render_release_notes.py` reads that exact section and uses it as the
GitHub release body for `vX.Y.Z` tags.

## [Unreleased]

### Added

- Added a read-only `tui` command for operator triage across latest runs,
  degraded sources, archive digest/stats/profile summaries, and sampled
  conversation drill-down.

## [0.1.0] - 2026-03-20

### Added

- Fixed `uv sync --frozen` followed by
  `uv build --sdist --wheel --out-dir dist --no-build-isolation` as the
  canonical wheel/sdist build path for versioned releases.
- Added `scripts/release_preflight.sh` to run tests, inspect artifacts, and
  confirm `uv tool install` works from the built wheel.
- Added a tag-triggered GitHub Actions release workflow that builds release
  assets and publishes them with notes extracted from this changelog.

### Changed

- Documented the version bump, tag naming, artifact inspection, and release
  ownership rules in `docs/releasing.md`.
