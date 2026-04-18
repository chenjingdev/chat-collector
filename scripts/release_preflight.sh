#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

version="$(
  uv run python - <<'PY'
import pathlib
import tomllib

pyproject = tomllib.loads(pathlib.Path("pyproject.toml").read_text(encoding="utf-8"))
print(pyproject["project"]["version"])
PY
)"

wheel_path="dist/llm_chat_archive-${version}-py3-none-any.whl"
sdist_path="dist/llm_chat_archive-${version}.tar.gz"

echo "[release] syncing locked environment"
uv sync --frozen

echo "[release] running test suite"
uv run pytest

echo "[release] running fixture smoke tests"
uv run pytest tests/test_ci_fixture_smoke.py -q

echo "[release] building wheel and sdist"
rm -rf dist
uv build --sdist --wheel --out-dir dist --no-build-isolation --no-build-logs

test -f "$wheel_path"
test -f "$sdist_path"

echo "[release] inspecting wheel contents"
uv run python -m zipfile -l "$wheel_path" | grep -q "llm_chat_archive/__init__.py"

echo "[release] inspecting sdist contents"
tar -tzf "$sdist_path" | grep -q "llm_chat_archive-${version}/pyproject.toml"
tar -tzf "$sdist_path" | grep -q "llm_chat_archive-${version}/CHANGELOG.md"
tar -tzf "$sdist_path" | grep -q "llm_chat_archive-${version}/docs/releasing.md"

echo "[release] verifying uv tool install from wheel"
temp_home="$(mktemp -d)"
trap 'rm -rf "$temp_home"' EXIT
tool_python="$(
  uv run python - <<'PY'
import sys

print(sys.executable)
PY
)"
HOME="$temp_home" uv tool install --python "$tool_python" --force "./${wheel_path}"
tool_bin_dir="$(HOME="$temp_home" uv tool dir --bin)"
"${tool_bin_dir}/llm-chat-archive" sources >/dev/null

echo "[release] release artifacts ready"
printf '%s\n' "$wheel_path"
printf '%s\n' "$sdist_path"
