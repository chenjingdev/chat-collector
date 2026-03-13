#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
symphony_root="$project_root/.symphony"
env_file="$symphony_root/.env"
workflow_template="$symphony_root/WORKFLOW.template.md"
rendered_workflow="$symphony_root/WORKFLOW.md"

if [[ -f "$env_file" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$env_file"
  set +a
else
  echo "Missing $env_file" >&2
  exit 1
fi

required_vars=(
  LINEAR_API_KEY
  SYMPHONY_LINEAR_PROJECT_SLUG
  SYMPHONY_SHARED_ROOT
  CODEX_BIN
  CHAT_ARCHIVE_ROOT
  SYMPHONY_PORT
  SYMPHONY_WORKSPACE_ROOT
  SYMPHONY_LOGS_ROOT
  SYMPHONY_SOURCE_REPO_URL
)

for name in "${required_vars[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "Set $name in .symphony/.env" >&2
    exit 1
  fi
done

export CODEX_BIN
export CHAT_ARCHIVE_ROOT
export LINEAR_API_KEY
export SYMPHONY_PORT
export SYMPHONY_WORKSPACE_ROOT
export SYMPHONY_LOGS_ROOT
export SYMPHONY_SOURCE_REPO_URL
export SYMPHONY_LINEAR_PROJECT_SLUG

if [[ ! -f "$workflow_template" ]]; then
  echo "Missing $workflow_template" >&2
  exit 1
fi

if [[ ! -d "$SYMPHONY_SHARED_ROOT" ]]; then
  echo "Shared Symphony runtime not found: $SYMPHONY_SHARED_ROOT" >&2
  exit 1
fi

mkdir -p "$symphony_root" "$SYMPHONY_LOGS_ROOT"

escaped_slug="$(printf '%s' "$SYMPHONY_LINEAR_PROJECT_SLUG" | sed -e 's/[\\/&]/\\&/g')"
escaped_workspace_root="$(printf '%s' "$SYMPHONY_WORKSPACE_ROOT" | sed -e 's/[\\/&]/\\&/g')"
escaped_archive_root="$(printf '%s' "$CHAT_ARCHIVE_ROOT" | sed -e 's/[\\/&]/\\&/g')"
sed -e "s|__SET_SYMPHONY_LINEAR_PROJECT_SLUG__|$escaped_slug|g" \
  -e "s|__SET_SYMPHONY_WORKSPACE_ROOT__|$escaped_workspace_root|g" \
  -e "s|__SET_CHAT_ARCHIVE_ROOT__|$escaped_archive_root|g" \
  "$workflow_template" > "$rendered_workflow"

if [[ ! -x "$SYMPHONY_SHARED_ROOT/bin/symphony" ]]; then
  if ! command -v mise >/dev/null 2>&1; then
    echo "mise is required to bootstrap the shared Symphony runtime." >&2
    exit 1
  fi

  (
    cd "$SYMPHONY_SHARED_ROOT"
    mise trust
    mise install
    mise exec -- mix setup
    mise exec -- mix build
  )
fi

cd "$SYMPHONY_SHARED_ROOT"
exec mise exec -- ./bin/symphony \
  --i-understand-that-this-will-be-running-without-the-usual-guardrails \
  "$rendered_workflow" \
  --logs-root "$SYMPHONY_LOGS_ROOT" \
  --port "$SYMPHONY_PORT"
