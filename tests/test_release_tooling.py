from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_pyproject() -> dict[str, object]:
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_project_version_has_matching_changelog_section() -> None:
    version = load_pyproject()["project"]["version"]
    changelog = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")

    assert f"## [{version}] -" in changelog


def test_hatch_build_targets_pin_release_artifact_inputs() -> None:
    pyproject = load_pyproject()
    assert pyproject["build-system"]["build-backend"] == "hatchling.build"
    assert pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == [
        "src/llm_chat_archive"
    ]
    include_paths = set(
        pyproject["tool"]["hatch"]["build"]["targets"]["sdist"]["include"]
    )
    assert {
        "/CHANGELOG.md",
        "/docs/releasing.md",
        "/scripts/release_preflight.sh",
        "/scripts/render_release_notes.py",
        "/src/llm_chat_archive",
        "/tests",
    } <= include_paths


def test_release_workflow_uses_preflight_and_changelog_notes() -> None:
    workflow = (REPO_ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")

    assert 'tags:' in workflow
    assert './scripts/release_preflight.sh' in workflow
    assert 'scripts/render_release_notes.py "${GITHUB_REF_NAME}" > RELEASE_NOTES.md' in workflow
    assert 'gh release create "${GITHUB_REF_NAME}"' in workflow


def test_release_doc_mentions_install_smoke_and_first_run_smoke() -> None:
    release_doc = (REPO_ROOT / "docs" / "releasing.md").read_text(encoding="utf-8")

    assert "Release checklist" in release_doc
    assert "Install smoke" in release_doc
    assert "First-run smoke" in release_doc
    assert "llm-chat-archive config init --archive-root /Users/chenjing/dev/chat-history --force" in release_doc
    assert "llm-chat-archive doctor --all --profile default" in release_doc
    assert "llm-chat-archive collect --all" in release_doc
    assert "llm-chat-archive runs latest --archive-root /Users/chenjing/dev/chat-history" in release_doc
    assert "llm-chat-archive archive export-memory" in release_doc


def test_render_release_notes_returns_current_version_section() -> None:
    version = load_pyproject()["project"]["version"]
    result = subprocess.run(
        [sys.executable, "scripts/render_release_notes.py", str(version)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.startswith(f"## [{version}] - ")
    assert "uv build --sdist --wheel --out-dir dist --no-build-isolation" in result.stdout


def test_render_release_notes_rejects_unknown_version() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/render_release_notes.py", "9.9.9"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "Missing changelog section" in result.stderr
