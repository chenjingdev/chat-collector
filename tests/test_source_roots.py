from __future__ import annotations

from llm_chat_archive.source_roots import resolve_source_roots
from llm_chat_archive.sources.claude_code_ide import CLAUDE_CODE_IDE_DESCRIPTOR
from llm_chat_archive.sources.codex_cli import CODEX_CLI_DESCRIPTOR
from llm_chat_archive.sources.cursor_cli import CURSOR_CLI_DESCRIPTOR
from llm_chat_archive.sources.windsurf_editor import WINDSURF_EDITOR_DESCRIPTOR


def test_resolve_source_roots_expands_darwin_home_paths() -> None:
    resolution = resolve_source_roots(
        CODEX_CLI_DESCRIPTOR,
        platform="darwin",
        env={"HOME": "/Users/tester"},
    )

    assert resolution.to_dict() == {
        "platform": "darwin",
        "resolution_source": "descriptor",
        "resolved_roots": ["/Users/tester/.codex"],
        "roots": [
            {
                "declared_path": "$HOME/.codex",
                "resolution_source": "descriptor",
                "path": "/Users/tester/.codex",
                "env_vars": ["HOME"],
            }
        ],
    }


def test_resolve_source_roots_expands_linux_xdg_override() -> None:
    resolution = resolve_source_roots(
        CLAUDE_CODE_IDE_DESCRIPTOR,
        platform="linux",
        env={
            "HOME": "/home/tester",
            "XDG_CONFIG_HOME": "/tmp/xdg-config",
        },
    )

    assert resolution.to_dict() == {
        "platform": "linux",
        "resolution_source": "descriptor",
        "resolved_roots": [
            "/home/tester/.claude",
            "/home/tester/.claude.json",
            "/tmp/xdg-config/Code/User/globalStorage",
            "/tmp/xdg-config/Cursor/User/globalStorage",
        ],
        "roots": [
            {
                "declared_path": "$HOME/.claude",
                "resolution_source": "descriptor",
                "path": "/home/tester/.claude",
                "env_vars": ["HOME"],
            },
            {
                "declared_path": "$HOME/.claude.json",
                "resolution_source": "descriptor",
                "path": "/home/tester/.claude.json",
                "env_vars": ["HOME"],
            },
            {
                "declared_path": "$XDG_CONFIG_HOME/Code/User/globalStorage",
                "resolution_source": "descriptor",
                "path": "/tmp/xdg-config/Code/User/globalStorage",
                "env_vars": ["XDG_CONFIG_HOME"],
            },
            {
                "declared_path": "$XDG_CONFIG_HOME/Cursor/User/globalStorage",
                "resolution_source": "descriptor",
                "path": "/tmp/xdg-config/Cursor/User/globalStorage",
                "env_vars": ["XDG_CONFIG_HOME"],
            },
        ],
    }


def test_resolve_source_roots_expands_windows_appdata_override() -> None:
    resolution = resolve_source_roots(
        CURSOR_CLI_DESCRIPTOR,
        platform="windows",
        env={
            "USERPROFILE": "C:/Users/tester",
            "APPDATA": "D:/Roaming",
        },
    )

    assert resolution.to_dict() == {
        "platform": "windows",
        "resolution_source": "descriptor",
        "resolved_roots": [
            "C:/Users/tester/.cursor",
            "D:/Roaming/Cursor",
        ],
        "roots": [
            {
                "declared_path": "$HOME/.cursor",
                "resolution_source": "descriptor",
                "path": "C:/Users/tester/.cursor",
                "env_vars": ["HOME"],
            },
            {
                "declared_path": "$APPDATA/Cursor",
                "resolution_source": "descriptor",
                "path": "D:/Roaming/Cursor",
                "env_vars": ["APPDATA"],
            },
        ],
    }


def test_resolve_source_roots_expands_windows_programdata_root() -> None:
    resolution = resolve_source_roots(
        WINDSURF_EDITOR_DESCRIPTOR,
        platform="windows",
        env={
            "USERPROFILE": "C:/Users/tester",
            "APPDATA": "D:/Roaming",
            "PROGRAMDATA": "E:/ProgramData",
        },
    )

    assert resolution.to_dict() == {
        "platform": "windows",
        "resolution_source": "descriptor",
        "resolved_roots": [
            "D:/Roaming/Codeium/windsurf",
            "E:/ProgramData/Windsurf",
        ],
        "roots": [
            {
                "declared_path": "$APPDATA/Codeium/windsurf",
                "resolution_source": "descriptor",
                "path": "D:/Roaming/Codeium/windsurf",
                "env_vars": ["APPDATA"],
            },
            {
                "declared_path": "$PROGRAMDATA/Windsurf",
                "resolution_source": "descriptor",
                "path": "E:/ProgramData/Windsurf",
                "env_vars": ["PROGRAMDATA"],
            },
        ],
    }
