from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_ROOT = REPO_ROOT / "tests" / "fixtures"


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "llm_chat_archive", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def load_json_output(*args: str) -> dict[str, object]:
    result = run_cli(*args)
    assert result.returncode == 0, result.stderr or result.stdout
    return json.loads(result.stdout)


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def collect_fixture(archive_root: Path, *, source: str, fixture_subpath: str) -> dict[str, object]:
    payload = load_json_output(
        "collect",
        source,
        "--archive-root",
        str(archive_root),
        "--input-root",
        str(FIXTURES_ROOT / fixture_subpath),
        "--execute",
    )
    output_path = Path(str(payload["output_path"]))
    assert output_path.exists()
    assert payload["source"] == source
    assert int(payload["conversation_count"]) > 0
    return payload


def test_codex_cli_fixture_collect_smoke(tmp_path: Path) -> None:
    payload = collect_fixture(
        tmp_path,
        source="codex_cli",
        fixture_subpath="codex_cli",
    )

    assert payload["scanned_artifact_count"] == 2
    assert payload["conversation_count"] == 2
    assert payload["message_count"] == 5
    rows = read_jsonl(Path(str(payload["output_path"])))
    assert [row["source_session_id"] for row in rows] == [
        "session-archived",
        "session-active",
    ]


def test_cursor_fixture_collect_smoke(tmp_path: Path) -> None:
    payload = collect_fixture(
        tmp_path,
        source="cursor",
        fixture_subpath="cursor_cli",
    )

    assert payload["scanned_artifact_count"] == 3
    assert payload["conversation_count"] == 3
    rows = read_jsonl(Path(str(payload["output_path"])))
    assert {row["source_session_id"] for row in rows} == {
        "20260314T120000",
        "20260314T121500",
        "20260314T160000",
    }
    assert {row.get("transcript_completeness", "complete") for row in rows} == {
        "complete",
        "partial",
        "unsupported",
    }


def test_gemini_code_assist_ide_fixture_collect_smoke(tmp_path: Path) -> None:
    payload = collect_fixture(
        tmp_path,
        source="gemini_code_assist_ide",
        fixture_subpath="gemini_code_assist_ide",
    )

    assert payload["scanned_artifact_count"] == 3
    assert payload["conversation_count"] == 3
    rows = read_jsonl(Path(str(payload["output_path"])))
    assert {row["source_session_id"] for row in rows} == {
        "vscode:global",
        "vscode:workspace-alpha:gemini-candidate",
        "vscode:workspace-alpha:residue",
    }
    assert sorted(len(row["messages"]) for row in rows) == [0, 0, 6]


def test_archive_commands_fixture_smoke(tmp_path: Path) -> None:
    collect_fixture(
        tmp_path,
        source="codex_cli",
        fixture_subpath="codex_cli",
    )
    collect_fixture(
        tmp_path,
        source="cursor",
        fixture_subpath="cursor_cli",
    )
    collect_fixture(
        tmp_path,
        source="gemini_code_assist_ide",
        fixture_subpath="gemini_code_assist_ide",
    )

    list_payload = load_json_output(
        "archive",
        "list",
        "--archive-root",
        str(tmp_path),
    )
    assert list_payload["conversation_count"] == 8
    assert {conversation["source"] for conversation in list_payload["conversations"]} == {
        "codex_cli",
        "cursor",
        "gemini_code_assist_ide",
    }

    show_payload = load_json_output(
        "archive",
        "show",
        "--archive-root",
        str(tmp_path),
        "--source",
        "codex_cli",
        "--session",
        "session-active",
    )
    assert show_payload["summary"]["source"] == "codex_cli"
    assert show_payload["conversation"]["messages"][0]["text"] == (
        "Follow the repository guardrails."
    )

    find_payload = load_json_output(
        "archive",
        "find",
        "--archive-root",
        str(tmp_path),
        "--text",
        "cli.log and bridge state",
    )
    assert find_payload["conversation_count"] == 1
    assert find_payload["conversations"][0]["source"] == "cursor"
    assert find_payload["conversations"][0]["source_session_id"] == "20260314T120000"

    stats_payload = load_json_output(
        "archive",
        "stats",
        "--archive-root",
        str(tmp_path),
    )
    assert stats_payload["source_count"] == 3
    assert stats_payload["conversation_count"] == 8
    assert stats_payload["message_count"] == 15
    assert stats_payload["transcript_completeness"] == {
        "complete": {"count": 4, "ratio": 4 / 8},
        "partial": {"count": 1, "ratio": 1 / 8},
        "unsupported": {"count": 3, "ratio": 3 / 8},
    }


def test_one_pass_first_run_fixture_smoke(tmp_path: Path) -> None:
    archive_root = tmp_path / "chat-history"
    config_path = tmp_path / "collector.toml"
    fixture_root = FIXTURES_ROOT / "codex_cli"

    config_payload = load_json_output(
        "config",
        "init",
        "--output",
        str(config_path),
        "--archive-root",
        str(archive_root),
    )
    assert config_payload == {
        "archive_root": str(archive_root),
        "output_path": str(config_path),
        "overwrote": False,
        "written": True,
    }

    doctor_payload = load_json_output(
        "doctor",
        "--all",
        "--profile",
        "default",
        "--source",
        "codex_cli",
        "--input-root",
        str(fixture_root),
    )
    assert doctor_payload["selected_sources"] == ["codex_cli"]
    assert doctor_payload["sources"][0]["source"] == "codex_cli"
    assert doctor_payload["sources"][0]["status"] == "ready"
    assert doctor_payload["sources"][0]["candidate_artifact_count"] == 2

    collect_payload = load_json_output(
        "collect",
        "--all",
        "--config",
        str(config_path),
        "--source",
        "codex_cli",
        "--input-root",
        str(fixture_root),
    )
    assert collect_payload["archive_root"] == str(archive_root)
    assert collect_payload["selected_sources"] == ["codex_cli"]
    assert collect_payload["sources"][0]["source"] == "codex_cli"
    assert collect_payload["sources"][0]["status"] == "complete"
    assert collect_payload["sources"][0]["written_conversation_count"] == 2

    latest_payload = load_json_output(
        "runs",
        "latest",
        "--archive-root",
        str(archive_root),
    )
    run_id = str(latest_payload["run_id"])
    assert run_id == collect_payload["run_id"]

    show_payload = load_json_output(
        "archive",
        "show",
        "--archive-root",
        str(archive_root),
        "--source",
        "codex_cli",
        "--session",
        "session-active",
    )
    assert show_payload["summary"]["source"] == "codex_cli"
    assert show_payload["conversation"]["messages"][0]["text"] == (
        "Follow the repository guardrails."
    )

    find_payload = load_json_output(
        "archive",
        "find",
        "--archive-root",
        str(archive_root),
        "--text",
        "Summarize the last change.",
    )
    assert find_payload["conversation_count"] == 1
    assert find_payload["conversations"][0]["source"] == "codex_cli"
    assert find_payload["conversations"][0]["source_session_id"] == "session-archived"

    export_dir = tmp_path / "memory-export"
    export_payload = load_json_output(
        "archive",
        "export-memory",
        "--archive-root",
        str(archive_root),
        "--run",
        run_id,
        "--output-dir",
        str(export_dir),
        "--execute",
    )
    assert export_payload["filters"]["run_id"] == run_id
    assert export_payload["conversation_count"] == 2
    assert export_payload["record_count"] == 2
    assert export_payload["records_path"] == str(export_dir / "memory-records.jsonl")
    assert export_payload["manifest_path"] == str(
        export_dir / "memory-export-manifest.json"
    )
    assert len(read_jsonl(export_dir / "memory-records.jsonl")) == 2
