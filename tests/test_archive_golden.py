from __future__ import annotations

import difflib
import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import pytest

from llm_chat_archive.sources import build_registry
from tests.fixture_cases import FIXTURE_CASES, FixtureCase, FIXTURES_ROOT, REPO_ROOT

GOLDEN_ROOT = REPO_ROOT / "tests" / "golden" / "archive"
UPDATE_ENV_VAR = "UPDATE_ARCHIVE_GOLDENS"
OUTPUT_FILENAME_PATTERN = re.compile(r"memory_chat_v1-\d{8}T\d{6}Z\.jsonl")
REPO_ROOT_ALIASES = tuple(
    dict.fromkeys((str(REPO_ROOT), "/Users/chenjing/dev/chat-collector"))
)


def expected_path(case: FixtureCase) -> Path:
    return GOLDEN_ROOT / f"{case.case_id}.json"


@lru_cache(maxsize=1)
def _collectors_by_key() -> dict[str, object]:
    return {
        collector.descriptor.key: collector
        for collector in build_registry().list()
    }


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    content = path.read_text(encoding="utf-8")
    if not content.strip():
        return []
    return [json.loads(line) for line in content.splitlines()]


def _build_snapshot(case: FixtureCase, tmp_path: Path) -> dict[str, object]:
    collector = _collectors_by_key()[case.source]
    archive_root = tmp_path / case.case_id
    result = collector.collect(archive_root, input_roots=(case.fixture_root,))
    rows = _read_jsonl(result.output_path)
    collected_at = rows[0]["collected_at"] if rows else None

    descriptor = {
        "display_name": collector.descriptor.display_name,
        "execution_context": collector.descriptor.execution_context,
        "support_level": collector.descriptor.support_level.value,
    }
    if collector.descriptor.support_metadata is not None:
        descriptor["support_metadata"] = collector.descriptor.support_metadata.to_dict()

    return {
        "case": case.case_id,
        "source": case.source,
        "fixture_subpath": case.fixture_subpath,
        "fixture_metadata": (
            None if case.fixture_metadata is None else case.fixture_metadata.to_dict()
        ),
        "descriptor": descriptor,
        "result": _normalize_value(
            result.to_dict(),
            case=case,
            archive_root=archive_root,
            collected_at=collected_at,
        ),
        "rows": _normalize_value(
            rows,
            case=case,
            archive_root=archive_root,
            collected_at=collected_at,
        ),
    }


def _normalize_value(
    value: Any,
    *,
    case: GoldenCase,
    archive_root: Path,
    collected_at: str | None,
) -> Any:
    if isinstance(value, dict):
        return {
            key: _normalize_value(
                nested_value,
                case=case,
                archive_root=archive_root,
                collected_at=collected_at,
            )
            for key, nested_value in value.items()
        }
    if isinstance(value, list):
        return [
            _normalize_value(
                nested_value,
                case=case,
                archive_root=archive_root,
                collected_at=collected_at,
            )
            for nested_value in value
        ]
    if isinstance(value, str):
        return _normalize_string(
            value,
            case=case,
            archive_root=archive_root,
            collected_at=collected_at,
        )
    return value


def _normalize_string(
    value: str,
    *,
    case: GoldenCase,
    archive_root: Path,
    collected_at: str | None,
) -> str:
    normalized = value
    normalized = normalized.replace(str(case.fixture_root), "<FIXTURE_ROOT>")
    normalized = normalized.replace(str(archive_root), "<ARCHIVE_ROOT>")
    for repo_root_alias in REPO_ROOT_ALIASES:
        normalized = normalized.replace(repo_root_alias, "<REPO_ROOT>")

    if collected_at is not None:
        normalized = normalized.replace(collected_at, "<COLLECTED_AT>")

    return OUTPUT_FILENAME_PATTERN.sub(
        "memory_chat_v1-<COLLECTED_AT>.jsonl",
        normalized,
    )


def _serialize_snapshot(snapshot: dict[str, object]) -> str:
    return json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _assert_matches_golden(case: FixtureCase, serialized: str) -> None:
    if os.getenv(UPDATE_ENV_VAR) == "1":
        expected_path(case).parent.mkdir(parents=True, exist_ok=True)
        expected_path(case).write_text(serialized, encoding="utf-8")
        return

    if not expected_path(case).is_file():
        pytest.fail(
            "missing golden snapshot "
            f"{expected_path(case)}. "
            f"Re-run with `{UPDATE_ENV_VAR}=1 uv run pytest tests/test_archive_golden.py`."
        )

    expected = expected_path(case).read_text(encoding="utf-8")
    if expected == serialized:
        return

    diff = "".join(
        difflib.unified_diff(
            expected.splitlines(keepends=True),
            serialized.splitlines(keepends=True),
            fromfile=str(expected_path(case)),
            tofile=f"{expected_path(case)} (actual)",
        )
    )
    pytest.fail(
        f"golden snapshot mismatch for source `{case.source}` "
        f"(case `{case.case_id}`).\n{diff}"
    )


@pytest.mark.parametrize("case", FIXTURE_CASES, ids=lambda case: case.case_id)
def test_source_archive_matches_golden_snapshot(
    case: FixtureCase,
    tmp_path: Path,
) -> None:
    snapshot = _build_snapshot(case, tmp_path)
    _assert_matches_golden(case, _serialize_snapshot(snapshot))
