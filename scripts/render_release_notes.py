from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

SECTION_HEADING = re.compile(r"^## \[(?P<version>[^\]]+)\](?: - .+)?$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the changelog section for a release version."
    )
    parser.add_argument("version", help="Release version, with or without a leading v")
    parser.add_argument(
        "--changelog",
        default="CHANGELOG.md",
        help="Path to the changelog file.",
    )
    return parser.parse_args()


def normalize_version(version: str) -> str:
    return version[1:] if version.startswith("v") else version


def extract_release_notes(changelog_text: str, version: str) -> str:
    normalized_version = normalize_version(version)
    lines = changelog_text.splitlines()
    start_index: int | None = None
    end_index = len(lines)

    for index, line in enumerate(lines):
        match = SECTION_HEADING.match(line)
        if match and match.group("version") == normalized_version:
            start_index = index
            break

    if start_index is None:
        raise ValueError(
            f"Missing changelog section for version {normalized_version!r} in CHANGELOG.md."
        )

    for index in range(start_index + 1, len(lines)):
        if SECTION_HEADING.match(lines[index]):
            end_index = index
            break

    section = "\n".join(lines[start_index:end_index]).strip()
    if not section:
        raise ValueError(
            f"Changelog section for version {normalized_version!r} is empty."
        )
    return section + "\n"


def main() -> int:
    args = parse_args()
    changelog_path = Path(args.changelog)
    try:
        changelog_text = changelog_path.read_text(encoding="utf-8")
        release_notes = extract_release_notes(changelog_text, args.version)
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    sys.stdout.write(release_notes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
