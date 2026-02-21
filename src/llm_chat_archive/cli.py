from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from llm_chat_archive.collector import CollectConfig, VALID_SOURCES, run_collect


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm-chat-archive")
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser("collect", help="Collect and archive chat sessions")
    collect.add_argument("--output", required=True, type=Path, help="Output archive repository path")
    collect.add_argument(
        "--full-rebuild",
        action="store_true",
        help="Regenerate archive tree and rebuild session index",
    )
    collect.add_argument("--timezone", default=None, help="IANA timezone (e.g. Asia/Seoul)")
    collect.add_argument("--since", default=None, help="Only include sessions on/after YYYY-MM-DD")
    collect.add_argument(
        "--sources",
        default=",".join(VALID_SOURCES),
        help=f"Comma-separated source list (default: {','.join(VALID_SOURCES)})",
    )
    collect.add_argument(
        "--copy-source-jsonl",
        action="store_true",
        help="Copy source JSONL files (Codex/Claude) into each session directory",
    )
    collect.add_argument(
        "--update-existing",
        action="store_true",
        help="Rewrite existing sessions matched by fingerprint (summary file stays preserved)",
    )

    collect.add_argument("--codex-home", type=Path, default=None, help=argparse.SUPPRESS)
    collect.add_argument("--claude-home", type=Path, default=None, help=argparse.SUPPRESS)
    collect.add_argument("--cursor-support", type=Path, default=None, help=argparse.SUPPRESS)
    collect.add_argument("--antigravity-support", type=Path, default=None, help=argparse.SUPPRESS)
    collect.add_argument("--gemini-home", type=Path, default=None, help=argparse.SUPPRESS)

    return parser


def _parse_since(value: str | None) -> date | None:
    if not value:
        return None
    return date.fromisoformat(value)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "collect":
        sources = [item.strip() for item in str(args.sources).split(",") if item.strip()]
        config = CollectConfig(
            output_root=args.output,
            sources=sources,
            full_rebuild=bool(args.full_rebuild),
            timezone=args.timezone,
            since=_parse_since(args.since),
            copy_source_jsonl=bool(args.copy_source_jsonl),
            update_existing=bool(args.update_existing),
            codex_home=args.codex_home,
            claude_home=args.claude_home,
            cursor_support=args.cursor_support,
            antigravity_support=args.antigravity_support,
            gemini_home=args.gemini_home,
        )
        result = run_collect(config)
        print(f"scanned={result.scanned} (by source: {result.sources_stats})")
        print(f"written={result.written}")
        print(f"skipped_since={result.skipped_since}")
        print(f"skipped_empty={result.skipped_empty}")
        print(f"skipped_existing={result.skipped_existing}")
        return 0

    parser.print_help()
    return 1
