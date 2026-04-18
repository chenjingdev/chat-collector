from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Callable, Sequence

from .config import (
    CollectorConfigError,
    default_collect_config_path,
    render_collect_config_template,
    resolve_collect_config,
    resolve_collect_config_output_path,
    resolve_scheduled_config,
    scaffold_collect_config,
)
from .acceptance import run_ship_acceptance
from .archive_anomalies import (
    ArchiveAnomalyThresholds,
    summarize_archive_anomalies,
)
from .archive_digest import summarize_archive_digest
from .archive_export import export_archive_subset
from .archive_identity_audit import audit_archive_identities
from .archive_index import inspect_archive_index, refresh_archive_index
from .archive_import import ArchiveImportError, import_archive_bundle
from .archive_memory_export import export_archive_memory_records
from .archive_migrate import ArchiveMigrateError, migrate_archive
from .archive_profile import summarize_archive_profile
from .archive_prune import DEFAULT_AUXILIARY_DIRECTORIES, ArchivePruneError, prune_archive
from .archive_quarantine_export import (
    ArchiveQuarantineExportError,
    export_archive_quarantine,
)
from .archive_sample import sample_archive_subset
from .archive_stats import summarize_archive_stats
from .archive_rewrite import rewrite_archive
from .archive_verify import ArchiveVerifyError, verify_archive
from .baseline_policy import (
    BASELINE_POLICY_FILENAME,
    baseline_policy_path,
    load_baseline_policy,
    merge_baseline_entries,
    save_baseline_policy,
    snapshot_entries_from_archive_anomalies,
    snapshot_entries_from_archive_verify,
    snapshot_entries_from_validate,
)
from .archive_inspect import (
    ArchiveInspectError,
    find_archive_conversations,
    list_archive_conversations,
    show_archive_conversation,
)
from .doctor import inspect_registry_readiness, inspect_source_readiness
from .execution_policy import collection_execution_policy_context
from .models import (
    ArchiveTargetPolicy,
    DEFAULT_ARCHIVE_ROOT,
    EffectiveCollectConfig,
    NormalizationContract,
    RerunSelectionReason,
    ScheduledRunMetadata,
    ScheduledRunMode,
    SourceSelectionProfile,
    TranscriptCompleteness,
    ValidationMode,
)
from .rerun import RerunSelectionError, plan_rerun, resolve_rerun_config
from .reporting import (
    RunReportingError,
    list_run_summaries,
    load_run_diff,
    load_latest_run_summary,
    load_run_summary,
    load_run_trend,
)
from .registry import ExecutableCollector
from .source_selection import build_source_selection_policy
from .source_roots import resolve_source_roots
from .source_support import (
    build_source_support_matrix,
    render_source_support_matrix_markdown,
)
from .scheduled import ScheduledLockError, acquire_scheduled_lock
from .runner import run_collection_batch
from .sources import build_registry
from .tui import (
    DEFAULT_SAMPLE_COUNT as TUI_DEFAULT_SAMPLE_COUNT,
    DEFAULT_SAMPLE_SEED as TUI_DEFAULT_SAMPLE_SEED,
    DEFAULT_SNAPSHOT_WIDTH as TUI_DEFAULT_SNAPSHOT_WIDTH,
    TuiError,
    render_tui_snapshot,
    run_operator_triage_tui,
    tui_view_choices,
)
from .validate import validate_run


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    registry = build_registry()
    parser = argparse.ArgumentParser(
        prog="llm_chat_archive",
        description=(
            "Collect local coding-agent chats into normalized archives stored outside "
            "this repository."
        ),
    )
    subparsers = parser.add_subparsers(dest="command")

    sources_parser = subparsers.add_parser("sources", help="List registered collectors")
    sources_parser.add_argument(
        "--format",
        choices=("tsv", "json", "markdown"),
        default="tsv",
        help=(
            "Output format. 'tsv' keeps the source/support/root listing, "
            "'json' emits the support matrix summary, and 'markdown' renders "
            "the support matrix document."
        ),
    )
    sources_parser.set_defaults(handler=handle_sources)

    contract_parser = subparsers.add_parser(
        "contract",
        help="Show the normalization schema and archive target policy",
    )
    contract_parser.set_defaults(handler=handle_contract)

    config_parser = subparsers.add_parser(
        "config",
        help="Scaffold or inspect collector config files",
    )
    config_subparsers = config_parser.add_subparsers(dest="config_command")
    config_parser.set_defaults(handler=_build_help_handler(config_parser))

    config_init_parser = config_subparsers.add_parser(
        "init",
        help="Write the default collector config scaffold",
    )
    config_init_parser.add_argument(
        "--archive-root",
        type=Path,
        default=DEFAULT_ARCHIVE_ROOT,
        help=(
            "Archive root to write into collect.archive_root. "
            f"Default: {DEFAULT_ARCHIVE_ROOT}"
        ),
    )
    config_init_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output path for the scaffolded config file. "
            f"Default: {default_collect_config_path()}"
        ),
    )
    config_init_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing config file at the target path.",
    )
    config_init_parser.add_argument(
        "--print",
        action="store_true",
        dest="print_template",
        help="Print the default config template to stdout without writing a file.",
    )
    config_init_parser.set_defaults(handler=handle_config_init)

    collect_parser = subparsers.add_parser(
        "collect",
        help="Emit a collection plan, execute a collector, or run a batch collection",
    )
    collect_parser.add_argument("target_source", nargs="?", choices=registry.keys())
    collect_parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Absolute path to a collector config file. If omitted, tries "
            f"{default_collect_config_path()}"
        ),
    )
    collect_parser.add_argument(
        "--all",
        action="store_true",
        help="Execute all registered collectors sequentially and write a run manifest.",
    )
    collect_parser.add_argument(
        "--profile",
        choices=("all", "default", "complete_only"),
        help=(
            "Batch selection profile. 'all' keeps current behavior, 'default' "
            "runs unattended-ready complete sources, and 'complete_only' keeps "
            "the same complete-only source set with explicit intent."
        ),
    )
    collect_parser.add_argument(
        "--source",
        dest="selected_sources",
        action="append",
        choices=registry.keys(),
        default=None,
        help="Batch allowlist. Repeat to run only specific sources with --all.",
    )
    collect_parser.add_argument(
        "--exclude-source",
        action="append",
        choices=registry.keys(),
        default=None,
        help="Batch denylist. Repeat to skip specific sources with --all.",
    )
    collect_parser.add_argument(
        "--archive-root",
        type=Path,
        default=None,
        help=(
            "Absolute archive root outside the repository. "
            f"Default: collector config value or {DEFAULT_ARCHIVE_ROOT}"
        ),
    )
    collect_parser.add_argument(
        "--incremental",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable or disable incremental dedupe before writes. "
            "Default: collector config value or enabled."
        ),
    )
    collect_parser.add_argument(
        "--redaction",
        choices=("on", "off"),
        default=None,
        help=(
            "Credential redaction policy for collected output. "
            "Default: collector config value or on."
        ),
    )
    collect_parser.add_argument(
        "--validation",
        choices=("off", "report", "strict"),
        default=None,
        help=(
            "Batch manifest validation policy after collection. "
            "Default: collector config value or off."
        ),
    )
    collect_parser.add_argument(
        "--input-root",
        action="append",
        type=Path,
        default=[],
        help=(
            "Optional source artifact root. Repeat to override default roots or point "
            "at test fixtures."
        ),
    )
    collect_parser.add_argument(
        "--execute",
        action="store_true",
        help="Run the collector and write normalized output to the archive root.",
    )
    collect_parser.set_defaults(handler=handle_collect)

    rerun_parser = subparsers.add_parser(
        "rerun",
        help="Re-execute a source subset derived from a previous recorded run",
    )
    rerun_parser.add_argument(
        "--run",
        required=True,
        dest="run_id",
        help="Recorded batch collection run identifier used as the rerun origin.",
    )
    rerun_parser.add_argument(
        "--reason",
        choices=tuple(reason.value for reason in RerunSelectionReason),
        help=(
            "Subset selector derived from the origin manifest. 'degraded' includes "
            "partial and unsupported sources. Defaults to collector config "
            "[rerun] selection_preset when present."
        ),
    )
    rerun_parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Absolute path to a collector config file. If omitted, tries "
            f"{default_collect_config_path()}"
        ),
    )
    rerun_parser.add_argument(
        "--source",
        dest="selected_sources",
        action="append",
        choices=registry.keys(),
        default=[],
        help=(
            "Additional source to include in the rerun allowlist. Repeat to widen "
            "the rerun target set."
        ),
    )
    rerun_parser.add_argument(
        "--exclude-source",
        action="append",
        choices=registry.keys(),
        default=[],
        help="Source to remove from the rerun target set. Repeat as needed.",
    )
    rerun_parser.add_argument(
        "--archive-root",
        type=Path,
        default=None,
        help=(
            "Absolute archive root outside the repository. "
            f"Default: collector config value or {DEFAULT_ARCHIVE_ROOT}"
        ),
    )
    rerun_parser.add_argument(
        "--incremental",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable or disable incremental dedupe before writes. "
            "Default: collector config value or enabled."
        ),
    )
    rerun_parser.add_argument(
        "--redaction",
        choices=("on", "off"),
        default=None,
        help=(
            "Credential redaction policy for collected output. "
            "Default: collector config value or on."
        ),
    )
    rerun_parser.add_argument(
        "--validation",
        choices=("off", "report", "strict"),
        default=None,
        help=(
            "Batch manifest validation policy after collection. "
            "Default: collector config value or off."
        ),
    )
    rerun_parser.add_argument(
        "--input-root",
        action="append",
        type=Path,
        default=[],
        help=(
            "Optional source artifact root. Repeat to override default roots or point "
            "at test fixtures."
        ),
    )
    rerun_parser.set_defaults(handler=handle_rerun)

    scheduled_parser = subparsers.add_parser(
        "scheduled",
        help="Run a non-interactive scheduled collection or rerun with overlap-safe locking",
    )
    scheduled_subparsers = scheduled_parser.add_subparsers(dest="scheduled_command")
    scheduled_parser.set_defaults(handler=_build_help_handler(scheduled_parser))

    scheduled_run_parser = scheduled_subparsers.add_parser(
        "run",
        help="Execute the configured scheduled batch or rerun mode under an archive lock",
    )
    scheduled_run_parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Absolute path to a collector config file. If omitted, tries "
            f"{default_collect_config_path()}"
        ),
    )
    scheduled_run_parser.add_argument(
        "--archive-root",
        type=Path,
        default=None,
        help=(
            "Absolute archive root outside the repository. "
            f"Default: scheduled config value or {DEFAULT_ARCHIVE_ROOT}"
        ),
    )
    scheduled_run_parser.add_argument(
        "--mode",
        choices=tuple(mode.value for mode in ScheduledRunMode),
        default=None,
        help=(
            "Override the configured scheduled mode. 'collect' runs the scheduled "
            "batch selection, and 'rerun' replays a prior run using the scheduled "
            "rerun preset."
        ),
    )
    scheduled_run_parser.add_argument(
        "--run",
        dest="run_id",
        default=None,
        help=(
            "Origin run identifier for scheduled rerun mode. Defaults to the latest "
            "recorded run when omitted."
        ),
    )
    scheduled_run_parser.add_argument(
        "--stale-after-seconds",
        type=int,
        default=None,
        help="Override the scheduled stale lock threshold in seconds.",
    )
    scheduled_run_parser.add_argument(
        "--force-unlock-stale",
        action="store_true",
        help=(
            "Replace an existing stale lock and continue. Active non-stale locks are "
            "never overridden."
        ),
    )
    scheduled_run_parser.add_argument(
        "--input-root",
        action="append",
        type=Path,
        default=[],
        help=(
            "Optional source artifact root. Repeat to override default roots or point "
            "at test fixtures."
        ),
    )
    scheduled_run_parser.set_defaults(handler=handle_scheduled_run)

    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Inspect source readiness without running collection writes",
    )
    doctor_parser.add_argument("target_source", nargs="?", choices=registry.keys())
    doctor_parser.add_argument(
        "--all",
        action="store_true",
        help="Inspect all registered collectors and report source readiness.",
    )
    doctor_parser.add_argument(
        "--profile",
        choices=("all", "default", "complete_only"),
        help=(
            "Batch selection profile. 'all' keeps current behavior, 'default' "
            "inspects unattended-ready complete sources, and 'complete_only' "
            "keeps the same complete-only source set with explicit intent."
        ),
    )
    doctor_parser.add_argument(
        "--source",
        dest="selected_sources",
        action="append",
        choices=registry.keys(),
        default=[],
        help="Batch allowlist. Repeat to inspect only specific sources with --all.",
    )
    doctor_parser.add_argument(
        "--exclude-source",
        action="append",
        choices=registry.keys(),
        default=[],
        help="Batch denylist. Repeat to skip specific sources with --all.",
    )
    doctor_parser.add_argument(
        "--input-root",
        action="append",
        type=Path,
        default=[],
        help=(
            "Optional source artifact root. Repeat to override default roots or point "
            "at test fixtures."
        ),
    )
    doctor_parser.set_defaults(handler=handle_doctor)

    acceptance_parser = subparsers.add_parser(
        "acceptance",
        help="Run the fixed ship-acceptance operator flow on a clean archive root",
    )
    acceptance_subparsers = acceptance_parser.add_subparsers(
        dest="acceptance_command"
    )
    acceptance_parser.set_defaults(handler=_build_help_handler(acceptance_parser))

    acceptance_ship_parser = acceptance_subparsers.add_parser(
        "ship",
        help=(
            "Run collect, validate, archive verify, archive digest, archive export, "
            "and archive export-memory against the pinned operator source set"
        ),
    )
    acceptance_ship_parser.add_argument(
        "--archive-root",
        required=True,
        type=Path,
        help="Absolute clean archive root outside the repository.",
    )
    acceptance_ship_parser.add_argument(
        "--snapshot-path",
        type=Path,
        default=None,
        help=(
            "Optional absolute path for the repo-safe golden acceptance snapshot JSON."
        ),
    )
    acceptance_ship_parser.set_defaults(handler=handle_acceptance_ship)

    runs_parser = subparsers.add_parser(
        "runs",
        help="Inspect recorded batch collection run manifests",
    )
    runs_subparsers = runs_parser.add_subparsers(dest="runs_command")
    runs_parser.set_defaults(handler=_build_help_handler(runs_parser))

    runs_list_parser = runs_subparsers.add_parser(
        "list",
        help="List recorded batch collection runs",
    )
    add_archive_root_argument(runs_list_parser)
    runs_list_parser.set_defaults(handler=handle_runs_list)

    runs_latest_parser = runs_subparsers.add_parser(
        "latest",
        help="Show the latest recorded batch collection run",
    )
    add_archive_root_argument(runs_latest_parser)
    runs_latest_parser.set_defaults(handler=handle_runs_latest)

    runs_show_parser = runs_subparsers.add_parser(
        "show",
        help="Show a specific recorded batch collection run",
    )
    runs_show_parser.add_argument("run_id")
    add_archive_root_argument(runs_show_parser)
    runs_show_parser.set_defaults(handler=handle_runs_show)

    runs_diff_parser = runs_subparsers.add_parser(
        "diff",
        help="Compare two recorded batch collection runs",
    )
    runs_diff_parser.add_argument(
        "--from",
        dest="from_run_id",
        default=None,
        help=(
            "Base run identifier to compare from. If omitted together with --to, "
            "compares latest vs previous."
        ),
    )
    runs_diff_parser.add_argument(
        "--to",
        dest="to_run_id",
        default=None,
        help="Target run identifier to compare to.",
    )
    add_archive_root_argument(runs_diff_parser)
    runs_diff_parser.set_defaults(handler=handle_runs_diff)

    runs_trend_parser = runs_subparsers.add_parser(
        "trend",
        help="Summarize source health trends across recorded runs",
    )
    runs_trend_parser.add_argument(
        "--source",
        dest="sources",
        action="append",
        default=[],
        help="Only include the given source key. Repeat to keep multiple sources.",
    )
    add_archive_root_argument(runs_trend_parser)
    runs_trend_parser.set_defaults(handler=handle_runs_trend)

    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate a recorded batch collection run manifest and its source outputs",
    )
    validate_parser.add_argument(
        "--run",
        required=True,
        dest="run_id",
        help="Recorded batch collection run identifier",
    )
    add_archive_root_argument(validate_parser)
    add_baseline_argument(validate_parser)
    validate_parser.set_defaults(handler=handle_validate)

    baseline_parser = subparsers.add_parser(
        "baseline",
        help="Create or update baseline policy entries from current operator reports",
    )
    baseline_subparsers = baseline_parser.add_subparsers(dest="baseline_command")
    baseline_parser.set_defaults(handler=_build_help_handler(baseline_parser))

    baseline_snapshot_parser = baseline_subparsers.add_parser(
        "snapshot",
        help="Snapshot current warnings or limitations into the baseline policy",
    )
    add_archive_root_argument(baseline_snapshot_parser)
    baseline_snapshot_parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help=(
            "Absolute baseline policy file path. "
            f"Default: <archive-root>/{BASELINE_POLICY_FILENAME}"
        ),
    )
    baseline_snapshot_parser.add_argument(
        "--from",
        dest="snapshot_from",
        required=True,
        choices=("validate", "archive-verify", "archive-anomalies"),
        help="Report surface to snapshot into the baseline policy.",
    )
    baseline_snapshot_parser.add_argument(
        "--reason",
        required=True,
        help="Operator-facing explanation stored on every new baseline entry.",
    )
    baseline_snapshot_parser.add_argument(
        "--run",
        default=None,
        dest="run_id",
        help="Recorded batch collection run identifier. Required with --from validate.",
    )
    baseline_snapshot_parser.add_argument(
        "--source",
        default=None,
        help="Optional source filter for archive-verify or archive-anomalies snapshots.",
    )
    baseline_snapshot_parser.add_argument(
        "--low-message-count",
        type=int,
        default=1,
        help="archive-anomalies threshold override for low message count snapshots.",
    )
    baseline_snapshot_parser.add_argument(
        "--limitations-count",
        type=int,
        default=2,
        help="archive-anomalies threshold override for limitation snapshots.",
    )
    baseline_snapshot_parser.add_argument(
        "--unsupported-count",
        type=int,
        default=2,
        help="archive-anomalies threshold override for unsupported ratio snapshots.",
    )
    baseline_snapshot_parser.add_argument(
        "--unsupported-ratio",
        type=float,
        default=0.5,
        help="archive-anomalies threshold override for unsupported ratio snapshots.",
    )
    baseline_snapshot_parser.set_defaults(handler=handle_baseline_snapshot)

    archive_parser = subparsers.add_parser(
        "archive",
        help="Inspect, verify, rewrite, export, or import normalized archive conversations",
    )
    archive_subparsers = archive_parser.add_subparsers(dest="archive_command")
    archive_parser.set_defaults(handler=_build_help_handler(archive_parser))

    archive_index_parser = archive_subparsers.add_parser(
        "index",
        help="Inspect or refresh the SQLite archive query index",
    )
    archive_index_subparsers = archive_index_parser.add_subparsers(
        dest="archive_index_command"
    )
    archive_index_parser.set_defaults(handler=_build_help_handler(archive_index_parser))

    archive_index_status_parser = archive_index_subparsers.add_parser(
        "status",
        help="Show archive index readiness, stale state, and rebuild requirements",
    )
    add_archive_root_argument(archive_index_status_parser)
    archive_index_status_parser.set_defaults(handler=handle_archive_index_status)

    archive_index_refresh_parser = archive_index_subparsers.add_parser(
        "refresh",
        help="Build or refresh the SQLite archive query index",
    )
    add_archive_root_argument(archive_index_refresh_parser)
    archive_index_refresh_parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild the index from scratch even when the current metadata looks usable.",
    )
    archive_index_refresh_parser.set_defaults(handler=handle_archive_index_refresh)

    archive_list_parser = archive_subparsers.add_parser(
        "list",
        help="List archived conversations",
    )
    add_archive_root_argument(archive_list_parser)
    add_archive_filter_arguments(archive_list_parser, include_session=True)
    archive_list_parser.set_defaults(handler=handle_archive_list)

    archive_show_parser = archive_subparsers.add_parser(
        "show",
        help="Show a specific archived conversation",
    )
    add_archive_root_argument(archive_show_parser)
    archive_show_parser.add_argument(
        "--source",
        required=True,
        help="Normalized source key recorded in the archive.",
    )
    archive_show_parser.add_argument(
        "--session",
        required=True,
        help="Source session identifier recorded as source_session_id.",
    )
    archive_show_parser.set_defaults(handler=handle_archive_show)

    archive_find_parser = archive_subparsers.add_parser(
        "find",
        help="Find archived conversations by normalized message text",
    )
    add_archive_root_argument(archive_find_parser)
    add_archive_filter_arguments(archive_find_parser, include_session=False)
    archive_find_parser.add_argument(
        "--text",
        required=True,
        help="Case-insensitive substring to search within normalized message text.",
    )
    archive_find_parser.set_defaults(handler=handle_archive_find)

    archive_sample_parser = archive_subparsers.add_parser(
        "sample",
        help="Sample archived conversations from a filtered subset",
    )
    add_archive_root_argument(archive_sample_parser)
    add_archive_filter_arguments(archive_sample_parser, include_session=False)
    archive_sample_parser.add_argument(
        "--text",
        default=None,
        help="Case-insensitive substring to filter normalized message text.",
    )
    archive_sample_parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="Maximum number of conversations to sample after filtering. Default: 5.",
    )
    archive_sample_parser.add_argument(
        "--seed",
        default=None,
        help=(
            "Deterministic sampling seed. When omitted, a random seed is generated "
            "and returned in the response."
        ),
    )
    archive_sample_parser.set_defaults(handler=handle_archive_sample)

    archive_stats_parser = archive_subparsers.add_parser(
        "stats",
        help="Summarize archive coverage and completeness by source",
    )
    add_archive_root_argument(archive_stats_parser)
    archive_stats_parser.add_argument(
        "--source",
        default=None,
        help="Only summarize rows from the given normalized source key.",
    )
    archive_stats_parser.set_defaults(handler=handle_archive_stats)

    archive_profile_parser = archive_subparsers.add_parser(
        "profile",
        help="Summarize message roles, completeness, and limitations by source",
    )
    add_archive_root_argument(archive_profile_parser)
    archive_profile_parser.add_argument(
        "--source",
        default=None,
        help="Only summarize rows from the given normalized source key.",
    )
    archive_profile_parser.set_defaults(handler=handle_archive_profile)

    archive_anomalies_parser = archive_subparsers.add_parser(
        "anomalies",
        help="Report low-signal conversations and suspicious source aggregates",
    )
    add_archive_root_argument(archive_anomalies_parser)
    archive_anomalies_parser.add_argument(
        "--source",
        default=None,
        help="Only inspect rows from the given normalized source key.",
    )
    archive_anomalies_parser.add_argument(
        "--low-message-count",
        type=int,
        default=1,
        help="Flag conversations whose message_count is at or below this threshold.",
    )
    archive_anomalies_parser.add_argument(
        "--limitations-count",
        type=int,
        default=2,
        help="Flag conversations whose limitations count meets or exceeds this threshold.",
    )
    archive_anomalies_parser.add_argument(
        "--unsupported-count",
        type=int,
        default=2,
        help="Flag sources whose unsupported conversation count meets this threshold.",
    )
    archive_anomalies_parser.add_argument(
        "--unsupported-ratio",
        type=float,
        default=0.5,
        help="Flag sources whose unsupported conversation ratio meets this threshold.",
    )
    add_baseline_argument(archive_anomalies_parser)
    archive_anomalies_parser.set_defaults(handler=handle_archive_anomalies)

    archive_digest_parser = archive_subparsers.add_parser(
        "digest",
        help="Generate a single operator digest across archive status signals",
    )
    add_archive_root_argument(archive_digest_parser)
    add_baseline_argument(archive_digest_parser)
    archive_digest_parser.set_defaults(handler=handle_archive_digest)

    archive_audit_identities_parser = archive_subparsers.add_parser(
        "audit-identities",
        help="Report source-local identity collision candidates across archive rows",
    )
    add_archive_root_argument(archive_audit_identities_parser)
    archive_audit_identities_parser.add_argument(
        "--source",
        default=None,
        help="Only inspect rows from the given normalized source key.",
    )
    archive_audit_identities_parser.set_defaults(handler=handle_archive_audit_identities)

    archive_verify_parser = archive_subparsers.add_parser(
        "verify",
        help="Scan archive source JSONL files and report integrity findings",
    )
    add_archive_root_argument(archive_verify_parser)
    archive_verify_parser.add_argument(
        "--source",
        default=None,
        help="Only verify rows from the given normalized source key.",
    )
    add_baseline_argument(archive_verify_parser)
    archive_verify_parser.set_defaults(handler=handle_archive_verify)

    archive_quarantine_export_parser = archive_subparsers.add_parser(
        "quarantine-export",
        help="Export archive rows referenced by archive verify findings into a quarantine bundle",
    )
    add_archive_root_argument(archive_quarantine_export_parser)
    archive_quarantine_export_parser.add_argument(
        "--source",
        default=None,
        help="Only export quarantine rows from the given normalized source key.",
    )
    archive_quarantine_export_parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help=(
            "Absolute bundle directory for quarantine output. Writes "
            "quarantine.jsonl and quarantine-manifest.json when --execute is set."
        ),
    )
    archive_quarantine_export_parser.add_argument(
        "--execute",
        action="store_true",
        help="Write the quarantine bundle. Without this flag, only emit a dry-run summary.",
    )
    archive_quarantine_export_parser.set_defaults(
        handler=handle_archive_quarantine_export
    )

    archive_migrate_parser = archive_subparsers.add_parser(
        "migrate",
        help="Upgrade archive rows to the latest schema_version without changing record counts",
    )
    add_archive_root_argument(archive_migrate_parser)
    archive_migrate_parser.add_argument(
        "--source",
        default=None,
        help="Only migrate rows from the given normalized source key.",
    )
    archive_migrate_parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Optional external staging root. When set, migrate the full archive and "
            "rewrite run manifests against the staging root."
        ),
    )
    archive_migrate_parser.add_argument(
        "--backup-dir",
        type=Path,
        default=None,
        help=(
            "Optional external backup root for changed source outputs during in-place "
            "execution."
        ),
    )
    archive_migrate_parser.add_argument(
        "--execute",
        action="store_true",
        help="Write migrated output. Without this flag, only emit a dry-run summary.",
    )
    archive_migrate_parser.set_defaults(handler=handle_archive_migrate)

    archive_rewrite_parser = archive_subparsers.add_parser(
        "rewrite",
        help="Compact append-only archive outputs into canonical source JSONL files",
    )
    add_archive_root_argument(archive_rewrite_parser)
    archive_rewrite_parser.add_argument(
        "--source",
        default=None,
        help="Only rewrite rows from the given normalized source key.",
    )
    archive_rewrite_parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Optional external staging root for canonical output. Defaults to the "
            "archive root for in-place rewrites."
        ),
    )
    archive_rewrite_parser.add_argument(
        "--execute",
        action="store_true",
        help="Write canonical output. Without this flag, only emit a dry-run summary.",
    )
    archive_rewrite_parser.set_defaults(handler=handle_archive_rewrite)

    archive_prune_parser = archive_subparsers.add_parser(
        "prune",
        help="Prune old run manifests and explicit auxiliary archive artifacts",
    )
    add_archive_root_argument(archive_prune_parser)
    archive_prune_parser.add_argument(
        "--keep-last-runs",
        type=int,
        default=None,
        help=(
            "Retain at least the newest N recorded runs. When combined with "
            "--older-than-days, only runs beyond this floor are eligible for age "
            "pruning."
        ),
    )
    archive_prune_parser.add_argument(
        "--older-than-days",
        type=int,
        default=None,
        help=(
            "Prune recorded runs older than N days using completed_at, started_at, "
            "or the run identifier timestamp."
        ),
    )
    archive_prune_parser.add_argument(
        "--prune-auxiliary",
        action="store_true",
        help=(
            "Also prune default auxiliary directories under the archive root: "
            f"{', '.join(DEFAULT_AUXILIARY_DIRECTORIES)}."
        ),
    )
    archive_prune_parser.add_argument(
        "--auxiliary-dir",
        action="append",
        default=[],
        help=(
            "Additional direct child directory name under the archive root to "
            "prune. Repeat as needed."
        ),
    )
    archive_prune_parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Delete matched run manifests and auxiliary directories. Without this "
            "flag, only emit a dry-run summary."
        ),
    )
    archive_prune_parser.set_defaults(handler=handle_archive_prune)

    archive_export_memory_parser = archive_subparsers.add_parser(
        "export-memory",
        help="Export archived conversations as a stable memory ingestion contract",
    )
    add_archive_root_argument(archive_export_memory_parser)
    add_archive_filter_arguments(archive_export_memory_parser, include_session=True)
    archive_export_memory_parser.add_argument(
        "--text",
        default=None,
        help="Case-insensitive substring to filter normalized message text.",
    )
    archive_export_memory_parser.add_argument(
        "--run",
        dest="run_id",
        default=None,
        help=(
            "Only export active rows referenced by the given recorded batch run "
            "manifest."
        ),
    )
    archive_export_memory_parser.add_argument(
        "--after-collected-at",
        default=None,
        help=(
            "Only export rows with collected_at after this exact ISO-8601 timestamp "
            "with timezone."
        ),
    )
    archive_export_memory_parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help=(
            "Absolute bundle directory for memory export output. Writes "
            "memory-records.jsonl and memory-export-manifest.json when --execute "
            "is set."
        ),
    )
    archive_export_memory_parser.add_argument(
        "--execute",
        action="store_true",
        help="Write the memory export bundle. Without this flag, only emit a dry-run summary.",
    )
    archive_export_memory_parser.set_defaults(handler=handle_archive_export_memory)

    archive_export_parser = archive_subparsers.add_parser(
        "export",
        help="Export a filtered portable bundle of normalized archive conversations",
    )
    add_archive_root_argument(archive_export_parser)
    add_archive_filter_arguments(archive_export_parser, include_session=True)
    archive_export_parser.add_argument(
        "--text",
        default=None,
        help="Case-insensitive substring to filter normalized message text.",
    )
    archive_export_parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help=(
            "Absolute bundle directory for export output. Writes "
            "conversations.jsonl and export-manifest.json when --execute is set."
        ),
    )
    archive_export_parser.add_argument(
        "--execute",
        action="store_true",
        help="Write the export bundle. Without this flag, only emit a dry-run summary.",
    )
    archive_export_parser.set_defaults(handler=handle_archive_export)

    archive_import_parser = archive_subparsers.add_parser(
        "import",
        help="Import a portable export bundle into the normalized archive root",
    )
    add_archive_root_argument(archive_import_parser)
    archive_import_parser.add_argument(
        "--bundle-dir",
        required=True,
        type=Path,
        help=(
            "Absolute export bundle directory containing conversations.jsonl and "
            "export-manifest.json."
        ),
    )
    archive_import_parser.add_argument(
        "--execute",
        action="store_true",
        help="Write merged canonical archive output. Without this flag, only emit a dry-run summary.",
    )
    archive_import_parser.set_defaults(handler=handle_archive_import)

    tui_parser = subparsers.add_parser(
        "tui",
        help="Launch the read-only operator triage terminal UI",
    )
    add_archive_root_argument(tui_parser)
    add_baseline_argument(tui_parser)
    tui_parser.add_argument(
        "--view",
        choices=tui_view_choices(),
        default="overview",
        help="Initial screen to open. Also used as the snapshot target with --snapshot.",
    )
    tui_parser.add_argument(
        "--run",
        dest="run_id",
        default=None,
        help="Optional initial run selection.",
    )
    tui_parser.add_argument(
        "--source",
        default=None,
        help="Optional initial source selection.",
    )
    tui_parser.add_argument(
        "--session",
        default=None,
        help="Optional initial source_session_id selection within the sample view.",
    )
    tui_parser.add_argument(
        "--sample-count",
        type=int,
        default=TUI_DEFAULT_SAMPLE_COUNT,
        help=(
            "Conversation sample size for the sample drill-down. "
            f"Default: {TUI_DEFAULT_SAMPLE_COUNT}"
        ),
    )
    tui_parser.add_argument(
        "--sample-seed",
        default=TUI_DEFAULT_SAMPLE_SEED,
        help=(
            "Stable sampling seed so repeated refreshes keep the same sample set. "
            f"Default: {TUI_DEFAULT_SAMPLE_SEED}"
        ),
    )
    tui_parser.add_argument(
        "--snapshot",
        action="store_true",
        help="Render the selected view as plain text and exit instead of opening curses.",
    )
    tui_parser.add_argument(
        "--width",
        type=int,
        default=TUI_DEFAULT_SNAPSHOT_WIDTH,
        help=(
            "Snapshot render width when --snapshot is set. "
            f"Default: {TUI_DEFAULT_SNAPSHOT_WIDTH}"
        ),
    )
    tui_parser.set_defaults(handler=handle_tui)

    return parser


def _build_help_handler(
    parser: argparse.ArgumentParser,
) -> Callable[[argparse.Namespace], int]:
    def _handler(_args: argparse.Namespace) -> int:
        parser.print_help()
        return 0

    return _handler


def add_archive_root_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=DEFAULT_ARCHIVE_ROOT,
        help=(
            "Absolute archive root outside the repository. "
            f"Default: {DEFAULT_ARCHIVE_ROOT}"
        ),
    )


def add_baseline_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help=(
            "Absolute baseline policy file path. "
            f"Default: <archive-root>/{BASELINE_POLICY_FILENAME}"
        ),
    )


def add_archive_filter_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_session: bool,
) -> None:
    parser.add_argument(
        "--source",
        default=None,
        help="Only inspect rows from the given normalized source key.",
    )
    if include_session:
        parser.add_argument(
            "--session",
            default=None,
            help="Only inspect rows with the given source_session_id.",
        )
    parser.add_argument(
        "--transcript-completeness",
        choices=tuple(
            completeness.value for completeness in TranscriptCompleteness
        ),
        default=None,
        help="Only inspect rows with the given transcript completeness status.",
    )


def validate_archive_root(candidate: Path) -> Path:
    policy = ArchiveTargetPolicy(repo_root=repository_root())
    return policy.validate(candidate)


def validate_absolute_path(candidate: Path, *, label: str) -> Path:
    resolved_candidate = candidate.expanduser()
    if not resolved_candidate.is_absolute():
        raise ValueError(f"{label} must be an absolute path")
    return resolved_candidate.resolve(strict=False)


def resolve_baseline_policy(
    archive_root: Path,
    *,
    baseline: Path | None,
    allow_missing: bool,
):
    resolved_baseline = (
        None
        if baseline is None
        else validate_absolute_path(baseline, label="baseline policy path")
    )
    resolved_path = baseline_policy_path(archive_root, baseline_path=resolved_baseline)
    return load_baseline_policy(resolved_path, allow_missing=allow_missing)


def emit_json(payload: dict[str, object]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def resolve_cli_input_roots(input_roots: Sequence[Path]) -> tuple[Path, ...]:
    return tuple(
        input_root.expanduser().resolve(strict=False) for input_root in input_roots
    )


def validate_target_selection(
    *,
    command_name: str,
    target_source: str | None,
    inspect_all: bool,
    has_batch_selection_options: bool,
) -> str | None:
    if inspect_all and target_source is not None:
        return "choose either a source or --all"
    if not inspect_all and target_source is None:
        return f"{command_name} requires a source or --all"
    if target_source is not None and has_batch_selection_options:
        return (
            "source selection options require --all and cannot be combined with a single source"
        )
    return None


def handle_sources(args: argparse.Namespace) -> int:
    registry = build_registry()
    if args.format == "json":
        emit_json(
            {
                "default_batch_profile": SourceSelectionProfile.DEFAULT.value,
                "sources": [
                    entry.to_dict() for entry in build_source_support_matrix(registry)
                ],
            }
        )
        return 0
    if args.format == "markdown":
        print(
            render_source_support_matrix_markdown(
                build_source_support_matrix(registry)
            ),
            end="",
        )
        return 0
    for collector in registry.list():
        root_resolution = resolve_source_roots(collector.descriptor)
        roots = ", ".join(root_resolution.to_dict()["resolved_roots"])
        if not roots and root_resolution.miss_reasons:
            roots = "; ".join(root_resolution.miss_reasons)
        print(
            f"{collector.descriptor.key}\t"
            f"{collector.descriptor.support_level.value}\t"
            f"{roots}"
        )
    return 0


def handle_contract(_args: argparse.Namespace) -> int:
    policy = ArchiveTargetPolicy(repo_root=repository_root())
    payload = {
        "archive_target_policy": policy.to_dict(),
        "normalization_contract": NormalizationContract().to_dict(),
        "sources": [
            collector.descriptor.to_dict() for collector in build_registry().list()
        ],
    }
    emit_json(payload)
    return 0


def handle_config_init(args: argparse.Namespace) -> int:
    output_path = resolve_collect_config_output_path(args.output)
    try:
        archive_root = validate_archive_root(args.archive_root)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.print_template:
        print(render_collect_config_template(archive_root=archive_root), end="")
        return 0

    existed_before_write = output_path.exists()
    try:
        scaffold_collect_config(
            output_path=output_path,
            archive_root=archive_root,
            force=args.force,
        )
    except CollectorConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except OSError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    emit_json(
        {
            "archive_root": str(archive_root),
            "output_path": str(output_path),
            "overwrote": existed_before_write,
            "written": True,
        }
    )
    return 0


def _build_batch_payload(
    result,
    *,
    archive_root: Path,
    validation_mode: ValidationMode,
) -> tuple[dict[str, object], int]:
    payload = result.to_dict()
    exit_code = 1 if any(source.failed for source in result.sources) else 0
    if validation_mode != ValidationMode.OFF:
        report = validate_run(
            archive_root,
            run_id=result.run_id,
            repo_root=repository_root(),
        )
        payload["validation"] = {
            "mode": validation_mode.value,
            "status": report.status.value,
            "success_count": report.success_count,
            "warning_count": report.warning_count,
            "error_count": report.error_count,
        }
        if validation_mode == ValidationMode.STRICT and report.error_count:
            exit_code = 1
    return payload, exit_code


def handle_collect(args: argparse.Namespace) -> int:
    try:
        effective_config = resolve_collect_config(
            config_path=args.config,
            cli_archive_root=args.archive_root,
            cli_profile=args.profile,
            cli_include_sources=args.selected_sources,
            cli_exclude_sources=args.exclude_source,
            cli_incremental=args.incremental,
            cli_redaction=args.redaction,
            cli_validation=args.validation,
        )
        archive_root = validate_archive_root(effective_config.archive_root)
        effective_config = replace(effective_config, archive_root=archive_root)
    except (CollectorConfigError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    has_batch_selection_options = (
        effective_config.selection_policy.profile != SourceSelectionProfile.ALL
        or bool(effective_config.selection_policy.include_sources)
        or bool(effective_config.selection_policy.exclude_sources)
    )
    validation_error = validate_target_selection(
        command_name="collect",
        target_source=args.target_source,
        inspect_all=args.all,
        has_batch_selection_options=has_batch_selection_options,
    )
    if validation_error is not None:
        print(validation_error, file=sys.stderr)
        return 2

    input_roots = resolve_cli_input_roots(args.input_root)
    registry = build_registry()

    if args.all:
        unknown_sources = _find_unknown_sources(
            registry.keys(),
            effective_config.selection_policy.include_sources,
            effective_config.selection_policy.exclude_sources,
        )
        if unknown_sources:
            print(
                f"unknown sources in selection policy: {', '.join(unknown_sources)}",
                file=sys.stderr,
            )
            return 2
        result = run_collection_batch(
            registry,
            archive_root,
            input_roots=input_roots or None,
            selection_policy=effective_config.selection_policy,
            execution_policy=effective_config.execution_policy,
            effective_config=effective_config,
        )
        payload, exit_code = _build_batch_payload(
            result,
            archive_root=archive_root,
            validation_mode=effective_config.execution_policy.validation,
        )
        emit_json(payload)
        return exit_code

    collector = registry.get(args.target_source)
    root_resolution = resolve_source_roots(
        collector.descriptor,
        input_roots=input_roots or None,
    )

    if args.execute:
        if not isinstance(collector, ExecutableCollector):
            print(
                f"real collection writes are not implemented for source: {args.target_source}",
                file=sys.stderr,
            )
            return 1
        with collection_execution_policy_context(effective_config.execution_policy):
            result = collector.collect(archive_root, input_roots=input_roots or None)
        payload = result.to_dict()
        payload["root_resolution"] = root_resolution.to_dict()
        payload["effective_config"] = effective_config.to_dict()
        emit_json(payload)
        return 0

    plan = collector.build_plan(archive_root)
    payload = plan.to_dict()
    payload["root_resolution"] = root_resolution.to_dict()
    if input_roots:
        payload["input_roots"] = [str(input_root) for input_root in input_roots]
    payload["effective_config"] = effective_config.to_dict()
    emit_json(payload)
    return 0


def handle_rerun(args: argparse.Namespace) -> int:
    try:
        effective_config = resolve_collect_config(
            config_path=args.config,
            cli_archive_root=args.archive_root,
            cli_profile=SourceSelectionProfile.ALL.value,
            cli_include_sources=(),
            cli_exclude_sources=(),
            cli_incremental=args.incremental,
            cli_redaction=args.redaction,
            cli_validation=args.validation,
        )
        archive_root = validate_archive_root(effective_config.archive_root)
        origin_summary = load_run_summary(
            archive_root,
            args.run_id,
            verify_output_paths=False,
        )
        effective_rerun_config = resolve_rerun_config(
            cli_reason=args.reason,
            configured_rerun=effective_config.rerun,
        )
        rerun = plan_rerun(
            origin_summary,
            selection_reason=effective_rerun_config.selection_reason,
            include_sources=args.selected_sources,
            exclude_sources=args.exclude_source,
        )
        effective_config = replace(
            effective_config,
            archive_root=archive_root,
            selection_policy=rerun.selection_policy,
            rerun=effective_rerun_config,
        )
    except RunReportingError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except RerunSelectionError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except (CollectorConfigError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    input_roots = resolve_cli_input_roots(args.input_root)
    registry = build_registry()
    unknown_sources = _find_unknown_sources(
        registry.keys(),
        effective_config.selection_policy.include_sources,
        effective_config.selection_policy.exclude_sources,
    )
    if unknown_sources:
        print(
            f"unknown sources in selection policy: {', '.join(unknown_sources)}",
            file=sys.stderr,
        )
        return 2

    result = run_collection_batch(
        registry,
        archive_root,
        input_roots=input_roots or None,
        selection_policy=effective_config.selection_policy,
        execution_policy=effective_config.execution_policy,
        effective_config=effective_config,
        rerun=rerun.metadata,
    )
    payload, exit_code = _build_batch_payload(
        result,
        archive_root=archive_root,
        validation_mode=effective_config.execution_policy.validation,
    )
    emit_json(payload)
    return exit_code


def handle_scheduled_run(args: argparse.Namespace) -> int:
    try:
        scheduled_config = resolve_scheduled_config(
            config_path=args.config,
            cli_archive_root=args.archive_root,
            cli_mode=args.mode,
            cli_stale_after_seconds=args.stale_after_seconds,
        )
        archive_root = validate_archive_root(scheduled_config.archive_root)
        scheduled_config = replace(scheduled_config, archive_root=archive_root)
    except (CollectorConfigError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    input_roots = resolve_cli_input_roots(args.input_root)
    registry = build_registry()

    try:
        with acquire_scheduled_lock(
            archive_root,
            mode=scheduled_config.mode,
            stale_after_seconds=scheduled_config.stale_after_seconds,
            force_unlock_stale=args.force_unlock_stale,
        ) as lock_acquisition:
            if scheduled_config.mode == ScheduledRunMode.COLLECT:
                effective_config = _scheduled_collect_config(scheduled_config)
                unknown_sources = _find_unknown_sources(
                    registry.keys(),
                    effective_config.selection_policy.include_sources,
                    effective_config.selection_policy.exclude_sources,
                )
                if unknown_sources:
                    print(
                        f"unknown sources in selection policy: {', '.join(unknown_sources)}",
                        file=sys.stderr,
                    )
                    return 2
                scheduled_metadata = ScheduledRunMetadata(
                    mode=scheduled_config.mode,
                    lock=lock_acquisition.lock,
                    stale_after_seconds=scheduled_config.stale_after_seconds,
                    config_source=scheduled_config.config_source,
                    force_unlocked_stale_lock=lock_acquisition.force_unlocked_stale_lock,
                    replaced_lock=lock_acquisition.replaced_lock,
                )
                result = run_collection_batch(
                    registry,
                    archive_root,
                    input_roots=input_roots or None,
                    selection_policy=effective_config.selection_policy,
                    execution_policy=effective_config.execution_policy,
                    effective_config=effective_config,
                    scheduled=scheduled_metadata,
                )
                payload, exit_code = _build_batch_payload(
                    result,
                    archive_root=archive_root,
                    validation_mode=effective_config.execution_policy.validation,
                )
                emit_json(payload)
                return exit_code

            if scheduled_config.rerun is None:
                print(
                    "scheduled rerun requires [scheduled.rerun] selection_preset or [rerun] selection_preset",
                    file=sys.stderr,
                )
                return 2

            try:
                origin_summary = (
                    load_latest_run_summary(archive_root, verify_output_paths=False)
                    if args.run_id is None
                    else load_run_summary(
                        archive_root,
                        args.run_id,
                        verify_output_paths=False,
                    )
                )
            except RunReportingError as exc:
                print(str(exc), file=sys.stderr)
                return 1

            try:
                rerun = plan_rerun(
                    origin_summary,
                    selection_reason=scheduled_config.rerun.selection_reason,
                )
            except RerunSelectionError as exc:
                print(str(exc), file=sys.stderr)
                return 1

            effective_config = _scheduled_collect_config(
                scheduled_config,
                selection_policy=rerun.selection_policy,
                rerun=scheduled_config.rerun,
            )
            unknown_sources = _find_unknown_sources(
                registry.keys(),
                effective_config.selection_policy.include_sources,
                effective_config.selection_policy.exclude_sources,
            )
            if unknown_sources:
                print(
                    f"unknown sources in selection policy: {', '.join(unknown_sources)}",
                    file=sys.stderr,
                )
                return 2
            scheduled_metadata = ScheduledRunMetadata(
                mode=scheduled_config.mode,
                lock=lock_acquisition.lock,
                stale_after_seconds=scheduled_config.stale_after_seconds,
                config_source=scheduled_config.config_source,
                origin_run_id=origin_summary.run_id,
                force_unlocked_stale_lock=lock_acquisition.force_unlocked_stale_lock,
                replaced_lock=lock_acquisition.replaced_lock,
            )
            result = run_collection_batch(
                registry,
                archive_root,
                input_roots=input_roots or None,
                selection_policy=effective_config.selection_policy,
                execution_policy=effective_config.execution_policy,
                effective_config=effective_config,
                rerun=rerun.metadata,
                scheduled=scheduled_metadata,
            )
            payload, exit_code = _build_batch_payload(
                result,
                archive_root=archive_root,
                validation_mode=effective_config.execution_policy.validation,
            )
            emit_json(payload)
            return exit_code
    except ScheduledLockError as exc:
        emit_json(
            {
                "archive_root": str(archive_root),
                "status": "skipped",
                "reason": (
                    "scheduled_lock_stale"
                    if exc.status == "stale"
                    else "scheduled_lock_held"
                ),
                "message": str(exc),
                "scheduled": {
                    "mode": scheduled_config.mode.value,
                    "stale_after_seconds": scheduled_config.stale_after_seconds,
                },
                "lock": exc.lock.to_dict(),
                "force_unlock_stale_available": exc.status == "stale",
            }
        )
        return 1


def _scheduled_collect_config(
    scheduled_config,
    *,
    selection_policy=None,
    rerun=None,
):
    return EffectiveCollectConfig(
        archive_root=scheduled_config.archive_root,
        selection_policy=selection_policy or scheduled_config.selection_policy,
        execution_policy=scheduled_config.execution_policy,
        rerun=scheduled_config.rerun if rerun is None else rerun,
        config_source=scheduled_config.config_source,
        config_path=scheduled_config.config_path,
    )


def handle_doctor(args: argparse.Namespace) -> int:
    validation_error = validate_target_selection(
        command_name="doctor",
        target_source=args.target_source,
        inspect_all=args.all,
        has_batch_selection_options=(
            args.profile is not None
            or bool(args.selected_sources)
            or bool(args.exclude_source)
        ),
    )
    if validation_error is not None:
        print(validation_error, file=sys.stderr)
        return 2

    input_roots = resolve_cli_input_roots(args.input_root)
    registry = build_registry()

    if args.all:
        selection_policy = build_source_selection_policy(
            profile=args.profile,
            include_sources=args.selected_sources,
            exclude_sources=args.exclude_source,
        )
        emit_json(
            inspect_registry_readiness(
                registry,
                input_roots=input_roots or None,
                selection_policy=selection_policy,
            ).to_dict()
        )
        return 0

    collector = registry.get(args.target_source)
    emit_json(
        inspect_source_readiness(
            collector,
            input_roots=input_roots or None,
        ).to_dict()
    )
    return 0


def handle_acceptance_ship(args: argparse.Namespace) -> int:
    try:
        archive_root = validate_archive_root(args.archive_root)
        snapshot_path = (
            None
            if args.snapshot_path is None
            else validate_absolute_path(
                args.snapshot_path,
                label="ship acceptance snapshot path",
            )
        )
        report = run_ship_acceptance(
            build_registry(),
            archive_root=archive_root,
            repo_root=repository_root(),
            snapshot_path=snapshot_path,
        )
    except (
        ArchiveInspectError,
        ArchiveImportError,
        ArchiveVerifyError,
        CollectorConfigError,
        OSError,
        RunReportingError,
        ValueError,
    ) as exc:
        print(str(exc), file=sys.stderr)
        return 2 if isinstance(exc, ValueError) else 1

    emit_json(report.to_dict())
    return 0 if report.status == "pass" else 1


def handle_runs_list(args: argparse.Namespace) -> int:
    try:
        archive_root = validate_archive_root(args.archive_root)
        runs = list_run_summaries(archive_root)
    except RunReportingError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    emit_json(
        {
            "archive_root": str(archive_root),
            "run_count": len(runs),
            "runs": [run.to_overview_dict() for run in runs],
        }
    )
    return 0


def handle_runs_latest(args: argparse.Namespace) -> int:
    return _handle_runs_summary(args, latest=True)


def handle_runs_show(args: argparse.Namespace) -> int:
    return _handle_runs_summary(args, latest=False)


def handle_runs_diff(args: argparse.Namespace) -> int:
    try:
        archive_root = validate_archive_root(args.archive_root)
        diff = load_run_diff(
            archive_root,
            from_run_id=args.from_run_id,
            to_run_id=args.to_run_id,
        )
    except RunReportingError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    emit_json(diff.to_dict())
    return 0


def handle_runs_trend(args: argparse.Namespace) -> int:
    try:
        archive_root = validate_archive_root(args.archive_root)
        report = load_run_trend(
            archive_root,
            sources=tuple(args.sources),
        )
    except RunReportingError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    emit_json(report.to_dict())
    return 0


def handle_validate(args: argparse.Namespace) -> int:
    try:
        archive_root = validate_archive_root(args.archive_root)
        baseline_policy = resolve_baseline_policy(
            archive_root,
            baseline=args.baseline,
            allow_missing=True,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    report = validate_run(
        archive_root,
        run_id=args.run_id,
        repo_root=repository_root(),
        baseline_policy=baseline_policy,
    )
    emit_json(report.to_dict())
    return 1 if report.error_count else 0


def handle_baseline_snapshot(args: argparse.Namespace) -> int:
    try:
        archive_root = validate_archive_root(args.archive_root)
        baseline_path = baseline_policy_path(
            archive_root,
            baseline_path=(
                None
                if args.baseline is None
                else validate_absolute_path(args.baseline, label="baseline policy path")
            ),
        )
        baseline_policy = load_baseline_policy(baseline_path, allow_missing=True)
        if args.snapshot_from == "validate":
            if args.run_id is None:
                raise ValueError("baseline snapshot --from validate requires --run")
            report = validate_run(
                archive_root,
                run_id=args.run_id,
                repo_root=repository_root(),
            )
            snapshot_entries = snapshot_entries_from_validate(report, reason=args.reason)
        elif args.snapshot_from == "archive-verify":
            report = verify_archive(
                archive_root,
                source=args.source,
            )
            snapshot_entries = snapshot_entries_from_archive_verify(
                report,
                reason=args.reason,
            )
        else:
            report = summarize_archive_anomalies(
                archive_root,
                source=args.source,
                thresholds=ArchiveAnomalyThresholds(
                    low_message_count=args.low_message_count,
                    limitations_count=args.limitations_count,
                    unsupported_count=args.unsupported_count,
                    unsupported_ratio=args.unsupported_ratio,
                ),
            )
            snapshot_entries = snapshot_entries_from_archive_anomalies(
                report,
                reason=args.reason,
            )
        merged_policy, added_entry_count = merge_baseline_entries(
            baseline_policy,
            path=baseline_path,
            entries=snapshot_entries,
        )
        save_baseline_policy(merged_policy)
    except (ArchiveInspectError, ArchiveVerifyError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except OSError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    emit_json(
        {
            "archive_root": str(archive_root),
            "baseline_path": str(baseline_path),
            "snapshot_from": args.snapshot_from,
            "reason": args.reason,
            "source_filter": args.source,
            "run_id": args.run_id,
            "snapshot_entry_count": len(snapshot_entries),
            "added_entry_count": added_entry_count,
            "entry_count": merged_policy.entry_count,
            "entries": [entry.to_dict() for entry in snapshot_entries],
        }
    )
    return 0


def handle_archive_index_status(args: argparse.Namespace) -> int:
    try:
        archive_root = validate_archive_root(args.archive_root)
        status = inspect_archive_index(archive_root)
    except ArchiveInspectError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    emit_json(status.to_dict())
    return 0


def handle_archive_index_refresh(args: argparse.Namespace) -> int:
    try:
        archive_root = validate_archive_root(args.archive_root)
        report = refresh_archive_index(
            archive_root,
            force=args.force,
        )
    except ArchiveInspectError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    emit_json(report.to_dict())
    return 0


def handle_archive_list(args: argparse.Namespace) -> int:
    try:
        archive_root = validate_archive_root(args.archive_root)
        conversations = list_archive_conversations(
            archive_root,
            source=args.source,
            session=args.session,
            transcript_completeness=args.transcript_completeness,
        )
    except ArchiveInspectError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    emit_json(
        {
            "archive_root": str(archive_root),
            "conversation_count": len(conversations),
            "conversations": [conversation.to_dict() for conversation in conversations],
        }
    )
    return 0


def handle_archive_show(args: argparse.Namespace) -> int:
    try:
        archive_root = validate_archive_root(args.archive_root)
        conversation = show_archive_conversation(
            archive_root,
            source=args.source,
            session=args.session,
        )
    except ArchiveInspectError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    emit_json(
        {
            "archive_root": str(archive_root),
            "summary": conversation.summary.to_dict(),
            "conversation": conversation.to_dict(),
        }
    )
    return 0


def handle_archive_find(args: argparse.Namespace) -> int:
    try:
        archive_root = validate_archive_root(args.archive_root)
        conversations = find_archive_conversations(
            archive_root,
            text=args.text,
            source=args.source,
            transcript_completeness=args.transcript_completeness,
        )
    except ArchiveInspectError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    emit_json(
        {
            "archive_root": str(archive_root),
            "text": args.text,
            "conversation_count": len(conversations),
            "conversations": [conversation.to_dict() for conversation in conversations],
        }
    )
    return 0


def handle_archive_sample(args: argparse.Namespace) -> int:
    try:
        archive_root = validate_archive_root(args.archive_root)
        report = sample_archive_subset(
            archive_root,
            count=args.count,
            source=args.source,
            transcript_completeness=args.transcript_completeness,
            text=args.text,
            seed=args.seed,
        )
    except ArchiveInspectError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    emit_json(report.to_dict())
    return 0


def handle_archive_stats(args: argparse.Namespace) -> int:
    try:
        archive_root = validate_archive_root(args.archive_root)
        report = summarize_archive_stats(
            archive_root,
            source=args.source,
        )
    except ArchiveInspectError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    emit_json(report.to_dict())
    return 0


def handle_archive_profile(args: argparse.Namespace) -> int:
    try:
        archive_root = validate_archive_root(args.archive_root)
        report = summarize_archive_profile(
            archive_root,
            source=args.source,
        )
    except ArchiveInspectError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    emit_json(report.to_dict())
    return 0


def handle_archive_anomalies(args: argparse.Namespace) -> int:
    try:
        archive_root = validate_archive_root(args.archive_root)
        baseline_policy = resolve_baseline_policy(
            archive_root,
            baseline=args.baseline,
            allow_missing=True,
        )
        report = summarize_archive_anomalies(
            archive_root,
            source=args.source,
            thresholds=ArchiveAnomalyThresholds(
                low_message_count=args.low_message_count,
                limitations_count=args.limitations_count,
                unsupported_count=args.unsupported_count,
                unsupported_ratio=args.unsupported_ratio,
            ),
            baseline_policy=baseline_policy,
        )
    except ArchiveInspectError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    emit_json(report.to_dict())
    return 0


def handle_archive_digest(args: argparse.Namespace) -> int:
    try:
        archive_root = validate_archive_root(args.archive_root)
        baseline_policy = resolve_baseline_policy(
            archive_root,
            baseline=args.baseline,
            allow_missing=True,
        )
        report = summarize_archive_digest(
            archive_root,
            baseline_policy=baseline_policy,
        )
    except (ArchiveInspectError, ArchiveVerifyError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    emit_json(report.to_dict())
    return 0


def handle_archive_audit_identities(args: argparse.Namespace) -> int:
    try:
        archive_root = validate_archive_root(args.archive_root)
        report = audit_archive_identities(
            archive_root,
            source=args.source,
        )
    except ArchiveInspectError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    emit_json(report.to_dict())
    return 0


def handle_archive_verify(args: argparse.Namespace) -> int:
    try:
        archive_root = validate_archive_root(args.archive_root)
        baseline_policy = resolve_baseline_policy(
            archive_root,
            baseline=args.baseline,
            allow_missing=True,
        )
        report = verify_archive(
            archive_root,
            source=args.source,
            baseline_policy=baseline_policy,
        )
    except ArchiveVerifyError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    emit_json(report.to_dict())
    return 1 if report.error_count else 0


def handle_archive_quarantine_export(args: argparse.Namespace) -> int:
    try:
        archive_root = validate_archive_root(args.archive_root)
        output_dir = validate_archive_root(args.output_dir)
        report = export_archive_quarantine(
            archive_root,
            output_dir=output_dir,
            source=args.source,
            execute=args.execute,
        )
    except (ArchiveQuarantineExportError, ArchiveVerifyError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except OSError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    emit_json(report.to_dict())
    return 0


def handle_archive_migrate(args: argparse.Namespace) -> int:
    try:
        archive_root = validate_archive_root(args.archive_root)
        output_root = (
            None
            if args.output_root is None
            else validate_archive_root(args.output_root)
        )
        backup_dir = (
            None
            if args.backup_dir is None
            else validate_archive_root(args.backup_dir)
        )
        report = migrate_archive(
            archive_root,
            output_root=output_root,
            source=args.source,
            backup_dir=backup_dir,
            execute=args.execute,
        )
    except (ArchiveInspectError, ArchiveMigrateError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except OSError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    emit_json(report.to_dict())
    return 0


def handle_archive_rewrite(args: argparse.Namespace) -> int:
    try:
        archive_root = validate_archive_root(args.archive_root)
        output_root = (
            archive_root
            if args.output_root is None
            else validate_archive_root(args.output_root)
        )
        report = rewrite_archive(
            archive_root,
            output_root=output_root,
            source=args.source,
            execute=args.execute,
        )
    except ArchiveInspectError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except OSError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    emit_json(report.to_dict())
    return 0


def handle_archive_export(args: argparse.Namespace) -> int:
    try:
        archive_root = validate_archive_root(args.archive_root)
        output_dir = validate_absolute_path(
            args.output_dir,
            label="archive export output_dir",
        )
        report = export_archive_subset(
            archive_root,
            output_dir=output_dir,
            source=args.source,
            session=args.session,
            transcript_completeness=args.transcript_completeness,
            text=args.text,
            execute=args.execute,
        )
    except ArchiveInspectError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except OSError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    emit_json(report.to_dict())
    return 0


def handle_archive_export_memory(args: argparse.Namespace) -> int:
    try:
        archive_root = validate_archive_root(args.archive_root)
        output_dir = validate_absolute_path(
            args.output_dir,
            label="archive export-memory output_dir",
        )
        report = export_archive_memory_records(
            archive_root,
            output_dir=output_dir,
            source=args.source,
            session=args.session,
            transcript_completeness=args.transcript_completeness,
            text=args.text,
            run_id=args.run_id,
            after_collected_at=args.after_collected_at,
            execute=args.execute,
        )
    except ArchiveInspectError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except RunReportingError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except OSError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    emit_json(report.to_dict())
    return 0


def handle_archive_import(args: argparse.Namespace) -> int:
    try:
        archive_root = validate_archive_root(args.archive_root)
        bundle_dir = validate_absolute_path(
            args.bundle_dir,
            label="archive import bundle_dir",
        )
        report = import_archive_bundle(
            archive_root,
            bundle_dir=bundle_dir,
            execute=args.execute,
        )
    except ArchiveImportError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except ArchiveInspectError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except OSError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    emit_json(report.to_dict())
    return 0


def handle_archive_prune(args: argparse.Namespace) -> int:
    try:
        archive_root = validate_archive_root(args.archive_root)
        report = prune_archive(
            archive_root,
            keep_last_runs=args.keep_last_runs,
            older_than_days=args.older_than_days,
            prune_auxiliary=args.prune_auxiliary,
            auxiliary_directories=tuple(args.auxiliary_dir),
            execute=args.execute,
        )
    except ArchivePruneError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except OSError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    emit_json(report.to_dict())
    return 0


def handle_tui(args: argparse.Namespace) -> int:
    try:
        archive_root = validate_archive_root(args.archive_root)
        baseline_policy = resolve_baseline_policy(
            archive_root,
            baseline=args.baseline,
            allow_missing=True,
        )
        if args.snapshot:
            print(
                render_tui_snapshot(
                    archive_root,
                    baseline_policy=baseline_policy,
                    view=args.view,
                    selected_run_id=args.run_id,
                    selected_source=args.source,
                    selected_session=args.session,
                    sample_count=args.sample_count,
                    sample_seed=args.sample_seed,
                    width=args.width,
                )
            )
        else:
            run_operator_triage_tui(
                archive_root,
                baseline_policy=baseline_policy,
                initial_view=args.view,
                selected_run_id=args.run_id,
                selected_source=args.source,
                selected_session=args.session,
                sample_count=args.sample_count,
                sample_seed=args.sample_seed,
            )
    except (ArchiveInspectError, ArchiveVerifyError, RunReportingError, TuiError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    return 0


def _handle_runs_summary(args: argparse.Namespace, *, latest: bool) -> int:
    try:
        archive_root = validate_archive_root(args.archive_root)
        summary = (
            load_latest_run_summary(archive_root)
            if latest
            else load_run_summary(archive_root, args.run_id)
        )
    except RunReportingError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    emit_json(summary.to_dict())
    return 0


def _find_unknown_sources(
    known_sources: Sequence[str],
    include_sources: Sequence[str],
    exclude_sources: Sequence[str],
) -> tuple[str, ...]:
    known = set(known_sources)
    unknown = sorted((set(include_sources) | set(exclude_sources)) - known)
    return tuple(unknown)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 0
    return handler(args)
