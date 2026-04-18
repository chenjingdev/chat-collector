from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
import textwrap

from .archive_digest import (
    ArchiveDigestReport,
    ArchiveDigestSourceReport,
    summarize_archive_digest,
)
from .archive_inspect import ArchiveConversationRecord, ArchiveInspectError
from .archive_profile import (
    ArchiveProfileReport,
    ArchiveSourceProfile,
    summarize_archive_profile,
)
from .archive_sample import (
    ArchiveConversationSample,
    ArchiveSampleReport,
    sample_archive_subset,
)
from .archive_stats import ArchiveSourceStats, ArchiveStatsReport, summarize_archive_stats
from .archive_verify import ArchiveVerifyError
from .baseline_policy import BaselinePolicy
from .reporting import RunReportingError, RunSummary, list_run_summaries

DEFAULT_SAMPLE_COUNT = 5
DEFAULT_SAMPLE_SEED = "operator-triage"
DEFAULT_SNAPSHOT_WIDTH = 100
_MISSING_RUN_HISTORY_PREFIXES = (
    "run manifests directory does not exist:",
    "no run manifests found under:",
)


class TuiError(ValueError):
    """Raised when the operator TUI cannot load archive state or start."""


class TuiView(StrEnum):
    OVERVIEW = "overview"
    RUNS = "runs"
    SOURCES = "sources"
    SAMPLES = "samples"
    HELP = "help"


@dataclass(frozen=True, slots=True)
class TuiBundle:
    archive_root: Path
    runs: tuple[RunSummary, ...]
    runs_error: str | None
    digest: ArchiveDigestReport
    stats: ArchiveStatsReport
    profile: ArchiveProfileReport

    @property
    def source_reports(self) -> tuple[ArchiveDigestSourceReport, ...]:
        return _sorted_source_reports(self.digest.sources)


@dataclass(frozen=True, slots=True)
class TuiSelectionSnapshot:
    bundle: TuiBundle
    view: TuiView
    sample_count: int
    sample_seed: str
    selected_run: RunSummary | None
    selected_source_name: str | None
    selected_source_report: ArchiveDigestSourceReport | None
    selected_source_stats: ArchiveSourceStats | None
    selected_source_profile: ArchiveSourceProfile | None
    sample_report: ArchiveSampleReport | None
    selected_sample: ArchiveConversationSample | None
    selected_conversation: ArchiveConversationRecord | None
    sample_error: str | None = None
    conversation_error: str | None = None


def tui_view_choices() -> tuple[str, ...]:
    return tuple(view.value for view in TuiView)


def load_tui_bundle(
    archive_root: Path,
    *,
    baseline_policy: BaselinePolicy | None = None,
) -> TuiBundle:
    resolved_root = archive_root.expanduser().resolve(strict=False)
    try:
        runs = list_run_summaries(resolved_root)
        runs_error = None
    except RunReportingError as exc:
        if str(exc).startswith(_MISSING_RUN_HISTORY_PREFIXES):
            runs = ()
            runs_error = None
        else:
            runs = ()
            runs_error = str(exc)

    try:
        digest = summarize_archive_digest(
            resolved_root,
            baseline_policy=baseline_policy,
        )
        stats = summarize_archive_stats(resolved_root)
        profile = summarize_archive_profile(resolved_root)
    except (ArchiveInspectError, ArchiveVerifyError, RunReportingError, ValueError) as exc:
        raise TuiError(str(exc)) from exc

    return TuiBundle(
        archive_root=resolved_root,
        runs=runs,
        runs_error=runs_error,
        digest=digest,
        stats=stats,
        profile=profile,
    )


def build_selection_snapshot(
    bundle: TuiBundle,
    *,
    view: TuiView,
    selected_run_id: str | None,
    selected_source: str | None,
    selected_session: str | None,
    sample_count: int,
    sample_seed: str,
) -> TuiSelectionSnapshot:
    selected_run = _select_run(bundle.runs, selected_run_id)
    selected_source_name = _select_source_name(
        bundle.source_reports,
        requested_source=selected_source,
    )
    source_report = _find_by_source(bundle.digest.sources, selected_source_name)
    source_stats = _find_by_source(bundle.stats.sources, selected_source_name)
    source_profile = _find_by_source(bundle.profile.sources, selected_source_name)

    sample_report: ArchiveSampleReport | None = None
    selected_sample: ArchiveConversationSample | None = None
    selected_conversation: ArchiveConversationRecord | None = None
    sample_error: str | None = None
    conversation_error: str | None = None

    if selected_source_name is not None:
        try:
            sample_report = sample_archive_subset(
                bundle.archive_root,
                count=sample_count,
                source=selected_source_name,
                seed=sample_seed,
            )
        except (ArchiveInspectError, ValueError) as exc:
            sample_error = str(exc)
        else:
            selected_sample = _select_sample(
                sample_report.conversations,
                requested_session=selected_session,
            )
            if (
                selected_sample is not None
                and selected_sample.conversation.source_session_id is not None
            ):
                try:
                    from .archive_inspect import show_archive_conversation

                    selected_conversation = show_archive_conversation(
                        bundle.archive_root,
                        source=selected_source_name,
                        session=selected_sample.conversation.source_session_id,
                    )
                except ArchiveInspectError as exc:
                    conversation_error = str(exc)

    return TuiSelectionSnapshot(
        bundle=bundle,
        view=view,
        sample_count=sample_count,
        sample_seed=sample_seed,
        selected_run=selected_run,
        selected_source_name=selected_source_name,
        selected_source_report=source_report,
        selected_source_stats=source_stats,
        selected_source_profile=source_profile,
        sample_report=sample_report,
        selected_sample=selected_sample,
        selected_conversation=selected_conversation,
        sample_error=sample_error,
        conversation_error=conversation_error,
    )


def render_tui_snapshot(
    archive_root: Path,
    *,
    baseline_policy: BaselinePolicy | None = None,
    view: TuiView | str = TuiView.OVERVIEW,
    selected_run_id: str | None = None,
    selected_source: str | None = None,
    selected_session: str | None = None,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    sample_seed: str = DEFAULT_SAMPLE_SEED,
    width: int = DEFAULT_SNAPSHOT_WIDTH,
) -> str:
    resolved_view = TuiView(view)
    bundle = load_tui_bundle(
        archive_root,
        baseline_policy=baseline_policy,
    )
    selection = build_selection_snapshot(
        bundle,
        view=resolved_view,
        selected_run_id=selected_run_id,
        selected_source=selected_source,
        selected_session=selected_session,
        sample_count=sample_count,
        sample_seed=sample_seed,
    )
    return render_selection_snapshot(selection, width=width)


class OperatorTriageTui:
    def __init__(
        self,
        archive_root: Path,
        *,
        baseline_policy: BaselinePolicy | None = None,
        initial_view: TuiView | str = TuiView.OVERVIEW,
        selected_run_id: str | None = None,
        selected_source: str | None = None,
        selected_session: str | None = None,
        sample_count: int = DEFAULT_SAMPLE_COUNT,
        sample_seed: str = DEFAULT_SAMPLE_SEED,
    ) -> None:
        if sample_count <= 0:
            raise TuiError("sample count must be greater than zero")

        self.archive_root = archive_root.expanduser().resolve(strict=False)
        self.baseline_policy = baseline_policy
        self.view = TuiView(initial_view)
        self.selected_run_id = selected_run_id
        self.selected_source = selected_source
        self.selected_session = selected_session
        self.sample_count = sample_count
        self.sample_seed = sample_seed
        self.notice: str | None = None
        self.bundle = load_tui_bundle(
            self.archive_root,
            baseline_policy=self.baseline_policy,
        )
        self._selection_cache: TuiSelectionSnapshot | None = None

    def render(self, *, width: int) -> str:
        selection = self._selection()
        body = render_selection_snapshot(selection, width=width)
        if self.notice is None:
            return body
        return "\n".join([f"Notice: {self.notice}", body])

    def refresh(self) -> None:
        self.bundle = load_tui_bundle(
            self.archive_root,
            baseline_policy=self.baseline_policy,
        )
        self.notice = "refreshed archive state from disk"
        self._invalidate_selection()

    def run(self) -> None:
        try:
            import curses
        except ImportError as exc:
            raise TuiError("curses is not available on this Python build") from exc

        curses.wrapper(self._curses_main)

    def _curses_main(self, stdscr) -> None:
        try:
            curses = __import__("curses")
            curses.curs_set(0)
        except Exception:
            curses = __import__("curses")
        stdscr.keypad(True)

        while True:
            height, width = stdscr.getmaxyx()
            stdscr.erase()
            for line_number, line in enumerate(
                _clip_lines(self.render(width=max(width - 1, 20)).splitlines(), height)
            ):
                attributes = curses.A_REVERSE if line.startswith("> ") else curses.A_NORMAL
                try:
                    stdscr.addnstr(line_number, 0, line, max(width - 1, 1), attributes)
                except Exception:
                    continue
            stdscr.refresh()

            key = stdscr.get_wch()
            if self._handle_key(key):
                break

    def _handle_key(self, key: object) -> bool:
        try:
            import curses
        except ImportError:
            curses = None
        previous_view = self.view

        if key in ("q", "Q"):
            return True
        if key == "1":
            self.view = TuiView.OVERVIEW
        elif key == "2":
            self.view = TuiView.RUNS
        elif key == "3":
            self.view = TuiView.SOURCES
        elif key == "4":
            self.view = TuiView.SAMPLES
        elif key in ("?", "h", "H"):
            self.view = TuiView.HELP
        elif key == "\t":
            self.view = _cycle_view(self.view)
        elif key in ("r", "R"):
            try:
                self.refresh()
            except TuiError as exc:
                self.notice = f"refresh failed: {exc}"
            return False
        elif key in ("\n", "\r"):
            self._activate_selection()
        elif key in ("j", "J") or (curses is not None and key == curses.KEY_DOWN):
            self._move_selection(1)
        elif key in ("k", "K") or (curses is not None and key == curses.KEY_UP):
            self._move_selection(-1)
        elif curses is not None and key == curses.KEY_RESIZE:
            return False
        if self.view != previous_view:
            self.notice = None
            self._invalidate_selection()
        return False

    def _move_selection(self, delta: int) -> None:
        selection = self._selection()
        if self.view in (TuiView.OVERVIEW, TuiView.SOURCES):
            source_reports = selection.bundle.source_reports
            source_name = _move_selected_name(
                source_reports,
                current_name=selection.selected_source_name,
                delta=delta,
            )
            if source_name is not None:
                self.selected_source = source_name
                self.selected_session = None
                self.notice = None
                self._invalidate_selection()
            return
        if self.view == TuiView.RUNS:
            run_id = _move_selected_name(
                selection.bundle.runs,
                current_name=None if selection.selected_run is None else selection.selected_run.run_id,
                delta=delta,
            )
            if run_id is not None:
                self.selected_run_id = run_id
                self.notice = None
                self._invalidate_selection()
            return
        if self.view != TuiView.SAMPLES or selection.sample_report is None:
            return
        session_id = _move_selected_name(
            selection.sample_report.conversations,
            current_name=(
                None
                if selection.selected_sample is None
                else selection.selected_sample.conversation.source_session_id
            ),
            delta=delta,
        )
        if session_id is not None:
            self.selected_session = session_id
            self.notice = None
            self._invalidate_selection()

    def _activate_selection(self) -> None:
        selection = self._selection()
        if self.view == TuiView.RUNS and selection.selected_run is not None:
            focus_source = _preferred_run_source(selection.selected_run)
            if focus_source is not None:
                self.selected_source = focus_source
                self.selected_session = None
                self.view = TuiView.SOURCES
                self._invalidate_selection()
            return
        if self.view in (TuiView.OVERVIEW, TuiView.SOURCES):
            self.view = TuiView.SAMPLES
            self._invalidate_selection()

    def _selection(self) -> TuiSelectionSnapshot:
        if self._selection_cache is None:
            self._selection_cache = build_selection_snapshot(
                self.bundle,
                view=self.view,
                selected_run_id=self.selected_run_id,
                selected_source=self.selected_source,
                selected_session=self.selected_session,
                sample_count=self.sample_count,
                sample_seed=self.sample_seed,
            )
        return self._selection_cache

    def _invalidate_selection(self) -> None:
        self._selection_cache = None


def run_operator_triage_tui(
    archive_root: Path,
    *,
    baseline_policy: BaselinePolicy | None = None,
    initial_view: TuiView | str = TuiView.OVERVIEW,
    selected_run_id: str | None = None,
    selected_source: str | None = None,
    selected_session: str | None = None,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    sample_seed: str = DEFAULT_SAMPLE_SEED,
) -> None:
    app = OperatorTriageTui(
        archive_root,
        baseline_policy=baseline_policy,
        initial_view=initial_view,
        selected_run_id=selected_run_id,
        selected_source=selected_source,
        selected_session=selected_session,
        sample_count=sample_count,
        sample_seed=sample_seed,
    )
    app.run()


def render_selection_snapshot(
    selection: TuiSelectionSnapshot,
    *,
    width: int,
) -> str:
    effective_width = max(width, 40)
    lines = _render_header(selection, width=effective_width)
    if selection.view == TuiView.OVERVIEW:
        lines.extend(_render_overview(selection, width=effective_width))
    elif selection.view == TuiView.RUNS:
        lines.extend(_render_runs(selection, width=effective_width))
    elif selection.view == TuiView.SOURCES:
        lines.extend(_render_sources(selection, width=effective_width))
    elif selection.view == TuiView.SAMPLES:
        lines.extend(_render_samples(selection, width=effective_width))
    else:
        lines.extend(_render_help(selection, width=effective_width))
    return "\n".join(_shorten(line, effective_width) for line in lines)


def _render_header(selection: TuiSelectionSnapshot, *, width: int) -> list[str]:
    nav = "1 overview | 2 runs | 3 sources | 4 samples | ? help | r refresh | q quit"
    lines = [
        f"llm-chat-archive operator triage [{selection.view.value}]",
        nav,
        f"Archive root: {selection.bundle.archive_root}",
    ]
    if selection.bundle.digest.baseline_path is not None:
        lines.append(
            "Baseline: "
            f"{selection.bundle.digest.baseline_path} "
            f"({selection.bundle.digest.baseline_entry_count} entries)"
        )
    lines.append("=" * min(width, 100))
    return lines


def _render_overview(selection: TuiSelectionSnapshot, *, width: int) -> list[str]:
    lines: list[str] = []
    lines.extend(_section("Latest Run", width=width))
    latest_run = selection.bundle.digest.latest_run
    if latest_run is None:
        lines.append("Latest run: unavailable")
    else:
        lines.append(
            "Latest run: "
            f"{latest_run.run_id} completed={latest_run.completed_at or '-'} "
            f"failed={latest_run.failed_source_count} "
            f"degraded={latest_run.degraded_source_count} "
            f"sources={latest_run.source_count}"
        )
        lines.append(
            "Failed sources: "
            f"{_list_or_dash(latest_run.failed_sources)}"
        )
        lines.append(
            "Degraded sources: "
            f"{_list_or_dash(latest_run.degraded_sources)}"
        )
    if selection.bundle.runs:
        lines.extend(_section("Recent Runs", width=width))
        lines.extend(
            _render_selectable_lines(
                selection.bundle.runs[:5],
                current_name=(
                    None
                    if selection.selected_run is None
                    else selection.selected_run.run_id
                ),
                width=width,
                label=_format_run_line,
            )
        )

    lines.extend(_section("Archive Health", width=width))
    overview = selection.bundle.digest.to_dict()["overview"]
    lines.append(
        "Digest: "
        f"status={selection.bundle.digest.status} "
        f"warnings={overview['warning_count']} "
        f"errors={overview['error_count']} "
        f"suspicious_sources={overview['suspicious_source_count']} "
        f"orphans={overview['orphan_file_count']}"
    )
    lines.append(
        "Archive totals: "
        f"conversations={overview['conversation_count']} "
        f"messages={overview['message_count']} "
        f"with_limitations={overview['conversation_with_limitations_count']}"
    )
    lines.append(
        "Completeness: "
        f"{_format_ratio_map(overview['transcript_completeness'])}"
    )

    lines.extend(_section("Sources", width=width))
    lines.extend(
        _render_selectable_lines(
            selection.bundle.source_reports,
            current_name=selection.selected_source_name,
            width=width,
            label=_format_digest_source_line,
        )
    )
    lines.extend(_render_selected_source_summary(selection, width=width))
    return lines


def _render_runs(selection: TuiSelectionSnapshot, *, width: int) -> list[str]:
    lines: list[str] = []
    lines.extend(_section("Latest Runs", width=width))
    if not selection.bundle.runs:
        lines.append(
            "No recorded runs are available yet."
            if selection.bundle.runs_error is None
            else f"Run history error: {selection.bundle.runs_error}"
        )
        return lines

    lines.extend(
        _render_selectable_lines(
            selection.bundle.runs,
            current_name=(
                None if selection.selected_run is None else selection.selected_run.run_id
            ),
            width=width,
            label=_format_run_line,
            limit=10,
        )
    )
    lines.append("Enter focuses the selected run's first degraded source.")
    lines.extend(_section("Selected Run Sources", width=width))
    if selection.selected_run is None:
        lines.append("No run selected.")
        return lines

    for source in _sorted_run_sources(selection.selected_run):
        lines.append(_format_run_source_line(source))
    return lines


def _render_sources(selection: TuiSelectionSnapshot, *, width: int) -> list[str]:
    lines: list[str] = []
    lines.extend(_section("Source Health", width=width))
    lines.extend(
        _render_selectable_lines(
            selection.bundle.source_reports,
            current_name=selection.selected_source_name,
            width=width,
            label=_format_digest_source_line,
            limit=12,
        )
    )
    lines.extend(_render_selected_source_summary(selection, width=width))
    lines.append("Enter opens the sample drill-down for the selected source.")
    return lines


def _render_samples(selection: TuiSelectionSnapshot, *, width: int) -> list[str]:
    lines: list[str] = []
    lines.extend(_section("Sample Drill-Down", width=width))
    if selection.selected_source_name is None:
        lines.append("No source is available to sample.")
        return lines

    lines.append(
        "Selection: "
        f"source={selection.selected_source_name} "
        f"seed={selection.sample_seed} "
        f"count={selection.sample_count}"
    )
    if selection.sample_error is not None:
        lines.append(f"Sample error: {selection.sample_error}")
        return lines
    if selection.sample_report is None:
        lines.append("Sample report is unavailable.")
        return lines

    lines.append(
        "Candidates: "
        f"{selection.sample_report.candidate_count} "
        f"returned={selection.sample_report.conversation_count} "
        f"messages={selection.sample_report.message_count}"
    )
    lines.extend(
        _render_selectable_lines(
            selection.sample_report.conversations,
            current_name=(
                None
                if selection.selected_sample is None
                else selection.selected_sample.conversation.source_session_id
            ),
            width=width,
            label=_format_sample_line,
            limit=8,
        )
    )

    lines.extend(_section("Conversation", width=width))
    if selection.selected_sample is None:
        lines.append("No sampled conversation is available for this source.")
        return lines

    summary = selection.selected_sample.conversation
    lines.append(
        "Summary: "
        f"session={summary.source_session_id or '-'} "
        f"collected_at={summary.collected_at} "
        f"completeness={summary.transcript_completeness} "
        f"messages={summary.message_count}"
    )
    if summary.limitations:
        lines.append("Limitations: " + ", ".join(summary.limitations))
    if selection.conversation_error is not None:
        lines.append(f"Conversation error: {selection.conversation_error}")
        return lines
    if selection.selected_conversation is None:
        lines.append("Conversation detail is unavailable.")
        return lines

    lines.extend(
        _render_conversation_messages(
            selection.selected_conversation,
            width=width,
            max_messages=6,
        )
    )
    return lines


def _render_help(selection: TuiSelectionSnapshot, *, width: int) -> list[str]:
    _ = selection
    return [
        "== Keybindings ==",
        "1: overview screen with latest run and archive health summary",
        "2: run history screen with per-run source status list",
        "3: source screen with digest, stats, and profile details",
        "4: sampled conversation drill-down for the selected source",
        "Up/Down or j/k: move within the active list",
        "Enter: from runs jump to the selected run's first degraded source",
        "Enter: from overview or sources open the sample drill-down",
        "r: reload manifests, digest, stats, profile, and sample state from disk",
        "q: quit",
        "",
        "The TUI is read-only. It reuses recorded run manifests and normalized archive",
        "JSONL output instead of creating a second source of truth.",
    ]


def _render_selected_source_summary(
    selection: TuiSelectionSnapshot,
    *,
    width: int,
) -> list[str]:
    lines = _section("Selected Source", width=width)
    source_report = selection.selected_source_report
    if source_report is None:
        lines.append("No source is selected.")
        return lines

    lines.append(
        "Status: "
        f"latest_run={source_report.latest_run_status or 'archive-only'} "
        f"support={source_report.support_level or '-'} "
        f"verify={source_report.verify_status or '-'} "
        f"attention={_yes_no(source_report.attention_required)}"
    )
    lines.append(
        "Counts: "
        f"files={source_report.file_count} "
        f"conversations={source_report.conversation_count} "
        f"messages={source_report.message_count} "
        f"warnings={source_report.warning_count} "
        f"errors={source_report.error_count}"
    )
    lines.append(
        "Completeness: "
        f"{_format_ratio_map(source_report.transcript_completeness)}"
    )

    if selection.selected_source_profile is not None:
        lines.append(
            "Roles: "
            f"{_format_ratio_map(selection.selected_source_profile.to_dict()['message_roles'])}"
        )

    if source_report.top_limitations:
        lines.append(
            "Top limitations: "
            + ", ".join(
                f"{item.limitation} ({item.count})" for item in source_report.top_limitations
            )
        )
    if source_report.source_reasons:
        for reason in source_report.source_reasons[:3]:
            lines.extend(
                _wrap_text(
                    f"Reason [{reason.code}]: {reason.message}",
                    width=width,
                    initial="",
                    subsequent="  ",
                )
            )
    return lines


def _render_selectable_lines(
    items,
    *,
    current_name: str | None,
    width: int,
    label,
    limit: int | None = None,
) -> list[str]:
    normalized_items = tuple(items)
    if not normalized_items:
        return ["  <none>"]

    lines: list[str] = []
    visible_items = normalized_items if limit is None else normalized_items[:limit]
    for item in visible_items:
        name = _name_for_item(item)
        prefix = "> " if name == current_name else "  "
        lines.append(prefix + label(item))
    hidden_count = len(normalized_items) - len(visible_items)
    if hidden_count > 0:
        lines.append(f"  ... {hidden_count} more")
    return [_shorten(line, width) for line in lines]


def _render_conversation_messages(
    conversation: ArchiveConversationRecord,
    *,
    width: int,
    max_messages: int,
) -> list[str]:
    lines: list[str] = []
    messages = conversation.messages[:max_messages]
    for index, message in enumerate(messages, start=1):
        role = message.get("role", "unknown")
        timestamp = message.get("timestamp")
        header = f"{index}. {role}"
        if isinstance(timestamp, str) and timestamp:
            header += f" @ {timestamp}"
        lines.append(header)
        text = message.get("text")
        if isinstance(text, str) and text.strip():
            lines.extend(_wrap_text(text, width=width, initial="   ", subsequent="   "))
        else:
            lines.append("   <non-text message>")
    if len(conversation.messages) > len(messages):
        lines.append(
            f"... {len(conversation.messages) - len(messages)} more messages omitted"
        )
    return lines


def _section(title: str, *, width: int) -> list[str]:
    line = f"== {title} =="
    return ["", _shorten(line, width)]


def _format_run_line(run: RunSummary) -> str:
    return (
        f"{run.run_id} completed={run.completed_at or '-'} "
        f"failed={run.failed_source_count} partial={run.partial_source_count} "
        f"unsupported={run.unsupported_source_count} "
        f"conv={run.conversation_count} msg={run.message_count}"
    )


def _format_run_source_line(source) -> str:
    detail = source.failure_reason or source.support_limitation_summary or "-"
    return (
        f"{source.source} status={source.status} support={source.support_level} "
        f"conv={source.conversation_count} msg={source.message_count} detail={detail}"
    )


def _format_digest_source_line(source: ArchiveDigestSourceReport) -> str:
    flags: list[str] = []
    if source.failed:
        flags.append("failed")
    elif source.run_degraded:
        flags.append("degraded")
    else:
        flags.append(source.latest_run_status or "archive-only")
    if source.verify_status not in (None, "success"):
        flags.append(f"verify={source.verify_status}")
    if source.suspicious:
        flags.append("suspicious")
    return (
        f"{source.source} [{' '.join(flags)}] "
        f"conv={source.conversation_count} msg={source.message_count}"
    )


def _format_sample_line(sample: ArchiveConversationSample) -> str:
    summary = sample.conversation
    preview = sample.preview or "-"
    return (
        f"{summary.source_session_id or '-'} "
        f"{summary.transcript_completeness} "
        f"{summary.collected_at} "
        f"messages={summary.message_count} "
        f"preview={preview}"
    )


def _format_ratio_map(payload: dict[str, dict[str, float | int]]) -> str:
    parts: list[str] = []
    for key, item in payload.items():
        count = int(item["count"])
        ratio = float(item["ratio"])
        parts.append(f"{key} {count} ({ratio:.0%})")
    return ", ".join(parts) if parts else "-"


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _list_or_dash(values: tuple[str, ...]) -> str:
    return ", ".join(values) if values else "-"


def _select_run(
    runs: tuple[RunSummary, ...],
    requested_run_id: str | None,
) -> RunSummary | None:
    if not runs:
        return None
    if requested_run_id is not None:
        for run in runs:
            if run.run_id == requested_run_id:
                return run
    return runs[0]


def _select_source_name(
    source_reports: tuple[ArchiveDigestSourceReport, ...],
    *,
    requested_source: str | None,
) -> str | None:
    if not source_reports:
        return None
    if requested_source is not None:
        for source_report in source_reports:
            if source_report.source == requested_source:
                return requested_source
    for source_report in source_reports:
        if source_report.attention_required:
            return source_report.source
    return source_reports[0].source


def _select_sample(
    samples: tuple[ArchiveConversationSample, ...],
    *,
    requested_session: str | None,
) -> ArchiveConversationSample | None:
    if not samples:
        return None
    if requested_session is not None:
        for sample in samples:
            if sample.conversation.source_session_id == requested_session:
                return sample
    return samples[0]


def _sorted_source_reports(
    source_reports: tuple[ArchiveDigestSourceReport, ...],
) -> tuple[ArchiveDigestSourceReport, ...]:
    return tuple(
        sorted(
            source_reports,
            key=lambda source: (
                not source.attention_required,
                not source.failed,
                not source.run_degraded,
                source.source,
            ),
        )
    )


def _sorted_run_sources(run: RunSummary):
    return tuple(
        sorted(
            run.sources,
            key=lambda source: (
                not source.failed,
                not (source.partial or source.unsupported),
                source.source,
            ),
        )
    )


def _preferred_run_source(run: RunSummary) -> str | None:
    if not run.sources:
        return None
    for source in _sorted_run_sources(run):
        if source.failed or source.partial or source.unsupported:
            return source.source
    return run.sources[0].source


def _find_by_source(items, source_name: str | None):
    if source_name is None:
        return None
    for item in items:
        if item.source == source_name:
            return item
    return None


def _name_for_item(item) -> str | None:
    if hasattr(item, "run_id"):
        return item.run_id
    if hasattr(item, "source"):
        return item.source
    if hasattr(item, "conversation"):
        return item.conversation.source_session_id
    return None


def _move_selected_name(
    items,
    *,
    current_name: str | None,
    delta: int,
) -> str | None:
    normalized_items = tuple(items)
    if not normalized_items:
        return None
    names = [_name_for_item(item) for item in normalized_items]
    if current_name not in names:
        target_index = 0 if delta >= 0 else len(names) - 1
        return names[target_index]
    current_index = names.index(current_name)
    next_index = max(0, min(len(names) - 1, current_index + delta))
    return names[next_index]


def _cycle_view(view: TuiView) -> TuiView:
    order = tuple(TuiView)
    index = order.index(view)
    return order[(index + 1) % len(order)]


def _wrap_text(
    text: str,
    *,
    width: int,
    initial: str,
    subsequent: str,
) -> list[str]:
    wrap_width = max(width, len(initial) + 10)
    wrapped = textwrap.wrap(
        text,
        width=wrap_width,
        initial_indent=initial,
        subsequent_indent=subsequent,
        replace_whitespace=False,
        drop_whitespace=True,
    )
    return wrapped or [initial.rstrip()]


def _shorten(text: str, width: int) -> str:
    if len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    return text[: width - 3] + "..."


def _clip_lines(lines: list[str], height: int) -> list[str]:
    if height <= 0:
        return []
    if len(lines) <= height:
        return lines
    visible = lines[: max(height - 1, 0)]
    remaining = len(lines) - len(visible)
    visible.append(f"... {remaining} more lines omitted")
    return visible


__all__ = [
    "DEFAULT_SAMPLE_COUNT",
    "DEFAULT_SAMPLE_SEED",
    "DEFAULT_SNAPSHOT_WIDTH",
    "OperatorTriageTui",
    "TuiBundle",
    "TuiError",
    "TuiSelectionSnapshot",
    "TuiView",
    "build_selection_snapshot",
    "load_tui_bundle",
    "render_selection_snapshot",
    "render_tui_snapshot",
    "run_operator_triage_tui",
    "tui_view_choices",
]
