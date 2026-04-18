from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Iterable

from .models import (
    CollectionExecutionPolicy,
    DEFAULT_ARCHIVE_ROOT,
    EffectiveCollectConfig,
    EffectiveRerunConfig,
    EffectiveScheduledConfig,
    RedactionMode,
    RerunSelectionPreset,
    ScheduledRunMode,
    SourceSelectionProfile,
    ValidationMode,
)
from .source_selection import build_source_selection_policy

DEFAULT_COLLECT_CONFIG_PATH = Path("~/.config/llm-chat-archive/collector.toml")
DEFAULT_SCHEDULED_STALE_AFTER_SECONDS = 21600


class CollectorConfigError(ValueError):
    """Raised when a collector config file cannot be loaded or validated."""


def default_collect_config_path() -> Path:
    return DEFAULT_COLLECT_CONFIG_PATH.expanduser().resolve(strict=False)


def render_collect_config_template(
    archive_root: Path = DEFAULT_ARCHIVE_ROOT,
) -> str:
    resolved_archive_root = archive_root.expanduser().resolve(strict=False)
    return "\n".join(
        (
            "# This file matches `uv run llm-chat-archive config init`.",
            "# Set collect.archive_root to an absolute path outside this repository.",
            "",
            "[collect]",
            f'archive_root = "{resolved_archive_root}"',
            "incremental = true",
            'redaction = "on"',
            'validation = "report"',
            "",
            "[collect.selection]",
            'profile = "default"',
            "",
            "[rerun]",
            'selection_preset = "failed_and_degraded"',
            "",
            "[scheduled]",
            'mode = "collect"',
            "incremental = true",
            'redaction = "on"',
            'validation = "report"',
            f"stale_after_seconds = {DEFAULT_SCHEDULED_STALE_AFTER_SECONDS}",
            "",
            "[scheduled.selection]",
            'profile = "default"',
            "",
            "[scheduled.rerun]",
            'selection_preset = "failed_and_degraded"',
            "",
        )
    )


def resolve_collect_config_output_path(output_path: Path | None) -> Path:
    if output_path is None:
        return default_collect_config_path()
    return output_path.expanduser().resolve(strict=False)


def scaffold_collect_config(
    *,
    output_path: Path,
    archive_root: Path = DEFAULT_ARCHIVE_ROOT,
    force: bool = False,
) -> Path:
    if output_path.exists():
        if not output_path.is_file():
            raise CollectorConfigError(f"collector config path is not a file: {output_path}")
        if not force:
            raise CollectorConfigError(
                f"collector config already exists: {output_path} (pass --force to overwrite)"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_collect_config_template(archive_root=archive_root),
        encoding="utf-8",
    )
    return output_path


def resolve_collect_config(
    *,
    config_path: Path | None,
    cli_archive_root: Path | None,
    cli_profile: str | None,
    cli_include_sources: Iterable[str] | None,
    cli_exclude_sources: Iterable[str] | None,
    cli_incremental: bool | None,
    cli_redaction: str | None,
    cli_validation: str | None,
    default_path: Path | None = None,
) -> EffectiveCollectConfig:
    loaded_config = _load_collect_config(
        config_path=config_path,
        default_path=default_path or default_collect_config_path(),
    )

    selection_policy = build_source_selection_policy(
        profile=cli_profile or loaded_config.profile,
        include_sources=(
            tuple(cli_include_sources)
            if cli_include_sources is not None
            else loaded_config.include_sources
        ),
        exclude_sources=(
            tuple(cli_exclude_sources)
            if cli_exclude_sources is not None
            else loaded_config.exclude_sources
        ),
    )
    execution_policy = CollectionExecutionPolicy(
        incremental=(
            cli_incremental
            if cli_incremental is not None
            else loaded_config.incremental
            if loaded_config.incremental is not None
            else True
        ),
        redaction=(
            RedactionMode(cli_redaction)
            if cli_redaction is not None
            else loaded_config.redaction or RedactionMode.ON
        ),
        validation=(
            ValidationMode(cli_validation)
            if cli_validation is not None
            else loaded_config.validation or ValidationMode.OFF
        ),
    )
    archive_root = cli_archive_root or loaded_config.archive_root or DEFAULT_ARCHIVE_ROOT

    return EffectiveCollectConfig(
        archive_root=archive_root,
        selection_policy=selection_policy,
        execution_policy=execution_policy,
        rerun=loaded_config.rerun,
        config_source=loaded_config.config_source,
        config_path=loaded_config.config_path,
    )


def resolve_scheduled_config(
    *,
    config_path: Path | None,
    cli_archive_root: Path | None,
    cli_mode: str | None,
    cli_stale_after_seconds: int | None,
    default_path: Path | None = None,
) -> EffectiveScheduledConfig:
    if cli_stale_after_seconds is not None and cli_stale_after_seconds <= 0:
        raise CollectorConfigError("scheduled stale-after-seconds must be a positive integer")
    loaded_config = _load_collect_config(
        config_path=config_path,
        default_path=default_path or default_collect_config_path(),
    )
    resolved_scheduled = loaded_config.scheduled or _LoadedScheduledConfig(
        profile=loaded_config.profile,
        include_sources=loaded_config.include_sources,
        exclude_sources=loaded_config.exclude_sources,
        incremental=loaded_config.incremental,
        redaction=loaded_config.redaction,
        validation=loaded_config.validation,
        rerun=loaded_config.rerun,
        source=(
            "collect_defaults"
            if loaded_config.config_source != "defaults"
            else "defaults"
        ),
    )
    selection_policy = build_source_selection_policy(
        profile=resolved_scheduled.profile,
        include_sources=resolved_scheduled.include_sources,
        exclude_sources=resolved_scheduled.exclude_sources,
    )
    execution_policy = CollectionExecutionPolicy(
        incremental=(
            resolved_scheduled.incremental
            if resolved_scheduled.incremental is not None
            else True
        ),
        redaction=resolved_scheduled.redaction or RedactionMode.ON,
        validation=resolved_scheduled.validation or ValidationMode.OFF,
    )
    archive_root = cli_archive_root or loaded_config.archive_root or DEFAULT_ARCHIVE_ROOT
    mode = (
        ScheduledRunMode(cli_mode)
        if cli_mode is not None
        else resolved_scheduled.mode or ScheduledRunMode.COLLECT
    )
    stale_after_seconds = (
        cli_stale_after_seconds
        if cli_stale_after_seconds is not None
        else resolved_scheduled.stale_after_seconds
        if resolved_scheduled.stale_after_seconds is not None
        else DEFAULT_SCHEDULED_STALE_AFTER_SECONDS
    )
    return EffectiveScheduledConfig(
        archive_root=archive_root,
        mode=mode,
        selection_policy=selection_policy,
        execution_policy=execution_policy,
        rerun=resolved_scheduled.rerun,
        stale_after_seconds=stale_after_seconds,
        source=resolved_scheduled.source,
        config_source=loaded_config.config_source,
        config_path=loaded_config.config_path,
    )


class _LoadedScheduledConfig:
    def __init__(
        self,
        *,
        mode: ScheduledRunMode | None = None,
        profile: str | SourceSelectionProfile | None = SourceSelectionProfile.DEFAULT,
        include_sources: tuple[str, ...] = (),
        exclude_sources: tuple[str, ...] = (),
        incremental: bool | None = None,
        redaction: RedactionMode | None = None,
        validation: ValidationMode | None = None,
        rerun: EffectiveRerunConfig | None = None,
        stale_after_seconds: int | None = None,
        source: str = "defaults",
    ) -> None:
        self.mode = mode
        self.profile = profile
        self.include_sources = include_sources
        self.exclude_sources = exclude_sources
        self.incremental = incremental
        self.redaction = redaction
        self.validation = validation
        self.rerun = rerun
        self.stale_after_seconds = stale_after_seconds
        self.source = source


class _LoadedCollectConfig:
    def __init__(
        self,
        *,
        archive_root: Path | None = None,
        profile: str | SourceSelectionProfile | None = None,
        include_sources: tuple[str, ...] = (),
        exclude_sources: tuple[str, ...] = (),
        incremental: bool | None = None,
        redaction: RedactionMode | None = None,
        validation: ValidationMode | None = None,
        rerun: EffectiveRerunConfig | None = None,
        scheduled: _LoadedScheduledConfig | None = None,
        config_source: str = "defaults",
        config_path: Path | None = None,
    ) -> None:
        self.archive_root = archive_root
        self.profile = profile
        self.include_sources = include_sources
        self.exclude_sources = exclude_sources
        self.incremental = incremental
        self.redaction = redaction
        self.validation = validation
        self.rerun = rerun
        self.scheduled = scheduled
        self.config_source = config_source
        self.config_path = config_path


def _load_collect_config(
    *,
    config_path: Path | None,
    default_path: Path,
) -> _LoadedCollectConfig:
    if config_path is not None:
        resolved_path = _resolve_explicit_config_path(config_path)
        return _parse_collect_config(
            resolved_path,
            config_source="explicit",
        )

    if default_path.exists():
        return _parse_collect_config(
            default_path,
            config_source="default_path",
        )

    return _LoadedCollectConfig()


def _resolve_explicit_config_path(config_path: Path) -> Path:
    if not config_path.is_absolute():
        raise CollectorConfigError("collector config path must be absolute")
    return config_path.expanduser().resolve(strict=False)


def _parse_collect_config(path: Path, *, config_source: str) -> _LoadedCollectConfig:
    if not path.exists():
        raise CollectorConfigError(f"collector config does not exist: {path}")
    if not path.is_file():
        raise CollectorConfigError(f"collector config is not a file: {path}")

    try:
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise CollectorConfigError(
            f"collector config is not valid TOML: {path}: {exc}"
        ) from exc
    except OSError as exc:
        raise CollectorConfigError(f"failed to read collector config {path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise CollectorConfigError(f"collector config root must be a table: {path}")

    _raise_for_unexpected_keys(
        payload,
        allowed={"collect", "rerun", "scheduled"},
        context=str(path),
    )
    collect = payload.get("collect", {})
    if not isinstance(collect, dict):
        raise CollectorConfigError(f"collector config [collect] must be a table: {path}")
    rerun = payload.get("rerun", {})
    if not isinstance(rerun, dict):
        raise CollectorConfigError(f"collector config [rerun] must be a table: {path}")
    scheduled = payload.get("scheduled", {})
    if not isinstance(scheduled, dict):
        raise CollectorConfigError(f"collector config [scheduled] must be a table: {path}")

    _raise_for_unexpected_keys(
        collect,
        allowed={
            "archive_root",
            "incremental",
            "redaction",
            "validation",
            "selection",
        },
        context=f"{path} [collect]",
    )

    selection = collect.get("selection", {})
    if not isinstance(selection, dict):
        raise CollectorConfigError(
            f"collector config [collect.selection] must be a table: {path}"
        )

    _raise_for_unexpected_keys(
        selection,
        allowed={"profile", "sources", "exclude_sources"},
        context=f"{path} [collect.selection]",
    )
    _raise_for_unexpected_keys(
        rerun,
        allowed={"selection_preset"},
        context=f"{path} [rerun]",
    )
    _raise_for_unexpected_keys(
        scheduled,
        allowed={
            "mode",
            "incremental",
            "redaction",
            "validation",
            "stale_after_seconds",
            "selection",
            "rerun",
        },
        context=f"{path} [scheduled]",
    )

    archive_root = _optional_absolute_path(
        collect,
        "archive_root",
        context=f"{path} [collect]",
    )
    scheduled_selection = scheduled.get("selection", {})
    if not isinstance(scheduled_selection, dict):
        raise CollectorConfigError(
            f"collector config [scheduled.selection] must be a table: {path}"
        )
    _raise_for_unexpected_keys(
        scheduled_selection,
        allowed={"profile", "sources", "exclude_sources"},
        context=f"{path} [scheduled.selection]",
    )
    scheduled_rerun = scheduled.get("rerun", {})
    if not isinstance(scheduled_rerun, dict):
        raise CollectorConfigError(
            f"collector config [scheduled.rerun] must be a table: {path}"
        )
    _raise_for_unexpected_keys(
        scheduled_rerun,
        allowed={"selection_preset"},
        context=f"{path} [scheduled.rerun]",
    )
    profile = _optional_enum(
        selection,
        "profile",
        enum_type=SourceSelectionProfile,
        context=f"{path} [collect.selection]",
    )
    include_sources = _optional_string_list(
        selection,
        "sources",
        context=f"{path} [collect.selection]",
    )
    exclude_sources = _optional_string_list(
        selection,
        "exclude_sources",
        context=f"{path} [collect.selection]",
    )
    incremental = _optional_bool(
        collect,
        "incremental",
        context=f"{path} [collect]",
    )
    redaction = _optional_enum(
        collect,
        "redaction",
        enum_type=RedactionMode,
        context=f"{path} [collect]",
    )
    validation = _optional_enum(
        collect,
        "validation",
        enum_type=ValidationMode,
        context=f"{path} [collect]",
    )
    rerun_selection_preset = _optional_enum(
        rerun,
        "selection_preset",
        enum_type=RerunSelectionPreset,
        context=f"{path} [rerun]",
    )
    scheduled_mode = _optional_enum(
        scheduled,
        "mode",
        enum_type=ScheduledRunMode,
        context=f"{path} [scheduled]",
    )
    scheduled_profile = _optional_enum(
        scheduled_selection,
        "profile",
        enum_type=SourceSelectionProfile,
        context=f"{path} [scheduled.selection]",
    )
    scheduled_include_sources = _optional_string_list(
        scheduled_selection,
        "sources",
        context=f"{path} [scheduled.selection]",
    )
    scheduled_exclude_sources = _optional_string_list(
        scheduled_selection,
        "exclude_sources",
        context=f"{path} [scheduled.selection]",
    )
    scheduled_incremental = _optional_bool(
        scheduled,
        "incremental",
        context=f"{path} [scheduled]",
    )
    scheduled_redaction = _optional_enum(
        scheduled,
        "redaction",
        enum_type=RedactionMode,
        context=f"{path} [scheduled]",
    )
    scheduled_validation = _optional_enum(
        scheduled,
        "validation",
        enum_type=ValidationMode,
        context=f"{path} [scheduled]",
    )
    scheduled_stale_after_seconds = _optional_positive_int(
        scheduled,
        "stale_after_seconds",
        context=f"{path} [scheduled]",
    )
    scheduled_rerun_selection_preset = _optional_enum(
        scheduled_rerun,
        "selection_preset",
        enum_type=RerunSelectionPreset,
        context=f"{path} [scheduled.rerun]",
    )

    return _LoadedCollectConfig(
        archive_root=archive_root,
        profile=profile,
        include_sources=include_sources,
        exclude_sources=exclude_sources,
        incremental=incremental,
        redaction=redaction,
        validation=validation,
        rerun=(
            EffectiveRerunConfig(
                selection_preset=rerun_selection_preset,
                selection_reason=rerun_selection_preset.selection_reason,
                source="config",
            )
            if rerun_selection_preset is not None
            else None
        ),
        scheduled=(
            _LoadedScheduledConfig(
                mode=scheduled_mode,
                profile=scheduled_profile or profile or SourceSelectionProfile.DEFAULT,
                include_sources=scheduled_include_sources or include_sources,
                exclude_sources=scheduled_exclude_sources or exclude_sources,
                incremental=(
                    scheduled_incremental
                    if scheduled_incremental is not None
                    else incremental
                ),
                redaction=scheduled_redaction or redaction,
                validation=scheduled_validation or validation,
                rerun=(
                    EffectiveRerunConfig(
                        selection_preset=scheduled_rerun_selection_preset,
                        selection_reason=scheduled_rerun_selection_preset.selection_reason,
                        source="config",
                    )
                    if scheduled_rerun_selection_preset is not None
                    else (
                        EffectiveRerunConfig(
                            selection_preset=rerun_selection_preset,
                            selection_reason=rerun_selection_preset.selection_reason,
                            source="config",
                        )
                        if rerun_selection_preset is not None
                        else None
                    )
                ),
                stale_after_seconds=scheduled_stale_after_seconds,
                source="config",
            )
            if "scheduled" in payload
            else None
        ),
        config_source=config_source,
        config_path=path,
    )


def _raise_for_unexpected_keys(
    payload: dict[str, object],
    *,
    allowed: set[str],
    context: str,
) -> None:
    unexpected = sorted(set(payload) - allowed)
    if unexpected:
        raise CollectorConfigError(
            f"unexpected keys in {context}: {', '.join(unexpected)}"
        )


def _optional_absolute_path(
    payload: dict[str, object],
    key: str,
    *,
    context: str,
) -> Path | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise CollectorConfigError(f"{context} field '{key}' must be a string")
    candidate = Path(value)
    if not candidate.is_absolute():
        raise CollectorConfigError(f"{context} field '{key}' must be an absolute path")
    return candidate.expanduser().resolve(strict=False)


def _optional_string_list(
    payload: dict[str, object],
    key: str,
    *,
    context: str,
) -> tuple[str, ...]:
    value = payload.get(key)
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise CollectorConfigError(f"{context} field '{key}' must be an array of strings")
    return tuple(value)


def _optional_bool(
    payload: dict[str, object],
    key: str,
    *,
    context: str,
) -> bool | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, bool):
        raise CollectorConfigError(f"{context} field '{key}' must be a boolean")
    return value


def _optional_positive_int(
    payload: dict[str, object],
    key: str,
    *,
    context: str,
) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    raise CollectorConfigError(f"{context} field '{key}' must be a positive integer")


def _optional_enum(
    payload: dict[str, object],
    key: str,
    *,
    enum_type: type[
        SourceSelectionProfile
        | RedactionMode
        | ValidationMode
        | RerunSelectionPreset
        | ScheduledRunMode
    ],
    context: str,
):
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise CollectorConfigError(f"{context} field '{key}' must be a string")
    try:
        return enum_type(value)
    except ValueError as exc:
        allowed_values = ", ".join(member.value for member in enum_type)
        raise CollectorConfigError(
            f"{context} field '{key}' must be one of: {allowed_values}"
        ) from exc


__all__ = [
    "CollectorConfigError",
    "DEFAULT_COLLECT_CONFIG_PATH",
    "DEFAULT_SCHEDULED_STALE_AFTER_SECONDS",
    "default_collect_config_path",
    "render_collect_config_template",
    "resolve_collect_config",
    "resolve_collect_config_output_path",
    "resolve_scheduled_config",
    "scaffold_collect_config",
]
