from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from .models import SourceDescriptor, SourceRootCandidate, SourceRootPlatform

_VARIABLE_PATTERN = re.compile(r"\$(\w+)|\$\{(\w+)\}")
_ALL_ROOT_PLATFORMS = tuple(platform for platform in SourceRootPlatform)


class _MissingEnvironmentVariable(ValueError):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.name = name


@dataclass(frozen=True, slots=True)
class ResolvedSourceRoot:
    declared_path: str
    path: str | None
    resolution_source: str
    env_vars: tuple[str, ...] = ()
    miss_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "declared_path": self.declared_path,
            "resolution_source": self.resolution_source,
            "path": self.path,
        }
        if self.env_vars:
            payload["env_vars"] = list(self.env_vars)
        if self.miss_reason is not None:
            payload["miss_reason"] = self.miss_reason
        return payload


@dataclass(frozen=True, slots=True)
class SourceRootResolution:
    platform: SourceRootPlatform
    resolution_source: str
    roots: tuple[ResolvedSourceRoot, ...]
    miss_reasons: tuple[str, ...] = ()

    @property
    def resolved_root_strings(self) -> tuple[str, ...]:
        return tuple(root.path for root in self.roots if root.path is not None)

    @property
    def resolved_paths(self) -> tuple[Path, ...]:
        return tuple(
            Path(root.path).expanduser().resolve(strict=False)
            for root in self.roots
            if root.path is not None
        )

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "platform": self.platform.value,
            "resolution_source": self.resolution_source,
            "resolved_roots": list(self.resolved_root_strings),
            "roots": [root.to_dict() for root in self.roots],
        }
        if self.miss_reasons:
            payload["miss_reasons"] = list(self.miss_reasons)
        return payload


def all_platform_root(path: str) -> SourceRootCandidate:
    return SourceRootCandidate(path=path, platforms=_ALL_ROOT_PLATFORMS)


def darwin_root(path: str) -> SourceRootCandidate:
    return SourceRootCandidate(path=path, platforms=(SourceRootPlatform.DARWIN,))


def linux_root(path: str) -> SourceRootCandidate:
    return SourceRootCandidate(path=path, platforms=(SourceRootPlatform.LINUX,))


def windows_root(path: str) -> SourceRootCandidate:
    return SourceRootCandidate(path=path, platforms=(SourceRootPlatform.WINDOWS,))


def resolve_explicit_input_roots(input_roots: Iterable[Path]) -> tuple[Path, ...]:
    return tuple(root.expanduser().resolve(strict=False) for root in input_roots)


def resolve_source_roots(
    descriptor: SourceDescriptor,
    *,
    input_roots: Iterable[Path] | None = None,
    platform: str | SourceRootPlatform | None = None,
    env: Mapping[str, str] | None = None,
) -> SourceRootResolution:
    resolved_platform = normalize_source_root_platform(platform)
    if input_roots is not None:
        resolved_paths = resolve_explicit_input_roots(input_roots)
        return SourceRootResolution(
            platform=resolved_platform,
            resolution_source="cli_input_root",
            roots=tuple(
                ResolvedSourceRoot(
                    declared_path=str(path),
                    path=str(path),
                    resolution_source="cli_input_root",
                )
                for path in resolved_paths
            ),
        )

    candidates = _applicable_root_candidates(descriptor, resolved_platform)
    if not candidates:
        return SourceRootResolution(
            platform=resolved_platform,
            resolution_source="descriptor",
            roots=(),
            miss_reasons=(
                f"no root candidates are declared for platform {resolved_platform.value}",
            ),
        )

    roots: list[ResolvedSourceRoot] = []
    miss_reasons: list[str] = []
    for candidate in candidates:
        path, env_vars, miss_reason = _expand_candidate_path(
            candidate.path,
            platform=resolved_platform,
            env=env,
        )
        roots.append(
            ResolvedSourceRoot(
                declared_path=candidate.path,
                path=path,
                resolution_source="descriptor",
                env_vars=env_vars,
                miss_reason=miss_reason,
            )
        )
        if miss_reason is not None and miss_reason not in miss_reasons:
            miss_reasons.append(miss_reason)

    return SourceRootResolution(
        platform=resolved_platform,
        resolution_source="descriptor",
        roots=tuple(roots),
        miss_reasons=tuple(miss_reasons),
    )


def default_descriptor_input_roots(
    descriptor: SourceDescriptor,
    *,
    platform: str | SourceRootPlatform | None = None,
    env: Mapping[str, str] | None = None,
) -> tuple[Path, ...]:
    return resolve_source_roots(
        descriptor,
        platform=platform,
        env=env,
    ).resolved_paths


def normalize_source_root_platform(
    platform: str | SourceRootPlatform | None = None,
) -> SourceRootPlatform:
    if isinstance(platform, SourceRootPlatform):
        return platform

    raw_value = (sys.platform if platform is None else platform).lower()
    if raw_value.startswith("darwin"):
        return SourceRootPlatform.DARWIN
    if raw_value.startswith("linux"):
        return SourceRootPlatform.LINUX
    if raw_value.startswith(("win", "windows")):
        return SourceRootPlatform.WINDOWS
    raise ValueError(f"unsupported source root platform: {raw_value}")


def _applicable_root_candidates(
    descriptor: SourceDescriptor,
    platform: SourceRootPlatform,
) -> tuple[SourceRootCandidate, ...]:
    if descriptor.artifact_root_candidates:
        return tuple(
            candidate
            for candidate in descriptor.artifact_root_candidates
            if candidate.applies_to(platform)
        )
    return tuple(
        SourceRootCandidate(path=root, platforms=(platform,))
        for root in descriptor.default_input_roots
    )


def _expand_candidate_path(
    declared_path: str,
    *,
    platform: SourceRootPlatform,
    env: Mapping[str, str] | None,
) -> tuple[str | None, tuple[str, ...], str | None]:
    environment = dict(env or {})
    referenced_env_vars: list[str] = []

    try:
        expanded_path = _expand_home_prefix(
            declared_path,
            platform=platform,
            env=environment,
        )

        def replace_variable(match: re.Match[str]) -> str:
            name = match.group(1) or match.group(2)
            if name is None:
                return ""
            if name not in referenced_env_vars:
                referenced_env_vars.append(name)
            value = _resolve_environment_variable(
                name,
                platform=platform,
                env=environment,
            )
            if value is None:
                raise _MissingEnvironmentVariable(name)
            return value

        expanded_path = _VARIABLE_PATTERN.sub(replace_variable, expanded_path)
    except _MissingEnvironmentVariable as exc:
        env_vars = tuple(sorted(set(referenced_env_vars + [exc.name])))
        return (
            None,
            env_vars,
            f"environment variable {exc.name} is not set",
        )

    return expanded_path, tuple(sorted(set(referenced_env_vars))), None


def _expand_home_prefix(
    declared_path: str,
    *,
    platform: SourceRootPlatform,
    env: Mapping[str, str],
) -> str:
    if declared_path == "~":
        home = _resolve_environment_variable("HOME", platform=platform, env=env)
        if home is None:
            raise _MissingEnvironmentVariable("HOME")
        return home
    if declared_path.startswith("~/"):
        home = _resolve_environment_variable("HOME", platform=platform, env=env)
        if home is None:
            raise _MissingEnvironmentVariable("HOME")
        return f"{home}/{declared_path[2:]}"
    return declared_path


def _resolve_environment_variable(
    name: str,
    *,
    platform: SourceRootPlatform,
    env: Mapping[str, str],
) -> str | None:
    direct_value = env.get(name)
    if direct_value:
        return direct_value

    if name == "HOME":
        user_profile = env.get("USERPROFILE")
        if user_profile:
            return user_profile
        return str(Path.home())

    if name == "USERPROFILE":
        home = env.get("HOME")
        if home:
            return home
        if platform == SourceRootPlatform.WINDOWS:
            return str(Path.home())
        return None

    if name == "XDG_CONFIG_HOME":
        home = _resolve_environment_variable("HOME", platform=platform, env=env)
        return None if home is None else f"{home}/.config"

    if name == "XDG_CACHE_HOME":
        home = _resolve_environment_variable("HOME", platform=platform, env=env)
        return None if home is None else f"{home}/.cache"

    if name == "XDG_STATE_HOME":
        home = _resolve_environment_variable("HOME", platform=platform, env=env)
        return None if home is None else f"{home}/.local/state"

    if name == "APPDATA":
        if platform != SourceRootPlatform.WINDOWS:
            return None
        user_profile = _resolve_environment_variable(
            "USERPROFILE",
            platform=platform,
            env=env,
        )
        return None if user_profile is None else f"{user_profile}/AppData/Roaming"

    if name == "LOCALAPPDATA":
        if platform != SourceRootPlatform.WINDOWS:
            return None
        user_profile = _resolve_environment_variable(
            "USERPROFILE",
            platform=platform,
            env=env,
        )
        return None if user_profile is None else f"{user_profile}/AppData/Local"

    if name == "PROGRAMDATA":
        if platform != SourceRootPlatform.WINDOWS:
            return None
        system_drive = env.get("SYSTEMDRIVE")
        if system_drive:
            return f"{system_drive}/ProgramData"
        return "C:/ProgramData"

    return None


__all__ = [
    "ResolvedSourceRoot",
    "SourceRootResolution",
    "all_platform_root",
    "darwin_root",
    "default_descriptor_input_roots",
    "linux_root",
    "normalize_source_root_platform",
    "resolve_explicit_input_roots",
    "resolve_source_roots",
    "windows_root",
]
