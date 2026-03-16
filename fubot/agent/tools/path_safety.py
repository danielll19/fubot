"""Helpers for consistent workspace path resolution."""

from __future__ import annotations

import os
from pathlib import Path


def resolve_user_path(path: str, *, base_dir: Path | None = None) -> Path:
    """Resolve a user-provided path using realpath semantics."""
    expanded = Path(os.path.expandvars(path)).expanduser()
    if not expanded.is_absolute() and base_dir is not None:
        expanded = base_dir / expanded
    return expanded.resolve(strict=False)


def resolve_within_directory(
    path: str,
    *,
    base_dir: Path | None = None,
    allowed_dir: Path | None = None,
    label: str = "Path",
) -> Path:
    """Resolve a path and enforce that it stays under the allowed directory."""
    resolved = resolve_user_path(path, base_dir=base_dir)
    if allowed_dir is None:
        return resolved

    allowed_root = allowed_dir.expanduser().resolve(strict=False)
    try:
        resolved.relative_to(allowed_root)
    except ValueError as exc:
        raise PermissionError(f"{label} {path} is outside allowed directory {allowed_root}") from exc
    return resolved
