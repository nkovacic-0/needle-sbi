import os
import tempfile
from pathlib import Path

from needle.utils.logging import ColorFormatter

logger = ColorFormatter.get_logger("ml")


def resolve_versioned_path(
    base_path: str | Path,
    fold_index: int | None,
    suffix: str,
    force: bool = False,
) -> Path | None:
    """Resolve a fold-aware, collision-safe output path for a per-fold artifact.

    Behavior:
        - The canonical name for a given fold is `{stem}_{fold_index}{suffix}`.
        - If that file doesn't exist yet, it is returned as-is (first save, no
          (intended as the first save in real runs).
        - If it already exists and `force=False`, returns None — signaling the
          caller to skip saving (a prior run for this fold already produced it).
        - If it already exists and `force=True`, a new, uniquely-suffixed path
          is minted (bash mktemp-style) and reserved atomically, so a
          duplicate is written rather than overwriting the original.


    Behavior:
        - The canonical name for a given fold is `{stem}_{fold_index}{suffix}`.
        - If that file doesn't exist yet, it is returned as-is (the first save,
          in real runs).
        - If it already exists and `force=False`, returns None — signaling the
          caller to skip saving (a prior run for this fold already produced it).
        - If it already exists and `force=True`, a new, uniquely-suffixed path
          is minted (bash mktemp-style) and reserved atomically, so a
          duplicate is written rather than overwriting the original.
        
    NOTE: 
        Unless the upstream (whatever process is calling resolve_versioned_path)
        is running from a point in the DAG were there's no per-fold parallelism
        a TOCTOU race between different processes for the resolved path might
        ensue.

    Args:
        base_path (str | Path): Base path, e.g. '/data/out/scaler.json'. The
            fold tag is inserted before the suffix.
        fold_index (int | None): Fold index to encode into the filename.
            If None, no fold tag is added (e.g. non-fold-based runs).
        suffix (str): File extension, including the leading dot (e.g. '.json').
        force (bool): Whether to write a duplicate if the canonical file exists.

    Returns:
        Path | None: Path to write to, or None to indicate "skip, already saved".
    """
    base_path = Path(base_path)
    stem = base_path.stem if base_path.suffix else base_path.name
    parent = base_path.parent
    parent.mkdir(parents=True, exist_ok=True)

    fold_tag = f"_{fold_index}" if fold_index is not None else ""
    canonical = parent / f"{stem}{fold_tag}{suffix}"

    if not canonical.exists():
        return canonical

    if not force:
        logger.info(f"{canonical} already exists; skipping save (pass force=True to override).")
        return None

    # Canonical name is taken and force=True: atomically reserve a new,
    # uniquely-tagged path (avoids TOCTOU races if multiple fold/ensemble
    # tasks run concurrently against the same directory).
    fd, tmp_path = tempfile.mkstemp(prefix=f"{stem}{fold_tag}_", suffix=suffix, dir=parent)
    os.close(fd)  # just reserving the name; caller does the actual write
    logger.info(f"{canonical} already exists and force=True; writing duplicate to {tmp_path}.")
    return Path(tmp_path)