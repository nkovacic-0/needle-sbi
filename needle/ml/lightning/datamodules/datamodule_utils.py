import os
import tempfile
from pathlib import Path

import torch

from needle.etl.column_normalization import ColumnScaler

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

def labels_naming_collate_fn(batch, label_names: list[str]):
    features, labels, weights = zip(*batch)
    features = torch.stack(features)
    labels = torch.stack(labels)     # (B,) if squeezed single-column, else (B, F) 
    weights = torch.stack(weights)   # always (B,), thanks to weights_combine
    if labels.ndim == 1:
        labels_dict = {label_names[0]: labels}
    else:
        labels_dict = {name: labels[:, i] for i, name in enumerate(label_names)}
    return features, labels_dict, weights

def labels_naming_test_collate_fn(
    batch: list[tuple],
    label_names: list[str],
    scaler: ColumnScaler,
) -> tuple:
    """Test-mode collate for GroupedDataModule when aux_feature_fields is set.
    Batches the (features, labels, weights, aux) 4-tuples GroupedTorchDataset/
    GroupedDaskDataset yield in that case, and reverts normalization on the aux
    tensors using the SAME scaler cache features were scaled with — aux values
    reach the caller restored to physical units, not scaled.
    """
    features, labels, weights, aux = zip(*batch)
    features = torch.stack(features)
    labels = torch.stack(labels)
    weights = torch.stack(weights)
    if labels.ndim == 1:
        labels_dict = {label_names[0]: labels}
    else:
        labels_dict = {name: labels[:, i] for i, name in enumerate(label_names)}

    aux_dict = {
        field: scaler.denormalize_field(torch.stack([event[field] for event in aux]), field)
        for field in aux[0]
    }
    return features, labels_dict, weights, aux_dict