"""
Compute normalization constants from Parquet file metadata.

Collects per-file (partition) column statistics such as min/max,
without reading any data into memory.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Any, ClassVar

import numpy as np
import awkward as ak
import dask_awkward as dak
import pyarrow.parquet as pq

from needle.etl.array import NestedArrayIndexer
from needle.utils.logging import ColorFormatter

logger = ColorFormatter.get_logger("etl")

#: Nested metadata structure: ``{field → {scaler_key → {class_index → value}}}``.
type MetaDataDict = dict[str, dict[str, dict[int, Any]]]


def clean_column_name(path: str) -> str:
    """
    Clean up pyarrow column names to the one actually in the parquet file

    Important:
        Will filter out the fields "list" and "item".

    Example:
        >>> clean_column_name('Lepton.list.item.eta')
        'Lepton.eta'

        >>> clean_column_name('event')
        'event'
    """
    parts = path.split(".")
    filtered = [p for p in parts if p not in {"list", "item"}]
    return ".".join(filtered)

# TODO: this helper functions seems unifinished, or rather, that it was never implemented in scalers
def gather_metadata(paths: list[str]) -> MetaDataDict:
    """
    Collect min/max stats for all columns in each Parquet file using pyarrow.

    Args:
        paths: list of Parquet file paths.

    Returns:
        dict:
            A nest dict of shape:
                {<column_name>: {"min": {<partition_id>: <val>}, "max": ...}, ...}, ...}
    """
    metadata: MetaDataDict = {}

    for pid, path in enumerate(paths):
        file_meta = pq.read_metadata(path)

        n_row_groups = file_meta.num_row_groups

        per_file_stats: dict[str, dict[str, Any]] = {}

        for row_group_index in range(n_row_groups):
            row_group = file_meta.row_group(row_group_index)

            for col_idx in range(row_group.num_columns):
                column = row_group.column(col_idx)
                name = clean_column_name(str(column.path_in_schema))
                stats = column.statistics

                if (stats is None) or (stats.min is None) or (stats.max is None):
                    continue

                cmin = stats.min if not hasattr(stats.min, "as_py") else stats.min.as_py()
                cmax = stats.max if not hasattr(stats.max, "as_py") else stats.max.as_py()

                if name not in per_file_stats:
                    per_file_stats[name] = {"min": cmin, "max": cmax}
                else:
                    per_file_stats[name]["min"] = min(per_file_stats[name]["min"], cmin)
                    per_file_stats[name]["max"] = max(per_file_stats[name]["max"], cmax)

        for col, stats in per_file_stats.items():
            col_meta = metadata.setdefault(col, {"min": {}, "max": {}})
            col_meta["min"][pid] = stats["min"]
            col_meta["max"][pid] = stats["max"]

    return metadata


def load_scaler(path: str | Path) -> BaseScaler:
    """Instantiate the correct scaler class and load its saved cache, without
    the caller needing to already know which scaler type was used at train time.
    """
    with open(path) as f:
        payload = json.load(f)

    registry = {"StandardScaler": StandardScaler, "MinMaxScaler": MinMaxScaler}
    scaler_type = payload.get("scaler_type")
    if scaler_type not in registry:
        raise ValueError(f"Unknown or missing scaler_type in {path}: {scaler_type!r}")

    scaler = registry[scaler_type]()
    scaler.cache = payload["cache"]
    return scaler


class BaseScaler(ABC):
    cache: dict[str, dict[str, Any]]
    stat_keys: ClassVar[tuple[str, str]]  # e.g. ("mean", "std") or ("min", "max")

    @abstractmethod
    def fit_stats(self, sample: ak.Array) -> dict[str, float]:
        """Compute this scaler's statistics (e.g. {'mean': ..., 'std': ...}) from an eager sample."""

    @abstractmethod
    def normalize(self, column: dak.Array, stats: dict[str, float]) -> dak.Array:
        """Apply the forward transformation given fitted/loaded statistics."""

    @abstractmethod
    def denormalize(self, column: dak.Array, stats: dict[str, float]) -> dak.Array:
        """Apply the inverse transformation given fitted/loaded statistics."""

    def apply(
        self,
        array: dak.Array,
        use_sampling: bool = False,
        sample_fraction: float = 0.02,
        force_avoid_partition_sampling: bool = True,
    ) -> dak.Array:
        """Fit statistics from `array` and apply the transformation.

        Args:
            array (dask_awkward.Array): The Array to normalize.
            use_sampling (bool): If True, estimate statistics from a subset of the
                data rather than the full dataset — substantially faster for large
                datasets, at the cost of an approximate (not bit-for-bit
                reproducible-from-all-data) estimate. Do not use if exact,
                fully-reproducible statistics are required.
            sample_fraction (float): Fraction of data to sample when `use_sampling=True`.
            force_avoid_partition_sampling (bool): If True, sample a contiguous
                slice of events instead of whole partitions — needed when the
                dataset has too few partitions for partition-level sampling to be
                meaningful (e.g. a single dask partition per file). Assumes the
                underlying data is already shuffled; a contiguous slice of an
                unshuffled dataset is not a representative sample.

        Returns:
            dask_awkward.Array: A NEW array with each field replaced by its
                normalized version (immutable, original `array` is unchanged —
                callers MUST use the returned value).
        """
        self.cache = {key: {} for key in self.stat_keys}
        return self._transform(
            array, fit=True,
            use_sampling=use_sampling,
            sample_fraction=sample_fraction,
            force_avoid_partition_sampling=force_avoid_partition_sampling,
        )

    def apply_with_cache(self, array: dak.Array, cache: dict[str, dict[str, Any]]) -> dak.Array:
        """Apply the transformation using externally-provided statistics
        (e.g. loaded via `load()`), rather than fitting fresh ones. Use for
        validation/test/inference data.
        """
        self.cache = cache
        return self._transform(array, fit=False)

    def _sample_column(
        self,
        column: dak.Array,
        use_sampling: bool,
        sample_fraction: float,
        force_avoid_partition_sampling: bool,
    ) -> ak.Array:
        """Return an EAGER ak.Array to fit statistics from — either the full
        column, or a subset chosen per `use_sampling`/`sample_fraction`.

        Always returns a materialized (computed) array; callers (fit_stats
        implementations) never need to call .compute() or worry about lazy
        dak.Array semantics.
        """
        if not use_sampling:
            return column.compute()

        if not (0.0 < sample_fraction <= 1.0):
            raise ValueError(f"sample_fraction must be in (0, 1], got {sample_fraction}")

        n_partitions = column.npartitions

        if n_partitions > 1 and (1 / n_partitions) < (sample_fraction + 0.01) and not force_avoid_partition_sampling:
            # NOTE: par-partition samplin is only meaningful when there are enough partitions to subsample from
            # NOTE: unless the data is shuffled/unordered over partitions this is not a good way to sample, but it should be fast
            n_sample = max(1, round(n_partitions * sample_fraction))
            rng = np.random.default_rng(42)
            sample_partition_ids = rng.choice(n_partitions, size=n_sample, replace=False)
            logger.debug(f"Sampling {n_sample}/{n_partitions} partition(s) (fraction={sample_fraction}).")
            return ak.concatenate([column.partitions[int(p)].compute() for p in sample_partition_ids])

        # Per-event slice sampling — needed when partition count is too low (e.g. a single dask partition spanning the whole file)
        # NOTE: assumes the underlying data is already shuffled, otherwise a contiguous slice of data is not a representative sample
        total_events = len(column)
        n_sample = max(1, round(total_events * sample_fraction))
        logger.debug(f"Sampling {n_sample}/{total_events} event(s) (fraction={sample_fraction}).")
        return column[:n_sample].compute()

    def _transform(
        self,
        array: dak.Array,
        fit: bool,
        use_sampling: bool = False,
        sample_fraction: float = 0.02,
        force_avoid_partition_sampling: bool = True,
    ) -> dak.Array:
        fields = NestedArrayIndexer.list_all_fields(array, as_tuple=False, separator=".")
        logger.debug(f"Scaler processing {len(fields)} field(s): {fields}")

        for i, field in enumerate(fields):
            logger.debug(f"[{i+1}/{len(fields)}] {'Fitting' if fit else 'Applying'} scaler for field '{field}'...")
            column = NestedArrayIndexer.get_nested_field(array, field, separator=".")

            if fit:
                sample = self._sample_column(column, use_sampling, sample_fraction, force_avoid_partition_sampling)
                stats = self.fit_stats(sample)
                for key, value in stats.items():
                    self.cache[key][field] = value
            else:
                stats = {key: self.cache[key][field] for key in self.stat_keys}

            normalized = self.normalize(column, stats)
            array = ak.with_field(array, normalized, where=tuple(field.split(".")))
            logger.debug(f"[{i+1}/{len(fields)}] Complete, done with field '{field}'.")

        return array

    def revert(self, array: dak.Array) -> dak.Array:
        """Undo the normalization transformation using the currently-fitted/loaded cache."""
        fields = NestedArrayIndexer.list_all_fields(array, as_tuple=False, separator=".")
        for field in fields:
            stats = {key: self.cache[key][field] for key in self.stat_keys}
            column = NestedArrayIndexer.get_nested_field(array, field, separator=".")
            restored = self.denormalize(column, stats)
            array = ak.with_field(array, restored, where=tuple(field.split(".")))
        return array

    def save(self, path: str | Path) -> None:
        """Persist the fitted cache (e.g. mean/std or min/max per field) to JSON.

        Raises:
            RuntimeError: If `apply()` (or `apply_with_cache()`) has not been called yet,
                i.e. there is nothing fitted to save.
        """
        if not getattr(self, "cache", None):
            raise RuntimeError(
                "Scaler has no fitted cache to save; call `apply()` (to fit on data) "
                "or `load()`/`apply_with_cache()` (to reuse existing statistics) first."
            )
        serializable = {
            "scaler_type": type(self).__name__,
            "cache": {
                metric: {field: float(value) for field, value in field_values.items()}
                for metric, field_values in self.cache.items()
            },
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(serializable, f, indent=2, sort_keys=True)
        logger.info(f"Saved scaler cache ({list(self.cache.keys())}) to {path}")

    def load(self, path: str | Path) -> None:
        """Load a previously-saved cache from JSON, without needing to call `apply()`.

        Important:
            Use this at inference/test time: load the training-fitted statistics and
            then call `apply_with_cache()` — NOT `apply()`, which would silently refit
            fresh statistics from whatever array it's given (e.g. the test set),
            rather than reusing the ones saved from training.
        """
        with open(path) as f:
            payload = json.load(f)
        saved_type = payload.get("scaler_type")
        if saved_type is not None and saved_type != type(self).__name__:
            raise ValueError(
                f"Scaler mismatch: file {path} was saved by '{saved_type}', "
                f"but load() was called on a '{type(self).__name__}' instance."
            )
        self.cache = payload["cache"]
        logger.info(f"Loaded scaler cache ({list(self.cache.keys())}) from {path}")

class StandardScaler(BaseScaler):
    stat_keys = ("mean", "std")

    def fit_stats(self, sample: ak.Array) -> dict[str, float]:
        return {"mean": ak.mean(sample), "std": ak.std(sample)}

    def normalize(self, column: dak.Array, stats: dict[str, float]) -> dak.Array:
        return (column - stats["mean"]) / stats["std"]

    def denormalize(self, column: dak.Array, stats: dict[str, float]) -> dak.Array:
        return column * stats["std"] + stats["mean"]


class MinMaxScaler(BaseScaler):
    stat_keys = ("min", "max")

    def fit_stats(self, sample: ak.Array) -> dict[str, float]:
        return {"min": ak.min(sample), "max": ak.max(sample)}

    def normalize(self, column: dak.Array, stats: dict[str, float]) -> dak.Array:
        return (column - stats["min"]) / (stats["max"] - stats["min"])

    def denormalize(self, column: dak.Array, stats: dict[str, float]) -> dak.Array:
        return column * (stats["max"] - stats["min"]) + stats["min"]