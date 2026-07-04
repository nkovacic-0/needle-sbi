"""
Compute normalization constants from Parquet file metadata.

Collects per-file (partition) column statistics such as min/max,
without reading any data into memory.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

import dask 
import awkward as ak
import dask_awkward as dak
import pyarrow.parquet as pq

from needle.etl.array import NestedArrayIndexer
from needle.utils.logging import ColorFormatter

logger = ColorFormatter.get_logger("ml")

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

def load_scaler(path: str | Path) -> ScalerProtocol:
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

class ScalerProtocol(Protocol):
    cache: dict[str, dict[str, Any]]
    """Format: `{"<metric>": {"<field>": <value>}, ...}`"""

    def apply(self, array: dak.Array) -> dak.Array:
        ...

    def revert(self, array: dak.Array) -> dak.Array:
        ...

    def apply_with_cache(self, array: dak.Array, cache: dict[str, dict[str, Any]]) -> dak.Array:
        ...

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


class MinMaxScaler(ScalerProtocol):
    def apply(self, array: dak.Array) -> dak.Array:
        """Fit min/max statistics from `array` and apply a min-max transformation.

        Transformation:
            (x - min) / (max - min)

        Where max, min are calculated independently for each field over the whole array.

        Args:
            array (dask_awkward.Array): The Array to normalize

        Returns:
            dask_awkward.Array: A NEW array with each field replaced by its
                normalized version. `dak.Array`/`ak.Array` are immutable, so the
                original `array` argument is never modified in place — callers
                MUST use the returned value (e.g. `features.array = scaler.apply(features.array)`),
                not assume the input was updated in place.

        Note:
            The way min and max are calculated are not optimal since we load in the full arrays each
            time for each field. On the other hand, using `dak.min` fails due to a known bug.
        """
        self.cache = {"min": dict(), "max": dict()}
        return self._transform(array, fit=True)

    def apply_with_cache(self, array: dak.Array, cache: dict[str, dict[str, Any]]) -> dak.Array:
        """Apply min-max normalization using externally-provided statistics
        (e.g. loaded via `load()` from a training run), rather than fitting
        fresh statistics from `array`. Use this for validation/test/inference data.
        """
        self.cache = cache
        return self._transform(array, fit=False)

    def _transform(self, array: dak.Array, fit: bool) -> dak.Array:
        fields = NestedArrayIndexer.list_all_fields(array, as_tuple=False, separator=".")
        logger.debug(f"Scaler processing {len(fields)} field(s): {fields}")

        for i, field in enumerate(fields):
            logger.debug(f"[{i+1}/{len(fields)}] Fitting scaler for field '{field}'...")
            column = NestedArrayIndexer.get_nested_field(array, field, separator=".")

            if fit:
                # TODO: move the two computes here into a single compute call
                minimum = ak.min(column.compute())
                maximum = ak.max(column.compute())
                self.cache["min"][field] = minimum
                self.cache["max"][field] = maximum
            else:
                minimum = self.cache["min"][field]
                maximum = self.cache["max"][field]

            normalized = (column - minimum) / (maximum - minimum)
            array = ak.with_field(array, normalized, where=tuple(field.split(".")))
            logger.debug(f"[{i+1}/{len(fields)}] Comlete, done with field '{field}'.")

        return array

    def revert(self, array: dak.Array) -> dak.Array:
        """Undo the normalization transformation

        Transformation:
            x * (max - min) + min

        Args:
            array (dask_awkward.Array): The Array to un-normalize

        Returns:
            dask_awkward.Array: A NEW array (see note on `apply()` regarding immutability).
        """
        fields = NestedArrayIndexer.list_all_fields(array, as_tuple=False, separator=".")

        for field in fields:
            minimum = self.cache["min"][field]
            maximum = self.cache["max"][field]

            column = NestedArrayIndexer.get_nested_field(array, field, separator=".")
            restored = column * (maximum - minimum) + minimum
            array = ak.with_field(array, restored, where=tuple(field.split(".")))

        return array


class StandardScaler(ScalerProtocol):
    def apply(self, array: dak.Array) -> dak.Array:
        """Fit mean/std statistics from `array` and apply a standard scaling transformation.

        Transformation:
            (x - mean) / std

        Where mean, std are calculated independently for each field over the whole array.

        Args:
            array (dask_awkward.Array): The Array to normalize

        Returns:
            dask_awkward.Array: A NEW array with each field replaced by its
                normalized version. `dak.Array`/`ak.Array` are immutable, so the
                original `array` argument is never modified in place — callers
                MUST use the returned value (e.g. `features.array = scaler.apply(features.array)`),
                not assume the input was updated in place.
        """
        self.cache = {"mean": dict(), "std": dict()}
        return self._transform(array, fit=True)

    def apply_with_cache(self, array: dak.Array, cache: dict[str, dict[str, Any]]) -> dak.Array:
        """Apply standard scaling using externally-provided statistics
        (e.g. loaded via `load()` from a training run), rather than fitting
        fresh statistics from `array`. Use this for validation/test/inference data.
        """
        self.cache = cache
        return self._transform(array, fit=False)

    def _transform(self, array: dak.Array, fit: bool) -> dak.Array:
        fields = NestedArrayIndexer.list_all_fields(array, as_tuple=False, separator=".")
        logger.debug(f"Scaler processing {len(fields)} field(s): {fields}")

        for i, field in enumerate(fields):
            logger.debug(f"[{i+1}/{len(fields)}] Fitting scaler for field '{field}'...")
            column = NestedArrayIndexer.get_nested_field(array, field, separator=".")

            if fit:
                # mean = ak.mean(column.compute())
                # std = ak.std(column.compute())
                # let's try putting both of these computes into one call:
                # NOTE: annoyingly, the ak.mean lazy is not supported by dask - it is literally left as TODO in dask awkward source! 
                # mean_lazy = ak.mean(column)
                # std_lazy = ak.std(column)
                # mean, std = dask.compute(mean_lazy, std_lazy)
                # for the reason above, we gotta do the mean/std lazy calc "manually":
                n = ak.count(column)
                total = ak.sum(column)
                total_sq = ak.sum(column ** 2)
                n_val, total_val, total_sq_val = dask.compute(n, total, total_sq)

                mean = total_val / n_val
                std = ((total_sq_val / n_val) - (mean ** 2))**0.5
                
                self.cache["mean"][field] = mean
                self.cache["std"][field] = std
            else:
                mean = self.cache["mean"][field]
                std = self.cache["std"][field]

            normalized = (column - mean) / std
            array = ak.with_field(array, normalized, where=tuple(field.split(".")))
            logger.debug(f"[{i+1}/{len(fields)}] Comlete, done with field '{field}'.")

        return array

    def revert(self, array: dak.Array) -> dak.Array:
        """Undo the normalization transformation

        Transformation:
            x * std + mean

        Args:
            array (dask_awkward.Array): The Array to un-normalize

        Returns:
            dask_awkward.Array: A NEW array (see note on `apply()` regarding immutability).
        """
        fields = NestedArrayIndexer.list_all_fields(array, as_tuple=False, separator=".")

        for field in fields:
            mean = self.cache["mean"][field]
            std = self.cache["std"][field]

            column = NestedArrayIndexer.get_nested_field(array, field, separator=".")
            restored = column * std + mean
            array = ak.with_field(array, restored, where=tuple(field.split(".")))

        return array