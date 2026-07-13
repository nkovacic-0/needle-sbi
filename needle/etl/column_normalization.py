"""
Per-column scaler for the grouped particle-feature pipeline. Fits ONE set of
statistics per REAL (non-schema-missing) raw column independently — no pooling
across particles/rows. 

Important:
    Assumes ingestor.resolve_sentinels() has already run and been reassigned
    to ingestor.array before apply()/apply_with_cache() are called — fitting
    here only excludes NaN, it does not itself detect sentinel values. Real
    sentinels must already have been converted to NaN upstream, or they will
    silently contaminate the fitted statistics.
"""

import json
import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import awkward as ak
import dask
import dask_awkward as dak

from needle.etl.array import NestedArrayIndexer
from needle.utils.logging import ColorFormatter

logger = ColorFormatter.get_logger("etl")


def _standard_fit(sample: np.ndarray) -> dict[str, float]:
    return {"mean": float(np.mean(sample)), "std": float(np.std(sample))}


def _standard_normalize(column: dak.Array, stats: dict[str, float]) -> dak.Array:
    return (column - stats["mean"]) / stats["std"]


def _standard_denormalize(column: dak.Array, stats: dict[str, float]) -> dak.Array:
    return column * stats["std"] + stats["mean"]


def _minmax_fit(sample: np.ndarray) -> dict[str, float]:
    return {"min": float(np.min(sample)), "max": float(np.max(sample))}


def _minmax_normalize(column: dak.Array, stats: dict[str, float]) -> dak.Array:
    return (column - stats["min"]) / (stats["max"] - stats["min"])


def _minmax_denormalize(column: dak.Array, stats: dict[str, float]) -> dak.Array:
    return column * (stats["max"] - stats["min"]) + stats["min"]


_MATH = {
    "standard": {"fit": _standard_fit, "normalize": _standard_normalize, "denormalize": _standard_denormalize},
    "minmax": {"fit": _minmax_fit, "normalize": _minmax_normalize, "denormalize": _minmax_denormalize},
}


def load_column_scaler(path: str | Path) -> "ColumnScaler":
    """Instantiate the correct ColumnScaler math type and load its saved cache."""
    with open(path) as f:
        payload = json.load(f)
    math_type = payload.get("math_type")
    if math_type not in _MATH:
        raise ValueError(f"Unknown or missing math_type in {path}: {math_type!r}")
    scaler = ColumnScaler(scaler_choice=math_type)
    scaler.cache = payload["cache"]
    return scaler


def _lazy_ravel_concat_compute(lazy_columns: list[dak.Array]) -> ak.Array:
    """Ravel + concatenate lazy columns, batched into one dask.compute() call.
    Mirrors the ak.ravel-on-dak.Array pattern already used in Ingestor
    (with the same DEBUG-gated warning suppression), since ak.ravel on a
    lazy dak.Array is not fully supported and emits warnings otherwise.
    Works fine with a single-element list (the common case here).
    """
    lazy_raveled = [ak.ravel(c) for c in lazy_columns]
    if logger.isEnabledFor(logging.DEBUG):
        (pooled,) = dask.compute(ak.concatenate(lazy_raveled))
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            (pooled,) = dask.compute(ak.concatenate(lazy_raveled))
    return pooled


class ColumnScaler:
    def __init__(self, scaler_choice: str = "standard"):
        if scaler_choice not in _MATH:
            err_msg = (f"Unknown scaler_choice '{scaler_choice}'. Available: {list(_MATH)}")
            logger.error(err_msg)
            raise ValueError(err_msg)
        self.scaler_choice = scaler_choice
        self.cache: dict[str, dict[str, float]] = {}

    def _build_sampling_plan(
        self, ingestor, use_sampling: bool, sample_fraction: float, force_avoid_partition_sampling: bool
    ) -> dict[str, Any] | None:
        """Computed ONCE per apply() call (not per column) — sampling
        granularity is a dataset-wide property, reused identically across
        every column for consistency. Returns None to mean "use everything".
        """
        if not use_sampling:
            return None
        if not (0.0 < sample_fraction <= 1.0):
            err_msg = (f"sample_fraction must be in (0, 1], got {sample_fraction}")
            logger.error(err_msg)
            raise ValueError(err_msg)

        array = ingestor.array
        n_partitions = array.npartitions

        if n_partitions > 1 and (1 / n_partitions) < (sample_fraction + 0.01) and not force_avoid_partition_sampling:
            n_sample = max(1, round(n_partitions * sample_fraction))
            rng = np.random.default_rng(42)
            partition_ids = rng.choice(n_partitions, size=n_sample, replace=False).tolist()
            logger.debug(f"[ColumnScaler] Sampling {n_sample}/{n_partitions} partition(s).")
            return {"mode": "partitions", "partition_ids": partition_ids}

        total_events = len(array)
        n_sample = max(1, round(total_events * sample_fraction))
        logger.debug(f"[ColumnScaler] Sampling {n_sample}/{total_events} event(s).")
        return {"mode": "events", "n_sample": n_sample}

    def _sample_field(self, array: dak.Array, field: str, separator: str, plan: dict[str, Any] | None) -> np.ndarray:
        """Sample a single field into a 1-D eager array (per the active
        sampling plan), then drop NaN. NaN here covers both real missing
        entries and sentinel values already converted to NaN upstream by
        ingestor.resolve_sentinels() — this method does not itself detect
        sentinels, only excludes NaN.
        """
        if plan is None:
            col = NestedArrayIndexer.get_nested_field(array, field, separator)
            pooled = _lazy_ravel_concat_compute([col])

        elif plan["mode"] == "partitions":
            col = NestedArrayIndexer.get_nested_field(array, field, separator)
            pieces = [ak.ravel(col.partitions[p].compute()) for p in plan["partition_ids"]]
            pooled = ak.concatenate(pieces)

        else:  # "events"
            n = plan["n_sample"]
            col = NestedArrayIndexer.get_nested_field(array, field, separator)[:n]
            pooled = _lazy_ravel_concat_compute([col])

        values = ak.to_numpy(pooled)
        return values[~np.isnan(values)]

    def apply(
        self,
        ingestor,
        use_sampling: bool = False,
        sample_fraction: float = 0.10,
        force_avoid_partition_sampling: bool = False,
    ) -> dak.Array:
        """Fit stats independently for every REAL column (ingestor.fields —
        already excludes schema-missing columns by construction) and
        normalize each one with its own fitted stats.

        Important:
            Assumes ingestor.resolve_sentinels() has already run and been
            reassigned to ingestor.array — see module docstring.

        Returns:
            dak.Array: a NEW array (immutable; caller must reassign, e.g.
                `features.array = scaler.apply(features)`).
        """
        array = ingestor.array
        plan = self._build_sampling_plan(ingestor, use_sampling, sample_fraction, force_avoid_partition_sampling)
        self.cache = {}

        logger.debug(f"[ColumnScaler] fitting {len(ingestor.fields)} real column(s), sampling_plan={plan}")

        for field in ingestor.fields:
            sample = self._sample_field(array, field, ingestor.SEPARATOR, plan)
            stats = _MATH[self.scaler_choice]["fit"](sample)
            logger.debug(f"[ColumnScaler] field='{field}' n_sampled={len(sample)} stats={stats}")
            self.cache[field] = stats

            column = NestedArrayIndexer.get_nested_field(array, field, ingestor.SEPARATOR)
            normalized = _MATH[self.scaler_choice]["normalize"](column, stats)
            array = ak.with_field(array, normalized, where=tuple(field.split(ingestor.SEPARATOR)))

        return array

    def _verify_cache_against_ingestor(self, ingestor) -> None:
        """Fail loudly if the cache doesn't match ingestor.fields exactly —
        either direction. A cache field absent from the ingestor suggests a
        stale/mismatched config or a cache from a different dataset; an
        ingestor field absent from the cache would otherwise be silently
        left unnormalized by apply_with_cache()/revert().
        """
        cache_fields = set(self.cache.keys())
        real_fields = set(ingestor.fields)

        unexpected = cache_fields - real_fields
        if unexpected:
            err_msg = (
                f"ColumnScaler cache has stats for field(s) {unexpected}, not present in "
                "the current ingestor. The cache may have been fit against a different "
                "config or dataset."
            )
            logger.error(err_msg)
            raise ValueError(err_msg)
        uncovered = real_fields - cache_fields
        if uncovered:
            err_msg = (
                f"ColumnScaler cache is missing stats for real field(s) {uncovered} present "
                "in the current ingestor. Applying this cache would silently leave them "
                "unnormalized."
            )
            logger.error(err_msg)
            raise ValueError(err_msg)
        logger.debug(f"[ColumnScaler] cache verified OK against ingestor: {len(cache_fields)} field(s).")

    def apply_with_cache(self, ingestor, cache: dict[str, dict[str, float]]) -> dak.Array:
        """Apply using externally-provided (e.g. loaded) statistics — no fitting."""
        self.cache = cache
        self._verify_cache_against_ingestor(ingestor)

        array = ingestor.array
        for field, stats in self.cache.items():
            column = NestedArrayIndexer.get_nested_field(array, field, ingestor.SEPARATOR)
            normalized = _MATH[self.scaler_choice]["normalize"](column, stats)
            array = ak.with_field(array, normalized, where=tuple(field.split(ingestor.SEPARATOR)))
        return array

    def revert(self, ingestor) -> dak.Array:
        """Undo normalization using the currently-fitted/loaded cache."""
        self._verify_cache_against_ingestor(ingestor)

        array = ingestor.array
        for field, stats in self.cache.items():
            column = NestedArrayIndexer.get_nested_field(array, field, ingestor.SEPARATOR)
            restored = _MATH[self.scaler_choice]["denormalize"](column, stats)
            array = ak.with_field(array, restored, where=tuple(field.split(ingestor.SEPARATOR)))
        return array
    
    def denormalize_field(self, value, field: str):
        """Revert normalization for a single field's already-scaled values,
        given directly (e.g. a torch.Tensor pulled from a batch) rather than
        as part of an Ingestor's dask/awkward array — unlike revert(), which
        operates on a whole ingestor's lazy array. Uses the same cached stats
        and math as revert()/apply(), so callers stay insulated from any
        future change to that math.
        """
        if field not in self.cache:
            raise KeyError(f"No cached stats for field '{field}'; cache has: {list(self.cache)}")
        return _MATH[self.scaler_choice]["denormalize"](value, self.cache[field])

    def save(self, path: str | Path) -> None:
        if not self.cache:
            err_msg = ("ColumnScaler has no fitted cache to save, call apply() first.")
            logger.error(err_msg)
            raise RuntimeError(err_msg)
        serializable = {
            "math_type": self.scaler_choice,
            "cache": {
                field: {k: float(v) for k, v in stats.items()}
                for field, stats in self.cache.items()
            },
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(serializable, f, indent=2, sort_keys=True)
        logger.info(f"Saved column scaler cache ({len(self.cache)} field(s)) to {path}")

    def load(self, path: str | Path) -> None:
        with open(path) as f:
            payload = json.load(f)
        saved_type = payload.get("math_type")
        if saved_type is not None and saved_type != self.scaler_choice:
            err_msg = (
                f"Scaler mismatch: file {path} was saved with math_type='{saved_type}', "
                f"but this ColumnScaler uses '{self.scaler_choice}'."
            )
            logger.error(err_msg)
            raise ValueError(err_msg)
        self.cache = payload["cache"]
        logger.info(f"Loaded column scaler cache ({len(self.cache)} field(s)) from {path}")