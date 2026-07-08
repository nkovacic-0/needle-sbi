"""
Scaler for the grouped particle-feature pipeline. Fits ONE set of statistics per
feature-row (pooling every real particle-type column for that row together),
rather than one per raw column.

Cache structure: self.cache is keyed by an arbitrary, purely cosmetic stand-in
name (feature_names, if provided and unique; otherwise "row_0", "row_1", ...).
Each entry is self-describing:
    {"fields": [<real column names in this row>], "scaler_params": {<stat>: <value>}}

Once fitted (or loaded), the scaler needs nothing else from the dataset config
to apply/revert — it reads its OWN "fields" list per entry, not
ingestor.feature_columns_grouped/feature_names. This makes a saved scaler
self-contained and its correctness checkable at load time (every field it
expects must actually exist in the target Ingestor), rather than depending on
an unenforced assumption that feature_names/row order still match.
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


def load_grouped_scaler(path: str | Path) -> "GroupedScaler":
    """Instantiate the correct GroupedScaler math type and load its saved cache."""
    with open(path) as f:
        payload = json.load(f)
    math_type = payload.get("math_type")
    if math_type not in _MATH:
        raise ValueError(f"Unknown or missing math_type in {path}: {math_type!r}")
    scaler = GroupedScaler(scaler_choice=math_type)
    scaler.cache = payload["cache"]
    return scaler


def _lazy_ravel_concat_compute(lazy_columns: list[dak.Array]) -> ak.Array:
    """Ravel + concatenate lazy columns, batched into one dask.compute() call.
    Mirrors the ak.ravel-on-dak.Array pattern already used in Ingestor
    (with the same DEBUG-gated warning suppression), since ak.ravel on a
    lazy dak.Array is not fully supported and emits warnings otherwise.
    """
    lazy_raveled = [ak.ravel(c) for c in lazy_columns]
    if logger.isEnabledFor(logging.DEBUG):
        (pooled,) = dask.compute(ak.concatenate(lazy_raveled))
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            (pooled,) = dask.compute(ak.concatenate(lazy_raveled))
    return pooled


def _resolve_row_keys(ingestor) -> list[str]:
    """Cosmetic cache keys, one per feature-row: ingestor.feature_names if
    provided and unique, otherwise "row_0", "row_1", ... . These are never
    used for lookup once cached — only for human-readable dumps.
    """
    names = getattr(ingestor, "feature_names", None)
    if names and len(set(names)) == len(names):
        return list(names)
    return [f"row_{i}" for i in range(len(ingestor.feature_columns_grouped))]


class GroupedScaler:
    def __init__(self, scaler_choice: str = "standard"):
        if scaler_choice not in _MATH:
            raise ValueError(f"Unknown scaler_choice '{scaler_choice}'. Available: {list(_MATH)}")
        self.scaler_choice = scaler_choice
        self.cache: dict[str, dict[str, Any]] = {}

    def _build_sampling_plan(
        self, ingestor, use_sampling: bool, sample_fraction: float, force_avoid_partition_sampling: bool
    ) -> dict[str, Any] | None:
        """Computed ONCE per apply() call (not per feature-row) — sampling
        granularity is a dataset-wide property, reused identically across
        every row for consistency. Returns None to mean "use everything".
        """
        if not use_sampling:
            return None
        if not (0.0 < sample_fraction <= 1.0):
            raise ValueError(f"sample_fraction must be in (0, 1], got {sample_fraction}")

        array = ingestor.array
        n_partitions = array.npartitions

        if n_partitions > 1 and (1 / n_partitions) < (sample_fraction + 0.01) and not force_avoid_partition_sampling:
            n_sample = max(1, round(n_partitions * sample_fraction))
            rng = np.random.default_rng(42)
            partition_ids = rng.choice(n_partitions, size=n_sample, replace=False).tolist()
            logger.debug(f"[GroupedScaler] Sampling {n_sample}/{n_partitions} partition(s).")
            return {"mode": "partitions", "partition_ids": partition_ids}

        total_events = len(array)
        n_sample = max(1, round(total_events * sample_fraction))
        logger.debug(f"[GroupedScaler] Sampling {n_sample}/{total_events} event(s).")
        return {"mode": "events", "n_sample": n_sample}

    def _pool_fields(self, array: dak.Array, fields: list[str], separator: str, plan: dict[str, Any] | None) -> np.ndarray:
        """Pool every listed field into a single 1-D eager array (per the
        active sampling plan), then drop NaN.
        """
        if plan is None:
            lazy = [NestedArrayIndexer.get_nested_field(array, c, separator) for c in fields]
            pooled = _lazy_ravel_concat_compute(lazy)

        elif plan["mode"] == "partitions":
            pieces = []
            for c in fields:
                col = NestedArrayIndexer.get_nested_field(array, c, separator)
                for p in plan["partition_ids"]:
                    pieces.append(ak.ravel(col.partitions[p].compute()))
            pooled = ak.concatenate(pieces)

        else:  # "events"
            n = plan["n_sample"]
            lazy = [NestedArrayIndexer.get_nested_field(array, c, separator)[:n] for c in fields]
            pooled = _lazy_ravel_concat_compute(lazy)

        values = ak.to_numpy(pooled)
        return values[~np.isnan(values)]

    def apply(
        self,
        ingestor,
        use_sampling: bool = False,
        sample_fraction: float = 0.10,
        force_avoid_partition_sampling: bool = False,
    ) -> dak.Array:
        """Fit pooled stats per feature-row (only over ingestor.fields — real,
        actually-read columns; schema-missing columns are untouched, filled
        later in raw space) and normalize each real column with its row's stat.

        Stores each row's fitted stats keyed by a cosmetic name (see
        _resolve_row_keys), alongside the EXACT list of fields it applies to —
        self-describing, so apply_with_cache()/revert() never again need to
        consult ingestor.feature_columns_grouped/feature_names.

        Important:
            Assumes ingestor.resolve_sentinels() has already run and been
            reassigned to ingestor.array — masking here only excludes NaN,
            it does not itself detect sentinels.

        Returns:
            dak.Array: a NEW array (immutable; caller must reassign, e.g.
                `features.array = scaler.apply(features)`).
        """
        array = ingestor.array
        plan = self._build_sampling_plan(ingestor, use_sampling, sample_fraction, force_avoid_partition_sampling)

        row_keys = _resolve_row_keys(ingestor)
        self.cache = {}

        logger.debug(f"[GroupedScaler] row_keys={row_keys} sampling_plan={plan}")


        for row_idx, row in enumerate(ingestor.feature_columns_grouped):
            key = row_keys[row_idx]
            real_cols = [c for c in row if c not in ingestor.missing_columns]

            sample = self._pool_fields(array, real_cols, ingestor.SEPARATOR, plan)
            stats = _MATH[self.scaler_choice]["fit"](sample)
            logger.debug(f"[GroupedScaler] key='{key}' fields={real_cols} n_pooled={len(sample)} stats={stats}")
            self.cache[key] = {"fields": real_cols, "scaler_params": stats}

            for c in real_cols:
                column = NestedArrayIndexer.get_nested_field(array, c, ingestor.SEPARATOR)
                normalized = _MATH[self.scaler_choice]["normalize"](column, stats)
                array = ak.with_field(array, normalized, where=tuple(c.split(ingestor.SEPARATOR)))

        return array

    def _verify_cache_against_ingestor(self, ingestor) -> None:
        """Fail loudly if the cache's expected fields don't match what's
        actually available on ingestor — e.g. a stale/mismatched config, or
        a cache loaded for a different dataset entirely. Every field listed
        in any cache entry must be a real, currently-read column.
        """
        available = set(ingestor.fields)
        seen: set[str] = set()
        for key, entry in self.cache.items():
            fields = set(entry["fields"])
            missing = fields - available
            if missing:
                raise ValueError(
                    f"GroupedScaler cache entry '{key}' expects field(s) {missing}, "
                    "which are not present in the current ingestor. The cache may have "
                    "been fit against a different config or dataset."
                )
            overlap = fields & seen
            if overlap:
                raise ValueError(
                    f"GroupedScaler cache entry '{key}' re-declares field(s) {overlap}, "
                    "already claimed by another cache entry. The cache is inconsistent."
                )
            seen |= fields
        logger.debug(f"[GroupedScaler] cache verified OK against ingestor: entries={list(self.cache.keys())}")

    def apply_with_cache(self, ingestor, cache: dict[str, dict[str, Any]]) -> dak.Array:
        """Apply using externally-provided (e.g. loaded) statistics — no
        fitting. Iterates the CACHE's own entries (each self-describing its
        fields), not ingestor.feature_columns_grouped — fully decoupled from
        the dataset config once a cache is in hand.
        """
        self.cache = cache
        self._verify_cache_against_ingestor(ingestor)

        array = ingestor.array
        for entry in self.cache.values():
            fields = entry["fields"]
            stats = entry["scaler_params"]
            for c in fields:
                column = NestedArrayIndexer.get_nested_field(array, c, ingestor.SEPARATOR)
                normalized = _MATH[self.scaler_choice]["normalize"](column, stats)
                array = ak.with_field(array, normalized, where=tuple(c.split(ingestor.SEPARATOR)))
        return array

    def revert(self, ingestor) -> dak.Array:
        """Undo normalization using the currently-fitted/loaded cache — same
        cache-entry-driven iteration as apply_with_cache().
        """
        self._verify_cache_against_ingestor(ingestor)

        array = ingestor.array
        for entry in self.cache.values():
            fields = entry["fields"]
            stats = entry["scaler_params"]
            for c in fields:
                column = NestedArrayIndexer.get_nested_field(array, c, ingestor.SEPARATOR)
                restored = _MATH[self.scaler_choice]["denormalize"](column, stats)
                array = ak.with_field(array, restored, where=tuple(c.split(ingestor.SEPARATOR)))
        return array

    def save(self, path: str | Path) -> None:
        if not self.cache:
            raise RuntimeError("GroupedScaler has no fitted cache to save; call apply() first.")
        serializable = {
            "math_type": self.scaler_choice,
            "cache": {
                key: {
                    "fields": list(entry["fields"]),
                    "scaler_params": {k: float(v) for k, v in entry["scaler_params"].items()},
                }
                for key, entry in self.cache.items()
            },
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(serializable, f, indent=2, sort_keys=True)
        logger.info(f"Saved grouped scaler cache ({list(self.cache.keys())}) to {path}")

    def load(self, path: str | Path) -> None:
        with open(path) as f:
            payload = json.load(f)
        saved_type = payload.get("math_type")
        if saved_type is not None and saved_type != self.scaler_choice:
            raise ValueError(
                f"Scaler mismatch: file {path} was saved with math_type='{saved_type}', "
                f"but this GroupedScaler uses '{self.scaler_choice}'."
            )
        self.cache = payload["cache"]
        logger.info(f"Loaded grouped scaler cache ({list(self.cache.keys())}) from {path}")