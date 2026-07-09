import reprlib
from typing import Any, Literal

import dask
import numpy as np
import awkward as ak
import dask_awkward as dak
import pyarrow.parquet as pq
import uproot

from needle.etl.array import NestedArrayIndexer, resolve_paths, brute_force_max_list_length
from needle.etl.dask_ingestor import Ingestor
from needle.utils.logging import ColorFormatter

logger = ColorFormatter.get_logger("etl")


class GroupedIngestor(Ingestor):
    """Ingestor variant for the grouped particle-feature pipeline.

    Reuses Ingestor's format resolution, length lookup, and array inspection
    unchanged (via inheritance) — only column resolution differs: columns
    declared in `replacements_missing_columns` are tolerated as schema-absent
    (never sent to the reader; warned about if actually found), while any
    other requested-but-missing column still hard-fails, same as Ingestor.
    """

    def __init__(
        self,
        paths: str | list[str],
        feature_columns_grouped: list[list[str]],
        particle_names: list[str] | None = None,
        feature_names: list[str] | None = None,
        replacements_missing_columns: dict[str, float] | None = None,
        replacements_default: float = 0.0,
        general_sentinel_value: float | None = None,
        replace_all_sentinel_values: bool = True,
        replacements_sentinel_default: float = float("nan"),
        replacements_sentinel_columns: dict[str, float | None] | None = None,
        format: Literal["parquet", "root", "automatic"] = "automatic",
        reader_kwargs: dict[str, Any] | None = None,
        max_number_events: int = -1,
        name: str = "unnamed",
    ) -> None:
        paths = [paths] if isinstance(paths, str) else paths
        self.paths = paths
        self.name = name

        self.feature_columns_grouped = feature_columns_grouped
        n_particles = len(feature_columns_grouped[0])
        self.particle_names = particle_names or [f"particle_{i}" for i in range(n_particles)]
        self.feature_names = feature_names or [f"feature_{i}" for i in range(len(feature_columns_grouped))]

        self.replacements_missing_columns = replacements_missing_columns or {}
        self.replacements_default = replacements_default
        self.missing_columns = set(self.replacements_missing_columns)
        logger.debug(f"[{self.name}] Declared missing_columns: {self.missing_columns}")

        self.general_sentinel_value = general_sentinel_value
        self.replace_all_sentinel_values = replace_all_sentinel_values
        self.replacements_sentinel_default = replacements_sentinel_default
        self.replacements_sentinel_columns = replacements_sentinel_columns or {}

        all_requested = [col for row in feature_columns_grouped for col in row]
        columns_to_read = [c for c in all_requested if c not in self.missing_columns]

        format = self._resolve_format(format, paths[0])
        reader_kwargs = reader_kwargs or {}

        logger.debug(f"[{self.name}] Building lazy dak.Array via {format}, {len(columns_to_read)} column(s)...")
        match format:
            case "parquet":
                self.array = dak.from_parquet(paths, columns=columns_to_read, **reader_kwargs)
                # logger.debug(f"[{self.name}] after from_parquet: npartitions={self.array.npartitions}, divisions={self.array.divisions}")
            case "root":
                self.array = uproot.dask(paths, columns=columns_to_read, **reader_kwargs)

        # NOTE: eager_compute_divisions breaks row groups from parquet's internal partitioning!
        # we'll comment it out here, but this needs to be addressed
        # also, NOTE: current _inspect_array overload for GroupedIngestor only works with single file-partitioning
        # if it can execute pyarrow.parquet reads on files
        # self.array.eager_compute_divisions()
        # logger.debug(f"[{self.name}] after eager_compute_divisions: npartitions={self.array.npartitions}, divisions={self.array.divisions}")

        self._inspect_array(self.array, paths)
        # pre_npartitions = self.array.npartitions
        # self._inspect_array(self.array, paths)
        # if self.array.npartitions < pre_npartitions:
        #     logger.warning(
        #         f"[{self.name}] _inspect_array's brute_force_divisions fallback collapsed "
        #         f"{pre_npartitions} partitions down to {self.array.npartitions}; this is a known "
        #         "issue with split_row_groups reads. Skipping the fallback."
        #     )
        # logger.debug(f"[{self.name}] after _inspect_array: npartitions={self.array.npartitions}, divisions={self.array.divisions}")

        loaded_columns = NestedArrayIndexer.list_all_fields(self.array, separator=self.SEPARATOR, as_tuple=False)

        found_but_declared_missing = self.missing_columns & set(loaded_columns)
        if found_but_declared_missing:
            logger.warning(
                f"[{self.name}] Column(s) declared in replacements_missing_columns but actually "
                f"present in the dataset: {found_but_declared_missing}. Config may be stale."
            )

        self._check_if_all_columns_found(columns_to_read, loaded_columns)
        self.fields = columns_to_read
        self.num_classes = len(self.fields)

        if max_number_events > 0:
            self.array = self.array[0:max_number_events]

        self.length = self._get_length(self[self.fields[0]], paths)
        logger.info(
            f"[{self.name}] Loaded {self.length} events, {len(all_requested)} requested field(s) "
            f"({len(self.missing_columns)} schema-missing): {reprlib.repr(all_requested)}"
        )

    def _inspect_array(self, array: dak.Array, paths: str | list[str]) -> None:
        """Overridden: base Ingestor's divisions fallback (brute_force_divisions)
        assumes 1 partition == 1 file, which is WRONG whenever partitions are
        finer than file-level (e.g. split_row_groups=True). Using it in that case
        silently collapses real partitions down to file-count, breaking chunked
        iteration (and, more subtly, potentially only ever reading partition 0's
        worth of data across the whole training run).

        Keeps the same "does the array have fields" check as the base method, but
        replaces the divisions-recovery fallback: only uses the file-level
        brute-force approach when partition count genuinely matches file count;
        otherwise recovers real per-partition lengths directly from dask's own
        already-built partition structure, which is correct regardless of what
        granularity partitions actually are.
        """
        if not hasattr(array, "fields") or not array.fields:
            try:
                err_msg = (
                    f"Input array does not have any fields: {array.fields}. "
                    "Please check the validity of the input columns. Available fields are: "
                    f"{dak.from_parquet(paths).fields}. "
                )
                logger.error(err_msg)
                raise ValueError(err_msg)
            except AttributeError:
                err_msg = ("Input array does not have attribute 'fields' or is empty.")
                logger.error(err_msg)
                raise ValueError(err_msg)

        if not any(array.divisions):
            n_partitions = array.npartitions
            resolved_paths = resolve_paths(paths)

            if n_partitions == len(resolved_paths):
                logger.debug(
                    f"[{self.name}] npartitions ({n_partitions}) matches file count; "
                    "file-level brute_force_divisions is safe here."
                )
                self.array._divisions = brute_force_divisions(resolved_paths)
            else:
                logger.debug(
                    f"[{self.name}] npartitions ({n_partitions}) != file count "
                    f"({len(resolved_paths)}); recovering divisions via map_partitions(len) "
                    "instead, to avoid collapsing real partitions down to file boundaries."
                )
                # partition_lengths = array.map_partitions(len).compute() # <-- this causes a type error crash!
                # self.array._divisions = tuple(np.cumsum([0] + list(partition_lengths)).tolist())
                # attempted fix below:
                partition_lengths = []

                for file_path in resolved_paths:
                    parquet_file = pq.ParquetFile(file_path)
                    metadata = parquet_file.metadata

                    for i in range(metadata.num_row_groups):
                        partition_lengths.append(metadata.row_group(i).num_rows)

                if len(partition_lengths) != n_partitions:
                    err_msg = (
                        f"Recovered {len(partition_lengths)} parquet partitions, "
                        f"but dask-awkward reports {n_partitions} partitions."
                    )
                    logger.error(err_msg)
                    raise RuntimeError(err_msg)

                self.array._divisions = tuple(np.cumsum([0] + partition_lengths).tolist())

            logger.debug(f"[{self.name}] divisions recovered: npartitions={self.array.npartitions}")

    # NOTE: Overridden: base Ingestor's padding-length methods cache by raw field
    # name, but the grouped pipeline has a differnet caching scheme. Rather than risk
    # the two key conventions mixing in the same self._padding_lengths dict,
    # both are overridden to fail loudly if called directly on a GroupedIngestor.

    def get_padding_length(self, field: str, method: Literal["dask", "pyarrow"] = "dask") -> int:
        raise NotImplementedError(
            "GroupedIngestor caches padding lengths by particle name, not raw field name. "
            "Use compute_padding_layout() (row-0-derived, then extrapolated) instead."
        )

    def compute_all_padding_lengths(self, fields: list[str]) -> dict[str, int]:
        raise NotImplementedError(
            "Use compute_padding_layout() instead — grouped padding is keyed by particle "
            "name and derived from feature_columns_grouped's row 0, not an arbitrary field list."
        )

    def compute_padding_layout(self) -> dict[str, Any]:
        """Per-particle-type padding length. Computation is still cheap — one
        reference field per particle (see _reference_field_for_particle) — but
        cache entries now record EVERY real field for that particle, plus the
        computed length, so the cache is self-describing:

            {particle_key: {"fields": [...], "length": int}, ..., "__total__": int}

        Downstream consumers (validate_padding_layout, a loaded cache) never
        again need to consult feature_columns_grouped/particle_names.

        A particle with NO real column anywhere (unsupported by convention) gets
        length 1 and an empty "fields" list, with a warning.
        """
        particle_keys = self._resolve_particle_keys()

        reference_fields = {}
        all_fields = {}
        for j, key in enumerate(particle_keys):
            ref_field = self._reference_field_for_particle(j)
            all_fields[key] = self._real_fields_for_particle(j)
            if ref_field is None:
                logger.warning(
                    f"[{self.name}] Particle '{key}' has no real column in any row; "
                    "defaulting padding length to 1."
                )
            reference_fields[key] = ref_field

        lengths = {key: 1 for key, field in reference_fields.items() if field is None}
        lazy_lengths = {}
        for key, field in reference_fields.items():
            if field is None:
                continue
            column = self._ensure_2d(self[field])
            lazy_lengths[key] = ak.max(ak.ravel(ak.num(column, axis=1)))

        if lazy_lengths:
            logger.debug(f"[{self.name}] Computing padding lengths for {len(lazy_lengths)} particle(s), batched...")
            try:
                computed = dask.compute(*lazy_lengths.values())
                lengths.update(dict(zip(lazy_lengths.keys(), (int(v) for v in computed))))
            except Exception as e:
                logger.warning(
                    f"[{self.name}] Batched dask padding computation failed ({e!r}); "
                    "falling back to pyarrow per-particle."
                )
                lengths.update({
                    key: brute_force_max_list_length(resolve_paths(self.paths), reference_fields[key])
                    for key in lazy_lengths.keys()
                })

        results = {key: {"fields": all_fields[key], "length": lengths[key]} for key in particle_keys}
        results["__total__"] = sum(lengths[key] for key in particle_keys)
        self._padding_lengths = results
        logger.info(f"[{self.name}] Padding layout: {results}")
        return results

    def validate_padding_layout(self, expected: dict[str, Any] | None = None) -> None:
        """Recompute padding length for EVERY real field listed in each cache
        entry — read from the cache itself, NOT re-derived from
        feature_columns_grouped/particle_names — and compare against the
        cached "length". Raises ValueError on the first mismatch. Opt-in —
        materially more expensive than compute_padding_layout (every real
        field per particle, not just one reference).
        """
        expected = expected if expected is not None else self._padding_lengths
        if not expected:
            err_msg = (
                "No padding layout to validate against; call compute_padding_layout() "
                "or set_padding_lengths() first."
            )
            logger.error(err_msg)
            raise RuntimeError(err_msg)

        for key, entry in expected.items():
            if key == "__total__":
                continue
            if "fields" not in entry:
                err_msg = (f"Padding cache entry '{key}' is missing required key 'fields'.")
                logger.error(err_msg)
                raise ValueError(err_msg)
            for field in entry["fields"]:
                actual = self._get_padding_length_dask(field)
                if actual != entry["length"]:
                    err_msg = (
                        f"[{self.name}] Padding length mismatch for particle '{key}', "
                        f"field '{field}': expected {entry['length']}, got {actual}."
                    )
                    logger.error(err_msg)
                    raise ValueError(err_msg)
        logger.info(f"[{self.name}] Padding layout validated across all cached particle(s): OK.")

    def _sentinel_target_for_field(self, field: str) -> float | None:
        """Whether `field` should be checked against general_sentinel_value at all —
        gated by replace_all_sentinel_values or explicit membership in
        replacements_sentinel_columns. The VALUE being detected is always
        general_sentinel_value (singular); the per-column dict only overrides
        the FILL value used on a match, not what's being matched against.
        """
        checked = self.replace_all_sentinel_values or (field in self.replacements_sentinel_columns)
        if not checked or self.general_sentinel_value is None:
            return None
        return self.general_sentinel_value

    def resolve_sentinels(self) -> dak.Array:
        """Replace real None entries (unconditionally, every real field)
        and, where applicable, sentinel-valued entries (e.g. -25.0) with NaN.

        Lazy — builds on self.array's existing graph, no compute triggered.
        Works identically for scalar or ragged fields (ak.where/ak.is_none
        broadcast into nested structures without special-casing).

        Returns:
            dak.Array: a NEW array (dak.Array is immutable — same convention
                as BaseScaler.apply/revert). Caller must reassign, e.g.
                `ingestor.array = ingestor.resolve_sentinels()`.
        """
        array = self.array
        for field in self.fields:
            column = NestedArrayIndexer.get_nested_field(array, field, self.SEPARATOR)
            column = ak.where(ak.is_none(column), np.nan, column)

            sentinel = self._sentinel_target_for_field(field)
            logger.debug(
                f"[{self.name}] resolve_sentinels: field='{field}' checked={sentinel is not None} "
                f"sentinel_value={sentinel}"
            )
            if sentinel is not None:
                fill = self.replacements_sentinel_columns.get(field)
                if fill is None:
                    fill = self.replacements_sentinel_default
                column = ak.where(column == sentinel, fill, column)

            array = ak.with_field(array, column, where=tuple(field.split(self.SEPARATOR)))

        return array

    def fill_missing_columns(self) -> dak.Array:
        """LAZILY synthesize every schema-missing column (declared in
        replacements_missing_columns) as a new field, applying the
        particle-wide-absence override — if every REAL sibling column at the
        same particle position (across other rows) is NaN for an event, the
        fixed fill is overridden to NaN there too, rather than fabricating a
        constant for an otherwise-absent particle.

        No compute triggered — builds on self.array's existing graph via
        ak.where/ak.with_field, same as resolve_sentinels(). Must run after
        resolve_sentinels() (siblings need to already be NaN-resolved for the
        absence check to be meaningful); scaler order doesn't affect the
        override itself (NaN survives arithmetic), this runs AFTER scaler 
        application (fill stays raw-space).

        No-op (returns self.array unchanged) if no columns are declared missing.

        Returns:
            dak.Array: a NEW array (immutable — caller must reassign, e.g.
                `ingestor.array = ingestor.fill_missing_columns()`).
        """
        if not self.missing_columns:
            return self.array

        array = self.array
        grid = self.feature_columns_grouped

        for r, row in enumerate(grid):
            for j, field in enumerate(row):
                if field not in self.missing_columns:
                    continue

                fill_value = self.replacements_missing_columns.get(field, self.replacements_default)
                sibling_fields = [
                    grid[r2][j] for r2 in range(len(grid))
                    if r2 != r and grid[r2][j] not in self.missing_columns
                ]
                logger.debug(
                    f"[{self.name}] fill_missing_columns: field='{field}' (row={r}, particle_pos={j}) "
                    f"fill_value={fill_value} sibling_fields={sibling_fields}"
                )
                if sibling_fields:
                    sibling_cols = [
                        NestedArrayIndexer.get_nested_field(array, sf, self.SEPARATOR)
                        for sf in sibling_fields
                    ]
                    all_nan = sibling_cols[0] != sibling_cols[0]
                    for sib in sibling_cols[1:]:
                        all_nan = all_nan & (sib != sib)
                    filled = ak.where(all_nan, np.nan, fill_value)
                else:
                    logger.warning(
                        f"[{self.name}] Particle at row {r}, position {j} ('{field}') has no "
                        "real column in any row — cannot determine presence; filling as an "
                        "always-present constant."
                    )
                    # shape/partitioning template only — condition is never actually
                    # used (both branches are fill_value), so this can't leak the
                    # reference field's own NaNs into the fabricated constant.
                    ref = NestedArrayIndexer.get_nested_field(array, self.fields[0], self.SEPARATOR)
                    filled = ak.where(ref == ref, fill_value, fill_value)

                array = ak.with_field(array, filled, where=tuple(field.split(self.SEPARATOR)))

        return array
    
    def _resolve_particle_keys(self) -> list[str]:
        """Cosmetic cache keys, one per particle: self.particle_names if unique,
        otherwise "particle_0", "particle_1", ... . Checked fresh here (not
        trusted from __init__, which only defaults None/empty — never checks
        a user-provided list for duplicates). Mirrors GroupedScaler's
        _resolve_row_keys; never used for lookup once cached, only for
        human-readable dumps.
        """
        names = self.particle_names
        if names and len(set(names)) == len(names):
            return list(names)
        return [f"particle_{i}" for i in range(len(self.feature_columns_grouped[0]))]

    def _real_fields_for_particle(self, j: int) -> list[str]:
        """Every REAL (non-missing) occurrence of particle j across all rows —
        unlike _reference_field_for_particle (first match only, used for the
        actual cheap computation), this collects all of them, for recording in
        the cache so validate_padding_layout() can later check every real field,
        not just the one that was computed against.
        """
        return [row[j] for row in self.feature_columns_grouped if row[j] not in self.missing_columns]

    def _reference_field_for_particle(self, j: int) -> str | None:
        """First row where particle j has a REAL (non-missing) column — used
        as the basis for padding-length computation. Raggedness/multiplicity
        is a property of the PARTICLE, not of any specific row, so any row
        with a real column for that particle works as the reference; row 0
        alone could itself declare that particle's column as schema-missing
        for some (equally valid) configs, which the old row-0-only logic
        didn't handle.

        Returns:
            str | None: the reference field name, or None if no row has a
                real column for this particle at all (unsupported/edge case
                — see fill_missing_columns' fallback branch).
        """
        for row in self.feature_columns_grouped:
            field = row[j]
            if field not in self.missing_columns:
                return field
        return None

    def _verify_padding_cache_against_ingestor(self, padding_lengths: dict[str, Any]) -> None:
        """Fail loudly if a (possibly loaded) padding cache's expected fields
        don't match this ingestor's real columns — stale/mismatched config, or
        a cache from a different dataset entirely. Mirrors
        GroupedScaler._verify_cache_against_ingestor.
        """
        available = set(self.fields)
        seen: set[str] = set()
        total_from_entries = 0

        for key, entry in padding_lengths.items():
            if key == "__total__":
                continue
            fields = set(entry.get("fields", []))
            missing = fields - available
            if missing:
                err_msg = (
                    f"Padding cache entry '{key}' expects field(s) {missing}, not present in "
                    "the current ingestor. The cache may have been computed against a "
                    "different config or dataset."
                )
                logger.error(err_msg)
                raise ValueError(err_msg)
            overlap = fields & seen
            if overlap:
                err_msg = (
                    f"Padding cache entry '{key}' re-declares field(s) {overlap}, already "
                    "claimed by another cache entry. The cache is inconsistent."
                )
                logger.error(err_msg)
                raise ValueError(err_msg)
            seen |= fields
            if "length" not in entry:
                err_msg = (f"Padding cache entry '{key}' is missing required key 'length'.")
                logger.error(err_msg)
                raise ValueError(err_msg)
            total_from_entries += entry["length"]

        if "__total__" in padding_lengths and padding_lengths["__total__"] != total_from_entries:
            err_msg = (
                f"Padding cache '__total__'={padding_lengths['__total__']} does not match the "
                f"sum of per-particle lengths ({total_from_entries}). The cache is inconsistent."
            )
            logger.error(err_msg)
            raise ValueError(err_msg)
        logger.debug(f"[{self.name}] Padding cache verified OK against ingestor.")

    def set_padding_lengths(self, padding_lengths: dict[str, Any]) -> None:
        """Pre-seed the padding-length cache from externally-provided values
        (e.g. loaded from JSON). Verifies the cache's self-described fields
        against this ingestor's real columns before accepting it.
        """
        self._verify_padding_cache_against_ingestor(padding_lengths)
        self._padding_lengths = dict(padding_lengths)