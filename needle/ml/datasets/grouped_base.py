"""
Base class for grouped (particles, features) Datasets. Mirrors PaddedDatasetBase,
reusing its ABC scaffolding and flat (labels/weights) conversion unchanged; only
the ragged-feature conversion differs, replaced by convert_grouped_ak_to_tensor.
"""

import awkward as ak
import numpy as np
import torch

from needle.etl.array import NestedArrayIndexer, is_ragged
from needle.etl.dask_grouped_ingestor import GroupedIngestor
from needle.ml.datasets.padded_base import PaddedDatasetBase
from needle.utils.logging import ColorFormatter

logger = ColorFormatter.get_logger("ml")


class GroupedDatasetBase(PaddedDatasetBase):
    """
    Base class for Datasets producing a dense (E, particle_max, n_features) tensor
    from grouped particle-feature columns. Subclasses provide __init__/__iter__
    (torch- or dask-multithreaded), same split as PaddedTorchDataset/PaddedDaskDataset.
    """

    features_ingestor: GroupedIngestor

    def convert_grouped_ak_to_tensor(self, array: ak.Array, ingestor: GroupedIngestor) -> torch.Tensor:
        """Step 5 only: gather each row's particle columns, concat+pad+stack.

        Important:
            `array` must already have had sentinel/None resolution (step 2),
            scaler application (step 3), AND missing-column fill (step 4)
            applied LAZILY upstream, on `ingestor.array`, before this
            partition was computed — see GroupedIngestor.resolve_sentinels()
            and GroupedIngestor.fill_missing_columns(). By this point every
            field named anywhere in feature_columns_grouped is a real,
            directly-gettable field; this method does no branching on
            missing/real columns at all.

            `ingestor` must have a cached padding layout (compute_padding_layout()
            or set_padding_lengths()) before this runs.

        Returns:
            torch.Tensor: shape (E, P=particle_max, F=n_features).
        """
        layout = ingestor.get_all_padding_lengths()
        if "__total__" not in layout:
            raise RuntimeError(
                "GroupedIngestor has no cached padding layout — call compute_padding_layout() "
                "or set_padding_lengths() before iterating."
            )
        target_total = layout["__total__"]
        logger.debug(f"[GroupedDatasetBase] target_total={target_total}")

        # replaced the concat_and_pad_row operation with a dask-based alternative, should be much faster
        # row_tensors = []
        # for row in ingestor.feature_columns_grouped:
        #     columns = [NestedArrayIndexer.get_nested_field(array, field, ingestor.SEPARATOR) for field in row]
        #     padded_row = concat_and_pad_row(columns, target_total)
        #     row_tensors.append(padded_row[..., np.newaxis])
        # events = ak.concatenate(row_tensors, axis=-1)
        # tensor = torch.tensor(ak.to_numpy(events), dtype=torch.float32)

        padded = ingestor.build_grouped_tensor(array, target_total)
        dense = np.stack(
            [ak.to_numpy(ak.fill_none(padded[name], np.nan)) for name in ingestor.feature_names],
            axis=-1,
        )
        tensor = torch.tensor(dense, dtype=torch.float32)

        if not getattr(self, "_grouped_debug_done", False):
            _nan = torch.isnan(tensor)
            logger.debug(
                f"[GroupedDatasetBase] first-partition tensor shape={tuple(tensor.shape)} "
                f"any_nan={_nan.any().item()} nan_fraction={_nan.float().mean().item():.4f}"
            )
            self._grouped_debug_done = True
        return tensor

    def extract_aux_fields(
        self,
        array: ak.Array,
        ingestor: GroupedIngestor,
        fields: list[str],
    ) -> dict[str, torch.Tensor]:
        """Validation/test-only counterpart to convert_grouped_ak_to_tensor: 
        extracts specific real columns from the same partition array, padding each 
        to its own particle-group length instead of concatenating groups into
        target_total (which makes per-group boundaries event-dependent and
        ambiguous to invert). Returns SCALED values — reverting normalization
        is the caller's job.
 
        Important:
            `array` must be the exact same eager, already-prepped partition
            object passed to convert_grouped_ak_to_tensor for this iteration
            step. This method never reads ingestor.array itself — deliberately,
            to avoid any risk of it reflecting a different point in the lazy,
            repeatedly-rebound resolve_sentinels/apply_with_cache/
            fill_missing_columns chain than the training tensor was built from.
 
        Args:
            array (ak.Array): already-computed partition (same object handed
                to convert_grouped_ak_to_tensor for this partition).
            ingestor (GroupedIngestor): source of feature_columns_grouped/
                particle layout and the cached per-particle padding lengths.
            fields (list[str]): raw column names to extract. Every field must
                be a REAL column (ingestor.fields) — schema-missing/filled
                columns were never scaled and have no meaningful value to
                extract or revert.
 
        Returns:
            dict[str, torch.Tensor]: one entry per requested field, each of
                shape (E, length_j), length_j being that field's own
                particle-group's padding length (NOT target_total).
        """
        if not fields:
            return {}
 
        unknown = [f for f in fields if f not in ingestor.fields]
        if unknown:
            raise ValueError(
                f"[GroupedDatasetBase] aux_feature_fields {unknown} are not real "
                f"columns (ingestor.fields); schema-missing/filled columns have no "
                "meaningful scaled value to extract or revert."
            )
 
        layout = ingestor.get_all_padding_lengths()
        if "__total__" not in layout:
            raise RuntimeError(
                "GroupedIngestor has no cached padding layout — call "
                "compute_padding_layout() or set_padding_lengths() before iterating."
            )
 
        particle_keys = ingestor._resolve_particle_keys()
        field_to_particle_pos = {
            col: j for row in ingestor.feature_columns_grouped for j, col in enumerate(row)
        }
 
        aux_tensors: dict[str, torch.Tensor] = {}
        for field in fields:
            particle_key = particle_keys[field_to_particle_pos[field]]
            length_j = layout[particle_key]["length"]
 
            column = NestedArrayIndexer.get_nested_field(array, field, ingestor.SEPARATOR)
            if not is_ragged(column):
                column = ak.singletons(column, axis=0)
            padded = ak.pad_none(column, axis=1, target=length_j, clip=True)
            dense = ak.to_numpy(ak.fill_none(padded, np.nan))
            aux_tensors[field] = torch.tensor(dense, dtype=torch.float32)
 
        logger.debug(
            f"[GroupedDatasetBase] extract_aux_fields: fields={list(aux_tensors.keys())} "
            f"shapes={[tuple(t.shape) for t in aux_tensors.values()]}"
        )
        return aux_tensors


    def convert_ragged_ak_to_tensor(self, *args, **kwargs):
        raise NotImplementedError(
            "GroupedDatasetBase uses convert_grouped_ak_to_tensor instead — "
            "the ragged/flat pipeline's per-field padding concept doesn't apply here."
        )

    def get_padding_length(self, field: str = None) -> int:
        raise NotImplementedError(
            "Grouped padding is a single total (all particle-types combined), not "
            "per-field. Use self.features_ingestor.get_all_padding_lengths()['__total__']."
        )


    def _verify_array_prepared(self, ingestor: GroupedIngestor) -> None:
        """Fail fast if ingestor.array is missing any column referenced in
        feature_columns_grouped — i.e., resolve_sentinels()/scaler.apply()/
        fill_missing_columns() haven't all been run (and reassigned to
        ingestor.array) yet. Call this once, e.g. in a Dataset subclass's
        __init__, rather than letting a missing field surface as a confusing
        FieldNotFoundError deep inside convert_grouped_ak_to_tensor on the
        first partition.
        """
        available = set(
            NestedArrayIndexer.list_all_fields(ingestor.array, separator=ingestor.SEPARATOR, as_tuple=False)
        )
        required = {field for row in ingestor.feature_columns_grouped for field in row}
        missing = required - available
        if missing:
            raise RuntimeError(
                f"ingestor.array is missing field(s) {missing} required by feature_columns_grouped. "
                "Did you forget to call resolve_sentinels()/scaler.apply()/fill_missing_columns() "
                "and reassign ingestor.array before constructing this Dataset?"
            )
        logger.debug(f"[GroupedDatasetBase] _verify_array_prepared: available={len(available)} required={len(required)}")
