"""
Base class for grouped (particles, features) Datasets. Mirrors PaddedDatasetBase,
reusing its ABC scaffolding and flat (labels/weights) conversion unchanged; only
the ragged-feature conversion differs, replaced by convert_grouped_ak_to_tensor.
"""

import awkward as ak
import numpy as np
import torch

from needle.etl.array import NestedArrayIndexer
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
