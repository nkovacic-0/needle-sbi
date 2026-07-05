"""
Base class for padded Datasets. Provided methods for other datasets to inherit from.

TODO Padding happens for each column independently, which improves memory usage. It might
not be finished yet for training and evaluation datasets, as they should have the same
padding.
"""

import logging
import warnings
from pathlib import Path
from typing import Literal
from abc import ABC, abstractmethod

import awkward as ak
import numpy as np
import torch
from torch.utils.data import Dataset

from needle.etl.array import NestedArrayIndexer
from needle.etl.dask_ingestor import Ingestor
from needle.utils.logging import ColorFormatter

logger = ColorFormatter.get_logger("ml")

class PaddedDatasetBase(Dataset, ABC):
    """
    Base class for Datasets handling ragged data from dask_awkward Record arrays.
    """

    SHUFFLE_ALLOWED: bool
    TORCH_MULTIPROCESSING_ALLOWED: bool

    features_ingestor: Ingestor
    labels_ingestor: Ingestor
    weights_ingestor: Ingestor

    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    def __len__(self) -> int:
        return self.features_ingestor.length

    def convert_ragged_ak_to_tensor(self, array: ak.Array, fields: list[str], ingestor: Ingestor) -> torch.Tensor:
        # TODO - check that the docstring explains what is going on (I'm continuosly making code updates, but not docstring updates)
        """For per-particle (jagged) fields. Pads axis=1 to the precomputed global
        max length. Shape: (E, P, F).

        Note:
            `array` must already be computed (eager), e.g. via `partition.compute()`.
            Padding target comes from `get_padding_length`, which reflects the max
            over the *entire* dataset, not just this partition/chunk.
        """
        event_list = []
        for field in fields:
            column = NestedArrayIndexer.get_nested_field(array, field, ingestor.SEPARATOR)
            column = self.add_innermost_dimension(column)
            padded = ak.pad_none(column, axis=1, target=self.get_padding_length(field), clip=True)
            event_list.append(padded[..., np.newaxis])

            # ============ TEMP DEBUG BODGE — gotta check the actual state of padding TODO - remove after testing!
            if not getattr(self, "_padding_debug_done", False):
                sample_list = ak.to_list(padded[:3])
                logger.warning(f"[PADDING DEBUG] field='{field}' first 3 events (raw, pre-numpy): {sample_list}")

                numeric = ak.to_numpy(padded, allow_missing=True)
                logger.warning(f"[PADDING DEBUG] field='{field}' ak.to_numpy() result type: {type(numeric)}")
                if hasattr(numeric, "fill_value"):
                    logger.warning(f"[PADDING DEBUG] field='{field}' MaskedArray.fill_value: {numeric.fill_value}")
                    logger.warning(f"[PADDING DEBUG] field='{field}' MaskedArray.data sample (first event): {numeric.data[0]}")
                    logger.warning(f"[PADDING DEBUG] field='{field}' MaskedArray.mask sample (first event): {numeric.mask[0] if numeric.mask is not np.ma.nomask else 'no mask (nomask)'}")

                test_tensor = torch.tensor(np.asarray(numeric), dtype=torch.float32)
                is_nan = torch.isnan(test_tensor)
                logger.warning(
                    f"[PADDING DEBUG] field='{field}' resulting tensor: "
                    f"any NaN present = {is_nan.any().item()}, "
                    f"NaN count = {is_nan.sum().item()} / {test_tensor.numel()}, "
                    f"tensor sample (first event) = {test_tensor[0]}"
                )

        if not getattr(self, "_padding_debug_done", False):
            overall_nan_check = torch.isnan(tensor)
            logger.warning(
                f"[PADDING DEBUG] FINAL combined tensor shape={tuple(tensor.shape)}, "
                f"any NaN = {overall_nan_check.any().item()}, "
                f"NaN fraction = {overall_nan_check.float().mean().item():.4f}"
            )
            self._padding_debug_done = True  # only spam this once per Dataset instance
        # ============ END TEMP DEBUG BODGE ============


        events = ak.concatenate(event_list, axis=-1)
        return torch.tensor(ak.to_numpy(events), dtype=torch.float32)

    def convert_flat_ak_to_tensor(
        self, array: ak.Array, fields: list[str], ingestor: Ingestor, 
        reduce: Literal["stack", "product", "sum"] = "stack",
    ) -> torch.Tensor:
        # TODO - needs its doctring checked, updated!
        """For per-event (regular, non-ragged) fields like labels/weights.

        Returns shape (E,) for a single field or a reduced ("product"/"sum") combination
        of multiple fields; shape (E, F) for "stack" with F > 1 fields.
        """
        columns = [
            ak.to_numpy(NestedArrayIndexer.get_nested_field(array, f, ingestor.SEPARATOR))
            for f in fields
        ]
        if len(columns) == 1:
            combined = columns[0]
        # stach is intended for labels, however, this will BREAK if anything other than 1-D fields are passed to labels
        # so, F colums/fields each of dim 1, is fine and yields (E, F)
        # TODO - see if we want to expand this functionality, or revert to padded labels?
        elif reduce == "stack":
            combined = np.stack(columns, axis=-1)
        elif reduce == "product":
            combined = np.prod(np.stack(columns, axis=-1), axis=-1)
        elif reduce == "sum":
            combined = np.sum(np.stack(columns, axis=-1), axis=-1)
        else:
            raise ValueError(f"Unknown reduce mode: {reduce}")
        # if combined.ndim == 2:
        #     assert combined.shape[-1] == len(fields), (
        #         f"Stacked label tensor has {combined.shape[-1]} columns but {len(fields)} fields were requested:"
        #         f" {fields}. "
        #         f"This would misalign downstream `collate_fn` label naming."
        #     )
        return torch.tensor(combined, dtype=torch.float32)

    def get_padding_length(self, field: str) -> int:
        return self.features_ingestor.get_padding_length(field)


    @staticmethod
    def add_innermost_dimension(array: ak.Array) -> ak.Array:
        """Potentially add an inner dimension if the array is not 2D.

        Works by trying to access the first element at axis 1. If that element
        does not exist, add an inner dimension at axis 0 using ak.singletons().

        Args:
            array: ak.Array: The input data. If of shape (E, 1), nothing happens
                and the original array is returned. If of shape (E,), an inner
                dimension is added and the array is reshaped to (E, 1).

        Returns:
            ak.Array: The same array now shaped 2D (E, 1).

        Example:
            Case where the input array is 1D and needs an inner dimension added:
            >>> print(PaddedEagerDataset.add_innermost_dimension([11, 52, 31]))
            ak.Array([[11], [52], [31]])

            Case where the input array is already 2D and nothing needs to be done:
            >>> print(PaddedEagerDataset.add_innermost_dimension([[11], [52], [31]]))
            ak.Array([[11], [52], [31]])
        """
        try:
            _ = array[0][0]
        except IndexError:
            logger.debug("Added an extra dimension to the array at depth 0.")
            array = ak.singletons(array, axis=0)
        finally:
            return array

    

    # NOTE: this is dead code, which delegates/esposes the intestor padding calculation at this class level
    # it was used by _compute_padding_lengths calls in padded dask/torch's inits, but that cal forcing has 
    # now been moved to padded_datamodule
    # def _compute_padding_lengths(self, fields: list[str]) -> None:
    #     """Trigger (and cache, via Ingestor) padding-length computation for ragged fields.
    #     Delegates to `self.features_ingestor.get_padding_length`, which owns the actual
    #     caching — shared across any Dataset instances built from the same Ingestor
    #     (e.g. train/val), so this only pays the real computation cost once per run.
    #     """
    #     for field in fields:
    #         self.features_ingestor.get_padding_length(field)

    # def _compute_padding_lengths(self, fields: list[str]) -> None:
    # # TODO - is this actually sane? i.e. I didn't see NEEDLE pre-computng the 
    # # padding scheme, but that is very much surprising, can it be that I missed it???
    # # -- seems like there realy was not pre-computed padding scheme... 
    # # this functions is now deprecated in favor of the new function above...
    # """Precompute and cache the global padding length for each ragged field.

    # Important:
    #     Must be called once, in `__init__`, using the full un-computed
    #     dask_awkward array (`self.features_ingestor.array`) — not a
    #     partition-level slice. Because the result is cached per field and
    #     reused across every partition and every worker, computing it from
    #     a single partition would silently apply the wrong padding length
    #     to all other partitions (risking truncation of real particles via
    #     `clip=True`), and independently-computed values across torch
    #     DataLoader workers could disagree entirely, producing inconsistent
    #     tensor shapes across a batch.
    # """
    #     if not hasattr(self, "_padding_lengths"):
    #         self._padding_lengths = {}
    #     for field in fields:
    #         if field in self._padding_lengths:
    #             continue
    #         column = NestedArrayIndexer.get_nested_field(
    #             self.features_ingestor.array, field, self.features_ingestor.SEPARATOR
    #         )
    #         column = self.add_innermost_dimension(column)
    #         def _get_length() -> int:
    #             return int(ak.max(ak.ravel(ak.num(column, axis=1))))
    #         if logger.isEnabledFor(logging.DEBUG):
    #             length = _get_length()
    #         else:
    #             with warnings.catch_warnings():
    #                 warnings.simplefilter("ignore")
    #                 length = _get_length()
    #         self._padding_lengths[field] = length