"""
Base class for padded Datasets. Provided methods for other datasets to inherit from.

TODO Padding happens for each column independently, which improves memory usage. It might
not be finished yet for training and evaluation datasets, as they should have the same
padding.
"""

import logging
import warnings
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
        self._compute_padding_lengths(self.feature_names)

    def convert_ragged_to_tensor(self, array: ak.Array, fields: list[str], ingestor: Ingestor) -> torch.Tensor:
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
        events = ak.concatenate(event_list, axis=-1)
        return torch.tensor(ak.to_numpy(events), dtype=torch.float32)

    def convert_flat_to_tensor(
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
        elif reduce == "stack":
            combined = np.stack(columns, axis=-1)
        elif reduce == "product":
            combined = np.prod(np.stack(columns, axis=-1), axis=-1)
        elif reduce == "sum":
            combined = np.sum(np.stack(columns, axis=-1), axis=-1)
        else:
            raise ValueError(f"Unknown reduce mode: {reduce}")
        return torch.tensor(combined, dtype=torch.float32)

    def _compute_padding_lengths(self, fields: list[str]) -> None:
    # TODO - is this actually sane? i.e. I didn't see NEEDLE pre-computng the 
    # padding scheme, but that is very much surprising, can it be that I missed it???
    """Precompute and cache the global padding length for each ragged field.

    Important:
        Must be called once, in `__init__`, using the full un-computed
        dask_awkward array (`self.features_ingestor.array`) — not a
        partition-level slice. Because the result is cached per field and
        reused across every partition and every worker, computing it from
        a single partition would silently apply the wrong padding length
        to all other partitions (risking truncation of real particles via
        `clip=True`), and independently-computed values across torch
        DataLoader workers could disagree entirely, producing inconsistent
        tensor shapes across a batch.
    """
        if not hasattr(self, "_padding_lengths"):
            self._padding_lengths = {}

        for field in fields:
            if field in self._padding_lengths:
                continue

            column = NestedArrayIndexer.get_nested_field(
                self.features_ingestor.array, field, self.features_ingestor.SEPARATOR
            )
            column = self.add_innermost_dimension(column)

            def _get_length() -> int:
                return int(ak.max(ak.ravel(ak.num(column, axis=1))))

            if logger.isEnabledFor(logging.DEBUG):
                length = _get_length()
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    length = _get_length()

            self._padding_lengths[field] = length