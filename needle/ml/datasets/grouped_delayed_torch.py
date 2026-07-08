"""
Out-of-memory grouped-feature Dataset using torch multiprocessing and dask
single-thread read-in. Mirrors PaddedTorchDataset — same PartitionQueue/
worker-splitting/kfold logic, reusing io.py unchanged — swapping only the
feature conversion call.
"""

import numpy as np
import torch
from typing import Literal
from torch.utils.data import IterableDataset

from needle.etl.dask_ingestor import Ingestor
from needle.etl.dask_grouped_ingestor import GroupedIngestor
from needle.ml.datasets.io import PartitionQueue
from needle.ml.datasets.kfold import KFold
from needle.ml.datasets.grouped_base import GroupedDatasetBase
from needle.utils.logging import ColorFormatter

logger = ColorFormatter.get_logger("ml")


class GroupedTorchDataset(IterableDataset, GroupedDatasetBase):
    """Out-of-memory grouped (particles, features) Dataset compatible with
    torch DataLoader multi-worker use (num_workers > 0)."""

    SHUFFLE_ALLOWED: bool = False
    TORCH_MULTIPROCESSING_ALLOWED: bool = True

    def __init__(
        self,
        features: GroupedIngestor,
        labels: Ingestor,
        weights: Ingestor,
        shuffle_partitions: bool = False,
        shuffle_events: bool = True,
        random_seed: int = 42,
        kfold: KFold | None = None,
        weights_combine: Literal["product", "sum"] = "product",
    ):
        """
        Args:
            features (GroupedIngestor): Must already have resolve_sentinels(),
                scaler.apply(), and fill_missing_columns() applied and
                reassigned to features.array, AND a cached padding layout
                (compute_padding_layout() or set_padding_lengths()) — see
                GroupedDatasetBase._verify_array_prepared, called below.
            labels (Ingestor): Plain Ingestor — labels are not grouped/padded.
            weights (Ingestor): Plain Ingestor — weights are not grouped/padded.
            shuffle_partitions (bool): Whether to shuffle partition order.
            shuffle_events (bool): Whether to shuffle events within each partition.
            random_seed (int): Random seed for reproducibility.
            kfold (KFold, optional): Instance of the KFold class.
            weights_combine (Literal["product", "sum"]): Combination mode when
                multiple weight columns are configured.

        Important:
            This dataset cannot be shuffled by the Dataloader, as chunks are
            loaded in sequence and shuffling would invalidate the chunking.
            Shuffling is configured via shuffle_partitions/shuffle_events instead.
        """
        self._verify_array_prepared(features)

        self.features_ingestor = features
        self.labels_ingestor = labels
        self.weights_ingestor = weights

        self.shuffle_partitions = shuffle_partitions
        self.shuffle_events = shuffle_events
        self.random_seed = random_seed
        self.kfold = kfold

        self.labels_names = labels.fields
        self.weights_names = weights.fields
        self.weights_combine = weights_combine

        self.features_queue = PartitionQueue(features.array)
        self.labels_queue = PartitionQueue(labels.array)
        self.weights_queue = PartitionQueue(weights.array)

        if features.array.npartitions == 1:
            logger.warning(
                f"[{features.name}] features array has only 1 partition — no real "
                "partition-level chunking will occur; each 'partition load' will "
                "materialize the ENTIRE dataset into memory at once. Consider "
                "dak_reader_kwargs={'split_row_groups': True} (or equivalent) to "
                "split the underlying parquet file(s) into multiple dask partitions."
            )

    def __iter__(self):
        """Yields individual events, loaded in per-partition chunks.

        If the Dataloader uses multiple workers, each worker processes a
        different set of partitions (torch.utils.data.get_worker_info()).

        Yields:
            tuple: (features, labels, weights) of torch Tensors. features has
                shape (particle_max, n_features); labels/weights are unchanged
                from the flat/ragged pipeline (see convert_flat_ak_to_tensor).
        """
        rng = np.random.default_rng(self.random_seed)
        self.worker_info = torch.utils.data.get_worker_info()

        if self.kfold:
            if self.worker_info:
                assigned_partition_mask = [
                    p % self.worker_info.num_workers == self.worker_info.id for p in self.kfold.partition_ids
                ]
                self.kfold.mask(assigned_partition_mask)
            if self.shuffle_partitions:
                self.kfold.shuffle(rng)
            partitions = self.kfold.partitions
        else:
            partition_ids = list(range(self.features_ingestor.array.npartitions))
            if self.worker_info:
                partition_ids = [p for p in partition_ids if p % self.worker_info.num_workers == self.worker_info.id]
            if self.shuffle_partitions:
                partition_ids = rng.permutation(partition_ids)
            partitions = {p: None for p in partition_ids}
        
        logger.debug(f"partitions assigned this worker: {list(partitions.keys())}")

        for partition_id, slicing_index in partitions.items():
            features_partition = self.features_queue.load_partition_thread_safe(partition_id, slicing_index)
            labels_partition = self.labels_queue.load_partition_thread_safe(partition_id, slicing_index)
            weights_partition = self.weights_queue.load_partition_thread_safe(partition_id, slicing_index)

            features_tensor = self.convert_grouped_ak_to_tensor(
                features_partition.compute(),
                self.features_ingestor,
            )
            labels_tensor = self.convert_flat_ak_to_tensor(
                labels_partition.compute(),
                self.labels_names,
                self.labels_ingestor,
                reduce="stack",
            )
            weights_tensor = self.convert_flat_ak_to_tensor(
                weights_partition.compute(),
                self.weights_names,
                self.weights_ingestor,
                reduce=self.weights_combine,
            )

            logger.debug(f"partition {partition_id}: features_tensor.shape={tuple(features_tensor.shape)}")

            if labels_tensor.ndim == 2 and labels_tensor.shape[-1] == 1:
                labels_tensor = labels_tensor.squeeze(-1)

            event_indices = range(len(features_tensor))

            if self.shuffle_events:
                event_indices = rng.permutation(event_indices)

            for event_idx in event_indices:
                yield (features_tensor[event_idx], labels_tensor[event_idx], weights_tensor[event_idx])