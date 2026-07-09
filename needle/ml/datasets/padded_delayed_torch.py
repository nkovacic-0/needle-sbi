import numpy as np
import torch
from typing import Literal
from torch.utils.data import IterableDataset

from needle.etl.dask_ingestor import Ingestor
from needle.ml.datasets.io import PartitionQueue
from needle.ml.datasets.kfold import KFold
from needle.ml.datasets.padded_base import PaddedDatasetBase


class PaddedTorchDataset(IterableDataset, PaddedDatasetBase):
    """Out-of-memory dataset using torch multiprocessing and dask single-thread read-in."""

    SHUFFLE_ALLOWED: bool = False
    TORCH_MULTIPROCESSING_ALLOWED: bool = True

    def __init__(
        self,
        features: Ingestor,
        labels: Ingestor,
        weights: Ingestor,
        shuffle_partitions: bool = False,
        shuffle_events: bool = True,
        random_seed: int = 42,
        kfold: KFold | None = None,
        weights_combine: Literal["product", "sum"] = "product",
    ):
        """Out-of-memory Dataset supporting multi-worker chunked iteration with torch.

        Data is loaded lazily, partition by partition, inside __iter__ -- not during
        __init__. Each partition is read via a PartitionQueue (io.py), which serializes
        reads with a lock to stay safe across forked DataLoader worker processes, then
        converted to tensors and yielded one event at a time. This is compatible with
        multiple workers when the Dataloader uses num_workers > 0.

        Args:
            features (Ingestor): Ingestor class instance holding the feature columns.
            labels (Ingestor): Ingestor class instance holding the label columns.
            weights (Ingestor): Ingestor class instance holding the weight columns.
            shuffle_partitions (bool): Whether to shuffle partition order before iterating.
            shuffle_events (bool): Whether to shuffle events within each partition.
            random_seed (int): Random seed for reproducibility.
            kfold (KFold, optional): Instance of the KFold class.
            weights_combine (Literal["product", "sum"]): Combination mode when multiple
                weight columns are configured.

        Important:
            This dataset cannot be shuffled by the Dataloader, as partitions are loaded
            in sequence and shuffling it would invalidate the partitioning. Instead,
            shuffling is configured via the `shuffle_partitions` and `shuffle_events`
            arguments.
        """
        self.features_ingestor = features
        self.labels_ingestor = labels
        self.weights_ingestor = weights

        self.shuffle_partitions = shuffle_partitions
        self.shuffle_events = shuffle_events
        self.random_seed = random_seed
        self.kfold = kfold

        self.feature_names = features.fields
        self.labels_names = labels.fields
        self.weights_names = weights.fields
        self.weights_combine = weights_combine

        self.features_queue = PartitionQueue(features.array)
        self.labels_queue = PartitionQueue(labels.array)
        self.weights_queue = PartitionQueue(weights.array)

    def __iter__(self):
        """Yields individual events, loaded in per-partition chunks.

        If the Dataloader uses multiple workers, each worker processes a
        different set of partitions (torch.utils.data.get_worker_info()).

        Yields:
            tuple: (features, labels, weights) of torch Tensors. features has
                shape (P, F); labels/weights follow convert_flat_ak_to_tensor.

        Note:
            Both partition order and events within a partition can optionally be
            shuffled before yielding; during regular training it's recommended to
            set both `shuffle_partitions` and `shuffle_events` to True.

            The splitting of the data works as follows:
            - Partitions are assigned to workers based on worker ID (partition_id
                % num_workers == worker_id). The order of assigned partitions can
                be shuffled beforehand via `shuffle_partitions`.
            - Each partition is loaded into memory and converted to tensors via
                `convert_ragged_ak_to_tensor`/`convert_flat_ak_to_tensor`, which pad
                the ragged feature fields to the global max particle count.
            - Events within the partition are then optionally shuffled by
                permuting their indices, and yielded one at a time as
                (features, labels, weights).
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

        for partition_id, slicing_index in partitions.items():
            features_partition = self.features_queue.load_partition_thread_safe(partition_id, slicing_index)
            labels_partition = self.labels_queue.load_partition_thread_safe(partition_id, slicing_index)
            weights_partition = self.weights_queue.load_partition_thread_safe(partition_id, slicing_index)

            features_tensor = self.convert_ragged_ak_to_tensor(
                features_partition.compute(), 
                self.feature_names, 
                self.features_ingestor
            )
            labels_tensor = self.convert_flat_ak_to_tensor(
                labels_partition.compute(), 
                self.labels_names, 
                self.labels_ingestor, reduce="stack"
            )
            weights_tensor = self.convert_flat_ak_to_tensor(
                weights_partition.compute(), 
                self.weights_names, 
                self.weights_ingestor, 
                reduce=self.weights_combine
            )

            if labels_tensor.ndim == 2 and labels_tensor.shape[-1] == 1:
                labels_tensor = labels_tensor.squeeze(-1)

            event_indices = range(len(features_tensor))

            if self.shuffle_events:
                event_indices = rng.permutation(event_indices)

            for event_idx in event_indices:
                yield (features_tensor[event_idx], labels_tensor[event_idx], weights_tensor[event_idx])
