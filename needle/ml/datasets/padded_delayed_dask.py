import numpy as np
from typing import Literal
from torch.utils.data import IterableDataset

from needle.etl.dask_ingestor import Ingestor
from needle.ml.datasets.io import load_partition
from needle.ml.datasets.kfold import KFold
from needle.ml.datasets.padded_base import PaddedDatasetBase


class PaddedDaskDataset(IterableDataset, PaddedDatasetBase):
    """Out-of-memory dataset using dask multithreading and single torch worker."""

    SHUFFLE_ALLOWED: bool = False
    TORCH_MULTIPROCESSING_ALLOWED: bool = False

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
        # TODO - update docstring!
        """This class extends the PaddedDatasetBase to support chunked iterating with Dask.

        Args:
            features (dak.Array): Dask_awkward array containing feature columns
            labels (dak.Array): Dask_awkward array containing label columns
            shuffle_partitions (bool): Whether to shuffle partition order
            shuffle_events (bool): Whether to shuffle events within each partition
            random_seed (int): Random seed for reproducibility
            kfold (KFold, optional): Instance of the KFold class

        Note:
            This dataset is designed for single-worker use (num_workers=0) to avoid
            multiprocessing complexity. For multi-worker processing, use PaddedTorchDataset.
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

        # moved the explicit computation force (call) to datamodule level, into setup() method
        # self._compute_padding_lengths(self.feature_names)

        # temp addition of a debug/dump
        # padding_lengths_path = '/data/dust/user/nkovacic/NEEDLE/NEEDLE_DATA/fair_universe_data_merged_customized/padding_info_HTT_train.json'
        # if padding_lengths_path is not None:
        #     self.dump_padding_lengths(padding_lengths_path)

    def __iter__(self):
        """Iterate through partitions sequentially

        Yields:
            tuple: A tuple (features, labels, weights) of torch Tensors with shape (P, F)
        """
        # note: this implementation of rng won't always be completely consistent 
        # as the result of the call can depend on the hardware (so, different results
        # could be observed when running on different cluster nodes, even with same seed)
        rng = np.random.default_rng(self.random_seed)

        if self.kfold:
            if self.shuffle_partitions:
                self.kfold.shuffle(rng)
            partitions = self.kfold.partitions
        else:
            partition_ids = list(range(self.features_ingestor.array.npartitions))
            if self.shuffle_partitions:
                partition_ids = rng.permutation(partition_ids)
            partitions = {p: None for p in partition_ids}

        for partition_id, slicing_index in partitions.items():
            features_partition = load_partition(self.features_ingestor.array, partition_id, slicing_index)
            labels_partition = load_partition(self.labels_ingestor.array, partition_id, slicing_index)
            weights_partition = load_partition(self.weights_ingestor.array, partition_id, slicing_index)

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

            event_indices = list(range(len(features_tensor)))

            if self.shuffle_events:
                event_indices = rng.permutation(event_indices)

            for event_idx in event_indices:
                yield (features_tensor[event_idx], labels_tensor[event_idx], weights_tensor[event_idx])
