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
        # TODO - update the docstring!
        """This class extends the PaddedEagerDataset to support chunked iterating with Torch.

        In the parent class, the data is loaded during instantiation. Instead, this class focusses
        on combining the __iter__ method of IterableDataset with the PaddedEagerDataset's functionality to
        yield chunks of data. The __iter__ method still yields individual events, same as the regular
        __getitem__ method of the parent class, but loads only discrete chunks of data to memory.
        This is compatible with multiple workers if the Dataloader is using num_workers > 0.

        Args:
            features (dict[str, dak.Array]): Dictionary of dask_awkward arrays for features
            labels (dict[str, dak.Array]): Dictionary of dask_awkward arrays for labels
            chunk_size (int): Number of events per chunk to load into memory at once.
            shuffle_chunks (bool): Whether to shuffle the chunks before yielding.
            shuffle_events (bool): Whether to shuffle the events within each chunk.
            random_seed (int): Random seed for reproducibility.
            kfold (KFold): Instance of the KFold class

        Important:
            This dataset cannot be shuffled by the Dataloader, as the chunks are loaded in sequence,
            and shuffling it would invalidate the chunking. Instead, the shuffling is configured by
            the arguments `shuffle_chunks` and `shuffle_events`.
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

        # moved the explicit computation force (call) to datamodule level, into setup() method
        # self._compute_padding_lengths(self.feature_names)

        # temp addition of a debug/dump
        # padding_lengths_path = '/data/dust/user/nkovacic/NEEDLE/NEEDLE_DATA/fair_universe_data_merged_customized/padding_info_HTT_train.json'
        # if padding_lengths_path is not None:
        #     self.dump_padding_lengths(padding_lengths_path)

    def __iter__(self):
        """Yields individual events from the dataset in after loading them in chunks.

        If the Dataloader is using multiple workers, each worker will process a different set of
        chunks based on the worker ID (as determined by torch.utils.data.get_worker_info()).

        Yields:
            tuple: A tuple (features, labels, weights) of torch Tensors

        Note:
            Optionally, both the chunks and the events can be shuffled before yielding.
            During regular training, it is recommended to set both to True.

            The splitting of the data works as follows:
            - The dataset is split into chunks of size `chunk_size`.
            - Each worker processes a different set of chunks based on its ID. The
                actual chunk index can be shuffled beforehand.
            - Within each chunk, the events are loaded into memory using the
                `_compute_tensors` method, which pads the features and labels to the
                maximum number of particles per event. Same as in the parent class,
                Afterwards, the events are optionally shuffled by permuting their
                indices. Finally, each event is yielded as a tuple of tensors
                (features, labels) of shape (P, F).
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
