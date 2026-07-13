
import numpy as np
from typing import Literal
from torch.utils.data import IterableDataset

from needle.etl.dask_ingestor import Ingestor
from needle.etl.dask_grouped_ingestor import GroupedIngestor
from needle.ml.datasets.io import load_partition
from needle.ml.datasets.kfold import KFold
from needle.ml.datasets.grouped_base import GroupedDatasetBase
from needle.utils.logging import ColorFormatter

logger = ColorFormatter.get_logger("ml")

class GroupedDaskDataset(IterableDataset, GroupedDatasetBase):
    """Out-of-memory grouped (particles, features) Dataset using dask
    multithreading and single torch worker (num_workers=0)."""

    SHUFFLE_ALLOWED: bool = False
    TORCH_MULTIPROCESSING_ALLOWED: bool = False

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
        aux_feature_fields: list[str] | None = None,
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
            aux_feature_fields (list[str], optional): raw (real) column names to
                additionally extract per event, already scaled, alongside the
                usual (features, labels, weights) — see
                GroupedDatasetBase.extract_aux_fields. Validation/test-only;
                None (default) leaves __iter__'s yield shape exactly as before
                this option existed.

        Note:
            Single-worker use (num_workers=0), same as PaddedDaskDataset — for
            multi-worker processing, use GroupedTorchDataset.
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
        self.aux_feature_fields = aux_feature_fields


        if features.array.npartitions == 1:
            logger.warning(
                f"[{features.name}] features array has only 1 partition — no real "
                "partition-level chunking will occur; each 'partition load' will "
                "materialize the ENTIRE dataset into memory at once. Consider "
                "dak_reader_kwargs={'split_row_groups': True} (or equivalent) to "
                "split the underlying parquet file(s) into multiple dask partitions."
            )

    def __iter__(self):
        """Yields individual events, iterating through partitions sequentially.

        Yields:
            tuple: (features, labels, weights) of torch Tensors, or
                (features, labels, weights, aux) if aux_feature_fields was set —
                aux is a dict[str, torch.Tensor] of per-event, still-scaled
                auxiliary values (see GroupedDatasetBase.extract_aux_fields).
                features has shape (particle_max, n_features), labels/weights
                are unchanged from the padded pipeline (see
                convert_flat_ak_to_tensor).

        """
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

        # logger.debug(f"partitions assigned this worker: {list(partitions.keys())}")

        for partition_id, slicing_index in partitions.items():
            features_partition = load_partition(self.features_ingestor.array, partition_id, slicing_index)
            labels_partition = load_partition(self.labels_ingestor.array, partition_id, slicing_index)
            weights_partition = load_partition(self.weights_ingestor.array, partition_id, slicing_index)

            if self.aux_feature_fields:
                features_array = features_partition.compute()
                features_tensor = self.convert_grouped_ak_to_tensor(
                    features_array,
                    self.features_ingestor,
                )
                aux_tensors = self.extract_aux_fields(
                    features_array, self.features_ingestor, self.aux_feature_fields
                )
                # only held for the extraction above, not the yield loop below
                # this fork does have a greater memory footprint, but it is intended
                # to be used only on test/downstream tasks
                del features_array  
            else:
                features_tensor = self.convert_grouped_ak_to_tensor(
                    features_partition.compute(),
                    self.features_ingestor,
                )
                aux_tensors = {}

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

            # logger.debug(f"partition {partition_id}: features_tensor.shape={tuple(features_tensor.shape)}")

            if labels_tensor.ndim == 2 and labels_tensor.shape[-1] == 1:
                labels_tensor = labels_tensor.squeeze(-1)

            event_indices = list(range(len(features_tensor)))

            if self.shuffle_events:
                event_indices = rng.permutation(event_indices)


            if self.aux_feature_fields:
                for event_idx in event_indices:
                    yield (
                        features_tensor[event_idx],
                        labels_tensor[event_idx],
                        weights_tensor[event_idx],
                        {name: t[event_idx] for name, t in aux_tensors.items()},
                    )
            else:
                for event_idx in event_indices:
                    yield (features_tensor[event_idx], labels_tensor[event_idx], weights_tensor[event_idx])
