import json
import logging
from pathlib import Path
from typing import Literal
from functools import partial

import torch
import lightning as L
from torch.utils.data import DataLoader

from needle.etl.dask_ingestor import Ingestor
from needle.etl.normalization import MinMaxScaler, StandardScaler, load_scaler
from needle.ml.datasets import PaddedDaskDataset, PaddedTorchDataset
from needle.ml.datasets.kfold import KFold
from needle.utils.config_schema import DatasetConfig
from needle.ml.lightning.datamodules.datamodule_utils import resolve_versioned_path

from needle.utils.logging import ColorFormatter

logger = ColorFormatter.get_logger("ml")

# Maps config-facing scaler names to their needle.etl.normalization classes.
SCALER_REGISTRY = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
}

class PaddedDataModule(L.LightningDataModule):
    """
    LightningDataModule for the flat/ragged (padded) pipeline. Builds Ingestor
    instances for features/labels/weights, then in setup() runs the shared
    prep sequence: scaler load-or-fit (load_scaler/apply_with_cache if
    scaler_load_path is set, otherwise BaseScaler.apply()) followed by
    padding-length load-or-fit (features.set_padding_lengths() if
    padding_lengths_load_path is set, otherwise compute_all_padding_lengths()
    with a sequential get_padding_length() fallback). Both scaler and padding
    artifacts are saved via resolve_versioned_path (see datamodule_utils.py)
    when a save path is configured, which skips re-saving if the canonical
    file already exists (unless force_resave_padding_scaler=True).

    Note: GroupedDataModule mirrors this file's structure for the grouped
    particle-feature pipeline, swapping in GroupedIngestor/ColumnScaler and an
    extended prep sequence (sentinel resolution -> scaling -> missing-column
    fill) in place of the single scaler-apply step here.
    """

    def __init__(
        self,
        dataset_config: dict,
        batch_size: int = 1024,
        fold_index: int | None = None,
        n_folds: int | None = None,
        n_workers: int = 0,
        multiprocessing_type: Literal["torch", "dask"] = "torch",
        shuffle_partitions: bool = True,
        shuffle_events: bool = True,
        # path to save padding info to, has to be a full path
        padding_lengths_save_path: str | Path | None = None,
        # path where to load padding from
        padding_lengths_load_path: str | Path | None = None,
        # variables determining scaler behaviour
        scaler_choice: Literal["standard", "minmax"] = "standard", # this is irrelevant if scaler_load_path != None
        scaler_save_path: str | Path | None = None, # full filepath to save scalers to
        scaler_load_path: str | Path | None = None, # full filepath to load scalers from
        # forces new saves of padding and scalers to new files, if the originally targeted file exists
        force_resave_padding_scaler: bool = False,
        scaler_use_sampling: bool = False,
        scaler_sample_fraction: float = 0.10,
        # if force_avoid_partition_sampling==True, it disallows the scaler-fitting sampler from selecting 
        # sample events partition-by-partition (avoids biased/uneven per-partition sampling)
        # however, the alternative is just a hard slice on the events
        force_avoid_partition_sampling: bool = True, 
    ) -> None:
        super().__init__()
        self.dataset_config = DatasetConfig(**dataset_config)
        self.batch_size = batch_size
        self.fold_index = fold_index

        self.n_folds = n_folds
        self.n_workers = n_workers
        self.multiprocessing_type = multiprocessing_type
        self.shuffle_partitions = shuffle_partitions
        self.shuffle_events = shuffle_events
        self.padding_lengths_save_path = padding_lengths_save_path
        self.padding_lengths_load_path = padding_lengths_load_path

        self.force_resave_padding_scaler = force_resave_padding_scaler

        self.scaler_save_path = scaler_save_path
        self.scaler_load_path = scaler_load_path
        self.scaler_name = scaler_choice
        if scaler_choice not in SCALER_REGISTRY:
            raise ValueError(f"Unknown scaler '{scaler_choice}'. Available: {list(SCALER_REGISTRY)}")
        self.scaler = SCALER_REGISTRY[scaler_choice]()
        self.scaler_use_sampling = scaler_use_sampling
        self.scaler_sample_fraction = scaler_sample_fraction
        self.force_avoid_partition_sampling = force_avoid_partition_sampling

        logger.debug(f"[Padded datamodule] n_folds: {n_folds}")
        logger.debug(f"[Padded datamodule] fold_index: {fold_index}")

    def setup(self, stage: str | None = None) -> None:
        features = Ingestor(
            self.dataset_config.paths,
            format=self.dataset_config.format,
            columns=self.dataset_config.features_columns,
            reader_kwargs=self.dataset_config.dak_reader_kwargs,
            max_number_events=self.dataset_config.max_number_events,
            name="FeaturesIngestor",
        )
        labels = Ingestor(
            self.dataset_config.paths,
            format=self.dataset_config.format,
            columns=self.dataset_config.labels_columns,
            reader_kwargs=self.dataset_config.dak_reader_kwargs,
            max_number_events=self.dataset_config.max_number_events,
            name="LabelsIngestor",
        )
        weights = Ingestor(
            self.dataset_config.paths,
            format=self.dataset_config.format,
            columns=self.dataset_config.weights_columns,
            reader_kwargs=self.dataset_config.dak_reader_kwargs,
            max_number_events=self.dataset_config.max_number_events,
            name="WeightsIngestor",
        )
        # Scaler handling: load pre-fitted statistics, or fit fresh
        if self.scaler_load_path is not None:
            logger.info(f"Loading pre-fitted scaler from {self.scaler_load_path}...")
            self.scaler = load_scaler(self.scaler_load_path)   # auto-detects StandardScaler/MinMaxScaler
            if type(self.scaler).__name__.lower().replace("scaler", "") != self.scaler_name:
                logger.warning(
                    f"scaler_choice='{self.scaler_name}' is being ignored, loaded "
                    f"{type(self.scaler).__name__} from disk instead."
                )
            self.scaler_name = type(self.scaler).__name__
            features.array = self.scaler.apply_with_cache(features.array, self.scaler.cache)
            logger.info(f"Scaler loaded and applied ({self.scaler_name}).")
        else:
            logger.info(f"Applying scaler ({self.scaler_name}) to features...")
            features.array = self.scaler.apply(
                features.array,
                use_sampling=self.scaler_use_sampling,
                sample_fraction=self.scaler_sample_fraction,
                force_avoid_partition_sampling=self.force_avoid_partition_sampling,
            )
            path = None
            if self.scaler_save_path is not None:
                path = resolve_versioned_path(self.scaler_save_path, self.fold_index, ".json", force=self.force_resave_padding_scaler)
                if path is not None:
                    self.scaler.save(path)
            logger.info(f"Scaler applied{f' and saved to {path}' if path is not None else ''}.")

        # no need for normalization of labels and weights
        # labels.array = self.scaler.apply(labels.array)
        # weights.array = self.scaler.apply(weights.array)

        self.features = features
        self.labels = labels
        self.weights = weights


        # Padding (lengths) handling: load pre-computed values, or compute fresh
        if self.padding_lengths_load_path is not None:
            logger.info(f"Loading pre-computed padding lengths from {self.padding_lengths_load_path}...")
            with open(self.padding_lengths_load_path) as f:
                saved_lengths = json.load(f)
            features.set_padding_lengths(saved_lengths)
            logger.info(f"Padding lengths loaded: {saved_lengths}")
        else:
            # Explicitly force padding-length computation here, once, rather than relying
            # on it happening implicitly as a side effect of whichever Dataset subclass
            # __init__ runs first (train or val). This also guarantees the dump below
            # reflects a fully-populated cache regardless of which backend/dataloader
            # gets constructed later, or in which order.
            logger.info("Computing padding lengths for all feature fields...")
            # try using the parallel compute_all_padding_lengths, if it fails revert to serial operation get_padding_length
            # where get_padding_length should have its own pyarrow fallback
            try:
                features.compute_all_padding_lengths(features.fields)
            except Exception as e:
                logger.warning(f"Batched padding-length computation failed ({e!r}); falling back to sequential per-field method.")
                for field in features.fields:
                    features.get_padding_length(field)

            path = None
            if self.padding_lengths_save_path is not None:
                path = resolve_versioned_path(self.padding_lengths_save_path, self.fold_index, ".json", force=self.force_resave_padding_scaler)
                if path is not None:
                    padding_lengths = features.get_all_padding_lengths()
                    with path.open("w") as f:
                        json.dump(padding_lengths, f, indent=2, sort_keys=True)
            logger.info(f"Padding lengths computed{f' and saved to {path}' if path is not None else ''}.")


    @staticmethod
    def get_dataset(name: str):
        match name:
            case "dask":
                return PaddedDaskDataset
            case "torch":
                return PaddedTorchDataset
            case _:
                raise ValueError(name)

    def train_dataloader(self) -> DataLoader:
        # the condition that has been commented out evaluates to false for fold_index == 0 
        # (n.b. from estimator task: List[FoldTask]: One task per fold (0 to n_folds-1))
        # and falls back to using the whole dataset, fixed with the new condition
        # if self.fold_index and self.n_folds:
        if self.fold_index is not None and self.n_folds and self.n_folds > 1:
            kfold = KFold(
                fold_index=self.fold_index,
                n_folds=self.n_folds,
                is_training=True,
                divisions=self.features.array.divisions,
            )
        else:
            kfold = None

        dataset = self.get_dataset(self.multiprocessing_type)(
            self.features,
            self.labels,
            self.weights,
            shuffle_partitions=self.shuffle_partitions,
            shuffle_events=self.shuffle_events,
            weights_combine=self.dataset_config.weights_combine,
            kfold=kfold,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=dataset.SHUFFLE_ALLOWED,
            num_workers=self.n_workers if dataset.TORCH_MULTIPROCESSING_ALLOWED else 0,
            collate_fn=partial(padded_collate_fn, label_names=self.labels.fields),
        )

    def val_dataloader(self) -> DataLoader:
        # the condition that has been commented out evaluates to false for fold_index == 0 
        # (n.b. from estimator task: List[FoldTask]: One task per fold (0 to n_folds-1))
        # and falls back to using the whole dataset, fixed with the new condition
        # if self.fold_index and self.n_folds:
        if self.fold_index is not None and self.n_folds and self.n_folds > 1:
            kfold = KFold(
                fold_index=self.fold_index,
                n_folds=self.n_folds,
                is_training=False,
                divisions=self.features.array.divisions,
            )
        else:
            kfold = None
        dataset = self.get_dataset(self.multiprocessing_type)(
            self.features,
            self.labels,
            self.weights,
            shuffle_partitions=self.shuffle_partitions,
            shuffle_events=self.shuffle_events,
            weights_combine=self.dataset_config.weights_combine,
            kfold=kfold,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=dataset.SHUFFLE_ALLOWED,
            num_workers=self.n_workers if dataset.TORCH_MULTIPROCESSING_ALLOWED else 0,
            collate_fn=partial(padded_collate_fn, label_names=self.labels.fields),
        )

def padded_collate_fn(batch, label_names: list[str]):
    features, labels, weights = zip(*batch)
    features = torch.stack(features)
    labels = torch.stack(labels)     # (B,) if squeezed single-column, else (B, F) 
    weights = torch.stack(weights)   # always (B,), thanks to weights_combine
    if labels.ndim == 1:
        labels_dict = {label_names[0]: labels}
    else:
        labels_dict = {name: labels[:, i] for i, name in enumerate(label_names)}
    return features, labels_dict, weights
