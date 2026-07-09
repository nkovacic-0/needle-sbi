import json
import logging
from pathlib import Path
from typing import Literal
from functools import partial

import lightning as L
from torch.utils.data import DataLoader

from needle.etl.dask_ingestor import Ingestor
from needle.etl.dask_grouped_ingestor import GroupedIngestor
from needle.etl.grouped_normalization import GroupedScaler, load_grouped_scaler
from needle.etl.column_normalization import ColumnScaler, load_column_scaler
from needle.etl.validate_grouped_dataset_config import validate_grouped_config
from needle.ml.datasets.grouped_delayed_dask import GroupedDaskDataset
from needle.ml.datasets.grouped_delayed_torch import GroupedTorchDataset
from needle.ml.datasets.kfold import KFold
from needle.ml.lightning.datamodules.datamodule_utils import resolve_versioned_path
from needle.ml.lightning.datamodules.padded_datamodule import padded_collate_fn
from needle.utils.config_schema import DatasetConfig
from needle.utils.logging import ColorFormatter

logger = ColorFormatter.get_logger("ml")

_DATASET_REGISTRY = {
    "dask": GroupedDaskDataset,
    "torch": GroupedTorchDataset,
}


class GroupedDataModule(L.LightningDataModule):
    """
    LightningDataModule for the grouped particle-feature pipeline. Mirrors
    PaddedDataModule's structure (setup/train_dataloader/val_dataloader, scaler
    and padding save/load), wired to GroupedIngestor/ColumnScaler and the
    explicit prep sequence (sentinel resolution -> scaling -> missing-
    column fill) established for this pipeline.

    Note: unlike PaddedDataModule, scaler-related settings (scaler_choice,
    scaler_save_path, scaler_load_path, scaler_use_sampling, scaler_sample_fraction,
    force_avoid_partition_sampling) live on DatasetConfig itself, not as separate
    constructor arguments here (since they're dataset-dependent).
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
        padding_lengths_save_path: str | Path | None = None,
        padding_lengths_load_path: str | Path | None = None,
        force_resave_padding_scaler: bool = False,
        apply_sentinel_resolution: bool = True,
        apply_missing_column_fill: bool = True,
        validate_padding_layout: bool = False,
    ) -> None:
        super().__init__()
        self.dataset_config = DatasetConfig(**dataset_config)
        validate_grouped_config(self.dataset_config)

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

        self.apply_sentinel_resolution = apply_sentinel_resolution
        self.apply_missing_column_fill = apply_missing_column_fill
        self.validate_padding_layout_flag = validate_padding_layout

        logger.debug(f"[Grouped datamodule] n_folds: {n_folds}")
        logger.debug(f"[Grouped datamodule] fold_index: {fold_index}")

    def setup(self, stage: str | None = None) -> None:
        cfg = self.dataset_config

        features = GroupedIngestor(
            cfg.paths,
            feature_columns_grouped=cfg.feature_columns_grouped,
            particle_names=cfg.particle_names,
            feature_names=cfg.feature_names,
            replacements_missing_columns=cfg.replacements_missing_columns,
            replacements_default=cfg.replacements_default,
            general_sentinel_value=cfg.general_sentinel_value,
            replace_all_sentinel_values=cfg.replace_all_sentinel_values,
            replacements_sentinel_default=cfg.replacements_sentinel_default,
            replacements_sentinel_columns=cfg.replacements_sentinel_columns,
            format=cfg.format,
            reader_kwargs=cfg.dak_reader_kwargs,
            max_number_events=cfg.max_number_events,
            name="FeaturesIngestor",
        )
        logger.debug(f"features.fields (real columns read)={features.fields}")
        logger.debug(f"features.missing_columns={features.missing_columns}")

        labels = Ingestor(
            cfg.paths,
            format=cfg.format,
            columns=cfg.labels_columns,
            reader_kwargs=cfg.dak_reader_kwargs,
            max_number_events=cfg.max_number_events,
            name="LabelsIngestor",
        )
        weights = Ingestor(
            cfg.paths,
            format=cfg.format,
            columns=cfg.weights_columns,
            reader_kwargs=cfg.dak_reader_kwargs,
            max_number_events=cfg.max_number_events,
            name="WeightsIngestor",
        )

        # logger.debug(f"[prep] after construction: npartitions={features.array.npartitions}, divisions={features.array.divisions}")
        # logger.debug(f"[LabelsIngestor] npartitions={labels.array.npartitions}, divisions={labels.array.divisions}")
        # logger.debug(f"[WeightsIngestor] npartitions={weights.array.npartitions}, divisions={weights.array.divisions}")

        # Step 2: sentinel/None resolution (lazy)
        if self.apply_sentinel_resolution:
            logger.info("Resolving sentinels/None on features...")
            features.array = features.resolve_sentinels()
            # logger.debug(f"[prep] after resolve_sentinels: npartitions={features.array.npartitions}, divisions={features.array.divisions}")
        else:
            logger.info("Skipping sentinel resolution (apply_sentinel_resolution=False).")
        
        # Step 3: scaler — load pre-fitted statistics, or fit fresh (lazy)
        if cfg.scaler_load_path is not None:
            logger.info(f"Loading pre-fitted grouped scaler from {cfg.scaler_load_path}...")
            self.scaler = load_column_scaler(cfg.scaler_load_path)
            if self.scaler.scaler_choice != cfg.scaler_choice:
                logger.warning(
                    f"scaler_choice='{cfg.scaler_choice}' is being ignored; loaded "
                    f"'{self.scaler.scaler_choice}' scaler from disk instead."
                )
            features.array = self.scaler.apply_with_cache(features, self.scaler.cache)
            logger.info(f"Scaler loaded and applied ({self.scaler.scaler_choice}).")
        else:
            logger.info(f"Fitting scaler ({cfg.scaler_choice}) on features...")
            self.scaler = ColumnScaler(scaler_choice=cfg.scaler_choice)
            # if cfg.force_avoid_partition_sampling==True, it disallows the scaler-fitting sampler from  
            # selecting sample events partition-by-partition (avoids biased/uneven per-partition sampling)
            # however, the alternative is just a hard slice on the events
            features.array = self.scaler.apply(
                features,
                use_sampling=cfg.scaler_use_sampling,
                sample_fraction=cfg.scaler_sample_fraction,
                force_avoid_partition_sampling=cfg.force_avoid_partition_sampling,
            )
            path = None
            if cfg.scaler_save_path is not None:
                path = resolve_versioned_path(
                    cfg.scaler_save_path, self.fold_index, ".json", force=self.force_resave_padding_scaler
                )
                if path is not None:
                    self.scaler.save(path)
            logger.info(f"Scaler fitted{f' and saved to {path}' if path is not None else ''}.")

        logger.debug(f"Scaler cache keys: {list(self.scaler.cache.keys())}")
        # logger.debug(f"[prep] after scaler: npartitions={features.array.npartitions}, divisions={features.array.divisions}")

        # Step 4: missing-column fill, with particle-wide-absence override (lazy)
        if self.apply_missing_column_fill:
            logger.info("Filling schema-missing columns...")
            features.array = features.fill_missing_columns()
        else:
            logger.info("Skipping missing-column fill (apply_missing_column_fill=False).")

        self.features = features
        self.labels = labels
        self.weights = weights
        # logger.debug(f"[prep] after fill_missing_columns: npartitions={features.array.npartitions}, divisions={features.array.divisions}")

        # Padding layout — load pre-computed values, or compute fresh
        if self.padding_lengths_load_path is not None:
            logger.info(f"Loading pre-computed padding layout from {self.padding_lengths_load_path}...")
            with open(self.padding_lengths_load_path) as f:
                saved_layout = json.load(f)
            features.set_padding_lengths(saved_layout)
            logger.info(f"Padding layout loaded: {saved_layout}")
        else:
            logger.info("Computing padding layout (per-particle reference field)...")
            features.compute_padding_layout()

            path = None
            if self.padding_lengths_save_path is not None:
                path = resolve_versioned_path(
                    self.padding_lengths_save_path, self.fold_index, ".json", force=self.force_resave_padding_scaler
                )
                if path is not None:
                    with path.open("w") as f:
                        json.dump(features.get_all_padding_lengths(), f, indent=2, sort_keys=True)
            logger.info(f"Padding layout computed{f' and saved to {path}' if path is not None else ''}.")

        if self.validate_padding_layout_flag:
            logger.info("Validating padding layout across all rows (opt-in, expensive)...")
            features.validate_padding_layout()

        logger.debug(f"Final padding layout: {features.get_all_padding_lengths()}")

    @staticmethod
    def get_dataset(name: str):
        if name not in _DATASET_REGISTRY:
            raise ValueError(name)
        return _DATASET_REGISTRY[name]

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