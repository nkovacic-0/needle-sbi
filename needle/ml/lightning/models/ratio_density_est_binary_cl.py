from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import lightning as L

from torchmetrics.classification import BinaryAccuracy

from needle.ml.lightning.models.src.transformer_model import TransformerModel
from needle.ml.lightning.models.src.training_configuration import custom_configure_optimizers
from needle.ml.lightning.models.model_utils import (
    unwrap_labels,
    WeightedBinaryAccuracy,
    ExpectedCalibrationError,
)

from needle.utils.config_schema import DatasetConfig
from needle.utils.logging import ColorFormatter

logger = ColorFormatter.get_logger("ml")

class RatioDensityEstimatorBinary(L.LightningModule):
    def __init__(
        self,
        transformer_model_configs: dict,
        optimizer_configs: dict,
        dataset_config: dict | None = None,
        weighted_loss: str | None = None,
        num_features_in: int | None = None,
        ECE_coarse_binning_bin_num: int = 10,
        ECE_fine_binning_bin_num: int = 30,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        # metrics form torchmetrics.classification
        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        # custom weighted metrics
        if weighted_loss is not None:
            self.train_weighted_acc = WeightedBinaryAccuracy()
            self.val_weighted_acc = WeightedBinaryAccuracy()

        # ECE: coarse + fine binning, computed once per epoch (see epoch-end hooks below)
        if weighted_loss is not None:
            _ece_weighted = True
            self._ece_prefix = "weighted_"
        else:
            _ece_weighted = False
            self._ece_prefix = ""
        self.train_ece_coarse = ExpectedCalibrationError(n_bins=ECE_coarse_binning_bin_num, weighted=_ece_weighted)
        self.val_ece_coarse   = ExpectedCalibrationError(n_bins=ECE_coarse_binning_bin_num, weighted=_ece_weighted)
        self.train_ece_fine   = ExpectedCalibrationError(n_bins=ECE_fine_binning_bin_num, weighted=_ece_weighted)
        self.val_ece_fine     = ExpectedCalibrationError(n_bins=ECE_fine_binning_bin_num, weighted=_ece_weighted)

        # validate optimizer_configs 
        _required_optimizer_keys = {"optimizer", "lr"}
        if not _required_optimizer_keys.issubset(optimizer_configs.keys()):
            err_msg = (
                f"optimizer_configs is missing required keys. "
                f"Expected at least: {_required_optimizer_keys}, "
                f"got: {set(optimizer_configs.keys())}"
            )
            logger.error(err_msg)
            raise ValueError(err_msg)
        self.optimizer_configs = optimizer_configs
        # validate the setting for weights in the loss function
        _allowed_weighted_loss = {None, "mean", "weighted_mean"}
        if weighted_loss not in _allowed_weighted_loss:
            err_msg = (
                f"Unsupported value for 'weighted_loss': {weighted_loss!r}. "
                f"Expected one of: {_allowed_weighted_loss}."
            )
            logger.error(err_msg)
            raise ValueError(err_msg)
        self.weighted_loss = weighted_loss

        # we want to add the number of input features to the transformer_model_configs
        # first, we make a copy so we don't mutate the original config dict
        self.transformer_model_configs = dict(transformer_model_configs)
        # determine input dimenson
        if num_features_in is not None:
            self.num_features_in = int(num_features_in)
        elif dataset_config is not None:
            ds_cfg = DatasetConfig(**dataset_config)
            if ds_cfg.feature_columns_grouped is not None:
                self.num_features_in = len(ds_cfg.feature_columns_grouped)
            else:
                self.num_features_in = len(ds_cfg.features_columns)
        else:
            err_msg = (
                "Expected at least one of 'num_features_in' or 'dataset_config' to be != None, "
                "but was provided with None for both. Cannot determine input tensor dimension!"
            )
            logger.error(err_msg)
            raise ValueError(err_msg)
                 
        self.transformer_model_configs["num_features_in"] = self.num_features_in

        # reduction for the loss is handled in the _shared_step method
        self.loss_function = nn.BCEWithLogitsLoss(reduction="none")

    def configure_model(self) -> None:
        self.model = TransformerModel(**self.transformer_model_configs)

    def configure_optimizers(self) -> dict:
        optimizer_config = custom_configure_optimizers(
            parameters = self.parameters(),
            optimizer_configs = self.optimizer_configs,
        )
        return optimizer_config

    # @property
    # def loss_function(self) -> nn.Module:
    #     # reduction in handled in the _shared_step method
    #     return nn.BCEWithLogitsLoss(reduction = "none")

    def forward(self, x: torch.Tensor, cond_x: torch.Tensor | None = None) -> torch.Tensor:
        return self.model(x, cond_x=cond_x)

    def _shared_step(self, batch: tuple[torch.Tensor, ...], stage: str) -> torch.Tensor:
        # this method is here so that the logic is shared between train and valid
        # batch should be in order: x, cond_x, labels, weights
        # note:  labels is now a dict[str, Tensor]!
        # if self.weighted_loss is None:
        #     if len(batch) == 2: # TODO
        #         x, labels = batch
        #         cond_x = None
        #     elif len(batch) == 3:
        #         x, cond_x, labels = batch
        #     else:
        #         err_msg = (
        #             f"Expected batch of length 2 (x, labels) or 3 (x, cond_x, labels), but got {len(batch)}"
        #         )
        #         logger.error(err_msg)
        #         raise ValueError(err_msg)
        # else:
        #     if len(batch) == 3:
        #         x, labels, weights = batch
        #         cond_x = None
        #     elif len(batch) == 4:
        #         x, cond_x, labels, weights = batch
        #     else:
        #         err_msg = (
        #             f"Expected batch of length 3 (x, labels, weights) or 4 (x, cond_x, labels, weights), but got {len(batch)}"
        #         )
        #         logger.error(err_msg)
        #         raise ValueError(err_msg)  
        # swapped the code above to always unpack weights, but then they get ingnored unless weighted_loss is specified
        if len(batch) == 3:
            x, labels, weights = batch
            cond_x = None
        elif len(batch) == 4:
            x, cond_x, labels, weights = batch
        else:
            err_msg = (
                f"Expected batch of length 3 (x, labels, weights) or 4 (x, cond_x, labels, weights), but got {len(batch)}"
            )
            logger.error(err_msg)
            raise ValueError(err_msg) 

        output = self(x, cond_x=cond_x)

        # here, our labels only have 1 dim, we can just unwrap dict{str: torch.tensor} with the util function
        labels = unwrap_labels(labels)
        # output: (B, 1), labels: (B,) or (B, 1) — align shapes for BCEWithLogitsLoss
        output = output.squeeze(-1)
        labels = labels.float()
        weights = weights.float()

        loss = self.loss_function(output, labels)
        if self.weighted_loss == 'mean':
            loss = (loss * weights).mean()
        elif self.weighted_loss == 'weighted_mean':
            loss = (loss * weights).sum() / weights.sum()
        else:
            loss = loss.mean()

        # this is needed in the callbacks
        self.log(f"{stage}_loss", loss, on_step=(stage == "train"), on_epoch=True, prog_bar=True)

        # other metrics...
        # accuracy
        metric = self.train_acc if stage == "train" else self.val_acc
        metric.update(output, labels.int())
        self.log(f"{stage}_accuracy", metric.compute(), on_epoch=True, prog_bar=True)
        if self.weighted_loss is not None:
            weighted_metric = (
                self.train_weighted_acc if stage == "train" else self.val_weighted_acc
            )
            weighted_metric.update(output, labels.int(), weights)
            self.log(f"{stage}_weighted_accuracy", weighted_metric.compute(), on_epoch=True)
        # ECE
        # accumulate only the compute/log/reset happens at epoch end (see hooks below)
        ece_coarse = self.train_ece_coarse if stage == "train" else self.val_ece_coarse
        ece_fine = self.train_ece_fine if stage == "train" else self.val_ece_fine
        if self.weighted_loss is not None:
            ece_coarse.update(output, labels, weights)
            ece_fine.update(output, labels, weights)
        else:
            ece_coarse.update(output, labels)
            ece_fine.update(output, labels)

        return loss


    def training_step(self, batch: tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: tuple[torch.Tensor, ...], batch_idx: int) -> None:
        return self._shared_step(batch, "val")

    # epoch start resets
    def on_validation_epoch_start(self):
        self.val_acc.reset()
        if self.weighted_loss is not None:
            self.val_weighted_acc.reset()
    def on_train_epoch_start(self):
        self.train_acc.reset()
        if self.weighted_loss is not None:
            self.train_weighted_acc.reset()

    def on_train_epoch_end(self):
        self.log(f"train_{self._ece_prefix}ece_coarse", self.train_ece_coarse.compute())
        self.log(f"train_{self._ece_prefix}ece_fine", self.train_ece_fine.compute())
        self.train_ece_coarse.reset()
        self.train_ece_fine.reset()

    def on_validation_epoch_end(self):
        self.log(f"val_{self._ece_prefix}ece_coarse", self.val_ece_coarse.compute())
        self.log(f"val_{self._ece_prefix}ece_fine", self.val_ece_fine.compute())
        self.val_ece_coarse.reset()
        self.val_ece_fine.reset()