from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import lightning as L

from needle.ml.lightning.models.src.transformer_model import TransformerModel
from needle.ml.lightning.models.src.training_configuration import custom_configure_optimizers
from needle.ml.lightning.models.model_utils import unwrap_labels

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
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

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
        if num_features_in is not None:
            self.num_features_in = int(num_features_in)
        if dataset_config is not None:
            self.num_features_in = len(DatasetConfig(**dataset_config).features_columns)
        elif dataset_config is None:
            err_msg = (
                f"Expected at least one of 'num_features_in' or 'dataset_config' to be != None, "
                f"But was provided with None fo rboth variables. Cannot determine input tensor dimension!"
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

        # i our labels only have 1 dim, we can just unwrap dict{str: torch.tensor} with the util function
        labels = _unwrap_labels(labels)
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

        # not sure if this is needed w.r.t. NEEDLE custom training imlementation? - TODO
        self.log(f"{stage}_loss", loss, on_step=(stage == "train"), on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch: tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: tuple[torch.Tensor, ...], batch_idx: int) -> None:
        return self._shared_step(batch, "val")