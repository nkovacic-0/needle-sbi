import torch
import torch.nn as nn

from typing import Any, Callable
from collections.abc import Sequence

from needle.ml.lightning.models.src.mlp_model import MLP

from needle.utils.logging import ColorFormatter

logger = ColorFormatter.get_logger("ml")

# this is literally just a pass-thourgh for the MLP from mlp_model
# certainly it is unnecessary, but it might make tracing the future imports
# like needle.ml.models.src.embedding import MLPEmbedding make more sense?
class MLPEmbedding(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_layers: Sequence[int],
        output_dim: int,
        activation: str | type[nn.Module] | nn.Module = "relu",
        activation_kwargs: dict[str, Any] | None = None,
        output_activation: str | None = None,
        output_activation_kwargs: dict[str, Any] | None = None,
        use_dropout: bool = False,
        dropout_prob: float = 0.5,
        normalization: str | None = None,
        init: str | Callable[[nn.Module], None] = "kaiming",
    ):
        super().__init__()
        # perhaps working with just args and kwargs would be easier, 
        # however, that doesn't seem to mesh well with PL
        self.mlp = MLP(
            input_dim = input_dim,
            hidden_layers = hidden_layers,
            output_dim = output_dim,
            activation = activation,
            activation_kwargs = activation_kwargs,
            output_activation = output_activation,
            output_activation_kwargs = output_activation_kwargs,
            use_dropout = use_dropout,
            dropout_prob = dropout_prob,
            normalization = normalization,
            init = init,
        )

    def forward(self, x):
        return self.mlp(x)