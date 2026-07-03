import torch
import torch.nn as nn

from typing import Any, Callable
from collections.abc import Sequence

from needle.utils.logging import ColorFormatter

logger = ColorFormatter.get_logger("ml")


class MLP(nn.Module):
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
        # parse and validate the input logic
        if output_activation_kwargs and output_activation is None:
            logger.warning("output_activation_kwargs provided but output_activation is None — kwargs will be ignored.")
        if dropout_prob != 0.5 and not use_dropout:  # 0.5 being the default for this class
            logger.warning("dropout_prob provided but use_dropout=False — dropout_prob will be ignored.")
        # expose the output dim, in case we need to call it
        self.output_dim = output_dim
        # accumulate layers, looping over the requested nums of nodes in the 'hidden_layers'
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            # note: hidden_layers=[] is valid and maps x->output_activation(linear(x))
            layers.append(nn.Linear(prev_dim, hidden_dim))
            # add normalization, note that batchnorm WON'T work with inputs of more than 2 dimensions i.e. (B, T, D)/(B, P, F)
            if normalization is not None:
                layers.append(self._get_normalization(normalization, hidden_dim))
            # add activ function, pass any kwargs that the user provided
            layers.append(self._get_activation(activation, **(activation_kwargs or {})))
            # add dropout
            if use_dropout:
                layers.append(nn.Dropout(dropout_prob))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        # technically, we will want to always include the output_activation
        if output_activation is not None:
            layers.append(self._get_activation(output_activation,**(output_activation_kwargs or {})))
        # let torch unpack the layers
        self.network = nn.Sequential(*layers)
        # initialize the network wgs with whatever the user provided in the 'init' arg
        self.reset_parameters(init, activation)

    # what follows are several boilerplate statics for activation, normalization, etc. selection
    @staticmethod
    def _get_activation(
        activation: str | type[nn.Module] | nn.Module,
        **kwargs
    ) -> nn.Module:
        # in case we get a string, we match it against a dict of commonly used activ functions
        if isinstance(activation, str):
            # a simple mapping of commonly used activation names to torch functions
            activations = {
                "relu": nn.ReLU,
                "leaky_relu": nn.LeakyReLU,
                "gelu": nn.GELU,
                "elu": nn.ELU,
                "tanh": nn.Tanh,
                "sigmoid": nn.Sigmoid,
                "selu": nn.SELU,
                "silu": nn.SiLU,  #<-aka swish
                "swish": nn.SiLU, #also catch someone asking for 'swish' :)
            }
            try:
                return activations[activation.lower()](**kwargs)
            except KeyError as e:
                logger.error(
                    f"Unsupported activation function key '{activation}'. "
                    f"Available activations (keys): {', '.join(activations)}"
                )
                raise
        elif isinstance(activation, nn.Module):
            return deepcopy(activation)
        elif isinstance(activation, type) and issubclass(activation, nn.Module):
            return activation(**kwargs)
        else:
            err_msg = (
                f"activation must be a string, nn.Module instance, "
                f"or nn.Module subclass, got {type(activation)}"
            )
            logger.error(err_msg)
            raise TypeError(err_msg)

    @staticmethod
    def _get_normalization(name: str, dim: int) -> nn.Module:
        norms = {
            "batchnorm": nn.BatchNorm1d,
            "layernorm": nn.LayerNorm,
        }
        try:
            return norms[name.lower()](dim)
        except KeyError as e:
            logger.error(
                f"Unsupported normalization key '{name}'. "
                f"Available normalizations (keys): {', '.join(norms)}"
            )
            raise

    @staticmethod
    def _init_weights(
            module: nn.Module, 
            init: str,
            activation: str | type[nn.Module] | nn.Module,
        ):
        if not isinstance(module, nn.Linear):
            return
        if init == "kaiming":
            # kaiming works better if gains are calculated with specific activation functions in mind
            # here we compare the provided activation with some of the supported ones
            # be extra careful as there is no check provided for the output layer innit, which can break the training
            if isinstance(activation, str):
                activation_map = {
                    "relu":  "relu",
                    "elu":   "relu", # approx
                    "gelu":  "relu", # approx
                    "silu":  "relu", # approx
                    "swish": "relu",
                    "tanh":  "tanh",
                    "sigmoid": "sigmoid",
                    "leaky_relu": "leaky_relu",
                    "selu": "linear",   # SELU uses its own scaling internally
                }
                nonlinearity = activation_map.get(activation.lower(), "relu")
            else:
                nonlinearity = "relu"
                logger.warning(f"Non-string activation provided for Kaiming init! Falling back to 'relu' nonlinearity.")
            nn.init.kaiming_normal_(module.weight, nonlinearity=nonlinearity)
        elif init == "xavier":
            nn.init.xavier_uniform_(module.weight)
        elif init == "normal":
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif init == "zeros":
            logger.warning("init='zeros' will set all weights to 0.0! This breaks gradient symmetry and will prevent learning if applied to the full network!")
            nn.init.zeros_(module.weight)
        else:
            logger.error(f"Unsupported string-based init option: {init}")
            raise ValueError(f"Unsupported string-based init option: {init}")
        if module.bias is not None:
            # technically, we could add a small uniform init for biases if we use xavier - todo
            nn.init.zeros_(module.bias)

    def reset_parameters(
            self, 
            init: str | Callable[[nn.Module], None], 
            activation: str | type[nn.Module] | nn.Module | None = None,
        ):
        # helper function to reset or init model weights
        # do note that it requires raw init param!
        if callable(init) and not isinstance(init, str):
            self.network.apply(init)
        else:
            if activation is None:
                err_msg = (
                    f"Passed 'None' as 'activation' argument while 'init' that was passed isn't callable!"
                    f"Expected behaviour in this case is to have 'activation' be one of the type hints str | type[nn.Module] | nn.Module"
                )
                logger.error(err_msg)
                raise ValueError(err_msg)                
            self.network.apply(lambda m: self._init_weights(m, init, activation))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)