import torch
import torch.nn as nn

from typing import Any
from collections.abc import Sequence

from needle.ml.lightning.models.src.mlp_model import MLP
from needle.ml.lightning.models.src.attention import Attention
from needle.ml.lightning.models.src.embedding import MLPEmbedding
from needle.ml.lightning.models.src.transformer_block import TransformerBlock
from needle.ml.lightning.models.src.pooling import Pooling


from needle.utils.logging import ColorFormatter

logger = ColorFormatter.get_logger("ml")


class TransformerModel(nn.Module):
    def __init__(
        self,
        # input dim
        num_features_in: int,
        # embedding
        embedding_type: str | None = None, # 'mlp' or None
        embedding_hidden_layers: Sequence[int] = (16, 16),
        embedding_output_dim: int = 16,
        embedding_kwargs: dict | None = None,
        # transformer blocks
        n_transformer_blocks: int = 4,
        transformer_block_n_heads: int = 4,
        transformer_block_attn_type: str = "default_torchMHA",
        transformer_block_attn_kwargs: dict | None = None,
        transformer_block_mlp_hidden_layers: Sequence[int] = (256, 256),
        transformer_block_mlp_kwargs: dict | None = None,
        transformer_block_dropout: float = 0.1,
        transformer_block_residual_mode: str = "standard",
        # conditioning
        use_conditioning: bool = False,
        cond_dim: int | None = None,
        condscale_startval: float = 0.0,
        # pooling
        pool_type: str = "mean",
        pool_dropout: float = 0.0,
        # MLP head
        mlp_head_hidden_layers: Sequence[int] = (128,),
        mlp_head_outdim: int = 1,
        mlp_head_kwargs: dict | None = None,
        # global
        batch_first: bool = True,
    ):
        super().__init__()
        
        # parse and validate inputs
        self.batch_first = batch_first
        # embedding
        if embedding_type is not None:
            self.skip_embedding = False
            _supported_embeddings = {"mlp"}
            if embedding_type not in _supported_embeddings:
                err_msg = (
                    f"Unsupported embedding_type '{embedding_type}'. "
                    f"Expected one of: {', '.join(sorted(_supported_embeddings))}"
                )
                logger.error(err_msg)
                raise ValueError(err_msg)
        else:
            self.skip_embedding = True
        # conditioning
        if use_conditioning and cond_dim is None:
            err_msg = (
                "use_conditioning=True but cond_dim is None. "
                "Provide the dimension of the conditioning input."
            )
            logger.error(err_msg)
            raise ValueError(err_msg)
        self.use_conditioning = use_conditioning
        if self.skip_embedding and num_features_in % transformer_block_n_heads != 0:
            err_msg = (
                f"When skip_embedding=True, num_features_in ({num_features_in}) must be "
                f"divisible by transformer_block_n_heads ({transformer_block_n_heads})."
            )
            logger.error(err_msg)
            raise ValueError(err_msg)

        # define model components
        # embedding
        emb_kwargs = embedding_kwargs or {}
        self.embedding = None
        if embedding_type == "mlp":
            self.embedding = MLPEmbedding(
                input_dim = num_features_in,
                hidden_layers = embedding_hidden_layers,
                output_dim = embedding_output_dim,
                **emb_kwargs,
            )
        # d_model in TransformerBlock is whatever the embedding projects to
        # if there's no enbedding it is the input dim
        if self.skip_embedding:
            d_model = num_features_in
        else:
            d_model = embedding_output_dim

        # transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                transformer_block_d_model = d_model,
                transformer_block_n_heads = transformer_block_n_heads,
                transformer_block_atttn_type = transformer_block_attn_type,
                transformer_block_atttn_kwargs = transformer_block_attn_kwargs,
                transformer_block_mlp_hidden_layers = transformer_block_mlp_hidden_layers,
                transformer_block_mlp_kwargs = transformer_block_mlp_kwargs,
                transformer_block_dropout = transformer_block_dropout,
                transformer_block_residual_mode = transformer_block_residual_mode,
                transformer_block_use_conditioning = use_conditioning,
                transformer_block_cond_dim = cond_dim,
                transformer_block_condscale_startval = condscale_startval,
                batch_first = batch_first,
            )
            for _ in range(n_transformer_blocks)
        ])
        # pooling
        self.pooling = Pooling(
            d_model = d_model,
            pool_type = pool_type,
            dropout = pool_dropout,
        )
        # MLP head
        mlp_head_kwargs = mlp_head_kwargs or {}
        self.classifier = MLP(
            input_dim = d_model,
            hidden_layers = mlp_head_hidden_layers,
            output_dim = mlp_head_outdim,
            **mlp_head_kwargs,
        )

    def forward(
        self,
        x: torch.Tensor,
        cond_x: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x: (B, T, F) i.e. (B, P, F) in our case

        # we need to prepare a padding mask from nans before setting them to 0.0
        # the convention is key_padding_mask: (B, T) bool, True = ignore (padding),
        # that matches nn.MHA convention
        key_padding_mask = torch.isnan(x).any(dim=-1) 
        # now we can remove all nans so they don't propagate through the matrix ops
        x = torch.nan_to_num(x, nan=0.0)
        # if we have embedding apply it
        # here (B, T, F) -> (B, T, D), D == d_model var value from above
        if not self.skip_embedding:
            x = self.embedding(x)
        # pass through transformer blocks, carrying mask and optional conditioning
        for block in self.transformer_blocks:
            x = block(
                x,
                cond_x=cond_x,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
            )
        # apply pooling, transforming (B, T, D) -> (B, D)
        # we laso need to pass the key_padding_mask
        x = self.pooling(x, mask=key_padding_mask)
        # apply the MLP head, which transforms (B, D) -> (B, mlp_head_outdim)
        return self.classifier(x)
        
