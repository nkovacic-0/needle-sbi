import torch
import torch.nn as nn
from typing import Sequence, Callable, Any

from needle.ml.lightning.models.src.mlp_model import MLP
from needle.ml.lightning.models.src.attention import Attention


from needle.utils.logging import ColorFormatter

logger = ColorFormatter.get_logger("ml")

class TransformerBlock(nn.Module):
    def __init__(
        self,
        transformer_block_d_model: int,
        transformer_block_n_heads: int,
        transformer_block_atttn_type: str = "default_torchMHA",
        transformer_block_atttn_kwargs: dict | None = None,
        transformer_block_mlp_hidden_layers: Sequence[int] = 1,
        transformer_block_mlp_kwargs: dict | None = None,
        transformer_block_dropout: float = 0.1,
        transformer_block_residual_mode: str = "standard",
        transformer_block_use_conditioning: bool = False,
        transformer_block_cond_dim: int | None = None,
        transformer_block_condscale_startval: float = 0.0,
        batch_first: bool = True
    ):
        super().__init__()
        # here we either have a skip connection:
        # 'coupled' from before the TransformerBlock into pre-MLP and post-TransformerBlock
        # 'standard' from before the TransformerBlock into pre-MLP and from pre-MLP into post-TransformerBlock
        assert transformer_block_residual_mode in ["standard", "coupled"]
        if transformer_block_use_conditioning and transformer_block_cond_dim is None:
            err_msg = (
                f"Transformer block is mandated to include conditioning, yet the conditioning "
                f"dimension variable is set to None!"
            )
            logger.error(err_msg)
            raise ValueError(err_msg)
        self.residual_mode = transformer_block_residual_mode
        self.attention_type = transformer_block_atttn_type
        self.use_conditioning = transformer_block_use_conditioning
        mlp_kwargs = transformer_block_mlp_kwargs or {}
        # if we're passing None as the kwargs, default to using the dropout for other 
        # elements of the block in the attn mechanism
        if transformer_block_atttn_kwargs is None:
            transformer_block_atttn_kwargs = {
                'dropout': transformer_block_dropout,
            }
        # define the block components
        self.norm1 = nn.LayerNorm(transformer_block_d_model)
        self.norm2 = nn.LayerNorm(transformer_block_d_model)
        if transformer_block_use_conditioning:
            self.norm_cond = nn.LayerNorm(transformer_block_d_model)

        # the 'default_torchMHA' is meant for debugging, bypassing the 
        # custom attetnion implementations in the Attention class
        if self.attention_type == "default_torchMHA":
            self.attn = nn.MultiheadAttention(
                embed_dim = transformer_block_d_model,
                num_heads = transformer_block_n_heads,
                batch_first = batch_first,
                **transformer_block_atttn_kwargs,
            )
        else:
            self.attn = Attention(
                embed_dim=transformer_block_d_model,
                num_heads = transformer_block_n_heads,
                attention_type = self.attention_type,
                batch_first = batch_first,
                **transformer_block_atttn_kwargs,
            )
        if transformer_block_use_conditioning:
            self.cross_scale = nn.Parameter(torch.tensor(transformer_block_condscale_startval))
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=transformer_block_d_model,
                num_heads=transformer_block_n_heads,
                dropout=transformer_block_dropout,
                batch_first=batch_first,
            )
            # project cond_x to model dimension if needed
            # self.cond_proj = (
            #     nn.Linear(transformer_block_cond_dim, transformer_block_d_model)
            #     if transformer_block_cond_dim is not None
            #     else nn.Identity()
            # )     
            self.cond_proj = nn.Linear(transformer_block_cond_dim, transformer_block_d_model)   

        self.mlp = MLP(
            input_dim=transformer_block_d_model,
            hidden_layers=transformer_block_mlp_hidden_layers,
            output_dim=transformer_block_d_model,
            **mlp_kwargs,
        )
        self.dropout = nn.Dropout(transformer_block_dropout)

    def forward(self, x, cond_x=None, attn_mask=None, key_padding_mask=None):
        # first part of the model, attention block
        x_norm = self.norm1(x)
        if self.attention_type == "default_torchMHA":
            attn_out, _ = self.attn(
                x_norm,
                x_norm,
                x_norm,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
        else:
            attn_out = self.attn(
                x_norm,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
            )           
        res1 = x + self.dropout(attn_out)

        # crossattention conditioning
        if self.use_conditioning:
            if cond_x is None:
                logger.warning("TransformerBlock configured with conditioning but cond_x is None — skipping cross-attention.")
            else:
                x_norm_cond = self.norm_cond(res1)

                cond = self.cond_proj(cond_x)
                # If cond is (B, D), expand to sequence
                if cond.dim() == 2:
                    cond = cond.unsqueeze(1)  # (B, 1, D)
                cross_attn_out, _ = self.cross_attn(
                    query=x_norm_cond,
                    key=cond,
                    value=cond,
                    need_weights=False,
                )
                res1 = res1 + self.dropout(cross_attn_out * self.cross_scale)
        # MLP 
        mlp_out = self.mlp(self.norm2(res1))
        # handle two different skip conection variants
        if self.residual_mode == "coupled":
            out = x + mlp_out
        elif self.residual_mode == "standard":
            out = res1 + mlp_out
        else:  
            # due to the earlier assert user should never actually get to this point
            logger.error(f"Unsupported residual_mode option '{self.residual_mode}'!")
            raise ValueError(f"Unsupported residual_mode option '{self.residual_mode}'!")
        return out


