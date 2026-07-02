import torch
import torch.nn as nn
import torch.nn.functional as F

from needle.utils.logging import ColorFormatter

logger = ColorFormatter.get_logger("ml")

# this class is completely untested and should not be used, for now as there might be a few more open issues with it
class Attention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attention_type: str = "sdpa",
        dropout: float = 0.0,
        num_kv_heads: int | None = None,
        bias: bool = True,
        batch_first: bool = True,
    ):
        super().__init__()

        # parse inputs and validate them
        _supported = {"mha", "sdpa", "gqa"}
        if attention_type not in _supported:
            err_msg = (
                f"Unsupported attention_type '{attention_type}'. "
                f"Expected one of: {', '.join(sorted(_supported))}"
            )
            logger.error(err_msg)
            raise ValueError(err_msg)
        self.attention_type = attention_type
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.batch_first = batch_first
        self.dropout_p = dropout
        self.dropout  = nn.Dropout(dropout)
        (self.attn, self.q_proj, self.k_proj, self.num_kv_heads,
         self.v_proj, self.out_proj, self.head_dim, ) = (None,) * 7

        if attention_type == "mha":
            # fully implemented by torch
            self.attn = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                bias=bias,
                batch_first=batch_first,
            )
            return

        # SDPA / GQA uses only projections + torch SDPA kernel
        # check if we're not doing batch first, our implementation supposes that to be true
        if not batch_first:
            err_msg = (
                "SDPA and GQA attention types require batch_first=True. "
                "Either set batch_first=True or use attention_type='mha', "
                "which handles both layouts natively."
            )
            logger.error(err_msg)
            raise ValueError(err_msg)
        # prepare dims, check them
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        )
        self.num_kv_heads = num_kv_heads or num_heads
        assert num_heads % self.num_kv_heads == 0, (
            f"num_heads ({num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads})"
        )

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        # key_padding_mask: (B, S) i.e. (B, T) i.e. (B, P) bool with True=ignore, matching torch.isnan convention.
        #                   Float masks are also accepted with states 1=valid, 0=padding

        # no need to use any custom logic with the torch-native MHA
        if self.attention_type == "mha":
            out, _ = self.attn(
                x, x, x,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            return out
        # SDPA / FlashAttention path will need custom definitions
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q, k, v = self._reshape_for_sdpa(q, k, v)
        # hard-define the attn mask
        attn_mask = self._merge_masks(attn_mask, key_padding_mask, q)
        # expand KV heads if doing GQA
        if self.num_kv_heads != self.num_heads:
            repeat = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        if attn_mask is not None:
            attn_mask = self._validate_and_convert_mask(attn_mask, q)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False,
        )

        return self.out_proj(self._merge_heads(out))

    def _reshape_for_sdpa(self, q, k, v):
        # (B, T, E) i.e. (B, P, F) -> (B, H, T, D)
        b, t, _ = q.shape
        d = self.embed_dim // self.num_heads

        q = q.view(b, t, self.num_heads, d).transpose(1, 2)
        k = k.view(b, t, self.num_kv_heads, d).transpose(1, 2)
        v = v.view(b, t, self.num_kv_heads, d).transpose(1, 2)

        return q, k, v

    def _merge_heads(self, x):
        # (B, H, T, D) -> (B, T, E) i.e. (B, P, F)
        b, h, t, d = x.shape
        return x.transpose(1, 2).contiguous().view(b, t, h * d)

    def _merge_masks(
        self,
        attn_mask: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None,
        q: torch.Tensor,
    ) -> torch.Tensor | None:
        # expected convention for padding mask: isnan-like mask: True = NaN/invalid
        if key_padding_mask is None:
            return attn_mask
        # fix the mask so that it becomes bool with True = should be masked OUT (i.e. is padding)
        if key_padding_mask.dtype != torch.bool:
            key_padding_mask = key_padding_mask == 0

        b, _, t, s = q.shape[0], q.shape[1], q.shape[2], key_padding_mask.shape[1]
        # (B, S) -> (B, 1, 1, S), then convert to additive float mask
        kpm = key_padding_mask[:, None, None, :].expand(b, self.num_heads, t, s)
        kpm_float = torch.zeros_like(kpm, dtype=q.dtype).masked_fill(kpm, float("-inf"))
        if attn_mask is None:
            return kpm_float
        # both exist: add them together (both are additive float masks at this point
        # after _validate_and_convert_mask will run on attn_mask immediately after)
        if attn_mask.dtype == torch.bool:
            attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype).masked_fill(
                attn_mask, float("-inf")
            )
        return attn_mask + kpm_float

    @staticmethod
    def _validate_and_convert_mask(
        attn_mask: torch.Tensor,
        q: torch.Tensor,
    ) -> torch.Tensor:
        b, h, t, _ = q.shape
        if attn_mask.dtype == torch.bool:
            attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype).masked_fill(
                attn_mask, float("-inf")
            )
        if attn_mask.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            err_msg = (
                f"attn_mask must be bool or a float dtype (float16, bfloat16, float32), "
                f"got {attn_mask.dtype}"
            )
            logger.error(err_msg)
            raise TypeError(err_msg)
        # check it is broadcastable to (B, H, T, S)
        if attn_mask.dim() == 2:
            # (T, S) — valid, will broadcast across B and H
            pass
        elif attn_mask.dim() == 4:
            mb, mh, mt, _ = attn_mask.shape
            if mb not in (1, b) or mh not in (1, h) or mt not in (1, t):
                err_msg = (
                    f"4D attn_mask shape {tuple(attn_mask.shape)} is not broadcastable "
                    f"to (B={b}, H={h}, T={t}, S)"
                )
                logger.error(err_msg)
                raise ValueError(err_msg)
        else:
            logger.error(f"attn_mask must be 2D (T, S) or 4D (B, H, T, S), got {attn_mask.dim()}D")
            raise ValueError(f"attn_mask must be 2D (T, S) or 4D (B, H, T, S), got {attn_mask.dim()}D")
        return attn_mask