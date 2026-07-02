import torch
import torch.nn as nn

from needle.utils.logging import ColorFormatter

logger = ColorFormatter.get_logger("ml")


class Pooling(nn.Module):
    def __init__(
        self,
        d_model: int,
        pool_type: str = "mean",
        dropout: float = 0.0,
    ):
        super().__init__()

        pool_type = pool_type.lower()
        _supported = ["mean", "max", "cls", "last", "attention"]
        if pool_type not in _supported:
            err_msg = f"Unsupported pooling mode '{pool_type}'. Available modes are: {_supported}"
            logger.error(err_msg)
            raise ValueError(err_msg)
        self.pool_type = pool_type

        self.dropout = nn.Dropout(dropout)

        if pool_type == "attention":
            self.score = nn.Linear(d_model, 1)

    def forward(self, x, mask=None):
        """
        x: (B, T, D) i.e. (B, P, F) in our case
        mask: None or
              (B, T) with true for nan/invalid
              (B, T, D) with true for nan/invalid (as you'd get from isnan method)
        """
        mask = self._fix_mask(mask, x)
        if self.pool_type != "attention":
            x = self.dropout(x)
        if self.pool_type == "mean":
            return self._mean(x, mask)
        elif self.pool_type == "max":
            return self._max(x, mask)
        elif self.pool_type == "cls":
            # here we only ASSUME that the CLS token was added to pos 0!
            return x[:, 0]
        elif self.pool_type == "last":
            # last is irrelevant for what we're doing here to be honest, kept it in for now
            return self._last(x, mask)
        elif self.pool_type == "attention":
            return self._attention(x, mask)

    def _mean(self, x, mask):
        if mask is None:
            return x.mean(dim=1)
        mask = mask.to(dtype=x.dtype).unsqueeze(-1)  # (B, T, 1)
        summed = (x * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        return summed / denom

    def _max(self, x, mask):
        if mask is None:
            return torch.amax(x, dim=1)
        # mask invalid tokens
        neg_inf = torch.finfo(x.dtype).min
        x = x.masked_fill(mask.unsqueeze(-1) == 0, neg_inf)
        return torch.amax(x, dim=1)

    def _last(self, x, mask):
        # if mask is None:
        #     return x[:, -1]
        # # compute last valid index per sequence
        # lengths = mask.sum(dim=1).long().clamp(min=1)
        # idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, x.size(-1))
        # return x.gather(1, idx).squeeze(1)
        if mask is None:
            lengths = torch.full((x.size(0),), x.size(1), dtype=torch.long, device=x.device)
        else:
            lengths = mask.sum(dim=1).long().clamp(min=1)
        idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, x.size(-1))
        return x.gather(1, idx).squeeze(1)

    def _attention(self, x, mask):
        scores = self.score(x)  # (B, T, 1) 
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(-1) == 0, float("-inf"))
        weights = torch.softmax(scores, dim=1)
        weights = torch.nan_to_num(weights, nan=0.0)
        return self.dropout((weights * x).sum(dim=1))
        
    def _fix_mask(self, mask: torch.Tensor | None, x: torch.Tensor) -> torch.Tensor | None:
        if mask is None:
            return None
        if mask.dim() == 3:
            # (B, T, E) i.e. (B, P, F) isnan-like mask: True = NaN/invalid
            # collapse feature dim and invert so 1 = valid, 0 = padding
            mask = ~mask.any(dim=-1) # (B, T), and is bool
        if mask.dtype == torch.bool:
            # incoming convention: True=invalid/padding (matches torch.isnan and nn.MHA)
            # inverting it here so pooling methods get 1 = valid, 0 = padding
            mask = (~mask).float()
        return mask