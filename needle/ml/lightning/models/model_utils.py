import torch
from torchmetrics import Metric

from needle.utils.logging import ColorFormatter

logger = ColorFormatter.get_logger("ml")

def unwrap_labels(labels: dict[str, torch.Tensor]) -> torch.Tensor | dict[str, torch.Tensor]:
    """Unwrap a single-entry labels dict into a plain tensor.

    Keeps `_shared_step` and similar training code agnostic to the label
    column's name for the common single-label case, without requiring
    `collate_fn` to special-case it. Multi-label dicts pass through unchanged.
    """
    if len(labels) == 1:
        return next(iter(labels.values()))
    return labels

class WeightedBinaryAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("weight_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds, target, weights):
        preds = (torch.sigmoid(preds) > 0.5).int()
        correct = (preds == target.int()).float()

        self.correct += (correct * weights).sum()
        self.weight_sum += weights.sum()

    def compute(self):
        return self.correct / self.weight_sum
    
class ExpectedCalibrationError(Metric):
    """Expected Calibration Error (ECE) for binary classifiers.

    By default, bins are formed on the predicted score interpreted as 
    probability p, then each bin compares its mean predicted probability
    against its empirical positive rate:

        ECE = sum_b (w_b / W) * |mean(p)_b - mean(target)_b|

    Args:
        n_bins: number of equal-width bins. Spans [0, 1] 
        weighted: if True, `update()` requires a `weights` tensor and it is
            used both for bin mass and for weighting each point's
            contribution within its bin. If False, `weights` is ignored
            (need not be passed, and is treated as all-ones if it is).
    """

    def __init__(self, n_bins: int = 15, weighted: bool = False):
        super().__init__()
        if n_bins < 1:
            err_msg = (f"n_bins must be >= 1, got {n_bins}")
            logger.error(err_msg)
            raise ValueError(err_msg)
        self.n_bins = n_bins
        self.weighted = weighted

        self.add_state(
            "bin_target", default=torch.zeros(n_bins), dist_reduce_fx="sum"
        )
        self.add_state(
            "bin_confidence", default=torch.zeros(n_bins), dist_reduce_fx="sum"
        )
        self.add_state(
            "bin_weight", default=torch.zeros(n_bins), dist_reduce_fx="sum"
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor, weights: torch.Tensor | None = None):
        if self.weighted:
            if weights is None:
                err_msg = (
                    "ExpectedCalibrationError was constructed with weighted=True, "
                    "so 'weights' must be provided to update()."
                )
                logger.error(err_msg)
                raise ValueError(err_msg)
            weights = weights.float().reshape(-1)
        else:
            weights = torch.ones_like(preds, dtype=torch.float).reshape(-1)

        probits = torch.sigmoid(preds).reshape(-1).float()
        target = target.reshape(-1).float()
        # bins over [0.0, 1.0]
        bin_idx = torch.clamp(
            (probits * self.n_bins).long(), 0, self.n_bins - 1
        )
        self.bin_target.scatter_add_(0, bin_idx, target * weights)
        self.bin_confidence.scatter_add_(0, bin_idx, probits * weights)
        self.bin_weight.scatter_add_(0, bin_idx, weights)

    def compute(self) -> torch.Tensor:
        total_weight = self.bin_weight.sum()
        nonempty = self.bin_weight > 0

        bin_outcome = torch.zeros_like(self.bin_weight)
        bin_conf = torch.zeros_like(self.bin_weight)
        bin_outcome[nonempty] = self.bin_target[nonempty] / self.bin_weight[nonempty]
        bin_conf[nonempty] = self.bin_confidence[nonempty] / self.bin_weight[nonempty]

        gap = torch.abs(bin_outcome - bin_conf)
        ece = (self.bin_weight * gap).sum() / total_weight
        return ece