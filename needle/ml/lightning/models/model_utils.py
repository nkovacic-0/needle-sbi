import torch
from torchmetrics import Metric


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