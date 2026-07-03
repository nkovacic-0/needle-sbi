import torch

def unwrap_labels(labels: dict[str, torch.Tensor]) -> torch.Tensor | dict[str, torch.Tensor]:
    """Unwrap a single-entry labels dict into a plain tensor.

    Keeps `_shared_step` and similar training code agnostic to the label
    column's name for the common single-label case, without requiring
    `collate_fn` to special-case it. Multi-label dicts pass through unchanged.
    """
    if len(labels) == 1:
        return next(iter(labels.values()))
    return labels