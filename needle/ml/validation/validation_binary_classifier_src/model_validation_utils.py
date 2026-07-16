import numpy as np
import torch

from needle.ml.lightning.models.model_utils import unwrap_labels

from needle.utils.logging import ColorFormatter
logger = ColorFormatter.get_logger("downstream-validation")

def collect_predictions(
    model_chunk: list[dict],
    dataloader,
    total_batches: int | None = None,
) -> tuple[list[dict], torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Single streaming pass over `dataloader`, evaluating every model in
    `model_chunk` against each batch as it streams by. This amortizes the test
    file's I/O cost across the whole chunk instead of once per model.

    Args:
        model_chunk: [{"model_path": str, "results_savepath": str, "model": nn.Module}, ...]
            "model" must already be .eval()'d and on its target device.
        dataloader: yields (features, labels, weights) or
            (features, labels, weights, aux) batches.
        total_batches: if known, enables a better progress log every 10% progress
            towards finishing the evaluations, otherwise every 1000 batches are reported

    Returns:
        chunk_results: [{"model_path": ..., "results_savepath": ..., "model_predictions": Tensor}, ...],
            same order as model_chunk.
        labels, weights: (N,) Tensors, shared across every model (same test set).
        aux_features: dict[str, Tensor], shared, already reverted to physical units.
    """
    for entry in model_chunk:
        entry["model"].eval()

    devices = [next(entry["model"].parameters()).device for entry in model_chunk]
    # group model indices by device so a batch is transferred once per device,
    # not once per model as most chunks share a single device in practice
    indices_by_device: dict[torch.device, list[int]] = {}
    for i, device in enumerate(devices):
        indices_by_device.setdefault(device, []).append(i)

    logger.info(
        f"[collect_predictions] Evaluating {len(model_chunk)} model(s) across "
        f"{len(indices_by_device)} device(s): {[str(d) for d in indices_by_device]}"
    )

    score_chunks: list[list[torch.Tensor]] = [[] for _ in model_chunk]
    label_chunks, weight_chunks = [], []
    aux_chunks: dict[str, list[torch.Tensor]] = {}

    n_batches = 0
    batch_interval_reporting = 1000
    last_reported_decile = 0  
    with torch.no_grad():
        for batch in dataloader:
            n_batches += 1

            if total_batches is not None:
                current_decile = min(n_batches * 10 // total_batches, 10)
                if current_decile > last_reported_decile:
                    last_reported_decile = current_decile
                    logger.info(
                        f"[collect_predictions] Processed {n_batches}/{total_batches} batches "
                        f"({100 * n_batches / total_batches:.1f}%)"
                    )
            else:
                if n_batches % batch_interval_reporting == 0:
                    logger.info(f"[collect_predictions] Processed {n_batches} batch(es)...")

            features, labels, weights, *aux = batch
            aux = aux[0] if aux else {}

            labels = unwrap_labels(labels)
            if not isinstance(labels, torch.Tensor):
                raise ValueError(
                    f"Expected a single label column for binary classification, got: {list(labels)}"
                )

            logger.debug(
                f"[collect_predictions] batch {n_batches}: features.shape={tuple(features.shape)}, "
                f"aux_fields={list(aux.keys())}"
            )

            label_chunks.append(labels.cpu())
            weight_chunks.append(weights.cpu())
            for name, values in aux.items():
                aux_chunks.setdefault(name, []).append(values.cpu())

            for device, indices in indices_by_device.items():
                device_features = features.to(device)
                for i in indices:
                    logits = model_chunk[i]["model"](device_features).squeeze(-1)
                    score_chunks[i].append(torch.sigmoid(logits).cpu())

    total_events = sum(t.shape[0] for t in label_chunks)
    logger.info(
        f"[collect_predictions] Done: {n_batches} batch(es), {total_events} event(s), "
        f"{len(model_chunk)} model(s) evaluated."
    )

    chunk_results = [
        {
            "model_path": entry["model_path"],
            "results_savepath": entry["results_savepath"],
            "model_predictions": torch.cat(score_chunks[i]),
        }
        for i, entry in enumerate(model_chunk)
    ]

    return (
        chunk_results,
        torch.cat(label_chunks),
        torch.cat(weight_chunks),
        {name: torch.cat(chunks) for name, chunks in aux_chunks.items()},
    )


_REQUIRED_STR_KEYS = ("model_class", "multiprocessing_type",)
_REQUIRED_POSITIVE_INT_KEYS = ("batch_size", "NN_model_validation_chunk_size",)
_REQUIRED_NONNEGATIVE_INT_KEYS = ("n_workers",)  # 0 is valid -- single-process loading, e.g. GroupedDaskDataset
_ALLOWED_MULTIPROCESSING_TYPES = {"torch", "dask"}


def check_validation_configs(validation_configs: dict) -> dict:
    """Validate the 6 scalar validation_configs entries plus validation_settings,
    then fold aux_feature_fields into a plain, mutable copy of validation_settings.

    Every entry except aux_feature_fields must be present and non-empty (None,
    "", or a non-positive int where positivity is required, while n_workers is
    the only one int allowed to be 0. aux_feature_fields is the sole exception to
    the "no None" rule, there None or an empty list both mean "no auxiliary features
    requested," not an error.

    Returns:
        dict: finalized validation_settings (aux_feature_fields folded in if set).
    """
    required_keys = _REQUIRED_STR_KEYS + _REQUIRED_POSITIVE_INT_KEYS + _REQUIRED_NONNEGATIVE_INT_KEYS + ("validation_settings",)
    missing = [k for k in required_keys if k not in validation_configs]
    if missing:
        err_msg = f"validation_configs is missing required key(s): {missing}"
        logger.error(err_msg)
        raise ValueError(err_msg)

    for key in _REQUIRED_STR_KEYS:
        value = validation_configs[key]
        if not isinstance(value, str) or not value:
            err_msg = f"validation_configs['{key}'] must be a non-empty str, got: {value!r}"
            logger.error(err_msg)
            raise ValueError(err_msg)

    if validation_configs["multiprocessing_type"] not in _ALLOWED_MULTIPROCESSING_TYPES:
        err_msg = (
            f"validation_configs['multiprocessing_type']={validation_configs['multiprocessing_type']!r} "
            f"not in {_ALLOWED_MULTIPROCESSING_TYPES}"
        )
        logger.error(err_msg)
        raise ValueError(err_msg)

    # isinstance(x, int) is True for bool too, so it is excluded explicitly in order for a stray
    # `batch_size: true` in YAML doesn't silently pass as a dtype check.
    for key in _REQUIRED_POSITIVE_INT_KEYS:
        value = validation_configs[key]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            err_msg = f"validation_configs['{key}'] must be a positive int, got: {value!r}"
            logger.error(err_msg)
            raise ValueError(err_msg)

    for key in _REQUIRED_NONNEGATIVE_INT_KEYS:
        value = validation_configs[key]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            err_msg = f"validation_configs['{key}'] must be a non-negative int, got: {value!r}"
            logger.error(err_msg)
            raise ValueError(err_msg)

    aux_feature_fields = validation_configs.get("aux_feature_fields")
    if aux_feature_fields is not None:
        if not isinstance(aux_feature_fields, (list, tuple)) or not all(isinstance(f, str) for f in aux_feature_fields):
            err_msg = f"validation_configs['aux_feature_fields'] must be a list of str or None, got: {aux_feature_fields!r}"
            logger.error(err_msg)
            raise ValueError(err_msg)

    validation_settings = validation_configs["validation_settings"]
    if not isinstance(validation_settings, dict):
        err_msg = f"validation_configs['validation_settings'] must be a dict, got: {type(validation_settings)}"
        logger.error(err_msg)
        raise ValueError(err_msg)

    validation_settings = dict(validation_settings)  # plain, mutable copy

    if aux_feature_fields:
        validation_settings["aux_feature_fields"] = list(aux_feature_fields)
        logger.debug(f"[check_validation_configs] Folded aux_feature_fields into validation_settings: {list(aux_feature_fields)}")
    else:
        logger.info("[check_validation_configs] aux_feature_fields is empty/None; no auxiliary features requested.")

    return validation_settings

def resolve_weights(weights: torch.Tensor | None, predictions: torch.Tensor, weighted: bool) -> torch.Tensor:
    """Shared across every ECE variant: weighted=True requires real weights;
    weighted=False treats every event as equally weighted (ones).
    """
    if weighted:
        if weights is None:
            err_msg = "weighted=True requires 'weights' to be provided."
            logger.error(err_msg)
            raise ValueError(err_msg)
        return weights.reshape(-1).float()
    return torch.ones_like(predictions)

def correct_prior_shift(scores: torch.Tensor, real_class_weight_sums: dict, epsilon: float = 1e-7) -> torch.Tensor:
    """Corrects scores trained under an implicit 50/50 class prior (per-class
    weight normalization) to be valid probabilities under the real class
    prior, per Saerens, Latinne & Decaestecker (2002), simplified to a
    constant log-odds shift since the training prior is was 50/50:

        s_corrected = sigmoid(logit(s) + log(W_real,1 / W_real,0))

    Args:
        scores: raw model scores (sigmoid probabilities), any shape.
        real_class_weight_sums: {"class_1": float, "class_0": float} 
            the real total weight per class over the full data set.
        epsilon: clamps scores to [epsilon, 1-epsilon] before the logit
            transform, to avoid inf/nan at s=0 or s=1.

    Returns:
        torch.Tensor: corrected scores, same shape as input.
    """
    w1 = real_class_weight_sums["class_1"]
    w0 = real_class_weight_sums["class_0"]
    if w1 <= 0 or w0 <= 0:
        err_msg = f"correct_prior_shift: real_class_weight_sums must both be positive, got {real_class_weight_sums}"
        logger.error(err_msg)
        raise ValueError(err_msg)

    log_k = torch.log(torch.tensor(w1 / w0, dtype=scores.dtype, device=scores.device))
    return torch.sigmoid(torch.logit(scores, eps=epsilon) + log_k)

def compute_probability_ratio(score: torch.Tensor, epsilon: float = 1e-5) -> torch.Tensor:
    """Probability ratio trick mechanism: (1-s)/s, clamped for numerical stability. 
    """
    score_clamped = torch.clamp(score, min=epsilon, max=1.0 - epsilon)
    return (1.0 - score_clamped) / score_clamped


def weighted_bin_statistics(
    values: np.ndarray,
    weights: np.ndarray,
    bin_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-bin weighted mean, weighted standard deviation, and total weight of
    `values`, for len(bin_edges)-1 bins. Companion to
    weighted_histogram_with_error -- that computes per-bin COUNTS/density;
    this computes per-bin POSITION (the weighted mean and spread of the
    values actually landing in each bin), needed when a bin's plotted
    x-position should reflect where its events truly sit rather than an
    assumed bin-center.

    - A bin whose NET total weight is <= 0 is discarded entirely. 
      A warning fires naming how many bins this happened to.
    - A bin will still be kept (net weight > 0) while containing a MIX of
      positive- and negative-weight events. 

    Args:
        values: (N,) raw, unbinned values (e.g. raw model scores).
        weights: (N,) corresponding event weights, can be signed.
        bin_edges: (n_bins + 1,) monotonically increasing bin edges.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: (weighted_mean,
            weighted_std, total_weight), each shape (n_bins,). mean/std are
            NaN for any discarded bin; total_weight never is.
    """
    n_bins = len(bin_edges) - 1
    bin_idx = np.clip(np.digitize(values, bin_edges[1:-1], right=False), 0, n_bins - 1)

    bin_weight = np.zeros(n_bins)
    bin_weighted_value = np.zeros(n_bins)
    bin_weighted_sq_value = np.zeros(n_bins)
    np.add.at(bin_weight, bin_idx, weights)
    np.add.at(bin_weighted_value, bin_idx, weights * values)
    np.add.at(bin_weighted_sq_value, bin_idx, weights * values ** 2)

    kept = bin_weight > 0
    negative = bin_weight < 0
    n_negative = int(negative.sum())
    if n_negative > 0:
        logger.warning(
            f"[weighted_bin_statistics] {n_negative}/{n_bins} bin(s) have a negative NET "
            "total weight and were discarded from mean/std calculation - weighted mean/std "
            "for those bins are NaN. total_weight is returned as-is (not NaN'd) for all bins."
        )

    mean = np.full(n_bins, np.nan)
    std = np.full(n_bins, np.nan)

    mean[kept] = bin_weighted_value[kept] / bin_weight[kept]
    variance = np.zeros(n_bins)
    variance[kept] = bin_weighted_sq_value[kept] / bin_weight[kept] - mean[kept] ** 2
    variance = np.clip(variance, a_min=0.0, a_max=None)
    std[kept] = np.sqrt(variance[kept])

    return mean, std, bin_weight