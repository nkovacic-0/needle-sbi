import torch

from needle.ml.validation.validation_binary_classifier_src.model_validation_utils import resolve_weights

from needle.utils.logging import ColorFormatter

logger = ColorFormatter.get_logger("downstream-validation")


def _sample_bin_counts(bin_num_min: int, bin_num_max: int, binning_scheme_sampling_points: int) -> list[int]:
    """Shared across every averaged-* variant: binning_scheme_sampling_points
    bin counts, evenly spaced between bin_num_min and bin_num_max inclusive.
    Note: for a narrow range relative to sampling_points, rounding can produce
    duplicate bin counts that is not deduplicated here, so those bins can get double 
    weight in the resulting average. 
    """
    if bin_num_min < 1 or bin_num_max < bin_num_min:
        err_msg = f"Expected 1 <= bin_num_min <= bin_num_max, got bin_num_min={bin_num_min}, bin_num_max={bin_num_max}"
        logger.error(err_msg)
        raise ValueError(err_msg)
    if binning_scheme_sampling_points < 1:
        err_msg = f"binning_scheme_sampling_points must be >= 1, got {binning_scheme_sampling_points}"
        logger.error(err_msg)
        raise ValueError(err_msg)
    return torch.linspace(bin_num_min, bin_num_max, binning_scheme_sampling_points).round().long().tolist()

def _average_over_schemes(bin_counts: list[int], values: list[float], weight_by_bin_count: bool) -> float:
    """Shared averaging step for averaged_* variants.

    weight_by_bin_count=False (default): plain unweighted mean across schemes,
    each binning scheme's error already represents a fraction-of-total-weight
    quantity via the w_b/W term inside _binned_calibration_gap, so no
    additional scaling by n_bins is applied by default.
    """
    if weight_by_bin_count:
        inv_bins = [1.0 / n for n in bin_counts]
        return sum(w * v for w, v in zip(inv_bins, values)) / sum(inv_bins)
    return sum(values) / len(values)

def _bin_statistics(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    bin_idx: torch.Tensor,
    n_bins: int,
    context: str,
) -> dict:
    """Shared per-bin statistics for every binned calibration function in
    this module. Handles signed weights (common in HEP, also negative weights):

    - A bin whose NET total weight is <= 0 is discarded entirely. It is excluded
      from both the numerator (its outcome/confidence stay 0) AND the
      total_weight denominator. A warning fires naming how many bins this
      happened to, noting that any n_eff-based debiasing correction for
      those bins is correspondingly zeroed too.
    - A bin can still be KEPT (net weight > 0) while containing a MIX of
      positive- and negative-weight events. bin_outcome/bin_conf are only
      guaranteed convex combinations (i.e. bounded in [0, 1]) when every
      weight in a bin shares a sign. A separate warning fires if any kept
      bin's outcome or confidence lands outside [0, 1]. 

    Args:
        context: short string naming the calling function, used only in
            log messages.

    Returns:
        dict with keys: bin_target, bin_confidence, bin_weight (zeroed at
        discarded bins), kept, negative, total_weight, bin_outcome, bin_conf.
    """
    bin_target = torch.zeros(n_bins, dtype=torch.float).scatter_add_(0, bin_idx, labels * weights)
    bin_confidence = torch.zeros(n_bins, dtype=torch.float).scatter_add_(0, bin_idx, predictions * weights)
    bin_weight = torch.zeros(n_bins, dtype=torch.float).scatter_add_(0, bin_idx, weights)

    kept = bin_weight > 0
    negative = bin_weight < 0

    n_negative = int(negative.sum())
    if n_negative > 0:
        logger.warning(
            f"[{context}] {n_negative}/{n_bins} bin(s) have a negative NET total weight "
            "(common with signed/negative-weighted HEP samples) and were discarded entirely and "
            "excluded from the gap computation, the total-weight denominator, and (for functions "
            "that use one) the effective sample size n_eff."
        )

    total_weight = bin_weight[kept].sum()

    bin_outcome = torch.zeros_like(bin_weight)
    bin_conf = torch.zeros_like(bin_weight)
    bin_outcome[kept] = bin_target[kept] / bin_weight[kept]
    bin_conf[kept] = bin_confidence[kept] / bin_weight[kept]

    out_of_range = kept & ((bin_outcome < 0) | (bin_outcome > 1) | (bin_conf < 0) | (bin_conf > 1))
    n_out_of_range = int(out_of_range.sum())
    if n_out_of_range > 0:
        logger.warning(
            f"[{context}] {n_out_of_range}/{n_bins} kept bin(s) have a per-bin outcome or "
            "confidence average outside [0, 1]. This happens when a bin's NET weight is positive "
            "but contains a MIX of positive- and negative-weight events, so the weighted average "
            "is no longer a proper convex combination. Not corrected here."
        )

    bin_weight = torch.where(kept, bin_weight, torch.zeros_like(bin_weight))

    return {
        "bin_target": bin_target,
        "bin_confidence": bin_confidence,
        "bin_weight": bin_weight,
        "kept": kept,
        "negative": negative,
        "total_weight": total_weight,
        "bin_outcome": bin_outcome,
        "bin_conf": bin_conf,
    }


def _binned_p_calibration_gap(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    bin_idx: torch.Tensor,
    n_bins: int,
    p_metric: float = 1.0,
    signed: bool = False,
    context: str = "_binned_p_calibration_gap",
) -> torch.Tensor:
    """Shared aggregation step for every calibration-error variant in this
    module: given a per-event bin assignment, returns the weighted Lp
    calibration error:

        Lp-CE = ( sum_b (w_b / W) * |confidence_b - outcome_b|^p )^(1/p)

    p_metric=1.0 (default) reduces to the original |gap| aggregation with no
    extra pow/root ops; p_metric=2.0 etc. generalize to any p >= 1 norm.

    signed=True: confidence_b - outcome_b per bin,
        only validfor p_metric=1.0, since any even power destroys sign information
        Positive means overconfident, negative means underconfident.
    
    Note:
    signed-weights handling, i.e. bin discarding, out-of-range warnings and alike
        are delegated to _bin_statistics
    """
    if signed and p_metric != 1.0:
        err_msg = (
            f"signed=True is only valid for p_metric=1.0 (sign is not meaningful "
            f"under p={p_metric}); got p_metric={p_metric}"
        )
        logger.error(err_msg)
        raise ValueError(err_msg)
    if p_metric < 1.0:
        err_msg = f"p_metric must be >= 1.0 to form a valid norm, got {p_metric}"
        logger.error(err_msg)
        raise ValueError(err_msg)

    bin_target = torch.zeros(n_bins, dtype=torch.float).scatter_add_(0, bin_idx, labels * weights)
    bin_confidence = torch.zeros(n_bins, dtype=torch.float).scatter_add_(0, bin_idx, predictions * weights)
    bin_weight = torch.zeros(n_bins, dtype=torch.float).scatter_add_(0, bin_idx, weights)

    total_weight = bin_weight.sum()
    nonempty = bin_weight > 0

    bin_outcome = torch.zeros_like(bin_weight)
    bin_conf = torch.zeros_like(bin_weight)
    bin_outcome[nonempty] = bin_target[nonempty] / bin_weight[nonempty]
    bin_conf[nonempty] = bin_confidence[nonempty] / bin_weight[nonempty]

    gap = bin_conf - bin_outcome

    if signed:
        return (bin_weight * gap).sum() / total_weight

    if p_metric == 1.0:
        return (bin_weight * torch.abs(gap)).sum() / total_weight

    weighted_mean_p = (bin_weight * torch.abs(gap) ** p_metric).sum() / total_weight
    return weighted_mean_p ** (1.0 / p_metric)


def expected_calibration_error(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor | None = None,
    n_bins: int = 15,
    weighted: bool = True,
    p_metric: float = 1.0,
) -> float:
    """Expected Calibration Error (ECE): fixed-width bins over [0, 1].

    predictions are assumed to be probabilities in range [0, 1]

        ECE = sum_b (w_b / W) * |mean(p)_b - mean(target)_b|
    """
    if n_bins < 1:
        err_msg = f"n_bins must be >= 1, got {n_bins}"
        logger.error(err_msg)
        raise ValueError(err_msg)

    predictions = predictions.reshape(-1).float()
    labels = labels.reshape(-1).float()
    weights = resolve_weights(weights, predictions, weighted)

    bin_idx = torch.clamp((predictions * n_bins).long(), 0, n_bins - 1)
    ece = _binned_p_calibration_gap(predictions, labels, weights, bin_idx, n_bins, p_metric=p_metric, signed=False, context="expected_calibration_error")
    logger.debug(f"[expected_calibration_error] n_bins={n_bins} weighted={weighted} ece={ece.item():.6f}")
    return ece.item()


def adaptive_calibration_error(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor | None = None,
    n_bins: int = 15,
    weighted: bool = True,
    p_metric: float = 1.0,
) -> float:
    """Adaptive / equal-mass Calibration Error: bin edges are chosen so each
    bin holds equal total weight (or equal count, when weighted=False) rather 
    than equal-width over [0, 1]. Uses the same weighted-gap aggregation as 
    expected_calibration_error.

    Boundaries are found by sequential nearest-target search over a shrinking
    suffix of the score-sorted, cumulatively-summed weights. Searching
    sequentially guarantees boundary indices are non-decreasing by
    construction, so bins stay contiguous in score order regardless of local
    non-monotonicity.
    """
    if n_bins < 1:
        err_msg = f"n_bins must be >= 1, got {n_bins}"
        logger.error(err_msg)
        raise ValueError(err_msg)

    predictions = predictions.reshape(-1).float()
    labels = labels.reshape(-1).float()
    weights = resolve_weights(weights, predictions, weighted)

    sort_idx = torch.argsort(predictions)
    sorted_predictions = predictions[sort_idx]
    sorted_labels = labels[sort_idx]
    sorted_weights = weights[sort_idx]

    cum_weights = torch.cumsum(sorted_weights, dim=0)
    total_weight = cum_weights[-1]
    n_events = sorted_weights.shape[0]

    boundaries = []
    start = 0
    for k in range(1, n_bins):
        target = total_weight * k / n_bins
        offset = torch.argmin(torch.abs(cum_weights[start:] - target)).item()
        boundary = min(start + offset, n_events - 1)
        boundaries.append(boundary)
        start = min(boundary + 1, n_events - 1)

    bin_idx = torch.zeros(n_events, dtype=torch.long)
    prev_boundary = -1
    for bin_num, boundary in enumerate(boundaries):
        bin_idx[prev_boundary + 1 : boundary + 1] = bin_num
        prev_boundary = boundary
    bin_idx[prev_boundary + 1 :] = len(boundaries)

    ace = _binned_p_calibration_gap(
        sorted_predictions, sorted_labels, sorted_weights, bin_idx, n_bins,
        p_metric=p_metric, signed=False, context="adaptive_calibration_error",
    )
    logger.debug(f"[adaptive_calibration_error] n_bins={n_bins} weighted={weighted} ace={ace.item():.6f}")
    return ace.item()

def expected_signed_calibration_error(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor | None = None,
    n_bins: int = 15,
    weighted: bool = True,
) -> float:
    """Signed Expected Calibration Error: same binning/aggregation as
    expected_calibration_error, without the absolute value. 
    Note: SECE is not as informative on its own (positive and negative bin-level 
    errors can cancel out across the [0, 1] range). It is, however, useful compared 
    against expected_calibration_error where a large gap between the two indicates 
    systematic over/underconfidence that a symmetric metric like plain ECE would mask.

        signed ECE = sum_b (w_b / W) * (mean(p)_b - mean(target)_b)

    Positive means model is, on average, overconfident. Negative means underconfident.
    """
    if n_bins < 1:
        err_msg = f"n_bins must be >= 1, got {n_bins}"
        logger.error(err_msg)
        raise ValueError(err_msg)

    predictions = predictions.reshape(-1).float()
    labels = labels.reshape(-1).float()
    weights = resolve_weights(weights, predictions, weighted)

    bin_idx = torch.clamp((predictions * n_bins).long(), 0, n_bins - 1)
    signed_ece = _binned_p_calibration_gap(predictions, labels, weights, bin_idx, n_bins, p_metric=1.0, signed=True, context="expected_signed_calibration_error")
    logger.debug(f"[expected_signed_calibration_error] n_bins={n_bins} weighted={weighted} signed_ece={signed_ece.item():.6f}")
    return signed_ece.item()

def averaged_expected_calibration_error(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor | None = None,
    weighted: bool = True,
    bin_num_min: int = 20,
    bin_num_max: int = 200,
    binning_scheme_sampling_points: int = 10,
    weight_by_bin_count: bool = False,
    p_metric: float = 1.0,
) -> float:
    """Mean of expected_calibration_error() over several uniformly-spaced bin
    counts between bin_num_min and bin_num_max, instead of one fixed n_bins 
    reduces sensitivity to an arbitrary choice of bin count.
    weight_by_bin_count: see _average_over_schemes, it is off by default.
    """
    bin_counts = _sample_bin_counts(bin_num_min, bin_num_max, binning_scheme_sampling_points)
    values = [
        expected_calibration_error(predictions, labels, weights, n_bins=n_bins, weighted=weighted, p_metric=p_metric)
        for n_bins in bin_counts
    ]
    averaged = _average_over_schemes(bin_counts, values, weight_by_bin_count)

    logger.debug(
        f"[averaged_expected_calibration_error] bin_counts={bin_counts} weighted={weighted} "
        f"weight_by_bin_count={weight_by_bin_count} per_scheme={['%.6f' % v for v in values]} averaged={averaged:.6f}"
    )
    return averaged

def averaged_expected_signed_calibration_error(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor | None = None,
    weighted: bool = True,
    bin_num_min: int = 20,
    bin_num_max: int = 200,
    binning_scheme_sampling_points: int = 10,
    weight_by_bin_count: bool = False,
) -> float:
    """Signed-ECE counterpart to averaged_expected_calibration_error() 
    """
    bin_counts = _sample_bin_counts(bin_num_min, bin_num_max, binning_scheme_sampling_points)
    values = [
        expected_signed_calibration_error(predictions, labels, weights, n_bins=n_bins, weighted=weighted)
        for n_bins in bin_counts
    ]
    averaged = _average_over_schemes(bin_counts, values, weight_by_bin_count)

    logger.debug(
        f"[averaged_expected_signed_calibration_error] bin_counts={bin_counts} weighted={weighted} "
        f"per_scheme={['%.6f' % v for v in values]} averaged={averaged:.6f}"
    )
    return averaged


def debiased_squared_calibration_error(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor | None = None,
    n_bins: int = 15,
    weighted: bool = False,
) -> float:
    """Debiased estimator of the squared (L_2) calibration error, per Definition
    5.2 of Kumar, Liang & Ma, "Verified Uncertainty Calibration," NeurIPS 2019
    (arXiv:1909.10155). 

    The plugin estimator squares an already-noisy sample mean per bin, which
    is a biased estimator of the true squared gap (E[mean(y)^2] overshoots by
    the mean's own sampling variance). This subtracts the standard
    Bessel-corrected unbiased variance estimate to remove that bias:

        debiased_bin_b = (mean_conf_b - mean(y)_b)^2
                          - mean(y)_b * (1 - mean(y)_b) / (n_b - 1)
        CE2_debiased = sum_b (w_b / W) * debiased_bin_b

    Note:
    This can legitimately return a small negative value for finite samples and this
    is expected (as per Lane, "A Comprehensive Review of Classifier Probability
    Calibration Metrics," 2025, arXiv:2504.18278, Sec 4.10), not a bug.

    weighted=False (default): n_b is the RAW per-bin event count; the Bessel
        correction above is exact.
    weighted=True: n_b is replaced by Kish's effective sample size,
        n_eff_b = (sum_i w_i)^2 / sum_i (w_i^2)
        (Kish, "Survey Sampling," Wiley, 1965), note: this is speculative extension and 
        not derived or validated in the source paper (which assumes unweighted i.i.d.
        samples): exact when weights are literal repetition counts, an
        approximation for general importance-style weights, degrading as the
        weight distribution grows more skewed. n_eff reduces exactly to n_b
        when every weight is 1, so this is a strict generalization of the
        unweighted case, not a different formula.

    Returns:
        float: CE2_debiased. May be negative, see note above.
    """
    if n_bins < 1:
        err_msg = f"n_bins must be >= 1, got {n_bins}"
        logger.error(err_msg)
        raise ValueError(err_msg)

    predictions = predictions.reshape(-1).float()
    labels = labels.reshape(-1).float()
    weights = resolve_weights(weights, predictions, weighted)

    bin_idx = torch.clamp((predictions * n_bins).long(), 0, n_bins - 1)

    stats = _bin_statistics(predictions, labels, weights, bin_idx, n_bins, context="debiased_squared_calibration_error")
    bin_weight = stats["bin_weight"]
    kept = stats["kept"]
    total_weight = stats["total_weight"]
    bin_outcome = stats["bin_outcome"]
    bin_conf = stats["bin_conf"]

    if weighted:
        bin_weight_sq = torch.zeros(n_bins, dtype=torch.float).scatter_add_(0, bin_idx, weights ** 2)
        n_eff = torch.zeros(n_bins, dtype=torch.float)
        n_eff[kept] = bin_weight[kept] ** 2 / bin_weight_sq[kept]
    else:
        n_eff = torch.zeros(n_bins, dtype=torch.float).scatter_add_(0, bin_idx, torch.ones_like(predictions))
        n_eff = torch.where(kept, n_eff, torch.zeros_like(n_eff))

    squared_gap = (bin_conf - bin_outcome) ** 2

    correction = torch.zeros(n_bins, dtype=torch.float)
    debiasable = kept & (n_eff > 1)
    correction[debiasable] = (
        bin_outcome[debiasable] * (1.0 - bin_outcome[debiasable]) / (n_eff[debiasable] - 1.0)
    )

    n_undebiased = int((kept & ~debiasable).sum())
    if n_undebiased > 0:
        logger.warning(
            f"[debiased_squared_calibration_error] {n_undebiased}/{n_bins} kept bin(s) have "
            f"n_eff <= 1 (too few events to debias); using the plain (biased) squared gap for those bins."
        )
    debiased_bin = squared_gap - correction
    ce2_debiased = (bin_weight * debiased_bin).sum() / total_weight

    logger.debug(
        f"[debiased_squared_calibration_error] n_bins={n_bins} weighted={weighted} "
        f"ce2_debiased={ce2_debiased.item():.6f}"
    )
    return ce2_debiased.item()


def debiased_rmsce(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor | None = None,
    n_bins: int = 15,
    weighted: bool = False,
) -> float:
    """Debiased Root Mean Squared Calibration Error (DRMSCE): simply sqrt of
    debiased_squared_calibration_error(), with the cavat being that it is 
    clamped at 0 nefore sqrt as CE2_debiased can be negative for finite samples.

    Naming and the sqrt-of-debiased-CE2 formulation is from Petersen et al., "On
    (assessing) the fairness of risk score models," ACM FAccT 2023
    see their implementation of get_unbiased_calibration_rmse, 
    github.com/e-pet/risk-score-fairness
    """
    ce2_debiased = debiased_squared_calibration_error(
        predictions, labels, weights, n_bins=n_bins, weighted=weighted
    )
    if ce2_debiased < 0:
        logger.debug(f"[debiased_rmsce] CE2_debiased={ce2_debiased:.6f} < 0; clamped to 0 before sqrt.")
    return float(max(ce2_debiased, 0.0) ** 0.5)

def debiased_expected_calibration_error(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor | None = None,
    n_bins: int = 15,
    weighted: bool = False,
    n_mc_samples: int = 1000,
    random_seed: int = 42,
) -> float:
    """Debiased estimator of the L1 (ordinary) Expected Calibration Error, per
    Section G.3 of Kumar, Liang & Ma, "Verified Uncertainty Calibration,"
    NeurIPS 2019 (arXiv:1909.10155).

    Paper's notation, mapped to this function's variables:
        n     = predictions.numel() (weighted: total weight)
        L_k   = per-bin count (this function's n_eff; L_k when weighted=False)
        p_k   = L_k / n (this function's bin_weight / total_weight)
        y_k   = per-bin empirical label average (this function's bin_outcome)
        s_k   = per-bin empirical score average (this function's bin_conf)
        R_k  ~ N(y_k, y_k*(1-y_k) / (n*p_k))  ==  N(y_k, y_k*(1-y_k) / L_k)
              (this function's per-bin Gaussian, using n_eff in place of L_k
              when weighted=True, see the weighted/n_eff note below)

    We calculate:
        ece      = sum_k p_k * |s_k - y_k|                   
        ece_DB   = 2*ece - E[sum_k p_k * |s_k - R_k|]

    The expectation E[...] has no closed form (unlike the squared-error case,
    debiased_squared_calibration_error) and is approximated here by Monte
    Carlo sampling from each bin's R_k, n_mc_samples draws per bin.

    weighted/n_eff: same construction, and the same "speculative" caveat, as
    debiased_squared_calibration_error -- see that function's docstring.

    random_seed: Monte Carlo sampling is otherwise non-deterministic; fixed
    by default so repeated validation runs are reproducible.

    Returns:
        float: ECE_debiased. Like debiased_squared_calibration_error, can
            come out negative for finite samples -- not clamped here.
    """
    if n_bins < 1:
        err_msg = f"n_bins must be >= 1, got {n_bins}"
        logger.error(err_msg)
        raise ValueError(err_msg)
    if n_mc_samples < 1:
        err_msg = f"n_mc_samples must be >= 1, got {n_mc_samples}"
        logger.error(err_msg)
        raise ValueError(err_msg)

    predictions = predictions.reshape(-1).float()
    labels = labels.reshape(-1).float()
    resolved_weights = resolve_weights(weights, predictions, weighted)

    ece_plugin = expected_calibration_error(predictions, labels, weights, n_bins=n_bins, weighted=weighted)

    bin_idx = torch.clamp((predictions * n_bins).long(), 0, n_bins - 1)

    stats = _bin_statistics(predictions, labels, resolved_weights, bin_idx, n_bins, context="debiased_expected_calibration_error")
    bin_weight = stats["bin_weight"]
    kept = stats["kept"]
    total_weight = stats["total_weight"]
    bin_outcome = stats["bin_outcome"]
    bin_conf = stats["bin_conf"]

    if weighted:
        bin_weight_sq = torch.zeros(n_bins, dtype=torch.float).scatter_add_(0, bin_idx, resolved_weights ** 2)
        n_eff = torch.zeros(n_bins, dtype=torch.float)
        n_eff[kept] = bin_weight[kept] ** 2 / bin_weight_sq[kept]
    else:
        n_eff = torch.zeros(n_bins, dtype=torch.float).scatter_add_(0, bin_idx, torch.ones_like(predictions))
        n_eff = torch.where(kept, n_eff, torch.zeros_like(n_eff))

    sigma = torch.zeros(n_bins, dtype=torch.float)
    sigma[kept] = torch.sqrt(
        torch.clamp(bin_outcome[kept] * (1.0 - bin_outcome[kept]) / n_eff[kept].clamp(min=1e-12), min=0.0)
    )

    generator = torch.Generator().manual_seed(random_seed)
    noise = torch.randn(n_bins, n_mc_samples, generator=generator)
    samples = bin_outcome.unsqueeze(1) + sigma.unsqueeze(1) * noise
    mc_gap = torch.abs(bin_conf.unsqueeze(1) - samples).mean(dim=1)

    expected_plugin = (bin_weight * mc_gap).sum() / total_weight

    bias_estimate = expected_plugin - ece_plugin
    ece_debiased = ece_plugin - bias_estimate

    logger.debug(
        f"[debiased_expected_calibration_error] n_bins={n_bins} weighted={weighted} "
        f"n_mc_samples={n_mc_samples} ece_plugin={ece_plugin:.6f} "
        f"expected_plugin={expected_plugin.item():.6f} ece_debiased={ece_debiased.item():.6f}"
    )
    return ece_debiased.item()