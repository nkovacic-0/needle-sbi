import torch

from needle.ml.validation.validation_binary_classifier_src.model_validation_utils import resolve_weights
from needle.utils.logging import ColorFormatter

logger = ColorFormatter.get_logger("downstream-validation")


def brier_score(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor | None = None,
    weighted: bool = True,
) -> float:
    """Brier score for binary classification: mean squared error between
    predicted probability and the binary outcome.

        BS = sum_i (w_i / W) * (p_i - y_i)^2

    Lower is better, with 0 being a perfect score, 1 is the worst possible 
    (fully confident and always wrong).
    
    Predictions p_i are assumed to be probabilities in range [0, 1].

    Reference: Brier, G.W. (1950). "Verification of forecasts expressed in
    terms of probability." Monthly Weather Review, 78(1), 1-3.

    Args:
        predictions: (N,) predicted probabilities in [0, 1].
        labels: (N,) binary targets (0/1).
        weights: (N,) event weights. Required if weighted=True 
                    ignored (may be omitted) if weighted=False.
        weighted: if True, each event's squared error is weighted by its
            event weight. If False, every event counts equally.

    Returns:
        float: the Brier score.
    """
    predictions = predictions.reshape(-1).float()
    labels = labels.reshape(-1).float()
    weights = resolve_weights(weights, predictions, weighted)

    squared_error = (predictions - labels) ** 2
    score = (weights * squared_error).sum() / weights.sum()

    logger.debug(f"[brier_score] weighted={weighted} brier_score={score.item():.6f}")
    return score.item()

def entropic_calibration_difference(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor | None = None,
    weighted: bool = True,
    epsilon: float = 1e-7,
) -> float:
    """Entropic Calibration Difference (ECD) for binary classification, per
    Sumler, Devlin, Maskell & Lane, "An Entropic Metric for Measuring
    Calibration of Machine Learning Models," 2025 (arXiv:2502.14545), eq. 15.

        ECD = (1/N) * sum_i (p_i - y_i) * log(p_i / (1 - p_i))
    
    Predictions p_i are assumed to be probabilities in range [0, 1].

    Unlike expected_signed_calibration_error, ECD is deliberately ASYMMETRIC:
    inspired by "safe" state-estimation scoring (Normalised Estimation Error
    Squared / NEES from target tracking), it penalizes overconfidence in the
    wrong class far more heavily than underconfidence in the right one
    Per the paper it is roughly bounded below at -0.2785 for a single datum, but
    unbounded above. A score of 0 results from EITHER perfect calibration OR
    perfect uncertainty (p=0.5 regardless of label), a near-zero ECD alone
    doesn't distinguish the two. It is then compare against 
    expected_calibration_error/brier_score to tell them apart.

    Positive means overconfident (on average). Negative means underconfident.

    The paper does not define a weighted variant (all reported results are
    unweighted), so it is extended here the same way as brier_score, via 
    moving to a simple weighted sum.

    Args:
        predictions: (N,) predicted probabilities in [0, 1].
        labels: (N,) binary targets (0/1).
        weights: (N,) event weights. Required if weighted=True; ignored (may
            be omitted) if weighted=False.
        weighted: if True, each event's term is weighted by its event weight.
            If False, every event counts equally, matching the paper exactly.
        epsilon: clamps predictions to [epsilon, 1-epsilon] before taking the 
            log-odds, to avoid inf/nan at p=0 or p=1.

    Returns:
        float: the ECD score.
    """
    predictions = predictions.reshape(-1).float()
    labels = labels.reshape(-1).float()
    weights = resolve_weights(weights, predictions, weighted)

    log_odds = torch.logit(predictions, eps=epsilon)
    per_event = (predictions - labels) * log_odds
    score = (weights * per_event).sum() / weights.sum()

    logger.debug(f"[entropic_calibration_difference] weighted={weighted} ecd={score.item():.6f}")
    return score.item()


def modified_huber_loss(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor | None = None,
    weighted: bool = True,
    epsilon: float = 1e-7,
) -> float:
    """Modified Huber loss (Zhang, 2004) for binary classification

        L_i = (1 - m_i)^2   for m_i > -1
            = -4 * m_i       otherwise

        modified_huber_loss = sum_i (w_i / W) * L_i

    ADAPTATION NOTE: our predictions are calibrated probabilities in [0, 1]
    (collect_predictions in the upstream applies a sigmoid), not a margin score
    So the original definition (meant for SVM) doesn't match our case. We choose 
    to map p -> logit(p) as the margin proxy (same transform used in
    entropic_calibration_difference), so a confidently-wrong prediction
    pushes the margin to -inf and correctly reaches the linear branch.
    Labels y_i are also mapped, y_i in [0, 1] -> (2y-1) -> Y_i in [-1, +1]
    Thus, in the formula above: with p_i predicted probability m_i = Y_i*logit(p_i) 

    Args:
        predictions: (N,) predicted probabilities in [0, 1].
        labels: (N,) binary targets (0/1).
        weights: (N,) event weights. Required if weighted=True; ignored (may
            be omitted) if weighted=False.
        weighted: if True, each event's loss is weighted by its event weight.
            If False, every event counts equally.
        epsilon: clamps predictions to [epsilon, 1-epsilon] before the logit transform,
            to avoid inf/nan at p=0 or p=1.

    Returns:
        float: the (weighted) mean modified Huber loss.
    """
    predictions = predictions.reshape(-1).float()
    labels = labels.reshape(-1).float()
    weights = resolve_weights(weights, predictions, weighted)

    y_pm1 = 2.0 * labels - 1.0
    margin = y_pm1 * torch.logit(predictions, eps=epsilon)

    quadratic = torch.clamp(1.0 - margin, min=0.0) ** 2
    linear = -4.0 * margin
    per_event = torch.where(margin > -1.0, quadratic, linear)

    loss = (weights * per_event).sum() / weights.sum()

    logger.debug(f"[modified_huber_loss] weighted={weighted} modified_huber_loss={loss.item():.6f}")
    return loss.item()