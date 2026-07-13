import numpy as np
import mplhep as hep
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from scipy.stats import wasserstein_distance

from needle.utils.logging import ColorFormatter

logger = ColorFormatter.get_logger("plotting")


def weighted_histogram_with_error(
    data: np.ndarray,
    weights: np.ndarray,
    edges: np.ndarray,
    hist_range: tuple[float, float],
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted histogram and its per-bin Poissonic error, both squared-error
    convention (caller takes sqrt for display), see weighted_histogram_comparison.

    Args:
        data: (N,) values to histogram.
        weights: (N,) event weights.
        edges: bin edges.
        hist_range: (min, max) histogram range.
        normalize: normalize counts (and correspondingly, squared errors) so
            the histogram integrates to unit area.

    Returns:
        tuple[np.ndarray, np.ndarray]: (counts, squared_errors), same shape,
            one entry per bin.
    """
    counts, _ = np.histogram(data, edges, hist_range, weights=weights)
    squared_errors, _ = np.histogram(data, edges, hist_range, weights=weights ** 2)

    if normalize:
        total = np.sum(counts)
        counts = counts / total
        squared_errors = squared_errors / (total ** 2)

    return counts, squared_errors


def weighted_quantile(data: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    """Weighted quantile via linear interpolation on the weighted empirical CDF
    quantile=1.0 returns (up to floating point) data.max(), quantile=0.0 returns data.min().
    """
    if not (0.0 <= quantile <= 1.0):
        err_msg = f"quantile must be in [0, 1], got {quantile}"
        logger.error(err_msg)
        raise ValueError(err_msg)

    order = np.argsort(data)
    sorted_data = data[order]
    cum_weights = np.cumsum(weights[order])
    cum_weights = cum_weights / cum_weights[-1]

    return float(np.interp(quantile, cum_weights, sorted_data))


def weighted_density_ratio_with_error(
    counts_a: np.ndarray,
    squared_errors_a: np.ndarray,
    counts_b: np.ndarray,
    squared_errors_b: np.ndarray,
    epsilon: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray]:
    """Ratio B/(A+B) of two independently-histogrammed densities, with
    propagated error (standard propagation of errors for f = B/(A+B)

        Var(f) = (df/dA)^2 Var(A) + (df/dB)^2 Var(B)
               = [B^2 Var(A) + A^2 Var(B)] / (A+B)^4

    Args:
        counts_a, counts_b: (n_bins,) histogram counts/densities for A and B.
        squared_errors_a, squared_errors_b: (n_bins,) corresponding Var(A),
            Var(B) e.g. from weighted_histogram_with_error.
        epsilon: floor on (A+B) before dividing, to avoid inf/nan in empty bins.

    Returns:
        tuple[np.ndarray, np.ndarray]: (ratio, ratio_error) 
        note: the error in return is a standard deviation (sqrt'd), not a variance.
    """
    total_safe = np.clip(counts_a + counts_b, a_min=epsilon, a_max=None)
    ratio = counts_b / total_safe
    ratio_variance = (counts_b ** 2 * squared_errors_a + counts_a ** 2 * squared_errors_b) / (total_safe ** 4)
    return ratio, np.sqrt(ratio_variance)


def weighted_histogram_ratio_with_error(
    counts_numerator: np.ndarray,
    squared_errors_numerator: np.ndarray,
    counts_denominator: np.ndarray,
    squared_errors_denominator: np.ndarray,
    epsilon: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray]:
    """Ratio C/D of two INDEPENDENTLY-sampled histogram densities, with
    standard propagated error. Notably NOT the disjoint-classes special case used
    by weighted_density_ratio_with_error (e.g. reweighted-vs-target here are
    independent samples, not a partition of one combined total):

        f = C/D
        Var(f) = Var(C)/D^2 + C^2*Var(D)/D^4

    Args:
        counts_numerator, counts_denominator: (n_bins,) histogram densities.
        squared_errors_numerator, squared_errors_denominator: corresponding
            Var(C), Var(D).
        epsilon: floor on the denominator magnitude before dividing.

    Returns:
        tuple[np.ndarray, np.ndarray]: (ratio, ratio_error), error already
            sqrt'd (standard deviation, not variance).
    """
    denom_safe = np.clip(counts_denominator, a_min=epsilon, a_max=None)
    ratio = counts_numerator / denom_safe
    variance = (
        squared_errors_numerator / denom_safe ** 2
        + (counts_numerator ** 2 * squared_errors_denominator) / denom_safe ** 4
    )
    return ratio, np.sqrt(variance)


def weighted_mode_region_bounds(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
    probe_bins: int,
) -> tuple[float, float]:
    """Find an "informative" (x_min, x_max) sub-range of `values` by growing
    a contiguous region outward from the weighted-histogram MODE bin, always
    extending into whichever immediate neighbor bin currently has the larger
    weighted height, until the region's cumulative weighted mass reaches
    `quantile` of the total. Returns that region's outer bin edges.

    Useful for trimming long, uninformative tails of distributions

    quantile >= 1.0 is a special case: returns the full (values.min(),
    values.max()) range, no region-growing performed.

    Args:
        values: raw (unbinned) values to probe.
        weights: corresponding event weights.
        quantile: target fraction of total weighted mass to capture, in
            (0, 1]. Ties between the two candidate neighbor bins favor the
            right-hand neighbor (arbitrary, deterministic tie-break).
        probe_bins: number of bins for the internal probe histogram used to
            find the mode and grow the region. Coarser only affects
            precision of the returned edges, not correctness.

    Returns:
        tuple[float, float]: (x_min, x_max).
    """
    if quantile >= 1.0:
        return float(values.min()), float(values.max())
    if not (0.0 < quantile < 1.0):
        err_msg = f"quantile must be in (0, 1], got {quantile}"
        logger.error(err_msg)
        raise ValueError(err_msg)

    edges = np.linspace(values.min(), values.max(), probe_bins + 1)
    hist, _ = np.histogram(values, edges, weights=weights)
    total = hist.sum()
    if total <= 0:
        err_msg = f"weighted_mode_region_bounds: total weighted mass is <= 0 ({total})"
        logger.error(err_msg)
        raise ValueError(err_msg)

    mode_idx = int(np.argmax(hist))
    left = right = mode_idx
    cumulative = float(hist[mode_idx])
    target = quantile * total

    while cumulative < target:
        left_candidate = left - 1
        right_candidate = right + 1
        left_height = hist[left_candidate] if left_candidate >= 0 else -np.inf
        right_height = hist[right_candidate] if right_candidate < len(hist) else -np.inf

        if left_height == -np.inf and right_height == -np.inf:
            logger.warning(
                f"[weighted_mode_region_bounds] Exhausted the full range before reaching "
                f"quantile={quantile} (reached {cumulative / total:.3f}); using full range."
            )
            break

        if right_height >= left_height:
            right = right_candidate
            cumulative += hist[right]
        else:
            left = left_candidate
            cumulative += hist[left]

    return float(edges[left]), float(edges[right + 1])

def weighted_histogram_comparison(
    samples: list[tuple[np.ndarray, np.ndarray, str]],
    title: str,
    xlabel: str,
    ylabel: str,
    x_min: float,
    x_max: float,
    bins: int,
    figure_size: tuple[float, float] = (8.0, 6.0),
    fill_opacity: float = 0.3,
    linewidth: float = 1.5,
    title_fontsize: float = 14.0,
    labels_axis_fontsize: float = 12.0,
    yscale: str = "log",
    colors: list[str] | None = None,
    normalize: bool = True,
) -> Figure:
    """Overlay N weighted distributions as histograms with Poissonic
    errorbars, on shared bin edges.

    Does NOT apply NEEDLE branding or save/show the figure, this
    funciton just helps prepares the plot by building and returning
    a bare Figure.

    Args:
        samples: list of (data, weights, label) triples, one per distribution
            to overlay. data/weights must be equal-length 1D arrays.
        title, xlabel, ylabel: plot text.
        x_min, x_max: shared histogram range across every sample.
        bins: number of bins, shared across every sample.
        figure_size: matplotlib figsize.
        fill_opacity: 0 = fully filled, 1 = outline only 
            (alpha = 1 - fill_opacity for the filled region).
        linewidth: outline linewidth, drawn as a second, unfilled histplot
            call on top of the filled region 
        title_fontsize, labels_axis_fontsize: text sizing.
        yscale: "log" or "linear".
        colors: one color per sample; defaults to matplotlib's active color
            cycle, recycled if there are more samples than colors given.
        normalize: normalize each sample's histogram to unit area
            independently, so shapes are comparable regardless of relative
            sample sizes (same default/meaning as the old function).

    Returns:
        Figure
    """
    if not samples:
        err_msg = "samples must contain at least one (data, weights, label) triple"
        logger.error(err_msg)
        raise ValueError(err_msg)
    if x_max <= x_min:
        err_msg = f"x_max must be > x_min, got x_min={x_min}, x_max={x_max}"
        logger.error(err_msg)
        raise ValueError(err_msg)
    if bins < 1:
        err_msg = f"bins must be >= 1, got {bins}"
        logger.error(err_msg)
        raise ValueError(err_msg)
    for data, weights, label in samples:
        if len(data) != len(weights):
            err_msg = f"sample '{label}': data and weights must be equal length, got {len(data)} vs {len(weights)}"
            logger.error(err_msg)
            raise ValueError(err_msg)

    bin_edges = np.linspace(x_min, x_max, num=bins + 1)
    hist_range = (x_min, x_max)

    fig, ax = plt.subplots(figsize=figure_size)

    resolved_colors = colors if colors else plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, (data, weights, label) in enumerate(samples):
        color = resolved_colors[i % len(resolved_colors)]
        counts, squared_errors = weighted_histogram_with_error(
            data=data, weights=weights, edges=bin_edges, hist_range=hist_range, normalize=normalize,
        )
        hep.histplot(
            counts, bin_edges, yerr=np.sqrt(squared_errors), label=label,
            histtype="fill", alpha=1 - fill_opacity, color=color, ax=ax,
        )
        hep.histplot(
            counts, bin_edges, yerr=np.sqrt(squared_errors), linewidth=linewidth, color=color, ax=ax,
        )

    ax.set_title(title, fontsize=title_fontsize, pad=30)
    ax.set_xlabel(xlabel, fontsize=labels_axis_fontsize)
    ax.set_ylabel(ylabel, fontsize=labels_axis_fontsize)
    ax.legend(loc="best", prop={"size": labels_axis_fontsize})
    if yscale == "log":
        ax.set_yscale("log")

    logger.debug(
        f"[weighted_histogram_comparison] {len(samples)} sample(s), bins={bins}, "
        f"range=({x_min}, {x_max}), normalize={normalize}"
    )
    return fig

def weighted_wasserstein_distance(
    bin_centers: np.ndarray,
    source_weights: np.ndarray,
    target_weights: np.ndarray,
) -> float:
    """Wasserstein-1 (Earth Mover's) distance between two histograms sharing
    the same bin_centers, given as (possibly signed, from HEP-style negative
    event weights) per-bin weights.

    scipy.stats.wasserstein_distance requires per-point weights to represent
    a non-negative probability mass, but does NOT validate this (docs only
    require the WEIGHT SUM to be positive and finite), thus silently feeding it
    negative per-bin weights doesn't raise: it breaks the internal empirical-
    CDF construction (assumes monotonically non-decreasing cumulative mass)
    and returns, silently, the wrong number. Bins with a negative weight, for
    both distributions, are therefore discarded (from that side only), with a
    warning naming how many.

    Args:
        bin_centers: (n_bins,) shared bin centers for both histograms.
        source_weights, target_weights: (n_bins,) per-bin weights (e.g.
            histogram heights) for the two distributions being compared.

    Returns:
        float: the Wasserstein-1 distance.
    """
    source_negative = source_weights < 0
    target_negative = target_weights < 0
    n_source_negative = int(source_negative.sum())
    n_target_negative = int(target_negative.sum())

    if n_source_negative > 0 or n_target_negative > 0:
        logger.warning(
            f"[weighted_wasserstein_distance] Discarding {n_source_negative} source bin(s) and "
            f"{n_target_negative} target bin(s) with negative weight (common with signed/negative-"
            "weighted HEP samples) -- see docstring for why this must be discarded, not clamped."
        )

    source_keep = ~source_negative
    target_keep = ~target_negative

    if source_weights[source_keep].sum() <= 0 or target_weights[target_keep].sum() <= 0:
        err_msg = (
            "weighted_wasserstein_distance: remaining (non-negative) weight sum is <= 0 on one or "
            "both sides after discarding negative bins."
        )
        logger.error(err_msg)
        raise ValueError(err_msg)

    return float(
        wasserstein_distance(
            u_values=bin_centers[source_keep],
            v_values=bin_centers[target_keep],
            u_weights=source_weights[source_keep],
            v_weights=target_weights[target_keep],
        )
    )

