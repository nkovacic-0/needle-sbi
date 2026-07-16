
from typing import  List

import numpy as np
import torch
import mplhep as hep
import matplotlib.pyplot as plt

from needle.ml.validation.validation_utils.plotting_wrapper import PlottingWrapper
from needle.ml.validation.validation_utils.metrics_utils import (
    weighted_histogram_with_error,
    weighted_density_ratio_with_error,
)
from needle.ml.validation.validation_binary_classifier_src.model_validation_utils import resolve_weights
from needle.utils.logging import ColorFormatter

logger = ColorFormatter.get_logger("plotting")


class CalibrationCurveBase(PlottingWrapper):
    """Shared scaffolding for calibration-curve-style plots: two-panel
    figure (curve + residual, in std-deviation units), diagonal reference,
    total_bins-driven variant handling.

    Subclasses differ ONLY in how the per-bin ratio + error are computed
    (see _compute_ratio_and_error), but everything else (figure layout,
    residual computation, styling, variant/config plumbing) is identical
    and lives here.

    Errors throughout are standard weighted-Poisson (sumw2) propagation on
    two independent per-bin counts, via weighted_histogram_with_error /
    weighted_density_ratio_with_error. 
    Note: This is not an approximation of a binomial-proportion error.
    For two disjoint classes (an event can never belong to both), independent
    -Poisson error propagation on their ratio is exactly the weighted 
    generalization of the binomial-proportion variance 
    """

    #: subclasses override to point at their own plotting_configs sub-dict key
    _CONFIG_KEY: str = "calibration_curve"

    def __init__(
        self,
        plot_save_dir: str,
        scores_class_0: torch.Tensor,
        scores_class_1: torch.Tensor,
        weights_class_0: torch.Tensor,
        weights_class_1: torch.Tensor,
        rlabel: str = "",
        formats: List[str] | None = None,
        plotting_configs: dict | None = None,
    ) -> None:
        super().__init__(
            plot_save_dir=plot_save_dir, 
            rlabel=rlabel, 
            formats = formats,
            plotting_configs=plotting_configs
        )

        self.scores_class_0 = scores_class_0.detach().cpu().numpy()
        self.scores_class_1 = scores_class_1.detach().cpu().numpy()

        use_weighted = self.plotting_configs.get("weighted", True)
        self.weights_class_0 = resolve_weights(weights_class_0, scores_class_0, use_weighted).detach().cpu().numpy()
        self.weights_class_1 = resolve_weights(weights_class_1, scores_class_1, use_weighted).detach().cpu().numpy()

    @property
    def _cfg(self) -> dict:
        return self.plotting_configs.get(self._CONFIG_KEY, {})

    @staticmethod
    def _bins_key(n_bins: int) -> str:
        return f"bins_{n_bins}"

    def _compute_ratio_and_error(
        self, bin_edges: np.ndarray, hist_range: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Subclasses implement: return (ratio, ratio_error) per bin, using
        self.scores_class_0/1 and self.weights_class_0/1.
        """
        raise NotImplementedError

    def _build_calibration_curve_figures(
        self, base_filename: str, title: str, xlabel: str, ylabel: str,
    ) -> list[tuple]:
        """Shared figure-building logic, called by each subclass's own
        @plot-decorated entry point. base_filename must be distinct PER
        SUBCLASS as both write into the same plot_save_dir, so an identical
        base_filename across HEP/prior-corrected would collide.
        """
        cfg = self._cfg
        figure_size = cfg.get("figure_size", (8.0, 8.0))
        linewidth = cfg.get("linewidth", 1.5)
        auxiliary_linewidth = cfg.get("auxiliary_linewidth", 1.0)
        line_color = cfg.get("line_color", "dodgerblue")
        title_fontsize = cfg.get("title_fontsize", 14.0)
        labels_axis_fontsize = cfg.get("labels_axis_fontsize", 12.0)
        y_lim_for_residue = cfg.get("y_lim_for_residue", (-5.0, 5.0))
        add_additional_residue_hlines = cfg.get("add_additional_residue_hlines", False)
        residue_hline_range = cfg.get("residue_hline_range", (-10.0, 10.0, 1.0))
        epsilon = cfg.get("epsilon", 1e-5)

        results = []
        for n_bins in cfg.get("total_bins", [10]):
            key = self._bins_key(n_bins)
            bin_edges = np.linspace(0.0, 1.0, num=n_bins + 1)
            hist_range = (0.0, 1.0)

            ratio, ratio_error = self._compute_ratio_and_error(bin_edges, hist_range)

            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            residuals = (ratio - bin_centers) / np.clip(ratio_error, a_min=epsilon, a_max=None)

            fig, (ax_curve, ax_residual) = plt.subplots(
                nrows=2, figsize=figure_size, sharex=True,
                gridspec_kw={"height_ratios": [3, 1]},
            )

            ax_curve.plot([0.0, 1.0], [0.0, 1.0], "--", color="black", linewidth=auxiliary_linewidth)
            hep.histplot(ratio, bin_edges, yerr=ratio_error, linewidth=linewidth, color=line_color, ax=ax_curve)
            ax_curve.set_xlim(0.0, 1.0)
            ax_curve.set_ylim(0.0, 1.0)
            ax_curve.set_ylabel(ylabel, fontsize=labels_axis_fontsize)
            ax_curve.set_title(title, fontsize=title_fontsize)

            ax_residual.errorbar(
                bin_centers, residuals, yerr=1.0,
                drawstyle="steps-mid", elinewidth=linewidth, color=line_color,
            )
            ax_residual.axhline(0.0, color="black", linewidth=auxiliary_linewidth)
            if add_additional_residue_hlines:
                start, stop, step = residue_hline_range
                for y_loc in np.arange(start, stop + step / 2, step):
                    if y_loc == 0:
                        continue
                    ax_residual.axhline(y_loc, color="black", linewidth=auxiliary_linewidth / 2, alpha=0.5)
            ax_residual.set_xlim(0.0, 1.0)
            ax_residual.set_ylim(*y_lim_for_residue)
            ax_residual.set_xlabel(xlabel, fontsize=labels_axis_fontsize)
            ax_residual.set_ylabel("Residual\n(std. deviations)", fontsize=labels_axis_fontsize, loc="center")

            fig = self.set_needle_plot_style(fig, axes=[ax_curve])
            fig.tight_layout()
            filename = self._resolve_plot_filename(f"{base_filename}_{key}")
            results.append((filename, fig))

        logger.debug(f"[{self.__class__.__name__}] Built {len(results)} calibration curve variant(s): {[name for name, _ in results]}")
        return results
    
    def _build_title(self, distinguishing_label: str) -> str:
        """This class no longer uses the base title-building mechanism!
        Titles are hardcoded per plot (see plot_scores/plot_ratios) 
        """
        logger.warning(
            "[CalibrationCurve] _build_title() was called, but titles in this class are hardcoded "
            "per plot, not derived from plotting_configs['title']/prefix/suffix. This call is "
            "likely leftover/accidental."
        )
        return distinguishing_label


class CalibrationCurvePlotsHEP(CalibrationCurveBase):
    """HEP-style calibration curve: per-class independently normalized score
    densities combined via Bayes' rule under an implicit 50/50 prior --

        P(class=1 | score in bin) = density_1(bin) / (density_0(bin) + density_1(bin))

    NOT a population-weighted fraction, this reconstructs exactly what a
    ratio-estimating binary classifier was trained to produce, independent of
    the real class balance. Invariant to whether normalized-training or
    real/unnormalized weights are passed in the per-class normalization
    cancels any overall per-class scale, so plotting_configs["weighted"] is
    mathematically inert here specifically (see CalibrationCurvePlotsPriorCorrected
    for where it isn't).

    plotting_configs keys used:
        weighted (bool, default True): see note above, inert for this class.
        calibration_curve (dict): total_bins, epsilon, line_color, linewidth,
            auxiliary_linewidth, figure_size, title_fontsize,
            labels_axis_fontsize, y_lim_for_residue,
            add_additional_residue_hlines, residue_hline_range, see
            CalibrationCurveBase._build_calibration_curve_figures.
    """

    _CONFIG_KEY = "calibration_curve"

    def _compute_ratio_and_error(self, bin_edges, hist_range):
        counts_0, sq_err_0 = weighted_histogram_with_error(
            data=self.scores_class_0, weights=self.weights_class_0,
            edges=bin_edges, hist_range=hist_range, normalize=True,
        )
        counts_1, sq_err_1 = weighted_histogram_with_error(
            data=self.scores_class_1, weights=self.weights_class_1,
            edges=bin_edges, hist_range=hist_range, normalize=True,
        )
        return weighted_density_ratio_with_error(
            counts_a=counts_0, squared_errors_a=sq_err_0,
            counts_b=counts_1, squared_errors_b=sq_err_1,
            epsilon=self._cfg.get("epsilon", 1e-5),
        )

    @PlottingWrapper.plot(name="calibration_curve_normalized", add_needle_plot_style=False)
    def plot_calibration_curve(self):
        return self._build_calibration_curve_figures(
            base_filename="calibration_curve",
            title="Calibration Curve (normalized)",
            xlabel="NN Score",
            ylabel="Bin label average\n(probability ratio)",
        )


class CalibrationCurvePlotsPriorCorrected(CalibrationCurveBase):
    """Standard (non-normalizing) reliability-diagram-style calibration
    curve, intended for use with PRIOR-SHIFT-CORRECTED scores (see
    model_validation_utils.correct_prior_shift) 

        P(class=1 | score in bin) = weight_1(bin) / (weight_0(bin) + weight_1(bin))

    Note:
    Unlike CalibrationCurvePlotsHEP, plotting_configs["weighted"] is NOT
    inert here! with no independent per-class normalization, the real
    weight magnitudes directly determine the pooled fraction plotted.
    weighted=False computes an unweighted (mock all-ones) empirical
    fraction instead, same convention as the point metrics.

    plotting_configs keys used: same shape as CalibrationCurvePlotsHEP, but
    under plotting_configs["calibration_curve_corrected_prior"] instead of
    plotting_configs["calibration_curve"], so the two plots' binning/styling
    can be configured independently.
    """

    _CONFIG_KEY = "calibration_curve_corrected_prior"

    def __init__(
        self,
        plot_save_dir: str,
        scores_class_0: torch.Tensor,
        scores_class_1: torch.Tensor,
        weights_class_0: torch.Tensor,
        weights_class_1: torch.Tensor,
        rlabel: str = "FAIR Universe HiggsML",
        formats: List[str] | None = None,
        plotting_configs: dict | None = None,
    ) -> None:
        super().__init__(
            plot_save_dir=plot_save_dir,
            scores_class_0=scores_class_0,
            scores_class_1=scores_class_1,
            weights_class_0=weights_class_0,
            weights_class_1=weights_class_1,
            rlabel=rlabel,
            formats=formats,
            plotting_configs=plotting_configs,
        )

    def _compute_ratio_and_error(self, bin_edges, hist_range):
        counts_0, sq_err_0 = weighted_histogram_with_error(
            data=self.scores_class_0, weights=self.weights_class_0,
            edges=bin_edges, hist_range=hist_range, normalize=False,
        )
        counts_1, sq_err_1 = weighted_histogram_with_error(
            data=self.scores_class_1, weights=self.weights_class_1,
            edges=bin_edges, hist_range=hist_range, normalize=False,
        )
        return weighted_density_ratio_with_error(
            counts_a=counts_0, squared_errors_a=sq_err_0,
            counts_b=counts_1, squared_errors_b=sq_err_1,
            epsilon=self._cfg.get("epsilon", 1e-5),
        )

    @PlottingWrapper.plot(name="calibration_curve_corrected_prior", add_needle_plot_style=False)
    def plot_calibration_curve_corrected_prior(self):
        return self._build_calibration_curve_figures(
            base_filename="calibration_curve_corrected_prior",
            title="Calibration Curve",
            xlabel="Corrected NN Score",
            ylabel="Empirical class-1 fraction\n(real weights)",
        )