
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
from needle.ml.validation.validation_binary_classifier_src.model_validation_utils import (
    resolve_weights,
    weighted_bin_statistics,
)
from needle.utils.logging import ColorFormatter

logger = ColorFormatter.get_logger("plotting")


class ReliabilityDiagramBase(PlottingWrapper):
    """Three-panel improvement on CalibrationCurveBase: plots each bin's
    point at its weighted-mean score position (see src of weighted_bin_statistics) 
    rather than assuming the bin's geometric center. 

    Panel 1 (main): scatter + (y, optionally x) errorbars against the y=x
        diagonal. y-position/error is _compute_ratio_and_error's output
        x-position/error is the weighted mean/std of scores landing in that bin.
    Panel 2 (residual): (ratio - x_mean) / ratio_error -- the vertical
        distance from each point to the diagonal, using its real x-position,
        not a bin-center approximation.
    Panel 3 (bin weight): normalized total weight per bin, y in [0, 1] it
        lets the reader see how much real weight backs each plotted point.

    Subclasses differ only in _compute_ratio_and_error (identical contract to
    CalibrationCurveBase), everything else is shared here.
    """

    #: subclasses override to point at their own plotting_configs sub-dict key
    _CONFIG_KEY: str = "reliability_diagram"

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

        # ALWAYS pooled -- see class docstring.
        self.pooled_scores = np.concatenate([self.scores_class_0, self.scores_class_1])
        self.pooled_weights = np.concatenate([self.weights_class_0, self.weights_class_1])

    @property
    def _cfg(self) -> dict:
        return self.plotting_configs.get(self._CONFIG_KEY, {})

    @staticmethod
    def _bins_key(n_bins: int) -> str:
        return f"bins_{n_bins}"

    def _compute_ratio_and_error(
        self, bin_edges: np.ndarray, hist_range: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Subclasses implement -- identical contract to
        CalibrationCurveBase._compute_ratio_and_error.
        """
        raise NotImplementedError

    def _render_weight_panel(self, ax, bin_total_weight: np.ndarray, bin_edges: np.ndarray, cfg: dict) -> None:
        """Panel 3: normalized bin weight, y in [0, 1]. Negative-net-weight
        bins are excluded from the normalization denominator AND rendered as 0.
        """
        color = cfg.get("weight_panel_color", "dodgerblue")
        fill_opacity = cfg.get("weight_panel_fill_opacity", 0.3)
        edge_linewidth = cfg.get("weight_panel_linewidth", 1.5)

        kept = bin_total_weight > 0
        total_kept_weight = bin_total_weight[kept].sum()
        normalized_weight = np.zeros_like(bin_total_weight)
        if total_kept_weight > 0:
            normalized_weight[kept] = bin_total_weight[kept] / total_kept_weight

        hep.histplot(normalized_weight, bin_edges, ax=ax, histtype="fill", facecolor=color, alpha=fill_opacity)
        hep.histplot(normalized_weight, bin_edges, ax=ax, histtype="step", color=color, linewidth=edge_linewidth)
        ax.set_ylim(0.0, 1.0)

    def _build_reliability_diagram_figures(
        self, base_filename: str, title: str, xlabel: str, ylabel: str,
    ) -> list[tuple]:
        """Shared figure-building logic, called by each subclass's own
        @plot-decorated entry point. base_filename must be distinct PER
        SUBCLASS, same reasoning as CalibrationCurveBase.
        """
        cfg = self._cfg
        figure_size = cfg.get("figure_size", (8.0, 10.0))
        auxiliary_linewidth = cfg.get("auxiliary_linewidth", 1.0)
        line_color = cfg.get("line_color", "dodgerblue")
        title_fontsize = cfg.get("title_fontsize", 14.0)
        labels_axis_fontsize = cfg.get("labels_axis_fontsize", 12.0)
        y_lim_for_residue = cfg.get("y_lim_for_residue", (-5.0, 5.0))
        add_additional_residue_hlines = cfg.get("add_additional_residue_hlines", False)
        residue_hline_range = cfg.get("residue_hline_range", (-10.0, 10.0, 1.0))
        epsilon = cfg.get("epsilon", 1e-5)

        scatter_point_size = cfg.get("scatter_point_size", 4.0)
        scatter_point_color = cfg.get("scatter_point_color", "dodgerblue")
        error_bar_color = cfg.get("error_bar_color", "dodgerblue")
        error_bar_linewidth = cfg.get("error_bar_linewidth", 1.5)
        error_bar_capsize = cfg.get("error_bar_capsize", 3.0)
        show_x_errorbars = cfg.get("show_x_errorbars", True)

        results = []
        for n_bins in cfg.get("total_bins", [10]):
            key = self._bins_key(n_bins)
            bin_edges = np.linspace(0.0, 1.0, num=n_bins + 1)
            hist_range = (0.0, 1.0)

            ratio, ratio_error = self._compute_ratio_and_error(bin_edges, hist_range)
            x_mean, x_std, bin_total_weight = weighted_bin_statistics(self.pooled_scores, self.pooled_weights, bin_edges)

            residuals = (ratio - x_mean) / np.clip(ratio_error, a_min=epsilon, a_max=None)

            fig, (ax_curve, ax_residual, ax_weight) = plt.subplots(
                nrows=3, figsize=figure_size, sharex=True,
                gridspec_kw={"height_ratios": [3, 1, 1]},
            )

            ax_curve.plot([0.0, 1.0], [0.0, 1.0], "--", color="black", linewidth=auxiliary_linewidth)
            ax_curve.errorbar(
                x_mean, ratio,
                yerr=ratio_error, xerr=(x_std if show_x_errorbars else None),
                fmt="o", markersize=scatter_point_size, color=scatter_point_color,
                ecolor=error_bar_color, elinewidth=error_bar_linewidth, capsize=error_bar_capsize,
            )
            ax_curve.set_xlim(0.0, 1.0)
            ax_curve.set_ylim(0.0, 1.0)
            ax_curve.set_ylabel(ylabel, fontsize=labels_axis_fontsize)
            ax_curve.set_title(title, fontsize=title_fontsize)

            ax_residual.errorbar(
                x_mean, residuals, yerr=1.0,
                drawstyle="steps-mid", elinewidth=error_bar_linewidth, color=line_color,
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
            ax_residual.set_ylabel("Residual\n(std. deviations)", fontsize=labels_axis_fontsize, loc="center")

            self._render_weight_panel(ax_weight, bin_total_weight, bin_edges, cfg)
            ax_weight.set_xlim(0.0, 1.0)
            ax_weight.set_xlabel(xlabel, fontsize=labels_axis_fontsize)
            ax_weight.set_ylabel("Normalized\nbin weight", fontsize=labels_axis_fontsize, loc="center")

            fig = self.set_needle_plot_style(fig, axes=[ax_curve])
            fig.tight_layout()
            filename = self._resolve_plot_filename(f"{base_filename}_{key}")
            results.append((filename, fig))

        logger.debug(f"[{self.__class__.__name__}] Built {len(results)} reliability diagram variant(s): {[name for name, _ in results]}")
        return results

    def _build_title(self, distinguishing_label: str) -> str:
        """This class no longer uses the base title-building mechanism!
        Titles are hardcoded per plot (see plot_reliability_diagram /
        plot_reliability_diagram_prior_corrected).
        """
        logger.warning(
            f"[{self.__class__.__name__}] _build_title() was called, but titles in this class "
            "are hardcoded per plot, not derived from plotting_configs['title']/prefix/suffix. "
            "This call is likely leftover/accidental."
        )
        return distinguishing_label


class ReliabilityDiagramHEP(ReliabilityDiagramBase):
    """Reliability-diagram counterpart to CalibrationCurvePlotsHEP.

    plotting_configs keys used:
        weighted (bool, default True): inert for the y-axis computation here
            (same reasoning as CalibrationCurvePlotsHEP), but NOT inert for
            the pooled x-position/x-error/weight-panel, which always use real
            per-event weights when weighted=True.
        reliability_diagram (dict): total_bins, epsilon, line_color,
            auxiliary_linewidth, figure_size, title_fontsize,
            labels_axis_fontsize, y_lim_for_residue,
            add_additional_residue_hlines, residue_hline_range,
            scatter_point_size, scatter_point_color, error_bar_color,
            error_bar_linewidth, error_bar_capsize, show_x_errorbars,
            weight_panel_color, weight_panel_fill_opacity,
            weight_panel_linewidth, also see
            ReliabilityDiagramBase._build_reliability_diagram_figures /
            _render_weight_panel.
    """

    _CONFIG_KEY = "reliability_diagram"

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

    @PlottingWrapper.plot(name="reliability_diagram_normalized", add_needle_plot_style=False)
    def plot_reliability_diagram(self):
        return self._build_reliability_diagram_figures(
            base_filename="reliability_diagram",
            title="Reliability Diagram (normalized)",
            xlabel="Mean NN Score (weighted, per bin)",
            ylabel="Bin label average\n(probability ratio)",
        )


class ReliabilityDiagramPriorCorrected(ReliabilityDiagramBase):
    """Reliability-diagram counterpart to CalibrationCurvePlotsPriorCorrected.

    plotting_configs keys used: same shape as ReliabilityDiagramHEP, but
    under plotting_configs["reliability_diagram_prior_corrected"] instead of
    plotting_configs["reliability_diagram"], so the two plots' binning/
    styling can be configured independently. Unlike ReliabilityDiagramHEP,
    weighted=False here ALSO changes the y-axis computation (same
    distinction as CalibrationCurvePlotsPriorCorrected).
    """

    _CONFIG_KEY = "reliability_diagram_prior_corrected"

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

    @PlottingWrapper.plot(name="reliability_diagram_corrected_prior", add_needle_plot_style=False)
    def plot_reliability_diagram_corrected_prior(self):
        return self._build_reliability_diagram_figures(
            base_filename="reliability_diagram_corrected_prior",
            title="Reliability Diagram",
            xlabel="Mean Corrected NN Score (weighted, per bin)",
            ylabel="Empirical class-1 fraction\n(real weights)",
        )