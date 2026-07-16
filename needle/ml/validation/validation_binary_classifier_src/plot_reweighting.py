
from typing import  List

import numpy as np
import torch
import mplhep as hep
import matplotlib.pyplot as plt

from needle.ml.validation.validation_utils.plotting_wrapper import PlottingWrapper
from needle.ml.validation.validation_utils.metrics_utils import (
    weighted_histogram_with_error,
    weighted_histogram_ratio_with_error,
    weighted_mode_region_bounds,
    weighted_wasserstein_distance,
)

from needle.utils.logging import ColorFormatter

logger = ColorFormatter.get_logger("plotting")


class ReweightingPlots(PlottingWrapper):
    """Reweighting diagnostic: for a given physical feature, checks whether
    multiplying SOURCE's event weights by an already-computed probability
    ratio reproduces TARGET's shape in that feature. One plot per entry in
    `features_source` (matched to `features_target` by feature_column).

    Also computes the Wasserstein-1 (Earth Mover's) distance between source
    and target both BEFORE and AFTER reweighting, as a scalar closure-test
    metric. See emd_results (populated as a side effect of generate_all_plots();

    Intended for closure tests with "real" (unnormalized) weights only 

    plotting_configs keys used:
        reweighting (dict):
            num_bins (int, default 40).
            epsilon (float, default 1e-5): ratio-SUBPLOT denominator floor
                only, it is NOT used for the reweighting ratio itself anymore,
                that's computed upstream 
                (see e.g. model_validation_utils.compute_probability_ratio)
            figure_size, linewidth, auxiliary_linewidth, title_fontsize,
                labels_axis_fontsize: styling, same meaning as elsewhere.
            y_lim_for_residue (tuple[float, float], default (0.0, 2.0)):
                ratio-subplot y-limits.
            yscale (str | list[str], default "log"): main histogram panel scale, passed
                directly to matplotlib's set_yscale (any valid value works, e.g.
                "linear"/"log"/"symlog", an invalid string raises there, not silently
                ignored). A list produces one figure per entry, filename suffixed with
                "_{yscale}"; a single str produces one figure, filename unchanged.
            colors (list[str], default ["black", "orangered", "dodgerblue"]):
                target / source-raw / source-reweighted, in order.
            display_quantile (float, default 1.0): see weighted_mode_region_bounds.
            add_additional_ratio_hlines (bool, default False): extra faint guide
                lines in the ratio panel, at regular intervals, in addition to the
                always-drawn ratio=1.0 reference line.
            ratio_hline_range (tuple[float, float, float], default derived from
                y_lim_for_residue as (y_min, y_max, 0.25) if not set): start/stop/step
                for the extra guide lines. 1.0 itself is always skipped (already
                drawn as the primary reference line).
    """

    def __init__(
        self,
        plot_save_dir: str,
        features_source: list[dict],
        features_target: list[dict],
        probability_ratios_source_to_target: torch.Tensor,
        weights_source: torch.Tensor | None = None,
        weights_target: torch.Tensor | None = None,
        dataset_source_label: str = "Source",
        dataset_target_label: str = "Target",
        rlabel: str = "",
        formats: List[str] | None = None,
        plotting_configs: dict | None = None,
    ) -> None:
        super().__init__(
            plot_save_dir=plot_save_dir, 
            rlabel=rlabel,
            formats = formats,
            plotting_configs=plotting_configs,
        )

        if not features_source or not features_target:
            err_msg = "ReweightingPlots requires at least one entry in BOTH features_source and features_target"
            logger.error(err_msg)
            raise ValueError(err_msg)

        source_columns = [f["feature_column"] for f in features_source]
        target_columns = [f["feature_column"] for f in features_target]
        if len(set(source_columns)) != len(source_columns):
            err_msg = f"Duplicate feature_column value(s) in features_source: {source_columns}"
            logger.error(err_msg)
            raise ValueError(err_msg)
        if len(set(target_columns)) != len(target_columns):
            err_msg = f"Duplicate feature_column value(s) in features_target: {target_columns}"
            logger.error(err_msg)
            raise ValueError(err_msg)
        if set(source_columns) != set(target_columns):
            err_msg = (
                f"features_source and features_target must cover the same feature_column set. "
                f"source-only: {set(source_columns) - set(target_columns)}, "
                f"target-only: {set(target_columns) - set(source_columns)}"
            )
            logger.error(err_msg)
            raise ValueError(err_msg)

        target_by_column = {f["feature_column"]: f for f in features_target}
        self.features_by_column = {
            f["feature_column"]: {"source": f, "target": target_by_column[f["feature_column"]]}
            for f in features_source
        }

        if weights_source is None and weights_target is None:
            logger.info(
                "[ReweightingPlots] No weights provided for either source or target; using all-ones mock "
                "weights (dataset is truly unweighted)."
            )
            weights_source = torch.ones_like(features_source[0]["values"])
            weights_target = torch.ones_like(features_target[0]["values"])
        elif weights_source is None or weights_target is None:
            err_msg = (
                "ReweightingPlots got weights for only one of source/target (weights_source and "
                "weights_target must either both be provided or both be None)."
            )
            logger.error(err_msg)
            raise ValueError(err_msg)

        self.weights_source = weights_source.detach().cpu().numpy()
        self.weights_target = weights_target.detach().cpu().numpy()
        self.reweighted_weights_source = (weights_source * probability_ratios_source_to_target).detach().cpu().numpy()

        self.dataset_source_label = dataset_source_label
        self.dataset_target_label = dataset_target_label
        
        self.emd_results: dict[str, dict] = {}

    @property
    def _cfg(self) -> dict:
        return self.plotting_configs.get("reweighting", {})

    def _feature_variant_keys(self) -> list[str]:
        return list(self.features_by_column.keys())

    def get_emd_results(self) -> dict[str, dict]:
        """Call AFTER generate_all_plots(); empty until then."""
        return self.emd_results

    @PlottingWrapper.plot(name="reweighting", add_needle_plot_style=False)
    def plot_reweighting(self):
        cfg = self._cfg
        num_bins = cfg.get("num_bins", 40)
        epsilon = cfg.get("epsilon", 1e-5)
        figure_size = cfg.get("figure_size", (8.0, 8.0))
        linewidth = cfg.get("linewidth", 1.5)
        auxiliary_linewidth = cfg.get("auxiliary_linewidth", 1.0)
        title_fontsize = cfg.get("title_fontsize", 14.0)
        labels_axis_fontsize = cfg.get("labels_axis_fontsize", 12.0)
        y_lim_for_residue = cfg.get("y_lim_for_residue", (0.0, 2.0))
        yscale = cfg.get("yscale", "log")
        legend_loc = cfg.get("legend_loc", "best")
        colors = cfg.get("colors", ["black", "orangered", "dodgerblue"])
        display_quantile = cfg.get("display_quantile", 1.0)
        add_additional_ratio_hlines = cfg.get("add_additional_ratio_hlines", False)
        ratio_hline_range = cfg.get("ratio_hline_range")
        if ratio_hline_range is None:
            ratio_hline_range = (y_lim_for_residue[0], y_lim_for_residue[1], 0.25)

        # yscale: str -> one figure per feature, filename UNCHANGED (backward
        # compatible with existing configs). list[str] -> one figure per
        # feature PER entry, filename gets a "_{yscale}" suffix. Suffixing is
        # decided by TYPE alone, not list length -- even a single-entry list
        # gets suffixed.
        yscale_config = cfg.get("yscale", "log")
        if isinstance(yscale_config, str):
            yscale_options = [yscale_config]
            suffix_yscale_in_filename = False
        else:
            yscale_options = list(yscale_config)
            suffix_yscale_in_filename = True

        results = []
        for key, pair in self.features_by_column.items():
            source_feature = pair["source"]
            target_feature = pair["target"]
            values_source_raw = source_feature["values"].detach().cpu().numpy()
            values_target_raw = target_feature["values"].detach().cpu().numpy()
            pretty_label = source_feature.get("pretty_label", key)

            keep_source = (
                ~np.isnan(values_source_raw)
                & ~np.isnan(self.weights_source)
                & ~np.isnan(self.reweighted_weights_source)
            )
            keep_target = ~np.isnan(values_target_raw) & ~np.isnan(self.weights_target)

            n_dropped_source = int((~keep_source).sum())
            n_dropped_target = int((~keep_target).sum())
            if n_dropped_source > 0 or n_dropped_target > 0:
                logger.debug(
                    f"[ReweightingPlots] '{key}': dropping {n_dropped_source} NaN source event(s), "
                    f"{n_dropped_target} NaN target event(s) (e.g. padding)."
                )

            values_source = values_source_raw[keep_source]
            weights_source = self.weights_source[keep_source]
            reweighted_weights_source = self.reweighted_weights_source[keep_source]

            values_target = values_target_raw[keep_target]
            weights_target = self.weights_target[keep_target]

            x_min, x_max = weighted_mode_region_bounds(
                values_target, weights_target, quantile=display_quantile, probe_bins=num_bins,
            )
            bin_edges = np.linspace(x_min, x_max, num=num_bins + 1)
            hist_range = (x_min, x_max)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            hist_target, sq_err_target = weighted_histogram_with_error(
                data=values_target, weights=weights_target, edges=bin_edges, hist_range=hist_range, normalize=True,
            )
            hist_source_raw, sq_err_source_raw = weighted_histogram_with_error(
                data=values_source, weights=weights_source, edges=bin_edges, hist_range=hist_range, normalize=True,
            )
            hist_source_reweighted, sq_err_source_reweighted = weighted_histogram_with_error(
                data=values_source, weights=reweighted_weights_source, edges=bin_edges, hist_range=hist_range, normalize=True,
            )

            ratio_raw, ratio_raw_err = weighted_histogram_ratio_with_error(
                counts_numerator=hist_source_raw, squared_errors_numerator=sq_err_source_raw,
                counts_denominator=hist_target, squared_errors_denominator=sq_err_target,
                epsilon=epsilon,
            )
            ratio_reweighted, ratio_reweighted_err = weighted_histogram_ratio_with_error(
                counts_numerator=hist_source_reweighted, squared_errors_numerator=sq_err_source_reweighted,
                counts_denominator=hist_target, squared_errors_denominator=sq_err_target,
                epsilon=epsilon,
            )

            # EMD is independent of yscale (rendering-only) we compute it once per
            # feature, and reuse across every yscale variant below.
            emd_before = weighted_wasserstein_distance(bin_centers, hist_source_raw, hist_target)
            emd_after = weighted_wasserstein_distance(bin_centers, hist_source_reweighted, hist_target)
            self.emd_results[key] = {"emd_before": emd_before, "emd_after": emd_after, "feature_label": pretty_label}

            for yscale in yscale_options:
                fig, (ax_hist, ax_ratio) = plt.subplots(
                    nrows=2, figsize=figure_size, sharex=True,
                    gridspec_kw={"height_ratios": [3, 1]},
                )

                hep.histplot(
                    hist_target, bin_edges, yerr=np.sqrt(sq_err_target),
                    label=self.dataset_target_label, linewidth=linewidth, color=colors[0], ax=ax_hist
                )
                hep.histplot(
                    hist_source_raw, bin_edges, yerr=np.sqrt(sq_err_source_raw),
                    label=f"{self.dataset_source_label} (raw)", linewidth=linewidth, color=colors[1], ax=ax_hist
                )
                hep.histplot(
                    hist_source_reweighted, bin_edges, yerr=np.sqrt(sq_err_source_reweighted),
                    label=f"{self.dataset_source_label} (reweighted)", linewidth=linewidth, color=colors[2], ax=ax_hist
                )

                ax_hist.set_ylabel("Normalized events", fontsize=labels_axis_fontsize)
                ax_hist.legend(loc=legend_loc, fontsize=labels_axis_fontsize)
                ax_hist.set_yscale(yscale)  # forwarded directly, see class docstring
                ax_hist.set_title(f"Reweighting of {pretty_label}", fontsize=title_fontsize)

                hep.histplot(ratio_raw, bin_edges, yerr=ratio_raw_err, label="Raw ratio", linewidth=linewidth, color=colors[1], ax=ax_ratio)
                hep.histplot(ratio_reweighted, bin_edges, yerr=ratio_reweighted_err, label="Reweighted ratio", linewidth=linewidth, color=colors[2], ax=ax_ratio)
                ax_ratio.axhline(1.0, color="black", linewidth=auxiliary_linewidth)
                if add_additional_ratio_hlines:
                    start, stop, step = ratio_hline_range
                    for y_loc in np.arange(start, stop + step / 2, step):
                        if y_loc == 1.0:
                            continue  # already drawn above, avoid double-drawing
                        ax_ratio.axhline(y_loc, color="black", linewidth=auxiliary_linewidth / 2, alpha=0.5, linestyle='--') 
                ax_ratio.set_ylim(*y_lim_for_residue)
                ax_ratio.set_xlim(x_min, x_max)
                ax_ratio.set_xlabel(pretty_label, fontsize=labels_axis_fontsize)
                ax_ratio.set_ylabel(f"Ratio to\n{self.dataset_target_label}", fontsize=labels_axis_fontsize, loc="center")
                ax_ratio.legend(loc=legend_loc, fontsize=int(0.85 * labels_axis_fontsize))

                fig = self.set_needle_plot_style(fig, axes=[ax_hist])
                fig.tight_layout()

                base_name = f"reweighting_{key}_{yscale}" if suffix_yscale_in_filename else f"reweighting_{key}"
                filename = self._resolve_plot_filename(base_name)
                results.append((filename, fig))

        logger.debug(f"[ReweightingPlots] Built {len(results)} reweighting plot(s): {[name for name, _ in results]}")
        return results
    
    def _build_title(self, distinguishing_label: str) -> str:
        """This class no longer uses the base title-building mechanism!
        Titles are hardcoded per plot (see plot_scores/plot_ratios) 
        """
        logger.warning(
            "[ReweightingPlots] _build_title() was called, but titles in this class are hardcoded "
            "per plot, not derived from plotting_configs['title']/prefix/suffix. This call is "
            "likely leftover/accidental."
        )
        return distinguishing_label