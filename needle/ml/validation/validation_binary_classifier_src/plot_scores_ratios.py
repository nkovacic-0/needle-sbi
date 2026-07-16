
from typing import  List

import numpy as np
import torch

from needle.ml.validation.validation_utils.plotting_wrapper import PlottingWrapper
from needle.ml.validation.validation_utils.metrics_utils import weighted_histogram_comparison, weighted_quantile
from needle.ml.validation.validation_binary_classifier_src.model_validation_utils import resolve_weights
from needle.utils.logging import ColorFormatter

logger = ColorFormatter.get_logger("plotting")


class ScoreRatioPlots(PlottingWrapper):
    """Plots NN scores (always) and, if plotting_configs["plot_ratios"] is
    True, likelihood-ratio-style transforms of those scores (r = p/(1-p+eps)),
    for one model's two labeled populations.

    Label separation (which events are class 0 vs class 1) is the caller's
    job (BinaryClassifierValidation). This class only ever sees already-
    split arrays, since that split would otherwise repeat across every plot
    family that needs it.

    plotting_configs keys used by this class:
        plot_savefile_prefix / plot_savefile_suffix (str): wrapped around the
            saved filename (via _resolve_plot_filename).
        weighted (bool, default True): if True, use the real weights passed
            in; if False, use all-ones mock weights (via resolve_weights,
            same convention as the metric functions) for both the histograms
            AND the weighted quantile computation below.
        plot_ratios (bool, default False): whether to also produce ratio plot(s).
        epsilon (float, default 1e-8): stabilizer in r = p / (1 - p + epsilon).
        ratio_display_quantiles (list[float], default [1.0, 0.66]): one ratio
            plot per entry. For quantile==1.0, x_max is the pooled data's
            actual max (no truncation, per an explicit special case rather
            than relying on the quantile formula's boundary behavior).
            Otherwise x_max is the (weighted, if weighted=True) quantile
            computed over class_0 and class_1's ratios POOLED together.
            x_min is always 0.
        weighted_histogram_comparison (dict): passed straight through as
            kwargs to weighted_histogram_comparison (e.g. xlabel/ylabel come
            from here untouched), EXCEPT title/x_min/x_max, which this class
            always computes and overrides on a per-call copy.
    """

    def __init__(
        self,
        plot_save_dir: str,
        scores_class_0: torch.Tensor,
        scores_class_1: torch.Tensor,
        weights_class_0: torch.Tensor,
        weights_class_1: torch.Tensor,
        class_0_dataset_label: str = "Class 0",
        class_1_dataset_label: str = "Class 1",
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

        self.scores_class_0 = scores_class_0.detach().cpu().numpy()
        self.scores_class_1 = scores_class_1.detach().cpu().numpy()

        self.class_0_dataset_label = class_0_dataset_label
        self.class_1_dataset_label = class_1_dataset_label

        use_weighted = self.plotting_configs.get("weighted", True)
        self.weights_class_0 = resolve_weights(weights_class_0, scores_class_0, use_weighted).detach().cpu().numpy()
        self.weights_class_1 = resolve_weights(weights_class_1, scores_class_1, use_weighted).detach().cpu().numpy()

    def _histogram_comparison_kwargs(self, title: str, x_min: float, x_max: float) -> dict:
        kwargs = dict(self.plotting_configs.get("weighted_histogram_comparison", {}))
        kwargs["title"] = title
        kwargs["x_min"] = x_min
        kwargs["x_max"] = x_max
        return kwargs

    @staticmethod
    def _quantile_key(quantile: float) -> str:
        return f"quantile_{quantile:.4g}".replace(".", "p")

    @staticmethod
    def _quantile_title_fragment(quantile: float) -> str:
        return f"{quantile:.2f}".lstrip("0") + "Q"

    @PlottingWrapper.plot(name="scores_distribution")
    def plot_scores(self):
        samples = [
            (self.scores_class_0, self.weights_class_0, self.class_0_dataset_label),
            (self.scores_class_1, self.weights_class_1, self.class_1_dataset_label),
        ]
        fig = weighted_histogram_comparison(
            samples=samples,
            **self._histogram_comparison_kwargs("NN scores distribution", x_min=0.0, x_max=1.0),
        )
        return [(self._resolve_plot_filename("scores"), fig)]
    
    @PlottingWrapper.plot(name="ratios_distribution")
    def plot_ratios(self):
        if not self.plotting_configs.get("plot_ratios", False):
            return []

        epsilon = self.plotting_configs.get("epsilon", 1e-8)
        ratios_class_0 = self.scores_class_0 / (1.0 - self.scores_class_0 + epsilon)
        ratios_class_1 = self.scores_class_1 / (1.0 - self.scores_class_1 + epsilon)

        pooled_ratios = np.concatenate([ratios_class_0, ratios_class_1])
        pooled_weights = np.concatenate([self.weights_class_0, self.weights_class_1])

        quantiles = self.plotting_configs.get("ratio_display_quantiles", [1.0, 0.66])
        results = []
        for quantile in quantiles:
            key = self._quantile_key(quantile)
            if quantile == 1.0:
                x_max = float(pooled_ratios.max())
            else:
                x_max = weighted_quantile(pooled_ratios, pooled_weights, quantile)

            samples = [
                (ratios_class_0, self.weights_class_0, self.class_0_dataset_label),
                (ratios_class_1, self.weights_class_1, self.class_1_dataset_label),
            ]
            title = f"Probability ratio distribution ({self._quantile_title_fragment(quantile)})"
            fig = weighted_histogram_comparison(
                samples=samples,
                **self._histogram_comparison_kwargs(title, x_min=0.0, x_max=x_max),
            )
            results.append((self._resolve_plot_filename(f"ratios_{key}"), fig))

        logger.debug(f"[ScoreRatioPlots] Built {len(results)} ratio plot variant(s): {[name for name, _ in results]}")
        return results
    
    def _build_title(self, distinguishing_label: str) -> str:
        """This class no longer uses the base title-building mechanism!
        Titles are hardcoded per plot (see plot_scores/plot_ratios) 
        """
        logger.warning(
            "[ScoreRatioPlots] _build_title() was called, but titles in this class are hardcoded "
            "per plot, not derived from plotting_configs['title']/prefix/suffix. This call is "
            "likely leftover/accidental."
        )
        return distinguishing_label