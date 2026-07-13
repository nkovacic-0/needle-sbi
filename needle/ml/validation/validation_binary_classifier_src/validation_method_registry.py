import os
import logging

from functools import partial

from needle.ml.validation.validation_binary_classifier_src.expected_calibration_error_w_variants import (
    expected_calibration_error,
    adaptive_calibration_error,
    expected_signed_calibration_error,
    averaged_expected_calibration_error,
    averaged_expected_signed_calibration_error,
    debiased_squared_calibration_error,
    debiased_rmsce,
    debiased_expected_calibration_error,
)
from needle.ml.validation.validation_binary_classifier_src.point_based_metrics import (
    brier_score,
    entropic_calibration_difference,
    modified_huber_loss,
)
from needle.ml.validation.validation_binary_classifier_src.plot_calibration_curve import (
    CalibrationCurvePlotsHEP,
    CalibrationCurvePlotsPriorCorrected,
)
from needle.ml.validation.validation_utils.plotting_wrapper import PlottingWrapper
from needle.ml.validation.validation_binary_classifier_src.plot_scores_ratios import ScoreRatioPlots
from needle.ml.validation.validation_binary_classifier_src.plot_reweighting import ReweightingPlots


from needle.utils.logging import ColorFormatter

logger = ColorFormatter.get_logger("downstream-validation")

# note: adapters: uniform contract with one per registry entry type
# signature: adapter(class/function, **kwargs) -> dict[str, float]
#   metric adapter    -> {"": value}
#   pure-plot adapter -> {}
#   reweighting-style -> {sub_key: value, ...}   (e.g. "emd_before_reweighting_jet_0_pt")

def _adapter_scalar_metric(metric_function, **kwargs) -> dict:
    """Uniform adapter for every function in expected_calibration_error_w_variants
    and point_based_metrics.
    """
    return {"": metric_function(**kwargs)}


def _adapter_plotting_function(plotting_class: type[PlottingWrapper], **kwargs) -> dict:
    """Uniform adapter for every pure-plot class (CalibrationCurvePlotsHEP,
    CalibrationCurvePlotsPriorCorrected, ScoreRatioPlots, ...). 
    """
    plotting_class(**kwargs).generate_all_plots()
    return {}
    # return {"": plotting_class(**kwargs)}


def _adapter_reweighting(**direction_kwargs) -> dict:
    """Reweighting is the only plot family that 
        (a) also contributes to metrics via get_emd_results() and 
        (b) may need MULTIPLE ReweightingPlots instances per call,
            one per enabled reweighting direction.
    direction_kwargs is keyed by direction name (e.g. "class_1_to_class_0"),
    (see also BinaryClassifierValidation._kwargs_for_reweighting).
    """
    flat = {}
    for direction_name, kwargs in direction_kwargs.items():
        plots = ReweightingPlots(**kwargs)
        plots.generate_all_plots()
        for feature_key, result in plots.get_emd_results().items():
            flat[f"{direction_name}_emd_before_{feature_key}"] = result["emd_before"]
            flat[f"{direction_name}_emd_after_{feature_key}"] = result["emd_after"]
    return flat


VALIDATION_METHOD_REGISTRY = {
    # point metrics registry
    "brier_score": {
        "function_call": partial(_adapter_scalar_metric, metric_function=brier_score),
        "use_corrected_scores": False,
        "expect_outputs": True,
        "input_shape": "scalar_metric",
    },
    "brier_score_prior_corrected": {
        "function_call": partial(_adapter_scalar_metric, metric_function=brier_score),
        "use_corrected_scores": True,
        "expect_outputs": True,
        "input_shape": "scalar_metric",
    },
    "entropic_calibration_difference": {
        "function_call": partial(_adapter_scalar_metric, metric_function=entropic_calibration_difference),
        "use_corrected_scores": False,
        "expect_outputs": True,
        "input_shape": "scalar_metric",
    },
    "entropic_calibration_difference_prior_corrected": {
        "function_call": partial(_adapter_scalar_metric, metric_function=entropic_calibration_difference),
        "use_corrected_scores": True,
        "expect_outputs": True,
        "input_shape": "scalar_metric",
    },
    "modified_huber_loss": {
        "function_call": partial(_adapter_scalar_metric, metric_function=modified_huber_loss),
        "use_corrected_scores": False,
        "expect_outputs": True,
        "input_shape": "scalar_metric",
    },
    "modified_huber_loss_prior_corrected": {
        "function_call": partial(_adapter_scalar_metric, metric_function=modified_huber_loss),
        "use_corrected_scores": True,
        "expect_outputs": True,
        "input_shape": "scalar_metric",
    },
    # ECE and ECE-like metrics registry
    "expected_calibration_error": {
        "function_call": partial(_adapter_scalar_metric, metric_function=expected_calibration_error),
        "use_corrected_scores": False,
        "expect_outputs": True,
        "input_shape": "scalar_metric",
    },
    "expected_calibration_error_prior_corrected": {
        "function_call": partial(_adapter_scalar_metric, metric_function=expected_calibration_error),
        "use_corrected_scores": True,
        "expect_outputs": True,
        "input_shape": "scalar_metric",
    },
    "adaptive_calibration_error": {
        "function_call": partial(_adapter_scalar_metric, metric_function=adaptive_calibration_error),
        "use_corrected_scores": False,
        "expect_outputs": True,
        "input_shape": "scalar_metric",
    },
    "adaptive_calibration_error_prior_corrected": {
        "function_call": partial(_adapter_scalar_metric, metric_function=adaptive_calibration_error),
        "use_corrected_scores": True,
        "expect_outputs": True,
        "input_shape": "scalar_metric",
    },
    "expected_signed_calibration_error": {
        "function_call": partial(_adapter_scalar_metric, metric_function=expected_signed_calibration_error),
        "use_corrected_scores": False,
        "expect_outputs": True,
        "input_shape": "scalar_metric",
    },
    "expected_signed_calibration_error_prior_corrected": {
        "function_call": partial(_adapter_scalar_metric, metric_function=expected_signed_calibration_error),
        "use_corrected_scores": True,
        "expect_outputs": True,
        "input_shape": "scalar_metric",
    },
    "averaged_expected_calibration_error": {
        "function_call": partial(_adapter_scalar_metric, metric_function=averaged_expected_calibration_error),
        "use_corrected_scores": False,
        "expect_outputs": True,
        "input_shape": "scalar_metric",
    },
    "averaged_expected_calibration_error_prior_corrected": {
        "function_call": partial(_adapter_scalar_metric, metric_function=averaged_expected_calibration_error),
        "use_corrected_scores": True,
        "expect_outputs": True,
        "input_shape": "scalar_metric",
    },
    "averaged_expected_signed_calibration_error": {
        "function_call": partial(_adapter_scalar_metric, metric_function=averaged_expected_signed_calibration_error),
        "use_corrected_scores": False,
        "expect_outputs": True,
        "input_shape": "scalar_metric",
    },
    "averaged_expected_signed_calibration_error_prior_corrected": {
        "function_call": partial(_adapter_scalar_metric, metric_function=averaged_expected_signed_calibration_error),
        "use_corrected_scores": True,
        "expect_outputs": True,
        "input_shape": "scalar_metric",
    },
    "debiased_squared_calibration_error": {
        "function_call": partial(_adapter_scalar_metric, metric_function=debiased_squared_calibration_error),
        "use_corrected_scores": False,
        "expect_outputs": True,
        "input_shape": "scalar_metric",
    },
    "debiased_squared_calibration_error_prior_corrected": {
        "function_call": partial(_adapter_scalar_metric, metric_function=debiased_squared_calibration_error),
        "use_corrected_scores": True,
        "expect_outputs": True,
        "input_shape": "scalar_metric",
    },
    "debiased_rmsce": {
        "function_call": partial(_adapter_scalar_metric, metric_function=debiased_rmsce),
        "use_corrected_scores": False,
        "expect_outputs": True,
        "input_shape": "scalar_metric",
    },
    "debiased_rmsce_prior_corrected": {
        "function_call": partial(_adapter_scalar_metric, metric_function=debiased_rmsce),
        "use_corrected_scores": True,
        "expect_outputs": True,
        "input_shape": "scalar_metric",
    },
    "debiased_expected_calibration_error": {
        "function_call": partial(_adapter_scalar_metric, metric_function=debiased_expected_calibration_error),
        "use_corrected_scores": False,
        "expect_outputs": True,
        "input_shape": "scalar_metric",
    },
    "debiased_expected_calibration_error_prior_corrected": {
        "function_call": partial(_adapter_scalar_metric, metric_function=debiased_expected_calibration_error),
        "use_corrected_scores": True,
        "expect_outputs": True,
        "input_shape": "scalar_metric",
    },
    # plotting registry
    "calibration_curve": {
        "function_call": partial(_adapter_plotting_function, plotting_class=CalibrationCurvePlotsHEP),
        "use_corrected_scores": False,
        "expect_outputs": False,
        "input_shape": "calibration_curve",
    },
    "calibration_curve_prior_corrected": {
        "function_call": partial(_adapter_plotting_function, plotting_class=CalibrationCurvePlotsPriorCorrected),
        "use_corrected_scores": True,
        "expect_outputs": False,
        "input_shape": "calibration_curve",
    },
    "scores_ratios_plot": {
        "function_call": partial(_adapter_plotting_function, plotting_class=ScoreRatioPlots),
        "use_corrected_scores": False,
        "expect_outputs": False,
        "input_shape": "score_ratio_plot",
    },
    "scores_ratios_plot_prior_corrected": {
        "function_call": partial(_adapter_plotting_function, plotting_class=ScoreRatioPlots),
        "use_corrected_scores": True,
        "expect_outputs": False,
        "input_shape": "score_ratio_plot",
    },
    "reweighting": {
        "function_call": _adapter_reweighting,
        "use_corrected_scores": False,
        "expect_outputs": True,
        "input_shape": "reweighting",
    },
    "reweighting_prior_corrected": {
        "function_call": _adapter_reweighting,
        "use_corrected_scores": True,
        "expect_outputs": True,
        "input_shape": "reweighting",
    },
}