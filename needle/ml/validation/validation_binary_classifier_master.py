import os
import json
import tempfile
from datetime import datetime

import torch

from needle.ml.validation.validation_binary_classifier_src.validation_method_registry import VALIDATION_METHOD_REGISTRY
from needle.ml.validation.validation_binary_classifier_src.model_validation_utils import correct_prior_shift, compute_probability_ratio

from needle.utils.logging import ColorFormatter
logger = ColorFormatter.get_logger("downstream-validation")

class BinaryClassifierValidation:
    """Runs every enabled check in validation_settings["make_validation_checks"]
    against one model's already-collected predictions, writes a single
    validation_results.json into results_savepath.
    """

    def __init__(
        self,
        validation_settings: dict,
        real_class_weight_sums: dict,
        node_id: str,
        model_result: dict,
        weights: torch.Tensor,
        labels: torch.Tensor,
        aux_features: dict,
        results_savepath: str,
    ) -> None:
        self.validation_settings = validation_settings
        self.real_class_weight_sums = real_class_weight_sums
        self.node_id = node_id
        self.model_result = model_result
        self.weights = weights
        self.labels = labels
        self.aux_features = aux_features
        self.results_savepath = results_savepath
        self.plots_savepath = {
            'nominal':          os.path.join(self.results_savepath, "Plots"),
            'prior_corrected':  os.path.join(self.results_savepath, "Plots_with_prior_corrected")
        }
        

        scores = self.model_result["model_predictions"]
        if torch.isnan(scores).any():
            err_msg = f"[BinaryClassifierValidation] NaN in model_predictions for node_id={node_id}"
            logger.error(err_msg)
            raise ValueError(err_msg)

        requested_checks = self.validation_settings.get("make_validation_checks", {})
        unknown_keys = set(requested_checks) - set(VALIDATION_METHOD_REGISTRY)
        if unknown_keys:
            err_msg = (
                f"[BinaryClassifierValidation] Unknown make_validation_checks key(s): {sorted(unknown_keys)}. "
                f"Available: {sorted(VALIDATION_METHOD_REGISTRY)}"
            )
            logger.error(err_msg)
            raise ValueError(err_msg)

        class_1_mask = self.labels == 1
        self.scores_class_1 = scores[class_1_mask]
        self.scores_class_0 = scores[~class_1_mask]
        self.weights_class_1 = self.weights[class_1_mask]
        self.weights_class_0 = self.weights[~class_1_mask]

        self.scores = scores  # full, unsplit needed by scalar metrics
        self.scores_corrected = correct_prior_shift(scores, real_class_weight_sums)
        self.scores_class_1_corrected = self.scores_corrected[class_1_mask]
        self.scores_class_0_corrected = self.scores_corrected[~class_1_mask]

        # aux_features split, mirroring scores/weights above -- a hard
        # prerequisite for reweighting's kwargs (needs feature values
        # pre-split per class).
        self.aux_features_class_0 = {name: values[~class_1_mask] for name, values in self.aux_features.items()}
        self.aux_features_class_1 = {name: values[class_1_mask] for name, values in self.aux_features.items()}

        self.dataset_labels = self.validation_settings.get("dataset_labels", {"class_0": "Class 0", "class_1": "Class 1"})
        self.rlabel = self.validation_settings.get("rlabel", "")

        self.metrics: dict[str, float] = {}
        self.validation_process_success: dict[str, bool] = {}

    def compute(self) -> dict:
        os.makedirs(self.results_savepath, exist_ok=True)

        requested_checks = self.validation_settings.get("make_validation_checks", {})
        for registry_key, enabled in requested_checks.items():
            if not enabled:
                continue

            entry = VALIDATION_METHOD_REGISTRY[registry_key]
            adapter = entry["function_call"]
            use_corrected = entry["use_corrected_scores"]
            expect_outputs = entry["expect_outputs"]

            try:
                # Shared config lookup for a check and its *_prior_corrected sibling:
                # derived by stripping the suffix rather than an explicit registry
                # field, to avoid users having to duplicate settings for both
                # variants. Accepted tradeoff: relies on every corrected entry
                # following this exact naming convention.
                base_key = registry_key
                if base_key.endswith("_prior_corrected"):
                    base_key = registry_key.removesuffix("_prior_corrected")
                current_configs = self.validation_settings.get("validation_checks_configs", {}).get(base_key, {})

                if entry["input_shape"] == "scalar_metric":
                    kwargs = self._kwargs_for_scalar_metric(use_corrected, current_configs)
                elif entry["input_shape"] == "score_ratio_plot":
                    kwargs = self._kwargs_for_score_ratio_plots(use_corrected, current_configs)
                elif entry["input_shape"] == "calibration_curve":
                    kwargs = self._kwargs_for_calibration_curve(use_corrected, current_configs)
                elif entry["input_shape"] == "reweighting":
                    kwargs = self._kwargs_for_reweighting(use_corrected, current_configs)
                else:
                    err_msg = (f"Unknown input_shape '{entry['input_shape']}' for '{registry_key}'")
                    logger.error(err_msg)
                    raise ValueError(err_msg)

                raw_dict = adapter(**kwargs)

                self.validation_process_success[registry_key] = True

                if expect_outputs:
                    if not raw_dict:
                        logger.warning(
                            f"[BinaryClassifierValidation] '{registry_key}' declares expect_outputs=True "
                            "but returned no values."
                        )
                    for sub_key, value in raw_dict.items():
                        flat_key = registry_key if sub_key == "" else f"{registry_key}_{sub_key}"
                        self.metrics[flat_key] = value

            except Exception:
                logger.exception(
                    f"[BinaryClassifierValidation] '{registry_key}' failed for node_id={self.node_id}"
                )
                self.validation_process_success[registry_key] = False
                if expect_outputs:
                    self.metrics[registry_key] = float("nan")

        results_dict = {
            "node_id": self.node_id,
            "metrics": self.metrics,
            "validation_process_success": self.validation_process_success,
        }
        output_path = self._resolve_json_output_path()
        with open(output_path, "w") as f:
            json.dump(results_dict, f, indent=2, sort_keys=True)
        logger.info(f"[BinaryClassifierValidation] Wrote results for node_id={self.node_id} to {output_path}")


        return results_dict


    def _kwargs_for_scalar_metric(self, use_corrected: bool, current_configs: dict) -> dict:
        predictions = self.scores_corrected if use_corrected else self.scores
        return {
            "predictions": predictions,
            "labels": self.labels,
            "weights": self.weights,
            **current_configs,
        }

    def _kwargs_for_score_ratio_plots(self, use_corrected: bool, current_configs: dict) -> dict:
        if use_corrected:
            scores_class_0, scores_class_1 = self.scores_class_0_corrected, self.scores_class_1_corrected
            plots_savepath = self.plots_savepath['prior_corrected']
        else:
            scores_class_0, scores_class_1 = self.scores_class_0, self.scores_class_1
            plots_savepath = self.plots_savepath['nominal']
        return {
            "plot_save_dir": plots_savepath,
            "scores_class_0": scores_class_0,
            "scores_class_1": scores_class_1,
            "weights_class_0": self.weights_class_0,
            "weights_class_1": self.weights_class_1,
            "class_0_dataset_label": self.dataset_labels.get("class_0", "Class 0"),
            "class_1_dataset_label": self.dataset_labels.get("class_1", "Class 1"),
            "rlabel": self.rlabel,
            "plotting_configs": current_configs,
        }

    def _kwargs_for_calibration_curve(self, use_corrected: bool, current_configs: dict) -> dict:
        if use_corrected:
            scores_class_0, scores_class_1 = self.scores_class_0_corrected, self.scores_class_1_corrected
            plots_savepath = self.plots_savepath['prior_corrected']
        else:
            scores_class_0, scores_class_1 = self.scores_class_0, self.scores_class_1
            plots_savepath = self.plots_savepath['nominal']
        return {
            "plot_save_dir": plots_savepath,
            "scores_class_0": scores_class_0,
            "scores_class_1": scores_class_1,
            "weights_class_0": self.weights_class_0,
            "weights_class_1": self.weights_class_1,
            "rlabel": self.rlabel,
            "plotting_configs": current_configs,
        }

    def _kwargs_for_reweighting(self, use_corrected: bool, current_configs: dict) -> dict:
        if use_corrected:
            plots_savepath = self.plots_savepath['prior_corrected']
        else:
            plots_savepath = self.plots_savepath['nominal']

        reweighting_cfg = self.validation_settings.get("reweighting_additional_configs", {})
        reweighting_direction = reweighting_cfg.get("reweighting_direction", {"class_1_to_class_0": True})
        resolved_features = self._resolve_reweighting_features()

        direction_kwargs = {}
        for direction, enabled in reweighting_direction.items():
            if not enabled:
                continue

            features_source, features_target = self._build_source_target_features(resolved_features, direction)
            ratio = self._reweighting_ratio_for_direction(direction, use_corrected, current_configs)

            if direction == "class_1_to_class_0":
                weights_source, weights_target = self.weights_class_1, self.weights_class_0
                dataset_source_label = self.dataset_labels.get("class_1", "Source")
                dataset_target_label = self.dataset_labels.get("class_0", "Target")
            else:
                weights_source, weights_target = self.weights_class_0, self.weights_class_1
                dataset_source_label = self.dataset_labels.get("class_0", "Source")
                dataset_target_label = self.dataset_labels.get("class_1", "Target")


            # each direction needs a distinct saved filename, or both would
            # collide writing into the same plot_save_dir under the same
            # prefix/suffix.
            direction_configs = dict(current_configs)
            existing_suffix = direction_configs.get("plot_savefile_suffix", "")
            direction_configs["plot_savefile_suffix"] = f"{existing_suffix}_{direction}"

            direction_kwargs[direction] = {
                "plot_save_dir": plots_savepath,
                "features_source": features_source,
                "features_target": features_target,
                "probability_ratios_source_to_target": ratio,
                "weights_source": weights_source,
                "weights_target": weights_target,
                "dataset_source_label": dataset_source_label,
                "dataset_target_label": dataset_target_label,
                "rlabel": self.rlabel,
                "plotting_configs": direction_configs,
            }

        return direction_kwargs
    
    def _resolve_reweighting_features(self) -> list[dict]:
            """Builds the `features` kwarg for ReweightingPlots from
            self.aux_features_class_0/1, applying reweighting_additional_configs'
            optional pretty-label and particle-slice overrides.

            Per field: pretty_label defaults to the raw field name unless
            overridden via feature_pretty_labels. Slice indices come from
            take_elements_from_aux_fields if the field is listed there, otherwise
            a shape check decides: 
                A single-column field is used as-is (no real "choice" was made, so no suffix)
                A multi-column field defaults to column 0 (an implicit choice, so the suffix IS applied
                A field with multiple requested slices fans out into multiple `features` entries.
            """
            reweighting_cfg = self.validation_settings.get("reweighting_additional_configs", {})
            pretty_labels = reweighting_cfg.get("feature_pretty_labels", {})
            slice_overrides = reweighting_cfg.get("take_elements_from_aux_fields", {})

            features = []
            for field_name in self.aux_features_class_0:
                values_class_0 = self.aux_features_class_0[field_name]
                values_class_1 = self.aux_features_class_1[field_name]
                base_label = pretty_labels.get(field_name, field_name)
                n_columns = values_class_0.shape[1]

                if field_name in slice_overrides:
                    slice_indices = slice_overrides[field_name]
                    apply_suffix = True
                elif n_columns == 1:
                    slice_indices = [0]
                    apply_suffix = False
                else:
                    slice_indices = [0]
                    apply_suffix = True

                for slice_index in slice_indices:
                    if slice_index < 0 or slice_index >= n_columns:
                        err_msg = (
                            f"[BinaryClassifierValidation] take_elements_from_aux_fields['{field_name}'] "
                            f"requested slice {slice_index}, but this field only has {n_columns} column(s)."
                        )
                        logger.error(err_msg)
                        raise ValueError(err_msg)

                    label = f"{base_label}$_{{,{slice_index}}}$" if apply_suffix else base_label
                    column_name = f"{field_name}_{slice_index}" if apply_suffix else field_name
                    features.append({
                        "feature_to_reweight_class_0": values_class_0[:, slice_index],
                        "feature_to_reweight_class_1": values_class_1[:, slice_index],
                        "feature_to_reweight_pretty_label": label,
                        "feature_column": column_name,
                    })

            return features

    def _reweighting_ratio_for_direction(self, direction: str, use_corrected: bool, current_configs: dict) -> torch.Tensor:
        epsilon = current_configs.get("epsilon", 1e-5)
        if direction == "class_1_to_class_0":
            score = self.scores_class_1_corrected if use_corrected else self.scores_class_1
            return compute_probability_ratio(score, epsilon=epsilon)
        elif direction == "class_0_to_class_1":
            score = self.scores_class_0_corrected if use_corrected else self.scores_class_0
            return compute_probability_ratio(1.0 - score, epsilon=epsilon)
        else:
            raise ValueError(f"Unknown reweighting direction '{direction}'")
        
    @staticmethod
    def _build_source_target_features(resolved_features: list[dict], direction: str) -> tuple[list[dict], list[dict]]:
        """Splits the class_0/class_1-keyed output of _resolve_reweighting_features
        into the source/target lists ReweightingPlots expects, per direction.
        """
        if direction == "class_1_to_class_0":
            source_key, target_key = "feature_to_reweight_class_1", "feature_to_reweight_class_0"
        elif direction == "class_0_to_class_1":
            source_key, target_key = "feature_to_reweight_class_0", "feature_to_reweight_class_1"
        else:
            raise ValueError(f"Unknown reweighting direction '{direction}'")

        features_source = [
            {"values": f[source_key], "pretty_label": f["feature_to_reweight_pretty_label"], "feature_column": f["feature_column"]}
            for f in resolved_features
        ]
        features_target = [
            {"values": f[target_key], "pretty_label": f["feature_to_reweight_pretty_label"], "feature_column": f["feature_column"]}
            for f in resolved_features
        ]
        return features_source, features_target
    
    def _resolve_json_output_path(self) -> str:
        """Returns results_savepath/validation_results.json, or a
        date-suffixed fallback if that already exists (e.g. a re-run against
        the same results_savepath), or a further mktemp-style random-suffixed
        fallback if even the date-suffixed name collides (e.g. two runs on
        the same day). Never overwrites an existing validation_results*.json.

        Uses exclusive-create ("x" mode) for the first two, deterministic
        tiers. Falls through to tempfile.mkstemp only for the final tier

        Each candidate is claimed as an empty placeholder file the moment
        this returns; compute()'s subsequent open(output_path, "w") is then
        writing to an already-reserved name, not racing to create it.
        """
        base_path = os.path.join(self.results_savepath, "validation_results.json")
        try:
            open(base_path, "x").close()
            return base_path
        except FileExistsError:
            pass

        date_suffix = datetime.now().strftime("%d_%m_%Y")
        dated_path = os.path.join(self.results_savepath, f"validation_results_{date_suffix}.json")
        try:
            open(dated_path, "x").close()
            logger.warning(
                f"[BinaryClassifierValidation] '{base_path}' already exists; writing to '{dated_path}' instead."
            )
            return dated_path
        except FileExistsError:
            pass

        fd, random_path = tempfile.mkstemp(
            prefix=f"validation_results_{date_suffix}_", suffix=".json", dir=self.results_savepath
        )
        os.close(fd)
        logger.warning(
            f"[BinaryClassifierValidation] '{dated_path}' also already exists; writing to '{random_path}' instead."
        )
        return random_path