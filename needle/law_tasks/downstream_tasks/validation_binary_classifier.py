import os
import json
from pathlib import Path
from functools import cached_property

import hydra
from omegaconf import OmegaConf
from needle.utils.config_utils import hydra_instantiate

import luigi
from luigi.freezing import recursively_unfreeze

import torch

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy("file_system")

from needle.ml.lightning.datamodules.grouped_datamodule import GroupedDataModule
from needle.ml.lightning.models.ratio_density_est_binary_cl import RatioDensityEstimatorBinary
from needle.ml.validation.validation_binary_classifier_src.model_validation_utils import collect_predictions, check_validation_configs
from needle.ml.validation.validation_binary_classifier_master import BinaryClassifierValidation

from needle.utils.logging import ColorFormatter
logger = ColorFormatter.get_logger("downstream-validation")


_LOADABLE_MODELS = {
    "ratio_density_est_binary": RatioDensityEstimatorBinary,
}

def get_model_class(name: str):
    if name not in _LOADABLE_MODELS:
        err_msg = (f"Unknown model_class '{name}'. Available: {list(_LOADABLE_MODELS)}")
        logger.error(err_msg)
        raise ValueError(err_msg)
    return _LOADABLE_MODELS[name]


class ValidationTestsBinaryClassifier(luigi.Task):
    """Hub for NN final-quality validation on a binary classifier's held-out
    test set. Registered as an ordinary downstream_tasks entry (_target_);
    snapshot_path is injected automatically by DownstreamTask.

    Validates every fold-level checkpoint belonging to estimator_names, found
    by filtering the (already-complete, by the time this runs) DAG snapshot 
    Runs serially over all matching folds right now, idea: create_branch_map()
    -based parallelization in DownstreamTask?

    validation_configs bundles every run-tuning knob (model_class, batch_size,
    n_workers, NN_model_validation_chunk_size, multiprocessing_type,
    aux_feature_fields) plus validation_settings, a nested dict passed through
    to the metric-computation step once that exists. Note: this is a single
    luigi.DictParameter, so overriding any one key in YAML means supplying the
    whole dict there is no per-key merge against the default below.
    """

    snapshot_path: str = luigi.Parameter(description="Path to the DAG snapshot JSON")
    output_path: str = luigi.Parameter(description="Directory where per-fold validation results are written")
    estimator_names: list = luigi.ListParameter(description="Which estimator(s) in the snapshot to validate")
    dataset_config: dict = luigi.DictParameter(description="Test-set DatasetConfig kwargs")
    padding_lengths_load_path: str = luigi.Parameter(description="Path to the saved padding lengths cache dump.")

    validation_configs: dict = luigi.DictParameter(
        default={
            "model_class": "ratio_density_est_binary",
            "batch_size": 1024,
            "n_workers": 2,
            "NN_model_validation_chunk_size": 1,
            "multiprocessing_type": "torch",
            "aux_feature_fields": [],
            "validation_settings": {},
        },
        description="Validation-run tuning knobs; validation_settings is passed through to the metric-computation step.",
    )

    def output(self):
        return {
            node_id: luigi.LocalTarget(f"{self._results_savepath(node_id)}/_SUCCESS")
            for node_id in self.fold_node_ids
        }

    @cached_property
    def snapshot(self) -> dict:
        with open(self.snapshot_path) as f:
            return json.load(f)

    @cached_property
    def validation_settings(self) -> dict:
        """Validated validation_settings, with aux_feature_fields folded in if set.

        validation_configs is a luigi.DictParameter, which recursively freezes
        every nested dict/list into FrozenOrderedDict/tuple (for task-id
        hashing). dict(self.validation_configs) only unfreezes the OUTER
        level i.e. validation_configs["validation_settings"] itself, and
        everything nested inside it, stays frozen, which
        check_validation_configs's isinstance(x, dict) check correctly
        rejects. recursively_unfreeze walks the whole structure.
        """
        return check_validation_configs(recursively_unfreeze(self.validation_configs))

    @cached_property
    def datamodule(self) -> GroupedDataModule:
        return GroupedDataModule(
            dataset_config=recursively_unfreeze(self.dataset_config),
            batch_size=self.validation_configs["batch_size"],
            n_workers=self.validation_configs["n_workers"],
            multiprocessing_type=self.validation_configs["multiprocessing_type"],
            padding_lengths_load_path=self.padding_lengths_load_path,
            mode="test",
            aux_feature_fields=list(self.validation_configs["aux_feature_fields"]),
        )

    @cached_property
    def fold_node_ids(self) -> list[str]:
        """Every fold-level node belonging to one of estimator_names."""
        return [
            node_id
            for node_id, node in self.snapshot["nodes"].items()
            if node["task_type"] == "fold" and node["estimator_name"] in self.estimator_names
        ]

    @staticmethod
    def _safe_filename(node_id: str) -> str:
        # node_id is a urlencoded query string ("est=X&syst=Y&..."); '&'/'=' aren't
        # safe filename chars on every filesystem. Swap for the est__/syst__ style
        # already used on disk by fold.py et al. for consistency
        return node_id.replace("&", "_").replace("=", "__")

    def _load_model(self, checkpoint_path: str):
        model_cls = get_model_class(self.validation_configs["model_class"])
        config_path = Path(checkpoint_path).with_name("model_config.yaml")

        if config_path.exists():
            model_config = OmegaConf.load(config_path)
            configured_cls = hydra.utils.get_class(model_config._target_)
            if configured_cls is not model_cls:
                err_msg = (
                    f"model_config.yaml at '{config_path}' declares _target_={configured_cls.__name__}, "
                    f"but model_class='{self.validation_configs['model_class']}' resolves to "
                    f"{model_cls.__name__}. Mismatch between estimator_names filtering and the "
                    "configured model_class."
                )
                logger.error(err_msg)
                raise ValueError(err_msg)

            model = hydra_instantiate(model_config)
            model.configure_model()  # self.model is built lazily and must happen before load_state_dict

            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["state_dict"])
            logger.debug(
                f"[ValidationTestsBinaryClassifier._load_model] Loaded model from checkpoint path"
                f"'{checkpoint_path}' via model_config.yaml ({model_cls.__name__})."
            )
        else:
            logger.warning(
                f"[ValidationTestsBinaryClassifier._load_model] No model_config.yaml found next to model"
                f"\ncheckpoint path '{checkpoint_path}'!"
                f"falling back to {model_cls.__name__}.load_from_checkpoint(), which relies on the checkpoint's "
                "own embedded hyperparameters and has NOT been verified to restore lazily-built submodules correctly!"
            )
            model = model_cls.load_from_checkpoint(checkpoint_path)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.debug(f"[ValidationTestsBinaryClassifier._load_model] Moving model to device '{device}'")
        return model.to(device).eval().to(torch.float32)

    def _results_savepath(self, node_id: str) -> str:
        return f"{self.output_path}/{self._safe_filename(node_id)}"

    @cached_property
    def model_manifest(self) -> list[dict]:
        """One entry per fold node this task is responsible for validating."""
        manifest = [
            {
                "node_id": node_id,
                "model_path": self.snapshot["nodes"][node_id]["checkpoint_path"],
                "results_savepath": self._results_savepath(node_id),
            }
            for node_id in self.fold_node_ids
        ]
        if not manifest:
            logger.warning(
                f"[ValidationTestsBinaryClassifier.model_manifest] No fold nodes matched "
                f"estimator_names={list(self.estimator_names)} in the snapshot -- validation will "
                f"run zero chunks. Check estimator_names and snapshot_path for a misconfiguration."
            )
        else:
            logger.info(f"[ValidationTestsBinaryClassifier.model_manifest] {len(manifest)} fold(s) to validate for estimator_names={list(self.estimator_names)}")
        logger.debug(f"[ValidationTestsBinaryClassifier.model_manifest] node_ids: {[entry['node_id'] for entry in manifest]}")
        return manifest

    def _get_clean_shared_arrays(self, labels: torch.Tensor, weights: torch.Tensor, aux_features: dict) -> tuple:
        """NaN-filtered labels/weights/aux_features, computed once and cached
        (identical across every chunk ). A dataset-quality issue, not a model 
        issue so this function warns and drops the offending events rather than 
        raising. Returns (clean_labels, clean_weights, clean_aux_features).
        self._nan_keep_mask is also cached for callers that need to filter their 
        own per-model arrays consistently.
        """
        if getattr(self, "_nan_keep_mask", None) is not None:
            return self._clean_labels, self._clean_weights, self._clean_aux_features

        n_events = labels.shape[0]
        nan_mask = torch.isnan(labels) | torch.isnan(weights)

        keep_mask = ~nan_mask
        n_dropped = int(nan_mask.sum())
        if n_dropped > 0:
            logger.warning(
                f"[run] Dropping {n_dropped}/{n_events} event(s) with NaN in labels/weights "
                "(this is a dataset issue, not a model issue). Those events are excluded from "
                "validation for every model in this run."
            )

        self._nan_keep_mask = keep_mask
        self._clean_labels = labels[keep_mask]
        self._clean_weights = weights[keep_mask]
        self._clean_aux_features = {name: values[keep_mask] for name, values in aux_features.items()}

        return self._clean_labels, self._clean_weights, self._clean_aux_features

    def run(self):
        requested_checks = self.validation_settings.get("make_validation_checks", {})
        logger.debug(f"[ValidationTestsBinaryClassifier] requested_checks status: {requested_checks}")

        self.datamodule.setup()
        dataloader = self.datamodule.test_dataloader()
        targets = self.output()

        manifest = self.model_manifest
        chunk_size = self.validation_configs["NN_model_validation_chunk_size"]
        n_chunks = (len(manifest) + chunk_size - 1) // chunk_size

        for chunk_idx, start in enumerate(range(0, len(manifest), chunk_size), start=1):
            chunk = [
                {**entry, "model": self._load_model(entry["model_path"])}
                for entry in manifest[start : start + chunk_size]
            ]
            logger.info(
                f"[run] Running model chunk {chunk_idx}/{n_chunks}:"
                f"\n\t{[entry['node_id'] for entry in chunk]}"
            )

            chunk_results, labels, weights, aux_features = collect_predictions(chunk, dataloader)

            clean_labels, clean_weights, clean_aux_features = self._get_clean_shared_arrays(labels, weights, aux_features)
            del labels, weights, aux_features
            
            if not torch.all(self._nan_keep_mask):
                for result in chunk_results:
                    result["model_predictions"] = result["model_predictions"][self._nan_keep_mask]

            class_1_mask = clean_labels == 1
            weights_class_1 = clean_weights[class_1_mask]
            weights_class_0 = clean_weights[~class_1_mask]

            if getattr(self, "_real_class_weight_sums", None) is None:
                self._real_class_weight_sums = {
                    "class_1": weights_class_1.sum().item(),
                    "class_0": weights_class_0.sum().item(),
                }
                logger.info(f"[run] Real (unnormalized) per-class weight sums: {self._real_class_weight_sums}")

            for i, (entry, result) in enumerate(zip(chunk, chunk_results), start=1):
                logger.info(
                    f"[run] Validating model {i}/{len(chunk)} in chunk {chunk_idx}/{n_chunks}: "
                    f"node_id={entry['node_id']}"
                )

                BinaryClassifierValidation(
                    validation_settings=self.validation_settings,
                    real_class_weight_sums=self._real_class_weight_sums,
                    node_id=entry["node_id"],
                    model_result=result,
                    weights=clean_weights,
                    labels=clean_labels,
                    aux_features=clean_aux_features,
                    results_savepath=result["results_savepath"],
                ).compute()

                with targets[entry["node_id"]].open("w") as f:
                    json.dump({"status": "predictions_ready", "node_id": entry["node_id"]}, f)
                logger.debug(f"[run] Wrote _SUCCESS for node_id={entry['node_id']}")

            del chunk, chunk_results
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        logger.info(f"[run] Validation complete: {len(manifest)} fold(s) across {n_chunks} chunk(s).")