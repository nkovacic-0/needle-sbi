"""This module lists the schema for the needle config by nesting appropriate python dataclasses. These
are then passed to hydra and OmegaConf to build the final container for the config (`MainConfig`) and
ensure accurate typehints throughout the framework.

Adding new fields is as simple as registering the
field in one of the dataclasses then adding the entry to the config. Fields that do not exist in the
config schema will raise an Error by hydra. Some fields are empty in the schema but are then filled
at runtime by other functions. This enables leaving that field empty in the config but also allowing
for manual overrides.

The sibling module to this one `config_utils` stores all the functions that
help construct the config from file.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Any, List, Optional, Literal, Dict

from omegaconf import MISSING

from needle.utils.dataclass import SerializableDataclass



class WeightsCombine(str, Enum):
    product = "product"
    sum = "sum"
class ScalerChoice(str, Enum):
    standard = "standard"
    minmax = "minmax"
 
 
@dataclass
class DatasetConfig(SerializableDataclass):
    """Optional container that is shared between Lightning Datamodule and Module to exchange information
    about the data used. For example, if your data requires a dimension to be a certain integer, this
    can be managed uniquely using this container.
 
    Note:
        Fields below the `### grouped particle-feature pipeline ###` marker are ALL optional and
        default to None/empty, specifically so that existing configs (using only the original,
        flat `features_columns` pipeline) continue to work unmodified. These fields are only
        consumed by the new grouped-feature pipeline the existing PaddedDataModule/Ingestor code ignores them entirely.
    Additional note:
        This was annotated/commented on by a clanker and is thus overly descriptive
    """

    paths: str = ""
    features_columns: Optional[List[str]] = field(default_factory=list)
    labels_columns: Optional[List[str]] = field(default_factory=list)
    weights_columns: Optional[List[str]] = field(default_factory=list)
    # issue: OmegaConf doesn't undersand Literal, so the workaround to enforce the two types is the
    # WeightsCombineSupport class
    # weights_combine: Optional[Literal["product", "sum"]] = "product"
    weights_combine: Optional[WeightsCombine] = WeightsCombine.product
    format: str = "automatic"
    dak_reader_kwargs: Optional[Dict[str, Any]] = field(default_factory=dict)
    max_number_events: Optional[int] = -1
 
    ##### grouped particle-feature pipeline (all optional; None/empty => unused) ####
    # for more information take a look at the config example under 
    # needle-sbi/needle/templates/conf/datasets/HTauTau_binary_classifier_dataset_propershaping.yaml

    # list[list[str]] form of feature_columns, for the grouped (particles, features)
    # pipeline. Kept as a SEPARATE field from `features_columns` (rather than trying
    # to overload that field's type) so existing configs and existing code paths that
    # read `features_columns: List[str]` are entirely unaffected. If this field is
    # set (non-None), the grouped pipeline is expected to be used instead of the
    # flat/ragged one for feature loading — that branching decision lives in
    # whichever DataModule consumes this config, not in this schema.
    feature_columns_grouped: Optional[List[List[str]]] = None
 
    # Purely descriptive labels for logging/debug messages — not used for any
    # lookup logic. Optional; if omitted, rows/columns are referred to by index.
    feature_names: Optional[List[str]] = None
    particle_names: Optional[List[str]] = None
 
    # Columns that are schema-absent (never sent to the parquet/dak reader) and
    # always filled with a fixed value, applied post-scaling. Example: MET has
    # no eta, so {"PRI_met_eta": 0.0}.
    replacements_missing_columns: Optional[Dict[str, float]] = field(default_factory=dict)
    replacements_default: Optional[float] = 0.0
 
    # Sentinel-value handling: a magic numeric placeholder (e.g. -25.0) that a
    # present column uses to mean "this particle is absent for this event",
    # distinct from a real awkward `None` (which is always detected and
    # NaN-filled unconditionally, regardless of these settings).
    general_sentinel_value: Optional[float] = None
    replace_all_sentinel_values: Optional[bool] = True
    replacements_sentinel_default: Optional[float] = float("nan")
    replacements_sentinel_columns: Optional[Dict[str, Optional[float]]] = field(default_factory=dict)
 
    ### scaler configuration
    scaler_choice: Optional[ScalerChoice] = ScalerChoice.standard
    scaler_save_path: Optional[str] = None
    scaler_load_path: Optional[str] = None
    scaler_use_sampling: Optional[bool] = False
    scaler_sample_fraction: Optional[float] = 0.10
    force_avoid_partition_sampling: Optional[bool] = True


# @dataclass
# class DatasetConfig(SerializableDataclass):
#     """Optional container that is shared between Lightning Datamodule and Module to exchange information
#     about the data used. For example, if your data requires a dimension to be a certain integer, this
#     can be managed uniquely using this container.
#     """

#     paths: str = ""
#     features_columns: Optional[List[str]] = field(default_factory=list)
#     labels_columns: Optional[List[str]] = field(default_factory=list)
#     weights_columns: Optional[List[str]] = field(default_factory=list)
#     # issue: OmegaConf doesn't undersand Literal, so the workaround to enforce the two types is the 
#     # WeightsCombineSupport class
#     # weights_combine: Optional[Literal["product", "sum"]] = "product"
#     weights_combine: Optional[WeightsCombine] = WeightsCombine.product
#     format: str = "automatic"
#     dak_reader_kwargs: dict[str, Any] = field(default_factory=dict)
#     max_number_events: int = -1


@dataclass
class SystematicConfig(SerializableDataclass):
    """In contrast to the `EstimatorConfig` dataclass, entries here can be inferred from the Asimov
    dataclass, e.g. by adopting the entries from the parent estimator.
    """

    datamodule: Optional[str] = None
    datamodule_override: Optional[Any] = None
    dataset: Optional[str] = None
    dataset_override: Optional[DatasetConfig] = None
    model: Optional[str] = None
    model_override: Optional[Any] = None
    trainer: Optional[str] = None
    trainer_override: Optional[Any] = None


@dataclass
class EnsembleConfig(SerializableDataclass):
    num_ensembles: int = 1
    aggregation_method: str | None = None


@dataclass
class ExpansionConfig(SerializableDataclass):
    ensembles: EnsembleConfig = field(default_factory=EnsembleConfig)
    systematics: dict[str, SystematicConfig] = field(default_factory=lambda: {"nominal": SystematicConfig()})
    folds: int = 1


@dataclass
class EstimatorConfig(SerializableDataclass):
    """Config for modules used during training.

    Important:
        The field to be used afterwards in the code is `*_override`, as this is the resolved field,
        the name is just a str and needs to be populated manually afterwards.

    Each field can be defined in two ways:

    1. By name (str): `dataset="fair_universe"`. This string is resolved at runtime using
        `needle.registry.resolve_defaults` to produce the actual DictConfig in the corresponding
        `*_override` field.

    2. By override (DictConfig / dataclass): Usual `dataset_override=DatasetConfig(...)`.
        If provided, the resolver will use this directly and not overwrite it.
    """

    datamodule: str = MISSING
    datamodule_override: Optional[Any] = None
    dataset: Optional[str] = None
    dataset_override: Optional[DatasetConfig] = field(default_factory=DatasetConfig)
    model: str = MISSING
    model_override: Optional[Any] = None
    trainer: str = MISSING
    trainer_override: Optional[Any] = None
    expands: ExpansionConfig = field(default_factory=ExpansionConfig)
    requires: Optional[List[str]] = None


@dataclass
class DownstreamTaskConfig(SerializableDataclass):
    requires: Optional[List[str]] = None
    args: Optional[dict[str, Any]] = field(default_factory=dict)
    expands: Optional[dict[str, Any]] = field(default_factory=dict)


@dataclass
class AggregationConfig(SerializableDataclass):
    """Configuration for model aggregation at each DAG level"""

    fold_method: str = "mean"
    ensemble_method: str = "mean"
    systematic_method: str = "mean"
    estimator_method: str = "sum"

    fold_weights: Optional[List[float]] = None
    ensemble_weights: Optional[List[float]] = None
    systematic_weights: Optional[List[float]] = None
    estimator_weights: Optional[List[float]] = None


@dataclass
class MainConfig(SerializableDataclass):
    estimators: dict[str, EstimatorConfig] = field(default_factory=dict)
    downstream_tasks: Optional[dict[str, DownstreamTaskConfig]] = field(default_factory=dict)
    aggregation: AggregationConfig = field(default_factory=AggregationConfig)
    results_path: Optional[str] = None
    results_path_downstream: Optional[str] = None
    custom_settings: Any = None
    _resolved: bool = False
