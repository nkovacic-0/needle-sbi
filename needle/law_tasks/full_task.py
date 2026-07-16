import os
from pathlib import Path
from typing import List, Union

import law

from needle.law_tasks.downstream import DownstreamTask
from needle.law_tasks.mixins import HydraMixin
from needle.law_tasks.snapshot import SnapshotTask
from needle.utils.logging import ColorFormatter, LogOnce

logger = ColorFormatter.get_logger("dag")


class FullTask(HydraMixin, law.WrapperTask):
    """Single entry point that runs the complete NEEDLE pipeline end-to-end.

    Requiring this task pulls in, transitively, the entire chain that previously had to be run
    by hand in three separate steps:

        FullTask
         ├─ (no downstream_tasks in config)
         │   └─ SnapshotTask → MainTask → EstimatorTask → SystematicTask → EnsembleTask → FoldTask
         └─ (downstream_tasks defined in config)
             └─ DownstreamTask(key) [one per selected key] → SnapshotTask → MainTask → ...

    `FullTask` does not read or write `config.yaml` itself, that caching/consistency-check
    responsibility stays with `MainTask.requires()` (via `cache_and_compare_config`), which every
    path above still passes through exactly once.
    """

    results_path: str = law.Parameter(
        description="Root directory where results are saved.",
        default="runs",
        significant=False,
    )  # type: ignore
    select_downstream_to_run: tuple = law.CSVParameter(
        description=(
            "Comma-separated subset of downstream_tasks keys to run, e.g. "
            "'downstream_A,downstream_B'. If omitted, every key defined under "
            "config.downstream_tasks is run. Ignored (must be empty) if no downstream_tasks "
            "are defined in the config."
        ),
        default=(),
        significant=False,
    )  # type: ignore

    @property
    def abs_results_path(self) -> Path:
        """Get the absolute path to the results directory.

        Same CLI-vs-config precedence rules as `MainTask.abs_results_path` /
        `SnapshotTask.abs_results_path`.

        Returns:
            Path: Absolute path to results directory.
        """
        cli_given = str(self.results_path) != "runs"
        config_value = self.config.results_path

        if cli_given and config_value and str(config_value) != str(self.results_path):
            LogOnce(logger).warn_once(
                f"Conflicting value for arg `--results-path`. Config indicates '{config_value}' "
                f"while CLI arg is '{self.results_path}'. The CLI value takes precedence. You can also "
                f"set this parameter using `--hydra-overrides='results_path={self.results_path}'`."
            )
            return Path(os.path.abspath(self.results_path))

        if config_value:
            return Path(os.path.abspath(config_value))

        return Path(os.path.abspath(self.results_path))

    def requires(self) -> List[Union[SnapshotTask, DownstreamTask]]:
        """Resolve which top-level task(s) to require based on the config and CLI selection.

        Returns:
            List[Union[SnapshotTask, DownstreamTask]]: A single SnapshotTask if no downstream_tasks
                are configured, otherwise one DownstreamTask per selected (or, by default, every)
                key in config.downstream_tasks.

        Raises:
            ValueError: If `select_downstream_to_run` references key(s) not present in
                `config.downstream_tasks` (this also fires if downstream_tasks is empty but a
                selection was given, since the valid-key set is then empty).
        """
        downstream_config = self.config.downstream_tasks or {}
        valid_keys = set(downstream_config.keys())
        requested_keys = set(self.select_downstream_to_run)

        unknown_keys = requested_keys - valid_keys
        if unknown_keys:
            raise ValueError(
                f"Unknown downstream task key(s) passed via --select-downstream-to-run: "
                f"{sorted(unknown_keys)}. Valid keys in config.downstream_tasks are: "
                f"{sorted(valid_keys) if valid_keys else '(none defined)'}"
            )

        if not valid_keys:
            logger.info("No downstream_tasks defined in config; FullTask will run like SnapshotTask.")
            return [
                SnapshotTask(
                    results_path=self.abs_results_path,
                    config_file=self.config_file,
                    hydra_overrides=self.hydra_overrides,
                )
            ]

        keys_to_run = requested_keys if requested_keys else valid_keys
        logger.info(f"FullTask will run downstream task(s): {sorted(keys_to_run)}")

        return [
            DownstreamTask(
                results_path=self.abs_results_path,
                downstream=key,
                config_file=self.config_file,
                hydra_overrides=self.hydra_overrides,
            )
            for key in sorted(keys_to_run)
        ]