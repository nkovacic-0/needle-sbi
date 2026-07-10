import os
from typing import List

import law
from law.contrib import slurm

from needle.law_tasks.workflows.common import (
    RemoteConfig,
    add_workflow_settings_from_cfg,
    get_script_dir,
)
from needle.utils.logging import ColorFormatter

logger = ColorFormatter.get_logger("slurm")


class SlurmWorkflow(slurm.SlurmWorkflow):
    results_path: str

    def slurm_output_directory(self) -> law.LocalDirectoryTarget:  # type: ignore
        if os.path.isabs(self.results_path):
            output_dir = os.path.join(self.abs_results_path, "slurm")  # type: ignore
            logger.warning(
                f"results_path ('{self.results_path}') is an absolute path. This overrides the "
                f"usual behaviour of anchoring the slurm output directory to the script directory "
                f"(get_script_dir()) which is silently dropped by os.path.join once an absolute"
                f" component follows it. Slurm output directory is set to: '{output_dir}'"
            )
        else:
            output_dir = os.path.join(get_script_dir(), self.results_path, "slurm", self.__class__.__name__)
        return law.LocalDirectoryTarget(output_dir)

    def slurm_create_job_file_factory(self, **kwargs):
        kwargs.setdefault("dir", os.path.join(self.abs_results_path, "slurm", "job_files"))  # type: ignore
        return super().slurm_create_job_file_factory(**kwargs)

    def slurm_bootstrap_file(self) -> law.JobInputFile:
        return law.JobInputFile(os.path.join(get_script_dir(), "setup.sh"))

    def slurm_log_stem(self) -> str:
        """Human-readable, grep-friendly stem identifying this specific job.

        Falls back to the class name if the expected identifying attributes aren't present, so
        this mixin isn't hard-tied to FoldTask's exact parameter set.
        """
        attrs = ("estimator", "systematic", "ensemble", "fold_index")
        parts = [f"{attr}-{getattr(self, attr)}" for attr in attrs if hasattr(self, attr)]
        return "_".join(parts) or self.__class__.__name__

    def slurm_job_config(
        self,
        config: RemoteConfig,
        job_num: int,
        branches: List[int],
    ):
        config = super().slurm_job_config(config, job_num, branches)
        config = add_workflow_settings_from_cfg(self, config, workflow_type="slurm")

        config.input_files["pyproject.toml"] = law.JobInputFile(
            os.path.join(get_script_dir(), "pyproject.toml"),
        )

        config.custom_content.append(("export", "ALL"))
        config.render_variables["script_dir"] = get_script_dir()

        log_stem = self.slurm_log_stem()
        config.stdout = f"{log_stem}_%j.out"  # %j = Slurm job id, auto-substituted by sbatch
        config.stderr = f"{log_stem}_%j.err"
        config.log = "slurm.log"

        return config
