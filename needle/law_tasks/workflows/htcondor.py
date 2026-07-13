import os
from typing import List

import law
from law.contrib import htcondor

from needle.law_tasks.workflows.common import (
    RemoteConfig,
    add_workflow_settings_from_cfg,
    get_script_dir,
)
from needle.utils.logging import ColorFormatter

logger = ColorFormatter.get_logger("htcondor")


class HTCondorWorkflow(htcondor.HTCondorWorkflow):
    results_path: str

    def htcondor_output_directory(self) -> law.LocalDirectoryTarget:  # type: ignore
        if os.path.isabs(self.results_path):
            output_dir = os.path.join(self.abs_results_path, "htcondor")  # type: ignore
            logger.warning(
                f"results_path ('{self.results_path}') is an absolute path. This overrides the "
                f"usual behaviour of anchoring the htcondor output directory to the script "
                f"directory (get_script_dir()) which is silently dropped by os.path.join once an "
                f"absolute component follows it. HTCondor output directory is set to: '{output_dir}'"
            )
        else:
            output_dir = os.path.join(get_script_dir(), self.results_path, "htcondor", self.__class__.__name__)
        return law.LocalDirectoryTarget(output_dir)

    def htcondor_create_job_file_factory(self, **kwargs):
        kwargs.setdefault("dir", os.path.join(self.abs_results_path, "htcondor", "job_files"))  # type: ignore
        return super().htcondor_create_job_file_factory(**kwargs)
    
    def htcondor_bootstrap_file(self) -> law.JobInputFile:
        return law.JobInputFile(os.path.join(get_script_dir(), "setup.sh"))

    def htcondor_log_stem(self) -> str:
        """Human-readable, grep-friendly stem identifying this specific job.
        Falls back to the class name if the expected identifying attributes aren't present, so
        this mixin isn't hard-tied to FoldTask's exact parameter set.
        """
        attrs = ("estimator", "systematic", "ensemble", "fold_index")
        parts = [f"{attr}-{getattr(self, attr)}" for attr in attrs if hasattr(self, attr)]
        return "_".join(parts) or self.__class__.__name__


    def htcondor_job_config(
        self,
        config: RemoteConfig,
        job_num: int,
        branches: List[int],
    ):
        config = super().htcondor_job_config(config, job_num, branches)
        config = add_workflow_settings_from_cfg(self, config, workflow_type="htcondor")

        config.input_files["pyproject.toml"] = law.JobInputFile(
            os.path.join(get_script_dir(), "pyproject.toml"),
        )

        config.custom_content.append(("getenv", "true"))
        config.render_variables["script_dir"] = get_script_dir()

        log_stem = self.htcondor_log_stem()
        # $(Cluster)/$(Process) are HTCondor's own ClassAd macros, substituted by the schedd at
        # submission time (HTCondor's equivalent of Slurm's %j). Needed because the branch-range
        # postfix law appends automatically is constant across every FoldTask instance (always
        # "_0To1", since FoldTask always has exactly one branch).
        config.stdout = f"{log_stem}_$(Cluster).$(Process).out"
        config.stderr = f"{log_stem}_$(Cluster).$(Process).err"
        config.log = f"{log_stem}_$(Cluster).$(Process).log"

        return config
