"""Configuration dataclass and YAML loader for the SynthSeg pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import yaml

from mengrowth.synthseg.exceptions import SynthSegConfigError

logger = logging.getLogger(__name__)


@dataclass
class SynthSegConfig:
    """Configuration for the MenGrowth SynthSeg parcellation pipeline.

    Attributes
    ----------
    input_root : str
        Root directory containing preprocessed MenGrowth studies. Expected
        layout: ``{input_root}/MenGrowth-XXXX/MenGrowth-XXXX-YYY/t1n.nii.gz``.
    output_root : str
        Root directory where per-study parcellation outputs are written.
        Mirrors the patient / study tree under ``input_root``.
    synthseg_repo : str
        Path to the cloned SynthSeg repository. The runner invokes
        ``{synthseg_repo}/scripts/commands/SynthSeg_predict.py``.
    conda_env_path : str
        Path to the dedicated SynthSeg conda environment (prefix, not a
        named env). Used by the SLURM worker to activate the interpreter.
    python_bin : str
        Optional override for the Python interpreter path. When empty, the
        runner falls back to ``"python"`` on ``PATH`` (i.e. whichever conda
        env is currently active).
    input_modality : str
        Modality filename stem (without extension). SynthSeg is evaluated on
        this modality only; defaults to ``"t1n"``.
    robust : bool
        Pass ``--robust`` to SynthSeg. Mandatory for pathology cohorts.
    threads : int
        Number of CPU threads; mapped to ``--threads``.
    timeout_seconds : int
        Hard cap per-volume subprocess timeout.
    use_gpu : bool
        If False, force CPU inference via ``--cpu``.
    parcellation_filename, volumes_filename, qc_filename : str
        Filenames for the three outputs written per study.
    posteriors_filename : str | None
        If set, also save posterior probability maps via ``--post``.
    log_dir : str
        Directory for SLURM launcher/worker logs and the patient manifest.
    """

    input_root: str = ""
    output_root: str = ""
    synthseg_repo: str = ""
    conda_env_path: str = ""
    python_bin: str = ""

    input_modality: str = "t1n"
    robust: bool = True
    threads: int = 2
    timeout_seconds: int = 900
    use_gpu: bool = True

    parcellation_filename: str = "synthseg_parc.nii.gz"
    volumes_filename: str = "synthseg_volumes.csv"
    qc_filename: str = "synthseg_qc.csv"
    posteriors_filename: Optional[str] = None

    log_dir: str = ""

    def __post_init__(self) -> None:
        if not self.input_root:
            raise SynthSegConfigError("input_root must be set")
        if not self.output_root:
            raise SynthSegConfigError("output_root must be set")
        if not self.synthseg_repo:
            raise SynthSegConfigError("synthseg_repo must be set")
        if not self.input_modality:
            raise SynthSegConfigError("input_modality must be set")
        if self.threads < 1:
            raise SynthSegConfigError(
                f"threads must be >= 1, got {self.threads}"
            )
        if self.timeout_seconds < 30:
            raise SynthSegConfigError(
                f"timeout_seconds must be >= 30, got {self.timeout_seconds}"
            )
        if not self.robust:
            # The meningioma cohort is out-of-distribution for non-robust
            # SynthSeg; allowing robust=False silently would produce
            # misleading QC scores. Hard-fail to force an explicit override.
            raise SynthSegConfigError(
                "robust=False is disallowed for the MenGrowth cohort "
                "(meningiomas are OOD; --robust is mandatory)"
            )
        for fname_field in (
            "parcellation_filename",
            "volumes_filename",
            "qc_filename",
        ):
            if not getattr(self, fname_field):
                raise SynthSegConfigError(f"{fname_field} must be non-empty")


def load_synthseg_config(yaml_path: Union[str, Path]) -> SynthSegConfig:
    """Load a :class:`SynthSegConfig` from a YAML file.

    The YAML must contain a top-level ``synthseg:`` key whose children map
    1-to-1 onto :class:`SynthSegConfig` fields.

    Parameters
    ----------
    yaml_path : str | Path
        Path to the YAML configuration file.

    Returns
    -------
    SynthSegConfig

    Raises
    ------
    FileNotFoundError
        If ``yaml_path`` does not exist.
    SynthSegConfigError
        If the YAML is malformed, missing the ``synthseg`` key, or contains
        invalid field values.
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"SynthSeg config file not found: {yaml_path}")

    logger.info("Loading SynthSeg config from %s", yaml_path)

    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise SynthSegConfigError(f"Failed to parse YAML: {e}") from e

    if not raw:
        raise SynthSegConfigError(f"Configuration file {yaml_path} is empty")
    if "synthseg" not in raw:
        raise SynthSegConfigError(
            f"Configuration {yaml_path} must contain a top-level 'synthseg' key"
        )

    data = raw["synthseg"]
    if not isinstance(data, dict):
        raise SynthSegConfigError("'synthseg' section must be a mapping")

    try:
        cfg = SynthSegConfig(**data)
    except TypeError as e:
        raise SynthSegConfigError(f"Invalid configuration: {e}") from e

    logger.info("SynthSeg configuration loaded (input_modality=%s, robust=%s)",
                cfg.input_modality, cfg.robust)
    return cfg
