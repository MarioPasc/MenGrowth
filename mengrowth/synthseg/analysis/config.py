"""Configuration dataclass and YAML loader for the Phase 2 analysis pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Union

import yaml

from mengrowth.synthseg.exceptions import SynthSegConfigError

logger = logging.getLogger(__name__)


DEFAULT_RELIABLE_REGIONS: List[str] = [
    "Left-Thalamus",
    "Right-Thalamus",
    "Left-Putamen",
    "Right-Putamen",
    "Left-Caudate",
    "Right-Caudate",
    "Left-Pallidum",
    "Right-Pallidum",
    "Left-Hippocampus",
    "Right-Hippocampus",
    "Left-Amygdala",
    "Right-Amygdala",
    "Left-Lateral-Ventricle",
    "Right-Lateral-Ventricle",
    "Left-Cerebral-White-Matter",
    "Right-Cerebral-White-Matter",
]

DEFAULT_DEEP_GREY_MATTER: List[str] = [
    "Left-Thalamus",
    "Right-Thalamus",
    "Left-Putamen",
    "Right-Putamen",
    "Left-Caudate",
    "Right-Caudate",
]

DEFAULT_MODALITIES: List[str] = ["t1n", "t1c", "t2w", "t2f"]


@dataclass
class AnalysisConfig:
    """Configuration for the Phase 2 SynthSeg QC analysis.

    Attributes
    ----------
    synthseg_output_root : str
        Root of the per-study SynthSeg outputs (finalized — no ``.tmp``).
    preprocessed_root : str
        Root of the preprocessed modality tree (same layout as the SynthSeg
        tree). Source of T1n / T1c / T2w / T2f NIfTIs for cross-modal
        analyses, and of ``seg.nii.gz`` when present.
    original_metadata_csv : str
        Pre-preprocessing per-study metrics CSV from the curation stage
        (typically ``quality/qc_analysis/per_study_metrics.csv``). Supplies
        original acquisition spacings per (patient, study, modality).
    output_dir : str
        Directory where Phase 2 outputs (CSVs, figures, report) are written.
    tumor_proximity_threshold_mm : float
        δ in the "tumor-distant vs tumor-adjacent" partition.
    modalities_for_roi_analysis : list[str]
        Modalities probed by ROI-based cross-modal metrics.
    reliable_regions : list[str]
        Subcortical regions used for cross-modal metrics (SYNTHSEG_LABEL_MAP
        keys). Restricted to large subcortical structures where SynthSeg is
        most reliable.
    deep_grey_matter_regions : list[str]
        Regions aggregated to form the headline acceptance CV metric.
    cv_acceptance_threshold : float
        Pass/fail threshold (plan §7) applied to the median CV of
        ``deep_grey_matter_regions`` under the tumor-distant partition.
    figure_format : str
        Image suffix for figure outputs. ``"png"`` is the default.
    figure_dpi : int
        DPI for saved figures.
    n_workers : int
        Process-pool size for the per-study region metrics stage.
    """

    synthseg_output_root: str = ""
    preprocessed_root: str = ""
    original_metadata_csv: str = ""
    output_dir: str = ""

    tumor_proximity_threshold_mm: float = 20.0

    modalities_for_roi_analysis: List[str] = field(
        default_factory=lambda: list(DEFAULT_MODALITIES)
    )
    reliable_regions: List[str] = field(
        default_factory=lambda: list(DEFAULT_RELIABLE_REGIONS)
    )
    deep_grey_matter_regions: List[str] = field(
        default_factory=lambda: list(DEFAULT_DEEP_GREY_MATTER)
    )

    cv_acceptance_threshold: float = 0.05
    figure_format: str = "png"
    figure_dpi: int = 300
    n_workers: int = 4

    # Phase 1 output filenames — reused so the collector can locate inputs.
    parcellation_filename: str = "synthseg_parc.nii.gz"
    volumes_filename: str = "synthseg_volumes.csv"
    qc_filename: str = "synthseg_qc.csv"

    # Preprocessed modality filename pattern ("<modality>.nii.gz").
    input_modality_for_discovery: str = "t1n"

    def __post_init__(self) -> None:
        for name in (
            "synthseg_output_root",
            "preprocessed_root",
            "original_metadata_csv",
            "output_dir",
        ):
            if not getattr(self, name):
                raise SynthSegConfigError(f"{name} must be set")
        if self.tumor_proximity_threshold_mm <= 0:
            raise SynthSegConfigError(
                f"tumor_proximity_threshold_mm must be > 0, got "
                f"{self.tumor_proximity_threshold_mm}"
            )
        if self.figure_dpi < 72:
            raise SynthSegConfigError(
                f"figure_dpi must be >= 72, got {self.figure_dpi}"
            )
        if self.cv_acceptance_threshold <= 0:
            raise SynthSegConfigError(
                f"cv_acceptance_threshold must be > 0, got "
                f"{self.cv_acceptance_threshold}"
            )
        if self.n_workers < 1:
            raise SynthSegConfigError(
                f"n_workers must be >= 1, got {self.n_workers}"
            )
        if not self.reliable_regions:
            raise SynthSegConfigError("reliable_regions must be non-empty")
        if not self.modalities_for_roi_analysis:
            raise SynthSegConfigError(
                "modalities_for_roi_analysis must be non-empty"
            )
        if not self.deep_grey_matter_regions:
            raise SynthSegConfigError(
                "deep_grey_matter_regions must be non-empty"
            )


def load_analysis_config(yaml_path: Union[str, Path]) -> AnalysisConfig:
    """Load an :class:`AnalysisConfig` from YAML.

    The file must contain a top-level ``synthseg_analysis:`` key whose
    children map 1-to-1 onto :class:`AnalysisConfig` fields.
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Analysis config not found: {yaml_path}")

    logger.info("Loading SynthSeg analysis config from %s", yaml_path)

    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise SynthSegConfigError(f"Failed to parse YAML: {e}") from e

    if not raw:
        raise SynthSegConfigError(f"Configuration file {yaml_path} is empty")
    if "synthseg_analysis" not in raw:
        raise SynthSegConfigError(
            f"Configuration {yaml_path} must contain a top-level "
            f"'synthseg_analysis' key"
        )
    data = raw["synthseg_analysis"]
    if not isinstance(data, dict):
        raise SynthSegConfigError("'synthseg_analysis' section must be a mapping")

    try:
        cfg = AnalysisConfig(**data)
    except TypeError as e:
        raise SynthSegConfigError(f"Invalid configuration: {e}") from e

    logger.info(
        "Analysis config loaded (output_dir=%s, modalities=%s, workers=%d)",
        cfg.output_dir,
        cfg.modalities_for_roi_analysis,
        cfg.n_workers,
    )
    return cfg
