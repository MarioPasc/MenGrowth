"""Configuration dataclasses for preprocessing pipeline.

This module defines configuration structures for the MenGrowth preprocessing pipeline,
including data harmonization, normalization, and other preprocessing steps.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, List
import yaml
import logging

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when configuration is invalid or incomplete."""
    pass


@dataclass
class BackgroundZeroingConfig:
    """Configuration for background removal.

    Supports two methods:
    - "border_connected_percentile": Conservative percentile-based approach
    - "self_head_mask": SELF algorithm for head-air separation

    Attributes:
        method: Background removal method

        # Parameters for "border_connected_percentile"
        percentile_low: Low percentile threshold for background detection [0.1-2.0]
        gaussian_sigma: Gaussian smoothing sigma before thresholding (voxels)
        min_comp_voxels: Minimum component size to consider (voxels)

        # Parameters for "self_head_mask"
        auto_fallback: Use simple fallback if SELF fails (default: True)
        fallback_threshold: Min coverage for SELF before fallback (default: 0.05)
        fill_value: Value to set for background voxels (default: 0.0)

        # Common parameters for controlling conservativeness
        air_border_margin: Voxels to erode air mask (MORE conservative - shrinks air) (default: 1)
        expand_air_mask: Voxels to dilate air mask (LESS conservative - expands air) (default: 0)
        Note: Use either air_border_margin OR expand_air_mask, not both > 0
    """
    method: Literal["border_connected_percentile", "self_head_mask"] = "border_connected_percentile"

    # Parameters for border_connected_percentile
    percentile_low: float = 0.7
    gaussian_sigma: float = 0.5
    min_comp_voxels: int = 500

    # Parameters for self_head_mask
    auto_fallback: bool = True
    fallback_threshold: float = 0.05
    fill_value: float = 0.0

    # Common parameters
    air_border_margin: int = 1
    expand_air_mask: int = 0

    def __post_init__(self) -> None:
        """Validate configuration values."""
        # Validate method
        if self.method not in ["border_connected_percentile", "self_head_mask"]:
            raise ConfigurationError(
                f"method must be 'border_connected_percentile' or 'self_head_mask', got {self.method}"
            )

        # Validate border_connected_percentile parameters
        if self.method == "border_connected_percentile":
            if not 0.1 <= self.percentile_low <= 2.0:
                raise ConfigurationError(
                    f"percentile_low must be in [0.1, 2.0], got {self.percentile_low}"
                )
            if self.gaussian_sigma < 0:
                raise ConfigurationError(
                    f"gaussian_sigma must be non-negative, got {self.gaussian_sigma}"
                )
            if self.min_comp_voxels < 0:
                raise ConfigurationError(
                    f"min_comp_voxels must be non-negative, got {self.min_comp_voxels}"
                )

        # Validate self_head_mask parameters
        if self.method == "self_head_mask":
            if not 0.0 <= self.fallback_threshold <= 1.0:
                raise ConfigurationError(
                    f"fallback_threshold must be in [0.0, 1.0], got {self.fallback_threshold}"
                )

        # Validate common parameters
        if self.air_border_margin < 0:
            raise ConfigurationError(
                f"air_border_margin must be non-negative, got {self.air_border_margin}"
            )
        if self.expand_air_mask < 0:
            raise ConfigurationError(
                f"expand_air_mask must be non-negative, got {self.expand_air_mask}"
            )

        # Warn if both erosion and dilation are enabled (conflicting intentions)
        if self.air_border_margin > 0 and self.expand_air_mask > 0:
            import logging
            logging.getLogger(__name__).warning(
                f"Both air_border_margin ({self.air_border_margin}) and expand_air_mask ({self.expand_air_mask}) "
                "are > 0. These have opposite effects. Consider using only one."
            )


@dataclass
class Step0DataHarmonizationConfig:
    """Configuration for Step 0: Data harmonization (NRRD to NIfTI, reorient, background removal).

    Attributes:
        save_visualization: Whether to save visualization outputs for this step
        reorient_to: Target orientation convention ("RAS" or "LPS")
        background_zeroing: Configuration for background removal
    """
    save_visualization: bool = True
    reorient_to: Literal["RAS", "LPS"] = "RAS"
    background_zeroing: BackgroundZeroingConfig = field(default_factory=BackgroundZeroingConfig)

    def __post_init__(self) -> None:
        """Ensure background_zeroing is a BackgroundZeroingConfig instance."""
        if isinstance(self.background_zeroing, dict):
            self.background_zeroing = BackgroundZeroingConfig(**self.background_zeroing)


@dataclass
class DataHarmonizationConfig:
    """Configuration for the data harmonization preprocessing stage.

    Attributes:
        enabled: Whether this stage is enabled
        patient_selector: Select "single" patient or "all" patients
        patient_id: Patient ID to process (used only if patient_selector == "single")
        mode: Operating mode - "test" (separate output) or "pipeline" (in-place)
        dataset_root: Root directory of the MenGrowth dataset
        output_root: Output directory for test mode
        viz_root: Directory for visualization outputs
        overwrite: Allow overwriting existing files
        modalities: List of modalities to process
        step0_data_harmonization: Configuration for harmonization operations
    """
    enabled: bool = True
    patient_selector: Literal["single", "all"] = "single"
    patient_id: str = "MenGrowth-0001"
    mode: Literal["test", "pipeline"] = "test"
    dataset_root: str = ""
    output_root: str = ""
    viz_root: str = ""
    overwrite: bool = False
    modalities: List[str] = field(default_factory=lambda: ["t1c", "t1n", "t2w", "t2f"])
    step0_data_harmonization: Step0DataHarmonizationConfig = field(
        default_factory=Step0DataHarmonizationConfig
    )

    def __post_init__(self) -> None:
        """Validate configuration and convert paths."""
        # Ensure step0 is a dataclass instance
        if isinstance(self.step0_data_harmonization, dict):
            self.step0_data_harmonization = Step0DataHarmonizationConfig(
                **self.step0_data_harmonization
            )

        # Validate dataset_root
        if not self.dataset_root:
            raise ConfigurationError("dataset_root must be specified")

        dataset_path = Path(self.dataset_root)
        if not dataset_path.exists():
            raise ConfigurationError(
                f"dataset_root does not exist: {self.dataset_root}"
            )

        # Validate output_root for test mode
        if self.mode == "test":
            if not self.output_root:
                raise ConfigurationError(
                    "output_root must be specified in test mode"
                )

        # Validate viz_root
        if not self.viz_root:
            raise ConfigurationError("viz_root must be specified")

        # Validate patient_id for single mode
        if self.patient_selector == "single" and not self.patient_id:
            raise ConfigurationError(
                "patient_id must be specified when patient_selector is 'single'"
            )

        # Validate modalities
        if not self.modalities:
            raise ConfigurationError("At least one modality must be specified")

        logger.info(f"DataHarmonizationConfig validated: mode={self.mode}, "
                   f"patient_selector={self.patient_selector}, "
                   f"overwrite={self.overwrite}")


@dataclass
class PreprocessingPipelineConfig:
    """Top-level configuration for the preprocessing pipeline.

    Attributes:
        data_harmonization: Configuration for data harmonization stage
    """
    data_harmonization: DataHarmonizationConfig = field(
        default_factory=DataHarmonizationConfig
    )

    def __post_init__(self) -> None:
        """Ensure data_harmonization is a DataHarmonizationConfig instance."""
        if isinstance(self.data_harmonization, dict):
            self.data_harmonization = DataHarmonizationConfig(**self.data_harmonization)


def load_preprocessing_pipeline_config(config_path: Path) -> PreprocessingPipelineConfig:
    """Load and validate preprocessing pipeline configuration from YAML.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Validated PreprocessingPipelineConfig object

    Raises:
        FileNotFoundError: If config file does not exist
        ConfigurationError: If configuration is invalid
        yaml.YAMLError: If YAML parsing fails
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    logger.info(f"Loading preprocessing pipeline config from {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Failed to parse YAML: {e}") from e

    if not yaml_data:
        raise ConfigurationError("Configuration file is empty")

    if "preprocessing" not in yaml_data:
        raise ConfigurationError(
            "Configuration must contain 'preprocessing' top-level key"
        )

    preprocessing_data = yaml_data["preprocessing"]

    try:
        config = PreprocessingPipelineConfig(**preprocessing_data)
    except TypeError as e:
        raise ConfigurationError(f"Invalid configuration structure: {e}") from e

    logger.info("Preprocessing pipeline configuration loaded successfully")

    return config
