"""Configuration dataclasses for preprocessing pipeline.

This module defines configuration structures for the MenGrowth preprocessing pipeline,
including data harmonization, normalization, and other preprocessing steps.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, List, Optional, Union
import yaml
import logging

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when configuration is invalid or incomplete."""
    pass


@dataclass
class BackgroundZeroingConfig:
    """Configuration for background removal.

    Supports three options:
    - None: Skip background removal entirely
    - "border_connected_percentile": Conservative percentile-based approach
    - "self_head_mask": SELF algorithm for head-air separation

    Attributes:
        method: Background removal method (None to skip)

        # Parameters for "border_connected_percentile"
        percentile_low: Low percentile threshold for background detection [0.1-2.0]
        gaussian_sigma: Gaussian smoothing sigma before thresholding (voxels)
        min_comp_voxels: Minimum component size to consider (voxels)

        # Parameters for "self_head_mask" - SELF algorithm
        auto_fallback: Use simple fallback if SELF fails (default: True)
        fallback_threshold: Min coverage for SELF before fallback (default: 0.05)
        fallback_method: Method for fallback mask: "otsu", "percentile", "zero" (default: "otsu")
        fallback_percentile: Percentile threshold for fallback when method="percentile" (default: 10.0)
        fill_value: Value to set for background voxels (default: 0.0)

        # SELF algorithm mask parameters (from MaskParams)
        air_p_low: Percentile threshold for seeding air at dark intensities (default: 1.0)
        air_p_high: Percentile threshold to permit flood-fill through dark voxels (default: 25.0)
        air_p_global: Global percentile for darkest voxels as fallback seeds (default: 0.2)
        erode_vox: Number of erosion iterations on head seed (0=conservative, default: 0)
        close_iters: Number of iterations for final morphological smoothing (default: 1)
        connectivity: Connectivity structure: 1=6-conn, 2=18-conn, 3=26-conn (default: 2)

        # Common parameters for controlling conservativeness
        air_border_margin: Voxels to erode air mask (MORE conservative - shrinks air) (default: 1)
        expand_air_mask: Voxels to dilate air mask (LESS conservative - expands air) (default: 0)
        Note: Use either air_border_margin OR expand_air_mask, not both > 0
    """
    method: Optional[Literal["border_connected_percentile", "self_head_mask"]] = "border_connected_percentile"

    # Parameters for border_connected_percentile
    percentile_low: float = 0.7
    gaussian_sigma: float = 0.5
    min_comp_voxels: int = 500

    # Parameters for self_head_mask - fallback control
    auto_fallback: bool = True
    fallback_threshold: float = 0.05
    fallback_method: str = "otsu"
    fallback_percentile: float = 10.0
    fill_value: float = 0.0

    # SELF algorithm mask parameters (from MaskParams)
    air_p_low: float = 1.0
    air_p_high: float = 25.0
    air_p_global: float = 0.2
    erode_vox: int = 0
    close_iters: int = 1
    connectivity: int = 2

    # Common parameters
    air_border_margin: int = 1
    expand_air_mask: int = 0

    def __post_init__(self) -> None:
        """Validate configuration values."""
        # Validate method
        if self.method is not None and self.method not in ["border_connected_percentile", "self_head_mask"]:
            raise ConfigurationError(
                f"method must be None, 'border_connected_percentile', or 'self_head_mask', got {self.method}"
            )

        # Skip validation if method is None (background removal disabled)
        if self.method is None:
            return

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
            if self.fallback_method not in ["otsu", "percentile", "zero"]:
                raise ConfigurationError(
                    f"fallback_method must be 'otsu', 'percentile', or 'zero', got {self.fallback_method}"
                )
            if not 0.0 <= self.fallback_percentile <= 100.0:
                raise ConfigurationError(
                    f"fallback_percentile must be in [0.0, 100.0], got {self.fallback_percentile}"
                )

            # Validate SELF algorithm mask parameters
            if not 0.0 <= self.air_p_low <= 100.0:
                raise ConfigurationError(
                    f"air_p_low must be in [0.0, 100.0], got {self.air_p_low}"
                )
            if not 0.0 <= self.air_p_high <= 100.0:
                raise ConfigurationError(
                    f"air_p_high must be in [0.0, 100.0], got {self.air_p_high}"
                )
            if not 0.0 <= self.air_p_global <= 100.0:
                raise ConfigurationError(
                    f"air_p_global must be in [0.0, 100.0], got {self.air_p_global}"
                )
            if self.erode_vox < 0:
                raise ConfigurationError(
                    f"erode_vox must be non-negative, got {self.erode_vox}"
                )
            if self.close_iters < 0:
                raise ConfigurationError(
                    f"close_iters must be non-negative, got {self.close_iters}"
                )
            if self.connectivity not in [1, 2, 3]:
                raise ConfigurationError(
                    f"connectivity must be 1, 2, or 3, got {self.connectivity}"
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
class BiasFieldCorrectionConfig:
    """Configuration for N4 bias field correction.

    Attributes:
        method: Bias field correction method ("n4" or None to skip)
        shrink_factor: Downsampling factor for computational efficiency [1-4]
        max_iterations: Maximum iterations per resolution level (4 levels)
        bias_field_fwhm: Full-width-at-half-maximum for Gaussian smoothing [0.15-0.5]
        convergence_threshold: Early stopping convergence threshold
    """
    method: Optional[Literal["n4"]] = "n4"
    shrink_factor: int = 4
    max_iterations: List[int] = field(default_factory=lambda: [50, 50, 50, 50])
    bias_field_fwhm: float = 0.15
    convergence_threshold: float = 0.001

    def __post_init__(self) -> None:
        """Validate configuration values."""
        # Validate method
        if self.method is not None and self.method != "n4":
            raise ConfigurationError(
                f"method must be None or 'n4', got {self.method}"
            )

        # Skip validation if method is None (bias correction disabled)
        if self.method is None:
            return

        # Validate shrink_factor
        if not 1 <= self.shrink_factor <= 8:
            raise ConfigurationError(
                f"shrink_factor must be in [1, 8], got {self.shrink_factor}"
            )

        # Validate max_iterations
        if not isinstance(self.max_iterations, list):
            raise ConfigurationError(
                f"max_iterations must be a list, got {type(self.max_iterations)}"
            )
        if len(self.max_iterations) != 4:
            raise ConfigurationError(
                f"max_iterations must have exactly 4 elements (one per level), got {len(self.max_iterations)}"
            )
        if any(iters < 1 for iters in self.max_iterations):
            raise ConfigurationError(
                f"All max_iterations values must be >= 1, got {self.max_iterations}"
            )

        # Validate bias_field_fwhm
        if not 0.01 <= self.bias_field_fwhm <= 1.0:
            raise ConfigurationError(
                f"bias_field_fwhm must be in [0.01, 1.0], got {self.bias_field_fwhm}"
            )

        # Validate convergence_threshold
        if not 0.0 < self.convergence_threshold < 1.0:
            raise ConfigurationError(
                f"convergence_threshold must be in (0.0, 1.0), got {self.convergence_threshold}"
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
class Step1BiasFieldCorrectionConfig:
    """Configuration for Step 1: Bias field correction.

    Attributes:
        save_visualization: Whether to save visualization outputs for this step
        save_artifact: Whether to save bias field NIfTI to artifacts directory
        bias_field_correction: Configuration for bias field correction method
    """
    save_visualization: bool = True
    save_artifact: bool = True
    bias_field_correction: BiasFieldCorrectionConfig = field(default_factory=BiasFieldCorrectionConfig)

    def __post_init__(self) -> None:
        """Ensure bias_field_correction is a BiasFieldCorrectionConfig instance."""
        if isinstance(self.bias_field_correction, dict):
            self.bias_field_correction = BiasFieldCorrectionConfig(**self.bias_field_correction)


@dataclass
class ResamplingConfig:
    """Configuration for resampling to isotropic resolution.

    Attributes:
        method: Resampling method ("bspline", "eclare", or None to skip)
        target_voxel_size: Target voxel size in mm [x, y, z]

        # Normalization parameters (applied BEFORE resampling)
        normalize_method: Normalization method to apply before resampling
                         ("zscore", "kde", "percentile_minmax", or None to skip)
        p1: Lower percentile for "percentile_minmax" normalization (default=1.0)
        p2: Upper percentile for "percentile_minmax" normalization (default=99.0)
        norm_value: Scaling factor for "zscore" and "kde" normalization (default=1.0)

        bspline_order: BSpline interpolation order [0-5] (used if method=="bspline")
                       0: nearest neighbor, 1: linear, 3: cubic (recommended)
        conda_environment_eclare: Conda environment with ECLARE installed (used if method=="eclare")
        batch_size: Batch size for ECLARE inference (used if method=="eclare")
        n_patches: Number of patches for ECLARE training (used if method=="eclare")
        patch_sampling: Patch sampling strategy for ECLARE (used if method=="eclare")
        suffix: Suffix to add to ECLARE output filename (used if method=="eclare")
        gpu_id: GPU ID(s) to use for ECLARE - int or list of ints (used if method=="eclare")
    """
    method: Optional[Literal["bspline", "eclare", "composite"]] = "bspline"
    target_voxel_size: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])

    # Normalization parameters (applied before resampling)
    normalize_method: Optional[Literal["zscore", "kde", "percentile_minmax"]] = None
    p1: float = 1.0
    p2: float = 99.0
    norm_value: float = 1.0

    # BSpline parameters
    bspline_order: int = 3

    # ECLARE parameters
    conda_environment_eclare: str = "eclare_env"
    batch_size: int = 128
    n_patches: int = 1000000
    patch_sampling: str = "gradient"
    suffix: str = ""
    gpu_id: Union[int, List[int]] = 0

    # COMPOSITE parameters
    composite_interpolator: str = "bspline"
    composite_dl_method: str = "eclare"
    max_mm_interpolator: float = 1.2
    max_mm_dl_method: float = 5.0
    resample_mm_to_interpolator_if_max_mm_dl_method: float = 3.0

    def __post_init__(self) -> None:
        """Validate configuration values."""
        # Validate method
        if self.method is not None and self.method not in ["bspline", "eclare", "composite"]:
            raise ConfigurationError(
                f"method must be None, 'bspline', 'eclare', or 'composite', got {self.method}"
            )

        # Validate normalization parameters
        if self.normalize_method is not None:
            if self.normalize_method not in ["zscore", "kde", "percentile_minmax"]:
                raise ConfigurationError(
                    f"normalize_method must be None, 'zscore', 'kde', or 'percentile_minmax', "
                    f"got {self.normalize_method}"
                )

            # Validate percentile parameters (used by percentile_minmax)
            if not 0.0 <= self.p1 < self.p2 <= 100.0:
                raise ConfigurationError(
                    f"Percentiles must satisfy 0 <= p1 < p2 <= 100, got p1={self.p1}, p2={self.p2}"
                )

            # Validate norm_value (used by zscore and kde)
            if self.norm_value <= 0:
                raise ConfigurationError(
                    f"norm_value must be positive, got {self.norm_value}"
                )

        # Skip validation if method is None (resampling disabled)
        if self.method is None:
            return

        # Validate target_voxel_size
        if not isinstance(self.target_voxel_size, list):
            raise ConfigurationError(
                f"target_voxel_size must be a list, got {type(self.target_voxel_size)}"
            )
        if len(self.target_voxel_size) != 3:
            raise ConfigurationError(
                f"target_voxel_size must have exactly 3 elements [x, y, z], got {len(self.target_voxel_size)}"
            )
        if any(size <= 0 for size in self.target_voxel_size):
            raise ConfigurationError(
                f"All target_voxel_size values must be > 0, got {self.target_voxel_size}"
            )

        # Validate BSpline parameters (if using BSpline)
        if self.method == "bspline":
            if not isinstance(self.bspline_order, int) or not (0 <= self.bspline_order <= 5):
                raise ConfigurationError(
                    f"bspline_order must be an integer in [0, 5], got {self.bspline_order}"
                )

        # Validate ECLARE parameters (if using ECLARE)
        if self.method == "eclare":
            if not isinstance(self.conda_environment_eclare, str) or not self.conda_environment_eclare:
                raise ConfigurationError(
                    f"conda_environment_eclare must be a non-empty string, got {self.conda_environment_eclare}"
                )

            if not isinstance(self.batch_size, int) or self.batch_size <= 0:
                raise ConfigurationError(
                    f"batch_size must be a positive integer, got {self.batch_size}"
                )

            if not isinstance(self.n_patches, int) or self.n_patches <= 0:
                raise ConfigurationError(
                    f"n_patches must be a positive integer, got {self.n_patches}"
                )

            # Validate gpu_id (can be int or list of ints)
            if isinstance(self.gpu_id, int):
                if self.gpu_id < 0:
                    raise ConfigurationError(
                        f"gpu_id must be non-negative, got {self.gpu_id}"
                    )
            elif isinstance(self.gpu_id, list):
                if not all(isinstance(gpu, int) and gpu >= 0 for gpu in self.gpu_id):
                    raise ConfigurationError(
                        "gpu_id list must contain only non-negative integers"
                    )
                if len(self.gpu_id) == 0:
                    raise ConfigurationError("gpu_id list cannot be empty")
            else:
                raise ConfigurationError(
                    f"gpu_id must be int or List[int], got {type(self.gpu_id)}"
                )

        # Validate COMPOSITE parameters (if using COMPOSITE)
        if self.method == "composite":
            # Validate composite_interpolator
            if self.composite_interpolator not in ["bspline"]:
                raise ConfigurationError(
                    f"composite_interpolator must be 'bspline', got {self.composite_interpolator}"
                )

            # Validate composite_dl_method
            if self.composite_dl_method not in ["eclare"]:
                raise ConfigurationError(
                    f"composite_dl_method must be 'eclare', got {self.composite_dl_method}"
                )

            # Validate threshold ordering
            if not 0 < self.max_mm_interpolator < self.max_mm_dl_method:
                raise ConfigurationError(
                    f"Must satisfy 0 < max_mm_interpolator < max_mm_dl_method, "
                    f"got max_mm_interpolator={self.max_mm_interpolator}, "
                    f"max_mm_dl_method={self.max_mm_dl_method}"
                )

            # Validate resample_mm_to_interpolator_if_max_mm_dl_method
            if not 0 < self.resample_mm_to_interpolator_if_max_mm_dl_method < self.max_mm_dl_method:
                raise ConfigurationError(
                    f"resample_mm_to_interpolator_if_max_mm_dl_method must be between 0 and max_mm_dl_method, "
                    f"got {self.resample_mm_to_interpolator_if_max_mm_dl_method} (max_mm_dl_method={self.max_mm_dl_method})"
                )


@dataclass
class Step2ResamplingConfig:
    """Configuration for Step 2: Resampling to isotropic resolution.

    Attributes:
        save_visualization: Whether to save visualization outputs for this step
        resampling: Configuration for resampling method
    """
    save_visualization: bool = True
    resampling: ResamplingConfig = field(default_factory=ResamplingConfig)

    def __post_init__(self) -> None:
        """Ensure resampling is a ResamplingConfig instance."""
        if isinstance(self.resampling, dict):
            self.resampling = ResamplingConfig(**self.resampling)


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
        preprocessing_artifacts_path: Directory for intermediate preprocessing artifacts
        viz_root: Directory for visualization outputs
        overwrite: Allow overwriting existing files
        modalities: List of modalities to process
        step0_data_harmonization: Configuration for harmonization operations
        step1_bias_field_correction: Configuration for bias field correction
        step2_resampling: Configuration for resampling to isotropic resolution
    """
    enabled: bool = True
    patient_selector: Literal["single", "all"] = "single"
    patient_id: str = "MenGrowth-0001"
    mode: Literal["test", "pipeline"] = "test"
    dataset_root: str = ""
    output_root: str = ""
    preprocessing_artifacts_path: str = ""
    viz_root: str = ""
    overwrite: bool = False
    modalities: List[str] = field(default_factory=lambda: ["t1c", "t1n", "t2w", "t2f"])
    step0_data_harmonization: Step0DataHarmonizationConfig = field(
        default_factory=Step0DataHarmonizationConfig
    )
    step1_bias_field_correction: Step1BiasFieldCorrectionConfig = field(
        default_factory=Step1BiasFieldCorrectionConfig
    )
    step2_resampling: Step2ResamplingConfig = field(
        default_factory=Step2ResamplingConfig
    )

    def __post_init__(self) -> None:
        """Validate configuration and convert paths."""
        # Ensure step0 is a dataclass instance
        if isinstance(self.step0_data_harmonization, dict):
            self.step0_data_harmonization = Step0DataHarmonizationConfig(
                **self.step0_data_harmonization
            )

        # Ensure step1 is a dataclass instance
        if isinstance(self.step1_bias_field_correction, dict):
            self.step1_bias_field_correction = Step1BiasFieldCorrectionConfig(
                **self.step1_bias_field_correction
            )

        # Ensure step2 is a dataclass instance
        if isinstance(self.step2_resampling, dict):
            self.step2_resampling = Step2ResamplingConfig(
                **self.step2_resampling
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

        # Validate preprocessing_artifacts_path
        if not self.preprocessing_artifacts_path:
            raise ConfigurationError("preprocessing_artifacts_path must be specified")

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
