"""Configuration dataclasses for preprocessing pipeline.

This module defines configuration structures for the MenGrowth preprocessing pipeline,
including data harmonization, normalization, and other preprocessing steps.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Literal,
    List,
    Optional,
    Union,
    Dict,
    Any,
    Callable,
    Tuple,
    TYPE_CHECKING,
)
import yaml
import logging

if TYPE_CHECKING:
    from mengrowth.preprocessing.src.preprocess import PreprocessingOrchestrator

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when configuration is invalid or incomplete."""

    pass


# ============================================================================
# Dynamic Pipeline Infrastructure
# ============================================================================


@dataclass
class StepMetadata:
    """Metadata describing step execution characteristics.

    Attributes:
        level: Whether step operates per-modality, at study level, or at patient level
        description: Human-readable description of the step
    """

    level: Literal["modality", "study", "patient"]
    description: str = ""


# Step metadata registry - defines execution level for each step type
# IMPORTANT: Order matters for substring matching! More specific patterns before general ones
STEP_METADATA: Dict[str, StepMetadata] = {
    "data_harmonization": StepMetadata(
        level="modality",
        description="NRRD→NIfTI conversion, reorientation, background removal",
    ),
    "bias_field_correction": StepMetadata(
        level="modality", description="N4 bias field correction"
    ),
    "resampling": StepMetadata(
        level="modality",
        description="Isotropic resampling (bspline, eclare, composite)",
    ),
    "cubic_padding": StepMetadata(
        level="study",
        description="Zero-pad volumes to cubic shape for registration stability",
    ),
    "longitudinal_registration": StepMetadata(
        level="patient",
        description="Longitudinal registration across patient timestamps",
    ),
    "registration": StepMetadata(
        level="study", description="Multi-modal coregistration and atlas registration"
    ),
    "skull_stripping": StepMetadata(
        level="study", description="Brain extraction (HD-BET, SynthStrip)"
    ),
    "intensity_normalization": StepMetadata(
        level="modality", description="Intensity normalization (zscore, kde, fcm, etc.)"
    ),
}


@dataclass
class StepExecutionContext:
    """Context passed to each step function containing all necessary state.

    Attributes:
        patient_id: Patient identifier
        study_dir: Path to study directory (None for patient-level steps)
        modality: Modality being processed (e.g., "t1c", "t1n"), None for study-level and patient-level steps
        paths: Dictionary of output paths from _get_output_paths()
        orchestrator: Reference to PreprocessingOrchestrator instance
        step_name: Full step name from config (e.g., "intensity_normalization_2")
        step_config: Step-specific configuration object
        all_study_dirs: List of all study directories for patient-level steps
    """

    patient_id: str
    study_dir: Optional[Path]
    modality: Optional[str]
    paths: Optional[Dict[str, Path]]
    orchestrator: "PreprocessingOrchestrator"
    step_name: str
    step_config: Any
    all_study_dirs: Optional[List[Path]] = None


class StepRegistry:
    """Registry mapping step name patterns to execution functions.

    Supports substring matching: if a registered pattern appears anywhere
    in a step name, that step's handler is invoked.

    Example:
        registry = StepRegistry()
        registry.register("intensity_normalization", normalize_func)

        # Both match the same handler:
        registry.get_handler("intensity_normalization_1")  # matches
        registry.get_handler("intensity_normalization_kde")  # matches
    """

    def __init__(self) -> None:
        """Initialize empty step registry."""
        self._registry: Dict[str, Callable] = {}

    def register(self, pattern: str, func: Callable) -> None:
        """Register a step function with a pattern name.

        Args:
            pattern: Pattern to match (e.g., "data_harmonization", "intensity_normalization")
            func: Function to call for this step
        """
        self._registry[pattern] = func
        logger.debug(f"Registered step pattern: {pattern}")

    def get_handler(self, step_name: str) -> Tuple[str, Callable]:
        """Find handler for a step name using substring matching.

        Args:
            step_name: Full step name from config (e.g., "intensity_normalization_1")

        Returns:
            Tuple of (matched_pattern, handler_function)

        Raises:
            ValueError: If no matching pattern found
        """
        for pattern, func in self._registry.items():
            if pattern in step_name:
                logger.debug(f"Matched '{step_name}' to pattern '{pattern}'")
                return pattern, func

        raise ValueError(
            f"No handler found for step '{step_name}'. "
            f"Available patterns: {list(self._registry.keys())}"
        )

    def list_patterns(self) -> List[str]:
        """List all registered step patterns.

        Returns:
            List of registered pattern names
        """
        return list(self._registry.keys())


# ============================================================================
# Configuration Dataclasses
# ============================================================================


@dataclass
class BackgroundZeroingConfig:
    """Configuration for background removal.

    Supports four options:
    - None: Skip background removal entirely
    - "border_connected_percentile": Conservative percentile-based approach
    - "self_head_mask": SELF algorithm for head-air separation
    - "otsu_foreground": Otsu-based foreground extraction (robust, recommended)

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

        # Parameters for "otsu_foreground" - Otsu-based extraction
        gaussian_sigma_px: Gaussian smoothing sigma in pixels (default: 1.0)
        min_component_voxels: Minimum component size in voxels (default: 1000)
        n_components_to_keep: Number of largest components to keep (default: 1)
        relaxed_threshold_factor: Factor for secondary components threshold (default: 0.1)
        p_low: Lower percentile for intensity scaling (default: 1.0)
        p_high: Upper percentile for intensity scaling (default: 99.0)

        # Common parameters for controlling conservativeness
        air_border_margin: Voxels to erode air mask (MORE conservative - shrinks air) (default: 1)
        expand_air_mask: Voxels to dilate air mask (LESS conservative - expands air) (default: 0)
        Note: Use either air_border_margin OR expand_air_mask, not both > 0
    """

    method: Optional[
        Literal["border_connected_percentile", "self_head_mask", "otsu_foreground"]
    ] = "border_connected_percentile"

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

    # Parameters for otsu_foreground
    gaussian_sigma_px: float = 1.0
    min_component_voxels: int = 1000
    n_components_to_keep: int = 1
    relaxed_threshold_factor: float = 0.1
    p_low: float = 1.0
    p_high: float = 99.0

    # Common parameters
    air_border_margin: int = 1
    expand_air_mask: int = 0

    def __post_init__(self) -> None:
        """Validate configuration values."""
        # Validate method
        valid_methods = [
            "border_connected_percentile",
            "self_head_mask",
            "otsu_foreground",
        ]
        if self.method is not None and self.method not in valid_methods:
            raise ConfigurationError(
                f"method must be None or one of {valid_methods}, got {self.method}"
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

        # Validate otsu_foreground parameters
        if self.method == "otsu_foreground":
            if self.gaussian_sigma_px < 0:
                raise ConfigurationError(
                    f"gaussian_sigma_px must be non-negative, got {self.gaussian_sigma_px}"
                )
            if self.min_component_voxels < 0:
                raise ConfigurationError(
                    f"min_component_voxels must be non-negative, got {self.min_component_voxels}"
                )
            if self.n_components_to_keep < 1:
                raise ConfigurationError(
                    f"n_components_to_keep must be >= 1, got {self.n_components_to_keep}"
                )
            if not 0.0 <= self.relaxed_threshold_factor <= 1.0:
                raise ConfigurationError(
                    f"relaxed_threshold_factor must be in [0.0, 1.0], got {self.relaxed_threshold_factor}"
                )
            if not 0.0 <= self.p_low <= 100.0:
                raise ConfigurationError(
                    f"p_low must be in [0.0, 100.0], got {self.p_low}"
                )
            if not 0.0 <= self.p_high <= 100.0:
                raise ConfigurationError(
                    f"p_high must be in [0.0, 100.0], got {self.p_high}"
                )
            if self.p_low >= self.p_high:
                raise ConfigurationError(
                    f"p_low ({self.p_low}) must be less than p_high ({self.p_high})"
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
            raise ConfigurationError(f"method must be None or 'n4', got {self.method}")

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
class DataHarmonizationStepConfig:
    """Configuration for data harmonization step (NRRD to NIfTI, reorient, background removal).

    Attributes:
        save_visualization: Whether to save visualization outputs for this step
        reorient_to: Target orientation convention ("RAS" or "LPS")
        background_zeroing: Configuration for background removal
    """

    save_visualization: bool = True
    reorient_to: Literal["RAS", "LPS"] = "RAS"
    background_zeroing: BackgroundZeroingConfig = field(
        default_factory=BackgroundZeroingConfig
    )

    def __post_init__(self) -> None:
        """Ensure background_zeroing is a BackgroundZeroingConfig instance."""
        if isinstance(self.background_zeroing, dict):
            self.background_zeroing = BackgroundZeroingConfig(**self.background_zeroing)


# Backwards compatibility alias
Step0DataHarmonizationConfig = DataHarmonizationStepConfig


@dataclass
class BiasFieldCorrectionStepConfig:
    """Configuration for bias field correction step.

    Attributes:
        save_visualization: Whether to save visualization outputs for this step
        save_artifact: Whether to save bias field NIfTI to artifacts directory
        bias_field_correction: Configuration for bias field correction method
    """

    save_visualization: bool = True
    save_artifact: bool = True
    bias_field_correction: BiasFieldCorrectionConfig = field(
        default_factory=BiasFieldCorrectionConfig
    )

    def __post_init__(self) -> None:
        """Ensure bias_field_correction is a BiasFieldCorrectionConfig instance."""
        if isinstance(self.bias_field_correction, dict):
            self.bias_field_correction = BiasFieldCorrectionConfig(
                **self.bias_field_correction
            )


# Backwards compatibility alias
Step1BiasFieldCorrectionConfig = BiasFieldCorrectionStepConfig


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
    normalize_method: Optional[
        Literal["zscore", "kde", "percentile_minmax", "whitestripe", "fcm", "lsq"]
    ] = None
    p1: float = 1.0
    p2: float = 99.0
    norm_value: float = 1.0

    # WhiteStripe parameters
    whitestripe_width: float = 0.05
    whitestripe_width_l: Optional[float] = None
    whitestripe_width_u: Optional[float] = None

    # FCM parameters
    fcm_n_clusters: int = 3
    fcm_tissue_type: str = "WM"
    fcm_max_iter: int = 50
    fcm_error_threshold: float = 0.005
    fcm_fuzziness: float = 2.0

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
        if self.method is not None and self.method not in [
            "bspline",
            "eclare",
            "composite",
        ]:
            raise ConfigurationError(
                f"method must be None, 'bspline', 'eclare', or 'composite', got {self.method}"
            )

        # Validate normalization parameters
        if self.normalize_method is not None:
            valid_methods = [
                "zscore",
                "kde",
                "percentile_minmax",
                "whitestripe",
                "fcm",
                "lsq",
            ]
            if self.normalize_method not in valid_methods:
                raise ConfigurationError(
                    f"normalize_method must be None or one of {valid_methods}, "
                    f"got {self.normalize_method}"
                )

            # Validate percentile parameters (used by percentile_minmax)
            if not 0.0 <= self.p1 < self.p2 <= 100.0:
                raise ConfigurationError(
                    f"Percentiles must satisfy 0 <= p1 < p2 <= 100, got p1={self.p1}, p2={self.p2}"
                )

            # Validate norm_value (used by zscore, kde, lsq)
            if self.norm_value <= 0:
                raise ConfigurationError(
                    f"norm_value must be positive, got {self.norm_value}"
                )

            # Validate WhiteStripe parameters
            if self.normalize_method == "whitestripe":
                if not 0.0 < self.whitestripe_width <= 1.0:
                    raise ConfigurationError(
                        f"whitestripe_width must be in (0.0, 1.0], got {self.whitestripe_width}"
                    )
                if (
                    self.whitestripe_width_l is not None
                    and not 0.0 < self.whitestripe_width_l <= 1.0
                ):
                    raise ConfigurationError(
                        f"whitestripe_width_l must be in (0.0, 1.0], got {self.whitestripe_width_l}"
                    )
                if (
                    self.whitestripe_width_u is not None
                    and not 0.0 < self.whitestripe_width_u <= 1.0
                ):
                    raise ConfigurationError(
                        f"whitestripe_width_u must be in (0.0, 1.0], got {self.whitestripe_width_u}"
                    )

            # Validate FCM parameters
            if self.normalize_method == "fcm":
                if not 2 <= self.fcm_n_clusters <= 10:
                    raise ConfigurationError(
                        f"fcm_n_clusters must be in [2, 10], got {self.fcm_n_clusters}"
                    )
                if self.fcm_tissue_type not in ["WM", "GM", "CSF"]:
                    raise ConfigurationError(
                        f"fcm_tissue_type must be 'WM', 'GM', or 'CSF', got {self.fcm_tissue_type}"
                    )
                if self.fcm_max_iter < 1:
                    raise ConfigurationError(
                        f"fcm_max_iter must be >= 1, got {self.fcm_max_iter}"
                    )
                if not 0.0 < self.fcm_error_threshold < 1.0:
                    raise ConfigurationError(
                        f"fcm_error_threshold must be in (0.0, 1.0), got {self.fcm_error_threshold}"
                    )
                if not 1.0 < self.fcm_fuzziness <= 10.0:
                    raise ConfigurationError(
                        f"fcm_fuzziness must be in (1.0, 10.0], got {self.fcm_fuzziness}"
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
            if not isinstance(self.bspline_order, int) or not (
                0 <= self.bspline_order <= 5
            ):
                raise ConfigurationError(
                    f"bspline_order must be an integer in [0, 5], got {self.bspline_order}"
                )

        # Validate ECLARE parameters (if using ECLARE)
        if self.method == "eclare":
            if (
                not isinstance(self.conda_environment_eclare, str)
                or not self.conda_environment_eclare
            ):
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
            if (
                not 0
                < self.resample_mm_to_interpolator_if_max_mm_dl_method
                < self.max_mm_dl_method
            ):
                raise ConfigurationError(
                    f"resample_mm_to_interpolator_if_max_mm_dl_method must be between 0 and max_mm_dl_method, "
                    f"got {self.resample_mm_to_interpolator_if_max_mm_dl_method} (max_mm_dl_method={self.max_mm_dl_method})"
                )


@dataclass
class ResamplingStepConfig:
    """Configuration for resampling step (isotropic resolution).

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


# Backwards compatibility alias
Step2ResamplingConfig = ResamplingStepConfig


@dataclass
class CubicPaddingConfig:
    """Configuration for cubic padding to normalize field-of-view.

    This step pads volumes to a cubic shape before registration to:
    - Normalize FOV across different sequences (T1, T2, FLAIR)
    - Reduce boundary artifacts during registration transforms
    - Prevent edge clipping during affine rotations/translations

    Attributes:
        method: Padding method ("symmetric" or None to skip)
            - "symmetric": Center the brain with equal padding on both sides
        fill_value_mode: How to determine the fill value for padding
            - "min": Use the minimum intensity value of each image (recommended)
            - "zero": Always use 0
        target_shape_mode: How to determine the target cubic size
            - "max_across_modalities": Use max dimension across all modalities in study
            - "max_per_modality": Use max dimension of each individual image
    """

    method: Optional[Literal["symmetric"]] = "symmetric"
    fill_value_mode: Literal["min", "zero"] = "min"
    target_shape_mode: Literal["max_across_modalities", "max_per_modality"] = (
        "max_across_modalities"
    )

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.method is not None and self.method not in ["symmetric"]:
            raise ConfigurationError(
                f"method must be None or 'symmetric', got {self.method}"
            )
        if self.fill_value_mode not in ["min", "zero"]:
            raise ConfigurationError(
                f"fill_value_mode must be 'min' or 'zero', got {self.fill_value_mode}"
            )
        if self.target_shape_mode not in ["max_across_modalities", "max_per_modality"]:
            raise ConfigurationError(
                f"target_shape_mode must be 'max_across_modalities' or 'max_per_modality', "
                f"got {self.target_shape_mode}"
            )


@dataclass
class CubicPaddingStepConfig:
    """Configuration for cubic padding step.

    Attributes:
        save_visualization: Whether to save visualization outputs
        cubic_padding: Configuration for padding method
    """

    save_visualization: bool = True
    cubic_padding: CubicPaddingConfig = field(default_factory=CubicPaddingConfig)

    def __post_init__(self) -> None:
        """Ensure cubic_padding is a CubicPaddingConfig instance."""
        if isinstance(self.cubic_padding, dict):
            self.cubic_padding = CubicPaddingConfig(**self.cubic_padding)


@dataclass
class IntraStudyToReferenceConfig:
    """Configuration for intra-study multi-modal coregistration to reference.

    This step registers all modalities within a study to a reference modality
    (e.g., T1n, T2w, T2-FLAIR → T1c).

    Attributes:
        method: Registration method ("ants" or None to skip)
        reference_modality_priority: Priority order for selecting reference modality
                                      e.g., "t1c > t1n > t2f > t2w"
        transform_type: Type of transform(s) - single string ("Rigid", "Affine", "SyN")
                       or list of transforms (e.g., ["Rigid", "Affine"])
        metric: Similarity metric for registration ("Mattes", "MI", "CC", etc.)
        metric_bins: Number of bins for mutual information metric
        sampling_strategy: Sampling strategy ("Random", "Regular", or "None")
        sampling_percentage: Percentage of voxels to sample (0.0-1.0)
        number_of_iterations: Iterations per resolution level (list of lists)
        shrink_factors: Downsampling factors per level (list of lists)
        smoothing_sigmas: Smoothing sigmas per level (list of lists)
        convergence_threshold: Convergence threshold for optimization
        convergence_window_size: Window size for convergence detection
        write_composite_transform: Write composite transform file (.h5)
        interpolation: Interpolation method ("Linear", "BSpline", "NearestNeighbor")
        use_center_of_mass_init: Enable center-of-mass initialization before registration
        validate_registration_quality: Compute and log registration quality metrics
        quality_warning_threshold: Warn if correlation dissimilarity > this threshold
    """

    method: Optional[Literal["ants"]] = "ants"
    engine: Optional[Literal["nipype", "antspyx"]] = None
    reference_modality_priority: str = "t1c > t1n > t2f > t2w"
    transform_type: Union[str, List[str]] = "Rigid"
    metric: str = "Mattes"
    metric_bins: int = 32
    sampling_strategy: str = "Random"
    sampling_percentage: float = 0.2
    number_of_iterations: List[List[int]] = field(
        default_factory=lambda: [[1000, 500, 250]]
    )
    shrink_factors: List[List[int]] = field(default_factory=lambda: [[8, 4, 2, 1]])
    smoothing_sigmas: List[List[int]] = field(default_factory=lambda: [[4, 2, 1, 0]])
    convergence_threshold: float = 1e-6
    convergence_window_size: int = 10
    write_composite_transform: bool = True
    interpolation: str = "Linear"
    save_detailed_registration_info: bool = False
    use_center_of_mass_init: bool = True
    validate_registration_quality: bool = True
    quality_warning_threshold: float = -0.3

    def __post_init__(self) -> None:
        """Validate configuration values."""
        # Validate method
        if self.method is not None and self.method not in ["ants"]:
            raise ConfigurationError(
                f"method must be None or 'ants', got {self.method}"
            )

        # Validate engine (if specified)
        if self.engine is not None and self.engine not in ["nipype", "antspyx"]:
            raise ConfigurationError(
                f"engine must be None, 'nipype', or 'antspyx', got {self.engine}"
            )

        # Skip validation if method is None (registration disabled)
        if self.method is None:
            return

        # Validate reference_modality_priority
        if not self.reference_modality_priority:
            raise ConfigurationError("reference_modality_priority cannot be empty")
        if ">" not in self.reference_modality_priority:
            # Single modality specified
            if not self.reference_modality_priority.strip():
                raise ConfigurationError(
                    "reference_modality_priority must specify at least one modality"
                )

        # Validate transform_type (can be string or list of strings)
        valid_transforms = ["Rigid", "Affine", "SyN"]
        if isinstance(self.transform_type, str):
            if self.transform_type not in valid_transforms:
                raise ConfigurationError(
                    f"transform_type must be one of {valid_transforms}, got {self.transform_type}"
                )
        elif isinstance(self.transform_type, list):
            if not self.transform_type:
                raise ConfigurationError("transform_type list cannot be empty")
            for t in self.transform_type:
                if t not in valid_transforms:
                    raise ConfigurationError(
                        f"Invalid transform '{t}' in transform_type. Must be one of {valid_transforms}"
                    )
            # Validate that multi-resolution parameters match the number of transforms
            if len(self.number_of_iterations) != len(self.transform_type):
                raise ConfigurationError(
                    f"number_of_iterations must have one sublist per transform. "
                    f"Got {len(self.number_of_iterations)} sublists for {len(self.transform_type)} transforms"
                )
            if len(self.shrink_factors) != len(self.transform_type):
                raise ConfigurationError(
                    f"shrink_factors must have one sublist per transform. "
                    f"Got {len(self.shrink_factors)} sublists for {len(self.transform_type)} transforms"
                )
            if len(self.smoothing_sigmas) != len(self.transform_type):
                raise ConfigurationError(
                    f"smoothing_sigmas must have one sublist per transform. "
                    f"Got {len(self.smoothing_sigmas)} sublists for {len(self.transform_type)} transforms"
                )
        else:
            raise ConfigurationError(
                f"transform_type must be a string or list of strings, got {type(self.transform_type)}"
            )

        # Validate metric
        valid_metrics = ["Mattes", "MI", "CC", "MeanSquares", "Demons"]
        if self.metric not in valid_metrics:
            raise ConfigurationError(
                f"metric must be one of {valid_metrics}, got {self.metric}"
            )

        # Validate metric_bins
        if not 8 <= self.metric_bins <= 128:
            raise ConfigurationError(
                f"metric_bins must be in [8, 128], got {self.metric_bins}"
            )

        # Validate sampling_strategy
        if self.sampling_strategy not in ["Random", "Regular", "None"]:
            raise ConfigurationError(
                f"sampling_strategy must be 'Random', 'Regular', or 'None', got {self.sampling_strategy}"
            )

        # Validate sampling_percentage
        if not 0.0 < self.sampling_percentage <= 1.0:
            raise ConfigurationError(
                f"sampling_percentage must be in (0.0, 1.0], got {self.sampling_percentage}"
            )

        # Validate number_of_iterations
        if not isinstance(self.number_of_iterations, list):
            raise ConfigurationError(
                f"number_of_iterations must be a list, got {type(self.number_of_iterations)}"
            )
        if not all(isinstance(level, list) for level in self.number_of_iterations):
            raise ConfigurationError("number_of_iterations must be a list of lists")

        # Validate shrink_factors
        if not isinstance(self.shrink_factors, list):
            raise ConfigurationError(
                f"shrink_factors must be a list, got {type(self.shrink_factors)}"
            )
        if not all(isinstance(level, list) for level in self.shrink_factors):
            raise ConfigurationError("shrink_factors must be a list of lists")

        # Validate smoothing_sigmas
        if not isinstance(self.smoothing_sigmas, list):
            raise ConfigurationError(
                f"smoothing_sigmas must be a list, got {type(self.smoothing_sigmas)}"
            )
        if not all(isinstance(level, list) for level in self.smoothing_sigmas):
            raise ConfigurationError("smoothing_sigmas must be a list of lists")

        # Validate convergence_threshold
        if self.convergence_threshold <= 0:
            raise ConfigurationError(
                f"convergence_threshold must be positive, got {self.convergence_threshold}"
            )

        # Validate convergence_window_size
        if self.convergence_window_size < 1:
            raise ConfigurationError(
                f"convergence_window_size must be >= 1, got {self.convergence_window_size}"
            )

        # Validate interpolation
        valid_interpolations = [
            "Linear",
            "BSpline",
            "NearestNeighbor",
            "MultiLabel",
            "Gaussian",
        ]
        if self.interpolation not in valid_interpolations:
            raise ConfigurationError(
                f"interpolation must be one of {valid_interpolations}, got {self.interpolation}"
            )


@dataclass
class IntraStudyToAtlasConfig:
    """Configuration for registering reference modality to atlas space.

    This step registers the reference modality to an atlas (e.g., SRI24) and
    propagates the transform to all other modalities, bringing the entire study
    into atlas space.

    Attributes:
        method: Registration method ("ants" or None to skip)
        atlas_path: Path to atlas NIfTI file (e.g., SRI24_T1.nii.gz)
        transforms: List of transform types to apply (e.g., ["Rigid", "Affine"])
        create_composite_transforms: Whether to create composite M→atlas transforms
        metric: Similarity metric for registration
        metric_bins: Number of bins for mutual information metric
        sampling_strategy: Sampling strategy ("Random", "Regular", or "None")
        sampling_percentage: Percentage of voxels to sample (0.0-1.0)
        number_of_iterations: Iterations per resolution level (one sublist per transform)
        shrink_factors: Downsampling factors per level (one sublist per transform)
        smoothing_sigmas: Smoothing sigmas per level (one sublist per transform)
        convergence_threshold: Convergence threshold for optimization
        convergence_window_size: Window size for convergence detection
        interpolation: Interpolation method for intensities
        use_center_of_mass_init: Enable center-of-mass initialization before registration
        validate_registration_quality: Compute and log registration quality metrics
        quality_warning_threshold: Warn if correlation dissimilarity > this threshold
    """

    method: Optional[Literal["ants"]] = "ants"
    engine: Optional[Literal["nipype", "antspyx"]] = None
    atlas_path: str = ""
    transforms: List[str] = field(default_factory=lambda: ["Rigid", "Affine"])
    create_composite_transforms: bool = False
    metric: str = "Mattes"
    metric_bins: int = 32
    sampling_strategy: str = "Random"
    sampling_percentage: float = 0.2
    number_of_iterations: List[List[int]] = field(
        default_factory=lambda: [[2000, 1000, 500, 250], [1000, 500, 250, 100]]
    )
    shrink_factors: List[List[int]] = field(
        default_factory=lambda: [[8, 4, 2, 1], [4, 2, 1, 1]]
    )
    smoothing_sigmas: List[List[int]] = field(
        default_factory=lambda: [[4, 2, 1, 0], [2, 1, 0, 0]]
    )
    convergence_threshold: float = 1e-6
    convergence_window_size: int = 10
    interpolation: str = "Linear"
    save_detailed_registration_info: bool = False
    use_center_of_mass_init: bool = True
    validate_registration_quality: bool = True
    quality_warning_threshold: float = -0.3

    def __post_init__(self) -> None:
        """Validate configuration values."""
        # Validate method
        if self.method is not None and self.method not in ["ants"]:
            raise ConfigurationError(
                f"method must be None or 'ants', got {self.method}"
            )

        # Validate engine (if specified)
        if self.engine is not None and self.engine not in ["nipype", "antspyx"]:
            raise ConfigurationError(
                f"engine must be None, 'nipype', or 'antspyx', got {self.engine}"
            )

        # Skip validation if method is None (atlas registration disabled)
        if self.method is None:
            return

        # Validate atlas_path
        if not self.atlas_path:
            raise ConfigurationError(
                "atlas_path must be specified when method is not None"
            )

        # Validate transforms
        if not self.transforms:
            raise ConfigurationError("transforms list cannot be empty")
        valid_transforms = ["Rigid", "Affine", "SyN"]
        for t in self.transforms:
            if t not in valid_transforms:
                raise ConfigurationError(
                    f"Invalid transform '{t}'. Must be one of {valid_transforms}"
                )

        # Validate that number_of_iterations, shrink_factors, smoothing_sigmas have same length as transforms
        if len(self.number_of_iterations) != len(self.transforms):
            raise ConfigurationError(
                f"number_of_iterations must have one sublist per transform. "
                f"Got {len(self.number_of_iterations)} sublists for {len(self.transforms)} transforms"
            )
        if len(self.shrink_factors) != len(self.transforms):
            raise ConfigurationError(
                f"shrink_factors must have one sublist per transform. "
                f"Got {len(self.shrink_factors)} sublists for {len(self.transforms)} transforms"
            )
        if len(self.smoothing_sigmas) != len(self.transforms):
            raise ConfigurationError(
                f"smoothing_sigmas must have one sublist per transform. "
                f"Got {len(self.smoothing_sigmas)} sublists for {len(self.transforms)} transforms"
            )

        # Validate metric
        valid_metrics = ["Mattes", "MI", "CC", "MeanSquares", "Demons"]
        if self.metric not in valid_metrics:
            raise ConfigurationError(
                f"metric must be one of {valid_metrics}, got {self.metric}"
            )

        # Validate metric_bins
        if not 8 <= self.metric_bins <= 128:
            raise ConfigurationError(
                f"metric_bins must be in [8, 128], got {self.metric_bins}"
            )

        # Validate sampling_strategy
        if self.sampling_strategy not in ["Random", "Regular", "None"]:
            raise ConfigurationError(
                f"sampling_strategy must be 'Random', 'Regular', or 'None', got {self.sampling_strategy}"
            )

        # Validate sampling_percentage
        if not 0.0 < self.sampling_percentage <= 1.0:
            raise ConfigurationError(
                f"sampling_percentage must be in (0.0, 1.0], got {self.sampling_percentage}"
            )

        # Validate convergence_threshold
        if self.convergence_threshold <= 0:
            raise ConfigurationError(
                f"convergence_threshold must be positive, got {self.convergence_threshold}"
            )

        # Validate convergence_window_size
        if self.convergence_window_size < 1:
            raise ConfigurationError(
                f"convergence_window_size must be >= 1, got {self.convergence_window_size}"
            )

        # Validate interpolation
        valid_interpolations = [
            "Linear",
            "BSpline",
            "NearestNeighbor",
            "MultiLabel",
            "Gaussian",
        ]
        if self.interpolation not in valid_interpolations:
            raise ConfigurationError(
                f"interpolation must be one of {valid_interpolations}, got {self.interpolation}"
            )


@dataclass
class RegistrationStepConfig:
    """Configuration for registration step (multi-modal coregistration and atlas registration).

    This step consists of two sub-steps:
    1. Intra-study to reference: Register all modalities within a study to a reference
    2. Intra-study to atlas: Register the reference (and all modalities) to atlas space

    Attributes:
        save_visualization: Whether to save visualization outputs for this step
        intra_study_to_reference: Configuration for intra-study coregistration
        intra_study_to_atlas: Configuration for atlas registration
    """

    save_visualization: bool = True
    save_detailed_registration_info: bool = False
    intra_study_to_reference: IntraStudyToReferenceConfig = field(
        default_factory=IntraStudyToReferenceConfig
    )
    intra_study_to_atlas: IntraStudyToAtlasConfig = field(
        default_factory=IntraStudyToAtlasConfig
    )

    def __post_init__(self) -> None:
        """Ensure sub-configs are proper dataclass instances."""
        if isinstance(self.intra_study_to_reference, dict):
            self.intra_study_to_reference = IntraStudyToReferenceConfig(
                **self.intra_study_to_reference
            )
        if isinstance(self.intra_study_to_atlas, dict):
            self.intra_study_to_atlas = IntraStudyToAtlasConfig(
                **self.intra_study_to_atlas
            )


# Backwards compatibility alias
Step3RegistrationConfig = RegistrationStepConfig


@dataclass
class SkullStrippingConfig:
    """Configuration for skull stripping (brain extraction).

    Attributes:
        method: Algorithm to use ("hdbet", "synthstrip", or None to skip)
        fill_value: Value for background voxels (default: 0.0)
        consensus_masking: Use consensus voting across modalities for brain mask
        consensus_threshold: Minimum number of modalities that must agree a voxel is brain

        # HD-BET parameters
        hdbet_mode: Mode for HD-BET ("fast" or "accurate")
        hdbet_device: Device for HD-BET (GPU id as int or "cpu")
        hdbet_do_tta: Enable test-time augmentation for HD-BET

        # SynthStrip parameters
        synthstrip_border: Border parameter for SynthStrip in mm
        synthstrip_device: Device for SynthStrip (GPU id as int or "cpu")
    """

    method: Optional[Literal["hdbet", "synthstrip"]] = "hdbet"
    fill_value: float = 0.0

    # Reference mask modality: if set, only skull-strip this modality and apply its mask to all others
    # (BraTS-standard approach). When None, each modality gets its own mask (legacy behavior).
    reference_mask_modality: Optional[str] = None

    # Consensus masking (ignored when reference_mask_modality is set)
    consensus_masking: bool = True
    consensus_threshold: int = 2

    # HD-BET parameters
    hdbet_mode: Literal["fast", "accurate"] = "accurate"
    hdbet_device: Union[int, str] = 0
    hdbet_do_tta: bool = True

    # SynthStrip parameters
    synthstrip_border: int = 1
    synthstrip_device: Union[int, str] = 0

    def __post_init__(self) -> None:
        """Validate configuration values."""
        # Validate method
        if self.method is not None and self.method not in ["hdbet", "synthstrip"]:
            raise ConfigurationError(
                f"method must be None, 'hdbet', or 'synthstrip', got {self.method}"
            )

        # Skip validation if disabled
        if self.method is None:
            return

        # Validate HD-BET parameters
        if self.method == "hdbet":
            if self.hdbet_mode not in ["fast", "accurate"]:
                raise ConfigurationError(
                    f"hdbet_mode must be 'fast' or 'accurate', got {self.hdbet_mode}"
                )

            if isinstance(self.hdbet_device, int) and self.hdbet_device < 0:
                raise ConfigurationError(
                    f"hdbet_device must be non-negative, got {self.hdbet_device}"
                )
            elif isinstance(self.hdbet_device, str) and self.hdbet_device != "cpu":
                raise ConfigurationError(
                    f"hdbet_device must be int or 'cpu', got {self.hdbet_device}"
                )

        # Validate SynthStrip parameters
        if self.method == "synthstrip":
            if self.synthstrip_border < 0:
                raise ConfigurationError(
                    f"synthstrip_border must be non-negative, got {self.synthstrip_border}"
                )

            if isinstance(self.synthstrip_device, int) and self.synthstrip_device < 0:
                raise ConfigurationError(
                    f"synthstrip_device must be non-negative, got {self.synthstrip_device}"
                )
            elif (
                isinstance(self.synthstrip_device, str)
                and self.synthstrip_device != "cpu"
            ):
                raise ConfigurationError(
                    f"synthstrip_device must be int or 'cpu', got {self.synthstrip_device}"
                )

        # Validate reference_mask_modality
        if self.reference_mask_modality is not None:
            if not isinstance(self.reference_mask_modality, str) or not self.reference_mask_modality.strip():
                raise ConfigurationError(
                    f"reference_mask_modality must be a non-empty string or None, "
                    f"got {self.reference_mask_modality!r}"
                )

        # Validate consensus masking parameters
        if self.consensus_masking and self.consensus_threshold < 1:
            raise ConfigurationError(
                f"consensus_threshold must be >= 1 when consensus_masking is enabled, "
                f"got {self.consensus_threshold}"
            )


@dataclass
class SkullStrippingStepConfig:
    """Configuration for skull stripping step (brain extraction).

    Attributes:
        save_visualization: Whether to save visualization PNGs
        save_mask: Whether to save brain mask NIfTI to artifacts directory
        skull_stripping: Configuration for skull stripping algorithm
    """

    save_visualization: bool = True
    save_mask: bool = True
    skull_stripping: SkullStrippingConfig = field(default_factory=SkullStrippingConfig)

    def __post_init__(self) -> None:
        """Ensure skull_stripping is a SkullStrippingConfig instance."""
        if isinstance(self.skull_stripping, dict):
            self.skull_stripping = SkullStrippingConfig(**self.skull_stripping)


# Backwards compatibility alias
Step4SkullStrippingConfig = SkullStrippingStepConfig


@dataclass
class IntensityNormalizationConfig:
    """Configuration for intensity normalization methods (Step 5).

    Attributes:
        method: Normalization method to apply
                ("zscore", "kde", "percentile_minmax", "whitestripe", "fcm", "lsq", or None to skip)

        # Common parameters
        norm_value: Scaling factor for zscore, kde, and lsq (default=1.0)
        p1: Lower percentile for percentile_minmax (default=1.0)
        p2: Upper percentile for percentile_minmax (default=99.0)

        # WhiteStripe parameters
        width: Quantile range width for white matter detection (default=0.05)
        width_l: Optional lower bound width override
        width_u: Optional upper bound width override

        # FCM parameters
        n_clusters: Number of tissue clusters (default=3)
        tissue_type: Target tissue type "WM"|"GM"|"CSF" (default="WM")
        max_iter: Maximum FCM iterations (default=50)
        error_threshold: Convergence threshold (default=0.005)
        fuzziness: Cluster membership fuzziness parameter (default=2.0)
    """

    method: Optional[
        Literal["zscore", "kde", "percentile_minmax", "whitestripe", "fcm", "lsq"]
    ] = None

    # Common parameters
    norm_value: float = 1.0
    p1: float = 1.0
    p2: float = 99.0

    # Z-score specific parameters
    clip_range: Optional[List[float]] = (
        None  # Optional [low, high] clipping after z-score (e.g., [-5.0, 5.0])
    )

    # WhiteStripe parameters
    width: float = 0.05
    width_l: Optional[float] = None
    width_u: Optional[float] = None

    # FCM parameters
    n_clusters: int = 3
    tissue_type: str = "WM"
    max_iter: int = 50
    error_threshold: float = 0.005
    fuzziness: float = 2.0

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.method is not None:
            valid_methods = [
                "zscore",
                "kde",
                "percentile_minmax",
                "whitestripe",
                "fcm",
                "lsq",
            ]
            if self.method not in valid_methods:
                raise ConfigurationError(
                    f"method must be None or one of {valid_methods}, got {self.method}"
                )

            # Validate percentile parameters (used by percentile_minmax)
            if not 0.0 <= self.p1 < self.p2 <= 100.0:
                raise ConfigurationError(
                    f"Percentiles must satisfy 0 <= p1 < p2 <= 100, got p1={self.p1}, p2={self.p2}"
                )

            # Validate norm_value (used by zscore, kde, lsq)
            if self.norm_value <= 0:
                raise ConfigurationError(
                    f"norm_value must be positive, got {self.norm_value}"
                )

            # Validate clip_range (used by zscore)
            if self.clip_range is not None:
                if (
                    len(self.clip_range) != 2
                    or self.clip_range[0] >= self.clip_range[1]
                ):
                    raise ConfigurationError(
                        f"clip_range must be [low, high] with low < high, got {self.clip_range}"
                    )

            # Validate WhiteStripe parameters
            if self.method == "whitestripe":
                if not 0.0 < self.width <= 1.0:
                    raise ConfigurationError(
                        f"width must be in (0.0, 1.0], got {self.width}"
                    )
                if self.width_l is not None and not 0.0 < self.width_l <= 1.0:
                    raise ConfigurationError(
                        f"width_l must be in (0.0, 1.0], got {self.width_l}"
                    )
                if self.width_u is not None and not 0.0 < self.width_u <= 1.0:
                    raise ConfigurationError(
                        f"width_u must be in (0.0, 1.0], got {self.width_u}"
                    )

            # Validate FCM parameters
            if self.method == "fcm":
                if not 2 <= self.n_clusters <= 10:
                    raise ConfigurationError(
                        f"n_clusters must be in [2, 10], got {self.n_clusters}"
                    )
                if self.tissue_type not in ["WM", "GM", "CSF"]:
                    raise ConfigurationError(
                        f"tissue_type must be 'WM', 'GM', or 'CSF', got {self.tissue_type}"
                    )
                if self.max_iter < 1:
                    raise ConfigurationError(
                        f"max_iter must be >= 1, got {self.max_iter}"
                    )
                if not 0.0 < self.error_threshold < 1.0:
                    raise ConfigurationError(
                        f"error_threshold must be in (0.0, 1.0), got {self.error_threshold}"
                    )
                if not 1.0 < self.fuzziness <= 10.0:
                    raise ConfigurationError(
                        f"fuzziness must be in (1.0, 10.0], got {self.fuzziness}"
                    )


@dataclass
class IntensityNormalizationStepConfig:
    """Configuration for intensity normalization step.

    Attributes:
        save_visualization: Whether to save visualization PNGs
        intensity_normalization: Configuration for normalization method
    """

    save_visualization: bool = True
    intensity_normalization: IntensityNormalizationConfig = field(
        default_factory=IntensityNormalizationConfig
    )

    def __post_init__(self) -> None:
        """Ensure intensity_normalization is an IntensityNormalizationConfig instance."""
        if isinstance(self.intensity_normalization, dict):
            self.intensity_normalization = IntensityNormalizationConfig(
                **self.intensity_normalization
            )


# Backwards compatibility alias
Step5IntensityNormalizationConfig = IntensityNormalizationStepConfig


@dataclass
class LongitudinalRegistrationConfig:
    """Configuration for longitudinal registration across timestamps.

    Attributes:
        method: Registration method ("ants" or None to skip)
        engine: Registration engine ("nipype" or "antspyx")
        reference_modality_priority: Either:
            - Priority string (e.g., "t1n > t1c > t2f > t2w") for single-reference mode
            - "per_modality" for per-modality independent registration
        reference_timestamp_per_study: Path to YAML file mapping patient_id to reference timestamp

        # Registration parameters (same as intra_study_to_reference)
        transform_type: Transform types (e.g., ["Rigid", "Affine"])
        metric: Similarity metric
        metric_bins: Number of bins for MI
        sampling_strategy: Sampling strategy
        sampling_percentage: Percentage of voxels to sample
        number_of_iterations: Iterations per resolution level
        shrink_factors: Downsampling factors per level
        smoothing_sigmas: Smoothing sigmas per level
        convergence_threshold: Convergence threshold
        convergence_window_size: Window size for convergence
        write_composite_transform: Write composite transform file
        interpolation: Interpolation method
        save_detailed_registration_info: Save detailed diagnostics

        # Mask propagation for QC
        propagate_reference_mask: Warp reference brain mask to other timestamps
        compute_mask_comparison: Compute Dice between independent and propagated masks

        # Reference selection (automatic)
        reference_selection_method: Method for automatic reference selection when not in YAML
            - "quality_based": Select based on image quality metrics (recommended)
            - "first": Use first (earliest) timestamp
            - "last": Use last (latest) timestamp
            - "midpoint": Use middle timestamp
        reference_selection_metrics: Metrics to use for quality-based selection
        reference_selection_prefer_earlier: Use earlier timestamp as tiebreaker
        reference_selection_validate_jacobian: Validate registration with Jacobian statistics
        reference_selection_jacobian_threshold: Max allowed |log(det(J))| mean
    """

    method: Optional[Literal["ants"]] = "ants"
    engine: Optional[Literal["nipype", "antspyx"]] = None
    reference_modality_priority: str = "t1n > t1c > t2f > t2w"
    reference_timestamp_per_study: Optional[str] = None

    # Reference selection configuration (automatic, used when patient not in YAML)
    reference_selection_method: str = "quality_based"
    reference_selection_metrics: List[str] = field(
        default_factory=lambda: [
            "snr_foreground",
            "cnr_high_low",
            "boundary_gradient_score",
            "brain_coverage_fraction",
            "laplacian_sharpness",
            "ghosting_score",
        ]
    )
    reference_selection_prefer_earlier: bool = True
    reference_selection_validate_jacobian: bool = True
    reference_selection_jacobian_threshold: float = 0.5

    # Transform parameters
    transform_type: Union[str, List[str]] = field(
        default_factory=lambda: ["Rigid", "Affine"]
    )

    # Metric parameters
    metric: str = "Mattes"
    metric_bins: int = 64
    sampling_strategy: str = "Random"
    sampling_percentage: float = 0.5

    # Multi-resolution schedule
    number_of_iterations: List[List[int]] = field(
        default_factory=lambda: [[1000, 500, 250, 0], [1000, 500, 250, 0]]
    )
    shrink_factors: List[List[int]] = field(
        default_factory=lambda: [[8, 4, 2, 1], [8, 4, 2, 1]]
    )
    smoothing_sigmas: List[List[int]] = field(
        default_factory=lambda: [[3, 2, 1, 0], [3, 2, 1, 0]]
    )

    # Convergence parameters
    convergence_threshold: float = 1.0e-6
    convergence_window_size: int = 10

    # Output parameters
    write_composite_transform: bool = True
    interpolation: str = "BSpline"
    save_detailed_registration_info: bool = False
    save_visualization: bool = True

    # Mask propagation parameters (for QC)
    propagate_reference_mask: bool = False
    compute_mask_comparison: bool = False

    def __post_init__(self) -> None:
        """Validate configuration values."""
        # Validate method
        if self.method is not None and self.method not in ["ants"]:
            raise ConfigurationError(
                f"method must be None or 'ants', got {self.method}"
            )

        # Validate engine (if specified)
        if self.engine is not None and self.engine not in ["nipype", "antspyx"]:
            raise ConfigurationError(
                f"engine must be None, 'nipype', or 'antspyx', got {self.engine}"
            )

        # Skip validation if method is None (registration disabled)
        if self.method is None:
            return

        # Validate reference_modality_priority
        if not self.reference_modality_priority:
            raise ConfigurationError("reference_modality_priority cannot be empty")

        # Validate transform_type (can be string or list of strings)
        valid_transforms = ["Rigid", "Affine", "SyN"]
        if isinstance(self.transform_type, str):
            if self.transform_type not in valid_transforms:
                raise ConfigurationError(
                    f"transform_type must be one of {valid_transforms}, got {self.transform_type}"
                )
        elif isinstance(self.transform_type, list):
            if not self.transform_type:
                raise ConfigurationError("transform_type list cannot be empty")
            for t in self.transform_type:
                if t not in valid_transforms:
                    raise ConfigurationError(
                        f"Invalid transform '{t}' in transform_type. Must be one of {valid_transforms}"
                    )
            # Validate that multi-resolution parameters match the number of transforms
            if len(self.number_of_iterations) != len(self.transform_type):
                raise ConfigurationError(
                    f"number_of_iterations must have one sublist per transform. "
                    f"Got {len(self.number_of_iterations)} sublists for {len(self.transform_type)} transforms"
                )
            if len(self.shrink_factors) != len(self.transform_type):
                raise ConfigurationError(
                    f"shrink_factors must have one sublist per transform. "
                    f"Got {len(self.shrink_factors)} sublists for {len(self.transform_type)} transforms"
                )
            if len(self.smoothing_sigmas) != len(self.transform_type):
                raise ConfigurationError(
                    f"smoothing_sigmas must have one sublist per transform. "
                    f"Got {len(self.smoothing_sigmas)} sublists for {len(self.transform_type)} transforms"
                )
        else:
            raise ConfigurationError(
                f"transform_type must be a string or list of strings, got {type(self.transform_type)}"
            )

        # Validate metric
        valid_metrics = ["Mattes", "MI", "CC", "MeanSquares", "Demons"]
        if self.metric not in valid_metrics:
            raise ConfigurationError(
                f"metric must be one of {valid_metrics}, got {self.metric}"
            )

        # Validate metric_bins
        if not 8 <= self.metric_bins <= 128:
            raise ConfigurationError(
                f"metric_bins must be in [8, 128], got {self.metric_bins}"
            )

        # Validate sampling_strategy
        if self.sampling_strategy not in ["Random", "Regular", "None"]:
            raise ConfigurationError(
                f"sampling_strategy must be 'Random', 'Regular', or 'None', got {self.sampling_strategy}"
            )

        # Validate sampling_percentage
        if not 0.0 < self.sampling_percentage <= 1.0:
            raise ConfigurationError(
                f"sampling_percentage must be in (0.0, 1.0], got {self.sampling_percentage}"
            )

        # Validate convergence_threshold
        if self.convergence_threshold <= 0:
            raise ConfigurationError(
                f"convergence_threshold must be positive, got {self.convergence_threshold}"
            )

        # Validate convergence_window_size
        if self.convergence_window_size < 1:
            raise ConfigurationError(
                f"convergence_window_size must be >= 1, got {self.convergence_window_size}"
            )

        # Validate interpolation
        valid_interpolations = [
            "Linear",
            "BSpline",
            "NearestNeighbor",
            "MultiLabel",
            "Gaussian",
        ]
        if self.interpolation not in valid_interpolations:
            raise ConfigurationError(
                f"interpolation must be one of {valid_interpolations}, got {self.interpolation}"
            )


@dataclass
class LongitudinalRegistrationStepConfig:
    """Configuration for longitudinal registration step.

    Attributes:
        save_visualization: Whether to save visualization outputs
        longitudinal_registration: Configuration for longitudinal registration method
    """

    save_visualization: bool = True
    longitudinal_registration: LongitudinalRegistrationConfig = field(
        default_factory=LongitudinalRegistrationConfig
    )

    def __post_init__(self) -> None:
        """Ensure longitudinal_registration is a LongitudinalRegistrationConfig instance."""
        if isinstance(self.longitudinal_registration, dict):
            self.longitudinal_registration = LongitudinalRegistrationConfig(
                **self.longitudinal_registration
            )


# ============================================================================
# QC Metrics Configuration
# ============================================================================


@dataclass
class QCGeometryMetricsConfig:
    """Configuration for geometry/header consistency metrics.

    Attributes:
        enabled: Whether geometry metrics are enabled
        check_orientation: Verify image orientation consistency
        check_spacing: Validate voxel spacing
        check_affine_det: Check affine determinant sign
    """

    enabled: bool = True
    check_orientation: bool = True
    check_spacing: bool = True
    check_affine_det: bool = True


@dataclass
class QCRegistrationSimilarityConfig:
    """Configuration for registration similarity metrics.

    Attributes:
        enabled: Whether registration similarity metrics are enabled
        nmi_multimodal: Compute NMI for multi-modal registration
        ncc_longitudinal: Compute NCC for longitudinal (same modality) registration
        use_mask: Apply skull-stripped mask when computing similarity
    """

    enabled: bool = True
    nmi_multimodal: bool = True
    ncc_longitudinal: bool = True
    use_mask: bool = True


@dataclass
class QCMaskPlausibilityConfig:
    """Configuration for mask plausibility metrics.

    Attributes:
        enabled: Whether mask plausibility metrics are enabled
        check_volume: Compute brain volume in cubic centimeters
        boundary_gradient_score: Compute gradient score at mask boundary
        longitudinal_dice: Compute Dice coefficient between warped longitudinal masks
    """

    enabled: bool = True
    check_volume: bool = True
    boundary_gradient_score: bool = True
    longitudinal_dice: bool = True


@dataclass
class QCIntensityStabilityConfig:
    """Configuration for intensity stability metrics.

    Attributes:
        enabled: Whether intensity stability metrics are enabled
        median_iqr: Compute masked median and IQR
        wasserstein_distance: Compute Wasserstein distance to reference distribution
        reference_mode: How to compute reference distribution ("site_modality" or "global")
        histogram_bins: Number of bins for histogram computation
        histogram_range_percentiles: Percentile range for clipping intensities (low, high)
    """

    enabled: bool = True
    median_iqr: bool = True
    wasserstein_distance: bool = True
    reference_mode: Literal["site_modality", "global"] = "site_modality"
    histogram_bins: int = 256
    histogram_range_percentiles: Tuple[float, float] = (0.5, 99.5)


@dataclass
class QCSNRCNRConfig:
    """Configuration for SNR/CNR quality metrics.

    SNR (Signal-to-Noise Ratio) and CNR (Contrast-to-Noise Ratio) provide
    quantitative measures of image quality.

    Attributes:
        enabled: Whether SNR/CNR metrics are enabled
        background_percentile: Percentile for background region detection (default: 5)
        foreground_percentile: Percentile for foreground/signal detection (default: 75)
        edge_erosion_iters: Erosion iterations for edge removal in Kaufman method (default: 3)
        intensity_low_pct: Lower percentile for CNR region 1 (default: 25)
        intensity_mid_pct: Mid percentile boundary for CNR (default: 50)
        intensity_high_pct: Upper percentile for CNR region 2 (default: 75)
    """

    enabled: bool = True
    background_percentile: float = 5.0
    foreground_percentile: float = 75.0
    edge_erosion_iters: int = 3
    intensity_low_pct: float = 25.0
    intensity_mid_pct: float = 50.0
    intensity_high_pct: float = 75.0


@dataclass
class QCBaselineConfig:
    """Configuration for baseline metrics capture.

    Baseline metrics are captured before any preprocessing to enable
    pre-vs-post comparison and track quality improvements/degradations.

    Attributes:
        enabled: Whether baseline capture is enabled
        capture_before_first_step: Capture metrics before the first preprocessing step
        metrics_to_capture: List of metric families to capture at baseline
            Options: "geometry", "intensity", "snr_cnr"
    """

    enabled: bool = True
    capture_before_first_step: bool = True
    metrics_to_capture: List[str] = field(
        default_factory=lambda: ["geometry", "intensity", "snr_cnr"]
    )


@dataclass
class QCComparisonConfig:
    """Configuration for pre-vs-post comparison metrics.

    Compares baseline metrics to post-processing metrics to quantify
    improvements or detect potential quality degradation.

    Attributes:
        enabled: Whether pre-post comparison is enabled
        compute_deltas: Compute absolute differences (post - baseline)
        compute_ratios: Compute ratios (post / baseline)
        flag_degradation_threshold_pct: Flag if metric degrades by more than this percentage
    """

    enabled: bool = True
    compute_deltas: bool = True
    compute_ratios: bool = True
    flag_degradation_threshold_pct: float = 10.0


@dataclass
class QCOutlierDetectionConfig:
    """Configuration for outlier detection.

    Attributes:
        enabled: Whether outlier detection is enabled
        method: Outlier detection method ("mad" for Median Absolute Deviation or "iqr")
        mad_threshold: MAD threshold for outlier detection (typically 3.5)
        iqr_multiplier: IQR multiplier for outlier detection (typically 3.0)
    """

    enabled: bool = True
    method: Literal["mad", "iqr"] = "mad"
    mad_threshold: float = 3.5
    iqr_multiplier: float = 3.0


@dataclass
class QCOutputConfig:
    """Configuration for QC output formats.

    Attributes:
        save_long_csv: Save tidy/long format CSV (one metric per row)
        save_wide_csv: Save wide format CSV (one row per image)
        save_summary_csv: Save summary CSV with aggregated stats and outlier flags
        save_metadata_json: Save run metadata JSON with config snapshot
    """

    save_long_csv: bool = True
    save_wide_csv: bool = True
    save_summary_csv: bool = True
    save_metadata_json: bool = True


@dataclass
class QCMetricsConfig:
    """Configuration for all QC metric families.

    Attributes:
        geometry: Geometry/header consistency metrics configuration
        registration_similarity: Registration similarity metrics configuration
        mask_plausibility: Mask plausibility metrics configuration
        intensity_stability: Intensity stability metrics configuration
        snr_cnr: SNR/CNR quality metrics configuration
        baseline: Baseline metrics capture configuration
        comparison: Pre-vs-post comparison configuration
    """

    geometry: QCGeometryMetricsConfig = field(default_factory=QCGeometryMetricsConfig)
    registration_similarity: QCRegistrationSimilarityConfig = field(
        default_factory=QCRegistrationSimilarityConfig
    )
    mask_plausibility: QCMaskPlausibilityConfig = field(
        default_factory=QCMaskPlausibilityConfig
    )
    intensity_stability: QCIntensityStabilityConfig = field(
        default_factory=QCIntensityStabilityConfig
    )
    snr_cnr: QCSNRCNRConfig = field(default_factory=QCSNRCNRConfig)
    baseline: QCBaselineConfig = field(default_factory=QCBaselineConfig)
    comparison: QCComparisonConfig = field(default_factory=QCComparisonConfig)

    def __post_init__(self) -> None:
        """Convert dict to dataclass instances if needed."""
        if isinstance(self.geometry, dict):
            self.geometry = QCGeometryMetricsConfig(**self.geometry)
        if isinstance(self.registration_similarity, dict):
            self.registration_similarity = QCRegistrationSimilarityConfig(
                **self.registration_similarity
            )
        if isinstance(self.mask_plausibility, dict):
            self.mask_plausibility = QCMaskPlausibilityConfig(**self.mask_plausibility)
        if isinstance(self.intensity_stability, dict):
            self.intensity_stability = QCIntensityStabilityConfig(
                **self.intensity_stability
            )
        if isinstance(self.snr_cnr, dict):
            self.snr_cnr = QCSNRCNRConfig(**self.snr_cnr)
        if isinstance(self.baseline, dict):
            self.baseline = QCBaselineConfig(**self.baseline)
        if isinstance(self.comparison, dict):
            self.comparison = QCComparisonConfig(**self.comparison)


@dataclass
class QCConfig:
    """Main configuration for QC metrics collection.

    This configuration enables lightweight, label-free quality control that runs
    after specified preprocessing steps. QC is optional and designed to be cheap
    via downsampling and masking.

    Attributes:
        enabled: Whether QC metrics collection is enabled
        output_dir: Directory for QC outputs (CSVs, JSONs)
        artifacts_dir: Directory for QC-specific artifacts
        overwrite: Allow overwriting existing QC files
        compute_after_steps: List of step names to trigger QC computation
        downsample_to_mm: Downsample to this resolution for cheap computation (e.g., 2.0 mm)
        max_voxels: Maximum voxels to process (for very large images)
        random_seed: Seed for reproducible downsampling
        mask_source: Mask source strategy
        site_metadata: Optional path to YAML mapping patient_id -> site
        outlier_detection: Outlier detection configuration
        metrics: Metrics families configuration
        outputs: Output formats configuration
    """

    enabled: bool = False
    output_dir: str = ""
    artifacts_dir: str = ""
    overwrite: bool = True
    compute_after_steps: List[str] = field(default_factory=list)
    downsample_to_mm: float = 2.0
    max_voxels: int = 250000
    random_seed: int = 1234
    mask_source: Literal[
        "skullstrip_else_otsu", "skullstrip_only", "otsu_only", "none"
    ] = "skullstrip_else_otsu"
    site_metadata: Optional[str] = None

    outlier_detection: QCOutlierDetectionConfig = field(
        default_factory=QCOutlierDetectionConfig
    )
    metrics: QCMetricsConfig = field(default_factory=QCMetricsConfig)
    outputs: QCOutputConfig = field(default_factory=QCOutputConfig)

    def __post_init__(self) -> None:
        """Validate QC configuration."""
        if self.enabled:
            # Validate required directories
            if not self.output_dir:
                raise ConfigurationError(
                    "qc_metrics.output_dir must be specified when enabled=True"
                )
            if not self.artifacts_dir:
                raise ConfigurationError(
                    "qc_metrics.artifacts_dir must be specified when enabled=True"
                )

            # Validate numeric parameters
            if self.downsample_to_mm <= 0:
                raise ConfigurationError(
                    f"downsample_to_mm must be positive, got {self.downsample_to_mm}"
                )
            if self.max_voxels <= 0:
                raise ConfigurationError(
                    f"max_voxels must be positive, got {self.max_voxels}"
                )

        # Convert nested configs from dicts if needed
        if isinstance(self.outlier_detection, dict):
            self.outlier_detection = QCOutlierDetectionConfig(**self.outlier_detection)
        if isinstance(self.metrics, dict):
            self.metrics = QCMetricsConfig(**self.metrics)
        if isinstance(self.outputs, dict):
            self.outputs = QCOutputConfig(**self.outputs)


@dataclass
class DetailedArchiveConfig:
    """Configuration for detailed patient archiving.

    When enabled, saves per-step MRI snapshots and artifacts to HDF5 archives
    for configured showcase patients. Used for publication figures and debugging
    without re-running the pipeline.

    Attributes:
        enabled: Whether archiving is enabled
        patient_ids: List of patient IDs to archive (showcase patients)
        compression: HDF5 compression algorithm
        compression_level: Compression level (1-9)
    """

    enabled: bool = False
    patient_ids: List[str] = field(default_factory=list)
    compression: str = "gzip"
    compression_level: int = 4

    def __post_init__(self) -> None:
        """Validate archive configuration."""
        if self.enabled and not self.patient_ids:
            raise ConfigurationError(
                "detailed_archive.patient_ids must be non-empty when enabled"
            )
        if self.compression_level < 1 or self.compression_level > 9:
            raise ConfigurationError(
                f"compression_level must be 1-9, got {self.compression_level}"
            )


@dataclass
class PipelineExecutionConfig:
    """Configuration for the preprocessing pipeline execution.

    This is the main configuration class that controls pipeline execution,
    including patient selection, I/O paths, and step ordering.

    Attributes:
        enabled: Whether the pipeline is enabled
        patient_selector: Select "single" patient or "all" patients
        patient_id: Patient ID to process (used only if patient_selector == "single")
        mode: Operating mode - "test" (separate output) or "pipeline" (in-place)
        dataset_root: Root directory of the MenGrowth dataset
        output_root: Output directory for test mode
        preprocessing_artifacts_path: Directory for intermediate preprocessing artifacts
        viz_root: Directory for visualization outputs
        overwrite: Allow overwriting existing files
        modalities: List of modalities to process
        steps: Ordered list of step names to execute
        step_configs: Dictionary mapping step patterns to their typed configurations
        qc_metrics: QC metrics configuration
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

    # Dynamic pipeline configuration
    steps: List[str] = field(default_factory=list)
    step_configs: Dict[str, Any] = field(default_factory=dict)
    quality_analysis: Optional[Dict[str, Any]] = None

    # QC metrics configuration
    qc_metrics: QCConfig = field(default_factory=QCConfig)

    # Checkpoint configuration
    checkpoints: Optional[Dict[str, Any]] = None

    # Detailed archive configuration
    detailed_archive: DetailedArchiveConfig = field(
        default_factory=DetailedArchiveConfig
    )

    def __post_init__(self) -> None:
        """Validate configuration and convert step configs to typed dataclasses."""
        # Convert detailed_archive from dict if needed
        if isinstance(self.detailed_archive, dict):
            self.detailed_archive = DetailedArchiveConfig(**self.detailed_archive)

        # Convert qc_metrics to QCConfig instance if needed
        if isinstance(self.qc_metrics, dict):
            self.qc_metrics = QCConfig(**self.qc_metrics)

        # Validate dynamic pipeline configuration if steps are provided
        if self.steps and self.step_configs:
            self._validate_dynamic_config()

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
                raise ConfigurationError("output_root must be specified in test mode")

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

        logger.info(
            f"PipelineExecutionConfig validated: mode={self.mode}, "
            f"patient_selector={self.patient_selector}, "
            f"overwrite={self.overwrite}"
        )

    def _validate_dynamic_config(self) -> None:
        """Validate dynamic pipeline configuration (steps and step_configs)."""
        # Validate steps list is not empty
        if not self.steps:
            raise ConfigurationError("'steps' list cannot be empty")

        # Validate each step name has a known handler
        registry = StepRegistry()
        # Register patterns temporarily for validation
        # IMPORTANT: Order matters! More specific patterns must come before general ones
        # (e.g., "longitudinal_registration" before "registration")
        registry.register("data_harmonization", lambda: None)
        registry.register("bias_field_correction", lambda: None)
        registry.register("intensity_normalization", lambda: None)
        registry.register("resampling", lambda: None)
        registry.register("cubic_padding", lambda: None)
        registry.register("longitudinal_registration", lambda: None)
        registry.register("registration", lambda: None)
        registry.register("skull_stripping", lambda: None)

        for step_name in self.steps:
            try:
                pattern, _ = registry.get_handler(step_name)
            except ValueError as e:
                raise ConfigurationError(f"Invalid step name '{step_name}': {e}") from e

        # Validate each step has a matching config
        for step_name in self.steps:
            matched = False
            for pattern in self.step_configs.keys():
                if pattern in step_name:
                    matched = True
                    break
            if not matched:
                raise ConfigurationError(
                    f"Step '{step_name}' has no matching configuration in step_configs. "
                    f"Available patterns: {list(self.step_configs.keys())}"
                )

        # Convert step_configs dict entries to proper dataclass instances
        self._convert_step_configs()

    def _convert_step_configs(self) -> None:
        """Convert step_configs dict entries to typed dataclass instances.

        Maps each step configuration to its corresponding typed dataclass,
        providing full type validation and IDE support.
        """
        # Registry mapping step patterns to their typed config classes
        # IMPORTANT: Order matters! More specific patterns must come before general ones
        # (e.g., "longitudinal_registration" before "registration")
        step_config_classes: Dict[str, type] = {
            "data_harmonization": DataHarmonizationStepConfig,
            "bias_field_correction": BiasFieldCorrectionStepConfig,
            "resampling": ResamplingStepConfig,
            "cubic_padding": CubicPaddingStepConfig,
            "longitudinal_registration": LongitudinalRegistrationStepConfig,
            "registration": RegistrationStepConfig,
            "skull_stripping": SkullStrippingStepConfig,
            "intensity_normalization": IntensityNormalizationStepConfig,
        }

        for step_name, config_data in list(self.step_configs.items()):
            if isinstance(config_data, dict):
                # Find matching config class using substring matching
                matched = False
                for pattern, config_class in step_config_classes.items():
                    if pattern in step_name:
                        try:
                            self.step_configs[step_name] = config_class(**config_data)
                            matched = True
                            logger.debug(
                                f"Converted '{step_name}' config to {config_class.__name__}"
                            )
                            break
                        except TypeError as e:
                            raise ConfigurationError(
                                f"Invalid configuration for step '{step_name}': {e}"
                            ) from e

                if not matched:
                    raise ConfigurationError(
                        f"No config class found for step '{step_name}'. "
                        f"Available patterns: {list(step_config_classes.keys())}"
                    )


# Backwards compatibility alias
DataHarmonizationConfig = PipelineExecutionConfig


@dataclass
class PreprocessingPipelineConfig:
    """Top-level configuration for the preprocessing pipeline.

    Attributes:
        steps: Ordered list of pipeline steps to execute
        general_configuration: Global configuration settings
        step_configs: Dictionary of step-specific configurations
        skull_stripping: Optional skull stripping configuration (top-level)
        intensity_normalization: Optional intensity normalization configuration (top-level)
        longitudinal_registration: Optional longitudinal registration configuration (top-level)
        quality_analysis: Optional quality analysis configuration
        qc_metrics: Optional per-step QC metrics configuration
        checkpoints: Optional checkpoint system configuration
    """

    steps: List[str] = field(default_factory=list)
    general_configuration: PipelineExecutionConfig = field(
        default_factory=PipelineExecutionConfig
    )
    step_configs: Dict[str, Any] = field(default_factory=dict)
    skull_stripping: Optional[Dict[str, Any]] = None
    intensity_normalization: Optional[Dict[str, Any]] = None
    longitudinal_registration: Optional[Dict[str, Any]] = None
    quality_analysis: Optional[Dict[str, Any]] = None
    qc_metrics: Optional[Dict[str, Any]] = None
    checkpoints: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Validate and convert configuration."""
        # Ensure general_configuration is a PipelineExecutionConfig instance
        if isinstance(self.general_configuration, dict):
            self.general_configuration = PipelineExecutionConfig(
                **self.general_configuration
            )

        # Add top-level configs to step_configs if they exist
        if self.skull_stripping:
            self.step_configs["skull_stripping"] = self.skull_stripping
        if self.intensity_normalization:
            self.step_configs["intensity_normalization"] = self.intensity_normalization
        if self.longitudinal_registration:
            self.step_configs["longitudinal_registration"] = (
                self.longitudinal_registration
            )

        # Add steps and step_configs to general_configuration for orchestrator access
        self.general_configuration.steps = self.steps
        self.general_configuration.step_configs = self.step_configs
        self.general_configuration.quality_analysis = self.quality_analysis

        # Convert qc_metrics dict to QCConfig if provided at this level
        if self.qc_metrics:
            if isinstance(self.qc_metrics, dict):
                self.general_configuration.qc_metrics = QCConfig(**self.qc_metrics)
            else:
                self.general_configuration.qc_metrics = self.qc_metrics

        # Pass checkpoints config (kept as dict for flexibility)
        if self.checkpoints:
            self.general_configuration.checkpoints = self.checkpoints

        # Now validate the dynamic config
        if self.steps and self.step_configs:
            self.general_configuration._validate_dynamic_config()


def load_preprocessing_pipeline_config(config_path: Path) -> PipelineExecutionConfig:
    """Load and validate preprocessing pipeline configuration from YAML.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Validated PipelineExecutionConfig object with steps and step_configs populated

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
        pipeline_config = PreprocessingPipelineConfig(**preprocessing_data)
    except TypeError as e:
        raise ConfigurationError(f"Invalid configuration structure: {e}") from e

    logger.info("Preprocessing pipeline configuration loaded successfully")

    # Return the general_configuration which now has steps and step_configs populated
    return pipeline_config.general_configuration
