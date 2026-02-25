"""Preprocessing pipeline orchestrator for MenGrowth dataset.

This module coordinates the execution of preprocessing steps on patient data,
managing file paths, mode semantics, and visualization outputs.
"""

import gc
from pathlib import Path
from typing import List, Optional, Any, Tuple, Dict, Union
import logging

from mengrowth.preprocessing.src.config import (
    PipelineExecutionConfig,
    StepRegistry,
    StepExecutionContext,
    STEP_METADATA,
    # Step config types for type hints
    DataHarmonizationStepConfig,
    BiasFieldCorrectionStepConfig,
    ResamplingStepConfig,
    CubicPaddingStepConfig,
    RegistrationStepConfig,
    SkullStrippingStepConfig,
    IntensityNormalizationStepConfig,
    LongitudinalRegistrationStepConfig,
)
from mengrowth.preprocessing.src.data_harmonization.io import NRRDtoNIfTIConverter
from mengrowth.preprocessing.src.data_harmonization.orient import Reorienter
from mengrowth.preprocessing.src.data_harmonization.head_masking.conservative import (
    ConservativeBackgroundRemover,
)
from mengrowth.preprocessing.src.data_harmonization.head_masking.self import (
    SELFBackgroundRemover,
)
from mengrowth.preprocessing.src.data_harmonization.head_masking.otsu_foreground import (
    OtsuForegroundRemover,
)
from mengrowth.preprocessing.src.bias_field_correction.n4_sitk import (
    N4BiasFieldCorrector,
)
from mengrowth.preprocessing.src.normalization.zscore import ZScoreNormalizer
from mengrowth.preprocessing.src.normalization.kde import KDENormalizer
from mengrowth.preprocessing.src.normalization.percentile_minmax import (
    PercentileMinMaxNormalizer,
)
from mengrowth.preprocessing.src.normalization.whitestripe import WhiteStripeNormalizer
from mengrowth.preprocessing.src.normalization.fcm import FCMNormalizer
from mengrowth.preprocessing.src.normalization.lsq import LSQNormalizer
from mengrowth.preprocessing.src.resampling.bspline import BSplineResampler
from mengrowth.preprocessing.src.resampling.eclare import EclareResampler
from mengrowth.preprocessing.src.resampling.composite import CompositeResampler
from mengrowth.preprocessing.src.skull_stripping.hdbet import HDBetSkullStripper
from mengrowth.preprocessing.src.skull_stripping.synthstrip import (
    SynthStripSkullStripper,
)

logger = logging.getLogger(__name__)


class PreprocessingOrchestrator:
    """Orchestrates preprocessing operations for a patient.

    This class manages the preprocessing pipeline execution, including:
    - File path resolution based on mode (test vs pipeline)
    - Step sequencing (NRRD to NIfTI -> Reorient -> Background removal)
    - Visualization generation
    - Overwrite protection
    """

    def __init__(self, config: PipelineExecutionConfig, verbose: bool = False) -> None:
        """Initialize preprocessing orchestrator with lazy component loading.

        Args:
            config: Pipeline execution configuration
            verbose: Enable verbose logging
        """
        self.config = config
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)

        # Initialize step registry
        self.step_registry = StepRegistry()
        self._register_step_handlers()

        # Component cache for lazy initialization
        self._components = {}

        # State variables
        self.selected_reference_modality = None

        # Initialize QC Manager if enabled
        self.qc_manager = None
        if config.qc_metrics.enabled:
            site_map = self._load_site_map(config.qc_metrics.site_metadata)
            from mengrowth.preprocessing.quality_analysis.qc_manager import QCManager

            self.qc_manager = QCManager(config.qc_metrics, site_map=site_map)
            self.logger.info("QC Manager initialized")

        # Initialize Checkpoint Manager if enabled
        self.checkpoint_manager = None
        checkpoint_config = getattr(config, "checkpoints", None)
        if checkpoint_config and checkpoint_config.get("enabled", False):
            from mengrowth.preprocessing.src.checkpoint import CheckpointManager

            checkpoint_dir = Path(checkpoint_config.get("checkpoint_dir", ""))
            if checkpoint_dir:
                self.checkpoint_manager = CheckpointManager(
                    checkpoint_dir=checkpoint_dir, enabled=True
                )
                self.logger.info(f"Checkpoint Manager initialized: {checkpoint_dir}")

        self.logger.info(
            "Preprocessing orchestrator initialized with dynamic pipeline execution"
        )

    def _register_step_handlers(self) -> None:
        """Register all step handler functions with the registry.

        IMPORTANT: Order matters! More specific patterns must be registered before general ones
        to ensure correct substring matching (e.g., "longitudinal_registration" before "registration").
        """
        from mengrowth.preprocessing.src.steps import (
            data_harmonization,
            bias_field_correction,
            intensity_normalization,
            resampling,
            cubic_padding,
            registration,
            skull_stripping,
            longitudinal_registration,
        )

        self.step_registry.register("data_harmonization", data_harmonization.execute)
        self.step_registry.register(
            "bias_field_correction", bias_field_correction.execute
        )
        self.step_registry.register(
            "intensity_normalization", intensity_normalization.execute
        )
        self.step_registry.register("resampling", resampling.execute)
        self.step_registry.register("cubic_padding", cubic_padding.execute)
        self.step_registry.register(
            "longitudinal_registration", longitudinal_registration.execute
        )
        self.step_registry.register("registration", registration.execute)
        self.step_registry.register("skull_stripping", skull_stripping.execute)

        self.logger.debug(
            f"Registered {len(self.step_registry.list_patterns())} step patterns"
        )

    def _load_site_map(self, yaml_path: Optional[str]) -> Dict[str, str]:
        """Load optional site metadata YAML mapping patient_id -> site.

        Args:
            yaml_path: Path to YAML file with site metadata

        Returns:
            Dictionary mapping patient_id to site, empty if file not found
        """
        if not yaml_path or not Path(yaml_path).exists():
            return {}

        try:
            import yaml

            with open(yaml_path, "r") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            self.logger.warning(f"Failed to load site metadata from {yaml_path}: {e}")
            return {}

    def _trigger_qc_if_needed(
        self, step_name: str, context: "StepExecutionContext", result: Dict[str, Any]
    ) -> None:
        """Trigger QC metric collection if this step is configured for QC.

        Args:
            step_name: Name of completed step
            context: Step execution context
            result: Step execution result dict
        """
        if not self.qc_manager:
            return

        # Build base case_metadata from context
        case_metadata = {
            "patient_id": context.patient_id,
        }

        # Add study_id if available
        if context.study_dir:
            case_metadata["study_id"] = context.study_dir.name

        # Add modality if available
        if context.modality:
            case_metadata["modality"] = context.modality

        # Check for is_longitudinal flag in result
        if result and "reference_timestamp" in result:
            case_metadata["is_longitudinal"] = True
            case_metadata["reference_timestamp"] = result.get("reference_timestamp")

        # Handle study-level steps specially (they have qc_paths in result)
        if result and "qc_paths" in result:
            qc_paths = result["qc_paths"]

            # Determine step base for routing
            step_base = self._extract_step_base(step_name)

            # For registration: iterate over modalities and create QC records
            if step_base == "registration":
                self._trigger_registration_qc(
                    step_name, case_metadata, qc_paths, result
                )
                return

            # For skull_stripping: iterate over modalities
            if step_base == "skull_stripping":
                self._trigger_skull_stripping_qc(step_name, case_metadata, qc_paths)
                return

        # Original logic for modality-level steps
        image_paths = {}
        if context.paths:
            # Try to find output path
            output_file = context.paths.get("nifti") or context.paths.get("resampled")
            if output_file and output_file.exists():
                image_paths["output"] = output_file

        # Build artifact_paths
        artifact_paths = {}
        if context.paths:
            # Check for mask artifacts (from skull stripping)
            mask_artifact = context.paths.get("mask_artifact")
            if mask_artifact and mask_artifact.exists():
                artifact_paths["mask"] = mask_artifact

        # Call QCManager
        try:
            self.qc_manager.on_step_completed(
                step_name=step_name,
                case_metadata=case_metadata,
                image_paths=image_paths,
                artifact_paths=artifact_paths,
            )
        except Exception as e:
            self.logger.warning(f"QC trigger failed for {step_name}: {e}")

    def _extract_step_base(self, step_name: str) -> str:
        """Extract base step type from full step name.

        Args:
            step_name: Full step name (e.g., "registration_static", "skull_stripping_1")

        Returns:
            Base step type (e.g., "registration", "skull_stripping")
        """
        # Check against known patterns in order of specificity
        for pattern in [
            "longitudinal_registration",
            "registration",
            "skull_stripping",
            "intensity_normalization",
            "bias_field_correction",
            "data_harmonization",
            "resampling",
            "cubic_padding",
        ]:
            if pattern in step_name:
                return pattern
        return step_name

    def _trigger_registration_qc(
        self,
        step_name: str,
        base_metadata: Dict[str, Any],
        qc_paths: Dict[str, Any],
        result: Dict[str, Any],
    ) -> None:
        """Trigger QC for registration step with proper paths.

        Args:
            step_name: Step name
            base_metadata: Base case metadata (patient_id, study_id)
            qc_paths: Paths dict from registration step
            result: Full result dict from registration step
        """
        atlas_path_str = qc_paths.get("atlas_path")
        atlas_path = Path(atlas_path_str) if atlas_path_str else None

        for modality, output_path_str in qc_paths.get("modality_outputs", {}).items():
            if not output_path_str:
                continue
            output_path = Path(output_path_str)
            if not output_path.exists():
                continue

            case_metadata = {**base_metadata, "modality": modality}

            image_paths = {"output": output_path}
            if atlas_path and atlas_path.exists():
                image_paths["reference"] = atlas_path

            artifact_paths = {}

            try:
                self.qc_manager.on_step_completed(
                    step_name=step_name,
                    case_metadata=case_metadata,
                    image_paths=image_paths,
                    artifact_paths=artifact_paths,
                )
            except Exception as e:
                self.logger.warning(
                    f"QC trigger failed for {step_name}/{modality}: {e}"
                )

    def _trigger_skull_stripping_qc(
        self, step_name: str, base_metadata: Dict[str, Any], qc_paths: Dict[str, Any]
    ) -> None:
        """Trigger QC for skull stripping step with proper paths.

        Args:
            step_name: Step name
            base_metadata: Base case metadata (patient_id, study_id)
            qc_paths: Paths dict from skull stripping step
        """
        for modality in qc_paths.get("image_outputs", {}).keys():
            output_path_str = qc_paths["image_outputs"].get(modality, "")
            if not output_path_str:
                continue
            output_path = Path(output_path_str)

            if not output_path.exists():
                continue

            case_metadata = {**base_metadata, "modality": modality}

            image_paths = {"output": output_path}
            artifact_paths = {}

            # Add mask path if available
            mask_path_str = qc_paths.get("mask_outputs", {}).get(modality, "")
            if mask_path_str:
                mask_path = Path(mask_path_str)
                if mask_path.exists():
                    artifact_paths["mask"] = mask_path

            try:
                self.qc_manager.on_step_completed(
                    step_name=step_name,
                    case_metadata=case_metadata,
                    image_paths=image_paths,
                    artifact_paths=artifact_paths,
                )
            except Exception as e:
                self.logger.warning(
                    f"QC trigger failed for {step_name}/{modality}: {e}"
                )

    # Prefixes of cached components that hold GPU memory or large CPU allocations
    _GPU_COMPONENT_PREFIXES = (
        "skull_stripper_",
        "resampler_eclare",
        "resampler_composite",
    )

    def _release_gpu_components(self) -> None:
        """Release cached components that hold GPU memory or large CPU allocations.

        Called between studies to prevent RAM/VRAM accumulation when processing
        patients with many longitudinal studies.
        """
        keys_to_delete = [
            k for k in self._components if k.startswith(self._GPU_COMPONENT_PREFIXES)
        ]
        if not keys_to_delete:
            return

        for key in keys_to_delete:
            del self._components[key]

        gc.collect()

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        self.logger.debug(f"Released GPU components: {keys_to_delete}")

    def _get_component(self, component_type: str, config: Any) -> Any:
        """Get or create a component instance (lazy initialization).

        Args:
            component_type: Type of component (e.g., "normalizer_kde", "resampler_bspline")
            config: Configuration for the component

        Returns:
            Component instance (cached for reuse)
        """
        if component_type not in self._components:
            self._components[component_type] = self._create_component(
                component_type, config
            )
        return self._components[component_type]

    def _create_component(self, component_type: str, config: Any) -> Any:
        """Create a component instance based on type and config.

        Args:
            component_type: Type identifier (e.g., "normalizer_kde", "resampler_bspline")
            config: Configuration dict or dataclass

        Returns:
            Initialized component instance
        """
        # Parse component type
        if component_type == "converter":
            return NRRDtoNIfTIConverter(verbose=self.verbose)
        elif component_type == "reorienter":
            return Reorienter(
                target_orientation=config.reorient_to, verbose=self.verbose
            )
        elif component_type.startswith("background_remover_"):
            method = component_type.split("_", 2)[
                2
            ]  # Extract method from "background_remover_METHOD"
            return self._create_background_remover(method, config)
        elif component_type.startswith("bias_corrector_"):
            parts = component_type.split("_", 3)
            method = parts[2] if len(parts) > 2 else "unknown"
            return self._create_bias_corrector(method, config)
        elif component_type.startswith("normalizer_"):
            # Extract method from "normalizer_METHOD_stepname"
            parts = component_type.split("_", 2)
            method = parts[1] if len(parts) > 1 else "unknown"
            return self._create_normalizer(method, config)
        elif component_type.startswith("resampler_"):
            parts = component_type.split("_", 2)
            method = parts[1] if len(parts) > 1 else "unknown"
            return self._create_resampler(method, config)
        elif component_type.startswith("intra_study_to_ref_"):
            return self._create_intra_study_registrator(config)
        elif component_type.startswith("intra_study_to_atlas_"):
            return self._create_atlas_registrator(config)
        elif component_type.startswith("skull_stripper_"):
            parts = component_type.split("_", 3)
            method = parts[2] if len(parts) > 2 else "unknown"
            return self._create_skull_stripper(method, config)
        elif component_type.startswith("longitudinal_reg_"):
            return self._create_longitudinal_registrator(config)
        else:
            raise ValueError(f"Unknown component type: {component_type}")

    def _create_background_remover(self, method: str, config: Any) -> Any:
        """Create background remover instance."""
        if method == "border_connected_percentile":
            return ConservativeBackgroundRemover(
                config=config.background_zeroing, verbose=self.verbose
            )
        elif method == "self_head_mask":
            return SELFBackgroundRemover(
                config=config.background_zeroing, verbose=self.verbose
            )
        elif method == "otsu_foreground":
            return OtsuForegroundRemover(
                config=config.background_zeroing, verbose=self.verbose
            )
        else:
            raise ValueError(f"Unknown background removal method: {method}")

    def _create_bias_corrector(self, method: str, config: Any) -> Any:
        """Create bias field corrector instance."""
        if method == "n4":
            bf_config_dict = {
                "shrink_factor": config.bias_field_correction.shrink_factor,
                "max_iterations": config.bias_field_correction.max_iterations,
                "bias_field_fwhm": config.bias_field_correction.bias_field_fwhm,
                "convergence_threshold": config.bias_field_correction.convergence_threshold,
            }
            return N4BiasFieldCorrector(config=bf_config_dict, verbose=self.verbose)
        else:
            raise ValueError(f"Unknown bias field correction method: {method}")

    def _create_normalizer(self, method: str, config: Any) -> Any:
        """Create normalizer instance based on method."""
        if method == "zscore":
            norm_config_dict = {
                "norm_value": config.intensity_normalization.norm_value,
                "clip_range": config.intensity_normalization.clip_range,
            }
            return ZScoreNormalizer(config=norm_config_dict, verbose=self.verbose)
        elif method == "kde":
            norm_config_dict = {
                "norm_value": config.intensity_normalization.norm_value,
            }
            return KDENormalizer(config=norm_config_dict, verbose=self.verbose)
        elif method == "percentile_minmax":
            norm_config_dict = {
                "p1": config.intensity_normalization.p1,
                "p2": config.intensity_normalization.p2,
            }
            return PercentileMinMaxNormalizer(
                config=norm_config_dict, verbose=self.verbose
            )
        elif method == "whitestripe":
            norm_config_dict = {
                "width": config.intensity_normalization.width,
                "width_l": config.intensity_normalization.width_l,
                "width_u": config.intensity_normalization.width_u,
            }
            return WhiteStripeNormalizer(config=norm_config_dict, verbose=self.verbose)
        elif method == "fcm":
            norm_config_dict = {
                "n_clusters": config.intensity_normalization.n_clusters,
                "tissue_type": config.intensity_normalization.tissue_type,
                "max_iter": config.intensity_normalization.max_iter,
                "error_threshold": config.intensity_normalization.error_threshold,
                "fuzziness": config.intensity_normalization.fuzziness,
            }
            return FCMNormalizer(config=norm_config_dict, verbose=self.verbose)
        elif method == "lsq":
            norm_config_dict = {
                "norm_value": config.intensity_normalization.norm_value,
            }
            return LSQNormalizer(config=norm_config_dict, verbose=self.verbose)
        else:
            raise ValueError(f"Unknown normalization method: {method}")

    def _create_resampler(self, method: str, config: Any) -> Any:
        """Create resampler instance based on method."""
        if method == "bspline":
            resample_config_dict = {
                "bspline_order": config.resampling.bspline_order,
            }
            return BSplineResampler(
                target_voxel_size=config.resampling.target_voxel_size,
                config=resample_config_dict,
                verbose=self.verbose,
            )
        elif method == "eclare":
            resample_config_dict = {
                "conda_environment_eclare": config.resampling.conda_environment_eclare,
                "batch_size": config.resampling.batch_size,
                "n_patches": config.resampling.n_patches,
                "patch_sampling": config.resampling.patch_sampling,
                "suffix": config.resampling.suffix,
                "gpu_id": config.resampling.gpu_id,
                "verbose": config.resampling.verbose
                if hasattr(config.resampling, "verbose")
                else self.verbose,
            }
            return EclareResampler(
                target_voxel_size=config.resampling.target_voxel_size,
                config=resample_config_dict,
                verbose=self.verbose,
            )
        elif method == "composite":
            resample_config_dict = {
                "composite_interpolator": config.resampling.composite_interpolator,
                "composite_dl_method": config.resampling.composite_dl_method,
                "max_mm_interpolator": config.resampling.max_mm_interpolator,
                "max_mm_dl_method": config.resampling.max_mm_dl_method,
                "resample_mm_to_interpolator_if_max_mm_dl_method": config.resampling.resample_mm_to_interpolator_if_max_mm_dl_method,
                "bspline_order": config.resampling.bspline_order,
                "conda_environment_eclare": config.resampling.conda_environment_eclare,
                "batch_size": config.resampling.batch_size,
                "n_patches": config.resampling.n_patches,
                "patch_sampling": config.resampling.patch_sampling,
                "suffix": config.resampling.suffix,
                "gpu_id": config.resampling.gpu_id,
            }
            return CompositeResampler(
                target_voxel_size=config.resampling.target_voxel_size,
                config=resample_config_dict,
                verbose=self.verbose,
            )
        else:
            raise ValueError(f"Unknown resampling method: {method}")

    def _create_intra_study_registrator(self, config: Any) -> Any:
        """Create intra-study to reference registrator."""
        from mengrowth.preprocessing.src.registration.factory import (
            create_multi_modal_coregistration,
        )
        from mengrowth.preprocessing.src.registration.constants import (
            DEFAULT_REGISTRATION_ENGINE,
        )

        intra_study_config = {
            "reference_modality_priority": config.intra_study_to_reference.reference_modality_priority,
            "transform_type": config.intra_study_to_reference.transform_type,
            "metric": config.intra_study_to_reference.metric,
            "metric_bins": config.intra_study_to_reference.metric_bins,
            "sampling_strategy": config.intra_study_to_reference.sampling_strategy,
            "sampling_percentage": config.intra_study_to_reference.sampling_percentage,
            "number_of_iterations": config.intra_study_to_reference.number_of_iterations,
            "shrink_factors": config.intra_study_to_reference.shrink_factors,
            "smoothing_sigmas": config.intra_study_to_reference.smoothing_sigmas,
            "convergence_threshold": config.intra_study_to_reference.convergence_threshold,
            "convergence_window_size": config.intra_study_to_reference.convergence_window_size,
            "write_composite_transform": config.intra_study_to_reference.write_composite_transform,
            "interpolation": config.intra_study_to_reference.interpolation,
            "engine": config.intra_study_to_reference.engine
            or DEFAULT_REGISTRATION_ENGINE,
            "save_detailed_registration_info": config.intra_study_to_reference.save_detailed_registration_info,
            "use_center_of_mass_init": config.intra_study_to_reference.use_center_of_mass_init,
            "validate_registration_quality": config.intra_study_to_reference.validate_registration_quality,
            "quality_warning_threshold": config.intra_study_to_reference.quality_warning_threshold,
        }
        return create_multi_modal_coregistration(
            config=intra_study_config, verbose=self.verbose
        )

    def _create_atlas_registrator(self, config: Any) -> Any:
        """Create intra-study to atlas registrator."""
        from mengrowth.preprocessing.src.registration.factory import (
            create_intra_study_to_atlas,
        )
        from mengrowth.preprocessing.src.registration.constants import (
            DEFAULT_REGISTRATION_ENGINE,
        )

        atlas_config = {
            "atlas_path": config.intra_study_to_atlas.atlas_path,
            "transforms": config.intra_study_to_atlas.transforms,
            "create_composite_transforms": config.intra_study_to_atlas.create_composite_transforms,
            "metric": config.intra_study_to_atlas.metric,
            "metric_bins": config.intra_study_to_atlas.metric_bins,
            "sampling_strategy": config.intra_study_to_atlas.sampling_strategy,
            "sampling_percentage": config.intra_study_to_atlas.sampling_percentage,
            "number_of_iterations": config.intra_study_to_atlas.number_of_iterations,
            "shrink_factors": config.intra_study_to_atlas.shrink_factors,
            "smoothing_sigmas": config.intra_study_to_atlas.smoothing_sigmas,
            "convergence_threshold": config.intra_study_to_atlas.convergence_threshold,
            "convergence_window_size": config.intra_study_to_atlas.convergence_window_size,
            "interpolation": config.intra_study_to_atlas.interpolation,
            "engine": config.intra_study_to_atlas.engine or DEFAULT_REGISTRATION_ENGINE,
            "save_detailed_registration_info": config.intra_study_to_atlas.save_detailed_registration_info,
            "use_center_of_mass_init": config.intra_study_to_atlas.use_center_of_mass_init,
            "validate_registration_quality": config.intra_study_to_atlas.validate_registration_quality,
            "quality_warning_threshold": config.intra_study_to_atlas.quality_warning_threshold,
        }
        return create_intra_study_to_atlas(
            config=atlas_config,
            reference_modality=self.selected_reference_modality,
            verbose=self.verbose,
        )

    def _create_skull_stripper(self, method: str, config: Any) -> Any:
        """Create skull stripper instance."""
        if method == "hdbet":
            skull_strip_config = {
                "mode": config.skull_stripping.hdbet_mode,
                "device": config.skull_stripping.hdbet_device,
                "do_tta": config.skull_stripping.hdbet_do_tta,
                "fill_value": config.skull_stripping.fill_value,
            }
            return HDBetSkullStripper(config=skull_strip_config, verbose=self.verbose)
        elif method == "synthstrip":
            skull_strip_config = {
                "border": config.skull_stripping.synthstrip_border,
                "device": config.skull_stripping.synthstrip_device,
                "fill_value": config.skull_stripping.fill_value,
            }
            return SynthStripSkullStripper(
                config=skull_strip_config, verbose=self.verbose
            )
        else:
            raise ValueError(f"Unknown skull stripping method: {method}")

    def _create_longitudinal_registrator(self, config: Any) -> Any:
        """Create longitudinal registrator instance.

        Args:
            config: LongitudinalRegistrationConfig object (not full StepExecutionConfig)
        """
        from mengrowth.preprocessing.src.registration.longitudinal_registration import (
            LongitudinalRegistration,
        )
        from mengrowth.preprocessing.src.registration.constants import (
            DEFAULT_REGISTRATION_ENGINE,
        )

        long_config = {
            "engine": config.engine or DEFAULT_REGISTRATION_ENGINE,
            "transform_type": config.transform_type,
            "metric": config.metric,
            "metric_bins": config.metric_bins,
            "sampling_strategy": config.sampling_strategy,
            "sampling_percentage": config.sampling_percentage,
            "number_of_iterations": config.number_of_iterations,
            "shrink_factors": config.shrink_factors,
            "smoothing_sigmas": config.smoothing_sigmas,
            "convergence_threshold": config.convergence_threshold,
            "convergence_window_size": config.convergence_window_size,
            "write_composite_transform": config.write_composite_transform,
            "interpolation": config.interpolation,
        }

        return LongitudinalRegistration(config=long_config, verbose=self.verbose)

    def _get_study_directories(self, patient_id: str) -> List[Path]:
        """Get list of study directories for a patient.

        Args:
            patient_id: Patient ID (e.g., "MenGrowth-0015")

        Returns:
            List of study directory paths

        Raises:
            FileNotFoundError: If patient directory does not exist
        """
        patient_dir = Path(self.config.dataset_root) / patient_id

        if not patient_dir.exists():
            raise FileNotFoundError(f"Patient directory not found: {patient_dir}")

        # Find all study directories (e.g., MenGrowth-0015-000, MenGrowth-0015-001)
        study_dirs = sorted([d for d in patient_dir.iterdir() if d.is_dir()])

        self.logger.info(f"Found {len(study_dirs)} study directories for {patient_id}")

        return study_dirs

    def _get_modality_file(self, study_dir: Path, modality: str) -> Optional[Path]:
        """Find modality file in study directory.

        Args:
            study_dir: Study directory path
            modality: Modality name (e.g., "t1c", "t2w")

        Returns:
            Path to modality file, or None if not found
        """
        # Look for NRRD file
        nrrd_file = study_dir / f"{modality}.nrrd"

        if nrrd_file.exists():
            return nrrd_file

        self.logger.debug(f"Modality {modality} not found in {study_dir.name}")
        return None

    def _get_output_paths(
        self, patient_id: str, study_dir: Path, modality: str
    ) -> dict:
        """Get output paths for all processing steps.

        Args:
            patient_id: Patient ID
            study_dir: Study directory path
            modality: Modality name

        Returns:
            Dictionary with output paths for each step
        """
        study_name = study_dir.name

        if self.config.mode == "test":
            # Test mode: write to separate output directory
            output_base = Path(self.config.output_root) / patient_id / study_name
            viz_base = Path(self.config.viz_root) / patient_id / study_name
            artifacts_base = (
                Path(self.config.preprocessing_artifacts_path) / patient_id / study_name
            )
        else:
            # Pipeline mode: write in-place
            output_base = study_dir
            viz_base = Path(self.config.viz_root) / patient_id / study_name
            artifacts_base = (
                Path(self.config.preprocessing_artifacts_path) / patient_id / study_name
            )

        # Ensure directories exist
        output_base.mkdir(parents=True, exist_ok=True)
        viz_base.mkdir(parents=True, exist_ok=True)
        artifacts_base.mkdir(parents=True, exist_ok=True)

        return {
            "nifti": output_base / f"{modality}.nii.gz",
            "bias_field": artifacts_base / f"{modality}_bias_field.nii.gz",
        }

    def run_patient(self, patient_id: str) -> None:
        """Run preprocessing pipeline for a single patient with dynamic step execution.

        Args:
            patient_id: Patient ID to process

        Raises:
            FileNotFoundError: If patient directory not found
            RuntimeError: If processing fails
        """
        self.logger.info(f"{'=' * 80}")
        self.logger.info(f"Processing patient: {patient_id}")
        self.logger.info(f"{'=' * 80}")

        # Get study directories
        study_dirs = self._get_study_directories(patient_id)

        # Log and save pipeline configuration
        self._log_pipeline_order(patient_id)

        # Separate patient-level from other steps
        modality_study_groups, patient_level_steps = self._categorize_step_groups()

        # PHASE 1: Process all studies (modality-level and study-level steps)
        total_processed, total_skipped, total_errors = self._process_studies(
            patient_id, study_dirs, modality_study_groups
        )

        # PHASE 2: Execute patient-level steps (new)
        if patient_level_steps:
            self.logger.info(f"\n{'=' * 80}")
            self.logger.info(f"Executing patient-level steps for {patient_id}")
            self.logger.info(f"{'=' * 80}")
            self._execute_patient_level_steps(
                patient_id, study_dirs, patient_level_steps
            )

        # PHASE 3: Finalize QC for this patient (if enabled)
        if self.qc_manager:
            try:
                qc_outputs = self.qc_manager.finalize()
                if qc_outputs:
                    self.logger.info(f"\nQC finalized: {list(qc_outputs.keys())}")
            except Exception as e:
                self.logger.error(f"QC finalization failed: {e}", exc_info=True)

        # Summary
        self.logger.info(f"\n{'=' * 80}")
        self.logger.info(f"Patient {patient_id} processing complete:")
        self.logger.info(f"  Processed: {total_processed}")
        self.logger.info(f"  Skipped:   {total_skipped}")
        self.logger.info(f"  Errors:    {total_errors}")
        self.logger.info(f"{'=' * 80}\n")

    def _process_studies(
        self,
        patient_id: str,
        study_dirs: List[Path],
        step_groups: List[Tuple[str, List[str]]],
    ) -> Tuple[int, int, int]:
        """Process all studies with modality-level and study-level steps.

        Args:
            patient_id: Patient ID
            study_dirs: List of all study directories
            step_groups: List of (level, [step_names]) for modality/study steps

        Returns:
            Tuple of (total_processed, total_skipped, total_errors)
        """
        total_processed = 0
        total_skipped = 0
        total_errors = 0

        # Process each study
        for study_idx, study_dir in enumerate(study_dirs, 1):
            self.logger.info(
                f"\n[Study {study_idx}/{len(study_dirs)}] {study_dir.name}"
            )

            # Track modalities that have been processed successfully
            processed_modalities = set()

            # Execute step groups in order (preserves interleaving of per-modality and study-level)
            for group_level, group_steps in step_groups:
                if group_level == "modality":
                    # Execute per-modality steps on all modalities
                    for modality in self.config.modalities:
                        self.logger.info(f"  Processing modality: {modality}")

                        try:
                            # Find input file (only check on first group)
                            input_file = self._get_modality_file(study_dir, modality)

                            if (
                                input_file is None
                                and modality not in processed_modalities
                            ):
                                self.logger.warning(
                                    f"    Modality {modality} not found - skipping"
                                )
                                total_skipped += 1
                                continue

                            # Get output paths
                            paths = self._get_output_paths(
                                patient_id, study_dir, modality
                            )

                            # Check overwrite conditions (only on first group)
                            if modality not in processed_modalities:
                                if (
                                    paths["nifti"].exists()
                                    and not self.config.overwrite
                                ):
                                    if self.config.mode == "pipeline":
                                        raise FileExistsError(
                                            f"Output file exists and overwrite=False: {paths['nifti']}\n"
                                            "Set overwrite=true in config or use test mode"
                                        )
                                    else:
                                        self.logger.warning(
                                            f"    Output exists and overwrite=False - skipping {modality}"
                                        )
                                        total_skipped += 1
                                        continue

                            # Execute per-modality steps for this group
                            self._execute_per_modality_steps(
                                patient_id=patient_id,
                                study_dir=study_dir,
                                modality=modality,
                                paths=paths,
                                steps=group_steps,
                            )

                            processed_modalities.add(modality)
                            self.logger.info(f"    Successfully processed {modality}")
                            total_processed += 1

                        except FileExistsError:
                            # Re-raise overwrite errors (halt execution)
                            raise
                        except Exception as e:
                            self.logger.error(f"   [Error] Processing {modality}: {e}")
                            total_errors += 1
                            # Continue with next modality

                else:
                    # Execute study-level steps
                    self._execute_study_level_steps(
                        patient_id=patient_id, study_dir=study_dir, steps=group_steps
                    )

            # Free GPU models and large allocations between studies to prevent
            # RAM/VRAM accumulation (HD-BET ensemble, ECLARE, etc.)
            self._release_gpu_components()

        return total_processed, total_skipped, total_errors

    def _log_pipeline_order(self, patient_id: str) -> None:
        """Log pipeline configuration to console and JSON file."""
        from mengrowth.preprocessing.src.utils.pipeline_logger import (
            PipelineOrderRecord,
        )

        # Console logging
        self.logger.info("=" * 80)
        self.logger.info("PIPELINE CONFIGURATION")
        self.logger.info("=" * 80)
        self.logger.info(
            f"Mode: {self.config.mode}, Overwrite: {self.config.overwrite}"
        )
        self.logger.info(f"Step Order ({len(self.config.steps)} steps):")
        for i, step_name in enumerate(self.config.steps, 1):
            # Find matching config pattern
            pattern = None
            for p in self.config.step_configs.keys():
                if p in step_name:
                    pattern = p
                    break
            self.logger.info(f"  {i}. {step_name} (config: {pattern})")

        self.logger.info(f"Modalities: {', '.join(self.config.modalities)}")
        self.logger.info("=" * 80)

        # JSON logging
        from pathlib import Path

        artifacts_path = Path(self.config.preprocessing_artifacts_path)
        json_path = artifacts_path / patient_id / "pipeline_order.json"

        record = PipelineOrderRecord.from_config(patient_id, self.config)
        record.save(json_path)

    def _categorize_step_groups(self) -> Tuple[List[Tuple[str, List[str]]], List[str]]:
        """Categorize steps into modality/study groups and patient-level steps.

        Returns:
            Tuple of (modality_study_groups, patient_level_steps)
            - modality_study_groups: List of (level, [step_names]) for modality/study steps
            - patient_level_steps: List of patient-level step names
        """
        modality_study_groups = []
        patient_level_steps = []

        current_level = None
        current_steps = []

        for step_name in self.config.steps:
            # Find step level from STEP_METADATA
            step_level = "modality"  # Default
            for pattern, metadata in STEP_METADATA.items():
                if pattern in step_name:
                    step_level = metadata.level
                    break

            # Separate patient-level steps
            if step_level == "patient":
                # Finalize any current group first
                if current_steps:
                    modality_study_groups.append((current_level, current_steps))
                    current_level = None
                    current_steps = []
                # Add to patient-level list
                patient_level_steps.append(step_name)
            else:
                # Group modality/study steps
                if step_level != current_level:
                    if current_steps:
                        modality_study_groups.append((current_level, current_steps))
                    current_level = step_level
                    current_steps = [step_name]
                else:
                    current_steps.append(step_name)

        # Finalize last group
        if current_steps:
            modality_study_groups.append((current_level, current_steps))

        return modality_study_groups, patient_level_steps

    def _execute_per_modality_steps(
        self,
        patient_id: str,
        study_dir: Path,
        modality: str,
        paths: Dict[str, Path],
        steps: List[str],
    ) -> None:
        """Execute per-modality steps in order.

        Args:
            patient_id: Patient ID
            study_dir: Study directory
            modality: Modality being processed
            paths: Output paths dict
            steps: List of step names to execute
        """
        total_steps = len(steps)

        # Capture baseline before first step (if QC enabled)
        if self.qc_manager and total_steps > 0:
            input_file = self._get_modality_file(study_dir, modality)
            if input_file and input_file.exists():
                case_metadata = {
                    "patient_id": patient_id,
                    "study_id": study_dir.name,
                    "modality": modality,
                }
                try:
                    self.qc_manager.capture_baseline(case_metadata, input_file)
                except Exception as e:
                    self.logger.warning(
                        f"    Baseline capture failed for {modality}: {e}"
                    )

        for step_num, step_name in enumerate(steps, 1):
            # Check if step can be skipped due to checkpoint
            if self.checkpoint_manager:
                if self.checkpoint_manager.is_step_completed(
                    patient_id, study_dir.name, modality, step_name
                ):
                    self.logger.info(
                        f"    [{step_num}/{total_steps}] {step_name} - skipped (checkpoint exists)"
                    )
                    continue

            # Get step config
            step_config = self._get_step_config(step_name)

            # Create execution context
            context = StepExecutionContext(
                patient_id=patient_id,
                study_dir=study_dir,
                modality=modality,
                paths=paths,
                orchestrator=self,
                step_name=step_name,
                step_config=step_config,
            )

            # Get handler function
            pattern, handler_func = self.step_registry.get_handler(step_name)

            # Execute step
            try:
                result = handler_func(context, total_steps, step_num)

                # Trigger QC if enabled
                self._trigger_qc_if_needed(step_name, context, result)

                # Save checkpoint after successful step completion
                if self.checkpoint_manager:
                    output_path = paths.get("nifti") or paths.get("resampled")
                    if output_path:
                        self.checkpoint_manager.save_checkpoint(
                            patient_id=patient_id,
                            study_id=study_dir.name,
                            modality=modality,
                            step_name=step_name,
                            output_path=output_path,
                            metadata=result,
                        )

            except Exception as e:
                self.logger.error(f"    Step '{step_name}' failed: {e}")
                raise RuntimeError(f"Step '{step_name}' failed") from e

    def _execute_study_level_steps(
        self, patient_id: str, study_dir: Path, steps: List[str]
    ) -> None:
        """Execute study-level steps (registration, skull stripping).

        Args:
            patient_id: Patient ID
            study_dir: Study directory
            steps: List of study-level step names
        """
        for step_name in steps:
            # Get step config
            step_config = self._get_step_config(step_name)

            # Create context (no specific modality)
            context = StepExecutionContext(
                patient_id=patient_id,
                study_dir=study_dir,
                modality=None,  # N/A for study-level
                paths=None,  # Will be computed per-modality internally
                orchestrator=self,
                step_name=step_name,
                step_config=step_config,
            )

            # Get handler and execute
            pattern, handler_func = self.step_registry.get_handler(step_name)
            try:
                result = handler_func(context, total_steps=0, current_step_num=0)

                # Trigger QC if enabled
                self._trigger_qc_if_needed(step_name, context, result)

            except Exception as e:
                self.logger.error(f"  Study-level step '{step_name}' failed: {e}")
                # Continue anyway - don't halt for study-level failures

    def _execute_patient_level_steps(
        self, patient_id: str, study_dirs: List[Path], steps: List[str]
    ) -> None:
        """Execute patient-level steps that operate across all timestamps.

        Args:
            patient_id: Patient ID
            study_dirs: List of all study directories for this patient
            steps: List of patient-level step names
        """
        for step_name in steps:
            self.logger.info(f"\n  Executing patient-level step: {step_name}")

            # Get step config
            step_config = self._get_step_config(step_name)

            # Create execution context
            context = StepExecutionContext(
                patient_id=patient_id,
                study_dir=None,  # N/A for patient-level
                modality=None,  # N/A for patient-level
                paths=None,  # N/A for patient-level
                orchestrator=self,
                step_name=step_name,
                step_config=step_config,
                all_study_dirs=study_dirs,  # NEW: Pass all study dirs
            )

            # Get handler and execute
            pattern, handler_func = self.step_registry.get_handler(step_name)
            try:
                result = handler_func(context, total_steps=0, current_step_num=0)

                # Trigger QC if enabled
                self._trigger_qc_if_needed(step_name, context, result)

                # Store warped masks info for longitudinal registration
                if result and "warped_masks" in result and self.qc_manager:
                    self.qc_manager.store_warped_masks_info(
                        patient_id=patient_id, warped_masks=result["warped_masks"]
                    )

                self.logger.info(f"  ✓ {step_name} completed successfully")
            except Exception as e:
                self.logger.error(f"  ✗ Patient-level step '{step_name}' failed: {e}")
                raise RuntimeError(f"Patient-level step '{step_name}' failed") from e

    def _get_step_config(
        self, step_name: str
    ) -> Union[
        DataHarmonizationStepConfig,
        BiasFieldCorrectionStepConfig,
        ResamplingStepConfig,
        CubicPaddingStepConfig,
        RegistrationStepConfig,
        SkullStrippingStepConfig,
        IntensityNormalizationStepConfig,
        LongitudinalRegistrationStepConfig,
    ]:
        """Get configuration for a step name using substring matching.

        Args:
            step_name: Full step name (e.g., "intensity_normalization_2")

        Returns:
            Typed configuration object for the step

        Raises:
            ValueError: If no matching config found
        """
        for pattern, config in self.config.step_configs.items():
            if pattern in step_name:
                return config

        raise ValueError(
            f"No configuration found for step '{step_name}'. "
            f"Available patterns: {list(self.config.step_configs.keys())}"
        )


def run_preprocessing(
    config: PipelineExecutionConfig, patient_id: Optional[str] = None
) -> None:
    """Run preprocessing pipeline on selected patients.

    Args:
        config: Pipeline execution configuration (returned by loader)
        patient_id: Specific patient to process (overrides config.patient_selector)

    Raises:
        ValueError: If patient_selector is invalid
        FileNotFoundError: If patient directory not found
        RuntimeError: If processing fails
    """
    if not config.enabled:
        logger.info("Pipeline is disabled in config - exiting")
        return

    # Override patient_id if provided
    if patient_id is not None:
        config.patient_id = patient_id
        config.patient_selector = "single"
        logger.info(f"Overriding config: processing single patient {patient_id}")

    # Initialize orchestrator
    orchestrator = PreprocessingOrchestrator(config, verbose=True)

    # Process patients
    if config.patient_selector == "single":
        orchestrator.run_patient(config.patient_id)

    elif config.patient_selector == "all":
        # Get all patient directories
        dataset_root = Path(config.dataset_root)
        patient_dirs = sorted(
            [
                d
                for d in dataset_root.iterdir()
                if d.is_dir() and d.name.startswith("MenGrowth-")
            ]
        )

        logger.info(f"Processing all patients: {len(patient_dirs)} found")

        for idx, patient_dir in enumerate(patient_dirs, 1):
            logger.info(f"\n\n{'#' * 80}")
            logger.info(f"# Patient {idx}/{len(patient_dirs)}: {patient_dir.name}")
            logger.info(f"{'#' * 80}")

            try:
                orchestrator.run_patient(patient_dir.name)
            except Exception as e:
                logger.error(f"Failed to process {patient_dir.name}: {e}")
                # Continue with next patient

        logger.info(f"\n\n{'#' * 80}")
        logger.info(
            f"# All patients processing complete ({len(patient_dirs)} patients)"
        )
        logger.info(f"{'#' * 80}")

    else:
        raise ValueError(
            f"Invalid patient_selector: {config.patient_selector}. Must be 'single' or 'all'"
        )
