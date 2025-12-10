"""Preprocessing pipeline orchestrator for MenGrowth dataset.

This module coordinates the execution of preprocessing steps on patient data,
managing file paths, mode semantics, and visualization outputs.
"""

from pathlib import Path
from typing import List, Optional, Any, Tuple, Dict, Union
import logging

from mengrowth.preprocessing.src.config import (
    PreprocessingPipelineConfig,
    PipelineExecutionConfig,
    DataHarmonizationConfig,  # Backwards compatibility alias
    StepRegistry,
    StepExecutionContext,
    STEP_METADATA,
    # Step config types for type hints
    DataHarmonizationStepConfig,
    BiasFieldCorrectionStepConfig,
    ResamplingStepConfig,
    RegistrationStepConfig,
    SkullStrippingStepConfig,
    IntensityNormalizationStepConfig,
)
from mengrowth.preprocessing.src.data_harmonization.io import NRRDtoNIfTIConverter
from mengrowth.preprocessing.src.data_harmonization.orient import Reorienter
from mengrowth.preprocessing.src.data_harmonization.head_masking.conservative import ConservativeBackgroundRemover
from mengrowth.preprocessing.src.data_harmonization.head_masking.self import SELFBackgroundRemover
from mengrowth.preprocessing.src.bias_field_correction.n4_sitk import N4BiasFieldCorrector
from mengrowth.preprocessing.src.normalization.zscore import ZScoreNormalizer
from mengrowth.preprocessing.src.normalization.kde import KDENormalizer
from mengrowth.preprocessing.src.normalization.percentile_minmax import PercentileMinMaxNormalizer
from mengrowth.preprocessing.src.normalization.whitestripe import WhiteStripeNormalizer
from mengrowth.preprocessing.src.normalization.fcm import FCMNormalizer
from mengrowth.preprocessing.src.normalization.lsq import LSQNormalizer
from mengrowth.preprocessing.src.resampling.bspline import BSplineResampler
from mengrowth.preprocessing.src.resampling.eclare import EclareResampler
from mengrowth.preprocessing.src.resampling.composite import CompositeResampler
from mengrowth.preprocessing.src.skull_stripping.hdbet import HDBetSkullStripper
from mengrowth.preprocessing.src.skull_stripping.synthstrip import SynthStripSkullStripper

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
        self._step_results = {}  # Store results from previous steps

        # State variables
        self.selected_reference_modality = None

        self.logger.info("Preprocessing orchestrator initialized with dynamic pipeline execution")

    def _register_step_handlers(self) -> None:
        """Register all step handler functions with the registry."""
        from mengrowth.preprocessing.src.steps import (
            data_harmonization,
            bias_field_correction,
            intensity_normalization,
            resampling,
            registration,
            skull_stripping
        )

        self.step_registry.register("data_harmonization", data_harmonization.execute)
        self.step_registry.register("bias_field_correction", bias_field_correction.execute)
        self.step_registry.register("intensity_normalization", intensity_normalization.execute)
        self.step_registry.register("resampling", resampling.execute)
        self.step_registry.register("registration", registration.execute)
        self.step_registry.register("skull_stripping", skull_stripping.execute)

        self.logger.debug(f"Registered {len(self.step_registry.list_patterns())} step patterns")

    def _get_component(self, component_type: str, config: Any) -> Any:
        """Get or create a component instance (lazy initialization).

        Args:
            component_type: Type of component (e.g., "normalizer_kde", "resampler_bspline")
            config: Configuration for the component

        Returns:
            Component instance (cached for reuse)
        """
        if component_type not in self._components:
            self._components[component_type] = self._create_component(component_type, config)
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
                target_orientation=config.reorient_to,
                verbose=self.verbose
            )
        elif component_type.startswith("background_remover_"):
            method = component_type.split("_", 2)[2]  # Extract method from "background_remover_METHOD"
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
        else:
            raise ValueError(f"Unknown component type: {component_type}")

    def _create_background_remover(self, method: str, config: Any) -> Any:
        """Create background remover instance."""
        if method == "border_connected_percentile":
            return ConservativeBackgroundRemover(
                config=config.background_zeroing,
                verbose=self.verbose
            )
        elif method == "self_head_mask":
            return SELFBackgroundRemover(
                config=config.background_zeroing,
                verbose=self.verbose
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
            return N4BiasFieldCorrector(
                config=bf_config_dict,
                verbose=self.verbose
            )
        else:
            raise ValueError(f"Unknown bias field correction method: {method}")

    def _create_normalizer(self, method: str, config: Any) -> Any:
        """Create normalizer instance based on method."""
        if method == "zscore":
            norm_config_dict = {
                "norm_value": config.intensity_normalization.norm_value,
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
            return PercentileMinMaxNormalizer(config=norm_config_dict, verbose=self.verbose)
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
                verbose=self.verbose
            )
        elif method == "eclare":
            resample_config_dict = {
                "conda_environment_eclare": config.resampling.conda_environment_eclare,
                "batch_size": config.resampling.batch_size,
                "n_patches": config.resampling.n_patches,
                "patch_sampling": config.resampling.patch_sampling,
                "suffix": config.resampling.suffix,
                "gpu_id": config.resampling.gpu_id,
                "verbose": config.resampling.verbose if hasattr(config.resampling, "verbose") else self.verbose,
            }
            return EclareResampler(
                target_voxel_size=config.resampling.target_voxel_size,
                config=resample_config_dict,
                verbose=self.verbose
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
                verbose=self.verbose
            )
        else:
            raise ValueError(f"Unknown resampling method: {method}")

    def _create_intra_study_registrator(self, config: Any) -> Any:
        """Create intra-study to reference registrator."""
        from mengrowth.preprocessing.src.registration.factory import create_multi_modal_coregistration
        from mengrowth.preprocessing.src.registration.constants import DEFAULT_REGISTRATION_ENGINE

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
            "engine": config.intra_study_to_reference.engine or DEFAULT_REGISTRATION_ENGINE,
        }
        return create_multi_modal_coregistration(config=intra_study_config, verbose=self.verbose)

    def _create_atlas_registrator(self, config: Any) -> Any:
        """Create intra-study to atlas registrator."""
        from mengrowth.preprocessing.src.registration.factory import create_intra_study_to_atlas
        from mengrowth.preprocessing.src.registration.constants import DEFAULT_REGISTRATION_ENGINE

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
        }
        return create_intra_study_to_atlas(
            config=atlas_config,
            reference_modality=self.selected_reference_modality,
            verbose=self.verbose
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
            return SynthStripSkullStripper(config=skull_strip_config, verbose=self.verbose)
        else:
            raise ValueError(f"Unknown skull stripping method: {method}")

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
        self,
        patient_id: str,
        study_dir: Path,
        modality: str
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
            artifacts_base = Path(self.config.preprocessing_artifacts_path) / patient_id / study_name
        else:
            # Pipeline mode: write in-place
            output_base = study_dir
            viz_base = Path(self.config.viz_root) / patient_id / study_name
            artifacts_base = Path(self.config.preprocessing_artifacts_path) / patient_id / study_name

        # Ensure directories exist
        output_base.mkdir(parents=True, exist_ok=True)
        viz_base.mkdir(parents=True, exist_ok=True)
        artifacts_base.mkdir(parents=True, exist_ok=True)

        return {
            "nifti": output_base / f"{modality}.nii.gz",
            "reoriented": output_base / f"{modality}.nii.gz",  # Same file, in-place
            "masked": output_base / f"{modality}.nii.gz",  # Same file, in-place
            "bias_corrected": output_base / f"{modality}.nii.gz",  # Same file, in-place
            "resampled": output_base / f"{modality}.nii.gz",  # Same file, in-place
            "bias_field": artifacts_base / f"{modality}_bias_field.nii.gz",  # Artifact: bias field
            "viz_convert": viz_base / f"step0_convert_{modality}.png",
            "viz_reorient": viz_base / f"step0_reorient_{modality}.png",
            "viz_background": viz_base / f"step0_background_{modality}.png",
            "viz_bias_field": viz_base / f"step1_bias_field_{modality}.png",
            "viz_resampling": viz_base / f"step2_resampling_{modality}.png",
            "viz_registration": viz_base / f"step3_registration_{modality}.png",
        }

    def _visualize_normalization_and_resampling(
        self,
        original_path: Path,
        normalized_path: Path,
        resampled_path: Path,
        output_path: Path,
        norm_result: dict,
        resample_result: dict
    ) -> None:
        """Generate 3-row visualization: original → normalized → resampled.

        Creates a comprehensive visualization showing all three stages of processing
        when normalization is enabled before resampling. Each row shows:
        - 3 anatomical views (axial, sagittal, coronal)
        - 1 intensity histogram

        Args:
            original_path: Path to original image
            normalized_path: Path to normalized image
            resampled_path: Path to resampled image
            output_path: Path to save visualization (PNG)
            norm_result: Normalization metadata from normalizer.execute()
            resample_result: Resampling metadata from resampler.execute()

        Raises:
            RuntimeError: If visualization generation fails
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import nibabel as nib

        self.logger.info(f"Generating combined normalization + resampling visualization: {output_path}")

        try:
            # Load all three images
            original_img = nib.load(str(original_path))
            original_data = original_img.get_fdata()

            normalized_img = nib.load(str(normalized_path))
            normalized_data = normalized_img.get_fdata()

            resampled_img = nib.load(str(resampled_path))
            resampled_data = resampled_img.get_fdata()

            # Get middle slices for each view - using original image dimensions
            # Axial (XY plane, slice along Z)
            mid_z_orig = original_data.shape[2] // 2
            axial_orig = original_data[:, :, mid_z_orig].T

            mid_z_norm = normalized_data.shape[2] // 2
            axial_norm = normalized_data[:, :, mid_z_norm].T

            mid_z_resamp = resampled_data.shape[2] // 2
            axial_resamp = resampled_data[:, :, mid_z_resamp].T

            # Sagittal (YZ plane, slice along X)
            mid_x_orig = original_data.shape[0] // 2
            sagittal_orig = original_data[mid_x_orig, :, :].T

            mid_x_norm = normalized_data.shape[0] // 2
            sagittal_norm = normalized_data[mid_x_norm, :, :].T

            mid_x_resamp = resampled_data.shape[0] // 2
            sagittal_resamp = resampled_data[mid_x_resamp, :, :].T

            # Coronal (XZ plane, slice along Y)
            mid_y_orig = original_data.shape[1] // 2
            coronal_orig = original_data[:, mid_y_orig, :].T

            mid_y_norm = normalized_data.shape[1] // 2
            coronal_norm = normalized_data[:, mid_y_norm, :].T

            mid_y_resamp = resampled_data.shape[1] // 2
            coronal_resamp = resampled_data[:, mid_y_resamp, :].T

            # Create figure: 3 rows x 4 columns (3 views + 1 histogram per row)
            fig, axes = plt.subplots(3, 4, figsize=(20, 15))

            # Determine normalization method for title
            norm_method = type(self.normalizer).__name__.replace("Normalizer", "")
            resample_method = type(self.resampler).__name__.replace("Resampler", "")

            fig.suptitle(
                f'Normalization ({norm_method}) + Resampling ({resample_method}): {original_path.stem}',
                fontsize=16,
                fontweight='bold'
            )

            # Row 1: Original image
            vmin_orig = original_data.min()
            vmax_orig = original_data.max()

            axes[0, 0].imshow(axial_orig, cmap='gray', origin='lower', vmin=vmin_orig, vmax=vmax_orig)
            axes[0, 0].set_title('Original - Axial', fontsize=11)
            axes[0, 0].axis('off')

            axes[0, 1].imshow(sagittal_orig, cmap='gray', origin='lower', vmin=vmin_orig, vmax=vmax_orig)
            axes[0, 1].set_title('Original - Sagittal', fontsize=11)
            axes[0, 1].axis('off')

            axes[0, 2].imshow(coronal_orig, cmap='gray', origin='lower', vmin=vmin_orig, vmax=vmax_orig)
            axes[0, 2].set_title('Original - Coronal', fontsize=11)
            axes[0, 2].axis('off')

            # Histogram for original
            orig_nonzero = original_data[original_data > 0]
            axes[0, 3].hist(orig_nonzero, bins=100, alpha=0.7, color='blue', density=True)
            axes[0, 3].set_xlabel('Intensity', fontsize=9)
            axes[0, 3].set_ylabel('Density', fontsize=9)
            axes[0, 3].set_title('Original Histogram', fontsize=11)
            axes[0, 3].grid(True, alpha=0.3)

            # Row 2: Normalized image
            vmin_norm = normalized_data.min()
            vmax_norm = normalized_data.max()

            axes[1, 0].imshow(axial_norm, cmap='gray', origin='lower', vmin=vmin_norm, vmax=vmax_norm)
            axes[1, 0].set_title('Normalized - Axial', fontsize=11)
            axes[1, 0].axis('off')

            axes[1, 1].imshow(sagittal_norm, cmap='gray', origin='lower', vmin=vmin_norm, vmax=vmax_norm)
            axes[1, 1].set_title('Normalized - Sagittal', fontsize=11)
            axes[1, 1].axis('off')

            axes[1, 2].imshow(coronal_norm, cmap='gray', origin='lower', vmin=vmin_norm, vmax=vmax_norm)
            axes[1, 2].set_title('Normalized - Coronal', fontsize=11)
            axes[1, 2].axis('off')

            # Histogram for normalized
            norm_nonzero = normalized_data[normalized_data > vmin_norm]
            axes[1, 3].hist(norm_nonzero, bins=100, alpha=0.7, color='green', density=True)
            axes[1, 3].set_xlabel('Intensity', fontsize=9)
            axes[1, 3].set_ylabel('Density', fontsize=9)
            axes[1, 3].set_title('Normalized Histogram', fontsize=11)
            axes[1, 3].grid(True, alpha=0.3)

            # Row 3: Resampled image
            vmin_resamp = resampled_data.min()
            vmax_resamp = resampled_data.max()

            axes[2, 0].imshow(axial_resamp, cmap='gray', origin='lower', vmin=vmin_resamp, vmax=vmax_resamp)
            axes[2, 0].set_title('Resampled - Axial', fontsize=11)
            axes[2, 0].axis('off')

            axes[2, 1].imshow(sagittal_resamp, cmap='gray', origin='lower', vmin=vmin_resamp, vmax=vmax_resamp)
            axes[2, 1].set_title('Resampled - Sagittal', fontsize=11)
            axes[2, 1].axis('off')

            axes[2, 2].imshow(coronal_resamp, cmap='gray', origin='lower', vmin=vmin_resamp, vmax=vmax_resamp)
            axes[2, 2].set_title('Resampled - Coronal', fontsize=11)
            axes[2, 2].axis('off')

            # Histogram for resampled
            resamp_nonzero = resampled_data[resampled_data > vmin_resamp]
            axes[2, 3].hist(resamp_nonzero, bins=100, alpha=0.7, color='orange', density=True)
            axes[2, 3].set_xlabel('Intensity', fontsize=9)
            axes[2, 3].set_ylabel('Density', fontsize=9)
            axes[2, 3].set_title('Resampled Histogram', fontsize=11)
            axes[2, 3].grid(True, alpha=0.3)

            # Add metadata text
            # Build metadata string based on normalization method
            norm_info = f"Normalization: {norm_method}\n"
            if 'mean' in norm_result:
                norm_info += f"  Mean={norm_result['mean']:.3f}, Std={norm_result['std']:.3f}\n"
            elif 'mode' in norm_result:
                norm_info += f"  Mode={norm_result['mode']:.3f}\n"
            elif 'p1_value' in norm_result:
                norm_info += f"  P{norm_result['p1_percentile']}={norm_result['p1_value']:.3f}, P{norm_result['p2_percentile']}={norm_result['p2_value']:.3f}\n"

            metadata_text = (
                f"{norm_info}\n"
                f"Resampling: {resample_method}\n"
                f"  Original Spacing: {resample_result['original_spacing']}\n"
                f"  Target Spacing: {resample_result['target_spacing']}\n"
                f"  Original Shape: {resample_result['original_shape']}\n"
                f"  Resampled Shape: {resample_result['resampled_shape']}"
            )

            fig.text(
                0.5, 0.01,
                metadata_text,
                ha='center',
                fontsize=9,
                family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            )

            plt.tight_layout(rect=[0, 0.08, 1, 0.98])

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save figure
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

            self.logger.info(f"Visualization saved to {output_path}")

        except Exception as e:
            self.logger.error(f"Visualization generation failed: {e}")
            raise RuntimeError(f"Visualization failed: {e}") from e

    def run_patient(self, patient_id: str) -> None:
        """Run preprocessing pipeline for a single patient with dynamic step execution.

        Args:
            patient_id: Patient ID to process

        Raises:
            FileNotFoundError: If patient directory not found
            RuntimeError: If processing fails
        """
        self.logger.info(f"{'='*80}")
        self.logger.info(f"Processing patient: {patient_id}")
        self.logger.info(f"{'='*80}")

        # Get study directories
        study_dirs = self._get_study_directories(patient_id)

        # Log and save pipeline configuration
        self._log_pipeline_order(patient_id)

        total_processed = 0
        total_skipped = 0
        total_errors = 0

        # Categorize steps into per-modality and study-level
        per_modality_steps, study_level_steps = self._categorize_steps()

        # Process each study
        for study_idx, study_dir in enumerate(study_dirs, 1):
            self.logger.info(f"\n[Study {study_idx}/{len(study_dirs)}] {study_dir.name}")

            # Execute per-modality steps
            for modality in self.config.modalities:
                self.logger.info(f"  Processing modality: {modality}")

                try:
                    # Find input file
                    input_file = self._get_modality_file(study_dir, modality)

                    if input_file is None:
                        self.logger.warning(f"    Modality {modality} not found - skipping")
                        total_skipped += 1
                        continue

                    # Get output paths
                    paths = self._get_output_paths(patient_id, study_dir, modality)

                    # Check overwrite conditions (strict: error and halt if exists)
                    if paths["nifti"].exists() and not self.config.overwrite:
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

                    # Execute per-modality steps dynamically
                    self._execute_per_modality_steps(
                        patient_id=patient_id,
                        study_dir=study_dir,
                        modality=modality,
                        paths=paths,
                        steps=per_modality_steps
                    )

                    self.logger.info(f"    Successfully processed {modality}")
                    total_processed += 1

                except FileExistsError as e:
                    # Re-raise overwrite errors (halt execution)
                    raise
                except Exception as e:
                    self.logger.error(f"   [Error] Processing {modality}: {e}")
                    total_errors += 1
                    # Continue with next modality

            # Execute study-level steps dynamically
            self._execute_study_level_steps(
                patient_id=patient_id,
                study_dir=study_dir,
                steps=study_level_steps
            )

        # Summary
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"Patient {patient_id} processing complete:")
        self.logger.info(f"  Processed: {total_processed}")
        self.logger.info(f"  Skipped:   {total_skipped}")
        self.logger.info(f"  Errors:    {total_errors}")
        self.logger.info(f"{'='*80}\n")

    def _log_pipeline_order(self, patient_id: str) -> None:
        """Log pipeline configuration to console and JSON file."""
        from mengrowth.preprocessing.src.utils.pipeline_logger import PipelineOrderRecord

        # Console logging
        self.logger.info("="*80)
        self.logger.info("PIPELINE CONFIGURATION")
        self.logger.info("="*80)
        self.logger.info(f"Mode: {self.config.mode}, Overwrite: {self.config.overwrite}")
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
        self.logger.info("="*80)

        # JSON logging
        from pathlib import Path
        artifacts_path = Path(self.config.preprocessing_artifacts_path)
        json_path = artifacts_path / patient_id / "pipeline_order.json"

        record = PipelineOrderRecord.from_config(patient_id, self.config)
        record.save(json_path)

    def _categorize_steps(self) -> Tuple[List[str], List[str]]:
        """Categorize steps into per-modality and study-level using STEP_METADATA.

        Returns:
            Tuple of (per_modality_steps, study_level_steps)
        """
        per_modality_steps = []
        study_level_steps = []

        for step_name in self.config.steps:
            # Find matching pattern in STEP_METADATA
            step_level = "modality"  # Default
            for pattern, metadata in STEP_METADATA.items():
                if pattern in step_name:
                    step_level = metadata.level
                    break
            
            if step_level == "study":
                study_level_steps.append(step_name)
            else:
                per_modality_steps.append(step_name)

        return per_modality_steps, study_level_steps

    def _execute_per_modality_steps(
        self,
        patient_id: str,
        study_dir: Path,
        modality: str,
        paths: Dict[str, Path],
        steps: List[str]
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

        for step_num, step_name in enumerate(steps, 1):
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
                step_config=step_config
            )

            # Get handler function
            pattern, handler_func = self.step_registry.get_handler(step_name)

            # Execute step
            try:
                result = handler_func(context, total_steps, step_num)
                # Store result for potential use by downstream steps
                self._step_results[step_name] = result
            except Exception as e:
                self.logger.error(f"    Step '{step_name}' failed: {e}")
                raise RuntimeError(f"Step '{step_name}' failed") from e

    def _execute_study_level_steps(
        self,
        patient_id: str,
        study_dir: Path,
        steps: List[str]
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
                paths=None,     # Will be computed per-modality internally
                orchestrator=self,
                step_name=step_name,
                step_config=step_config
            )

            # Get handler and execute
            pattern, handler_func = self.step_registry.get_handler(step_name)
            try:
                result = handler_func(context, total_steps=0, current_step_num=0)
                self._step_results[step_name] = result
            except Exception as e:
                self.logger.error(f"  Study-level step '{step_name}' failed: {e}")
                # Continue anyway - don't halt for study-level failures

    def _get_step_config(
        self, 
        step_name: str
    ) -> Union[
        DataHarmonizationStepConfig,
        BiasFieldCorrectionStepConfig,
        ResamplingStepConfig,
        RegistrationStepConfig,
        SkullStrippingStepConfig,
        IntensityNormalizationStepConfig,
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


def run_preprocessing(config: PipelineExecutionConfig, patient_id: Optional[str] = None) -> None:
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
        patient_dirs = sorted([d for d in dataset_root.iterdir() if d.is_dir() and d.name.startswith("MenGrowth-")])

        logger.info(f"Processing all patients: {len(patient_dirs)} found")

        for idx, patient_dir in enumerate(patient_dirs, 1):
            logger.info(f"\n\n{'#'*80}")
            logger.info(f"# Patient {idx}/{len(patient_dirs)}: {patient_dir.name}")
            logger.info(f"{'#'*80}")

            try:
                orchestrator.run_patient(patient_dir.name)
            except Exception as e:
                logger.error(f"Failed to process {patient_dir.name}: {e}")
                # Continue with next patient

        logger.info(f"\n\n{'#'*80}")
        logger.info(f"# All patients processing complete ({len(patient_dirs)} patients)")
        logger.info(f"{'#'*80}")

    else:
        raise ValueError(
            f"Invalid patient_selector: {config.patient_selector}. Must be 'single' or 'all'"
        )
