"""Preprocessing pipeline orchestrator for MenGrowth dataset.

This module coordinates the execution of preprocessing steps on patient data,
managing file paths, mode semantics, and visualization outputs.
"""

from pathlib import Path
from typing import List, Optional, Union
import logging

from mengrowth.preprocessing.src.config import PreprocessingPipelineConfig, DataHarmonizationConfig
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

    def __init__(self, config: DataHarmonizationConfig, verbose: bool = False) -> None:
        """Initialize preprocessing orchestrator.

        Args:
            config: Data harmonization configuration
            verbose: Enable verbose logging
        """
        self.config = config
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)

        # Initialize preprocessing steps
        self.converter = NRRDtoNIfTIConverter(verbose=verbose)
        self.reorienter = Reorienter(
            target_orientation=config.step0_data_harmonization.reorient_to,
            verbose=verbose
        )

        # Select background removal algorithm based on method
        bg_method = config.step0_data_harmonization.background_zeroing.method
        if bg_method is None:
            self.background_remover = None
            self.logger.info("Preprocessing orchestrator initialized with background removal disabled")
        elif bg_method == "border_connected_percentile":
            self.background_remover = ConservativeBackgroundRemover(
                config=config.step0_data_harmonization.background_zeroing,
                verbose=verbose
            )
            self.logger.info(f"Preprocessing orchestrator initialized with background method: {bg_method}")
        elif bg_method == "self_head_mask":
            self.background_remover = SELFBackgroundRemover(
                config=config.step0_data_harmonization.background_zeroing,
                verbose=verbose
            )
            self.logger.info(f"Preprocessing orchestrator initialized with background method: {bg_method}")
        else:
            raise ValueError(
                f"Unknown background removal method: {bg_method}. "
                "Must be None, 'border_connected_percentile', or 'self_head_mask'"
            )

        # Select bias field correction algorithm based on method
        bf_method = config.step1_bias_field_correction.bias_field_correction.method
        if bf_method is None:
            self.bias_corrector = None
            self.logger.info("Preprocessing orchestrator initialized with bias field correction disabled")
        elif bf_method == "n4":
            # Convert BiasFieldCorrectionConfig to dictionary for initializer
            bf_config_dict = {
                "shrink_factor": config.step1_bias_field_correction.bias_field_correction.shrink_factor,
                "max_iterations": config.step1_bias_field_correction.bias_field_correction.max_iterations,
                "bias_field_fwhm": config.step1_bias_field_correction.bias_field_correction.bias_field_fwhm,
                "convergence_threshold": config.step1_bias_field_correction.bias_field_correction.convergence_threshold,
            }
            self.bias_corrector = N4BiasFieldCorrector(
                config=bf_config_dict,
                verbose=verbose
            )
            self.logger.info(f"Preprocessing orchestrator initialized with bias correction method: {bf_method}")
        else:
            raise ValueError(
                f"Unknown bias field correction method: {bf_method}. Must be None or 'n4'"
            )

        # Select normalization algorithm based on method (applied before resampling)
        norm_method = config.step2_resampling.resampling.normalize_method
        if norm_method is None:
            self.normalizer = None
            self.logger.info("Preprocessing orchestrator initialized with normalization disabled")
        elif norm_method == "zscore":
            # Convert config to dictionary for initializer
            norm_config_dict = {
                "norm_value": config.step2_resampling.resampling.norm_value,
            }
            self.normalizer = ZScoreNormalizer(
                config=norm_config_dict,
                verbose=verbose
            )
            self.logger.info(f"Preprocessing orchestrator initialized with normalization method: {norm_method}")
        elif norm_method == "kde":
            # Convert config to dictionary for initializer
            norm_config_dict = {
                "norm_value": config.step2_resampling.resampling.norm_value,
            }
            self.normalizer = KDENormalizer(
                config=norm_config_dict,
                verbose=verbose
            )
            self.logger.info(f"Preprocessing orchestrator initialized with normalization method: {norm_method}")
        elif norm_method == "percentile_minmax":
            # Convert config to dictionary for initializer
            norm_config_dict = {
                "p1": config.step2_resampling.resampling.p1,
                "p2": config.step2_resampling.resampling.p2,
            }
            self.normalizer = PercentileMinMaxNormalizer(
                config=norm_config_dict,
                verbose=verbose
            )
            self.logger.info(f"Preprocessing orchestrator initialized with normalization method: {norm_method}")
        elif norm_method == "whitestripe":
            # Convert config to dictionary for initializer
            norm_config_dict = {
                "width": config.step2_resampling.resampling.whitestripe_width,
                "width_l": config.step2_resampling.resampling.whitestripe_width_l,
                "width_u": config.step2_resampling.resampling.whitestripe_width_u,
            }
            self.normalizer = WhiteStripeNormalizer(
                config=norm_config_dict,
                verbose=verbose
            )
            self.logger.info(f"Preprocessing orchestrator initialized with normalization method: {norm_method}")
        elif norm_method == "fcm":
            # Convert config to dictionary for initializer
            norm_config_dict = {
                "n_clusters": config.step2_resampling.resampling.fcm_n_clusters,
                "tissue_type": config.step2_resampling.resampling.fcm_tissue_type,
                "max_iter": config.step2_resampling.resampling.fcm_max_iter,
                "error_threshold": config.step2_resampling.resampling.fcm_error_threshold,
                "fuzziness": config.step2_resampling.resampling.fcm_fuzziness,
            }
            self.normalizer = FCMNormalizer(
                config=norm_config_dict,
                verbose=verbose
            )
            self.logger.info(f"Preprocessing orchestrator initialized with normalization method: {norm_method}")
        elif norm_method == "lsq":
            # Convert config to dictionary for initializer
            norm_config_dict = {
                "norm_value": config.step2_resampling.resampling.norm_value,
            }
            self.normalizer = LSQNormalizer(
                config=norm_config_dict,
                verbose=verbose
            )
            self.logger.info(f"Preprocessing orchestrator initialized with normalization method: {norm_method} (population-based)")
        else:
            raise ValueError(
                f"Unknown normalization method: {norm_method}. "
                "Must be None, 'zscore', 'kde', 'percentile_minmax', 'whitestripe', 'fcm', or 'lsq'"
            )

        # Select resampling algorithm based on method
        resample_method = config.step2_resampling.resampling.method
        if resample_method is None:
            self.resampler = None
            self.logger.info("Preprocessing orchestrator initialized with resampling disabled")
        elif resample_method == "bspline":
            # Convert ResamplingConfig to dictionary for initializer
            resample_config_dict = {
                "bspline_order": config.step2_resampling.resampling.bspline_order,
            }
            self.resampler = BSplineResampler(
                target_voxel_size=config.step2_resampling.resampling.target_voxel_size,
                config=resample_config_dict,
                verbose=verbose
            )
            self.logger.info(f"Preprocessing orchestrator initialized with resampling method: {resample_method}")
        elif resample_method == "eclare":
            # Convert ResamplingConfig to dictionary for initializer
            resample_config_dict = {
                "conda_environment_eclare": config.step2_resampling.resampling.conda_environment_eclare,
                "batch_size": config.step2_resampling.resampling.batch_size,
                "n_patches": config.step2_resampling.resampling.n_patches,
                "patch_sampling": config.step2_resampling.resampling.patch_sampling,
                "suffix": config.step2_resampling.resampling.suffix,
                "gpu_id": config.step2_resampling.resampling.gpu_id,
                "verbose": config.step2_resampling.resampling.verbose if hasattr(config.step2_resampling.resampling, "verbose") else verbose,
            }
            self.resampler = EclareResampler(
                target_voxel_size=config.step2_resampling.resampling.target_voxel_size,
                config=resample_config_dict,
                verbose=verbose
            )
            self.logger.info(f"Preprocessing orchestrator initialized with resampling method: {resample_method}")
        elif resample_method == "composite":
            # Convert ResamplingConfig to dictionary for initializer
            # Composite needs parameters for both BSpline and ECLARE
            resample_config_dict = {
                # Composite-specific parameters
                "composite_interpolator": config.step2_resampling.resampling.composite_interpolator,
                "composite_dl_method": config.step2_resampling.resampling.composite_dl_method,
                "max_mm_interpolator": config.step2_resampling.resampling.max_mm_interpolator,
                "max_mm_dl_method": config.step2_resampling.resampling.max_mm_dl_method,
                "resample_mm_to_interpolator_if_max_mm_dl_method": config.step2_resampling.resampling.resample_mm_to_interpolator_if_max_mm_dl_method,
                # BSpline parameters
                "bspline_order": config.step2_resampling.resampling.bspline_order,
                # ECLARE parameters
                "conda_environment_eclare": config.step2_resampling.resampling.conda_environment_eclare,
                "batch_size": config.step2_resampling.resampling.batch_size,
                "n_patches": config.step2_resampling.resampling.n_patches,
                "patch_sampling": config.step2_resampling.resampling.patch_sampling,
                "suffix": config.step2_resampling.resampling.suffix,
                "gpu_id": config.step2_resampling.resampling.gpu_id,
            }
            self.resampler = CompositeResampler(
                target_voxel_size=config.step2_resampling.resampling.target_voxel_size,
                config=resample_config_dict,
                verbose=verbose
            )
            self.logger.info(f"Preprocessing orchestrator initialized with resampling method: {resample_method}")
        else:
            raise ValueError(
                f"Unknown resampling method: {resample_method}. Must be None, 'bspline', 'eclare', or 'composite'"
            )

        # Step 3a: Intra-study to reference registration
        intra_study_to_ref_method = config.step3_registration.intra_study_to_reference.method
        if intra_study_to_ref_method is None:
            self.intra_study_to_ref_registrator = None
            self.selected_reference_modality = None
            self.logger.info("Intra-study to reference registration disabled")
        elif intra_study_to_ref_method == "ants":
            from mengrowth.preprocessing.src.registration.factory import create_multi_modal_coregistration
            from mengrowth.preprocessing.src.registration.constants import DEFAULT_REGISTRATION_ENGINE
            # Convert config to dictionary
            intra_study_to_ref_config = {
                "reference_modality_priority": config.step3_registration.intra_study_to_reference.reference_modality_priority,
                "transform_type": config.step3_registration.intra_study_to_reference.transform_type,
                "metric": config.step3_registration.intra_study_to_reference.metric,
                "metric_bins": config.step3_registration.intra_study_to_reference.metric_bins,
                "sampling_strategy": config.step3_registration.intra_study_to_reference.sampling_strategy,
                "sampling_percentage": config.step3_registration.intra_study_to_reference.sampling_percentage,
                "number_of_iterations": config.step3_registration.intra_study_to_reference.number_of_iterations,
                "shrink_factors": config.step3_registration.intra_study_to_reference.shrink_factors,
                "smoothing_sigmas": config.step3_registration.intra_study_to_reference.smoothing_sigmas,
                "convergence_threshold": config.step3_registration.intra_study_to_reference.convergence_threshold,
                "convergence_window_size": config.step3_registration.intra_study_to_reference.convergence_window_size,
                "write_composite_transform": config.step3_registration.intra_study_to_reference.write_composite_transform,
                "interpolation": config.step3_registration.intra_study_to_reference.interpolation,
                "engine": config.step3_registration.intra_study_to_reference.engine or DEFAULT_REGISTRATION_ENGINE,
            }
            self.intra_study_to_ref_registrator = create_multi_modal_coregistration(
                config=intra_study_to_ref_config,
                verbose=verbose
            )
            self.selected_reference_modality = None  # Will be set during execution
            self.logger.info(f"Intra-study to reference registration initialized: {intra_study_to_ref_method}")
        else:
            raise ValueError(
                f"Unknown intra-study to reference method: {intra_study_to_ref_method}. Must be None or 'ants'"
            )

        # Step 3b: Intra-study to atlas registration
        intra_study_to_atlas_method = config.step3_registration.intra_study_to_atlas.method
        if intra_study_to_atlas_method is None:
            self.intra_study_to_atlas_registrator = None
            self.logger.info("Intra-study to atlas registration disabled")
        elif intra_study_to_atlas_method == "ants":
            from mengrowth.preprocessing.src.registration.constants import DEFAULT_REGISTRATION_ENGINE
            # Convert config to dictionary
            intra_study_to_atlas_config = {
                "atlas_path": config.step3_registration.intra_study_to_atlas.atlas_path,
                "transforms": config.step3_registration.intra_study_to_atlas.transforms,
                "create_composite_transforms": config.step3_registration.intra_study_to_atlas.create_composite_transforms,
                "metric": config.step3_registration.intra_study_to_atlas.metric,
                "metric_bins": config.step3_registration.intra_study_to_atlas.metric_bins,
                "sampling_strategy": config.step3_registration.intra_study_to_atlas.sampling_strategy,
                "sampling_percentage": config.step3_registration.intra_study_to_atlas.sampling_percentage,
                "number_of_iterations": config.step3_registration.intra_study_to_atlas.number_of_iterations,
                "shrink_factors": config.step3_registration.intra_study_to_atlas.shrink_factors,
                "smoothing_sigmas": config.step3_registration.intra_study_to_atlas.smoothing_sigmas,
                "convergence_threshold": config.step3_registration.intra_study_to_atlas.convergence_threshold,
                "convergence_window_size": config.step3_registration.intra_study_to_atlas.convergence_window_size,
                "interpolation": config.step3_registration.intra_study_to_atlas.interpolation,
                "engine": config.step3_registration.intra_study_to_atlas.engine or DEFAULT_REGISTRATION_ENGINE,
            }
            # Note: reference_modality will be set after step 3a completes
            self.intra_study_to_atlas_registrator = None  # Will be initialized with reference_modality later
            self.intra_study_to_atlas_config = intra_study_to_atlas_config
            self.logger.info(f"Intra-study to atlas registration will be initialized after reference selection")
        else:
            raise ValueError(
                f"Unknown intra-study to atlas method: {intra_study_to_atlas_method}. Must be None or 'ants'"
            )

        # Step 4: Skull stripping (brain extraction)
        skull_strip_method = config.step4_skull_stripping.skull_stripping.method
        if skull_strip_method is None:
            self.skull_stripper = None
            self.logger.info("Skull stripping disabled")
        elif skull_strip_method == "hdbet":
            skull_strip_config_dict = {
                "mode": config.step4_skull_stripping.skull_stripping.hdbet_mode,
                "device": config.step4_skull_stripping.skull_stripping.hdbet_device,
                "do_tta": config.step4_skull_stripping.skull_stripping.hdbet_do_tta,
                "fill_value": config.step4_skull_stripping.skull_stripping.fill_value,
            }
            self.skull_stripper = HDBetSkullStripper(
                config=skull_strip_config_dict,
                verbose=verbose
            )
            self.logger.info(f"Initialized skull stripper: {skull_strip_method}")
        elif skull_strip_method == "synthstrip":
            skull_strip_config_dict = {
                "border": config.step4_skull_stripping.skull_stripping.synthstrip_border,
                "device": config.step4_skull_stripping.skull_stripping.synthstrip_device,
                "fill_value": config.step4_skull_stripping.skull_stripping.fill_value,
            }
            self.skull_stripper = SynthStripSkullStripper(
                config=skull_strip_config_dict,
                verbose=verbose
            )
            self.logger.info(f"Initialized skull stripper: {skull_strip_method}")
        else:
            raise ValueError(
                f"Unknown skull stripping method: {skull_strip_method}. "
                f"Must be None, 'hdbet', or 'synthstrip'"
            )

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
        """Run preprocessing pipeline for a single patient.

        Args:
            patient_id: Patient ID to process

        Raises:
            FileNotFoundError: If patient directory not found
            RuntimeError: If processing fails
        """
        self.logger.info(f"{'='*80}")
        self.logger.info(f"Processing patient: {patient_id}")
        self.logger.info(f"Mode: {self.config.mode}, Overwrite: {self.config.overwrite}")
        self.logger.info(f"{'='*80}")

        # Get study directories
        study_dirs = self._get_study_directories(patient_id)

        total_processed = 0
        total_skipped = 0
        total_errors = 0

        # Process each study
        for study_idx, study_dir in enumerate(study_dirs, 1):
            self.logger.info(f"\n[Study {study_idx}/{len(study_dirs)}] {study_dir.name}")

            # Process each modality
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

                    # Determine total steps (Step 0: convert + reorient + background, Step 1: bias field, Step 2: resampling)
                    step0_substeps = 2 if self.background_remover is None else 3
                    total_steps = step0_substeps + (1 if self.bias_corrector is not None else 0) + (1 if self.resampler is not None else 0)

                    # Step 0.1: NRRD -> NIfTI
                    self.logger.info(f"    [1/{total_steps}] Converting NRRD to NIfTI...")
                    self.converter.execute(
                        input_file,
                        paths["nifti"],
                        allow_overwrite=self.config.overwrite
                    )

                    if self.config.step0_data_harmonization.save_visualization:
                        self.converter.visualize(
                            input_file,
                            paths["nifti"],
                            paths["viz_convert"]
                        )

                    # Step 0.2: Reorient (in-place)
                    self.logger.info(
                        f"    [2/{total_steps}] Reorienting to {self.config.step0_data_harmonization.reorient_to}..."
                    )
                    # Create temp path for reorientation
                    temp_reoriented = paths["reoriented"].parent / f"_temp_{modality}_reoriented.nii.gz"

                    self.reorienter.execute(
                        paths["nifti"],
                        temp_reoriented,
                        allow_overwrite=True
                    )

                    if self.config.step0_data_harmonization.save_visualization:
                        self.reorienter.visualize(
                            paths["nifti"],
                            temp_reoriented,
                            paths["viz_reorient"]
                        )

                    # Replace original with reoriented
                    temp_reoriented.replace(paths["nifti"])

                    # Step 0.3: Background removal (in-place) - only if enabled
                    if self.background_remover is not None:
                        self.logger.info(f"    [3/{total_steps}] Removing background...")
                        temp_masked = paths["masked"].parent / f"_temp_{modality}_masked.nii.gz"

                        self.background_remover.execute(
                            paths["nifti"],
                            temp_masked,
                            allow_overwrite=True
                        )

                        if self.config.step0_data_harmonization.save_visualization:
                            self.background_remover.visualize(
                                paths["nifti"],
                                temp_masked,
                                paths["viz_background"]
                            )

                        # Replace with masked version
                        temp_masked.replace(paths["nifti"])
                    else:
                        self.logger.info("    [Background removal skipped - method is None]")

                    # Step 1: Bias field correction (in-place) - only if enabled
                    if self.bias_corrector is not None:
                        step_num = step0_substeps + 1
                        self.logger.info(f"    [{step_num}/{total_steps}] Applying N4 bias field correction...")
                        temp_corrected = paths["bias_corrected"].parent / f"_temp_{modality}_bias_corrected.nii.gz"

                        # Determine where to save bias field artifact
                        if self.config.step1_bias_field_correction.save_artifact:
                            bias_field_path = paths["bias_field"]
                            self.logger.debug(f"    Saving bias field artifact to: {bias_field_path}")
                        else:
                            # Use temporary file that will be deleted
                            import tempfile
                            temp_dir = Path(tempfile.gettempdir())
                            bias_field_path = temp_dir / f"_temp_{modality}_bias_field.nii.gz"
                            self.logger.debug("    Bias field artifact will not be saved (save_artifact=False)")

                        # Execute bias correction and get results
                        result = self.bias_corrector.execute(
                            paths["nifti"],
                            temp_corrected,
                            allow_overwrite=True,
                            bias_field_output_path=bias_field_path
                        )

                        if self.config.step1_bias_field_correction.save_visualization:
                            self.bias_corrector.visualize(
                                paths["nifti"],
                                temp_corrected,
                                paths["viz_bias_field"],
                                bias_field_path=result["bias_field_path"],
                                convergence_data=result["convergence_data"]
                            )

                        # Replace with bias-corrected version
                        temp_corrected.replace(paths["nifti"])

                        # Clean up temporary bias field if not saving
                        if not self.config.step1_bias_field_correction.save_artifact:
                            if result["bias_field_path"].exists():
                                result["bias_field_path"].unlink()
                                self.logger.debug("    Temporary bias field artifact deleted")
                    else:
                        self.logger.info("    [Bias field correction skipped - method is None]")

                    # Step 2a: Normalization (before resampling) - only if enabled
                    norm_result = None
                    temp_normalized = None
                    if self.normalizer is not None:
                        step_num = total_steps - 1 if self.resampler is not None else total_steps
                        self.logger.info(f"    [{step_num}/{total_steps}] Applying intensity normalization...")
                        temp_normalized = paths["resampled"].parent / f"_temp_{modality}_normalized.nii.gz"

                        # Execute normalization and get results
                        norm_result = self.normalizer.execute(
                            paths["nifti"],
                            temp_normalized,
                            allow_overwrite=True
                        )

                        # Note: Visualization will be done after resampling (if both are enabled)
                        # to show original → normalized → resampled in one figure

                    # Step 2b: Resampling (in-place) - only if enabled
                    if self.resampler is not None:
                        step_num = total_steps
                        self.logger.info(f"    [{step_num}/{total_steps}] Resampling to isotropic resolution...")
                        temp_resampled = paths["resampled"].parent / f"_temp_{modality}_resampled.nii.gz"

                        # Use normalized image if normalization was applied, otherwise use original
                        input_for_resampling = temp_normalized if temp_normalized is not None else paths["nifti"]

                        # Execute resampling and get results
                        resample_result = self.resampler.execute(
                            input_for_resampling,
                            temp_resampled,
                            allow_overwrite=True
                        )

                        # Generate visualization
                        if self.config.step2_resampling.save_visualization:
                            if self.normalizer is not None:
                                # Combined visualization: original → normalized → resampled
                                self._visualize_normalization_and_resampling(
                                    original_path=paths["nifti"],
                                    normalized_path=temp_normalized,
                                    resampled_path=temp_resampled,
                                    output_path=paths["viz_resampling"],
                                    norm_result=norm_result,
                                    resample_result=resample_result
                                )
                            else:
                                # Standard visualization: original → resampled (with histogram)
                                self.resampler.visualize(
                                    paths["nifti"],
                                    temp_resampled,
                                    paths["viz_resampling"],
                                    original_spacing=resample_result["original_spacing"],
                                    target_spacing=resample_result["target_spacing"],
                                    original_shape=resample_result["original_shape"],
                                    resampled_shape=resample_result["resampled_shape"]
                                )

                        # Replace with resampled version
                        temp_resampled.replace(paths["nifti"])

                        # Clean up temporary normalized file if it exists
                        if temp_normalized is not None and temp_normalized.exists():
                            temp_normalized.unlink()
                            self.logger.debug("    Temporary normalized file deleted")
                    else:
                        if self.normalizer is not None:
                            # Only normalization, no resampling
                            # Replace with normalized version
                            temp_normalized.replace(paths["nifti"])
                            self.logger.info("    [Resampling skipped - method is None]")
                        else:
                            self.logger.info("    [Normalization and resampling skipped - methods are None]")

                    self.logger.info(f"    Successfully processed {modality}")
                    total_processed += 1

                except FileExistsError as e:
                    # Re-raise overwrite errors (halt execution)
                    raise
                except Exception as e:
                    self.logger.error(f"   [Error] Processing {modality}: {e}")
                    total_errors += 1
                    # Continue with next modality

            # Step 3a: Intra-study multi-modal coregistration to reference
            intra_study_transforms = {}  # Store transforms for step 3b
            if self.intra_study_to_ref_registrator is not None:
                self.logger.info(f"\n  [Step 3a] Intra-study multi-modal coregistration to reference")
                try:
                    # Determine study output directory based on mode
                    if self.config.mode == "test":
                        study_output_dir = Path(self.config.output_root) / patient_id / study_dir.name
                        artifacts_base = Path(self.config.preprocessing_artifacts_path) / patient_id / study_dir.name
                        viz_base = Path(self.config.viz_root) / patient_id / study_dir.name
                    else:
                        study_output_dir = study_dir
                        artifacts_base = Path(self.config.preprocessing_artifacts_path) / patient_id / study_dir.name
                        viz_base = Path(self.config.viz_root) / patient_id / study_dir.name

                    # Execute intra-study to reference registration
                    reg_result = self.intra_study_to_ref_registrator.execute(
                        study_dir=study_output_dir,
                        artifacts_dir=artifacts_base,
                        modalities=self.config.modalities
                    )

                    # Store selected reference modality and transforms for step 3b
                    self.selected_reference_modality = reg_result["reference_modality"]
                    intra_study_transforms = reg_result["transforms"]
                    self.logger.info(f"  Reference modality: {self.selected_reference_modality}")

                    # Generate visualizations if enabled
                    if self.config.step3_registration.save_visualization:
                        reference_path = study_output_dir / f"{self.selected_reference_modality}.nii.gz"

                        for modality in reg_result["registered_modalities"]:
                            # Note: The file has already been replaced with registered version
                            moving_path = study_output_dir / f"{modality}.nii.gz"  # Now contains registered version
                            registered_path = moving_path  # Same (already replaced)
                            transform_path = reg_result["transforms"].get(modality)

                            viz_output = viz_base / f"step3a_intra_study_to_ref_{modality}_to_{self.selected_reference_modality}.png"

                            try:
                                self.intra_study_to_ref_registrator.visualize(
                                    reference_path=reference_path,
                                    moving_path=moving_path,  # Actually the registered version
                                    registered_path=registered_path,
                                    output_path=viz_output,
                                    modality=modality,
                                    transform_path=transform_path
                                )
                            except Exception as viz_error:
                                self.logger.warning(f"  Failed to generate visualization for {modality}: {viz_error}")

                    self.logger.info(
                        f"  Successfully registered {len(reg_result['registered_modalities'])} modalities to reference"
                    )

                except Exception as e:
                    self.logger.error(f"  [Error] Intra-study to reference registration failed: {e}")
                    total_errors += 1
                    # Continue with next study
            else:
                self.logger.info("  [Step 3a: Intra-study to reference registration skipped - method is None]")

            # Step 3b: Intra-study to atlas registration (register reference to atlas, propagate transforms)
            if self.intra_study_to_atlas_config is not None and self.selected_reference_modality is not None:
                self.logger.info(f"\n  [Step 3b] Intra-study to atlas registration")
                try:
                    # Initialize atlas registrator with reference modality from step 3a
                    from mengrowth.preprocessing.src.registration.factory import create_intra_study_to_atlas

                    atlas_registrator = create_intra_study_to_atlas(
                        config=self.intra_study_to_atlas_config,
                        reference_modality=self.selected_reference_modality,
                        verbose=self.verbose
                    )

                    # Determine study output directory based on mode
                    if self.config.mode == "test":
                        study_output_dir = Path(self.config.output_root) / patient_id / study_dir.name
                        artifacts_base = Path(self.config.preprocessing_artifacts_path) / patient_id / study_dir.name
                        viz_base = Path(self.config.viz_root) / patient_id / study_dir.name
                    else:
                        study_output_dir = study_dir
                        artifacts_base = Path(self.config.preprocessing_artifacts_path) / patient_id / study_dir.name
                        viz_base = Path(self.config.viz_root) / patient_id / study_dir.name

                    # Execute atlas registration
                    atlas_result = atlas_registrator.execute(
                        study_dir=study_output_dir,
                        artifacts_dir=artifacts_base,
                        modalities=self.config.modalities,
                        intra_study_transforms=intra_study_transforms
                    )

                    self.logger.info(f"  Reference registered to atlas: {atlas_result['atlas_path']}")

                    # Generate visualizations if enabled
                    if self.config.step3_registration.save_visualization:
                        atlas_path = Path(self.intra_study_to_atlas_config["atlas_path"])
                        reference_path = study_output_dir / f"{self.selected_reference_modality}.nii.gz"

                        # Visualize reference to atlas alignment
                        viz_output_ref = viz_base / f"step3b_atlas_registration_reference_{self.selected_reference_modality}.png"
                        try:
                            atlas_registrator.visualize_reference_to_atlas(
                                atlas_path=atlas_path,
                                reference_path=reference_path,
                                output_path=viz_output_ref,
                                ref_to_atlas_transform=atlas_result["ref_to_atlas_transform"]
                            )
                        except Exception as viz_error:
                            self.logger.warning(f"  Failed to generate reference→atlas visualization: {viz_error}")

                        # Visualize each modality in atlas space
                        for modality in atlas_result["registered_modalities"]:
                            modality_path = study_output_dir / f"{modality}.nii.gz"
                            viz_output = viz_base / f"step3b_atlas_space_{modality}.png"

                            try:
                                atlas_registrator.visualize_modality_in_atlas_space(
                                    atlas_path=atlas_path,
                                    modality_path=modality_path,
                                    output_path=viz_output,
                                    modality=modality
                                )
                            except Exception as viz_error:
                                self.logger.warning(f"  Failed to generate atlas space visualization for {modality}: {viz_error}")

                    self.logger.info(
                        f"  Successfully registered {len(atlas_result['registered_modalities'])} modalities to atlas space"
                    )

                except Exception as e:
                    self.logger.error(f"  [Error] Intra-study to atlas registration failed: {e}")
                    total_errors += 1
                    # Continue with next study
            elif self.intra_study_to_atlas_config is not None and self.selected_reference_modality is None:
                self.logger.warning("  [Step 3b: Atlas registration skipped - no reference modality available from step 3a]")
            else:
                self.logger.info("  [Step 3b: Intra-study to atlas registration skipped - method is None]")

            # Step 4: Skull stripping (brain extraction)
            if self.skull_stripper is not None:
                self.logger.info(f"\n  [Step 4] Skull stripping (brain extraction)")
                try:
                    # Determine output directories based on mode
                    if self.config.mode == "test":
                        study_output_dir = Path(self.config.output_root) / patient_id / study_dir.name
                        artifacts_base = Path(self.config.preprocessing_artifacts_path) / patient_id / study_dir.name
                        viz_base = Path(self.config.viz_root) / patient_id / study_dir.name
                    else:
                        study_output_dir = study_dir
                        artifacts_base = Path(self.config.preprocessing_artifacts_path) / patient_id / study_dir.name
                        viz_base = Path(self.config.viz_root) / patient_id / study_dir.name

                    # Process each modality
                    for modality in self.config.modalities:
                        modality_path = study_output_dir / f"{modality}.nii.gz"

                        # Skip if file doesn't exist
                        if not modality_path.exists():
                            self.logger.warning(f"  Skipping {modality} - file not found")
                            continue

                        self.logger.info(f"  Processing {modality}...")

                        # Create temporary output path
                        temp_skull_stripped = modality_path.parent / f"_temp_{modality}_skull_stripped.nii.gz"

                        # Determine mask path
                        if self.config.step4_skull_stripping.save_mask:
                            mask_path = artifacts_base / f"{modality}_brain_mask.nii.gz"
                        else:
                            import tempfile
                            temp_dir = Path(tempfile.gettempdir())
                            mask_path = temp_dir / f"_temp_{modality}_brain_mask.nii.gz"

                        # Execute skull stripping
                        result = self.skull_stripper.execute(
                            modality_path,
                            temp_skull_stripped,
                            mask_path=mask_path,
                            allow_overwrite=True
                        )

                        # Generate visualization if enabled
                        if self.config.step4_skull_stripping.save_visualization:
                            viz_output = viz_base / f"step4_skull_stripping_{modality}.png"
                            self.skull_stripper.visualize(
                                modality_path,
                                temp_skull_stripped,
                                viz_output,
                                **result
                            )

                        # Replace original with skull-stripped version (in-place)
                        temp_skull_stripped.replace(modality_path)

                        # Clean up temporary mask if not saving
                        if not self.config.step4_skull_stripping.save_mask and mask_path.exists():
                            mask_path.unlink()

                        self.logger.info(
                            f"  {modality}: brain_volume={result['brain_volume_mm3']:.1f} mm³, "
                            f"coverage={result['brain_coverage_percent']:.1f}%"
                        )

                    self.logger.info(f"  Step 4 completed successfully")

                except Exception as e:
                    self.logger.error(f"  [Error] Skull stripping failed: {e}")
                    total_errors += 1
            else:
                self.logger.info("  [Step 4: Skull stripping skipped - method is None]")

        # Summary
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"Patient {patient_id} processing complete:")
        self.logger.info(f"  Processed: {total_processed}")
        self.logger.info(f"  Skipped:   {total_skipped}")
        self.logger.info(f"  Errors:    {total_errors}")
        self.logger.info(f"{'='*80}\n")


def run_preprocessing(config: PreprocessingPipelineConfig, patient_id: Optional[str] = None) -> None:
    """Run preprocessing pipeline on selected patients.

    Args:
        config: Preprocessing pipeline configuration
        patient_id: Specific patient to process (overrides config.patient_selector)

    Raises:
        ValueError: If patient_selector is invalid
        FileNotFoundError: If patient directory not found
        RuntimeError: If processing fails
    """
    # Extract data harmonization config
    dh_config = config.data_harmonization

    if not dh_config.enabled:
        logger.info("Data harmonization is disabled in config - exiting")
        return

    # Override patient_id if provided
    if patient_id is not None:
        dh_config.patient_id = patient_id
        dh_config.patient_selector = "single"
        logger.info(f"Overriding config: processing single patient {patient_id}")

    # Initialize orchestrator
    orchestrator = PreprocessingOrchestrator(dh_config, verbose=True)

    # Process patients
    if dh_config.patient_selector == "single":
        orchestrator.run_patient(dh_config.patient_id)

    elif dh_config.patient_selector == "all":
        # Get all patient directories
        dataset_root = Path(dh_config.dataset_root)
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
            f"Invalid patient_selector: {dh_config.patient_selector}. Must be 'single' or 'all'"
        )
