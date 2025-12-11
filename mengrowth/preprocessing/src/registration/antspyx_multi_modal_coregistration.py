"""Multi-modal intra-study coregistration using AntsPyX.

This module implements rigid registration of multiple MRI modalities
to a reference sequence within the same study/time-point using the
AntsPyX library instead of nipype.
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple
import logging
import time
import json
from datetime import datetime

import nibabel as nib
import matplotlib.pyplot as plt

from mengrowth.preprocessing.src.registration.base import BaseRegistrator
from mengrowth.preprocessing.src.registration.stdout_capture import capture_stdout
from mengrowth.preprocessing.src.registration.diagnostic_parser import (
    parse_ants_diagnostic_output,
    extract_transform_types
)

logger = logging.getLogger(__name__)


class AntsPyXMultiModalCoregistration(BaseRegistrator):
    """Intra-study rigid multi-modal coregistration using AntsPyX.

    This is the AntsPyX-based implementation, functionally equivalent
    to MultiModalCoregistration but using antspyx library instead of nipype.

    For each study, registers all modalities (T1n, T2w, T2-FLAIR, etc.)
    to a reference modality (typically T1c) using rigid registration.

    The reference modality is selected based on a priority list, allowing
    graceful handling of missing modalities.

    Attributes:
        config: Registration configuration parameters
        verbose: Whether to enable verbose logging
    """

    def __init__(self, config: Dict[str, Any], verbose: bool = False) -> None:
        """Initialize multi-modal coregistration step with AntsPyX.

        Args:
            config: Configuration dictionary from RegistrationConfig
            verbose: Enable verbose logging
        """
        super().__init__(config=config, verbose=verbose)
        self.logger = logging.getLogger(__name__)
        self.save_detailed_info = config.get("save_detailed_registration_info", False)

        # Validate antspyx is available
        try:
            import ants
        except ImportError:
            raise ImportError(
                "AntsPyX is required for this registration engine. "
                "Install with: pip install antspyx"
            )

    def execute(
        self,
        study_dir: Path,
        artifacts_dir: Path,
        modalities: List[str],
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Execute multi-modal coregistration for a single study.

        Args:
            study_dir: Directory containing modality files (*.nii.gz)
            artifacts_dir: Directory to save transform artifacts
            modalities: List of expected modalities
            **kwargs: Additional parameters (unused)

        Returns:
            Dictionary with:
            - reference_modality: Selected reference modality
            - registered_modalities: List of registered modalities
            - transforms: Dict mapping modality to transform path

        Raises:
            ValueError: If fewer than 2 modalities found
            RuntimeError: If registration fails
        """
        start_time = time.time()
        self.logger.info(f"[AntsPyX] Starting multi-modal coregistration in {study_dir.name}")

        # 1. Discover available modality files
        modality_files = self._discover_modality_files(study_dir, modalities)

        if len(modality_files) < 2:
            raise ValueError(
                f"Registration requires at least 2 modalities, found {len(modality_files)}: "
                f"{list(modality_files.keys())}"
            )

        # 2. Select reference modality
        reference_modality = self._select_reference_modality(
            available_modalities=list(modality_files.keys()),
            priority_str=self.config.get("reference_modality_priority", "t1c > t1n > t2f > t2w")
        )
        reference_path = modality_files[reference_modality]

        self.logger.info(
            f"Reference: {reference_modality}, "
            f"Moving: {[m for m in modality_files if m != reference_modality]}"
        )

        # 3. Create registration artifacts directory
        registration_dir = artifacts_dir / "registration"
        registration_dir.mkdir(parents=True, exist_ok=True)

        # 4. Register each non-reference modality
        transforms = {}
        registered_modalities = []

        for modality, moving_path in modality_files.items():
            if modality == reference_modality:
                continue  # Skip reference modality

            self.logger.info(f"Registering {modality} to {reference_modality}...")

            # Define transform output path
            transform_path = registration_dir / f"{modality}_to_{reference_modality}.h5"

            try:
                # Perform registration
                # _register_modality returns (registered_path, actual_transform_path)
                registered_path, actual_transform = self._register_modality(
                    fixed_path=reference_path,
                    moving_path=moving_path,
                    transform_path=transform_path,
                    study_dir=study_dir,
                    modality=modality
                )

                transforms[modality] = actual_transform
                registered_modalities.append(modality)

                self.logger.info(f"✓ {modality} registered successfully")

            except Exception as e:
                self.logger.error(f"✗ Failed to register {modality}: {e}")
                # Continue with other modalities rather than failing completely
                continue

        elapsed = time.time() - start_time
        self.logger.info(
            f"Completed registration in {elapsed:.1f}s. "
            f"Registered {len(registered_modalities)}/{len(modality_files)-1} modalities"
        )

        return {
            "reference_modality": reference_modality,
            "registered_modalities": registered_modalities,
            "transforms": transforms
        }

    def _register_modality(
        self,
        fixed_path: Path,
        moving_path: Path,
        transform_path: Path,
        study_dir: Path,
        modality: str
    ) -> Tuple[Path, Path]:
        """Register a single modality to the reference using AntsPyX.

        Args:
            fixed_path: Path to reference (fixed) image
            moving_path: Path to moving image
            transform_path: Path to save transform file
            study_dir: Study directory for output
            modality: Modality name

        Returns:
            Tuple of (registered_output_file_path, actual_transform_path)

        Raises:
            RuntimeError: If registration fails
        """
        import ants

        if self.verbose:
            self.logger.debug(f"[DEBUG] [AntsPyX] Registration setup for {modality}:")
            self.logger.debug(f"  Fixed image:     {fixed_path}")
            self.logger.debug(f"  Moving image:    {moving_path}")
            self.logger.debug(f"  Transform (req): {transform_path}")

        try:
            # Load images
            fixed_img = ants.image_read(str(fixed_path))
            moving_img = ants.image_read(str(moving_path))

            # Get transform types from config (can be string or list)
            transform_type_config = self.config.get("transform_type", "Rigid")

            # Normalize to list for uniform handling
            if isinstance(transform_type_config, str):
                transforms = [transform_type_config]
            else:
                transforms = transform_type_config

            # Map to AntsPyX type_of_transform
            # For ["Rigid", "Affine"], use "Affine" (includes rigid initialization)
            # For more complex cases, may need sequential calls
            if transforms == ["Rigid"]:
                type_of_transform = "Rigid"
            elif transforms == ["Rigid", "Affine"] or transforms == ["Affine"]:
                type_of_transform = "Affine"  # Includes rigid
            elif "SyN" in transforms:
                type_of_transform = "SyN"  # Includes affine and rigid
            else:
                # Default to the last transform type
                type_of_transform = transforms[-1] if transforms else "Rigid"

            # Extract parameters
            metric = self.config.get("metric", "Mattes").lower()
            metric_bins = self.config.get("metric_bins", 32)
            sampling_percentage = self.config.get("sampling_percentage", 0.2)

            # Multi-resolution parameters
            number_of_iterations_list = self.config.get(
                "number_of_iterations",
                [[1000, 500, 250]]
            )
            shrink_factors_list = self.config.get(
                "shrink_factors",
                [[4, 2, 1]]
            )
            smoothing_sigmas_list = self.config.get(
                "smoothing_sigmas",
                [[2, 1, 0]]
            )

            # For multi-stage, use parameters from the final stage
            # AntsPyX handles multi-resolution internally for composite transforms
            if len(number_of_iterations_list) > 0:
                # For Affine (which includes Rigid), use the affine parameters
                # Typically the second set of parameters if available
                if len(number_of_iterations_list) > 1 and type_of_transform in ["Affine", "SyN"]:
                    aff_iterations = tuple(number_of_iterations_list[-1])
                    aff_shrink_factors = tuple(shrink_factors_list[-1])
                    aff_smoothing_sigmas = tuple(smoothing_sigmas_list[-1])
                else:
                    aff_iterations = tuple(number_of_iterations_list[0])
                    aff_shrink_factors = tuple(shrink_factors_list[0])
                    aff_smoothing_sigmas = tuple(smoothing_sigmas_list[0])
            else:
                aff_iterations = (1000, 500, 250)
                aff_shrink_factors = (4, 2, 1)
                aff_smoothing_sigmas = (2, 1, 0)

            # Construct output prefix (without extension)
            # AntsPyX will create files like: prefix0GenericAffine.mat or prefixComposite.h5
            transform_prefix = str(transform_path.with_suffix("").with_suffix(""))

            if self.verbose:
                self.logger.debug(f"[DEBUG] [AntsPyX] Calling ants.registration with:")
                self.logger.debug(f"  transforms: {transforms}")
                self.logger.debug(f"  type_of_transform: {type_of_transform}")
                self.logger.debug(f"  aff_metric: {metric}")
                self.logger.debug(f"  aff_sampling: {metric_bins}")
                self.logger.debug(f"  aff_random_sampling_rate: {sampling_percentage}")
                self.logger.debug(f"  aff_iterations: {aff_iterations}")
                self.logger.debug(f"  aff_shrink_factors: {aff_shrink_factors}")
                self.logger.debug(f"  aff_smoothing_sigmas: {aff_smoothing_sigmas}")
                self.logger.debug(f"  outprefix: {transform_prefix}")

            # Perform registration
            write_composite = self.config.get("write_composite_transform", True)

            # Capture stdout if detailed info is enabled
            captured_stdout = None
            if self.save_detailed_info:
                with capture_stdout() as captured:
                    result = ants.registration(
                        fixed=fixed_img,
                        moving=moving_img,
                        type_of_transform=type_of_transform,
                        outprefix=transform_prefix,
                        aff_metric=metric,
                        aff_sampling=metric_bins,
                        aff_random_sampling_rate=sampling_percentage,
                        aff_iterations=aff_iterations,
                        aff_shrink_factors=aff_shrink_factors,
                        aff_smoothing_sigmas=aff_smoothing_sigmas,
                        write_composite_transform=write_composite,
                        verbose=True  # Force verbose for stdout capture
                    )
                captured_stdout = captured.getvalue()
            else:
                result = ants.registration(
                    fixed=fixed_img,
                    moving=moving_img,
                    type_of_transform=type_of_transform,
                    outprefix=transform_prefix,
                    aff_metric=metric,
                    aff_sampling=metric_bins,
                    aff_random_sampling_rate=sampling_percentage,
                    aff_iterations=aff_iterations,
                    aff_shrink_factors=aff_shrink_factors,
                    aff_smoothing_sigmas=aff_smoothing_sigmas,
                    write_composite_transform=write_composite,
                    verbose=self.verbose
                )

            if self.verbose:
                self.logger.debug(f"[DEBUG] [AntsPyX] Registration completed")

            # Extract warped image
            warped_img = result['warpedmovout']

            # Get transform file paths
            fwd_transforms = result['fwdtransforms']  # List of paths

            if self.verbose:
                self.logger.debug(f"[DEBUG] [AntsPyX] Forward transforms: {fwd_transforms}")

            # Determine actual transform path
            # If write_composite_transform=True, look for Composite.h5
            # Otherwise, use the first transform in the list
            if write_composite:
                actual_transform_path = Path(transform_prefix + "Composite.h5")
                if not actual_transform_path.exists():
                    # Fallback to first transform
                    actual_transform_path = Path(fwd_transforms[0]) if fwd_transforms else None
            else:
                actual_transform_path = Path(fwd_transforms[0]) if fwd_transforms else None

            if not actual_transform_path or not actual_transform_path.exists():
                raise RuntimeError(f"Transform file not created: {actual_transform_path}")

            # Save warped image to temp location
            temp_output = study_dir / f"_temp_{modality}_registered.nii.gz"
            ants.image_write(warped_img, str(temp_output))

            if not temp_output.exists():
                raise RuntimeError(f"Registered image not created: {temp_output}")

            if self.verbose:
                self.logger.debug(f"[DEBUG] [AntsPyX] Outputs verified successfully")
                self.logger.debug(f"[DEBUG] [AntsPyX] Replacing {moving_path.name} with registered version")

            # Replace original with registered version (in-place)
            final_output = moving_path
            temp_output.replace(final_output)

            if self.verbose:
                self.logger.debug(f"[DEBUG] [AntsPyX] File replacement complete")

            # Save detailed registration info if enabled
            if self.save_detailed_info and captured_stdout:
                self._save_detailed_registration_info(
                    stdout=captured_stdout,
                    fixed_path=fixed_path,
                    moving_path=moving_path,
                    transform_path=actual_transform_path,
                    config_params={
                        "transforms": transforms,
                        "type_of_transform": type_of_transform,
                        "metric": metric,
                        "metric_bins": metric_bins,
                        "sampling_strategy": self.config.get("sampling_strategy", "Random"),
                        "sampling_percentage": sampling_percentage,
                        "number_of_iterations": number_of_iterations_list,
                        "shrink_factors": shrink_factors_list,
                        "smoothing_sigmas": smoothing_sigmas_list,
                        "convergence_threshold": self.config.get("convergence_threshold", 1e-6),
                        "convergence_window_size": self.config.get("convergence_window_size", 10),
                        "write_composite_transform": write_composite,
                        "interpolation": self.config.get("interpolation", "Linear")
                    },
                    registration_type="intra_study_to_reference"
                )

            return final_output, actual_transform_path

        except Exception as e:
            error_msg = f"AntsPyX registration failed for {modality}: {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def _save_detailed_registration_info(
        self,
        stdout: str,
        fixed_path: Path,
        moving_path: Path,
        transform_path: Path,
        config_params: Dict[str, Any],
        registration_type: str
    ) -> None:
        """Save detailed registration information to JSON.

        Args:
            stdout: Captured stdout from registration
            fixed_path: Path to fixed image
            moving_path: Path to moving image
            transform_path: Path to transform file
            config_params: Registration parameters used
            registration_type: "intra_study_to_reference" or "intra_study_to_atlas"
        """
        try:
            # Parse diagnostic output
            parsed_diagnostics = parse_ants_diagnostic_output(stdout)
            transform_types = extract_transform_types(stdout)

            # Build JSON structure
            info = {
                "metadata": {
                    "registration_type": registration_type,
                    "timestamp": datetime.now().isoformat(),
                    "engine": "antspyx",
                    "fixed_image": fixed_path.name,
                    "moving_image": moving_path.name,
                    "output_image": moving_path.name,  # In-place replacement
                    "transform_file": transform_path.name,
                    "success": True,
                    "error_message": None
                },
                "parameters": config_params,
                "timing": {
                    "total_elapsed_time_seconds": parsed_diagnostics.get("total_elapsed_time_seconds"),
                    "stages": [
                        {
                            "stage_index": stage["stage_index"],
                            "transform_type": transform_types[stage["stage_index"]] if stage["stage_index"] < len(transform_types) else "Unknown",
                            "elapsed_time_seconds": stage.get("elapsed_time_seconds")
                        }
                        for stage in parsed_diagnostics.get("stages", [])
                    ]
                },
                "convergence": {
                    "stages": parsed_diagnostics.get("stages", [])
                },
                "stdout_capture": {
                    "full_output": stdout,
                    "command_lines_ok": parsed_diagnostics.get("command_lines_ok", False)
                }
            }

        except Exception as parse_error:
            self.logger.warning(f"Failed to parse diagnostic output: {parse_error}")
            # Fall back to minimal JSON with raw stdout
            info = {
                "metadata": {
                    "registration_type": registration_type,
                    "timestamp": datetime.now().isoformat(),
                    "engine": "antspyx",
                    "fixed_image": fixed_path.name,
                    "moving_image": moving_path.name,
                    "error_message": f"Parsing failed: {str(parse_error)}"
                },
                "stdout_capture": {
                    "full_output": stdout
                }
            }

        try:
            # Determine output path
            # Extract modality names from paths
            fixed_modality = fixed_path.stem.replace(".nii", "")
            moving_modality = moving_path.stem.replace(".nii", "")

            # Create detailed_info directory
            detailed_info_dir = transform_path.parent / "detailed_info"
            detailed_info_dir.mkdir(parents=True, exist_ok=True)

            # Construct filename
            json_filename = f"{moving_modality}_to_{fixed_modality}_info.json"
            json_path = detailed_info_dir / json_filename

            # Write JSON
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(info, f, indent=2, default=str)  # default=str handles inf, datetime

            self.logger.info(f"Saved detailed registration info: {json_path.name}")

        except Exception as write_error:
            self.logger.error(f"Failed to write detailed registration info: {write_error}")
            # Don't raise - registration succeeded, just diagnostic save failed

    def visualize(
        self,
        reference_path: Path,
        moving_path: Path,
        registered_path: Path,
        output_path: Path,
        **kwargs: Any
    ) -> None:
        """Generate visualization comparing reference, moving, and registered images.

        Creates a 1x3 subplot showing:
        1. Reference (fixed) image
        2. Moving image (before registration)
        3. Registered image (after registration)

        Args:
            reference_path: Path to reference image
            moving_path: Path to moving image (original)
            registered_path: Path to registered image
            output_path: Path to save visualization PNG
            **kwargs: Additional parameters (transform_path, modality)
        """
        try:
            # Load images
            ref_img = nib.load(str(reference_path))
            mov_img = nib.load(str(moving_path))
            reg_img = nib.load(str(registered_path))

            ref_data = ref_img.get_fdata()
            mov_data = mov_img.get_fdata()
            reg_data = reg_img.get_fdata()

            # Select middle slice (axial view)
            slice_idx = ref_data.shape[2] // 2

            ref_slice = ref_data[:, :, slice_idx]
            mov_slice = mov_data[:, :, slice_idx]
            reg_slice = reg_data[:, :, slice_idx]

            # Create figure
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Reference image
            axes[0].imshow(ref_slice.T, cmap="gray", origin="lower")
            axes[0].set_title("Reference (Fixed)")
            axes[0].axis("off")

            # Moving image
            axes[1].imshow(mov_slice.T, cmap="gray", origin="lower")
            axes[1].set_title("Moving (Original)")
            axes[1].axis("off")

            # Registered image
            axes[2].imshow(reg_slice.T, cmap="gray", origin="lower")
            axes[2].set_title("Registered")
            axes[2].axis("off")

            # Add metadata
            modality = kwargs.get("modality", "unknown")
            transform_path = kwargs.get("transform_path", "N/A")

            fig.suptitle(
                f"Multi-Modal Coregistration (AntsPyX): {modality}\n"
                f"Transform: {transform_path.name if hasattr(transform_path, 'name') else transform_path}",
                fontsize=12
            )

            plt.tight_layout()

            # Save
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()

            self.logger.info(f"Saved visualization: {output_path.name}")

        except Exception as e:
            self.logger.error(f"Failed to generate visualization: {e}")
            raise RuntimeError(f"Visualization failed: {e}") from e
