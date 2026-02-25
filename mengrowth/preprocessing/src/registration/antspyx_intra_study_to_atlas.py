"""Register reference modality to atlas space and propagate transforms using AntsPyX.

This module implements atlas registration using the AntsPyX library instead of nipype.
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import logging
import time
import json
from datetime import datetime

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

from mengrowth.preprocessing.src.registration.base import BaseRegistrator
from mengrowth.preprocessing.src.registration.stdout_capture import capture_stdout
from mengrowth.preprocessing.src.registration.diagnostic_parser import (
    parse_ants_diagnostic_output,
    extract_transform_types,
)

logger = logging.getLogger(__name__)


class AntsPyXIntraStudyToAtlas(BaseRegistrator):
    """Register reference modality to atlas and propagate to all modalities using AntsPyX.

    This class performs two main operations:
    1. Register the reference modality (e.g., T1c) to an atlas template
    2. Apply the composed transform (M→ref→atlas) to all other modalities

    The result is that all modalities from the study are brought into atlas space.

    Attributes:
        config: Atlas registration configuration parameters
        reference_modality: The reference modality selected in step 3a
        verbose: Whether to enable verbose logging
    """

    def __init__(
        self, config: Dict[str, Any], reference_modality: str, verbose: bool = False
    ) -> None:
        """Initialize atlas registration step with AntsPyX.

        Args:
            config: Configuration dictionary from IntraStudyToAtlasConfig
            reference_modality: Reference modality name (from step 3a)
            verbose: Enable verbose logging
        """
        super().__init__(config=config, verbose=verbose)
        self.reference_modality = reference_modality
        self.logger = logging.getLogger(__name__)
        self.save_detailed_info = config.get("save_detailed_registration_info", False)
        self.use_center_of_mass_init = config.get("use_center_of_mass_init", True)
        self.validate_registration_quality = config.get(
            "validate_registration_quality", True
        )
        self.quality_warning_threshold = config.get("quality_warning_threshold", -0.3)

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
        intra_study_transforms: Dict[str, Path],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Execute atlas registration for a single study.

        Args:
            study_dir: Directory containing modality files
            artifacts_dir: Directory to save transform artifacts
            modalities: List of expected modalities
            intra_study_transforms: Dict mapping modality to M→ref transform path
            **kwargs: Additional parameters (unused)

        Returns:
            Dictionary with:
            - atlas_path: Path to atlas file used
            - reference_to_atlas_transform: Path to ref→atlas transform
            - atlas_transforms: Dict mapping modality to transform info
            - registered_modalities: List of modalities now in atlas space

        Raises:
            ValueError: If atlas file not found or reference modality missing
            RuntimeError: If registration fails
        """
        start_time = time.time()
        self.logger.info(f"[AntsPyX] Starting atlas registration for {study_dir.name}")

        # Validate atlas file exists
        atlas_path = Path(self.config.get("atlas_path", ""))
        if not atlas_path.exists():
            raise ValueError(f"Atlas file not found: {atlas_path}")

        if self.verbose:
            self.logger.debug(f"[DEBUG] [AntsPyX] Atlas path: {atlas_path}")
            self.logger.debug(
                f"[DEBUG] [AntsPyX] Reference modality: {self.reference_modality}"
            )

        # 1. Register reference modality to atlas
        reference_file = study_dir / f"{self.reference_modality}.nii.gz"
        if not reference_file.exists():
            raise ValueError(f"Reference modality file not found: {reference_file}")

        registration_dir = artifacts_dir / "registration"
        registration_dir.mkdir(parents=True, exist_ok=True)

        ref_to_atlas_transform_path = (
            registration_dir / f"{self.reference_modality}_to_atlasComposite.h5"
        )

        try:
            self.logger.info(f"Registering {self.reference_modality} to atlas...")
            ref_registered_path, ref_to_atlas_transform, quality_metrics = (
                self._register_reference_to_atlas(
                    reference_path=reference_file,
                    atlas_path=atlas_path,
                    transform_path=ref_to_atlas_transform_path,
                    study_dir=study_dir,
                )
            )
            self.logger.info("✓ Reference registered to atlas")

        except Exception as e:
            self.logger.error(f"✗ Failed to register reference to atlas: {e}")
            raise RuntimeError(f"Atlas registration failed: {e}") from e

        quality_metrics = quality_metrics or {}

        # 2. Apply transforms to all other modalities
        atlas_transforms = {}
        registered_modalities = [
            self.reference_modality
        ]  # Reference is already registered

        for modality in modalities:
            if modality == self.reference_modality:
                # Reference already registered above
                atlas_transforms[modality] = ref_to_atlas_transform
                continue

            modality_file = study_dir / f"{modality}.nii.gz"
            if not modality_file.exists():
                if self.verbose:
                    self.logger.debug(
                        f"[DEBUG] [AntsPyX] Modality {modality} not found, skipping"
                    )
                continue

            # Get the M→ref transform from step 3a
            m_to_ref_transform = intra_study_transforms.get(modality)
            if not m_to_ref_transform or not m_to_ref_transform.exists():
                self.logger.warning(f"Transform for {modality} not found, skipping")
                continue

            self.logger.info(f"Transforming {modality} to atlas space...")

            try:
                # Apply both transforms: M→ref→atlas
                self._apply_transforms_to_modality(
                    modality_path=modality_file,
                    atlas_path=atlas_path,
                    m_to_ref_transform=m_to_ref_transform,
                    ref_to_atlas_transform=ref_to_atlas_transform,
                    study_dir=study_dir,
                    modality=modality,
                )

                atlas_transforms[modality] = {
                    "m_to_ref": m_to_ref_transform,
                    "ref_to_atlas": ref_to_atlas_transform,
                }
                registered_modalities.append(modality)
                self.logger.info(f"✓ {modality} transformed to atlas space")

            except Exception as e:
                self.logger.error(f"✗ Failed to transform {modality}: {e}")
                continue

        elapsed = time.time() - start_time
        self.logger.info(
            f"Atlas registration completed in {elapsed:.1f}s. "
            f"Registered {len(registered_modalities)}/{len(modalities)} modalities"
        )

        return {
            "atlas_path": atlas_path,
            "ref_to_atlas_transform": ref_to_atlas_transform,
            "atlas_transforms": atlas_transforms,
            "registered_modalities": registered_modalities,
            "quality_metrics": quality_metrics,
        }

    def _register_reference_to_atlas(
        self,
        reference_path: Path,
        atlas_path: Path,
        transform_path: Path,
        study_dir: Path,
    ) -> Tuple[Path, Path, Optional[Dict[str, float]]]:
        """Register reference modality to atlas using AntsPyX.

        Handles multi-stage registration (e.g., Rigid + Affine).

        Args:
            reference_path: Path to reference modality (moving image)
            atlas_path: Path to atlas template (fixed image)
            transform_path: Base path for transform output (without "Composite.h5")
            study_dir: Study directory for output

        Returns:
            Tuple of (registered_reference_path, actual_transform_path, quality_metrics)
            where quality_metrics is None if validation is disabled

        Raises:
            RuntimeError: If registration fails
        """
        import ants

        if self.verbose:
            self.logger.debug("[DEBUG] [AntsPyX] Reference→Atlas registration setup:")
            self.logger.debug(f"  Fixed (atlas):   {atlas_path}")
            self.logger.debug(f"  Moving (ref):    {reference_path}")

        try:
            # Load images
            atlas_img = ants.image_read(str(atlas_path))
            reference_img = ants.image_read(str(reference_path))

            # Validate spacing — ITK crashes with an opaque error on zero-valued spacing
            for label, img, path in [
                ("atlas", atlas_img, atlas_path),
                ("reference", reference_img, reference_path),
            ]:
                if any(s <= 0 for s in img.spacing):
                    raise RuntimeError(
                        f"{label} image has invalid spacing {img.spacing}: {path}. "
                        "Zero or negative spacing is not supported by ITK."
                    )

            # Get transform types from config
            transforms = self.config.get("transforms", ["Rigid", "Affine"])

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
                type_of_transform = transforms[-1] if transforms else "Affine"

            # Extract parameters
            metric = self.config.get("metric", "Mattes").lower()
            metric_bins = self.config.get("metric_bins", 32)
            sampling_percentage = self.config.get("sampling_percentage", 0.2)

            # Multi-resolution parameters
            number_of_iterations_list = self.config.get(
                "number_of_iterations", [[1000, 500, 250], [500, 250, 100]]
            )
            shrink_factors_list = self.config.get(
                "shrink_factors", [[4, 2, 1], [2, 1, 1]]
            )
            smoothing_sigmas_list = self.config.get(
                "smoothing_sigmas", [[2, 1, 0], [1, 0, 0]]
            )

            # For multi-stage, use parameters from the final stage
            # AntsPyX handles multi-resolution internally for composite transforms
            if len(number_of_iterations_list) > 0:
                # For Affine (which includes Rigid), use the affine parameters
                # Typically the second set of parameters
                if len(number_of_iterations_list) > 1:
                    aff_iterations = tuple(number_of_iterations_list[1])
                    aff_shrink_factors = tuple(shrink_factors_list[1])
                    aff_smoothing_sigmas = tuple(smoothing_sigmas_list[1])
                else:
                    aff_iterations = tuple(number_of_iterations_list[0])
                    aff_shrink_factors = tuple(shrink_factors_list[0])
                    aff_smoothing_sigmas = tuple(smoothing_sigmas_list[0])
            else:
                aff_iterations = (1000, 500, 250)
                aff_shrink_factors = (4, 2, 1)
                aff_smoothing_sigmas = (2, 1, 0)

            transform_prefix = str(transform_path.with_suffix("").with_suffix(""))

            if self.verbose:
                self.logger.debug("[DEBUG] [AntsPyX] Atlas registration parameters:")
                self.logger.debug(f"  type_of_transform: {type_of_transform}")
                self.logger.debug(f"  aff_metric: {metric}")
                self.logger.debug(f"  aff_iterations: {aff_iterations}")

            # Compute center-of-mass alignment as initial transform if enabled
            # NOTE: ants.registration() expects initial_transform as a file path string,
            # not an ANTsTransform object. We write the transform to a temp file.
            initial_tx = None
            if self.use_center_of_mass_init:
                self.logger.info("  Computing center-of-mass initialization...")
                try:
                    atlas_com = ants.get_center_of_mass(atlas_img)
                    ref_com = ants.get_center_of_mass(reference_img)

                    # Create translation to align centers
                    translation = [a - r for a, r in zip(atlas_com, ref_com)]

                    com_tx = ants.create_ants_transform(
                        transform_type="Euler3DTransform",
                        center=ref_com,
                        translation=translation,
                    )
                    # Write to temp file — ants.registration() needs a file path, not an object
                    com_tx_path = (
                        study_dir
                        / f"_temp_{self.reference_modality}_atlas_com_init.mat"
                    )
                    ants.write_transform(com_tx, str(com_tx_path))
                    initial_tx = str(com_tx_path)
                    if self.verbose:
                        self.logger.debug(
                            f"[DEBUG] [AntsPyX] COM translation: {translation}"
                        )
                except Exception as com_error:
                    self.logger.warning(
                        f"  COM initialization failed, proceeding without: {com_error}"
                    )
                    initial_tx = None

            # Perform registration
            # Capture stdout if detailed info is enabled
            captured_stdout = None
            if self.save_detailed_info:
                with capture_stdout() as captured:
                    result = ants.registration(
                        fixed=atlas_img,
                        moving=reference_img,
                        type_of_transform=type_of_transform,
                        initial_transform=initial_tx,
                        outprefix=transform_prefix,
                        aff_metric=metric,
                        aff_sampling=metric_bins,
                        aff_random_sampling_rate=sampling_percentage,
                        aff_iterations=aff_iterations,
                        aff_shrink_factors=aff_shrink_factors,
                        aff_smoothing_sigmas=aff_smoothing_sigmas,
                        write_composite_transform=True,
                        verbose=True,  # Force verbose for stdout capture
                    )
                captured_stdout = captured.getvalue()
            else:
                result = ants.registration(
                    fixed=atlas_img,
                    moving=reference_img,
                    type_of_transform=type_of_transform,
                    initial_transform=initial_tx,
                    outprefix=transform_prefix,
                    aff_metric=metric,
                    aff_sampling=metric_bins,
                    aff_random_sampling_rate=sampling_percentage,
                    aff_iterations=aff_iterations,
                    aff_shrink_factors=aff_shrink_factors,
                    aff_smoothing_sigmas=aff_smoothing_sigmas,
                    write_composite_transform=True,
                    verbose=self.verbose,
                )

            if self.verbose:
                self.logger.debug("[DEBUG] [AntsPyX] Registration completed")

            # Extract outputs
            warped_img = result["warpedmovout"]
            fwd_transforms = result["fwdtransforms"]

            if self.verbose:
                self.logger.debug(
                    f"[DEBUG] [AntsPyX] Forward transforms: {fwd_transforms}"
                )

            # Determine actual transform path
            actual_transform_path = Path(transform_prefix + "Composite.h5")
            if not actual_transform_path.exists() and fwd_transforms:
                actual_transform_path = Path(fwd_transforms[0])

            if not actual_transform_path.exists():
                raise RuntimeError(f"Transform not created: {actual_transform_path}")

            # Compute and log registration quality if enabled
            quality_metrics = None
            if self.validate_registration_quality:
                quality_metrics = self._compute_registration_quality(
                    atlas_img, warped_img
                )
                corr_sim = quality_metrics.get("correlation_similarity")
                corr_dissim = quality_metrics.get("correlation_dissimilarity")
                self.logger.info(
                    f"  Registration quality: MI={quality_metrics['mi_dissimilarity']:.4f}, "
                    f"Corr={corr_sim:.4f}"
                    if corr_sim is not None
                    else f"  Registration quality: MI={quality_metrics['mi_dissimilarity']:.4f}, "
                    f"Corr=N/A"
                )

                # Warn if quality is poor (correlation_dissimilarity > threshold means poor alignment)
                if (
                    corr_dissim is not None
                    and corr_dissim > self.quality_warning_threshold
                ):
                    self.logger.warning(
                        f"  WARNING: Registration quality may be poor! "
                        f"Correlation dissimilarity ({corr_dissim:.4f}) "
                        f"> threshold ({self.quality_warning_threshold})"
                    )

            # Save warped reference to temp location
            temp_output = study_dir / f"_temp_{self.reference_modality}_atlas.nii.gz"
            ants.image_write(warped_img, str(temp_output))

            if not temp_output.exists():
                raise RuntimeError(f"Registered image not created: {temp_output}")

            # Replace original with atlas-space version
            final_output = reference_path
            temp_output.replace(final_output)

            if self.verbose:
                self.logger.debug(
                    f"[DEBUG] [AntsPyX] Replaced {reference_path.name} with atlas-space version"
                )

            # Save detailed registration info if enabled
            if self.save_detailed_info and captured_stdout:
                self._save_detailed_registration_info(
                    stdout=captured_stdout,
                    fixed_path=atlas_path,
                    moving_path=reference_path,
                    transform_path=actual_transform_path,
                    config_params={
                        "transforms": transforms,
                        "type_of_transform": type_of_transform,
                        "metric": metric,
                        "metric_bins": metric_bins,
                        "sampling_strategy": self.config.get(
                            "sampling_strategy", "Random"
                        ),
                        "sampling_percentage": sampling_percentage,
                        "number_of_iterations": number_of_iterations_list,
                        "shrink_factors": shrink_factors_list,
                        "smoothing_sigmas": smoothing_sigmas_list,
                        "convergence_threshold": self.config.get(
                            "convergence_threshold", 1e-6
                        ),
                        "convergence_window_size": self.config.get(
                            "convergence_window_size", 10
                        ),
                        "interpolation": self.config.get("interpolation", "Linear"),
                    },
                    registration_type="intra_study_to_atlas",
                )

            return final_output, actual_transform_path, quality_metrics

        except Exception as e:
            error_msg = f"AntsPyX reference→atlas registration failed: {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def _compute_registration_quality(
        self, fixed_img: "ants.ANTsImage", warped_img: "ants.ANTsImage"
    ) -> Dict[str, float]:
        """Compute quality metrics for registration result.

        Args:
            fixed_img: Fixed (atlas) image
            warped_img: Warped (registered) moving image

        Returns:
            Dictionary with quality metrics:
            - mi_dissimilarity: Mattes Mutual Information dissimilarity (lower = more similar)
            - correlation_dissimilarity: Correlation dissimilarity (returns -1 for perfect match)
            - correlation_similarity: Correlation similarity (0-1 scale, higher = better)
        """
        import ants

        # Compute Mattes Mutual Information (lower = more similar in ANTs)
        mi_dissimilarity = ants.image_similarity(
            fixed_img, warped_img, metric_type="MattesMutualInformation"
        )

        # Compute correlation (returns -1 for perfect match)
        corr_dissimilarity = ants.image_similarity(
            fixed_img, warped_img, metric_type="Correlation"
        )

        return {
            "mi_dissimilarity": float(mi_dissimilarity),
            "correlation_dissimilarity": float(corr_dissimilarity),
            # Convert to similarity (0-1 scale, higher = better)
            "correlation_similarity": float(-corr_dissimilarity)
            if corr_dissimilarity is not None
            else None,
        }

    def _apply_transforms_to_modality(
        self,
        modality_path: Path,
        atlas_path: Path,
        m_to_ref_transform: Path,
        ref_to_atlas_transform: Path,
        study_dir: Path,
        modality: str,
    ) -> Path:
        """Apply ref-to-atlas transform to bring modality to atlas space using AntsPyX.

        After step 3a, all modality files have been coregistered to reference space
        in-place. Therefore we only need to apply the ref_to_atlas transform here.
        The m_to_ref_transform parameter is retained for logging/provenance but is
        NOT applied (it was already applied in-place during step 3a).

        Args:
            modality_path: Path to modality file (already in reference space after step 3a)
            atlas_path: Path to atlas (defines output space)
            m_to_ref_transform: Transform from modality to reference (kept for provenance, not applied)
            ref_to_atlas_transform: Transform from reference to atlas
            study_dir: Study directory
            modality: Modality name

        Returns:
            Path to transformed modality file

        Raises:
            RuntimeError: If transform application fails
        """
        import ants

        if self.verbose:
            self.logger.debug(
                f"[DEBUG] [AntsPyX] Applying atlas transform for {modality}:"
            )
            self.logger.debug(
                f"  Transform: {ref_to_atlas_transform.name} (m_to_ref already applied in step 3a)"
            )

        try:
            # Load images
            atlas_img = ants.image_read(str(atlas_path))
            modality_img = ants.image_read(str(modality_path))

            # Get interpolation method
            interpolation = self.config.get("interpolation", "Linear")
            # Map to AntsPyX format (lowercase)
            interp_map = {
                "Linear": "linear",
                "BSpline": "bSpline",
                "NearestNeighbor": "nearestNeighbor",
                "Gaussian": "gaussian",
                "MultiLabel": "multiLabel",
            }
            ants_interpolator = interp_map.get(interpolation, "linear")

            # Apply only ref_to_atlas transform.
            # The modality file is already in reference space (step 3a replaced it in-place),
            # so applying m_to_ref again would double-transform and cause catastrophic displacement.
            transformed = ants.apply_transforms(
                fixed=atlas_img,
                moving=modality_img,
                transformlist=[str(ref_to_atlas_transform)],
                interpolator=ants_interpolator,
                defaultvalue=0,
            )

            # Save to temp location
            temp_output = study_dir / f"_temp_{modality}_atlas.nii.gz"
            ants.image_write(transformed, str(temp_output))

            if not temp_output.exists():
                raise RuntimeError(f"Transformed image not created: {temp_output}")

            # Replace original
            final_output = modality_path
            temp_output.replace(final_output)

            if self.verbose:
                self.logger.debug(
                    f"[DEBUG] [AntsPyX] Replaced {modality_path.name} with atlas-space version"
                )

            return final_output

        except Exception as e:
            error_msg = f"AntsPyX transform application failed for {modality}: {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def _save_detailed_registration_info(
        self,
        stdout: str,
        fixed_path: Path,
        moving_path: Path,
        transform_path: Path,
        config_params: Dict[str, Any],
        registration_type: str,
    ) -> None:
        """Save detailed registration information to JSON.

        Args:
            stdout: Captured stdout from registration
            fixed_path: Path to fixed image (atlas)
            moving_path: Path to moving image (reference modality)
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
                    "error_message": None,
                },
                "parameters": config_params,
                "timing": {
                    "total_elapsed_time_seconds": parsed_diagnostics.get(
                        "total_elapsed_time_seconds"
                    ),
                    "stages": [
                        {
                            "stage_index": stage["stage_index"],
                            "transform_type": transform_types[stage["stage_index"]]
                            if stage["stage_index"] < len(transform_types)
                            else "Unknown",
                            "elapsed_time_seconds": stage.get("elapsed_time_seconds"),
                        }
                        for stage in parsed_diagnostics.get("stages", [])
                    ],
                },
                "convergence": {"stages": parsed_diagnostics.get("stages", [])},
                "stdout_capture": {
                    "full_output": stdout,
                    "command_lines_ok": parsed_diagnostics.get(
                        "command_lines_ok", False
                    ),
                },
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
                    "error_message": f"Parsing failed: {str(parse_error)}",
                },
                "stdout_capture": {"full_output": stdout},
            }

        try:
            # Determine output path
            # For atlas registration, filename is: {reference_modality}_to_atlas_info.json
            moving_modality = moving_path.stem.replace(".nii", "")

            # Create detailed_info directory
            detailed_info_dir = transform_path.parent / "detailed_info"
            detailed_info_dir.mkdir(parents=True, exist_ok=True)

            # Construct filename
            json_filename = f"{moving_modality}_to_atlas_info.json"
            json_path = detailed_info_dir / json_filename

            # Write JSON
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(
                    info, f, indent=2, default=str
                )  # default=str handles inf, datetime

            self.logger.info(f"Saved detailed registration info: {json_path.name}")

        except Exception as write_error:
            self.logger.error(
                f"Failed to write detailed registration info: {write_error}"
            )
            # Don't raise - registration succeeded, just diagnostic save failed

    def visualize(
        self,
        atlas_path: Optional[Path] = None,
        reference_path: Optional[Path] = None,
        modality_paths: Optional[Dict[str, Path]] = None,
        output_dir: Optional[Path] = None,
        **kwargs: Any,
    ) -> None:
        """Generate visualizations for atlas registration.

        Supports two modes:
        1. Batch mode (original): output_dir, modality_paths, etc.
        2. Single mode (registration.py): output_path, atlas_path, reference_path/modality_path

        Args:
            atlas_path: Path to atlas template
            reference_path: Path to reference modality (now in atlas space)
            modality_paths: Dict mapping modality to path (all in atlas space)
            output_dir: Directory to save visualizations
            **kwargs: Additional parameters (e.g., output_path, modality_path, modality)
        """
        # Handle single mode (registration.py usage)
        output_path = kwargs.get("output_path")
        if output_path is not None:
            modality_path = kwargs.get("modality_path")

            if atlas_path and reference_path and not modality_path:
                self.visualize_reference_to_atlas(
                    atlas_path, reference_path, output_path
                )
                return
            elif atlas_path and modality_path:
                modality = kwargs.get("modality", "unknown")
                self.visualize_modality_in_atlas_space(
                    atlas_path, modality_path, output_path, modality
                )
                return

        try:
            # 1. Visualize reference→atlas alignment
            if output_dir and atlas_path and reference_path:
                output_dir.mkdir(parents=True, exist_ok=True)
                ref_output = output_dir / "atlas_registration_reference.png"
                self.visualize_reference_to_atlas(
                    atlas_path, reference_path, ref_output
                )

            # 2. Visualize each modality in atlas space
            if output_dir and atlas_path and modality_paths:
                for modality, mod_path in modality_paths.items():
                    if modality == self.reference_modality:
                        continue

                    if not mod_path.exists():
                        continue

                    mod_output = output_dir / f"atlas_registration_{modality}.png"
                    self.visualize_modality_in_atlas_space(
                        atlas_path, mod_path, mod_output, modality
                    )

        except Exception as e:
            self.logger.error(f"Failed to generate visualization: {e}")
            raise RuntimeError(f"Visualization failed: {e}") from e

    def visualize_reference_to_atlas(
        self, atlas_path: Path, reference_path: Path, output_path: Path
    ) -> None:
        """Visualize reference modality registration to atlas."""
        try:
            atlas_img = nib.load(str(atlas_path))
            ref_img = nib.load(str(reference_path))

            atlas_data = atlas_img.get_fdata()
            ref_data = ref_img.get_fdata()

            slice_idx = atlas_data.shape[2] // 2

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))

            # Atlas - squeeze to ensure 2D for imshow
            atlas_slice = np.squeeze(atlas_data[:, :, slice_idx])
            axes[0].imshow(atlas_slice.T, cmap="gray", origin="lower")
            axes[0].set_title("Atlas Template")
            axes[0].axis("off")

            # Reference in atlas space - squeeze to ensure 2D for imshow
            ref_slice = np.squeeze(ref_data[:, :, slice_idx])
            axes[1].imshow(ref_slice.T, cmap="gray", origin="lower")
            axes[1].set_title(f"Reference ({self.reference_modality}) in Atlas Space")
            axes[1].axis("off")

            fig.suptitle(
                "Atlas Registration (AntsPyX): Reference Alignment", fontsize=12
            )
            plt.tight_layout()

            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()

            self.logger.info(f"Saved reference→atlas visualization: {output_path.name}")
        except Exception as e:
            self.logger.error(f"Failed to visualize reference to atlas: {e}")
            raise

    def visualize_modality_in_atlas_space(
        self, atlas_path: Path, modality_path: Path, output_path: Path, modality: str
    ) -> None:
        """Visualize modality in atlas space."""
        try:
            atlas_img = nib.load(str(atlas_path))
            mod_img = nib.load(str(modality_path))

            atlas_data = atlas_img.get_fdata()
            mod_data = mod_img.get_fdata()

            slice_idx = atlas_data.shape[2] // 2

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))

            # Atlas - squeeze to ensure 2D for imshow
            atlas_slice = np.squeeze(atlas_data[:, :, slice_idx])
            axes[0].imshow(atlas_slice.T, cmap="gray", origin="lower")
            axes[0].set_title("Atlas Template")
            axes[0].axis("off")

            # Modality in atlas space - squeeze to ensure 2D for imshow
            mod_slice = np.squeeze(mod_data[:, :, slice_idx])
            axes[1].imshow(mod_slice.T, cmap="gray", origin="lower")
            axes[1].set_title(f"{modality} in Atlas Space")
            axes[1].axis("off")

            fig.suptitle(
                f"Atlas Registration (AntsPyX): {modality} Alignment", fontsize=12
            )
            plt.tight_layout()

            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()

            self.logger.info(f"Saved {modality} visualization: {output_path.name}")
        except Exception as e:
            self.logger.error(f"Failed to visualize modality {modality}: {e}")
            raise
