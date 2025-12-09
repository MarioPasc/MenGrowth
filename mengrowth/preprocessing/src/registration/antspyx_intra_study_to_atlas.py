"""Register reference modality to atlas space and propagate transforms using AntsPyX.

This module implements atlas registration using the AntsPyX library instead of nipype.
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple
import logging
import time

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

from mengrowth.preprocessing.src.registration.base import BaseRegistrator

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
        self,
        config: Dict[str, Any],
        reference_modality: str,
        verbose: bool = False
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
        **kwargs: Any
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
            self.logger.debug(f"[DEBUG] [AntsPyX] Reference modality: {self.reference_modality}")

        # 1. Register reference modality to atlas
        reference_file = study_dir / f"{self.reference_modality}.nii.gz"
        if not reference_file.exists():
            raise ValueError(f"Reference modality file not found: {reference_file}")

        registration_dir = artifacts_dir / "registration"
        registration_dir.mkdir(parents=True, exist_ok=True)

        ref_to_atlas_transform_path = registration_dir / f"{self.reference_modality}_to_atlasComposite.h5"

        try:
            self.logger.info(f"Registering {self.reference_modality} to atlas...")
            ref_registered_path, ref_to_atlas_transform = self._register_reference_to_atlas(
                reference_path=reference_file,
                atlas_path=atlas_path,
                transform_path=ref_to_atlas_transform_path,
                study_dir=study_dir
            )
            self.logger.info(f"✓ Reference registered to atlas")

        except Exception as e:
            self.logger.error(f"✗ Failed to register reference to atlas: {e}")
            raise RuntimeError(f"Atlas registration failed: {e}") from e

        # 2. Apply transforms to all other modalities
        atlas_transforms = {}
        registered_modalities = [self.reference_modality]  # Reference is already registered

        for modality in modalities:
            if modality == self.reference_modality:
                # Reference already registered above
                atlas_transforms[modality] = ref_to_atlas_transform
                continue

            modality_file = study_dir / f"{modality}.nii.gz"
            if not modality_file.exists():
                if self.verbose:
                    self.logger.debug(f"[DEBUG] [AntsPyX] Modality {modality} not found, skipping")
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
                    modality=modality
                )

                atlas_transforms[modality] = {
                    "m_to_ref": m_to_ref_transform,
                    "ref_to_atlas": ref_to_atlas_transform
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
            "reference_to_atlas_transform": ref_to_atlas_transform,
            "atlas_transforms": atlas_transforms,
            "registered_modalities": registered_modalities
        }

    def _register_reference_to_atlas(
        self,
        reference_path: Path,
        atlas_path: Path,
        transform_path: Path,
        study_dir: Path
    ) -> Tuple[Path, Path]:
        """Register reference modality to atlas using AntsPyX.

        Handles multi-stage registration (e.g., Rigid + Affine).

        Args:
            reference_path: Path to reference modality (moving image)
            atlas_path: Path to atlas template (fixed image)
            transform_path: Base path for transform output (without "Composite.h5")
            study_dir: Study directory for output

        Returns:
            Tuple of (registered_reference_path, actual_transform_path)

        Raises:
            RuntimeError: If registration fails
        """
        import ants

        if self.verbose:
            self.logger.debug(f"[DEBUG] [AntsPyX] Reference→Atlas registration setup:")
            self.logger.debug(f"  Fixed (atlas):   {atlas_path}")
            self.logger.debug(f"  Moving (ref):    {reference_path}")

        try:
            # Load images
            atlas_img = ants.image_read(str(atlas_path))
            reference_img = ants.image_read(str(reference_path))

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
                "number_of_iterations",
                [[1000, 500, 250], [500, 250, 100]]
            )
            shrink_factors_list = self.config.get(
                "shrink_factors",
                [[4, 2, 1], [2, 1, 1]]
            )
            smoothing_sigmas_list = self.config.get(
                "smoothing_sigmas",
                [[2, 1, 0], [1, 0, 0]]
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
                self.logger.debug(f"[DEBUG] [AntsPyX] Atlas registration parameters:")
                self.logger.debug(f"  type_of_transform: {type_of_transform}")
                self.logger.debug(f"  aff_metric: {metric}")
                self.logger.debug(f"  aff_iterations: {aff_iterations}")

            # Perform registration
            result = ants.registration(
                fixed=atlas_img,
                moving=reference_img,
                type_of_transform=type_of_transform,
                outprefix=transform_prefix,
                aff_metric=metric,
                aff_sampling=metric_bins,
                aff_random_sampling_rate=sampling_percentage,
                aff_iterations=aff_iterations,
                aff_shrink_factors=aff_shrink_factors,
                aff_smoothing_sigmas=aff_smoothing_sigmas,
                write_composite_transform=True,
                verbose=self.verbose
            )

            if self.verbose:
                self.logger.debug(f"[DEBUG] [AntsPyX] Registration completed")

            # Extract outputs
            warped_img = result['warpedmovout']
            fwd_transforms = result['fwdtransforms']

            if self.verbose:
                self.logger.debug(f"[DEBUG] [AntsPyX] Forward transforms: {fwd_transforms}")

            # Determine actual transform path
            actual_transform_path = Path(transform_prefix + "Composite.h5")
            if not actual_transform_path.exists() and fwd_transforms:
                actual_transform_path = Path(fwd_transforms[0])

            if not actual_transform_path.exists():
                raise RuntimeError(f"Transform not created: {actual_transform_path}")

            # Save warped reference to temp location
            temp_output = study_dir / f"_temp_{self.reference_modality}_atlas.nii.gz"
            ants.image_write(warped_img, str(temp_output))

            if not temp_output.exists():
                raise RuntimeError(f"Registered image not created: {temp_output}")

            # Replace original with atlas-space version
            final_output = reference_path
            temp_output.replace(final_output)

            if self.verbose:
                self.logger.debug(f"[DEBUG] [AntsPyX] Replaced {reference_path.name} with atlas-space version")

            return final_output, actual_transform_path

        except Exception as e:
            error_msg = f"AntsPyX reference→atlas registration failed: {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def _apply_transforms_to_modality(
        self,
        modality_path: Path,
        atlas_path: Path,
        m_to_ref_transform: Path,
        ref_to_atlas_transform: Path,
        study_dir: Path,
        modality: str
    ) -> Path:
        """Apply composed transforms to bring modality to atlas space using AntsPyX.

        Args:
            modality_path: Path to modality file (in reference space)
            atlas_path: Path to atlas (defines output space)
            m_to_ref_transform: Transform from modality to reference
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
            self.logger.debug(f"[DEBUG] [AntsPyX] Applying transforms for {modality}:")
            self.logger.debug(f"  Transforms: [{m_to_ref_transform.name}, {ref_to_atlas_transform.name}]")

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
                "MultiLabel": "multiLabel"
            }
            ants_interpolator = interp_map.get(interpolation, "linear")

            # Apply transforms
            # AntsPyX applies in reverse order (same as ANTs)
            # We want: modality → ref → atlas
            # So list is: [ref_to_atlas, m_to_ref]
            transformed = ants.apply_transforms(
                fixed=atlas_img,
                moving=modality_img,
                transformlist=[
                    str(ref_to_atlas_transform),
                    str(m_to_ref_transform)
                ],
                interpolator=ants_interpolator,
                defaultvalue=0
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
                self.logger.debug(f"[DEBUG] [AntsPyX] Replaced {modality_path.name} with atlas-space version")

            return final_output

        except Exception as e:
            error_msg = f"AntsPyX transform application failed for {modality}: {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def visualize(
        self,
        atlas_path: Path,
        reference_path: Path,
        modality_paths: Dict[str, Path],
        output_dir: Path,
        **kwargs: Any
    ) -> None:
        """Generate visualizations for atlas registration.

        Creates visualizations showing:
        1. Reference modality in atlas space
        2. Each modality in atlas space

        Args:
            atlas_path: Path to atlas template
            reference_path: Path to reference modality (now in atlas space)
            modality_paths: Dict mapping modality to path (all in atlas space)
            output_dir: Directory to save visualizations
            **kwargs: Additional parameters
        """
        try:
            # Load atlas
            atlas_img = nib.load(str(atlas_path))
            atlas_data = atlas_img.get_fdata()

            # 1. Visualize reference→atlas alignment
            ref_img = nib.load(str(reference_path))
            ref_data = ref_img.get_fdata()

            slice_idx = atlas_data.shape[2] // 2

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))

            # Atlas
            axes[0].imshow(atlas_data[:, :, slice_idx].T, cmap="gray", origin="lower")
            axes[0].set_title("Atlas Template")
            axes[0].axis("off")

            # Reference in atlas space
            axes[1].imshow(ref_data[:, :, slice_idx].T, cmap="gray", origin="lower")
            axes[1].set_title(f"Reference ({self.reference_modality}) in Atlas Space")
            axes[1].axis("off")

            fig.suptitle(f"Atlas Registration (AntsPyX): Reference Alignment", fontsize=12)
            plt.tight_layout()

            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"atlas_registration_reference.png"
            plt.savefig(output_file, dpi=150, bbox_inches="tight")
            plt.close()

            self.logger.info(f"Saved reference→atlas visualization: {output_file.name}")

            # 2. Visualize each modality in atlas space
            for modality, mod_path in modality_paths.items():
                if modality == self.reference_modality:
                    continue

                if not mod_path.exists():
                    continue

                mod_img = nib.load(str(mod_path))
                mod_data = mod_img.get_fdata()

                fig, axes = plt.subplots(1, 2, figsize=(10, 5))

                # Atlas
                axes[0].imshow(atlas_data[:, :, slice_idx].T, cmap="gray", origin="lower")
                axes[0].set_title("Atlas Template")
                axes[0].axis("off")

                # Modality in atlas space
                axes[1].imshow(mod_data[:, :, slice_idx].T, cmap="gray", origin="lower")
                axes[1].set_title(f"{modality} in Atlas Space")
                axes[1].axis("off")

                fig.suptitle(f"Atlas Registration (AntsPyX): {modality} Alignment", fontsize=12)
                plt.tight_layout()

                output_file = output_dir / f"atlas_registration_{modality}.png"
                plt.savefig(output_file, dpi=150, bbox_inches="tight")
                plt.close()

                self.logger.info(f"Saved {modality} visualization: {output_file.name}")

        except Exception as e:
            self.logger.error(f"Failed to generate visualization: {e}")
            raise RuntimeError(f"Visualization failed: {e}") from e
