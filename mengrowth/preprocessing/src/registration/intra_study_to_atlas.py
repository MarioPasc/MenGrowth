"""Register reference modality to atlas space and propagate transforms.

This module implements atlas registration where the reference modality is registered
to an atlas template (e.g., SRI24), and all other modalities are transformed to atlas
space by composing the intra-study and atlas transforms.
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple
import logging
import time

import nibabel as nib
import matplotlib.pyplot as plt

from mengrowth.preprocessing.src.registration.base import BaseRegistrator

logger = logging.getLogger(__name__)


class IntraStudyToAtlas(BaseRegistrator):
    """Register reference modality to atlas and propagate to all modalities.

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
        """Initialize atlas registration step.

        Args:
            config: Configuration dictionary from IntraStudyToAtlasConfig
            reference_modality: Reference modality name (from step 3a)
            verbose: Enable verbose logging
        """
        super().__init__(config=config, verbose=verbose)
        self.reference_modality = reference_modality
        self.logger = logging.getLogger(__name__)

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
        self.logger.info(f"Starting atlas registration for {study_dir.name}")

        # Validate atlas file exists
        atlas_path = Path(self.config.get("atlas_path", ""))
        if not atlas_path.exists():
            raise ValueError(f"Atlas file not found: {atlas_path}")

        if self.verbose:
            self.logger.debug(f"[DEBUG] Atlas path: {atlas_path}")
            self.logger.debug(f"[DEBUG] Reference modality: {self.reference_modality}")

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
            ref_registered_path, ref_to_atlas_transform = (
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
                        f"[DEBUG] Modality {modality} not found, skipping"
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
            "reference_to_atlas_transform": ref_to_atlas_transform,
            "atlas_transforms": atlas_transforms,
            "registered_modalities": registered_modalities,
        }

    def _register_reference_to_atlas(
        self,
        reference_path: Path,
        atlas_path: Path,
        transform_path: Path,
        study_dir: Path,
    ) -> Tuple[Path, Path]:
        """Register reference modality to atlas using ANTs.

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
        from nipype.interfaces import ants

        # Temp output for registered reference
        temp_output = study_dir / f"_temp_{self.reference_modality}_atlas.nii.gz"

        if self.verbose:
            self.logger.debug("[DEBUG] Reference→Atlas registration setup:")
            self.logger.debug(f"  Fixed (atlas):   {atlas_path}")
            self.logger.debug(f"  Moving (ref):    {reference_path}")
            self.logger.debug(f"  Transform path:  {transform_path}")
            self.logger.debug(f"  Temp output:     {temp_output}")

        # Initialize ANTs Registration
        reg = ants.Registration()
        reg.inputs.dimension = 3
        reg.inputs.fixed_image = str(atlas_path)
        reg.inputs.moving_image = str(reference_path)

        # Get transforms from config
        transforms = self.config.get("transforms", ["Rigid", "Affine"])
        reg.inputs.transforms = transforms

        # Transform parameters (one per transform)
        reg.inputs.transform_parameters = [(0.1,)] * len(transforms)

        # Metric configuration
        metric = self.config.get("metric", "Mattes")
        metric_bins = self.config.get("metric_bins", 32)
        reg.inputs.metric = [metric] * len(transforms)
        reg.inputs.metric_weight = [1.0] * len(transforms)
        reg.inputs.radius_or_number_of_bins = [metric_bins] * len(transforms)

        # Sampling
        sampling_strategy = self.config.get("sampling_strategy", "Random")
        sampling_percentage = self.config.get("sampling_percentage", 0.2)
        reg.inputs.sampling_strategy = [sampling_strategy] * len(transforms)
        reg.inputs.sampling_percentage = [sampling_percentage] * len(transforms)

        # Multi-resolution schedule
        reg.inputs.number_of_iterations = self.config.get(
            "number_of_iterations", [[1000, 500, 250], [500, 250, 100]]
        )
        reg.inputs.shrink_factors = self.config.get(
            "shrink_factors", [[4, 2, 1], [2, 1, 1]]
        )
        reg.inputs.smoothing_sigmas = self.config.get(
            "smoothing_sigmas", [[2, 1, 0], [1, 0, 0]]
        )
        reg.inputs.sigma_units = ["vox"] * len(transforms)

        # Convergence
        convergence_threshold = self.config.get("convergence_threshold", 1e-6)
        convergence_window_size = self.config.get("convergence_window_size", 10)
        reg.inputs.convergence_threshold = [convergence_threshold] * len(transforms)
        reg.inputs.convergence_window_size = [convergence_window_size] * len(transforms)

        # Output configuration
        reg.inputs.write_composite_transform = True
        transform_prefix = str(
            transform_path.with_suffix("").with_suffix("")
        )  # Remove .h5 if present
        reg.inputs.output_transform_prefix = transform_prefix
        reg.inputs.output_warped_image = str(temp_output)

        # ANTs will append "Composite.h5"
        actual_transform_path = Path(str(transform_prefix) + "Composite.h5")

        # Interpolation
        interpolation = self.config.get("interpolation", "Linear")
        reg.inputs.interpolation = interpolation

        # Verbose
        reg.inputs.verbose = self.verbose

        if self.verbose:
            self.logger.debug("[DEBUG] ANTs parameters:")
            self.logger.debug(f"  Transforms: {transforms}")
            self.logger.debug(f"  Metric: {metric} (bins={metric_bins})")
            self.logger.debug(
                f"  Sampling: {sampling_strategy} ({sampling_percentage * 100}%)"
            )
            self.logger.debug(f"  Interpolation: {interpolation}")
            self.logger.debug(f"  Actual transform: {actual_transform_path}")

        try:
            if self.verbose:
                self.logger.debug("[DEBUG] Executing ANTs registration...")

            result = reg.run()

            # Verify transform created
            if not actual_transform_path.exists():
                if self.verbose:
                    transform_dir = actual_transform_path.parent
                    if transform_dir.exists():
                        self.logger.debug(f"[DEBUG] Files in {transform_dir}:")
                        for f in transform_dir.iterdir():
                            self.logger.debug(f"  - {f.name}")
                raise RuntimeError(f"Transform not created: {actual_transform_path}")

            # Verify output image created
            if not temp_output.exists():
                raise RuntimeError(f"Registered image not created: {temp_output}")

            # Replace original with atlas-space version
            final_output = reference_path
            temp_output.replace(final_output)

            if self.verbose:
                self.logger.debug(
                    f"[DEBUG] Replaced {reference_path.name} with atlas-space version"
                )

            return final_output, actual_transform_path

        except Exception as e:
            if temp_output.exists():
                temp_output.unlink()
            raise RuntimeError(f"Reference→atlas registration failed: {e}") from e

    def _apply_transforms_to_modality(
        self,
        modality_path: Path,
        atlas_path: Path,
        m_to_ref_transform: Path,
        ref_to_atlas_transform: Path,
        study_dir: Path,
        modality: str,
    ) -> Path:
        """Apply ref-to-atlas transform to bring modality to atlas space.

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
        from nipype.interfaces import ants

        temp_output = study_dir / f"_temp_{modality}_atlas.nii.gz"

        if self.verbose:
            self.logger.debug(f"[DEBUG] Applying atlas transform for {modality}:")
            self.logger.debug(f"  Input: {modality_path}")
            self.logger.debug(f"  Atlas: {atlas_path}")
            self.logger.debug(
                f"  Transform: {ref_to_atlas_transform.name} (m_to_ref already applied in step 3a)"
            )
            self.logger.debug(f"  Temp output: {temp_output}")

        # Initialize ApplyTransforms
        apply_xfm = ants.ApplyTransforms()
        apply_xfm.inputs.dimension = 3
        apply_xfm.inputs.input_image = str(modality_path)
        apply_xfm.inputs.reference_image = str(atlas_path)

        # Only apply ref_to_atlas transform.
        # The modality file is already in reference space (step 3a replaced it in-place),
        # so applying m_to_ref again would double-transform and cause catastrophic displacement.
        apply_xfm.inputs.transforms = [str(ref_to_atlas_transform)]

        apply_xfm.inputs.interpolation = self.config.get("interpolation", "Linear")
        apply_xfm.inputs.default_value = 0
        apply_xfm.inputs.output_image = str(temp_output)

        try:
            if self.verbose:
                self.logger.debug("[DEBUG] Executing ApplyTransforms...")

            result = apply_xfm.run()

            if not temp_output.exists():
                raise RuntimeError(f"Transformed image not created: {temp_output}")

            # Replace original with atlas-space version
            final_output = modality_path
            temp_output.replace(final_output)

            if self.verbose:
                self.logger.debug(
                    f"[DEBUG] Replaced {modality_path.name} with atlas-space version"
                )

            return final_output

        except Exception as e:
            if temp_output.exists():
                temp_output.unlink()
            raise RuntimeError(f"Transform application failed: {e}") from e

    def visualize(
        self,
        atlas_path: Path,
        reference_path: Path,
        modality_paths: Dict[str, Path],
        output_dir: Path,
        **kwargs: Any,
    ) -> None:
        """Generate visualizations for atlas registration.

        Creates two types of visualizations:
        1. Reference→Atlas alignment (shows registration quality)
        2. Each modality in atlas space (shows final atlas-space images)

        Args:
            atlas_path: Path to atlas template
            reference_path: Path to reference in atlas space
            modality_paths: Dict mapping modality names to atlas-space paths
            output_dir: Directory to save visualizations
            **kwargs: Additional parameters

        Raises:
            RuntimeError: If visualization generation fails
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 1. Reference→Atlas visualization
            self._visualize_reference_to_atlas(
                atlas_path=atlas_path,
                reference_path=reference_path,
                output_path=output_dir / "step3b_atlas_registration_reference.png",
            )

            # 2. Modality in atlas space visualizations
            for modality, modality_path in modality_paths.items():
                self._visualize_modality_in_atlas(
                    atlas_path=atlas_path,
                    modality_path=modality_path,
                    modality_name=modality,
                    output_path=output_dir / f"step3b_atlas_space_{modality}.png",
                )

        except Exception as e:
            self.logger.error(f"Failed to generate visualizations: {e}")
            raise RuntimeError(f"Visualization failed: {e}") from e

    def _visualize_reference_to_atlas(
        self, atlas_path: Path, reference_path: Path, output_path: Path
    ) -> None:
        """Visualize reference modality registration to atlas.

        Creates 1×2 subplot: Atlas | Reference (in atlas space)

        Args:
            atlas_path: Path to atlas
            reference_path: Path to reference in atlas space
            output_path: Output PNG path
        """
        atlas_img = nib.load(str(atlas_path))
        ref_img = nib.load(str(reference_path))

        atlas_data = atlas_img.get_fdata()
        ref_data = ref_img.get_fdata()

        # Middle axial slice
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

        fig.suptitle(
            f"Atlas Registration: {self.reference_modality} → Atlas", fontsize=12
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        self.logger.info(f"Saved visualization: {output_path.name}")

    def _visualize_modality_in_atlas(
        self,
        atlas_path: Path,
        modality_path: Path,
        modality_name: str,
        output_path: Path,
    ) -> None:
        """Visualize modality in atlas space.

        Creates 1×2 subplot: Atlas | Modality (in atlas space)

        Args:
            atlas_path: Path to atlas
            modality_path: Path to modality in atlas space
            modality_name: Name of modality
            output_path: Output PNG path
        """
        atlas_img = nib.load(str(atlas_path))
        mod_img = nib.load(str(modality_path))

        atlas_data = atlas_img.get_fdata()
        mod_data = mod_img.get_fdata()

        # Middle axial slice
        slice_idx = atlas_data.shape[2] // 2

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # Atlas
        axes[0].imshow(atlas_data[:, :, slice_idx].T, cmap="gray", origin="lower")
        axes[0].set_title("Atlas Template")
        axes[0].axis("off")

        # Modality in atlas space
        axes[1].imshow(mod_data[:, :, slice_idx].T, cmap="gray", origin="lower")
        axes[1].set_title(f"{modality_name.upper()} in Atlas Space")
        axes[1].axis("off")

        fig.suptitle(f"Atlas Space: {modality_name.upper()}", fontsize=12)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        self.logger.info(f"Saved visualization: {output_path.name}")
