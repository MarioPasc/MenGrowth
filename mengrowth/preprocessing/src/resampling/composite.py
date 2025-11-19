"""Composite resampling combining interpolation and deep learning methods.

This module implements a composite resampling strategy that intelligently combines
traditional interpolation (BSpline) with deep learning-based super-resolution (ECLARE).
The strategy applies different methods based on the current resolution of each dimension
relative to the target resolution.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import logging
import tempfile

import SimpleITK as sitk
import nibabel as nib
import numpy as np

from mengrowth.preprocessing.src.resampling.base import BaseResampler
from mengrowth.preprocessing.src.resampling.bspline import BSplineResampler
from mengrowth.preprocessing.src.resampling.eclare import EclareResampler

logger = logging.getLogger(__name__)


class CompositeResampler(BaseResampler):
    """Composite resampling combining interpolation and deep learning methods.

    This resampler implements a hybrid strategy that applies different resampling
    methods based on the resolution gap between current and target voxel sizes:

    Decision Rules (per dimension, with spacing d and target t):
    1. d < t: DL method only (upsampling from better resolution)
    2. t ≤ d ≤ max_mm_interpolator: Interpolation to target (small gap)
    3. max_mm_interpolator < d ≤ max_mm_dl_method: DL method only (medium gap)
    4. d > max_mm_dl_method: Interpolation to intermediate, then DL to target (large gap)

    The composite method uses staged processing:
    - Stage 1: Apply interpolation to eligible dimensions (intermediate resolution)
    - Stage 2: Apply DL method to reach target resolution

    This approach leverages the strengths of both methods:
    - Interpolation: Fast, reliable for small resolution changes
    - DL: High quality for larger resolution changes, especially anisotropic data

    Reference:
        The composite strategy is inspired by multi-scale processing approaches
        commonly used in medical image analysis.
    """

    def __init__(
        self,
        target_voxel_size: List[float],
        config: Dict[str, Any],
        verbose: bool = False
    ) -> None:
        """Initialize composite resampler.

        Args:
            target_voxel_size: Target voxel size in mm [x, y, z]
            config: Configuration dictionary containing:
                - composite_interpolator: Interpolator method ("bspline")
                - composite_dl_method: DL method ("eclare")
                - max_mm_interpolator: Max spacing for interpolation-only (mm)
                - max_mm_dl_method: Max spacing for DL-only (mm)
                - resample_mm_to_interpolator_if_max_mm_dl_method: Intermediate
                  resolution for large gaps (mm)
                - Plus all parameters for BSpline and ECLARE methods
            verbose: Enable verbose logging
        """
        super().__init__(
            target_voxel_size=target_voxel_size,
            config=config,
            verbose=verbose
        )

        # Extract composite-specific parameters
        self.composite_interpolator = config.get("composite_interpolator", "bspline")
        self.composite_dl_method = config.get("composite_dl_method", "eclare")
        self.max_mm_interpolator = config.get("max_mm_interpolator", 1.2)
        self.max_mm_dl_method = config.get("max_mm_dl_method", 5.0)
        self.resample_mm_to_interpolator_if_max_mm_dl_method = config.get(
            "resample_mm_to_interpolator_if_max_mm_dl_method", 3.0
        )

        # Store full config for sub-resamplers
        self.full_config = config

        # Validate composite parameters
        self._validate_composite_params()

        self.logger.info(
            f"Initialized CompositeResampler: "
            f"interpolator={self.composite_interpolator}, "
            f"dl_method={self.composite_dl_method}, "
            f"max_mm_interpolator={self.max_mm_interpolator}, "
            f"max_mm_dl_method={self.max_mm_dl_method}, "
            f"intermediate_mm={self.resample_mm_to_interpolator_if_max_mm_dl_method}"
        )

    def _validate_composite_params(self) -> None:
        """Validate composite-specific parameters.

        Raises:
            ValueError: If parameters are invalid
        """
        if self.composite_interpolator not in ["bspline"]:
            raise ValueError(
                f"composite_interpolator must be 'bspline', "
                f"got {self.composite_interpolator}"
            )

        if self.composite_dl_method not in ["eclare"]:
            raise ValueError(
                f"composite_dl_method must be 'eclare', "
                f"got {self.composite_dl_method}"
            )

        if not 0 < self.max_mm_interpolator < self.max_mm_dl_method:
            raise ValueError(
                f"Must satisfy 0 < max_mm_interpolator < max_mm_dl_method, "
                f"got max_mm_interpolator={self.max_mm_interpolator}, "
                f"max_mm_dl_method={self.max_mm_dl_method}"
            )

        if not 0 < self.resample_mm_to_interpolator_if_max_mm_dl_method < self.max_mm_dl_method:
            raise ValueError(
                f"resample_mm_to_interpolator_if_max_mm_dl_method must be "
                f"between 0 and max_mm_dl_method, got "
                f"{self.resample_mm_to_interpolator_if_max_mm_dl_method}"
            )

    def _analyze_dimension_strategy(
        self,
        current_spacing: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze current spacing and determine resampling strategy.

        Applies the 4-rule decision logic to each dimension to determine
        which method(s) to apply.

        Args:
            current_spacing: Current voxel spacing [x, y, z]

        Returns:
            Dictionary containing:
                - 'needs_interpolation': Whether interpolation stage is needed
                - 'needs_dl': Whether DL stage is needed
                - 'intermediate_spacing': Intermediate spacing after interpolation
                - 'dimension_strategies': Per-dimension strategies
        """
        target = np.array(self.target_voxel_size)
        current = np.array(current_spacing)

        needs_interpolation = False
        needs_dl = False
        intermediate_spacing = current.copy()
        dimension_strategies = []

        for i, (d, t) in enumerate(zip(current, target)):
            dim_name = ['X', 'Y', 'Z'][i]

            if d < t:
                # Rule 1: DL only (upsampling from better resolution)
                strategy = "dl_only"
                needs_dl = True
                intermediate_spacing[i] = d  # No interpolation change
                self.logger.info(
                    f"  {dim_name}: {d:.3f}mm < {t:.3f}mm → DL only (Rule 1)"
                )

            elif d <= self.max_mm_interpolator:
                # Rule 2: Interpolation to target (small gap)
                strategy = "interp_to_target"
                needs_interpolation = True
                intermediate_spacing[i] = t
                self.logger.info(
                    f"  {dim_name}: {d:.3f}mm ≤ {self.max_mm_interpolator}mm → Interp to {t:.3f}mm (Rule 2)"
                )

            elif d <= self.max_mm_dl_method:
                # Rule 3: DL only (medium gap)
                strategy = "dl_only"
                needs_dl = True
                intermediate_spacing[i] = d  # No interpolation change
                self.logger.info(
                    f"  {dim_name}: {d:.3f}mm ≤ {self.max_mm_dl_method}mm → DL only (Rule 3)"
                )

            else:
                # Rule 4: Interpolation to intermediate, then DL to target
                strategy = "interp_then_dl"
                needs_interpolation = True
                needs_dl = True
                intermediate_spacing[i] = self.resample_mm_to_interpolator_if_max_mm_dl_method
                self.logger.info(
                    f"  {dim_name}: {d:.3f}mm > {self.max_mm_dl_method}mm → Interp to {intermediate_spacing[i]:.3f}mm, then DL to {t:.3f}mm (Rule 4)"
                )

            dimension_strategies.append({
                "dimension": dim_name,
                "current_spacing": float(d),
                "target_spacing": float(t),
                "strategy": strategy
            })

        self.logger.info(
            f"Strategy: needs_interpolation={needs_interpolation}, "
            f"needs_dl={needs_dl}"
        )

        return {
            "needs_interpolation": needs_interpolation,
            "needs_dl": needs_dl,
            "intermediate_spacing": intermediate_spacing.tolist(),
            "dimension_strategies": dimension_strategies
        }

    def execute(
        self,
        input_path: Path,
        output_path: Path,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Execute composite resampling with staged processing.

        Args:
            input_path: Path to input NIfTI file
            output_path: Path to output resampled NIfTI file
            **kwargs: Additional parameters:
                - allow_overwrite: Allow overwriting existing files (bool)

        Returns:
            Dictionary containing:
                - 'original_spacing': Original voxel spacing [x, y, z]
                - 'target_spacing': Target voxel spacing [x, y, z]
                - 'original_shape': Original image shape [x, y, z]
                - 'resampled_shape': Final resampled image shape [x, y, z]
                - 'strategy': Strategy dictionary from _analyze_dimension_strategy
                - 'intermediate_spacing': Spacing after interpolation (if applied)
                - 'intermediate_shape': Shape after interpolation (if applied)
                - 'stages_applied': List of stages applied ("interpolation", "dl")

        Raises:
            FileNotFoundError: If input file does not exist
            RuntimeError: If resampling fails
        """
        allow_overwrite = kwargs.get("allow_overwrite", False)

        # Validate inputs
        self.validate_inputs(input_path)
        self.validate_outputs(output_path, allow_overwrite=allow_overwrite)

        # Log execution
        self.log_execution(input_path, output_path)

        try:
            # Load image to get current spacing
            self.logger.debug(f"Loading image to analyze spacing: {input_path}")
            image_sitk = sitk.ReadImage(str(input_path))
            original_spacing = np.array(image_sitk.GetSpacing())
            original_shape = np.array(image_sitk.GetSize())

            self.logger.info(
                f"Original spacing: {original_spacing}, shape: {original_shape}"
            )

            # Analyze dimension-wise strategy
            strategy = self._analyze_dimension_strategy(original_spacing)

            stages_applied = []
            intermediate_path: Optional[Path] = None
            intermediate_spacing: Optional[List[float]] = None
            intermediate_shape: Optional[List[int]] = None

            # Determine current input for next stage
            current_input = input_path

            # Stage 1: Apply interpolation if needed
            if strategy["needs_interpolation"]:
                self.logger.info("Stage 1: Applying interpolation...")

                # Create temporary file for intermediate result
                temp_dir = output_path.parent

                # Handle double extension (.nii.gz)
                if output_path.suffix == '.gz' and output_path.stem.endswith('.nii'):
                    base_name = output_path.stem[:-4]  # Remove .nii
                    temp_suffix = '.nii.gz'
                else:
                    base_name = output_path.stem
                    temp_suffix = output_path.suffix

                temp_fd, temp_path_str = tempfile.mkstemp(
                    suffix=temp_suffix,
                    prefix=f"{base_name}_after_interp_",
                    dir=temp_dir
                )
                intermediate_path = Path(temp_path_str)
                import os
                os.close(temp_fd)  # Close file descriptor

                # Create BSpline resampler with intermediate target spacing
                bspline_config = {
                    "bspline_order": self.full_config.get("bspline_order", 3)
                }
                bspline_resampler = BSplineResampler(
                    target_voxel_size=strategy["intermediate_spacing"],
                    config=bspline_config,
                    verbose=self.verbose
                )

                # Execute interpolation
                interp_result = bspline_resampler.execute(
                    input_path=current_input,
                    output_path=intermediate_path,
                    allow_overwrite=True
                )

                intermediate_spacing = interp_result["target_spacing"]
                intermediate_shape = interp_result["resampled_shape"]
                stages_applied.append("interpolation")

                self.logger.info(
                    f"Interpolation complete: spacing={intermediate_spacing}, "
                    f"shape={intermediate_shape}"
                )

                # Update current input for next stage
                current_input = intermediate_path

            # Stage 2: Apply DL method if needed
            if strategy["needs_dl"]:
                self.logger.info("Stage 2: Applying DL method...")

                # Create ECLARE resampler with final target spacing
                eclare_config = {
                    "conda_environment_eclare": self.full_config.get(
                        "conda_environment_eclare", "eclare_env"
                    ),
                    "batch_size": self.full_config.get("batch_size", 128),
                    "n_patches": self.full_config.get("n_patches", 50000),
                    "patch_sampling": self.full_config.get("patch_sampling", "gradient"),
                    "suffix": self.full_config.get("suffix", ""),
                    "gpu_id": self.full_config.get("gpu_id", 0)
                }
                eclare_resampler = EclareResampler(
                    target_voxel_size=self.target_voxel_size,
                    config=eclare_config,
                    verbose=self.verbose
                )

                # Execute DL resampling
                dl_result = eclare_resampler.execute(
                    input_path=current_input,
                    output_path=output_path,
                    allow_overwrite=True
                )

                final_spacing = dl_result["target_spacing"]
                final_shape = dl_result["resampled_shape"]
                stages_applied.append("dl")

                self.logger.info(
                    f"DL method complete: spacing={final_spacing}, "
                    f"shape={final_shape}"
                )

            else:
                # No DL stage needed, interpolation output is final
                # Move intermediate result to final output
                if intermediate_path is not None:
                    import shutil
                    shutil.move(str(intermediate_path), str(output_path))
                    intermediate_path = None  # Prevent cleanup

                    final_spacing = intermediate_spacing
                    final_shape = intermediate_shape
                else:
                    # Edge case: no stages needed (already at target resolution)
                    # Just copy input to output
                    import shutil
                    shutil.copy2(str(input_path), str(output_path))

                    final_spacing = self.target_voxel_size
                    final_shape = original_shape.tolist()

            # Clean up temporary file if it still exists
            if intermediate_path is not None and intermediate_path.exists():
                intermediate_path.unlink()

            # Construct result metadata
            result = {
                "original_spacing": original_spacing.tolist(),
                "target_spacing": self.target_voxel_size,
                "original_shape": original_shape.tolist(),
                "resampled_shape": final_shape,
                "strategy": strategy,
                "stages_applied": stages_applied
            }

            # Add intermediate metadata if interpolation was applied
            if strategy["needs_interpolation"]:
                result["intermediate_spacing"] = intermediate_spacing
                result["intermediate_shape"] = intermediate_shape

            self.logger.info(
                f"Composite resampling complete: "
                f"stages={stages_applied}, "
                f"final_spacing={final_spacing}, "
                f"final_shape={final_shape}"
            )

            return result

        except Exception as e:
            self.logger.error(f"Composite resampling failed: {e}")
            # Clean up temporary file if it exists
            if 'intermediate_path' in locals() and intermediate_path is not None:
                if intermediate_path.exists():
                    intermediate_path.unlink()
            raise RuntimeError(f"Composite resampling failed: {e}") from e

    def visualize(
        self,
        before_path: Path,
        after_path: Path,
        output_path: Path,
        **kwargs: Any
    ) -> None:
        """Generate adaptive visualization based on stages applied.

        Creates either 2-row or 3-row visualization:
        - 2-row: If only one stage applied (original → final)
        - 3-row: If both stages applied (original → intermediate → final)

        Each row shows: axial, sagittal, coronal views
        Bottom: Intensity histogram overlay
        Footer: Metadata with spacing and shape information

        Args:
            before_path: Path to input file (before resampling)
            after_path: Path to output file (after resampling)
            output_path: Path to save visualization (PNG)
            **kwargs: Additional parameters:
                - 'original_spacing': Original voxel spacing
                - 'target_spacing': Target voxel spacing
                - 'original_shape': Original image shape
                - 'resampled_shape': Resampled image shape
                - 'strategy': Strategy dictionary
                - 'intermediate_spacing': Intermediate spacing (if applicable)
                - 'intermediate_shape': Intermediate shape (if applicable)
                - 'intermediate_path': Path to intermediate file (if applicable)
                - 'stages_applied': List of stages applied

        Raises:
            FileNotFoundError: If input files do not exist
            RuntimeError: If visualization generation fails
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        original_spacing = kwargs.get("original_spacing")
        target_spacing = kwargs.get("target_spacing")
        original_shape = kwargs.get("original_shape")
        resampled_shape = kwargs.get("resampled_shape")
        strategy = kwargs.get("strategy", {})
        intermediate_spacing = kwargs.get("intermediate_spacing")
        intermediate_shape = kwargs.get("intermediate_shape")
        intermediate_path = kwargs.get("intermediate_path")
        stages_applied = kwargs.get("stages_applied", [])

        self.logger.info(f"Generating Composite visualization: {output_path}")

        try:
            # Load original and final images
            before_img = nib.load(str(before_path))
            before_data = before_img.get_fdata()

            after_img = nib.load(str(after_path))
            after_data = after_img.get_fdata()

            # Determine if we need 3-row visualization
            show_intermediate = (
                len(stages_applied) == 2 and
                intermediate_path is not None and
                Path(intermediate_path).exists()
            )

            # Load intermediate image if needed
            if show_intermediate:
                intermediate_img = nib.load(str(intermediate_path))
                intermediate_data = intermediate_img.get_fdata()
            else:
                intermediate_data = None

            # Determine number of rows (2 or 3 for images, +1 for histogram)
            n_image_rows = 3 if show_intermediate else 2
            n_total_rows = n_image_rows + 1

            # Create figure
            fig = plt.figure(figsize=(18, 6 * n_image_rows))
            gs = fig.add_gridspec(n_total_rows, 3, hspace=0.3, wspace=0.2)

            fig.suptitle(
                f'Composite Resampling: {before_path.stem}',
                fontsize=16,
                fontweight='bold'
            )

            # Compute shared intensity range
            vmin = before_data.min()
            vmax = before_data.max()
            if show_intermediate:
                vmin = min(vmin, intermediate_data.min(), after_data.min())
                vmax = max(vmax, intermediate_data.max(), after_data.max())
            else:
                vmin = min(vmin, after_data.min())
                vmax = max(vmax, after_data.max())

            # Helper function to get middle slices
            def get_slices(data):
                mid_z = data.shape[2] // 2
                mid_x = data.shape[0] // 2
                mid_y = data.shape[1] // 2
                return (
                    data[:, :, mid_z].T,  # Axial
                    data[mid_x, :, :].T,  # Sagittal
                    data[:, mid_y, :].T   # Coronal
                )

            # Get slices for all images
            axial_before, sagittal_before, coronal_before = get_slices(before_data)
            axial_after, sagittal_after, coronal_after = get_slices(after_data)

            if show_intermediate:
                axial_inter, sagittal_inter, coronal_inter = get_slices(intermediate_data)

            # Row 0: Original image
            ax00 = fig.add_subplot(gs[0, 0])
            ax00.imshow(axial_before, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
            ax00.set_title('Original - Axial', fontsize=12)
            ax00.axis('off')

            ax01 = fig.add_subplot(gs[0, 1])
            ax01.imshow(sagittal_before, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
            ax01.set_title('Original - Sagittal', fontsize=12)
            ax01.axis('off')

            ax02 = fig.add_subplot(gs[0, 2])
            ax02.imshow(coronal_before, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
            ax02.set_title('Original - Coronal', fontsize=12)
            ax02.axis('off')

            # Row 1: Intermediate or Final image
            if show_intermediate:
                # Row 1: Intermediate (after interpolation)
                ax10 = fig.add_subplot(gs[1, 0])
                ax10.imshow(axial_inter, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
                ax10.set_title('After Interpolation - Axial', fontsize=12)
                ax10.axis('off')

                ax11 = fig.add_subplot(gs[1, 1])
                ax11.imshow(sagittal_inter, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
                ax11.set_title('After Interpolation - Sagittal', fontsize=12)
                ax11.axis('off')

                ax12 = fig.add_subplot(gs[1, 2])
                ax12.imshow(coronal_inter, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
                ax12.set_title('After Interpolation - Coronal', fontsize=12)
                ax12.axis('off')

                # Row 2: Final (after DL)
                ax20 = fig.add_subplot(gs[2, 0])
                ax20.imshow(axial_after, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
                ax20.set_title('Final (After DL) - Axial', fontsize=12)
                ax20.axis('off')

                ax21 = fig.add_subplot(gs[2, 1])
                ax21.imshow(sagittal_after, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
                ax21.set_title('Final (After DL) - Sagittal', fontsize=12)
                ax21.axis('off')

                ax22 = fig.add_subplot(gs[2, 2])
                ax22.imshow(coronal_after, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
                ax22.set_title('Final (After DL) - Coronal', fontsize=12)
                ax22.axis('off')

            else:
                # Row 1: Final only
                ax10 = fig.add_subplot(gs[1, 0])
                ax10.imshow(axial_after, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
                ax10.set_title('Final - Axial', fontsize=12)
                ax10.axis('off')

                ax11 = fig.add_subplot(gs[1, 1])
                ax11.imshow(sagittal_after, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
                ax11.set_title('Final - Sagittal', fontsize=12)
                ax11.axis('off')

                ax12 = fig.add_subplot(gs[1, 2])
                ax12.imshow(coronal_after, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
                ax12.set_title('Final - Coronal', fontsize=12)
                ax12.axis('off')

            # Last row: Histogram overlay (spans all 3 columns)
            ax_hist = fig.add_subplot(gs[n_image_rows, :])

            # Flatten data for histogram (remove zeros)
            before_nonzero = before_data.flatten()
            before_nonzero = before_nonzero[before_nonzero > 0]

            after_nonzero = after_data.flatten()
            after_nonzero = after_nonzero[after_nonzero > 0]

            # Compute histogram range
            hist_range = (
                min(before_nonzero.min(), after_nonzero.min()),
                max(before_nonzero.max(), after_nonzero.max())
            )

            # Plot histograms
            bins = 100
            ax_hist.hist(
                before_nonzero,
                bins=bins,
                range=hist_range,
                alpha=0.5,
                label='Original',
                color='blue',
                density=True
            )

            if show_intermediate:
                intermediate_nonzero = intermediate_data.flatten()
                intermediate_nonzero = intermediate_nonzero[intermediate_nonzero > 0]
                ax_hist.hist(
                    intermediate_nonzero,
                    bins=bins,
                    range=hist_range,
                    alpha=0.5,
                    label='After Interpolation',
                    color='green',
                    density=True
                )

            ax_hist.hist(
                after_nonzero,
                bins=bins,
                range=hist_range,
                alpha=0.5,
                label='Final',
                color='red',
                density=True
            )

            ax_hist.set_xlabel('Intensity', fontsize=11)
            ax_hist.set_ylabel('Density', fontsize=11)
            ax_hist.set_title('Intensity Distribution Comparison (non-zero voxels)', fontsize=12)
            ax_hist.legend(fontsize=10)
            ax_hist.grid(True, alpha=0.3)

            # Add metadata text
            metadata_lines = [
                f"Original:",
                f"  Spacing: {original_spacing}",
                f"  Shape: {original_shape}",
            ]

            if show_intermediate:
                metadata_lines.extend([
                    f"",
                    f"After Interpolation:",
                    f"  Spacing: {intermediate_spacing}",
                    f"  Shape: {intermediate_shape}",
                ])

            metadata_lines.extend([
                f"",
                f"Final:",
                f"  Spacing: {target_spacing}",
                f"  Shape: {resampled_shape}",
                f"",
                f"Composite Config:",
                f"  Stages: {' → '.join(stages_applied)}",
                f"  max_mm_interpolator: {self.max_mm_interpolator}",
                f"  max_mm_dl_method: {self.max_mm_dl_method}",
            ])

            metadata_text = "\n".join(metadata_lines)

            fig.text(
                0.5, 0.01,
                metadata_text,
                ha='center',
                fontsize=9,
                family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            )

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save figure
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

            self.logger.info(f"Composite visualization saved to {output_path}")

        except Exception as e:
            self.logger.error(f"Composite visualization generation failed: {e}")
            raise RuntimeError(f"Composite visualization failed: {e}") from e
