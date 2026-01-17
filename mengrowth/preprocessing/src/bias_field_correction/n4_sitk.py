"""N4 bias field correction using SimpleITK.

This module implements N4 bias field correction following the OOP design pattern
specified in CLAUDE.md. It wraps SimpleITK's N4BiasFieldCorrectionImageFilter
with convergence monitoring and comprehensive visualization.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple
import logging
import tempfile

import SimpleITK as sitk
import nibabel as nib
import numpy as np

from mengrowth.preprocessing.src.bias_field_correction.base import BaseBiasFieldCorrector

logger = logging.getLogger(__name__)


class N4ConvergenceMonitor:
    """Command callback to track iteration-level N4 convergence by resolution level."""

    def __init__(self, n4_filter: sitk.N4BiasFieldCorrectionImageFilter) -> None:
        """Initialize convergence monitor.

        Args:
            n4_filter: N4 filter instance to monitor
        """
        self.n4_filter = n4_filter
        # Store data per level: {level: [(iteration, convergence), ...]}
        self.level_data: Dict[int, List[Tuple[int, float]]] = {}

    def __call__(self) -> None:
        """Callback executed at each iteration."""
        iteration = self.n4_filter.GetElapsedIterations()
        level = self.n4_filter.GetCurrentLevel()
        conv = self.n4_filter.GetCurrentConvergenceMeasurement()

        if level not in self.level_data:
            self.level_data[level] = []

        self.level_data[level].append((iteration, conv))


class N4BiasFieldCorrector(BaseBiasFieldCorrector):
    """N4 bias field correction using SimpleITK's N4BiasFieldCorrectionImageFilter.

    This corrector estimates and removes intensity non-uniformities using the
    N4 algorithm (Tustison et al., 2010) with multi-resolution processing and
    convergence monitoring.

    Reference:
        Tustison NJ, et al. "N4ITK: Improved N3 Bias Correction" IEEE TMI 2010.
    """

    def __init__(self, config: Dict[str, Any], verbose: bool = False) -> None:
        """Initialize N4 bias field corrector.

        Args:
            config: Configuration dictionary containing:
                - shrink_factor: Downsampling factor for speed (int, default=4)
                - max_iterations: Iterations per level (List[int], default=[50,50,50,50])
                - bias_field_fwhm: FWHM for Gaussian smoothing (float, default=0.15)
                - convergence_threshold: Early stopping threshold (float, default=0.001)
            verbose: Enable verbose logging
        """
        super().__init__(config=config, verbose=verbose)

        # Extract parameters with defaults
        self.shrink_factor = config.get("shrink_factor", 4)
        self.max_iterations = config.get("max_iterations", [50, 50, 50, 50])
        self.bias_field_fwhm = config.get("bias_field_fwhm", 0.15)
        self.convergence_threshold = config.get("convergence_threshold", 0.001)

        self.logger.info(
            f"Initialized N4BiasFieldCorrector: shrink_factor={self.shrink_factor}, "
            f"max_iterations={self.max_iterations}, fwhm={self.bias_field_fwhm}"
        )

    def execute(
        self,
        input_path: Path,
        output_path: Path,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Execute N4 bias field correction.

        Args:
            input_path: Path to input NIfTI file
            output_path: Path to output corrected NIfTI file
            **kwargs: Additional parameters:
                - allow_overwrite: Allow overwriting existing files (bool)
                - bias_field_output_path: Optional path to save bias field (Path)

        Returns:
            Dictionary containing:
                - 'bias_field_path': Path to saved bias field NIfTI
                - 'convergence_data': Dict mapping level to [(iter, conv), ...]

        Raises:
            FileNotFoundError: If input file does not exist
            RuntimeError: If correction fails
        """
        allow_overwrite = kwargs.get("allow_overwrite", False)
        bias_field_output_path = kwargs.get("bias_field_output_path", None)

        # Validate inputs
        self.validate_inputs(input_path)
        self.validate_outputs(output_path, allow_overwrite=allow_overwrite)

        # Log execution
        self.log_execution(input_path, output_path)

        try:
            # Load NIfTI with SimpleITK
            self.logger.debug(f"Loading image: {input_path}")
            image_sitk = sitk.ReadImage(str(input_path))

            # Convert to float32 if needed
            if image_sitk.GetPixelID() != sitk.sitkFloat32:
                self.logger.debug("Converting image to float32")
                image_sitk = sitk.Cast(image_sitk, sitk.sitkFloat32)

            # Initialize N4 filter
            n4_filter = sitk.N4BiasFieldCorrectionImageFilter()
            n4_filter.SetMaximumNumberOfIterations(self.max_iterations)
            n4_filter.SetBiasFieldFullWidthAtHalfMaximum(self.bias_field_fwhm)
            n4_filter.SetConvergenceThreshold(self.convergence_threshold)

            # Set number of control points (4 per dimension by default)
            n4_filter.SetNumberOfControlPoints([4] * image_sitk.GetDimension())

            # Attach convergence monitor
            monitor = N4ConvergenceMonitor(n4_filter)
            n4_filter.AddCommand(sitk.sitkIterationEvent, monitor)

            self.logger.info(
                f"Running N4 correction: shrink_factor={self.shrink_factor}, "
                f"max_iters={self.max_iterations}"
            )

            # Shrink image for computational efficiency
            if self.shrink_factor > 1:
                self.logger.debug(f"Shrinking image by factor {self.shrink_factor}")
                shrunk_img = sitk.Shrink(
                    image_sitk,
                    [self.shrink_factor] * image_sitk.GetDimension()
                )
            else:
                shrunk_img = image_sitk

            # Execute N4 on shrunk image (no mask)
            self.logger.debug("Executing N4 filter...")
            _ = n4_filter.Execute(shrunk_img)

            # Extract log bias field at full resolution
            self.logger.debug("Extracting bias field at full resolution")
            log_bias_field = n4_filter.GetLogBiasFieldAsImage(image_sitk)
            bias_field = sitk.Exp(log_bias_field)

            # Compute corrected image
            corrected = image_sitk / bias_field

            # Save corrected image
            self.logger.debug(f"Saving corrected image: {output_path}")
            sitk.WriteImage(corrected, str(output_path))

            # Save bias field
            if bias_field_output_path is None:
                # Create temporary file for bias field
                bias_field_output_path = output_path.parent / f"{output_path.stem}_bias_field.nii.gz"

            self.logger.debug(f"Saving bias field: {bias_field_output_path}")
            sitk.WriteImage(bias_field, str(bias_field_output_path))

            # Log convergence info
            final_level = n4_filter.GetCurrentLevel()
            final_conv = n4_filter.GetCurrentConvergenceMeasurement()
            self.logger.info(
                f"N4 completed: final_level={final_level}, "
                f"final_convergence={final_conv:.6f}"
            )

            # Compute bias field statistics for QC
            bias_field_arr = sitk.GetArrayFromImage(bias_field)
            image_arr = sitk.GetArrayFromImage(image_sitk)

            # Compute statistics only in non-zero (brain) region
            nonzero_mask = image_arr > 0
            if np.any(nonzero_mask):
                bias_in_brain = bias_field_arr[nonzero_mask]
                # Mean deviation from 1.0 (ideal bias field value)
                bias_field_magnitude_mean = float(np.mean(np.abs(bias_in_brain - 1.0)))
                bias_field_range = float(np.max(bias_in_brain) - np.min(bias_in_brain))
            else:
                bias_field_magnitude_mean = 0.0
                bias_field_range = 0.0

            # Count total iterations across all levels
            n_iterations_total = sum(
                len(level_data) for level_data in monitor.level_data.values()
            )

            # Check if convergence was achieved (final convergence < threshold)
            convergence_achieved = final_conv < self.convergence_threshold

            return {
                "bias_field_path": bias_field_output_path,
                "convergence_data": monitor.level_data,
                # QC-relevant intermediate metrics
                "bias_field_magnitude_mean": bias_field_magnitude_mean,
                "bias_field_range": bias_field_range,
                "n_iterations_total": n_iterations_total,
                "convergence_achieved": convergence_achieved,
                "final_convergence_value": float(final_conv),
            }

        except Exception as e:
            self.logger.error(f"N4 bias field correction failed: {e}")
            raise RuntimeError(f"N4 correction failed: {e}") from e

    def visualize(
        self,
        before_path: Path,
        after_path: Path,
        output_path: Path,
        **kwargs: Any
    ) -> None:
        """Generate 4-column visualization of bias field correction.

        Creates visualization with:
        1. Original image (middle axial slice)
        2. Original with bias field overlay (alpha=0.5)
        3. Bias-field-corrected image
        4. Convergence monitoring plot

        Args:
            before_path: Path to input file (before correction)
            after_path: Path to output file (after correction)
            output_path: Path to save visualization (PNG)
            **kwargs: Additional parameters:
                - bias_field_path: Path to bias field file (required)
                - convergence_data: Convergence monitoring data (required)

        Raises:
            FileNotFoundError: If input files do not exist
            RuntimeError: If visualization generation fails
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        bias_field_path = kwargs.get("bias_field_path")
        convergence_data = kwargs.get("convergence_data")

        if bias_field_path is None:
            raise ValueError("bias_field_path is required for visualization")
        if convergence_data is None:
            raise ValueError("convergence_data is required for visualization")

        self.logger.info(f"Generating bias field correction visualization: {output_path}")

        try:
            # Load images
            before_img = nib.load(str(before_path))
            before_data = before_img.get_fdata()

            after_img = nib.load(str(after_path))
            after_data = after_img.get_fdata()

            bias_field_img = nib.load(str(bias_field_path))
            bias_field_data = bias_field_img.get_fdata()

            # Get middle axial slice
            mid_z = before_data.shape[2] // 2

            before_slice = before_data[:, :, mid_z].T
            after_slice = after_data[:, :, mid_z].T
            bias_slice = bias_field_data[:, :, mid_z].T

            # Create figure: 1 row x 4 columns
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            fig.suptitle(
                f'N4 Bias Field Correction: {before_path.stem}',
                fontsize=16,
                fontweight='bold'
            )

            # Column 1: Original image
            im1 = axes[0].imshow(before_slice, cmap='gray', origin='lower')
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

            # Column 2: Original with bias field overlay (alpha=0.5)
            axes[1].imshow(before_slice, cmap='gray', origin='lower', alpha=1.0)
            im2 = axes[1].imshow(bias_slice, cmap='hot', origin='lower', alpha=0.5)
            axes[1].set_title('Original + Bias Field Overlay')
            axes[1].axis('off')
            plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04, label='Bias Field')

            # Column 3: Corrected image
            im3 = axes[2].imshow(after_slice, cmap='gray', origin='lower')
            axes[2].set_title('Bias-Field-Corrected')
            axes[2].axis('off')
            plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

            # Column 4: Convergence monitoring
            for level, data in convergence_data.items():
                if data:
                    iterations, convergences = zip(*data)
                    axes[3].plot(iterations, convergences, marker='o', label=f'Level {level}')

            axes[3].set_xlabel('Iteration')
            axes[3].set_ylabel('Convergence Measure')
            axes[3].set_title('Convergence Monitoring')
            axes[3].legend()
            axes[3].grid(True, alpha=0.3)
            axes[3].set_yscale('log')

            plt.tight_layout()

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save figure
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

            self.logger.info(f"Visualization saved to {output_path}")

        except Exception as e:
            self.logger.error(f"Visualization generation failed: {e}")
            raise RuntimeError(f"Visualization failed: {e}") from e
