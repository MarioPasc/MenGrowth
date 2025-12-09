"""SynthStrip skull stripping implementation.

This module provides skull stripping using the SynthStrip algorithm
from the brainles_preprocessing package (nipreps/synthstrip).
"""

from pathlib import Path
from typing import Dict, Any
import logging
import numpy as np
import nibabel as nib
import torch

from mengrowth.preprocessing.src.skull_stripping.base import BaseSkullStripper

logger = logging.getLogger(__name__)


class SynthStripSkullStripper(BaseSkullStripper):
    """SynthStrip brain extraction implementation.

    Uses SynthStrip algorithm which performs surface distance transform (SDT)
    prediction using deep learning. Requires conformation to model input space.

    Note:
        This is an optional dependency. Install with:
        pip install brainles_preprocessing[synthstrip]

    References:
        Hoopes A, et al. "SynthStrip: Skull-Stripping for Any Brain Image."
        NeuroImage 2022.
    """

    def __init__(self, config: Dict[str, Any], verbose: bool = False) -> None:
        """Initialize SynthStrip skull stripper.

        Args:
            config: Configuration dictionary containing:
                - border: Border parameter in mm (int, default=1)
                - device: Device for computation (int for GPU id or 'cpu', default=0)
                - fill_value: Background fill value (float, default=0.0)
            verbose: Enable verbose logging
        """
        super().__init__(config=config, verbose=verbose)

        # Extract parameters with defaults
        self.border = config.get("border", 1)
        self.device = config.get("device", 0)

        # Validate border
        if self.border < 0:
            raise ValueError(f"border must be non-negative, got {self.border}")

        self.logger.info(
            f"Initialized SynthStripSkullStripper: border={self.border}, "
            f"device={self.device}, fill_value={self.fill_value}"
        )

    def execute(
        self,
        input_path: Path,
        output_path: Path,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Execute SynthStrip skull stripping.

        Args:
            input_path: Path to input NIfTI file
            output_path: Path to output brain-extracted NIfTI file
            **kwargs: Additional parameters:
                - mask_path: Path where brain mask should be saved (required)
                - allow_overwrite: Allow overwriting existing files (bool)

        Returns:
            Dictionary containing:
                - 'mask_path': Path to saved brain mask NIfTI
                - 'algorithm': 'synthstrip'
                - 'parameters': Dict of parameters used
                - 'brain_volume_mm3': Volume of extracted brain in mm³
                - 'brain_coverage_percent': Percentage of original volume kept

        Raises:
            FileNotFoundError: If input file does not exist
            RuntimeError: If skull stripping fails or GPU unavailable
            ImportError: If brainles_preprocessing[synthstrip] not installed
        """
        allow_overwrite = kwargs.get("allow_overwrite", False)
        mask_path = kwargs.get("mask_path", None)

        if mask_path is None:
            raise ValueError("mask_path must be provided in kwargs")

        mask_path = Path(mask_path)

        # Validate inputs
        self.validate_inputs(input_path)
        self.validate_outputs(output_path, allow_overwrite=allow_overwrite)

        # Log execution
        self.log_execution(input_path, output_path)

        try:
            # Import SynthStrip
            try:
                from brainles_preprocessing.brain_extraction.synthstrip import SynthStripExtractor
            except ImportError as e:
                raise ImportError(
                    "SynthStrip dependencies not found. Install with:\n"
                    "  pip install brainles_preprocessing[synthstrip]\n"
                    f"Original error: {e}"
                ) from e

            # Validate GPU availability if GPU specified
            if isinstance(self.device, int):
                if not torch.cuda.is_available():
                    raise RuntimeError(
                        f"GPU {self.device} requested but CUDA is not available. "
                        f"Either install CUDA-enabled PyTorch or set device='cpu'"
                    )
                if self.device >= torch.cuda.device_count():
                    raise RuntimeError(
                        f"GPU {self.device} requested but only {torch.cuda.device_count()} "
                        f"GPU(s) available (0-{torch.cuda.device_count()-1})"
                    )

            # Create temporary output path for SynthStrip
            temp_output = output_path.parent / f"_temp_{output_path.name}"

            # Ensure mask output directory exists
            mask_path.parent.mkdir(parents=True, exist_ok=True)

            self.logger.info(
                f"Running SynthStrip: border={self.border}, device={self.device}"
            )

            # Instantiate and run SynthStrip
            extractor = SynthStripExtractor(border=self.border)

            # Convert device to torch.device if needed
            if isinstance(self.device, int):
                device_obj = torch.device(f"cuda:{self.device}")
            else:
                device_obj = torch.device("cpu")

            # SynthStrip extract method signature: (input_path, masked_output_path, mask_path, device)
            # Note: Need to check actual API
            extractor.extract(
                input_image_path=input_path,
                masked_image_path=temp_output,
                brain_mask_path=mask_path,
                device=device_obj
            )

            # Load images for statistics and custom fill_value application
            self.logger.debug("Loading images for statistics computation")
            input_img = nib.load(str(input_path))
            masked_img = nib.load(str(temp_output))
            mask_img = nib.load(str(mask_path))

            input_data = input_img.get_fdata()
            masked_data = masked_img.get_fdata()
            mask_data = mask_img.get_fdata()

            # Compute brain volume statistics
            voxel_volume_mm3 = np.prod(input_img.header.get_zooms())
            brain_voxels = np.sum(mask_data > 0)
            brain_volume_mm3 = brain_voxels * voxel_volume_mm3

            # Compute coverage percentage (brain volume / non-zero original volume)
            original_nonzero = np.sum(input_data > 0)
            if original_nonzero > 0:
                brain_coverage_percent = (brain_voxels / original_nonzero) * 100.0
            else:
                brain_coverage_percent = 0.0

            self.logger.info(
                f"Brain statistics: volume={brain_volume_mm3:.1f} mm³, "
                f"coverage={brain_coverage_percent:.1f}%"
            )

            # Apply custom fill_value if different from 0
            if abs(self.fill_value) > 1e-6:
                self.logger.debug(f"Applying custom fill_value={self.fill_value}")
                final_data = np.where(mask_data > 0, input_data, self.fill_value)
            else:
                # Use SynthStrip output directly (already has 0 background)
                final_data = masked_data

            # Save final output
            self.logger.debug(f"Saving skull-stripped image: {output_path}")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            final_img = nib.Nifti1Image(final_data, input_img.affine, input_img.header)
            nib.save(final_img, str(output_path))

            # Clean up temporary file
            if temp_output.exists():
                temp_output.unlink()

            self.logger.info("SynthStrip skull stripping completed successfully")

            return {
                "mask_path": mask_path,
                "algorithm": "synthstrip",
                "parameters": {
                    "border": self.border,
                    "device": str(self.device),
                    "fill_value": self.fill_value
                },
                "brain_volume_mm3": float(brain_volume_mm3),
                "brain_coverage_percent": float(brain_coverage_percent)
            }

        except Exception as e:
            self.logger.error(f"SynthStrip skull stripping failed: {e}")
            raise RuntimeError(f"SynthStrip skull stripping failed: {e}") from e

    def visualize(
        self,
        before_path: Path,
        after_path: Path,
        output_path: Path,
        **kwargs: Any
    ) -> None:
        """Generate 3×4 visualization of skull stripping results.

        Creates visualization with:
        - Row 1 (axial): Original | Mask overlay | Skull-stripped | Histogram
        - Row 2 (sagittal): Original | Mask overlay | Skull-stripped | Histogram
        - Row 3 (coronal): Original | Mask overlay | Skull-stripped | Histogram

        Args:
            before_path: Path to input file (before skull stripping)
            after_path: Path to output file (after skull stripping)
            output_path: Path to save visualization output (PNG)
            **kwargs: Visualization parameters:
                - mask_path: Path to brain mask file (required)
                - algorithm: Algorithm name
                - parameters: Parameters used
                - brain_volume_mm3: Brain volume
                - brain_coverage_percent: Coverage percentage

        Raises:
            FileNotFoundError: If input files do not exist
            RuntimeError: If visualization generation fails
        """
        try:
            import matplotlib
            matplotlib.use('Agg')  # Headless rendering
            import matplotlib.pyplot as plt
            from matplotlib.colors import ListedColormap

            # Extract metadata
            mask_path = kwargs.get("mask_path")
            if mask_path is None:
                raise ValueError("mask_path must be provided in kwargs")

            algorithm = kwargs.get("algorithm", "synthstrip")
            parameters = kwargs.get("parameters", {})
            brain_volume_mm3 = kwargs.get("brain_volume_mm3", 0.0)
            brain_coverage_percent = kwargs.get("brain_coverage_percent", 0.0)

            # Load images
            self.logger.debug(f"Loading images for visualization")
            original_img = nib.load(str(before_path))
            skull_stripped_img = nib.load(str(after_path))
            mask_img = nib.load(str(mask_path))

            original_data = original_img.get_fdata()
            skull_stripped_data = skull_stripped_img.get_fdata()
            mask_data = mask_img.get_fdata()

            # Get middle slices for each view
            shape = original_data.shape
            axial_slice = shape[2] // 2
            sagittal_slice = shape[0] // 2
            coronal_slice = shape[1] // 2

            # Create figure with 3 rows × 4 columns
            fig, axes = plt.subplots(3, 4, figsize=(20, 15))

            # Define views and slices
            views = [
                ("Axial", original_data[:, :, axial_slice], skull_stripped_data[:, :, axial_slice], mask_data[:, :, axial_slice]),
                ("Sagittal", original_data[sagittal_slice, :, :], skull_stripped_data[sagittal_slice, :, :], mask_data[sagittal_slice, :, :]),
                ("Coronal", original_data[:, coronal_slice, :], skull_stripped_data[:, coronal_slice, :], mask_data[:, coronal_slice, :])
            ]

            for row, (view_name, orig_slice, stripped_slice, mask_slice) in enumerate(views):
                # Column 0: Original
                axes[row, 0].imshow(orig_slice.T, cmap='gray', origin='lower')
                axes[row, 0].set_title(f'{view_name}: Original')
                axes[row, 0].axis('off')

                # Column 1: Mask overlay
                axes[row, 1].imshow(orig_slice.T, cmap='gray', origin='lower')
                # Create red overlay for mask
                red_cmap = ListedColormap(['none', 'red'])
                axes[row, 1].imshow(mask_slice.T, cmap=red_cmap, alpha=0.5, origin='lower')
                axes[row, 1].set_title(f'{view_name}: Mask Overlay')
                axes[row, 1].axis('off')

                # Column 2: Skull-stripped
                axes[row, 2].imshow(stripped_slice.T, cmap='gray', origin='lower')
                axes[row, 2].set_title(f'{view_name}: Skull-stripped')
                axes[row, 2].axis('off')

                # Column 3: Histogram (only for first row)
                if row == 0:
                    # Compute histograms for non-zero voxels
                    orig_nonzero = original_data[original_data > 0]
                    stripped_nonzero = skull_stripped_data[skull_stripped_data > 0]

                    axes[row, 3].hist(orig_nonzero, bins=50, alpha=0.7, color='blue', label='Original', density=True)
                    axes[row, 3].hist(stripped_nonzero, bins=50, alpha=0.7, color='orange', label='Skull-stripped', density=True)
                    axes[row, 3].set_xlabel('Intensity')
                    axes[row, 3].set_ylabel('Density')
                    axes[row, 3].set_title('Intensity Distribution')
                    axes[row, 3].legend()
                    axes[row, 3].grid(True, alpha=0.3)
                else:
                    # Hide histogram for other rows
                    axes[row, 3].axis('off')

            # Add metadata text box at bottom
            metadata_text = (
                f"Algorithm: {algorithm.upper()} | "
                f"Border: {parameters.get('border', 'N/A')} mm | "
                f"Device: {parameters.get('device', 'N/A')}\n"
                f"Brain Volume: {brain_volume_mm3:.1f} mm³ | "
                f"Brain Coverage: {brain_coverage_percent:.1f}% of original | "
                f"Fill Value: {parameters.get('fill_value', 0.0)}"
            )
            fig.text(0.5, 0.02, metadata_text, ha='center', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            # Adjust layout and save
            plt.tight_layout(rect=[0, 0.04, 1, 1])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

            self.logger.info(f"Visualization saved: {output_path}")

        except Exception as e:
            self.logger.error(f"Visualization generation failed: {e}")
            raise RuntimeError(f"Visualization generation failed: {e}") from e
