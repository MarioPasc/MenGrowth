"""Format conversion operations for data harmonization.

This module implements format converters that wrap existing utilities
in mengrowth.preprocessing.src.utils following the OOP design pattern.
"""

from pathlib import Path
from typing import Any
import logging
import nibabel as nib
import numpy as np

from mengrowth.preprocessing.src.data_harmonization.base import BaseConverter
from mengrowth.preprocessing.src.utils.nrrd_to_nifti import nifti_write_3d

logger = logging.getLogger(__name__)


class NRRDtoNIfTIConverter(BaseConverter):
    """Convert NRRD files to NIfTI format.

    This converter wraps the existing nifti_write_3d utility and provides
    OOP interface with visualization capabilities. It preserves all medical
    imaging metadata and performs orientation flips to RAS.
    """

    def __init__(self, verbose: bool = False) -> None:
        """Initialize NRRD to NIfTI converter.

        Args:
            verbose: Enable verbose logging
        """
        super().__init__(verbose=verbose)
        self.logger.info("Initialized NRRDtoNIfTIConverter")

    def execute(
        self,
        input_path: Path,
        output_path: Path,
        **kwargs: Any
    ) -> None:
        """Convert NRRD file to NIfTI format.

        Args:
            input_path: Path to input NRRD file
            output_path: Path to output NIfTI file (.nii.gz)
            **kwargs: Additional parameters (allow_overwrite)

        Raises:
            FileNotFoundError: If input file does not exist
            ValueError: If input is not 3D NRRD data
            RuntimeError: If conversion fails
        """
        allow_overwrite = kwargs.get("allow_overwrite", False)

        # Validate inputs
        self.validate_inputs(input_path)
        self.validate_outputs(output_path, allow_overwrite=allow_overwrite)

        # Log execution
        self.log_execution(input_path, output_path)

        try:
            # Call the existing utility
            nifti_write_3d(
                volume=str(input_path),
                out_file=str(output_path),
                verbose=self.verbose
            )

            self.logger.info(
                f"Successfully converted {input_path.name} to {output_path.name}"
            )

        except Exception as e:
            self.logger.error(f"Conversion failed: {e}")
            raise RuntimeError(f"NRRD to NIfTI conversion failed: {e}") from e

    def visualize(
        self,
        before_path: Path,
        after_path: Path,
        output_path: Path,
        **kwargs: Any
    ) -> None:
        """Generate visualization comparing NRRD and NIfTI formats.

        For format conversion, visualization shows that the data and
        orientation are preserved. Displays axial, sagittal, and coronal
        mid-slices from both formats.

        Args:
            before_path: Path to input NRRD file
            after_path: Path to output NIfTI file
            output_path: Path to save visualization (PNG)
            **kwargs: Additional visualization parameters

        Raises:
            FileNotFoundError: If input files do not exist
            RuntimeError: If visualization generation fails
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import nrrd

        self.logger.info(f"Generating conversion visualization: {output_path}")

        try:
            # Load NRRD (before)
            nrrd_data, nrrd_header = nrrd.read(str(before_path))

            # Load NIfTI (after)
            nifti_img = nib.load(str(after_path))
            nifti_data = nifti_img.get_fdata()

            # Get mid-slices for each orientation
            mid_x = nrrd_data.shape[0] // 2
            mid_y = nrrd_data.shape[1] // 2
            mid_z = nrrd_data.shape[2] // 2

            # Create figure with 2 rows (NRRD, NIfTI) x 3 columns (axial, sagittal, coronal)
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle(f'Format Conversion: {before_path.stem}', fontsize=16, fontweight='bold')

            # NRRD slices (row 0)
            axes[0, 0].imshow(nrrd_data[:, :, mid_z].T, cmap='gray', origin='lower')
            axes[0, 0].set_title('NRRD - Axial')
            axes[0, 0].axis('off')

            axes[0, 1].imshow(nrrd_data[:, mid_y, :].T, cmap='gray', origin='lower')
            axes[0, 1].set_title('NRRD - Sagittal')
            axes[0, 1].axis('off')

            axes[0, 2].imshow(nrrd_data[mid_x, :, :].T, cmap='gray', origin='lower')
            axes[0, 2].set_title('NRRD - Coronal')
            axes[0, 2].axis('off')

            # NIfTI slices (row 1)
            axes[1, 0].imshow(nifti_data[:, :, mid_z].T, cmap='gray', origin='lower')
            axes[1, 0].set_title('NIfTI - Axial')
            axes[1, 0].axis('off')

            axes[1, 1].imshow(nifti_data[:, mid_y, :].T, cmap='gray', origin='lower')
            axes[1, 1].set_title('NIfTI - Sagittal')
            axes[1, 1].axis('off')

            axes[1, 2].imshow(nifti_data[mid_x, :, :].T, cmap='gray', origin='lower')
            axes[1, 2].set_title('NIfTI - Coronal')
            axes[1, 2].axis('off')

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
