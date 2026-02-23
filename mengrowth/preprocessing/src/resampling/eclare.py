"""ECLARE deep learning-based super-resolution resampling.

This module implements ECLARE (Enhanced Contrastive Learning for Anisotropic REsampling),
a deep learning-based super-resolution method for medical imaging. It wraps the ECLARE
command-line tool via subprocess execution in a specified conda environment.

ECLARE must be installed in a separate conda environment specified in the configuration.
The method is particularly useful for high-quality super-resolution of anisotropic MRI scans.

Reference:
    Sanchez, I., et al. "ECLARE: Extreme Classification with Label Graph Correlations" (2023).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging
import subprocess
import os
import tempfile

import SimpleITK as sitk
import nibabel as nib
import numpy as np

from mengrowth.preprocessing.src.resampling.base import BaseResampler

logger = logging.getLogger(__name__)


class EclareResampler(BaseResampler):
    """ECLARE deep learning-based super-resolution resampler.

    This resampler uses the ECLARE deep learning model for super-resolution resampling
    of anisotropic medical images. It executes ECLARE via a conda environment subprocess,
    supporting both single and multi-GPU execution for parallel processing.

    The method computes the relative slice thickness from the input volume and passes
    it to ECLARE along with other user-defined parameters.
    """

    def __init__(
        self,
        target_voxel_size: List[float],
        config: Dict[str, Any],
        verbose: bool = False
    ) -> None:
        """Initialize ECLARE resampler.

        Args:
            target_voxel_size: Target voxel size in mm [x, y, z]
            config: Configuration dictionary containing:
                - conda_environment_eclare: Name of conda environment with ECLARE installed
                - batch_size: Batch size for ECLARE inference (default=128)
                - n_patches: Number of patches for training (default=1000000)
                - patch_sampling: Patch sampling strategy (default="gradient")
                - suffix: Suffix to add to output filename (default="")
                - gpu_id: GPU ID(s) to use - int or list of ints (default=0)
            verbose: Enable verbose logging

        Raises:
            ValueError: If required parameters are missing or invalid
        """
        super().__init__(
            target_voxel_size=target_voxel_size,
            config=config,
            verbose=verbose
        )

        # Extract ECLARE-specific parameters with defaults
        self.conda_env = config.get("conda_environment_eclare", "eclare_env")
        self.batch_size = config.get("batch_size", 128)
        self.n_patches = config.get("n_patches", 1000000)
        self.patch_sampling = config.get("patch_sampling", "gradient")
        self.suffix = config.get("suffix", "")
        self.gpu_id = config.get("gpu_id", 0)
        self.eclare_verbose = config.get("verbose", verbose)

        # Validate parameters
        if not isinstance(self.conda_env, str) or not self.conda_env:
            raise ValueError("conda_environment_eclare must be a non-empty string")

        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError(f"batch_size must be a positive integer, got {self.batch_size}")

        if not isinstance(self.n_patches, int) or self.n_patches <= 0:
            raise ValueError(f"n_patches must be a positive integer, got {self.n_patches}")

        if self.patch_sampling not in ["uniform", "gradient", "random"]:
            self.logger.warning(
                f"patch_sampling='{self.patch_sampling}' may not be supported by ECLARE. "
                "Common values are: 'uniform', 'gradient', 'random'"
            )

        # Validate gpu_id (can be int or list of ints)
        if isinstance(self.gpu_id, int):
            if self.gpu_id < 0:
                raise ValueError(f"gpu_id must be non-negative, got {self.gpu_id}")
        elif isinstance(self.gpu_id, list):
            if not all(isinstance(gpu, int) and gpu >= 0 for gpu in self.gpu_id):
                raise ValueError("gpu_id list must contain only non-negative integers")
            if len(self.gpu_id) == 0:
                raise ValueError("gpu_id list cannot be empty")
        else:
            raise ValueError(f"gpu_id must be int or List[int], got {type(self.gpu_id)}")

        self.logger.info(
            f"Initialized EclareResampler: "
            f"conda_env={self.conda_env}, "
            f"batch_size={self.batch_size}, "
            f"n_patches={self.n_patches}, "
            f"patch_sampling={self.patch_sampling}, "
            f"gpu_id={self.gpu_id}, "
            f"target_spacing={target_voxel_size}"
        )

    def _compute_relative_slice_thickness(self, input_path: Path) -> float:
        """Compute relative slice thickness for ECLARE's Gaussian blur kernel.

        This represents the FWHM of the slice profile relative to the minimum
        in-plane resolution.  The through-plane axis is identified dynamically
        as the dimension with the largest spacing — this handles axial, sagittal,
        and coronal acquisitions without assuming Z is always through-plane.
        """
        try:
            image_sitk = sitk.ReadImage(str(input_path))
            spacing = np.array(image_sitk.GetSpacing())

            # Identify worst (largest spacing) dimension — orientation-agnostic
            worst_dim = int(np.argmax(spacing))
            through_plane = spacing[worst_dim]
            in_plane_dims = [i for i in range(3) if i != worst_dim]
            min_in_plane = min(spacing[in_plane_dims[0]], spacing[in_plane_dims[1]])

            # This is the FWHM for the Gaussian kernel
            relative_thickness = through_plane / min_in_plane

            dim_labels = ['X', 'Y', 'Z']
            self.logger.info(
                f"Computed relative slice thickness: {relative_thickness:.3f} "
                f"(through-plane={dim_labels[worst_dim]}={through_plane:.3f}mm / "
                f"min_in_plane={min_in_plane:.3f}mm)"
            )

            return relative_thickness  # Return as float, not rounded int

        except Exception as e:
            self.logger.error(f"Failed to compute relative slice thickness: {e}")
            raise RuntimeError(f"Slice thickness computation failed: {e}") from e
        
    def _run_eclare_subprocess(
        self,
        input_path: Path,
        output_dir: Path,
        relative_slice_thickness: float,
        gpu_id: int,
        inplane_acq_res: Optional[List[float]] = None
    ) -> None:
        """Run ECLARE as a subprocess via conda run.

        Args:
            input_path: Path to input NIfTI file
            output_dir: Directory for ECLARE output
            relative_slice_thickness: Relative slice thickness (float)
            gpu_id: GPU ID to use (single integer)
            inplane_acq_res: In-plane acquisition resolution [dim_a, dim_b] in mm.
                These are the target resolutions for the two non-worst dimensions,
                identified dynamically from the input spacing (orientation-agnostic).
                If None, falls back to target_voxel_size[0:2] (legacy behavior).

        Raises:
            RuntimeError: If ECLARE subprocess fails
        """
        # Determine in-plane acquisition resolution
        if inplane_acq_res is None:
            inplane_acq_res = [self.target_voxel_size[0], self.target_voxel_size[1]]

        # Build ECLARE command
        cmd = [
            "conda", "run", "-n", self.conda_env,
            "run-eclare",
            "--in-fpath", str(input_path),
            "--out-dir", str(output_dir),
            "--batch-size", str(self.batch_size),
            "--n-patches", str(self.n_patches),
            "--inplane-acq-res", f"{inplane_acq_res[0]}", f"{inplane_acq_res[1]}",
            "--patch-sampling", self.patch_sampling,
            #"--relative-slice-thickness", str(relative_slice_thickness),
            "--gpu-id", str(gpu_id)
        ]

        # Add optional suffix if specified
        if self.suffix:
            cmd.extend(["--suffix", self.suffix])

        # Add verbose flag if enabled
        if self.eclare_verbose:
            cmd.append("--verbose")

        self.logger.info(f"Running ECLARE command: {' '.join(cmd)}")

        try:
            # Run subprocess with output capture
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                env=os.environ.copy()
            )

            # Log stdout if available
            if result.stdout:
                self.logger.debug(f"ECLARE stdout:\n{result.stdout}")

            self.logger.info("ECLARE subprocess completed successfully")

        except subprocess.CalledProcessError as e:
            error_msg = (
                f"ECLARE subprocess failed with return code {e.returncode}.\n"
                f"Command: {' '.join(cmd)}\n"
                f"Stderr: {e.stderr}\n"
                f"Stdout: {e.stdout}"
            )
            self.logger.error(error_msg)
            raise RuntimeError(f"ECLARE execution failed: {e.stderr}") from e

        except FileNotFoundError as e:
            self.logger.error(
                f"ECLARE command not found. Ensure ECLARE is installed in conda "
                f"environment '{self.conda_env}' and 'run-eclare' is in PATH."
            )
            raise RuntimeError(
                f"ECLARE not found in conda environment '{self.conda_env}': {e}"
            ) from e

    def execute(
        self,
        input_path: Path,
        output_path: Path,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Execute ECLARE deep learning-based resampling.

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
                - 'resampled_shape': Resampled image shape [x, y, z]

        Raises:
            FileNotFoundError: If input file does not exist
            RuntimeError: If resampling fails

        Note:
            For multi-GPU support (when gpu_id is a list), this method currently
            uses only the first GPU. Full parallelization across multiple GPUs
            should be implemented at the orchestrator level for processing multiple
            modalities simultaneously.
        """
        allow_overwrite = kwargs.get("allow_overwrite", False)

        # Validate inputs
        self.validate_inputs(input_path)
        self.validate_outputs(output_path, allow_overwrite=allow_overwrite)

        # Log execution
        self.log_execution(input_path, output_path)

        try:
            # Load input image to get original metadata
            self.logger.debug(f"Loading image: {input_path}")
            image_sitk = sitk.ReadImage(str(input_path))

            original_spacing = np.array(image_sitk.GetSpacing())
            original_size = np.array(image_sitk.GetSize())

            self.logger.info(
                f"Original spacing: {original_spacing}, shape: {original_size}"
            )

            # Pre-flight: identify through-plane axis (orientation-agnostic).
            # ECLARE requires a unique worst-resolution axis.  Sagittal and
            # coronal acquisitions may have the worst axis on X or Y, not Z.
            worst_dim = int(np.argmax(original_spacing))
            in_plane_dims = [i for i in range(3) if i != worst_dim]
            sorted_spacing = np.sort(original_spacing)
            dim_labels = ['X', 'Y', 'Z']

            if (sorted_spacing[-1] - sorted_spacing[-2]) < 1e-3:
                raise RuntimeError(
                    f"ECLARE requires anisotropic input but spacing "
                    f"{original_spacing.tolist()} has no unique worst-resolution "
                    f"axis (sorted: {sorted_spacing.tolist()}).  Consider using "
                    f"BSpline resampling or the composite method instead."
                )

            self.logger.info(
                f"Through-plane axis: {dim_labels[worst_dim]} "
                f"({original_spacing[worst_dim]:.3f}mm), "
                f"in-plane: {dim_labels[in_plane_dims[0]]}="
                f"{original_spacing[in_plane_dims[0]]:.3f}mm, "
                f"{dim_labels[in_plane_dims[1]]}="
                f"{original_spacing[in_plane_dims[1]]:.3f}mm"
            )

            # Compute in-plane target resolution from the non-worst dimensions
            inplane_acq_res = [
                self.target_voxel_size[in_plane_dims[0]],
                self.target_voxel_size[in_plane_dims[1]]
            ]

            # Compute relative slice thickness
            relative_slice_thickness = self._compute_relative_slice_thickness(input_path)

            # Select GPU (use first one if multiple GPUs specified)
            if isinstance(self.gpu_id, list):
                if len(self.gpu_id) > 1:
                    self.logger.warning(
                        f"Multiple GPUs specified {self.gpu_id}, but single-image "
                        f"execution will use only GPU {self.gpu_id[0]}. For multi-GPU "
                        "parallel processing, use orchestrator-level parallelization."
                    )
                selected_gpu = self.gpu_id[0]
            else:
                selected_gpu = self.gpu_id

            # Create temporary output directory for ECLARE
            # ECLARE outputs to a directory, not a specific file path
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_output_dir = Path(tmp_dir)

                self.logger.info(
                    f"Running ECLARE super-resolution with GPU {selected_gpu}..."
                )

                # Run ECLARE subprocess
                self._run_eclare_subprocess(
                    input_path=input_path,
                    output_dir=tmp_output_dir,
                    relative_slice_thickness=relative_slice_thickness,
                    gpu_id=selected_gpu,
                    inplane_acq_res=inplane_acq_res
                )

                # Find ECLARE output file
                # ECLARE typically outputs: <input_stem><suffix>.nii.gz
                expected_output_name = f"{input_path.stem}{self.suffix}.nii.gz"
                # Remove .nii from stem if present (since stem includes everything before final suffix)
                if input_path.stem.endswith('.nii'):
                    expected_output_name = f"{input_path.stem[:-4]}{self.suffix}.nii.gz"

                eclare_output_path = tmp_output_dir / expected_output_name

                # Search for output file if not found at expected location
                if not eclare_output_path.exists():
                    self.logger.warning(
                        f"Expected output not found at {eclare_output_path}. "
                        "Searching for ECLARE output..."
                    )
                    output_files = list(tmp_output_dir.glob("*.nii.gz"))
                    if len(output_files) == 1:
                        eclare_output_path = output_files[0]
                        self.logger.info(f"Found ECLARE output: {eclare_output_path}")
                    elif len(output_files) == 0:
                        raise RuntimeError(
                            f"No ECLARE output found in {tmp_output_dir}"
                        )
                    else:
                        raise RuntimeError(
                            f"Multiple outputs found in {tmp_output_dir}: {output_files}. "
                            "Cannot determine which is the correct output."
                        )

                # Load ECLARE output to get new metadata
                resampled_image = sitk.ReadImage(str(eclare_output_path))
                output_spacing = np.array(resampled_image.GetSpacing())
                output_size = np.array(resampled_image.GetSize())

                self.logger.info(
                    f"ECLARE output spacing: {output_spacing}, shape: {output_size}"
                )

                # Copy output to final destination
                sitk.WriteImage(resampled_image, str(output_path))
                self.logger.info(f"Saved resampled image to: {output_path}")

            # Return metadata
            target_spacing = np.array(self.target_voxel_size)

            return {
                "original_spacing": original_spacing.tolist(),
                "target_spacing": target_spacing.tolist(),
                "original_shape": original_size.tolist(),
                "resampled_shape": output_size.tolist()
            }

        except Exception as e:
            if isinstance(e, RuntimeError):
                raise  # Re-raise RuntimeErrors (already logged)
            self.logger.error(f"ECLARE resampling failed: {e}")
            raise RuntimeError(f"ECLARE resampling failed: {e}") from e

    def visualize(
        self,
        before_path: Path,
        after_path: Path,
        output_path: Path,
        **kwargs: Any
    ) -> None:
        """Generate 3-view comparison visualization with intensity histogram overlay.

        Creates visualization with:
        - Row 1: Original image (axial, sagittal, coronal)
        - Row 2: Resampled image (axial, sagittal, coronal)
        - Row 3 (subplot 4): Overlayed intensity histograms (pre vs post)
        - Metadata text with spacing and shape information

        The histogram overlay is crucial for deep learning-based methods to verify
        that the intensity distribution is preserved after super-resolution.

        Args:
            before_path: Path to input file (before resampling)
            after_path: Path to output file (after resampling)
            output_path: Path to save visualization (PNG)
            **kwargs: Additional parameters:
                - 'original_spacing': Original voxel spacing
                - 'target_spacing': Target voxel spacing
                - 'original_shape': Original image shape
                - 'resampled_shape': Resampled image shape

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

        self.logger.info(f"Generating ECLARE visualization: {output_path}")

        try:
            # Load images
            before_img = nib.load(str(before_path))
            before_data = before_img.get_fdata()

            after_img = nib.load(str(after_path))
            after_data = after_img.get_fdata()

            # Get middle slices for each view
            # Axial (XY plane, slice along Z)
            mid_z_before = before_data.shape[2] // 2
            mid_z_after = after_data.shape[2] // 2
            axial_before = before_data[:, :, mid_z_before].T
            axial_after = after_data[:, :, mid_z_after].T

            # Sagittal (YZ plane, slice along X)
            mid_x_before = before_data.shape[0] // 2
            mid_x_after = after_data.shape[0] // 2
            sagittal_before = before_data[mid_x_before, :, :].T
            sagittal_after = after_data[mid_x_after, :, :].T

            # Coronal (XZ plane, slice along Y)
            mid_y_before = before_data.shape[1] // 2
            mid_y_after = after_data.shape[1] // 2
            coronal_before = before_data[:, mid_y_before, :].T
            coronal_after = after_data[:, mid_y_after, :].T

            # Create figure: 3 rows x 3 columns (last row has histogram in middle)
            fig = plt.figure(figsize=(18, 16))
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.2)

            fig.suptitle(
                f'ECLARE Super-Resolution: {before_path.stem}',
                fontsize=16,
                fontweight='bold'
            )

            # Compute shared intensity range for consistent visualization
            vmin = min(before_data.min(), after_data.min())
            vmax = max(before_data.max(), after_data.max())

            # Row 1: Original image
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

            # Row 2: Resampled image
            ax10 = fig.add_subplot(gs[1, 0])
            ax10.imshow(axial_after, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
            ax10.set_title('ECLARE Resampled - Axial', fontsize=12)
            ax10.axis('off')

            ax11 = fig.add_subplot(gs[1, 1])
            ax11.imshow(sagittal_after, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
            ax11.set_title('ECLARE Resampled - Sagittal', fontsize=12)
            ax11.axis('off')

            ax12 = fig.add_subplot(gs[1, 2])
            ax12.imshow(coronal_after, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
            ax12.set_title('ECLARE Resampled - Coronal', fontsize=12)
            ax12.axis('off')

            # Row 3: Overlayed histogram (center column, spans all 3 columns)
            ax2 = fig.add_subplot(gs[2, :])

            # Flatten data for histogram
            before_flat = before_data.flatten()
            after_flat = after_data.flatten()

            # Remove zero/background values for better histogram
            before_nonzero = before_flat[before_flat > 0]
            after_nonzero = after_flat[after_flat > 0]

            # Compute histogram bins
            bins = 100
            hist_range = (
                min(before_nonzero.min(), after_nonzero.min()),
                max(before_nonzero.max(), after_nonzero.max())
            )

            # Plot overlayed histograms with transparency
            ax2.hist(
                before_nonzero,
                bins=bins,
                range=hist_range,
                alpha=0.5,
                label='Original',
                color='blue',
                density=True
            )
            ax2.hist(
                after_nonzero,
                bins=bins,
                range=hist_range,
                alpha=0.5,
                label='ECLARE Resampled',
                color='red',
                density=True
            )

            ax2.set_xlabel('Intensity', fontsize=11)
            ax2.set_ylabel('Density', fontsize=11)
            ax2.set_title('Intensity Distribution Comparison (non-zero voxels)', fontsize=12)
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)

            # Add metadata text
            metadata_text = (
                f"Original:\n"
                f"  Spacing: {original_spacing}\n"
                f"  Shape: {original_shape}\n\n"
                f"ECLARE Resampled:\n"
                f"  Spacing: {target_spacing}\n"
                f"  Shape: {resampled_shape}\n\n"
                f"ECLARE Config:\n"
                f"  Batch size: {self.batch_size}\n"
                f"  N patches: {self.n_patches}\n"
                f"  Patch sampling: {self.patch_sampling}"
            )

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

            self.logger.info(f"Visualization saved to {output_path}")

        except Exception as e:
            self.logger.error(f"Visualization generation failed: {e}")
            raise RuntimeError(f"Visualization failed: {e}") from e
