"""Background removal operations for data harmonization.

This module implements SELF background removal that sets only air/background
voxels to zero without eroding anatomical structures. This is NOT skull-stripping.

Implementation
--------------
This wrapper uses the SELF-based head–air separation implemented in
``head_mask.compute_brain_mask``. The head mask marks all anatomical structures
(brain, skull, soft tissue) as foreground and primarily labels air as background.

Here we convert the head mask into an "air mask" (its logical complement) and
set those voxels to a constant value (typically 0.0). Intensities inside the
head mask are not modified.
"""

from pathlib import Path
from typing import Any, Dict
import logging

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion, generate_binary_structure

from mengrowth.preprocessing.src.data_harmonization.base import BaseBackgroundRemover
from mengrowth.preprocessing.src.config import BackgroundZeroingConfig
from mengrowth.preprocessing.src.utils.head_mask import compute_brain_mask


logger = logging.getLogger(__name__)


class SELFBackgroundRemover(BaseBackgroundRemover):
    """Remove background/air voxels SELFly without eroding anatomy.

    This class is responsible for I/O and logging. The actual head–air
    separation is performed by :func:`compute_brain_mask` in ``head_mask.py``:

    1. A SELF head mask (head vs. air) is computed from the input
       volume using the SELF algorithm.
    2. The air mask is obtained as the logical complement of the head mask.
    3. Optionally, the air mask is eroded a few voxels to be even more
       SELF with respect to anatomy (shrink air, not head).
    4. Voxels in the air mask are set to a constant value (fill_value),
       leaving all voxels inside the head mask unchanged.

    The input data:
        - May contain skull (no skull-stripping required).
        - Are assumed to be ~1 mm^3 isotropic.
        - Are assumed to be intensity-normalized (or at least stable).

    Configuration notes
    -------------------
    The previous implementation used ``percentile_low``, ``gaussian_sigma``,
    and ``min_comp_voxels`` for a border-connected percentile algorithm. Those
    hyperparameters are no longer used by this SELF-based implementation.

    New/used parameters from ``BackgroundZeroingConfig``:

        - ``air_border_margin`` (int): number of erosions applied to the
          *air mask* (default 0). This shrinks the air region and is safe
          for anatomy (it never erodes the head).
        - ``auto_fallback`` (bool, optional): if present, passed to
          ``compute_brain_mask``; defaults to True.
        - ``fallback_threshold`` (float, optional): minimum acceptable head
          coverage before falling back to a simple Otsu-based mask; defaults
          to 0.05 (5% of volume).
        - ``fill_value`` (float, optional): value used to zero background
          voxels; defaults to 0.0.

    Existing YAML files may continue to include ``percentile_low``,
    ``gaussian_sigma``, and ``min_comp_voxels`` but they are ignored by this
    implementation.
    """

    def __init__(self, config: BackgroundZeroingConfig, verbose: bool = False) -> None:
        """Initialize SELF background remover.

        Parameters
        ----------
        config:
            BackgroundZeroingConfig instance parsed from YAML.
        verbose:
            If True, enable more detailed logging.
        """
        # Convert dataclass to dict for base class; keep legacy fields for
        # compatibility even though some are unused in the new algorithm.
        config_dict: Dict[str, Any] = {
            "method": config.method,
            "percentile_low": getattr(config, "percentile_low", None),
            "gaussian_sigma": getattr(config, "gaussian_sigma", None),
            "min_comp_voxels": getattr(config, "min_comp_voxels", None),
            "air_border_margin": config.air_border_margin,
            "auto_fallback": getattr(config, "auto_fallback", True),
            "fallback_threshold": getattr(config, "fallback_threshold", 0.05),
            "fill_value": getattr(config, "fill_value", 0.0),
        }
        super().__init__(config=config_dict, verbose=verbose)

        self.bg_config = config
        self.verbose = verbose

        auto_fallback = config_dict["auto_fallback"]
        fallback_threshold = config_dict["fallback_threshold"]
        fill_value = config_dict["fill_value"]

        logger.info(
            "Initialized SELFBackgroundRemover (SELF-based): "
            f"method={config.method}, air_margin={config.air_border_margin}, "
            f"auto_fallback={auto_fallback}, "
            f"fallback_threshold={fallback_threshold}, "
            f"fill_value={fill_value}"
        )

    def _build_air_mask(self, volume: np.ndarray) -> np.ndarray:
        """Build air/background mask from a 3D MRI volume.

        This function calls :func:`compute_brain_mask` to obtain a SELF
        head mask and then returns its logical complement (air mask). Optionally
        the air mask is eroded by ``air_border_margin`` voxels to further reduce
        the risk of eroding anatomical structures.

        Parameters
        ----------
        volume:
            3D MRI volume array (D, H, W).

        Returns
        -------
        np.ndarray
            Boolean mask where True = air/background, False = head/anatomy.
        """
        self.logger.info("Building air mask from SELF head mask algorithm...")

        # Optional parameters with robust defaults for backward compatibility
        auto_fallback = getattr(self.bg_config, "auto_fallback", True)
        fallback_threshold = getattr(self.bg_config, "fallback_threshold", 0.05)

        # Compute SELF head mask (True=head, False=air)
        head_mask = compute_brain_mask(
            volume,
            verbose=self.verbose,
            auto_fallback=auto_fallback,
            fallback_threshold=fallback_threshold,
        )

        if head_mask.dtype != bool:
            head_mask = head_mask.astype(bool)

        head_coverage = head_mask.sum() / head_mask.size
        self.logger.info("Head mask coverage: %.2f%%", head_coverage * 100.0)

        if head_mask.sum() == 0:
            self.logger.warning(
                "Head mask is empty; returning all-background air mask (no changes will be applied)."
            )
            return np.ones_like(volume, dtype=bool)

        # Air mask is the logical complement of the head mask.
        air_mask = ~head_mask

        # Optionally erode air mask to be extra SELF with respect to anatomy.
        air_margin = getattr(self.bg_config, "air_border_margin", 0)
        if air_margin > 0:
            self.logger.debug("Eroding air mask by %d voxels", air_margin)
            erosion_struct = generate_binary_structure(3, 1)  # 6-connectivity
            for _ in range(air_margin):
                air_mask = binary_erosion(air_mask, structure=erosion_struct)

        final_size = air_mask.sum()
        self.logger.info(
            "Final air mask: %d voxels (%.2f%% of volume)",
            final_size,
            100.0 * final_size / volume.size,
        )

        return air_mask

    def execute(
        self,
        input_path: Path,
        output_path: Path,
        **kwargs: Any,
    ) -> None:
        """Remove background from NIfTI volume SELFly.

        Parameters
        ----------
        input_path:
            Path to input NIfTI file.
        output_path:
            Path to output NIfTI file.
        **kwargs:
            Additional parameters (e.g. allow_overwrite).

        Raises
        ------
        RuntimeError
            If reading, masking, or writing fails.
        """
        allow_overwrite: bool = bool(kwargs.get("allow_overwrite", False))

        # Check overwrite policy
        if output_path.exists() and not allow_overwrite:
            msg = (
                f"Output file {output_path} already exists and allow_overwrite is False. "
                "Background removal aborted."
            )
            self.logger.error(msg)
            raise RuntimeError(msg)

        # Log execution in base class bookkeeping
        self.log_execution(input_path, output_path)

        try:
            # Load NIfTI
            img = nib.load(str(input_path))
            volume = img.get_fdata()

            self.logger.info(
                "Loaded volume: shape=%s, range=[%.4f, %.4f]",
                volume.shape,
                float(volume.min()),
                float(volume.max()),
            )

            # Build air mask (True=air, False=head)
            air_mask = self._build_air_mask(volume)

            # Apply mask: set only air voxels to fill_value
            fill_value: float = float(getattr(self.bg_config, "fill_value", 0.0))
            masked_volume = volume.copy()
            masked_volume[air_mask] = fill_value

            n_air = air_mask.sum()
            self.logger.info(
                "Zeroed %d background voxels (%.2f%% of volume)",
                n_air,
                100.0 * n_air / air_mask.size,
            )

            # Preserve header and affine
            output_img = nib.Nifti1Image(masked_volume, affine=img.affine, header=img.header)

            # Save output
            nib.save(output_img, str(output_path))
            self.logger.info("Successfully removed background from %s", input_path.name)

        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Background removal failed: %s", exc)
            raise RuntimeError(f"Background removal failed: {exc}") from exc

    def visualize(
        self,
        before_path: Path,
        after_path: Path,
        output_path: Path,
        **kwargs: Any,
    ) -> None:
        """Generate visualization showing background removal with mask overlay.

        The visualization shows 4 depth slices for each of 3 orientations
        (axial, sagittal, coronal), with the computed air mask overlaid on
        the *original* volume in red.

        Parameters
        ----------
        before_path:
            NIfTI path before background removal.
        after_path:
            NIfTI path after background removal (not strictly required for the
            mask computation but kept for symmetry and potential QC checks).
        output_path:
            Path to save the resulting PNG figure.
        **kwargs:
            Unused; kept for interface compatibility.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore[import]

        self.logger.info("Generating background removal visualization: %s", output_path)

        try:
            # Load "before" and "after" images
            before_img = nib.load(str(before_path))
            before_data = before_img.get_fdata()

            after_img = nib.load(str(after_path))
            after_data = after_img.get_fdata()

            # Compute air mask directly from the original volume
            air_mask = self._build_air_mask(before_data)

            # For logging/QC: how many voxels actually changed to fill_value?
            fill_value = float(getattr(self.bg_config, "fill_value", 0.0))
            changed_to_fill = (before_data != after_data) & np.isclose(after_data, fill_value)
            self.logger.info(
                "Visualization stats: air_mask=%d voxels (%.2f%%), newly_zeroed=%d voxels",
                air_mask.sum(),
                100.0 * air_mask.sum() / air_mask.size,
                changed_to_fill.sum(),
            )

            # Define slice positions (fractions of each axis).
            depth_fractions = [0.25, 0.4, 0.5, 0.6]

            fig, axes = plt.subplots(3, 4, figsize=(16, 12))
            fig.suptitle(
                f"Background Removal: {before_path.stem}\n"
                "Air mask overlaid on original (red=air/background)",
                fontsize=16,
                fontweight="bold",
            )

            # Axial slices (row 0)
            for col, frac in enumerate(depth_fractions):
                z_idx = int(before_data.shape[2] * frac)
                slice_img = before_data[:, :, z_idx].T
                slice_mask = air_mask[:, :, z_idx].T

                axes[0, col].imshow(slice_img, cmap="gray", origin="lower")
                overlay = np.zeros((*slice_mask.shape, 4), dtype=float)
                overlay[slice_mask, :] = [0.7, 0.0, 0.0, 0.8]  # red with alpha
                axes[0, col].imshow(overlay, origin="lower")
                axes[0, col].set_title(f"Axial z={z_idx}")
                axes[0, col].axis("off")

            # Sagittal slices (row 1)
            for col, frac in enumerate(depth_fractions):
                x_idx = int(before_data.shape[0] * frac)
                slice_img = before_data[x_idx, :, :].T
                slice_mask = air_mask[x_idx, :, :].T

                axes[1, col].imshow(slice_img, cmap="gray", origin="lower")
                overlay = np.zeros((*slice_mask.shape, 4), dtype=float)
                overlay[slice_mask, :] = [0.7, 0.0, 0.0, 0.8]
                axes[1, col].imshow(overlay, origin="lower")
                axes[1, col].set_title(f"Sagittal x={x_idx}")
                axes[1, col].axis("off")

            # Coronal slices (row 2)
            for col, frac in enumerate(depth_fractions):
                y_idx = int(before_data.shape[1] * frac)
                slice_img = before_data[:, y_idx, :].T
                slice_mask = air_mask[:, y_idx, :].T

                axes[2, col].imshow(slice_img, cmap="gray", origin="lower")
                overlay = np.zeros((*slice_mask.shape, 4), dtype=float)
                overlay[slice_mask, :] = [0.7, 0.0, 0.0, 0.8]
                axes[2, col].imshow(overlay, origin="lower")
                axes[2, col].set_title(f"Coronal y={y_idx}")
                axes[2, col].axis("off")

            plt.tight_layout()

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save figure
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

            self.logger.info("Visualization saved to %s", output_path)

        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Visualization generation failed: %s", exc)
            raise RuntimeError(f"Visualization failed: {exc}") from exc
