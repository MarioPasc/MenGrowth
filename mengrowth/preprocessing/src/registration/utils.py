"""Utility functions for the registration module."""

from typing import Optional
import logging

import numpy as np

logger = logging.getLogger(__name__)


def detect_thick_slice_volume(
    img: "ants.ANTsImage",
    threshold_ratio: float = 3.0,
    log: Optional[logging.Logger] = None,
) -> bool:
    """Detect if a volume was likely acquired with thick slices.

    After isotropic resampling, thick-slice volumes retain low gradient energy
    in the original slice-select direction (z) compared to in-plane directions
    (x, y) due to interpolation smoothing between the few original slices.

    This heuristic computes the ratio of mean in-plane gradient energy to
    z-gradient energy. A high ratio indicates thick-slice origin.

    Args:
        img: ANTs image (already resampled to isotropic spacing).
        threshold_ratio: If in-plane gradient energy exceeds z-gradient
            energy by this factor, consider it a thick-slice volume.
        log: Optional logger for diagnostic output.

    Returns:
        True if the volume appears to originate from a thick-slice acquisition.
    """
    data = img.numpy()

    # Compute mean squared gradient along each axis
    energy_x = float(np.mean(np.diff(data, axis=0) ** 2))
    energy_y = float(np.mean(np.diff(data, axis=1) ** 2))
    energy_z = float(np.mean(np.diff(data, axis=2) ** 2))

    if energy_z < 1e-10:
        if log:
            log.info("    Gradient energy: z ≈ 0 → thick-slice detected")
        return True

    in_plane_energy = (energy_x + energy_y) / 2.0
    ratio = in_plane_energy / energy_z

    if log:
        log.info(
            f"    Gradient energy ratio (in-plane/z) = {ratio:.2f} "
            f"(threshold={threshold_ratio})"
        )

    return ratio > threshold_ratio
