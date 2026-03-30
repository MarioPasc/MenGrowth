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


def drop_coarsest_levels(
    number_of_iterations: list[int],
    shrink_factors: list[int],
    smoothing_sigmas: list[int],
    max_shrink_factor: int = 4,
) -> tuple[list[int], list[int], list[int]]:
    """Drop multi-resolution levels where shrink factor exceeds threshold.

    Thick-slice volumes have near-zero gradient energy at coarse resolution
    levels. Starting at shrink_factor=8 on 5mm-original data means 8mm
    effective resolution — the z-direction is constant and MI optimization
    receives no gradient signal.

    This function removes leading (coarsest) levels from the multi-resolution
    schedule where `shrink_factors[i] > max_shrink_factor`, keeping at least
    the finest level.

    Ref: Avants et al., NeuroImage 2011, DOI: 10.1016/j.neuroimage.2010.09.025

    Args:
        number_of_iterations: Iterations per resolution level (coarsest first).
        shrink_factors: Shrink factors per resolution level (coarsest first).
        smoothing_sigmas: Smoothing sigmas per resolution level (coarsest first).
        max_shrink_factor: Maximum allowed shrink factor. Levels with higher
            shrink factors are dropped.

    Returns:
        Trimmed copies of (number_of_iterations, shrink_factors, smoothing_sigmas).
        At least one level (the finest) is always preserved.
    """
    if not shrink_factors:
        return list(number_of_iterations), list(shrink_factors), list(smoothing_sigmas)

    # Find first index where shrink factor is within threshold
    first_valid = 0
    for i, sf in enumerate(shrink_factors):
        if sf <= max_shrink_factor:
            first_valid = i
            break
    else:
        # All levels exceed threshold — keep only the last (finest)
        first_valid = len(shrink_factors) - 1

    return (
        list(number_of_iterations[first_valid:]),
        list(shrink_factors[first_valid:]),
        list(smoothing_sigmas[first_valid:]),
    )
