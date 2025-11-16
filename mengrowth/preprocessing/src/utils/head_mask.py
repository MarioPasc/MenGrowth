# -*- coding: utf-8 -*-
"""
Background removal utilities for MRI head volumes.

Goal
----
Compute a conservative head-air mask and set voxels outside the head region
to zero. This is *not* skull-stripping in the classical sense: the mask is
intended to keep all anatomical structures (brain, skull, soft tissue) and
only remove air/background. The input data:

- May contain skull (no skull-stripping required).
- Is assumed to be ~1 mm^3 isotropic.
- Is assumed to be intensity-normalized (or at least with stable percentiles).

Design principles
-----------------
- Conservative: it is preferable to leave some background voxels inside the
  mask than to mistakenly remove anatomical voxels.
- No changes to voxel intensities inside the mask; only outside voxels are
  set to a constant (typically 0.0).
- Robust to multi-center, multi-scanner intensity variations via
  percentile-based thresholds and border seeding.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy.ndimage import (
    binary_fill_holes,
    binary_dilation,
    binary_erosion,
    label,
    generate_binary_structure,
    binary_propagation,
    binary_opening,
    binary_closing,
)
from skimage.filters import sobel, threshold_otsu
from skimage.segmentation import watershed

Array = np.ndarray


@dataclass(frozen=True)
class _MaskParams:
    """
    Internal parameters for robust head-air separation.

    Tuned for conservative behaviour on non-skull-stripped, normalized MRI:
    - air_p_low / air_p_high control how strict the "air" definition is.
      Lower values => only very dark voxels are considered air.
    - erode_vox = 0 to avoid shrinking the head marker before watershed
      (conservative: do not eat into anatomy at the boundary).
    """
    air_p_low: float = 1.0       # seed air at very dark intensities (1st percentile)
    air_p_high: float = 25.0     # permit flood-fill through "dark-enough" voxels
    air_p_global: float = 0.2    # global darkest voxels to use as fallback seeds
    erode_vox: int = 0           # do NOT erode head seed (conservative boundary)
    close_iters: int = 1         # final morphological smoothing
    connectivity: int = 2        # 6/18/26-connectivity selector


def _normalize01(v: np.ndarray) -> np.ndarray:
    """Percentile-based robust [0,1] rescale to reduce bias-field effects."""
    v = np.asarray(v, dtype=np.float32)
    v[~np.isfinite(v)] = 0.0
    lo, hi = np.percentile(v, [0.5, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        v = v - np.nanmin(v)
        hi = np.nanmax(v)
        lo = 0.0
    result: np.ndarray = np.clip((v - lo) / (hi - lo + np.finfo(float).eps), 0.0, 1.0)
    return result

def compute_brain_mask_simple(
    volume: np.ndarray,
    method: str = "otsu",
    percentile_threshold: float = 10.0,
    verbose: bool = False,
) -> np.ndarray:
    """
    Simple intensity-based head extraction for volumes with clear background.

    This method is intended as a *fallback* when the border-based SELF algorithm
    fails (e.g. pathological cases). It assumes:

    - Volume includes background air near zero.
    - Head (brain + skull + soft tissue) is significantly brighter than air.
    - Data can be non-skull-stripped; the mask will approximate the head.

    Pipeline
    --------
    1) Threshold: Otsu, percentile, or zero-based.
    2) Remove small noise with morphological opening.
    3) Keep largest connected component (head).
    4) Fill holes.
    5) Smooth with morphological closing.

    Parameters
    ----------
    volume : np.ndarray
        3D MRI volume (D, H, W).
    method : str
        Thresholding method:
        - 'otsu': Otsu's automatic thresholding (default).
        - 'percentile': Use percentile_threshold parameter.
        - 'zero': Simple > 0 threshold (for strictly zero-padded background).
    percentile_threshold : float
        Percentile for 'percentile' method (default: 10.0).
    verbose : bool
        If True, prints diagnostic information.

    Returns
    -------
    mask : np.ndarray of bool
        True for head, False for background.
        Same shape as input volume.
    """
    v = np.asanyarray(volume, dtype=np.float32)
    v[~np.isfinite(v)] = 0.0

    if verbose:
        print(f"   Simple head mask: volume shape {v.shape}")
        print(f"   Input range: [{v.min():.4f}, {v.max():.4f}]")
        print(f"   Method: {method}")

    # 1) Compute threshold
    if method == "otsu":
        try:
            threshold = threshold_otsu(v)
        except Exception:
            threshold = np.percentile(v[v > 0], 10.0) if (v > 0).any() else 0.0
        if verbose:
            print(f"   Otsu threshold: {threshold:.4f}")
    elif method == "percentile":
        threshold = np.percentile(v, percentile_threshold)
        if verbose:
            print(f"   Percentile ({percentile_threshold}%) threshold: {threshold:.4f}")
    elif method == "zero":
        threshold = 0.0
        if verbose:
            print("   Zero threshold: 0.0")
    else:
        raise ValueError("Unknown method: {method}. Use 'otsu', 'percentile', or 'zero'")

    # Create initial mask
    mask = v > threshold

    if verbose:
        print(f"   Initial mask: {mask.sum()} voxels ({100 * mask.sum() / mask.size:.1f}%)")

    # 2) Remove small noise with morphological opening
    st = generate_binary_structure(3, 1)  # 6-connectivity
    mask = binary_opening(mask, structure=st, iterations=2)

    if verbose:
        print(f"   After opening: {mask.sum()} voxels ({100 * mask.sum() / mask.size:.1f}%)")

    # 3) Keep largest connected component (head)
    labels, nlab = label(mask, structure=st)

    if nlab == 0:
        if verbose:
            print("   ⚠️  No components found - returning empty mask")
        return np.zeros_like(mask, dtype=bool)

    counts = np.bincount(labels.ravel())
    counts[0] = 0  # Ignore background
    largest_label = int(counts.argmax())
    mask = labels == largest_label

    if verbose:
        print(f"   Found {nlab} components, keeping largest ({counts.max()} voxels)")

    # 4) Fill holes
    mask = binary_fill_holes(mask)

    if verbose:
        print(f"   After hole filling: {mask.sum()} voxels ({100 * mask.sum() / mask.size:.1f}%)")

    # 5) Smooth with morphological closing
    mask = binary_closing(mask, structure=st, iterations=3)

    if verbose:
        print(f"   Final mask: {mask.sum()} voxels ({100 * mask.sum() / mask.size:.1f}%)")

    result: np.ndarray = mask
    return result


def _compute_brain_mask_self(volume: np.ndarray, verbose: bool = False) -> np.ndarray:
    """
    Border-based SELF algorithm for head mask extraction (internal function).

    Idea
    ----
    Detect background air by:
    - Seeding very dark voxels on the volume border.
    - Flood-filling through voxels that remain "dark enough".
    - Labeling as head everything that is *not* connected dark air.

    This returns a mask that keeps the entire head (brain + skull + soft tissue)
    and removes primarily air. It is suitable for non-skull-stripped data.

    Pipeline
    --------
      1) Robustly normalize intensities to [0,1].
      2) Seed outside air on the volume border at very low intensities.
      3) Flood-fill air through 'dark-enough' voxels to label exterior.
      4) Head = complement of exterior; keep largest 3D component.
      5) Edge refinement: watershed on the gradient with air vs. head markers.
      6) Fill holes and light morphological closing.

    Parameters
    ----------
    volume : np.ndarray
        3D MRI volume (D, H, W) in any orientation.
    verbose : bool
        If True, prints diagnostic information.

    Returns
    -------
    mask : np.ndarray of bool
        True for head/anatomy, False for background (air).
        Same shape as input volume.
    """
    p = _MaskParams()
    v = np.asanyarray(volume, dtype=np.float32)
    v[~np.isfinite(v)] = 0.0

    if verbose:
        print(f"   Head mask extraction (SELF): volume shape {v.shape}")
        print(f"   Input range: [{v.min():.4f}, {v.max():.4f}]")

    # 1) Normalize to [0,1] for robust percentile operations
    v = _normalize01(v)

    # 2) Background air seeds
    air_thr_seed = np.percentile(v, p.air_p_low)
    air_thr_pass = np.percentile(v, p.air_p_high)

    # Define border mask
    border = np.zeros_like(v, dtype=bool)
    border[[0, -1], :, :] = True
    border[:, [0, -1], :] = True
    border[:, :, [0, -1]] = True

    air_seed = border & (v <= air_thr_seed)

    if verbose:
        print(f"   Air threshold (seed): {air_thr_seed:.4f}")
        print(f"   Air threshold (pass): {air_thr_pass:.4f}")
        print(f"   Border air seed voxels: {air_seed.sum()}")

    # Fallback: if no border seeds found, use global darkest voxels near the border
    if air_seed.sum() == 0:
        if verbose:
            print("   No border seeds found - using global background detection")

        global_thr = np.percentile(v, p.air_p_global)
        dark_global = v <= global_thr

        if verbose:
            print(f"   Global background threshold: {global_thr:.4f}")
            print(f"   Global dark voxels: {dark_global.sum()}")

        st_dilate = generate_binary_structure(3, 1)
        border_expanded = binary_dilation(border, structure=st_dilate, iterations=3)
        air_seed = dark_global & border_expanded

        if verbose:
            print(f"   Near-border dark seeds: {air_seed.sum()}")

        if air_seed.sum() == 0:
            if verbose:
                print("   Still no seeds - using darkest voxels globally (last resort)")
            n_seeds = max(100, int(0.001 * v.size))
            flat_v = v.ravel()
            darkest_idx = np.argpartition(flat_v, n_seeds)[:n_seeds]
            air_seed_flat = np.zeros(v.size, dtype=bool)
            air_seed_flat[darkest_idx] = True
            air_seed = air_seed_flat.reshape(v.shape)
            if verbose:
                print(f"   Global darkest seeds: {air_seed.sum()}")

    # 3) Flood-fill exterior through dark voxels
    st = generate_binary_structure(3, p.connectivity)
    dark = v <= air_thr_pass
    exterior = binary_propagation(air_seed, mask=dark, structure=st)

    if verbose:
        print(f"   Exterior voxels: {exterior.sum()} ({100 * exterior.sum() / v.size:.1f}%)")

    # 4) Raw head and largest connected component
    head0 = ~exterior
    labels, nlab = label(head0.astype(np.uint8), structure=st)

    if nlab == 0:
        if verbose:
            print("   ⚠️  No head components found - returning empty mask")
        return np.zeros_like(head0, dtype=bool)

    counts = np.bincount(labels.ravel())
    counts[0] = 0  # Ignore background
    head_lcc = labels == int(counts.argmax())

    if verbose:
        print(f"   Found {nlab} head components, largest has {counts.max()} voxels")

    # 5) Watershed edge refinement
    if p.erode_vox > 0:
        head_marker = binary_erosion(head_lcc, structure=st, iterations=p.erode_vox)
    else:
        # Conservative: do not shrink the head marker
        head_marker = head_lcc

    markers = np.zeros_like(labels, dtype=np.int32)
    markers[exterior] = 1
    markers[head_marker] = 2

    grad = sobel(v)  # Isotropic gradient magnitude
    w = watershed(grad, markers=markers, connectivity=1, mask=(exterior | head_lcc))
    mask = w == 2

    # 6) Fill holes and light closing
    mask = binary_fill_holes(mask)
    if p.close_iters > 0:
        mask = binary_dilation(mask, structure=st, iterations=p.close_iters)
        mask = binary_erosion(mask, structure=st, iterations=p.close_iters)

    if verbose:
        print(f"   Final SELF mask voxels: {mask.sum()} ({100 * mask.sum() / v.size:.1f}%)")

    result: np.ndarray = mask

    return result


def compute_brain_mask(
    volume: np.ndarray,
    verbose: bool = False,
    auto_fallback: bool = True,
    fallback_threshold: float = 0.05,
) -> np.ndarray:
    """
    Compute a conservative head mask (head vs. air) with optional fallback.

    Strategy
    --------
    1. Run the SELF (border-based) algorithm.
    2. Measure coverage = (#mask voxels / total voxels).
    3. If coverage is extremely small (< fallback_threshold), assume failure
       and fall back to a simple intensity-based mask.
    4. Otherwise, keep the SELF mask, even if coverage is high: this is
       conservative and preferred (we do not want to over-remove voxels).

    This function is designed for:
    - Non-skull-stripped, normalized MRI volumes.
    - 1 mm^3 isotropic spacing.
    - Use as a *background remover*: only voxels outside the mask will be
      set to zero; intensities inside the mask are preserved.

    Parameters
    ----------
    volume : np.ndarray
        3D MRI volume (D, H, W).
    verbose : bool
        If True, prints diagnostic information.
    auto_fallback : bool
        If True, use simple method when SELF coverage is too low.
    fallback_threshold : float
        Minimum acceptable coverage for SELF (default: 0.05 = 5% of volume).
        If SELF returns a mask smaller than this, it is considered a failure
        and a simple method is used instead.

    Returns
    -------
    mask : np.ndarray of bool
        True for head/anatomy, False for background.
        Same shape as input volume.
    """
    if verbose:
        print("   Trying SELF (border-based) head mask...")

    mask_self = _compute_brain_mask_self(volume, verbose=verbose)
    coverage = mask_self.sum() / mask_self.size

    if verbose:
        print(f"   SELF coverage: {coverage * 100:.1f}%")

    # Fallback only if coverage is *too low* (under-segmentation / failure).
    if auto_fallback and coverage < fallback_threshold:
        if verbose:
            print(
                f"   SELF coverage {coverage * 100:.1f}% < "
                f"{fallback_threshold * 100:.1f}% threshold - using simple fallback"
            )
        mask = compute_brain_mask_simple(volume, method="otsu", verbose=verbose)
        if verbose:
            new_coverage = mask.sum() / mask.size
            print(f"   Simple method coverage: {new_coverage * 100:.1f}%")
        return mask
    else:
        # Conservative: keep SELF result even if it covers a large fraction
        # of the volume; we prefer under-removal of background to over-removal
        # of anatomical voxels.
        return mask_self


def apply_brain_mask(
    volume: np.ndarray,
    mask: np.ndarray,
    fill_value: float = 0.0,
) -> np.ndarray:
    """
    Apply a binary head mask to a volume, setting background voxels to fill_value.

    This function does not change intensities inside the mask.

    Parameters
    ----------
    volume : np.ndarray
        Volume to process (any shape compatible with mask).
    mask : np.ndarray
        Binary mask (True=head/anatomy, False=background).
    fill_value : float
        Value to set for background voxels (typically 0.0).

    Returns
    -------
    masked_volume : np.ndarray
        Copy of volume with background removed (set to fill_value).
    """
    masked = volume.copy()
    masked[~mask] = fill_value
    return masked


def extract_and_mask_volume(
    volume: np.ndarray,
    source: str = "SELF",
    reference_volume: Optional[np.ndarray] = None,
    fill_value: float = 0.0,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove background from a non-skull-stripped MRI volume by masking.

    Computes a conservative head mask to identify the anatomical region
    and sets all voxels outside this region to fill_value (typically 0.0).

    Importantly, this does NOT perform skull-stripping: the mask is designed
    to keep brain, skull and surrounding soft tissue, and to remove only air.

    Parameters
    ----------
    volume : np.ndarray
        Volume to process (with skull present).
    source : str
        Source for mask computation:
        - "SELF": Compute mask from volume itself (default, recommended).
        - "REFERENCE": Compute mask from reference_volume (e.g. HR volume).
    reference_volume : np.ndarray, optional
        Reference volume for mask computation when source="REFERENCE".
        Required if source="REFERENCE".
    fill_value : float
        Value to set for background voxels (typically 0.0).
    verbose : bool
        Print diagnostic information.

    Returns
    -------
    masked_volume : np.ndarray
        Volume with background removed (set to fill_value).
    mask : np.ndarray
        The computed binary mask (True=head, False=background).

    Raises
    ------
    ValueError
        If source="REFERENCE" but reference_volume is None.
    """
    if source.upper() == "REFERENCE":
        if reference_volume is None:
            raise ValueError("reference_volume must be provided when source='REFERENCE'")
        if verbose:
            print("   Extracting head mask from REFERENCE volume")
        mask = compute_brain_mask(reference_volume, verbose=verbose)
    elif source.upper() == "SELF":
        if verbose:
            print("   Extracting head mask from volume itself")
        mask = compute_brain_mask(volume, verbose=verbose)
    else:
        raise ValueError("Unknown source: {source}. Must be 'SELF' or 'REFERENCE'")

    # Apply mask without modifying intensities inside the head region
    masked_volume = apply_brain_mask(volume, mask, fill_value=fill_value)
    return masked_volume, mask
