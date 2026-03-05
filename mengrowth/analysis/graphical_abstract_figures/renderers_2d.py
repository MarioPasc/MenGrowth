"""2D slice extraction and rendering for graphical abstract figures.

Each renderer produces standalone single-image figures (no subplots)
suitable for external composition into a graphical abstract.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from scipy.ndimage import zoom

from .config import SliceConfig, StepFigureConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Core utilities
# =============================================================================


def compute_slice_index(
    shape: Tuple[int, ...],
    view: str,
    frac: Optional[float] = None,
) -> int:
    """Compute the slice index for a given view and fractional position.

    Args:
        shape: 3D volume shape (I, J, K).
        view: One of "axial", "sagittal", "coronal".
        frac: Fractional position along the slicing axis (0-1). None = 0.5 (center).

    Returns:
        Integer slice index.
    """
    axis_map = {"axial": 2, "sagittal": 0, "coronal": 1}
    axis = axis_map[view]
    f = frac if frac is not None else 0.5
    return int(np.clip(f * (shape[axis] - 1), 0, shape[axis] - 1))


def extract_slice(volume: np.ndarray, view: str, index: int) -> np.ndarray:
    """Extract a 2D slice from a 3D volume.

    Args:
        volume: 3D array.
        view: One of "axial", "sagittal", "coronal".
        index: Slice index along the view axis.

    Returns:
        2D array (transposed for standard radiological display).
    """
    if view == "axial":
        return volume[:, :, index].T
    elif view == "sagittal":
        return volume[index, :, :].T
    elif view == "coronal":
        return volume[:, index, :].T
    else:
        raise ValueError(f"Unknown view: {view}")


def normalize_for_display(
    slice_2d: np.ndarray,
    plow: float = 1.0,
    phigh: float = 99.0,
    nonzero_only: bool = True,
) -> np.ndarray:
    """Normalize a 2D slice to [0, 1] using percentile windowing.

    Args:
        slice_2d: 2D float array.
        plow: Low percentile for clipping.
        phigh: High percentile for clipping.
        nonzero_only: If True, compute percentiles from nonzero voxels only.

    Returns:
        2D float array in [0, 1].
    """
    values = slice_2d[slice_2d > 0] if nonzero_only else slice_2d.ravel()
    if values.size == 0:
        return np.zeros_like(slice_2d)

    vmin = np.percentile(values, plow)
    vmax = np.percentile(values, phigh)
    if vmax <= vmin:
        return np.zeros_like(slice_2d)

    normed = (slice_2d - vmin) / (vmax - vmin)
    return np.clip(normed, 0.0, 1.0)


def get_slice_frac(slice_config: SliceConfig, view: str) -> Optional[float]:
    """Get the fractional slice position for a given view from the config.

    Args:
        slice_config: SliceConfig instance.
        view: One of "axial", "sagittal", "coronal".

    Returns:
        Fractional position, or None for center.
    """
    frac_map = {
        "axial": slice_config.axial_frac,
        "sagittal": slice_config.sagittal_frac,
        "coronal": slice_config.coronal_frac,
    }
    return frac_map.get(view)


def get_pre_reg_slice_frac(slice_config: SliceConfig, view: str) -> Optional[float]:
    """Get the fractional slice position for pre-registration steps.

    Falls back to the standard frac if no pre-reg frac is configured.

    Args:
        slice_config: SliceConfig instance.
        view: One of "axial", "sagittal", "coronal".

    Returns:
        Fractional position, or None for center.
    """
    pre_reg_map = {
        "axial": slice_config.pre_reg_axial_frac,
        "sagittal": slice_config.pre_reg_sagittal_frac,
        "coronal": slice_config.pre_reg_coronal_frac,
    }
    frac = pre_reg_map.get(view)
    if frac is not None:
        return frac
    return get_slice_frac(slice_config, view)


def make_clean_figure(
    slice_2d: np.ndarray,
    cmap: str = "gray",
    title: Optional[str] = None,
    dpi: int = 300,
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> Figure:
    """Create a clean, standalone figure with a single image.

    No axes decorations, ticks, or spines — just the image with an optional title.

    Args:
        slice_2d: 2D array to display.
        cmap: Matplotlib colormap name.
        title: Optional title string.
        dpi: Figure resolution.
        vmin: Display minimum.
        vmax: Display maximum.

    Returns:
        matplotlib Figure.
    """
    h, w = slice_2d.shape
    fig_w = w / dpi * 3  # Scale up for readability
    fig_h = h / dpi * 3
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

    ax.imshow(slice_2d, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower", aspect="equal")
    ax.axis("off")

    if title:
        ax.set_title(title, fontsize=10, pad=4)

    fig.tight_layout(pad=0.1)
    return fig


# =============================================================================
# Per-step-type renderers
# =============================================================================


def render_standard_slice(
    volume: np.ndarray,
    view: str,
    frac: Optional[float],
    plow: float,
    phigh: float,
    dpi: int = 300,
    title: Optional[str] = None,
) -> List[Tuple[Figure, str]]:
    """Render a standard MRI slice in grayscale.

    Args:
        volume: 3D float array.
        view: Anatomical view.
        frac: Fractional slice position (None = center).
        plow: Low percentile for normalization.
        phigh: High percentile for normalization.
        dpi: Figure resolution.
        title: Optional title.

    Returns:
        List of (Figure, filename_suffix) tuples. Always length 1.
    """
    idx = compute_slice_index(volume.shape, view, frac)
    sl = extract_slice(volume, view, idx)
    normed = normalize_for_display(sl, plow, phigh)
    fig = make_clean_figure(normed, cmap="gray", title=title, dpi=dpi)
    return [(fig, "")]


def render_bias_field_step(
    corrected_volume: np.ndarray,
    bias_field: Optional[np.ndarray],
    view: str,
    frac: Optional[float],
    plow: float,
    phigh: float,
    step_options: StepFigureConfig,
    dpi: int = 300,
) -> List[Tuple[Figure, str]]:
    """Render bias field correction: corrected MRI + bias field overlay.

    Args:
        corrected_volume: 3D array after bias field correction.
        bias_field: 3D multiplicative bias field (optional).
        view: Anatomical view.
        frac: Fractional slice position.
        plow: Low percentile.
        phigh: High percentile.
        step_options: StepFigureConfig with bias field rendering options.
        dpi: Figure resolution.

    Returns:
        List of (Figure, suffix) tuples. 1 image if no bias field, 2 otherwise.
    """
    idx = compute_slice_index(corrected_volume.shape, view, frac)
    sl_corrected = extract_slice(corrected_volume, view, idx)
    normed = normalize_for_display(sl_corrected, plow, phigh)

    results: List[Tuple[Figure, str]] = []

    # (1) Corrected MRI
    fig1 = make_clean_figure(normed, cmap="gray", dpi=dpi)
    results.append((fig1, ""))

    # (2) Bias field overlay on corrected MRI
    if bias_field is not None:
        sl_bias = extract_slice(bias_field, view, idx)

        h, w = normed.shape
        fig2, ax = plt.subplots(figsize=(w / dpi * 3, h / dpi * 3), dpi=dpi)
        ax.imshow(normed, cmap="gray", vmin=0, vmax=1, origin="lower", aspect="equal")

        # Diverging colormap centered at 1.0 (N4 multiplicative field)
        bias_min = float(np.min(sl_bias[sl_bias > 0])) if np.any(sl_bias > 0) else 0.5
        bias_max = float(np.max(sl_bias))
        # Center at 1.0: symmetric range around 1
        extent = max(abs(bias_min - 1.0), abs(bias_max - 1.0), 0.01)
        ax.imshow(
            sl_bias,
            cmap=step_options.bias_field_cmap,
            vmin=1.0 - extent,
            vmax=1.0 + extent,
            origin="lower",
            aspect="equal",
            alpha=step_options.bias_field_alpha,
        )
        ax.axis("off")
        fig2.tight_layout(pad=0.1)
        results.append((fig2, "bias_overlay"))

    return results


def render_registration_step(
    pre_reg_volume: np.ndarray,
    post_reg_volume: np.ndarray,
    atlas_volume: Optional[np.ndarray],
    view: str,
    frac: Optional[float],
    plow: float,
    phigh: float,
    step_options: StepFigureConfig,
    dpi: int = 300,
    pre_reg_frac: Optional[float] = None,
) -> List[Tuple[Figure, str]]:
    """Render registration: pre-reg, alpha-blend, and atlas.

    Pre-reg and post-reg may differ in shape. The pre-reg 2D slice is resized
    to match post-reg 2D slice shape via scipy.ndimage.zoom before blending.

    Args:
        pre_reg_volume: 3D array before registration (e.g., step4 cubic_padding).
        post_reg_volume: 3D array after registration (e.g., step5).
        atlas_volume: 3D atlas array (optional).
        view: Anatomical view.
        frac: Fractional slice position (used for post-reg and atlas).
        plow: Low percentile.
        phigh: High percentile.
        step_options: StepFigureConfig with registration rendering options.
        dpi: Figure resolution.
        pre_reg_frac: Fractional slice position for the pre-reg volume.
            None = uses frac.

    Returns:
        List of (Figure, suffix) tuples: pre, blend, and optionally atlas.
    """
    results: List[Tuple[Figure, str]] = []

    # Pre-registration slice (uses pre_reg_frac if provided)
    effective_pre_frac = pre_reg_frac if pre_reg_frac is not None else frac
    pre_idx = compute_slice_index(pre_reg_volume.shape, view, effective_pre_frac)
    sl_pre = extract_slice(pre_reg_volume, view, pre_idx)
    normed_pre = normalize_for_display(sl_pre, plow, phigh)

    fig_pre = make_clean_figure(normed_pre, cmap="gray", dpi=dpi)
    results.append((fig_pre, "pre"))

    # Post-registration slice
    post_idx = compute_slice_index(post_reg_volume.shape, view, frac)
    sl_post = extract_slice(post_reg_volume, view, post_idx)
    normed_post = normalize_for_display(sl_post, plow, phigh)

    # Alpha blend: resize pre-reg to match post-reg shape, then overlay
    if normed_pre.shape != normed_post.shape:
        zoom_factors = (
            normed_post.shape[0] / normed_pre.shape[0],
            normed_post.shape[1] / normed_pre.shape[1],
        )
        normed_pre_resized = zoom(normed_pre, zoom_factors, order=1)
    else:
        normed_pre_resized = normed_pre

    h, w = normed_post.shape
    fig_blend, ax = plt.subplots(figsize=(w / dpi * 3, h / dpi * 3), dpi=dpi)
    ax.imshow(
        normed_pre_resized, cmap="gray", vmin=0, vmax=1, origin="lower", aspect="equal"
    )
    ax.imshow(
        normed_post,
        cmap=step_options.registration_overlay_cmap,
        vmin=0,
        vmax=1,
        origin="lower",
        aspect="equal",
        alpha=step_options.registration_alpha,
    )
    ax.axis("off")
    fig_blend.tight_layout(pad=0.1)
    results.append((fig_blend, "blend"))

    # Atlas slice
    if atlas_volume is not None:
        atlas_figs = render_atlas_slice(atlas_volume, view, frac, plow, phigh, dpi)
        results.extend(atlas_figs)

    return results


def render_skull_stripping_step(
    full_head_volume: np.ndarray,
    stripped_volume: np.ndarray,
    brain_mask: Optional[np.ndarray],
    view: str,
    frac: Optional[float],
    plow: float,
    phigh: float,
    step_options: StepFigureConfig,
    dpi: int = 300,
) -> List[Tuple[Figure, str]]:
    """Render skull stripping: full head with mask contour + stripped brain.

    Args:
        full_head_volume: 3D array before skull stripping (registration output).
        stripped_volume: 3D array after skull stripping.
        brain_mask: 3D binary brain mask (optional).
        view: Anatomical view.
        frac: Fractional slice position.
        plow: Low percentile.
        phigh: High percentile.
        step_options: StepFigureConfig with mask contour options.
        dpi: Figure resolution.

    Returns:
        List of (Figure, suffix) tuples: with_mask and stripped.
    """
    results: List[Tuple[Figure, str]] = []
    idx = compute_slice_index(full_head_volume.shape, view, frac)

    # (1) Full head with mask contour
    sl_head = extract_slice(full_head_volume, view, idx)
    normed_head = normalize_for_display(sl_head, plow, phigh)

    h, w = normed_head.shape
    fig1, ax1 = plt.subplots(figsize=(w / dpi * 3, h / dpi * 3), dpi=dpi)
    ax1.imshow(normed_head, cmap="gray", vmin=0, vmax=1, origin="lower", aspect="equal")

    if brain_mask is not None:
        sl_mask = extract_slice(brain_mask.astype(np.float32), view, idx)
        ax1.contour(
            sl_mask,
            levels=[0.5],
            colors=[step_options.mask_contour_color],
            linewidths=[step_options.mask_contour_linewidth],
            origin="lower",
        )

    ax1.axis("off")
    fig1.tight_layout(pad=0.1)
    results.append((fig1, "with_mask"))

    # (2) Skull-stripped brain
    sl_stripped = extract_slice(stripped_volume, view, idx)
    normed_stripped = normalize_for_display(sl_stripped, plow, phigh)
    fig2 = make_clean_figure(normed_stripped, cmap="gray", dpi=dpi)
    results.append((fig2, ""))

    return results


def render_atlas_slice(
    atlas_volume: np.ndarray,
    view: str,
    frac: Optional[float],
    plow: float = 1.0,
    phigh: float = 99.0,
    dpi: int = 300,
) -> List[Tuple[Figure, str]]:
    """Render an atlas T1 reference slice.

    Args:
        atlas_volume: 3D float array.
        view: Anatomical view.
        frac: Fractional slice position.
        plow: Low percentile.
        phigh: High percentile.
        dpi: Figure resolution.

    Returns:
        List with one (Figure, "atlas") tuple.
    """
    idx = compute_slice_index(atlas_volume.shape, view, frac)
    sl = extract_slice(atlas_volume, view, idx)
    normed = normalize_for_display(sl, plow, phigh)
    fig = make_clean_figure(normed, cmap="gray", dpi=dpi)
    return [(fig, "atlas")]


def render_segmentation_overlay(
    volume: np.ndarray,
    segmentation: np.ndarray,
    view: str,
    frac: Optional[float],
    plow: float,
    phigh: float,
    step_options: StepFigureConfig,
    dpi: int = 300,
) -> List[Tuple[Figure, str]]:
    """Render MRI with tumor segmentation overlay (contours + filled regions).

    Produces two images: (1) MRI with colored contours per label,
    (2) MRI with semi-transparent filled overlay per label.

    Args:
        volume: 3D float MRI array.
        segmentation: 3D uint8 label array (0=bg, 1=NET, 2=SNFH, 3=ET).
        view: Anatomical view.
        frac: Fractional slice position.
        plow: Low percentile.
        phigh: High percentile.
        step_options: StepFigureConfig with segmentation colors/alpha.
        dpi: Figure resolution.

    Returns:
        List of (Figure, suffix) tuples: contour and filled overlay.
    """
    import matplotlib.colors as mcolors

    results: List[Tuple[Figure, str]] = []
    idx = compute_slice_index(volume.shape, view, frac)
    sl_mri = extract_slice(volume, view, idx)
    normed = normalize_for_display(sl_mri, plow, phigh)
    sl_seg = extract_slice(segmentation.astype(np.float32), view, idx)

    h, w = normed.shape
    label_colors = step_options.segmentation_colors

    # (1) Contour overlay
    fig1, ax1 = plt.subplots(figsize=(w / dpi * 3, h / dpi * 3), dpi=dpi)
    ax1.imshow(normed, cmap="gray", vmin=0, vmax=1, origin="lower", aspect="equal")
    for label_id, color in sorted(label_colors.items()):
        binary = (sl_seg == label_id).astype(np.float32)
        if binary.max() > 0:
            ax1.contour(
                binary,
                levels=[0.5],
                colors=[color],
                linewidths=[step_options.segmentation_linewidth],
                origin="lower",
            )
    ax1.axis("off")
    fig1.tight_layout(pad=0.1)
    results.append((fig1, "seg_contour"))

    # (2) Filled overlay
    fig2, ax2 = plt.subplots(figsize=(w / dpi * 3, h / dpi * 3), dpi=dpi)
    ax2.imshow(normed, cmap="gray", vmin=0, vmax=1, origin="lower", aspect="equal")

    # Build RGBA overlay
    overlay = np.zeros((h, w, 4), dtype=np.float32)
    for label_id, color_hex in sorted(label_colors.items()):
        mask = sl_seg == label_id
        if mask.any():
            rgb = mcolors.to_rgb(color_hex)
            overlay[mask, 0] = rgb[0]
            overlay[mask, 1] = rgb[1]
            overlay[mask, 2] = rgb[2]
            overlay[mask, 3] = step_options.segmentation_alpha

    ax2.imshow(overlay, origin="lower", aspect="equal")
    ax2.axis("off")
    fig2.tight_layout(pad=0.1)
    results.append((fig2, "seg_filled"))

    return results
