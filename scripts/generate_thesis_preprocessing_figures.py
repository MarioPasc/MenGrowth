#!/usr/bin/env python
"""Generate thesis preprocessing figures (1x3 panels) from the HDF5 archive.

Produces one PDF per preprocessing step, each showing axial / sagittal / coronal
views side by side.  Bias-field figures use a 2x3 layout (bottom row = colorbar).

Usage:
    ~/.conda/envs/growth/bin/python scripts/generate_thesis_preprocessing_figures.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from scipy.ndimage import binary_dilation, zoom

# ── project imports ──────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mengrowth.analysis.graphical_abstract_figures.config import (
    GraphicalAbstractConfig,
    load_graphical_abstract_config,
)
from mengrowth.analysis.graphical_abstract_figures.loader import ArchiveLoader
from mengrowth.analysis.graphical_abstract_figures.renderers_2d import (
    compute_slice_index,
    extract_slice,
    normalize_for_display,
)
from mengrowth.preprocessing.utils.settings import apply_ieee_style

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ── constants ────────────────────────────────────────────────────────────
CONFIG_PATH = Path("configs/templates/graphical_abstract.yaml")
OUTPUT_ROOT = Path(
    "/media/mpascual/PortableSSD/Meningiomas/visualizations/mengrowth"
)
VIEWS = ["axial", "sagittal", "coronal"]
VIEW_LABELS = {"axial": "Axial", "sagittal": "Sagittal", "coronal": "Coronal"}
DPI = 300
FMT = "pdf"


# ── helpers ──────────────────────────────────────────────────────────────


def _slice_for_view(
    volume: np.ndarray,
    view: str,
    frac: Optional[float],
    plow: float = 1.0,
    phigh: float = 99.0,
) -> np.ndarray:
    """Extract and normalise a 2D slice from a 3D volume."""
    idx = compute_slice_index(volume.shape, view, frac)
    sl = extract_slice(volume, view, idx)
    return normalize_for_display(sl, plow, phigh)


def _make_1x3_panel(
    slices: List[np.ndarray],
    cmap: str = "gray",
    vmin: float = 0.0,
    vmax: float = 1.0,
    dpi: int = DPI,
    pad_to_common: bool = True,
    frame_color: Optional[str] = None,
    frame_width: int = 1,
) -> Figure:
    """Create a clean 1×3 panel figure.

    Args:
        slices: List of three 2D arrays (axial, sagittal, coronal).
        pad_to_common: If True, pad all slices to the same h×w (post-reg steps).
            If False, render at native size with width_ratios (pre-reg steps).
        frame_color: If set, draw a thin rectangular frame around each image.
        frame_width: Width in pixels of the frame border.
    """
    import matplotlib.gridspec as gridspec

    if pad_to_common:
        max_h = max(s.shape[0] for s in slices)
        max_w = max(s.shape[1] for s in slices)

        padded: List[np.ndarray] = []
        for s in slices:
            pad_h = max_h - s.shape[0]
            pad_w = max_w - s.shape[1]
            top = pad_h // 2
            left = pad_w // 2
            p = np.zeros((max_h, max_w), dtype=s.dtype)
            p[top : top + s.shape[0], left : left + s.shape[1]] = s
            padded.append(p)

        cell_w = max_w / dpi
        cell_h = max_h / dpi
        scale = 3.0
        fig_w = cell_w * scale * 3
        fig_h = cell_h * scale

        fig, axes = plt.subplots(1, 3, figsize=(fig_w, fig_h), dpi=dpi)
        for ax, img in zip(axes, padded):
            ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower", aspect="equal")
            ax.axis("off")
        fig.subplots_adjust(wspace=0.02, left=0.0, right=1.0, top=1.0, bottom=0.0)
        return fig
    else:
        # Native aspect ratios with gridspec width_ratios
        max_h = max(s.shape[0] for s in slices)
        width_ratios = [s.shape[1] for s in slices]
        total_w = sum(width_ratios)
        scale = 3.0
        fig_h = max_h / dpi * scale
        fig_w = total_w / dpi * scale

        fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
        gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=width_ratios,
                               wspace=0.03, left=0.0, right=1.0, top=1.0, bottom=0.0)

        for col, s in enumerate(slices):
            ax = fig.add_subplot(gs[0, col])
            ax.imshow(s, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower", aspect="equal")
            if frame_color:
                import matplotlib.patches as mpatches
                rect = mpatches.Rectangle(
                    (0, 0), s.shape[1] - 1, s.shape[0] - 1,
                    linewidth=frame_width, edgecolor=frame_color,
                    facecolor="none", clip_on=False,
                )
                ax.add_patch(rect)
            ax.axis("off")
        return fig


def _save(fig: Figure, path: Path) -> None:
    """Save figure and close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    logger.info("Saved %s", path)


# ── figure generators ────────────────────────────────────────────────────


def fig_data_harmonization(
    loader: ArchiveLoader,
    cfg: GraphicalAbstractConfig,
    out_dir: Path,
) -> None:
    """Step 1: data harmonisation — T1n, 1×3 panel."""
    vol = loader.load_step_volume("step1_data_harmonization", "t1n")
    plow, phigh = cfg.intensity_percentile_low, cfg.intensity_percentile_high
    pre_fracs = {
        "axial": cfg.slice.pre_reg_axial_frac or cfg.slice.axial_frac,
        "sagittal": cfg.slice.pre_reg_sagittal_frac or cfg.slice.sagittal_frac,
        "coronal": cfg.slice.pre_reg_coronal_frac or cfg.slice.coronal_frac,
    }
    slices = [_slice_for_view(vol.data, v, pre_fracs[v], plow, phigh) for v in VIEWS]
    fig = _make_1x3_panel(slices, pad_to_common=False, frame_color="#555555")
    _save(fig, out_dir / f"step_data_harmonization.{FMT}")


def fig_bias_field_correction(
    loader: ArchiveLoader,
    cfg: GraphicalAbstractConfig,
    out_dir: Path,
) -> None:
    """Step 2: bias field correction — T1n with bias field overlay, 2×3 (bottom = cbar)."""
    vol = loader.load_step_volume("step2_bias_field_correction", "t1n")
    bias = loader.load_bias_field("t1n")
    plow, phigh = cfg.intensity_percentile_low, cfg.intensity_percentile_high
    pre_fracs = {
        "axial": cfg.slice.pre_reg_axial_frac or cfg.slice.axial_frac,
        "sagittal": cfg.slice.pre_reg_sagittal_frac or cfg.slice.sagittal_frac,
        "coronal": cfg.slice.pre_reg_coronal_frac or cfg.slice.coronal_frac,
    }

    # Extract slices at native resolution
    mri_slices: List[np.ndarray] = []
    bias_slices: List[np.ndarray] = []
    fg_masks: List[np.ndarray] = []

    for v in VIEWS:
        mri_slices.append(_slice_for_view(vol.data, v, pre_fracs[v], plow, phigh))

        # Foreground mask from raw data
        idx_raw = compute_slice_index(vol.data.shape, v, pre_fracs[v])
        sl_raw = extract_slice(vol.data, v, idx_raw)
        fg_masks.append(sl_raw > 0)

        if bias is not None:
            idx = compute_slice_index(bias.shape, v, pre_fracs[v])
            sl_b = extract_slice(bias, v, idx)
            # Mask non-foreground as NaN
            sl_b_masked = sl_b.astype(np.float32).copy()
            sl_b_masked[~fg_masks[-1]] = np.nan
            bias_slices.append(sl_b_masked)

    # Compute global bias range from foreground voxels only (symmetric around 1.0)
    if bias is not None and bias_slices:
        fg_vals = np.concatenate(
            [b[m] for b, m in zip(bias_slices, fg_masks) if m.any()]
        )
        fg_vals = fg_vals[np.isfinite(fg_vals)]
        if fg_vals.size > 0:
            bias_min = float(np.min(fg_vals[fg_vals > 0])) if np.any(fg_vals > 0) else 0.5
            bias_max = float(np.max(fg_vals))
        else:
            bias_min, bias_max = 0.9, 1.1
        extent = max(abs(bias_min - 1.0), abs(bias_max - 1.0), 0.01)
    else:
        extent = 0.1

    cbar_cmap = cfg.step_options.bias_field_cmap
    alpha = cfg.step_options.bias_field_alpha

    # Native aspect ratios via gridspec
    import matplotlib.gridspec as gridspec
    import matplotlib as mpl

    width_ratios = [s.shape[1] for s in mri_slices]
    max_h = max(s.shape[0] for s in mri_slices)
    total_w = sum(width_ratios)
    scale = 3.0
    fig_h = max_h / DPI * scale
    fig_w = total_w / DPI * scale

    cbar_height_ratio = 0.08
    fig = plt.figure(figsize=(fig_w, fig_h * (1.0 + cbar_height_ratio)), dpi=DPI)
    gs = gridspec.GridSpec(
        2, 3, figure=fig, width_ratios=width_ratios,
        height_ratios=[1.0, cbar_height_ratio],
        wspace=0.03, hspace=0.05,
        left=0.0, right=1.0, top=1.0, bottom=0.10,
    )

    bias_cmap_obj = mpl.colormaps[cbar_cmap].copy()
    bias_cmap_obj.set_bad(alpha=0)  # NaN → fully transparent

    frame_color = "#555555"
    for col in range(3):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(mri_slices[col], cmap="gray", vmin=0, vmax=1, origin="lower", aspect="equal")
        if bias_slices:
            ax.imshow(
                bias_slices[col],
                cmap=bias_cmap_obj,
                vmin=1.0 - extent,
                vmax=1.0 + extent,
                origin="lower",
                aspect="equal",
                alpha=alpha,
            )
        # Frame
        import matplotlib.patches as mpatches
        h, w = mri_slices[col].shape
        rect = mpatches.Rectangle(
            (0, 0), w - 1, h - 1,
            linewidth=1, edgecolor=frame_color,
            facecolor="none", clip_on=False,
        )
        ax.add_patch(rect)
        ax.axis("off")

    # Bottom row: hidden axes + centered colorbar
    for col in range(3):
        ax = fig.add_subplot(gs[1, col])
        ax.axis("off")

    cbar_ax = fig.add_axes([0.15, 0.02, 0.7, 0.04])
    norm = plt.Normalize(vmin=1.0 - extent, vmax=1.0 + extent)
    sm = plt.cm.ScalarMappable(cmap=cbar_cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label(r"Bias field $\hat{B}(\mathbf{x})$", fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    _save(fig, out_dir / f"step_bias_field_correction.{FMT}")


def fig_resampling(
    loader: ArchiveLoader,
    cfg: GraphicalAbstractConfig,
    out_dir: Path,
) -> None:
    """Step 3: resampling — T1n, 1×3 panel."""
    vol = loader.load_step_volume("step3_resampling", "t1n")
    plow, phigh = cfg.intensity_percentile_low, cfg.intensity_percentile_high
    pre_fracs = {
        "axial": cfg.slice.pre_reg_axial_frac or cfg.slice.axial_frac,
        "sagittal": cfg.slice.pre_reg_sagittal_frac or cfg.slice.sagittal_frac,
        "coronal": cfg.slice.pre_reg_coronal_frac or cfg.slice.coronal_frac,
    }
    slices = [_slice_for_view(vol.data, v, pre_fracs[v], plow, phigh) for v in VIEWS]
    fig = _make_1x3_panel(slices, pad_to_common=False, frame_color="#555555")
    _save(fig, out_dir / f"step_resampling.{FMT}")


def fig_cubic_padding(
    loader: ArchiveLoader,
    cfg: GraphicalAbstractConfig,
    out_dir: Path,
) -> None:
    """Step 4: cubic padding — T1n, 1×3 panel."""
    vol = loader.load_step_volume("step4_cubic_padding", "t1n")
    plow, phigh = cfg.intensity_percentile_low, cfg.intensity_percentile_high
    pre_fracs = {
        "axial": cfg.slice.pre_reg_axial_frac or cfg.slice.axial_frac,
        "sagittal": cfg.slice.pre_reg_sagittal_frac or cfg.slice.sagittal_frac,
        "coronal": cfg.slice.pre_reg_coronal_frac or cfg.slice.coronal_frac,
    }
    slices = [_slice_for_view(vol.data, v, pre_fracs[v], plow, phigh) for v in VIEWS]
    fig = _make_1x3_panel(slices)
    _save(fig, out_dir / f"step_cubic_padding.{FMT}")


def fig_registration(
    loader: ArchiveLoader,
    cfg: GraphicalAbstractConfig,
    out_dir: Path,
) -> None:
    """Step 5: registration — T1n after atlas registration, 1×3 panel."""
    vol = loader.load_step_volume("step5_registration_static", "t1n")
    plow, phigh = cfg.intensity_percentile_low, cfg.intensity_percentile_high
    frac = {"axial": cfg.slice.axial_frac, "sagittal": cfg.slice.sagittal_frac, "coronal": cfg.slice.coronal_frac}
    slices = [_slice_for_view(vol.data, v, frac[v], plow, phigh) for v in VIEWS]
    fig = _make_1x3_panel(slices)
    _save(fig, out_dir / f"step_registration.{FMT}")


def fig_atlas_overlay(
    loader: ArchiveLoader,
    cfg: GraphicalAbstractConfig,
    out_dir: Path,
) -> None:
    """Atlas T1 with grey/white/CSF tissue probability map contours, 1×3 panel."""
    import nibabel as nib

    atlas = loader.load_atlas()
    if atlas is None:
        logger.warning("Atlas not available — skipping atlas figure")
        return

    # Load tissue probability maps
    tpm_root = Path("/media/mpascual/PortableSSD/Meningiomas/ATLAS/sri24_spm8/tpm")
    tpm_info = [
        ("grey", tpm_root / "grey.nii", "#4477AA"),   # blue
        ("white", tpm_root / "white.nii", "#EE6677"),  # red
        ("csf", tpm_root / "csf.nii", "#228833"),      # green
    ]
    tpm_volumes: List[Tuple[str, np.ndarray, str]] = []
    for name, path, color in tpm_info:
        if not path.exists():
            logger.warning("TPM not found: %s", path)
            continue
        img = nib.load(str(path))
        data = np.asarray(img.dataobj, dtype=np.float32).squeeze()
        # Normalise 0-255 → 0-1
        data = data / 255.0
        tpm_volumes.append((name, data, color))

    plow, phigh = cfg.intensity_percentile_low, cfg.intensity_percentile_high
    frac = {"axial": cfg.slice.axial_frac, "sagittal": cfg.slice.sagittal_frac, "coronal": cfg.slice.coronal_frac}

    atlas_slices = [_slice_for_view(atlas, v, frac[v], plow, phigh) for v in VIEWS]

    # All have the same shape (240x240x155), so no padding needed
    max_h = max(s.shape[0] for s in atlas_slices)
    max_w = max(s.shape[1] for s in atlas_slices)

    cell_w = max_w / DPI
    cell_h = max_h / DPI
    scale = 3.0
    fig, axes = plt.subplots(1, 3, figsize=(cell_w * scale * 3, cell_h * scale), dpi=DPI)

    for col, (ax, v) in enumerate(zip(axes, VIEWS)):
        ax.imshow(atlas_slices[col], cmap="gray", vmin=0, vmax=1, origin="lower", aspect="equal")

        # Overlay TPM contours at the tissue boundary (0.5 threshold)
        for name, vol, color in tpm_volumes:
            idx = compute_slice_index(vol.shape, v, frac[v])
            sl = extract_slice(vol, v, idx)
            if sl.max() > 0.3:
                ax.contour(
                    sl,
                    levels=[0.5],
                    colors=[color],
                    linewidths=[0.8],
                    origin="lower",
                )
        ax.axis("off")

    fig.subplots_adjust(wspace=0.02, left=0.0, right=1.0, top=1.0, bottom=0.0)
    _save(fig, out_dir / f"step_atlas_overlay.{FMT}")


def fig_skull_stripping(
    loader: ArchiveLoader,
    cfg: GraphicalAbstractConfig,
    out_dir: Path,
) -> None:
    """Step 6: skull stripping — T1c with brain mask contour, 1×3 panel."""
    # Full head = registration output (step5)
    full_head = loader.load_step_volume("step5_registration_static", "t1c")
    brain_mask = loader.load_brain_mask()
    if brain_mask is None:
        brain_mask = loader.load_brain_mask_nifti("t1c")

    plow, phigh = cfg.intensity_percentile_low, cfg.intensity_percentile_high
    frac = {"axial": cfg.slice.axial_frac, "sagittal": cfg.slice.sagittal_frac, "coronal": cfg.slice.coronal_frac}

    mri_slices = [_slice_for_view(full_head.data, v, frac[v], plow, phigh) for v in VIEWS]

    max_h = max(s.shape[0] for s in mri_slices)
    max_w = max(s.shape[1] for s in mri_slices)

    def _pad(s: np.ndarray) -> np.ndarray:
        pad_h = max_h - s.shape[0]
        pad_w = max_w - s.shape[1]
        top = pad_h // 2
        left = pad_w // 2
        p = np.zeros((max_h, max_w), dtype=s.dtype)
        p[top : top + s.shape[0], left : left + s.shape[1]] = s
        return p

    padded_mri = [_pad(s) for s in mri_slices]

    # Extract mask slices
    mask_slices_raw = []
    if brain_mask is not None:
        for v in VIEWS:
            idx = compute_slice_index(brain_mask.shape, v, frac[v])
            sl = extract_slice(brain_mask.astype(np.float32), v, idx)
            mask_slices_raw.append(sl)

    padded_mask = [_pad(s) for s in mask_slices_raw] if mask_slices_raw else []

    cell_w = max_w / DPI
    cell_h = max_h / DPI
    scale = 3.0
    fig, axes = plt.subplots(1, 3, figsize=(cell_w * scale * 3, cell_h * scale), dpi=DPI)

    mask_color = cfg.step_options.mask_contour_color
    mask_lw = cfg.step_options.mask_contour_linewidth

    for col, ax in enumerate(axes):
        ax.imshow(padded_mri[col], cmap="gray", vmin=0, vmax=1, origin="lower", aspect="equal")
        if padded_mask:
            ax.contour(
                padded_mask[col],
                levels=[0.5],
                colors=[mask_color],
                linewidths=[mask_lw],
                origin="lower",
            )
        ax.axis("off")

    fig.subplots_adjust(wspace=0.02, left=0.0, right=1.0, top=1.0, bottom=0.0)
    _save(fig, out_dir / f"step_skull_stripping.{FMT}")


def fig_longitudinal_registration(
    loader_ref: ArchiveLoader,
    loader_mov: ArchiveLoader,
    cfg: GraphicalAbstractConfig,
    out_dir: Path,
) -> None:
    """Step 8: longitudinal registration — reference (grey) + moving (colour-washed), 1×3."""
    vol_ref = loader_ref.load_longitudinal_volume("t1n")
    vol_mov = loader_mov.load_longitudinal_volume("t1n")
    if vol_ref is None or vol_mov is None:
        logger.warning("Longitudinal volumes not available — skipping")
        return

    plow, phigh = cfg.intensity_percentile_low, cfg.intensity_percentile_high
    frac = {"axial": cfg.slice.axial_frac, "sagittal": cfg.slice.sagittal_frac, "coronal": cfg.slice.coronal_frac}

    ref_slices = [_slice_for_view(vol_ref.data, v, frac[v], plow, phigh) for v in VIEWS]
    mov_slices = [_slice_for_view(vol_mov.data, v, frac[v], plow, phigh) for v in VIEWS]

    max_h = max(s.shape[0] for s in ref_slices)
    max_w = max(s.shape[1] for s in ref_slices)

    def _pad(s: np.ndarray) -> np.ndarray:
        pad_h = max_h - s.shape[0]
        pad_w = max_w - s.shape[1]
        top = pad_h // 2
        left = pad_w // 2
        p = np.zeros((max_h, max_w), dtype=s.dtype)
        p[top : top + s.shape[0], left : left + s.shape[1]] = s
        return p

    padded_ref = [_pad(s) for s in ref_slices]
    padded_mov = [_pad(s) for s in mov_slices]

    overlay_cmap = cfg.step_options.registration_overlay_cmap
    overlay_alpha = cfg.step_options.registration_alpha

    cell_w = max_w / DPI
    cell_h = max_h / DPI
    scale = 3.0
    fig, axes = plt.subplots(1, 3, figsize=(cell_w * scale * 3, cell_h * scale), dpi=DPI)

    for ax, ref_img, mov_img in zip(axes, padded_ref, padded_mov):
        ax.imshow(ref_img, cmap="gray", vmin=0, vmax=1, origin="lower", aspect="equal")
        ax.imshow(
            mov_img,
            cmap=overlay_cmap,
            vmin=0,
            vmax=1,
            origin="lower",
            aspect="equal",
            alpha=overlay_alpha,
        )
        ax.axis("off")

    fig.subplots_adjust(wspace=0.02, left=0.0, right=1.0, top=1.0, bottom=0.0)
    _save(fig, out_dir / f"step_longitudinal_registration.{FMT}")


def fig_presegmentation(
    loader: ArchiveLoader,
    cfg: GraphicalAbstractConfig,
    out_dir: Path,
) -> None:
    """Pre-segmentation — T1c with tumour segmentation overlay, 1×3 panel."""
    # Use final preprocessed volume (post-longitudinal registration)
    vol = loader.load_longitudinal_volume("t1c")
    if vol is None:
        # Fall back to skull-stripped
        vol = loader.load_step_volume("step6_skull_stripping", "t1c")

    seg = loader.load_segmentation()
    if seg is None:
        logger.warning("No segmentation available — skipping pre-segmentation figure")
        return

    plow, phigh = cfg.intensity_percentile_low, cfg.intensity_percentile_high
    frac = {"axial": cfg.slice.axial_frac, "sagittal": cfg.slice.sagittal_frac, "coronal": cfg.slice.coronal_frac}

    mri_slices = [_slice_for_view(vol.data, v, frac[v], plow, phigh) for v in VIEWS]

    # Extract seg slices (no normalisation)
    seg_slices_raw = []
    for v in VIEWS:
        idx = compute_slice_index(seg.shape, v, frac[v])
        sl = extract_slice(seg.astype(np.float32), v, idx)
        seg_slices_raw.append(sl)

    max_h = max(s.shape[0] for s in mri_slices)
    max_w = max(s.shape[1] for s in mri_slices)

    def _pad(s: np.ndarray) -> np.ndarray:
        pad_h = max_h - s.shape[0]
        pad_w = max_w - s.shape[1]
        top = pad_h // 2
        left = pad_w // 2
        p = np.zeros((max_h, max_w), dtype=s.dtype)
        p[top : top + s.shape[0], left : left + s.shape[1]] = s
        return p

    padded_mri = [_pad(s) for s in mri_slices]
    padded_seg = [_pad(s) for s in seg_slices_raw]

    label_colors = cfg.step_options.segmentation_colors
    seg_alpha = cfg.step_options.segmentation_alpha
    seg_lw = cfg.step_options.segmentation_linewidth

    cell_w = max_w / DPI
    cell_h = max_h / DPI
    scale = 3.0
    fig, axes = plt.subplots(1, 3, figsize=(cell_w * scale * 3, cell_h * scale), dpi=DPI)

    for ax, mri_img, seg_img in zip(axes, padded_mri, padded_seg):
        ax.imshow(mri_img, cmap="gray", vmin=0, vmax=1, origin="lower", aspect="equal")

        # Build RGBA overlay
        h, w = mri_img.shape
        overlay = np.zeros((h, w, 4), dtype=np.float32)
        for label_id, color_hex in sorted(label_colors.items()):
            mask = seg_img == label_id
            if mask.any():
                rgb = mcolors.to_rgb(color_hex)
                overlay[mask, 0] = rgb[0]
                overlay[mask, 1] = rgb[1]
                overlay[mask, 2] = rgb[2]
                overlay[mask, 3] = seg_alpha

        ax.imshow(overlay, origin="lower", aspect="equal")

        # Contours
        for label_id, color_hex in sorted(label_colors.items()):
            binary = (seg_img == label_id).astype(np.float32)
            if binary.max() > 0:
                ax.contour(
                    binary,
                    levels=[0.5],
                    colors=[color_hex],
                    linewidths=[seg_lw],
                    origin="lower",
                )
        ax.axis("off")

    fig.subplots_adjust(wspace=0.02, left=0.0, right=1.0, top=1.0, bottom=0.0)
    _save(fig, out_dir / f"step_presegmentation.{FMT}")


def _build_synthseg_color_table(label_ids: np.ndarray) -> Dict[int, np.ndarray]:
    """Distinct RGBA colour per non-zero SynthSeg label.

    Concatenates ``tab20`` + ``tab20b`` + ``tab20c`` so ~30 labels land on
    visually separable hues (homologous L/R pairs fall on different
    entries). Mirrors ``fig_qualitative._build_label_color_table``.
    """
    palettes = [plt.cm.tab20.colors, plt.cm.tab20b.colors, plt.cm.tab20c.colors]
    table_colors = [c for p in palettes for c in p]
    mapping: Dict[int, np.ndarray] = {}
    idx = 0
    for lid in label_ids:
        if int(lid) == 0:
            continue
        rgb = np.asarray(table_colors[idx % len(table_colors)], dtype=np.float32)
        mapping[int(lid)] = np.concatenate([rgb, [1.0]]).astype(np.float32)
        idx += 1
    return mapping


def fig_synthseg_overlay(
    loader: ArchiveLoader,
    cfg: GraphicalAbstractConfig,
    out_dir: Path,
    synthseg_root: Path,
    modality: str = "t1n",
    fill_alpha: float = 0.20,
    contour_alpha: float = 1.0,
) -> None:
    """Final preprocessed volume with overlayed SynthSeg parcellation, 1×3 panel.

    Loads ``synthseg_parc.nii.gz`` from ``synthseg_root / patient_id /
    study_id``. For each of the three canonical views, renders the T1
    slice in greyscale, per-label filled interior at ``fill_alpha`` and a
    one-voxel-thick per-label boundary at ``contour_alpha``. Layout and
    colour assignment follow ``fig_qualitative._render_cell``.
    """
    import nibabel as nib

    vol = loader.load_longitudinal_volume(modality)
    if vol is None:
        vol = loader.load_step_volume("step6_skull_stripping", modality)

    parc_path = synthseg_root / cfg.patient_id / cfg.study_ids[1] / "synthseg_parc.nii.gz"
    if not parc_path.exists():
        logger.warning("SynthSeg parcellation not found at %s — skipping", parc_path)
        return
    parc = np.asarray(nib.load(str(parc_path)).dataobj, dtype=np.int32)

    plow, phigh = cfg.intensity_percentile_low, cfg.intensity_percentile_high
    frac = {
        "axial": cfg.slice.axial_frac,
        "sagittal": cfg.slice.sagittal_frac,
        "coronal": cfg.slice.coronal_frac,
    }

    mri_slices = [_slice_for_view(vol.data, v, frac[v], plow, phigh) for v in VIEWS]
    parc_slices: List[np.ndarray] = []
    for v in VIEWS:
        idx = compute_slice_index(parc.shape, v, frac[v])
        parc_slices.append(extract_slice(parc, v, idx).astype(np.int32))

    # Common color table across the three views so homologous regions match.
    all_labels = np.unique(np.concatenate([p.ravel() for p in parc_slices]))
    label_colors = _build_synthseg_color_table(all_labels)

    max_h = max(s.shape[0] for s in mri_slices)
    max_w = max(s.shape[1] for s in mri_slices)

    def _pad(s: np.ndarray, dtype: Optional[type] = None) -> np.ndarray:
        pad_h = max_h - s.shape[0]
        pad_w = max_w - s.shape[1]
        top = pad_h // 2
        left = pad_w // 2
        p = np.zeros((max_h, max_w), dtype=dtype or s.dtype)
        p[top : top + s.shape[0], left : left + s.shape[1]] = s
        return p

    padded_mri = [_pad(s) for s in mri_slices]
    padded_parc = [_pad(s, dtype=np.int32) for s in parc_slices]

    cell_w = max_w / DPI
    cell_h = max_h / DPI
    scale = 3.0
    fig, axes = plt.subplots(1, 3, figsize=(cell_w * scale * 3, cell_h * scale), dpi=DPI)

    for ax, mri_img, parc_img in zip(axes, padded_mri, padded_parc):
        ax.imshow(mri_img, cmap="gray", vmin=0, vmax=1, origin="lower", aspect="equal")

        fill = np.zeros((*parc_img.shape, 4), dtype=np.float32)
        contour = np.zeros((*parc_img.shape, 4), dtype=np.float32)
        for lid, rgba in label_colors.items():
            mask = parc_img == lid
            if not mask.any():
                continue
            boundary = binary_dilation(mask, iterations=1) & ~mask
            interior = mask & ~boundary
            fill[interior, :3] = rgba[:3]
            fill[interior, 3] = fill_alpha
            contour[boundary, :3] = rgba[:3]
            contour[boundary, 3] = contour_alpha

        ax.imshow(fill, origin="lower", aspect="equal", interpolation="nearest")
        ax.imshow(contour, origin="lower", aspect="equal", interpolation="nearest")
        ax.axis("off")

    fig.subplots_adjust(wspace=0.02, left=0.0, right=1.0, top=1.0, bottom=0.0)
    _save(fig, out_dir / f"step_synthseg.{FMT}")


# ── main ─────────────────────────────────────────────────────────────────


def main() -> None:
    """Generate all thesis preprocessing figures."""
    apply_ieee_style()
    plt.rcParams.update(
        {
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )

    cfg = load_graphical_abstract_config(CONFIG_PATH)
    out_dir = OUTPUT_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)

    # Primary study loader (MenGrowth-0009-000)
    study_id = cfg.study_ids[1]  # MenGrowth-0009-000
    archive_path = Path(cfg.archive_root) / cfg.patient_id / study_id / "archive.h5"
    artifacts_dir = Path(cfg.artifacts_root) / cfg.patient_id / study_id
    atlas_path = Path(cfg.atlas_path) if cfg.atlas_path else None
    preprocessed_dir = Path(cfg.preprocessed_root) / cfg.patient_id / study_id

    loader = ArchiveLoader(archive_path, artifacts_dir, atlas_path, preprocessed_dir)

    # Ensure segmentation is in the archive
    if cfg.show_segmentation:
        loader.ensure_segmentation_in_archive()

    logger.info("Generating thesis preprocessing figures...")
    logger.info("Output: %s", out_dir)

    # Generate all figures
    fig_data_harmonization(loader, cfg, out_dir)
    fig_bias_field_correction(loader, cfg, out_dir)
    fig_resampling(loader, cfg, out_dir)
    fig_cubic_padding(loader, cfg, out_dir)
    fig_registration(loader, cfg, out_dir)
    fig_atlas_overlay(loader, cfg, out_dir)
    fig_skull_stripping(loader, cfg, out_dir)

    # Longitudinal: need a second study loader
    if len(cfg.study_ids) >= 2:
        study_id_mov = cfg.study_ids[1]  # MenGrowth-0009-001
        archive_mov = Path(cfg.archive_root) / cfg.patient_id / study_id_mov / "archive.h5"
        artifacts_mov = Path(cfg.artifacts_root) / cfg.patient_id / study_id_mov
        preprocessed_mov = Path(cfg.preprocessed_root) / cfg.patient_id / study_id_mov
        loader_mov = ArchiveLoader(archive_mov, artifacts_mov, atlas_path, preprocessed_mov)
        fig_longitudinal_registration(loader, loader_mov, cfg, out_dir)

    # Pre-segmentation
    fig_presegmentation(loader, cfg, out_dir)

    # SynthSeg parcellation overlay on final preprocessed volume.
    # synthseg_root is the sibling of preprocessed_root (…/v5_final/synthseg).
    synthseg_root = Path(cfg.preprocessed_root).parent / "synthseg"
    fig_synthseg_overlay(loader, cfg, out_dir, synthseg_root=synthseg_root)

    logger.info("Done! All figures saved to %s", out_dir)


if __name__ == "__main__":
    main()
