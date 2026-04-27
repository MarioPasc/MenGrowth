#!/usr/bin/env python3
"""Generate side-by-side curated vs. processed montage for a single patient.

Layout: 8 columns (4 curated + 4 processed), studies as row groups.
Each study has a colored header row followed by 3 image rows (axial,
sagittal, coronal).

Usage:
    python scripts/visualize_pre_post_comparison.py --patient MenGrowth-0009
    python scripts/visualize_pre_post_comparison.py --patient MenGrowth-0009 \
        --output-dir /path/to/output --formats png pdf
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from mengrowth.preprocessing.utils.settings import PLOT_SETTINGS, apply_ieee_style

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

CURATED_DIR = Path(
    "/media/mpascual/PortableSSD/Meningiomas/MenGrowth/curated/dataset/MenGrowth-2025"
)
PROCESSED_DIR = Path(
    "/media/mpascual/PortableSSD/Meningiomas/MenGrowth/v5_final/MenGrowth-2025"
)
OUTPUT_DIR = Path(
    "/media/mpascual/PortableSSD/Meningiomas/visualizations/mengrowth"
)
MODALITY_ORDER = ["t1n", "t1c", "t2w", "t2f"]
VIEW_NAMES = ["Axial", "Sagittal", "Coronal"]


def load_volume(path: Path) -> Optional[np.ndarray]:
    """Load image, reorient to RAI, resample to 1 mm isotropic (BSpline).

    Parameters
    ----------
    path : Path
        Path to the image file (.nrrd or .nii.gz).

    Returns
    -------
    np.ndarray or None
        3D array in (z, y, x) order, or None if file missing.
    """
    if not path.exists():
        return None
    img = sitk.ReadImage(str(path))
    img = sitk.DICOMOrient(img, "RAI")

    orig_spacing = img.GetSpacing()
    orig_size = img.GetSize()
    new_spacing = (1.0, 1.0, 1.0)
    new_size = [
        int(round(osz * osp / nsp))
        for osz, osp, nsp in zip(orig_size, orig_spacing, new_spacing)
    ]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetInterpolator(sitk.sitkBSpline)
    resampler.SetDefaultPixelValue(0)
    img = resampler.Execute(img)

    arr = sitk.GetArrayFromImage(img)
    return arr.astype(np.float32)


def get_middle_slices(
    vol: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract middle axial, sagittal, and coronal slices.

    Parameters
    ----------
    vol : np.ndarray
        3D array in (z, y, x) order.

    Returns
    -------
    tuple of np.ndarray
        (axial, sagittal, coronal) 2D slices.
    """
    nz, ny, nx = vol.shape
    axial = vol[nz // 2, :, :]
    sagittal = vol[:, :, nx // 2]
    coronal = vol[:, ny // 2, :]
    return axial, sagittal, coronal


def normalize_slice(s: np.ndarray) -> np.ndarray:
    """Robust percentile normalization to [0, 1].

    Parameters
    ----------
    s : np.ndarray
        2D slice array.

    Returns
    -------
    np.ndarray
        Normalized array clipped to [0, 1].
    """
    p2, p98 = np.percentile(s, [2, 98])
    if p98 - p2 < 1e-6:
        return np.zeros_like(s, dtype=np.float32)
    return np.clip((s - p2) / (p98 - p2), 0, 1)


def _render_image_cell(
    ax: plt.Axes,
    sl: Optional[np.ndarray],
    view_name: str,
    show_ylabel: bool,
) -> None:
    """Render a single image cell in the montage grid."""
    ax.set_facecolor("black")
    if sl is not None:
        ax.imshow(
            normalize_slice(sl),
            cmap="gray", origin="upper",
            aspect="equal", interpolation="bilinear",
        )
    else:
        ax.text(
            0.5, 0.5, "N/A",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=7, color="0.4",
        )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if show_ylabel:
        ax.set_ylabel(
            view_name,
            fontsize=PLOT_SETTINGS["tick_labelsize"],
            color="white", rotation=90, labelpad=4,
        )


def generate_comparison_montage(
    patient_id: str,
    curated_patient_dir: Path,
    processed_patient_dir: Path,
    output_stem: Path,
    formats: list[str] | None = None,
) -> None:
    """Generate side-by-side curated vs. processed vertical montage.

    Layout: 8 columns split into two halves (curated | processed), each
    with 4 modality columns. Studies appear as row groups, each with a
    colored header row and 3 image rows (axial, sagittal, coronal).

    Parameters
    ----------
    patient_id : str
        Patient identifier (e.g. "MenGrowth-0009").
    curated_patient_dir : Path
        Path to curated patient directory (NRRD files).
    processed_patient_dir : Path
        Path to processed patient directory (NIfTI files).
    output_stem : Path
        Output path without extension.
    formats : list[str] or None
        Output formats (default: ["png", "pdf"]).
    """
    if formats is None:
        formats = ["png", "pdf"]

    curated_studies: set[str] = set()
    processed_studies: set[str] = set()
    if curated_patient_dir.exists():
        curated_studies = {d.name for d in curated_patient_dir.iterdir() if d.is_dir()}
    if processed_patient_dir.exists():
        processed_studies = {d.name for d in processed_patient_dir.iterdir() if d.is_dir()}
    all_studies = sorted(curated_studies | processed_studies)

    if not all_studies:
        logger.warning(f"No studies found for {patient_id}")
        return

    n_studies = len(all_studies)
    n_mods = len(MODALITY_ORDER)
    n_views = len(VIEW_NAMES)

    apply_ieee_style()

    cell_size = 1.2
    group_header_h = 0.32
    mod_header_h = 0.30
    study_header_h = 0.28
    label_w = 0.45
    half_w = n_mods * cell_size
    gap_frac = 0.12

    study_block_h = study_header_h + n_views * cell_size
    total_studies_h = n_studies * study_block_h

    fig_w = 2 * half_w + label_w + 0.4
    fig_h = group_header_h + mod_header_h + total_studies_h + 0.2

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="black")

    # ── Top-level grid: 3 rows (group hdr, mod hdr, content) ──
    outer_gs = gridspec.GridSpec(
        3, 1,
        figure=fig,
        height_ratios=[group_header_h, mod_header_h, total_studies_h],
        hspace=0.005,
        left=label_w / fig_w,
        right=0.99,
        top=0.97,
        bottom=0.01,
    )

    # ── Row 0: Group headers (Curated | Processed) ──
    group_gs = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer_gs[0], wspace=gap_frac,
    )
    group_labels = ["Curated", "Processed"]
    group_colors = ["#2b5b84", "#84542b"]
    for gi, (label, color) in enumerate(zip(group_labels, group_colors)):
        ax_g = fig.add_subplot(group_gs[0, gi])
        ax_g.set_facecolor(color)
        ax_g.text(
            0.5, 0.5, label,
            transform=ax_g.transAxes, ha="center", va="center",
            fontsize=PLOT_SETTINGS["axes_labelsize"],
            fontweight="bold", color="white",
        )
        ax_g.set_xticks([])
        ax_g.set_yticks([])
        for spine in ax_g.spines.values():
            spine.set_visible(True)
            spine.set_color("0.4")
            spine.set_linewidth(0.8)

    # ── Row 1: Modality sub-headers (4 + 4) ──
    mod_split_gs = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer_gs[1], wspace=gap_frac,
    )
    for hi in range(2):
        half_mod_gs = gridspec.GridSpecFromSubplotSpec(
            1, n_mods, subplot_spec=mod_split_gs[0, hi], wspace=0.02,
        )
        for mi, mod in enumerate(MODALITY_ORDER):
            ax_m = fig.add_subplot(half_mod_gs[0, mi])
            ax_m.set_facecolor("0.15")
            ax_m.text(
                0.5, 0.5, mod.upper(),
                transform=ax_m.transAxes, ha="center", va="center",
                fontsize=PLOT_SETTINGS["font_size"],
                fontweight="bold", color="white",
            )
            ax_m.set_xticks([])
            ax_m.set_yticks([])
            for spine in ax_m.spines.values():
                spine.set_visible(True)
                spine.set_color("0.3")
                spine.set_linewidth(0.8)

    # ── Row 2: Study blocks ──
    content_gs = gridspec.GridSpecFromSubplotSpec(
        n_studies, 1, subplot_spec=outer_gs[2], hspace=0.03,
    )
    study_colors = plt.cm.Set2(np.linspace(0, 0.6, max(n_studies, 2)))

    for si, study_name in enumerate(all_studies):
        # Per-study block: header row + 3 image rows
        block_gs = gridspec.GridSpecFromSubplotSpec(
            1 + n_views, 1,
            subplot_spec=content_gs[si],
            height_ratios=[study_header_h] + [cell_size] * n_views,
            hspace=0.01,
        )

        # Study header (full width)
        ax_h = fig.add_subplot(block_gs[0, 0])
        ax_h.set_facecolor(study_colors[si])
        study_label = study_name.split("-")[-1]
        ax_h.text(
            0.5, 0.5, f"Study {study_label}",
            transform=ax_h.transAxes, ha="center", va="center",
            fontsize=PLOT_SETTINGS["font_size"],
            fontweight="bold", color="black",
        )
        ax_h.set_xticks([])
        ax_h.set_yticks([])
        for spine in ax_h.spines.values():
            spine.set_visible(True)
            spine.set_color("0.3")
            spine.set_linewidth(0.8)

        # Image rows: each row split into curated | processed halves
        curated_study_dir = curated_patient_dir / study_name
        processed_study_dir = processed_patient_dir / study_name

        for vi, view_name in enumerate(VIEW_NAMES):
            row_gs = gridspec.GridSpecFromSubplotSpec(
                1, 2,
                subplot_spec=block_gs[1 + vi, 0],
                wspace=gap_frac,
            )

            for hi, (src_dir, ext) in enumerate([
                (curated_study_dir, ".nrrd"),
                (processed_study_dir, ".nii.gz"),
            ]):
                half_gs = gridspec.GridSpecFromSubplotSpec(
                    1, n_mods,
                    subplot_spec=row_gs[0, hi],
                    wspace=0.02,
                )
                for mi, mod in enumerate(MODALITY_ORDER):
                    vol_path = src_dir / f"{mod}{ext}"
                    vol = load_volume(vol_path)
                    sl = get_middle_slices(vol)[vi] if vol is not None else None

                    ax = fig.add_subplot(half_gs[0, mi])
                    show_ylabel = (hi == 0 and mi == 0)
                    _render_image_cell(ax, sl, view_name, show_ylabel)

    # ── Save ──
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        save_path = output_stem.with_suffix(f".{fmt}")
        fig.savefig(
            save_path,
            dpi=PLOT_SETTINGS["dpi_print"],
            facecolor="black",
            bbox_inches="tight",
            pad_inches=0.05,
        )
        logger.info(f"  Saved {save_path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate curated vs. processed comparison montage."
    )
    parser.add_argument(
        "--patient", type=str, required=True,
        help="Patient ID (e.g. MenGrowth-0009).",
    )
    parser.add_argument(
        "--curated-dir", type=Path, default=CURATED_DIR,
        help="Path to curated dataset directory.",
    )
    parser.add_argument(
        "--processed-dir", type=Path, default=PROCESSED_DIR,
        help="Path to processed dataset directory.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR,
        help="Path for output images.",
    )
    parser.add_argument(
        "--formats", nargs="+", default=["png", "pdf"],
        help="Output formats (default: png pdf).",
    )
    args = parser.parse_args()

    curated_patient = args.curated_dir / args.patient
    processed_patient = args.processed_dir / args.patient

    if not curated_patient.exists() and not processed_patient.exists():
        logger.error(f"Patient not found in either dataset: {args.patient}")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = args.output_dir / f"{args.patient}_comparison"

    generate_comparison_montage(
        args.patient,
        curated_patient,
        processed_patient,
        output_stem,
        formats=args.formats,
    )
    logger.info("Done.")


if __name__ == "__main__":
    main()
