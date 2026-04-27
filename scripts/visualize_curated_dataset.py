#!/usr/bin/env python3
"""Generate axial-sagittal-coronal montage per patient from the curated dataset.

For each patient, produces an image with:
  - Default (horizontal): Columns = modalities grouped by study, Rows = views
  - Vertical (--vertical): Columns = modalities, Rows = views grouped by study

Usage:
    python scripts/visualize_curated_dataset.py --patient MenGrowth-0001
    python scripts/visualize_curated_dataset.py --patient MenGrowth-0001 --vertical
    python scripts/visualize_curated_dataset.py --workers 8
    python scripts/visualize_curated_dataset.py --patient MenGrowth-0009 --vertical --formats png pdf
"""

import argparse
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
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

DATASET_DIR = Path(
    "/media/mpascual/PortableSSD/Meningiomas/MenGrowth/curated/dataset/MenGrowth-2025"
)
OUTPUT_DIR = Path(
    "/media/mpascual/PortableSSD/Meningiomas/MenGrowth/curated/quality/patient_montages"
)
MODALITY_ORDER = ["t1n", "t1c", "t2w", "t2f"]
VIEW_NAMES = ["Axial", "Sagittal", "Coronal"]


def load_volume(path: Path) -> Optional[np.ndarray]:
    """Load image, reorient to RAI, resample to 1 mm isotropic (BSpline).

    RAI = Right-Anterior-Inferior gives standard radiological indexing:
      - axis 0 = axial slice index (inferior -> superior)
      - axis 1 = coronal slice index (anterior -> posterior)
      - axis 2 = sagittal slice index (right -> left)

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

    arr = sitk.GetArrayFromImage(img)  # (z, y, x) in RAI
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


def _save_figure(
    fig: plt.Figure,
    output_stem: Path,
    formats: list[str],
) -> None:
    """Save figure in multiple formats."""
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


def generate_patient_montage(
    patient_dir: Path,
    output_stem: Path,
    formats: list[str] | None = None,
) -> None:
    """Horizontal layout: studies as column groups, views as rows.

    Layout: 3 rows (axial, sagittal, coronal) x (n_studies * 4) columns.
    Top boxes group columns by study.

    Parameters
    ----------
    patient_dir : Path
        Patient directory containing study subdirectories.
    output_stem : Path
        Output path without extension (extensions added per format).
    formats : list[str] or None
        Output formats (default: ["png"]).
    """
    if formats is None:
        formats = ["png"]
    patient_id = patient_dir.name
    studies = sorted(
        [d for d in patient_dir.iterdir() if d.is_dir()],
        key=lambda p: p.name,
    )
    if not studies:
        logger.warning(f"No studies found for {patient_id}")
        return

    n_studies = len(studies)
    n_mods = len(MODALITY_ORDER)
    n_cols = n_studies * n_mods
    n_rows = 3  # axial, sagittal, coronal

    apply_ieee_style()

    cell_w = min(1.2, 8.0 / max(n_cols, 1))
    cell_h = cell_w
    label_w = 0.45
    fig_w = n_cols * cell_w + label_w
    header_h = 0.35
    fig_h = n_rows * cell_h + header_h + 0.1

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="black")

    outer_gs = gridspec.GridSpec(
        2,
        1,
        figure=fig,
        height_ratios=[header_h, n_rows * cell_h],
        hspace=0.01,
        left=label_w / fig_w,
        right=1.0,
        top=0.97,
        bottom=0.01,
    )

    # ── Study header boxes ──
    header_gs = gridspec.GridSpecFromSubplotSpec(
        1,
        n_studies,
        subplot_spec=outer_gs[0],
        wspace=0.06,
    )
    study_colors = plt.cm.Set2(np.linspace(0, 0.6, max(n_studies, 2)))
    for si, study_dir in enumerate(studies):
        ax_h = fig.add_subplot(header_gs[0, si])
        ax_h.set_facecolor(study_colors[si])
        study_label = study_dir.name.split("-")[-1]
        ax_h.text(
            0.5,
            0.5,
            f"Study {study_label}",
            transform=ax_h.transAxes,
            ha="center",
            va="center",
            fontsize=PLOT_SETTINGS["font_size"],
            fontweight="bold",
            color="black",
        )
        ax_h.set_xticks([])
        ax_h.set_yticks([])
        for spine in ax_h.spines.values():
            spine.set_visible(True)
            spine.set_color("0.3")
            spine.set_linewidth(0.8)

    # ── Image grid ──
    img_gs = gridspec.GridSpecFromSubplotSpec(
        n_rows,
        n_cols,
        subplot_spec=outer_gs[1],
        wspace=0.02,
        hspace=0.02,
    )

    for si, study_dir in enumerate(studies):
        for mi, mod in enumerate(MODALITY_ORDER):
            col = si * n_mods + mi
            vol_path = study_dir / f"{mod}.nrrd"
            vol = load_volume(vol_path)

            if vol is not None:
                slices = get_middle_slices(vol)
            else:
                slices = (None, None, None)

            for ri, (sl, view_name) in enumerate(zip(slices, VIEW_NAMES)):
                ax = fig.add_subplot(img_gs[ri, col])
                ax.set_facecolor("black")

                if sl is not None:
                    img = normalize_slice(sl)
                    ax.imshow(
                        img,
                        cmap="gray",
                        origin="upper",
                        aspect="equal",
                        interpolation="bilinear",
                    )
                else:
                    ax.text(
                        0.5,
                        0.5,
                        "N/A",
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="0.4",
                    )

                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)

                if col == 0:
                    ax.set_ylabel(
                        view_name,
                        fontsize=PLOT_SETTINGS["tick_labelsize"],
                        color="white",
                        rotation=90,
                        labelpad=4,
                    )

    _save_figure(fig, output_stem, formats)
    plt.close(fig)


def generate_patient_montage_vertical(
    patient_dir: Path,
    output_stem: Path,
    formats: list[str] | None = None,
) -> None:
    """Vertical layout: studies as rows, modalities as columns.

    For each study, generates a colored header row followed by 3 image rows
    (axial, sagittal, coronal), yielding 4 x n_studies total rows and
    4 columns (one per modality).

    Parameters
    ----------
    patient_dir : Path
        Patient directory containing study subdirectories.
    output_stem : Path
        Output path without extension.
    formats : list[str] or None
        Output formats (default: ["png"]).
    """
    if formats is None:
        formats = ["png"]
    patient_id = patient_dir.name
    studies = sorted(
        [d for d in patient_dir.iterdir() if d.is_dir()],
        key=lambda p: p.name,
    )
    if not studies:
        logger.warning(f"No studies found for {patient_id}")
        return

    n_studies = len(studies)
    n_mods = len(MODALITY_ORDER)
    n_views = len(VIEW_NAMES)

    apply_ieee_style()

    cell_size = 1.2
    mod_header_h = 0.35
    study_header_h = 0.30
    label_w = 0.45
    study_block_h = study_header_h + n_views * cell_size
    total_studies_h = n_studies * study_block_h

    fig_w = n_mods * cell_size + label_w
    fig_h = mod_header_h + total_studies_h + 0.2

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="black")

    outer_gs = gridspec.GridSpec(
        2,
        1,
        figure=fig,
        height_ratios=[mod_header_h, total_studies_h],
        hspace=0.01,
        left=label_w / fig_w,
        right=0.99,
        top=0.97,
        bottom=0.01,
    )

    # ── Modality headers ──
    mod_gs = gridspec.GridSpecFromSubplotSpec(
        1, n_mods, subplot_spec=outer_gs[0], wspace=0.02,
    )
    for mi, mod in enumerate(MODALITY_ORDER):
        ax_m = fig.add_subplot(mod_gs[0, mi])
        ax_m.set_facecolor("0.15")
        ax_m.text(
            0.5, 0.5, mod.upper(),
            transform=ax_m.transAxes, ha="center", va="center",
            fontsize=PLOT_SETTINGS["font_size"], fontweight="bold", color="white",
        )
        ax_m.set_xticks([])
        ax_m.set_yticks([])
        for spine in ax_m.spines.values():
            spine.set_visible(True)
            spine.set_color("0.3")
            spine.set_linewidth(0.8)

    # ── Study blocks ──
    studies_gs = gridspec.GridSpecFromSubplotSpec(
        n_studies, 1, subplot_spec=outer_gs[1], hspace=0.03,
    )
    study_colors = plt.cm.Set2(np.linspace(0, 0.6, max(n_studies, 2)))

    for si, study_dir in enumerate(studies):
        block_gs = gridspec.GridSpecFromSubplotSpec(
            1 + n_views, n_mods,
            subplot_spec=studies_gs[si],
            height_ratios=[study_header_h] + [cell_size] * n_views,
            hspace=0.01, wspace=0.02,
        )

        # Study header (spans all modality columns)
        ax_h = fig.add_subplot(block_gs[0, :])
        ax_h.set_facecolor(study_colors[si])
        study_label = study_dir.name.split("-")[-1]
        ax_h.text(
            0.5, 0.5, f"Study {study_label}",
            transform=ax_h.transAxes, ha="center", va="center",
            fontsize=PLOT_SETTINGS["font_size"], fontweight="bold", color="black",
        )
        ax_h.set_xticks([])
        ax_h.set_yticks([])
        for spine in ax_h.spines.values():
            spine.set_visible(True)
            spine.set_color("0.3")
            spine.set_linewidth(0.8)

        # Image rows
        for mi, mod in enumerate(MODALITY_ORDER):
            vol_path = study_dir / f"{mod}.nrrd"
            vol = load_volume(vol_path)
            slices = get_middle_slices(vol) if vol is not None else (None, None, None)

            for vi, (sl, view_name) in enumerate(zip(slices, VIEW_NAMES)):
                ax = fig.add_subplot(block_gs[1 + vi, mi])
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

                if mi == 0:
                    ax.set_ylabel(
                        view_name,
                        fontsize=PLOT_SETTINGS["tick_labelsize"],
                        color="white", rotation=90, labelpad=4,
                    )

    _save_figure(fig, output_stem, formats)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate patient montage visualizations from curated dataset."
    )
    parser.add_argument(
        "--patient",
        type=str,
        default=None,
        help="Single patient ID (e.g. MenGrowth-0001). If omitted, process all.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DATASET_DIR,
        help="Path to curated dataset directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Path for output images.",
    )
    parser.add_argument(
        "--vertical",
        action="store_true",
        help="Use vertical layout (studies as rows, modalities as columns).",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png"],
        help="Output formats (default: png). E.g. --formats png pdf",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Number of parallel workers (default: min(8, cpu_count)).",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    montage_fn = (
        generate_patient_montage_vertical if args.vertical
        else generate_patient_montage
    )

    if args.patient:
        patient_dir = args.dataset_dir / args.patient
        if not patient_dir.exists():
            logger.error(f"Patient directory not found: {patient_dir}")
            sys.exit(1)
        montage_fn(
            patient_dir,
            output_dir / args.patient,
            formats=args.formats,
        )
    else:
        patients = sorted(
            [d for d in args.dataset_dir.iterdir() if d.is_dir()],
            key=lambda p: p.name,
        )
        logger.info(f"Processing {len(patients)} patients with {args.workers} workers ...")
        tasks = [(pd, output_dir / pd.name) for pd in patients]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(montage_fn, p, o, args.formats): p.name
                for p, o in tasks
            }
            done = 0
            for fut in as_completed(futures):
                done += 1
                pid = futures[fut]
                try:
                    fut.result()
                except Exception:
                    logger.exception(f"Failed for {pid}")
                if done % 20 == 0:
                    logger.info(f"  Progress: {done}/{len(patients)}")
        logger.info(f"Done. {len(patients)} montages saved to {output_dir}")


if __name__ == "__main__":
    main()
