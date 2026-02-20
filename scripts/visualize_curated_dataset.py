#!/usr/bin/env python3
"""Generate axial-sagittal-coronal montage per patient from the curated dataset.

For each patient, produces a single PNG with:
  - Columns: modalities (t1n, t1c, t2w, t2f) grouped by longitudinal study
  - Rows: axial, sagittal, coronal (middle slices)
  - Top boxes group columns by study

Usage:
    # Single patient (for testing):
    python scripts/visualize_curated_dataset.py --patient MenGrowth-0001

    # All patients:
    python scripts/visualize_curated_dataset.py
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

DATASET_DIR = Path(
    "/media/mpascual/PortableSSD/Meningiomas/MenGrowth/curated/dataset/MenGrowth-2025"
)
OUTPUT_DIR = Path(
    "/media/mpascual/PortableSSD/Meningiomas/MenGrowth/curated/quality/patient_montages"
)
MODALITY_ORDER = ["t1n", "t1c", "t2w", "t2f"]
VIEW_NAMES = ["Axial", "Sagittal", "Coronal"]


def load_volume(path: Path) -> Optional[np.ndarray]:
    """Load NRRD, reorient to RAI, resample to 1 mm³ isotropic (NN).

    RAI = Right-Anterior-Inferior gives standard radiological indexing:
      - axis 0 = axial slice index (inferior → superior)
      - axis 1 = coronal slice index (anterior → posterior)
      - axis 2 = sagittal slice index (right → left)

    Nearest-neighbor resampling to 1 mm³ ensures sagittal/coronal views
    have correct proportions without blurring.

    Args:
        path: Path to the .nrrd file.

    Returns:
        3D numpy array in (z, y, x) order, or None if file missing.
    """
    if not path.exists():
        return None
    img = sitk.ReadImage(str(path))
    img = sitk.DICOMOrient(img, "RAI")

    # Resample to 1 mm³ isotropic with nearest-neighbor
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
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    img = resampler.Execute(img)

    arr = sitk.GetArrayFromImage(img)  # (z, y, x) in RAI
    return arr.astype(np.float32)


def get_middle_slices(
    vol: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract middle axial, sagittal, and coronal slices.

    Assumes vol shape is (z, y, x) from RAI-oriented SimpleITK.

    Args:
        vol: 3D numpy array in (z, y, x) order.

    Returns:
        Tuple of (axial, sagittal, coronal) 2D slices.
    """
    nz, ny, nx = vol.shape
    axial = vol[nz // 2, :, :]  # fix z → (y, x)
    sagittal = vol[:, :, nx // 2]  # fix x → (z, y)
    coronal = vol[:, ny // 2, :]  # fix y → (z, x)
    return axial, sagittal, coronal


def normalize_slice(s: np.ndarray) -> np.ndarray:
    """Robust percentile normalization to [0, 1].

    Args:
        s: 2D slice array.

    Returns:
        Normalized array clipped to [0, 1].
    """
    p2, p98 = np.percentile(s, [2, 98])
    if p98 - p2 < 1e-6:
        return np.zeros_like(s, dtype=np.float32)
    return np.clip((s - p2) / (p98 - p2), 0, 1)


def generate_patient_montage(
    patient_dir: Path,
    output_path: Path,
) -> None:
    """Generate a multi-study montage PNG for one patient.

    Layout: 3 rows (axial, sagittal, coronal) x (n_studies * n_modalities) columns.
    Top-level boxes group columns by study.

    Args:
        patient_dir: Path to patient directory containing study subdirectories.
        output_path: Path for the output PNG.
    """
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

    # Sizing: scale cell width down for wide patients
    cell_w = min(1.2, 8.0 / max(n_cols, 1))
    cell_h = cell_w  # square cells for aspect='equal' isotropic data
    label_w = 0.45
    fig_w = n_cols * cell_w + label_w
    header_h = 0.35
    fig_h = n_rows * cell_h + header_h + 0.1

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="black")

    # Main gridspec: header row + image grid
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
        study_label = study_dir.name.split("-")[-1]  # "000", "001", ...
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

                # Row labels on leftmost column
                if col == 0:
                    ax.set_ylabel(
                        view_name,
                        fontsize=PLOT_SETTINGS["tick_labelsize"],
                        color="white",
                        rotation=90,
                        labelpad=4,
                    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        dpi=PLOT_SETTINGS["dpi_print"],
        facecolor="black",
        bbox_inches="tight",
        pad_inches=0.05,
    )
    plt.close(fig)
    logger.info(f"  Saved {output_path}")


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
        help="Path for output PNGs.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.patient:
        patient_dir = args.dataset_dir / args.patient
        if not patient_dir.exists():
            logger.error(f"Patient directory not found: {patient_dir}")
            sys.exit(1)
        generate_patient_montage(
            patient_dir,
            output_dir / f"{args.patient}.png",
        )
    else:
        patients = sorted(
            [d for d in args.dataset_dir.iterdir() if d.is_dir()],
            key=lambda p: p.name,
        )
        logger.info(f"Processing {len(patients)} patients ...")
        for patient_dir in patients:
            generate_patient_montage(
                patient_dir,
                output_dir / f"{patient_dir.name}.png",
            )
        logger.info(f"Done. {len(patients)} montages saved to {output_dir}")


if __name__ == "__main__":
    main()
