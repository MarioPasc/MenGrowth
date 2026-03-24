"""Patient-level summary montage visualization.

Generates a montage with 3 rows (axial, sagittal, coronal) and N*4 columns
(one group of 4 modalities per study), enabling quick visual QC of all
preprocessed volumes for a patient in a single image.
"""

from pathlib import Path
from typing import List, Optional
import logging

import numpy as np
import nibabel as nib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

logger = logging.getLogger(__name__)

# View labels and slice axis mapping
VIEWS = [
    ("Axial", 2),  # slice along z-axis
    ("Sagittal", 0),  # slice along x-axis
    ("Coronal", 1),  # slice along y-axis
]

MODALITY_LABELS = {
    "t1c": "T1c",
    "t1n": "T1n",
    "t2w": "T2w",
    "t2f": "FLAIR",
}


def generate_patient_summary(
    patient_id: str,
    study_dirs: List[Path],
    modalities: List[str],
    output_path: Path,
    output_root: Optional[Path] = None,
) -> Optional[Path]:
    """Generate a patient summary montage of all preprocessed volumes.

    Creates a figure with 3 rows (axial, sagittal, coronal) and N*M columns
    where N = number of studies and M = number of modalities. Each cell shows
    the middle slice of the corresponding view.

    Args:
        patient_id: Patient identifier (e.g., "MenGrowth-0001").
        study_dirs: List of study directory paths (source of NRRD input).
        modalities: List of modality names (e.g., ["t1c", "t1n", "t2w", "t2f"]).
        output_path: Path to save the output PNG.
        output_root: If set, look for NIfTI outputs here instead of study_dirs.
            Used in test mode where outputs go to a separate directory.

    Returns:
        Path to saved PNG, or None if no data was found.
    """
    n_studies = len(study_dirs)
    n_mods = len(modalities)
    n_cols = n_studies * n_mods
    n_rows = len(VIEWS)

    if n_cols == 0:
        logger.warning(f"No studies to visualize for {patient_id}")
        return None

    # Collect all volumes: volumes[study_idx][mod_idx] = 3D array or None
    volumes: List[List[Optional[np.ndarray]]] = []
    study_labels: List[str] = []

    for study_dir in sorted(study_dirs, key=lambda d: d.name):
        study_name = study_dir.name
        study_labels.append(study_name)
        study_vols: List[Optional[np.ndarray]] = []

        for modality in modalities:
            # Determine where to find the preprocessed NIfTI
            if output_root is not None:
                nifti_path = (
                    output_root / patient_id / study_name / f"{modality}.nii.gz"
                )
            else:
                nifti_path = study_dir / f"{modality}.nii.gz"

            if nifti_path.exists():
                try:
                    img = nib.load(str(nifti_path))
                    data = np.asarray(img.dataobj, dtype=np.float32)
                    study_vols.append(data)
                except Exception as e:
                    logger.warning(f"Failed to load {nifti_path}: {e}")
                    study_vols.append(None)
            else:
                study_vols.append(None)

        volumes.append(study_vols)

    # Check we have at least one volume
    if all(v is None for study_vols in volumes for v in study_vols):
        logger.warning(f"No preprocessed NIfTI files found for {patient_id}")
        return None

    # Layout constants — leave space between study groups
    group_gap = 0.03  # fraction of figure width between study groups
    left_margin = 0.01
    right_margin = 0.01
    top_margin = 0.07  # space for study headers
    bottom_margin = 0.01

    col_width = 2.2
    row_height = 2.2
    fig_width = n_cols * col_width + (n_studies - 1) * col_width * 0.5 + 1.0
    fig_height = n_rows * row_height + 1.5

    fig = plt.figure(figsize=(fig_width, fig_height), facecolor="black")

    # Compute per-study GridSpec regions with gaps between groups
    usable_w = 1.0 - left_margin - right_margin
    total_gap = group_gap * (n_studies - 1) if n_studies > 1 else 0
    study_width = (usable_w - total_gap) / n_studies

    from matplotlib.patches import FancyBboxPatch

    for study_idx, (study_vols, study_label) in enumerate(zip(volumes, study_labels)):
        # Compute horizontal bounds for this study group
        group_left = left_margin + study_idx * (study_width + group_gap)
        group_right = group_left + study_width

        # Create a GridSpec for this study's columns
        study_gs = GridSpec(
            n_rows,
            n_mods,
            figure=fig,
            wspace=0.04,
            hspace=0.08,
            left=group_left,
            right=group_right,
            top=1.0 - top_margin,
            bottom=bottom_margin,
        )

        for mod_idx, (modality, vol) in enumerate(zip(modalities, study_vols)):
            for row, (view_name, axis) in enumerate(VIEWS):
                ax = fig.add_subplot(study_gs[row, mod_idx])
                ax.set_facecolor("black")

                if vol is not None:
                    mid = vol.shape[axis] // 2
                    slc = _extract_slice(vol, axis, mid)
                    vmin, vmax = _robust_range(vol)
                    ax.imshow(
                        slc.T,
                        cmap="gray",
                        origin="lower",
                        vmin=vmin,
                        vmax=vmax,
                        aspect="auto",
                    )

                ax.set_xticks([])
                ax.set_yticks([])
                ax.spines[:].set_visible(False)

        # Study header above group
        x_center = (group_left + group_right) / 2.0
        fig.text(
            x_center,
            1.0 - top_margin + 0.02,
            study_label,
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="white",
        )

        # Bounding box around this study group
        pad = 0.006
        rect = FancyBboxPatch(
            (group_left - pad, bottom_margin - pad),
            study_width + 2 * pad,
            (1.0 - top_margin - bottom_margin) + 2 * pad,
            boxstyle="round,pad=0.004",
            linewidth=1.0,
            edgecolor="white",
            facecolor="none",
            alpha=0.4,
            transform=fig.transFigure,
            clip_on=False,
        )
        fig.patches.append(rect)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="black")
    plt.close(fig)

    logger.info(f"  Patient summary saved: {output_path}")
    return output_path


def _extract_slice(vol: np.ndarray, axis: int, index: int) -> np.ndarray:
    """Extract a 2D slice from a 3D volume along the given axis.

    Args:
        vol: 3D numpy array.
        axis: Axis to slice along (0=x, 1=y, 2=z).
        index: Slice index.

    Returns:
        2D numpy array.
    """
    if axis == 0:
        return vol[index, :, :]
    elif axis == 1:
        return vol[:, index, :]
    else:
        return vol[:, :, index]


def _robust_range(vol: np.ndarray) -> tuple:
    """Compute robust intensity range using percentiles.

    Uses P1 and P99 of nonzero voxels to avoid outlier-dominated display.

    Args:
        vol: 3D numpy array.

    Returns:
        Tuple of (vmin, vmax) for display.
    """
    nonzero = vol[vol > 0]
    if len(nonzero) == 0:
        return 0.0, 1.0
    vmin = float(np.percentile(nonzero, 1))
    vmax = float(np.percentile(nonzero, 99))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax
