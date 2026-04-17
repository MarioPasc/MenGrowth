"""Figure — Per-label tumor showcase across longitudinal timepoints.

One row per BraTS-MEN segmentation label (ET, SNFH, NETC when present),
each showing the patient with the largest volume of that label across
all timepoints. The axial slice is chosen per-row as the slice where
the target label has maximum area at t0.

SynthSeg parcellation is rendered with low opacity as anatomical
context; the target tumor label is rendered prominently.

Run standalone::

    python -m mengrowth.synthseg.analysis.figures.fig_tumor_labels \
        --data-dir outputs/synthseg_analysis \
        --preprocessed-root /media/mpascual/PortableSSD/Meningiomas/MenGrowth/v5_final/MenGrowth-2025 \
        --synthseg-root    /media/mpascual/PortableSSD/Meningiomas/MenGrowth/v5_final/synthseg \
        --output outputs/synthseg_analysis/figures/figure_tumor_labels.pdf
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation

from mengrowth.synthseg.analysis.figures.common import (
    apply_pub_style,
    save_figure,
)

logger = logging.getLogger(__name__)

# Model output labels (after BraTS-MEN → BSF conversion):
#   1 = SNFH, 2 = ET.  NETC was dropped; label 3 is never produced.
BRATS_LABELS: dict[int, str] = {
    1: "SNFH / Edema",
    2: "Enhancing Tumor (ET)",
}

LABEL_COLORS: dict[int, tuple[float, float, float]] = {
    1: (0.20, 0.70, 0.95),
    2: (0.95, 0.20, 0.15),
}


# ---------------------------------------------------------------------------
# Patient selection
# ---------------------------------------------------------------------------


def _find_best_patient_per_label(
    preprocessed_root: Path,
    qc_df: pd.DataFrame,
) -> dict[int, str]:
    """For each BraTS label, return the patient_id with the largest total voxel count."""
    patient_totals: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    patient_ids = sorted(qc_df["patient_id"].unique())
    for pid in patient_ids:
        studies = sorted((preprocessed_root / pid).glob("*/seg.nii.gz"))
        for seg_path in studies:
            arr = nib.load(str(seg_path)).get_fdata()
            for lab in BRATS_LABELS:
                count = int((arr == lab).sum())
                if count > 0:
                    patient_totals[lab][pid] += count

    result: dict[int, str] = {}
    for lab in sorted(patient_totals):
        best_pid = max(patient_totals[lab], key=patient_totals[lab].__getitem__)
        result[lab] = best_pid
        logger.info(
            "Label %d (%s): best patient %s (%d total voxels)",
            lab, BRATS_LABELS[lab], best_pid, patient_totals[lab][best_pid],
        )
    return result


# ---------------------------------------------------------------------------
# Volume loading
# ---------------------------------------------------------------------------


def _load_study(
    preprocessed_root: Path,
    synthseg_root: Path,
    patient_id: str,
    study_id: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Load (t1n, parc, seg) arrays for a study."""
    t1n_path = preprocessed_root / patient_id / study_id / "t1n.nii.gz"
    parc_path = synthseg_root / patient_id / study_id / "synthseg_parc.nii.gz"
    seg_path = preprocessed_root / patient_id / study_id / "seg.nii.gz"

    for p, name in ((t1n_path, "t1n"), (parc_path, "parc")):
        if not p.exists():
            raise FileNotFoundError(f"Missing {name}: {p}")

    t1n = nib.load(str(t1n_path)).get_fdata(dtype=np.float32)
    parc = np.asarray(nib.load(str(parc_path)).get_fdata(), dtype=np.int32)
    seg = None
    if seg_path.exists():
        seg = np.asarray(nib.load(str(seg_path)).get_fdata(), dtype=np.int32)
    return t1n, parc, seg


def _get_patient_studies(
    qc_df: pd.DataFrame,
    patient_id: str,
    preprocessed_root: Path,
    synthseg_root: Path,
) -> list[dict]:
    """Load all studies for a patient, sorted by timepoint."""
    rows = (
        qc_df[qc_df["patient_id"] == patient_id]
        .sort_values("timepoint_index")
        .to_dict(orient="records")
    )
    studies: list[dict] = []
    for r in rows:
        t1n, parc, seg = _load_study(
            preprocessed_root, synthseg_root, patient_id, str(r["study_id"]),
        )
        studies.append({
            "study_id": str(r["study_id"]),
            "timepoint_index": int(r["timepoint_index"]),
            "qc_score": float(r["qc_score"]),
            "t1n": t1n,
            "parc": parc,
            "seg": seg,
        })
    return studies


# ---------------------------------------------------------------------------
# Slice selection
# ---------------------------------------------------------------------------


def _best_slice_for_label(studies: list[dict], label: int) -> int:
    """Axial slice where the target label has maximum area at t0."""
    seg = studies[0]["seg"]
    if seg is None:
        return studies[0]["t1n"].shape[2] // 2
    counts = (seg == label).sum(axis=(0, 1))
    if counts.max() == 0:
        return studies[0]["t1n"].shape[2] // 2
    return int(np.argmax(counts))


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _build_label_color_table(label_ids: np.ndarray) -> dict[int, np.ndarray]:
    """Parcellation label → RGBA (same logic as fig_qualitative)."""
    palettes = [plt.cm.tab20.colors, plt.cm.tab20b.colors, plt.cm.tab20c.colors]
    table_colors = [c for p in palettes for c in p]
    mapping: dict[int, np.ndarray] = {}
    idx = 0
    for lid in label_ids:
        if int(lid) == 0:
            continue
        rgb = np.asarray(table_colors[idx % len(table_colors)], dtype=np.float32)
        mapping[int(lid)] = np.concatenate([rgb, [1.0]]).astype(np.float32)
        idx += 1
    return mapping


def _render_cell(
    ax: plt.Axes,
    t1n: np.ndarray,
    parc: np.ndarray,
    seg: np.ndarray | None,
    slice_idx: int,
    target_label: int,
    label_colors: dict[int, np.ndarray],
    qc_score: float,
) -> None:
    """Render one montage cell: T1n + faint parcellation + prominent tumor label."""
    ax.set_facecolor("black")
    t1n_slice = t1n[:, :, slice_idx]
    parc_slice = parc[:, :, slice_idx]

    lo, hi = np.percentile(t1n_slice, [1.0, 99.0])
    if hi <= lo:
        hi = lo + 1e-6
    ax.imshow(
        t1n_slice.T, cmap="gray", origin="lower",
        vmin=lo, vmax=hi, interpolation="nearest",
    )

    # SynthSeg parcellation: very faint fill + subdued contours
    parc_fill = np.zeros((*parc_slice.shape, 4), dtype=np.float32)
    parc_contour = np.zeros((*parc_slice.shape, 4), dtype=np.float32)
    for lid, rgba in label_colors.items():
        mask = parc_slice == lid
        if not mask.any():
            continue
        boundary = binary_dilation(mask, iterations=1) & ~mask
        interior = mask & ~boundary
        parc_fill[interior, :3] = rgba[:3]
        parc_fill[interior, 3] = 0.08
        parc_contour[boundary, :3] = rgba[:3]
        parc_contour[boundary, 3] = 0.6
    ax.imshow(parc_fill.transpose(1, 0, 2), origin="lower", interpolation="nearest")
    ax.imshow(parc_contour.transpose(1, 0, 2), origin="lower", interpolation="nearest")

    # Target tumor label: prominent fill + solid contour
    if seg is not None:
        tumor_slice = (seg[:, :, slice_idx] == target_label)
        if tumor_slice.any():
            rgb = LABEL_COLORS[target_label]
            tumor_boundary = binary_dilation(tumor_slice, iterations=1) & ~tumor_slice
            tumor_interior = tumor_slice & ~tumor_boundary
            tumor_overlay = np.zeros((*tumor_slice.shape, 4), dtype=np.float32)
            tumor_overlay[tumor_interior, :3] = rgb
            tumor_overlay[tumor_interior, 3] = 0.8
            tumor_overlay[tumor_boundary, :3] = rgb
            tumor_overlay[tumor_boundary, 3] = 1.0
            ax.imshow(
                tumor_overlay.transpose(1, 0, 2),
                origin="lower", interpolation="nearest",
            )

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ("top", "right", "left", "bottom"):
        ax.spines[spine].set_visible(False)

    txt = ax.text(
        0.97, 0.03, f"q = {qc_score:.3f}",
        transform=ax.transAxes, color="white",
        fontsize=9, fontweight="bold", ha="right", va="bottom",
    )
    txt.set_path_effects([
        pe.Stroke(linewidth=1.8, foreground="black"),
        pe.Normal(),
    ])


def _blank_cell(ax: plt.Axes) -> None:
    ax.set_facecolor("black")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ("top", "right", "left", "bottom"):
        ax.spines[spine].set_visible(False)


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


def make_figure(
    data_dir: Path,
    preprocessed_root: Path,
    synthseg_root: Path,
) -> plt.Figure:
    """Build the per-label tumor showcase figure."""
    qc_csv = data_dir / "cohort" / "cohort_qc.csv"
    if not qc_csv.exists():
        raise FileNotFoundError(qc_csv)
    qc_df = pd.read_csv(qc_csv)

    best_per_label = _find_best_patient_per_label(preprocessed_root, qc_df)
    if not best_per_label:
        raise ValueError("No tumor segmentations found in the dataset.")

    active_labels = sorted(best_per_label.keys())
    rows_data: list[tuple[int, str, list[dict]]] = []
    for lab in active_labels:
        pid = best_per_label[lab]
        studies = _get_patient_studies(qc_df, pid, preprocessed_root, synthseg_root)
        rows_data.append((lab, pid, studies))

    n_rows = len(rows_data)
    n_cols = max(len(s) for _, _, s in rows_data)

    apply_pub_style()
    fig_w = max(7.0, 2.6 * n_cols + 1.3)
    fig_h = 2.55 * n_rows + 0.65
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(fig_w, fig_h),
        gridspec_kw={
            "wspace": 0.03, "hspace": 0.04,
            "left": 0.08, "right": 0.985,
            "bottom": 0.02, "top": 1 - 0.55 / fig_h,
        },
        squeeze=False,
    )
    fig.patch.set_facecolor("black")

    # Shared parcellation colour table
    all_parc = [np.unique(s["parc"]) for _, _, studies in rows_data for s in studies]
    all_labels_arr = np.unique(np.concatenate(all_parc))
    parc_colors = _build_label_color_table(all_labels_arr)

    # Column headers
    for col in range(n_cols):
        axes[0, col].set_title(
            rf"$t_{{{col}}}$", fontsize=18, pad=6,
            color="white", fontweight="bold",
        )

    panel_letters = ["(A)", "(B)", "(C)"]

    for row_idx, (lab, pid, studies) in enumerate(rows_data):
        slice_idx = _best_slice_for_label(studies, lab)
        rgb = LABEL_COLORS[lab]

        for col in range(n_cols):
            ax = axes[row_idx, col]
            if col >= len(studies):
                _blank_cell(ax)
                continue
            s = studies[col]
            _render_cell(
                ax, s["t1n"], s["parc"], s["seg"],
                slice_idx, lab, parc_colors, s["qc_score"],
            )

        # Row label
        label_name = BRATS_LABELS[lab]
        color_hex = "#{:02x}{:02x}{:02x}".format(
            int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255),
        )
        row_label = f"Label {lab}: {label_name}\n{pid}"
        left_ax = axes[row_idx, 0]
        left_ax.text(
            -0.07, 0.5, row_label,
            transform=left_ax.transAxes,
            fontsize=10, fontweight="bold",
            ha="right", va="center",
            color=color_hex,
        )
        y_panel = 1.12 if row_idx == 0 else 1.04
        letter = panel_letters[row_idx] if row_idx < len(panel_letters) else ""
        left_ax.text(
            -0.28, y_panel, letter,
            transform=left_ax.transAxes,
            fontsize=13, fontweight="bold",
            ha="left", va="bottom", color="white",
        )

    return fig


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path("outputs/synthseg_analysis"),
    )
    parser.add_argument(
        "--preprocessed-root", type=Path,
        default=Path(
            "/media/mpascual/PortableSSD/Meningiomas/MenGrowth/v5_final/MenGrowth-2025"
        ),
    )
    parser.add_argument(
        "--synthseg-root", type=Path,
        default=Path(
            "/media/mpascual/PortableSSD/Meningiomas/MenGrowth/v5_final/synthseg"
        ),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("outputs/synthseg_analysis/figures/figure_tumor_labels.pdf"),
    )
    parser.add_argument("--also-png", action="store_true")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    fig = make_figure(args.data_dir, args.preprocessed_root, args.synthseg_root)
    out = save_figure(fig, args.output, dpi=args.dpi)
    logger.info("Saved %s", out)

    if args.also_png:
        fig2 = make_figure(args.data_dir, args.preprocessed_root, args.synthseg_root)
        save_figure(fig2, args.output.with_suffix(".png"), dpi=args.dpi)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
