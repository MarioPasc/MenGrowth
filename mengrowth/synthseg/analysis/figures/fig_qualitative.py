"""Figure 2 — Qualitative parcellation overlay montage.

Axial slices of the T1n volume with SynthSeg parcellation boundaries
overlaid, rendered for two showcase patients (one best-case, one
worst-case, both with ≥ 3 studies). Each row shows a patient across
their longitudinal timepoints; each column is a timepoint.

Run standalone::

    python -m mengrowth.synthseg.analysis.figures.fig_qualitative \
        --data-dir outputs/synthseg_analysis \
        --preprocessed-root /media/mpascual/PortableSSD/Meningiomas/MenGrowth/v5_final/MenGrowth-2025 \
        --synthseg-root    /media/mpascual/PortableSSD/Meningiomas/MenGrowth/v5_final/synthseg \
        --output outputs/synthseg_analysis/figures/figure2_qualitative.pdf
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation

from mengrowth.synthseg.analysis.figures.common import (
    apply_pub_style,
    paul_tol_color,
    save_figure,
    select_showcase_patients,
)

logger = logging.getLogger(__name__)


# BraTS-MEN segmentation labels (per BraTS 2025 Meningioma challenge):
#   1 = Enhancing Tumor (ET)
#   2 = Surrounding Non-enhancing FLAIR Hyperintensity (SNFH / edema)
#   3 = Non-Enhancing Tumor Core (NETC)
# The conventional aggregated annotations are:
#   WT (Whole Tumor)  = {1, 2, 3}
#   TC (Tumor Core)   = {1, 3}      ← excludes edema
#   ET (Enhancing)    = {1}
TUMOR_REGIONS: dict[str, tuple[int, ...]] = {
    "WT": (1, 2, 3),
    "TC": (1, 3),
    "ET": (1,),
}


def _tumor_mask_from_seg(seg: np.ndarray, region: str) -> np.ndarray:
    """Boolean mask for the requested BraTS aggregated region (WT/TC/ET)."""
    if region not in TUMOR_REGIONS:
        raise ValueError(f"unknown tumor region {region!r}; expected one of {list(TUMOR_REGIONS)}")
    labels = TUMOR_REGIONS[region]
    seg_int = np.asarray(seg, dtype=np.int32)
    return np.isin(seg_int, labels)


@dataclass(frozen=True)
class StudyAssets:
    """Paths + loaded volumes for a single study cell."""

    patient_id: str
    study_id: str
    timepoint_index: int
    qc_score: float
    t1n: np.ndarray
    parc: np.ndarray
    tumor: np.ndarray | None  # may be None when seg.nii.gz is absent


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


def _load_volumes(
    preprocessed_root: Path,
    synthseg_root: Path,
    patient_id: str,
    study_id: str,
    tumor_region: str = "TC",
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Load (t1n, parc, tumor) arrays for a study; tumor may be None.

    ``tumor_region`` selects the BraTS aggregated region (WT, TC, ET). Default
    ``TC`` (tumor core) excludes peritumoral edema.
    """
    t1n_path = preprocessed_root / patient_id / study_id / "t1n.nii.gz"
    parc_path = synthseg_root / patient_id / study_id / "synthseg_parc.nii.gz"
    tumor_path = preprocessed_root / patient_id / study_id / "seg.nii.gz"

    for p, label in ((t1n_path, "t1n"), (parc_path, "parc")):
        if not p.exists():
            raise FileNotFoundError(f"Missing {label}: {p}")

    t1n = nib.load(str(t1n_path)).get_fdata(dtype=np.float32)
    parc = np.asarray(nib.load(str(parc_path)).get_fdata(), dtype=np.int32)
    if t1n.shape != parc.shape:
        raise ValueError(
            f"Shape mismatch t1n {t1n.shape} vs parc {parc.shape} "
            f"for {patient_id}/{study_id}"
        )
    tumor = None
    if tumor_path.exists():
        tumor_arr = nib.load(str(tumor_path)).get_fdata()
        tumor = _tumor_mask_from_seg(tumor_arr, tumor_region)
    return t1n, parc, tumor


def _assemble_patient_studies(
    qc_df: pd.DataFrame,
    patient_id: str,
    preprocessed_root: Path,
    synthseg_root: Path,
    tumor_region: str = "TC",
) -> list[StudyAssets]:
    rows = (
        qc_df[qc_df["patient_id"] == patient_id]
        .sort_values("timepoint_index")
        .to_dict(orient="records")
    )
    out: list[StudyAssets] = []
    for r in rows:
        t1n, parc, tumor = _load_volumes(
            preprocessed_root, synthseg_root, patient_id, r["study_id"],
            tumor_region=tumor_region,
        )
        out.append(
            StudyAssets(
                patient_id=patient_id,
                study_id=str(r["study_id"]),
                timepoint_index=int(r["timepoint_index"]),
                qc_score=float(r["qc_score"]),
                t1n=t1n,
                parc=parc,
                tumor=tumor,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _boundary_mask(parc_slice: np.ndarray) -> np.ndarray:
    """Return a 2D boolean mask of per-label parcellation boundaries.

    Implementation: for every label ≥ 1, ``dilate(mask) & ~mask`` yields a
    single-voxel-thick outer boundary; the union over all labels forms
    inter-region edges exactly once.
    """
    boundary = np.zeros_like(parc_slice, dtype=bool)
    labels = np.unique(parc_slice)
    for label_id in labels:
        if label_id == 0:
            continue
        mask = parc_slice == label_id
        dilated = binary_dilation(mask, iterations=1)
        boundary |= dilated & ~mask
    return boundary


def _build_label_color_table(label_ids: np.ndarray) -> dict[int, np.ndarray]:
    """Map each non-zero label id to a distinct RGBA colour (alpha=1.0).

    Uses ``matplotlib`` ``tab20`` + ``tab20b`` + ``tab20c`` concatenated so
    all ~30 SynthSeg labels receive visually separable hues while
    homologous left/right pairs land on different entries of the cycle.
    """
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
    study: StudyAssets,
    slice_idx: int,
    label_colors: dict[int, np.ndarray],
    tumor_color: tuple[float, float, float, float],
    fill_alpha: float = 0.20,
    contour_alpha: float = 1.0,
) -> None:
    """Render a single montage cell into ``ax``."""
    ax.set_facecolor("black")
    t1n_slice = study.t1n[:, :, slice_idx]
    parc_slice = study.parc[:, :, slice_idx]

    lo, hi = np.percentile(t1n_slice, [1.0, 99.0])
    if hi <= lo:
        hi = lo + 1e-6
    ax.imshow(
        t1n_slice.T,
        cmap="gray",
        origin="lower",
        vmin=lo,
        vmax=hi,
        interpolation="nearest",
    )

    # Interior fills and per-label boundaries in the same hue: fills at
    # ``fill_alpha`` so the T1n anatomy stays visible, boundaries at
    # ``contour_alpha`` so region separations remain sharp.
    fill_overlay = np.zeros((*parc_slice.shape, 4), dtype=np.float32)
    contour_overlay = np.zeros((*parc_slice.shape, 4), dtype=np.float32)
    for lid, rgba in label_colors.items():
        mask = parc_slice == lid
        if not mask.any():
            continue
        boundary = binary_dilation(mask, iterations=1) & ~mask
        interior = mask & ~boundary
        fill_overlay[interior, :3] = rgba[:3]
        fill_overlay[interior, 3] = fill_alpha
        contour_overlay[boundary, :3] = rgba[:3]
        contour_overlay[boundary, 3] = contour_alpha
    ax.imshow(
        fill_overlay.transpose(1, 0, 2), origin="lower", interpolation="nearest"
    )
    ax.imshow(
        contour_overlay.transpose(1, 0, 2), origin="lower", interpolation="nearest"
    )

    # Optional tumor (whole-tumor mask): filled interior + sharp contour.
    if study.tumor is not None:
        tumor_slice = study.tumor[:, :, slice_idx]
        if tumor_slice.any():
            tumor_boundary = binary_dilation(tumor_slice, iterations=1) & ~tumor_slice
            tumor_interior = tumor_slice & ~tumor_boundary
            tumor_overlay = np.zeros((*tumor_slice.shape, 4), dtype=np.float32)
            tumor_overlay[tumor_interior, :3] = tumor_color[:3]
            tumor_overlay[tumor_interior, 3] = 0.45
            tumor_overlay[tumor_boundary, :3] = tumor_color[:3]
            tumor_overlay[tumor_boundary, 3] = 1.0
            ax.imshow(
                tumor_overlay.transpose(1, 0, 2),
                origin="lower",
                interpolation="nearest",
            )

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ("top", "right", "left", "bottom"):
        ax.spines[spine].set_visible(False)

    # QC annotation, lower-right, white-over-outline for legibility.
    txt = ax.text(
        0.97, 0.03,
        f"q = {study.qc_score:.3f}",
        transform=ax.transAxes,
        color="white",
        fontsize=9,
        fontweight="bold",
        ha="right", va="bottom",
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


def _mid_axial_slice(studies: list[StudyAssets]) -> int:
    """Default slice picker: mid-axial of the t0 volume."""
    return studies[0].t1n.shape[2] // 2 if studies else 0


def _max_tumor_slice(studies: list[StudyAssets]) -> int:
    """Slice picker: index of the t0 axial slice with the largest tumor area.

    Falls back to the mid-axial slice when no tumor is present.
    """
    if not studies or studies[0].tumor is None:
        return _mid_axial_slice(studies)
    counts = studies[0].tumor.sum(axis=(0, 1))
    if counts.max() == 0:
        return _mid_axial_slice(studies)
    return 101
    #return int(np.argmax(counts))


def make_figure(
    data_dir: Path,
    preprocessed_root: Path,
    synthseg_root: Path,
    best_override: str | None = None,
    worst_override: str | None = None,
    tumor_patient: str | None = "MenGrowth-0015",
    tumor_region: str = "TC",
) -> plt.Figure:
    """Build the qualitative montage figure (best / worst / tumor showcase).

    ``tumor_region`` controls which BraTS aggregated label is overlaid in
    the tumor-showcase row. Default ``TC`` (tumor core) excludes edema.
    """
    qc_csv = data_dir / "cohort" / "cohort_qc.csv"
    if not qc_csv.exists():
        raise FileNotFoundError(qc_csv)
    qc_df = pd.read_csv(qc_csv)

    if best_override and worst_override:
        best_pid, worst_pid = best_override, worst_override
    else:
        best_pid, worst_pid = select_showcase_patients(qc_df, min_studies=3)
        if best_override:
            best_pid = best_override
        if worst_override:
            worst_pid = worst_override
    logger.info(
        "Rendering best=%s worst=%s tumor=%s",
        best_pid, worst_pid, tumor_patient,
    )

    best_studies = _assemble_patient_studies(
        qc_df, best_pid, preprocessed_root, synthseg_root,
        tumor_region=tumor_region,
    )
    worst_studies = _assemble_patient_studies(
        qc_df, worst_pid, preprocessed_root, synthseg_root,
        tumor_region=tumor_region,
    )
    tumor_studies: list[StudyAssets] = []
    if tumor_patient:
        tumor_studies = _assemble_patient_studies(
            qc_df, tumor_patient, preprocessed_root, synthseg_root,
            tumor_region=tumor_region,
        )

    rows_spec = [
        ("Best-case", best_pid, best_studies, _mid_axial_slice),
        ("Worst-case", worst_pid, worst_studies, _mid_axial_slice),
    ]
    if tumor_studies:
        rows_spec.append(
            ("Tumor", tumor_patient, tumor_studies, _max_tumor_slice)
        )

    n_rows = len(rows_spec)
    n_cols = max(len(s) for _, _, s, _ in rows_spec)

    apply_pub_style()
    fig_w = max(7.0, 2.6 * n_cols + 1.3)
    fig_h = 2.55 * n_rows + 0.65
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(fig_w, fig_h),
        gridspec_kw={"wspace": 0.03, "hspace": 0.04,
                     "left": 0.08, "right": 0.985,
                     "bottom": 0.02, "top": 1 - 0.55 / fig_h},
        squeeze=False,
    )
    fig.patch.set_facecolor("black")

    tumor_color = (0.95, 0.15, 0.15, 0.9)

    # Single label→colour table across every patient so the same region
    # always lands on the same hue.
    all_parc = [
        np.unique(s.parc)
        for _, _, studies, _ in rows_spec
        for s in studies
    ]
    all_labels = np.unique(np.concatenate(all_parc))
    label_colors = _build_label_color_table(all_labels)

    # Column headers (t0, t1, …) — larger and white for black bg.
    for col in range(n_cols):
        axes[0, col].set_title(
            rf"$t_{{{col}}}$", fontsize=18, pad=6, color="white",
            fontweight="bold",
        )

    panel_letters = ["(A)", "(B)", "(C)", "(D)", "(E)"]

    for row_idx, (label, pid, studies, slice_picker) in enumerate(rows_spec):
        row_axes = axes[row_idx]
        slice_idx = slice_picker(studies)
        q_values = [s.qc_score for s in studies]
        mean_qc = float(np.mean(q_values)) if q_values else float("nan")
        std_qc = float(np.std(q_values, ddof=1)) if len(q_values) > 1 else float("nan")
        for col in range(n_cols):
            ax = row_axes[col]
            if col >= len(studies):
                _blank_cell(ax)
                continue
            _render_cell(
                ax, studies[col], slice_idx,
                label_colors=label_colors,
                tumor_color=tumor_color,
            )
        qc_summary = (
            rf"$\overline{{q}} = {mean_qc:.3f} \pm {std_qc:.3f}$"
            if not np.isnan(std_qc)
            else rf"$\overline{{q}} = {mean_qc:.3f}$"
        )
        # Tumor row gets a different annotation style — emphasise tumor.
        if label == "Tumor":
            row_label = (
                f"Tumor showcase ({tumor_region})\n{pid}\n{qc_summary}"
            )
        else:
            row_label = f"{label}\n{pid}\n{qc_summary}"
        left_ax = row_axes[0]
        left_ax.text(
            -0.07, 0.5,
            row_label,
            transform=left_ax.transAxes,
            fontsize=10, fontweight="bold",
            ha="right", va="center",
            color="white",
        )
        # Panel label: lift only the top row's letter further to clear the
        # column titles; subsequent rows sit just above their own row.
        y_panel = 1.12 if row_idx == 0 else 1.04
        row_axes[0].text(
            -0.28, y_panel, panel_letters[row_idx],
            transform=row_axes[0].transAxes,
            fontsize=13, fontweight="bold", ha="left", va="bottom",
            color="white",
        )

    return fig


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("outputs/synthseg_analysis"),
    )
    parser.add_argument(
        "--preprocessed-root",
        type=Path,
        default=Path(
            "/media/mpascual/PortableSSD/Meningiomas/MenGrowth/v5_final/MenGrowth-2025"
        ),
    )
    parser.add_argument(
        "--synthseg-root",
        type=Path,
        default=Path(
            "/media/mpascual/PortableSSD/Meningiomas/MenGrowth/v5_final/synthseg"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/synthseg_analysis/figures/figure2_qualitative.pdf"),
    )
    parser.add_argument("--also-png", action="store_true")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--best-patient",
        type=str,
        default=None,
        help="Override automatic best-case patient selection.",
    )
    parser.add_argument(
        "--worst-patient",
        type=str,
        default=None,
        help="Override automatic worst-case patient selection.",
    )
    parser.add_argument(
        "--tumor-patient",
        type=str,
        default="MenGrowth-0015",
        help="Patient ID for the tumor showcase row (pass empty string to disable).",
    )
    parser.add_argument(
        "--tumor-region",
        type=str,
        choices=("WT", "TC", "ET"),
        default="TC",
        help="BraTS aggregated label to overlay: WT (whole tumor incl. edema), "
             "TC (tumor core, no edema; default), or ET (enhancing only).",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    tumor_patient = args.tumor_patient.strip() or None if args.tumor_patient else None
    fig = make_figure(
        args.data_dir,
        args.preprocessed_root,
        args.synthseg_root,
        best_override=args.best_patient,
        worst_override=args.worst_patient,
        tumor_patient=tumor_patient,
        tumor_region=args.tumor_region,
    )
    out_pdf = save_figure(fig, args.output, dpi=args.dpi)
    logger.info("Saved %s", out_pdf)
    if args.also_png:
        png = args.output.with_suffix(".png")
        fig2 = make_figure(
            args.data_dir,
            args.preprocessed_root,
            args.synthseg_root,
            best_override=args.best_patient,
            worst_override=args.worst_patient,
            tumor_patient=tumor_patient,
            tumor_region=args.tumor_region,
        )
        save_figure(fig2, png, dpi=args.dpi)
        logger.info("Saved %s", png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
