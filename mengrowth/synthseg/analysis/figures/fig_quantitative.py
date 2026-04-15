"""Figure 1 — Quantitative preprocessing quality assessment.

Two-panel publication figure:

* **Panel A** — Joint plot of SynthSeg QC score versus original T1n
  maximum voxel spacing (central scatter + marginal histograms), with
  the linear mixed-effects model regression line and 95 % confidence
  band superimposed.
* **Panel B** — 17 × 58 heatmap of within-subject volume CV, bilateral
  regions grouped by anatomy, patients ordered by mean QC.

Run standalone::

    python -m mengrowth.synthseg.analysis.figures.fig_quantitative \\
        --data-dir outputs/synthseg_analysis \\
        --output outputs/synthseg_analysis/figures/figure1_quantitative.pdf
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.ticker import MultipleLocator

from mengrowth.synthseg.analysis.figures.common import (
    apply_pub_style,
    bilateral_merge_cv,
    build_heatmap_layout,
    mean_qc_per_patient,
    paul_tol_color,
    save_figure,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _resolve_paths(data_dir: Path) -> tuple[Path, Path, Path]:
    qc_spacing = data_dir / "qc_analysis" / "qc_vs_spacing_merged.csv"
    cv_csv = data_dir / "longitudinal" / "within_subject_cv.csv"
    cohort_qc = data_dir / "cohort" / "cohort_qc.csv"
    for p in (qc_spacing, cv_csv, cohort_qc):
        if not p.exists():
            raise FileNotFoundError(f"Required CSV missing: {p}")
    return qc_spacing, cv_csv, cohort_qc


def _fit_lmm(
    df: pd.DataFrame,
) -> tuple[float, float, float, tuple[float, float], np.ndarray]:
    """Refit the mixed-effects model and return key quantities.

    Returns
    -------
    intercept, slope, slope_p_value, slope_ci, cov_fixed
    """
    import statsmodels.formula.api as smf

    model = smf.mixedlm("qc_score ~ max_spacing", df, groups=df["patient_id"])
    result = model.fit(reml=True, method="lbfgs")
    intercept = float(result.fe_params["Intercept"])
    slope = float(result.fe_params["max_spacing"])
    p_slope = float(result.pvalues["max_spacing"])
    ci = result.conf_int().loc["max_spacing"].tolist()
    cov = np.asarray(result.cov_params().loc[
        ["Intercept", "max_spacing"], ["Intercept", "max_spacing"]
    ])
    return intercept, slope, p_slope, (float(ci[0]), float(ci[1])), cov


# ---------------------------------------------------------------------------
# Panel A — joint plot
# ---------------------------------------------------------------------------


def _draw_panel_a(
    fig: plt.Figure,
    subspec,
    merged: pd.DataFrame,
) -> None:
    """Draw the joint plot (scatter + marginals) into ``subspec``."""
    nested = GridSpecFromSubplotSpec(
        2,
        2,
        subplot_spec=subspec,
        width_ratios=[4, 1],
        height_ratios=[1, 4],
        wspace=0.04,
        hspace=0.04,
    )
    ax_scatter = fig.add_subplot(nested[1, 0])
    ax_top = fig.add_subplot(nested[0, 0], sharex=ax_scatter)
    ax_right = fig.add_subplot(nested[1, 1], sharey=ax_scatter)

    blue = "#4878CF"
    red = "#E24A33"

    # Scatter
    ax_scatter.scatter(
        merged["max_spacing"],
        merged["qc_score"],
        s=28,
        alpha=0.65,
        color=blue,
        edgecolors="white",
        linewidths=0.3,
        zorder=3,
    )

    # LMM regression line + 95 % CI band
    intercept, slope, p_slope, slope_ci, cov_fixed = _fit_lmm(merged)
    logger.info("LMM: intercept=%.4f slope=%.4f p=%.2e CI=%s",
                intercept, slope, p_slope, slope_ci)
    x_grid = np.linspace(
        float(merged["max_spacing"].min()),
        float(merged["max_spacing"].max()),
        200,
    )
    y_hat = intercept + slope * x_grid
    # Variance of the predicted mean: Var(b0 + b1 x) = Var(b0) + 2x Cov + x^2 Var(b1)
    var_pred = (
        cov_fixed[0, 0]
        + 2.0 * x_grid * cov_fixed[0, 1]
        + (x_grid**2) * cov_fixed[1, 1]
    )
    se_pred = np.sqrt(np.clip(var_pred, 0.0, None))
    y_lo = y_hat - 1.96 * se_pred
    y_hi = y_hat + 1.96 * se_pred

    ax_scatter.fill_between(
        x_grid, y_lo, y_hi, color=red, alpha=0.15, linewidth=0, zorder=2
    )
    ax_scatter.plot(
        x_grid,
        y_hat,
        color=red,
        linewidth=1.5,
        label="LMM fit",
        zorder=4,
    )

    # Outlier annotation (MenGrowth-0007, lowest QC)
    outlier = merged.loc[merged["qc_score"].idxmin()]
    ox, oy = float(outlier["max_spacing"]), float(outlier["qc_score"])
    label_xy = (ox - 0.6, oy + 0.07)
    ax_scatter.annotate(
        f"{outlier['patient_id']}\n(large meningioma)",
        xy=(ox, oy),
        xytext=label_xy,
        fontsize=7.5,
        ha="right",
        color="0.15",
        arrowprops=dict(
            arrowstyle="-",
            color="0.35",
            linewidth=0.7,
            shrinkA=2,
            shrinkB=3,
        ),
    )

    # Model annotation: full fitted equation + p-value.
    p_label = "p < 0.001" if p_slope < 1e-3 else f"p = {p_slope:.3f}"
    sign = "-" if slope < 0 else "+"
    annotation = (
        rf"$\hat{{q}} = {intercept:.3f} {sign} {abs(slope):.3f}\,s$"
        "\n"
        rf"$({p_label})$"
    )
    ax_scatter.text(
        0.02,
        0.04,
        annotation,
        transform=ax_scatter.transAxes,
        fontsize=9,
        ha="left",
        va="bottom",
        bbox=dict(
            facecolor="white", alpha=0.9, edgecolor="0.6", linewidth=0.6,
            boxstyle="round,pad=0.35",
        ),
    )

    ax_scatter.set_xlabel("Original T1n max voxel spacing (mm)")
    ax_scatter.set_ylabel("SynthSeg QC score")
    ax_scatter.set_ylim(0.5, 0.9)
    ax_scatter.xaxis.set_major_locator(MultipleLocator(1.0))
    # Interior y-ticks only: drop the 0.50 and 0.90 endpoints.
    ax_scatter.set_yticks(np.arange(0.55, 0.90, 0.05))
    for spine in ("top", "right", "left", "bottom"):
        ax_scatter.spines[spine].set_visible(True)
        ax_scatter.spines[spine].set_linewidth(0.7)
    ax_scatter.grid(True, linestyle=":", linewidth=0.6, color="0.55", alpha=0.5)
    ax_scatter.set_axisbelow(True)

    # Marginal histograms — full frame + grid, labels on all 4 sides hidden
    # where they would duplicate the scatter's axes.
    ax_top.hist(
        merged["max_spacing"], bins=25, color=blue, alpha=0.85, edgecolor="white",
        linewidth=0.4,
    )
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_top.tick_params(axis="y", labelsize=7)
    ax_top.set_ylabel("count", fontsize=8)
    for spine in ("top", "right", "left", "bottom"):
        ax_top.spines[spine].set_visible(True)
        ax_top.spines[spine].set_linewidth(0.7)
    ax_top.grid(True, linestyle=":", linewidth=0.6, color="0.55", alpha=0.5)
    ax_top.set_axisbelow(True)

    ax_right.hist(
        merged["qc_score"], bins=25, color=blue, alpha=0.85,
        edgecolor="white", linewidth=0.4, orientation="horizontal",
    )
    ax_right.tick_params(axis="y", labelleft=False)
    ax_right.tick_params(axis="x", labelsize=7)
    ax_right.set_xlabel("count", fontsize=8)
    for spine in ("top", "right", "left", "bottom"):
        ax_right.spines[spine].set_visible(True)
        ax_right.spines[spine].set_linewidth(0.7)
    ax_right.grid(True, linestyle=":", linewidth=0.6, color="0.55", alpha=0.5)
    ax_right.set_axisbelow(True)

    # Panel label (A) — figure-coord, lifted well above the top marginal.
    pos = subspec.get_position(fig)
    fig.text(
        pos.x0 - 0.010, pos.y1 + 0.055,
        "(A)", fontsize=13, fontweight="bold", ha="left", va="top",
    )


# ---------------------------------------------------------------------------
# Panel B — heatmap
# ---------------------------------------------------------------------------


def _draw_panel_b(
    fig: plt.Figure,
    subspec,
    cv_df: pd.DataFrame,
    qc_df: pd.DataFrame,
) -> None:
    """Draw the region × patient CV heatmap into ``subspec``."""
    layout = build_heatmap_layout()
    merged = bilateral_merge_cv(cv_df)
    patient_order = (
        mean_qc_per_patient(qc_df).sort_values(ascending=True).index.tolist()
    )
    # Restrict to patients that actually have CV data (n_timepoints >= 2).
    patient_order = [p for p in patient_order if p in set(merged["patient_id"])]

    # Build the 17 × N matrix.
    pivot = merged.pivot_table(
        index="region_base", columns="patient_id", values="cv", aggfunc="mean",
    )
    matrix = pivot.reindex(
        index=layout.region_bases_ordered, columns=patient_order
    ).to_numpy(dtype=float)

    # Split heatmap + colorbar inside this subspec.
    nested = GridSpecFromSubplotSpec(
        1, 2, subplot_spec=subspec, width_ratios=[40, 1.2], wspace=0.04,
    )
    ax_hm = fig.add_subplot(nested[0, 0])
    ax_cb = fig.add_subplot(nested[0, 1])

    vmin, vmax = 0.0, 0.30
    # Use 'rocket_r' (sequential, perceptually uniform); low CV = pale.
    import seaborn as sns  # local import keeps seaborn optional

    cmap = sns.color_palette("rocket_r", as_cmap=True)
    im = ax_hm.imshow(
        matrix,
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        origin="upper",
    )

    # Thin white cell separators: overlay via a grid of minor ticks.
    ax_hm.set_xticks(np.arange(matrix.shape[1] + 1) - 0.5, minor=True)
    ax_hm.set_yticks(np.arange(matrix.shape[0] + 1) - 0.5, minor=True)
    ax_hm.grid(which="minor", color="white", linewidth=0.25)
    ax_hm.tick_params(which="minor", bottom=False, left=False)

    # Y-axis: region labels
    ax_hm.set_yticks(np.arange(matrix.shape[0]))
    ax_hm.set_yticklabels(layout.display_labels, fontsize=7.5)
    ax_hm.set_ylabel("")

    # X-axis: patient ticks every 10
    n_patients = matrix.shape[1]
    tick_step = max(1, n_patients // 6)
    xticks = list(range(0, n_patients, tick_step))
    if xticks[-1] != n_patients - 1:
        xticks.append(n_patients - 1)
    ax_hm.set_xticks(xticks)
    ax_hm.set_xticklabels([str(i + 1) for i in xticks], fontsize=7)
    ax_hm.set_xlabel("Patient (ordered by ascending mean QC)")

    # Group separators + left-side group labels
    for i, (label, start) in enumerate(
        zip(layout.group_labels, layout.group_row_starts)
    ):
        if i > 0:
            ax_hm.axhline(start - 0.5, color="white", linewidth=1.6, zorder=4)
            ax_hm.axhline(start - 0.5, color="0.25", linewidth=0.7, zorder=5)
        end = (
            layout.group_row_starts[i + 1] - 1
            if i + 1 < len(layout.group_row_starts)
            else matrix.shape[0] - 1
        )
        mid = (start + end) / 2.0
        # Group label sits well to the left of the y-tick labels.
        ax_hm.text(
            -0.27, mid,
            label,
            transform=ax_hm.get_yaxis_transform(),
            ha="center", va="center", rotation=90,
            fontsize=8.5, fontweight="bold", color="0.15",
            clip_on=False,
        )
        # Short connector tick marking the group span.
        ax_hm.plot(
            [-0.235, -0.235],
            [start - 0.4, end + 0.4],
            transform=ax_hm.get_yaxis_transform(),
            color="0.4", linewidth=0.9, clip_on=False,
        )

    for spine in ("top", "right", "left", "bottom"):
        ax_hm.spines[spine].set_linewidth(0.6)

    # Colorbar
    cbar = fig.colorbar(im, cax=ax_cb, extend="max")
    cbar.set_label("Within-subject volume CV", fontsize=8.5)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_ticks([0.0, 0.05, 0.10, 0.20, 0.30])
    cbar.set_ticklabels(["0", "0.05", "0.10", "0.20", r"$\geq$0.30"])

    # Panel label (B) — anchored in figure coords, lifted above the heatmap.
    pos_b = subspec.get_position(fig)
    fig.text(
        pos_b.x0 - 0.004, pos_b.y1 + 0.055,
        "(B)", fontsize=13, fontweight="bold", ha="left", va="top",
    )


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


def make_figure(data_dir: Path) -> plt.Figure:
    """Assemble the two-panel figure from Phase-2 analysis outputs."""
    qc_spacing_csv, cv_csv, cohort_qc_csv = _resolve_paths(data_dir)
    merged = pd.read_csv(qc_spacing_csv)
    cv_df = pd.read_csv(cv_csv)
    qc_df = pd.read_csv(cohort_qc_csv)

    apply_pub_style()

    fig = plt.figure(figsize=(14.5, 6.0))
    top = GridSpec(
        1, 2, figure=fig,
        width_ratios=[1.0, 1.55],
        wspace=0.45,
        left=0.055, right=0.985, bottom=0.13, top=0.88,
    )
    _draw_panel_a(fig, top[0, 0], merged)
    _draw_panel_b(fig, top[0, 1], cv_df, qc_df)
    return fig


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("outputs/synthseg_analysis"),
        help="Root of Phase-2 analysis outputs (cohort/, longitudinal/, qc_analysis/).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/synthseg_analysis/figures/figure1_quantitative.pdf"),
        help="Path of the main (PDF) output figure.",
    )
    parser.add_argument(
        "--also-png",
        action="store_true",
        help="Additionally save a PNG at the same stem.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    fig = make_figure(args.data_dir)
    out_pdf = save_figure(fig, args.output, dpi=args.dpi)
    logger.info("Saved %s", out_pdf)
    if args.also_png:
        png = args.output.with_suffix(".png")
        fig2 = make_figure(args.data_dir)
        save_figure(fig2, png, dpi=args.dpi)
        logger.info("Saved %s", png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
