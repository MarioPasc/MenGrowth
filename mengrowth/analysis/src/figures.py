"""Phase 2: Publication-quality quantitative figures for MenGrowth analysis."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import List

import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from mengrowth.preprocessing.utils.settings import (
    PAUL_TOL_BRIGHT,
    PLOT_SETTINGS,
    apply_ieee_style,
    get_figure_size,
)

from .types import (
    LABEL_COLORS_HEX,
    LABEL_IDS,
    LABEL_NAMES,
    MIDLINE_BAND,
    MIDLINE_X,
    DatasetMetrics,
    PatientMetrics,
)

logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _setup_style() -> None:
    """Apply IEEE publication style and override save defaults."""
    apply_ieee_style()
    plt.rcParams.update(
        {
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )


def _save(fig: plt.Figure, path: Path) -> None:
    """Save figure, create parent dirs, close."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    logger.info("Saved: %s", path)


def _sorted_patients(metrics: DatasetMetrics) -> List[PatientMetrics]:
    """Return patients sorted by ID."""
    return [pm for _, pm in sorted(metrics.patients.items())]


def _max_timepoints(metrics: DatasetMetrics) -> int:
    """Maximum number of studies across all patients."""
    return max(pm.n_studies for pm in metrics.patients.values())


def _patient_cmap(n: int) -> np.ndarray:
    """Generate *n* distinguishable colors for patient lines."""
    return plt.cm.tab20(np.linspace(0, 1, max(n, 1)))


def _tp_labels(n: int) -> List[str]:
    """Generate LaTeX timepoint labels $t_0$ ... $t_{n-1}$."""
    return [f"$t_{{{i}}}$" for i in range(n)]


# ═════════════════════════════════════════════════════════════════════════════
# Individual figure functions
# ═════════════════════════════════════════════════════════════════════════════


def fig01_volume_trajectories(m: DatasetMetrics, out: Path) -> None:
    """Spaghetti plot of total tumor volume across timepoints."""
    _setup_style()
    fig, ax = plt.subplots(figsize=get_figure_size("double"))

    max_tp = _max_timepoints(m)
    all_vols: List[List[float]] = [[] for _ in range(max_tp)]
    colors = _patient_cmap(m.n_patients)

    for i, pm in enumerate(_sorted_patients(m)):
        c = colors[i]
        xs = [s.timepoint_index for s in pm.studies]
        ys = [s.total_volume for s in pm.studies]
        ax.plot(xs, ys, "-", color=c, alpha=0.35, linewidth=0.8)
        for s in pm.studies:
            if s.is_empty:
                ax.plot(
                    s.timepoint_index,
                    0,
                    "o",
                    color=c,
                    alpha=0.6,
                    markersize=4,
                    markerfacecolor="none",
                    markeredgewidth=1,
                )
            else:
                ax.plot(
                    s.timepoint_index,
                    s.total_volume,
                    "o",
                    color=c,
                    alpha=0.6,
                    markersize=3,
                )
                all_vols[s.timepoint_index].append(s.total_volume)

    # Population mean ± std
    mx, my, sy = [], [], []
    for tp in range(max_tp):
        if all_vols[tp]:
            mx.append(tp)
            my.append(np.mean(all_vols[tp]))
            sy.append(np.std(all_vols[tp]))
    my_a, sy_a = np.array(my), np.array(sy)
    ax.plot(mx, my_a, "k-", linewidth=2, label="Population mean", zorder=5)
    ax.fill_between(mx, my_a - sy_a, my_a + sy_a, alpha=0.15, color="black", zorder=4)

    # Cosmetics
    ax.set_xlabel(r"Timepoint ($t_i$)")
    ax.set_ylabel(r"Total tumor volume (mm$^3$)")
    ax.set_xticks(range(max_tp))
    ax.set_xticklabels(_tp_labels(max_tp))
    ax.plot(
        [],
        [],
        "o",
        color="gray",
        markerfacecolor="none",
        markeredgewidth=1,
        markersize=5,
        label="Empty segmentation",
    )
    ax.legend(loc="upper left")
    _save(fig, out)


def fig02_volume_trajectories_per_label(m: DatasetMetrics, out: Path) -> None:
    """Three-panel spaghetti plots for NET, SNFH, ET volumes."""
    _setup_style()
    fig, axes = plt.subplots(
        1, 3, figsize=(get_figure_size("double")[0], 3.5), sharey=False
    )

    max_tp = _max_timepoints(m)
    colors = _patient_cmap(m.n_patients)

    for ax, label in zip(axes, LABEL_IDS):
        all_vols: List[List[float]] = [[] for _ in range(max_tp)]
        for i, pm in enumerate(_sorted_patients(m)):
            c = colors[i]
            xs = [s.timepoint_index for s in pm.studies]
            ys = [s.volumes.get(label, 0) for s in pm.studies]
            ax.plot(xs, ys, "-", color=c, alpha=0.35, linewidth=0.8)
            for s in pm.studies:
                if not s.is_empty:
                    ax.plot(
                        s.timepoint_index,
                        s.volumes.get(label, 0),
                        "o",
                        color=c,
                        alpha=0.6,
                        markersize=2.5,
                    )
                    all_vols[s.timepoint_index].append(s.volumes.get(label, 0))

        # Mean
        mx, my, sy = [], [], []
        for tp in range(max_tp):
            if all_vols[tp]:
                mx.append(tp)
                my.append(np.mean(all_vols[tp]))
                sy.append(np.std(all_vols[tp]))
        if mx:
            my_a, sy_a = np.array(my), np.array(sy)
            ax.plot(mx, my_a, "-", color=LABEL_COLORS_HEX[label], linewidth=2, zorder=5)
            ax.fill_between(
                mx,
                my_a - sy_a,
                my_a + sy_a,
                alpha=0.2,
                color=LABEL_COLORS_HEX[label],
                zorder=4,
            )

        ax.set_title(LABEL_NAMES[label], fontsize=PLOT_SETTINGS["axes_titlesize"])
        ax.set_xlabel(r"Timepoint ($t_i$)")
        ax.set_xticks(range(max_tp))
        ax.set_xticklabels(_tp_labels(max_tp))
        if label == 1:
            ax.set_ylabel(r"Volume (mm$^3$)")

    fig.tight_layout()
    _save(fig, out)


def fig03_volume_distributions(m: DatasetMetrics, out: Path) -> None:
    """Violin plot of total volume + stacked bar of per-label mean contribution."""
    _setup_style()
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=get_figure_size("double"), gridspec_kw={"width_ratios": [2, 1]}
    )

    # Collect non-empty volumes
    all_vols = [
        s.total_volume
        for pm in m.patients.values()
        for s in pm.studies
        if not s.is_empty
    ]

    # Left: violin
    vp = ax1.violinplot([all_vols], positions=[0], showmeans=True, showmedians=True)
    for body in vp["bodies"]:
        body.set_facecolor(PAUL_TOL_BRIGHT["blue"])
        body.set_alpha(0.6)
    ax1.set_xticks([0])
    ax1.set_xticklabels(["All non-empty studies"])
    ax1.set_ylabel(r"Total volume (mm$^3$)")
    ax1.set_title("Volume distribution")

    # Right: stacked bar of mean label contribution
    label_means = {}
    for label in LABEL_IDS:
        vols = [
            s.volumes[label]
            for pm in m.patients.values()
            for s in pm.studies
            if not s.is_empty
        ]
        label_means[label] = np.mean(vols) if vols else 0.0
    bottom = 0.0
    for label in LABEL_IDS:
        ax2.bar(
            0,
            label_means[label],
            bottom=bottom,
            color=LABEL_COLORS_HEX[label],
            label=LABEL_NAMES[label],
            width=0.5,
        )
        bottom += label_means[label]
    ax2.set_xticks([0])
    ax2.set_xticklabels(["Mean composition"])
    ax2.set_ylabel(r"Volume (mm$^3$)")
    ax2.set_title("Label contributions")
    ax2.legend(fontsize=PLOT_SETTINGS["legend_fontsize"])

    fig.tight_layout()
    _save(fig, out)


def fig04_tumor_heatmap(m: DatasetMetrics, out: Path) -> None:
    """Three-view MIP of tumor frequency overlaid on mean brain template."""
    _setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(get_figure_size("double")[0], 3.5))
    fig.patch.set_facecolor("black")

    hm = m.tumor_heatmap
    brain = m.mean_brain
    hm_max = float(hm.max()) if hm.max() > 0 else 1.0

    view_specs = [
        ("Axial", 2),  # MIP along I→S axis
        ("Coronal", 1),  # MIP along P→A axis
        ("Sagittal", 0),  # MIP along L→R axis
    ]

    for ax, (title, proj_axis) in zip(axes, view_specs):
        ax.set_facecolor("black")

        # Brain underlay MIP
        brain_mip = np.max(brain, axis=proj_axis).T
        brain_norm = np.zeros_like(brain_mip)
        nz = brain_mip[brain_mip > 0]
        if nz.size > 0:
            brain_norm = brain_mip / np.percentile(nz, 99)
            brain_norm = np.clip(brain_norm, 0, 1)

        ax.imshow(
            brain_norm, cmap="gray", vmin=0, vmax=1, origin="lower", aspect="equal"
        )

        # Tumor frequency overlay
        tumor_mip = np.max(hm, axis=proj_axis).T
        norm = mcolors.Normalize(vmin=1, vmax=hm_max)
        cmap = plt.cm.viridis
        overlay_rgba = cmap(norm(tumor_mip))
        overlay_rgba[..., 3] = np.where(tumor_mip > 0, 0.75, 0.0)
        ax.imshow(overlay_rgba, origin="lower", aspect="equal")

        ax.set_title(title, color="white", fontsize=PLOT_SETTINGS["axes_titlesize"])
        ax.axis("off")

    # Colorbar
    sm = plt.cm.ScalarMappable(
        cmap="viridis", norm=mcolors.Normalize(vmin=1, vmax=hm_max)
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.03, pad=0.04, shrink=0.8)
    cbar.set_label(
        "No. studies with tumor",
        color="white",
        fontsize=PLOT_SETTINGS["axes_labelsize"],
    )
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    _save(fig, out)


def fig05_dice_consistency(m: DatasetMetrics, out: Path) -> None:
    """Heatmap of Dice coefficients (patients × consecutive pairs)."""
    _setup_style()
    patients = _sorted_patients(m)
    max_pairs = max((pm.n_studies - 1 for pm in patients), default=0)
    if max_pairs == 0:
        logger.warning("No Dice pairs to plot")
        return

    # Build matrix (patients × pairs), NaN for missing
    mat = np.full((len(patients), max_pairs), np.nan)
    pids = []
    for r, pm in enumerate(patients):
        pids.append(pm.patient_id.replace("MenGrowth-", ""))
        for dp in pm.dice_pairs:
            mat[r, dp.pair_index] = dp.dice_total

    fig, ax = plt.subplots(
        figsize=(max(3.5, max_pairs * 0.9), max(4, len(patients) * 0.35))
    )

    # Masked array for gray background on NaN
    masked = np.ma.masked_invalid(mat)
    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad(color="0.85")

    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    # Annotate cells
    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            val = mat[r, c]
            if not np.isnan(val):
                color = "white" if val < 0.4 else "black"
                ax.text(
                    c,
                    r,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=color,
                )

    ax.set_yticks(range(len(pids)))
    ax.set_yticklabels(pids, fontsize=7)
    ax.set_xticks(range(max_pairs))
    ax.set_xticklabels(
        [f"$t_{{{i}}}$–$t_{{{i + 1}}}$" for i in range(max_pairs)], fontsize=8
    )
    ax.set_xlabel("Consecutive timepoint pair")
    ax.set_ylabel("Patient")
    ax.set_title("Inter-timepoint Dice coefficient")

    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04, label="Dice")
    fig.tight_layout()
    _save(fig, out)


def fig06_growth_rates(m: DatasetMetrics, out: Path) -> None:
    """Violin plots of absolute and relative volume change per step."""
    _setup_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(get_figure_size("double")[0], 5))

    abs_vals = [
        v for pm in m.patients.values() for v in pm.growth_absolute if not math.isnan(v)
    ]
    rel_vals = [
        v for pm in m.patients.values() for v in pm.growth_relative if not math.isnan(v)
    ]

    # Absolute
    if abs_vals:
        pos_abs = [v for v in abs_vals if v >= 0]
        neg_abs = [v for v in abs_vals if v < 0]
        data_abs, labels_abs, colors_abs = [], [], []
        if pos_abs:
            data_abs.append(pos_abs)
            labels_abs.append(f"Increase\n(n={len(pos_abs)})")
            colors_abs.append(PAUL_TOL_BRIGHT["red"])
        if neg_abs:
            data_abs.append(neg_abs)
            labels_abs.append(f"Decrease\n(n={len(neg_abs)})")
            colors_abs.append(PAUL_TOL_BRIGHT["blue"])

        if data_abs:
            vp = ax1.violinplot(
                data_abs,
                positions=range(len(data_abs)),
                showmeans=True,
                showmedians=True,
            )
            for body, c in zip(vp["bodies"], colors_abs):
                body.set_facecolor(c)
                body.set_alpha(0.6)
            ax1.set_xticks(range(len(labels_abs)))
            ax1.set_xticklabels(labels_abs)
    ax1.axhline(0, color="0.5", linewidth=0.5, linestyle="--")
    ax1.set_ylabel(r"$\Delta V$ (mm$^3$/step)")
    ax1.set_title("Absolute volume change")

    # Relative
    if rel_vals:
        pos_rel = [v for v in rel_vals if v >= 0]
        neg_rel = [v for v in rel_vals if v < 0]
        data_rel, labels_rel, colors_rel = [], [], []
        if pos_rel:
            data_rel.append(pos_rel)
            labels_rel.append(f"Increase\n(n={len(pos_rel)})")
            colors_rel.append(PAUL_TOL_BRIGHT["red"])
        if neg_rel:
            data_rel.append(neg_rel)
            labels_rel.append(f"Decrease\n(n={len(neg_rel)})")
            colors_rel.append(PAUL_TOL_BRIGHT["blue"])

        if data_rel:
            vp = ax2.violinplot(
                data_rel,
                positions=range(len(data_rel)),
                showmeans=True,
                showmedians=True,
            )
            for body, c in zip(vp["bodies"], colors_rel):
                body.set_facecolor(c)
                body.set_alpha(0.6)
            ax2.set_xticks(range(len(labels_rel)))
            ax2.set_xticklabels(labels_rel)
    ax2.axhline(0, color="0.5", linewidth=0.5, linestyle="--")
    ax2.set_ylabel(r"$\Delta V / V$ (%/step)")
    ax2.set_title("Relative volume change")

    fig.tight_layout()
    _save(fig, out)


def fig07_label_composition(m: DatasetMetrics, out: Path) -> None:
    """Small-multiples stacked area (per patient) + population donut chart."""
    _setup_style()
    patients = _sorted_patients(m)
    n = len(patients)
    ncols = 6
    nrows = math.ceil(n / ncols)

    fig = plt.figure(figsize=(get_figure_size("double")[0] + 2, nrows * 1.4 + 1.5))
    gs = gridspec.GridSpec(
        nrows + 1,
        ncols + 1,
        figure=fig,
        width_ratios=[1] * ncols + [1.5],
        height_ratios=[1] * nrows + [0.3],
    )

    label_colors = [LABEL_COLORS_HEX[l] for l in LABEL_IDS]

    for idx, pm in enumerate(patients):
        row, col = divmod(idx, ncols)
        ax = fig.add_subplot(gs[row, col])

        x = [s.timepoint_index for s in pm.studies]
        stacks = [
            [s.label_proportions.get(l, 0) for s in pm.studies] for l in LABEL_IDS
        ]

        if len(x) >= 2:
            ax.stackplot(x, *stacks, colors=label_colors, alpha=0.85)
        elif len(x) == 1:
            bottom = 0.0
            for li, label in enumerate(LABEL_IDS):
                val = stacks[li][0]
                ax.bar(0, val, bottom=bottom, color=label_colors[li], width=0.5)
                bottom += val

        ax.set_xlim(-0.3, max(x) + 0.3 if x else 0.5)
        ax.set_ylim(0, 1)
        ax.set_title(pm.patient_id.replace("MenGrowth-", ""), fontsize=7, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])

    # Population donut (spanning right column)
    ax_donut = fig.add_subplot(gs[:nrows, ncols])
    pop_means = []
    for label in LABEL_IDS:
        vals = [
            s.volumes[label]
            for pm in m.patients.values()
            for s in pm.studies
            if not s.is_empty
        ]
        pop_means.append(np.mean(vals) if vals else 0.0)
    total = sum(pop_means)
    fracs = [v / total * 100 if total > 0 else 0 for v in pop_means]
    wedges, texts, autotexts = ax_donut.pie(
        fracs,
        labels=[LABEL_NAMES[l] for l in LABEL_IDS],
        colors=label_colors,
        autopct="%1.1f%%",
        pctdistance=0.75,
        startangle=90,
        textprops={"fontsize": 8},
    )
    centre_circle = plt.Circle((0, 0), 0.50, fc="white")
    ax_donut.add_artist(centre_circle)
    ax_donut.set_title("Population", fontsize=9)

    # Legend row
    ax_leg = fig.add_subplot(gs[nrows, :])
    ax_leg.axis("off")
    handles = [
        mpatches.Patch(color=LABEL_COLORS_HEX[l], label=LABEL_NAMES[l])
        for l in LABEL_IDS
    ]
    ax_leg.legend(handles=handles, loc="center", ncol=3, frameon=False, fontsize=9)

    fig.tight_layout()
    _save(fig, out)


def fig08_spatial_analysis(m: DatasetMetrics, out: Path) -> None:
    """Laterality bar chart + centroid scatter on axial projection."""
    _setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=get_figure_size("double"))

    centroids_x, centroids_y, sizes = [], [], []
    left, right, midline = 0, 0, 0

    for pm in m.patients.values():
        for s in pm.studies:
            if s.centroid is None:
                continue
            cx, cy, cz = s.centroid
            centroids_x.append(cx)
            centroids_y.append(cy)
            sizes.append(s.total_volume)

            if cx < MIDLINE_X - MIDLINE_BAND:
                left += 1
            elif cx > MIDLINE_X + MIDLINE_BAND:
                right += 1
            else:
                midline += 1

    # Left: laterality bar chart
    cats = ["Left", "Midline", "Right"]
    counts = [left, midline, right]
    bar_colors = [
        PAUL_TOL_BRIGHT["blue"],
        PAUL_TOL_BRIGHT["grey"],
        PAUL_TOL_BRIGHT["red"],
    ]
    bars = ax1.bar(cats, counts, color=bar_colors, width=0.6)
    for bar, cnt in zip(bars, counts):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            str(cnt),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax1.set_ylabel("No. studies")
    ax1.set_title("Tumor laterality")

    # Right: centroid scatter (axial projection)
    if centroids_x:
        sz = np.array(sizes)
        sz_norm = 10 + 200 * (sz - sz.min()) / (sz.max() - sz.min() + 1e-9)
        scatter = ax2.scatter(
            centroids_x,
            centroids_y,
            s=sz_norm,
            c=PAUL_TOL_BRIGHT["purple"],
            alpha=0.5,
            edgecolors="0.3",
            linewidths=0.3,
        )
        ax2.axvline(
            MIDLINE_X, color="0.5", linestyle="--", linewidth=0.5, label="Midline"
        )
        ax2.set_xlabel("L ← X (voxel) → R")
        ax2.set_ylabel("P ← Y (voxel) → A")
        ax2.set_title("Tumor centroid locations (axial)")
        ax2.legend(fontsize=PLOT_SETTINGS["legend_fontsize"])

    fig.tight_layout()
    _save(fig, out)


def fig09_empty_segmentation_analysis(m: DatasetMetrics, out: Path) -> None:
    """Table-figure summarising the empty segmentation studies."""
    _setup_style()
    n_empty = len(m.empty_studies)
    fig_height = max(1.5, 0.3 * n_empty + 1.2)
    fig, ax = plt.subplots(figsize=(get_figure_size("double")[0], fig_height))
    ax.axis("off")

    if not m.empty_studies:
        ax.text(
            0.5, 0.5, "No empty segmentations", ha="center", va="center", fontsize=12
        )
        _save(fig, out)
        return

    headers = ["Patient", "Study", "Timepoint", "Other non-empty?"]
    rows = []
    for pid, sid, tp in m.empty_studies:
        pm = m.patients[pid]
        has_non_empty = any(not s.is_empty for s in pm.studies)
        rows.append([pid, sid, f"t{tp}", "Yes" if has_non_empty else "No"])

    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)

    # Highlight fully-empty patients
    for r, (pid, sid, tp) in enumerate(m.empty_studies, start=1):
        pm = m.patients[pid]
        if all(s.is_empty for s in pm.studies):
            for c in range(len(headers)):
                table[r, c].set_facecolor("#FFCCCC")

    ax.set_title(
        f"Empty segmentations ({n_empty}/{m.n_studies} studies)",
        fontsize=PLOT_SETTINGS["axes_titlesize"],
        pad=20,
    )
    fig.tight_layout()
    _save(fig, out)


def fig10_studies_per_patient(m: DatasetMetrics, out: Path) -> None:
    """Histogram of the number of studies per patient."""
    _setup_style()
    fig, ax = plt.subplots(figsize=get_figure_size("single"))

    counts = [pm.n_studies for pm in m.patients.values()]
    mean_c = np.mean(counts)
    median_c = np.median(counts)

    bins = np.arange(min(counts) - 0.5, max(counts) + 1.5, 1)
    ax.hist(
        counts, bins=bins, color=PAUL_TOL_BRIGHT["blue"], edgecolor="white", alpha=0.85
    )
    ax.axvline(
        mean_c,
        color=PAUL_TOL_BRIGHT["red"],
        linestyle="--",
        linewidth=1.2,
        label=f"Mean={mean_c:.1f}",
    )
    ax.axvline(
        median_c,
        color=PAUL_TOL_BRIGHT["green"],
        linestyle="-.",
        linewidth=1.2,
        label=f"Median={median_c:.1f}",
    )

    ax.set_xlabel("No. studies per patient")
    ax.set_ylabel("Count")
    ax.set_title("Studies per patient distribution")
    ax.legend(fontsize=PLOT_SETTINGS["legend_fontsize"])
    ax.set_xticks(range(min(counts), max(counts) + 1))

    fig.tight_layout()
    _save(fig, out)


def fig11_dataset_overview(m: DatasetMetrics, out: Path) -> None:
    """Multi-panel dataset overview: demographics, volume stats, summary."""
    _setup_style()
    fig = plt.figure(figsize=get_figure_size("double", height_ratio=0.6))
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 1])

    # Panel A: Key numbers
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.axis("off")
    n_with_meta = sum(1 for pm in m.patients.values() if pm.clinical is not None)
    text_a = (
        f"Patients:  {m.n_patients}\n"
        f"Total studies:  {m.n_studies}\n"
        f"Non-empty segs:  {m.n_non_empty}\n"
        f"Empty segs:  {len(m.empty_studies)}\n"
        f"Clinical metadata:  {n_with_meta}/{m.n_patients}\n"
        f"Modalities:  t1c, t1n, t2f, t2w\n"
        f"Voxel size:  1 mm³ isotropic\n"
        f"Atlas:  SRI24"
    )
    ax_a.text(
        0.1,
        0.95,
        text_a,
        transform=ax_a.transAxes,
        fontsize=9,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="0.95", edgecolor="0.7"),
    )
    ax_a.set_title("Dataset summary", fontsize=PLOT_SETTINGS["axes_titlesize"])

    # Panel B: Volume statistics box
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.axis("off")
    vols = [
        s.total_volume
        for pm in m.patients.values()
        for s in pm.studies
        if not s.is_empty
    ]
    if vols:
        varr = np.array(vols)
        text_b = (
            f"N = {len(vols)} non-empty studies\n\n"
            f"Mean:   {np.mean(varr):>10,.0f} mm³\n"
            f"Median: {np.median(varr):>10,.0f} mm³\n"
            f"Std:    {np.std(varr):>10,.0f} mm³\n"
            f"Min:    {np.min(varr):>10,.0f} mm³\n"
            f"Max:    {np.max(varr):>10,.0f} mm³\n"
            f"Q25:    {np.percentile(varr, 25):>10,.0f} mm³\n"
            f"Q75:    {np.percentile(varr, 75):>10,.0f} mm³"
        )
    else:
        text_b = "No non-empty studies"
    ax_b.text(
        0.1,
        0.95,
        text_b,
        transform=ax_b.transAxes,
        fontsize=9,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="0.95", edgecolor="0.7"),
    )
    ax_b.set_title("Volume statistics", fontsize=PLOT_SETTINGS["axes_titlesize"])

    # Panel C: Demographics (the 4 patients with clinical data)
    ax_c = fig.add_subplot(gs[0, 2])
    ax_c.axis("off")
    lines = []
    for pid, pm in sorted(m.patients.items()):
        if pm.clinical is None:
            continue
        age = pm.clinical.get("age", "?")
        sex_code = pm.clinical.get("sex", "?")
        sex = {"0": "F", "1": "M"}.get(str(sex_code), "?")
        lines.append(f"{pid}: age {age}, sex {sex}")
    if lines:
        text_c = "Patients with metadata:\n\n" + "\n".join(lines)
    else:
        text_c = "No clinical metadata available"
    ax_c.text(
        0.1,
        0.95,
        text_c,
        transform=ax_c.transAxes,
        fontsize=9,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="0.95", edgecolor="0.7"),
    )
    ax_c.set_title("Demographics", fontsize=PLOT_SETTINGS["axes_titlesize"])

    fig.tight_layout()
    _save(fig, out)


# ═════════════════════════════════════════════════════════════════════════════
# Orchestrator
# ═════════════════════════════════════════════════════════════════════════════


def generate_all_figures(m: DatasetMetrics, output_dir: Path) -> None:
    """Generate all 11 quantitative analysis figures.

    Args:
        m: Computed dataset metrics from the discovery phase.
        output_dir: Directory to save figures into (created if needed).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    figures = [
        ("fig01_volume_trajectories.pdf", fig01_volume_trajectories),
        (
            "fig02_volume_trajectories_per_label.pdf",
            fig02_volume_trajectories_per_label,
        ),
        ("fig03_volume_distributions.pdf", fig03_volume_distributions),
        ("fig04_tumor_heatmap.pdf", fig04_tumor_heatmap),
        ("fig05_dice_consistency.pdf", fig05_dice_consistency),
        ("fig06_growth_rates.pdf", fig06_growth_rates),
        ("fig07_label_composition.pdf", fig07_label_composition),
        ("fig08_spatial_analysis.pdf", fig08_spatial_analysis),
        ("fig09_empty_segmentation_analysis.pdf", fig09_empty_segmentation_analysis),
        ("fig10_studies_per_patient.pdf", fig10_studies_per_patient),
        ("fig11_dataset_overview.pdf", fig11_dataset_overview),
    ]

    for name, func in figures:
        logger.info("Generating %s...", name)
        try:
            func(m, output_dir / name)
        except Exception:
            logger.exception("Failed to generate %s", name)
