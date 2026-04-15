"""Figure generation for the Phase 2 SynthSeg QC analysis.

All plot functions follow the same protocol: they receive the relevant
DataFrame(s), an output directory, and a DPI; they save the figure under
``{out_dir}/<name>.<ext>`` and return the :class:`pathlib.Path` so the
report assembler can embed it.

Styling is set once at module import (seaborn whitegrid + colorblind
palette, 300 DPI save). This matches the convention in
``mengrowth/preprocessing/quality_analysis/visualize.py``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# Module-level style (called once on import).
sns.set_style("whitegrid")
sns.set_palette("colorblind")
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)


def _save(fig: plt.Figure, out: Path, dpi: int) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Analysis 1 — QC vs original T1n acquisition spacing
# ---------------------------------------------------------------------------


def plot_qc_vs_spacing_scatter(
    merged: pd.DataFrame,
    out_dir: Path,
    dpi: int,
    name: str = "qc_vs_original_spacing.png",
) -> Path:
    """Scatter of SynthSeg QC vs. original T1n max spacing.

    Points are coloured by patient; a linear regression with 95% CI band
    is overlaid via :func:`seaborn.regplot`.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(
        data=merged,
        x="max_spacing",
        y="qc_score",
        hue="patient_id",
        ax=ax,
        legend=False,
        s=28,
        alpha=0.75,
    )
    sns.regplot(
        data=merged,
        x="max_spacing",
        y="qc_score",
        ax=ax,
        scatter=False,
        color="black",
        line_kws={"linewidth": 1.5},
    )
    ax.set_xlabel("Original T1n max voxel spacing (mm)")
    ax.set_ylabel("SynthSeg QC score (0 = bad, 1 = good)")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("SynthSeg QC vs. original T1n acquisition spacing")
    return _save(fig, out_dir / name, dpi)


def plot_qc_vs_spacing_boxplot(
    merged: pd.DataFrame,
    out_dir: Path,
    dpi: int,
    name: str = "qc_vs_spacing_boxplot.png",
) -> Path:
    """Boxplot of QC scores stratified by spacing bin (<1.5, 1.5-3, >3 mm)."""
    df = merged.copy()
    df["spacing_bin"] = pd.cut(
        df["max_spacing"],
        bins=[-np.inf, 1.5, 3.0, np.inf],
        labels=["<1.5 mm", "1.5–3 mm", ">3 mm"],
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.boxplot(
        data=df,
        x="spacing_bin",
        y="qc_score",
        ax=ax,
        showfliers=True,
        width=0.55,
    )
    sns.stripplot(
        data=df,
        x="spacing_bin",
        y="qc_score",
        ax=ax,
        color="black",
        size=3,
        alpha=0.5,
        jitter=0.2,
    )
    ax.set_xlabel("Original T1n max spacing bin")
    ax.set_ylabel("SynthSeg QC score")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("QC distribution by acquisition spacing bin")
    return _save(fig, out_dir / name, dpi)


# ---------------------------------------------------------------------------
# Analysis 2 — longitudinal CV
# ---------------------------------------------------------------------------


def plot_region_cv_violin(
    cv_df: pd.DataFrame,
    out_dir: Path,
    dpi: int,
    name: str = "region_cv_violin.png",
) -> Path:
    """Violin of within-subject CV by region, hue = tumor proximity."""
    fig_h = max(5.0, 0.22 * cv_df["region_name"].nunique())
    fig, ax = plt.subplots(figsize=(9, fig_h))
    hue = "proximity_class" if "proximity_class" in cv_df.columns else None
    sns.violinplot(
        data=cv_df,
        y="region_name",
        x="cv",
        hue=hue,
        split=(hue is not None and cv_df[hue].nunique() == 2),
        inner="quartile",
        ax=ax,
        cut=0,
    )
    ax.set_xlabel("Within-subject volume CV")
    ax.set_ylabel("Region")
    ax.set_title("Longitudinal CV distribution per region")
    return _save(fig, out_dir / name, dpi)


def plot_region_cv_heatmap(
    cv_df: pd.DataFrame,
    out_dir: Path,
    dpi: int,
    name: str = "region_cv_heatmap.png",
) -> Path:
    """Heatmap regions × patients. Cell value = median CV across regions."""
    pivot = cv_df.pivot_table(
        index="region_name",
        columns="patient_id",
        values="cv",
        aggfunc="first",
    )
    fig_h = max(5.0, 0.22 * pivot.shape[0])
    fig_w = max(8.0, 0.18 * pivot.shape[1])
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(
        pivot,
        cmap="rocket_r",
        vmin=0.0,
        vmax=min(0.5, float(np.nanmax(pivot.values)) if pivot.size else 0.5),
        cbar_kws={"label": "CV"},
        ax=ax,
    )
    ax.set_title("Within-subject volume CV — regions × patients")
    ax.set_xlabel("Patient")
    ax.set_ylabel("Region")
    ax.tick_params(axis="x", labelrotation=90)
    return _save(fig, out_dir / name, dpi)


def plot_tumor_distant_mean_cv_bar(
    cv_df: pd.DataFrame,
    deep_grey_matter: Sequence[str],
    out_dir: Path,
    dpi: int,
    threshold: float,
    name: str = "tumor_distant_mean_cv_bar.png",
) -> Path:
    """Headline bar chart: median CV per region for tumor-distant deep GM."""
    dgm = cv_df[cv_df["region_name"].isin(deep_grey_matter)].copy()
    if "proximity_class" in dgm.columns:
        dgm = dgm[dgm["proximity_class"] == "distant"]

    agg = (
        dgm.groupby("region_name")["cv"]
        .agg(["median", "mean", "std", "count"])
        .reset_index()
        .sort_values("region_name")
    )
    fig, ax = plt.subplots(figsize=(7, 4.2))
    sns.barplot(
        data=agg,
        y="region_name",
        x="median",
        ax=ax,
        color=sns.color_palette("colorblind")[0],
    )
    ax.axvline(threshold, linestyle="--", color="red", linewidth=1.2, label=f"threshold {threshold:g}")
    ax.set_xlabel("Median within-subject volume CV (tumor-distant)")
    ax.set_ylabel("Region")
    ax.set_title("Longitudinal CV — tumor-distant deep grey matter")
    ax.legend(loc="lower right", frameon=True)
    return _save(fig, out_dir / name, dpi)


def plot_contralateral_delta_cv(
    delta_df: pd.DataFrame,
    out_dir: Path,
    dpi: int,
    name: str = "contralateral_delta_cv.png",
) -> Path:
    """Ipsilateral vs contralateral CV (paired box/strip by region)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(
        data=delta_df,
        x="region_pair",
        y="delta_cv",
        ax=ax,
        width=0.5,
    )
    sns.stripplot(
        data=delta_df,
        x="region_pair",
        y="delta_cv",
        ax=ax,
        color="black",
        alpha=0.5,
        size=3,
        jitter=0.2,
    )
    ax.axhline(0, linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("Region pair")
    ax.set_ylabel("CV(ipsi) − CV(contra)")
    ax.set_title("Contralateral CV difference by region pair")
    return _save(fig, out_dir / name, dpi)


# ---------------------------------------------------------------------------
# Analysis 3 — cross-modal ROI metrics
# ---------------------------------------------------------------------------


def plot_roi_gradient_vs_spacing(
    gradient_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    modality: str,
    region_name: str,
    out_dir: Path,
    dpi: int,
    name: Optional[str] = None,
) -> Optional[Path]:
    """Scatter of boundary-gradient energy vs. original acquisition spacing.

    Returns ``None`` if there are no rows (e.g. modality missing from the
    cohort).
    """
    mod_meta = meta_df[meta_df["sequence"].str.lower() == modality.lower()]
    mod_grad = gradient_df[
        (gradient_df["modality"].str.lower() == modality.lower())
        & (gradient_df["region_name"] == region_name)
    ]
    if mod_grad.empty or mod_meta.empty:
        logger.warning(
            "plot_roi_gradient_vs_spacing: no data for modality=%s region=%s",
            modality,
            region_name,
        )
        return None

    merged = mod_grad.merge(
        mod_meta[["patient_id", "study_id", "max_spacing"]],
        on=["patient_id", "study_id"],
        how="inner",
    )
    if merged.empty:
        return None

    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.regplot(
        data=merged,
        x="max_spacing",
        y="boundary_gradient",
        ax=ax,
        scatter_kws={"alpha": 0.75, "s": 26},
        color=sns.color_palette("colorblind")[2],
    )
    ax.set_xlabel(f"Original {modality.upper()} max spacing (mm)")
    ax.set_ylabel(f"Boundary gradient in {region_name}")
    ax.set_title(f"{modality.upper()} boundary gradient vs. original spacing")
    default_name = f"{modality.lower()}_gradient_vs_spacing.png"
    return _save(fig, out_dir / (name or default_name), dpi)


def plot_roi_cv_vs_spacing(
    cv_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    modality: str,
    region_name: str,
    out_dir: Path,
    dpi: int,
    name: Optional[str] = None,
) -> Optional[Path]:
    """Scatter of within-region intensity CV vs. original acquisition spacing."""
    mod_meta = meta_df[meta_df["sequence"].str.lower() == modality.lower()]
    mod_cv = cv_df[
        (cv_df["modality"].str.lower() == modality.lower())
        & (cv_df["region_name"] == region_name)
    ]
    if mod_cv.empty or mod_meta.empty:
        return None

    merged = mod_cv.merge(
        mod_meta[["patient_id", "study_id", "max_spacing"]],
        on=["patient_id", "study_id"],
        how="inner",
    )
    if merged.empty:
        return None

    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.regplot(
        data=merged,
        x="max_spacing",
        y="within_region_cv",
        ax=ax,
        scatter_kws={"alpha": 0.75, "s": 26},
        color=sns.color_palette("colorblind")[1],
    )
    ax.set_xlabel(f"Original {modality.upper()} max spacing (mm)")
    ax.set_ylabel(f"Intensity CV in {region_name}")
    ax.set_title(f"{modality.upper()} intensity CV vs. original spacing")
    default_name = f"{modality.lower()}_cv_vs_spacing.png"
    return _save(fig, out_dir / (name or default_name), dpi)
