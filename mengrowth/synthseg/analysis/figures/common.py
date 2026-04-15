"""Shared helpers for publication figures.

Centralises styling, bilateral-merge logic, anatomical grouping, and
patient-selection heuristics that are reused between the quantitative
(:mod:`.fig_quantitative`) and qualitative (:mod:`.fig_qualitative`)
figure scripts.

All plot functions here honour the project-wide IEEE palette defined in
:mod:`mengrowth.synthseg.analysis.style`. The top-level
``apply_ieee_style`` helper in that module references an undefined
``CONDITION_COLORS`` symbol, so we route through the safe fallback
instead and apply a few extra rcParams tuned for dense multi-panel
layouts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from mengrowth.synthseg.analysis.style import (
    PAUL_TOL_BRIGHT,
    PAUL_TOL_MUTED,
    PLOT_SETTINGS,
    _apply_fallback_ieee_style,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bilateral merge + anatomical grouping
# ---------------------------------------------------------------------------

#: Midline regions that are not laterality-paired.
MIDLINE_REGIONS: tuple[str, ...] = ("3rd-Ventricle", "4th-Ventricle", "Brain-Stem", "CSF")

#: Display names keyed by the bilateral "base" name (after stripping
#: Left-/Right-). Midline labels map to themselves with prettier casing.
DISPLAY_NAME: dict[str, str] = {
    "Cerebral-Cortex": "Cerebral Cortex",
    "Cerebral-White-Matter": "Cerebral White Matter",
    "Thalamus": "Thalamus",
    "Putamen": "Putamen",
    "Caudate": "Caudate",
    "Pallidum": "Pallidum",
    "Hippocampus": "Hippocampus",
    "Amygdala": "Amygdala",
    "Accumbens-area": "Accumbens Area",
    "VentralDC": "Ventral DC",
    "Lateral-Ventricle": "Lateral Ventricle",
    "3rd-Ventricle": "3rd Ventricle",
    "4th-Ventricle": "4th Ventricle",
    "CSF": "CSF",
    "Cerebellum-Cortex": "Cerebellum Cortex",
    "Cerebellum-White-Matter": "Cerebellum White Matter",
    "Brain-Stem": "Brain Stem",
}

#: Anatomical groups, in the order they appear from top to bottom of the
#: heatmap. Within each group the regions are ordered from cortex/shell
#: outwards to fine structures so bilateral homologues cluster visually.
REGION_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Parenchyma",
        ("Cerebral-Cortex", "Cerebral-White-Matter"),
    ),
    (
        "Deep Grey",
        (
            "Thalamus",
            "Putamen",
            "Caudate",
            "Pallidum",
            "Hippocampus",
            "Amygdala",
            "Accumbens-area",
            "VentralDC",
        ),
    ),
    (
        "Ventricular",
        ("Lateral-Ventricle", "3rd-Ventricle", "4th-Ventricle", "CSF"),
    ),
    (
        "Cerebellum",
        ("Cerebellum-Cortex", "Cerebellum-White-Matter", "Brain-Stem"),
    ),
)


def _strip_laterality(region_name: str) -> str:
    """Strip Left-/Right- prefix; return midline names unchanged."""
    if region_name.startswith("Left-"):
        return region_name[len("Left-") :]
    if region_name.startswith("Right-"):
        return region_name[len("Right-") :]
    return region_name


def bilateral_merge_cv(cv_df: pd.DataFrame) -> pd.DataFrame:
    """Average left/right homologous CV values per (patient, region base).

    Parameters
    ----------
    cv_df
        Long-format DataFrame with at least ``patient_id``, ``region_name``
        and ``cv`` columns (output of
        :func:`mengrowth.synthseg.analysis.longitudinal.compute_within_subject_cv`).

    Returns
    -------
    pandas.DataFrame
        Columns ``patient_id``, ``region_base``, ``display_name``, ``cv``.
    """
    df = cv_df.copy()
    df["region_base"] = df["region_name"].apply(_strip_laterality)
    merged = (
        df.groupby(["patient_id", "region_base"], as_index=False)["cv"]
        .mean()
    )
    merged["display_name"] = merged["region_base"].map(DISPLAY_NAME)
    return merged


@dataclass(frozen=True)
class HeatmapLayout:
    """Indices and labels needed to render the grouped CV heatmap."""

    region_bases_ordered: tuple[str, ...]
    display_labels: tuple[str, ...]
    group_labels: tuple[str, ...]
    #: Index (in ``region_bases_ordered``) of the first row of each group.
    group_row_starts: tuple[int, ...]


def build_heatmap_layout() -> HeatmapLayout:
    """Flatten :data:`REGION_GROUPS` into the row order used by the heatmap."""
    region_bases: list[str] = []
    group_labels: list[str] = []
    group_row_starts: list[int] = []
    for label, regions in REGION_GROUPS:
        group_row_starts.append(len(region_bases))
        group_labels.append(label)
        region_bases.extend(regions)
    display_labels = tuple(DISPLAY_NAME[r] for r in region_bases)
    return HeatmapLayout(
        region_bases_ordered=tuple(region_bases),
        display_labels=display_labels,
        group_labels=tuple(group_labels),
        group_row_starts=tuple(group_row_starts),
    )


# ---------------------------------------------------------------------------
# Patient ordering / selection
# ---------------------------------------------------------------------------


def mean_qc_per_patient(qc_df: pd.DataFrame) -> pd.Series:
    """Return per-patient mean QC score, indexed by ``patient_id``."""
    return qc_df.groupby("patient_id")["qc_score"].mean()


def order_patients_by_mean_qc(qc_df: pd.DataFrame, patient_ids: Iterable[str]) -> list[str]:
    """Sort ``patient_ids`` ascending by mean QC score (worst → best)."""
    means = mean_qc_per_patient(qc_df)
    return sorted(patient_ids, key=lambda pid: float(means.get(pid, np.nan)))


def select_showcase_patients(
    qc_df: pd.DataFrame,
    min_studies: int = 3,
) -> tuple[str, str]:
    """Pick one best-case and one worst-case patient for the overlay figure.

    Parameters
    ----------
    qc_df
        ``cohort_qc.csv`` as a DataFrame.
    min_studies
        Minimum number of studies a patient must have to be eligible.

    Returns
    -------
    tuple
        ``(best_patient_id, worst_patient_id)``.

    Raises
    ------
    ValueError
        If fewer than two patients satisfy the minimum-studies criterion.
    """
    agg = (
        qc_df.groupby("patient_id")
        .agg(n_studies=("study_id", "nunique"), mean_qc=("qc_score", "mean"))
        .reset_index()
    )
    eligible = agg[agg["n_studies"] >= min_studies].copy()
    if len(eligible) < 2:
        raise ValueError(
            f"Fewer than two patients have ≥ {min_studies} studies "
            f"(eligible={len(eligible)})."
        )
    # Best: highest mean QC, break ties by more studies.
    eligible_sorted_best = eligible.sort_values(
        ["mean_qc", "n_studies"], ascending=[False, False]
    )
    best = str(eligible_sorted_best.iloc[0]["patient_id"])
    # Worst: lowest mean QC, prefer patient with more studies for visual impact.
    eligible_sorted_worst = eligible.sort_values(
        ["mean_qc", "n_studies"], ascending=[True, False]
    )
    worst = str(eligible_sorted_worst.iloc[0]["patient_id"])
    logger.info("Showcase patients: best=%s, worst=%s", best, worst)
    return best, worst


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------


def apply_pub_style() -> None:
    """Apply the IEEE fallback rcParams without the buggy colour-cycle patch.

    ``style.apply_ieee_style`` references an undefined ``CONDITION_COLORS``
    symbol and crashes on call. We apply the safe fallback directly and
    add a Paul-Tol-muted colour cycle.
    """
    import matplotlib.pyplot as plt

    _apply_fallback_ieee_style()
    plt.rcParams.update(
        {
            "axes.prop_cycle": plt.cycler(color=PAUL_TOL_MUTED),
            "axes.grid": False,  # grids clutter heatmaps and joint plots
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def paul_tol_color(name: str) -> str:
    """Convenience accessor for a Paul Tol "bright" palette hex."""
    return PAUL_TOL_BRIGHT[name]


def save_figure(fig, out: Path, dpi: int = 300) -> Path:
    """Write ``fig`` to ``out`` (creating parent dirs), close it, return path.

    Preserves ``fig.patch`` facecolor/edgecolor so figures with a black
    canvas background are saved faithfully (matplotlib defaults otherwise
    force a white background).
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out,
        dpi=dpi,
        facecolor=fig.get_facecolor(),
        edgecolor=fig.get_edgecolor(),
    )
    import matplotlib.pyplot as plt

    plt.close(fig)
    return out


def plot_settings() -> dict:
    """Expose :data:`PLOT_SETTINGS` without re-importing elsewhere."""
    return PLOT_SETTINGS
