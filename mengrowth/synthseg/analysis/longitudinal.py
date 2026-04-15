"""Analysis 2 — longitudinal within-subject CV.

Per (patient, region), CV is computed across all timepoints:

    CV_{i,r} = std_{i,r}(V) / mean_{i,r}(V)

Patients with <2 studies are dropped (CV undefined). Each region is
further classified as *tumor-adjacent* (min distance from the region
boundary to the tumor-mask boundary ≤ δ) or *tumor-distant* (> δ) using
:func:`scipy.ndimage.distance_transform_edt` with ``sampling=voxel_zooms_mm``.

When the preprocessed study directory has no ``seg.nii.gz``, every region
is labelled ``"distant"`` with ``proximity_source="fallback_no_seg"``.
This is the branch exercised on the local workstation; the full
stratification runs once segmentation is attached on Picasso.

The contralateral delta-CV is

    δCV_{i,r} = CV_{i,ipsi} − CV_{i,contra}

where the ipsi side is determined by the sign of the tumor-centroid x
coordinate in the parcellation's study-space (RAS: +x = right).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from mengrowth.synthseg.analysis.config import AnalysisConfig
from mengrowth.synthseg.primitives import SYNTHSEG_LABEL_MAP

logger = logging.getLogger(__name__)

_LATERAL_RE = re.compile(r"^(Left|Right)-(.+)$")


# ---------------------------------------------------------------------------
# CV across timepoints
# ---------------------------------------------------------------------------


def compute_within_subject_cv(volumes_df: pd.DataFrame) -> pd.DataFrame:
    """Per-(patient, region) CV of ``volume_mm3`` across timepoints.

    Parameters
    ----------
    volumes_df : pd.DataFrame
        Long-form DataFrame from :func:`collect_cohort_volumes` with columns
        ``patient_id, study_id, region_name, region_id, volume_mm3``.

    Returns
    -------
    pd.DataFrame
        Columns: ``patient_id, region_name, region_id, n_timepoints, mean,
        std, cv``. Rows for patients with fewer than 2 timepoints are
        dropped.
    """
    if volumes_df.empty:
        return pd.DataFrame(
            columns=[
                "patient_id",
                "region_name",
                "region_id",
                "n_timepoints",
                "mean",
                "std",
                "cv",
            ]
        )

    grp = volumes_df.groupby(["patient_id", "region_name", "region_id"])[
        "volume_mm3"
    ]
    agg = grp.agg(["count", "mean", "std"]).reset_index()
    agg = agg.rename(columns={"count": "n_timepoints"})

    # Need at least 2 timepoints for a defined CV.
    agg = agg[agg["n_timepoints"] >= 2].copy()
    agg["cv"] = np.where(agg["mean"] > 0, agg["std"] / agg["mean"], np.nan)

    n_dropped = (
        volumes_df.groupby("patient_id")["study_id"].nunique() < 2
    ).sum()
    if n_dropped:
        logger.info(
            "compute_within_subject_cv: %d patient(s) dropped (<2 studies)",
            int(n_dropped),
        )

    return agg.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Tumor proximity classification
# ---------------------------------------------------------------------------


def _region_min_distance_mm(
    parc_data: np.ndarray,
    tumor_mask: np.ndarray,
    voxel_zooms_mm: Sequence[float],
) -> Dict[int, float]:
    """Return ``{label_id: min_dist_mm}`` between region boundary and tumor.

    Distance is computed once via
    ``distance_transform_edt(~tumor_mask, sampling=voxel_zooms_mm)``; for
    each label the minimum within that label's voxels is taken.
    """
    if tumor_mask.sum() == 0:
        return {lid: float("nan") for lid in np.unique(parc_data) if lid != 0}

    dist = ndi.distance_transform_edt(~tumor_mask, sampling=voxel_zooms_mm)
    out: Dict[int, float] = {}
    for lid in np.unique(parc_data):
        if lid == 0:
            continue
        region = parc_data == lid
        if not region.any():
            continue
        out[int(lid)] = float(dist[region].min())
    return out


def classify_regions_by_tumor_proximity(
    cfg: AnalysisConfig,
) -> pd.DataFrame:
    """Per (patient, study, region) tumor-proximity classification.

    Returns
    -------
    pd.DataFrame
        Columns: ``patient_id, study_id, region_name, region_id,
        min_dist_mm, proximity_class, proximity_source``.
        ``proximity_class`` ∈ {"adjacent", "distant"};
        ``proximity_source`` ∈ {"seg", "fallback_no_seg", "error"}.
    """
    from mengrowth.synthseg.analysis.collector import (
        _studies_with_finalized_outputs,
    )

    studies = _studies_with_finalized_outputs(cfg)
    rows: List[Dict] = []
    threshold = cfg.tumor_proximity_threshold_mm
    id_to_name = {v: k for k, v in SYNTHSEG_LABEL_MAP.items()}

    for s in studies:
        parc_path = (
            Path(cfg.synthseg_output_root)
            / s.patient_id
            / s.study_id
            / cfg.parcellation_filename
        )
        seg_path = (
            Path(cfg.preprocessed_root)
            / s.patient_id
            / s.study_id
            / "seg.nii.gz"
        )

        try:
            parc_img = nib.load(str(parc_path))
            parc_data = np.asarray(parc_img.dataobj).astype(np.int32)
            zooms = parc_img.header.get_zooms()[:3]
        except Exception as e:
            logger.warning(
                "Skipping proximity for %s/%s (parc load failed: %s)",
                s.patient_id,
                s.study_id,
                e,
            )
            continue

        if seg_path.exists():
            try:
                seg_data = np.asarray(nib.load(str(seg_path)).dataobj) > 0
                dists = _region_min_distance_mm(
                    parc_data, seg_data.astype(bool), zooms
                )
                source = "seg"
            except Exception as e:
                logger.warning(
                    "Proximity: seg load failed for %s/%s (%s); "
                    "falling back to distant-only",
                    s.patient_id,
                    s.study_id,
                    e,
                )
                dists = {lid: float("nan") for lid in id_to_name}
                source = "error"
        else:
            dists = {lid: float("nan") for lid in id_to_name}
            source = "fallback_no_seg"

        for lid, name in id_to_name.items():
            d = dists.get(lid, float("nan"))
            if np.isnan(d):
                cls = "distant"
            else:
                cls = "adjacent" if d <= threshold else "distant"
            rows.append(
                {
                    "patient_id": s.patient_id,
                    "study_id": s.study_id,
                    "region_id": lid,
                    "region_name": name,
                    "min_dist_mm": d,
                    "proximity_class": cls,
                    "proximity_source": source,
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "patient_id",
                "study_id",
                "region_id",
                "region_name",
                "min_dist_mm",
                "proximity_class",
                "proximity_source",
            ]
        )
    return pd.DataFrame(rows)


def attach_proximity_to_cv(
    cv_df: pd.DataFrame, proximity_df: pd.DataFrame
) -> pd.DataFrame:
    """Attach a per-(patient, region) proximity class to ``cv_df``.

    A region is called *adjacent* for the patient iff **any** of that
    patient's studies classifies it as adjacent. Otherwise distant. The
    source is "fallback_no_seg" only if every study used the fallback.
    """
    if proximity_df.empty or cv_df.empty:
        out = cv_df.copy()
        out["proximity_class"] = "distant"
        out["proximity_source"] = "fallback_no_seg"
        return out

    agg = (
        proximity_df.groupby(["patient_id", "region_name"])
        .agg(
            adjacent_any=("proximity_class", lambda s: (s == "adjacent").any()),
            source_any_seg=(
                "proximity_source",
                lambda s: (s == "seg").any(),
            ),
        )
        .reset_index()
    )
    agg["proximity_class"] = np.where(
        agg["adjacent_any"], "adjacent", "distant"
    )
    agg["proximity_source"] = np.where(
        agg["source_any_seg"], "seg", "fallback_no_seg"
    )
    merged = cv_df.merge(
        agg[["patient_id", "region_name", "proximity_class", "proximity_source"]],
        on=["patient_id", "region_name"],
        how="left",
    )
    merged["proximity_class"] = merged["proximity_class"].fillna("distant")
    merged["proximity_source"] = merged["proximity_source"].fillna(
        "fallback_no_seg"
    )
    return merged


# ---------------------------------------------------------------------------
# Contralateral delta CV
# ---------------------------------------------------------------------------


def _tumor_ipsi_side(
    cfg: AnalysisConfig, patient_id: str, study_ids: Sequence[str]
) -> Optional[str]:
    """Return ``"Left"`` or ``"Right"`` from the average tumor centroid sign.

    Uses RAS convention: +x = right. Returns ``None`` if no seg is
    available for any of the given studies.
    """
    xs: List[float] = []
    for sid in study_ids:
        seg_path = (
            Path(cfg.preprocessed_root)
            / patient_id
            / sid
            / "seg.nii.gz"
        )
        if not seg_path.exists():
            continue
        try:
            img = nib.load(str(seg_path))
            data = np.asarray(img.dataobj) > 0
            if not data.any():
                continue
            ijk = np.argwhere(data).mean(axis=0)
            aff = img.affine
            ras = aff[:3, :3] @ ijk + aff[:3, 3]
            xs.append(float(ras[0]))
        except Exception:
            continue
    if not xs:
        return None
    return "Right" if np.mean(xs) > 0 else "Left"


def compute_contralateral_delta_cv(
    cv_df: pd.DataFrame, cfg: AnalysisConfig
) -> pd.DataFrame:
    """Per (patient, region-pair) δCV = CV(ipsi) − CV(contra).

    Skips patients without any ``seg.nii.gz`` (ipsi side undefined).
    """
    from mengrowth.synthseg.analysis.collector import (
        _studies_with_finalized_outputs,
    )

    if cv_df.empty:
        return pd.DataFrame(
            columns=[
                "patient_id",
                "region_pair",
                "cv_ipsi",
                "cv_contra",
                "delta_cv",
                "ipsi_side",
            ]
        )

    studies = _studies_with_finalized_outputs(cfg)
    by_patient: Dict[str, List[str]] = {}
    for s in studies:
        by_patient.setdefault(s.patient_id, []).append(s.study_id)

    # Build (bare_region, left_cv, right_cv) per patient via pivot.
    lateral = cv_df.copy()
    m = lateral["region_name"].str.extract(_LATERAL_RE.pattern)
    lateral["side"] = m[0]
    lateral["bare_region"] = m[1]
    lateral = lateral.dropna(subset=["side", "bare_region"])

    pivot = lateral.pivot_table(
        index=["patient_id", "bare_region"],
        columns="side",
        values="cv",
        aggfunc="first",
    ).reset_index()

    rows: List[Dict] = []
    for _, r in pivot.iterrows():
        pid = r["patient_id"]
        ipsi_side = _tumor_ipsi_side(cfg, pid, by_patient.get(pid, []))
        if ipsi_side is None:
            continue
        cv_left = r.get("Left", np.nan)
        cv_right = r.get("Right", np.nan)
        if np.isnan(cv_left) or np.isnan(cv_right):
            continue
        cv_ipsi = cv_right if ipsi_side == "Right" else cv_left
        cv_contra = cv_left if ipsi_side == "Right" else cv_right
        rows.append(
            {
                "patient_id": pid,
                "region_pair": r["bare_region"],
                "cv_ipsi": float(cv_ipsi),
                "cv_contra": float(cv_contra),
                "delta_cv": float(cv_ipsi - cv_contra),
                "ipsi_side": ipsi_side,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Acceptance criterion (plan §7)
# ---------------------------------------------------------------------------


@dataclass
class AcceptanceResult:
    """Outcome of the headline tumor-distant deep-GM CV acceptance test."""

    threshold: float = 0.05
    median_cv: float = float("nan")
    mean_cv: float = float("nan")
    n_values: int = 0
    regions_used: List[str] = field(default_factory=list)
    passed: bool = False
    note: str = ""


def evaluate_acceptance(
    cv_df_with_proximity: pd.DataFrame,
    deep_grey_matter: Sequence[str],
    threshold: float,
) -> AcceptanceResult:
    """Median CV across tumor-distant deep grey matter vs. threshold."""
    sub = cv_df_with_proximity[
        cv_df_with_proximity["region_name"].isin(deep_grey_matter)
    ]
    if "proximity_class" in sub.columns:
        sub = sub[sub["proximity_class"] == "distant"]
    sub = sub.dropna(subset=["cv"])

    if sub.empty:
        return AcceptanceResult(
            threshold=threshold,
            note="no tumor-distant deep-GM CV values available",
            regions_used=list(deep_grey_matter),
        )

    med = float(sub["cv"].median())
    mean = float(sub["cv"].mean())
    return AcceptanceResult(
        threshold=threshold,
        median_cv=med,
        mean_cv=mean,
        n_values=int(len(sub)),
        regions_used=list(deep_grey_matter),
        passed=bool(med < threshold),
        note=(
            "all sources = fallback_no_seg"
            if (sub.get("proximity_source") == "fallback_no_seg").all()
            else ""
        ),
    )
