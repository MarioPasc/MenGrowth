"""Analysis 3 — ROI-based cross-modal quality metrics.

Uses the T1n SynthSeg parcellation as a structural template to probe
preprocessed T1c / T2w / T2f without running SynthSeg on them.

Three per-(modality, region, study) metrics are produced:

1. **Within-region intensity CV** — ``std(I[L=r]) / mean(I[L=r])``. A
   stable intracranial reference should have moderate, similar CV across
   studies; outliers flag normalization failures or bias.
2. **Boundary gradient energy** — mean of ``||∇I||`` on the one-voxel
   shell of each region, with ``∇I`` built from three Sobel passes. Sharp
   tissue transitions → high gradient.
3. **Longitudinal within-region intensity stability** — CV over
   timepoints of the mean regional intensity. Flags drifting
   normalization.

Per-study work parallelises cleanly — 179 studies × 4 modalities × 17
regions ≈ 12k region evaluations. The helper
:func:`compute_region_metrics_for_study` is picklable and safe for
:class:`concurrent.futures.ProcessPoolExecutor`.
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from mengrowth.synthseg.analysis.config import AnalysisConfig
from mengrowth.synthseg.primitives import SYNTHSEG_LABEL_MAP

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure array helpers (unit-testable in isolation)
# ---------------------------------------------------------------------------


def compute_gradient_magnitude(img: np.ndarray) -> np.ndarray:
    """Return ``||∇I||`` via three Sobel passes (float32)."""
    f = img.astype(np.float32, copy=False)
    gx = ndi.sobel(f, axis=0, mode="nearest")
    gy = ndi.sobel(f, axis=1, mode="nearest")
    gz = ndi.sobel(f, axis=2, mode="nearest")
    return np.sqrt(gx * gx + gy * gy + gz * gz)


def region_boundary_mask(region_mask: np.ndarray) -> np.ndarray:
    """Return the one-voxel shell ``mask XOR binary_erosion(mask)``."""
    if not region_mask.any():
        return np.zeros_like(region_mask, dtype=bool)
    eroded = ndi.binary_erosion(region_mask, iterations=1, border_value=0)
    return np.logical_xor(region_mask, eroded)


def within_region_cv(img: np.ndarray, region_mask: np.ndarray) -> float:
    """CV of intensities within ``region_mask`` (NaN if empty or mean≈0)."""
    if not region_mask.any():
        return float("nan")
    vals = img[region_mask].astype(np.float64, copy=False)
    mu = float(vals.mean())
    if not np.isfinite(mu) or abs(mu) < 1e-12:
        return float("nan")
    return float(vals.std(ddof=1) / mu) if vals.size > 1 else float("nan")


def boundary_gradient_energy(
    grad_mag: np.ndarray, region_mask: np.ndarray
) -> float:
    """Mean ``||∇I||`` on the boundary shell of ``region_mask``."""
    boundary = region_boundary_mask(region_mask)
    if not boundary.any():
        return float("nan")
    return float(grad_mag[boundary].mean())


# ---------------------------------------------------------------------------
# Per-study worker
# ---------------------------------------------------------------------------


@dataclass
class _StudyJob:
    """Picklable unit of work for the parallel region-metrics pass."""

    patient_id: str
    study_id: str
    parc_path: str
    preprocessed_dir: str
    modalities: Tuple[str, ...]
    region_names: Tuple[str, ...]


def _compute_one_study(job: _StudyJob) -> List[Dict]:
    """Compute all per-(modality, region) metrics for a single study."""
    rows: List[Dict] = []
    try:
        parc = np.asarray(nib.load(job.parc_path).dataobj).astype(np.int32)
    except Exception as exc:
        logger.warning(
            "Failed to load parc for %s/%s: %s",
            job.patient_id,
            job.study_id,
            exc,
        )
        return rows

    label_ids = {name: SYNTHSEG_LABEL_MAP[name] for name in job.region_names}

    for mod in job.modalities:
        mod_path = Path(job.preprocessed_dir) / f"{mod}.nii.gz"
        if not mod_path.exists():
            continue
        try:
            img = np.asarray(nib.load(str(mod_path)).dataobj).astype(
                np.float32
            )
        except Exception as exc:
            logger.warning(
                "Failed to load %s for %s/%s: %s",
                mod,
                job.patient_id,
                job.study_id,
                exc,
            )
            continue

        if img.shape != parc.shape:
            logger.warning(
                "Shape mismatch parc=%s vs %s=%s for %s/%s — skipping",
                parc.shape,
                mod,
                img.shape,
                job.patient_id,
                job.study_id,
            )
            continue

        grad_mag = compute_gradient_magnitude(img)

        for region_name, lid in label_ids.items():
            mask = parc == lid
            if not mask.any():
                continue
            rows.append(
                {
                    "patient_id": job.patient_id,
                    "study_id": job.study_id,
                    "modality": mod,
                    "region_name": region_name,
                    "region_id": int(lid),
                    "mean_intensity": float(img[mask].mean()),
                    "within_region_cv": within_region_cv(img, mask),
                    "boundary_gradient": boundary_gradient_energy(
                        grad_mag, mask
                    ),
                    "n_voxels": int(mask.sum()),
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_region_metrics(
    cfg: AnalysisConfig,
    region_names: Optional[Sequence[str]] = None,
    modalities: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Compute all region metrics across the cohort.

    Returns a long-form DataFrame with columns
    ``patient_id, study_id, modality, region_name, region_id,
    mean_intensity, within_region_cv, boundary_gradient, n_voxels``.
    """
    from mengrowth.synthseg.analysis.collector import (
        _studies_with_finalized_outputs,
    )

    rnames = tuple(region_names or cfg.reliable_regions)
    mods = tuple(m.lower() for m in (modalities or cfg.modalities_for_roi_analysis))

    studies = _studies_with_finalized_outputs(cfg)
    jobs: List[_StudyJob] = []
    for s in studies:
        parc = (
            Path(cfg.synthseg_output_root)
            / s.patient_id
            / s.study_id
            / cfg.parcellation_filename
        )
        prep = Path(cfg.preprocessed_root) / s.patient_id / s.study_id
        jobs.append(
            _StudyJob(
                patient_id=s.patient_id,
                study_id=s.study_id,
                parc_path=str(parc),
                preprocessed_dir=str(prep),
                modalities=mods,
                region_names=rnames,
            )
        )

    rows: List[Dict] = []
    if cfg.n_workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=cfg.n_workers) as pool:
            futures = [pool.submit(_compute_one_study, j) for j in jobs]
            for i, fut in enumerate(as_completed(futures), start=1):
                rows.extend(fut.result())
                if i % 25 == 0 or i == len(futures):
                    logger.info(
                        "Region metrics: %d/%d studies done",
                        i,
                        len(futures),
                    )
    else:
        for i, j in enumerate(jobs, start=1):
            rows.extend(_compute_one_study(j))
            if i % 25 == 0 or i == len(jobs):
                logger.info("Region metrics: %d/%d studies done", i, len(jobs))

    if not rows:
        logger.warning("compute_region_metrics: no rows produced")
        return pd.DataFrame(
            columns=[
                "patient_id",
                "study_id",
                "modality",
                "region_name",
                "region_id",
                "mean_intensity",
                "within_region_cv",
                "boundary_gradient",
                "n_voxels",
            ]
        )

    return pd.DataFrame(rows).sort_values(
        ["patient_id", "study_id", "modality", "region_id"]
    ).reset_index(drop=True)


def compute_longitudinal_intensity_cv(
    metrics_df: pd.DataFrame,
) -> pd.DataFrame:
    """Per-(patient, modality, region) CV of the mean-intensity timeseries.

    A drift in ``mean_intensity`` across timepoints exposes normalization
    failures: an ideal normalization yields near-constant regional means.
    Patients with <2 timepoints per (modality, region) cell are dropped.
    """
    if metrics_df.empty:
        return pd.DataFrame(
            columns=[
                "patient_id",
                "modality",
                "region_name",
                "region_id",
                "n_timepoints",
                "mean_of_means",
                "std_of_means",
                "intensity_cv",
            ]
        )

    grp = metrics_df.groupby(
        ["patient_id", "modality", "region_name", "region_id"]
    )["mean_intensity"]
    out = grp.agg(["count", "mean", "std"]).reset_index()
    out = out.rename(
        columns={
            "count": "n_timepoints",
            "mean": "mean_of_means",
            "std": "std_of_means",
        }
    )
    out = out[out["n_timepoints"] >= 2].copy()
    out["intensity_cv"] = np.where(
        out["mean_of_means"].abs() > 1e-12,
        out["std_of_means"] / out["mean_of_means"].abs(),
        np.nan,
    )
    return out.reset_index(drop=True)
