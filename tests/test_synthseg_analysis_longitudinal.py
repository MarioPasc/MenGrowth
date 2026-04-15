"""Tests for :mod:`mengrowth.synthseg.analysis.longitudinal`.

Covers:
* CV arithmetic on a tiny volume DataFrame.
* Tumor-proximity EDT distance with anisotropic voxel sampling.
* no-seg fallback path returns all-distant.
* Acceptance evaluation thresholding.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from mengrowth.synthseg.analysis.config import AnalysisConfig
from mengrowth.synthseg.analysis.longitudinal import (
    _region_min_distance_mm,
    attach_proximity_to_cv,
    classify_regions_by_tumor_proximity,
    compute_within_subject_cv,
    evaluate_acceptance,
)


# ---------------------------------------------------------------------------
# CV arithmetic
# ---------------------------------------------------------------------------


def test_within_subject_cv_basic() -> None:
    vols = pd.DataFrame(
        [
            dict(patient_id="P1", study_id="S0", region_name="L",
                 region_id=10, volume_mm3=100.0),
            dict(patient_id="P1", study_id="S1", region_name="L",
                 region_id=10, volume_mm3=110.0),
            dict(patient_id="P1", study_id="S2", region_name="L",
                 region_id=10, volume_mm3=90.0),
        ]
    )
    out = compute_within_subject_cv(vols)
    assert len(out) == 1
    row = out.iloc[0]
    # std(ddof=1)=10, mean=100 → CV=0.1
    assert row["mean"] == pytest.approx(100.0)
    assert row["std"] == pytest.approx(10.0)
    assert row["cv"] == pytest.approx(0.1)


def test_within_subject_cv_drops_single_timepoint() -> None:
    vols = pd.DataFrame(
        [
            dict(patient_id="P1", study_id="S0", region_name="L",
                 region_id=10, volume_mm3=100.0),
        ]
    )
    out = compute_within_subject_cv(vols)
    assert out.empty


# ---------------------------------------------------------------------------
# Tumor-proximity EDT with anisotropic sampling
# ---------------------------------------------------------------------------


def test_region_min_distance_respects_voxel_sampling() -> None:
    parc = np.zeros((10, 10, 10), dtype=np.int32)
    parc[0, 0, 0] = 10  # region at corner
    tumor = np.zeros_like(parc, dtype=bool)
    tumor[5, 0, 0] = True  # 5 voxels away along x

    # Isotropic 1 mm → distance 5 mm.
    d_iso = _region_min_distance_mm(parc, tumor, voxel_zooms_mm=(1.0, 1.0, 1.0))
    assert d_iso[10] == pytest.approx(5.0, abs=1e-6)

    # 2 mm along x → distance should double.
    d_aniso = _region_min_distance_mm(parc, tumor, voxel_zooms_mm=(2.0, 1.0, 1.0))
    assert d_aniso[10] == pytest.approx(10.0, abs=1e-6)


# ---------------------------------------------------------------------------
# No-seg fallback
# ---------------------------------------------------------------------------


def _make_parc(path: Path, labels_grid: np.ndarray) -> None:
    nib.save(nib.Nifti1Image(labels_grid.astype(np.int32), np.eye(4)), str(path))


def test_proximity_fallback_when_no_seg(tmp_path: Path) -> None:
    pre = tmp_path / "pre"
    ss = tmp_path / "ss"
    pid, sid = "MenGrowth-0001", "MenGrowth-0001-000"
    (pre / pid / sid).mkdir(parents=True)
    (ss / pid / sid).mkdir(parents=True)

    grid = np.zeros((4, 4, 4), dtype=np.int32)
    grid[0, 0, 0] = 10  # Left-Thalamus label id
    _make_parc(ss / pid / sid / "synthseg_parc.nii.gz", grid)
    # Matching t1n reference (same affine / shape).
    _make_parc(pre / pid / sid / "t1n.nii.gz", grid)

    # Minimum viable cohort metadata.
    meta = tmp_path / "meta.csv"
    pd.DataFrame(
        [dict(patient_id=pid, study_id=sid, sequence="t1n",
              spacing_x=1.0, spacing_y=1.0, spacing_z=1.0)]
    ).to_csv(meta, index=False)

    cfg = AnalysisConfig(
        synthseg_output_root=str(ss),
        preprocessed_root=str(pre),
        original_metadata_csv=str(meta),
        output_dir=str(tmp_path / "out"),
    )

    # Also need the volumes and QC csvs so _studies_with_finalized_outputs finds it.
    (ss / pid / sid / "synthseg_volumes.csv").write_text(
        "subject,left thalamus\n/x/t1n.nii.gz,9000.0\n"
    )
    (ss / pid / sid / "synthseg_qc.csv").write_text(
        "subject,qc\n/x/t1n.nii.gz,0.7\n"
    )

    df = classify_regions_by_tumor_proximity(cfg)
    assert (df["proximity_source"] == "fallback_no_seg").all()
    assert (df["proximity_class"] == "distant").all()
    assert df["min_dist_mm"].isna().all()


# ---------------------------------------------------------------------------
# Acceptance evaluation
# ---------------------------------------------------------------------------


def test_evaluate_acceptance_pass_and_fail() -> None:
    deep_gm = ["Left-Thalamus", "Right-Thalamus"]
    cv_df = pd.DataFrame(
        [
            dict(patient_id="P1", region_name="Left-Thalamus", cv=0.02,
                 proximity_class="distant", proximity_source="fallback_no_seg"),
            dict(patient_id="P1", region_name="Right-Thalamus", cv=0.03,
                 proximity_class="distant", proximity_source="fallback_no_seg"),
            dict(patient_id="P2", region_name="Left-Thalamus", cv=0.04,
                 proximity_class="distant", proximity_source="fallback_no_seg"),
        ]
    )
    res = evaluate_acceptance(cv_df, deep_gm, threshold=0.05)
    assert res.passed
    assert res.n_values == 3

    # Push median above threshold.
    cv_df.loc[:, "cv"] = [0.08, 0.09, 0.10]
    res2 = evaluate_acceptance(cv_df, deep_gm, threshold=0.05)
    assert not res2.passed


def test_attach_proximity_falls_back_on_empty_proximity_df() -> None:
    cv_df = pd.DataFrame(
        [dict(patient_id="P1", region_name="Left-Thalamus", cv=0.03)]
    )
    out = attach_proximity_to_cv(cv_df, pd.DataFrame())
    assert (out["proximity_class"] == "distant").all()
    assert (out["proximity_source"] == "fallback_no_seg").all()
