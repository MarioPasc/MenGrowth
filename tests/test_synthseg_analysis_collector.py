"""Tests for :mod:`mengrowth.synthseg.analysis.collector`.

Focused on the column-name normalisation bridge between SynthSeg's
human-readable volumes CSV and the canonical
:data:`SYNTHSEG_LABEL_MAP` keys, plus the timepoint-index assignment.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from mengrowth.synthseg.analysis.collector import (
    _assign_timepoint_index,
    _csv_key_to_canonical,
    _normalize,
    collect_cohort_metadata,
    collect_cohort_qc,
    collect_cohort_volumes,
)
from mengrowth.synthseg.analysis.config import AnalysisConfig
from mengrowth.synthseg.primitives import SYNTHSEG_LABEL_MAP


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def test_normalize_strips_whitespace_and_dashes() -> None:
    assert _normalize("Left-Thalamus") == "leftthalamus"
    assert _normalize("left thalamus") == "leftthalamus"
    assert _normalize("Brain-Stem") == "brainstem"
    assert _normalize("brain-stem") == "brainstem"


@pytest.mark.parametrize(
    "csv_key, expected",
    [
        ("left thalamus", "Left-Thalamus"),
        ("right thalamus", "Right-Thalamus"),
        ("brain-stem", "Brain-Stem"),
        ("Left-Thalamus", "Left-Thalamus"),
        ("3rd ventricle", "3rd-Ventricle"),
        ("CSF", "CSF"),
        ("total intracranial", None),
        ("unknown region", None),
    ],
)
def test_csv_key_to_canonical(csv_key: str, expected: str | None) -> None:
    assert _csv_key_to_canonical(csv_key) == expected


# ---------------------------------------------------------------------------
# Timepoint indexing
# ---------------------------------------------------------------------------


def test_timepoint_index_monotone_per_patient() -> None:
    df = pd.DataFrame(
        {
            "patient_id": [
                "P1",
                "P1",
                "P2",
                "P1",
                "P2",
            ],
            "study_id": [
                "P1-002",
                "P1-000",
                "P2-000",
                "P1-001",
                "P2-001",
            ],
        }
    )
    out = _assign_timepoint_index(df)
    p1 = out[out["patient_id"] == "P1"].sort_values("study_id")
    assert p1["timepoint_index"].tolist() == [0, 1, 2]
    p2 = out[out["patient_id"] == "P2"].sort_values("study_id")
    assert p2["timepoint_index"].tolist() == [0, 1]


# ---------------------------------------------------------------------------
# End-to-end on a tiny synthetic tree
# ---------------------------------------------------------------------------


def _make_nifti(path: Path, shape=(4, 4, 4)) -> None:
    nib.save(
        nib.Nifti1Image(np.zeros(shape, dtype=np.int16), np.eye(4)), str(path)
    )


def _make_volumes_csv(path: Path) -> None:
    # Keys taken from real SynthSeg --vol output: human-readable + extras.
    path.write_text(
        "subject,left thalamus,right thalamus,brain-stem,total intracranial\n"
        "/x/t1n.nii.gz,9000.0,9050.0,21000.0,1.4e6\n"
    )


def _make_qc_csv(path: Path, val: float = 0.7) -> None:
    path.write_text(f"subject,qc\n/x/t1n.nii.gz,{val}\n")


def _build_cohort(tmp_path: Path) -> AnalysisConfig:
    pre_root = tmp_path / "preprocessed"
    ss_root = tmp_path / "synthseg"
    meta_csv = tmp_path / "per_study_metrics.csv"

    # Two patients, 2 and 1 studies.
    for pid, sids in [("MenGrowth-0001", ["MenGrowth-0001-000", "MenGrowth-0001-001"]),
                      ("MenGrowth-0002", ["MenGrowth-0002-000"])]:
        for sid in sids:
            pre = pre_root / pid / sid
            out = ss_root / pid / sid
            pre.mkdir(parents=True, exist_ok=True)
            out.mkdir(parents=True, exist_ok=True)
            _make_nifti(pre / "t1n.nii.gz")
            _make_nifti(out / "synthseg_parc.nii.gz")
            _make_volumes_csv(out / "synthseg_volumes.csv")
            _make_qc_csv(out / "synthseg_qc.csv", val=0.8)

    # Metadata for t1n only (Analysis 1 consumer); spacing differs per study.
    pd.DataFrame(
        [
            dict(patient_id="MenGrowth-0001", study_id="MenGrowth-0001-000",
                 sequence="t1n", spacing_x=1.0, spacing_y=1.0, spacing_z=1.5),
            dict(patient_id="MenGrowth-0001", study_id="MenGrowth-0001-001",
                 sequence="t1n", spacing_x=1.0, spacing_y=1.0, spacing_z=3.2),
            dict(patient_id="MenGrowth-0002", study_id="MenGrowth-0002-000",
                 sequence="t1n", spacing_x=0.9, spacing_y=0.9, spacing_z=1.0),
        ]
    ).to_csv(meta_csv, index=False)

    return AnalysisConfig(
        synthseg_output_root=str(ss_root),
        preprocessed_root=str(pre_root),
        original_metadata_csv=str(meta_csv),
        output_dir=str(tmp_path / "out"),
    )


def test_collect_cohort_volumes_drops_non_label_map_rows(tmp_path: Path) -> None:
    cfg = _build_cohort(tmp_path)
    df = collect_cohort_volumes(cfg)
    # 3 studies × 3 valid labels (left thalamus, right thalamus, brain-stem).
    # "total intracranial" must be dropped.
    assert len(df) == 9
    assert set(df["region_name"]) == {
        "Left-Thalamus",
        "Right-Thalamus",
        "Brain-Stem",
    }
    assert df["volume_mm3"].gt(0).all()


def test_collect_cohort_qc_gives_one_row_per_study(tmp_path: Path) -> None:
    cfg = _build_cohort(tmp_path)
    df = collect_cohort_qc(cfg)
    assert len(df) == 3
    assert df["qc_score"].between(0, 1).all()
    # Timepoint index resets per patient.
    p1 = df[df["patient_id"] == "MenGrowth-0001"].sort_values("study_id")
    assert p1["timepoint_index"].tolist() == [0, 1]


def test_collect_cohort_metadata_derived_columns(tmp_path: Path) -> None:
    cfg = _build_cohort(tmp_path)
    df = collect_cohort_metadata(cfg)
    assert {"max_spacing", "min_spacing", "anisotropy_ratio"}.issubset(df.columns)
    row = df[df["study_id"] == "MenGrowth-0001-001"].iloc[0]
    assert row["max_spacing"] == pytest.approx(3.2)
    assert row["anisotropy_ratio"] == pytest.approx(3.2)
