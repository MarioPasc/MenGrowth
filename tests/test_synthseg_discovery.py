"""Tests for :mod:`mengrowth.synthseg.discovery`."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from mengrowth.synthseg.config import SynthSegConfig
from mengrowth.synthseg.discovery import (
    discover_patients,
    discover_studies,
    studies_for_patient,
)


def _make_nifti(path: Path, shape: tuple = (8, 8, 4)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.zeros(shape, dtype=np.float32)
    img = nib.Nifti1Image(data, affine=np.eye(4))
    nib.save(img, str(path))


def _make_cohort(root: Path, patient_studies: dict) -> None:
    """Create a fake cohort tree. ``patient_studies`` maps pid -> list of sid."""
    for pid, studies in patient_studies.items():
        for sid in studies:
            _make_nifti(root / pid / sid / "t1n.nii.gz")


@pytest.fixture
def cfg(tmp_path: Path) -> SynthSegConfig:
    return SynthSegConfig(
        input_root=str(tmp_path / "in"),
        output_root=str(tmp_path / "out"),
        synthseg_repo=str(tmp_path / "SynthSeg"),
    )


def test_discover_patients_sorted(tmp_path: Path, cfg: SynthSegConfig) -> None:
    root = tmp_path / "in"
    _make_cohort(
        root,
        {
            "MenGrowth-0003": ["MenGrowth-0003-000"],
            "MenGrowth-0001": ["MenGrowth-0001-000"],
            "MenGrowth-0002": ["MenGrowth-0002-000"],
        },
    )
    pts = discover_patients(root)
    assert pts == ["MenGrowth-0001", "MenGrowth-0002", "MenGrowth-0003"]


def test_discover_patients_skips_temp(tmp_path: Path) -> None:
    root = tmp_path / "in"
    root.mkdir()
    (root / "_temp_something").mkdir()
    (root / "MenGrowth-0001").mkdir()
    (root / "notapatient").mkdir()
    (root / "MenGrowth-0001-000").mkdir()  # study-shaped name at wrong level
    assert discover_patients(root) == ["MenGrowth-0001"]


def test_discover_studies_flags_missing_input(
    tmp_path: Path, cfg: SynthSegConfig
) -> None:
    root = tmp_path / "in"
    # One patient with t1n, one without
    _make_cohort(root, {"MenGrowth-0001": ["MenGrowth-0001-000"]})
    (root / "MenGrowth-0002" / "MenGrowth-0002-000").mkdir(parents=True)

    studies = discover_studies(root, cfg)
    assert len(studies) == 2

    by_pid = {s.patient_id: s for s in studies}
    assert by_pid["MenGrowth-0001"].has_input_modality is True
    assert by_pid["MenGrowth-0002"].has_input_modality is False


def test_studies_for_patient_filters(tmp_path: Path, cfg: SynthSegConfig) -> None:
    root = tmp_path / "in"
    _make_cohort(
        root,
        {
            "MenGrowth-0001": ["MenGrowth-0001-000", "MenGrowth-0001-001"],
            "MenGrowth-0002": ["MenGrowth-0002-000"],
        },
    )
    studies = studies_for_patient(root, cfg, "MenGrowth-0001")
    assert len(studies) == 2
    assert all(s.patient_id == "MenGrowth-0001" for s in studies)
    # Sort stability
    assert [s.study_id for s in studies] == [
        "MenGrowth-0001-000",
        "MenGrowth-0001-001",
    ]


def test_discover_studies_missing_root(tmp_path: Path, cfg: SynthSegConfig) -> None:
    with pytest.raises(FileNotFoundError):
        discover_studies(tmp_path / "nope", cfg)


def test_discover_studies_alternate_modality(tmp_path: Path) -> None:
    """A different ``input_modality`` flips which files are expected."""
    root = tmp_path / "in"
    (root / "MenGrowth-0001" / "MenGrowth-0001-000").mkdir(parents=True)
    _make_nifti(root / "MenGrowth-0001" / "MenGrowth-0001-000" / "t2w.nii.gz")

    cfg_t1 = SynthSegConfig(
        input_root=str(root),
        output_root=str(tmp_path / "out"),
        synthseg_repo=str(tmp_path / "SS"),
        input_modality="t1n",
    )
    cfg_t2 = SynthSegConfig(
        input_root=str(root),
        output_root=str(tmp_path / "out"),
        synthseg_repo=str(tmp_path / "SS"),
        input_modality="t2w",
    )
    assert discover_studies(root, cfg_t1)[0].has_input_modality is False
    assert discover_studies(root, cfg_t2)[0].has_input_modality is True
