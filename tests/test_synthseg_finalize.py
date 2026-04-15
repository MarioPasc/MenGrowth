"""Tests for :mod:`mengrowth.synthseg.finalize`.

Uses a tiny synthetic study dir to exercise:

* Atomic ``.tmp → final`` rename on a clean happy path.
* Rejection when parc geometry differs from the reference T1n.
* Rejection when the QC value is outside ``[0, 1]``.
* ``already_final`` when outputs are already in place.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from mengrowth.synthseg.config import SynthSegConfig
from mengrowth.synthseg.finalize import (
    finalize_one_study,
    finalize_synthseg_outputs,
)


def _make_nifti(path: Path, shape=(8, 8, 8), affine=np.eye(4)) -> None:
    data = np.zeros(shape, dtype=np.int16)
    nib.save(nib.Nifti1Image(data, affine), str(path))


def _write_csv(path: Path, header: str, row: str) -> None:
    path.write_text(f"{header}\n{row}\n")


def _make_study(
    root_in: Path, root_out: Path, pid="MenGrowth-0001", sid="MenGrowth-0001-000"
):
    in_dir = root_in / pid / sid
    out_dir = root_out / pid / sid
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    _make_nifti(in_dir / "t1n.nii.gz")
    return in_dir, out_dir


def _cfg_for(root_in: Path, root_out: Path) -> SynthSegConfig:
    return SynthSegConfig(
        input_root=str(root_in),
        output_root=str(root_out),
        synthseg_repo="unused",
        input_modality="t1n",
        parcellation_filename="synthseg_parc.nii.gz",
        volumes_filename="synthseg_volumes.csv",
        qc_filename="synthseg_qc.csv",
    )


def test_finalize_happy_path_renames_tmp(tmp_path: Path) -> None:
    root_in = tmp_path / "in"
    root_out = tmp_path / "out"
    in_dir, out_dir = _make_study(root_in, root_out)
    cfg = _cfg_for(root_in, root_out)

    # matching-geometry parc as .tmp, valid CSVs as .tmp.csv (SynthSeg variant)
    _make_nifti(out_dir / "synthseg_parc.nii.tmp.nii.gz")
    _write_csv(
        out_dir / "synthseg_qc.csv.tmp.csv",
        "subject,qc",
        "/some/t1n.nii.gz,0.73",
    )
    _write_csv(
        out_dir / "synthseg_volumes.csv.tmp.csv",
        "subject,left thalamus,right thalamus",
        "/some/t1n.nii.gz,9000.0,9100.0",
    )

    result = finalize_one_study(out_dir, in_dir / "t1n.nii.gz", cfg)
    assert result.status == "finalized", result.error
    assert (out_dir / "synthseg_parc.nii.gz").exists()
    assert (out_dir / "synthseg_qc.csv").exists()
    assert (out_dir / "synthseg_volumes.csv").exists()
    # Originals are gone after os.replace.
    assert not (out_dir / "synthseg_parc.nii.tmp.nii.gz").exists()


def test_finalize_already_final_no_tmp(tmp_path: Path) -> None:
    root_in = tmp_path / "in"
    root_out = tmp_path / "out"
    in_dir, out_dir = _make_study(root_in, root_out)
    cfg = _cfg_for(root_in, root_out)

    _make_nifti(out_dir / "synthseg_parc.nii.gz")
    _write_csv(out_dir / "synthseg_qc.csv", "subject,qc", "/s/t1n.nii.gz,0.8")
    _write_csv(
        out_dir / "synthseg_volumes.csv",
        "subject,left thalamus",
        "/s/t1n.nii.gz,9000.0",
    )

    result = finalize_one_study(out_dir, in_dir / "t1n.nii.gz", cfg)
    assert result.status == "already_final"


def test_finalize_rejects_geometry_mismatch(tmp_path: Path) -> None:
    root_in = tmp_path / "in"
    root_out = tmp_path / "out"
    in_dir, out_dir = _make_study(root_in, root_out)
    cfg = _cfg_for(root_in, root_out)

    # Mismatched shape vs. T1n (8,8,8)
    _make_nifti(out_dir / "synthseg_parc.nii.tmp.nii.gz", shape=(16, 8, 8))
    _write_csv(
        out_dir / "synthseg_qc.csv.tmp.csv", "subject,qc", "/s/t1n.nii.gz,0.7"
    )
    _write_csv(
        out_dir / "synthseg_volumes.csv.tmp.csv",
        "subject,left thalamus",
        "/s/t1n.nii.gz,9000.0",
    )

    result = finalize_one_study(out_dir, in_dir / "t1n.nii.gz", cfg)
    assert result.status == "failed"
    assert "shape" in (result.error or "")


def test_finalize_rejects_qc_out_of_range(tmp_path: Path) -> None:
    root_in = tmp_path / "in"
    root_out = tmp_path / "out"
    in_dir, out_dir = _make_study(root_in, root_out)
    cfg = _cfg_for(root_in, root_out)

    _make_nifti(out_dir / "synthseg_parc.nii.tmp.nii.gz")
    _write_csv(
        out_dir / "synthseg_qc.csv.tmp.csv", "subject,qc", "/s/t1n.nii.gz,1.5"
    )
    _write_csv(
        out_dir / "synthseg_volumes.csv.tmp.csv",
        "subject,left thalamus",
        "/s/t1n.nii.gz,9000.0",
    )

    result = finalize_one_study(out_dir, in_dir / "t1n.nii.gz", cfg)
    assert result.status == "failed"
    assert "qc" in (result.error or "").lower()


def test_finalize_report_exit_code(tmp_path: Path) -> None:
    root_in = tmp_path / "in"
    root_out = tmp_path / "out"
    _make_study(root_in, root_out)  # no outputs written
    cfg = _cfg_for(root_in, root_out)

    report = finalize_synthseg_outputs(cfg)
    assert report.exit_code() == 0  # no_outputs ≠ failure
    assert report.n_no_outputs == 1
