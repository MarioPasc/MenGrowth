"""Tests for :mod:`mengrowth.synthseg.runner`.

The SynthSeg subprocess is mocked; the tests exercise orchestration logic:
temp-file-then-rename, geometry validation, idempotent skip, per-study
failure isolation, and patient-level exit-code contract.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List

import nibabel as nib
import numpy as np
import pytest

from mengrowth.synthseg.config import SynthSegConfig
from mengrowth.synthseg.discovery import StudyInfo
from mengrowth.synthseg.runner import (
    PatientResult,
    StudyResult,
    run_patient,
    run_study,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SHAPE = (8, 8, 4)
AFFINE = np.diag([1.0, 1.0, 1.0, 1.0])


def _write_nifti(path: Path, shape=SHAPE, affine=AFFINE, dtype=np.float32) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.zeros(shape, dtype=dtype)
    nib.save(nib.Nifti1Image(data, affine), str(path))


def _make_study(tmp_path: Path, pid="MenGrowth-0001", sid="MenGrowth-0001-000") -> StudyInfo:
    study_dir = tmp_path / "in" / pid / sid
    input_path = study_dir / "t1n.nii.gz"
    _write_nifti(input_path)
    return StudyInfo(
        patient_id=pid,
        study_id=sid,
        study_path=study_dir,
        has_input_modality=True,
        input_path=input_path,
    )


def _make_config(tmp_path: Path, **overrides) -> SynthSegConfig:
    base = dict(
        input_root=str(tmp_path / "in"),
        output_root=str(tmp_path / "out"),
        synthseg_repo=str(tmp_path / "SS"),
        conda_env_path="",
        python_bin="",
    )
    base.update(overrides)
    return SynthSegConfig(**base)


@pytest.fixture
def synthseg_script(tmp_path: Path) -> Path:
    """Create a stub ``SynthSeg_predict.py`` location so check_command_available passes."""
    script = tmp_path / "SS" / "scripts" / "commands" / "SynthSeg_predict.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("# stub\n")
    return script


def _fake_synthseg_success(argv: List[str], input_shape=SHAPE, affine=AFFINE, qc=0.8):
    """Simulate a successful SynthSeg run: write parc + vol + qc tmp files."""
    def _run(*args, **kwargs):
        cmd = args[0]
        out_parc = Path(cmd[cmd.index("--o") + 1])
        out_vol = Path(cmd[cmd.index("--vol") + 1])
        out_qc = Path(cmd[cmd.index("--qc") + 1])
        _write_nifti(out_parc, shape=input_shape, affine=affine,
                     dtype=np.int32)
        out_vol.write_text(
            "subject,Left-Thalamus,Right-Thalamus\n"
            f"{cmd[cmd.index('--i') + 1]},7000.0,7100.0\n"
        )
        out_qc.write_text(f"subject,qc\n{cmd[cmd.index('--i') + 1]},{qc}\n")
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")
    return _run


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_run_study_success_writes_final_outputs(
    tmp_path: Path, synthseg_script: Path, monkeypatch
) -> None:
    cfg = _make_config(tmp_path)
    study = _make_study(tmp_path)

    monkeypatch.setattr(subprocess, "run", _fake_synthseg_success([]))
    result = run_study(cfg, study)

    assert result.status == "success"
    assert result.qc_score == pytest.approx(0.8)
    out_dir = Path(cfg.output_root) / study.patient_id / study.study_id
    assert (out_dir / cfg.parcellation_filename).exists()
    assert (out_dir / cfg.volumes_filename).exists()
    assert (out_dir / cfg.qc_filename).exists()
    # No stray temp files
    assert not any(p.name.endswith(".tmp.nii.gz") for p in out_dir.iterdir())
    assert not any(p.name.endswith(".tmp") for p in out_dir.iterdir())


def test_run_study_skipped_missing_input(
    tmp_path: Path, synthseg_script: Path
) -> None:
    cfg = _make_config(tmp_path)
    study = StudyInfo(
        patient_id="MenGrowth-0099",
        study_id="MenGrowth-0099-000",
        study_path=tmp_path / "in" / "MenGrowth-0099" / "MenGrowth-0099-000",
        has_input_modality=False,
        input_path=(
            tmp_path / "in" / "MenGrowth-0099" / "MenGrowth-0099-000" / "t1n.nii.gz"
        ),
    )
    result = run_study(cfg, study)
    assert result.status == "skipped_missing_input"


def test_run_study_idempotent_skip(
    tmp_path: Path, synthseg_script: Path, monkeypatch
) -> None:
    """Second call on the same study must not re-run SynthSeg."""
    cfg = _make_config(tmp_path)
    study = _make_study(tmp_path)

    monkeypatch.setattr(subprocess, "run", _fake_synthseg_success([]))
    run_study(cfg, study)  # first call produces outputs

    call_count = {"n": 0}

    def _should_not_be_called(*args, **kwargs):
        call_count["n"] += 1
        raise AssertionError("subprocess.run should not be invoked on skip")

    monkeypatch.setattr(subprocess, "run", _should_not_be_called)
    result = run_study(cfg, study)
    assert result.status == "skipped_already_done"
    assert call_count["n"] == 0
    assert result.qc_score == pytest.approx(0.8)


def test_run_study_nonzero_exit(
    tmp_path: Path, synthseg_script: Path, monkeypatch
) -> None:
    cfg = _make_config(tmp_path)
    study = _make_study(tmp_path)

    def _fail(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], returncode=2, stdout="", stderr="synthseg broke")

    monkeypatch.setattr(subprocess, "run", _fail)
    result = run_study(cfg, study)
    assert result.status == "failed"
    assert "rc=2" in (result.error or "")

    # Final outputs must NOT be committed on failure
    out_dir = Path(cfg.output_root) / study.patient_id / study.study_id
    final = out_dir / cfg.parcellation_filename
    assert not final.exists()


def test_run_study_timeout(
    tmp_path: Path, synthseg_script: Path, monkeypatch
) -> None:
    cfg = _make_config(tmp_path, timeout_seconds=30)
    study = _make_study(tmp_path)

    def _timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=30)

    monkeypatch.setattr(subprocess, "run", _timeout)
    result = run_study(cfg, study)
    assert result.status == "failed"
    assert "timed out" in (result.error or "").lower()


def test_run_study_geometry_mismatch(
    tmp_path: Path, synthseg_script: Path, monkeypatch
) -> None:
    """If SynthSeg writes a parcellation with a wrong shape, we must refuse to commit."""
    cfg = _make_config(tmp_path)
    study = _make_study(tmp_path)

    # Produce a parc with a DIFFERENT shape than the input
    monkeypatch.setattr(
        subprocess, "run", _fake_synthseg_success([], input_shape=(16, 16, 8))
    )
    result = run_study(cfg, study)
    assert result.status == "failed"
    assert "shape" in (result.error or "").lower()

    out_dir = Path(cfg.output_root) / study.patient_id / study.study_id
    final = out_dir / cfg.parcellation_filename
    assert not final.exists()


def test_run_study_qc_out_of_range(
    tmp_path: Path, synthseg_script: Path, monkeypatch
) -> None:
    cfg = _make_config(tmp_path)
    study = _make_study(tmp_path)
    monkeypatch.setattr(subprocess, "run", _fake_synthseg_success([], qc=1.5))
    result = run_study(cfg, study)
    assert result.status == "failed"
    assert "outside" in (result.error or "").lower()


# ---------------------------------------------------------------------------
# run_patient — exit-code contract
# ---------------------------------------------------------------------------


def _make_patient_cohort(
    tmp_path: Path,
    pid: str,
    study_ids: List[str],
    missing_input: List[str] = None,
) -> None:
    missing_input = missing_input or []
    for sid in study_ids:
        study_dir = tmp_path / "in" / pid / sid
        study_dir.mkdir(parents=True)
        if sid not in missing_input:
            _write_nifti(study_dir / "t1n.nii.gz")


def test_run_patient_all_success(
    tmp_path: Path, synthseg_script: Path, monkeypatch
) -> None:
    _make_patient_cohort(
        tmp_path,
        "MenGrowth-0001",
        ["MenGrowth-0001-000", "MenGrowth-0001-001"],
    )
    cfg = _make_config(tmp_path)
    monkeypatch.setattr(subprocess, "run", _fake_synthseg_success([]))

    result = run_patient(cfg, "MenGrowth-0001")
    assert result.n_success == 2
    assert result.n_failed == 0
    assert result.exit_code() == 0


def test_run_patient_partial_failure(
    tmp_path: Path, synthseg_script: Path, monkeypatch
) -> None:
    _make_patient_cohort(
        tmp_path,
        "MenGrowth-0001",
        ["MenGrowth-0001-000", "MenGrowth-0001-001"],
    )
    cfg = _make_config(tmp_path)

    calls = {"n": 0}

    def _mixed(*args, **kwargs):
        calls["n"] += 1
        # First call succeeds, second fails — verifies the loop continues
        # past a study-level failure.
        if calls["n"] == 1:
            return _fake_synthseg_success([])(*args, **kwargs)
        return subprocess.CompletedProcess(args[0], 5, "", "boom")

    monkeypatch.setattr(subprocess, "run", _mixed)

    result = run_patient(cfg, "MenGrowth-0001")
    assert result.n_success == 1
    assert result.n_failed == 1
    assert result.exit_code() == 2


def test_run_patient_all_missing_input(
    tmp_path: Path, synthseg_script: Path
) -> None:
    _make_patient_cohort(
        tmp_path,
        "MenGrowth-0001",
        ["MenGrowth-0001-000"],
        missing_input=["MenGrowth-0001-000"],
    )
    cfg = _make_config(tmp_path)
    result = run_patient(cfg, "MenGrowth-0001")
    assert result.n_skipped_missing == 1
    assert result.exit_code() == 3


def test_run_patient_no_studies(
    tmp_path: Path, synthseg_script: Path
) -> None:
    (tmp_path / "in").mkdir()
    cfg = _make_config(tmp_path)
    result = run_patient(cfg, "MenGrowth-9999")
    assert result.exit_code() == 3
    assert result.studies == []
