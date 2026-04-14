"""Tests for :mod:`mengrowth.synthseg.primitives`."""

from __future__ import annotations

from pathlib import Path

import pytest

from mengrowth.synthseg.primitives import (
    SYNTHSEG_LABEL_MAP,
    build_command,
    check_command_available,
    parse_qc_csv,
    parse_qc_value,
    parse_volumes_csv,
    parse_volumes_row,
)


# ---------------------------------------------------------------------------
# Label map
# ---------------------------------------------------------------------------


def test_label_map_has_30_regions() -> None:
    assert len(SYNTHSEG_LABEL_MAP) == 30


def test_label_map_lateralized_pairs() -> None:
    # Every left region has a right counterpart, except midline structures.
    left = {k for k in SYNTHSEG_LABEL_MAP if k.startswith("Left-")}
    right = {k for k in SYNTHSEG_LABEL_MAP if k.startswith("Right-")}
    assert len(left) == len(right)


# ---------------------------------------------------------------------------
# build_command
# ---------------------------------------------------------------------------


def _cmd_args(**overrides):
    base = dict(
        synthseg_repo=Path("/tools/SynthSeg"),
        input_nii=Path("/in/t1n.nii.gz"),
        output_parc=Path("/out/parc.nii.gz"),
        vol_csv=Path("/out/vol.csv"),
        qc_csv=Path("/out/qc.csv"),
        robust=True,
        threads=2,
        use_gpu=True,
        python_bin=Path("/env/bin/python"),
    )
    base.update(overrides)
    return base


def test_build_command_has_required_flags() -> None:
    cmd = build_command(**_cmd_args())
    assert cmd[0] == "/env/bin/python"
    assert cmd[1].endswith("scripts/commands/SynthSeg_predict.py")
    assert "--i" in cmd
    assert cmd[cmd.index("--i") + 1] == "/in/t1n.nii.gz"
    assert "--o" in cmd
    assert "--vol" in cmd
    assert "--qc" in cmd


def test_build_command_robust_flag() -> None:
    assert "--robust" in build_command(**_cmd_args(robust=True))
    assert "--robust" not in build_command(**_cmd_args(robust=False))


def test_build_command_threads_flag() -> None:
    cmd = build_command(**_cmd_args(threads=4))
    idx = cmd.index("--threads")
    assert cmd[idx + 1] == "4"


def test_build_command_cpu_flag() -> None:
    assert "--cpu" not in build_command(**_cmd_args(use_gpu=True))
    assert "--cpu" in build_command(**_cmd_args(use_gpu=False))


def test_build_command_posteriors() -> None:
    cmd_no = build_command(**_cmd_args(posteriors_path=None))
    cmd_yes = build_command(**_cmd_args(posteriors_path=Path("/out/post.nii.gz")))
    assert "--post" not in cmd_no
    assert "--post" in cmd_yes
    assert cmd_yes[cmd_yes.index("--post") + 1] == "/out/post.nii.gz"


# ---------------------------------------------------------------------------
# check_command_available
# ---------------------------------------------------------------------------


def test_check_command_available_absolute_python(tmp_path: Path) -> None:
    py = tmp_path / "python"
    py.write_text("#!/bin/sh\nexit 0\n")
    py.chmod(0o755)
    script = tmp_path / "SynthSeg_predict.py"
    script.write_text("# stub\n")
    assert check_command_available([str(py), str(script), "--i", "a"])


def test_check_command_available_missing_script(tmp_path: Path) -> None:
    py = tmp_path / "python"
    py.write_text("#!/bin/sh\nexit 0\n")
    py.chmod(0o755)
    assert not check_command_available([str(py), str(tmp_path / "nope.py")])


def test_check_command_available_missing_python() -> None:
    assert not check_command_available(["/nonexistent/python", "/also/nope"])


# ---------------------------------------------------------------------------
# CSV parsers — real SynthSeg output shapes
# ---------------------------------------------------------------------------


SINGLE_QC_CSV = "subject,qc\n/data/t1n.nii.gz,0.87\n"

ROBUST_QC_CSV = (
    "subject,general white matter,general grey matter,cerebrospinal fluid\n"
    "/data/t1n.nii.gz,0.91,0.85,0.77\n"
)

VOL_CSV = (
    "subject,Left-Cerebral-White-Matter,Right-Cerebral-White-Matter,Left-Thalamus\n"
    "/data/t1n.nii.gz,250000.0,248000.0,7200.5\n"
)


def test_parse_qc_csv_single_column(tmp_path: Path) -> None:
    f = tmp_path / "qc.csv"
    f.write_text(SINGLE_QC_CSV)
    scores = parse_qc_csv(f)
    assert set(scores.values()) == {0.87}


def test_parse_qc_csv_robust_multi_column(tmp_path: Path) -> None:
    f = tmp_path / "qc.csv"
    f.write_text(ROBUST_QC_CSV)
    scores = parse_qc_csv(f)
    # Mean of 0.91, 0.85, 0.77 = 0.8433...
    val = next(iter(scores.values()))
    assert abs(val - 0.8433333333) < 1e-5


def test_parse_qc_value_single_row(tmp_path: Path) -> None:
    f = tmp_path / "qc.csv"
    f.write_text(SINGLE_QC_CSV)
    assert parse_qc_value(f) == 0.87


def test_parse_qc_value_empty(tmp_path: Path) -> None:
    f = tmp_path / "qc.csv"
    f.write_text("subject,qc\n")
    with pytest.raises(ValueError):
        parse_qc_value(f)


def test_parse_volumes_csv(tmp_path: Path) -> None:
    f = tmp_path / "vol.csv"
    f.write_text(VOL_CSV)
    vols = parse_volumes_csv(f)
    subject_vols = next(iter(vols.values()))
    assert subject_vols["Left-Cerebral-White-Matter"] == 250000.0
    assert subject_vols["Left-Thalamus"] == 7200.5


def test_parse_volumes_row(tmp_path: Path) -> None:
    f = tmp_path / "vol.csv"
    f.write_text(VOL_CSV)
    vols = parse_volumes_row(f)
    assert vols["Right-Cerebral-White-Matter"] == 248000.0
