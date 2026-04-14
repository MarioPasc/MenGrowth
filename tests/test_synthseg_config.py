"""Tests for :mod:`mengrowth.synthseg.config`."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from mengrowth.synthseg.config import SynthSegConfig, load_synthseg_config
from mengrowth.synthseg.exceptions import SynthSegConfigError


def _minimal_kwargs(**overrides) -> dict:
    base = {
        "input_root": "/tmp/in",
        "output_root": "/tmp/out",
        "synthseg_repo": "/tmp/SynthSeg",
    }
    base.update(overrides)
    return base


def test_defaults_roundtrip() -> None:
    cfg = SynthSegConfig(**_minimal_kwargs())
    assert cfg.input_modality == "t1n"
    assert cfg.robust is True
    assert cfg.threads == 2
    assert cfg.use_gpu is True
    assert cfg.parcellation_filename == "synthseg_parc.nii.gz"


def test_rejects_empty_input_root() -> None:
    with pytest.raises(SynthSegConfigError):
        SynthSegConfig(**_minimal_kwargs(input_root=""))


def test_rejects_empty_output_root() -> None:
    with pytest.raises(SynthSegConfigError):
        SynthSegConfig(**_minimal_kwargs(output_root=""))


def test_rejects_empty_synthseg_repo() -> None:
    with pytest.raises(SynthSegConfigError):
        SynthSegConfig(**_minimal_kwargs(synthseg_repo=""))


def test_rejects_robust_false() -> None:
    # Meningioma cohort is OOD without --robust.
    with pytest.raises(SynthSegConfigError):
        SynthSegConfig(**_minimal_kwargs(robust=False))


def test_rejects_bad_threads() -> None:
    with pytest.raises(SynthSegConfigError):
        SynthSegConfig(**_minimal_kwargs(threads=0))


def test_rejects_bad_timeout() -> None:
    with pytest.raises(SynthSegConfigError):
        SynthSegConfig(**_minimal_kwargs(timeout_seconds=10))


def test_rejects_empty_filenames() -> None:
    with pytest.raises(SynthSegConfigError):
        SynthSegConfig(**_minimal_kwargs(parcellation_filename=""))


def test_load_from_yaml(tmp_path: Path) -> None:
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "synthseg": {
                    "input_root": "/data/preproc",
                    "output_root": "/data/synthseg",
                    "synthseg_repo": "/tools/SynthSeg",
                    "threads": 4,
                    "use_gpu": False,
                }
            }
        )
    )
    cfg = load_synthseg_config(yaml_path)
    assert cfg.input_root == "/data/preproc"
    assert cfg.threads == 4
    assert cfg.use_gpu is False


def test_load_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_synthseg_config(tmp_path / "nope.yaml")


def test_load_missing_synthseg_key(tmp_path: Path) -> None:
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(yaml.safe_dump({"other": {}}))
    with pytest.raises(SynthSegConfigError):
        load_synthseg_config(yaml_path)


def test_load_empty_file(tmp_path: Path) -> None:
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text("")
    with pytest.raises(SynthSegConfigError):
        load_synthseg_config(yaml_path)


def test_load_unknown_field(tmp_path: Path) -> None:
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "synthseg": {
                    "input_root": "/a",
                    "output_root": "/b",
                    "synthseg_repo": "/c",
                    "unknown_field": 42,
                }
            }
        )
    )
    with pytest.raises(SynthSegConfigError):
        load_synthseg_config(yaml_path)
