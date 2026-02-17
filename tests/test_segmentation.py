"""Tests for the BraTS meningioma segmentation pipeline.

Tests study discovery, shape validation, shape correction, name mapping,
and post-processing using synthetic data (no container required).
"""

import json
import os
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from mengrowth.segmentation.config import (
    SegmentationConfig,
    SegmentationConfigError,
    load_segmentation_config,
)
from mengrowth.segmentation.prepare import (
    StudyInfo,
    correct_shape,
    discover_studies,
    prepare_brats_input,
    validate_study,
)
from mengrowth.segmentation.postprocess import (
    cleanup_temp_files,
    postprocess_outputs,
)


# ========================================================================
# Fixtures
# ========================================================================


def _make_nifti(path: Path, shape: tuple = (240, 240, 155)) -> None:
    """Create a minimal NIfTI file with given shape."""
    data = np.zeros(shape, dtype=np.float32)
    img = nib.Nifti1Image(data, np.eye(4))
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(path))


@pytest.fixture
def config() -> SegmentationConfig:
    """Default segmentation config for tests."""
    return SegmentationConfig(
        input_root="/tmp/test_seg_input",
        sif_path="/tmp/test.sif",
        expected_shape=(240, 240, 155),
        shape_tolerance=2,
    )


@pytest.fixture
def sample_dataset(tmp_path: Path) -> Path:
    """Create a synthetic preprocessed dataset.

    Structure:
        tmp_path/
          MenGrowth-0001/
            MenGrowth-0001-000/  (complete, correct shape)
            MenGrowth-0001-001/  (complete, wrong shape)
          MenGrowth-0002/
            MenGrowth-0002-000/  (incomplete, missing t2f)
    """
    # Patient 1, Study 000 — complete, correct shape
    for mod in ["t1c", "t1n", "t2w", "t2f"]:
        _make_nifti(
            tmp_path / "MenGrowth-0001" / "MenGrowth-0001-000" / f"{mod}.nii.gz",
            shape=(240, 240, 155),
        )

    # Patient 1, Study 001 — complete, slightly wrong shape
    for mod in ["t1c", "t1n", "t2w", "t2f"]:
        _make_nifti(
            tmp_path / "MenGrowth-0001" / "MenGrowth-0001-001" / f"{mod}.nii.gz",
            shape=(238, 240, 157),
        )

    # Patient 2, Study 000 — incomplete (missing t2f)
    for mod in ["t1c", "t1n", "t2w"]:
        _make_nifti(
            tmp_path / "MenGrowth-0002" / "MenGrowth-0002-000" / f"{mod}.nii.gz",
            shape=(240, 240, 155),
        )

    return tmp_path


# ========================================================================
# Config tests
# ========================================================================


class TestSegmentationConfig:
    def test_defaults(self) -> None:
        cfg = SegmentationConfig()
        assert cfg.modalities == ["t1c", "t1n", "t2w", "t2f"]
        assert cfg.expected_shape == (240, 240, 155)
        assert cfg.shape_tolerance == 2

    def test_list_to_tuple_conversion(self) -> None:
        cfg = SegmentationConfig(expected_shape=[240, 240, 155])
        assert isinstance(cfg.expected_shape, tuple)

    def test_invalid_shape_length(self) -> None:
        with pytest.raises(SegmentationConfigError, match="3 elements"):
            SegmentationConfig(expected_shape=(240, 240))

    def test_negative_tolerance(self) -> None:
        with pytest.raises(SegmentationConfigError, match="shape_tolerance"):
            SegmentationConfig(shape_tolerance=-1)

    def test_empty_modalities(self) -> None:
        with pytest.raises(SegmentationConfigError, match="modalities"):
            SegmentationConfig(modalities=[])

    def test_load_from_yaml(self, tmp_path: Path) -> None:
        yaml_content = """
segmentation:
  input_root: "/data/preprocessed"
  sif_path: "/images/test.sif"
  expected_shape: [240, 240, 155]
  shape_tolerance: 3
"""
        yaml_path = tmp_path / "test_config.yaml"
        yaml_path.write_text(yaml_content)

        cfg = load_segmentation_config(yaml_path)
        assert cfg.input_root == "/data/preprocessed"
        assert cfg.shape_tolerance == 3
        assert cfg.expected_shape == (240, 240, 155)

    def test_load_missing_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_segmentation_config("/nonexistent/path.yaml")

    def test_load_missing_key(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "bad.yaml"
        yaml_path.write_text("other_key: value\n")
        with pytest.raises(SegmentationConfigError, match="segmentation"):
            load_segmentation_config(yaml_path)


# ========================================================================
# Discovery tests
# ========================================================================


class TestDiscoverStudies:
    def test_discovers_all(
        self, sample_dataset: Path, config: SegmentationConfig
    ) -> None:
        config.input_root = str(sample_dataset)
        studies = discover_studies(sample_dataset, config)
        assert len(studies) == 3

    def test_completeness_flag(
        self, sample_dataset: Path, config: SegmentationConfig
    ) -> None:
        config.input_root = str(sample_dataset)
        studies = discover_studies(sample_dataset, config)
        study_map = {s.study_id: s for s in studies}

        assert study_map["MenGrowth-0001-000"].is_complete is True
        assert study_map["MenGrowth-0001-001"].is_complete is True
        assert study_map["MenGrowth-0002-000"].is_complete is False

    def test_shape_recording(
        self, sample_dataset: Path, config: SegmentationConfig
    ) -> None:
        config.input_root = str(sample_dataset)
        studies = discover_studies(sample_dataset, config)
        study_map = {s.study_id: s for s in studies}

        shapes = study_map["MenGrowth-0001-001"].shape_per_modality
        assert shapes["t1c"] == (238, 240, 157)

    def test_patient_filter(
        self, sample_dataset: Path, config: SegmentationConfig
    ) -> None:
        config.input_root = str(sample_dataset)
        studies = discover_studies(
            sample_dataset, config, patient_filter="MenGrowth-0001"
        )
        assert len(studies) == 2
        assert all(s.patient_id == "MenGrowth-0001" for s in studies)

    def test_ignores_temp_dirs(
        self, sample_dataset: Path, config: SegmentationConfig
    ) -> None:
        temp_dir = sample_dataset / "_temp_processing"
        temp_dir.mkdir()
        (temp_dir / "junk.txt").write_text("temp")

        config.input_root = str(sample_dataset)
        studies = discover_studies(sample_dataset, config)
        assert not any(s.patient_id == "_temp_processing" for s in studies)


# ========================================================================
# Validation tests
# ========================================================================


class TestValidateStudy:
    def test_valid_study(self, config: SegmentationConfig) -> None:
        study = StudyInfo(
            patient_id="MenGrowth-0001",
            study_id="MenGrowth-0001-000",
            available_modalities=["t1c", "t1n", "t2w", "t2f"],
            is_complete=True,
            shape_per_modality={
                "t1c": (240, 240, 155),
                "t1n": (240, 240, 155),
                "t2w": (240, 240, 155),
                "t2f": (240, 240, 155),
            },
        )
        is_valid, issues = validate_study(study, config)
        assert is_valid is True
        assert issues == []

    def test_within_tolerance(self, config: SegmentationConfig) -> None:
        study = StudyInfo(
            patient_id="P1",
            study_id="S1",
            available_modalities=["t1c", "t1n", "t2w", "t2f"],
            is_complete=True,
            shape_per_modality={
                "t1c": (238, 240, 155),
                "t1n": (240, 242, 155),
                "t2w": (240, 240, 153),
                "t2f": (240, 240, 155),
            },
        )
        is_valid, issues = validate_study(study, config)
        assert is_valid is True

    def test_out_of_tolerance(self, config: SegmentationConfig) -> None:
        study = StudyInfo(
            patient_id="P1",
            study_id="S1",
            available_modalities=["t1c", "t1n", "t2w", "t2f"],
            is_complete=True,
            shape_per_modality={
                "t1c": (230, 240, 155),  # 10 off
                "t1n": (240, 240, 155),
                "t2w": (240, 240, 155),
                "t2f": (240, 240, 155),
            },
        )
        is_valid, issues = validate_study(study, config)
        assert is_valid is False
        assert any("t1c" in i and "axis 0" in i for i in issues)

    def test_missing_modality(self, config: SegmentationConfig) -> None:
        study = StudyInfo(
            patient_id="P1",
            study_id="S1",
            available_modalities=["t1c", "t1n", "t2w"],
            is_complete=False,
            shape_per_modality={
                "t1c": (240, 240, 155),
                "t1n": (240, 240, 155),
                "t2w": (240, 240, 155),
            },
        )
        is_valid, issues = validate_study(study, config)
        assert is_valid is False
        assert any("Missing modalities" in i for i in issues)


# ========================================================================
# Shape correction tests
# ========================================================================


class TestCorrectShape:
    def test_pad_to_target(self, tmp_path: Path) -> None:
        src = tmp_path / "input.nii.gz"
        dst = tmp_path / "output.nii.gz"
        _make_nifti(src, shape=(236, 240, 153))

        result = correct_shape(src, dst, (240, 240, 155))

        assert dst.exists()
        img = nib.load(str(dst))
        assert img.shape == (240, 240, 155)
        assert result["original_shape"] == [236, 240, 153]

    def test_crop_to_target(self, tmp_path: Path) -> None:
        src = tmp_path / "input.nii.gz"
        dst = tmp_path / "output.nii.gz"
        _make_nifti(src, shape=(244, 240, 157))

        result = correct_shape(src, dst, (240, 240, 155))

        img = nib.load(str(dst))
        assert img.shape == (240, 240, 155)
        assert result["original_shape"] == [244, 240, 157]

    def test_mixed_pad_crop(self, tmp_path: Path) -> None:
        src = tmp_path / "input.nii.gz"
        dst = tmp_path / "output.nii.gz"
        _make_nifti(src, shape=(238, 242, 155))

        result = correct_shape(src, dst, (240, 240, 155))

        img = nib.load(str(dst))
        assert img.shape == (240, 240, 155)

    def test_no_change_needed(self, tmp_path: Path) -> None:
        src = tmp_path / "input.nii.gz"
        dst = tmp_path / "output.nii.gz"
        _make_nifti(src, shape=(240, 240, 155))

        result = correct_shape(src, dst, (240, 240, 155))

        img = nib.load(str(dst))
        assert img.shape == (240, 240, 155)
        assert result["pad_before"] == [0, 0, 0]
        assert result["crop_before"] == [0, 0, 0]


# ========================================================================
# BraTS input preparation tests
# ========================================================================


class TestPrepareBratsInput:
    def test_creates_symlinks(
        self, sample_dataset: Path, config: SegmentationConfig
    ) -> None:
        config.input_root = str(sample_dataset)
        studies = discover_studies(sample_dataset, config)
        complete = [s for s in studies if s.is_complete]

        with tempfile.TemporaryDirectory() as work_base:
            config.work_dir = os.path.join(work_base, "work")
            work_dir, name_map = prepare_brats_input(complete, config)

            # Check directory structure
            input_dir = work_dir / "input"
            assert input_dir.exists()

            # Check name mapping
            assert len(name_map) == 2  # Two complete studies

            # Check symlinks exist for first entry
            first_brats = list(name_map.keys())[0]
            subj_dir = input_dir / first_brats
            assert subj_dir.exists()

            for mod in config.modalities:
                link = subj_dir / f"{first_brats}-{mod}.nii.gz"
                assert link.exists()

    def test_name_map_json_saved(
        self, sample_dataset: Path, config: SegmentationConfig
    ) -> None:
        config.input_root = str(sample_dataset)
        studies = discover_studies(sample_dataset, config)
        complete = [s for s in studies if s.is_complete]

        with tempfile.TemporaryDirectory() as work_base:
            config.work_dir = os.path.join(work_base, "work")
            work_dir, name_map = prepare_brats_input(complete, config)

            map_file = work_dir / "name_map.json"
            assert map_file.exists()

            with open(map_file) as f:
                saved = json.load(f)
            assert len(saved) == len(name_map)

            # Verify structure
            for brats_name, info in saved.items():
                assert "patient_id" in info
                assert "study_id" in info
                assert "study_path" in info
                assert "shape_corrected" in info

    def test_shape_correction_applied(
        self, sample_dataset: Path, config: SegmentationConfig
    ) -> None:
        config.input_root = str(sample_dataset)
        config.shape_tolerance = 0  # Force strict matching
        studies = discover_studies(sample_dataset, config)

        # MenGrowth-0001-001 has shape (238, 240, 157), should be corrected
        wrong_shape = [s for s in studies if s.study_id == "MenGrowth-0001-001"]
        assert len(wrong_shape) == 1

        with tempfile.TemporaryDirectory() as work_base:
            config.work_dir = os.path.join(work_base, "work")
            work_dir, name_map = prepare_brats_input(wrong_shape, config)

            # Check that the entry is marked as shape-corrected
            entry = list(name_map.values())[0]
            assert entry["shape_corrected"] is True

            # Check that files are actual files (not symlinks) since correction was needed
            brats_name = list(name_map.keys())[0]
            for mod in config.modalities:
                path = work_dir / "input" / brats_name / f"{brats_name}-{mod}.nii.gz"
                assert path.exists()
                assert not path.is_symlink()


# ========================================================================
# Post-processing tests
# ========================================================================


class TestPostprocessOutputs:
    def test_remap_outputs(
        self, sample_dataset: Path, config: SegmentationConfig
    ) -> None:
        config.input_root = str(sample_dataset)
        studies = discover_studies(sample_dataset, config)
        complete = [s for s in studies if s.is_complete]

        with tempfile.TemporaryDirectory() as work_base:
            config.work_dir = os.path.join(work_base, "work")
            work_dir, name_map = prepare_brats_input(complete, config)

            # Simulate BraTS output: create fake segmentation files
            output_dir = work_dir / "output"
            for brats_name in name_map:
                seg_path = output_dir / f"{brats_name}.nii.gz"
                _make_nifti(seg_path, shape=(240, 240, 155))

            # Run postprocess
            results = postprocess_outputs(work_dir, config)

            assert len(results) == len(name_map)
            assert all(r.success for r in results)

            # Check files were copied
            for r in results:
                assert Path(r.output_path).exists()

    def test_missing_output_reported(
        self, tmp_path: Path, config: SegmentationConfig
    ) -> None:
        work_dir = tmp_path / "work"
        (work_dir / "output").mkdir(parents=True)

        # Create name_map.json referencing a study with no output
        name_map = {
            "BraTS-MEN-00000-000": {
                "patient_id": "MenGrowth-0001",
                "study_id": "MenGrowth-0001-000",
                "study_path": str(tmp_path / "MenGrowth-0001" / "MenGrowth-0001-000"),
                "shape_corrected": False,
                "original_shape": {},
            }
        }
        with open(work_dir / "name_map.json", "w") as f:
            json.dump(name_map, f)

        results = postprocess_outputs(work_dir, config)
        assert len(results) == 1
        assert results[0].success is False
        assert "No output found" in results[0].error


# ========================================================================
# Cleanup tests
# ========================================================================


class TestCleanupTempFiles:
    def test_removes_temp_files(self, tmp_path: Path) -> None:
        # Create some temp files
        (tmp_path / "_temp_processing").mkdir()
        (tmp_path / "_temp_processing" / "file.txt").write_text("temp")
        (tmp_path / "real_file.nii.gz").write_text("keep")
        (tmp_path / "file.tmp.nii.gz").write_text("temp2")

        count = cleanup_temp_files(tmp_path)

        assert count >= 2
        assert (tmp_path / "real_file.nii.gz").exists()
        assert not (tmp_path / "file.tmp.nii.gz").exists()
