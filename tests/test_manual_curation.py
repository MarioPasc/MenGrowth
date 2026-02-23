"""Tests for manual curation of the curated dataset."""

import csv
import textwrap
from pathlib import Path

import pytest

from mengrowth.preprocessing.utils.manual_curation import (
    MANUAL_CURATION_STAGE,
    ManualCurationConfig,
    ManualExclusion,
    apply_manual_curation,
    load_manual_curation_config,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_dataset(tmp_path: Path) -> Path:
    """Create a minimal curated dataset structure for testing.

    Layout:
        MenGrowth-2025/
            MenGrowth-0001/
                MenGrowth-0001-000/ (t1c.nrrd, t1n.nrrd)
                MenGrowth-0001-001/ (t1c.nrrd, t1n.nrrd)
                MenGrowth-0001-002/ (t1c.nrrd, t1n.nrrd)
            MenGrowth-0002/
                MenGrowth-0002-000/ (t1c.nrrd)
                MenGrowth-0002-001/ (t1c.nrrd)
            MenGrowth-0003/
                MenGrowth-0003-000/ (t1c.nrrd, t2w.nrrd)
                MenGrowth-0003-001/ (t1c.nrrd, t2w.nrrd)
    """
    mengrowth_dir = tmp_path / "dataset" / "MenGrowth-2025"

    patients = {
        "MenGrowth-0001": {
            "MenGrowth-0001-000": ["t1c.nrrd", "t1n.nrrd"],
            "MenGrowth-0001-001": ["t1c.nrrd", "t1n.nrrd"],
            "MenGrowth-0001-002": ["t1c.nrrd", "t1n.nrrd"],
        },
        "MenGrowth-0002": {
            "MenGrowth-0002-000": ["t1c.nrrd"],
            "MenGrowth-0002-001": ["t1c.nrrd"],
        },
        "MenGrowth-0003": {
            "MenGrowth-0003-000": ["t1c.nrrd", "t2w.nrrd"],
            "MenGrowth-0003-001": ["t1c.nrrd", "t2w.nrrd"],
        },
    }

    for patient_id, studies in patients.items():
        for study_id, files in studies.items():
            study_dir = mengrowth_dir / patient_id / study_id
            study_dir.mkdir(parents=True)
            for filename in files:
                (study_dir / filename).write_text("dummy")

    return mengrowth_dir


@pytest.fixture
def quality_dir(tmp_path: Path) -> Path:
    """Create quality output directory."""
    qd = tmp_path / "quality"
    qd.mkdir(parents=True)
    return qd


@pytest.fixture
def valid_yaml(tmp_path: Path) -> Path:
    """Create a valid manual curation YAML file."""
    content = textwrap.dedent("""\
        exclusions:
          - patient_id: "MenGrowth-0001"
            study_id: "MenGrowth-0001-000"
            reason: "Poor axial resolution"
            sequence: [t1c]
          - patient_id: "MenGrowth-0003"
            study_id: "MenGrowth-0003-001"
            reason: "Motion artifacts"
            sequence: [t1c, t2w]
        min_studies_per_patient: 2
    """)
    yaml_path = tmp_path / "manual_curation.yaml"
    yaml_path.write_text(content)
    return yaml_path


# ── YAML Parsing Tests ────────────────────────────────────────────────────────


class TestLoadManualCurationConfig:
    def test_valid_config(self, valid_yaml: Path) -> None:
        config = load_manual_curation_config(valid_yaml)
        assert len(config.exclusions) == 2
        assert config.exclusions[0].patient_id == "MenGrowth-0001"
        assert config.exclusions[0].study_id == "MenGrowth-0001-000"
        assert config.exclusions[0].reason == "Poor axial resolution"
        assert config.exclusions[0].sequence == ["t1c"]
        assert config.exclusions[1].sequence == ["t1c", "t2w"]
        assert config.min_studies_per_patient == 2

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_manual_curation_config(tmp_path / "nonexistent.yaml")

    def test_empty_file(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "empty.yaml"
        yaml_path.write_text("")
        with pytest.raises(ValueError, match="Empty"):
            load_manual_curation_config(yaml_path)

    def test_missing_required_fields(self, tmp_path: Path) -> None:
        content = textwrap.dedent("""\
            exclusions:
              - patient_id: "MenGrowth-0001"
                reason: "Missing study_id field"
        """)
        yaml_path = tmp_path / "bad.yaml"
        yaml_path.write_text(content)
        with pytest.raises(ValueError, match="missing required fields"):
            load_manual_curation_config(yaml_path)

    def test_invalid_exclusions_type(self, tmp_path: Path) -> None:
        content = "exclusions: not_a_list\n"
        yaml_path = tmp_path / "bad.yaml"
        yaml_path.write_text(content)
        with pytest.raises(ValueError, match="must be a list"):
            load_manual_curation_config(yaml_path)

    def test_default_min_studies(self, tmp_path: Path) -> None:
        content = textwrap.dedent("""\
            exclusions:
              - patient_id: "MenGrowth-0001"
                study_id: "MenGrowth-0001-000"
                reason: "Test"
        """)
        yaml_path = tmp_path / "minimal.yaml"
        yaml_path.write_text(content)
        config = load_manual_curation_config(yaml_path)
        assert config.min_studies_per_patient == 2

    def test_empty_exclusions_list(self, tmp_path: Path) -> None:
        content = "exclusions: []\n"
        yaml_path = tmp_path / "empty_list.yaml"
        yaml_path.write_text(content)
        config = load_manual_curation_config(yaml_path)
        assert len(config.exclusions) == 0

    def test_sequence_omitted_defaults_empty(self, tmp_path: Path) -> None:
        content = textwrap.dedent("""\
            exclusions:
              - patient_id: "MenGrowth-0001"
                study_id: "MenGrowth-0001-000"
                reason: "No sequence info"
        """)
        yaml_path = tmp_path / "no_seq.yaml"
        yaml_path.write_text(content)
        config = load_manual_curation_config(yaml_path)
        assert config.exclusions[0].sequence == []

    def test_single_sequence_string_normalized_to_list(self, tmp_path: Path) -> None:
        content = textwrap.dedent("""\
            exclusions:
              - patient_id: "MenGrowth-0001"
                study_id: "MenGrowth-0001-000"
                reason: "Test"
                sequence: t1c
        """)
        yaml_path = tmp_path / "single_str.yaml"
        yaml_path.write_text(content)
        config = load_manual_curation_config(yaml_path)
        assert config.exclusions[0].sequence == ["t1c"]


# ── Apply Manual Curation Tests ──────────────────────────────────────────────


class TestApplyManualCuration:
    def test_remove_single_study(self, sample_dataset: Path, quality_dir: Path) -> None:
        """Remove one study from a patient with 3 studies — patient kept."""
        config = ManualCurationConfig(
            exclusions=[
                ManualExclusion(
                    patient_id="MenGrowth-0001",
                    study_id="MenGrowth-0001-000",
                    reason="Bad quality",
                )
            ],
            min_studies_per_patient=2,
        )

        stats = apply_manual_curation(config, sample_dataset, quality_dir)

        assert stats.studies_removed == 1
        assert stats.patients_cascade_removed == 0
        assert stats.patients_remaining == 3
        # Study directory should be gone
        assert not (sample_dataset / "MenGrowth-0001" / "MenGrowth-0001-000").exists()
        # Other studies should remain
        assert (sample_dataset / "MenGrowth-0001" / "MenGrowth-0001-001").exists()
        assert (sample_dataset / "MenGrowth-0001" / "MenGrowth-0001-002").exists()

    def test_cascade_patient_removal(
        self, sample_dataset: Path, quality_dir: Path
    ) -> None:
        """Remove one study from patient with 2 studies → patient removed."""
        config = ManualCurationConfig(
            exclusions=[
                ManualExclusion(
                    patient_id="MenGrowth-0002",
                    study_id="MenGrowth-0002-000",
                    reason="Bad quality",
                )
            ],
            min_studies_per_patient=2,
        )

        stats = apply_manual_curation(config, sample_dataset, quality_dir)

        assert stats.studies_removed == 1
        assert stats.patients_cascade_removed == 1
        assert stats.patients_remaining == 2
        # Entire patient directory should be gone
        assert not (sample_dataset / "MenGrowth-0002").exists()

    def test_nonexistent_study_skipped(
        self, sample_dataset: Path, quality_dir: Path
    ) -> None:
        """Exclusion referencing non-existent study is skipped."""
        config = ManualCurationConfig(
            exclusions=[
                ManualExclusion(
                    patient_id="MenGrowth-0001",
                    study_id="MenGrowth-0001-999",
                    reason="Doesn't exist",
                )
            ],
            min_studies_per_patient=2,
        )

        stats = apply_manual_curation(config, sample_dataset, quality_dir)

        assert stats.studies_removed == 0
        assert stats.exclusions_skipped == 1
        assert stats.patients_remaining == 3

    def test_rejected_files_csv_created(
        self, sample_dataset: Path, quality_dir: Path
    ) -> None:
        """Rejection records are written to rejected_files.csv."""
        config = ManualCurationConfig(
            exclusions=[
                ManualExclusion(
                    patient_id="MenGrowth-0001",
                    study_id="MenGrowth-0001-000",
                    reason="Test removal",
                )
            ],
            min_studies_per_patient=2,
        )

        apply_manual_curation(config, sample_dataset, quality_dir)

        rejected_csv = quality_dir / "rejected_files.csv"
        assert rejected_csv.exists()

        with open(rejected_csv, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # MenGrowth-0001-000 had 2 files (t1c.nrrd, t1n.nrrd)
        assert len(rows) == 2
        for row in rows:
            assert row["stage"] == str(MANUAL_CURATION_STAGE)
            assert row["source_type"] == "manual_curation"
            assert "Test removal" in row["rejection_reason"]
            assert row["patient_id"] == "MenGrowth-0001"

    def test_sequence_traced_in_csv(
        self, sample_dataset: Path, quality_dir: Path
    ) -> None:
        """Sequence tag appears in the rejection_reason column of the CSV."""
        config = ManualCurationConfig(
            exclusions=[
                ManualExclusion(
                    patient_id="MenGrowth-0001",
                    study_id="MenGrowth-0001-000",
                    reason="Poor resolution",
                    sequence=["t1c", "t2w"],
                )
            ],
            min_studies_per_patient=2,
        )

        apply_manual_curation(config, sample_dataset, quality_dir)

        rejected_csv = quality_dir / "rejected_files.csv"
        with open(rejected_csv, "r") as f:
            rows = list(csv.DictReader(f))

        for row in rows:
            assert "[t1c, t2w]" in row["rejection_reason"]

    def test_no_sequence_tag_when_omitted(
        self, sample_dataset: Path, quality_dir: Path
    ) -> None:
        """No sequence bracket tag when sequence is empty."""
        config = ManualCurationConfig(
            exclusions=[
                ManualExclusion(
                    patient_id="MenGrowth-0001",
                    study_id="MenGrowth-0001-000",
                    reason="Generic issue",
                )
            ],
            min_studies_per_patient=2,
        )

        apply_manual_curation(config, sample_dataset, quality_dir)

        rejected_csv = quality_dir / "rejected_files.csv"
        with open(rejected_csv, "r") as f:
            rows = list(csv.DictReader(f))

        for row in rows:
            assert row["rejection_reason"] == "Manual curation: Generic issue"
            assert "[" not in row["rejection_reason"]

    def test_cascade_rejection_records(
        self, sample_dataset: Path, quality_dir: Path
    ) -> None:
        """Cascade removal creates rejection records for remaining studies."""
        config = ManualCurationConfig(
            exclusions=[
                ManualExclusion(
                    patient_id="MenGrowth-0002",
                    study_id="MenGrowth-0002-000",
                    reason="Bad quality",
                )
            ],
            min_studies_per_patient=2,
        )

        apply_manual_curation(config, sample_dataset, quality_dir)

        rejected_csv = quality_dir / "rejected_files.csv"
        with open(rejected_csv, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # 1 file from explicit removal + 1 file from cascade removal
        assert len(rows) == 2
        cascade_rows = [
            r for r in rows if "Insufficient studies" in r["rejection_reason"]
        ]
        assert len(cascade_rows) == 1

    def test_multiple_exclusions(self, sample_dataset: Path, quality_dir: Path) -> None:
        """Multiple exclusions are applied correctly."""
        config = ManualCurationConfig(
            exclusions=[
                ManualExclusion(
                    patient_id="MenGrowth-0001",
                    study_id="MenGrowth-0001-000",
                    reason="Poor resolution",
                ),
                ManualExclusion(
                    patient_id="MenGrowth-0003",
                    study_id="MenGrowth-0003-001",
                    reason="Motion artifacts",
                ),
            ],
            min_studies_per_patient=2,
        )

        stats = apply_manual_curation(config, sample_dataset, quality_dir)

        assert stats.studies_removed == 2
        # MenGrowth-0001 still has 2 studies, MenGrowth-0003 has 1 → cascade remove
        assert stats.patients_cascade_removed == 1
        assert stats.patients_remaining == 2
        assert not (sample_dataset / "MenGrowth-0003").exists()

    def test_empty_exclusions(self, sample_dataset: Path, quality_dir: Path) -> None:
        """Empty exclusions list does nothing."""
        config = ManualCurationConfig(exclusions=[], min_studies_per_patient=2)

        stats = apply_manual_curation(config, sample_dataset, quality_dir)

        assert stats.studies_removed == 0
        assert stats.patients_cascade_removed == 0
        assert stats.patients_remaining == 3

    def test_source_path_traces_back_to_original_id(
        self, sample_dataset: Path, quality_dir: Path, tmp_path: Path
    ) -> None:
        """When id_mapping_path is given, source_path uses pre-reID P{N} paths."""
        import json

        # Write an id_mapping.json that maps P10 → MenGrowth-0001
        id_mapping = {
            "P10": {
                "new_id": "MenGrowth-0001",
                "studies": {
                    "0": "MenGrowth-0001-000",
                    "3": "MenGrowth-0001-001",
                    "5": "MenGrowth-0001-002",
                },
            },
            "P20": {
                "new_id": "MenGrowth-0002",
                "studies": {
                    "0": "MenGrowth-0002-000",
                    "1": "MenGrowth-0002-001",
                },
            },
            "P30": {
                "new_id": "MenGrowth-0003",
                "studies": {
                    "0": "MenGrowth-0003-000",
                    "2": "MenGrowth-0003-001",
                },
            },
        }
        id_mapping_path = tmp_path / "dataset" / "id_mapping.json"
        id_mapping_path.parent.mkdir(parents=True, exist_ok=True)
        with open(id_mapping_path, "w") as f:
            json.dump(id_mapping, f)

        config = ManualCurationConfig(
            exclusions=[
                ManualExclusion(
                    patient_id="MenGrowth-0001",
                    study_id="MenGrowth-0001-000",
                    reason="Bad quality",
                )
            ],
            min_studies_per_patient=2,
        )

        apply_manual_curation(
            config, sample_dataset, quality_dir, id_mapping_path=id_mapping_path
        )

        rejected_csv = quality_dir / "rejected_files.csv"
        with open(rejected_csv, "r") as f:
            rows = list(csv.DictReader(f))

        # source_path should reference P10/0/ (the original IDs)
        for row in rows:
            assert "P10" in row["source_path"]
            assert "/0/" in row["source_path"]
            assert row["patient_id"] == "P10"
            assert row["study_name"] == "0"

    def test_source_path_uses_mengrowth_id_without_mapping(
        self, sample_dataset: Path, quality_dir: Path
    ) -> None:
        """Without id_mapping_path, source_path uses current MenGrowth IDs."""
        config = ManualCurationConfig(
            exclusions=[
                ManualExclusion(
                    patient_id="MenGrowth-0001",
                    study_id="MenGrowth-0001-000",
                    reason="Bad quality",
                )
            ],
            min_studies_per_patient=2,
        )

        apply_manual_curation(config, sample_dataset, quality_dir)

        rejected_csv = quality_dir / "rejected_files.csv"
        with open(rejected_csv, "r") as f:
            rows = list(csv.DictReader(f))

        for row in rows:
            assert "MenGrowth-0001" in row["source_path"]
            assert row["patient_id"] == "MenGrowth-0001"
            assert row["study_name"] == "MenGrowth-0001-000"

    def test_cascade_removal_traces_back_to_original_id(
        self, sample_dataset: Path, quality_dir: Path, tmp_path: Path
    ) -> None:
        """Cascade-removed patient files also trace back to original IDs."""
        import json

        id_mapping = {
            "P10": {
                "new_id": "MenGrowth-0001",
                "studies": {
                    "0": "MenGrowth-0001-000",
                    "3": "MenGrowth-0001-001",
                    "5": "MenGrowth-0001-002",
                },
            },
            "P20": {
                "new_id": "MenGrowth-0002",
                "studies": {
                    "0": "MenGrowth-0002-000",
                    "1": "MenGrowth-0002-001",
                },
            },
            "P30": {
                "new_id": "MenGrowth-0003",
                "studies": {
                    "0": "MenGrowth-0003-000",
                    "2": "MenGrowth-0003-001",
                },
            },
        }
        id_mapping_path = tmp_path / "dataset" / "id_mapping.json"
        id_mapping_path.parent.mkdir(parents=True, exist_ok=True)
        with open(id_mapping_path, "w") as f:
            json.dump(id_mapping, f)

        # Remove one study from MenGrowth-0002 (has only 2) → cascade removal
        config = ManualCurationConfig(
            exclusions=[
                ManualExclusion(
                    patient_id="MenGrowth-0002",
                    study_id="MenGrowth-0002-000",
                    reason="Bad quality",
                )
            ],
            min_studies_per_patient=2,
        )

        apply_manual_curation(
            config, sample_dataset, quality_dir, id_mapping_path=id_mapping_path
        )

        rejected_csv = quality_dir / "rejected_files.csv"
        with open(rejected_csv, "r") as f:
            rows = list(csv.DictReader(f))

        # Both the explicit removal and cascade should trace to P20
        for row in rows:
            assert row["patient_id"] == "P20"
            assert "P20" in row["source_path"]


# ── Re-ID After Manual Curation Tests ────────────────────────────────────────


class TestReidAfterManualCuration:
    @pytest.fixture
    def dataset_with_gap(self, tmp_path: Path) -> tuple:
        """Create a dataset with a gap in MenGrowth IDs + id_mapping.json.

        Simulates removal of MenGrowth-0002, leaving 0001 and 0003.
        """
        import json

        from mengrowth.preprocessing.utils.manual_curation import (
            reid_after_manual_curation,
        )

        mengrowth_dir = tmp_path / "dataset" / "MenGrowth-2025"

        # Create MenGrowth-0001 with 2 studies
        for sid in ["MenGrowth-0001-000", "MenGrowth-0001-001"]:
            (mengrowth_dir / "MenGrowth-0001" / sid).mkdir(parents=True)
            (mengrowth_dir / "MenGrowth-0001" / sid / "t1c.nrrd").write_text("dummy")

        # Skip MenGrowth-0002 (simulates removal)

        # Create MenGrowth-0003 with 2 studies
        for sid in ["MenGrowth-0003-000", "MenGrowth-0003-001"]:
            (mengrowth_dir / "MenGrowth-0003" / sid).mkdir(parents=True)
            (mengrowth_dir / "MenGrowth-0003" / sid / "t1c.nrrd").write_text("dummy")

        # Write id_mapping.json (original P{N} → MenGrowth-{XXXX})
        id_mapping = {
            "P10": {
                "new_id": "MenGrowth-0001",
                "studies": {"0": "MenGrowth-0001-000", "1": "MenGrowth-0001-001"},
            },
            "P20": {
                "new_id": "MenGrowth-0002",
                "studies": {"0": "MenGrowth-0002-000", "3": "MenGrowth-0002-001"},
            },
            "P30": {
                "new_id": "MenGrowth-0003",
                "studies": {"0": "MenGrowth-0003-000", "2": "MenGrowth-0003-001"},
            },
        }
        id_mapping_path = tmp_path / "dataset" / "id_mapping.json"
        with open(id_mapping_path, "w") as f:
            json.dump(id_mapping, f)

        return mengrowth_dir, id_mapping_path, reid_after_manual_curation

    def test_closes_gap(self, dataset_with_gap: tuple) -> None:
        """MenGrowth-0003 is renumbered to MenGrowth-0002."""
        mengrowth_dir, id_mapping_path, reid_fn = dataset_with_gap

        rename_map = reid_fn(mengrowth_dir, id_mapping_path)

        assert rename_map == {"MenGrowth-0003": "MenGrowth-0002"}
        assert (mengrowth_dir / "MenGrowth-0001").exists()
        assert (mengrowth_dir / "MenGrowth-0002").exists()
        assert not (mengrowth_dir / "MenGrowth-0003").exists()

    def test_study_dirs_renamed(self, dataset_with_gap: tuple) -> None:
        """Study directories are renamed to match new patient ID."""
        mengrowth_dir, id_mapping_path, reid_fn = dataset_with_gap

        reid_fn(mengrowth_dir, id_mapping_path)

        # Old MenGrowth-0003 is now MenGrowth-0002
        studies = sorted(
            d.name for d in (mengrowth_dir / "MenGrowth-0002").iterdir() if d.is_dir()
        )
        assert studies == ["MenGrowth-0002-000", "MenGrowth-0002-001"]

    def test_id_mapping_updated(self, dataset_with_gap: tuple) -> None:
        """id_mapping.json reflects the new IDs."""
        import json

        mengrowth_dir, id_mapping_path, reid_fn = dataset_with_gap

        reid_fn(mengrowth_dir, id_mapping_path)

        with open(id_mapping_path) as f:
            mapping = json.load(f)

        # P10 still maps to MenGrowth-0001
        assert mapping["P10"]["new_id"] == "MenGrowth-0001"
        # P30 now maps to MenGrowth-0002 (was 0003)
        assert mapping["P30"]["new_id"] == "MenGrowth-0002"
        # P20 should be removed from mapping (patient was deleted)
        assert "P20" not in mapping

    def test_no_gap_no_rename(self, tmp_path: Path) -> None:
        """No renaming when IDs are already continuous."""
        import json

        from mengrowth.preprocessing.utils.manual_curation import (
            reid_after_manual_curation,
        )

        mengrowth_dir = tmp_path / "dataset" / "MenGrowth-2025"
        for pid in ["MenGrowth-0001", "MenGrowth-0002"]:
            sid = f"{pid}-000"
            (mengrowth_dir / pid / sid).mkdir(parents=True)
            (mengrowth_dir / pid / sid / "t1c.nrrd").write_text("dummy")

        id_mapping_path = tmp_path / "dataset" / "id_mapping.json"
        mapping = {
            "P1": {"new_id": "MenGrowth-0001", "studies": {"0": "MenGrowth-0001-000"}},
            "P2": {"new_id": "MenGrowth-0002", "studies": {"0": "MenGrowth-0002-000"}},
        }
        with open(id_mapping_path, "w") as f:
            json.dump(mapping, f)

        rename_map = reid_after_manual_curation(mengrowth_dir, id_mapping_path)

        assert rename_map == {}
        assert (mengrowth_dir / "MenGrowth-0001").exists()
        assert (mengrowth_dir / "MenGrowth-0002").exists()
