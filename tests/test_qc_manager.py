"""Integration tests for QCManager."""

import pytest
import tempfile
import shutil
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import pandas as pd

from mengrowth.preprocessing.src.config import QCConfig
from mengrowth.preprocessing.quality_analysis.qc_manager import QCManager


@pytest.fixture
def temp_dir():
    """Create temporary directory for test outputs."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    # Cleanup
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def qc_config(temp_dir):
    """Create QC configuration for testing."""
    return QCConfig(
        enabled=True,
        output_dir=str(temp_dir / "qc"),
        artifacts_dir=str(temp_dir / "qc_artifacts"),
        overwrite=True,
        compute_after_steps=["data_harmonization", "skull_stripping"],
        downsample_to_mm=2.0,
        max_voxels=100000,
        random_seed=42,
        mask_source="otsu_only"
    )


@pytest.fixture
def synthetic_image(temp_dir):
    """Create synthetic test image."""
    # Create simple 3D image
    arr = np.random.rand(64, 64, 64).astype(np.float32) * 100
    image = sitk.GetImageFromArray(arr)
    image.SetSpacing([1.0, 1.0, 1.0])
    image.SetOrigin([0, 0, 0])
    image.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])

    # Save to temporary file
    image_path = temp_dir / "test_image.nii.gz"
    sitk.WriteImage(image, str(image_path))

    return image_path


class TestQCManagerInitialization:
    """Test QCManager initialization."""

    def test_qc_manager_init(self, qc_config):
        """Test QCManager initializes correctly."""
        qc_manager = QCManager(qc_config)

        assert qc_manager.config.enabled == True
        assert len(qc_manager.metrics_accumulator) == 0
        assert len(qc_manager.histogram_accumulator) == 0

    def test_output_directories_created(self, qc_config):
        """Test that output directories are created."""
        qc_manager = QCManager(qc_config)

        assert Path(qc_config.output_dir).exists()
        assert Path(qc_config.artifacts_dir).exists()


class TestQCManagerAccumulation:
    """Test QC metric accumulation."""

    def test_on_step_completed_single_metric(self, qc_config, synthetic_image):
        """Test that on_step_completed accumulates metrics."""
        qc_manager = QCManager(qc_config)

        case_metadata = {
            "patient_id": "TEST-0001",
            "study_id": "000",
            "modality": "t1c"
        }

        image_paths = {
            "output": synthetic_image
        }

        qc_manager.on_step_completed(
            step_name="data_harmonization_1",
            case_metadata=case_metadata,
            image_paths=image_paths,
            artifact_paths={}
        )

        # Check that metric was accumulated
        assert len(qc_manager.metrics_accumulator) == 1
        metric = qc_manager.metrics_accumulator[0]

        assert metric["patient_id"] == "TEST-0001"
        assert metric["modality"] == "t1c"
        assert metric["status"] == "success"

    def test_skips_unconfigured_steps(self, qc_config, synthetic_image):
        """Test that steps not in compute_after_steps are skipped."""
        qc_manager = QCManager(qc_config)

        case_metadata = {"patient_id": "TEST-0001"}
        image_paths = {"output": synthetic_image}

        qc_manager.on_step_completed(
            step_name="bias_field_correction",  # Not in compute_after_steps
            case_metadata=case_metadata,
            image_paths=image_paths,
            artifact_paths={}
        )

        # Should not accumulate
        assert len(qc_manager.metrics_accumulator) == 0

    def test_handles_missing_image(self, qc_config):
        """Test graceful handling of missing image."""
        qc_manager = QCManager(qc_config)

        case_metadata = {"patient_id": "TEST-0001"}
        image_paths = {"output": Path("/nonexistent/image.nii.gz")}

        qc_manager.on_step_completed(
            step_name="data_harmonization",
            case_metadata=case_metadata,
            image_paths=image_paths,
            artifact_paths={}
        )

        # Should accumulate with error status
        assert len(qc_manager.metrics_accumulator) == 1
        assert qc_manager.metrics_accumulator[0]["status"] == "missing_output"


class TestQCManagerFinalization:
    """Test QC finalization and output writing."""

    def test_finalize_writes_outputs(self, qc_config, synthetic_image):
        """Test that finalize() writes output files."""
        qc_manager = QCManager(qc_config)

        # Add some metrics
        for i in range(3):
            case_metadata = {
                "patient_id": f"TEST-{i:04d}",
                "study_id": "000",
                "modality": "t1c"
            }
            qc_manager.on_step_completed(
                step_name="data_harmonization",
                case_metadata=case_metadata,
                image_paths={"output": synthetic_image},
                artifact_paths={}
            )

        # Finalize
        output_paths = qc_manager.finalize()

        # Check outputs exist
        assert "long_csv" in output_paths
        assert "metadata_json" in output_paths
        assert output_paths["long_csv"].exists()
        assert output_paths["metadata_json"].exists()

    def test_csv_schema_correct(self, qc_config, synthetic_image):
        """Test that CSV has correct schema."""
        qc_manager = QCManager(qc_config)

        case_metadata = {
            "patient_id": "TEST-0001",
            "study_id": "000",
            "modality": "t1c"
        }
        qc_manager.on_step_completed(
            step_name="data_harmonization",
            case_metadata=case_metadata,
            image_paths={"output": synthetic_image},
            artifact_paths={}
        )

        output_paths = qc_manager.finalize()

        # Read CSV
        df = pd.read_csv(output_paths["long_csv"])

        # Check required columns exist
        required_cols = ["patient_id", "modality", "step_name", "status", "pipeline_run_id"]
        for col in required_cols:
            assert col in df.columns

        # Check data
        assert df["patient_id"].iloc[0] == "TEST-0001"
        assert df["modality"].iloc[0] == "t1c"

    def test_finalize_empty_accumulator(self, qc_config):
        """Test finalize with no metrics accumulated."""
        qc_manager = QCManager(qc_config)

        # Finalize without adding any metrics
        output_paths = qc_manager.finalize()

        # Should return empty dict
        assert output_paths == {}


class TestQCManagerTwoPassWasserstein:
    """Test two-pass Wasserstein computation."""

    def test_histogram_accumulation(self, qc_config, synthetic_image):
        """Test that histograms are accumulated for Wasserstein."""
        # Enable intensity stability metrics
        qc_config.metrics.intensity_stability.enabled = True
        qc_config.metrics.intensity_stability.wasserstein_distance = True

        qc_manager = QCManager(qc_config)

        # Add multiple cases to accumulate histograms
        for i in range(5):
            case_metadata = {
                "patient_id": f"TEST-{i:04d}",
                "study_id": "000",
                "modality": "t1c"
            }
            qc_manager.on_step_completed(
                step_name="data_harmonization",
                case_metadata=case_metadata,
                image_paths={"output": synthetic_image},
                artifact_paths={}
            )

        # Check histograms accumulated
        assert len(qc_manager.histogram_accumulator) > 0

        # Finalize to compute references
        qc_manager.finalize()

        # Check reference histograms computed
        assert len(qc_manager.reference_histograms) > 0


class TestQCManagerErrorHandling:
    """Test error handling."""

    def test_handles_computation_errors(self, qc_config, tmp_path):
        """Test that computation errors don't crash the manager."""
        qc_manager = QCManager(qc_config)

        # Create corrupted image file
        bad_image_path = tmp_path / "bad_image.nii.gz"
        bad_image_path.write_text("not a valid image")

        case_metadata = {"patient_id": "TEST-0001"}

        # Should not raise exception
        qc_manager.on_step_completed(
            step_name="data_harmonization",
            case_metadata=case_metadata,
            image_paths={"output": bad_image_path},
            artifact_paths={}
        )

        # Should have error in accumulator
        assert len(qc_manager.metrics_accumulator) == 1
        assert qc_manager.metrics_accumulator[0]["status"] == "error"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
