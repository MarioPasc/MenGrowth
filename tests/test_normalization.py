"""Unit tests for intensity normalization brain mask fixes.

Tests verify that all normalizers correctly use brain masks to compute
statistics on brain voxels only, and that background voxels remain at 0.
"""

import pytest
import numpy as np
import nibabel as nib
from pathlib import Path
import tempfile
import shutil


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d)


def create_skull_stripped_image(temp_dir, filename="t1c.nii.gz", shape=(64, 64, 64)):
    """Create a synthetic skull-stripped MRI image.

    ~15% brain voxels with realistic intensity range, ~85% background zeros.
    """
    data = np.zeros(shape, dtype=np.float32)
    # Create a spherical brain region in the center (~15% of voxels)
    center = np.array(shape) // 2
    coords = np.mgrid[0 : shape[0], 0 : shape[1], 0 : shape[2]]
    dist = np.sqrt(sum((coords[i] - center[i]) ** 2 for i in range(3)))
    radius = min(shape) * 0.33  # ~15% of voxels in sphere
    brain_mask = dist < radius

    # Fill brain with realistic MRI intensities (mean=500, std=150)
    rng = np.random.RandomState(42)
    data[brain_mask] = rng.normal(500, 150, size=int(brain_mask.sum())).clip(50, 1200)

    affine = np.eye(4)
    img = nib.Nifti1Image(data, affine)
    img_path = temp_dir / filename
    nib.save(img, str(img_path))

    mask_data = brain_mask.astype(np.uint8)
    mask_img = nib.Nifti1Image(mask_data, affine)
    mask_path = temp_dir / f"{Path(filename).stem.split('.')[0]}_brain_mask.nii.gz"
    nib.save(mask_img, str(mask_path))

    brain_pct = 100.0 * brain_mask.sum() / data.size
    return img_path, mask_path, brain_mask, brain_pct


class TestZScoreNormalizer:
    """Tests for z-score normalizer with brain mask."""

    def test_zscore_uses_brain_mask(self, temp_dir):
        """Z-score should compute mean/std on brain voxels only, background stays 0."""
        from mengrowth.preprocessing.src.normalization.zscore import ZScoreNormalizer

        img_path, mask_path, brain_mask, _ = create_skull_stripped_image(temp_dir)
        output_path = temp_dir / "normalized.nii.gz"

        normalizer = ZScoreNormalizer(config={"norm_value": 1.0})
        result = normalizer.execute(
            img_path, output_path, allow_overwrite=True, mask_path=mask_path
        )

        # Load result
        out_data = nib.load(str(output_path)).get_fdata()

        # Background should be exactly 0
        assert np.all(out_data[~brain_mask] == 0.0), "Background voxels should be 0"

        # Brain voxels should have mean~0 and std~1
        brain_voxels = out_data[brain_mask]
        assert abs(np.mean(brain_voxels)) < 0.05, (
            f"Mean should be ~0, got {np.mean(brain_voxels):.3f}"
        )
        assert abs(np.std(brain_voxels) - 1.0) < 0.05, (
            f"Std should be ~1, got {np.std(brain_voxels):.3f}"
        )

        # Result should contain brain coverage metrics
        assert "mean" in result
        assert "std" in result
        assert "brain_voxel_count" in result
        assert "brain_coverage_percent" in result
        assert result["mask_source"] == "skull_stripping"

    def test_zscore_clip_range(self, temp_dir):
        """Z-score with clip_range should clip values within brain mask."""
        from mengrowth.preprocessing.src.normalization.zscore import ZScoreNormalizer

        img_path, mask_path, brain_mask, _ = create_skull_stripped_image(temp_dir)
        output_path = temp_dir / "normalized_clipped.nii.gz"

        normalizer = ZScoreNormalizer(
            config={"norm_value": 1.0, "clip_range": [-2.0, 2.0]}
        )
        result = normalizer.execute(
            img_path, output_path, allow_overwrite=True, mask_path=mask_path
        )

        out_data = nib.load(str(output_path)).get_fdata()
        brain_voxels = out_data[brain_mask]

        assert brain_voxels.min() >= -2.0, (
            f"Min should be >= -2.0, got {brain_voxels.min():.3f}"
        )
        assert brain_voxels.max() <= 2.0, (
            f"Max should be <= 2.0, got {brain_voxels.max():.3f}"
        )
        assert result["clip_range"] == [-2.0, 2.0]

    def test_zscore_invalid_clip_range(self):
        """Invalid clip_range should raise ValueError."""
        from mengrowth.preprocessing.src.normalization.zscore import ZScoreNormalizer

        with pytest.raises(ValueError, match="clip_range"):
            ZScoreNormalizer(config={"norm_value": 1.0, "clip_range": [2.0, -2.0]})

        with pytest.raises(ValueError, match="clip_range"):
            ZScoreNormalizer(config={"norm_value": 1.0, "clip_range": [1.0]})


class TestPercentileMinMaxNormalizer:
    """Tests for percentile min-max normalizer with brain mask."""

    def test_percentile_minmax_uses_brain_mask(self, temp_dir):
        """Percentiles should be computed on brain voxels only, not on zeros."""
        from mengrowth.preprocessing.src.normalization.percentile_minmax import (
            PercentileMinMaxNormalizer,
        )

        img_path, mask_path, brain_mask, _ = create_skull_stripped_image(temp_dir)
        output_path = temp_dir / "normalized.nii.gz"

        normalizer = PercentileMinMaxNormalizer(config={"p1": 1.0, "p2": 99.0})
        result = normalizer.execute(
            img_path, output_path, allow_overwrite=True, mask_path=mask_path
        )

        # P1 should NOT be 0 (the bug was computing percentiles on all voxels including 85% zeros)
        assert result["p1_value"] > 0, (
            f"P1 should be > 0 (computed on brain voxels), got {result['p1_value']:.3f}"
        )

        # Load and check background
        out_data = nib.load(str(output_path)).get_fdata()
        assert np.all(out_data[~brain_mask] == 0.0), "Background voxels should be 0"

        # Brain coverage metrics should be present
        assert "brain_voxel_count" in result
        assert "brain_coverage_percent" in result
        assert result["mask_source"] == "skull_stripping"

    def test_percentile_minmax_without_mask_uses_nonzero(self, temp_dir):
        """Without mask_path, should use image > 0 as fallback."""
        from mengrowth.preprocessing.src.normalization.percentile_minmax import (
            PercentileMinMaxNormalizer,
        )

        img_path, _, brain_mask, _ = create_skull_stripped_image(temp_dir)
        output_path = temp_dir / "normalized.nii.gz"

        normalizer = PercentileMinMaxNormalizer(config={"p1": 1.0, "p2": 99.0})
        result = normalizer.execute(
            img_path,
            output_path,
            allow_overwrite=True,
            # No mask_path
        )

        # Should still not be 0 because nonzero fallback is used
        assert result["p1_value"] > 0
        assert result["mask_source"] == "nonzero_fallback"


class TestFCMNormalizer:
    """Tests for FCM normalizer with brain mask."""

    def test_fcm_uses_brain_mask(self, temp_dir):
        """FCM should receive brain mask and produce brain coverage metrics."""
        from mengrowth.preprocessing.src.normalization.fcm import FCMNormalizer

        img_path, mask_path, brain_mask, _ = create_skull_stripped_image(temp_dir)
        output_path = temp_dir / "normalized.nii.gz"

        normalizer = FCMNormalizer(
            config={
                "n_clusters": 3,
                "tissue_type": "WM",
                "max_iter": 50,
                "error_threshold": 0.005,
                "fuzziness": 2.0,
            }
        )
        result = normalizer.execute(
            img_path, output_path, allow_overwrite=True, mask_path=mask_path
        )

        # Check background is zero
        out_data = nib.load(str(output_path)).get_fdata()
        assert np.all(out_data[~brain_mask] == 0.0), "Background voxels should be 0"

        # Brain coverage metrics
        assert "brain_voxel_count" in result
        assert "brain_coverage_percent" in result
        assert result["mask_source"] == "skull_stripping"


class TestNormalizersReturnBrainCoverage:
    """All normalizers should return brain_voxel_count and brain_coverage_percent."""

    def test_zscore_returns_brain_coverage(self, temp_dir):
        from mengrowth.preprocessing.src.normalization.zscore import ZScoreNormalizer

        img_path, mask_path, _, brain_pct = create_skull_stripped_image(temp_dir)
        output_path = temp_dir / "out.nii.gz"

        normalizer = ZScoreNormalizer(config={"norm_value": 1.0})
        result = normalizer.execute(
            img_path, output_path, allow_overwrite=True, mask_path=mask_path
        )

        assert isinstance(result["brain_voxel_count"], int)
        assert result["brain_voxel_count"] > 0
        assert 0 < result["brain_coverage_percent"] < 100
        assert abs(result["brain_coverage_percent"] - brain_pct) < 1.0

    def test_percentile_minmax_returns_brain_coverage(self, temp_dir):
        from mengrowth.preprocessing.src.normalization.percentile_minmax import (
            PercentileMinMaxNormalizer,
        )

        img_path, mask_path, _, _ = create_skull_stripped_image(temp_dir)
        output_path = temp_dir / "out.nii.gz"

        normalizer = PercentileMinMaxNormalizer(config={"p1": 1.0, "p2": 99.0})
        result = normalizer.execute(
            img_path, output_path, allow_overwrite=True, mask_path=mask_path
        )

        assert isinstance(result["brain_voxel_count"], int)
        assert result["brain_voxel_count"] > 0
        assert 0 < result["brain_coverage_percent"] < 100


class TestResolveBrainMaskPath:
    """Test mask resolution from artifacts directory."""

    def test_resolve_finds_existing_mask(self, temp_dir):
        """Should find mask when it exists in artifacts directory."""
        from mengrowth.preprocessing.src.steps.intensity_normalization import (
            _resolve_brain_mask_path,
        )
        from unittest.mock import MagicMock
        from mengrowth.preprocessing.src.config import StepExecutionContext

        # Create artifacts structure
        artifacts_dir = temp_dir / "artifacts"
        patient_dir = artifacts_dir / "MenGrowth-0006" / "MenGrowth-0006-000"
        patient_dir.mkdir(parents=True)
        mask_file = patient_dir / "t1c_brain_mask.nii.gz"
        mask_file.touch()

        # Create mock context
        context = MagicMock(spec=StepExecutionContext)
        context.patient_id = "MenGrowth-0006"
        context.study_dir = MagicMock()
        context.study_dir.name = "MenGrowth-0006-000"
        context.modality = "t1c"
        context.orchestrator = MagicMock()
        context.orchestrator.config.preprocessing_artifacts_path = str(artifacts_dir)

        result = _resolve_brain_mask_path(context)
        assert result is not None
        assert result.exists()

    def test_resolve_returns_none_when_no_mask(self, temp_dir):
        """Should return None when mask file doesn't exist."""
        from mengrowth.preprocessing.src.steps.intensity_normalization import (
            _resolve_brain_mask_path,
        )
        from unittest.mock import MagicMock
        from mengrowth.preprocessing.src.config import StepExecutionContext

        artifacts_dir = temp_dir / "artifacts"
        artifacts_dir.mkdir(parents=True)

        context = MagicMock(spec=StepExecutionContext)
        context.patient_id = "MenGrowth-0006"
        context.study_dir = MagicMock()
        context.study_dir.name = "MenGrowth-0006-000"
        context.modality = "t1c"
        context.orchestrator = MagicMock()
        context.orchestrator.config.preprocessing_artifacts_path = str(artifacts_dir)

        result = _resolve_brain_mask_path(context)
        assert result is None


class TestMaskFallbackToNonzero:
    """When no mask file exists, normalizers should use image > 0."""

    def test_zscore_fallback(self, temp_dir):
        from mengrowth.preprocessing.src.normalization.zscore import ZScoreNormalizer

        img_path, _, brain_mask, _ = create_skull_stripped_image(temp_dir)
        output_path = temp_dir / "out.nii.gz"

        normalizer = ZScoreNormalizer(config={"norm_value": 1.0})
        result = normalizer.execute(
            img_path,
            output_path,
            allow_overwrite=True,
            # No mask_path provided
        )

        assert result["mask_source"] == "nonzero_fallback"

        # Brain voxels should still have mean~0, std~1
        out_data = nib.load(str(output_path)).get_fdata()
        brain_voxels = out_data[brain_mask]
        assert abs(np.mean(brain_voxels)) < 0.05
        assert abs(np.std(brain_voxels) - 1.0) < 0.05

    def test_zscore_fallback_with_nonexistent_path(self, temp_dir):
        from mengrowth.preprocessing.src.normalization.zscore import ZScoreNormalizer

        img_path, _, _, _ = create_skull_stripped_image(temp_dir)
        output_path = temp_dir / "out.nii.gz"

        normalizer = ZScoreNormalizer(config={"norm_value": 1.0})
        result = normalizer.execute(
            img_path,
            output_path,
            allow_overwrite=True,
            mask_path="/nonexistent/mask.nii.gz",
        )

        assert result["mask_source"] == "nonzero_fallback"


class TestConfigClipRange:
    """Test clip_range configuration validation."""

    def test_valid_clip_range(self):
        from mengrowth.preprocessing.src.config import IntensityNormalizationConfig

        config = IntensityNormalizationConfig(method="zscore", clip_range=[-5.0, 5.0])
        assert config.clip_range == [-5.0, 5.0]

    def test_null_clip_range(self):
        from mengrowth.preprocessing.src.config import IntensityNormalizationConfig

        config = IntensityNormalizationConfig(method="zscore", clip_range=None)
        assert config.clip_range is None

    def test_invalid_clip_range_reversed(self):
        from mengrowth.preprocessing.src.config import (
            IntensityNormalizationConfig,
            ConfigurationError,
        )

        with pytest.raises(ConfigurationError, match="clip_range"):
            IntensityNormalizationConfig(method="zscore", clip_range=[5.0, -5.0])

    def test_invalid_clip_range_length(self):
        from mengrowth.preprocessing.src.config import (
            IntensityNormalizationConfig,
            ConfigurationError,
        )

        with pytest.raises(ConfigurationError, match="clip_range"):
            IntensityNormalizationConfig(method="zscore", clip_range=[1.0])


class TestMaskShapeMismatchDefense:
    """Tests that normalizers handle brain mask shape != image shape via BSpline resampling."""

    def _create_mismatched_image_and_mask(
        self, temp_dir, img_shape, mask_shape, filename="t1c.nii.gz"
    ):
        """Create image and mask with different shapes to simulate atlas registration mismatch."""
        affine = np.eye(4)

        # Create image with brain region
        data = np.zeros(img_shape, dtype=np.float32)
        center = np.array(img_shape) // 2
        coords = np.mgrid[0 : img_shape[0], 0 : img_shape[1], 0 : img_shape[2]]
        dist = np.sqrt(sum((coords[i] - center[i]) ** 2 for i in range(3)))
        radius = min(img_shape) * 0.33
        brain_region = dist < radius
        rng = np.random.RandomState(42)
        data[brain_region] = rng.normal(500, 150, size=int(brain_region.sum())).clip(
            50, 1200
        )

        img = nib.Nifti1Image(data, affine)
        img_path = temp_dir / filename
        nib.save(img, str(img_path))

        # Create mask at different shape
        mask_data_full = np.zeros(mask_shape, dtype=np.uint8)
        mask_center = np.array(mask_shape) // 2
        mask_coords = np.mgrid[0 : mask_shape[0], 0 : mask_shape[1], 0 : mask_shape[2]]
        mask_dist = np.sqrt(
            sum((mask_coords[i] - mask_center[i]) ** 2 for i in range(3))
        )
        mask_radius = min(mask_shape) * 0.33
        mask_data_full[mask_dist < mask_radius] = 1

        mask_img = nib.Nifti1Image(mask_data_full, affine)
        mask_path = temp_dir / f"{Path(filename).stem.split('.')[0]}_brain_mask.nii.gz"
        nib.save(mask_img, str(mask_path))

        return img_path, mask_path

    def test_zscore_resamples_mask_on_shape_mismatch(self, temp_dir):
        """Z-score should resample mask when shapes differ, not crash."""
        from mengrowth.preprocessing.src.normalization.zscore import ZScoreNormalizer

        img_path, mask_path = self._create_mismatched_image_and_mask(
            temp_dir, img_shape=(64, 64, 64), mask_shape=(48, 48, 48)
        )
        output_path = temp_dir / "normalized.nii.gz"

        normalizer = ZScoreNormalizer(config={"norm_value": 1.0})
        result = normalizer.execute(
            img_path, output_path, allow_overwrite=True, mask_path=mask_path
        )

        out_data = nib.load(str(output_path)).get_fdata()
        assert out_data.shape == (64, 64, 64), "Output shape should match input image"
        assert result["mask_source"] == "skull_stripping"

    def test_atlas_shape_scenario(self, temp_dir):
        """Simulate atlas registration scenario: image (240,240,155), mask (225,225,225)."""
        from mengrowth.preprocessing.src.normalization.zscore import ZScoreNormalizer

        img_path, mask_path = self._create_mismatched_image_and_mask(
            temp_dir, img_shape=(240, 240, 155), mask_shape=(225, 225, 225)
        )
        output_path = temp_dir / "normalized.nii.gz"

        normalizer = ZScoreNormalizer(config={"norm_value": 1.0})
        result = normalizer.execute(
            img_path, output_path, allow_overwrite=True, mask_path=mask_path
        )

        out_data = nib.load(str(output_path)).get_fdata()
        assert out_data.shape == (240, 240, 155), (
            "Output should match atlas-space image shape"
        )

    def test_mask_resampling_preserves_binary(self, temp_dir):
        """After BSpline zoom + threshold, mask should be strictly binary."""
        from scipy.ndimage import zoom

        # Create a binary mask
        mask = np.zeros((48, 48, 48), dtype=np.float32)
        center = np.array([24, 24, 24])
        coords = np.mgrid[0:48, 0:48, 0:48]
        dist = np.sqrt(sum((coords[i] - center[i]) ** 2 for i in range(3)))
        mask[dist < 15] = 1.0

        # Resample with BSpline order=3 + threshold
        factors = (64 / 48, 64 / 48, 64 / 48)
        resampled = zoom(mask, factors, order=3) > 0.5

        # Check binary
        unique_values = np.unique(resampled)
        assert set(unique_values).issubset({True, False}), (
            f"Mask should be binary, got {unique_values}"
        )
        assert resampled.shape == (64, 64, 64)
        # Should have some True values (brain) and some False (background)
        assert resampled.sum() > 0, "Resampled mask should have some True voxels"
        assert resampled.sum() < resampled.size, (
            "Resampled mask should have some False voxels"
        )
