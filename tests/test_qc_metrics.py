"""Unit tests for QC metrics computation functions."""

import pytest
import numpy as np
import SimpleITK as sitk
from pathlib import Path
import tempfile

from mengrowth.preprocessing.quality_analysis.qc_metrics import (
    downsample_image_for_qc,
    compute_geometry_metrics,
    _compute_nmi,
    _compute_ncc,
    detect_outliers_mad,
    detect_outliers_iqr,
    compute_intensity_stats_for_wasserstein,
)
from mengrowth.preprocessing.src.config import QCGeometryMetricsConfig, QCIntensityStabilityConfig


class TestDownsampling:
    """Test downsampling utilities."""

    def test_downsample_image_reduces_voxel_count(self):
        """Test that downsampling reduces voxel count."""
        # Create 256x256x180 image with 1mm spacing
        image = sitk.Image(256, 256, 180, sitk.sitkFloat32)
        image.SetSpacing([1.0, 1.0, 1.0])
        image.SetOrigin([0, 0, 0])
        image.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])

        # Fill with random data
        arr = np.random.rand(180, 256, 256).astype(np.float32)
        image = sitk.GetImageFromArray(arr)
        image.SetSpacing([1.0, 1.0, 1.0])

        downsampled, factor = downsample_image_for_qc(image, target_mm=2.0)

        assert downsampled.GetSize() == (128, 128, 90)
        assert factor == pytest.approx(2.0, rel=0.1)

    def test_downsample_respects_max_voxels(self):
        """Test that max_voxels cap works."""
        # Create large image
        image = sitk.Image(512, 512, 360, sitk.sitkFloat32)
        image.SetSpacing([1.0, 1.0, 1.0])

        # Force small max_voxels
        downsampled, factor = downsample_image_for_qc(
            image,
            target_mm=2.0,
            max_voxels=100000  # Much smaller than 256*256*180
        )

        total_voxels = np.prod(downsampled.GetSize())
        assert total_voxels <= 100000


class TestGeometryMetrics:
    """Test geometry metric computation."""

    def test_geometry_metrics_spacing(self):
        """Test spacing extraction."""
        image = sitk.Image(100, 100, 100, sitk.sitkFloat32)
        image.SetSpacing([1.0, 1.0, 1.5])

        config = QCGeometryMetricsConfig()
        metrics = compute_geometry_metrics(image, config)

        assert "spacing_x" in metrics
        assert metrics["spacing_x"] == 1.0
        assert metrics["spacing_z"] == 1.5
        assert "spacing_max_diff" in metrics
        assert metrics["spacing_max_diff"] == pytest.approx(0.5)

    def test_geometry_metrics_affine_det(self):
        """Test affine determinant computation."""
        image = sitk.Image(100, 100, 100, sitk.sitkFloat32)
        image.SetSpacing([1.0, 1.0, 1.0])
        image.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])  # Identity

        config = QCGeometryMetricsConfig()
        metrics = compute_geometry_metrics(image, config)

        assert "affine_det" in metrics
        # Determinant should be positive for valid orientation
        assert abs(metrics["affine_det"]) > 0


class TestRegistrationSimilarity:
    """Test registration similarity metrics."""

    def test_nmi_identical_images(self):
        """Test NMI between identical images."""
        # Generate random image
        fixed = np.random.rand(1000) * 255
        moving_identical = fixed.copy()

        nmi = _compute_nmi(fixed, moving_identical)

        # Identical images should have NMI close to 1
        assert nmi > 0.99

    def test_nmi_random_images(self):
        """Test NMI between uncorrelated images."""
        fixed = np.random.rand(1000) * 255
        moving_random = np.random.rand(1000) * 255

        nmi = _compute_nmi(fixed, moving_random)

        # Random images should have low NMI
        assert nmi < 0.5

    def test_ncc_identical_images(self):
        """Test NCC between identical images."""
        fixed = np.random.rand(1000) * 255
        moving_identical = fixed.copy()

        ncc = _compute_ncc(fixed, moving_identical)

        # Identical images should have NCC close to 1
        assert ncc > 0.99

    def test_ncc_anticorrelated_images(self):
        """Test NCC between negatively correlated images."""
        fixed = np.random.rand(1000) * 255
        moving_neg = -fixed

        ncc = _compute_ncc(fixed, moving_neg)

        # Negatively correlated should have NCC close to -1
        assert ncc < -0.9


class TestOutlierDetection:
    """Test outlier detection methods."""

    def test_mad_outlier_detection(self):
        """Test MAD outlier detection."""
        # Create data with one clear outlier
        values = np.array([1, 2, 2, 3, 3, 3, 2, 2, 100])
        outliers = detect_outliers_mad(values, threshold=3.0)

        # The value 100 should be detected as outlier
        assert outliers[-1] == True
        # Most other values should not be outliers
        assert np.sum(outliers) <= 2  # Allow for some edge cases

    def test_iqr_outlier_detection(self):
        """Test IQR outlier detection."""
        # Create data with outliers
        values = np.array([10, 11, 12, 13, 14, 15, 16, 100, 200])
        outliers = detect_outliers_iqr(values, multiplier=1.5)

        # The extreme values should be detected
        assert outliers[-1] == True
        assert outliers[-2] == True

    def test_mad_no_outliers_normal_data(self):
        """Test MAD on normal distribution with no extreme outliers."""
        np.random.seed(42)
        values = np.random.randn(100)
        outliers = detect_outliers_mad(values, threshold=3.5)

        # Should detect very few outliers in normal data
        assert np.sum(outliers) < 5


class TestIntensityMetrics:
    """Test intensity stability metrics."""

    def test_intensity_stats_computation(self):
        """Test median and IQR computation."""
        # Create simple image with known statistics
        arr = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]).astype(np.float32)
        image = sitk.GetImageFromArray(arr)

        config = QCIntensityStabilityConfig()
        metrics, hist, bin_edges = compute_intensity_stats_for_wasserstein(image, None, config)

        assert "median" in metrics
        assert "iqr" in metrics
        assert metrics["median"] == pytest.approx(5.0)

    def test_histogram_generation(self):
        """Test histogram generation for Wasserstein."""
        # Create image with uniform distribution
        arr = np.random.rand(20, 20, 20).astype(np.float32) * 100
        image = sitk.GetImageFromArray(arr)

        config = QCIntensityStabilityConfig(histogram_bins=50)
        metrics, hist, bin_edges = compute_intensity_stats_for_wasserstein(image, None, config)

        # Check histogram is generated
        assert len(hist) == 50
        assert len(bin_edges) == 51  # n_bins + 1
        assert np.sum(hist) > 0  # Non-empty histogram


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_values_intensity_stats(self):
        """Test intensity stats with empty mask."""
        # Create image of all zeros
        arr = np.zeros((10, 10, 10)).astype(np.float32)
        image = sitk.GetImageFromArray(arr)

        config = QCIntensityStabilityConfig()
        metrics, hist, bin_edges = compute_intensity_stats_for_wasserstein(image, None, config)

        # Should handle empty data gracefully
        assert metrics["median"] is None
        assert metrics["iqr"] is None
        assert len(hist) == 0

    def test_mad_constant_values(self):
        """Test MAD with constant values (MAD = 0)."""
        values = np.array([5, 5, 5, 5, 5])
        outliers = detect_outliers_mad(values, threshold=3.0)

        # Should not crash and should not detect outliers
        assert np.sum(outliers) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
