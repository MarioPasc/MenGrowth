"""Tests for reference selection quality metrics.

Tests use real preprocessed data from /tmp/mengrowth_preprocess/ when available,
and fall back to synthetic data otherwise. All real-data tests are skipped
if the expected data directory does not exist.
"""

from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from mengrowth.preprocessing.src.registration.reference_selection import (
    ReferenceSelectionConfig,
    ReferenceSelector,
)

# Paths to real preprocessed data
_OUTPUT_ROOT = Path("/tmp/mengrowth_preprocess/output")
_ARTIFACTS_ROOT = Path("/tmp/mengrowth_preprocess/artifacts")
_PATIENT_ID = "MenGrowth-0004"
_STUDY_ID = "MenGrowth-0004-000"
_STUDY_DIR = _OUTPUT_ROOT / _PATIENT_ID / _STUDY_ID
_ARTIFACTS_DIR = _ARTIFACTS_ROOT / _PATIENT_ID / _STUDY_ID
_MASK_PATH = _ARTIFACTS_DIR / "consensus_brain_mask.nii.gz"
_IMAGE_PATH = _STUDY_DIR / "t1c.nii.gz"

_REAL_DATA_AVAILABLE = _IMAGE_PATH.exists() and _MASK_PATH.exists()
_skip_no_data = pytest.mark.skipif(
    not _REAL_DATA_AVAILABLE,
    reason="Real preprocessing data not found at /tmp/mengrowth_preprocess/",
)

ALL_SIX_METRICS = [
    "snr_foreground",
    "cnr_high_low",
    "boundary_gradient_score",
    "brain_coverage_fraction",
    "laplacian_sharpness",
    "ghosting_score",
]


def _make_selector(metrics: list[str] | None = None) -> ReferenceSelector:
    """Create a ReferenceSelector with the specified metrics."""
    config = ReferenceSelectionConfig(
        method="quality_based",
        quality_metrics=metrics or ALL_SIX_METRICS,
    )
    return ReferenceSelector(config=config, verbose=True)


# ---- Real data tests -------------------------------------------------------


@_skip_no_data
def test_brain_coverage_fraction_from_real_mask() -> None:
    """Load consensus_brain_mask from MenGrowth-0004-000, verify 0 < result < 1."""
    selector = _make_selector()
    val = selector._compute_brain_coverage_fraction(_ARTIFACTS_DIR)
    assert not np.isnan(val), "brain_coverage_fraction returned NaN"
    assert 0.0 < val < 1.0, f"Expected 0 < val < 1, got {val}"


@_skip_no_data
def test_laplacian_sharpness_from_real_image() -> None:
    """Load t1c + mask from MenGrowth-0004-000, verify positive float."""
    selector = _make_selector()
    arr = sitk.GetArrayFromImage(sitk.ReadImage(str(_IMAGE_PATH)))
    mask = sitk.GetArrayFromImage(sitk.ReadImage(str(_MASK_PATH))) > 0
    val = selector._compute_laplacian_sharpness(arr, mask)
    assert not np.isnan(val), "laplacian_sharpness returned NaN"
    assert val > 0.0, f"Expected positive value, got {val}"


@_skip_no_data
def test_ghosting_score_from_real_image() -> None:
    """Load t1c + mask from MenGrowth-0004-000, verify reasonable range."""
    selector = _make_selector()
    arr = sitk.GetArrayFromImage(sitk.ReadImage(str(_IMAGE_PATH)))
    mask = sitk.GetArrayFromImage(sitk.ReadImage(str(_MASK_PATH))) > 0
    val = selector._compute_ghosting_score(arr, mask)
    assert not np.isnan(val), "ghosting_score returned NaN"
    # Score should be roughly in [-1, 2] range for real data
    assert -2.0 < val < 2.0, f"Ghosting score out of reasonable range: {val}"


@_skip_no_data
def test_all_six_metrics_computed() -> None:
    """Run _compute_metrics_from_image with all 6 metrics, verify all present."""
    selector = _make_selector()
    metrics = selector._compute_metrics_from_image(
        _IMAGE_PATH,
        artifacts_base=_ARTIFACTS_ROOT / _PATIENT_ID,
        study_dir=_STUDY_DIR,
    )
    for metric_name in ALL_SIX_METRICS:
        assert metric_name in metrics, (
            f"Metric '{metric_name}' missing from computed metrics. "
            f"Got: {list(metrics.keys())}"
        )


@_skip_no_data
def test_quality_ranking_with_new_metrics() -> None:
    """Run _select_quality_based on MenGrowth-0004 (3 studies), verify ranking."""
    patient_dir = _OUTPUT_ROOT / _PATIENT_ID
    study_dirs = sorted(patient_dir.iterdir())
    assert len(study_dirs) >= 2, f"Expected >=2 studies, found {len(study_dirs)}"

    selector = _make_selector()
    timestamps = [d.name.split("-")[-1] for d in study_dirs]

    selected, info = selector._select_quality_based(
        study_dirs=study_dirs,
        timestamps=timestamps,
        modalities=["t1c"],
        qc_metrics_path=None,
        artifacts_base=_ARTIFACTS_ROOT / _PATIENT_ID,
    )

    assert selected in timestamps
    assert "ranking" in info
    assert len(info["ranking"]) == len(study_dirs)


@_skip_no_data
def test_metrics_differ_across_timestamps() -> None:
    """Verify at least some new metrics differ between studies."""
    patient_dir = _OUTPUT_ROOT / _PATIENT_ID
    study_dirs = sorted(patient_dir.iterdir())
    assert len(study_dirs) >= 2

    selector = _make_selector()
    all_metrics = {}
    for study_dir in study_dirs:
        ts = study_dir.name.split("-")[-1]
        all_metrics[ts] = selector._compute_metrics_from_image(
            study_dir / "t1c.nii.gz",
            artifacts_base=_ARTIFACTS_ROOT / _PATIENT_ID,
            study_dir=study_dir,
        )

    # At least one of the new metrics should differ across timestamps
    new_metrics = ["brain_coverage_fraction", "laplacian_sharpness", "ghosting_score"]
    timestamps = list(all_metrics.keys())
    any_differ = False
    for metric in new_metrics:
        values = [
            all_metrics[ts].get(metric)
            for ts in timestamps
            if metric in all_metrics[ts]
        ]
        if len(values) >= 2 and len(set(values)) > 1:
            any_differ = True
            break

    assert any_differ, (
        "No new metrics differed across timestamps — unexpected for real data"
    )


# ---- Synthetic / fallback tests --------------------------------------------


def test_graceful_fallback_no_mask() -> None:
    """Synthetic image, no artifacts_base → new metrics use nonzero_mask fallback."""
    import tempfile

    selector = _make_selector()

    # Create a small synthetic image with some structure
    arr = np.zeros((32, 32, 32), dtype=np.float32)
    arr[8:24, 8:24, 8:24] = np.random.RandomState(42).normal(100, 20, (16, 16, 16))
    arr = np.clip(arr, 0, None)

    image = sitk.GetImageFromArray(arr)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.nii.gz"
        sitk.WriteImage(image, str(path))

        # No artifacts_base → should still compute laplacian and ghosting via nonzero fallback
        metrics = selector._compute_metrics_from_image(path)

    # Traditional metrics should be present
    assert "snr_foreground" in metrics
    # New metrics that don't need brain mask should use nonzero fallback
    assert "laplacian_sharpness" in metrics
    assert "ghosting_score" in metrics
    # brain_coverage_fraction needs artifacts, so it should be missing
    assert "brain_coverage_fraction" not in metrics


def test_normalize_new_metrics() -> None:
    """Verify _normalize_metric returns [0,1] for each new metric with typical values."""
    selector = _make_selector()

    test_cases = {
        "brain_coverage_fraction": [0.01, 0.10, 0.25, 0.40, 0.60],
        "laplacian_sharpness": [10.0, 100.0, 1000.0, 5000.0, 10000.0],
        "ghosting_score": [-0.5, 0.0, 0.5, 0.8, 1.0, 1.5],
    }

    for metric_name, values in test_cases.items():
        for val in values:
            normalized = selector._normalize_metric(metric_name, val)
            assert 0.0 <= normalized <= 1.0, (
                f"_normalize_metric('{metric_name}', {val}) = {normalized}, "
                f"expected [0, 1]"
            )
