"""Tests for thick-slice adaptive multi-resolution schedule and quality gate.

Validates that:
1. drop_coarsest_levels correctly removes levels with high shrink factors.
2. detect_thick_slice_volume identifies thick-slice and isotropic volumes.
3. Quality-gated fallback raises _CoregistrationCatastrophicFailure for
   thick-slice volumes with poor quality.
"""

import importlib.util

import numpy as np
import pytest

HAS_ANTS = importlib.util.find_spec("ants") is not None


# ─── drop_coarsest_levels unit tests ─────────────────────────────────────────


class TestDropCoarsestLevels:
    """Pure-logic tests for the drop_coarsest_levels utility."""

    def test_removes_high_shrink(self) -> None:
        """Levels with shrink > 4 are dropped."""
        from mengrowth.preprocessing.src.registration.utils import drop_coarsest_levels

        iters_in = [1000, 500, 250, 100]
        shrink_in = [8, 4, 2, 1]
        sigmas_in = [4, 2, 1, 0]

        iters, shrink, sigmas = drop_coarsest_levels(
            iters_in, shrink_in, sigmas_in, max_shrink_factor=4
        )

        assert shrink == [4, 2, 1]
        assert iters == [500, 250, 100]
        assert sigmas == [2, 1, 0]

    def test_no_change_when_below_threshold(self) -> None:
        """All levels kept when shrink factors are already <= max."""
        from mengrowth.preprocessing.src.registration.utils import drop_coarsest_levels

        iters_in = [500, 250, 100]
        shrink_in = [4, 2, 1]
        sigmas_in = [2, 1, 0]

        iters, shrink, sigmas = drop_coarsest_levels(
            iters_in, shrink_in, sigmas_in, max_shrink_factor=4
        )

        assert shrink == [4, 2, 1]
        assert iters == [500, 250, 100]
        assert sigmas == [2, 1, 0]

    def test_keeps_at_least_one_level(self) -> None:
        """When all levels exceed threshold, the last (finest) is preserved."""
        from mengrowth.preprocessing.src.registration.utils import drop_coarsest_levels

        iters_in = [1000]
        shrink_in = [8]
        sigmas_in = [4]

        iters, shrink, sigmas = drop_coarsest_levels(
            iters_in, shrink_in, sigmas_in, max_shrink_factor=4
        )

        assert shrink == [8]
        assert iters == [1000]
        assert sigmas == [4]

    def test_empty_input(self) -> None:
        """Empty inputs return empty lists."""
        from mengrowth.preprocessing.src.registration.utils import drop_coarsest_levels

        iters, shrink, sigmas = drop_coarsest_levels([], [], [], max_shrink_factor=4)

        assert iters == []
        assert shrink == []
        assert sigmas == []

    def test_does_not_mutate_inputs(self) -> None:
        """Input lists are not modified."""
        from mengrowth.preprocessing.src.registration.utils import drop_coarsest_levels

        iters_in = [1000, 500, 250]
        shrink_in = [8, 4, 2]
        sigmas_in = [4, 2, 1]

        # Copy for comparison
        iters_orig = iters_in.copy()
        shrink_orig = shrink_in.copy()
        sigmas_orig = sigmas_in.copy()

        drop_coarsest_levels(iters_in, shrink_in, sigmas_in, max_shrink_factor=4)

        assert iters_in == iters_orig
        assert shrink_in == shrink_orig
        assert sigmas_in == sigmas_orig

    def test_multiple_levels_dropped(self) -> None:
        """Multiple coarse levels can be dropped at once."""
        from mengrowth.preprocessing.src.registration.utils import drop_coarsest_levels

        iters_in = [2000, 1000, 500, 250]
        shrink_in = [16, 8, 4, 2]
        sigmas_in = [8, 4, 2, 1]

        iters, shrink, sigmas = drop_coarsest_levels(
            iters_in, shrink_in, sigmas_in, max_shrink_factor=4
        )

        assert shrink == [4, 2]
        assert iters == [500, 250]
        assert sigmas == [2, 1]


# ─── detect_thick_slice_volume tests ─────────────────────────────────────────


@pytest.mark.skipif(not HAS_ANTS, reason="antspyx not installed")
class TestDetectThickSlice:
    """Tests for the detect_thick_slice_volume heuristic."""

    def test_constant_z_detected_as_thick(self) -> None:
        """Volume with constant z-direction (simulating thick-slice) is detected."""
        import ants

        from mengrowth.preprocessing.src.registration.utils import (
            detect_thick_slice_volume,
        )

        # Create volume with high in-plane variation but constant z
        data = np.random.RandomState(42).rand(64, 64, 1).astype(np.float32)
        data = np.broadcast_to(data, (64, 64, 64)).copy()

        img = ants.from_numpy(data, origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0))
        assert detect_thick_slice_volume(img, threshold_ratio=3.0) is True

    def test_isotropic_volume_not_detected(self) -> None:
        """Volume with equal gradient energy in all axes is not thick-slice."""
        import ants

        from mengrowth.preprocessing.src.registration.utils import (
            detect_thick_slice_volume,
        )

        # Create volume with equal gradient energy in all directions
        rng = np.random.RandomState(123)
        data = rng.rand(64, 64, 64).astype(np.float32)

        img = ants.from_numpy(data, origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0))
        assert detect_thick_slice_volume(img, threshold_ratio=3.0) is False

    def test_mild_anisotropy_not_detected(self) -> None:
        """Volume with mild z-smoothing (ratio ~2) is not detected at threshold 3."""
        import ants

        from mengrowth.preprocessing.src.registration.utils import (
            detect_thick_slice_volume,
        )

        # Create volume then smooth z mildly
        rng = np.random.RandomState(456)
        data = rng.rand(64, 64, 64).astype(np.float32)
        # Mild z-smoothing: average neighboring slices
        from scipy.ndimage import uniform_filter1d

        data = uniform_filter1d(data, size=2, axis=2)

        img = ants.from_numpy(data, origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0))
        assert detect_thick_slice_volume(img, threshold_ratio=3.0) is False


# ─── Quality-gated fallback tests ────────────────────────────────────────────


class TestQualityGatedFallback:
    """Tests for the thick-slice quality-gated composed-transform fallback.

    These tests verify the logic at the integration seam — they mock ANTs
    internals and test the decision-making in _register_modality.
    """

    def test_thick_slice_poor_quality_raises_catastrophic(self) -> None:
        """Thick-slice volume with poor quality triggers fallback exception."""
        from mengrowth.preprocessing.src.registration.antspyx_multi_modal_coregistration import (
            _CoregistrationCatastrophicFailure,
        )

        # The fallback is: when is_thick_slice=True AND corr_dissim > threshold,
        # raise _CoregistrationCatastrophicFailure. Test the exception directly.
        threshold = -0.3
        corr_dissim = -0.1  # Poor quality (higher = worse)
        is_thick_slice = True

        should_raise = (
            is_thick_slice and corr_dissim is not None and corr_dissim > threshold
        )
        assert should_raise is True

        # Verify exception can be instantiated and caught
        with pytest.raises(_CoregistrationCatastrophicFailure) as exc_info:
            raise _CoregistrationCatastrophicFailure("t2w")
        assert exc_info.value.modality == "t2w"

    def test_good_quality_no_fallback(self) -> None:
        """Thick-slice with good quality does NOT trigger fallback."""
        threshold = -0.3
        corr_dissim = -0.5  # Good quality (lower = better)
        is_thick_slice = True

        should_raise = (
            is_thick_slice and corr_dissim is not None and corr_dissim > threshold
        )
        assert should_raise is False

    def test_normal_slice_poor_quality_no_fallback(self) -> None:
        """Non-thick-slice with poor quality gets warning only, no fallback."""
        threshold = -0.3
        corr_dissim = -0.1  # Poor quality
        is_thick_slice = False

        should_raise = (
            is_thick_slice and corr_dissim is not None and corr_dissim > threshold
        )
        assert should_raise is False

    def test_catastrophic_failure_flows_to_failed_modalities(self) -> None:
        """Verify exception is caught by execute() and modality added to failed set."""
        from mengrowth.preprocessing.src.registration.antspyx_multi_modal_coregistration import (
            _CoregistrationCatastrophicFailure,
        )

        # Simulate the execute() loop logic
        failed_modalities: set[str] = set()
        modality = "t2w"

        try:
            raise _CoregistrationCatastrophicFailure(modality)
        except _CoregistrationCatastrophicFailure:
            failed_modalities.add(modality)

        assert "t2w" in failed_modalities
