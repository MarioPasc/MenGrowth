"""Tests for :mod:`mengrowth.synthseg.analysis.region_metrics`.

Exercises the pure-array helpers in isolation: boundary shell, within-
region CV, and boundary-gradient energy.
"""

from __future__ import annotations

import numpy as np
import pytest

from mengrowth.synthseg.analysis.region_metrics import (
    boundary_gradient_energy,
    compute_gradient_magnitude,
    region_boundary_mask,
    within_region_cv,
)


def test_boundary_shell_equals_mask_xor_erosion() -> None:
    mask = np.zeros((6, 6, 6), dtype=bool)
    mask[1:5, 1:5, 1:5] = True  # 4x4x4 cube interior surface

    boundary = region_boundary_mask(mask)
    # Interior voxel (3,3,3) cannot be on boundary.
    assert not boundary[3, 3, 3]
    # Corner (1,1,1) is on boundary.
    assert boundary[1, 1, 1]
    # Boundary is a subset of the mask.
    assert np.all(boundary[~mask] == False)  # noqa: E712


def test_within_region_cv_constant_is_zero() -> None:
    img = np.full((8, 8, 8), 100.0, dtype=np.float32)
    mask = np.zeros_like(img, dtype=bool)
    mask[2:6, 2:6, 2:6] = True
    assert within_region_cv(img, mask) == pytest.approx(0.0, abs=1e-6)


def test_within_region_cv_empty_mask_is_nan() -> None:
    img = np.zeros((4, 4, 4), dtype=np.float32)
    mask = np.zeros_like(img, dtype=bool)
    assert np.isnan(within_region_cv(img, mask))


def test_gradient_on_constant_image_is_zero() -> None:
    img = np.full((8, 8, 8), 42.0, dtype=np.float32)
    grad = compute_gradient_magnitude(img)
    assert np.allclose(grad, 0.0, atol=1e-5)


def test_gradient_on_step_image_is_positive_at_step() -> None:
    img = np.zeros((8, 8, 8), dtype=np.float32)
    img[4:, :, :] = 100.0  # sharp step along x=4
    grad = compute_gradient_magnitude(img)
    # On the step face, gradient magnitude must exceed zero.
    assert grad[3:5, 4, 4].max() > 0
    # Far from the step the gradient vanishes.
    assert grad[0, 0, 0] == pytest.approx(0.0, abs=1e-5)


def test_boundary_gradient_energy_nonzero_on_contrasting_region() -> None:
    img = np.zeros((8, 8, 8), dtype=np.float32)
    img[2:6, 2:6, 2:6] = 100.0

    mask = np.zeros_like(img, dtype=bool)
    mask[2:6, 2:6, 2:6] = True

    grad = compute_gradient_magnitude(img)
    energy = boundary_gradient_energy(grad, mask)
    assert energy > 0


def test_boundary_gradient_energy_on_constant_is_zero() -> None:
    img = np.full((8, 8, 8), 100.0, dtype=np.float32)
    mask = np.zeros_like(img, dtype=bool)
    mask[2:6, 2:6, 2:6] = True
    grad = compute_gradient_magnitude(img)
    assert boundary_gradient_energy(grad, mask) == pytest.approx(0.0, abs=1e-5)
