"""Tests for longitudinal brain mask harmonization fix.

Validates that:
1. _warp_artifact_through_longitudinal_transform correctly applies transforms.
2. compute_and_apply_union_brain_mask warps masks AND pre-stripped backups
   before union when transforms are provided.
3. The volume-spread threshold triggers at 2%.
4. Pre-stripped backups are warped with BSpline (not NN).
5. Reference study masks are NOT warped.
6. Backwards compatibility: when no transforms are provided, behavior unchanged.
7. Warped intermediates are cleaned up unless save_intermediates=True.
"""

import importlib.util
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

HAS_ANTS = importlib.util.find_spec("ants") is not None


def _make_nifti(data: np.ndarray, affine: np.ndarray, path: Path) -> None:
    """Save a numpy array as a NIfTI file."""
    nii = nib.Nifti1Image(data.astype(np.float32), affine)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nii, str(path))


def _sphere_mask(
    shape: tuple[int, ...], center: tuple[int, ...], radius: int
) -> np.ndarray:
    """Create a binary sphere mask."""
    grid = np.mgrid[tuple(slice(0, s) for s in shape)]
    dist = np.sqrt(sum((g - c) ** 2 for g, c in zip(grid, center)))
    return (dist <= radius).astype(np.uint8)


# ─── Helper function tests ───────────────────────────────────────────────────


@pytest.mark.skipif(not HAS_ANTS, reason="antspyx not installed")
class TestWarpArtifactHelper:
    """Tests for _warp_artifact_through_longitudinal_transform."""

    def test_warp_mask_nearest_neighbor(self, tmp_path: Path) -> None:
        """Mask warping uses nearestNeighbor and produces binary output."""
        import ants

        from mengrowth.preprocessing.src.steps.skull_stripping import (
            _warp_artifact_through_longitudinal_transform,
        )

        # Create a simple binary mask
        mask_data = np.zeros((64, 64, 64), dtype=np.float32)
        mask_data[20:40, 20:40, 20:40] = 1.0

        mask_img = ants.from_numpy(
            mask_data, origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0)
        )
        mask_path = tmp_path / "mask.nii.gz"
        ants.image_write(mask_img, str(mask_path))

        # Create identity transform (no displacement)
        identity_tx = ants.create_ants_transform(
            transform_type="AffineTransform", dimension=3
        )
        tx_path = tmp_path / "identity.h5"
        ants.write_transform(identity_tx, str(tx_path))

        # Create reference image
        ref_data = np.ones((64, 64, 64), dtype=np.float32)
        ref_img = ants.from_numpy(
            ref_data, origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0)
        )
        ref_path = tmp_path / "ref.nii.gz"
        ants.image_write(ref_img, str(ref_path))

        output_path = tmp_path / "warped_mask.nii.gz"

        result = _warp_artifact_through_longitudinal_transform(
            artifact_path=mask_path,
            transform_path=tx_path,
            reference_path=ref_path,
            output_path=output_path,
            is_mask=True,
        )

        assert result is True
        assert output_path.exists()

        # Verify output is binary (only 0 and 1 from NN interpolation)
        warped = nib.load(str(output_path)).get_fdata()
        unique_vals = np.unique(warped)
        assert all(v in [0.0, 1.0] for v in unique_vals)

    def test_warp_intensity_bspline(self, tmp_path: Path) -> None:
        """Intensity warping uses BSpline and produces non-integer values."""
        import ants

        from mengrowth.preprocessing.src.steps.skull_stripping import (
            _warp_artifact_through_longitudinal_transform,
        )

        # Create a continuous-valued volume with gradient
        x = np.linspace(0, 1, 64)
        intensity_data = (x[:, None, None] * x[None, :, None] * 100).astype(np.float32)
        intensity_data = np.broadcast_to(intensity_data, (64, 64, 64)).copy()

        img = ants.from_numpy(
            intensity_data, origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0)
        )
        img_path = tmp_path / "intensity.nii.gz"
        ants.image_write(img, str(img_path))

        # Create a small translation transform
        tx = ants.create_ants_transform(transform_type="AffineTransform", dimension=3)
        params = list(tx.parameters)
        # Set a small translation (last 3 params)
        params[-3] = 1.5  # 1.5mm translation in x
        tx.set_parameters(params)
        tx_path = tmp_path / "translate.h5"
        ants.write_transform(tx, str(tx_path))

        ref = ants.from_numpy(
            np.ones((64, 64, 64), dtype=np.float32),
            origin=(0.0, 0.0, 0.0),
            spacing=(1.0, 1.0, 1.0),
        )
        ref_path = tmp_path / "ref.nii.gz"
        ants.image_write(ref, str(ref_path))

        output_path = tmp_path / "warped_intensity.nii.gz"

        result = _warp_artifact_through_longitudinal_transform(
            artifact_path=img_path,
            transform_path=tx_path,
            reference_path=ref_path,
            output_path=output_path,
            is_mask=False,
        )

        assert result is True
        assert output_path.exists()

        # BSpline should produce non-integer values due to interpolation
        warped = nib.load(str(output_path)).get_fdata()
        nonzero = warped[warped > 0]
        if len(nonzero) > 0:
            fractional = nonzero - np.floor(nonzero)
            # At least some values should have fractional parts from BSpline
            assert np.any(fractional > 1e-6), (
                "BSpline should produce non-integer values"
            )

    def test_missing_artifact_returns_false(self, tmp_path: Path) -> None:
        """Returns False when artifact file does not exist."""
        from mengrowth.preprocessing.src.steps.skull_stripping import (
            _warp_artifact_through_longitudinal_transform,
        )

        result = _warp_artifact_through_longitudinal_transform(
            artifact_path=tmp_path / "nonexistent.nii.gz",
            transform_path=tmp_path / "tx.h5",
            reference_path=tmp_path / "ref.nii.gz",
            output_path=tmp_path / "out.nii.gz",
            is_mask=True,
        )
        assert result is False

    def test_ants_exception_returns_false(self, tmp_path: Path) -> None:
        """Returns False when ANTs raises an exception (never propagates)."""
        from mengrowth.preprocessing.src.steps.skull_stripping import (
            _warp_artifact_through_longitudinal_transform,
        )

        # Create artifact file but no valid transform
        dummy = np.zeros((4, 4, 4), dtype=np.float32)
        _make_nifti(dummy, np.eye(4), tmp_path / "artifact.nii.gz")
        # Create a bogus transform file
        (tmp_path / "bad_tx.h5").touch()
        _make_nifti(dummy, np.eye(4), tmp_path / "ref.nii.gz")

        result = _warp_artifact_through_longitudinal_transform(
            artifact_path=tmp_path / "artifact.nii.gz",
            transform_path=tmp_path / "bad_tx.h5",
            reference_path=tmp_path / "ref.nii.gz",
            output_path=tmp_path / "out.nii.gz",
            is_mask=True,
        )
        assert result is False


# ─── Union mask integration tests ────────────────────────────────────────────


def _setup_two_study_patient(
    tmp_path: Path,
    modality: str = "t1c",
    ref_mask_center: tuple[int, int, int] = (32, 32, 32),
    nonref_mask_center: tuple[int, int, int] = (32, 32, 32),
    mask_radius: int = 15,
    create_transforms_dir: bool = False,
    create_pre_stripped: bool = True,
) -> dict:
    """Set up a synthetic two-study patient directory structure.

    Returns a dict with all paths needed for testing.
    """
    patient_id = "MenGrowth-0001"
    ref_study = "MenGrowth-0001-000"
    nonref_study = "MenGrowth-0001-001"

    affine = np.eye(4)
    shape = (64, 64, 64)

    # Study output dirs (where preprocessed NIfTIs live)
    ref_output = tmp_path / "output" / patient_id / ref_study
    nonref_output = tmp_path / "output" / patient_id / nonref_study
    ref_output.mkdir(parents=True)
    nonref_output.mkdir(parents=True)

    # Artifacts dirs
    artifacts = tmp_path / "artifacts"
    ref_art = artifacts / patient_id / ref_study
    nonref_art = artifacts / patient_id / nonref_study
    ref_art.mkdir(parents=True)
    nonref_art.mkdir(parents=True)

    # Create masks
    ref_mask = _sphere_mask(shape, ref_mask_center, mask_radius)
    nonref_mask = _sphere_mask(shape, nonref_mask_center, mask_radius)
    _make_nifti(ref_mask, affine, ref_art / f"{modality}_brain_mask.nii.gz")
    _make_nifti(nonref_mask, affine, nonref_art / f"{modality}_brain_mask.nii.gz")

    # Create images (simple intensity inside mask)
    ref_img_data = ref_mask.astype(np.float32) * 100.0
    nonref_img_data = nonref_mask.astype(np.float32) * 80.0
    _make_nifti(ref_img_data, affine, ref_output / f"{modality}.nii.gz")
    _make_nifti(nonref_img_data, affine, nonref_output / f"{modality}.nii.gz")

    # Create pre-stripped backups (full-head data before skull stripping)
    if create_pre_stripped:
        full_head = np.random.RandomState(42).rand(*shape).astype(np.float32) * 200
        _make_nifti(full_head, affine, ref_art / f"{modality}_pre_stripped.nii.gz")
        _make_nifti(full_head, affine, nonref_art / f"{modality}_pre_stripped.nii.gz")

    # Create longitudinal transforms dir if requested
    transforms_dir = None
    if create_transforms_dir:
        transforms_dir = artifacts / patient_id / "longitudinal_registration"
        transforms_dir.mkdir(parents=True)
        # Only the non-reference study has a transform (reference has none)
        # We create a dummy .h5 file — tests that actually use ANTs will
        # need real transforms; mock-based tests just check for existence
        (transforms_dir / f"001_{modality}_to_ref.h5").touch()

    return {
        "patient_id": patient_id,
        "ref_study": ref_study,
        "nonref_study": nonref_study,
        "study_dirs": [ref_output, nonref_output],
        "artifacts_root": artifacts,
        "transforms_dir": transforms_dir,
        "ref_art": ref_art,
        "nonref_art": nonref_art,
        "modality": modality,
    }


class TestUnionMaskLogic:
    """Tests for compute_and_apply_union_brain_mask behavior."""

    def test_volume_spread_threshold_02_triggers(self, tmp_path: Path) -> None:
        """Union IS applied when volume spread is 3% (above 2% threshold) in always mode."""
        from mengrowth.preprocessing.src.steps.skull_stripping import (
            compute_and_apply_union_brain_mask,
        )

        setup = _setup_two_study_patient(tmp_path)
        mod = setup["modality"]

        # Create masks with ~3% volume spread (above 2% threshold)
        shape = (64, 64, 64)
        big_mask = _sphere_mask(shape, (32, 32, 32), 15)
        big_vol = int(big_mask.sum())

        # Smaller mask: ~3% fewer voxels
        small_mask = _sphere_mask(shape, (32, 32, 32), 14)
        small_vol = int(small_mask.sum())
        spread = (big_vol - small_vol) / big_vol
        # Verify our setup produces meaningful spread (may not be exactly 3%
        # but should be above 2%)
        assert spread > 0.02, f"Test setup issue: spread={spread:.1%} <= 2%"

        # Write masks
        _make_nifti(
            big_mask,
            np.eye(4),
            setup["artifacts_root"]
            / setup["patient_id"]
            / setup["ref_study"]
            / f"{mod}_brain_mask.nii.gz",
        )
        _make_nifti(
            small_mask,
            np.eye(4),
            setup["artifacts_root"]
            / setup["patient_id"]
            / setup["nonref_study"]
            / f"{mod}_brain_mask.nii.gz",
        )

        result = compute_and_apply_union_brain_mask(
            patient_id=setup["patient_id"],
            study_dirs=setup["study_dirs"],
            artifacts_root=setup["artifacts_root"],
            modalities=[mod],
            longitudinal_transforms_dir=None,
            union_mode="always",
        )

        assert result is True, "Union should be applied when spread > 2%"

    def test_volume_spread_threshold_02_skips(self, tmp_path: Path) -> None:
        """Union is NOT applied when volume spread is ~1% (below 2% threshold) in always mode."""
        from mengrowth.preprocessing.src.steps.skull_stripping import (
            compute_and_apply_union_brain_mask,
        )

        setup = _setup_two_study_patient(tmp_path)
        mod = setup["modality"]

        # Create nearly identical masks (spread < 2%)
        shape = (64, 64, 64)
        mask1 = _sphere_mask(shape, (32, 32, 32), 15)
        # Tiny difference: flip a few border voxels
        mask2 = mask1.copy()
        mask2[17, 32, 32] = 0  # Remove a few voxels
        mask2[17, 33, 32] = 0

        vol1 = int(mask1.sum())
        vol2 = int(mask2.sum())
        spread = (vol1 - vol2) / vol1
        assert spread < 0.02, f"Test setup issue: spread={spread:.1%} >= 2%"

        _make_nifti(
            mask1,
            np.eye(4),
            setup["artifacts_root"]
            / setup["patient_id"]
            / setup["ref_study"]
            / f"{mod}_brain_mask.nii.gz",
        )
        _make_nifti(
            mask2,
            np.eye(4),
            setup["artifacts_root"]
            / setup["patient_id"]
            / setup["nonref_study"]
            / f"{mod}_brain_mask.nii.gz",
        )

        result = compute_and_apply_union_brain_mask(
            patient_id=setup["patient_id"],
            study_dirs=setup["study_dirs"],
            artifacts_root=setup["artifacts_root"],
            modalities=[mod],
            longitudinal_transforms_dir=None,
            union_mode="always",
        )

        assert result is False, "Union should NOT be applied when spread < 2%"

    def test_backward_compat_no_transforms(self, tmp_path: Path) -> None:
        """When longitudinal_transforms_dir=None, no warping occurs."""
        from mengrowth.preprocessing.src.steps.skull_stripping import (
            compute_and_apply_union_brain_mask,
        )

        setup = _setup_two_study_patient(
            tmp_path, mask_radius=15, nonref_mask_center=(32, 32, 28)
        )
        mod = setup["modality"]

        # Create masks with significant volume spread to trigger union
        shape = (64, 64, 64)
        big_mask = _sphere_mask(shape, (32, 32, 32), 18)
        small_mask = _sphere_mask(shape, (32, 32, 32), 14)

        _make_nifti(
            big_mask,
            np.eye(4),
            setup["artifacts_root"]
            / setup["patient_id"]
            / setup["ref_study"]
            / f"{mod}_brain_mask.nii.gz",
        )
        _make_nifti(
            small_mask,
            np.eye(4),
            setup["artifacts_root"]
            / setup["patient_id"]
            / setup["nonref_study"]
            / f"{mod}_brain_mask.nii.gz",
        )

        result = compute_and_apply_union_brain_mask(
            patient_id=setup["patient_id"],
            study_dirs=setup["study_dirs"],
            artifacts_root=setup["artifacts_root"],
            modalities=[mod],
            longitudinal_transforms_dir=None,  # No transforms
            union_mode="always",
        )

        # Should still compute union (from unwarped masks)
        assert result is True

        # Verify no warped intermediates were created
        for study_id in [setup["ref_study"], setup["nonref_study"]]:
            art_dir = setup["artifacts_root"] / setup["patient_id"] / study_id
            warped_mask = art_dir / f"{mod}_brain_mask_warped.nii.gz"
            warped_pre = art_dir / f"{mod}_pre_stripped_warped.nii.gz"
            assert not warped_mask.exists(), (
                "No warped mask should exist without transforms"
            )
            assert not warped_pre.exists(), (
                "No warped pre-stripped should exist without transforms"
            )

    def test_reference_study_not_warped(self, tmp_path: Path) -> None:
        """Reference study (no transform file) mask is used as-is."""
        from mengrowth.preprocessing.src.steps.skull_stripping import (
            compute_and_apply_union_brain_mask,
        )

        setup = _setup_two_study_patient(tmp_path, create_transforms_dir=True)
        mod = setup["modality"]

        # Create masks with enough spread to trigger union
        shape = (64, 64, 64)
        big_mask = _sphere_mask(shape, (32, 32, 32), 18)
        small_mask = _sphere_mask(shape, (32, 32, 32), 14)

        _make_nifti(
            big_mask,
            np.eye(4),
            setup["artifacts_root"]
            / setup["patient_id"]
            / setup["ref_study"]
            / f"{mod}_brain_mask.nii.gz",
        )
        _make_nifti(
            small_mask,
            np.eye(4),
            setup["artifacts_root"]
            / setup["patient_id"]
            / setup["nonref_study"]
            / f"{mod}_brain_mask.nii.gz",
        )

        # The reference study (000) should NOT have a warped mask
        # (it has no transform file in the transforms dir)
        # The non-reference study (001) has a transform but it's a dummy .h5,
        # so warping will fail gracefully. The key test is that the reference
        # study's mask directory has NO warped files.

        compute_and_apply_union_brain_mask(
            patient_id=setup["patient_id"],
            study_dirs=setup["study_dirs"],
            artifacts_root=setup["artifacts_root"],
            modalities=[mod],
            longitudinal_transforms_dir=setup["transforms_dir"],
        )

        # Reference study should never have warped artifacts
        ref_art = setup["artifacts_root"] / setup["patient_id"] / setup["ref_study"]
        assert not (ref_art / f"{mod}_brain_mask_warped.nii.gz").exists()
        assert not (ref_art / f"{mod}_pre_stripped_warped.nii.gz").exists()

    @pytest.mark.skipif(not HAS_ANTS, reason="antspyx not installed")
    def test_union_warps_pre_stripped_for_nonref(self, tmp_path: Path) -> None:
        """Non-reference study pre-stripped backup is warped and used for recovery."""
        import ants

        from mengrowth.preprocessing.src.steps.skull_stripping import (
            compute_and_apply_union_brain_mask,
        )

        patient_id = "MenGrowth-0001"
        ref_study = "MenGrowth-0001-000"
        nonref_study = "MenGrowth-0001-001"
        mod = "t1c"
        shape = (64, 64, 64)
        affine = np.eye(4)

        # Setup dirs
        artifacts = tmp_path / "artifacts"
        ref_art = artifacts / patient_id / ref_study
        nonref_art = artifacts / patient_id / nonref_study
        ref_output = tmp_path / "output" / patient_id / ref_study
        nonref_output = tmp_path / "output" / patient_id / nonref_study
        for d in [ref_art, nonref_art, ref_output, nonref_output]:
            d.mkdir(parents=True)

        # Create identity transform for non-reference study
        transforms_dir = artifacts / patient_id / "longitudinal_registration"
        transforms_dir.mkdir(parents=True)
        identity_tx = ants.create_ants_transform(
            transform_type="AffineTransform", dimension=3
        )
        ants.write_transform(identity_tx, str(transforms_dir / f"001_{mod}_to_ref.h5"))

        # Reference: large mask
        big_mask = _sphere_mask(shape, (32, 32, 32), 18)
        # Non-reference: smaller mask (will gain voxels from union)
        small_mask = _sphere_mask(shape, (32, 32, 32), 14)

        _make_nifti(big_mask, affine, ref_art / f"{mod}_brain_mask.nii.gz")
        _make_nifti(small_mask, affine, nonref_art / f"{mod}_brain_mask.nii.gz")

        # Images (skull-stripped)
        _make_nifti(big_mask * 100.0, affine, ref_output / f"{mod}.nii.gz")
        _make_nifti(small_mask * 80.0, affine, nonref_output / f"{mod}.nii.gz")

        # Pre-stripped backups with known sentinel values
        ref_pre = np.ones(shape, dtype=np.float32) * 42.0
        nonref_pre = np.ones(shape, dtype=np.float32) * 99.0
        _make_nifti(ref_pre, affine, ref_art / f"{mod}_pre_stripped.nii.gz")
        _make_nifti(nonref_pre, affine, nonref_art / f"{mod}_pre_stripped.nii.gz")

        result = compute_and_apply_union_brain_mask(
            patient_id=patient_id,
            study_dirs=[ref_output, nonref_output],
            artifacts_root=artifacts,
            modalities=[mod],
            longitudinal_transforms_dir=transforms_dir,
            save_intermediates=True,  # Keep for inspection
            union_mode="always",
        )

        assert result is True

        # Non-reference study should have gained voxels
        nonref_img = nib.load(str(nonref_output / f"{mod}.nii.gz")).get_fdata()

        # The union mask includes the big_mask region. Voxels that were NOT
        # in small_mask but ARE in big_mask should have been recovered.
        new_voxels = big_mask.astype(bool) & ~small_mask.astype(bool)
        n_new = int(new_voxels.sum())
        assert n_new > 0, "Test setup: should have new voxels to recover"

        # Check that warped pre-stripped was created
        warped_ps = nonref_art / f"{mod}_pre_stripped_warped.nii.gz"
        assert warped_ps.exists(), "Warped pre-stripped should exist"

        # The recovered voxels should come from the warped pre-stripped (identity
        # transform, so values should be ~99.0 from our sentinel)
        recovered = nonref_img[new_voxels]
        assert np.allclose(recovered, 99.0, atol=1.0), (
            f"Recovered voxels should be from warped pre-stripped (~99.0), "
            f"got mean={recovered.mean():.1f}"
        )

    def test_cleanup_intermediates(self, tmp_path: Path) -> None:
        """Warped intermediate files are deleted when save_intermediates=False."""
        from mengrowth.preprocessing.src.steps.skull_stripping import (
            compute_and_apply_union_brain_mask,
        )

        setup = _setup_two_study_patient(tmp_path, create_transforms_dir=True)
        mod = setup["modality"]
        shape = (64, 64, 64)

        # Large spread to trigger union
        big_mask = _sphere_mask(shape, (32, 32, 32), 18)
        small_mask = _sphere_mask(shape, (32, 32, 32), 14)
        _make_nifti(
            big_mask,
            np.eye(4),
            setup["artifacts_root"]
            / setup["patient_id"]
            / setup["ref_study"]
            / f"{mod}_brain_mask.nii.gz",
        )
        _make_nifti(
            small_mask,
            np.eye(4),
            setup["artifacts_root"]
            / setup["patient_id"]
            / setup["nonref_study"]
            / f"{mod}_brain_mask.nii.gz",
        )

        compute_and_apply_union_brain_mask(
            patient_id=setup["patient_id"],
            study_dirs=setup["study_dirs"],
            artifacts_root=setup["artifacts_root"],
            modalities=[mod],
            longitudinal_transforms_dir=setup["transforms_dir"],
            save_intermediates=False,
        )

        # Warped intermediates should be cleaned up regardless of whether
        # warping succeeded or failed (dummy .h5 will fail gracefully)
        for study_id in [setup["ref_study"], setup["nonref_study"]]:
            art_dir = setup["artifacts_root"] / setup["patient_id"] / study_id
            assert not (art_dir / f"{mod}_brain_mask_warped.nii.gz").exists()
            assert not (art_dir / f"{mod}_pre_stripped_warped.nii.gz").exists()

    def test_single_study_returns_false(self, tmp_path: Path) -> None:
        """Returns False immediately when only 1 study exists."""
        from mengrowth.preprocessing.src.steps.skull_stripping import (
            compute_and_apply_union_brain_mask,
        )

        study_dir = tmp_path / "output" / "MenGrowth-0001" / "MenGrowth-0001-000"
        study_dir.mkdir(parents=True)

        result = compute_and_apply_union_brain_mask(
            patient_id="MenGrowth-0001",
            study_dirs=[study_dir],
            artifacts_root=tmp_path / "artifacts",
            modalities=["t1c"],
        )
        assert result is False


# ─── Protective mode tests ──────────────────────────────────────────────────


def _setup_three_study_patient(
    tmp_path: Path,
    modality: str = "t1c",
    radii: tuple[int, int, int] = (18, 14, 16),
) -> dict:
    """Set up a synthetic three-study patient with different mask sizes.

    Returns a dict with all paths needed for testing.
    """
    patient_id = "MenGrowth-0002"
    studies = [
        "MenGrowth-0002-000",
        "MenGrowth-0002-001",
        "MenGrowth-0002-002",
    ]
    affine = np.eye(4)
    shape = (64, 64, 64)

    study_dirs = []
    for study_id, radius in zip(studies, radii):
        out_dir = tmp_path / "output" / patient_id / study_id
        art_dir = tmp_path / "artifacts" / patient_id / study_id
        out_dir.mkdir(parents=True)
        art_dir.mkdir(parents=True)

        mask = _sphere_mask(shape, (32, 32, 32), radius)
        _make_nifti(mask, affine, art_dir / f"{modality}_brain_mask.nii.gz")
        _make_nifti(
            mask.astype(np.float32) * 100.0, affine, out_dir / f"{modality}.nii.gz"
        )
        # Pre-stripped backup (full head)
        full_head = np.random.RandomState(42).rand(*shape).astype(np.float32) * 200
        _make_nifti(full_head, affine, art_dir / f"{modality}_pre_stripped.nii.gz")

        study_dirs.append(out_dir)

    return {
        "patient_id": patient_id,
        "studies": studies,
        "study_dirs": study_dirs,
        "artifacts_root": tmp_path / "artifacts",
        "modality": modality,
    }


class TestProtectiveMaskUnion:
    """Tests for the protective mask union mode."""

    def test_protective_only_expands_deficit_studies(self, tmp_path: Path) -> None:
        """In protective mode, only the study with volume deficit gets expanded."""
        from mengrowth.preprocessing.src.steps.skull_stripping import (
            compute_and_apply_union_brain_mask,
        )

        setup = _setup_two_study_patient(tmp_path)
        mod = setup["modality"]
        shape = (64, 64, 64)

        # Big mask (ref study) and small mask (nonref study, ~30% deficit)
        big_mask = _sphere_mask(shape, (32, 32, 32), 18)
        small_mask = _sphere_mask(shape, (32, 32, 32), 14)
        big_vol = int(big_mask.sum())
        small_vol = int(small_mask.sum())
        deficit = 1.0 - small_vol / big_vol
        assert deficit > 0.05, f"Test setup: deficit={deficit:.1%} should be > 5%"

        _make_nifti(
            big_mask,
            np.eye(4),
            setup["artifacts_root"]
            / setup["patient_id"]
            / setup["ref_study"]
            / f"{mod}_brain_mask.nii.gz",
        )
        _make_nifti(
            small_mask,
            np.eye(4),
            setup["artifacts_root"]
            / setup["patient_id"]
            / setup["nonref_study"]
            / f"{mod}_brain_mask.nii.gz",
        )

        # Save original ref image data for comparison
        ref_img_before = (
            nib.load(str(setup["study_dirs"][0] / f"{mod}.nii.gz")).get_fdata().copy()
        )

        result = compute_and_apply_union_brain_mask(
            patient_id=setup["patient_id"],
            study_dirs=setup["study_dirs"],
            artifacts_root=setup["artifacts_root"],
            modalities=[mod],
            longitudinal_transforms_dir=None,
            union_mode="protective",
            protective_threshold=0.05,
        )

        assert result is True

        # The ref study (big mask) should NOT have been modified
        ref_img_after = nib.load(
            str(setup["study_dirs"][0] / f"{mod}.nii.gz")
        ).get_fdata()
        np.testing.assert_array_equal(
            ref_img_before,
            ref_img_after,
            err_msg="Ref study with adequate mask should NOT be modified",
        )

        # The nonref study (small mask) SHOULD have been expanded
        nonref_mask_after = nib.load(
            str(
                setup["artifacts_root"]
                / setup["patient_id"]
                / setup["nonref_study"]
                / f"{mod}_brain_mask.nii.gz"
            )
        ).get_fdata()
        expanded_vol = int((nonref_mask_after > 0).sum())
        assert expanded_vol > small_vol, (
            f"Nonref mask should have been expanded: {small_vol} -> {expanded_vol}"
        )

    def test_protective_uses_largest_mask_not_union(self, tmp_path: Path) -> None:
        """Protective mode uses the largest individual mask, not the OR union."""
        from mengrowth.preprocessing.src.steps.skull_stripping import (
            compute_and_apply_union_brain_mask,
        )

        setup = _setup_three_study_patient(tmp_path, radii=(18, 12, 16))
        mod = setup["modality"]
        shape = (64, 64, 64)

        # Study 000: r=18 (largest), Study 001: r=12 (deficit), Study 002: r=16
        # In protective mode, expansion source = Study 000's mask (largest)
        # NOT the union of all three

        result = compute_and_apply_union_brain_mask(
            patient_id=setup["patient_id"],
            study_dirs=setup["study_dirs"],
            artifacts_root=setup["artifacts_root"],
            modalities=[mod],
            longitudinal_transforms_dir=None,
            union_mode="protective",
            protective_threshold=0.05,
        )

        assert result is True

        # The deficit study (001, r=12) should be expanded to match the largest
        # mask (000, r=18), not the union of all three
        largest_mask = _sphere_mask(shape, (32, 32, 32), 18)
        largest_vol = int(largest_mask.sum())

        expanded_mask = nib.load(
            str(
                setup["artifacts_root"]
                / setup["patient_id"]
                / setup["studies"][1]
                / f"{mod}_brain_mask.nii.gz"
            )
        ).get_fdata()
        expanded_vol = int((expanded_mask > 0).sum())

        # Should be close to largest mask volume (OR of small + largest = largest)
        assert expanded_vol == largest_vol, (
            f"Expanded mask should match largest individual mask "
            f"({largest_vol}), got {expanded_vol}"
        )

    def test_protective_no_expansion_when_all_adequate(self, tmp_path: Path) -> None:
        """No expansion when all studies are within the protective threshold."""
        from mengrowth.preprocessing.src.steps.skull_stripping import (
            compute_and_apply_union_brain_mask,
        )

        # All masks identical — zero deficit
        setup = _setup_three_study_patient(tmp_path, radii=(15, 15, 15))
        mod = setup["modality"]

        result = compute_and_apply_union_brain_mask(
            patient_id=setup["patient_id"],
            study_dirs=setup["study_dirs"],
            artifacts_root=setup["artifacts_root"],
            modalities=[mod],
            longitudinal_transforms_dir=None,
            union_mode="protective",
            protective_threshold=0.05,
        )

        assert result is False, "No expansion should occur when all within threshold"

    def test_always_mode_backwards_compatible(self, tmp_path: Path) -> None:
        """'always' mode applies union to ALL studies (original behaviour)."""
        from mengrowth.preprocessing.src.steps.skull_stripping import (
            compute_and_apply_union_brain_mask,
        )

        setup = _setup_two_study_patient(tmp_path)
        mod = setup["modality"]
        shape = (64, 64, 64)

        big_mask = _sphere_mask(shape, (32, 32, 32), 18)
        small_mask = _sphere_mask(shape, (32, 32, 32), 14)

        _make_nifti(
            big_mask,
            np.eye(4),
            setup["artifacts_root"]
            / setup["patient_id"]
            / setup["ref_study"]
            / f"{mod}_brain_mask.nii.gz",
        )
        _make_nifti(
            small_mask,
            np.eye(4),
            setup["artifacts_root"]
            / setup["patient_id"]
            / setup["nonref_study"]
            / f"{mod}_brain_mask.nii.gz",
        )

        result = compute_and_apply_union_brain_mask(
            patient_id=setup["patient_id"],
            study_dirs=setup["study_dirs"],
            artifacts_root=setup["artifacts_root"],
            modalities=[mod],
            longitudinal_transforms_dir=None,
            union_mode="always",
        )

        assert result is True

        # In "always" mode, BOTH studies get the union mask applied.
        # The ref study mask should now equal the union (= big_mask since
        # big_mask is a superset of small_mask).
        ref_mask_after = nib.load(
            str(
                setup["artifacts_root"]
                / setup["patient_id"]
                / setup["ref_study"]
                / f"{mod}_brain_mask.nii.gz"
            )
        ).get_fdata()
        union_expected = (big_mask | small_mask).astype(np.uint8)
        np.testing.assert_array_equal(
            (ref_mask_after > 0).astype(np.uint8),
            union_expected,
            err_msg="In 'always' mode, ref study should receive union mask",
        )

    def test_disabled_mode_skips(self, tmp_path: Path) -> None:
        """'disabled' mode returns False without modifying anything."""
        from mengrowth.preprocessing.src.steps.skull_stripping import (
            compute_and_apply_union_brain_mask,
        )

        setup = _setup_two_study_patient(tmp_path)
        mod = setup["modality"]
        shape = (64, 64, 64)

        big_mask = _sphere_mask(shape, (32, 32, 32), 18)
        small_mask = _sphere_mask(shape, (32, 32, 32), 14)

        _make_nifti(
            big_mask,
            np.eye(4),
            setup["artifacts_root"]
            / setup["patient_id"]
            / setup["ref_study"]
            / f"{mod}_brain_mask.nii.gz",
        )
        _make_nifti(
            small_mask,
            np.eye(4),
            setup["artifacts_root"]
            / setup["patient_id"]
            / setup["nonref_study"]
            / f"{mod}_brain_mask.nii.gz",
        )

        result = compute_and_apply_union_brain_mask(
            patient_id=setup["patient_id"],
            study_dirs=setup["study_dirs"],
            artifacts_root=setup["artifacts_root"],
            modalities=[mod],
            longitudinal_transforms_dir=None,
            union_mode="disabled",
        )

        assert result is False

        # Masks should be untouched
        ref_mask = nib.load(
            str(
                setup["artifacts_root"]
                / setup["patient_id"]
                / setup["ref_study"]
                / f"{mod}_brain_mask.nii.gz"
            )
        ).get_fdata()
        np.testing.assert_array_equal(
            (ref_mask > 0).astype(np.uint8),
            big_mask,
            err_msg="Disabled mode should not modify masks",
        )

    def test_backwards_compat_bool_false_means_disabled(self) -> None:
        """Old cross_study_mask_union=False resolves to mode='disabled'."""
        from mengrowth.preprocessing.src.config import SkullStrippingStepConfig

        cfg = SkullStrippingStepConfig(cross_study_mask_union=False)
        assert cfg.cross_study_mask_union_mode == "disabled"

    def test_backwards_compat_bool_true_keeps_mode(self) -> None:
        """Old cross_study_mask_union=True keeps the default audit mode."""
        from mengrowth.preprocessing.src.config import SkullStrippingStepConfig

        cfg = SkullStrippingStepConfig(cross_study_mask_union=True)
        assert cfg.cross_study_mask_union_mode == "audit"

    def test_config_explicit_mode_overrides_bool(self) -> None:
        """Explicit mode setting takes priority regardless of bool value."""
        from mengrowth.preprocessing.src.config import SkullStrippingStepConfig

        cfg = SkullStrippingStepConfig(
            cross_study_mask_union=True,
            cross_study_mask_union_mode="disabled",
        )
        assert cfg.cross_study_mask_union_mode == "disabled"

    def test_config_invalid_mode_raises(self) -> None:
        """Invalid mode raises ValueError."""
        from mengrowth.preprocessing.src.config import SkullStrippingStepConfig

        with pytest.raises(ValueError, match="cross_study_mask_union_mode"):
            SkullStrippingStepConfig(cross_study_mask_union_mode="bogus")

    def test_audit_mode_writes_report_no_mask_changes(self, tmp_path: Path) -> None:
        """Audit mode writes flag report but does NOT modify any masks."""
        import json

        from mengrowth.preprocessing.src.steps.skull_stripping import (
            compute_and_apply_union_brain_mask,
        )

        setup = _setup_two_study_patient(tmp_path)
        mod = setup["modality"]
        shape = (64, 64, 64)

        # Large deficit to trigger flagging
        big_mask = _sphere_mask(shape, (32, 32, 32), 18)
        small_mask = _sphere_mask(shape, (32, 32, 32), 12)

        _make_nifti(
            big_mask,
            np.eye(4),
            setup["artifacts_root"]
            / setup["patient_id"]
            / setup["ref_study"]
            / f"{mod}_brain_mask.nii.gz",
        )
        _make_nifti(
            small_mask,
            np.eye(4),
            setup["artifacts_root"]
            / setup["patient_id"]
            / setup["nonref_study"]
            / f"{mod}_brain_mask.nii.gz",
        )

        # Save originals for comparison
        ref_img_before = (
            nib.load(str(setup["study_dirs"][0] / f"{mod}.nii.gz")).get_fdata().copy()
        )
        nonref_img_before = (
            nib.load(str(setup["study_dirs"][1] / f"{mod}.nii.gz")).get_fdata().copy()
        )

        result = compute_and_apply_union_brain_mask(
            patient_id=setup["patient_id"],
            study_dirs=setup["study_dirs"],
            artifacts_root=setup["artifacts_root"],
            modalities=[mod],
            longitudinal_transforms_dir=None,
            union_mode="audit",
            protective_threshold=0.05,
        )

        # Audit mode returns False (no masks modified)
        assert result is False

        # Masks and images should be UNCHANGED
        ref_img_after = nib.load(
            str(setup["study_dirs"][0] / f"{mod}.nii.gz")
        ).get_fdata()
        nonref_img_after = nib.load(
            str(setup["study_dirs"][1] / f"{mod}.nii.gz")
        ).get_fdata()
        np.testing.assert_array_equal(ref_img_before, ref_img_after)
        np.testing.assert_array_equal(nonref_img_before, nonref_img_after)

        # Audit report should exist
        report_path = (
            setup["artifacts_root"] / setup["patient_id"] / "mask_volume_audit.json"
        )
        assert report_path.exists(), "Audit report should be written"

        report = json.loads(report_path.read_text())
        assert report["patient_id"] == setup["patient_id"]
        assert report["n_flagged"] > 0, "Should flag the small-mask study"
        assert report["flagged_studies"][0]["study_id"] == setup["nonref_study"]
