#!/usr/bin/env python3
"""
Targeted t2f fix for MenGrowth-0003-000.

Preprocesses only the t2f sequence through harmonization → bias correction →
resampling → padding, then registers it to the Picasso t1n (already in atlas
space). Caches intermediate results for fast iteration.

Usage:
    python scripts/fix_0003_t2f.py --stage prep     # steps 1-4
    python scripts/fix_0003_t2f.py --stage coreg    # intra-study registration
    python scripts/fix_0003_t2f.py --stage atlas    # atlas registration
    python scripts/fix_0003_t2f.py --stage all      # everything
"""

import argparse
import logging
import shutil
from pathlib import Path

import ants
import nibabel as nib
import nrrd
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Paths
CURATED = Path(
    "/media/mpascual/PortableSSD/Meningiomas/MenGrowth/curated/dataset/MenGrowth-2025/MenGrowth-0003/MenGrowth-0003-000"
)
PICASSO = Path(
    "/media/mpascual/PortableSSD/Meningiomas/MenGrowth/preprocessed/test_picasso/MenGrowth-2025/MenGrowth-0003/MenGrowth-0003-000"
)
WORKDIR = Path(
    "/media/mpascual/PortableSSD/Meningiomas/MenGrowth/preprocessed/test/MenGrowth-0003/_t2f_fix"
)
VIZDIR = Path(
    "/media/mpascual/PortableSSD/Meningiomas/MenGrowth/preprocessed/test/viz/MenGrowth-0003/MenGrowth-0003-000"
)

# Checkpoints
CKPT_PREP = WORKDIR / "t2f_after_prep.nii.gz"  # after harmonize+bias+resample+pad
CKPT_COREG = WORKDIR / "t2f_after_coreg.nii.gz"  # after intra-study registration
CKPT_ATLAS = WORKDIR / "t2f_after_atlas.nii.gz"  # after atlas registration


def visualize_registration(
    fixed_path: str, moving_path: str, warped_path: str, title: str, output_path: Path
) -> None:
    """Save a 3-panel registration comparison (axial middle slice)."""
    fixed = nib.load(fixed_path).get_fdata()
    moving = nib.load(moving_path).get_fdata()
    warped = nib.load(warped_path).get_fdata()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    mid_f = fixed.shape[2] // 2
    mid_m = moving.shape[2] // 2
    mid_w = warped.shape[2] // 2

    axes[0].imshow(fixed[:, :, mid_f].T, cmap="gray", origin="lower")
    axes[0].set_title("Reference (Fixed)")
    axes[1].imshow(moving[:, :, mid_m].T, cmap="gray", origin="lower")
    axes[1].set_title("Moving (Original)")
    axes[2].imshow(warped[:, :, mid_w].T, cmap="gray", origin="lower")
    axes[2].set_title("Registered")
    fig.suptitle(title)
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Viz saved: {output_path}")


def stage_prep() -> Path:
    """Steps 1-4: Harmonize, bias correct, resample, pad t2f."""
    if CKPT_PREP.exists():
        logger.info(f"[PREP] Using cached: {CKPT_PREP}")
        return CKPT_PREP

    WORKDIR.mkdir(parents=True, exist_ok=True)
    logger.info("[PREP] Processing t2f through steps 1-4...")

    # Step 1: NRRD → NIfTI + RAS
    src_nrrd = CURATED / "t2f.nrrd"
    data, header = nrrd.read(str(src_nrrd))
    space_dirs = np.array(header["space directions"])
    spacing = [np.linalg.norm(d) for d in space_dirs]
    origin = header.get("space origin", [0, 0, 0])
    logger.info(
        f"  Source: shape={data.shape}, spacing={[round(s, 2) for s in spacing]}"
    )

    # Convert via SimpleITK for proper orientation handling
    sitk_img = sitk.ReadImage(str(src_nrrd))
    sitk_img = sitk.DICOMOrient(sitk_img, "RAS")

    # Intensity clipping (p95 for hot pixel removal)
    arr = sitk.GetArrayFromImage(sitk_img).astype(np.float64)
    nonzero_vals = arr[arr > 0]
    if len(nonzero_vals) > 0:
        clip_val = np.percentile(nonzero_vals, 95.0)
        clipped = np.sum(arr > clip_val)
        arr = np.clip(arr, None, clip_val)
        logger.info(f"  Clipped {clipped:,} voxels at p95={clip_val:.1f}")

    clipped_img = sitk.GetImageFromArray(arr)
    clipped_img.CopyInformation(sitk_img)

    # Step 2: N4 bias field correction
    logger.info("  Running N4 bias correction...")
    mask = sitk.OtsuThreshold(clipped_img, 0, 1, 200)
    shrunk = sitk.Shrink(clipped_img, [4, 4, 4])
    mask_shrunk = sitk.Shrink(mask, [4, 4, 4])
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([50, 50, 50, 50])
    corrector.SetBiasFieldFullWidthAtHalfMaximum(0.15)
    corrector.SetConvergenceThreshold(0.001)
    corrected_shrunk = corrector.Execute(shrunk, mask_shrunk)

    log_bias = corrector.GetLogBiasFieldAsImage(clipped_img)
    corrected = sitk.Cast(clipped_img, sitk.sitkFloat32) / sitk.Cast(
        sitk.Exp(log_bias), sitk.sitkFloat32
    )
    logger.info("  N4 complete")

    # Step 3: BSpline resampling to 1mm isotropic
    logger.info("  Resampling to 1mm isotropic (BSpline order 3)...")
    orig_size = corrected.GetSize()
    orig_spacing = corrected.GetSpacing()
    new_spacing = [1.0, 1.0, 1.0]
    new_size = [
        int(round(osz * ospc / nspc))
        for osz, ospc, nspc in zip(orig_size, orig_spacing, new_spacing)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetOutputOrigin(corrected.GetOrigin())
    resampler.SetOutputDirection(corrected.GetDirection())
    resampler.SetInterpolator(sitk.sitkBSpline)
    resampler.SetDefaultPixelValue(0.0)
    resampled = resampler.Execute(corrected)
    logger.info(
        f"  Resampled: {orig_size} ({[round(s, 2) for s in orig_spacing]}) -> {new_size} (1mm)"
    )

    # Step 4: Cubic padding to match Picasso reference shape
    # Get the target shape from the Picasso t1n
    ref_nii = nib.load(str(PICASSO / "t1n.nii.gz"))
    # Pad to a cubic shape first (max dim), matching the pipeline behavior
    arr_resampled = sitk.GetArrayFromImage(resampled)  # ZYX
    shape_xyz = arr_resampled.shape[::-1]  # XYZ
    max_dim = max(shape_xyz)
    logger.info(f"  Pre-pad shape: {shape_xyz}, padding to cube: {max_dim}³")

    padded = np.zeros((max_dim, max_dim, max_dim), dtype=arr_resampled.dtype)
    offsets = [(max_dim - s) // 2 for s in arr_resampled.shape]
    padded[
        offsets[0] : offsets[0] + arr_resampled.shape[0],
        offsets[1] : offsets[1] + arr_resampled.shape[1],
        offsets[2] : offsets[2] + arr_resampled.shape[2],
    ] = arr_resampled

    # Save as NIfTI with same orientation as resampled
    padded_sitk = sitk.GetImageFromArray(padded)
    new_origin = list(resampled.GetOrigin())
    for i in range(3):
        new_origin[i] -= offsets[2 - i] * new_spacing[i]  # ZYX offset order
    padded_sitk.SetOrigin(new_origin)
    padded_sitk.SetSpacing(new_spacing)
    padded_sitk.SetDirection(resampled.GetDirection())

    sitk.WriteImage(padded_sitk, str(CKPT_PREP))
    logger.info(f"  Saved prep checkpoint: {CKPT_PREP}")
    logger.info(f"  Final shape: {padded.shape[::-1]}, nz={np.count_nonzero(padded):,}")

    return CKPT_PREP


def stage_coreg() -> Path:
    """Intra-study coregistration: t2f → Picasso t1n (Rigid)."""
    if CKPT_COREG.exists():
        logger.info(f"[COREG] Using cached: {CKPT_COREG}")
        return CKPT_COREG

    prep_path = stage_prep()

    logger.info("[COREG] Rigid registration: t2f -> Picasso t1n...")

    # Fixed = Picasso t1n (already in atlas space, skull-stripped)
    fixed = ants.image_read(str(PICASSO / "t1n.nii.gz"))
    moving = ants.image_read(str(prep_path))

    logger.info(
        f"  Fixed (t1n):  shape={fixed.shape}, nz={np.count_nonzero(fixed.numpy()):,}"
    )
    logger.info(
        f"  Moving (t2f): shape={moving.shape}, nz={np.count_nonzero(moving.numpy()):,}"
    )

    # Rigid registration with conservative parameters
    reg = ants.registration(
        fixed=fixed,
        moving=moving,
        type_of_transform="Rigid",
        aff_metric="mattes",
        aff_sampling=64,
        aff_random_sampling_rate=0.5,
        aff_iterations=(2000, 1000, 500, 250),
        aff_shrink_factors=(8, 4, 2, 1),
        aff_smoothing_sigmas=(3, 2, 1, 0),
        verbose=True,
    )

    warped = reg["warpedmovout"]
    nz = np.count_nonzero(warped.numpy())
    logger.info(f"  Registered t2f: nz={nz:,} ({nz / warped.numpy().size * 100:.1f}%)")

    # Save
    ants.image_write(warped, str(CKPT_COREG))
    logger.info(f"  Saved coreg checkpoint: {CKPT_COREG}")

    # Save moving pre-registration for viz comparison
    moving_viz = WORKDIR / "t2f_moving_before_coreg.nii.gz"
    if not moving_viz.exists():
        shutil.copy2(str(prep_path), str(moving_viz))

    # Visualize
    visualize_registration(
        str(PICASSO / "t1n.nii.gz"),
        str(prep_path),
        str(CKPT_COREG),
        "Intra-study coregistration: t2f → t1n (Rigid)",
        VIZDIR / "fix_t2f_coreg.png",
    )

    return CKPT_COREG


def stage_atlas() -> Path:
    """Apply skull-strip mask and finalize."""
    coreg_path = stage_coreg()

    logger.info("[ATLAS] Applying skull-strip mask and saving final result...")

    # The Picasso t1n is already in atlas space — our coreg put t2f into
    # the same space. We just need to apply the skull-stripping mask.
    mask_path = PICASSO / "_temp__temp_t1c_skull_stripped_skull_stripping_mask.nii.gz"
    if not mask_path.exists():
        logger.warning("  No mask found — saving unmasked result")
        shutil.copy2(str(coreg_path), str(CKPT_ATLAS))
        return CKPT_ATLAS

    coreg_nii = nib.load(str(coreg_path))
    mask_nii = nib.load(str(mask_path))

    coreg_data = coreg_nii.get_fdata()
    mask_data = mask_nii.get_fdata() > 0

    # Check shapes match
    if coreg_data.shape != mask_data.shape:
        logger.warning(
            f"  Shape mismatch: t2f={coreg_data.shape} mask={mask_data.shape}"
        )
        # Resample t2f to mask grid
        coreg_ants = ants.image_read(str(coreg_path))
        mask_ants = ants.image_read(str(mask_path))
        resampled = ants.resample_image_to_target(
            coreg_ants, mask_ants, interp_type="bSpline"
        )
        coreg_data = resampled.numpy()
        logger.info(f"  Resampled t2f to mask grid: {coreg_data.shape}")

    masked = coreg_data * mask_data
    nz = np.count_nonzero(masked)
    logger.info(f"  Masked t2f: nz={nz:,} ({nz / masked.size * 100:.1f}%)")

    # Save to final location
    out_nii = nib.Nifti1Image(
        masked.astype(np.float32), mask_nii.affine, mask_nii.header
    )
    nib.save(out_nii, str(CKPT_ATLAS))
    logger.info(f"  Saved atlas checkpoint: {CKPT_ATLAS}")

    # Also place in the output study dir
    final_path = Path(
        "/media/mpascual/PortableSSD/Meningiomas/MenGrowth/preprocessed/test/MenGrowth-0003/MenGrowth-0003-000/t2f.nii.gz"
    )
    nib.save(out_nii, str(final_path))
    logger.info(f"  Installed to: {final_path}")

    # Visualize
    atlas_path = (
        "/media/mpascual/PortableSSD/Meningiomas/ATLAS/sri24_spm8/templates/T1.nii"
    )
    if Path(atlas_path).exists():
        visualize_registration(
            atlas_path,
            str(coreg_path),
            str(CKPT_ATLAS),
            "Atlas alignment: t2f in atlas space (masked)",
            VIZDIR / "fix_t2f_atlas.png",
        )

    return CKPT_ATLAS


def clear_cache(stage: str) -> None:
    """Remove checkpoint for a specific stage to force re-run."""
    ckpts = {"prep": CKPT_PREP, "coreg": CKPT_COREG, "atlas": CKPT_ATLAS}
    if stage == "all":
        for p in ckpts.values():
            p.unlink(missing_ok=True)
        logger.info("Cleared all checkpoints")
    elif stage in ckpts:
        ckpts[stage].unlink(missing_ok=True)
        # Also clear downstream
        order = ["prep", "coreg", "atlas"]
        idx = order.index(stage)
        for s in order[idx:]:
            ckpts[s].unlink(missing_ok=True)
        logger.info(f"Cleared {stage} and downstream checkpoints")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fix MenGrowth-0003-000 t2f registration"
    )
    parser.add_argument(
        "--stage", choices=["prep", "coreg", "atlas", "all"], default="all"
    )
    parser.add_argument(
        "--clear", action="store_true", help="Clear cache for this stage first"
    )
    args = parser.parse_args()

    WORKDIR.mkdir(parents=True, exist_ok=True)
    VIZDIR.mkdir(parents=True, exist_ok=True)

    if args.clear:
        clear_cache(args.stage)

    if args.stage == "prep" or args.stage == "all":
        stage_prep()
    if args.stage == "coreg" or args.stage == "all":
        stage_coreg()
    if args.stage == "atlas" or args.stage == "all":
        stage_atlas()

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
