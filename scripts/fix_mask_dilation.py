#!/usr/bin/env python3
"""Post-process brain mask: recover tumor excluded by skull-stripping.

For convexity meningiomas where HD-BET excludes the extra-axial tumor,
this script uses the mask from a neighboring study (where the tumor is included)
to identify and recover the missing tumor region.

Approach:
  1. Load masks from the target study and a reference study (where tumor IS included)
  2. Compute the difference: reference_mask - target_mask = tumor region
  3. Dilate the tumor region slightly (2mm) to capture boundary voxels
  4. For each modality in the target study:
     - Keep the final (longitudinally-registered) volume for brain voxels
     - Paste tumor-region voxels from the pre-stripped backup

Usage:
    python scripts/fix_mask_dilation.py \
        --target-study MenGrowth-0056-002 \
        --donor-study MenGrowth-0056-001 \
        --output-root /path/to/preprocessed/test \
        --artifacts-root /path/to/artifacts \
        --dilation-mm 2.0
"""

import argparse
import logging
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MODALITIES = ["t1c", "t1n", "t2w", "t2f"]


def fix_tumor_mask(
    target_study_dir: Path,
    target_artifacts: Path,
    donor_artifacts: Path,
    dilation_mm: float = 2.0,
    ref_modality: str = "t1c",
) -> None:
    """Recover tumor region excluded by skull-stripping.

    Uses the mask from a donor study (where tumor is included) to identify
    the missing region, then pastes those voxels from the pre-stripped backup
    into the final longitudinally-registered volumes.

    Args:
        target_study_dir: Study dir with final NIfTI files (to fix).
        target_artifacts: Artifacts dir for the target study.
        donor_artifacts: Artifacts dir for the donor study (has tumor in mask).
        dilation_mm: Dilation of the tumor region in mm.
        ref_modality: Modality used for mask reference.
    """
    # Load masks
    target_mask_path = target_artifacts / f"{ref_modality}_brain_mask.nii.gz"
    donor_mask_path = donor_artifacts / f"{ref_modality}_brain_mask.nii.gz"

    target_mask_nii = nib.load(str(target_mask_path))
    donor_mask_nii = nib.load(str(donor_mask_path))

    target_mask = target_mask_nii.get_fdata() > 0
    donor_mask = donor_mask_nii.get_fdata() > 0

    # Verify alignment
    assert target_mask.shape == donor_mask.shape, "Mask shapes differ"
    assert np.allclose(target_mask_nii.affine, donor_mask_nii.affine), "Affines differ"

    logger.info(f"Target mask: {np.sum(target_mask):,} voxels")
    logger.info(f"Donor mask:  {np.sum(donor_mask):,} voxels")

    # Tumor region = in donor but not in target
    tumor_region = donor_mask & ~target_mask
    logger.info(f"Tumor region (donor \\ target): {np.sum(tumor_region):,} voxels")

    if np.sum(tumor_region) == 0:
        logger.warning(
            "No tumor region found — masks are identical or target is larger"
        )
        return

    # Dilate tumor region to capture boundary voxels
    voxel_size = target_mask_nii.header.get_zooms()[:3]
    iterations = max(1, int(round(dilation_mm / min(voxel_size))))
    struct = generate_binary_structure(3, 2)
    tumor_dilated = binary_dilation(
        tumor_region, structure=struct, iterations=iterations
    )
    # Don't include voxels already in the target mask (those are fine)
    new_voxels = tumor_dilated & ~target_mask
    logger.info(
        f"Tumor region dilated ({dilation_mm}mm, {iterations} iter): "
        f"{np.sum(new_voxels):,} new voxels to add"
    )

    # Expanded mask = original + new tumor voxels
    expanded_mask = target_mask | new_voxels
    logger.info(
        f"Expanded mask: {np.sum(expanded_mask):,} voxels "
        f"(+{np.sum(new_voxels) / np.sum(target_mask) * 100:.1f}%)"
    )

    # For each modality: paste tumor voxels from pre-stripped into final volume
    for mod in MODALITIES:
        final_path = target_study_dir / f"{mod}.nii.gz"
        pre_stripped_path = target_artifacts / f"{mod}_pre_stripped.nii.gz"

        if not final_path.exists():
            logger.warning(f"  {mod}: final file not found, skipping")
            continue
        if not pre_stripped_path.exists():
            logger.warning(f"  {mod}: no pre-stripped backup, skipping")
            continue

        final_nii = nib.load(str(final_path))
        pre_stripped_nii = nib.load(str(pre_stripped_path))

        final_data = final_nii.get_fdata().copy()
        pre_data = pre_stripped_nii.get_fdata()

        # Paste: only new_voxels get values from pre-stripped
        final_data[new_voxels] = pre_data[new_voxels]

        nz_before = np.count_nonzero(final_nii.get_fdata())
        nz_after = np.count_nonzero(final_data)
        logger.info(f"  {mod}: {nz_before:,} -> {nz_after:,} nonzero voxels")

        # Write via temp file
        temp_path = final_path.with_suffix(".tmp.nii.gz")
        out_nii = nib.Nifti1Image(
            final_data.astype(final_nii.get_fdata().dtype),
            final_nii.affine,
            final_nii.header,
        )
        nib.save(out_nii, str(temp_path))
        temp_path.rename(final_path)

    # Save expanded mask
    expanded_nii = nib.Nifti1Image(
        expanded_mask.astype(np.uint8), target_mask_nii.affine, target_mask_nii.header
    )
    nib.save(expanded_nii, str(target_mask_path))
    logger.info(f"  Updated mask: {target_mask_path}")

    study_mask = (
        target_study_dir
        / f"_temp__temp_{ref_modality}_skull_stripped_skull_stripping_mask.nii.gz"
    )
    if study_mask.exists():
        nib.save(expanded_nii, str(study_mask))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recover tumor region excluded by skull-stripping"
    )
    parser.add_argument(
        "--target-study",
        required=True,
        help="Target study ID (e.g., MenGrowth-0056-002)",
    )
    parser.add_argument(
        "--donor-study", required=True, help="Donor study ID (e.g., MenGrowth-0056-001)"
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument("--dilation-mm", type=float, default=2.0)
    args = parser.parse_args()

    # MenGrowth-XXXX-YYY -> patient = MenGrowth-XXXX
    parts = args.target_study.split("-")
    patient_id = f"{parts[0]}-{parts[1]}"

    fix_tumor_mask(
        target_study_dir=args.output_root / patient_id / args.target_study,
        target_artifacts=args.artifacts_root / patient_id / args.target_study,
        donor_artifacts=args.artifacts_root / patient_id / args.donor_study,
        dilation_mm=args.dilation_mm,
    )
    logger.info("Done.")


if __name__ == "__main__":
    main()
