"""Post-processing for BraTS meningioma segmentation outputs.

Remaps BraTS-formatted segmentation outputs back to the original study
directories and handles shape correction reversal and temp file cleanup.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List
import json
import logging
import shutil

import nibabel as nib
import numpy as np

from mengrowth.segmentation.config import SegmentationConfig

logger = logging.getLogger(__name__)


@dataclass
class SegmentationResult:
    """Result of post-processing a single segmentation output.

    Attributes:
        patient_id: Patient identifier.
        study_id: Study identifier.
        brats_name: BraTS internal name used during inference.
        output_path: Final path of the segmentation file.
        shape_corrected: Whether the output was cropped back to original shape.
        success: Whether the output was found and copied.
        error: Error message if not successful.
    """

    patient_id: str
    study_id: str
    brats_name: str
    output_path: str = ""
    shape_corrected: bool = False
    success: bool = False
    error: str = ""


def _find_brats_output(brats_output_dir: Path, brats_name: str) -> Path | None:
    """Find the segmentation output file for a BraTS subject.

    Checks multiple patterns since different BraTS algorithms may use
    different output conventions.

    Args:
        brats_output_dir: Directory containing BraTS outputs.
        brats_name: BraTS subject name (e.g., BraTS-MEN-00000-000).

    Returns:
        Path to the output file, or None if not found.
    """
    candidates = [
        brats_output_dir / f"{brats_name}.nii.gz",
        brats_output_dir / brats_name / f"{brats_name}.nii.gz",
        brats_output_dir / brats_name / "seg.nii.gz",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Fallback: search by identifier substring
    identifier = "-".join(brats_name.split("-")[-2:])
    for path in sorted(brats_output_dir.rglob("*.nii.gz")):
        if identifier in path.name:
            return path

    return None


def _reverse_shape_correction(
    seg_path: Path,
    output_path: Path,
    original_shape: List[int],
    target_shape: tuple,
) -> None:
    """Crop a segmentation mask back to its original pre-correction shape.

    This reverses the padding/cropping done during prepare_brats_input
    to ensure the segmentation aligns with the original preprocessed volumes.

    Args:
        seg_path: Path to the BraTS output segmentation.
        output_path: Path to write the cropped segmentation.
        original_shape: Original volume shape before correction.
        target_shape: The shape the volume was corrected to.
    """
    img = nib.load(str(seg_path))
    data = np.asarray(img.dataobj)
    affine = img.affine.copy()

    for axis in range(3):
        diff = target_shape[axis] - original_shape[axis]
        if diff > 0:
            # Was padded -> crop back
            pad_before = diff // 2
            slc = [slice(None)] * 3
            slc[axis] = slice(pad_before, pad_before + original_shape[axis])
            data = data[tuple(slc)]

            # Adjust affine
            shift = np.zeros(3)
            shift[axis] = pad_before
            affine[:3, 3] -= affine[:3, :3] @ shift
        elif diff < 0:
            # Was cropped -> pad back
            crop_before = (-diff) // 2
            pad_widths = [(0, 0)] * 3
            pad_widths[axis] = (
                crop_before,
                original_shape[axis] - data.shape[axis] - crop_before,
            )
            data = np.pad(data, pad_widths, mode="constant", constant_values=0)

            shift = np.zeros(3)
            shift[axis] = -crop_before
            affine[:3, 3] -= affine[:3, :3] @ shift

    result_img = nib.Nifti1Image(data, affine, img.header)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(result_img, str(output_path))


def postprocess_outputs(
    work_dir: str | Path,
    config: SegmentationConfig,
) -> List[SegmentationResult]:
    """Remap BraTS segmentation outputs to original study directories.

    Loads name_map.json from work_dir, finds each BraTS output, reverses
    shape correction if needed, and copies the result to the study directory.

    Args:
        work_dir: Working directory containing input/, output/, name_map.json.
        config: Segmentation configuration.

    Returns:
        List of SegmentationResult objects.
    """
    work_dir = Path(work_dir)
    brats_output_dir = work_dir / "output"
    map_file = work_dir / "name_map.json"

    if not map_file.exists():
        raise FileNotFoundError(f"Name mapping not found: {map_file}")

    with open(map_file, "r", encoding="utf-8") as f:
        name_map = json.load(f)

    results: List[SegmentationResult] = []

    for brats_name, info in sorted(name_map.items()):
        result = SegmentationResult(
            patient_id=info["patient_id"],
            study_id=info["study_id"],
            brats_name=brats_name,
        )

        # Find the output file
        output_file = _find_brats_output(brats_output_dir, brats_name)
        if output_file is None:
            result.error = f"No output found for {brats_name}"
            logger.warning(f"  {result.error}")
            results.append(result)
            continue

        # Destination: alongside preprocessed data
        study_path = Path(info["study_path"])
        dst_path = study_path / config.output_filename
        result.output_path = str(dst_path)

        # Handle shape correction reversal
        if info.get("shape_corrected", False) and info.get("original_shape"):
            # Use the first modality's original shape (they should all match)
            original_shapes = info["original_shape"]
            first_mod_shape = next(iter(original_shapes.values()))
            result.shape_corrected = True

            logger.info(
                f"  Reversing shape correction for {info['study_id']}: "
                f"{config.expected_shape} -> {first_mod_shape}"
            )
            _reverse_shape_correction(
                output_file, dst_path, first_mod_shape, config.expected_shape
            )
        else:
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(output_file, dst_path)

        result.success = True
        logger.info(f"  {brats_name} -> {dst_path}")
        results.append(result)

    succeeded = sum(1 for r in results if r.success)
    logger.info(
        f"Post-processing complete: {succeeded}/{len(results)} segmentations copied"
    )
    return results


def cleanup_temp_files(input_root: str | Path) -> int:
    """Remove temporary files from the input directory.

    Removes files matching _temp_* and *.tmp.* patterns recursively.

    Args:
        input_root: Root directory to clean.

    Returns:
        Number of files removed.
    """
    input_root = Path(input_root)
    count = 0

    for pattern in ["_temp_*", "*.tmp.*"]:
        for path in sorted(input_root.rglob(pattern)):
            if path.is_file():
                logger.debug(f"  Removing: {path}")
                path.unlink()
                count += 1
            elif path.is_dir():
                logger.debug(f"  Removing directory: {path}")
                shutil.rmtree(path)
                count += 1

    logger.info(f"Cleanup: removed {count} temporary file(s) from {input_root}")
    return count
