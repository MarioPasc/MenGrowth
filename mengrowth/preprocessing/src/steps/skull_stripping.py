"""Skull stripping step: brain extraction using HD-BET or SynthStrip.

This is a study-level step that processes all modalities.

Two modes of operation:
  1. Reference mask mode (BraTS-standard): skull-strip only the reference modality
     (e.g. T1c) and apply its brain mask to all co-registered modalities.
     Activated by setting `reference_mask_modality` in config.
  2. Per-modality mode (legacy): each modality gets its own brain mask, with
     optional consensus voting across modalities.
"""

import shutil
from typing import Dict, Any, Optional
from pathlib import Path
import logging

import numpy as np
import nibabel as nib
from scipy.ndimage import label as ndimage_label

from mengrowth.preprocessing.src.config import StepExecutionContext
from mengrowth.preprocessing.src.steps.utils import (
    get_output_dir,
    get_temp_path,
    get_artifact_path,
    get_visualization_path,
)

logger = logging.getLogger(__name__)


def _largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component in a binary mask.

    Args:
        mask: Binary numpy array.

    Returns:
        Binary mask with only the largest connected component.
    """
    labeled, num_features = ndimage_label(mask)
    if num_features <= 1:
        return mask
    component_sizes = np.bincount(labeled.ravel())
    # Ignore background (label 0)
    component_sizes[0] = 0
    largest_label = component_sizes.argmax()
    return (labeled == largest_label).astype(mask.dtype)


def _execute_reference_mask_mode(
    context: StepExecutionContext,
    skull_stripper: Any,
    study_output_dir: Path,
    artifacts_base: Path,
) -> Dict[str, Any]:
    """Execute skull stripping using a single reference modality mask for all modalities.

    BraTS-standard approach: skull-strip only the reference modality (e.g. T1c),
    then apply that mask to all co-registered modalities.

    Args:
        context: Execution context
        skull_stripper: Initialized skull stripping component
        study_output_dir: Directory containing modality NIfTI files
        artifacts_base: Directory for saving mask artifacts

    Returns:
        Dict with skull stripping results per modality
    """
    config = context.step_config
    orchestrator = context.orchestrator
    ss_config = config.skull_stripping
    ref_modality = ss_config.reference_mask_modality

    logger.info(f"  Reference mask mode: using {ref_modality} mask for all modalities")

    # ── Phase 1: Extract brain mask from reference modality only ──
    ref_path = study_output_dir / f"{ref_modality}.nii.gz"
    if not ref_path.exists():
        raise RuntimeError(
            f"Reference modality '{ref_modality}' not found at {ref_path}. "
            f"Cannot perform reference-mask skull stripping."
        )

    logger.info(f"  [Phase 1] Extracting brain mask from {ref_modality}...")

    temp_ref_stripped = get_temp_path(context, ref_modality, "skull_stripped")
    ref_mask_path = get_artifact_path(context, f"{ref_modality}_brain_mask")

    ref_result = skull_stripper.execute(
        ref_path,
        temp_ref_stripped,
        mask_path=ref_mask_path,
        allow_overwrite=True,
    )

    logger.info(
        f"  {ref_modality}: brain_volume={ref_result['brain_volume_mm3']:.1f} mm³, "
        f"coverage={ref_result['brain_coverage_percent']:.1f}%"
    )

    # Load the reference mask
    ref_mask_nii = nib.load(str(ref_mask_path))
    ref_mask_data = (ref_mask_nii.get_fdata() > 0).astype(bool)

    # ── Phase 2: Skipped (single authoritative mask) ──
    logger.info("  [Phase 2] Skipped — using single reference mask")

    # ── Phase 3: Apply reference mask to ALL modalities ──
    results = {}
    processed_modalities = []

    for modality in orchestrator.config.modalities:
        modality_path = study_output_dir / f"{modality}.nii.gz"
        if not modality_path.exists():
            logger.warning(f"  Skipping {modality} — file not found")
            continue

        processed_modalities.append(modality)

        if modality == ref_modality:
            # For reference modality, visualize before replacing
            if config.save_visualization:
                viz_output = get_visualization_path(context, suffix=f"_{modality}")
                skull_stripper.visualize(
                    modality_path, temp_ref_stripped, viz_output, **ref_result
                )

            # Replace original with skull-stripped output
            logger.info(f"  [Phase 3] Using skull-stripped output for {modality} (reference)")
            temp_ref_stripped.replace(modality_path)
            results[modality] = ref_result
        else:
            # Apply reference mask to this modality
            logger.info(f"  [Phase 3] Applying {ref_modality} mask to {modality}...")
            original_nii = nib.load(str(modality_path))
            original_data = original_nii.get_fdata()

            stripped = np.where(ref_mask_data, original_data, ss_config.fill_value)
            stripped_nii = nib.Nifti1Image(
                stripped.astype(original_data.dtype),
                original_nii.affine,
                original_nii.header,
            )

            temp_out = get_temp_path(context, modality, "ref_mask_stripped")
            nib.save(stripped_nii, str(temp_out))

            # Compute result metrics for this modality
            brain_voxels = int(ref_mask_data.sum())
            total_voxels = int(ref_mask_data.size)
            voxel_volume = float(np.prod(original_nii.header.get_zooms()[:3]))
            results[modality] = {
                "mask_path": ref_mask_path,
                "brain_volume_mm3": brain_voxels * voxel_volume,
                "brain_coverage_percent": 100.0 * brain_voxels / total_voxels,
                "reference_mask_from": ref_modality,
            }

            # Visualize before replacing original
            if config.save_visualization:
                viz_output = get_visualization_path(context, suffix=f"_{modality}")
                skull_stripper.visualize(
                    modality_path, temp_out, viz_output, **results[modality]
                )

            # Replace original with masked output
            temp_out.replace(modality_path)

        # Save per-modality mask copy for downstream compatibility (e.g. intensity normalization)
        if config.save_mask and modality != ref_modality:
            modality_mask_path = get_artifact_path(context, f"{modality}_brain_mask")
            shutil.copy2(str(ref_mask_path), str(modality_mask_path))
            logger.debug(f"  Saved {ref_modality} mask as {modality}_brain_mask")

    # ── Diagnostic: log final file state ──
    for modality in processed_modalities:
        final_path = study_output_dir / f"{modality}.nii.gz"
        if final_path.exists():
            diag_nii = nib.load(str(final_path))
            diag_data = diag_nii.get_fdata()
            nonzero_frac = np.count_nonzero(diag_data) / diag_data.size
            logger.info(
                f"  [DIAG] {modality} after skull stripping: "
                f"shape={diag_data.shape}, "
                f"nonzero={nonzero_frac:.3%}, "
                f"range=[{diag_data.min():.2f}, {diag_data.max():.2f}], "
                f"path={final_path}"
            )

    logger.info("  Skull stripping completed successfully (reference mask mode)")

    # Add qc_paths for QC system
    results["qc_paths"] = {
        "study_output_dir": str(study_output_dir),
        "mask_outputs": {
            mod: str(artifacts_base / f"{mod}_brain_mask.nii.gz")
            for mod in processed_modalities
        }
        if config.save_mask
        else {},
        "image_outputs": {
            mod: str(study_output_dir / f"{mod}.nii.gz") for mod in processed_modalities
        },
    }

    return results


def execute(
    context: StepExecutionContext, total_steps: int, current_step_num: int
) -> Dict[str, Any]:
    """Execute skull stripping step (study-level operation).

    Two modes:
      - Reference mask mode: skull-strip reference modality only, apply to all.
      - Per-modality mode: individual masks with optional consensus voting.

    Args:
        context: Execution context (modality will be None for study-level steps)
        total_steps: Not used for study-level steps
        current_step_num: Not used for study-level steps

    Returns:
        Dict with skull stripping results per modality

    Raises:
        RuntimeError: If skull stripping fails
    """
    config = context.step_config
    orchestrator = context.orchestrator
    patient_id = context.patient_id
    study_dir = context.study_dir

    # Check if skull stripping is enabled
    method = config.skull_stripping.method
    if method is None:
        logger.info(
            f"\n  Executing study-level step: {context.step_name} - skipped (method=None)"
        )
        return {}

    logger.info(f"\n  Executing study-level step: {context.step_name} ({method})")

    # Get or create skull stripper component
    skull_stripper = orchestrator._get_component(
        f"skull_stripper_{method}_{context.step_name}", config
    )

    # Determine output directories based on mode
    study_output_dir = get_output_dir(context=context)
    artifacts_base = (
        Path(orchestrator.config.preprocessing_artifacts_path)
        / patient_id
        / study_dir.name
    )

    ss_config = config.skull_stripping

    # ── Reference mask mode (BraTS-standard) ──
    if ss_config.reference_mask_modality is not None:
        return _execute_reference_mask_mode(
            context, skull_stripper, study_output_dir, artifacts_base
        )

    # ── Per-modality mode (legacy) ──
    use_consensus = ss_config.consensus_masking

    results = {}
    mask_paths: Dict[str, Path] = {}
    temp_stripped_paths: Dict[str, Path] = {}

    # ── Phase 1: Extract individual masks (do NOT replace originals yet) ──
    for modality in orchestrator.config.modalities:
        modality_path = study_output_dir / f"{modality}.nii.gz"
        if not modality_path.exists():
            logger.warning(f"  Skipping {modality} - file not found")
            continue

        logger.info(f"  [Phase 1] Extracting mask for {modality}...")

        # Create temporary output path for skull-stripped image
        temp_skull_stripped = get_temp_path(context, modality, "skull_stripped")

        # Determine mask path
        if config.save_mask:
            mask_path = get_artifact_path(context, f"{modality}_brain_mask")
        else:
            mask_path = get_temp_path(context, modality, "brain_mask")

        # Execute skull stripping (produces mask + stripped image)
        result = skull_stripper.execute(
            modality_path,
            temp_skull_stripped,
            mask_path=mask_path,
            allow_overwrite=True,
        )

        results[modality] = result
        mask_paths[modality] = mask_path
        temp_stripped_paths[modality] = temp_skull_stripped

        # Generate visualization (uses original vs stripped)
        if config.save_visualization:
            viz_output = get_visualization_path(context, suffix=f"_{modality}")
            skull_stripper.visualize(
                modality_path, temp_skull_stripped, viz_output, **result
            )

        logger.info(
            f"  {modality}: brain_volume={result['brain_volume_mm3']:.1f} mm³, "
            f"coverage={result['brain_coverage_percent']:.1f}%"
        )

    num_masks = len(mask_paths)

    # ── Phase 2: Compute consensus mask ──
    consensus_mask_nii = None
    if use_consensus and num_masks > 1:
        logger.info(
            f"  [Phase 2] Computing consensus mask from {num_masks} modalities "
            f"(threshold={ss_config.consensus_threshold})"
        )

        # Load all individual binary masks
        mask_arrays = {}
        reference_nii = None
        for modality, mp in sorted(mask_paths.items()):
            nii = nib.load(str(mp))
            mask_arrays[modality] = (nii.get_fdata() > 0).astype(np.uint8)
            if reference_nii is None:
                reference_nii = nii

        # Vote map: sum of binary masks → values 0..N
        vote_map = np.zeros_like(next(iter(mask_arrays.values())), dtype=np.int32)
        for arr in mask_arrays.values():
            vote_map += arr.astype(np.int32)

        # Effective threshold: clamp to num_masks for graceful fallback with <4 modalities
        effective_threshold = min(ss_config.consensus_threshold, num_masks)
        consensus = (vote_map >= effective_threshold).astype(np.uint8)

        # Largest connected component to remove outlier islands
        consensus = _largest_connected_component(consensus)

        consensus_mask_nii = nib.Nifti1Image(
            consensus, reference_nii.affine, reference_nii.header
        )

        brain_voxels = int(consensus.sum())
        total_voxels = int(consensus.size)
        logger.info(
            f"  Consensus mask: {brain_voxels} voxels "
            f"({100.0 * brain_voxels / total_voxels:.1f}% coverage), "
            f"effective threshold={effective_threshold}/{num_masks}"
        )

        # Save consensus mask
        consensus_path = artifacts_base / "consensus_brain_mask.nii.gz"
        consensus_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(consensus_mask_nii, str(consensus_path))
        logger.info(f"  Saved consensus mask: {consensus_path}")

        # Archive individual masks for QC, overwrite per-modality masks with consensus
        for modality, mp in sorted(mask_paths.items()):
            individual_path = (
                artifacts_base / f"{modality}_brain_mask_individual.nii.gz"
            )
            shutil.copy2(str(mp), str(individual_path))
            logger.debug(f"  Archived individual mask: {individual_path.name}")

            # Overwrite per-modality mask with consensus (downstream compatibility)
            nib.save(consensus_mask_nii, str(mp))

    elif use_consensus and num_masks == 1:
        logger.info(
            "  [Phase 2] Only 1 modality — skipping consensus, using individual mask"
        )
    elif not use_consensus:
        logger.info("  [Phase 2] Consensus masking disabled — using per-modality masks")

    # ── Phase 3: Apply mask and replace originals ──
    for modality in list(results.keys()):
        modality_path = study_output_dir / f"{modality}.nii.gz"

        if consensus_mask_nii is not None:
            # Apply consensus mask to original image
            logger.info(f"  [Phase 3] Applying consensus mask to {modality}...")
            original_nii = nib.load(str(modality_path))
            original_data = original_nii.get_fdata()
            consensus_data = consensus_mask_nii.get_fdata().astype(bool)

            stripped = np.where(consensus_data, original_data, ss_config.fill_value)
            stripped_nii = nib.Nifti1Image(
                stripped.astype(original_data.dtype),
                original_nii.affine,
                original_nii.header,
            )

            temp_out = get_temp_path(context, modality, "consensus_stripped")
            nib.save(stripped_nii, str(temp_out))
            temp_out.replace(modality_path)

            # Clean up the Phase 1 temp stripped file (not needed)
            temp_sp = temp_stripped_paths.get(modality)
            if temp_sp and temp_sp.exists():
                temp_sp.unlink()
        else:
            # No consensus — use per-modality skull-stripped output (original behavior)
            temp_sp = temp_stripped_paths.get(modality)
            if temp_sp and temp_sp.exists():
                temp_sp.replace(modality_path)

        # Clean up temporary mask if not saving
        if not config.save_mask:
            mp = mask_paths.get(modality)
            if mp and mp.exists():
                mp.unlink()
                logger.debug("    Temporary brain mask deleted")

        # Clean up any additional temporary files created by skull stripping libraries
        temp_pattern_files = list(study_output_dir.glob(f"_temp_*{modality}*.nii.gz"))
        for temp_file in temp_pattern_files:
            if temp_file != modality_path and temp_file.exists():
                temp_file.unlink()
                logger.debug(f"    Cleaned up temporary file: {temp_file.name}")

    # ── Diagnostic: log final file state after skull stripping ──
    for modality in list(results.keys()):
        final_path = study_output_dir / f"{modality}.nii.gz"
        if final_path.exists():
            diag_nii = nib.load(str(final_path))
            diag_data = diag_nii.get_fdata()
            nonzero_frac = np.count_nonzero(diag_data) / diag_data.size
            logger.info(
                f"  [DIAG] {modality} after skull stripping: "
                f"shape={diag_data.shape}, "
                f"nonzero={nonzero_frac:.3%}, "
                f"range=[{diag_data.min():.2f}, {diag_data.max():.2f}], "
                f"path={final_path}"
            )

    logger.info("  Skull stripping completed successfully")

    # Add qc_paths for QC system (study-level step output paths)
    processed_modalities = list(results.keys())
    results["qc_paths"] = {
        "study_output_dir": str(study_output_dir),
        "mask_outputs": {
            mod: str(artifacts_base / f"{mod}_brain_mask.nii.gz")
            for mod in processed_modalities
        }
        if config.save_mask
        else {},
        "image_outputs": {
            mod: str(study_output_dir / f"{mod}.nii.gz") for mod in processed_modalities
        },
    }

    return results
