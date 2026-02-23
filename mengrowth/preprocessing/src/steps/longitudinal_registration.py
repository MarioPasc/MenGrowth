"""Longitudinal registration step: register studies across timestamps for same patient.

This is a patient-level step that operates on all studies (timestamps) together:
1. Load reference timestamp from YAML override OR select automatically based on quality
2. Determine reference modality/modalities based on reference_modality_priority
3. Register all other timestamps to the reference timestamp
4. Optionally validate registration quality with Jacobian determinant statistics

Reference Selection Priority:
1. Manual override: Check reference_timestamp_per_study YAML for patient-specific setting
2. Automatic selection: Use reference_selection_method to choose optimal reference
   - quality_based: Select based on SNR, CNR, boundary sharpness (recommended)
   - first/last/midpoint: Simple chronological selection
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
import logging
import yaml
import shutil
import tempfile

from mengrowth.preprocessing.src.config import StepExecutionContext
from mengrowth.preprocessing.src.registration.reference_selection import (
    ReferenceSelector,
    ReferenceSelectionConfig,
    compute_jacobian_statistics,
    validate_registration_quality,
)

logger = logging.getLogger(__name__)


def execute(
    context: StepExecutionContext, total_steps: int, current_step_num: int
) -> Dict[str, Any]:
    """Execute longitudinal registration step (patient-level operation).

    Args:
        context: Execution context (modality and study_dir will be None, all_study_dirs contains all timestamps)
        total_steps: Not used for patient-level steps
        current_step_num: Not used for patient-level steps

    Returns:
        Dict with longitudinal registration results

    Raises:
        RuntimeError: If longitudinal registration fails
    """
    config = context.step_config.longitudinal_registration
    orchestrator = context.orchestrator
    patient_id = context.patient_id
    all_study_dirs = context.all_study_dirs

    logger.info(
        f"  Processing {len(all_study_dirs)} timestamps for patient {patient_id}"
    )

    results = {
        "reference_timestamp": None,
        "registered_studies": [],
        "transforms": {},
    }

    # Skip if method is None
    if config.method is None:
        logger.info("  Longitudinal registration skipped (method=None)")
        return results

    # Need at least 2 timestamps for longitudinal registration
    if len(all_study_dirs) < 2:
        logger.warning(
            f"  Only {len(all_study_dirs)} timestamp(s) found - skipping longitudinal registration"
        )
        return results

    # Step 1: Determine reference timestamp (YAML override or automatic selection)
    artifacts_base = Path(orchestrator.config.preprocessing_artifacts_path) / patient_id
    reference_timestamp, selection_info = _get_reference_timestamp(
        patient_id=patient_id,
        all_study_dirs=all_study_dirs,
        yaml_path=config.reference_timestamp_per_study,
        config=config,
        orchestrator=orchestrator,
        artifacts_base=artifacts_base,
    )
    results["reference_timestamp"] = reference_timestamp
    results["reference_selection_info"] = selection_info

    logger.info(
        f"  Reference timestamp: {reference_timestamp} (source: {selection_info.get('source', 'unknown')})"
    )

    # Find reference study directory
    reference_study_dir = None
    for study_dir in all_study_dirs:
        if study_dir.name.endswith(f"-{reference_timestamp}"):
            reference_study_dir = study_dir
            break

    if reference_study_dir is None:
        raise ValueError(
            f"Reference timestamp '{reference_timestamp}' not found in study directories: "
            f"{[s.name for s in all_study_dirs]}"
        )

    # Step 2: Determine registration mode and reference modality/modalities
    # Need to use output directory where preprocessed files are located
    reference_output_dir = _get_study_output_dir(
        orchestrator, patient_id, reference_study_dir
    )
    mode, reference_modalities = _determine_registration_mode(
        config=config,
        reference_study_dir=reference_output_dir,
        modalities=orchestrator.config.modalities,
    )

    logger.info(f"  Registration mode: {mode}")
    if mode == "single_reference":
        logger.info(f"  Reference modality: {reference_modalities[0]}")
    else:  # per_modality
        logger.info(f"  Per-modality registration: {reference_modalities}")

    # Step 3: Setup output directories
    artifacts_base = Path(orchestrator.config.preprocessing_artifacts_path) / patient_id
    viz_base = Path(orchestrator.config.viz_root) / patient_id

    # Step 4: Execute longitudinal registration
    if mode == "single_reference":
        # Single reference mode: register all modalities from other timestamps to one reference
        _execute_single_reference_registration(
            config=config,
            orchestrator=orchestrator,
            patient_id=patient_id,
            reference_study_dir=reference_study_dir,
            reference_modality=reference_modalities[0],
            all_study_dirs=all_study_dirs,
            artifacts_base=artifacts_base,
            viz_base=viz_base,
            results=results,
        )
    else:
        # Per-modality mode: register each modality type independently
        _execute_per_modality_registration(
            config=config,
            orchestrator=orchestrator,
            patient_id=patient_id,
            reference_study_dir=reference_study_dir,
            reference_modalities=reference_modalities,
            all_study_dirs=all_study_dirs,
            artifacts_base=artifacts_base,
            viz_base=viz_base,
            results=results,
        )

    logger.info(
        f"  Successfully registered {len(results['registered_studies'])} timestamps"
    )

    return results


def _get_reference_timestamp(
    patient_id: str,
    all_study_dirs: List[Path],
    yaml_path: Optional[str],
    config: Any = None,
    orchestrator: Any = None,
    artifacts_base: Optional[Path] = None,
) -> tuple:
    """Get reference timestamp for this patient from YAML override or automatic selection.

    Priority:
    1. Check YAML override file for patient-specific setting
    2. If not found, use automatic selection based on config.reference_selection_method

    Args:
        patient_id: Patient ID (e.g., "MenGrowth-0006")
        all_study_dirs: List of all study directories
        yaml_path: Path to YAML file with patient->timestamp mapping (override)
        config: LongitudinalRegistrationConfig (for automatic selection settings)
        orchestrator: PreprocessingOrchestrator (for accessing output directories)
        artifacts_base: Artifacts directory for saving selection info

    Returns:
        Tuple of (reference_timestamp, selection_info_dict)
    """
    selection_info: Dict[str, Any] = {"patient_id": patient_id}

    # Step 1: Try to load from YAML override if provided
    if yaml_path and Path(yaml_path).exists():
        try:
            with open(yaml_path, "r") as f:
                timestamp_map = yaml.safe_load(f)

            if timestamp_map and patient_id in timestamp_map:
                ref_timestamp = str(timestamp_map[patient_id]).zfill(3)
                logger.info(
                    f"  Reference timestamp from YAML override: {ref_timestamp}"
                )
                selection_info["source"] = "yaml_override"
                selection_info["yaml_path"] = str(yaml_path)
                selection_info["timestamp"] = ref_timestamp
                return ref_timestamp, selection_info
        except Exception as e:
            logger.warning(f"  Failed to load reference timestamp from YAML: {e}")

    # Step 2: Automatic selection based on config
    if config is None:
        # Fallback to default if no config
        logger.info("  Using default reference timestamp: 000 (no config provided)")
        selection_info["source"] = "default"
        selection_info["timestamp"] = "000"
        return "000", selection_info

    selection_method = getattr(config, "reference_selection_method", "quality_based")
    logger.info(f"  Automatic reference selection using method: {selection_method}")

    # Build ReferenceSelectionConfig from longitudinal registration config
    ref_selection_config = ReferenceSelectionConfig(
        method=selection_method,
        quality_metrics=getattr(
            config,
            "reference_selection_metrics",
            [
                "snr_foreground",
                "cnr_high_low",
                "boundary_gradient_score",
                "brain_coverage_fraction",
                "laplacian_sharpness",
                "ghosting_score",
            ],
        ),
        prefer_earlier=getattr(config, "reference_selection_prefer_earlier", True),
        validate_jacobian=getattr(
            config, "reference_selection_validate_jacobian", True
        ),
        jacobian_log_threshold=getattr(
            config, "reference_selection_jacobian_threshold", 0.5
        ),
    )

    # Create selector
    selector = ReferenceSelector(config=ref_selection_config, verbose=True)

    # Get output directories for each study (depends on mode: test vs pipeline)
    study_output_dirs = []
    for study_dir in all_study_dirs:
        if orchestrator:
            output_dir = _get_study_output_dir(orchestrator, patient_id, study_dir)
        else:
            output_dir = study_dir
        study_output_dirs.append(output_dir)

    # Get modalities from orchestrator
    modalities = (
        orchestrator.config.modalities if orchestrator else ["t1n", "t1c", "t2w", "t2f"]
    )

    # Check for QC metrics file
    qc_metrics_path = None
    if orchestrator and hasattr(orchestrator, "qc_manager") and orchestrator.qc_manager:
        qc_output_dir = Path(orchestrator.qc_manager.config.output_dir)
        potential_qc_path = qc_output_dir / "qc_metrics_wide.csv"
        if potential_qc_path.exists():
            qc_metrics_path = potential_qc_path

    try:
        ref_timestamp, auto_selection_info = selector.select_reference(
            study_dirs=study_output_dirs,
            patient_id=patient_id,
            modalities=modalities,
            qc_metrics_path=qc_metrics_path,
            artifacts_base=artifacts_base,
        )

        selection_info["source"] = "automatic"
        selection_info["method"] = selection_method
        selection_info.update(auto_selection_info)

        logger.info(f"  Selected reference timestamp: {ref_timestamp}")
        return ref_timestamp, selection_info

    except Exception as e:
        logger.warning(f"  Automatic reference selection failed: {e}")
        logger.info("  Falling back to default reference timestamp: 000")
        selection_info["source"] = "default_fallback"
        selection_info["fallback_reason"] = str(e)
        selection_info["timestamp"] = "000"
        return "000", selection_info


def _determine_registration_mode(
    config: Any, reference_study_dir: Path, modalities: List[str]
) -> tuple:
    """Determine registration mode and reference modality/modalities.

    Args:
        config: Longitudinal registration configuration
        reference_study_dir: Reference study directory
        modalities: List of expected modalities

    Returns:
        Tuple of (mode, reference_modalities)
        - mode: "single_reference" or "per_modality"
        - reference_modalities: List of reference modality names
    """
    priority_str = config.reference_modality_priority

    if priority_str == "per_modality":
        # Per-modality mode: find all available modalities in reference study
        available_modalities = []
        for modality in modalities:
            modality_file = reference_study_dir / f"{modality}.nii.gz"
            if modality_file.exists():
                available_modalities.append(modality)

        if not available_modalities:
            raise ValueError(
                f"No modalities found in reference study {reference_study_dir}"
            )

        return "per_modality", available_modalities

    else:
        # Single reference mode: use priority to select one modality
        # Parse priority string
        if ">" in priority_str:
            priorities = [m.strip() for m in priority_str.split(">")]
        else:
            priorities = [priority_str.strip()]

        # Find highest-priority available modality
        for modality in priorities:
            modality_file = reference_study_dir / f"{modality}.nii.gz"
            if modality_file.exists():
                return "single_reference", [modality]

        raise ValueError(
            f"No modality from priority list '{priority_str}' found in "
            f"reference study {reference_study_dir}"
        )


def _validate_jacobian(
    config: Any,
    transform_path: Path,
    reference_path: Path,
    timestamp: str,
    modality: str,
    results: Dict[str, Any],
) -> None:
    """Validate registration quality using Jacobian determinant statistics.

    Args:
        config: Longitudinal registration config
        transform_path: Path to computed transform
        reference_path: Path to reference image
        timestamp: Timestamp identifier for results key
        modality: Modality name for results key
        results: Results dict to store Jacobian stats into
    """
    try:
        jacobian_stats = compute_jacobian_statistics(
            transform_path=transform_path,
            reference_image_path=reference_path,
        )

        if "jacobian_stats" not in results:
            results["jacobian_stats"] = {}
        results["jacobian_stats"][f"{timestamp}_{modality}"] = jacobian_stats

        ref_config = ReferenceSelectionConfig(
            validate_jacobian=True,
            jacobian_log_threshold=config.reference_selection_jacobian_threshold,
        )
        is_valid, validation_msg = validate_registration_quality(
            jacobian_stats, ref_config
        )

        if is_valid:
            logger.debug(f"      ✓ Jacobian validation passed: {validation_msg}")
        else:
            logger.warning(f"      ! Jacobian validation warning: {validation_msg}")

    except Exception as jac_err:
        logger.warning(f"      ! Jacobian computation failed: {jac_err}")


def _warp_mask_for_qc(
    orchestrator: Any,
    artifacts_base: Path,
    reference_study_dir: Path,
    study_dir: Path,
    modality: str,
    timestamp: str,
    transform_dir: Path,
    transform_path: Path,
    reference_path: Path,
    results: Dict[str, Any],
) -> None:
    """Warp skull-strip mask for QC longitudinal Dice computation.

    Args:
        orchestrator: PreprocessingOrchestrator instance
        artifacts_base: Base artifacts directory for this patient
        reference_study_dir: Reference study directory
        study_dir: Moving study directory
        modality: Modality name
        timestamp: Timestamp identifier
        transform_dir: Directory containing transforms
        transform_path: Path to transform file
        reference_path: Path to reference image
        results: Results dict to store warped mask paths into
    """
    if not (
        orchestrator.qc_manager
        and orchestrator.qc_manager.config.metrics.mask_plausibility.longitudinal_dice
    ):
        return

    ref_mask_path = (
        artifacts_base.parent
        / reference_study_dir.name
        / f"{modality}_brain_mask.nii.gz"
    )
    moving_mask_path = (
        artifacts_base.parent / study_dir.name / f"{modality}_brain_mask.nii.gz"
    )

    if not (ref_mask_path.exists() and moving_mask_path.exists()):
        return

    warped_mask_dir = transform_dir / "warped_masks"
    warped_mask_dir.mkdir(parents=True, exist_ok=True)
    warped_mask_path = warped_mask_dir / f"{timestamp}_{modality}_mask_warped.nii.gz"

    try:
        _warp_mask(
            mask_path=moving_mask_path,
            transform_path=transform_path,
            reference_path=reference_path,
            output_path=warped_mask_path,
        )

        if "warped_masks" not in results:
            results["warped_masks"] = {}
        results["warped_masks"][f"{timestamp}_{modality}"] = {
            "ref_mask": ref_mask_path,
            "warped_mask": warped_mask_path,
        }
        logger.info(f"      ✓ {modality} mask warped for QC")
    except Exception as e:
        logger.warning(f"      ! Failed to warp mask for {modality}: {e}")


def _register_modality_pair(
    registrator: Any,
    config: Any,
    orchestrator: Any,
    reference_path: Path,
    moving_path: Path,
    transform_path: Path,
    transform_dir: Path,
    timestamp: str,
    modality: str,
    artifacts_base: Path,
    reference_study_dir: Path,
    study_dir: Path,
    viz_base: Path,
    results: Dict[str, Any],
) -> bool:
    """Register a single modality pair and perform post-registration QC.

    Handles: registration, Jacobian validation, mask warping, and visualization.

    Args:
        registrator: LongitudinalRegistration instance
        config: Longitudinal registration config
        orchestrator: PreprocessingOrchestrator instance
        reference_path: Path to fixed (reference) image
        moving_path: Path to moving image (overwritten in-place)
        transform_path: Path to save transform
        transform_dir: Directory containing transforms
        timestamp: Timestamp identifier
        modality: Modality name
        artifacts_base: Base artifacts directory
        reference_study_dir: Reference study directory
        study_dir: Moving study directory
        viz_base: Base visualization directory
        results: Results dict to update

    Returns:
        True if registration succeeded, False otherwise
    """
    # Save pre-registration image for visualization
    pre_registration_path = None
    if config.save_visualization:
        temp_file = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False)
        temp_file.close()
        pre_registration_path = Path(temp_file.name)
        shutil.copy(moving_path, pre_registration_path)

    try:
        registrator.register_pair(
            fixed_path=reference_path,
            moving_path=moving_path,
            output_path=moving_path,
            transform_path=transform_path,
        )

        results["transforms"][f"{timestamp}_{modality}"] = str(transform_path)

        # Jacobian validation
        if config.reference_selection_validate_jacobian:
            _validate_jacobian(
                config, transform_path, reference_path, timestamp, modality, results
            )

        # Warp mask for QC
        _warp_mask_for_qc(
            orchestrator, artifacts_base, reference_study_dir, study_dir,
            modality, timestamp, transform_dir, transform_path, reference_path, results,
        )

        # Visualization
        if config.save_visualization and pre_registration_path:
            viz_output = (
                viz_base / study_dir.name / f"longitudinal_registration_{modality}.png"
            )
            viz_output.parent.mkdir(parents=True, exist_ok=True)
            registrator.visualize(
                reference_path=reference_path,
                pre_registration_path=pre_registration_path,
                post_registration_path=moving_path,
                output_path=viz_output,
            )

        return True

    except Exception as e:
        logger.error(f"      ✗ Failed to register {modality}: {e}")
        return False
    finally:
        if pre_registration_path and pre_registration_path.exists():
            pre_registration_path.unlink()


def _execute_single_reference_registration(
    config: Any,
    orchestrator: Any,
    patient_id: str,
    reference_study_dir: Path,
    reference_modality: str,
    all_study_dirs: List[Path],
    artifacts_base: Path,
    viz_base: Path,
    results: Dict[str, Any],
) -> None:
    """Execute single-reference mode: register all modalities to one reference.

    Args:
        config: Configuration
        orchestrator: PreprocessingOrchestrator instance
        patient_id: Patient ID
        reference_study_dir: Reference study directory
        reference_modality: Reference modality name
        all_study_dirs: All study directories
        artifacts_base: Base artifacts directory
        viz_base: Base visualization directory
        results: Results dictionary to update
    """
    registrator = orchestrator._get_component(
        f"longitudinal_reg_{reference_modality}", config
    )

    reference_output_dir = _get_study_output_dir(
        orchestrator, patient_id, reference_study_dir
    )
    reference_path = reference_output_dir / f"{reference_modality}.nii.gz"

    for study_dir in all_study_dirs:
        if study_dir == reference_study_dir:
            continue

        logger.info(f"    Registering study {study_dir.name}...")
        study_output_dir = _get_study_output_dir(orchestrator, patient_id, study_dir)

        for modality in orchestrator.config.modalities:
            moving_path = study_output_dir / f"{modality}.nii.gz"
            if not moving_path.exists():
                logger.debug(f"      Modality {modality} not found in {study_dir.name}")
                continue

            timestamp = study_dir.name.split("-")[-1]
            transform_dir = artifacts_base / "longitudinal_registration"
            transform_dir.mkdir(parents=True, exist_ok=True)
            transform_path = transform_dir / f"{timestamp}_{modality}_to_ref.h5"

            success = _register_modality_pair(
                registrator=registrator,
                config=config,
                orchestrator=orchestrator,
                reference_path=reference_path,
                moving_path=moving_path,
                transform_path=transform_path,
                transform_dir=transform_dir,
                timestamp=timestamp,
                modality=modality,
                artifacts_base=artifacts_base,
                reference_study_dir=reference_study_dir,
                study_dir=study_dir,
                viz_base=viz_base,
                results=results,
            )

            if success:
                logger.info(f"      ✓ {modality} registered")

        results["registered_studies"].append(study_dir.name)


def _execute_per_modality_registration(
    config: Any,
    orchestrator: Any,
    patient_id: str,
    reference_study_dir: Path,
    reference_modalities: List[str],
    all_study_dirs: List[Path],
    artifacts_base: Path,
    viz_base: Path,
    results: Dict[str, Any],
) -> None:
    """Execute per-modality mode: register each modality type independently.

    Args:
        config: Configuration
        orchestrator: PreprocessingOrchestrator instance
        patient_id: Patient ID
        reference_study_dir: Reference study directory
        reference_modalities: List of available modalities in reference
        all_study_dirs: All study directories
        artifacts_base: Base artifacts directory
        viz_base: Base visualization directory
        results: Results dictionary to update
    """
    registrator = orchestrator._get_component(f"longitudinal_reg_{patient_id}", config)

    reference_output_dir = _get_study_output_dir(
        orchestrator, patient_id, reference_study_dir
    )

    registered_studies_set = set()

    for modality in reference_modalities:
        logger.info(f"    Processing modality: {modality}")

        reference_path = reference_output_dir / f"{modality}.nii.gz"
        if not reference_path.exists():
            logger.warning(f"      Reference {modality} not found - skipping")
            continue

        for study_dir in all_study_dirs:
            if study_dir == reference_study_dir:
                continue

            study_output_dir = _get_study_output_dir(
                orchestrator, patient_id, study_dir
            )
            moving_path = study_output_dir / f"{modality}.nii.gz"
            if not moving_path.exists():
                logger.debug(f"      {modality} not found in {study_dir.name}")
                continue

            timestamp = study_dir.name.split("-")[-1]
            transform_dir = artifacts_base / "longitudinal_registration"
            transform_dir.mkdir(parents=True, exist_ok=True)
            transform_path = (
                transform_dir / f"{timestamp}_{modality}_to_ref_{modality}.h5"
            )

            success = _register_modality_pair(
                registrator=registrator,
                config=config,
                orchestrator=orchestrator,
                reference_path=reference_path,
                moving_path=moving_path,
                transform_path=transform_path,
                transform_dir=transform_dir,
                timestamp=timestamp,
                modality=modality,
                artifacts_base=artifacts_base,
                reference_study_dir=reference_study_dir,
                study_dir=study_dir,
                viz_base=viz_base,
                results=results,
            )

            if success:
                logger.info(f"      ✓ {study_dir.name}/{modality} → ref/{modality}")
                registered_studies_set.add(study_dir.name)

    results["registered_studies"] = sorted(list(registered_studies_set))


def _get_study_output_dir(orchestrator: Any, patient_id: str, study_dir: Path) -> Path:
    """Get output directory for a study based on mode.

    Args:
        orchestrator: PreprocessingOrchestrator instance
        patient_id: Patient ID
        study_dir: Study directory

    Returns:
        Output directory path
    """
    if orchestrator.config.mode == "test":
        return Path(orchestrator.config.output_root) / patient_id / study_dir.name
    else:  # pipeline mode
        return study_dir


def _warp_mask(
    mask_path: Path, transform_path: Path, reference_path: Path, output_path: Path
) -> None:
    """Warp binary mask using computed transform.

    Uses nearest neighbor interpolation to preserve binary values.
    Uses ANTsPy for transform application since the transforms are saved
    in ANTs-native .h5 format (not compatible with SimpleITK).

    Args:
        mask_path: Path to moving mask
        transform_path: Path to transform file (.h5 or .mat)
        reference_path: Path to reference image
        output_path: Path to save warped mask

    Raises:
        FileNotFoundError: If transform file does not exist (registration failed upstream)
    """
    if not transform_path.exists():
        raise FileNotFoundError(
            f"Transform file not found: {transform_path}. "
            "Registration likely failed for this pair — cannot warp mask."
        )

    import ants

    mask = ants.image_read(str(mask_path))
    reference = ants.image_read(str(reference_path))

    warped = ants.apply_transforms(
        fixed=reference,
        moving=mask,
        transformlist=[str(transform_path)],
        interpolator="nearestNeighbor",
    )

    ants.image_write(warped, str(output_path))
