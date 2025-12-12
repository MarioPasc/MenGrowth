"""Longitudinal registration step: register studies across timestamps for same patient.

This is a patient-level step that operates on all studies (timestamps) together:
1. Load reference timestamp from YAML or use default "000"
2. Determine reference modality/modalities based on reference_modality_priority
3. Register all other timestamps to the reference timestamp
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
import logging
import yaml
import shutil
import tempfile

from mengrowth.preprocessing.src.config import StepExecutionContext

logger = logging.getLogger(__name__)


def execute(
    context: StepExecutionContext,
    total_steps: int,
    current_step_num: int
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

    logger.info(f"  Processing {len(all_study_dirs)} timestamps for patient {patient_id}")

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
        logger.warning(f"  Only {len(all_study_dirs)} timestamp(s) found - skipping longitudinal registration")
        return results

    # Step 1: Determine reference timestamp
    reference_timestamp = _get_reference_timestamp(
        patient_id=patient_id,
        all_study_dirs=all_study_dirs,
        yaml_path=config.reference_timestamp_per_study
    )
    results["reference_timestamp"] = reference_timestamp

    logger.info(f"  Reference timestamp: {reference_timestamp}")

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
    reference_output_dir = _get_study_output_dir(orchestrator, patient_id, reference_study_dir)
    mode, reference_modalities = _determine_registration_mode(
        config=config,
        reference_study_dir=reference_output_dir,
        modalities=orchestrator.config.modalities
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
            results=results
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
            results=results
        )

    logger.info(f"  Successfully registered {len(results['registered_studies'])} timestamps")

    return results


def _get_reference_timestamp(
    patient_id: str,
    all_study_dirs: List[Path],
    yaml_path: Optional[str]
) -> str:
    """Get reference timestamp for this patient from YAML or use default.

    Args:
        patient_id: Patient ID (e.g., "MenGrowth-0006")
        all_study_dirs: List of all study directories
        yaml_path: Path to YAML file with patient->timestamp mapping

    Returns:
        Reference timestamp string (e.g., "000", "001")
    """
    # Try to load from YAML if provided
    if yaml_path and Path(yaml_path).exists():
        try:
            with open(yaml_path, 'r') as f:
                timestamp_map = yaml.safe_load(f)

            if patient_id in timestamp_map:
                ref_timestamp = str(timestamp_map[patient_id]).zfill(3)
                logger.info(f"  Reference timestamp from YAML: {ref_timestamp}")
                return ref_timestamp
        except Exception as e:
            logger.warning(f"  Failed to load reference timestamp from YAML: {e}")

    # Default to "000"
    logger.info("  Using default reference timestamp: 000")
    return "000"


def _determine_registration_mode(
    config: Any,
    reference_study_dir: Path,
    modalities: List[str]
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


def _execute_single_reference_registration(
    config: Any,
    orchestrator: Any,
    patient_id: str,
    reference_study_dir: Path,
    reference_modality: str,
    all_study_dirs: List[Path],
    artifacts_base: Path,
    viz_base: Path,
    results: Dict[str, Any]
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
    # Get or create longitudinal registrator
    registrator = orchestrator._get_component(
        f"longitudinal_reg_{reference_modality}",
        config
    )

    # Get output directory where preprocessed files are located
    reference_output_dir = _get_study_output_dir(orchestrator, patient_id, reference_study_dir)
    reference_path = reference_output_dir / f"{reference_modality}.nii.gz"

    # Register each non-reference study
    for study_dir in all_study_dirs:
        if study_dir == reference_study_dir:
            continue  # Skip reference study

        logger.info(f"    Registering study {study_dir.name}...")

        # Get output directory (depends on mode: test vs pipeline)
        study_output_dir = _get_study_output_dir(orchestrator, patient_id, study_dir)

        # Register all modalities in this study to reference
        for modality in orchestrator.config.modalities:
            moving_path = study_output_dir / f"{modality}.nii.gz"

            if not moving_path.exists():
                logger.debug(f"      Modality {modality} not found in {study_dir.name}")
                continue

            # Define transform path
            timestamp = study_dir.name.split('-')[-1]
            transform_dir = artifacts_base / "longitudinal_registration"
            transform_dir.mkdir(parents=True, exist_ok=True)
            transform_path = transform_dir / f"{timestamp}_{modality}_to_ref.h5"

            # Save pre-registration image for visualization
            pre_registration_path = None
            if config.save_visualization:
                temp_file = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False)
                temp_file.close()
                pre_registration_path = Path(temp_file.name)
                shutil.copy(moving_path, pre_registration_path)

            try:
                # Perform registration (implementation depends on registrator)
                registrator.register_pair(
                    fixed_path=reference_path,
                    moving_path=moving_path,
                    output_path=moving_path,  # Overwrite in-place
                    transform_path=transform_path
                )

                results["transforms"][f"{timestamp}_{modality}"] = str(transform_path)
                logger.info(f"      ✓ {modality} registered")

                # Generate visualization if enabled
                if config.save_visualization and pre_registration_path:
                    viz_output = viz_base / "longitudinal_registration" / f"{timestamp}_{modality}_to_ref.png"
                    viz_output.parent.mkdir(parents=True, exist_ok=True)
                    registrator.visualize(
                        reference_path=reference_path,
                        pre_registration_path=pre_registration_path,
                        post_registration_path=moving_path,
                        output_path=viz_output
                    )

            except Exception as e:
                logger.error(f"      ✗ Failed to register {modality}: {e}")
                continue
            finally:
                # Clean up temporary pre-registration file
                if pre_registration_path and pre_registration_path.exists():
                    pre_registration_path.unlink()

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
    results: Dict[str, Any]
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
    # Get or create longitudinal registrator
    registrator = orchestrator._get_component(
        f"longitudinal_reg_{patient_id}",
        config
    )

    # Get output directory where preprocessed files are located
    reference_output_dir = _get_study_output_dir(orchestrator, patient_id, reference_study_dir)

    # Track which studies have been registered (at least one modality succeeded)
    registered_studies_set = set()

    # For each modality type, register across timestamps
    for modality in reference_modalities:
        logger.info(f"    Processing modality: {modality}")

        reference_path = reference_output_dir / f"{modality}.nii.gz"

        if not reference_path.exists():
            logger.warning(f"      Reference {modality} not found - skipping")
            continue

        # Register this modality from each non-reference study
        for study_dir in all_study_dirs:
            if study_dir == reference_study_dir:
                continue  # Skip reference study

            # Get output directory
            study_output_dir = _get_study_output_dir(orchestrator, patient_id, study_dir)
            moving_path = study_output_dir / f"{modality}.nii.gz"

            if not moving_path.exists():
                logger.debug(f"      {modality} not found in {study_dir.name}")
                continue

            # Define transform path
            timestamp = study_dir.name.split('-')[-1]
            transform_dir = artifacts_base / "longitudinal_registration"
            transform_dir.mkdir(parents=True, exist_ok=True)
            transform_path = transform_dir / f"{timestamp}_{modality}_to_ref_{modality}.h5"

            # Save pre-registration image for visualization
            pre_registration_path = None
            if config.save_visualization:
                temp_file = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False)
                temp_file.close()
                pre_registration_path = Path(temp_file.name)
                shutil.copy(moving_path, pre_registration_path)

            try:
                # Perform registration
                registrator.register_pair(
                    fixed_path=reference_path,
                    moving_path=moving_path,
                    output_path=moving_path,  # Overwrite in-place
                    transform_path=transform_path
                )

                results["transforms"][f"{timestamp}_{modality}"] = str(transform_path)
                logger.info(f"      ✓ {study_dir.name}/{modality} → ref/{modality}")

                # Mark this study as having at least one successful registration
                registered_studies_set.add(study_dir.name)

                # Generate visualization if enabled
                if config.save_visualization and pre_registration_path:
                    viz_output = viz_base / "longitudinal_registration" / f"{timestamp}_{modality}_to_ref_{modality}.png"
                    viz_output.parent.mkdir(parents=True, exist_ok=True)
                    registrator.visualize(
                        reference_path=reference_path,
                        pre_registration_path=pre_registration_path,
                        post_registration_path=moving_path,
                        output_path=viz_output
                    )

            except Exception as e:
                logger.error(f"      ✗ Failed: {e}")
                continue
            finally:
                # Clean up temporary pre-registration file
                if pre_registration_path and pre_registration_path.exists():
                    pre_registration_path.unlink()

    # Convert set to list for results
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
