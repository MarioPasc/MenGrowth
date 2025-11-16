"""Preprocessing pipeline orchestrator for MenGrowth dataset.

This module coordinates the execution of preprocessing steps on patient data,
managing file paths, mode semantics, and visualization outputs.
"""

from pathlib import Path
from typing import List, Optional
import logging

from mengrowth.preprocessing.src.config import PreprocessingPipelineConfig, DataHarmonizationConfig
from mengrowth.preprocessing.src.data_harmonization.io import NRRDtoNIfTIConverter
from mengrowth.preprocessing.src.data_harmonization.orient import Reorienter
from mengrowth.preprocessing.src.data_harmonization.background import ConservativeBackgroundRemover

logger = logging.getLogger(__name__)


class PreprocessingOrchestrator:
    """Orchestrates preprocessing operations for a patient.

    This class manages the preprocessing pipeline execution, including:
    - File path resolution based on mode (test vs pipeline)
    - Step sequencing (NRRD to NIfTI -> Reorient -> Background removal)
    - Visualization generation
    - Overwrite protection
    """

    def __init__(self, config: DataHarmonizationConfig, verbose: bool = False) -> None:
        """Initialize preprocessing orchestrator.

        Args:
            config: Data harmonization configuration
            verbose: Enable verbose logging
        """
        self.config = config
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)

        # Initialize preprocessing steps
        self.converter = NRRDtoNIfTIConverter(verbose=verbose)
        self.reorienter = Reorienter(
            target_orientation=config.step0_data_harmonization.reorient_to,
            verbose=verbose
        )
        self.background_remover = ConservativeBackgroundRemover(
            config=config.step0_data_harmonization.background_zeroing,
            verbose=verbose
        )

        self.logger.info("Preprocessing orchestrator initialized")

    def _get_study_directories(self, patient_id: str) -> List[Path]:
        """Get list of study directories for a patient.

        Args:
            patient_id: Patient ID (e.g., "MenGrowth-0015")

        Returns:
            List of study directory paths

        Raises:
            FileNotFoundError: If patient directory does not exist
        """
        patient_dir = Path(self.config.dataset_root) / patient_id

        if not patient_dir.exists():
            raise FileNotFoundError(f"Patient directory not found: {patient_dir}")

        # Find all study directories (e.g., MenGrowth-0015-000, MenGrowth-0015-001)
        study_dirs = sorted([d for d in patient_dir.iterdir() if d.is_dir()])

        self.logger.info(f"Found {len(study_dirs)} study directories for {patient_id}")

        return study_dirs

    def _get_modality_file(self, study_dir: Path, modality: str) -> Optional[Path]:
        """Find modality file in study directory.

        Args:
            study_dir: Study directory path
            modality: Modality name (e.g., "t1c", "t2w")

        Returns:
            Path to modality file, or None if not found
        """
        # Look for NRRD file
        nrrd_file = study_dir / f"{modality}.nrrd"

        if nrrd_file.exists():
            return nrrd_file

        self.logger.debug(f"Modality {modality} not found in {study_dir.name}")
        return None

    def _get_output_paths(
        self,
        patient_id: str,
        study_dir: Path,
        modality: str
    ) -> dict:
        """Get output paths for all processing steps.

        Args:
            patient_id: Patient ID
            study_dir: Study directory path
            modality: Modality name

        Returns:
            Dictionary with output paths for each step
        """
        study_name = study_dir.name

        if self.config.mode == "test":
            # Test mode: write to separate output directory
            output_base = Path(self.config.output_root) / patient_id / study_name
            viz_base = Path(self.config.viz_root) / patient_id / study_name
        else:
            # Pipeline mode: write in-place
            output_base = study_dir
            viz_base = Path(self.config.viz_root) / patient_id / study_name

        # Ensure directories exist
        output_base.mkdir(parents=True, exist_ok=True)
        viz_base.mkdir(parents=True, exist_ok=True)

        return {
            "nifti": output_base / f"{modality}.nii.gz",
            "reoriented": output_base / f"{modality}.nii.gz",  # Same file, in-place
            "masked": output_base / f"{modality}.nii.gz",  # Same file, in-place
            "viz_convert": viz_base / f"step0_convert_{modality}.png",
            "viz_reorient": viz_base / f"step0_reorient_{modality}.png",
            "viz_background": viz_base / f"step0_background_{modality}.png",
        }

    def run_patient(self, patient_id: str) -> None:
        """Run preprocessing pipeline for a single patient.

        Args:
            patient_id: Patient ID to process

        Raises:
            FileNotFoundError: If patient directory not found
            RuntimeError: If processing fails
        """
        self.logger.info(f"{'='*80}")
        self.logger.info(f"Processing patient: {patient_id}")
        self.logger.info(f"Mode: {self.config.mode}, Overwrite: {self.config.overwrite}")
        self.logger.info(f"{'='*80}")

        # Get study directories
        study_dirs = self._get_study_directories(patient_id)

        total_processed = 0
        total_skipped = 0
        total_errors = 0

        # Process each study
        for study_idx, study_dir in enumerate(study_dirs, 1):
            self.logger.info(f"\n[Study {study_idx}/{len(study_dirs)}] {study_dir.name}")

            # Process each modality
            for modality in self.config.modalities:
                self.logger.info(f"  Processing modality: {modality}")

                try:
                    # Find input file
                    input_file = self._get_modality_file(study_dir, modality)

                    if input_file is None:
                        self.logger.warning(f"    Modality {modality} not found - skipping")
                        total_skipped += 1
                        continue

                    # Get output paths
                    paths = self._get_output_paths(patient_id, study_dir, modality)

                    # Check overwrite conditions (strict: error and halt if exists)
                    if paths["nifti"].exists() and not self.config.overwrite:
                        if self.config.mode == "pipeline":
                            raise FileExistsError(
                                f"Output file exists and overwrite=False: {paths['nifti']}\n"
                                "Set overwrite=true in config or use test mode"
                            )
                        else:
                            self.logger.warning(
                                f"    Output exists and overwrite=False - skipping {modality}"
                            )
                            total_skipped += 1
                            continue

                    # Step 1: NRRD � NIfTI
                    self.logger.info("    [1/3] Converting NRRD to NIfTI...")
                    self.converter.execute(
                        input_file,
                        paths["nifti"],
                        allow_overwrite=self.config.overwrite
                    )

                    if self.config.step0_data_harmonization.save_visualization:
                        self.converter.visualize(
                            input_file,
                            paths["nifti"],
                            paths["viz_convert"]
                        )

                    # Step 2: Reorient (in-place)
                    self.logger.info(
                        f"    [2/3] Reorienting to {self.config.step0_data_harmonization.reorient_to}..."
                    )
                    # Create temp path for reorientation
                    temp_reoriented = paths["reoriented"].parent / f"_temp_{modality}_reoriented.nii.gz"

                    self.reorienter.execute(
                        paths["nifti"],
                        temp_reoriented,
                        allow_overwrite=True
                    )

                    if self.config.step0_data_harmonization.save_visualization:
                        self.reorienter.visualize(
                            paths["nifti"],
                            temp_reoriented,
                            paths["viz_reorient"]
                        )

                    # Replace original with reoriented
                    temp_reoriented.replace(paths["nifti"])

                    # Step 3: Background removal (in-place)
                    self.logger.info("    [3/3] Removing background...")
                    temp_masked = paths["masked"].parent / f"_temp_{modality}_masked.nii.gz"

                    self.background_remover.execute(
                        paths["nifti"],
                        temp_masked,
                        allow_overwrite=True
                    )

                    if self.config.step0_data_harmonization.save_visualization:
                        self.background_remover.visualize(
                            paths["nifti"],
                            temp_masked,
                            paths["viz_background"]
                        )

                    # Replace with masked version
                    temp_masked.replace(paths["nifti"])

                    self.logger.info(f"     Successfully processed {modality}")
                    total_processed += 1

                except FileExistsError as e:
                    # Re-raise overwrite errors (halt execution)
                    raise
                except Exception as e:
                    self.logger.error(f"     Error processing {modality}: {e}")
                    total_errors += 1
                    # Continue with next modality

        # Summary
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"Patient {patient_id} processing complete:")
        self.logger.info(f"  Processed: {total_processed}")
        self.logger.info(f"  Skipped:   {total_skipped}")
        self.logger.info(f"  Errors:    {total_errors}")
        self.logger.info(f"{'='*80}\n")


def run_preprocessing(config: PreprocessingPipelineConfig, patient_id: Optional[str] = None) -> None:
    """Run preprocessing pipeline on selected patients.

    Args:
        config: Preprocessing pipeline configuration
        patient_id: Specific patient to process (overrides config.patient_selector)

    Raises:
        ValueError: If patient_selector is invalid
        FileNotFoundError: If patient directory not found
        RuntimeError: If processing fails
    """
    # Extract data harmonization config
    dh_config = config.data_harmonization

    if not dh_config.enabled:
        logger.info("Data harmonization is disabled in config - exiting")
        return

    # Override patient_id if provided
    if patient_id is not None:
        dh_config.patient_id = patient_id
        dh_config.patient_selector = "single"
        logger.info(f"Overriding config: processing single patient {patient_id}")

    # Initialize orchestrator
    orchestrator = PreprocessingOrchestrator(dh_config, verbose=True)

    # Process patients
    if dh_config.patient_selector == "single":
        orchestrator.run_patient(dh_config.patient_id)

    elif dh_config.patient_selector == "all":
        # Get all patient directories
        dataset_root = Path(dh_config.dataset_root)
        patient_dirs = sorted([d for d in dataset_root.iterdir() if d.is_dir() and d.name.startswith("MenGrowth-")])

        logger.info(f"Processing all patients: {len(patient_dirs)} found")

        for idx, patient_dir in enumerate(patient_dirs, 1):
            logger.info(f"\n\n{'#'*80}")
            logger.info(f"# Patient {idx}/{len(patient_dirs)}: {patient_dir.name}")
            logger.info(f"{'#'*80}")

            try:
                orchestrator.run_patient(patient_dir.name)
            except Exception as e:
                logger.error(f"Failed to process {patient_dir.name}: {e}")
                # Continue with next patient

        logger.info(f"\n\n{'#'*80}")
        logger.info(f"# All patients processing complete ({len(patient_dirs)} patients)")
        logger.info(f"{'#'*80}")

    else:
        raise ValueError(
            f"Invalid patient_selector: {dh_config.patient_selector}. Must be 'single' or 'all'"
        )
