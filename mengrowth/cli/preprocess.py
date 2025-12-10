"""Command-line interface for preprocessing MRI data.

This script provides a CLI for running the preprocessing pipeline on MenGrowth
dataset, including data harmonization (NRRD→NIfTI, reorientation, background removal).

Usage:
    mengrowth-preprocess --config configs/preprocessing.yaml
    mengrowth-preprocess --config configs/preprocessing.yaml --patient MenGrowth-0015
    mengrowth-preprocess --config configs/preprocessing.yaml --dry-run
"""

import argparse
import logging
import sys
from pathlib import Path

from mengrowth.preprocessing.src.config import load_preprocessing_pipeline_config
from mengrowth.preprocessing.src.preprocess import run_preprocessing


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the CLI.

    Args:
        verbose: If True, set logging level to DEBUG; otherwise INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Run preprocessing pipeline on MenGrowth dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single patient (from config)
  mengrowth-preprocess --config configs/preprocessing.yaml

  # Override patient from command line
  mengrowth-preprocess --config configs/preprocessing.yaml --patient MenGrowth-0015

  # Dry run to check configuration
  mengrowth-preprocess --config configs/preprocessing.yaml --dry-run

  # Verbose logging for debugging
  mengrowth-preprocess --config configs/preprocessing.yaml --verbose

Processing steps (data harmonization):
  1. NRRD → NIfTI conversion (preserving metadata)
  2. Reorientation to RAS or LPS (configurable)
  3. Conservative background removal (no skull-stripping)

Modes:
  - test: Write outputs to separate directory (safe, no overwrite)
  - pipeline: In-place processing with strict overwrite protection

Output structure:
  Test mode:
    {output_root}/{patient_id}/{study}/modality.nii.gz
  Pipeline mode:
    {dataset_root}/{patient_id}/{study}/modality.nii.gz (in-place)
  Visualizations:
    {viz_root}/{patient_id}/{study}/step0_*.png
        """,
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/preprocessing.yaml"),
        help="Path to preprocessing configuration YAML file (default: configs/preprocessing.yaml).",
    )

    parser.add_argument(
        "--patient",
        type=str,
        help="Patient ID to process (overrides config patient_selector and patient_id).",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and display processing plan without executing.",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging (DEBUG level).",
    )

    return parser.parse_args()


def main() -> int:
    """Main CLI entry point.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    args = parse_arguments()

    # Setup logging
    setup_logging(verbose=args.verbose)
    logger = logging.getLogger(__name__)

    try:
        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        config = load_preprocessing_pipeline_config(args.config)

        # Display configuration summary
        dh_config = config  # loader now returns DataHarmonizationConfig directly
        logger.info("")
        logger.info("="*80)
        logger.info("PREPROCESSING CONFIGURATION")
        logger.info("="*80)
        logger.info(f"Enabled:          {dh_config.enabled}")
        logger.info(f"Mode:             {dh_config.mode}")
        logger.info(f"Patient selector: {dh_config.patient_selector}")
        if dh_config.patient_selector == "single" or args.patient:
            patient_display = args.patient if args.patient else dh_config.patient_id
            logger.info(f"Patient ID:       {patient_display}")
        logger.info(f"Dataset root:     {dh_config.dataset_root}")
        logger.info(f"Output root:      {dh_config.output_root}")
        logger.info(f"Viz root:         {dh_config.viz_root}")
        logger.info(f"Overwrite:        {dh_config.overwrite}")
        logger.info(f"Modalities:       {', '.join(dh_config.modalities)}")
        logger.info("")
        logger.info("Pipeline Configuration:")
        if hasattr(dh_config, 'steps') and dh_config.steps:
            logger.info(f"  Pipeline steps ({len(dh_config.steps)}):")
            for i, step in enumerate(dh_config.steps, 1):
                logger.info(f"    {i}. {step}")
            logger.info(f"  Step configs:   {len(dh_config.step_configs)} configured")
        else:
            logger.info("  Using legacy step configuration")
        logger.info("="*80)
        logger.info("")

        # Dry run mode
        if args.dry_run:
            logger.info("DRY RUN MODE - No processing will be performed")
            logger.info("")
            logger.info("Configuration validation: PASSED ✓")

            # Display what would be processed
            if args.patient or dh_config.patient_selector == "single":
                patient = args.patient if args.patient else dh_config.patient_id
                logger.info(f"Would process: {patient}")
            else:
                logger.info("Would process: ALL patients in dataset")

            logger.info("")
            logger.info("To execute, run without --dry-run flag")
            return 0

        # Run preprocessing
        logger.info("Starting preprocessing pipeline...")
        logger.info("")

        run_preprocessing(config, patient_id=args.patient)

        logger.info("")
        logger.info("="*80)
        logger.info("PREPROCESSING COMPLETE ✓")
        logger.info("="*80)

        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return 1
    except FileExistsError as e:
        logger.error(f"Overwrite protection: {e}")
        logger.error("Set overwrite=true in config or use test mode")
        return 1
    except KeyboardInterrupt:
        logger.warning("Processing interrupted by user")
        return 130
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
