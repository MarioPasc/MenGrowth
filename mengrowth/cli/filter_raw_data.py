"""Command-line interface for filtering reorganized MRI data.

This script provides a CLI for filtering reorganized MRI data to ensure studies
meet completeness requirements, apply orientation priority rules, and enforce
minimum longitudinal study requirements per patient.

Usage:
    mengrowth-filter --data-root /path/to/organized
    mengrowth-filter --data-root /data/organized --dry-run
    mengrowth-filter --data-root /data/organized --config custom.yaml --verbose
"""

import argparse
import logging
import sys
from pathlib import Path

from mengrowth.preprocessing.config import load_preprocessing_config
from mengrowth.preprocessing.utils.filter_raw_data import filter_raw_data


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
        description="Filter reorganized MRI data based on quality and completeness criteria.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic filtering (operates on reorganized data)
  mengrowth-filter --data-root /data/organized

  # Dry run to preview operations
  mengrowth-filter --data-root /data/organized --dry-run

  # Use custom configuration file
  mengrowth-filter --data-root /data/organized --config custom.yaml

  # Verbose logging for debugging
  mengrowth-filter --data-root /data/organized --verbose

Filtering criteria (configured in YAML):
  - Required sequences: List of MRI sequences needed (e.g., t1c, t1n, t2f, t2w)
  - Allowed missing sequences per study: Maximum sequences that can be missing
  - Minimum studies per patient: Minimum longitudinal studies required
  - Orientation priority: Priority order for selecting orientations

Input structure expected:
  {data-root}/MenGrowth-2025/P{patient_id}/{study_number}/*.nrrd

Output:
  - Studies/patients not meeting criteria are deleted
  - Rejections are appended to {data-root}/rejected_files.csv with stage=1
  - Oriented sequences are normalized (e.g., t1c-axial.nrrd -> t1c.nrrd)

IMPORTANT: This operates on the OUTPUT of reorganization, not the source data.
Run mengrowth-reorganize first before running this filtering step.
        """,
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Root directory containing reorganized data (output from mengrowth-reorganize).",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to preprocessing YAML config file (default: configs/raw_data.yaml in repo root).",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate operations without actually deleting files or renaming (useful for testing).",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose (DEBUG level) logging output.",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point for the CLI.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    args = parse_arguments()

    # Setup logging
    setup_logging(verbose=args.verbose)
    logger = logging.getLogger(__name__)

    try:
        # Determine config path
        if args.config:
            config_path = args.config
        else:
            # Default to configs/raw_data.yaml in repository root
            # Assuming this script is in mengrowth/cli/, repo root is 2 levels up
            repo_root = Path(__file__).parent.parent.parent
            config_path = repo_root / "configs" / "raw_data.yaml"

        logger.info(f"Loading configuration from: {config_path}")

        # Load configuration
        preprocessing_config = load_preprocessing_config(config_path)

        # Validate that filtering config exists
        if preprocessing_config.filtering is None:
            logger.error(
                "Filtering configuration not found in config file. "
                "Please add 'filtering:' section to your YAML config."
            )
            return 1

        # Validate data directory
        if not args.data_root.exists():
            logger.error(f"Data root directory does not exist: {args.data_root}")
            return 1

        mengrowth_dir = args.data_root / "MenGrowth-2025"
        if not mengrowth_dir.exists():
            logger.error(
                f"MenGrowth-2025 directory not found in {args.data_root}. "
                "Please run mengrowth-reorganize first."
            )
            return 1

        # Log configuration
        logger.info(f"Data root: {args.data_root}")
        logger.info(f"Dry run mode: {args.dry_run}")
        logger.info(
            f"Required sequences: {preprocessing_config.filtering.sequences}"
        )
        logger.info(
            f"Allowed missing per study: "
            f"{preprocessing_config.filtering.allowed_missing_sequences_per_study}"
        )
        logger.info(
            f"Min studies per patient: "
            f"{preprocessing_config.filtering.min_studies_per_patient}"
        )
        logger.info(
            f"Keep only required sequences: "
            f"{preprocessing_config.filtering.keep_only_required_sequences}"
        )
        logger.info(
            f"Re-identify patients: "
            f"{preprocessing_config.filtering.reid_patients}"
        )

        # Execute filtering
        stats = filter_raw_data(
            data_root=args.data_root,
            config=preprocessing_config,
            dry_run=args.dry_run,
        )

        # Report results
        logger.info("=" * 60)
        logger.info("FILTERING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Patients kept: {stats['patients_kept']}")
        logger.info(f"Patients removed: {stats['patients_removed']}")
        logger.info(f"Total studies processed: {stats['studies_processed']}")

        if preprocessing_config.filtering.keep_only_required_sequences:
            logger.info(f"Non-required sequences removed: {stats['sequences_removed']}")

        if preprocessing_config.filtering.reid_patients:
            logger.info(f"Patients re-identified: {stats['patients_renamed']}")
            logger.info(f"ID mapping saved to: {args.data_root}/id_mapping.json")

        logger.info("")
        logger.info(f"Updated rejected files report: {args.data_root}/rejected_files.csv")
        logger.info(
            "This CSV now includes both reorganization rejections (stage=0) "
            "and filtering rejections (stage=1)."
        )

        if args.dry_run:
            logger.info("\nThis was a DRY RUN - no files were actually deleted or renamed.")
            logger.info("Remove --dry-run flag to perform actual filtering.")

        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return 1

    except KeyError as e:
        logger.error(f"Configuration error: {e}")
        return 1

    except Exception as e:
        logger.exception(f"Unexpected error occurred: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
