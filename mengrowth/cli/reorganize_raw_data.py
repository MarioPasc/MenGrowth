"""Command-line interface for reorganizing raw MRI data.

This script provides a CLI for reorganizing multimodal MRI data from various
input structures into a standardized directory format for reproducible research.

Usage:
    mengrowth-reorganize --input-root /path/to/raw/processed --output-root /path/to/organized
    mengrowth-reorganize --input-root /data/raw --output-root /data/clean --dry-run
    mengrowth-reorganize --input-root /data/raw --output-root /data/clean --config custom.yaml
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from mengrowth.preprocessing.config import load_preprocessing_config
from mengrowth.preprocessing.utils.metadata import MetadataManager
from mengrowth.preprocessing.utils.reorganize_raw_data import reorganize_raw_data


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
        description="Reorganize raw MRI data into standardized directory structure.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic reorganization
  mengrowth-reorganize --input-root /data/raw/processed --output-root /data/organized

  # Dry run to preview operations
  mengrowth-reorganize --input-root /data/raw --output-root /data/clean --dry-run

  # Use custom configuration file
  mengrowth-reorganize --input-root /data/raw --output-root /data/clean --config custom.yaml

  # Verbose logging for debugging
  mengrowth-reorganize --input-root /data/raw --output-root /data/clean --verbose

Output structure:
  {output-root}/MenGrowth-2025/P{patient_id}/{study_number}/*.nrrd

Input sources expected:
  - {input-root}/source/baseline/RM/{modality}/{patient}/
  - {input-root}/source/baseline/TC/{patient}/
  - {input-root}/source/controls/{patient}/{control1,control2,...}/
  - {input-root}/extension_1/{patient_id}/{primera,segunda,tercera,...}/
        """,
    )

    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Root directory containing 'source' and 'extension_1' folders with raw data.",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Root directory for reorganized output (will create MenGrowth-2025 subdirectory).",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to preprocessing YAML config file (default: configs/preprocessing.yaml in repo root).",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate operations without actually copying files (useful for testing).",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose (DEBUG level) logging output.",
    )

    parser.add_argument(
        "--metadata-xlsx",
        type=Path,
        default=None,
        help="Path to clinical metadata xlsx file (overrides config setting).",
    )

    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Disable metadata processing even if configured.",
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
            # Default to configs/preprocessing.yaml in repository root
            # Assuming this script is in mengrowth/cli/, repo root is 2 levels up
            repo_root = Path(__file__).parent.parent.parent
            config_path = repo_root / "configs" / "preprocessing.yaml"

        logger.info(f"Loading configuration from: {config_path}")

        # Load configuration
        preprocessing_config = load_preprocessing_config(config_path)

        # Validate input directory
        if not args.input_root.exists():
            logger.error(f"Input root directory does not exist: {args.input_root}")
            return 1

        # Log configuration
        logger.info(f"Input root: {args.input_root}")
        logger.info(f"Output root: {args.output_root}")
        logger.info(f"Dry run mode: {args.dry_run}")

        # Initialize metadata manager if configured and not disabled
        metadata_manager: Optional[MetadataManager] = None
        if not args.no_metadata and preprocessing_config.metadata and preprocessing_config.metadata.enabled:
            xlsx_path = args.metadata_xlsx or (
                Path(preprocessing_config.metadata.xlsx_path)
                if preprocessing_config.metadata.xlsx_path
                else None
            )

            if xlsx_path and xlsx_path.exists():
                logger.info(f"Loading clinical metadata from: {xlsx_path}")
                metadata_manager = MetadataManager()
                metadata_manager.load_from_xlsx(xlsx_path)
                logger.info(f"Loaded metadata for {len(metadata_manager.get_patient_ids())} patients")
            elif xlsx_path:
                logger.warning(f"Metadata xlsx file not found: {xlsx_path}")
            else:
                logger.warning("Metadata enabled but no xlsx_path configured")

        # Execute reorganization
        stats = reorganize_raw_data(
            input_root=args.input_root,
            output_root=args.output_root,
            config=preprocessing_config.raw_data,
            dry_run=args.dry_run,
        )

        # Export metadata if loaded
        if metadata_manager and not args.dry_run:
            output_csv = args.output_root / preprocessing_config.metadata.output_csv_name
            output_json = args.output_root / preprocessing_config.metadata.output_json_name
            metadata_manager.export_enriched_csv(output_csv)
            metadata_manager.export_json(output_json)
            logger.info(f"Exported enriched metadata to: {output_csv}")
            logger.info(f"Exported metadata JSON to: {output_json}")

        # Report results
        logger.info("=" * 60)
        logger.info("REORGANIZATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Files copied: {stats['copied']}")
        logger.info(f"Files skipped (duplicates): {stats['skipped']}")
        logger.info(f"Errors encountered: {stats['errors']}")
        logger.info("")
        logger.info(f"Rejected files report: {args.output_root}/rejected_files.csv")
        logger.info("This CSV contains all files that were excluded and the reasons for rejection.")

        if metadata_manager:
            summary = metadata_manager.get_clinical_summary()
            logger.info("")
            logger.info(f"Clinical metadata: {summary['total_patients']} patients loaded")

        if args.dry_run:
            logger.info("\nThis was a DRY RUN - no files were actually copied.")
            logger.info("Remove --dry-run flag to perform actual reorganization.")

        if stats["errors"] > 0:
            logger.warning(f"\n{stats['errors']} errors occurred during processing.")
            return 1

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
