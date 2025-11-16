"""Command-line interface for dataset quality analysis.

This module provides a CLI for analyzing MRI dataset quality metrics,
including spatial properties, intensity statistics, and consistency checks.
"""

import argparse
import logging
import sys
from pathlib import Path

from mengrowth.preprocessing.config import load_quality_analysis_config
from mengrowth.preprocessing.quality_analysis.analyzer import QualityAnalyzer


def setup_logging(verbose: bool) -> None:
    """Configure logging with appropriate level and format.

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
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Analyze MRI dataset quality metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze dataset using default config
  mengrowth-analyze-dataset \\
    --input /path/to/dataset \\
    --output /path/to/results

  # Use custom configuration file
  mengrowth-analyze-dataset \\
    --input /path/to/dataset \\
    --output /path/to/results \\
    --config configs/custom_quality_analysis.yaml

  # Dry-run mode (scan dataset without loading images)
  mengrowth-analyze-dataset \\
    --input /path/to/dataset \\
    --output /path/to/results \\
    --dry-run

  # Enable verbose logging
  mengrowth-analyze-dataset \\
    --input /path/to/dataset \\
    --output /path/to/results \\
    --verbose

Expected dataset structure:
  input_dir/
    MenGrowth-0001/
      MenGrowth-0001-000/
        t1c.nrrd
        t1n.nrrd
        t2w.nrrd
      MenGrowth-0001-001/
        ...
    MenGrowth-0002/
      ...

Output structure:
  output_dir/
    per_study_metrics.csv
    per_patient_summary.csv
    summary_statistics.json
    per_sequence_statistics.json
    analysis_metadata.json
        """,
    )

    parser.add_argument(
        "--input",
        type=Path,
        help="Path to input dataset directory (overrides config file)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="Path to output directory for analysis results (overrides config file)",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/quality_analysis.yaml"),
        help="Path to quality analysis configuration file (default: configs/quality_analysis.yaml)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan dataset structure without loading images or computing metrics",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG level) logging",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point for quality analysis CLI.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    args = parse_arguments()
    setup_logging(verbose=args.verbose)
    logger = logging.getLogger(__name__)

    try:
        # Load configuration
        logger.info(f"Loading configuration from: {args.config}")
        config = load_quality_analysis_config(args.config)

        # Override config with command-line arguments if provided
        if args.input:
            config.input_dir = args.input
            logger.info(f"Using input directory from CLI: {args.input}")

        if args.output:
            config.output_dir = args.output
            logger.info(f"Using output directory from CLI: {args.output}")

        if args.dry_run:
            config.dry_run = True
            logger.info("Dry-run mode enabled")

        if args.verbose:
            config.verbose = True

        # Log configuration summary
        logger.info("=" * 60)
        logger.info("Quality Analysis Configuration:")
        logger.info(f"  Input directory:    {config.input_dir}")
        logger.info(f"  Output directory:   {config.output_dir}")
        logger.info(f"  File format:        {config.file_format}")
        logger.info(f"  Expected sequences: {', '.join(config.expected_sequences)}")
        logger.info(f"  Parallel enabled:   {config.parallel.enabled}")
        if config.parallel.enabled:
            logger.info(f"  Number of workers:  {config.parallel.n_workers or 'all CPUs'}")
        logger.info(f"  Dry-run mode:       {config.dry_run}")
        logger.info("=" * 60)

        # Validate input directory
        if not config.input_dir.exists():
            logger.error(f"Input directory does not exist: {config.input_dir}")
            return 1

        if not config.input_dir.is_dir():
            logger.error(f"Input path is not a directory: {config.input_dir}")
            return 1

        # Create analyzer
        logger.info("Initializing quality analyzer...")
        analyzer = QualityAnalyzer(config)

        # Run analysis
        logger.info("Starting analysis...")
        results = analyzer.run_analysis()

        # Save results
        if not config.dry_run:
            logger.info("Saving analysis results...")
            saved_files = analyzer.save_results()

            logger.info("=" * 60)
            logger.info("Analysis Complete!")
            logger.info("=" * 60)
            logger.info("Saved files:")
            for file_type, file_path in saved_files.items():
                logger.info(f"  {file_type}: {file_path}")
            logger.info("=" * 60)

            # Print summary statistics
            if "summary" in results and "patient_statistics" in results["summary"]:
                patient_stats = results["summary"]["patient_statistics"]
                logger.info("Dataset Summary:")
                logger.info(f"  Total patients: {patient_stats['total_patients']}")
                logger.info(f"  Total studies:  {patient_stats['total_studies']}")
                logger.info(f"  Mean studies per patient: {patient_stats['mean_studies']:.2f} ± {patient_stats['std_studies']:.2f}")
                logger.info(f"  Studies per patient range: {patient_stats['min_studies']} - {patient_stats['max_studies']}")
                logger.info("=" * 60)

            if "summary" in results and "missing_sequences" in results["summary"]:
                missing_stats = results["summary"]["missing_sequences"]
                logger.info("Missing Sequences Summary:")
                for seq in config.expected_sequences:
                    if seq in missing_stats:
                        stats = missing_stats[seq]
                        logger.info(
                            f"  {seq}: {stats['missing_count']} missing ({stats['missing_fraction']:.1%})"
                        )
                logger.info("=" * 60)
        else:
            logger.info("=" * 60)
            logger.info("Dry-run Complete!")
            logger.info("=" * 60)
            logger.info("Dataset structure scanned successfully.")
            logger.info("No images were loaded or analyzed (dry-run mode).")
            logger.info("=" * 60)

        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Invalid value: {e}")
        return 1
    except KeyError as e:
        logger.error(f"Missing configuration key: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Unexpected error during quality analysis: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
