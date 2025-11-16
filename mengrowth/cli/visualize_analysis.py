"""Command-line interface for visualizing quality analysis results.

This module provides a CLI for generating plots and HTML reports from
previously computed quality analysis results.
"""

import argparse
import logging
import sys
from pathlib import Path

from mengrowth.preprocessing.config import load_quality_analysis_config
from mengrowth.preprocessing.quality_analysis.visualize import QualityVisualizer


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
        description="Generate visualizations from quality analysis results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate visualizations from analysis results
  mengrowth-visualize-analysis \\
    --input /path/to/results \\
    --output /path/to/visualizations

  # Use custom configuration file
  mengrowth-visualize-analysis \\
    --input /path/to/results \\
    --output /path/to/visualizations \\
    --config configs/custom_quality_analysis.yaml

  # Enable verbose logging
  mengrowth-visualize-analysis \\
    --input /path/to/results \\
    --output /path/to/visualizations \\
    --verbose

Expected input structure:
  input_dir/
    per_study_metrics.csv
    per_patient_summary.csv
    summary_statistics.json
    per_sequence_statistics.json

Output structure:
  output_dir/
    figures/
      studies_per_patient.png
      missing_sequences.png
      spacing_distributions.png
      intensity_distributions.png
      dimension_consistency.png
      snr_distribution.png
    quality_analysis_report.html
        """,
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to directory containing analysis results (CSV and JSON files)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="Path to output directory for visualizations (default: same as input)",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/quality_analysis.yaml"),
        help="Path to quality analysis configuration file (default: configs/quality_analysis.yaml)",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG level) logging",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point for visualization CLI.

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

        # Override output directory if provided
        if args.output:
            config.output_dir = args.output
            logger.info(f"Using output directory from CLI: {args.output}")
        else:
            # Use input directory as output directory by default
            config.output_dir = args.input
            logger.info(f"Using input directory as output: {args.input}")

        if args.verbose:
            config.verbose = True

        # Log configuration summary
        logger.info("=" * 60)
        logger.info("Visualization Configuration:")
        logger.info(f"  Input directory:  {args.input}")
        logger.info(f"  Output directory: {config.output_dir}")
        logger.info(f"  Figure format:    {config.visualization.figure.format}")
        logger.info(f"  Figure DPI:       {config.visualization.figure.dpi}")
        logger.info(f"  HTML report:      {config.visualization.html_report.enabled}")
        logger.info("=" * 60)

        # Validate input directory
        if not args.input.exists():
            logger.error(f"Input directory does not exist: {args.input}")
            return 1

        if not args.input.is_dir():
            logger.error(f"Input path is not a directory: {args.input}")
            return 1

        # Check for required input files
        required_files = []
        optional_files = [
            "per_study_metrics.csv",
            "per_patient_summary.csv",
            "summary_statistics.json",
            "per_sequence_statistics.json",
        ]

        missing_files = []
        for filename in required_files:
            if not (args.input / filename).exists():
                missing_files.append(filename)

        if missing_files:
            logger.error(f"Missing required input files: {', '.join(missing_files)}")
            return 1

        # Warn about missing optional files
        for filename in optional_files:
            if not (args.input / filename).exists():
                logger.warning(f"Optional input file not found: {filename}")

        # Create visualizer
        logger.info("Initializing visualizer...")
        visualizer = QualityVisualizer(config, results_dir=args.input)

        # Run visualization
        logger.info("Generating visualizations...")
        output_paths = visualizer.run_visualization()

        # Report results
        logger.info("=" * 60)
        logger.info("Visualization Complete!")
        logger.info("=" * 60)

        if "plots" in output_paths and output_paths["plots"]:
            logger.info("Generated plots:")
            for plot_name, plot_path in output_paths["plots"].items():
                logger.info(f"  {plot_name}: {plot_path}")

        if "html_report" in output_paths:
            logger.info("=" * 60)
            logger.info(f"HTML report: {output_paths['html_report']}")
            logger.info("=" * 60)
            logger.info("Open the HTML report in your browser to view all visualizations.")

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
        logger.exception(f"Unexpected error during visualization: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
