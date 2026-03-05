"""CLI for generating graphical abstract figures from HDF5 detailed patient archives.

Usage:
    mengrowth-graphical-abstract --config configs/templates/graphical_abstract.yaml
    mengrowth-graphical-abstract --config configs/templates/graphical_abstract.yaml --verbose
"""

import argparse
import logging
import sys
from pathlib import Path

from mengrowth.analysis.graphical_abstract_figures import (
    GraphicalAbstractGenerator,
    load_graphical_abstract_config,
)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Generate graphical abstract figures from HDF5 detailed patient archives.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to graphical abstract YAML configuration file.",
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

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    try:
        config = load_graphical_abstract_config(args.config)
        generator = GraphicalAbstractGenerator(config)
        paths = generator.run()

        logger.info("=" * 60)
        logger.info("GRAPHICAL ABSTRACT GENERATION COMPLETE")
        logger.info("Generated %d figures:", len(paths))
        for p in sorted(paths):
            logger.info("  %s", p.name)
        logger.info("=" * 60)
        return 0

    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        return 1
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 130
    except Exception as e:
        logger.exception("Unexpected error: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
