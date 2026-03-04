#!/usr/bin/env python3
"""Comprehensive post-processing dataset analysis for MenGrowth.

Runs four phases:
  1. Dataset discovery  — walk preprocessed data, compute metrics
  2. Quantitative figures — 11 publication-quality PDF plots
  3. Octant grids         — PyVista 3-D renders (optional, slow)
  4. Export               — JSON metrics + LaTeX tables

Usage:
    # Quick run (no octant grids, ~2 min):
    python scripts/analyze_dataset.py \
        --dataset-root /path/to/preprocessed/MenGrowth-2025 \
        --output-dir /path/to/analysis \
        --metadata-root /path/to/curated/dataset \
        --skip-octant

    # Full run (with octant grids, ~10 min):
    python scripts/analyze_dataset.py \
        --dataset-root /path/to/preprocessed/MenGrowth-2025 \
        --output-dir /path/to/analysis \
        --metadata-root /path/to/curated/dataset
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="MenGrowth post-processing dataset analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/media/mpascual/PortableSSD/Meningiomas/MenGrowth/preprocessed/MenGrowth-2025"),
        help="Root of preprocessed dataset (contains MenGrowth-XXXX/ dirs)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/media/mpascual/PortableSSD/Meningiomas/MenGrowth/analysis"),
        help="Analysis output directory",
    )
    parser.add_argument(
        "--metadata-root",
        type=Path,
        default=Path("/media/mpascual/PortableSSD/Meningiomas/MenGrowth/curated/dataset"),
        help="Path to curated metadata (id_mapping.json, metadata_clean.json)",
    )
    parser.add_argument(
        "--skip-octant",
        action="store_true",
        help="Skip Phase 3 (PyVista octant grids) for faster execution",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable DEBUG logging",
    )
    return parser.parse_args()


def main() -> None:
    """Run the full analysis pipeline."""
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    from mengrowth.analysis.src.discovery import discover_dataset
    from mengrowth.analysis.src.export import export_all
    from mengrowth.analysis.src.figures import generate_all_figures
    from mengrowth.analysis.src.octant_grids import generate_octant_grids

    t0 = time.time()

    # Phase 1: Dataset discovery
    logger.info("=" * 60)
    logger.info("Phase 1: Dataset discovery")
    logger.info("=" * 60)
    metrics = discover_dataset(
        args.dataset_root,
        metadata_root=args.metadata_root,
        compute_mean_brain=True,
    )

    # Phase 2: Quantitative figures
    logger.info("=" * 60)
    logger.info("Phase 2: Generating figures")
    logger.info("=" * 60)
    generate_all_figures(metrics, args.output_dir / "figures")

    # Phase 3: Octant grids (optional)
    if not args.skip_octant:
        logger.info("=" * 60)
        logger.info("Phase 3: Generating octant grids")
        logger.info("=" * 60)
        generate_octant_grids(
            args.dataset_root, args.output_dir / "octant_grids", metrics
        )
    else:
        logger.info("Phase 3: Skipped (--skip-octant)")

    # Phase 4: Export
    logger.info("=" * 60)
    logger.info("Phase 4: Exporting results")
    logger.info("=" * 60)
    export_all(metrics, args.output_dir)

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("Done in %.1f s. Outputs: %s", elapsed, args.output_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
