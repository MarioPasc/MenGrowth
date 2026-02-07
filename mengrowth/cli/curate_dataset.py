"""Command-line interface for complete dataset curation pipeline.

This script provides a single entry point for the full data curation workflow:
    1. Reorganize: Convert raw data to standardized structure
    2. Filter: Apply quality and completeness criteria
    2.5. Quality Filter: Validate data quality (SNR, contrast, geometry, etc.)
    3. Reid: Re-identify patients (P* -> MenGrowth-XXXX, after quality filtering for continuous IDs)
    4. Analyze: Compute quality metrics
    5. Visualize: Generate plots and HTML report

Output is organized into:
    {output-root}/dataset/   - patient data, metadata, id_mapping
    {output-root}/quality/   - QC outputs, metrics, reports

Clinical metadata is processed throughout the pipeline, tracking patient
inclusion/exclusion and mapping original IDs to MenGrowth IDs.

Usage:
    mengrowth-curate-dataset --config configs/raw_data.yaml --input-root /path/to/raw --output-root /path/to/output
    mengrowth-curate-dataset --config configs/raw_data.yaml --skip-reorganize  # Resume from filtering
    mengrowth-curate-dataset --config configs/raw_data.yaml --dry-run
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

from mengrowth.preprocessing.config import (
    load_preprocessing_config,
    load_quality_analysis_config,
)
from mengrowth.preprocessing.quality_analysis.analyzer import QualityAnalyzer
from mengrowth.preprocessing.quality_analysis.visualize import QualityVisualizer
from mengrowth.preprocessing.utils.filter_raw_data import (
    filter_raw_data,
    reid_patients_and_studies,
)
from mengrowth.preprocessing.utils.metadata import MetadataManager
from mengrowth.preprocessing.utils.quality_filtering import (
    export_quality_issues,
    export_quality_metrics,
    run_quality_filtering,
)
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


def _format_duration(seconds: float) -> str:
    """Format elapsed seconds into human-readable duration string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    if minutes < 60:
        return f"{minutes}m {secs:.0f}s"
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours}h {mins}m {secs:.0f}s"


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Complete dataset curation pipeline: reorganize -> filter -> analyze -> visualize",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline
  mengrowth-curate-dataset \\
    --config configs/raw_data.yaml \\
    --input-root /path/to/raw \\
    --output-root /path/to/output

  # Skip reorganization (resume from filtering)
  mengrowth-curate-dataset \\
    --config configs/raw_data.yaml \\
    --output-root /path/to/output \\
    --skip-reorganize

  # Only reorganize and filter (no analysis/visualization)
  mengrowth-curate-dataset \\
    --config configs/raw_data.yaml \\
    --input-root /path/to/raw \\
    --output-root /path/to/output \\
    --skip-analyze --skip-visualize

  # Dry run to preview all operations
  mengrowth-curate-dataset \\
    --config configs/raw_data.yaml \\
    --input-root /path/to/raw \\
    --output-root /path/to/output \\
    --dry-run

Pipeline steps:
  1. REORGANIZE: Convert raw data to standardized structure
     - Source: {input-root}/source/..., {input-root}/extension_1/...
     - Output: {output-root}/dataset/MenGrowth-2025/P{id}/{study}/modality.nrrd

  2. FILTER: Apply quality and completeness criteria
     - Removes studies missing too many sequences
     - Removes patients with insufficient longitudinal data

  2.5. QUALITY FILTER: Validate data quality (SNR, contrast, geometry)
     - Removes blocked studies/patients
     - Exports quality metrics JSON and issues CSV

  3. REID: Re-identify patients after quality filtering
     - P1 -> MenGrowth-0001 (continuous numbering, no gaps)

  4. ANALYZE: Compute quality metrics
     - Spatial properties, intensity statistics, SNR
     - Output: {output-root}/quality/qc_analysis/*.csv, *.json

  5. VISUALIZE: Generate plots and HTML report
     - QC plots + clinical metadata + quality filtering plots
     - Output: {output-root}/quality/qc_analysis/figures/, quality_analysis_report.html

Output structure:
  {output-root}/dataset/    - MenGrowth-2025/, id_mapping.json, metadata
  {output-root}/quality/    - rejected_files.csv, quality_issues.csv, quality_metrics.json, qc_analysis/
        """,
    )

    # Input/output paths
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to raw_data.yaml configuration file.",
    )

    parser.add_argument(
        "--qa-config",
        type=Path,
        default=Path("configs/templates/quality_analysis.yaml"),
        help="Path to quality_analysis.yaml (default: configs/quality_analysis.yaml).",
    )

    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("/media/mpascual/PortableSSD/Meningiomas/MenGrowth/raw/processed"),
        help="Root directory containing raw data (required unless --skip-reorganize).",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/media/mpascual/PortableSSD/Meningiomas/MenGrowth/curated"),
        help="Root directory for all outputs (reorganized data, analysis, visualizations).",
    )

    # Pipeline control
    parser.add_argument(
        "--skip-reorganize",
        action="store_true",
        help="Skip reorganization step (assume data already reorganized in output-root).",
    )

    parser.add_argument(
        "--skip-filter",
        action="store_true",
        help="Skip filtering step.",
    )

    parser.add_argument(
        "--skip-quality-filter",
        action="store_true",
        help="Skip quality filtering step.",
    )

    parser.add_argument(
        "--skip-analyze",
        action="store_true",
        help="Skip quality analysis step.",
    )

    parser.add_argument(
        "--skip-visualize",
        action="store_true",
        help="Skip visualization step.",
    )

    # Metadata control
    parser.add_argument(
        "--metadata-xlsx",
        type=Path,
        default=None,
        help="Override metadata xlsx path from config.",
    )

    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Disable metadata processing entirely.",
    )

    # Common options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate operations without making changes.",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG level) logging.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: half of CPU cores).",
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

    pipeline_start = time.monotonic()
    step_durations = {}

    try:
        # Validate arguments
        if not args.skip_reorganize and args.input_root is None:
            logger.error("--input-root is required unless --skip-reorganize is set")
            return 1

        # Resolve worker count
        n_workers = args.workers if args.workers is not None else max(1, (os.cpu_count() or 4) // 2)
        logger.info(f"Parallel workers: {n_workers} (of {os.cpu_count()} CPUs)")

        # Load configurations
        logger.info(f"Loading configuration from: {args.config}")
        preprocessing_config = load_preprocessing_config(args.config)

        # Determine QA config path and load once (reused by analyze + visualize)
        qa_config_path = args.qa_config
        if qa_config_path is None:
            repo_root = Path(__file__).parent.parent.parent
            qa_config_path = repo_root / "configs" / "quality_analysis.yaml"

        qa_config = None
        if qa_config_path.exists():
            qa_config = load_quality_analysis_config(qa_config_path)

        # Define output directory structure
        dataset_dir = args.output_root / "dataset"
        quality_dir = args.output_root / "quality"
        mengrowth_dir = dataset_dir / "MenGrowth-2025"
        qa_output_dir = quality_dir / "qc_analysis"

        # Initialize metadata manager
        metadata_manager: Optional[MetadataManager] = None
        if not args.no_metadata and preprocessing_config.metadata and preprocessing_config.metadata.enabled:
            xlsx_path = args.metadata_xlsx or (
                Path(preprocessing_config.metadata.xlsx_path)
                if preprocessing_config.metadata.xlsx_path
                else None
            )

            if xlsx_path and xlsx_path.exists():
                logger.info("=" * 60)
                logger.info("LOADING CLINICAL METADATA")
                logger.info("=" * 60)
                logger.info(f"Metadata file: {xlsx_path}")
                metadata_manager = MetadataManager()
                metadata_manager.load_from_xlsx(xlsx_path)
                logger.info(f"Loaded metadata for {len(metadata_manager.get_patient_ids())} patients")

                # Show clinical summary
                summary = metadata_manager.get_clinical_summary()
                logger.info(f"  Age range: {summary['age_stats'].get('min', 'N/A')} - {summary['age_stats'].get('max', 'N/A')}")
                logger.info(f"  Sex distribution: {summary['sex_distribution']}")
                logger.info(f"  Growth: {summary['growth_distribution']}")
            elif xlsx_path:
                logger.warning(f"Metadata xlsx file not found: {xlsx_path}")
            else:
                logger.info("Metadata processing disabled (no xlsx_path configured)")

        # ====================================================================
        # STEP 1: REORGANIZE
        # ====================================================================
        if not args.skip_reorganize:
            logger.info("")
            logger.info("=" * 60)
            logger.info("STEP 1: REORGANIZE")
            logger.info("=" * 60)
            logger.info(f"Input root: {args.input_root}")
            logger.info(f"Output: {dataset_dir}")
            logger.info(f"Dry run: {args.dry_run}")

            if not args.input_root.exists():
                logger.error(f"Input root does not exist: {args.input_root}")
                return 1

            step_start = time.monotonic()
            reorg_stats = reorganize_raw_data(
                input_root=args.input_root,
                output_root=dataset_dir,
                config=preprocessing_config.raw_data,
                dry_run=args.dry_run,
                n_workers=n_workers,
            )
            step_durations["reorganize"] = time.monotonic() - step_start

            logger.info(f"Files copied: {reorg_stats['copied']}")
            logger.info(f"Files skipped: {reorg_stats['skipped']}")
            logger.info(f"Errors: {reorg_stats['errors']}")
            logger.info(f"Step duration: {_format_duration(step_durations['reorganize'])}")

        # ====================================================================
        # STEP 2: FILTER (reid deferred to Step 3)
        # ====================================================================
        if not args.skip_filter:
            logger.info("")
            logger.info("=" * 60)
            logger.info("STEP 2: FILTER")
            logger.info("=" * 60)

            if preprocessing_config.filtering is None:
                logger.warning("Filtering configuration not found, skipping filter step")
            else:
                if not mengrowth_dir.exists():
                    logger.error(
                        f"MenGrowth-2025 directory not found in {dataset_dir}. "
                        "Run without --skip-reorganize first."
                    )
                    return 1

                logger.info(f"Required sequences: {preprocessing_config.filtering.sequences}")
                logger.info(f"Min studies per patient: {preprocessing_config.filtering.min_studies_per_patient}")
                logger.info(f"Re-identify patients: {preprocessing_config.filtering.reid_patients} (deferred to after quality filtering)")

                step_start = time.monotonic()
                filter_stats = filter_raw_data(
                    data_root=dataset_dir,
                    config=preprocessing_config,
                    metadata_manager=metadata_manager,
                    dry_run=args.dry_run,
                    skip_reid=True,
                    rejected_csv_path=quality_dir / "rejected_files.csv",
                )
                step_durations["filter"] = time.monotonic() - step_start

                logger.info(f"Patients kept: {filter_stats['patients_kept']}")
                logger.info(f"Patients removed: {filter_stats['patients_removed']}")
                logger.info(f"Step duration: {_format_duration(step_durations['filter'])}")

        # ====================================================================
        # STEP 2.5: QUALITY FILTERING
        # ====================================================================
        qf_reports = None
        if not args.skip_quality_filter:
            logger.info("")
            logger.info("=" * 60)
            logger.info("STEP 2.5: QUALITY FILTERING")
            logger.info("=" * 60)

            if preprocessing_config.quality_filtering is None:
                logger.info("Quality filtering not configured, using defaults")
                from mengrowth.preprocessing.config import QualityFilteringConfig
                qf_config = QualityFilteringConfig()
            else:
                qf_config = preprocessing_config.quality_filtering

            if not qf_config.enabled:
                logger.info("Quality filtering is disabled in configuration")
            else:
                if not mengrowth_dir.exists():
                    logger.warning(
                        f"MenGrowth-2025 directory not found in {dataset_dir}. "
                        "Skipping quality filtering."
                    )
                else:
                    logger.info(f"Input: {mengrowth_dir}")
                    logger.info(f"Remove blocked: {qf_config.remove_blocked}")
                    logger.info(f"Min studies per patient: {qf_config.min_studies_per_patient}")
                    logger.info(f"Running quality validation checks...")

                    step_start = time.monotonic()
                    qf_stats, qf_reports = run_quality_filtering(
                        data_root=mengrowth_dir,
                        config=qf_config,
                        metadata_manager=metadata_manager,
                        dry_run=args.dry_run,
                        quality_output_dir=quality_dir,
                        n_workers=n_workers,
                    )
                    step_durations["quality_filter"] = time.monotonic() - step_start

                    # Export quality issues and metrics
                    export_quality_issues(qf_reports, quality_dir / "quality_issues.csv")
                    export_quality_metrics(qf_reports, quality_dir / "quality_metrics.json")

                    # Log summary
                    logger.info(f"Quality filtering complete:")
                    logger.info(f"  Files: {qf_stats.files_passed} passed, {qf_stats.files_warned} warned, {qf_stats.files_blocked} blocked")
                    logger.info(f"  Studies: {qf_stats.studies_passed} passed, {qf_stats.studies_blocked} blocked")
                    logger.info(f"  Patients: {qf_stats.patients_passed} passed, {qf_stats.patients_blocked} blocked")

                    if qf_stats.issues_by_type:
                        logger.info(f"  Issues by type: {qf_stats.issues_by_type}")

                    if qf_stats.studies_removed or qf_stats.patients_removed:
                        logger.info(f"  Studies removed: {qf_stats.studies_removed}")
                        logger.info(f"  Patients removed: {qf_stats.patients_removed}")

                    logger.info(f"Step duration: {_format_duration(step_durations['quality_filter'])}")

        # ====================================================================
        # ENSURE ALL DATASET PATIENTS HAVE METADATA
        # ====================================================================
        if metadata_manager and mengrowth_dir.exists():
            for patient_dir in mengrowth_dir.iterdir():
                if patient_dir.is_dir():
                    metadata_manager.ensure_patient_exists(patient_dir.name)

        # ====================================================================
        # STEP 3: REID (after quality filtering for continuous IDs)
        # ====================================================================
        if preprocessing_config.filtering and preprocessing_config.filtering.reid_patients:
            logger.info("")
            logger.info("=" * 60)
            logger.info("STEP 3: REID")
            logger.info("=" * 60)

            if not mengrowth_dir.exists():
                logger.warning(
                    f"MenGrowth-2025 directory not found in {dataset_dir}. "
                    "Skipping re-identification."
                )
            else:
                logger.info(f"Input: {mengrowth_dir}")

                step_start = time.monotonic()
                id_mapping = reid_patients_and_studies(mengrowth_dir, args.dry_run)
                step_durations["reid"] = time.monotonic() - step_start

                # Apply ID mapping to metadata
                if metadata_manager and id_mapping:
                    metadata_mapping = {}
                    for original_id, mapping_info in id_mapping.items():
                        new_id = mapping_info["new_id"]
                        metadata_mapping[new_id] = {
                            "original_id": original_id,
                            "studies": mapping_info.get("studies", {}),
                        }
                    metadata_manager.apply_id_mapping(metadata_mapping)
                    logger.info(f"Applied ID mapping to metadata for {len(metadata_mapping)} patients")

                logger.info(f"Re-identified {len(id_mapping)} patients")
                logger.info(f"Step duration: {_format_duration(step_durations['reid'])}")

        # ====================================================================
        # EXPORT METADATA
        # ====================================================================
        if metadata_manager and not args.dry_run:
            logger.info("")
            logger.info("=" * 60)
            logger.info("EXPORTING METADATA")
            logger.info("=" * 60)

            output_csv = dataset_dir / preprocessing_config.metadata.output_csv_name
            output_json = dataset_dir / preprocessing_config.metadata.output_json_name

            metadata_manager.export_enriched_csv(output_csv)
            metadata_manager.export_json(output_json)

            # Show final summary
            summary = metadata_manager.get_clinical_summary()
            logger.info(f"Total patients in metadata: {summary['total_patients']}")
            logger.info(f"Patients included: {summary['included_patients']}")
            logger.info(f"Patients excluded: {summary['excluded_patients']}")
            if summary['exclusion_reasons']:
                logger.info(f"Exclusion reasons: {summary['exclusion_reasons']}")

        # ====================================================================
        # STEP 4: ANALYZE
        # ====================================================================
        if not args.skip_analyze:
            logger.info("")
            logger.info("=" * 60)
            logger.info("STEP 4: ANALYZE")
            logger.info("=" * 60)

            if qa_config is None:
                logger.warning(f"QA config not found at {qa_config_path}, skipping analysis")
            else:
                if not mengrowth_dir.exists():
                    logger.warning(f"Data directory not found: {mengrowth_dir}, skipping analysis")
                else:
                    logger.info(f"Input: {mengrowth_dir}")
                    logger.info(f"Output: {qa_output_dir}")

                    if not args.dry_run:
                        step_start = time.monotonic()
                        # Override config paths for this run
                        qa_config.input_dir = mengrowth_dir
                        qa_config.output_dir = qa_output_dir
                        analyzer = QualityAnalyzer(config=qa_config)
                        analyzer.run_analysis()
                        analyzer.save_results()
                        step_durations["analyze"] = time.monotonic() - step_start
                        logger.info(f"Quality analysis complete ({_format_duration(step_durations['analyze'])})")
                    else:
                        logger.info("Dry run - skipping actual analysis")

        # ====================================================================
        # STEP 5: VISUALIZE
        # ====================================================================
        if not args.skip_visualize:
            logger.info("")
            logger.info("=" * 60)
            logger.info("STEP 5: VISUALIZE")
            logger.info("=" * 60)

            if not qa_output_dir.exists():
                logger.warning(f"Analysis results not found: {qa_output_dir}, skipping visualization")
            elif qa_config is None:
                logger.warning(f"QA config not found at {qa_config_path}, skipping visualization")
            else:
                # Override output_dir to match the analysis output location
                qa_config.output_dir = qa_output_dir

                logger.info(f"Input: {qa_output_dir}")
                logger.info(f"Output: {qa_output_dir}")

                if not args.dry_run:
                    step_start = time.monotonic()
                    visualizer = QualityVisualizer(
                        config=qa_config,
                        results_dir=qa_output_dir,
                    )

                    # Pass quality metrics path for enhanced report
                    quality_metrics_path = quality_dir / "quality_metrics.json"
                    visualizer.run_visualization(
                        metadata_manager=metadata_manager,
                        quality_metrics_path=quality_metrics_path if quality_metrics_path.exists() else None,
                    )
                    step_durations["visualize"] = time.monotonic() - step_start
                    logger.info(f"Visualization complete ({_format_duration(step_durations['visualize'])})")
                else:
                    logger.info("Dry run - skipping actual visualization")

        # ====================================================================
        # SUMMARY
        # ====================================================================
        pipeline_duration = time.monotonic() - pipeline_start
        logger.info("")
        logger.info("=" * 60)
        logger.info("CURATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Output directory: {args.output_root}")
        logger.info(f"Total pipeline duration: {_format_duration(pipeline_duration)}")
        if step_durations:
            logger.info("Step durations:")
            for step_name, duration in step_durations.items():
                logger.info(f"  {step_name}: {_format_duration(duration)}")
        logger.info("")
        logger.info("Generated files:")
        logger.info(f"  - {dataset_dir}/MenGrowth-2025/  (reorganized data)")
        logger.info(f"  - {quality_dir}/rejected_files.csv")
        if preprocessing_config.filtering and preprocessing_config.filtering.reid_patients:
            logger.info(f"  - {dataset_dir}/id_mapping.json")
        if not args.skip_quality_filter:
            logger.info(f"  - {quality_dir}/quality_issues.csv  (quality validation)")
            logger.info(f"  - {quality_dir}/quality_metrics.json  (all computed metrics)")
        if metadata_manager:
            logger.info(f"  - {dataset_dir}/{preprocessing_config.metadata.output_csv_name}")
            logger.info(f"  - {dataset_dir}/{preprocessing_config.metadata.output_json_name}")
        if not args.skip_analyze:
            logger.info(f"  - {qa_output_dir}/  (analysis results)")
        if not args.skip_visualize:
            logger.info(f"  - {qa_output_dir}/figures/  (plots)")
            logger.info(f"  - {qa_output_dir}/quality_analysis_report.html")

        if args.dry_run:
            logger.info("")
            logger.info("This was a DRY RUN - no actual changes were made.")
            logger.info("Remove --dry-run flag to perform actual curation.")

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
        logger.exception(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
