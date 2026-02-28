"""Command-line interface for BraTS meningioma segmentation.

Provides subcommands for the three-stage segmentation workflow:
  prepare     — discover studies, validate, create BraTS-format input
  postprocess — remap BraTS outputs back to study directories
  cleanup     — remove temporary files from input directory
  run         — all-in-one (prepare + singularity + postprocess) for local testing

Usage:
    mengrowth-segment prepare --config configs/picasso/segmentation.yaml
    mengrowth-segment postprocess --config configs/picasso/segmentation.yaml --work-dir /path/to/work
    mengrowth-segment cleanup --config configs/picasso/segmentation.yaml
    mengrowth-segment run --config configs/picasso/segmentation.yaml
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from mengrowth.segmentation.config import load_segmentation_config
from mengrowth.segmentation.prepare import (
    discover_studies,
    prepare_brats_input,
    validate_study,
)
from mengrowth.segmentation.postprocess import (
    cleanup_temp_files,
    postprocess_outputs,
)


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


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments common to all subcommands."""
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to segmentation configuration YAML file.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging (DEBUG level).",
    )


def cmd_prepare(args: argparse.Namespace) -> int:
    """Execute the prepare subcommand.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Exit code.
    """
    logger = logging.getLogger(__name__)
    config = load_segmentation_config(args.config)

    logger.info("=" * 60)
    logger.info("SEGMENTATION PREPARE")
    logger.info("=" * 60)
    logger.info(f"Input root: {config.input_root}")
    logger.info(f"Modalities: {config.modalities}")
    logger.info(f"Expected shape: {config.expected_shape}")
    if args.patient:
        logger.info(f"Patient filter: {args.patient}")

    # Discover studies
    studies = discover_studies(config.input_root, config, patient_filter=args.patient)

    if not studies:
        logger.error("No studies found.")
        return 1

    # Validate and filter
    valid_studies = []
    for study in studies:
        is_valid, issues = validate_study(study, config)
        if is_valid:
            valid_studies.append(study)
        elif not study.is_complete and config.skip_incomplete:
            logger.debug(f"  Skipping {study.study_id}: {', '.join(issues)}")
        else:
            logger.warning(f"  Issues with {study.study_id}: {', '.join(issues)}")
            # Include studies with shape issues (they'll be corrected)
            if study.is_complete:
                valid_studies.append(study)

    logger.info(f"Validated: {len(valid_studies)}/{len(studies)} studies ready")

    if not valid_studies:
        logger.error("No valid studies to process.")
        return 1

    # Create BraTS input
    work_dir, name_map = prepare_brats_input(valid_studies, config)

    logger.info("=" * 60)
    logger.info("PREPARE COMPLETE")
    logger.info("=" * 60)
    return 0


def cmd_postprocess(args: argparse.Namespace) -> int:
    """Execute the postprocess subcommand.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Exit code.
    """
    logger = logging.getLogger(__name__)
    config = load_segmentation_config(args.config)

    logger.info("=" * 60)
    logger.info("SEGMENTATION POST-PROCESS")
    logger.info("=" * 60)
    logger.info(f"Work dir: {args.work_dir}")
    logger.info(f"Output filename: {config.output_filename}")

    results = postprocess_outputs(args.work_dir, config)

    failed = [r for r in results if not r.success]
    if failed:
        logger.warning(f"{len(failed)} segmentation(s) not found:")
        for r in failed:
            logger.warning(f"  {r.study_id}: {r.error}")

    logger.info("=" * 60)
    logger.info("POST-PROCESS COMPLETE")
    logger.info("=" * 60)
    return 0


def cmd_cleanup(args: argparse.Namespace) -> int:
    """Execute the cleanup subcommand.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Exit code.
    """
    logger = logging.getLogger(__name__)
    config = load_segmentation_config(args.config)

    logger.info("=" * 60)
    logger.info("SEGMENTATION CLEANUP")
    logger.info("=" * 60)

    count = cleanup_temp_files(config.input_root)
    logger.info(f"Removed {count} temporary file(s)")

    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Execute the full run subcommand (prepare + singularity + postprocess).

    Intended for local testing. On the cluster, use the SLURM scripts instead.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Exit code.
    """
    logger = logging.getLogger(__name__)
    config = load_segmentation_config(args.config)

    # Step 1: Prepare
    logger.info("=" * 60)
    logger.info("STEP 1: PREPARE")
    logger.info("=" * 60)

    studies = discover_studies(config.input_root, config, patient_filter=args.patient)
    valid_studies = []
    for study in studies:
        is_valid, issues = validate_study(study, config)
        if is_valid:
            valid_studies.append(study)
        elif study.is_complete:
            valid_studies.append(study)

    if not valid_studies:
        logger.error("No valid studies found.")
        return 1

    work_dir, name_map = prepare_brats_input(valid_studies, config)

    # Step 2: Run Singularity
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 2: SINGULARITY INFERENCE")
    logger.info("=" * 60)

    sif_path = config.sif_path
    if not Path(sif_path).exists():
        logger.error(f"SIF image not found: {sif_path}")
        return 1

    brats_input = work_dir / "input"
    brats_output = work_dir / "output"

    cmd = [
        "singularity",
        "run",
        "--nv",
        "--cleanenv",
        "--no-home",
        "--writable-tmpfs",
        "--pwd", "/app",
        "--bind",
        f"{brats_input}:/input:rw",
        "--bind",
        f"{brats_output}:/output:rw",
        str(sif_path),
    ]
    logger.info(f"Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        logger.error(f"Singularity inference failed with exit code {result.returncode}")
        return result.returncode

    # Step 3: Post-process
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 3: POST-PROCESS")
    logger.info("=" * 60)

    results = postprocess_outputs(work_dir, config)

    failed = [r for r in results if not r.success]
    if failed:
        logger.warning(f"{len(failed)} segmentation(s) not found")

    logger.info("")
    logger.info("=" * 60)
    logger.info("SEGMENTATION RUN COMPLETE")
    logger.info("=" * 60)
    return 0


def cmd_attach_to_archive(args: argparse.Namespace) -> int:
    """Attach segmentation to an existing HDF5 archive.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Exit code.
    """
    logger = logging.getLogger(__name__)

    archive_path = Path(args.archive)
    seg_path = Path(args.seg)

    if not archive_path.exists():
        logger.error(f"Archive not found: {archive_path}")
        return 1
    if not seg_path.exists():
        logger.error(f"Segmentation not found: {seg_path}")
        return 1

    from mengrowth.preprocessing.src.archiver import DetailedPatientArchiver

    label_map = {1: "necrotic_core", 2: "peritumoral_edema", 3: "enhancing_tumor"}
    DetailedPatientArchiver.attach_segmentation(archive_path, seg_path, label_map)
    logger.info(f"Attached {seg_path.name} to {archive_path}")
    return 0


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="BraTS 2025 Meningioma Segmentation for MenGrowth.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Subcommands:
  prepare      Discover studies, validate, create BraTS-format input directory
  postprocess  Remap BraTS outputs back to study directories
  cleanup      Remove temporary files from input directory
  run          All-in-one: prepare + singularity + postprocess (local testing)

Examples:
  # Prepare input for SLURM job
  mengrowth-segment prepare --config configs/picasso/segmentation.yaml

  # Prepare single patient
  mengrowth-segment prepare --config configs/picasso/segmentation.yaml --patient MenGrowth-0015

  # Post-process after SLURM job
  mengrowth-segment postprocess --config configs/picasso/segmentation.yaml --work-dir /path/to/work

  # Local test run (requires SIF + GPU)
  mengrowth-segment run --config configs/picasso/segmentation.yaml
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Subcommand to run")

    # prepare
    p_prepare = subparsers.add_parser("prepare", help="Create BraTS-format input")
    _add_common_args(p_prepare)
    p_prepare.add_argument(
        "--patient",
        type=str,
        default=None,
        help="Process only this patient ID (e.g., MenGrowth-0015).",
    )

    # postprocess
    p_post = subparsers.add_parser("postprocess", help="Remap outputs to study dirs")
    _add_common_args(p_post)
    p_post.add_argument(
        "--work-dir",
        type=Path,
        required=True,
        help="Work directory from the prepare step.",
    )

    # cleanup
    p_cleanup = subparsers.add_parser("cleanup", help="Remove temp files")
    _add_common_args(p_cleanup)

    # run
    p_run = subparsers.add_parser("run", help="Full pipeline (local testing)")
    _add_common_args(p_run)
    p_run.add_argument(
        "--patient",
        type=str,
        default=None,
        help="Process only this patient ID.",
    )

    # attach-to-archive
    p_attach = subparsers.add_parser(
        "attach-to-archive", help="Attach segmentation to HDF5 archive"
    )
    _add_common_args(p_attach)
    p_attach.add_argument(
        "--archive",
        type=Path,
        required=True,
        help="Path to archive.h5 file.",
    )
    p_attach.add_argument(
        "--seg",
        type=Path,
        required=True,
        help="Path to seg.nii.gz file.",
    )

    return parser.parse_args()


def main() -> int:
    """Main CLI entry point.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    args = parse_arguments()

    if not args.command:
        print("Error: No subcommand specified. Use --help for usage.", file=sys.stderr)
        return 1

    setup_logging(verbose=args.verbose)

    try:
        if args.command == "prepare":
            return cmd_prepare(args)
        elif args.command == "postprocess":
            return cmd_postprocess(args)
        elif args.command == "cleanup":
            return cmd_cleanup(args)
        elif args.command == "run":
            return cmd_run(args)
        elif args.command == "attach-to-archive":
            return cmd_attach_to_archive(args)
        else:
            print(f"Unknown command: {args.command}", file=sys.stderr)
            return 1

    except FileNotFoundError as e:
        logging.getLogger(__name__).error(f"File not found: {e}")
        return 1
    except KeyboardInterrupt:
        logging.getLogger(__name__).warning("Interrupted by user")
        return 130
    except Exception as e:
        logging.getLogger(__name__).exception(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
