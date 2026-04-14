"""Command-line interface for the MenGrowth SynthSeg parcellation pipeline.

Subcommands
-----------
``discover``
    Enumerate patient IDs (one per line) under ``input_root``; used by the
    SLURM launcher to size the array job.
``run-patient``
    Process every study of a given patient; used by the SLURM worker.
``run``
    Process every patient serially; convenience for local testing and
    small-scale runs.
``verify``
    Audit produced outputs against expected inputs and print a summary
    report; non-zero exit iff any mismatch is detected.

All subcommands share ``--config <path> --verbose``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

from mengrowth.synthseg.config import SynthSegConfig, load_synthseg_config
from mengrowth.synthseg.discovery import (
    discover_patients,
    discover_studies,
    summarize_discovery,
)
from mengrowth.synthseg.primitives import parse_qc_value, parse_volumes_row
from mengrowth.synthseg.runner import PatientResult, run_patient


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to SynthSeg configuration YAML.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def cmd_discover(args: argparse.Namespace) -> int:
    """Write the list of discovered patient IDs to ``--output`` (or stdout).

    Used by the SLURM launcher: the output is one patient ID per line, which
    becomes the index source for the array job.
    """
    logger = logging.getLogger(__name__)
    config = load_synthseg_config(args.config)

    patients = discover_patients(config.input_root)
    if not patients:
        logger.error("No patients found under %s", config.input_root)
        return 1

    lines = "\n".join(patients) + "\n"

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(lines, encoding="utf-8")
        logger.info("Wrote %d patient IDs to %s", len(patients), out_path)
    else:
        sys.stdout.write(lines)

    logger.info("Total patients: %d", len(patients))
    return 0


def cmd_run_patient(args: argparse.Namespace) -> int:
    """Run SynthSeg on every study of one patient.

    Exit code contract is defined by :meth:`PatientResult.exit_code`.
    """
    logger = logging.getLogger(__name__)
    config = load_synthseg_config(args.config)

    logger.info("=" * 60)
    logger.info("SYNTHSEG RUN-PATIENT")
    logger.info("=" * 60)
    logger.info("Config:       %s", args.config)
    logger.info("Patient:      %s", args.patient)
    logger.info("Input root:   %s", config.input_root)
    logger.info("Output root:  %s", config.output_root)
    logger.info("SynthSeg:     %s", config.synthseg_repo)
    logger.info("Modality:     %s", config.input_modality)
    logger.info("Robust:       %s", config.robust)
    logger.info("Use GPU:      %s", config.use_gpu)

    result = run_patient(config, args.patient)
    return result.exit_code()


def cmd_run(args: argparse.Namespace) -> int:
    """Run SynthSeg on every patient sequentially.

    Intended for local / small-scale testing. On the cluster, use the SLURM
    launcher which parallelizes across patients.
    """
    logger = logging.getLogger(__name__)
    config = load_synthseg_config(args.config)

    patients: List[str]
    if args.patient:
        patients = [args.patient]
    else:
        patients = discover_patients(config.input_root)

    logger.info("Running SynthSeg on %d patient(s) serially", len(patients))

    agg_success = 0
    agg_failed = 0
    agg_skipped = 0
    any_failure = False

    for pid in patients:
        pr = run_patient(config, pid)
        agg_success += pr.n_success
        agg_failed += pr.n_failed
        agg_skipped += pr.n_skipped_missing
        if pr.exit_code() != 0:
            any_failure = True

    logger.info("=" * 60)
    logger.info("COHORT SUMMARY")
    logger.info("=" * 60)
    logger.info("Patients:           %d", len(patients))
    logger.info("Studies succeeded:  %d", agg_success)
    logger.info("Studies failed:     %d", agg_failed)
    logger.info("Studies skipped:    %d", agg_skipped)

    return 2 if any_failure else 0


def _verify_one_study(
    config: SynthSegConfig,
    study_dir: Path,
    patient_id: str,
    study_id: str,
) -> dict:
    """Return a status dict for one study's outputs."""
    parc = study_dir / config.parcellation_filename
    vol = study_dir / config.volumes_filename
    qc = study_dir / config.qc_filename

    row = {
        "patient_id": patient_id,
        "study_id": study_id,
        "parc_exists": parc.exists(),
        "vol_exists": vol.exists(),
        "qc_exists": qc.exists(),
        "qc_value": None,
        "n_regions": None,
        "issues": [],
    }

    if parc.exists() and vol.exists() and qc.exists():
        try:
            row["qc_value"] = parse_qc_value(qc)
            if not (0.0 <= row["qc_value"] <= 1.0):
                row["issues"].append(
                    f"qc_value {row['qc_value']} outside [0,1]"
                )
        except Exception as e:  # noqa: BLE001
            row["issues"].append(f"qc parse error: {e}")

        try:
            volumes = parse_volumes_row(vol)
            row["n_regions"] = sum(1 for v in volumes.values() if v > 0)
            if row["n_regions"] < 25:
                row["issues"].append(
                    f"only {row['n_regions']} non-zero regions (<25)"
                )
        except Exception as e:  # noqa: BLE001
            row["issues"].append(f"vol parse error: {e}")
    else:
        if not parc.exists():
            row["issues"].append("missing parcellation")
        if not vol.exists():
            row["issues"].append("missing volumes CSV")
        if not qc.exists():
            row["issues"].append("missing qc CSV")

    return row


def cmd_verify(args: argparse.Namespace) -> int:
    """Audit produced outputs; non-zero exit if anything's missing or invalid."""
    logger = logging.getLogger(__name__)
    config = load_synthseg_config(args.config)

    studies = discover_studies(config.input_root, config)
    total, with_input = summarize_discovery(studies)
    logger.info("Expected %d studies (%d with %s)",
                total, with_input, config.input_modality)

    rows = []
    for s in studies:
        if not s.has_input_modality:
            continue  # nothing expected for these
        out_dir = Path(config.output_root) / s.patient_id / s.study_id
        rows.append(
            _verify_one_study(config, out_dir, s.patient_id, s.study_id)
        )

    produced = sum(1 for r in rows if not r["issues"])
    with_issues = [r for r in rows if r["issues"]]
    qc_values = [r["qc_value"] for r in rows if r["qc_value"] is not None]

    logger.info("=" * 60)
    logger.info("VERIFY SUMMARY")
    logger.info("=" * 60)
    logger.info("Expected studies (with T1n): %d", with_input)
    logger.info("Produced (no issues):        %d", produced)
    logger.info("With issues:                 %d", len(with_issues))

    if qc_values:
        qc_arr = np.asarray(qc_values)
        logger.info(
            "QC scores: min=%.3f  median=%.3f  mean=%.3f  max=%.3f  (n=%d)",
            float(qc_arr.min()),
            float(np.median(qc_arr)),
            float(qc_arr.mean()),
            float(qc_arr.max()),
            len(qc_arr),
        )
        low = [r for r in rows if r["qc_value"] is not None and r["qc_value"] < 0.3]
        if low:
            logger.warning("QC < 0.3 (review manually):")
            for r in low:
                logger.warning(
                    "  %s/%s  qc=%.3f", r["patient_id"], r["study_id"], r["qc_value"]
                )

    if with_issues:
        logger.warning("Studies with issues:")
        for r in with_issues:
            logger.warning(
                "  %s/%s  %s",
                r["patient_id"],
                r["study_id"],
                "; ".join(r["issues"]),
            )
        return 2

    logger.info("All %d expected studies produced valid outputs.", produced)
    return 0


# ---------------------------------------------------------------------------
# argparse wiring
# ---------------------------------------------------------------------------


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MenGrowth SynthSeg parcellation pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Subcommands:\n"
            "  discover      Write patient IDs (for SLURM launcher)\n"
            "  run-patient   Process all studies of one patient (for SLURM worker)\n"
            "  run           Process every patient serially (local testing)\n"
            "  verify        Audit produced outputs\n"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", help="Subcommand")

    p_disc = subparsers.add_parser("discover", help="List patient IDs")
    _add_common_args(p_disc)
    p_disc.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write patient list to this file (default: stdout).",
    )

    p_rp = subparsers.add_parser("run-patient", help="Process all studies of one patient")
    _add_common_args(p_rp)
    p_rp.add_argument(
        "--patient",
        type=str,
        required=True,
        help="Patient ID, e.g. MenGrowth-0017.",
    )

    p_run = subparsers.add_parser("run", help="Process every patient serially")
    _add_common_args(p_run)
    p_run.add_argument(
        "--patient",
        type=str,
        default=None,
        help="Process only this patient ID (optional; default: all).",
    )

    p_ver = subparsers.add_parser("verify", help="Audit produced outputs")
    _add_common_args(p_ver)

    return parser.parse_args()


def main() -> int:
    args = parse_arguments()

    if not args.command:
        print("Error: no subcommand specified. Use --help.", file=sys.stderr)
        return 1

    setup_logging(verbose=getattr(args, "verbose", False))

    try:
        if args.command == "discover":
            return cmd_discover(args)
        if args.command == "run-patient":
            return cmd_run_patient(args)
        if args.command == "run":
            return cmd_run(args)
        if args.command == "verify":
            return cmd_verify(args)
        print(f"Unknown command: {args.command}", file=sys.stderr)
        return 1
    except FileNotFoundError as e:
        logging.getLogger(__name__).error("File not found: %s", e)
        return 1
    except KeyboardInterrupt:
        logging.getLogger(__name__).warning("Interrupted by user")
        return 130
    except Exception as e:  # noqa: BLE001
        logging.getLogger(__name__).exception("Unexpected error: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
