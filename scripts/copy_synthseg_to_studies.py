#!/usr/bin/env python3
"""Copy SynthSeg parcellations into each preprocessed patient-study directory.

Walks the SynthSeg output tree and copies every ``synthseg_parc.nii.gz`` to
the matching preprocessed study directory under the new name
``synthseg.nii.gz`` (i.e. sitting alongside ``t1n.nii.gz``, ``t1c.nii.gz``,
``t2w.nii.gz``, ``t2f.nii.gz``, ``seg.nii.gz``).

Usage::

    python scripts/copy_synthseg_to_studies.py \\
        --synthseg-root /media/.../v5_final/synthseg \\
        --preprocessed-root /media/.../v5_final/MenGrowth-2025
        [--overwrite] [--dry-run] [--workers 4]

Defaults match the local portable SSD layout. Source files remain in place.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

DEFAULT_SYNTHSEG_ROOT = Path(
    "/media/mpascual/PortableSSD/Meningiomas/MenGrowth/v5_final/synthseg"
)
DEFAULT_PREPROCESSED_ROOT = Path(
    "/media/mpascual/PortableSSD/Meningiomas/MenGrowth/v5_final/MenGrowth-2025"
)
SOURCE_NAME = "synthseg_parc.nii.gz"
DEST_NAME = "synthseg.nii.gz"


@dataclass(frozen=True)
class CopyJob:
    patient_id: str
    study_id: str
    src: Path
    dst: Path


@dataclass(frozen=True)
class CopyResult:
    job: CopyJob
    status: str  # "copied" | "skipped_exists" | "missing_dest_dir" | "failed"
    error: str = ""


def discover_jobs(
    synthseg_root: Path, preprocessed_root: Path
) -> list[CopyJob]:
    """Enumerate every (patient, study) with a SynthSeg parcellation."""
    jobs: list[CopyJob] = []
    for patient_dir in sorted(synthseg_root.iterdir()):
        if not patient_dir.is_dir() or not patient_dir.name.startswith("MenGrowth-"):
            continue
        for study_dir in sorted(patient_dir.iterdir()):
            if not study_dir.is_dir() or not study_dir.name.startswith(patient_dir.name):
                continue
            src = study_dir / SOURCE_NAME
            if not src.exists():
                logger.debug("No parcellation at %s", src)
                continue
            dst = preprocessed_root / patient_dir.name / study_dir.name / DEST_NAME
            jobs.append(CopyJob(patient_dir.name, study_dir.name, src, dst))
    return jobs


def copy_one(job: CopyJob, overwrite: bool, dry_run: bool) -> CopyResult:
    if not job.dst.parent.exists():
        return CopyResult(job, "missing_dest_dir", f"no dir {job.dst.parent}")
    if job.dst.exists() and not overwrite:
        return CopyResult(job, "skipped_exists")
    if dry_run:
        return CopyResult(job, "copied")
    try:
        tmp = job.dst.with_suffix(".tmp.nii.gz")
        shutil.copy2(job.src, tmp)
        tmp.replace(job.dst)
        return CopyResult(job, "copied")
    except OSError as e:
        return CopyResult(job, "failed", str(e))


def run(
    jobs: Iterable[CopyJob], overwrite: bool, dry_run: bool, workers: int
) -> list[CopyResult]:
    jobs = list(jobs)
    results: list[CopyResult] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(copy_one, j, overwrite, dry_run): j for j in jobs}
        for fut in as_completed(futures):
            results.append(fut.result())
    results.sort(key=lambda r: (r.job.patient_id, r.job.study_id))
    return results


def summarise(results: list[CopyResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    return counts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--synthseg-root", type=Path, default=DEFAULT_SYNTHSEG_ROOT)
    p.add_argument("--preprocessed-root", type=Path, default=DEFAULT_PREPROCESSED_ROOT)
    p.add_argument("--overwrite", action="store_true",
                   help="Replace existing synthseg.nii.gz files.")
    p.add_argument("--dry-run", action="store_true",
                   help="Plan only — do not write any files.")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    if not args.synthseg_root.exists():
        logger.error("synthseg root does not exist: %s", args.synthseg_root)
        return 2
    if not args.preprocessed_root.exists():
        logger.error("preprocessed root does not exist: %s", args.preprocessed_root)
        return 2

    jobs = discover_jobs(args.synthseg_root, args.preprocessed_root)
    logger.info("Discovered %d parcellations to copy", len(jobs))
    if not jobs:
        return 0

    results = run(jobs, args.overwrite, args.dry_run, args.workers)
    counts = summarise(results)
    logger.info(
        "%s summary: copied=%d, skipped_exists=%d, missing_dest_dir=%d, failed=%d",
        "DRY-RUN" if args.dry_run else "Copy",
        counts.get("copied", 0),
        counts.get("skipped_exists", 0),
        counts.get("missing_dest_dir", 0),
        counts.get("failed", 0),
    )
    for r in results:
        if r.status in ("missing_dest_dir", "failed"):
            logger.warning(
                "%s/%s: %s — %s",
                r.job.patient_id, r.job.study_id, r.status, r.error,
            )
    return 2 if counts.get("failed", 0) else 0


if __name__ == "__main__":
    sys.exit(main())
