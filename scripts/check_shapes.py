#!/usr/bin/env python3
"""Check NIfTI shapes across the preprocessed dataset.

Scans the preprocessed output directory and reports per-study, per-modality
shapes. Flags any volume that deviates from the expected BraTS shape.

Usage (login node):
    python scripts/check_shapes.py /path/to/preprocessed/MenGrowth-2025
    python scripts/check_shapes.py /path/to/preprocessed/MenGrowth-2025 --expected 240 240 155
    python scripts/check_shapes.py /path/to/preprocessed/MenGrowth-2025 --csv shapes.csv
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Tuple

import nibabel as nib


EXPECTED_SHAPE = (240, 240, 155)
MODALITIES = ("t1c", "t1n", "t2w", "t2f")


def scan_dataset(
    dataset_root: Path, expected: Tuple[int, int, int]
) -> List[dict]:
    """Scan preprocessed dataset and collect shape info.

    Args:
        dataset_root: Path to preprocessed MenGrowth-2025 directory
        expected: Expected BraTS shape

    Returns:
        List of dicts with patient_id, study_id, modality, shape, match fields
    """
    rows = []
    for patient_dir in sorted(dataset_root.iterdir()):
        if not patient_dir.is_dir() or not patient_dir.name.startswith("MenGrowth-"):
            continue
        for study_dir in sorted(patient_dir.iterdir()):
            if not study_dir.is_dir():
                continue
            for mod in MODALITIES:
                nifti = study_dir / f"{mod}.nii.gz"
                if not nifti.exists():
                    rows.append({
                        "patient_id": patient_dir.name,
                        "study_id": study_dir.name,
                        "modality": mod,
                        "shape": "MISSING",
                        "match": False,
                    })
                    continue
                header = nib.load(str(nifti)).header
                shape = tuple(int(d) for d in header.get_data_shape()[:3])
                rows.append({
                    "patient_id": patient_dir.name,
                    "study_id": study_dir.name,
                    "modality": mod,
                    "shape": f"{shape[0]}x{shape[1]}x{shape[2]}",
                    "match": shape == expected,
                })
    return rows


def main() -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Check NIfTI shapes in preprocessed MenGrowth dataset."
    )
    parser.add_argument(
        "dataset_root", type=Path, help="Path to preprocessed MenGrowth-2025 directory"
    )
    parser.add_argument(
        "--expected", type=int, nargs=3, default=list(EXPECTED_SHAPE),
        metavar=("X", "Y", "Z"), help=f"Expected shape (default: {EXPECTED_SHAPE})"
    )
    parser.add_argument(
        "--csv", type=Path, default=None, help="Save results to CSV file"
    )
    args = parser.parse_args()

    if not args.dataset_root.is_dir():
        print(f"ERROR: Not a directory: {args.dataset_root}", file=sys.stderr)
        return 1

    expected = tuple(args.expected)
    rows = scan_dataset(args.dataset_root, expected)

    if not rows:
        print("No studies found.")
        return 1

    # Aggregate stats
    total = len(rows)
    missing = sum(1 for r in rows if r["shape"] == "MISSING")
    matched = sum(1 for r in rows if r["match"])
    mismatched = total - missing - matched

    # Print per-study summary (one line per study, compact)
    print(f"Expected shape: {expected[0]}x{expected[1]}x{expected[2]}")
    print(f"{'=' * 72}")

    current_patient = None
    for r in rows:
        if r["patient_id"] != current_patient:
            current_patient = r["patient_id"]
            print(f"\n{current_patient}")

        flag = ""
        if r["shape"] == "MISSING":
            flag = " !! MISSING"
        elif not r["match"]:
            flag = " << MISMATCH"

        # Print one modality per line within study grouping
        if r["modality"] == MODALITIES[0]:
            print(f"  {r['study_id']}")
        print(f"    {r['modality']:4s}  {r['shape']:>15s}{flag}")

    # Summary
    print(f"\n{'=' * 72}")
    print(f"Total volumes: {total}")
    print(f"  Match:    {matched:4d} ({100 * matched / total:.1f}%)")
    print(f"  Mismatch: {mismatched:4d} ({100 * mismatched / total:.1f}%)")
    print(f"  Missing:  {missing:4d} ({100 * missing / total:.1f}%)")

    # Unique mismatch shapes
    bad_shapes = sorted({r["shape"] for r in rows if not r["match"] and r["shape"] != "MISSING"})
    if bad_shapes:
        print(f"\nUnique mismatch shapes: {', '.join(bad_shapes)}")

    # Save CSV if requested
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["patient_id", "study_id", "modality", "shape", "match"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved to {args.csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
