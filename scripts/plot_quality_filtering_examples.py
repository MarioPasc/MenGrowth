#!/usr/bin/env python
"""Generate a publication-ready grid of quality filtering examples.

Creates a 3×N figure (axial/sagittal/coronal × selected rejection categories
+ 1 high-quality) for inclusion in the MenGrowth thesis. Uses IEEE-compliant
styles from ``mengrowth.preprocessing.utils.settings``.

Each rejection column displays the **worst case** for the filter — the scan
with the most extreme metric value — using the actual modality that triggered
the block.

Default columns (all five filters + high quality):
    (a) Low SNR
    (b) Intensity outliers
    (c) Motion artifacts
    (d) Asymmetric FOV
    (e) Insufficient brain coverage
    (f) High quality (accepted)

Rows:
    Axial, sagittal, coronal central slices.

Usage::

    # All filters (default):
    python scripts/plot_quality_filtering_examples.py \
        --output-root /media/mpascual/PortableSSD/Meningiomas/MenGrowth/curated \
        --output figures/quality_filtering_examples.pdf

    # Subset of filters:
    python scripts/plot_quality_filtering_examples.py \
        --filters motion,fov,braincov \
        --output-root /media/mpascual/PortableSSD/Meningiomas/MenGrowth/curated \
        --output figures/quality_filtering_subset.pdf

Available filter names: snr, intensity, motion, fov, braincov

Notes:
    Rejected files are removed during quality filtering. This script searches
    for the original raw data in ``--raw-root`` using the study/modality
    mappings from the config. If raw files cannot be found, run the curation
    pipeline with ``remove_blocked: false`` to restore them.
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk  # noqa: N813 — used in _reorient_to_ras / _resample_to_isotropic

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mengrowth.preprocessing.utils.settings import (
    PLOT_SETTINGS,
    apply_ieee_style,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Quality-issue categories mapped to check names in quality_issues.csv.
# Dict key = short CLI name; value = (csv check_name, display label).
FILTER_REGISTRY: Dict[str, Tuple[str, str]] = {
    "snr": ("snr_filtering", "Low SNR"),
    "intensity": ("intensity_outliers", "Intensity Outliers"),
    "motion": ("motion_artifact", "Motion Artifacts"),
    "ghosting": ("ghosting_detection", "Ghosting"),
    "fov": ("fov_consistency", "Asymmetric FOV"),
    "braincov": ("brain_coverage", "Insuf. Brain Coverage"),
}

# Default filter order when --filters is not specified.
DEFAULT_FILTERS = ["snr", "intensity", "motion", "ghosting", "fov", "braincov"]

# Reverse mapping: standardized modality → possible raw file stems
MODALITY_RAW_NAMES: Dict[str, List[str]] = {
    "t1c": ["T1ce", "T1-ce", "T1post", "T1-post", "T1"],
    "t1n": ["T1pre", "T1-pre", "T1SIN", "T1sin"],
    "t2w": ["T2", "t2", "T2-weighted"],
    "t2f": ["FLAIR", "flair", "T2-FLAIR", "T2flair"],
}

# Reverse mapping: study number → possible raw directory names
STUDY_NUM_TO_NAMES: Dict[str, List[str]] = {
    "0": ["primera", "primero", "primera1", "primero1", "baseline"],
    "1": ["segunda", "segundo", "segunda1", "segundo1", "control1"],
    "2": ["tercera", "tercero", "control2"],
    "3": ["cuarta", "cuarto", "control3"],
    "4": ["quinta", "quinto", "control4"],
    "5": ["sexta", "sexto", "control5"],
    "6": ["septima", "septimo", "control6"],
    "7": ["octava", "octavo", "control7"],
    "8": ["novena", "noveno", "control8"],
    "9": ["decima", "decimo", "control9"],
    "10": ["undecima", "undecimo", "control10"],
}


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def load_quality_issues(quality_dir: Path) -> List[Dict[str, Any]]:
    """Load quality_issues.csv and return list of dicts."""
    csv_path = quality_dir / "quality_issues.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"quality_issues.csv not found at {csv_path}")
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _extract_metric_value(message: str) -> Optional[float]:
    """Extract the numeric metric value from a quality-issue message.

    Supported formats:
        "SNR 3.80 < 8.0 (t1c)"               → 3.80
        "Extreme outlier: max/99th = 22.3 ..." → 22.3
        "Low gradient entropy 1.35 < 2.7 ..."  → 1.35
        "Extreme FOV asymmetry: ratio 5.15 ..."→ 5.15
        "Min physical extent 36.3mm < 100.0mm" → 36.3
    """

    # "= <value>" pattern (intensity outliers)
    m = re.search(r"=\s*([\d.]+)", message)
    if m:
        return float(m.group(1))
    # "ratio <value>" pattern (FOV)
    m = re.search(r"ratio\s+([\d.]+)", message)
    if m:
        return float(m.group(1))
    # "extent <value>mm" pattern (brain coverage)
    m = re.search(r"extent\s+([\d.]+)\s*mm", message)
    if m:
        return float(m.group(1))
    # "entropy <value>" pattern (motion)
    m = re.search(r"entropy\s+([\d.]+)", message)
    if m:
        return float(m.group(1))
    # "SNR <value>" pattern
    m = re.search(r"SNR\s+([\d.]+)", message)
    if m:
        return float(m.group(1))
    # Generic: first float in the message
    m = re.search(r"([\d]+\.[\d]+)", message)
    if m:
        return float(m.group(1))
    return None


# Checks where a LOWER metric value means worse quality.
_LOWER_IS_WORSE = {"snr_filtering", "motion_artifact", "brain_coverage"}


def find_best_candidate(
    issues: List[Dict[str, Any]],
    check_name: str,
    used_patients: Optional[set] = None,
) -> Optional[Dict[str, Any]]:
    """Pick the **worst** blocked file for *check_name*.

    Selects the candidate with the most extreme metric value so the figure
    showcases the clearest example of each quality issue.

    "Worst" means:
        - Lowest value for SNR, motion entropy, brain coverage
        - Highest value for intensity outlier ratio, FOV asymmetry ratio

    Args:
        issues: Rows from quality_issues.csv.
        check_name: Quality check to look for (e.g. ``"snr_filtering"``).
        used_patients: Patient IDs already claimed by earlier columns
            (unused — kept for API compatibility).

    Returns:
        Row with the most extreme metric, or *None*.
    """
    blocked = [
        row
        for row in issues
        if row["check_name"] == check_name and row["action"] == "block"
    ]
    if not blocked:
        return None

    # Annotate with parsed metric value
    scored = []
    for row in blocked:
        val = _extract_metric_value(row["message"])
        if val is not None:
            scored.append((val, row))

    if not scored:
        return blocked[0]

    # Sort: lower-is-worse checks → ascending (worst first);
    #        higher-is-worse checks → descending (worst first).
    if check_name in _LOWER_IS_WORSE:
        scored.sort(key=lambda x: x[0])  # smallest first
    else:
        scored.sort(key=lambda x: x[0], reverse=True)  # largest first

    return scored[0][1]


# ---------------------------------------------------------------------------
# Raw-file search helpers
# ---------------------------------------------------------------------------


def _match_modality_file(
    nrrd_files: List[Path],
    modality: str,
) -> Optional[Path]:
    """Match a raw NRRD file to a standardized modality name.

    Handles suffixed filenames (e.g. ``T1_P16_01.nrrd``, ``FLAIR_P16_01.nrrd``)
    by checking if the filename starts with a known synonym.  Uses longest-match
    to avoid confusing ``T1SIN`` (t1n) with ``T1`` (t1c).
    """
    raw_names = MODALITY_RAW_NAMES.get(modality, [modality])

    # Prefixes belonging to OTHER modalities → exclusion set.
    other_prefixes: List[str] = []
    for other_mod, other_names in MODALITY_RAW_NAMES.items():
        if other_mod != modality:
            other_prefixes.extend(other_names)

    # Most-specific first
    for rname in sorted(raw_names, key=len, reverse=True):
        rname_lower = rname.lower()
        for fpath in nrrd_files:
            stem = fpath.stem.lower()
            if not stem.startswith(rname_lower):
                continue
            # Reject if a longer OTHER-modality prefix also matches
            is_other = any(
                stem.startswith(op.lower()) and len(op) > len(rname)
                for op in other_prefixes
            )
            if not is_other:
                return fpath

    return None


def _search_raw_file(
    patient_id: str,
    study_num: str,
    modality: str,
    raw_root: Path,
) -> Optional[Path]:
    """Search for the raw NRRD file across known directory layouts."""
    num = patient_id.lstrip("P")
    study_names = STUDY_NUM_TO_NAMES.get(str(study_num), [])

    candidate_dirs: List[Path] = []
    for sname in study_names:
        candidate_dirs.extend(
            [
                raw_root / "processed" / "extension_1" / num / sname,
                raw_root / "processed" / "source" / "controls" / patient_id / sname,
                raw_root / "source" / "men" / patient_id / sname,
                raw_root
                / "source"
                / "original_ordered"
                / "MenGrowth"
                / "controls"
                / patient_id
                / sname,
            ]
        )

    for cdir in candidate_dirs:
        if not cdir.exists():
            continue
        nrrd_files = list(cdir.glob("*.nrrd"))
        if not nrrd_files:
            continue
        # Exact name first (extension_1 style: T1ce.nrrd, FLAIR.nrrd)
        for rmod in MODALITY_RAW_NAMES.get(modality, [modality]):
            exact = cdir / f"{rmod}.nrrd"
            if exact.exists():
                return exact
        # Fuzzy match (controls style: T1_P16_01.nrrd, FLAIR_P16_01.nrrd)
        match = _match_modality_file(nrrd_files, modality)
        if match is not None:
            return match

    # Recursive glob fallback
    for rmod in MODALITY_RAW_NAMES.get(modality, [modality]):
        for pattern in [f"**/{rmod}.nrrd", f"**/{rmod}_*.nrrd", f"**/{rmod}*.nrrd"]:
            for match in raw_root.glob(pattern):
                if num in str(match) or patient_id in str(match):
                    return match

    return None


def find_image_file(
    candidate: Dict[str, Any],
    raw_root: Optional[Path],
) -> Optional[Path]:
    """Locate the NRRD file for *candidate*'s own modality on disk.

    Tries the recorded curated path first, then searches raw directories.
    """
    modality = candidate["modality"]

    # 1. Recorded curated path (may still exist for partial rejections)
    recorded_path = Path(candidate["file_path"])
    if recorded_path.exists():
        return recorded_path

    # 2. Search raw data
    if raw_root is not None:
        found = _search_raw_file(
            candidate["patient_id"],
            candidate["study_id"],
            modality,
            raw_root,
        )
        if found:
            return found

    return None


def find_high_quality_file(
    output_root: Path,
    sequence: str = "t1c",
) -> Optional[Path]:
    """Find a high-quality (accepted) file from the curated dataset."""
    dataset_dir = output_root / "dataset"
    patient_dirs = sorted(dataset_dir.glob("MenGrowth-*/MenGrowth-*"))
    if not patient_dirs:
        patient_dirs = sorted(
            dataset_dir.glob("MenGrowth-2025/MenGrowth-*/MenGrowth-*")
        )
    for study_dir in patient_dirs:
        candidate = study_dir / f"{sequence}.nrrd"
        if candidate.exists():
            return candidate
    for match in sorted(dataset_dir.rglob(f"{sequence}.nrrd")):
        return match
    return None


# ---------------------------------------------------------------------------
# Image loading (RAS + 1 mm³)
# ---------------------------------------------------------------------------


def _reorient_to_ras(img: sitk.Image) -> sitk.Image:
    """Reorient a SimpleITK image to RAS orientation."""
    orient_filter = sitk.DICOMOrientImageFilter()
    orient_filter.SetDesiredCoordinateOrientation("RAS")
    return orient_filter.Execute(img)


def _resample_to_isotropic(
    img: sitk.Image,
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> sitk.Image:
    """Resample to isotropic spacing with nearest-neighbor interpolation."""
    original_spacing = img.GetSpacing()
    original_size = img.GetSize()
    new_size = [
        int(round(osz * ospc / tspc))
        for osz, ospc, tspc in zip(original_size, original_spacing, target_spacing)
    ]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(img)


def load_nrrd_image(file_path: Path) -> Optional[np.ndarray]:
    """Load a 3D image, reorient to RAS, resample to 1 mm³ isotropic."""
    try:
        img = sitk.ReadImage(str(file_path))
        img = _reorient_to_ras(img)
        img = _resample_to_isotropic(img, target_spacing=(1.0, 1.0, 1.0))
        data = sitk.GetArrayFromImage(img)  # shape: (Z, Y, X)
        return data.astype(np.float32)
    except Exception as e:
        logger.warning("Failed to load %s: %s", file_path, e)
        return None


# ---------------------------------------------------------------------------
# Slice extraction and normalization
# ---------------------------------------------------------------------------


def get_central_slices(
    volume: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract central axial, sagittal, and coronal slices.

    Expects SimpleITK array ordering (Z, Y, X) after RAS reorientation:
      - axis 0 = inferior→superior  (axial slice index)
      - axis 1 = posterior→anterior (coronal slice index)
      - axis 2 = left→right        (sagittal slice index)

    Returns:
        (axial, sagittal, coronal) — each a 2D array.
    """
    cz = volume.shape[0] // 2
    cy = volume.shape[1] // 2
    cx = volume.shape[2] // 2

    axial = volume[cz, :, :]  # Y × X
    sagittal = volume[:, :, cx]  # Z × Y
    coronal = volume[:, cy, :]  # Z × X
    return axial, sagittal, coronal


def normalize_slice(
    slice_2d: np.ndarray,
    percentile_clip: float = 99.5,
) -> np.ndarray:
    """Normalize a 2D slice to [0, 1] with percentile clipping."""
    vmin = 0.0
    vmax = (
        np.percentile(slice_2d[slice_2d > 0], percentile_clip)
        if np.any(slice_2d > 0)
        else 1.0
    )
    if vmax <= vmin:
        vmax = vmin + 1.0
    clipped = np.clip(slice_2d, vmin, vmax)
    return (clipped - vmin) / (vmax - vmin)


# ---------------------------------------------------------------------------
# Main plotting function
# ---------------------------------------------------------------------------


def plot_quality_filtering_grid(
    output_root: str,
    output_path: str = "quality_filtering_examples.pdf",
    raw_root: Optional[str] = None,
    filters: Optional[List[str]] = None,
    hq_sequence: str = "t1c",
    dpi: int = 300,
) -> None:
    """Generate the quality filtering example grid figure.

    Each rejection column shows the **actual modality that caused the block**,
    so the artifact is directly visible.  The last column always shows a
    high-quality accepted scan for comparison.

    Args:
        output_root: Path to the curated output root (containing quality/).
        output_path: Destination file (PDF or PNG).
        raw_root: Raw data root for locating rejected files.
        filters: Ordered list of filter short-names to include
            (e.g. ``["snr", "motion", "fov"]``).  Defaults to all five.
        hq_sequence: Modality for the high-quality column (default ``t1c``).
        dpi: Output DPI.
    """
    if filters is None:
        filters = list(DEFAULT_FILTERS)

    # Validate filter names
    for f in filters:
        if f not in FILTER_REGISTRY:
            raise ValueError(
                f"Unknown filter '{f}'. Available: {', '.join(FILTER_REGISTRY)}"
            )

    # Build the active category list in the requested order
    active_categories: List[Tuple[str, str]] = [FILTER_REGISTRY[f] for f in filters]

    # Column labels: (a), (b), ... for rejection cols + last for HQ
    n_cols = len(active_categories) + 1  # +1 for high-quality
    col_labels = [f"({chr(ord('a') + i)})" for i in range(n_cols)]

    output_root_path = Path(output_root)
    quality_dir = output_root_path / "quality"
    raw_root_path = Path(raw_root) if raw_root else None

    # Infer raw_root from output_root if not specified
    if raw_root_path is None:
        candidate_raw = output_root_path.parent / "raw"
        if candidate_raw.exists():
            raw_root_path = candidate_raw

    # ------------------------------------------------------------------ #
    # Load quality issues
    # ------------------------------------------------------------------ #
    issues = load_quality_issues(quality_dir)
    logger.info("Loaded %d quality issues", len(issues))

    # ------------------------------------------------------------------ #
    # Select worst case per category
    # ------------------------------------------------------------------ #
    volumes: List[Optional[np.ndarray]] = []
    file_info: List[str] = []

    for check_name, label in active_categories:
        candidate = find_best_candidate(issues, check_name)
        if candidate is None:
            logger.warning("No blocked files for check '%s'", check_name)
            volumes.append(None)
            file_info.append(f"{label}: no example found")
            continue

        file_path = find_image_file(candidate, raw_root_path)
        if file_path is None:
            logger.warning(
                "File not found for %s/%s/%s (check: %s). "
                "Pass --raw-root or re-run curation with remove_blocked=false.",
                candidate["patient_id"],
                candidate["study_id"],
                candidate["modality"],
                check_name,
            )
            volumes.append(None)
            file_info.append(
                f"{label}: {candidate['patient_id']}/{candidate['study_id']}/"
                f"{candidate['modality']} (file not found)"
            )
            continue

        vol = load_nrrd_image(file_path)
        volumes.append(vol)
        file_info.append(
            f"{label}: {candidate['patient_id']}/{candidate['study_id']}/"
            f"{candidate['modality']} — {candidate['message']}"
        )
        logger.info("Loaded %s from %s", label, file_path)

    # High-quality example (always the last column)
    hq_path = find_high_quality_file(output_root_path, hq_sequence)
    if hq_path is not None:
        hq_vol = load_nrrd_image(hq_path)
        volumes.append(hq_vol)
        file_info.append(f"High Quality: {hq_path.parent.name}/{hq_sequence}")
        logger.info("Loaded high-quality from %s", hq_path)
    else:
        volumes.append(None)
        file_info.append("High Quality: no example found")
        logger.warning("No high-quality file found for sequence=%s", hq_sequence)

    # ------------------------------------------------------------------ #
    # Print summary
    # ------------------------------------------------------------------ #
    print("\n--- Quality Filtering Grid Summary ---")
    for i, info in enumerate(file_info):
        status = "OK" if volumes[i] is not None else "MISSING"
        print(f"  [{status}] Column {col_labels[i]}: {info}")
    print()

    # ------------------------------------------------------------------ #
    # Create figure
    # ------------------------------------------------------------------ #
    apply_ieee_style()

    n_rows = 3  # axial, sagittal, coronal

    # Figure sizing: full IEEE text width
    fig_width = PLOT_SETTINGS["figure_width_double"]
    cell_size = fig_width / n_cols
    fig_height = cell_size * n_rows + 0.3  # small padding for labels

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_width, fig_height),
        facecolor="black",
    )
    fig.patch.set_facecolor("black")

    # Handle single-column edge case (axes would be 1-D)
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for col_idx in range(n_cols):
        vol = volumes[col_idx]

        if vol is not None:
            axial, sagittal, coronal = get_central_slices(vol)
            slices = [axial, sagittal, coronal]
        else:
            slices = [None, None, None]

        for row_idx in range(n_rows):
            ax = axes[row_idx, col_idx]
            ax.set_facecolor("black")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            if slices[row_idx] is not None:
                normalized = normalize_slice(slices[row_idx])
                # origin='lower' puts Z=0 (inferior) at the bottom →
                # neck at the bottom, vertex at the top.
                ax.imshow(
                    normalized,
                    cmap="gray",
                    origin="lower",
                    aspect="equal",
                    interpolation="bilinear",
                )

        # Column label above the top row
        axes[0, col_idx].set_title(
            col_labels[col_idx],
            fontsize=PLOT_SETTINGS["panel_label_fontsize"],
            fontweight="bold",
            color="white",
            pad=4,
        )

    plt.subplots_adjust(
        left=0.01,
        right=0.99,
        top=0.94,
        bottom=0.01,
        wspace=0.02,
        hspace=0.02,
    )

    # Save
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        str(out),
        dpi=dpi,
        facecolor="black",
        edgecolor="none",
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(fig)
    print(f"Figure saved to {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate quality filtering example grid for thesis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/templates/raw_data.yaml",
        help="Path to raw_data.yaml config file.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="/media/mpascual/PortableSSD/Meningiomas/MenGrowth/curated",
        help="Path to curated output root (containing quality/ and dataset/).",
    )
    parser.add_argument(
        "--raw-root",
        type=str,
        default=None,
        help=(
            "Path to the raw data root for locating rejected files. "
            "If not given, inferred as {output-root}/../raw."
        ),
    )
    parser.add_argument(
        "--filters",
        type=str,
        default=",".join(DEFAULT_FILTERS),
        help=(
            "Comma-separated list of filters to include. "
            f"Available: {', '.join(FILTER_REGISTRY)}. "
            f"Default: {','.join(DEFAULT_FILTERS)}."
        ),
    )
    parser.add_argument(
        "--hq-sequence",
        type=str,
        default="t1c",
        choices=["t1c", "t1n", "t2w", "t2f"],
        help="Modality for the high-quality column (default: t1c).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="figures/quality_filtering_examples.pdf",
        help="Output figure path (PDF or PNG).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output DPI.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    selected_filters = [f.strip() for f in args.filters.split(",") if f.strip()]

    plot_quality_filtering_grid(
        output_root=args.output_root,
        output_path=args.output,
        raw_root=args.raw_root,
        filters=selected_filters,
        hq_sequence=args.hq_sequence,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
