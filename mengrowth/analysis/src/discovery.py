"""Phase 1: Dataset discovery — walk preprocessed data and compute all metrics."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np

from .types import (
    LABEL_IDS,
    DatasetMetrics,
    DicePair,
    PatientMetrics,
    StudyMetrics,
)

logger = logging.getLogger(__name__)


# ── NIfTI I/O ───────────────────────────────────────────────────────────────


def _load_seg(path: Path) -> np.ndarray:
    """Load a segmentation NIfTI as int32 in RAS orientation."""
    img = nib.load(str(path))
    img = nib.as_closest_canonical(img)
    return np.asanyarray(img.get_fdata()).astype(np.int32)


def _load_volume(path: Path) -> np.ndarray:
    """Load a NIfTI volume as float32 in RAS orientation."""
    img = nib.load(str(path))
    img = nib.as_closest_canonical(img)
    return np.asanyarray(img.get_fdata()).astype(np.float32)


# ── Dice coefficient ────────────────────────────────────────────────────────


def compute_dice(
    seg_a: np.ndarray, seg_b: np.ndarray
) -> Tuple[float, Dict[int, float]]:
    """Compute Dice coefficient between two atlas-registered segmentations.

    Args:
        seg_a: First segmentation (integer labels 0-3).
        seg_b: Second segmentation (integer labels 0-3).

    Returns:
        Tuple of (total_dice, {label_id: dice}) where NaN means both
        sides are empty and 0.0 means one side is empty.
    """
    a_empty = not np.any(seg_a > 0)
    b_empty = not np.any(seg_b > 0)

    if a_empty and b_empty:
        return float("nan"), {l: float("nan") for l in LABEL_IDS}
    if a_empty or b_empty:
        return 0.0, {l: 0.0 for l in LABEL_IDS}

    # Binary (total) Dice
    a_bin = seg_a > 0
    b_bin = seg_b > 0
    total_dice = (
        2.0 * float(np.sum(a_bin & b_bin)) / float(np.sum(a_bin) + np.sum(b_bin))
    )

    # Per-label Dice
    per_label: Dict[int, float] = {}
    for label in LABEL_IDS:
        a_l = seg_a == label
        b_l = seg_b == label
        tot = float(np.sum(a_l) + np.sum(b_l))
        per_label[label] = (
            2.0 * float(np.sum(a_l & b_l)) / tot if tot > 0 else float("nan")
        )

    return total_dice, per_label


# ── Per-study metrics ────────────────────────────────────────────────────────


def _compute_study_metrics(
    seg: np.ndarray,
    patient_id: str,
    study_id: str,
    tp_idx: int,
) -> StudyMetrics:
    """Compute volume, centroid, and label proportions for one study."""
    is_empty = not np.any(seg > 0)

    volumes: Dict[int, float] = {}
    for label in LABEL_IDS:
        volumes[label] = float(np.sum(seg == label))  # 1 mm³ isotropic

    total_volume = sum(volumes.values())

    label_proportions: Dict[int, float] = {}
    for label in LABEL_IDS:
        label_proportions[label] = (
            volumes[label] / total_volume if total_volume > 0 else 0.0
        )

    centroid: Optional[Tuple[float, float, float]] = None
    if not is_empty:
        coords = np.argwhere(seg > 0)
        centroid = tuple(coords.mean(axis=0).tolist())

    return StudyMetrics(
        patient_id=patient_id,
        study_id=study_id,
        timepoint_index=tp_idx,
        volumes=volumes,
        total_volume=total_volume,
        centroid=centroid,
        is_empty=is_empty,
        label_proportions=label_proportions,
    )


# ── Clinical metadata ───────────────────────────────────────────────────────


def load_clinical_metadata(
    metadata_root: Path,
) -> Dict[str, Dict[str, Any]]:
    """Load clinical metadata keyed by MenGrowth patient ID.

    Args:
        metadata_root: Directory with id_mapping.json and metadata_clean.json.

    Returns:
        Dict mapping ``MenGrowth-XXXX`` → clinical info dict.
    """
    mapping_path = metadata_root / "id_mapping.json"
    metadata_path = metadata_root / "metadata_clean.json"

    if not mapping_path.exists() or not metadata_path.exists():
        logger.warning("Clinical metadata files not found at %s", metadata_root)
        return {}

    with open(mapping_path) as f:
        id_mapping = json.load(f)
    with open(metadata_path) as f:
        metadata_clean = json.load(f)

    # Reverse map: MenGrowth-XXXX → original P-ID
    reverse_map: Dict[str, str] = {}
    for orig_id, info in id_mapping.items():
        reverse_map[info["new_id"]] = orig_id

    clinical: Dict[str, Dict[str, Any]] = {}
    for men_id, orig_id in reverse_map.items():
        if orig_id not in metadata_clean:
            continue
        raw = metadata_clean[orig_id]
        general = raw.get("general", {})

        # Extract study dates
        dates: List[Optional[str]] = []
        first_study = raw.get("first_study", {})
        dates.append(first_study.get("rm", {}).get("date"))

        for key in sorted(raw.keys()):
            if key.startswith("c") and key[1:].isdigit():
                dates.append(raw[key].get("date"))

        clinical[men_id] = {
            "age": general.get("age"),
            "sex": general.get("sex"),
            "dates": dates,
            "original_id": orig_id,
        }

    logger.info(
        "Loaded clinical metadata for %d/%d patients",
        len(clinical),
        len(reverse_map),
    )
    return clinical


# ── Main discovery ───────────────────────────────────────────────────────────


def discover_dataset(
    dataset_root: Path,
    metadata_root: Optional[Path] = None,
    compute_mean_brain: bool = True,
) -> DatasetMetrics:
    """Walk the preprocessed dataset and compute all analysis metrics.

    Args:
        dataset_root: Root containing ``MenGrowth-XXXX/`` patient directories.
        metadata_root: Optional path to curated metadata for clinical info.
        compute_mean_brain: Whether to accumulate a mean t1c template.

    Returns:
        Fully populated ``DatasetMetrics``.
    """
    patient_dirs = sorted(
        d
        for d in dataset_root.iterdir()
        if d.is_dir() and d.name.startswith("MenGrowth-")
    )
    if not patient_dirs:
        raise FileNotFoundError(f"No MenGrowth-* directories in {dataset_root}")

    # Determine volume shape from first segmentation
    first_study = sorted(
        d
        for d in patient_dirs[0].iterdir()
        if d.is_dir() and d.name.startswith("MenGrowth-")
    )[0]
    shape = tuple(nib.load(str(first_study / "seg.nii.gz")).shape)
    logger.info("Volume shape: %s", shape)

    # Accumulators
    heatmap = np.zeros(shape, dtype=np.float32)
    label_heatmaps = {l: np.zeros(shape, dtype=np.float32) for l in LABEL_IDS}
    mean_brain_acc = np.zeros(shape, dtype=np.float64) if compute_mean_brain else None
    n_brain = 0

    patients: Dict[str, PatientMetrics] = {}
    empty_studies: List[Tuple[str, str, int]] = []
    total_studies = 0
    total_non_empty = 0

    clinical = load_clinical_metadata(metadata_root) if metadata_root else {}

    for patient_dir in patient_dirs:
        patient_id = patient_dir.name
        study_dirs = sorted(
            d
            for d in patient_dir.iterdir()
            if d.is_dir() and d.name.startswith("MenGrowth-")
        )

        logger.info("Processing %s (%d studies)", patient_id, len(study_dirs))

        study_metrics_list: List[StudyMetrics] = []
        dice_pairs: List[DicePair] = []
        prev_seg: Optional[np.ndarray] = None
        prev_study_id: Optional[str] = None

        for tp_idx, study_dir in enumerate(study_dirs):
            study_id = study_dir.name
            total_studies += 1

            seg_path = study_dir / "seg.nii.gz"
            if not seg_path.exists():
                logger.warning("No seg.nii.gz in %s, skipping", study_dir)
                continue

            seg = _load_seg(seg_path)
            sm = _compute_study_metrics(seg, patient_id, study_id, tp_idx)
            study_metrics_list.append(sm)

            if sm.is_empty:
                empty_studies.append((patient_id, study_id, tp_idx))
                logger.info("  %s: EMPTY segmentation", study_id)
            else:
                total_non_empty += 1
                heatmap += (seg > 0).astype(np.float32)
                for label in LABEL_IDS:
                    label_heatmaps[label] += (seg == label).astype(np.float32)
                logger.info(
                    "  %s: %.0f mm³ (NET=%.0f, SNFH=%.0f, ET=%.0f)",
                    study_id,
                    sm.total_volume,
                    sm.volumes[1],
                    sm.volumes[2],
                    sm.volumes[3],
                )

            # Dice with previous timepoint
            if prev_seg is not None:
                d_total, d_labels = compute_dice(prev_seg, seg)
                dice_pairs.append(
                    DicePair(
                        patient_id=patient_id,
                        study_a=prev_study_id,
                        study_b=study_id,
                        pair_index=tp_idx - 1,
                        dice_total=d_total,
                        dice_per_label=d_labels,
                    )
                )

            prev_seg = seg
            prev_study_id = study_id

            # Mean brain accumulation
            if compute_mean_brain:
                t1c_path = study_dir / "t1c.nii.gz"
                if t1c_path.exists():
                    mean_brain_acc += _load_volume(t1c_path).astype(np.float64)
                    n_brain += 1

        # Growth rates
        growth_abs: List[float] = []
        growth_rel: List[float] = []
        for i in range(1, len(study_metrics_list)):
            prev_vol = study_metrics_list[i - 1].total_volume
            curr_vol = study_metrics_list[i].total_volume
            abs_change = curr_vol - prev_vol
            growth_abs.append(abs_change)
            growth_rel.append(
                100.0 * abs_change / prev_vol if prev_vol > 0 else float("nan")
            )

        patients[patient_id] = PatientMetrics(
            patient_id=patient_id,
            studies=study_metrics_list,
            n_studies=len(study_metrics_list),
            dice_pairs=dice_pairs,
            growth_absolute=growth_abs,
            growth_relative=growth_rel,
            clinical=clinical.get(patient_id),
        )

    # Finalize mean brain
    mean_brain = np.zeros(shape, dtype=np.float32)
    if compute_mean_brain and n_brain > 0:
        mean_brain = (mean_brain_acc / n_brain).astype(np.float32)
        logger.info("Mean brain template computed from %d volumes", n_brain)

    logger.info(
        "Dataset: %d patients, %d studies (%d non-empty, %d empty)",
        len(patients),
        total_studies,
        total_non_empty,
        len(empty_studies),
    )

    return DatasetMetrics(
        patients=patients,
        tumor_heatmap=heatmap,
        label_heatmaps=label_heatmaps,
        mean_brain=mean_brain,
        volume_shape=shape,
        empty_studies=empty_studies,
        n_patients=len(patients),
        n_studies=total_studies,
        n_non_empty=total_non_empty,
    )
