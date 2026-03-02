"""Data types for MenGrowth post-processing dataset analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Final, List, Optional, Tuple

import numpy as np

# ── Segmentation label definitions ──────────────────────────────────────────

LABEL_IDS: Final[List[int]] = [1, 2, 3]

LABEL_NAMES: Final[Dict[int, str]] = {
    1: "NET",
    2: "SNFH",
    3: "ET",
}

LABEL_COLORS_HEX: Final[Dict[int, str]] = {
    1: "#DDCC77",  # NET  — yellow/sand
    2: "#44AA99",  # SNFH — teal
    3: "#CC3311",  # ET   — red
}

LABEL_COLORS_RGB: Final[Dict[int, Tuple[float, float, float]]] = {
    1: (0.867, 0.800, 0.467),
    2: (0.267, 0.667, 0.600),
    3: (0.800, 0.200, 0.067),
}

MODALITIES: Final[List[str]] = ["t1n", "t1c", "t2f", "t2w"]

MIDLINE_X: Final[int] = 120  # RAS axis-0 midpoint for 240-wide volumes
MIDLINE_BAND: Final[int] = 5  # ±5 voxels around midline


# ── Dataclasses ─────────────────────────────────────────────────────────────


@dataclass
class StudyMetrics:
    """Metrics for a single study (patient × timepoint)."""

    patient_id: str
    study_id: str
    timepoint_index: int
    volumes: Dict[int, float]  # label_id → volume in mm³ (= voxel count at 1mm³)
    total_volume: float
    centroid: Optional[Tuple[float, float, float]]  # (x, y, z) voxel coords
    is_empty: bool
    label_proportions: Dict[int, float]  # label_id → fraction of total


@dataclass
class DicePair:
    """Dice coefficients between consecutive timepoints of the same patient."""

    patient_id: str
    study_a: str
    study_b: str
    pair_index: int  # 0 = t₀–t₁, 1 = t₁–t₂, …
    dice_total: float
    dice_per_label: Dict[int, float]


@dataclass
class PatientMetrics:
    """Aggregated metrics for one patient across all timepoints."""

    patient_id: str
    studies: List[StudyMetrics]
    n_studies: int
    dice_pairs: List[DicePair]
    growth_absolute: List[float]  # mm³ change per consecutive step
    growth_relative: List[float]  # % change per step (NaN when denominator is 0)
    clinical: Optional[Dict[str, Any]] = None


@dataclass
class DatasetMetrics:
    """Full dataset analysis results."""

    patients: Dict[str, PatientMetrics]
    tumor_heatmap: np.ndarray  # 3-D frequency map (count of patients with tumor)
    label_heatmaps: Dict[int, np.ndarray]  # per-label frequency maps
    mean_brain: np.ndarray  # mean t1c template
    volume_shape: Tuple[int, int, int]
    empty_studies: List[Tuple[str, str, int]]  # (patient_id, study_id, tp_idx)
    n_patients: int
    n_studies: int
    n_non_empty: int
