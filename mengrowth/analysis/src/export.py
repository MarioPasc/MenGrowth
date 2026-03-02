"""Phase 4: Export analysis results to JSON and LaTeX tables."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

from .types import LABEL_NAMES, DatasetMetrics

logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _safe(v: float) -> Any:
    """Convert float for JSON (NaN / Inf → None)."""
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return round(v, 2)


# ── JSON export ──────────────────────────────────────────────────────────────


def export_json(metrics: DatasetMetrics, output_path: Path) -> None:
    """Export all analysis metrics as a single JSON file.

    Args:
        metrics: Computed dataset metrics.
        output_path: Destination file (e.g. ``analysis/dataset_analysis.json``).
    """
    all_volumes: list[float] = []

    data: dict[str, Any] = {
        "summary": {
            "n_patients": metrics.n_patients,
            "n_studies": metrics.n_studies,
            "n_non_empty": metrics.n_non_empty,
            "n_empty": len(metrics.empty_studies),
            "volume_shape": list(metrics.volume_shape),
        },
        "empty_studies": [
            {"patient_id": pid, "study_id": sid, "timepoint": tp}
            for pid, sid, tp in metrics.empty_studies
        ],
        "patients": {},
    }

    for pid, pm in sorted(metrics.patients.items()):
        p: dict[str, Any] = {
            "n_studies": pm.n_studies,
            "studies": [],
            "growth_absolute": [_safe(v) for v in pm.growth_absolute],
            "growth_relative": [_safe(v) for v in pm.growth_relative],
            "dice_pairs": [],
        }

        for s in pm.studies:
            p["studies"].append(
                {
                    "study_id": s.study_id,
                    "timepoint": s.timepoint_index,
                    "is_empty": s.is_empty,
                    "total_volume": _safe(s.total_volume),
                    "volumes_per_label": {
                        LABEL_NAMES[l]: _safe(v) for l, v in s.volumes.items()
                    },
                    "label_proportions": {
                        LABEL_NAMES[l]: _safe(v) for l, v in s.label_proportions.items()
                    },
                    "centroid": (
                        [round(c, 1) for c in s.centroid] if s.centroid else None
                    ),
                }
            )
            if not s.is_empty:
                all_volumes.append(s.total_volume)

        for dp in pm.dice_pairs:
            p["dice_pairs"].append(
                {
                    "pair": f"{dp.study_a}-{dp.study_b}",
                    "pair_index": dp.pair_index,
                    "dice_total": _safe(dp.dice_total),
                    "dice_per_label": {
                        LABEL_NAMES[l]: _safe(v) for l, v in dp.dice_per_label.items()
                    },
                }
            )

        if pm.clinical:
            p["clinical"] = pm.clinical

        data["patients"][pid] = p

    # Volume summary statistics
    if all_volumes:
        arr = np.array(all_volumes)
        data["summary"]["volume_stats"] = {
            "mean": _safe(float(np.mean(arr))),
            "median": _safe(float(np.median(arr))),
            "std": _safe(float(np.std(arr))),
            "min": _safe(float(np.min(arr))),
            "max": _safe(float(np.max(arr))),
            "q25": _safe(float(np.percentile(arr, 25))),
            "q75": _safe(float(np.percentile(arr, 75))),
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Exported JSON: %s", output_path)


# ── LaTeX tables ─────────────────────────────────────────────────────────────


def export_latex_tables(metrics: DatasetMetrics, output_dir: Path) -> None:
    """Export dataset summary and per-patient tables as ``.tex`` files.

    Args:
        metrics: Computed dataset metrics.
        output_dir: Directory to write ``dataset_summary.tex`` and
            ``per_patient_summary.tex``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Gather stats ──
    all_volumes: list[float] = []
    studies_per_patient: list[int] = []
    for pm in metrics.patients.values():
        studies_per_patient.append(pm.n_studies)
        for s in pm.studies:
            if not s.is_empty:
                all_volumes.append(s.total_volume)

    vol = np.array(all_volumes) if all_volumes else np.array([0.0])
    spp = np.array(studies_per_patient)

    # ── Table 1: Dataset Summary ──
    summary = (
        r"\begin{table}[htbp]"
        "\n"
        r"\centering"
        "\n"
        r"\caption{Dataset summary statistics.}"
        "\n"
        r"\label{tab:dataset_summary}"
        "\n"
        r"\begin{tabular}{lr}"
        "\n"
        r"\toprule"
        "\n"
        r"\textbf{Property} & \textbf{Value} \\"
        "\n"
        r"\midrule"
        "\n"
        f"Patients & {metrics.n_patients} \\\\\n"
        f"Total studies & {metrics.n_studies} \\\\\n"
        f"Non-empty segmentations & {metrics.n_non_empty} \\\\\n"
        f"Empty segmentations & {len(metrics.empty_studies)} \\\\\n"
        r"\midrule"
        "\n"
        f"Studies/patient (mean $\\pm$ std) & ${np.mean(spp):.1f} \\pm {np.std(spp):.1f}$ \\\\\n"
        f"Studies/patient (range) & ${int(spp.min())}$--${int(spp.max())}$ \\\\\n"
        r"\midrule"
        "\n"
        f"Volume, mm\\textsuperscript{{3}} (mean $\\pm$ std) & "
        f"${np.mean(vol):,.0f} \\pm {np.std(vol):,.0f}$ \\\\\n"
        f"Volume, mm\\textsuperscript{{3}} (median [IQR]) & "
        f"${np.median(vol):,.0f}$ [${np.percentile(vol, 25):,.0f}$--${np.percentile(vol, 75):,.0f}$] \\\\\n"
        f"Volume, mm\\textsuperscript{{3}} (range) & "
        f"${np.min(vol):,.0f}$--${np.max(vol):,.0f}$ \\\\\n"
        r"\bottomrule"
        "\n"
        r"\end{tabular}"
        "\n"
        r"\end{table}"
        "\n"
    )
    (output_dir / "dataset_summary.tex").write_text(summary)

    # ── Table 2: Per-patient summary ──
    rows = []
    for pid, pm in sorted(metrics.patients.items()):
        non_empty = [s for s in pm.studies if not s.is_empty]
        has_empty = any(s.is_empty for s in pm.studies)
        if non_empty:
            vmin = min(s.total_volume for s in non_empty)
            vmax = max(s.total_volume for s in non_empty)
            vol_range = f"${vmin:,.0f}$--${vmax:,.0f}$"
        else:
            vol_range = "---"
        rows.append(
            f"        {pid} & {pm.n_studies} & {vol_range} & "
            f"{'Yes' if has_empty else 'No'} \\\\\n"
        )

    per_patient = (
        r"\begin{table}[htbp]"
        "\n"
        r"\centering"
        "\n"
        r"\caption{Per-patient summary.}"
        "\n"
        r"\label{tab:per_patient}"
        "\n"
        r"\begin{tabular}{lccc}"
        "\n"
        r"\toprule"
        "\n"
        r"\textbf{Patient} & \textbf{Studies} & \textbf{Volume range (mm\textsuperscript{3})} & \textbf{Empty segs} \\"
        "\n"
        r"\midrule"
        "\n" + "".join(rows) + r"\bottomrule"
        "\n"
        r"\end{tabular}"
        "\n"
        r"\end{table}"
        "\n"
    )
    (output_dir / "per_patient_summary.tex").write_text(per_patient)

    logger.info("Exported LaTeX tables to %s", output_dir)


# ── Orchestrator ─────────────────────────────────────────────────────────────


def export_all(metrics: DatasetMetrics, output_dir: Path) -> None:
    """Export JSON metrics and LaTeX tables.

    Args:
        metrics: Computed dataset metrics.
        output_dir: Top-level analysis output directory.
    """
    export_json(metrics, output_dir / "dataset_analysis.json")
    export_latex_tables(metrics, output_dir / "tables")
