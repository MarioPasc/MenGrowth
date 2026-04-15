"""End-to-end orchestration of the Phase 2 SynthSeg QC analysis.

This module ties together :mod:`collector`, :mod:`qc_analysis`,
:mod:`longitudinal`, :mod:`region_metrics`, :mod:`figures`, and
:mod:`report`. It mirrors the CLI entry point — there is intentionally no
fancy state machine, just an ordered series of banners and persistence
calls.

Exit convention: :func:`run_analysis` returns 0 on full success and 2
when the plan-§7 acceptance criterion fails (the report is still written
in that case).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from mengrowth.synthseg.analysis import figures
from mengrowth.synthseg.analysis.collector import persist_cohort
from mengrowth.synthseg.analysis.config import AnalysisConfig
from mengrowth.synthseg.analysis.longitudinal import (
    AcceptanceResult,
    attach_proximity_to_cv,
    classify_regions_by_tumor_proximity,
    compute_contralateral_delta_cv,
    compute_within_subject_cv,
    evaluate_acceptance,
)
from mengrowth.synthseg.analysis.qc_analysis import run_qc_analysis
from mengrowth.synthseg.analysis.region_metrics import (
    compute_longitudinal_intensity_cv,
    compute_region_metrics,
)
from mengrowth.synthseg.analysis.report import SynthSegAnalysisReport

logger = logging.getLogger(__name__)


def _banner(msg: str) -> None:
    logger.info("=" * 72)
    logger.info(msg)
    logger.info("=" * 72)


def run_analysis(
    cfg: AnalysisConfig, skip_collect: bool = False
) -> int:
    """Run the full Phase 2 pipeline end-to-end.

    Parameters
    ----------
    cfg : AnalysisConfig
    skip_collect : bool, optional
        If True and prior cohort CSVs exist under ``{output_dir}/cohort``,
        reload them instead of re-parsing SynthSeg outputs.

    Returns
    -------
    int
        0 on success, 2 if the headline CV acceptance fails.
    """
    out_root = Path(cfg.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    cohort_dir = out_root / "cohort"
    qc_dir = out_root / "qc_analysis"
    long_dir = out_root / "longitudinal"
    roi_dir = out_root / "roi_metrics"
    figures_dir = out_root / "figures"
    for d in (cohort_dir, qc_dir, long_dir, roi_dir, figures_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Stage 1 — cohort collection
    # ------------------------------------------------------------------
    _banner("Stage 1 — cohort collection")
    if (
        skip_collect
        and (cohort_dir / "cohort_volumes.csv").exists()
        and (cohort_dir / "cohort_qc.csv").exists()
        and (cohort_dir / "cohort_metadata.csv").exists()
    ):
        volumes_df = pd.read_csv(cohort_dir / "cohort_volumes.csv")
        qc_df = pd.read_csv(cohort_dir / "cohort_qc.csv")
        meta_df = pd.read_csv(cohort_dir / "cohort_metadata.csv")
        logger.info(
            "Reused cohort CSVs: volumes=%d, qc=%d, meta=%d",
            len(volumes_df),
            len(qc_df),
            len(meta_df),
        )
    else:
        cohort = persist_cohort(cfg)
        volumes_df = cohort["volumes"]
        qc_df = cohort["qc"]
        meta_df = cohort["metadata"]

    # ------------------------------------------------------------------
    # Stage 2 — Analysis 1: QC vs. acquisition spacing
    # ------------------------------------------------------------------
    _banner("Stage 2 — Analysis 1: QC vs. original T1n spacing")
    qc_result = run_qc_analysis(qc_df, meta_df, cfg, out_dir=qc_dir)

    # ------------------------------------------------------------------
    # Stage 3 — Analysis 2: longitudinal CV
    # ------------------------------------------------------------------
    _banner("Stage 3 — Analysis 2: longitudinal within-subject CV")
    cv_df = compute_within_subject_cv(volumes_df)
    cv_df.to_csv(long_dir / "within_subject_cv.csv", index=False)

    proximity_df = classify_regions_by_tumor_proximity(cfg)
    proximity_df.to_csv(long_dir / "tumor_proximity.csv", index=False)

    cv_df_prox = attach_proximity_to_cv(cv_df, proximity_df)
    cv_df_prox.to_csv(long_dir / "within_subject_cv_with_proximity.csv", index=False)

    contralateral_df = compute_contralateral_delta_cv(cv_df, cfg)
    contralateral_df.to_csv(
        long_dir / "contralateral_delta_cv.csv", index=False
    )

    acceptance = evaluate_acceptance(
        cv_df_prox,
        cfg.deep_grey_matter_regions,
        cfg.cv_acceptance_threshold,
    )
    logger.info(
        "Acceptance: median CV=%.4f (N=%d) %s threshold %.3f",
        acceptance.median_cv,
        acceptance.n_values,
        "<" if acceptance.passed else ">=",
        acceptance.threshold,
    )

    long_figures: List[Optional[Path]] = []
    if not cv_df_prox.empty:
        long_figures.append(
            figures.plot_region_cv_violin(cv_df_prox, long_dir, cfg.figure_dpi)
        )
        long_figures.append(
            figures.plot_region_cv_heatmap(cv_df_prox, long_dir, cfg.figure_dpi)
        )
        long_figures.append(
            figures.plot_tumor_distant_mean_cv_bar(
                cv_df_prox,
                cfg.deep_grey_matter_regions,
                long_dir,
                cfg.figure_dpi,
                threshold=cfg.cv_acceptance_threshold,
            )
        )
    if not contralateral_df.empty:
        long_figures.append(
            figures.plot_contralateral_delta_cv(
                contralateral_df, long_dir, cfg.figure_dpi
            )
        )

    # ------------------------------------------------------------------
    # Stage 4 — Analysis 3: cross-modal ROI metrics
    # ------------------------------------------------------------------
    _banner("Stage 4 — Analysis 3: cross-modal ROI metrics")
    metrics_df = compute_region_metrics(cfg)
    metrics_df.to_csv(roi_dir / "region_metrics_long.csv", index=False)

    stability_df = compute_longitudinal_intensity_cv(metrics_df)
    stability_df.to_csv(roi_dir / "longitudinal_stability.csv", index=False)

    # Pivot-style splits for convenient downstream use.
    if not metrics_df.empty:
        metrics_df.pivot_table(
            index=["patient_id", "study_id", "region_name"],
            columns="modality",
            values="within_region_cv",
        ).to_csv(roi_dir / "within_region_cv.csv")
        metrics_df.pivot_table(
            index=["patient_id", "study_id", "region_name"],
            columns="modality",
            values="boundary_gradient",
        ).to_csv(roi_dir / "boundary_gradient.csv")

    roi_figures: List[Optional[Path]] = []
    if not metrics_df.empty:
        headline_region = cfg.reliable_regions[0]
        for mod in cfg.modalities_for_roi_analysis:
            roi_figures.append(
                figures.plot_roi_gradient_vs_spacing(
                    metrics_df,
                    meta_df,
                    modality=mod,
                    region_name=headline_region,
                    out_dir=roi_dir,
                    dpi=cfg.figure_dpi,
                )
            )
            roi_figures.append(
                figures.plot_roi_cv_vs_spacing(
                    metrics_df,
                    meta_df,
                    modality=mod,
                    region_name=headline_region,
                    out_dir=roi_dir,
                    dpi=cfg.figure_dpi,
                )
            )

    # ------------------------------------------------------------------
    # Stage 5 — report assembly
    # ------------------------------------------------------------------
    _banner("Stage 5 — report assembly")
    report = SynthSegAnalysisReport(cfg=cfg)
    report.add_cohort_summary(volumes_df, qc_df, meta_df)
    report.add_qc_analysis(qc_result, figures_dir)
    report.add_longitudinal(
        cv_df=cv_df_prox,
        contralateral_df=contralateral_df,
        acceptance=acceptance,
        figure_paths=long_figures,
        figures_dir=figures_dir,
        proximity_sources=proximity_df["proximity_source"]
        if not proximity_df.empty
        else None,
    )
    report.add_roi_metrics(
        metrics_df=metrics_df,
        stability_df=stability_df,
        figure_paths=roi_figures,
        figures_dir=figures_dir,
    )
    report.generate(out_root / "report.html", out_root / "summary_report.md")

    exit_code = 0 if acceptance.passed else 2
    logger.info(
        "Phase 2 analysis complete (exit_code=%d, acceptance=%s)",
        exit_code,
        "PASS" if acceptance.passed else "FAIL",
    )
    return exit_code
