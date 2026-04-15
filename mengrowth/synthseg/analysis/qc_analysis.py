"""Analysis 1 — QC score vs. original T1n acquisition spacing.

Fits a patient-level random-intercept linear model

    q ~ max_spacing + (1 | patient_id)

via :func:`statsmodels.formula.api.mixedlm` and reports the fixed-effect
slope, its p-value, and an approximate marginal R² (Nakagawa & Schielzeth,
2013). Produces the two plan-§4.3 figures.

If the model fails to converge, the slope/p-value are returned as NaN
and a warning is logged — the figures are still generated from the raw
scatter so the report degrades gracefully.

References
----------
Nakagawa, S., & Schielzeth, H. "A general and simple method for obtaining
R² from generalized linear mixed-effects models." *Methods in Ecology and
Evolution*, 4(2), 133–142 (2013).
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from mengrowth.synthseg.analysis import figures
from mengrowth.synthseg.analysis.config import AnalysisConfig

logger = logging.getLogger(__name__)


@dataclass
class QCAnalysisResult:
    """Summary of Analysis 1."""

    n_studies: int = 0
    n_patients: int = 0
    beta_spacing: float = float("nan")
    pvalue_spacing: float = float("nan")
    intercept: float = float("nan")
    marginal_r2: float = float("nan")
    conditional_r2: float = float("nan")
    converged: bool = False
    model_summary: str = ""
    pearson_r_in_plane: float = float("nan")
    pearson_r_anisotropy: float = float("nan")
    figure_paths: List[Path] = field(default_factory=list)
    message: str = ""


def _merge_qc_with_t1n_meta(
    qc_df: pd.DataFrame, meta_df: pd.DataFrame
) -> pd.DataFrame:
    """Inner-join QC (one row per study) with the T1n metadata row."""
    t1n = meta_df[meta_df["sequence"].str.lower() == "t1n"].copy()
    merged = qc_df.merge(
        t1n[
            [
                "patient_id",
                "study_id",
                "max_spacing",
                "min_spacing",
                "anisotropy_ratio",
                "in_plane_resolution",
            ]
        ],
        on=["patient_id", "study_id"],
        how="inner",
    )
    return merged


def _fit_mixedlm(df: pd.DataFrame) -> Dict[str, float]:
    """Fit ``qc_score ~ max_spacing`` with a patient random intercept.

    Returns a dict with keys ``beta, pvalue, intercept, marginal_r2,
    conditional_r2, converged, summary, n_groups``. On failure, numeric
    entries are NaN and ``converged`` is False.
    """
    import statsmodels.formula.api as smf  # lazy import — heavy dep

    out: Dict[str, float] = {
        "beta": float("nan"),
        "pvalue": float("nan"),
        "intercept": float("nan"),
        "marginal_r2": float("nan"),
        "conditional_r2": float("nan"),
        "converged": False,
        "summary": "",
        "n_groups": float(df["patient_id"].nunique()),
    }

    if df["patient_id"].nunique() < 2 or len(df) < 3:
        out["summary"] = (
            f"mixedlm not fit: need >=2 patients and >=3 rows "
            f"(got {df['patient_id'].nunique()} patients, {len(df)} rows)"
        )
        return out

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = smf.mixedlm(
                "qc_score ~ max_spacing", df, groups=df["patient_id"]
            )
            fit = model.fit(reml=True, method="lbfgs")

        beta = float(fit.fe_params.get("max_spacing", float("nan")))
        pvalue = float(fit.pvalues.get("max_spacing", float("nan")))
        intercept = float(fit.fe_params.get("Intercept", float("nan")))

        # Nakagawa & Schielzeth marginal / conditional R² for linear mixed
        # models with a random intercept:
        #   marginal    = var(fixed_pred) / (var(fixed_pred) + var_u + var_e)
        #   conditional = (var(fixed_pred) + var_u) / (denom)
        fixed_pred = (
            intercept + beta * df["max_spacing"].to_numpy()
            if not np.isnan(beta)
            else np.array([])
        )
        var_f = float(np.var(fixed_pred, ddof=0)) if fixed_pred.size else 0.0
        var_u = float(fit.cov_re.iloc[0, 0]) if fit.cov_re.size else 0.0
        var_e = float(fit.scale)
        denom = var_f + var_u + var_e
        if denom > 0:
            out["marginal_r2"] = var_f / denom
            out["conditional_r2"] = (var_f + var_u) / denom

        out["beta"] = beta
        out["pvalue"] = pvalue
        out["intercept"] = intercept
        out["converged"] = bool(getattr(fit, "converged", True))
        out["summary"] = str(fit.summary())
    except Exception as exc:  # pragma: no cover — numerical failure branch
        logger.warning("mixedlm fit failed: %s", exc)
        out["summary"] = f"mixedlm fit failed: {exc}"

    return out


def _pearson(df: pd.DataFrame, col: str) -> float:
    """Return Pearson r between ``qc_score`` and ``col`` (NaN-safe)."""
    sub = df[["qc_score", col]].dropna()
    if len(sub) < 3 or sub["qc_score"].std() == 0 or sub[col].std() == 0:
        return float("nan")
    return float(sub["qc_score"].corr(sub[col]))


def run_qc_analysis(
    qc_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    cfg: AnalysisConfig,
    out_dir: Optional[Path] = None,
) -> QCAnalysisResult:
    """Run Analysis 1 end-to-end and return the summary + figure paths."""
    if out_dir is None:
        out_dir = Path(cfg.output_dir) / "qc_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    if qc_df.empty or meta_df.empty:
        logger.warning(
            "run_qc_analysis: empty qc (%d) or meta (%d) DataFrame",
            len(qc_df),
            len(meta_df),
        )
        return QCAnalysisResult(message="empty input DataFrame(s)")

    merged = _merge_qc_with_t1n_meta(qc_df, meta_df)
    if merged.empty:
        return QCAnalysisResult(
            message="no (patient, study) matches between QC and T1n metadata"
        )

    merged.to_csv(out_dir / "qc_vs_spacing_merged.csv", index=False)

    fit = _fit_mixedlm(merged)

    if fit["summary"]:
        (out_dir / "mixedlm_summary.txt").write_text(fit["summary"])

    # Figures — always produced (regplot is independent of the mixedlm fit).
    fig_paths: List[Path] = [
        figures.plot_qc_vs_spacing_scatter(merged, out_dir, cfg.figure_dpi),
        figures.plot_qc_vs_spacing_boxplot(merged, out_dir, cfg.figure_dpi),
    ]

    result = QCAnalysisResult(
        n_studies=int(len(merged)),
        n_patients=int(merged["patient_id"].nunique()),
        beta_spacing=fit["beta"],
        pvalue_spacing=fit["pvalue"],
        intercept=fit["intercept"],
        marginal_r2=fit["marginal_r2"],
        conditional_r2=fit["conditional_r2"],
        converged=bool(fit["converged"]),
        model_summary=fit["summary"],
        pearson_r_in_plane=_pearson(merged, "in_plane_resolution"),
        pearson_r_anisotropy=_pearson(merged, "anisotropy_ratio"),
        figure_paths=fig_paths,
    )
    logger.info(
        "Analysis 1: N=%d studies, %d patients, beta=%.4f p=%.4f R2_marg=%.3f",
        result.n_studies,
        result.n_patients,
        result.beta_spacing,
        result.pvalue_spacing,
        result.marginal_r2,
    )
    return result
