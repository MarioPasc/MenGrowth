"""HTML + Markdown report assembly for the Phase 2 SynthSeg QC analysis.

Mirrors the f-string-template pattern used by
``mengrowth/preprocessing/quality_analysis/html_report.py``: a module-level
:data:`_HTML_TEMPLATE` carries structural boilerplate and a
``<!-- SECTIONS -->`` placeholder which is replaced with the joined
per-analysis HTML fragments. No Jinja2 is used.

Figures are embedded via ``<img src="figures/<name>.png">`` — ``figures/``
is a sibling of the HTML file and contains relative-path copies (or the
original paths if they already live under that directory).
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

from mengrowth.synthseg.analysis.config import AnalysisConfig
from mengrowth.synthseg.analysis.longitudinal import AcceptanceResult
from mengrowth.synthseg.analysis.qc_analysis import QCAnalysisResult

logger = logging.getLogger(__name__)


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
 body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
         Helvetica, Arial, sans-serif; margin: 2em auto; max-width: 1100px;
         color: #222; }}
 h1 {{ border-bottom: 2px solid #333; padding-bottom: 0.2em; }}
 h2 {{ margin-top: 2em; border-bottom: 1px solid #aaa; padding-bottom: 0.15em; }}
 h3 {{ margin-top: 1.2em; color: #444; }}
 table {{ border-collapse: collapse; margin: 0.5em 0; font-size: 0.9em; }}
 th, td {{ border: 1px solid #ccc; padding: 0.35em 0.7em; text-align: right; }}
 th {{ background: #f3f3f3; }}
 td.key {{ text-align: left; font-weight: 600; }}
 img {{ max-width: 100%; height: auto; margin: 0.5em 0; border: 1px solid #eee; }}
 .pass {{ color: #126e2c; font-weight: 700; }}
 .fail {{ color: #a4161a; font-weight: 700; }}
 .warn {{ color: #b56200; font-weight: 600; }}
 .caveat {{ background: #fff7e0; border-left: 4px solid #e0b020;
            padding: 0.6em 1em; margin: 1em 0; }}
 .kv td {{ text-align: left; }}
</style>
</head>
<body>
<h1>{title}</h1>
<p><em>Generated {timestamp}. Plan reference:
<code>docs/synthseg/synthseg-parcellation-qc-plan.md</code> §4–§7.</em></p>
<!-- SECTIONS -->
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fmt(x: float, nd: int = 4) -> str:
    if x is None:
        return "—"
    try:
        if pd.isna(x):
            return "—"
    except (TypeError, ValueError):
        pass
    return f"{x:.{nd}f}"


def _copy_figure(
    src: Optional[Path], dest_dir: Path
) -> Optional[str]:
    """Copy ``src`` into ``dest_dir`` and return the relative HTML src.

    Returns ``None`` if ``src`` is None or does not exist.
    """
    if src is None:
        return None
    src = Path(src)
    if not src.exists():
        return None
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    try:
        if dest.resolve() != src.resolve():
            shutil.copy2(src, dest)
    except OSError as e:
        logger.warning("Could not copy %s → %s: %s", src, dest, e)
    return f"figures/{src.name}"


def _kv_table(rows: Iterable[tuple]) -> str:
    body = "\n".join(
        f'<tr><td class="key">{k}</td><td>{v}</td></tr>' for k, v in rows
    )
    return f'<table class="kv">{body}</table>'


def _df_as_html(df: pd.DataFrame, float_fmt: str = "{:.4f}") -> str:
    if df is None or df.empty:
        return "<p><em>(no rows)</em></p>"
    return df.to_html(
        index=False,
        float_format=float_fmt.format,
        border=0,
        classes="data",
    )


# ---------------------------------------------------------------------------
# Report assembler
# ---------------------------------------------------------------------------


@dataclass
class SynthSegAnalysisReport:
    """Accumulates HTML fragments, writes the final HTML + Markdown files."""

    cfg: AnalysisConfig
    title: str = "SynthSeg Phase 2 — Statistical QC Analysis"
    sections: List[str] = field(default_factory=list)
    md_lines: List[str] = field(default_factory=list)
    acceptance: Optional[AcceptanceResult] = None

    # ------- cohort -------

    def add_cohort_summary(
        self,
        volumes_df: pd.DataFrame,
        qc_df: pd.DataFrame,
        meta_df: pd.DataFrame,
    ) -> None:
        n_studies = (
            volumes_df[["patient_id", "study_id"]].drop_duplicates().shape[0]
            if not volumes_df.empty
            else 0
        )
        n_patients = (
            volumes_df["patient_id"].nunique() if not volumes_df.empty else 0
        )
        n_regions = (
            volumes_df["region_name"].nunique() if not volumes_df.empty else 0
        )
        n_qc = len(qc_df)
        n_meta = len(meta_df)
        modalities = (
            ", ".join(sorted(meta_df["sequence"].unique()))
            if not meta_df.empty
            else "—"
        )

        rows = [
            ("Patients", n_patients),
            ("Studies", n_studies),
            ("Canonical regions", n_regions),
            ("QC rows", n_qc),
            ("Metadata rows", n_meta),
            ("Modalities in metadata", modalities),
            ("Tumor-proximity threshold δ", f"{self.cfg.tumor_proximity_threshold_mm:g} mm"),
            ("CV acceptance threshold", _fmt(self.cfg.cv_acceptance_threshold, 3)),
        ]
        self.sections.append(
            f'<div class="section"><h2>Cohort</h2>{_kv_table(rows)}</div>'
        )

    # ------- Analysis 1 -------

    def add_qc_analysis(
        self, result: QCAnalysisResult, figures_dir: Path
    ) -> None:
        imgs = "\n".join(
            f'<img src="{rel}" alt="">'
            for rel in (
                _copy_figure(p, figures_dir) for p in result.figure_paths
            )
            if rel is not None
        )
        rows = [
            ("N (studies)", result.n_studies),
            ("N (patients)", result.n_patients),
            ("β<sub>max_spacing</sub>", _fmt(result.beta_spacing)),
            ("p-value", _fmt(result.pvalue_spacing)),
            ("Intercept", _fmt(result.intercept)),
            ("Marginal R²", _fmt(result.marginal_r2, 3)),
            ("Conditional R²", _fmt(result.conditional_r2, 3)),
            ("mixedlm converged", "yes" if result.converged else "no"),
            ("Pearson r (in-plane res.)", _fmt(result.pearson_r_in_plane, 3)),
            ("Pearson r (anisotropy)", _fmt(result.pearson_r_anisotropy, 3)),
        ]
        caveat = (
            f'<div class="caveat">{result.message}</div>'
            if result.message
            else ""
        )
        self.sections.append(
            f"""
<div class="section">
<h2>Analysis 1 — QC vs. original T1n acquisition spacing</h2>
{caveat}
<p>Random-intercept linear model <code>qc_score ~ max_spacing + (1 | patient_id)</code>.</p>
{_kv_table(rows)}
{imgs}
</div>
"""
        )

    # ------- Analysis 2 -------

    def add_longitudinal(
        self,
        cv_df: pd.DataFrame,
        contralateral_df: pd.DataFrame,
        acceptance: AcceptanceResult,
        figure_paths: Iterable[Optional[Path]],
        figures_dir: Path,
        proximity_sources: Optional[pd.Series] = None,
    ) -> None:
        self.acceptance = acceptance

        src_counts = {}
        if proximity_sources is not None and not proximity_sources.empty:
            src_counts = proximity_sources.value_counts().to_dict()
        proximity_line = ", ".join(
            f"{k}: {v}" for k, v in sorted(src_counts.items())
        ) or "—"

        verdict_cls = "pass" if acceptance.passed else "fail"
        verdict_text = (
            f"PASS (median CV {acceptance.median_cv:.4f} "
            f"< threshold {acceptance.threshold:g})"
            if acceptance.passed
            else f"FAIL (median CV {_fmt(acceptance.median_cv)} "
                 f"≥ threshold {acceptance.threshold:g})"
        )
        rows = [
            ("Regions evaluated", ", ".join(acceptance.regions_used)),
            ("Values used (N)", acceptance.n_values),
            ("Median CV (tumor-distant deep GM)", _fmt(acceptance.median_cv)),
            ("Mean CV (tumor-distant deep GM)", _fmt(acceptance.mean_cv)),
            ("Threshold", _fmt(acceptance.threshold, 3)),
            ("Verdict", f'<span class="{verdict_cls}">{verdict_text}</span>'),
            ("Proximity source counts", proximity_line),
        ]
        if acceptance.note:
            rows.append(("Note", acceptance.note))

        imgs = "\n".join(
            f'<img src="{rel}" alt="">'
            for rel in (_copy_figure(p, figures_dir) for p in figure_paths)
            if rel is not None
        )

        # Region summary table (median CV per region, tumor-distant + all).
        summary_cv = pd.DataFrame()
        if not cv_df.empty:
            summary_cv = (
                cv_df.groupby("region_name")["cv"]
                .agg(["median", "mean", "std", "count"])
                .reset_index()
                .rename(columns={"count": "n_patients"})
                .sort_values("median", ascending=False)
            )

        self.sections.append(
            f"""
<div class="section">
<h2>Analysis 2 — Longitudinal within-subject CV</h2>
<h3>Headline acceptance (plan §7)</h3>
{_kv_table(rows)}
{imgs}
<h3>Per-region CV summary (all patients, all proximity classes)</h3>
{_df_as_html(summary_cv)}
"""
            + (
                f"<h3>Contralateral δCV (ipsi − contra)</h3>"
                f"{_df_as_html(contralateral_df)}</div>"
                if not contralateral_df.empty
                else "</div>"
            )
        )

    # ------- Analysis 3 -------

    def add_roi_metrics(
        self,
        metrics_df: pd.DataFrame,
        stability_df: pd.DataFrame,
        figure_paths: Iterable[Optional[Path]],
        figures_dir: Path,
    ) -> None:
        imgs = "\n".join(
            f'<img src="{rel}" alt="">'
            for rel in (_copy_figure(p, figures_dir) for p in figure_paths)
            if rel is not None
        )

        per_mod = pd.DataFrame()
        if not metrics_df.empty:
            per_mod = (
                metrics_df.groupby(["modality", "region_name"])
                .agg(
                    within_cv_median=("within_region_cv", "median"),
                    boundary_grad_median=("boundary_gradient", "median"),
                    n=("within_region_cv", "count"),
                )
                .reset_index()
                .sort_values(["modality", "region_name"])
            )

        self.sections.append(
            f"""
<div class="section">
<h2>Analysis 3 — Cross-modal ROI metrics</h2>
<p>T1n parcellation re-applied to each modality to probe preprocessing
quality without re-running SynthSeg. Restricted to the configured
reliable regions (subcortical structures).</p>
{imgs}
<h3>Per-modality × region summary</h3>
{_df_as_html(per_mod)}
<h3>Longitudinal intensity CV (mean-intensity drift across timepoints)</h3>
{_df_as_html(stability_df.head(60))}
</div>
"""
        )

    # ------- emit -------

    def generate(self, html_path: Path, md_path: Path) -> None:
        """Write the HTML and a short Markdown sibling."""
        html_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.parent.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        body = self._base_html(timestamp).replace(
            "<!-- SECTIONS -->", "\n".join(self.sections)
        )
        html_path.write_text(body, encoding="utf-8")

        md_lines: List[str] = [f"# {self.title}", "", f"_{timestamp}_", ""]
        if self.acceptance is not None:
            md_lines.append("## Acceptance (plan §7)")
            verdict = "PASS" if self.acceptance.passed else "FAIL"
            md_lines.append(
                f"- **Headline:** {verdict} — median tumor-distant deep-GM "
                f"CV = {_fmt(self.acceptance.median_cv)} "
                f"vs threshold {self.acceptance.threshold:g}"
            )
            md_lines.append(
                f"- N values used: {self.acceptance.n_values}, "
                f"regions: {', '.join(self.acceptance.regions_used)}"
            )
            if self.acceptance.note:
                md_lines.append(f"- Note: {self.acceptance.note}")
            md_lines.append("")
        md_lines.append(f"Full HTML report: `{html_path.name}`.")
        md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

        logger.info("Wrote report: %s (and %s)", html_path, md_path)

    def _base_html(self, timestamp: str) -> str:
        return _HTML_TEMPLATE.format(title=self.title, timestamp=timestamp)
