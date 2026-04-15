"""Phase 2 statistical QC analysis on SynthSeg parcellations.

See ``docs/synthseg/synthseg-parcellation-qc-plan.md`` §4–§7 for the full
specification. Three analyses are produced, each saving per-study CSVs and
publication-quality figures, plus an HTML + Markdown summary report.
"""

from mengrowth.synthseg.analysis.config import (
    AnalysisConfig,
    load_analysis_config,
)
from mengrowth.synthseg.analysis.pipeline import run_analysis

__all__ = ["AnalysisConfig", "load_analysis_config", "run_analysis"]
