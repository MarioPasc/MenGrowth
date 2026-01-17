"""Quality analysis module for MRI dataset quality assessment.

This module provides tools for analyzing MRI dataset quality metrics,
including spatial properties, intensity statistics, and generating visualizations.

It also provides QCManager for per-step quality control during preprocessing,
SNRCNRCalculator for signal/contrast-to-noise ratio metrics, and
mask comparison utilities for longitudinal mask consistency analysis.
"""

from mengrowth.preprocessing.quality_analysis.analyzer import QualityAnalyzer
from mengrowth.preprocessing.quality_analysis.visualize import QualityVisualizer
from mengrowth.preprocessing.quality_analysis.qc_manager import QCManager
from mengrowth.preprocessing.quality_analysis.snr_cnr_metrics import SNRCNRCalculator
from mengrowth.preprocessing.quality_analysis.mask_comparison import (
    compute_mask_comparison_metrics,
    compute_all_mask_comparisons,
    summarize_mask_comparisons,
)
from mengrowth.preprocessing.quality_analysis.html_report import (
    HTMLReportGenerator,
    generate_qc_report,
)

__all__ = [
    "QualityAnalyzer",
    "QualityVisualizer",
    "QCManager",
    "SNRCNRCalculator",
    "compute_mask_comparison_metrics",
    "compute_all_mask_comparisons",
    "summarize_mask_comparisons",
    "HTMLReportGenerator",
    "generate_qc_report",
]
