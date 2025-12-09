"""Step 3: Intensity normalization.

This module provides intensity normalization operations for harmonizing MRI
intensities across subjects, scanners, and acquisition parameters. Normalization
is applied before resampling to ensure consistent intensity scales for
super-resolution or interpolation methods.
"""

from mengrowth.preprocessing.src.normalization.base import BaseNormalizer
from mengrowth.preprocessing.src.normalization.zscore import ZScoreNormalizer
from mengrowth.preprocessing.src.normalization.kde import KDENormalizer
from mengrowth.preprocessing.src.normalization.percentile_minmax import PercentileMinMaxNormalizer
from mengrowth.preprocessing.src.normalization.whitestripe import WhiteStripeNormalizer
from mengrowth.preprocessing.src.normalization.fcm import FCMNormalizer
from mengrowth.preprocessing.src.normalization.lsq import LSQNormalizer

__all__ = [
    "BaseNormalizer",
    "ZScoreNormalizer",
    "KDENormalizer",
    "PercentileMinMaxNormalizer",
    "WhiteStripeNormalizer",
    "FCMNormalizer",
    "LSQNormalizer",
]
