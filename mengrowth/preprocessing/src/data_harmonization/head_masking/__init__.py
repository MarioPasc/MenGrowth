"""Head masking / background removal modules.

This package provides conservative background removal methods that set
air/background voxels to zero without eroding anatomical structures.

Available methods:
- ConservativeBackgroundRemover: Border-connected percentile method
- SELFBackgroundRemover: SELF algorithm for head-air separation
- OtsuForegroundRemover: Otsu-based foreground extraction
"""

from mengrowth.preprocessing.src.data_harmonization.head_masking.conservative import (
    ConservativeBackgroundRemover,
)
from mengrowth.preprocessing.src.data_harmonization.head_masking.self import (
    SELFBackgroundRemover,
)
from mengrowth.preprocessing.src.data_harmonization.head_masking.otsu_foreground import (
    OtsuForegroundRemover,
)

__all__ = [
    "ConservativeBackgroundRemover",
    "SELFBackgroundRemover",
    "OtsuForegroundRemover",
]
