"""Registration preprocessing module.

This module provides image registration capabilities for the MenGrowth preprocessing pipeline,
including multi-modal coregistration, atlas registration, and longitudinal registration.
"""

from mengrowth.preprocessing.src.registration.base import BaseRegistrator
from mengrowth.preprocessing.src.registration.multi_modal_coregistration import MultiModalCoregistration

__all__ = [
    "BaseRegistrator",
    "MultiModalCoregistration",
]
