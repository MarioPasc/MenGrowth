"""Registration preprocessing module.

This module provides image registration capabilities for the MenGrowth preprocessing pipeline,
including multi-modal coregistration, atlas registration, and longitudinal registration.
"""

from mengrowth.preprocessing.src.registration.base import BaseRegistrator
from mengrowth.preprocessing.src.registration.multi_modal_coregistration import MultiModalCoregistration
from mengrowth.preprocessing.src.registration.intra_study_to_atlas import IntraStudyToAtlas

__all__ = [
    "BaseRegistrator",
    "MultiModalCoregistration",
    "IntraStudyToAtlas",
]
