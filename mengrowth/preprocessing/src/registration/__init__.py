"""Registration preprocessing module.

This module provides image registration capabilities for the MenGrowth preprocessing pipeline,
including multi-modal coregistration, atlas registration, and longitudinal registration.

Supports both nipype (default) and antspyx registration engines.
"""

from mengrowth.preprocessing.src.registration.base import BaseRegistrator
from mengrowth.preprocessing.src.registration.multi_modal_coregistration import MultiModalCoregistration
from mengrowth.preprocessing.src.registration.intra_study_to_atlas import IntraStudyToAtlas
from mengrowth.preprocessing.src.registration.antspyx_multi_modal_coregistration import AntsPyXMultiModalCoregistration
from mengrowth.preprocessing.src.registration.antspyx_intra_study_to_atlas import AntsPyXIntraStudyToAtlas
from mengrowth.preprocessing.src.registration.factory import (
    create_multi_modal_coregistration,
    create_intra_study_to_atlas
)
from mengrowth.preprocessing.src.registration.constants import (
    DEFAULT_REGISTRATION_ENGINE,
    VALID_ENGINES
)
from mengrowth.preprocessing.src.registration.reference_selection import (
    ReferenceSelector,
    ReferenceSelectionConfig,
    compute_jacobian_statistics,
    validate_registration_quality
)

__all__ = [
    "BaseRegistrator",
    "MultiModalCoregistration",
    "IntraStudyToAtlas",
    "AntsPyXMultiModalCoregistration",
    "AntsPyXIntraStudyToAtlas",
    "create_multi_modal_coregistration",
    "create_intra_study_to_atlas",
    "DEFAULT_REGISTRATION_ENGINE",
    "VALID_ENGINES",
    "ReferenceSelector",
    "ReferenceSelectionConfig",
    "compute_jacobian_statistics",
    "validate_registration_quality",
]
