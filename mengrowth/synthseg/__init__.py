"""MenGrowth SynthSeg parcellation pipeline.

Runs SynthSeg (Billot et al., Medical Image Analysis 2023) with the
``--robust`` flag on preprocessed T1n volumes to produce per-study brain
parcellations, per-region volume tables, and QC scores.
"""

from mengrowth.synthseg.config import SynthSegConfig, load_synthseg_config
from mengrowth.synthseg.discovery import (
    StudyInfo,
    discover_patients,
    discover_studies,
    studies_for_patient,
)
from mengrowth.synthseg.exceptions import (
    SynthSegConfigError,
    SynthSegError,
    SynthSegInputError,
    SynthSegOutputError,
    SynthSegRuntimeError,
)
from mengrowth.synthseg.primitives import (
    SYNTHSEG_LABEL_MAP,
    build_command,
    parse_qc_csv,
    parse_qc_value,
    parse_volumes_csv,
    parse_volumes_row,
)
from mengrowth.synthseg.runner import (
    PatientResult,
    StudyResult,
    run_patient,
    run_study,
)

__all__ = [
    "SYNTHSEG_LABEL_MAP",
    "PatientResult",
    "StudyInfo",
    "StudyResult",
    "SynthSegConfig",
    "SynthSegConfigError",
    "SynthSegError",
    "SynthSegInputError",
    "SynthSegOutputError",
    "SynthSegRuntimeError",
    "build_command",
    "discover_patients",
    "discover_studies",
    "load_synthseg_config",
    "parse_qc_csv",
    "parse_qc_value",
    "parse_volumes_csv",
    "parse_volumes_row",
    "run_patient",
    "run_study",
    "studies_for_patient",
]
