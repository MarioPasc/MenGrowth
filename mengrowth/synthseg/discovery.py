"""Patient / study discovery for the SynthSeg pipeline.

The preprocessed dataset is laid out as::

    {input_root}/
        MenGrowth-{XXXX}/                    # patient
            MenGrowth-{XXXX}-{YYY}/          # study timepoint
                t1n.nii.gz                   # SynthSeg input
                ...

Unlike the BraTS segmentation pipeline, SynthSeg operates on a single
modality (``t1n`` by default), so we do not validate the other modalities'
presence. Studies missing the requested modality are flagged but not
dropped here — the runner handles them gracefully at execution time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

from mengrowth.synthseg.config import SynthSegConfig

logger = logging.getLogger(__name__)


@dataclass
class StudyInfo:
    """A discovered study-timepoint.

    Attributes
    ----------
    patient_id : str
        e.g. ``"MenGrowth-0017"``.
    study_id : str
        e.g. ``"MenGrowth-0017-002"``.
    study_path : Path
        Absolute path to the study directory.
    has_input_modality : bool
        Whether the configured input modality file exists.
    input_path : Path
        Expected path of the input modality NIfTI (may not exist; check
        :attr:`has_input_modality`).
    """

    patient_id: str = ""
    study_id: str = ""
    study_path: Path = field(default_factory=Path)
    has_input_modality: bool = False
    input_path: Path = field(default_factory=Path)


def _iter_patient_dirs(input_root: Path) -> List[Path]:
    """Return sorted patient directories under ``input_root``.

    Excludes temp directories. Intentionally does not descend; each returned
    path should match ``MenGrowth-XXXX``.
    """
    if not input_root.exists():
        raise FileNotFoundError(f"input_root does not exist: {input_root}")
    return sorted(
        d
        for d in input_root.iterdir()
        if d.is_dir()
        and d.name.startswith("MenGrowth-")
        and not d.name.startswith("_temp_")
        # Patient dirs have exactly one dash-separator after the project tag;
        # study dirs (should they ever appear here) would have two.
        and d.name.count("-") == 1
    )


def discover_patients(
    input_root: Union[str, Path],
) -> List[str]:
    """Return sorted patient IDs under ``input_root``.

    Used by the SLURM launcher to size the array job.

    Parameters
    ----------
    input_root : str | Path
        Root of the preprocessed dataset.

    Returns
    -------
    list[str]
        Sorted patient IDs (directory names).
    """
    root = Path(input_root)
    patients = [d.name for d in _iter_patient_dirs(root)]
    logger.info("Discovered %d patients in %s", len(patients), root)
    return patients


def discover_studies(
    input_root: Union[str, Path],
    config: SynthSegConfig,
    patient_filter: Optional[str] = None,
) -> List[StudyInfo]:
    """Walk ``input_root`` and return one :class:`StudyInfo` per study.

    Parameters
    ----------
    input_root : str | Path
        Root of the preprocessed dataset.
    config : SynthSegConfig
        Supplies the input modality whose presence is checked per study.
    patient_filter : str, optional
        If set, restrict discovery to this patient ID.

    Returns
    -------
    list[StudyInfo]
        Sorted by ``(patient_id, study_id)``.
    """
    root = Path(input_root)
    patient_dirs = _iter_patient_dirs(root)
    if patient_filter:
        patient_dirs = [d for d in patient_dirs if d.name == patient_filter]
        if not patient_dirs:
            logger.warning("Patient %s not found in %s", patient_filter, root)

    modality_filename = f"{config.input_modality}.nii.gz"
    studies: List[StudyInfo] = []

    for patient_dir in patient_dirs:
        study_dirs = sorted(
            d
            for d in patient_dir.iterdir()
            if d.is_dir()
            and d.name.startswith(f"{patient_dir.name}-")
            and not d.name.startswith("_temp_")
        )

        for study_dir in study_dirs:
            input_path = study_dir / modality_filename
            info = StudyInfo(
                patient_id=patient_dir.name,
                study_id=study_dir.name,
                study_path=study_dir,
                has_input_modality=input_path.exists(),
                input_path=input_path,
            )
            studies.append(info)

    total = len(studies)
    with_input = sum(1 for s in studies if s.has_input_modality)
    logger.info(
        "Discovered %d studies (%d with %s)",
        total,
        with_input,
        modality_filename,
    )
    return studies


def studies_for_patient(
    input_root: Union[str, Path],
    config: SynthSegConfig,
    patient_id: str,
) -> List[StudyInfo]:
    """Return studies for a single patient, sorted by ``study_id``."""
    return discover_studies(input_root, config, patient_filter=patient_id)


def summarize_discovery(studies: List[StudyInfo]) -> Tuple[int, int]:
    """Return ``(total_studies, studies_with_input)`` for logging."""
    total = len(studies)
    with_input = sum(1 for s in studies if s.has_input_modality)
    return total, with_input
