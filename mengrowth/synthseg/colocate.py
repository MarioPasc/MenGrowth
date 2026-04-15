"""Optional helper: colocate SynthSeg parcellations into preprocessed study dirs.

The Phase 1 design intentionally writes SynthSeg outputs to a separate
``synthseg_output_root``; the preprocessed modality tree under
``{input_root}/MenGrowth-XXXX/MenGrowth-XXXX-YYY/`` is left untouched. This
module is an opt-in convenience for workflows that prefer the parcellation
mask sitting alongside ``t1n.nii.gz`` etc. Only the parcellation is
colocated (the QC and volumes CSVs stay external — they're cohort-level
artifacts).

Symlinks are created as *relative* paths so they survive if the containing
drive (e.g. the portable SSD hosting both trees) is remounted at a different
mount point.
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal

from mengrowth.synthseg.config import SynthSegConfig
from mengrowth.synthseg.discovery import discover_studies

logger = logging.getLogger(__name__)

ColocateMode = Literal["symlink", "copy"]


@dataclass
class StudyColocateResult:
    patient_id: str
    study_id: str
    status: str  # "linked" | "copied" | "already_present" | "missing_source" | "failed"
    error: str = ""


@dataclass
class ColocateReport:
    results: List[StudyColocateResult] = field(default_factory=list)
    mode: ColocateMode = "symlink"

    @property
    def n_linked(self) -> int:
        return sum(1 for r in self.results if r.status == "linked")

    @property
    def n_copied(self) -> int:
        return sum(1 for r in self.results if r.status == "copied")

    @property
    def n_already_present(self) -> int:
        return sum(1 for r in self.results if r.status == "already_present")

    @property
    def n_missing_source(self) -> int:
        return sum(1 for r in self.results if r.status == "missing_source")

    @property
    def n_failed(self) -> int:
        return sum(1 for r in self.results if r.status == "failed")

    def exit_code(self) -> int:
        return 2 if self.n_failed > 0 else 0


def _colocate_one_study(
    patient_id: str,
    study_id: str,
    source_parc: Path,
    dest_parc: Path,
    mode: ColocateMode,
) -> StudyColocateResult:
    if not source_parc.exists():
        return StudyColocateResult(
            patient_id=patient_id,
            study_id=study_id,
            status="missing_source",
            error=f"no parc at {source_parc}",
        )

    if dest_parc.exists() or dest_parc.is_symlink():
        return StudyColocateResult(
            patient_id=patient_id, study_id=study_id, status="already_present"
        )

    dest_parc.parent.mkdir(parents=True, exist_ok=True)

    try:
        if mode == "symlink":
            rel = os.path.relpath(source_parc, start=dest_parc.parent)
            os.symlink(rel, dest_parc)
            return StudyColocateResult(
                patient_id=patient_id, study_id=study_id, status="linked"
            )
        if mode == "copy":
            shutil.copy2(source_parc, dest_parc)
            return StudyColocateResult(
                patient_id=patient_id, study_id=study_id, status="copied"
            )
        raise ValueError(f"unknown mode {mode!r}")
    except OSError as e:
        return StudyColocateResult(
            patient_id=patient_id,
            study_id=study_id,
            status="failed",
            error=str(e),
        )


def colocate_synthseg_outputs(
    cfg: SynthSegConfig, mode: ColocateMode = "symlink"
) -> ColocateReport:
    """Place (or link) the parcellation file alongside the preprocessed modalities.

    Parameters
    ----------
    cfg : SynthSegConfig
        Supplies ``input_root`` (preprocessed modalities), ``output_root``
        (SynthSeg outputs), and ``parcellation_filename``.
    mode : "symlink" | "copy"
        Link (default — zero extra disk) or copy.

    Returns
    -------
    ColocateReport
    """
    report = ColocateReport(mode=mode)
    studies = discover_studies(cfg.input_root, cfg)

    for s in studies:
        source = Path(cfg.output_root) / s.patient_id / s.study_id / cfg.parcellation_filename
        dest = Path(cfg.input_root) / s.patient_id / s.study_id / cfg.parcellation_filename
        report.results.append(
            _colocate_one_study(s.patient_id, s.study_id, source, dest, mode)
        )

    logger.info(
        "Colocate summary (mode=%s): linked=%d, copied=%d, already_present=%d, "
        "missing_source=%d, failed=%d",
        mode,
        report.n_linked,
        report.n_copied,
        report.n_already_present,
        report.n_missing_source,
        report.n_failed,
    )
    return report
