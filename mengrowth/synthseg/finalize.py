"""Finalize Phase 1 SynthSeg outputs by renaming ``.tmp`` variants in place.

The runner writes outputs via the atomic temp-file pattern (see
:mod:`mengrowth.synthseg.runner`), then validates and renames. When a SLURM
array task is interrupted after SynthSeg has written the temp files but
before :func:`os.replace` runs, the final filenames are never materialised.

This module walks ``{output_root}/MenGrowth-XXXX/MenGrowth-XXXX-YYY/`` and,
for each study, attempts to commit any orphaned temp outputs provided they
pass the same validation the runner would have applied:

* Parcellation geometry matches the preprocessed T1n volume
  (``{input_root}/MenGrowth-XXXX/MenGrowth-XXXX-YYY/t1n.nii.gz``).
* QC score parses and is in ``[0, 1]``.
* Volumes CSV parses and is non-empty.

Expected / accepted temp filename patterns
------------------------------------------
Two variants are accepted because SynthSeg auto-appends ``.csv`` to ``--vol``
/ ``--qc`` arguments that don't already end in ``.csv``:

* Pythonic temp pattern (what the runner constructs via ``Path.with_suffix``):
  ``synthseg_parc.nii.tmp.nii.gz``, ``synthseg_volumes.csv.tmp``,
  ``synthseg_qc.csv.tmp``.
* As-landed-on-disk pattern (after SynthSeg auto-appends):
  ``synthseg_parc.nii.tmp.nii.gz``, ``synthseg_volumes.csv.tmp.csv``,
  ``synthseg_qc.csv.tmp.csv``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import nibabel as nib
import numpy as np

from mengrowth.synthseg.config import SynthSegConfig
from mengrowth.synthseg.discovery import discover_studies
from mengrowth.synthseg.exceptions import SynthSegOutputError
from mengrowth.synthseg.primitives import parse_qc_value, parse_volumes_row

logger = logging.getLogger(__name__)


@dataclass
class StudyFinalizeResult:
    """Outcome for a single study's finalization attempt."""

    patient_id: str
    study_id: str
    status: str  # "finalized" | "already_final" | "no_outputs" | "failed"
    error: Optional[str] = None


@dataclass
class FinalizeReport:
    """Aggregated outcomes across the cohort."""

    results: List[StudyFinalizeResult] = field(default_factory=list)

    @property
    def n_finalized(self) -> int:
        return sum(1 for r in self.results if r.status == "finalized")

    @property
    def n_already_final(self) -> int:
        return sum(1 for r in self.results if r.status == "already_final")

    @property
    def n_no_outputs(self) -> int:
        return sum(1 for r in self.results if r.status == "no_outputs")

    @property
    def n_failed(self) -> int:
        return sum(1 for r in self.results if r.status == "failed")

    def exit_code(self) -> int:
        """0 if everything is final (or already was); 2 on any failure."""
        return 2 if self.n_failed > 0 else 0


def _candidate_tmp_paths(
    out_dir: Path, final_name: str, is_nifti: bool
) -> List[Path]:
    """Return tmp filename variants to look for, in preference order.

    Parameters
    ----------
    out_dir : Path
        Study output directory.
    final_name : str
        The desired final filename (e.g. ``"synthseg_volumes.csv"``).
    is_nifti : bool
        NIfTI files use the ``{stem}.tmp.nii.gz`` pattern; CSVs have two
        legitimate on-disk variants depending on whether SynthSeg
        auto-appended ``.csv``.
    """
    final_path = Path(final_name)
    if is_nifti:
        # "synthseg_parc.nii.gz" → "synthseg_parc.nii.tmp.nii.gz"
        tmp = final_path.with_suffix(".tmp.nii.gz")
        return [out_dir / tmp.name]
    # CSV: accept both the Pythonic ("synthseg_volumes.csv.tmp") and the
    # SynthSeg-auto-suffixed ("synthseg_volumes.csv.tmp.csv") variants.
    return [
        out_dir / f"{final_name}.tmp.csv",
        out_dir / f"{final_name}.tmp",
    ]


def _first_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists() and p.stat().st_size > 0:
            return p
    return None


def _validate_parc_geometry(parc_tmp: Path, t1n_path: Path) -> None:
    """Raise :class:`SynthSegOutputError` if parc geometry differs from T1n."""
    parc_img = nib.load(str(parc_tmp))
    ref_img = nib.load(str(t1n_path))
    if parc_img.shape[:3] != ref_img.shape[:3]:
        raise SynthSegOutputError(
            f"parc shape {parc_img.shape[:3]} != t1n shape {ref_img.shape[:3]}"
        )
    if not np.allclose(parc_img.affine, ref_img.affine, atol=1e-4):
        raise SynthSegOutputError(
            "parc affine differs from t1n affine (atol=1e-4)"
        )


def finalize_one_study(
    out_dir: Path,
    t1n_path: Path,
    cfg: SynthSegConfig,
) -> StudyFinalizeResult:
    """Finalize a single study's SynthSeg outputs.

    Parameters
    ----------
    out_dir : Path
        Study output directory under ``{synthseg_output_root}/{pid}/{sid}``.
    t1n_path : Path
        Preprocessed T1n volume used as geometric reference.
    cfg : SynthSegConfig
        Supplies final output filenames.

    Returns
    -------
    StudyFinalizeResult
    """
    pid = out_dir.parent.name
    sid = out_dir.name

    parc_final = out_dir / cfg.parcellation_filename
    vol_final = out_dir / cfg.volumes_filename
    qc_final = out_dir / cfg.qc_filename

    if all(p.exists() and p.stat().st_size > 0 for p in (parc_final, vol_final, qc_final)):
        return StudyFinalizeResult(
            patient_id=pid, study_id=sid, status="already_final"
        )

    parc_tmp = _first_existing(
        _candidate_tmp_paths(out_dir, cfg.parcellation_filename, is_nifti=True)
    )
    vol_tmp = _first_existing(
        _candidate_tmp_paths(out_dir, cfg.volumes_filename, is_nifti=False)
    )
    qc_tmp = _first_existing(
        _candidate_tmp_paths(out_dir, cfg.qc_filename, is_nifti=False)
    )

    if parc_tmp is None and vol_tmp is None and qc_tmp is None:
        return StudyFinalizeResult(
            patient_id=pid, study_id=sid, status="no_outputs"
        )

    missing: List[str] = []
    if parc_tmp is None:
        missing.append("parc")
    if vol_tmp is None:
        missing.append("volumes")
    if qc_tmp is None:
        missing.append("qc")
    if missing:
        err = f"incomplete temp outputs (missing: {', '.join(missing)})"
        return StudyFinalizeResult(
            patient_id=pid, study_id=sid, status="failed", error=err
        )

    # Validate before committing.
    try:
        if not t1n_path.exists():
            raise SynthSegOutputError(f"reference t1n not found: {t1n_path}")
        _validate_parc_geometry(parc_tmp, t1n_path)  # type: ignore[arg-type]

        qc_val = parse_qc_value(qc_tmp)  # type: ignore[arg-type]
        if not (0.0 <= qc_val <= 1.0):
            raise SynthSegOutputError(f"qc {qc_val} outside [0, 1]")

        _ = parse_volumes_row(vol_tmp)  # type: ignore[arg-type]
    except (SynthSegOutputError, ValueError) as e:
        logger.error("Validation failed for %s/%s: %s", pid, sid, e)
        return StudyFinalizeResult(
            patient_id=pid, study_id=sid, status="failed", error=str(e)
        )

    try:
        os.replace(parc_tmp, parc_final)  # type: ignore[arg-type]
        os.replace(vol_tmp, vol_final)  # type: ignore[arg-type]
        os.replace(qc_tmp, qc_final)  # type: ignore[arg-type]
    except OSError as e:
        logger.error("Rename failed for %s/%s: %s", pid, sid, e)
        return StudyFinalizeResult(
            patient_id=pid, study_id=sid, status="failed", error=str(e)
        )

    logger.info("finalized %s/%s  (qc=%.3f)", pid, sid, qc_val)
    return StudyFinalizeResult(
        patient_id=pid, study_id=sid, status="finalized"
    )


def finalize_synthseg_outputs(cfg: SynthSegConfig) -> FinalizeReport:
    """Walk the SynthSeg output tree and finalize any orphaned temp outputs.

    Uses :func:`discover_studies` to enumerate the cohort; the preprocessed
    T1n at ``{input_root}/<pid>/<sid>/t1n.nii.gz`` is the geometric
    reference for parcellation validation.
    """
    report = FinalizeReport()

    studies = discover_studies(cfg.input_root, cfg)
    if not studies:
        logger.warning("No studies discovered under %s", cfg.input_root)
        return report

    for s in studies:
        out_dir = Path(cfg.output_root) / s.patient_id / s.study_id
        if not out_dir.exists():
            report.results.append(
                StudyFinalizeResult(
                    patient_id=s.patient_id,
                    study_id=s.study_id,
                    status="no_outputs",
                )
            )
            continue
        report.results.append(
            finalize_one_study(out_dir, s.input_path, cfg)
        )

    logger.info(
        "Finalize summary: finalized=%d, already_final=%d, no_outputs=%d, failed=%d",
        report.n_finalized,
        report.n_already_final,
        report.n_no_outputs,
        report.n_failed,
    )
    return report
