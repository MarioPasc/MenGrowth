"""SynthSeg execution logic — per-study and per-patient drivers.

The runner calls ``SynthSeg_predict.py`` in a subprocess, writes outputs via
the atomic temp-file pattern (mirroring the preprocessing convention), and
validates parcellation geometry against the T1n input. Per-study failures
are caught and reported without propagating, so a single-patient SLURM task
continues through its remaining studies even if one volume fails.

Exit-code contract of :func:`run_patient`:

* ``0`` — at least one successful study, no failures
* ``2`` — one or more studies failed
* ``3`` — no studies had the requested input modality

"Skipped because output already exists" counts as success.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional

import nibabel as nib
import numpy as np

from mengrowth.synthseg.config import SynthSegConfig
from mengrowth.synthseg.discovery import StudyInfo, studies_for_patient
from mengrowth.synthseg.exceptions import (
    SynthSegInputError,
    SynthSegOutputError,
    SynthSegRuntimeError,
)
from mengrowth.synthseg.primitives import (
    build_command,
    check_command_available,
    parse_qc_value,
    parse_volumes_row,
)

logger = logging.getLogger(__name__)

StudyStatus = Literal[
    "success",
    "skipped_already_done",
    "skipped_missing_input",
    "failed",
]


@dataclass
class StudyResult:
    """Outcome of a single SynthSeg run on one study."""

    patient_id: str
    study_id: str
    status: StudyStatus
    qc_score: Optional[float] = None
    duration_s: Optional[float] = None
    error: Optional[str] = None
    parcellation_path: Optional[Path] = None


@dataclass
class PatientResult:
    """Aggregated outcome across all studies of one patient."""

    patient_id: str
    studies: List[StudyResult] = field(default_factory=list)

    @property
    def n_success(self) -> int:
        return sum(
            1
            for s in self.studies
            if s.status in ("success", "skipped_already_done")
        )

    @property
    def n_failed(self) -> int:
        return sum(1 for s in self.studies if s.status == "failed")

    @property
    def n_skipped_missing(self) -> int:
        return sum(1 for s in self.studies if s.status == "skipped_missing_input")

    def exit_code(self) -> int:
        """Map aggregated outcomes to a process exit code."""
        if not self.studies:
            return 3
        if self.n_failed > 0:
            return 2
        if self.n_success == 0:
            # All studies were skipped for missing input.
            return 3
        return 0


def _resolve_python_bin(config: SynthSegConfig) -> Path:
    """Pick the Python interpreter for the SynthSeg subprocess.

    Precedence: explicit ``python_bin`` config field, then
    ``{conda_env_path}/bin/python``, then the currently active interpreter
    (``sys.executable``). The SLURM worker activates the SynthSeg env
    before invoking the CLI, so ``sys.executable`` is usually correct, but
    the explicit paths let us run from outside the env (e.g. during local
    testing from the ``growth`` env).
    """
    if config.python_bin:
        return Path(config.python_bin)
    if config.conda_env_path:
        candidate = Path(config.conda_env_path) / "bin" / "python"
        if candidate.exists():
            return candidate
    return Path(sys.executable)


def _outputs_already_present(output_dir: Path, config: SynthSegConfig) -> bool:
    """Return True iff all three final outputs exist and are non-empty."""
    parc = output_dir / config.parcellation_filename
    vol = output_dir / config.volumes_filename
    qc = output_dir / config.qc_filename
    for p in (parc, vol, qc):
        if not p.exists() or p.stat().st_size == 0:
            return False
    return True


def _validate_parcellation_geometry(
    parc_path: Path,
    input_path: Path,
    *,
    affine_atol: float = 1e-4,
) -> None:
    """Assert the parcellation's shape and affine match the input T1n.

    SynthSeg preserves geometry when ``--o`` is a single file; mismatches
    indicate either an interrupted write or an unexpected SynthSeg
    resampling path. Either way we must refuse to commit the output.

    Raises
    ------
    SynthSegOutputError
        If shape or affine deviate from the input.
    """
    parc_img = nib.load(str(parc_path))
    ref_img = nib.load(str(input_path))
    if parc_img.shape[:3] != ref_img.shape[:3]:
        raise SynthSegOutputError(
            f"Parcellation shape {parc_img.shape[:3]} "
            f"!= input shape {ref_img.shape[:3]} ({parc_path})"
        )
    if not np.allclose(parc_img.affine, ref_img.affine, atol=affine_atol):
        raise SynthSegOutputError(
            f"Parcellation affine differs from input affine "
            f"(atol={affine_atol}) for {parc_path}"
        )


def _atomic_rename(tmp: Path, final: Path) -> None:
    """Atomic rename honoring the project's temp-file write convention."""
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(tmp, final)


def run_study(
    config: SynthSegConfig,
    study: StudyInfo,
) -> StudyResult:
    """Run SynthSeg on a single study's T1n volume.

    Parameters
    ----------
    config : SynthSegConfig
    study : StudyInfo
        Discovery record; ``study.input_path`` may or may not exist.

    Returns
    -------
    StudyResult
    """
    out_dir = Path(config.output_root) / study.patient_id / study.study_id
    parc_final = out_dir / config.parcellation_filename
    vol_final = out_dir / config.volumes_filename
    qc_final = out_dir / config.qc_filename

    if not study.has_input_modality:
        msg = (
            f"Missing {config.input_modality}.nii.gz in {study.study_path}; "
            "skipping"
        )
        logger.warning(msg)
        return StudyResult(
            patient_id=study.patient_id,
            study_id=study.study_id,
            status="skipped_missing_input",
            error=msg,
        )

    if _outputs_already_present(out_dir, config):
        logger.info(
            "Outputs already present for %s/%s; skipping",
            study.patient_id,
            study.study_id,
        )
        try:
            qc = parse_qc_value(qc_final)
        except Exception:
            qc = None
        return StudyResult(
            patient_id=study.patient_id,
            study_id=study.study_id,
            status="skipped_already_done",
            qc_score=qc,
            parcellation_path=parc_final,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    parc_tmp = parc_final.with_suffix(".tmp.nii.gz")
    vol_tmp = vol_final.with_suffix(vol_final.suffix + ".tmp")
    qc_tmp = qc_final.with_suffix(qc_final.suffix + ".tmp")

    # Best-effort cleanup of stale temp files from an aborted prior run.
    for p in (parc_tmp, vol_tmp, qc_tmp):
        if p.exists():
            try:
                p.unlink()
            except OSError:
                logger.warning("Could not remove stale temp file %s", p)

    python_bin = _resolve_python_bin(config)
    argv = build_command(
        synthseg_repo=Path(config.synthseg_repo),
        input_nii=study.input_path,
        output_parc=parc_tmp,
        vol_csv=vol_tmp,
        qc_csv=qc_tmp,
        robust=config.robust,
        threads=config.threads,
        use_gpu=config.use_gpu,
        python_bin=python_bin,
    )

    if not check_command_available(argv):
        err = (
            f"SynthSeg command not executable: python={argv[0]}, "
            f"script={argv[1] if len(argv) > 1 else '?'}"
        )
        logger.error(err)
        return StudyResult(
            patient_id=study.patient_id,
            study_id=study.study_id,
            status="failed",
            error=err,
        )

    logger.info(
        "Running SynthSeg: %s -> %s",
        study.input_path,
        parc_final,
    )
    logger.debug("argv: %s", " ".join(argv))

    t0 = time.time()
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=config.timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        err = f"SynthSeg timed out after {config.timeout_seconds}s on {study.input_path}"
        logger.error(err)
        return StudyResult(
            patient_id=study.patient_id,
            study_id=study.study_id,
            status="failed",
            duration_s=time.time() - t0,
            error=err,
        )
    except FileNotFoundError as e:
        err = f"SynthSeg binary not found: {e}"
        logger.error(err)
        return StudyResult(
            patient_id=study.patient_id,
            study_id=study.study_id,
            status="failed",
            error=err,
        )

    duration = time.time() - t0

    if proc.returncode != 0:
        stderr_tail = (proc.stderr or "")[-800:]
        err = (
            f"SynthSeg returned rc={proc.returncode} on {study.input_path}: "
            f"{stderr_tail}"
        )
        logger.error(err)
        return StudyResult(
            patient_id=study.patient_id,
            study_id=study.study_id,
            status="failed",
            duration_s=duration,
            error=err,
        )

    # Post-run validation before committing.
    try:
        if not parc_tmp.exists():
            raise SynthSegOutputError(f"Parcellation not written: {parc_tmp}")
        if not vol_tmp.exists():
            raise SynthSegOutputError(f"Volumes CSV not written: {vol_tmp}")
        if not qc_tmp.exists():
            raise SynthSegOutputError(f"QC CSV not written: {qc_tmp}")

        _validate_parcellation_geometry(parc_tmp, study.input_path)

        qc_val = parse_qc_value(qc_tmp)
        if not (0.0 <= qc_val <= 1.0):
            raise SynthSegOutputError(
                f"QC score {qc_val} outside [0, 1] in {qc_tmp}"
            )

        _ = parse_volumes_row(vol_tmp)  # sanity-check parseability

    except (SynthSegOutputError, ValueError) as e:
        logger.error("Output validation failed for %s/%s: %s",
                     study.patient_id, study.study_id, e)
        return StudyResult(
            patient_id=study.patient_id,
            study_id=study.study_id,
            status="failed",
            duration_s=duration,
            error=str(e),
        )

    # Atomic commit of all three outputs.
    _atomic_rename(parc_tmp, parc_final)
    _atomic_rename(vol_tmp, vol_final)
    _atomic_rename(qc_tmp, qc_final)

    logger.info(
        "  OK  %s/%s  (qc=%.3f, %.1fs)",
        study.patient_id,
        study.study_id,
        qc_val,
        duration,
    )

    return StudyResult(
        patient_id=study.patient_id,
        study_id=study.study_id,
        status="success",
        qc_score=qc_val,
        duration_s=duration,
        parcellation_path=parc_final,
    )


def run_patient(
    config: SynthSegConfig,
    patient_id: str,
) -> PatientResult:
    """Run SynthSeg on every study of a single patient.

    Failures on individual studies are caught and recorded; the loop
    continues. Logs a summary table at the end.
    """
    result = PatientResult(patient_id=patient_id)

    studies = studies_for_patient(config.input_root, config, patient_id)
    if not studies:
        logger.error("No studies found for patient %s", patient_id)
        return result

    logger.info("=" * 60)
    logger.info("SYNTHSEG — PATIENT %s (%d studies)", patient_id, len(studies))
    logger.info("=" * 60)

    for study in studies:
        try:
            sr = run_study(config, study)
        except Exception as e:  # noqa: BLE001 — last-ditch barrier
            logger.exception("Unexpected error on %s/%s",
                             study.patient_id, study.study_id)
            sr = StudyResult(
                patient_id=study.patient_id,
                study_id=study.study_id,
                status="failed",
                error=f"uncaught: {e}",
            )
        result.studies.append(sr)

    logger.info("-" * 60)
    logger.info(
        "Patient %s summary: success=%d, failed=%d, skipped_missing=%d "
        "(total studies=%d)",
        patient_id,
        result.n_success,
        result.n_failed,
        result.n_skipped_missing,
        len(result.studies),
    )
    for sr in result.studies:
        qc_str = f"{sr.qc_score:.3f}" if sr.qc_score is not None else "  n/a"
        dur_str = f"{sr.duration_s:.1f}s" if sr.duration_s is not None else "   n/a"
        logger.info("  %-26s  %-22s  qc=%s  %s",
                    sr.study_id, sr.status, qc_str, dur_str)
    logger.info("=" * 60)

    return result
