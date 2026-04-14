"""Low-level SynthSeg primitives preserved from the legacy PaCS-SR code.

These are the only artifacts kept from the original ``synthseg_evaluation``
module after rebasing onto MenGrowth's filesystem-driven pipeline:

* :data:`SYNTHSEG_LABEL_MAP` — mapping of SynthSeg region name → FreeSurfer
  label ID for the 30 deep / cortical structures SynthSeg segments.
* :func:`build_command` — construct the ``SynthSeg_predict.py`` argv.
* :func:`parse_qc_csv` / :func:`parse_qc_value` — parse the QC CSV SynthSeg
  writes via ``--qc``. Supports both the single-column ``qc`` layout and
  the per-region layout produced by ``--robust``.
* :func:`parse_volumes_csv` / :func:`parse_volumes_row` — parse the per-region
  volume CSV SynthSeg writes via ``--vol``.

References
----------
Billot, B., et al. "SynthSeg: Segmentation of brain MRI scans of any
contrast and resolution without retraining." *Medical Image Analysis*, 86,
102789 (2023). DOI: 10.1016/j.media.2023.102789
"""

from __future__ import annotations

import csv
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# FreeSurfer label IDs for SynthSeg's 30 structural labels.
#
# Kept as a module-level constant because downstream analysis code needs a
# canonical mapping between region names and label integers when applying the
# T1n parcellation as an ROI template to other modalities.
# -----------------------------------------------------------------------------
SYNTHSEG_LABEL_MAP: Dict[str, int] = {
    "Left-Cerebral-White-Matter": 2,
    "Left-Cerebral-Cortex": 3,
    "Left-Lateral-Ventricle": 4,
    "Left-Cerebellum-White-Matter": 7,
    "Left-Cerebellum-Cortex": 8,
    "Left-Thalamus": 10,
    "Left-Caudate": 11,
    "Left-Putamen": 12,
    "Left-Pallidum": 13,
    "3rd-Ventricle": 14,
    "4th-Ventricle": 15,
    "Brain-Stem": 16,
    "Left-Hippocampus": 17,
    "Left-Amygdala": 18,
    "CSF": 24,
    "Left-Accumbens-area": 26,
    "Left-VentralDC": 28,
    "Right-Cerebral-White-Matter": 41,
    "Right-Cerebral-Cortex": 42,
    "Right-Lateral-Ventricle": 43,
    "Right-Cerebellum-White-Matter": 46,
    "Right-Cerebellum-Cortex": 47,
    "Right-Thalamus": 49,
    "Right-Caudate": 50,
    "Right-Putamen": 51,
    "Right-Pallidum": 52,
    "Right-Hippocampus": 53,
    "Right-Amygdala": 54,
    "Right-Accumbens-area": 58,
    "Right-VentralDC": 60,
}


def build_command(
    *,
    synthseg_repo: Path,
    input_nii: Path,
    output_parc: Path,
    vol_csv: Path,
    qc_csv: Path,
    robust: bool,
    threads: int,
    use_gpu: bool,
    python_bin: Path = Path("python"),
    posteriors_path: Optional[Path] = None,
) -> List[str]:
    """Construct the argv for ``SynthSeg_predict.py``.

    Parameters
    ----------
    synthseg_repo : Path
        Root of the SynthSeg repository. The script is expected at
        ``{repo}/scripts/commands/SynthSeg_predict.py``.
    input_nii : Path
        Input T1n NIfTI file.
    output_parc : Path
        Destination for the 32-region parcellation (NIfTI).
    vol_csv : Path
        Destination CSV for per-region volumes (``--vol`` flag).
    qc_csv : Path
        Destination CSV for the QC score (``--qc`` flag).
    robust : bool
        Append ``--robust`` — mandatory for pathology-containing cohorts.
    threads : int
        Number of CPU threads; passed as ``--threads`` when ``>= 1``.
    use_gpu : bool
        If False, append ``--cpu`` to force CPU inference.
    python_bin : Path, optional
        Python interpreter to invoke. Defaults to ``python`` on PATH.
    posteriors_path : Path, optional
        If provided, append ``--post`` to save posterior probability maps.

    Returns
    -------
    list[str]
        argv ready for :func:`subprocess.run`.
    """
    script = Path(synthseg_repo) / "scripts" / "commands" / "SynthSeg_predict.py"

    cmd: List[str] = [
        str(python_bin),
        str(script),
        "--i",
        str(input_nii),
        "--o",
        str(output_parc),
        "--vol",
        str(vol_csv),
        "--qc",
        str(qc_csv),
    ]
    if robust:
        cmd.append("--robust")
    if threads >= 1:
        cmd.extend(["--threads", str(threads)])
    if not use_gpu:
        cmd.append("--cpu")
    if posteriors_path is not None:
        cmd.extend(["--post", str(posteriors_path)])

    return cmd


def check_command_available(argv: Sequence[str]) -> bool:
    """Return True if the first two argv entries (python + script) exist.

    Lightweight pre-flight check used by the runner before launching the
    subprocess. Absence of either is a hard error because the subprocess
    would fail with an obscure message otherwise.

    Parameters
    ----------
    argv : Sequence[str]
        Command as built by :func:`build_command`.
    """
    if len(argv) < 2:
        return False
    py_bin = argv[0]
    if not Path(py_bin).is_absolute():
        if shutil.which(py_bin) is None:
            return False
    elif not Path(py_bin).exists():
        return False
    return Path(argv[1]).exists()


# -----------------------------------------------------------------------------
# CSV parsers. SynthSeg writes one-row-per-input CSVs; when we drive it per
# volume, each CSV contains exactly one data row.
# -----------------------------------------------------------------------------


def parse_qc_csv(qc_csv: Path) -> Dict[str, float]:
    """Parse the SynthSeg QC CSV into a ``subject -> score`` dict.

    Handles both CSV layouts produced by SynthSeg:

    1. Single-column ``qc`` (non-robust mode).
    2. Per-region columns (e.g. ``general white matter``, ``general grey
       matter``) emitted by ``--robust``. In that case the mean across
       numeric columns is used.

    Parameters
    ----------
    qc_csv : Path
        Path to the CSV written by SynthSeg's ``--qc``.

    Returns
    -------
    dict[str, float]
        Subject id (stem of the input NIfTI path, minus ``.nii``) mapped to
        the scalar QC value in ``[0, 1]``.
    """
    scores: Dict[str, float] = {}
    with open(qc_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            subject_raw = row.get("subject", row.get("", ""))
            subject = Path(subject_raw).stem.replace(".nii", "")

            qc_val = row.get("qc", row.get("qc_score", None))
            if qc_val is not None and qc_val != "":
                try:
                    scores[subject] = float(qc_val)
                    continue
                except (ValueError, TypeError):
                    pass

            vals: List[float] = []
            for key, val in row.items():
                if key in ("subject", ""):
                    continue
                try:
                    vals.append(float(val))
                except (ValueError, TypeError):
                    pass
            if vals:
                scores[subject] = float(np.mean(vals))
            else:
                logger.warning("No valid QC values parsed from %s", qc_csv)
    return scores


def parse_qc_value(qc_csv: Path) -> float:
    """Parse a single-volume QC CSV and return the scalar score.

    Convenience wrapper for the common case (one input per run).

    Raises
    ------
    ValueError
        If the CSV cannot be parsed or contains no valid score.
    """
    scores = parse_qc_csv(qc_csv)
    if not scores:
        raise ValueError(f"No QC score found in {qc_csv}")
    if len(scores) > 1:
        logger.warning(
            "Expected 1 subject in %s but found %d; returning first", qc_csv, len(scores)
        )
    return float(next(iter(scores.values())))


def parse_volumes_csv(csv_path: Path) -> Dict[str, Dict[str, float]]:
    """Parse a SynthSeg volumes CSV into nested ``{subject: {region: mm3}}``.

    Parameters
    ----------
    csv_path : Path
        Path to the CSV written by SynthSeg's ``--vol``.
    """
    volumes: Dict[str, Dict[str, float]] = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            subject_raw = row.get("subject", row.get("", ""))
            subject = Path(subject_raw).stem.replace(".nii", "")
            region_vols: Dict[str, float] = {}
            for key, val in row.items():
                if key in ("subject", ""):
                    continue
                try:
                    region_vols[key] = float(val)
                except (ValueError, TypeError):
                    pass
            volumes[subject] = region_vols
    return volumes


def parse_volumes_row(csv_path: Path) -> Dict[str, float]:
    """Parse a single-volume volumes CSV and return ``{region: mm3}``.

    Raises
    ------
    ValueError
        If the CSV is empty or unparseable.
    """
    volumes = parse_volumes_csv(csv_path)
    if not volumes:
        raise ValueError(f"No volumes parsed from {csv_path}")
    if len(volumes) > 1:
        logger.warning(
            "Expected 1 subject in %s but found %d; returning first",
            csv_path,
            len(volumes),
        )
    return next(iter(volumes.values()))
