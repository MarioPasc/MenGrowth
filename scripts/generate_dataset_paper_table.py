#!/usr/bin/env python
"""Dataset-paper appendix: per-study characteristics of the unprocessed MenGrowth-2025 data.

Reads the raw NRRD volumes from the curated dataset tree (which is the
*original*, non-preprocessed data after re-identification) and emits:

* ``<out>/mengrowth_study_characteristics.csv`` — full per-study table.
* ``<out>/mengrowth_study_characteristics.tex`` — LaTeX ``longtable`` for
  the dataset paper appendix.

Fields exposed per study (justification for a dataset paper):

* ``patient_id`` / ``study_id``  — longitudinal cohort identifiers.
* ``age``, ``sex``               — subject demographics.
* ``scanner``                    — site / acquisition scanner label.
* ``num_sequences``              — number of modalities available (0–4).
* Per modality ∈ {t1c, t1n, t2w, t2f}, a single string ``W×H×D @ sx×sy×sz``
  combining the in-plane matrix size, number of slices, and voxel spacing
  in millimetres. An em-dash marks absent modalities.
* ``max_slice_thickness_mm``     — max(z-spacing) across available modalities
  (coarse through-plane resolution indicator; distinguishes thin-slice 3D
  acquisitions from thick 2D stacks).
* ``isotropic``                  — True if all available modalities have
  max(sx, sy, sz) / min(sx, sy, sz) ≤ 1.5 (near-isotropic heuristic).

Usage::

    ~/.conda/envs/growth/bin/python scripts/generate_dataset_paper_table.py \\
        --dataset-root /media/mpascual/PortableSSD/Meningiomas/MenGrowth/curated/dataset/MenGrowth-2025 \\
        --metadata-csv /media/mpascual/PortableSSD/Meningiomas/MenGrowth/curated/dataset/metadata_enriched.csv \\
        --out-dir      /media/mpascual/PortableSSD/Meningiomas/visualizations/mengrowth/dataset_paper
"""

from __future__ import annotations

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import nrrd
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

MODALITIES: tuple[str, ...] = ("t1c", "t1n", "t2w", "t2f")
ISOTROPY_TOL: float = 1.5


@dataclass(frozen=True)
class ModalityHeader:
    """Header-level geometry for a single NRRD volume."""

    shape: tuple[int, int, int]
    spacing: tuple[float, float, float]


def _read_header(path: Path) -> Optional[ModalityHeader]:
    """Return shape and voxel spacing from an NRRD header (no pixel data)."""
    try:
        header = nrrd.read_header(str(path))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to read %s: %s", path, exc)
        return None

    sizes = header.get("sizes")
    directions = header.get("space directions")
    if sizes is None or directions is None:
        return None

    shape = tuple(int(s) for s in sizes[:3])
    directions = np.asarray(directions, dtype=np.float64)
    # "space directions" may carry a None row for scalar sequence axis.
    rows: list[np.ndarray] = []
    for row in directions:
        if row is None or (isinstance(row, np.ndarray) and row.dtype == object):
            continue
        rows.append(np.asarray(row, dtype=np.float64))
    if len(rows) < 3:
        spacing_mm = tuple(float(s) for s in header.get("spacings", (np.nan,) * 3)[:3])
    else:
        spacing_mm = tuple(float(np.linalg.norm(r)) for r in rows[:3])

    return ModalityHeader(shape=shape, spacing=spacing_mm)  # type: ignore[arg-type]


def _scan_study(args: tuple[Path, str, str]) -> dict[str, object]:
    """Collect per-modality headers for a single (patient, study) directory."""
    study_dir, patient_id, study_id = args
    row: dict[str, object] = {"patient_id": patient_id, "study_id": study_id}
    n_available = 0
    max_z: float = 0.0
    iso_flags: list[bool] = []
    for modality in MODALITIES:
        path = study_dir / f"{modality}.nrrd"
        if not path.exists():
            row[f"{modality}_shape"] = None
            row[f"{modality}_spacing"] = None
            continue
        h = _read_header(path)
        if h is None:
            row[f"{modality}_shape"] = None
            row[f"{modality}_spacing"] = None
            continue
        n_available += 1
        row[f"{modality}_shape"] = h.shape
        row[f"{modality}_spacing"] = h.spacing
        max_z = max(max_z, h.spacing[2])
        sp = np.asarray(h.spacing, dtype=np.float64)
        if np.all(sp > 0):
            iso_flags.append(float(sp.max() / sp.min()) <= ISOTROPY_TOL)

    row["num_sequences"] = n_available
    row["max_slice_thickness_mm"] = max_z if n_available > 0 else np.nan
    row["isotropic"] = bool(iso_flags) and all(iso_flags)
    return row


def _fmt_modality(shape: Optional[tuple[int, int, int]],
                  spacing: Optional[tuple[float, float, float]]) -> str:
    """Render a modality cell as ``W×H×D @ sx×sy×sz`` (mm). ``—`` if absent."""
    if shape is None or spacing is None:
        return "—"
    s = "×".join(str(int(x)) for x in shape)
    sp = "×".join(f"{x:.2f}" for x in spacing)
    return f"{s} @ {sp}"


def _load_clinical_metadata(path: Optional[Path]) -> pd.DataFrame:
    """Build a per-patient clinical frame keyed by MenGrowth_ID."""
    if path is None or not path.exists():
        logger.warning("No clinical metadata CSV available (looked at %s)", path)
        return pd.DataFrame(columns=["MenGrowth_ID", "age", "sex", "scanner"])

    df = pd.read_csv(path)
    keep = [c for c in ("MenGrowth_ID", "age", "sex", "scanner") if c in df.columns]
    df = df[keep].copy()
    df = df.dropna(subset=["MenGrowth_ID"])
    if "sex" in df.columns:
        df["sex"] = df["sex"].map({0.0: "F", 1.0: "M"}).fillna("?")
    df["MenGrowth_ID"] = df["MenGrowth_ID"].astype(str)
    return df


def build_table(dataset_root: Path, metadata_csv: Optional[Path],
                workers: int = 4) -> pd.DataFrame:
    """Walk the curated dataset tree and return a per-study DataFrame."""
    study_jobs: list[tuple[Path, str, str]] = []
    for patient_dir in sorted(dataset_root.iterdir()):
        if not patient_dir.is_dir() or not patient_dir.name.startswith("MenGrowth-"):
            continue
        for study_dir in sorted(patient_dir.iterdir()):
            if not study_dir.is_dir():
                continue
            study_jobs.append((study_dir, patient_dir.name, study_dir.name))

    logger.info("Scanning %d studies across %d patients...",
                len(study_jobs),
                len({j[1] for j in study_jobs}))

    rows: list[dict[str, object]] = []
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_scan_study, j) for j in study_jobs]
            for f in as_completed(futures):
                rows.append(f.result())
    else:
        rows = [_scan_study(j) for j in study_jobs]

    df = pd.DataFrame(rows)
    df = df.sort_values(["patient_id", "study_id"]).reset_index(drop=True)

    clinical = _load_clinical_metadata(metadata_csv)
    df = df.merge(clinical, how="left", left_on="patient_id", right_on="MenGrowth_ID")
    df = df.drop(columns=["MenGrowth_ID"], errors="ignore")

    # Wide string columns for the paper table.
    for m in MODALITIES:
        df[m] = [
            _fmt_modality(sh, sp)
            for sh, sp in zip(df[f"{m}_shape"], df[f"{m}_spacing"])
        ]
    return df


def to_latex(df: pd.DataFrame, out_path: Path) -> None:
    """Emit a ``longtable`` suitable for the dataset-paper appendix."""
    cols = ["patient_id", "study_id", "age", "sex", "scanner",
            *MODALITIES, "max_slice_thickness_mm", "isotropic"]
    display = df[cols].copy()
    display = display.rename(columns={
        "patient_id": "Patient",
        "study_id": "Study",
        "age": "Age",
        "sex": "Sex",
        "scanner": "Scanner",
        "t1c": r"T1\textsubscript{c} (W$\times$H$\times$D @ mm)",
        "t1n": r"T1\textsubscript{n} (W$\times$H$\times$D @ mm)",
        "t2w": r"T2\textsubscript{w} (W$\times$H$\times$D @ mm)",
        "t2f": r"T2\textsubscript{f} (W$\times$H$\times$D @ mm)",
        "max_slice_thickness_mm": r"$\max \Delta z$ (mm)",
        "isotropic": "Iso.",
    })
    if "Age" in display.columns:
        display["Age"] = display["Age"].apply(
            lambda x: f"{x:.0f}" if pd.notna(x) else "—"
        )
    display[r"$\max \Delta z$ (mm)"] = display[r"$\max \Delta z$ (mm)"].apply(
        lambda x: f"{x:.2f}" if pd.notna(x) else "—"
    )
    display["Iso."] = display["Iso."].map({True: r"\checkmark", False: ""})
    for c in ("Sex", "Scanner"):
        if c in display.columns:
            display[c] = display[c].fillna("—").astype(str)

    n_cols = len(display.columns)
    col_spec = "l l r c l " + " ".join(["p{2.6cm}"] * len(MODALITIES)) + " r c"
    assert len(col_spec.split()) == n_cols, (len(col_spec.split()), n_cols)

    header = " & ".join(display.columns) + r" \\"
    body_rows = []
    for _, row in display.iterrows():
        vals = [str(v).replace("×", r"$\times$").replace("@", r"@")
                for v in row.tolist()]
        body_rows.append(" & ".join(vals) + r" \\")

    caption = (
        "Per-study acquisition characteristics of the MenGrowth-2025 cohort "
        "(unprocessed, after re-identification only). Modality cells report "
        "matrix size $W{\\times}H{\\times}D$ in voxels and voxel spacing "
        "$s_x{\\times}s_y{\\times}s_z$ in millimetres. ``Iso.'' marks studies "
        "whose available modalities are all near-isotropic "
        "($\\max s / \\min s \\le 1.5$)."
    )

    tex = [
        r"% Auto-generated by scripts/generate_dataset_paper_table.py — do not edit.",
        r"\begin{small}",
        r"\begin{longtable}{" + col_spec + "}",
        r"\caption{" + caption + r"} \label{tab:mengrowth_study_characteristics} \\",
        r"\toprule",
        header,
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{" + str(n_cols) + r"}{c}%",
        r"{\tablename\ \thetable{} -- continued from previous page} \\",
        r"\toprule",
        header,
        r"\midrule",
        r"\endhead",
        r"\midrule \multicolumn{" + str(n_cols) + r"}{r}{Continued on next page} \\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
        *body_rows,
        r"\end{longtable}",
        r"\end{small}",
    ]
    out_path.write_text("\n".join(tex) + "\n", encoding="utf-8")
    logger.info("Wrote LaTeX table → %s", out_path)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, required=True,
                   help="Root of curated MenGrowth-2025 NRRD tree.")
    p.add_argument("--metadata-csv", type=Path, default=None,
                   help="Path to metadata_enriched.csv (age/sex/scanner).")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Directory for CSV + LaTeX outputs.")
    p.add_argument("--workers", type=int, default=4)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = build_table(args.dataset_root, args.metadata_csv, workers=args.workers)

    csv_path = args.out_dir / "mengrowth_study_characteristics.csv"
    csv_cols = ["patient_id", "study_id", "age", "sex", "scanner",
                "num_sequences", *MODALITIES,
                "max_slice_thickness_mm", "isotropic"]
    df[csv_cols].to_csv(csv_path, index=False)
    logger.info("Wrote CSV → %s (%d rows)", csv_path, len(df))

    tex_path = args.out_dir / "mengrowth_study_characteristics.tex"
    to_latex(df, tex_path)


if __name__ == "__main__":
    main()
