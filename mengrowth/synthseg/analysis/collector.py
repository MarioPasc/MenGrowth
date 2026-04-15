"""Cohort-level collection of SynthSeg outputs and acquisition metadata.

Produces three tidy DataFrames and persists them as CSVs under
``{output_dir}/cohort/``:

* ``cohort_volumes.csv`` — long form, one row per (patient, study,
  canonical region).
* ``cohort_qc.csv`` — one row per study.
* ``cohort_metadata.csv`` — one row per (patient, study, modality), derived
  from the curation-stage ``per_study_metrics.csv``.

SynthSeg's volumes CSV uses human-readable lowercase column names
(e.g. ``"left thalamus"``, ``"brain-stem"``) that differ from the
:data:`SYNTHSEG_LABEL_MAP` keys (``"Left-Thalamus"``, ``"Brain-Stem"``).
:func:`_csv_key_to_canonical` bridges the two representations by stripping
all punctuation and whitespace and lowercasing, yielding a stable key.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

from mengrowth.synthseg.analysis.config import AnalysisConfig
from mengrowth.synthseg.discovery import StudyInfo, discover_studies
from mengrowth.synthseg.config import SynthSegConfig
from mengrowth.synthseg.primitives import (
    SYNTHSEG_LABEL_MAP,
    parse_qc_value,
    parse_volumes_row,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Column-name normalisation
# ---------------------------------------------------------------------------


def _normalize(s: str) -> str:
    """Strip whitespace, dashes, and lowercase — yields a stable join key."""
    return "".join(ch for ch in s.lower() if ch.isalnum())


_CANONICAL_BY_NORM: Dict[str, str] = {
    _normalize(name): name for name in SYNTHSEG_LABEL_MAP.keys()
}


def _csv_key_to_canonical(csv_key: str) -> Optional[str]:
    """Map a SynthSeg volumes-CSV column name to the canonical label name.

    Returns ``None`` if the column is not one of the 30 labels in
    :data:`SYNTHSEG_LABEL_MAP` (e.g. "total intracranial",
    "left inferior lateral ventricle").
    """
    return _CANONICAL_BY_NORM.get(_normalize(csv_key))


# ---------------------------------------------------------------------------
# Internal helper: build a SynthSegConfig surrogate for discovery
# ---------------------------------------------------------------------------


def _discovery_config(cfg: AnalysisConfig) -> SynthSegConfig:
    """Build a throwaway :class:`SynthSegConfig` for ``discover_studies``.

    ``discover_studies`` takes a :class:`SynthSegConfig` to read the input
    modality name — we reuse it here without running the full SynthSeg
    pipeline.
    """
    return SynthSegConfig(
        input_root=cfg.preprocessed_root,
        output_root=cfg.synthseg_output_root,
        synthseg_repo="unused",
        input_modality=cfg.input_modality_for_discovery,
        parcellation_filename=cfg.parcellation_filename,
        volumes_filename=cfg.volumes_filename,
        qc_filename=cfg.qc_filename,
    )


def _studies_with_finalized_outputs(cfg: AnalysisConfig) -> List[StudyInfo]:
    """Discover studies that have all three finalized SynthSeg outputs."""
    disc_cfg = _discovery_config(cfg)
    studies = discover_studies(cfg.preprocessed_root, disc_cfg)
    ready: List[StudyInfo] = []
    for s in studies:
        out_dir = Path(cfg.synthseg_output_root) / s.patient_id / s.study_id
        parc = out_dir / cfg.parcellation_filename
        vol = out_dir / cfg.volumes_filename
        qc = out_dir / cfg.qc_filename
        if parc.exists() and vol.exists() and qc.exists():
            ready.append(s)
    logger.info(
        "Collector: %d/%d studies have finalized SynthSeg outputs",
        len(ready),
        len(studies),
    )
    return ready


def _assign_timepoint_index(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``timepoint_index`` as the within-patient rank of ``study_id``."""
    df = df.sort_values(["patient_id", "study_id"]).reset_index(drop=True)
    df["timepoint_index"] = df.groupby("patient_id").cumcount()
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def collect_cohort_volumes(cfg: AnalysisConfig) -> pd.DataFrame:
    """Assemble a long-form DataFrame of per-region volumes across the cohort.

    Columns: ``patient_id, study_id, timepoint_index, region_name,
    region_id, volume_mm3``. Region names are normalised to the canonical
    :data:`SYNTHSEG_LABEL_MAP` keys. Rows with no match in the label map
    (e.g. "total intracranial") are dropped.
    """
    studies = _studies_with_finalized_outputs(cfg)

    rows: List[Dict] = []
    for s in studies:
        vol_csv = (
            Path(cfg.synthseg_output_root)
            / s.patient_id
            / s.study_id
            / cfg.volumes_filename
        )
        try:
            raw_vols = parse_volumes_row(vol_csv)
        except (ValueError, OSError) as e:
            logger.warning(
                "Skipping volumes for %s/%s: %s", s.patient_id, s.study_id, e
            )
            continue

        for csv_key, value in raw_vols.items():
            canonical = _csv_key_to_canonical(csv_key)
            if canonical is None:
                continue
            rows.append(
                {
                    "patient_id": s.patient_id,
                    "study_id": s.study_id,
                    "region_name": canonical,
                    "region_id": SYNTHSEG_LABEL_MAP[canonical],
                    "volume_mm3": float(value),
                }
            )

    if not rows:
        logger.warning("collect_cohort_volumes: no rows produced")
        return pd.DataFrame(
            columns=[
                "patient_id",
                "study_id",
                "timepoint_index",
                "region_name",
                "region_id",
                "volume_mm3",
            ]
        )

    df = pd.DataFrame(rows)
    tp = _assign_timepoint_index(
        df[["patient_id", "study_id"]].drop_duplicates()
    )
    df = df.merge(tp, on=["patient_id", "study_id"], how="left")
    return df[
        [
            "patient_id",
            "study_id",
            "timepoint_index",
            "region_name",
            "region_id",
            "volume_mm3",
        ]
    ].sort_values(["patient_id", "study_id", "region_id"]).reset_index(drop=True)


def collect_cohort_qc(cfg: AnalysisConfig) -> pd.DataFrame:
    """One row per study with the SynthSeg scalar QC score."""
    studies = _studies_with_finalized_outputs(cfg)

    rows: List[Dict] = []
    for s in studies:
        qc_csv = (
            Path(cfg.synthseg_output_root)
            / s.patient_id
            / s.study_id
            / cfg.qc_filename
        )
        try:
            qc_val = parse_qc_value(qc_csv)
        except (ValueError, OSError) as e:
            logger.warning(
                "Skipping QC for %s/%s: %s", s.patient_id, s.study_id, e
            )
            continue
        rows.append(
            {
                "patient_id": s.patient_id,
                "study_id": s.study_id,
                "qc_score": float(qc_val),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=["patient_id", "study_id", "timepoint_index", "qc_score"]
        )

    df = _assign_timepoint_index(pd.DataFrame(rows))
    return df[["patient_id", "study_id", "timepoint_index", "qc_score"]]


def collect_cohort_metadata(cfg: AnalysisConfig) -> pd.DataFrame:
    """Per-(patient, study, modality) acquisition metadata.

    Source: ``per_study_metrics.csv`` produced by the dataset-curation
    Phase 5. Filters to the modalities requested for Phase 2 analyses and
    adds derived columns ``max_spacing`` and ``anisotropy_ratio``.
    """
    meta_path = Path(cfg.original_metadata_csv)
    if not meta_path.exists():
        raise FileNotFoundError(
            f"original_metadata_csv not found: {meta_path}"
        )

    df = pd.read_csv(meta_path)
    required = {"patient_id", "study_id", "sequence", "spacing_x", "spacing_y", "spacing_z"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"metadata CSV {meta_path} missing required columns: {sorted(missing)}"
        )

    modalities = [m.lower() for m in cfg.modalities_for_roi_analysis]
    df = df[df["sequence"].str.lower().isin(modalities)].copy()

    df["max_spacing"] = df[["spacing_x", "spacing_y", "spacing_z"]].max(axis=1)
    df["min_spacing"] = df[["spacing_x", "spacing_y", "spacing_z"]].min(axis=1)
    df["anisotropy_ratio"] = df["max_spacing"] / df["min_spacing"]
    # In-plane resolution (min of x, y — the thin axes)
    df["in_plane_resolution"] = df[["spacing_x", "spacing_y"]].min(axis=1)

    keep = [
        "patient_id",
        "study_id",
        "sequence",
        "spacing_x",
        "spacing_y",
        "spacing_z",
        "max_spacing",
        "min_spacing",
        "anisotropy_ratio",
        "in_plane_resolution",
    ]
    return df[keep].sort_values(["patient_id", "study_id", "sequence"]).reset_index(drop=True)


def persist_cohort(cfg: AnalysisConfig) -> Dict[str, pd.DataFrame]:
    """Run all three collectors and write the CSVs under ``{output_dir}/cohort``.

    Returns the dict ``{"volumes": ..., "qc": ..., "metadata": ...}``.
    """
    out = Path(cfg.output_dir) / "cohort"
    out.mkdir(parents=True, exist_ok=True)

    vols = collect_cohort_volumes(cfg)
    qc = collect_cohort_qc(cfg)
    meta = collect_cohort_metadata(cfg)

    vols.to_csv(out / "cohort_volumes.csv", index=False)
    qc.to_csv(out / "cohort_qc.csv", index=False)
    meta.to_csv(out / "cohort_metadata.csv", index=False)

    logger.info(
        "Cohort persisted: volumes=%d rows (%d studies × %d regions), "
        "qc=%d rows, metadata=%d rows",
        len(vols),
        vols[["patient_id", "study_id"]].drop_duplicates().shape[0] if len(vols) else 0,
        vols["region_name"].nunique() if len(vols) else 0,
        len(qc),
        len(meta),
    )
    return {"volumes": vols, "qc": qc, "metadata": meta}
