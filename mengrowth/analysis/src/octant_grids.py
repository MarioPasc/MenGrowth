"""Phase 3: Qualitative octant-grid visualizations using PyVista."""

from __future__ import annotations

import logging
import sys
from dataclasses import fields as dc_fields
from pathlib import Path
from typing import Dict, Final, List, Optional, Tuple

import numpy as np

from .types import DatasetMetrics

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────

# 5 showcase patients chosen during exploration.
# Format: (patient_id, description, camera_azimuth, camera_elevation)
# Set azimuth/elevation to None for auto camera.
# Use scripts snippet below to find good angles interactively.
SHOWCASE_PATIENTS: Final[List[Tuple[str, str, Optional[float], Optional[float]]]] = [
    ("MenGrowth-0009", "Clear grower (37k→82k, 5 tp)", 38.3, -37.1),
    ("MenGrowth-0015", "Most timepoints (6 studies)", 43.1, -20.1),
    ("MenGrowth-0020", "Steady grower (9k→13k, 5 tp)", -119.8, -20.9),
    #("MenGrowth-0032", "Largest tumor at baseline (48k)", None, None),
    #("MenGrowth-0007", "Small stable (1.4k→1.1k, 4 tp)", None, None),
]

# Row ordering for grids (matches plan)
GRID_MODALITIES: Final[List[str]] = ["t1n", "t1c", "t2f", "t2w"]

MODALITY_DISPLAY: Final[Dict[str, str]] = {
    "t1n": "t1n",
    "t1c": "t1c",
    "t2f": "t2f",
    "t2w": "t2w",
}

# Distinct grayscale brain surface tint per modality so rows are
# visually distinguishable even without reading the labels.
MODALITY_MESH_COLOR: Final[Dict[str, Tuple[float, float, float]]] = {
    "t1n": (0.92, 0.92, 0.92),  # near-white
    "t1c": (0.78, 0.78, 0.78),  # medium gray
    "t2f": (0.64, 0.64, 0.64),  # dark gray
    "t2w": (0.50, 0.50, 0.50),  # darkest
}


# ── Lazy imports from the octant visualization script ────────────────────────


def _import_octant_helpers() -> dict:
    """Import helpers from scripts/visualize_meningioma_octant.py.

    Returns:
        Dict with keys: load_volume, compute_brain_mask, find_optimal_octant,
        render_meningioma_octant, create_legend, RenderConfig.
    """
    scripts_dir = str(Path(__file__).resolve().parents[3] / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    from visualize_meningioma_octant import (
        RenderConfig,
        compute_brain_mask,
        create_legend,
        find_optimal_octant,
        load_volume,
        render_meningioma_octant,
    )

    return {
        "load_volume": load_volume,
        "compute_brain_mask": compute_brain_mask,
        "find_optimal_octant": find_optimal_octant,
        "render_meningioma_octant": render_meningioma_octant,
        "create_legend": create_legend,
        "RenderConfig": RenderConfig,
    }


def _derive_config(base_cfg: object, RenderConfig: type, **overrides) -> object:
    """Create a new RenderConfig with selected fields overridden.

    Works around the frozen dataclass by reconstructing it.
    """
    kw = {f.name: getattr(base_cfg, f.name) for f in dc_fields(base_cfg)}
    kw.update(overrides)
    return RenderConfig(**kw)


# ── Per-patient grid ─────────────────────────────────────────────────────────


def render_patient_grid(
    patient_id: str,
    dataset_root: Path,
    output_path: Path,
    metrics: DatasetMetrics,
    camera_azimuth: Optional[float] = None,
    camera_elevation: Optional[float] = None,
) -> bool:
    """Render a 4-row x N-column octant grid for one patient.

    Rows = modalities (t1n, t1c, t2f, t2w) with distinct surface tints.
    Columns = timepoints (t_0 ... t_n).
    Camera angle is either provided explicitly or auto-computed.

    Args:
        patient_id: MenGrowth patient ID.
        dataset_root: Preprocessed dataset root.
        output_path: Where to save the composite PDF/PNG.
        metrics: Dataset metrics (for study list).
        camera_azimuth: Explicit camera azimuth (degrees), or None for auto.
        camera_elevation: Explicit camera elevation (degrees), or None for auto.

    Returns:
        True if the grid was saved successfully.
    """
    try:
        import pyvista as pv

        pv.OFF_SCREEN = True
    except ImportError:
        logger.error("PyVista not installed — skipping octant grids")
        return False

    import matplotlib.pyplot as plt

    helpers = _import_octant_helpers()
    load_volume = helpers["load_volume"]
    compute_brain_mask = helpers["compute_brain_mask"]
    find_optimal_octant = helpers["find_optimal_octant"]
    render_meningioma_octant = helpers["render_meningioma_octant"]
    RenderConfig = helpers["RenderConfig"]

    pm = metrics.patients.get(patient_id)
    if pm is None:
        logger.warning("Patient %s not found in metrics", patient_id)
        return False

    patient_dir = dataset_root / patient_id
    study_dirs = sorted(
        d
        for d in patient_dir.iterdir()
        if d.is_dir() and d.name.startswith("MenGrowth-")
    )
    n_tp = len(study_dirs)

    # ── Determine consistent camera from first non-empty study ──
    ref_t1c: Optional[np.ndarray] = None
    ref_seg: Optional[np.ndarray] = None
    for study_dir in study_dirs:
        seg_path = study_dir / "seg.nii.gz"
        t1c_path = study_dir / "t1c.nii.gz"
        if seg_path.exists() and t1c_path.exists():
            seg = load_volume(seg_path).astype(np.int32)
            if np.any(seg > 0):
                ref_t1c = load_volume(t1c_path)
                ref_seg = seg
                break

    # Fall back to first study if all segs are empty
    if ref_t1c is None:
        ref_t1c = load_volume(study_dirs[0] / "t1c.nii.gz")
        ref_seg = load_volume(study_dirs[0] / "seg.nii.gz").astype(np.int32)

    brain_mask = compute_brain_mask(ref_t1c)
    slice_indices, octant = find_optimal_octant(ref_t1c, ref_seg)

    # Camera: use explicit angles if provided, otherwise auto-camera
    base_cfg = RenderConfig(
        octant=octant,
        window_size=(800, 700),
        zoom=3.5,
        camera_azimuth=camera_azimuth,
        camera_elevation=camera_elevation,
    )

    # ── Render all cells ──
    cell_images: Dict[Tuple[int, int], np.ndarray] = {}

    for col, study_dir in enumerate(study_dirs):
        for row, modality in enumerate(GRID_MODALITIES):
            mod_path = study_dir / f"{modality}.nii.gz"
            seg_path = study_dir / "seg.nii.gz"
            if not mod_path.exists() or not seg_path.exists():
                continue

            logger.info("  Rendering %s %s t%d", patient_id, modality, col)
            volume = load_volume(mod_path)
            seg = load_volume(seg_path).astype(np.int32)

            # Per-modality surface color
            cfg = _derive_config(
                base_cfg,
                RenderConfig,
                mesh_color=MODALITY_MESH_COLOR[modality],
            )

            try:
                plotter = render_meningioma_octant(
                    volume,
                    seg,
                    slice_indices,
                    cfg=cfg,
                    brain_mask=brain_mask,
                    off_screen=True,
                )
                plotter.set_background("black")
                plotter.render()
                img = plotter.screenshot(return_img=True)
                plotter.close()
                cell_images[(row, col)] = img
            except Exception as e:
                logger.warning(
                    "  Render failed %s %s t%d: %s", patient_id, modality, col, e
                )

    if not cell_images:
        logger.warning("No cells rendered for %s", patient_id)
        return False

    # ── Compose grid with matplotlib ──
    # Extra width on the left for row labels
    label_width_ratio = 0.12
    col_ratios = [label_width_ratio] + [1.0] * n_tp

    fig, axes = plt.subplots(
        4,
        n_tp + 1,  # +1 for the label column
        figsize=(2.5 * n_tp + 1.0, 2.5 * 4),
        gridspec_kw={
            "wspace": 0.02,
            "hspace": 0.05,
            "width_ratios": col_ratios,
        },
    )
    fig.patch.set_facecolor("black")

    for row in range(4):
        modality = GRID_MODALITIES[row]

        # Label column (col 0)
        ax_lbl = axes[row, 0]
        ax_lbl.set_facecolor("black")
        ax_lbl.axis("off")
        ax_lbl.text(
            0.9,
            0.5,
            MODALITY_DISPLAY[modality],
            color="white",
            fontsize=12,
            fontweight="bold",
            ha="right",
            va="center",
            transform=ax_lbl.transAxes,
        )

        # Render columns (col 1 .. n_tp)
        for col in range(n_tp):
            ax = axes[row, col + 1]
            ax.set_facecolor("black")
            ax.axis("off")

            img = cell_images.get((row, col))
            if img is not None:
                ax.imshow(img)

            # Timepoint labels on top row
            if row == 0:
                ax.set_title(f"$t_{{{col}}}$", color="white", fontsize=12, pad=5)

    # Patient ID as super-title
    fig.suptitle(patient_id, color="white", fontsize=14, y=1.01)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
        facecolor="black",
        edgecolor="none",
    )
    plt.close(fig)
    logger.info("Saved octant grid: %s", output_path)
    return True


# ── Orchestrator ─────────────────────────────────────────────────────────────


def generate_octant_grids(
    dataset_root: Path,
    output_dir: Path,
    metrics: DatasetMetrics,
) -> None:
    """Generate octant grids for showcase patients + standalone legend.

    Args:
        dataset_root: Preprocessed dataset root.
        output_dir: Directory to write octant grid PDFs.
        metrics: Pre-computed dataset metrics.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Standalone label legend
    helpers = _import_octant_helpers()
    fig_dir = output_dir.parent / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    helpers["create_legend"](save=fig_dir / "meningioma_legend.png")

    for patient_id, desc, cam_azim, cam_elev in SHOWCASE_PATIENTS:
        logger.info("Octant grid: %s (%s)", patient_id, desc)
        out_path = output_dir / f"{patient_id}.pdf"
        success = render_patient_grid(
            patient_id,
            dataset_root,
            out_path,
            metrics,
            camera_azimuth=cam_azim,
            camera_elevation=cam_elev,
        )
        if not success:
            logger.warning("Failed to generate octant grid for %s", patient_id)
