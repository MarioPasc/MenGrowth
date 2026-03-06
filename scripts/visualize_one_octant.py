#!/usr/bin/env python3
"""Interactive single-octant preview for tuning all rendering parameters.

Rotate the view in the interactive window, then close it.
Press 'p' at any time to print the full ``visualize_meningioma_octant.py``
command with every tunable parameter, ready to copy-paste.

Usage:
    python scripts/visualize_one_octant.py

Edit the configuration variables below.
"""

import sys
from pathlib import Path
from typing import Tuple

import numpy as np

sys.path.insert(0, "scripts")
from visualize_meningioma_octant import (
    MENINGIOMA_ALPHA,
    RenderConfig,
    compute_brain_mask,
    find_optimal_octant,
    load_volume,
    render_meningioma_octant,
)

ROOT = "/media/mpascual/PortableSSD/Meningiomas/MenGrowth/preprocessed/MenGrowth-2025"

# === EDIT THESE ===
PATIENT = "MenGrowth-0009"
STUDY = "MenGrowth-0009-000"
MOD = "t1c"
AZIM = None  # degrees, or None for auto
ELEV = None  # degrees, or None for auto
ZOOM = 2.5
MESH_ALPHA = 1.0
MESH_COLOR = (0.82, 0.82, 0.82)
TUMOR_ALPHA = 0.85
SNFH_ALPHA = 0.35
SLICE_ALPHA = 0.95
SPECULAR = 0.3
CMAP = "gray"
WINDOW_SIZE = (1400, 1200)
MARGIN = 0.05
# ==================


def _camera_to_angles(
    plotter, focus: Tuple[float, float, float]
) -> Tuple[float, float]:
    """Extract azimuth and elevation from current camera position."""
    pos = np.array(plotter.camera.position)
    foc = np.array(focus)
    d = pos - foc
    r = np.linalg.norm(d)
    if r < 1e-6:
        return 0.0, 0.0
    elev_deg = float(np.degrees(np.arcsin(d[2] / r)))
    azim_deg = float(np.degrees(np.arctan2(d[1], d[0])))
    return azim_deg, elev_deg


def _octant_to_str(octant: Tuple[bool, bool, bool]) -> str:
    """Convert octant bools to CLI string (e.g., (True, False, False) -> 'npp')."""
    return "".join("n" if inv else "p" for inv in octant)


def _build_command(
    subject_dir: Path,
    az: float,
    el: float,
    octant: Tuple[bool, bool, bool],
    cfg: RenderConfig,
    modality: str,
    output: str = "<OUTPUT>",
) -> str:
    """Build a full visualize_meningioma_octant.py CLI command string."""
    parts = [
        "python scripts/visualize_meningioma_octant.py",
        f"  --subject-dir {subject_dir}",
        f"  --modality {modality}",
        f"  --octant {_octant_to_str(octant)}",
        f"  --camera-azimuth {az:.1f}",
        f"  --camera-elevation {el:.1f}",
        f"  --zoom {cfg.zoom}",
        f"  --mesh-alpha {cfg.mesh_alpha}",
        f"  --mesh-color {cfg.mesh_color[0]:.2f}",
        f"  --tumor-alpha {cfg.tumor_alpha}",
        f"  --snfh-alpha {MENINGIOMA_ALPHA[2]:.2f}",
        f"  --slice-alpha {cfg.slice_alpha}",
        f"  --specular {cfg.specular}",
        f"  --cmap {cfg.cmap}",
        f"  --margin {MARGIN}",
        f"  --window-size {cfg.window_size[0]} {cfg.window_size[1]}",
        f"  --output {output}",
    ]
    return " \\\n".join(parts)


def main() -> None:
    """Main entry point."""
    subject_dir = Path(ROOT) / PATIENT / STUDY
    vol = load_volume(subject_dir / f"{MOD}.nii.gz")
    seg = load_volume(subject_dir / "seg.nii.gz").astype(np.int32)
    mask = compute_brain_mask(vol)
    si, octant = find_optimal_octant(vol, seg, margin_frac=MARGIN)

    MENINGIOMA_ALPHA[2] = SNFH_ALPHA

    cfg = RenderConfig(
        octant=octant,
        zoom=ZOOM,
        camera_azimuth=AZIM,
        camera_elevation=ELEV,
        mesh_alpha=MESH_ALPHA,
        mesh_color=MESH_COLOR,
        tumor_alpha=TUMOR_ALPHA,
        slice_alpha=SLICE_ALPHA,
        specular=SPECULAR,
        cmap=CMAP,
        window_size=WINDOW_SIZE,
    )
    p = render_meningioma_octant(
        vol, seg, si, cfg=cfg, brain_mask=mask, off_screen=False
    )
    p.set_background("black")

    # Save focus point set by render function (brain center)
    focus = p.camera.focal_point

    # Key callback: press 'p' to print full reproducible command
    def print_params() -> None:
        az, el = _camera_to_angles(p, focus)
        # Current zoom from camera (PyVista parallel scale changes, but
        # view angle stays fixed — re-derive effective zoom from distance)
        print()
        print("=" * 70)
        print(f"  azimuth={az:.1f}  elevation={el:.1f}  zoom={cfg.zoom}")
        print(f"  octant={_octant_to_str(octant)}  modality={MOD}")
        print()
        print(_build_command(subject_dir, az, el, octant, cfg, MOD))
        print("=" * 70)

    p.add_key_event("p", print_params)

    print()
    print("=" * 70)
    print(f"Patient: {PATIENT}   Study: {STUDY}   Mod: {MOD}")
    print(f"Octant:  {_octant_to_str(octant)}  ({octant})")
    print()
    print("Controls:")
    print("  - Rotate/pan/zoom the 3D view")
    print("  - Press 'p' to print full CLI command")
    print("  - Close the window to print final command")
    print("=" * 70)

    p.show()

    # After window closes, print final command
    az, el = _camera_to_angles(p, focus)
    print()
    print("=" * 70)
    print("FINAL COMMAND (copy-paste to generate figure):")
    print()
    print(_build_command(subject_dir, az, el, octant, cfg, MOD))
    print("=" * 70)


if __name__ == "__main__":
    main()
