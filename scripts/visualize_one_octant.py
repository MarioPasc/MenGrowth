#!/usr/bin/env python3
"""Interactive single-octant preview for tuning camera angles.

Rotate the view in the interactive window, then close it.
The script prints the azimuth/elevation to paste into SHOWCASE_PATIENTS.

You can also press 'p' at any time to print current camera angles.

Usage:
    python scripts/visualize_one_octant.py

Edit the PATIENT / STUDY / MOD / AZIM / ELEV variables below.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
from visualize_meningioma_octant import (
    RenderConfig,
    compute_brain_mask,
    find_optimal_octant,
    load_volume,
    render_meningioma_octant,
)

ROOT = "/media/mpascual/PortableSSD/Meningiomas/MenGrowth/preprocessed/MenGrowth-2025"

# === EDIT THESE ===
PATIENT = "MenGrowth-0032"
STUDY = "MenGrowth-0032-001"
MOD = "t1c"
AZIM = None  # degrees, or None for auto
ELEV = None  # degrees, or None for auto
# ==================


def _camera_to_angles(plotter, focus):
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


def main() -> None:
    d = Path(ROOT) / PATIENT / STUDY
    vol = load_volume(d / f"{MOD}.nii.gz")
    seg = load_volume(d / "seg.nii.gz").astype(np.int32)
    mask = compute_brain_mask(vol)
    si, octant = find_optimal_octant(vol, seg)

    cfg = RenderConfig(
        octant=octant, zoom=2.5, camera_azimuth=AZIM, camera_elevation=ELEV
    )
    p = render_meningioma_octant(
        vol, seg, si, cfg=cfg, brain_mask=mask, off_screen=False
    )
    p.set_background("black")

    # Save focus point set by render function (brain center)
    focus = p.camera.focal_point

    # Key callback: press 'p' to print current angles
    def print_angles():
        az, el = _camera_to_angles(p, focus)
        print()
        print(f"  Current camera:  azimuth={az:.1f}  elevation={el:.1f}")
        print(f'  >>> ("{PATIENT}", "...", {az:.1f}, {el:.1f}),')

    p.add_key_event("p", print_angles)

    print()
    print("=" * 60)
    print(f"Patient: {PATIENT}   Study: {STUDY}   Mod: {MOD}")
    print(f"Octant:  {octant}")
    print()
    print("Rotate the view, then:")
    print("  - Press 'p' to print current azimuth/elevation")
    print("  - Close the window to print final angles")
    print("=" * 60)

    p.show()

    # After window closes, print final angles
    az, el = _camera_to_angles(p, focus)
    print()
    print("=" * 60)
    print(f"FINAL camera:  azimuth={az:.1f}  elevation={el:.1f}")
    print()
    print(">>> Copy into SHOWCASE_PATIENTS:")
    print(f'    ("{PATIENT}", "...", {az:.1f}, {el:.1f}),')
    print("=" * 60)


if __name__ == "__main__":
    main()
