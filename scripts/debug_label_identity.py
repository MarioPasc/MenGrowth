"""Diagnostic figure to visually identify what model labels 1 and 2 represent.

For a study with substantial amounts of both labels, renders a 3x3 grid:
  Row 1: T1c (post-contrast), T1n (pre-contrast), T2-FLAIR
  Row 2: T1c with label 1 (red) and label 2 (blue) overlaid
  Row 3: T1c-T1n difference (enhancement map) with label contours

If label 1 overlaps with the bright enhancing mass on T1c → label 1 = ET.
If label 2 overlaps with the bright enhancing mass on T1c → label 2 = ET.
The other label is SNFH (edema, bright on FLAIR, not enhancing).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation


def make_diagnostic(
    pid: str = "MenGrowth-0056",
    sid: str = "MenGrowth-0056-001",
    prep_root: str = "/media/mpascual/PortableSSD/Meningiomas/MenGrowth/v5_final/MenGrowth-2025",
    out: str = "outputs/synthseg_analysis/figures/debug_label_identity.png",
) -> None:
    root = Path(prep_root) / pid / sid
    t1c = nib.load(str(root / "t1c.nii.gz")).get_fdata(dtype=np.float32)
    t1n = nib.load(str(root / "t1n.nii.gz")).get_fdata(dtype=np.float32)
    t2f = nib.load(str(root / "t2f.nii.gz")).get_fdata(dtype=np.float32)
    seg = nib.load(str(root / "seg.nii.gz")).get_fdata().astype(int)

    mask1 = seg == 1
    mask2 = seg == 2

    # Pick axial slice with maximum combined tumor area
    combined = (mask1 | mask2).sum(axis=(0, 1))
    z = int(np.argmax(combined))

    t1c_s = t1c[:, :, z].T
    t1n_s = t1n[:, :, z].T
    t2f_s = t2f[:, :, z].T
    m1 = mask1[:, :, z].T
    m2 = mask2[:, :, z].T

    enh_map = t1c_s - t1n_s

    boundary1 = binary_dilation(m1, iterations=1) & ~m1
    boundary2 = binary_dilation(m2, iterations=1) & ~m2

    fig, axes = plt.subplots(3, 3, figsize=(16, 16))
    fig.suptitle(
        f"Label identity diagnostic: {pid}/{sid}  (axial slice z={z})\n"
        f"Label 1 voxels: {int(mask1.sum()):,}  |  Label 2 voxels: {int(mask2.sum()):,}",
        fontsize=14, fontweight="bold",
    )

    def show(ax, img, title, cmap="gray"):
        lo, hi = np.percentile(img, [1, 99])
        ax.imshow(img, cmap=cmap, origin="lower", vmin=lo, vmax=hi)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

    # Row 1: raw modalities
    show(axes[0, 0], t1c_s, "T1c (post-contrast)\nET is BRIGHT here")
    show(axes[0, 1], t1n_s, "T1n (pre-contrast)\nBaseline, no enhancement")
    show(axes[0, 2], t2f_s, "T2-FLAIR\nSNFH/edema is BRIGHT here")

    # Row 2: T1c with label overlays
    for col, (title, overlay_m1, overlay_m2) in enumerate([
        ("T1c + Label 1 (RED)", True, False),
        ("T1c + Label 2 (BLUE)", False, True),
        ("T1c + Both labels", True, True),
    ]):
        ax = axes[1, col]
        lo, hi = np.percentile(t1c_s, [1, 99])
        ax.imshow(t1c_s, cmap="gray", origin="lower", vmin=lo, vmax=hi)
        if overlay_m1:
            fill1 = np.zeros((*m1.shape, 4), dtype=np.float32)
            fill1[m1, :] = [1.0, 0.2, 0.15, 0.45]
            fill1[boundary1, :] = [1.0, 0.2, 0.15, 1.0]
            ax.imshow(fill1, origin="lower")
        if overlay_m2:
            fill2 = np.zeros((*m2.shape, 4), dtype=np.float32)
            fill2[m2, :] = [0.2, 0.6, 1.0, 0.45]
            fill2[boundary2, :] = [0.2, 0.6, 1.0, 1.0]
            ax.imshow(fill2, origin="lower")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

    # Row 3: enhancement map + FLAIR with overlays
    show(axes[2, 0], enh_map, "Enhancement map (T1c - T1n)\nBright = contrast uptake")
    ax_enh = axes[2, 1]
    lo, hi = np.percentile(enh_map, [1, 99])
    ax_enh.imshow(enh_map, cmap="gray", origin="lower", vmin=lo, vmax=hi)
    fill1 = np.zeros((*m1.shape, 4), dtype=np.float32)
    fill1[boundary1, :] = [1.0, 0.2, 0.15, 1.0]
    ax_enh.imshow(fill1, origin="lower")
    fill2 = np.zeros((*m2.shape, 4), dtype=np.float32)
    fill2[boundary2, :] = [0.2, 0.6, 1.0, 1.0]
    ax_enh.imshow(fill2, origin="lower")
    ax_enh.set_title("Enhancement + contours\nRED=label1, BLUE=label2", fontsize=12, fontweight="bold")
    ax_enh.set_xticks([])
    ax_enh.set_yticks([])

    ax_flair = axes[2, 2]
    lo, hi = np.percentile(t2f_s, [1, 99])
    ax_flair.imshow(t2f_s, cmap="gray", origin="lower", vmin=lo, vmax=hi)
    fill1 = np.zeros((*m1.shape, 4), dtype=np.float32)
    fill1[boundary1, :] = [1.0, 0.2, 0.15, 1.0]
    ax_flair.imshow(fill1, origin="lower")
    fill2 = np.zeros((*m2.shape, 4), dtype=np.float32)
    fill2[boundary2, :] = [0.2, 0.6, 1.0, 1.0]
    ax_flair.imshow(fill2, origin="lower")
    ax_flair.set_title("T2-FLAIR + contours\nRED=label1, BLUE=label2", fontsize=12, fontweight="bold")
    ax_flair.set_xticks([])
    ax_flair.set_yticks([])

    # Intensity stats annotation
    stats_text = (
        f"Label 1 (RED):  T1c={np.mean(t1c_s[m1]):.0f}  T1n={np.mean(t1n_s[m1]):.0f}"
        f"  enh={np.mean(t1c_s[m1])-np.mean(t1n_s[m1]):.0f}  T2f={np.mean(t2f_s[m1]):.0f}\n"
        f"Label 2 (BLUE): T1c={np.mean(t1c_s[m2]):.0f}  T1n={np.mean(t1n_s[m2]):.0f}"
        f"  enh={np.mean(t1c_s[m2])-np.mean(t1n_s[m2]):.0f}  T2f={np.mean(t2f_s[m2]):.0f}"
    ) if m1.any() and m2.any() else "One label empty on this slice"
    fig.text(
        0.5, 0.01, stats_text, ha="center", fontsize=11,
        fontfamily="monospace", bbox=dict(facecolor="lightyellow", alpha=0.9),
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    make_diagnostic()
