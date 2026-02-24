#!/usr/bin/env python3
"""
visualize_meningioma_octant.py - 3D octant visualization for BraTS meningioma data.

Creates a 3D PyVista visualization showing a brain surface with a cutaway octant
revealing three orthogonal MRI slices inside. Three meningioma tumor compartments
are rendered as colored surface meshes:

    Label 1: Non-Enhancing Tumor (NET)         — yellow (#DDCC77)
    Label 2: Surrounding FLAIR Hyperintensity   — teal   (#44AA99)
    Label 3: Enhancing Tumor (ET)               — red    (#CC3311)

Input: BraTS-MEN subject directory containing {id}-{modality}.nii.gz + {id}-seg.nii.gz

Usage:
    # Single subject (auto-center on tumor):
    python scripts/visualize_meningioma_octant.py \
        --subject-dir /path/to/BraTS-MEN-00027-000 \
        --output /path/to/output.png

    # Batch mode (all subjects in a directory):
    python scripts/visualize_meningioma_octant.py \
        --dataset /path/to/BraTS_Men_Train \
        --output-dir /path/to/visualizations

    # Custom camera / octant / modality:
    python scripts/visualize_meningioma_octant.py \
        --subject-dir /path/to/BraTS-MEN-00027-000 \
        --modality t1c \
        --octant npp \
        --camera-azimuth 130 --camera-elevation 15 --zoom 2.4 \
        --output output.png
"""

from __future__ import annotations

import argparse
import logging
import pathlib
from dataclasses import dataclass
from typing import Any, Dict, Final, List, Optional, Tuple

import numpy as np

try:
    import nibabel as nib
except ModuleNotFoundError:
    nib = None

try:
    from scipy import ndimage

    HAS_SCIPY = True
except ModuleNotFoundError:
    HAS_SCIPY = False

try:
    import pyvista as pv
    from skimage import measure

    HAS_PYVISTA = True
except ModuleNotFoundError:
    HAS_PYVISTA = False
    pv = None

logger = logging.getLogger(__name__)

# =============================================================================
# BraTS Meningioma Label Configuration
# =============================================================================

# BraTS-MEN segmentation labels
MENINGIOMA_LABELS: Final[Dict[int, str]] = {
    1: "Non-Enhancing Tumor (NET)",
    2: "SNFH",
    3: "Enhancing Tumor (ET)",
}

# Colorblind-friendly palette (Paul Tol's vibrant)
MENINGIOMA_COLORS: Final[Dict[int, Tuple[float, float, float]]] = {
    1: (0.867, 0.800, 0.467),  # Yellow  (#DDCC77) — NET
    2: (0.267, 0.667, 0.600),  # Teal    (#44AA99) — SNFH
    3: (0.800, 0.200, 0.067),  # Red     (#CC3311) — ET
}

# Per-label opacity: ET and NET are fully opaque, SNFH is semi-transparent
# so the tumor core is visible through the surrounding edema.
# SNFH alpha can be overridden via --snfh-alpha CLI flag.
MENINGIOMA_ALPHA: Dict[int, float] = {
    1: 0.7,  # NET — fully opaque (tumor core)
    2: 0.5,  # SNFH — semi-transparent (surrounding edema/hyperintensity)
    3: 1.0,  # ET — fully opaque (enhancing tumor core)
}

# Valid BraTS modalities
VALID_MODALITIES: Final[List[str]] = ["t1c", "t1n", "t2w", "t2f"]


@dataclass(frozen=True)
class RenderConfig:
    """Configuration for PyVista surface rendering."""

    iso_level: float = 0.5
    mesh_alpha: float = 1.0
    mesh_color: Tuple[float, float, float] = (0.82, 0.82, 0.82)
    slice_alpha: float = 0.95
    cmap: str = "gray"
    specular: float = 0.3
    specular_power: float = 20.0
    plane_bias: float = 0.01
    smooth_iterations: int = 5
    smooth_lambda: float = 0.5
    # Tumor surface rendering
    tumor_alpha: float = 0.85
    tumor_smooth_iterations: int = 3
    # Octant: (invert_x, invert_y, invert_z). False = cut positive side
    octant: Tuple[bool, bool, bool] = (False, False, False)
    # Camera
    camera_azimuth: Optional[float] = None
    camera_elevation: Optional[float] = None
    zoom: float = 2.2
    # Window
    window_size: Tuple[int, int] = (1400, 1200)


# =============================================================================
# Data I/O
# =============================================================================


def resolve_brats_paths(
    subject_dir: pathlib.Path, modality: str = "t1c"
) -> Tuple[pathlib.Path, pathlib.Path]:
    """Resolve image and segmentation paths from a BraTS-MEN subject directory.

    Args:
        subject_dir: Path to BraTS subject directory (e.g., BraTS-MEN-00027-000/)
        modality: MRI modality to load (t1c, t1n, t2w, t2f)

    Returns:
        Tuple of (image_path, segmentation_path)

    Raises:
        FileNotFoundError: If files don't exist
        ValueError: If invalid modality
    """
    if modality not in VALID_MODALITIES:
        raise ValueError(
            f"Invalid modality '{modality}'. Must be one of {VALID_MODALITIES}"
        )

    subject_id = subject_dir.name
    image_path = subject_dir / f"{subject_id}-{modality}.nii.gz"
    seg_path = subject_dir / f"{subject_id}-seg.nii.gz"

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not seg_path.exists():
        raise FileNotFoundError(f"Segmentation not found: {seg_path}")

    return image_path, seg_path


def load_volume(path: pathlib.Path) -> np.ndarray:
    """Load a 3D NIfTI volume, converting to RAS orientation.

    Args:
        path: Path to NIfTI file

    Returns:
        3D float64 array in RAS orientation
    """
    if nib is None:
        raise ImportError("Reading NIfTI requires nibabel: pip install nibabel")
    img = nib.load(str(path))
    img = nib.as_closest_canonical(img)
    return np.asanyarray(img.get_fdata())


# =============================================================================
# Brain Mask & Mesh Helpers
# =============================================================================


def compute_brain_mask(
    volume: np.ndarray, threshold_percentile: float = 5.0
) -> np.ndarray:
    """Compute a brain mask via thresholding + morphological cleanup.

    Args:
        volume: 3D MRI volume
        threshold_percentile: Intensity percentile for foreground detection

    Returns:
        Binary mask array
    """
    if not HAS_SCIPY:
        raise ImportError("Brain mask computation requires scipy")

    finite_vals = volume[np.isfinite(volume)]
    if finite_vals.size == 0:
        return np.ones(volume.shape, dtype=bool)

    threshold = np.percentile(finite_vals, threshold_percentile)
    mask = volume > threshold

    struct = ndimage.generate_binary_structure(3, 1)
    mask = ndimage.binary_fill_holes(mask)
    mask = ndimage.binary_opening(mask, structure=struct, iterations=2)

    labeled, num_features = ndimage.label(mask)
    if num_features > 1:
        sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
        largest = np.argmax(sizes) + 1
        mask = labeled == largest

    mask = ndimage.binary_closing(mask, structure=struct, iterations=2)
    return mask.astype(bool)


def laplacian_smooth(
    vertices: np.ndarray, faces: np.ndarray, iterations: int = 5, lam: float = 0.5
) -> np.ndarray:
    """Apply Laplacian smoothing to a triangle mesh.

    Args:
        vertices: (N, 3) vertex positions
        faces: (F, 3) triangle indices
        iterations: Smoothing iterations
        lam: Smoothing factor (0=none, 1=full averaging)

    Returns:
        Smoothed vertex positions
    """
    n_verts = len(vertices)
    verts = vertices.copy()

    adjacency: List[set] = [set() for _ in range(n_verts)]
    for f in faces:
        for idx in range(3):
            adjacency[f[idx]].add(f[(idx + 1) % 3])
            adjacency[f[idx]].add(f[(idx + 2) % 3])

    for _ in range(iterations):
        new_verts = verts.copy()
        for idx in range(n_verts):
            neighbors = list(adjacency[idx])
            if neighbors:
                centroid = verts[neighbors].mean(axis=0)
                new_verts[idx] = (1 - lam) * verts[idx] + lam * centroid
        verts = new_verts

    return verts


def faces_to_pyvista(faces_tri: np.ndarray) -> np.ndarray:
    """Convert (F, 3) triangle array to PyVista face format [3, i, j, k, ...]."""
    f = np.hstack(
        [
            np.full((faces_tri.shape[0], 1), 3, dtype=np.int64),
            faces_tri.astype(np.int64),
        ]
    )
    return f.ravel()


def find_tumor_center(segmentation: np.ndarray) -> Tuple[int, int, int]:
    """Find the centroid of the tumor (all non-zero labels).

    Args:
        segmentation: 3D label array

    Returns:
        (x, y, z) centroid voxel indices
    """
    coords = np.array(np.where(segmentation > 0))
    if coords.size == 0:
        raise ValueError("No tumor found in segmentation")
    center = coords.mean(axis=1).astype(int)
    return tuple(center)


def find_optimal_octant(
    volume: np.ndarray,
    segmentation: np.ndarray,
    margin_frac: float = 0.05,
) -> Tuple[Tuple[int, int, int], Tuple[bool, bool, bool]]:
    """Find optimal slice indices and octant orientation for tumor visualization.

    For meningiomas, positions the octant cut to reveal the maximum tumor
    cross-section. The slices are placed at the inner edge of the tumor
    (closest to brain center) so the entire tumor is exposed in the octant.

    Args:
        volume: 3D MRI volume
        segmentation: 3D label array
        margin_frac: Extra margin beyond tumor inner edge (fraction of brain extent)

    Returns:
        Tuple of (slice_indices_kij, octant_bools)
    """
    nx, ny, nz = volume.shape

    if not np.any(segmentation > 0):
        return (nz // 2, nx // 2, ny // 2), (False, False, False)

    cx, cy, cz = find_tumor_center(segmentation)

    # Determine which octant to cut: cut toward the tumor hemisphere
    inv_x = cx < nx // 2
    inv_y = cy < ny // 2
    inv_z = cz < nz // 2

    # Find tumor bounding box to position slices at the inner edge
    tumor_coords = np.argwhere(segmentation > 0)
    t_min = tumor_coords.min(axis=0)  # (x_min, y_min, z_min)
    t_max = tumor_coords.max(axis=0)  # (x_max, y_max, z_max)

    # Margin in voxels
    margin = np.array([nx, ny, nz]) * margin_frac

    # For each axis, place the slice at the inner edge of the tumor
    # (the side closest to brain center) with some margin.
    # Cap at 35% from the cut side so we never remove more than ~65% of the brain.
    cap_min = np.array([nx, ny, nz]) * 0.35
    cap_max = np.array([nx, ny, nz]) * 0.65

    if inv_x:
        sx = int(np.clip(t_max[0] + margin[0], cap_min[0], nx - 2))
    else:
        sx = int(np.clip(t_min[0] - margin[0], 1, cap_max[0]))

    if inv_y:
        sy = int(np.clip(t_max[1] + margin[1], cap_min[1], ny - 2))
    else:
        sy = int(np.clip(t_min[1] - margin[1], 1, cap_max[1]))

    if inv_z:
        sz = int(np.clip(t_max[2] + margin[2], cap_min[2], nz - 2))
    else:
        sz = int(np.clip(t_min[2] - margin[2], 1, cap_max[2]))

    # Slice indices: (k_axial, i_coronal, j_sagittal)
    logger.info(
        f"  Tumor bbox: x=[{t_min[0]},{t_max[0]}] y=[{t_min[1]},{t_max[1]}] z=[{t_min[2]},{t_max[2]}]"
    )
    return (sz, sx, sy), (inv_x, inv_y, inv_z)


# =============================================================================
# PyVista Rendering
# =============================================================================


def render_meningioma_octant(
    volume: np.ndarray,
    segmentation: np.ndarray,
    slice_indices: Tuple[int, int, int],
    *,
    cfg: RenderConfig = RenderConfig(),
    brain_mask: Optional[np.ndarray] = None,
    save: Optional[pathlib.Path] = None,
    off_screen: bool = True,
) -> Any:
    """Render 3D octant visualization with brain surface and multi-label tumor.

    Args:
        volume: 3D MRI volume (nx, ny, nz), RAS orientation
        segmentation: 3D integer label array (0=bg, 1=NET, 2=SNFH, 3=ET)
        slice_indices: (k_axial, i_coronal, j_sagittal)
        cfg: Rendering configuration
        brain_mask: Pre-computed brain mask (if None, computed from volume)
        save: Output file path (PNG or PDF)
        off_screen: Render without display window

    Returns:
        pv.Plotter object
    """
    if not HAS_PYVISTA:
        raise ImportError("Requires: pip install pyvista scikit-image scipy")

    nx, ny, nz = volume.shape
    k_a, i_c, j_s = slice_indices

    margin = 1
    k = int(np.clip(k_a, margin, nz - 1 - margin))
    i = int(np.clip(i_c, margin, nx - 1 - margin))
    j = int(np.clip(j_s, margin, ny - 1 - margin))

    logger.info(f"Volume shape: {volume.shape}, octant origin: (i={i}, j={j}, k={k})")

    # Brain mask
    if brain_mask is not None:
        mask = brain_mask
        logger.info(f"Brain mask (provided): {mask.sum()} voxels")
    else:
        mask = compute_brain_mask(volume)
    logger.info(
        f"Brain mask: {mask.sum()} voxels ({100 * mask.sum() / mask.size:.1f}%)"
    )

    # Masked volume for slice rendering
    vol_masked = volume.copy().astype(np.float32)
    vol_masked[~mask] = np.nan

    inside = vol_masked[np.isfinite(vol_masked)]
    if inside.size > 0:
        vmin, vmax = (
            float(np.percentile(inside, [1.0, 99.0])[0]),
            float(np.percentile(inside, [1.0, 99.0])[1]),
        )
    else:
        vmin, vmax = float(np.nanmin(vol_masked)), float(np.nanmax(vol_masked))

    # ── Plotter setup ──
    pv.global_theme.background = "white"
    plotter = pv.Plotter(
        off_screen=off_screen or (save is not None),
        window_size=list(cfg.window_size),
    )

    try:
        plotter.enable_anti_aliasing("msaa")
    except Exception:
        pass
    try:
        plotter.enable_depth_peeling()
    except Exception:
        pass

    # ── Brain surface mesh ──
    verts, faces, _, _ = measure.marching_cubes(
        mask.astype(np.float32), level=cfg.iso_level
    )
    verts = laplacian_smooth(
        verts, faces, iterations=cfg.smooth_iterations, lam=cfg.smooth_lambda
    )
    mesh_full = pv.PolyData(verts, faces_to_pyvista(faces))

    # Octant clipping bounds
    inv_x, inv_y, inv_z = cfg.octant
    bx = (-0.5, i + 0.5) if inv_x else (i - 0.5, nx - 0.5)
    by = (-0.5, j + 0.5) if inv_y else (j - 0.5, ny - 0.5)
    bz = (-0.5, k + 0.5) if inv_z else (k - 0.5, nz - 0.5)
    bounds = (bx[0], bx[1], by[0], by[1], bz[0], bz[1])

    mesh_clip = mesh_full.clip_box(bounds=bounds, invert=True, merge_points=True)
    logger.info(f"Clipped brain mesh: {mesh_clip.n_points} vertices")

    plotter.add_mesh(
        mesh_clip,
        color=cfg.mesh_color,
        opacity=cfg.mesh_alpha,
        smooth_shading=True,
        specular=float(np.clip(cfg.specular, 0.0, 1.0)),
        specular_power=float(cfg.specular_power),
        show_edges=False,
    )

    # ── Tumor surface meshes (one per label) ──
    # Meningiomas are extra-axial: they grow from meninges on the brain surface.
    # We clip the tumor the same way as the brain so it appears on the visible
    # surface, and the octant cut reveals the tumor-brain interface on MRI slices.
    for label_id, color in MENINGIOMA_COLORS.items():
        label_mask = (segmentation == label_id).astype(np.float32)
        if label_mask.sum() < 10:
            logger.info(
                f"Label {label_id} ({MENINGIOMA_LABELS[label_id]}): too few voxels, skipping"
            )
            continue

        try:
            # Remove small disconnected components (BraTS segmentations often have
            # tiny scattered label fragments that create visual noise)
            if HAS_SCIPY:
                binary = label_mask > 0.5
                labeled_cc, n_cc = ndimage.label(binary)
                if n_cc > 1:
                    cc_sizes = ndimage.sum(binary, labeled_cc, range(1, n_cc + 1))
                    largest_cc = np.argmax(cc_sizes) + 1
                    # Keep only components >= 1% of the largest
                    min_size = max(100, cc_sizes[largest_cc - 1] * 0.05)
                    for cc_id in range(1, n_cc + 1):
                        if cc_sizes[cc_id - 1] < min_size:
                            label_mask[labeled_cc == cc_id] = 0.0
                    logger.debug(
                        f"  Label {label_id}: kept {sum(1 for s in cc_sizes if s >= min_size)}/{n_cc} components"
                    )

            # Slight Gaussian smooth for better surface
            if HAS_SCIPY:
                label_mask = ndimage.gaussian_filter(label_mask, sigma=0.5)

            t_verts, t_faces, _, _ = measure.marching_cubes(label_mask, level=0.5)

            if cfg.tumor_smooth_iterations > 0:
                t_verts = laplacian_smooth(
                    t_verts,
                    t_faces,
                    iterations=cfg.tumor_smooth_iterations,
                    lam=cfg.smooth_lambda,
                )

            tumor_mesh = pv.PolyData(t_verts, faces_to_pyvista(t_faces))

            # Render the FULL tumor mesh (no octant clipping). Meningiomas are
            # large extra-axial tumors — they should always be fully visible,
            # with the octant MRI slices visible behind/through them.
            # Per-label alpha: SNFH is semi-transparent so cores are visible through edema.
            label_alpha = MENINGIOMA_ALPHA.get(label_id, cfg.tumor_alpha)
            plotter.add_mesh(
                tumor_mesh,
                color=color,
                opacity=label_alpha,
                smooth_shading=True,
                show_edges=False,
            )

            logger.info(
                f"Label {label_id} ({MENINGIOMA_LABELS[label_id]}): "
                f"{tumor_mesh.n_points} verts, color={color}"
            )
        except Exception as e:
            logger.warning(f"Could not create surface for label {label_id}: {e}")

    # ── MRI slice surfaces ──
    grid = pv.ImageData()
    grid.dimensions = np.array(vol_masked.shape, dtype=int) + 1
    grid.spacing = (1.0, 1.0, 1.0)
    grid.origin = (-0.5, -0.5, -0.5)
    grid.cell_data["I"] = vol_masked.ravel(order="F")
    grid = grid.cell_data_to_point_data(pass_cell_data=True)

    clip_x_normal = (-1, 0, 0) if inv_x else (1, 0, 0)
    clip_y_normal = (0, -1, 0) if inv_y else (0, 1, 0)
    clip_z_normal = (0, 0, -1) if inv_z else (0, 0, 1)

    slice_kwargs = dict(
        scalars="I",
        cmap=cfg.cmap,
        clim=(vmin, vmax),
        opacity=cfg.slice_alpha,
        nan_opacity=0.0,
        show_scalar_bar=False,
    )

    # Axial (XY @ z=k)
    z0 = float(k) + (cfg.plane_bias if not inv_z else -cfg.plane_bias)
    slc = grid.slice(normal=(0, 0, 1), origin=(0.0, 0.0, z0))
    slc = slc.clip(normal=clip_x_normal, origin=(float(i), 0.0, 0.0), invert=False)
    slc = slc.clip(normal=clip_y_normal, origin=(0.0, float(j), 0.0), invert=False)
    plotter.add_mesh(slc, **slice_kwargs)

    # Coronal (YZ @ x=i)
    x0 = float(i) + (cfg.plane_bias if not inv_x else -cfg.plane_bias)
    slc = grid.slice(normal=(1, 0, 0), origin=(x0, 0.0, 0.0))
    slc = slc.clip(normal=clip_y_normal, origin=(0.0, float(j), 0.0), invert=False)
    slc = slc.clip(normal=clip_z_normal, origin=(0.0, 0.0, float(k)), invert=False)
    plotter.add_mesh(slc, **slice_kwargs)

    # Sagittal (XZ @ y=j)
    y0 = float(j) + (cfg.plane_bias if not inv_y else -cfg.plane_bias)
    slc = grid.slice(normal=(0, 1, 0), origin=(0.0, y0, 0.0))
    slc = slc.clip(normal=clip_x_normal, origin=(float(i), 0.0, 0.0), invert=False)
    slc = slc.clip(normal=clip_z_normal, origin=(0.0, 0.0, float(k)), invert=False)
    plotter.add_mesh(slc, **slice_kwargs)

    # ── Camera ──
    if mask.any():
        idx_arr = np.argwhere(mask)
        (xmin, ymin, zmin), (xmax, ymax, zmax) = idx_arr.min(0), idx_arr.max(0)
        center = ((xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2)
        brain_size = max(xmax - xmin, ymax - ymin, zmax - zmin)
    else:
        center = (nx / 2, ny / 2, nz / 2)
        brain_size = max(nx, ny, nz)

    plotter.set_focus(center)
    dist = 1.8 * brain_size

    if cfg.camera_azimuth is not None and cfg.camera_elevation is not None:
        azim_rad = np.radians(cfg.camera_azimuth)
        elev_rad = np.radians(cfg.camera_elevation)
        cam_pos = (
            center[0] + dist * np.cos(elev_rad) * np.cos(azim_rad),
            center[1] + dist * np.cos(elev_rad) * np.sin(azim_rad),
            center[2] + dist * np.sin(elev_rad),
        )
    else:
        x_sign = -1 if inv_x else 1
        y_sign = -1 if inv_y else 1
        z_sign = 0.7 if inv_z else 0.9
        cam_pos = (
            center[0] + x_sign * dist * 0.7,
            center[1] + y_sign * dist * 0.7,
            center[2] + z_sign * dist * 0.5,
        )

    plotter.set_position(cam_pos)
    plotter.set_viewup((0, 0, 1))
    plotter.camera.SetViewAngle(35)
    plotter.reset_camera_clipping_range()

    if cfg.zoom != 1.0:
        plotter.camera.zoom(cfg.zoom)

    logger.info(f"Camera: pos={cam_pos}, focus={center}, zoom={cfg.zoom}")

    # ── Save ──
    if save is not None:
        save.parent.mkdir(parents=True, exist_ok=True)
        suffix = save.suffix.lower()
        if suffix == ".pdf":
            plotter.save_graphic(str(save))
        else:
            plotter.screenshot(str(save))
        logger.info(f"Saved: {save}")

    return plotter


def create_legend(
    save: Optional[pathlib.Path] = None,
    figsize: Tuple[float, float] = (6, 0.7),
) -> None:
    """Create a standalone legend for meningioma tumor labels.

    Args:
        save: Output path for legend image
        figsize: Figure dimensions in inches
    """
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    handles = [
        mpatches.Patch(facecolor=MENINGIOMA_COLORS[lid], edgecolor="black", label=name)
        for lid, name in MENINGIOMA_LABELS.items()
    ]

    fig, ax = plt.subplots(figsize=figsize)
    ax.legend(
        handles=handles,
        loc="center",
        ncol=len(handles),
        frameon=False,
        handlelength=1.5,
        fontsize=11,
    )
    ax.axis("off")

    if save is not None:
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=300, bbox_inches="tight", transparent=True)
        logger.info(f"Saved legend: {save}")

    plt.close(fig)


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="3D octant visualization for BraTS meningioma data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input
    input_grp = parser.add_argument_group("Input")
    input_grp.add_argument(
        "--subject-dir",
        type=pathlib.Path,
        help="Path to single BraTS-MEN subject directory",
    )
    input_grp.add_argument(
        "--dataset",
        type=pathlib.Path,
        help="Path to BraTS dataset directory (batch mode, processes all subjects)",
    )
    input_grp.add_argument(
        "--modality",
        type=str,
        default="t1c",
        choices=VALID_MODALITIES,
        help="MRI modality to visualize (default: t1c)",
    )
    input_grp.add_argument(
        "--max-subjects",
        type=int,
        default=None,
        help="Max number of subjects to process in batch mode",
    )

    # Slice / octant
    slice_grp = parser.add_argument_group("Slice selection")
    slice_grp.add_argument("--axial", "-z", type=int, default=None)
    slice_grp.add_argument("--coronal", "-x", type=int, default=None)
    slice_grp.add_argument("--sagittal", "-y", type=int, default=None)
    slice_grp.add_argument(
        "--octant",
        type=str,
        default=None,
        help=(
            "Octant to cut as 3-char string: p=positive, n=negative per axis (xyz). "
            "E.g., 'npp' cuts the -x,+y,+z octant. "
            "Auto-detected from tumor location if not specified."
        ),
    )

    # Rendering
    vis_grp = parser.add_argument_group("Rendering")
    vis_grp.add_argument(
        "--mesh-color",
        type=str,
        default=None,
        help="Brain surface color as hex (#888888) or gray float 0-1 (default: 0.82)",
    )
    vis_grp.add_argument(
        "--mesh-alpha",
        type=float,
        default=1.0,
        help="Brain surface opacity (default: 1.0)",
    )
    vis_grp.add_argument(
        "--tumor-alpha",
        type=float,
        default=0.85,
        help="Default tumor opacity (default: 0.85)",
    )
    vis_grp.add_argument(
        "--snfh-alpha",
        type=float,
        default=0.35,
        help="SNFH (label 2) opacity — semi-transparent to see cores through edema (default: 0.35)",
    )
    vis_grp.add_argument(
        "--cmap", type=str, default="gray", help="MRI slice colormap (default: gray)"
    )
    vis_grp.add_argument(
        "--slice-alpha",
        type=float,
        default=0.95,
        help="MRI slice opacity (default: 0.95)",
    )
    vis_grp.add_argument(
        "--specular",
        type=float,
        default=0.3,
        help="Surface specular reflection (default: 0.3)",
    )
    vis_grp.add_argument(
        "--camera-azimuth", type=float, default=None, help="Camera azimuth in degrees"
    )
    vis_grp.add_argument(
        "--camera-elevation",
        type=float,
        default=None,
        help="Camera elevation in degrees",
    )
    vis_grp.add_argument(
        "--zoom",
        type=float,
        default=2.2,
        help="Zoom factor (>1 = closer, default: 2.2)",
    )
    vis_grp.add_argument(
        "--margin",
        type=float,
        default=0.05,
        help="Octant margin beyond tumor edge as fraction of brain size (default: 0.05)",
    )
    vis_grp.add_argument(
        "--window-size",
        type=int,
        nargs=2,
        default=[1400, 1200],
        help="Render window size in pixels (default: 1400 1200)",
    )

    # Output
    out_grp = parser.add_argument_group("Output")
    out_grp.add_argument(
        "--output", "-o", type=pathlib.Path, help="Output file (single subject mode)"
    )
    out_grp.add_argument(
        "--output-dir", type=pathlib.Path, help="Output directory (batch mode)"
    )
    out_grp.add_argument(
        "--legend", action="store_true", help="Also generate a standalone legend"
    )
    out_grp.add_argument(
        "--all-sequences",
        action="store_true",
        help=(
            "Generate one image per modality (t1c, t1n, t2w, t2f) with identical POV. "
            "Requires --output-dir. Brain mask is computed once from t1c and reused."
        ),
    )
    out_grp.add_argument("--verbose", "-v", action="store_true")

    return parser.parse_args()


def process_subject(
    subject_dir: pathlib.Path,
    modality: str,
    output_path: pathlib.Path,
    cfg: RenderConfig,
    *,
    axial: Optional[int] = None,
    coronal: Optional[int] = None,
    sagittal: Optional[int] = None,
    margin_frac: float = 0.05,
    brain_mask: Optional[np.ndarray] = None,
) -> bool:
    """Process a single BraTS-MEN subject.

    Args:
        subject_dir: Path to subject directory
        modality: MRI modality
        output_path: Where to save the visualization
        cfg: Render configuration
        axial/coronal/sagittal: Optional manual slice indices
        margin_frac: Octant margin as fraction of brain extent
        brain_mask: Pre-computed brain mask for consistent POV across modalities

    Returns:
        True if successful, False otherwise
    """
    subject_id = subject_dir.name

    try:
        image_path, seg_path = resolve_brats_paths(subject_dir, modality)
    except (FileNotFoundError, ValueError) as e:
        logger.warning(f"[{subject_id}] Skipping: {e}")
        return False

    logger.info(f"[{subject_id}] Loading {modality}: {image_path.name}")
    volume = load_volume(image_path)
    segmentation = load_volume(seg_path).astype(np.int32)

    # Log tumor label stats
    for lid, name in MENINGIOMA_LABELS.items():
        count = int(np.sum(segmentation == lid))
        if count > 0:
            logger.info(f"  Label {lid} ({name}): {count} voxels")

    # Determine slice indices and octant
    if all(v is not None for v in [axial, coronal, sagittal]):
        slice_indices = (axial, coronal, sagittal)
    else:
        slice_indices, auto_octant = find_optimal_octant(
            volume, segmentation, margin_frac=margin_frac
        )
        # Only override octant if not manually specified
        if cfg.octant == (False, False, False) and cfg.camera_azimuth is None:
            cfg = RenderConfig(
                **{
                    **{
                        f.name: getattr(cfg, f.name)
                        for f in cfg.__dataclass_fields__.values()
                    },
                    "octant": auto_octant,
                }
            )
        logger.info(
            f"  Auto slices: z={slice_indices[0]}, x={slice_indices[1]}, y={slice_indices[2]}"
        )

    logger.info(
        f"  Octant: inv_x={cfg.octant[0]}, inv_y={cfg.octant[1]}, inv_z={cfg.octant[2]}"
    )

    render_meningioma_octant(
        volume,
        segmentation,
        slice_indices,
        cfg=cfg,
        brain_mask=brain_mask,
        save=output_path,
        off_screen=True,
    )

    print(f"[OK] {subject_id} -> {output_path}")
    return True


def main() -> None:
    """Main entry point."""
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    # Build render config
    mesh_color = RenderConfig.mesh_color
    if args.mesh_color is not None:
        if args.mesh_color.startswith("#"):
            import matplotlib.colors as mcolors

            mesh_color = mcolors.to_rgb(args.mesh_color)
        else:
            g = float(args.mesh_color)
            mesh_color = (g, g, g)

    # Parse octant: 'n' = negative (invert=True), 'p' = positive (invert=False)
    octant_manual = False
    octant = (False, False, False)  # default, will be auto-detected
    if args.octant is not None:
        o = args.octant.lower().strip()
        if len(o) != 3 or not all(c in "np" for c in o):
            raise ValueError(
                f"--octant must be 3 chars of 'n' or 'p' (e.g., 'npp'), got '{args.octant}'"
            )
        octant = (o[0] == "n", o[1] == "n", o[2] == "n")
        octant_manual = True

    # Override per-label alpha for SNFH from CLI
    MENINGIOMA_ALPHA[2] = args.snfh_alpha

    cfg = RenderConfig(
        mesh_color=mesh_color,
        mesh_alpha=args.mesh_alpha,
        tumor_alpha=args.tumor_alpha,
        cmap=args.cmap,
        slice_alpha=args.slice_alpha,
        specular=args.specular,
        octant=octant,
        camera_azimuth=args.camera_azimuth,
        camera_elevation=args.camera_elevation,
        zoom=args.zoom,
        window_size=tuple(args.window_size),
    )

    # Single subject mode
    if args.subject_dir is not None:
        if args.all_sequences:
            # Generate all 4 modalities with identical POV (shared brain mask from t1c)
            out_dir = args.output_dir or args.subject_dir.parent
            out_dir.mkdir(parents=True, exist_ok=True)
            subject_id = args.subject_dir.name

            # Compute brain mask once from t1c for consistent surface across modalities
            ref_path, seg_path = resolve_brats_paths(args.subject_dir, "t1c")
            ref_vol = load_volume(ref_path)
            shared_mask = compute_brain_mask(ref_vol)
            seg = load_volume(seg_path).astype(np.int32)

            # Compute slice indices and octant once from the reference volume
            if all(v is not None for v in [args.axial, args.coronal, args.sagittal]):
                shared_slices = (args.axial, args.coronal, args.sagittal)
            else:
                shared_slices, auto_octant = find_optimal_octant(
                    ref_vol, seg, margin_frac=args.margin
                )
                if not octant_manual and cfg.camera_azimuth is None:
                    cfg = RenderConfig(
                        **{
                            **{
                                f.name: getattr(cfg, f.name)
                                for f in cfg.__dataclass_fields__.values()
                            },
                            "octant": auto_octant,
                        }
                    )

            # Determine output extension from --output if given, else png
            ext = args.output.suffix if args.output else ".png"
            for mod in VALID_MODALITIES:
                out_path = out_dir / f"{subject_id}_{mod}_octant{ext}"
                print(f"[{mod}] Rendering {subject_id}...")
                try:
                    img_path, _ = resolve_brats_paths(args.subject_dir, mod)
                    volume = load_volume(img_path)
                    render_meningioma_octant(
                        volume,
                        seg,
                        shared_slices,
                        cfg=cfg,
                        brain_mask=shared_mask,
                        save=out_path,
                        off_screen=True,
                    )
                    print(f"  [OK] -> {out_path}")
                except (FileNotFoundError, ValueError) as e:
                    print(f"  [SKIP] {mod}: {e}")

            print(f"\nDone: all sequences -> {out_dir}")
        else:
            if args.output is None:
                args.output = pathlib.Path(f"{args.subject_dir.name}_octant.png")

            process_subject(
                args.subject_dir,
                args.modality,
                args.output,
                cfg,
                axial=args.axial,
                coronal=args.coronal,
                sagittal=args.sagittal,
                margin_frac=args.margin,
            )

    # Batch mode
    elif args.dataset is not None:
        if args.output_dir is None:
            raise ValueError("Batch mode requires --output-dir")

        args.output_dir.mkdir(parents=True, exist_ok=True)

        subjects = sorted(
            d
            for d in args.dataset.iterdir()
            if d.is_dir() and d.name.startswith("BraTS-MEN-")
        )

        if args.max_subjects is not None:
            subjects = subjects[: args.max_subjects]

        total = len(subjects)
        success = 0

        for idx, subj_dir in enumerate(subjects, 1):
            print(f"[{idx}/{total}] Processing {subj_dir.name}...")
            out_path = args.output_dir / f"{subj_dir.name}_{args.modality}_octant.png"

            if process_subject(
                subj_dir, args.modality, out_path, cfg, margin_frac=args.margin
            ):
                success += 1

        print(f"\nDone: {success}/{total} subjects visualized -> {args.output_dir}")

    else:
        raise ValueError("Must specify either --subject-dir or --dataset")

    # Legend
    if args.legend:
        legend_dir = (
            args.output_dir
            if args.output_dir
            else (args.output.parent if args.output else pathlib.Path("."))
        )
        create_legend(save=legend_dir / "meningioma_legend.png")


if __name__ == "__main__":
    main()
