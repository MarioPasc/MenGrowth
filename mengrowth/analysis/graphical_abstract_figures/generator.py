"""Orchestrator: config -> loader -> renderers -> output files.

Generates standalone 2D slice figures for each pipeline step,
plus an optional combined grid overview. Supports multiple studies.

Output directory structure:
    {output_dir}/{study_id}/{step_name}/{view}/{modality}[_suffix].{fmt}
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from mengrowth.preprocessing.utils.settings import apply_ieee_style

from .config import GraphicalAbstractConfig
from .loader import ArchiveLoader, StepVolume
from .renderers_2d import (
    compute_slice_index,
    extract_slice,
    get_pre_reg_slice_frac,
    get_slice_frac,
    normalize_for_display,
    render_bias_field_step,
    render_registration_step,
    render_segmentation_overlay,
    render_skull_stripping_step,
    render_standard_slice,
)

logger = logging.getLogger(__name__)


def _clean_step_name(step_key: str) -> str:
    """Strip 'stepN_' prefix from an HDF5 group key.

    Args:
        step_key: e.g. "step2_bias_field_correction".

    Returns:
        Clean name, e.g. "bias_field_correction".
    """
    return step_key.split("_", 1)[1] if "_" in step_key else step_key


class GraphicalAbstractGenerator:
    """Generates publication-quality graphical abstract figures.

    Iterates over multiple studies, reads volumes from HDF5 archives
    and NIfTI artifacts, renders 2D slices per step, and saves
    standalone images in an organized directory tree.

    Args:
        config: GraphicalAbstractConfig instance.
    """

    def __init__(self, config: GraphicalAbstractConfig) -> None:
        self.config = config
        self.output_root = Path(config.output.output_dir)
        self.output_root.mkdir(parents=True, exist_ok=True)

        # Set up publication style once
        apply_ieee_style()
        plt.rcParams.update(
            {
                "savefig.dpi": config.output.dpi,
                "savefig.bbox": "tight",
                "savefig.pad_inches": 0.05,
            }
        )

    def run(self) -> List[Path]:
        """Generate figures for all configured studies.

        Returns:
            List of Path objects for all generated figure files.
        """
        all_saved: List[Path] = []

        study_ids = self.config.study_ids
        if not study_ids:
            logger.error("No study_ids configured")
            return all_saved

        for study_id in study_ids:
            logger.info("=" * 60)
            logger.info("Processing study: %s", study_id)
            logger.info("=" * 60)
            try:
                paths = self._run_study(study_id)
                all_saved.extend(paths)
            except Exception:
                logger.exception("Failed to process study %s", study_id)

        logger.info(
            "Generated %d total figures across %d studies in %s",
            len(all_saved),
            len(study_ids),
            self.output_root,
        )
        return all_saved

    def _run_study(self, study_id: str) -> List[Path]:
        """Generate all figures for a single study.

        Args:
            study_id: Study identifier (e.g., "MenGrowth-0009-000").

        Returns:
            List of saved file paths.
        """
        cfg = self.config
        saved: List[Path] = []

        # Build per-study loader
        archive_path = Path(cfg.archive_root) / cfg.patient_id / study_id / "archive.h5"
        artifacts_dir = Path(cfg.artifacts_root) / cfg.patient_id / study_id
        atlas_path = Path(cfg.atlas_path) if cfg.atlas_path else None
        preprocessed_dir = None
        if cfg.preprocessed_root:
            preprocessed_dir = Path(cfg.preprocessed_root) / cfg.patient_id / study_id

        loader = ArchiveLoader(
            archive_path, artifacts_dir, atlas_path, preprocessed_dir
        )

        # Attach segmentation if requested
        if cfg.show_segmentation:
            loader.ensure_segmentation_in_archive()

        info = loader.discover()
        logger.info(
            "  Archive: %d steps, modalities=%s, mask=%s, seg=%s",
            len(info["steps"]),
            info["modalities"],
            info["has_mask"],
            info["has_segmentation"],
        )

        # Study output root
        study_dir = self.output_root / study_id

        # Get ordered steps
        ordered_steps = loader.get_ordered_steps()
        if cfg.steps:
            ordered_steps = [
                s for s in ordered_steps if any(req in s for req in cfg.steps)
            ]
        logger.info("  Steps to render: %s", ordered_steps)

        for modality in cfg.modalities:
            for view in cfg.slice.views:
                frac = get_slice_frac(cfg.slice, view)
                pre_reg_frac = get_pre_reg_slice_frac(cfg.slice, view)

                combined_slices: List[Tuple[str, np.ndarray]] = []

                for step_key in ordered_steps:
                    step_name = _clean_step_name(step_key)

                    try:
                        figures = self._render_step(
                            loader,
                            step_key,
                            modality,
                            view,
                            frac,
                            ordered_steps,
                            pre_reg_frac,
                        )
                    except Exception:
                        logger.exception(
                            "Failed to render %s/%s/%s", step_key, modality, view
                        )
                        continue

                    for fig, suffix in figures:
                        path = self._save_figure(
                            fig, study_dir, step_name, view, modality, suffix
                        )
                        saved.append(path)

                    # Collect slice for combined grid
                    if figures:
                        vol = self._load_volume_for_step(loader, step_key, modality)
                        if vol is not None:
                            step_is_pre_reg = self._is_pre_registration(step_key)
                            grid_frac = pre_reg_frac if step_is_pre_reg else frac
                            idx = compute_slice_index(vol.data.shape, view, grid_frac)
                            sl = extract_slice(vol.data, view, idx)
                            normed = normalize_for_display(
                                sl,
                                cfg.intensity_percentile_low,
                                cfg.intensity_percentile_high,
                            )
                            combined_slices.append((step_name, normed))
                            del vol

                # Segmentation overlay
                if cfg.show_segmentation:
                    seg_paths = self._render_segmentation(
                        loader, study_dir, modality, view, frac
                    )
                    saved.extend(seg_paths)

                # Combined grid
                if cfg.output.generate_combined and combined_slices:
                    try:
                        combined_path = self._generate_combined(
                            combined_slices, study_dir, modality, view
                        )
                        saved.append(combined_path)
                    except Exception:
                        logger.exception("Failed to generate combined figure")

        logger.info("  Generated %d figures for %s", len(saved), study_id)
        return saved

    # ── Step classification & volume loading ─────────────────────────────

    def _classify_step(self, step_name: str) -> str:
        """Classify a step for renderer dispatch.

        Args:
            step_name: Full HDF5 group key (e.g., "step2_bias_field_correction").

        Returns:
            One of "bias_field", "registration", "skull_stripping",
            "longitudinal_registration", "standard".
        """
        if "bias_field" in step_name:
            return "bias_field"
        if "longitudinal" in step_name:
            return "longitudinal_registration"
        if "registration" in step_name:
            return "registration"
        if "skull_stripping" in step_name:
            return "skull_stripping"
        return "standard"

    def _is_pre_registration(self, step_key: str) -> bool:
        """Check if a step occurs before registration (different volume shape).

        Pre-registration steps (data_harmonization, bias_field_correction,
        resampling, cubic_padding) have different shapes than post-registration
        steps, so they may need separate slice fractions.

        Args:
            step_key: Full HDF5 group key.

        Returns:
            True if the step is before registration.
        """
        step_type = self._classify_step(step_key)
        return (
            step_type in ("standard", "bias_field") and "registration" not in step_key
        )

    def _load_volume_for_step(
        self, loader: ArchiveLoader, step_key: str, modality: str
    ) -> Optional[StepVolume]:
        """Load volume, dispatching to HDF5 or NIfTI for longitudinal."""
        if "longitudinal" in step_key:
            return loader.load_longitudinal_volume(modality)
        try:
            return loader.load_step_volume(step_key, modality)
        except KeyError:
            logger.warning("Volume not found: %s/%s", step_key, modality)
            return None

    def _find_previous_step_volume(
        self,
        loader: ArchiveLoader,
        current_step_key: str,
        modality: str,
        all_steps: List[str],
    ) -> Optional[StepVolume]:
        """Load the volume from the step immediately before the current one."""
        try:
            idx = all_steps.index(current_step_key)
        except ValueError:
            return None
        if idx == 0:
            return None
        return self._load_volume_for_step(loader, all_steps[idx - 1], modality)

    # ── Rendering dispatch ───────────────────────────────────────────────

    def _render_step(
        self,
        loader: ArchiveLoader,
        step_key: str,
        modality: str,
        view: str,
        frac: Optional[float],
        all_steps: List[str],
        pre_reg_frac: Optional[float] = None,
    ) -> List[Tuple[Figure, str]]:
        """Render figures for a single step.

        Args:
            pre_reg_frac: Separate slice frac for pre-registration steps.
                Used for steps before registration (different volume shape)
                so the displayed slice can be tuned to match the post-reg view.
        """
        step_type = self._classify_step(step_key)
        plow = self.config.intensity_percentile_low
        phigh = self.config.intensity_percentile_high
        dpi = self.config.output.dpi

        # Use pre_reg_frac for steps before registration
        effective_frac = pre_reg_frac if self._is_pre_registration(step_key) else frac

        if step_type == "bias_field":
            vol = loader.load_step_volume(step_key, modality)
            bias = loader.load_bias_field(modality)
            result = render_bias_field_step(
                vol.data,
                bias,
                view,
                effective_frac,
                plow,
                phigh,
                self.config.step_options,
                dpi,
            )
            del vol
            return result

        elif step_type == "registration":
            post_vol = loader.load_step_volume(step_key, modality)
            pre_vol = self._find_previous_step_volume(
                loader, step_key, modality, all_steps
            )
            atlas = loader.load_atlas()
            if pre_vol is not None:
                result = render_registration_step(
                    pre_vol.data,
                    post_vol.data,
                    atlas,
                    view,
                    frac,
                    plow,
                    phigh,
                    self.config.step_options,
                    dpi,
                    pre_reg_frac=pre_reg_frac,
                )
                del pre_vol
            else:
                result = render_standard_slice(
                    post_vol.data, view, frac, plow, phigh, dpi
                )
                if atlas is not None:
                    from .renderers_2d import render_atlas_slice

                    result.extend(
                        render_atlas_slice(atlas, view, frac, plow, phigh, dpi)
                    )
            del post_vol
            return result

        elif step_type == "skull_stripping":
            stripped_vol = loader.load_step_volume(step_key, modality)
            full_head = self._find_previous_step_volume(
                loader, step_key, modality, all_steps
            )
            brain_mask = loader.load_brain_mask()
            if brain_mask is None:
                brain_mask = loader.load_brain_mask_nifti(modality)
            if full_head is not None:
                result = render_skull_stripping_step(
                    full_head.data,
                    stripped_vol.data,
                    brain_mask,
                    view,
                    frac,
                    plow,
                    phigh,
                    self.config.step_options,
                    dpi,
                )
                del full_head
            else:
                result = render_standard_slice(
                    stripped_vol.data, view, frac, plow, phigh, dpi
                )
            del stripped_vol
            return result

        elif step_type == "longitudinal_registration":
            vol = loader.load_longitudinal_volume(modality)
            if vol is None:
                logger.warning("No longitudinal volume for %s/%s", step_key, modality)
                return []
            result = render_standard_slice(vol.data, view, frac, plow, phigh, dpi)
            del vol
            return result

        else:  # standard (pre-registration steps)
            vol = loader.load_step_volume(step_key, modality)
            result = render_standard_slice(
                vol.data, view, effective_frac, plow, phigh, dpi
            )
            del vol
            return result

    def _render_segmentation(
        self,
        loader: ArchiveLoader,
        study_dir: Path,
        modality: str,
        view: str,
        frac: Optional[float],
    ) -> List[Path]:
        """Render segmentation overlay on the final volume."""
        saved: List[Path] = []
        plow = self.config.intensity_percentile_low
        phigh = self.config.intensity_percentile_high
        dpi = self.config.output.dpi

        seg = loader.load_segmentation()
        if seg is None:
            logger.warning("No segmentation available for overlay")
            return saved

        vol = loader.load_longitudinal_volume(modality)
        if vol is None:
            ordered = loader.get_ordered_steps()
            last_key = ordered[-1] if ordered else None
            if last_key and "longitudinal" not in last_key:
                try:
                    vol = loader.load_step_volume(last_key, modality)
                except KeyError:
                    pass
        if vol is None:
            logger.warning("No volume available for segmentation overlay")
            return saved

        figures = render_segmentation_overlay(
            vol.data,
            seg,
            view,
            frac,
            plow,
            phigh,
            self.config.step_options,
            dpi,
        )
        del vol

        for fig, suffix in figures:
            path = self._save_figure(
                fig, study_dir, "segmentation", view, modality, suffix
            )
            saved.append(path)

        return saved

    # ── File I/O ─────────────────────────────────────────────────────────

    def _save_figure(
        self,
        fig: Figure,
        study_dir: Path,
        step_name: str,
        view: str,
        modality: str,
        suffix: str,
    ) -> Path:
        """Save a figure into the organized directory tree.

        Structure: {study_dir}/{step_name}/{view}/{modality}[_{suffix}].{fmt}

        Special case: atlas images go to {study_dir}/atlas/{view}/t1.{fmt}

        Args:
            fig: Matplotlib Figure.
            study_dir: Study-level output directory.
            step_name: Clean step name (e.g., "bias_field_correction").
            view: Anatomical view name.
            modality: Modality name.
            suffix: Filename suffix (e.g., "blend"). Empty for main image.

        Returns:
            Path to the saved file.
        """
        fmt = self.config.output.format

        if suffix == "atlas":
            out_dir = study_dir / "atlas" / view
            filename = f"t1.{fmt}"
        else:
            out_dir = study_dir / step_name / view
            if suffix:
                filename = f"{modality}_{suffix}.{fmt}"
            else:
                filename = f"{modality}.{fmt}"

        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / filename
        fig.savefig(
            path, dpi=self.config.output.dpi, bbox_inches="tight", pad_inches=0.02
        )
        plt.close(fig)
        logger.info("Saved: %s", path)
        return path

    def _generate_combined(
        self,
        slices: List[Tuple[str, np.ndarray]],
        study_dir: Path,
        modality: str,
        view: str,
    ) -> Path:
        """Generate a combined 2-row grid overview of all pipeline steps.

        Saved to: {study_dir}/pipeline_overview/{view}/{modality}.{fmt}

        Args:
            slices: List of (step_name, normalized_2d_slice) tuples.
            study_dir: Study-level output directory.
            modality: Modality name.
            view: Anatomical view name.

        Returns:
            Path to the saved combined figure.
        """
        n = len(slices)
        if n == 0:
            raise ValueError("No slices to combine")

        mid = (n + 1) // 2
        row1 = slices[:mid]
        row2 = slices[mid:]

        row1_widths = [sl.shape[1] for _, sl in row1]
        row2_widths = [sl.shape[1] for _, sl in row2]

        ncols = max(len(row1), len(row2))
        while len(row1_widths) < ncols:
            row1_widths.append(1)
        while len(row2_widths) < ncols:
            row2_widths.append(1)

        width_ratios = [max(row1_widths[i], row2_widths[i]) for i in range(ncols)]

        fig = plt.figure(figsize=(14, 6), dpi=self.config.output.dpi)
        gs = gridspec.GridSpec(
            2, ncols, figure=fig, width_ratios=width_ratios, hspace=0.15, wspace=0.05
        )

        for row_idx, row_data in enumerate([row1, row2]):
            for col_idx, (name, sl) in enumerate(row_data):
                ax = fig.add_subplot(gs[row_idx, col_idx])
                ax.imshow(
                    sl, cmap="gray", vmin=0, vmax=1, origin="lower", aspect="equal"
                )
                ax.set_title(name.replace("_", " ").title(), fontsize=7, pad=2)
                ax.axis("off")

        for row_idx, row_data in enumerate([row1, row2]):
            for col_idx in range(len(row_data), ncols):
                ax = fig.add_subplot(gs[row_idx, col_idx])
                ax.axis("off")

        fmt = self.config.output.format
        out_dir = study_dir / "pipeline_overview" / view
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{modality}.{fmt}"
        fig.savefig(
            path, dpi=self.config.output.dpi, bbox_inches="tight", pad_inches=0.05
        )
        plt.close(fig)
        logger.info("Saved combined: %s", path)
        return path
