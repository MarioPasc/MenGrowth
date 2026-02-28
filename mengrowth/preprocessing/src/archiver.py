"""Detailed patient archiver for HDF5 snapshots.

Archives per-step MRI snapshots and artifacts for showcase patients,
enabling publication-quality figure generation without re-running the pipeline.

Archive structure per study:
    /metadata/          — patient_id, study_id, modalities, pipeline_steps, created_at
    /steps/{step}/      — per-modality 3D volumes (float32, gzip) + affine attrs
    /masks/             — brain_mask (uint8, gzip)
    /transforms/        — raw bytes of ANTs .h5 transforms
    /segmentation/      — added later via attach_segmentation()
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class DetailedPatientArchiver:
    """Archives per-step MRI snapshots and artifacts for showcase patients.

    Attributes:
        config: Archive configuration (patient_ids, compression settings)
        output_root: Pipeline output root directory
        modalities: List of modalities being processed
    """

    def __init__(
        self,
        config: "DetailedArchiveConfig",  # noqa: F821
        output_root: str,
        modalities: List[str],
    ) -> None:
        """Initialize the archiver.

        Args:
            config: DetailedArchiveConfig with patient_ids and compression settings
            output_root: Pipeline output root directory
            modalities: List of modalities being processed
        """
        self.config = config
        self.output_root = Path(output_root)
        self.modalities = modalities
        self._handles: Dict[str, "h5py.File"] = {}  # noqa: F821

        logger.info(
            f"DetailedPatientArchiver initialized for patients: {config.patient_ids}"
        )

    def should_archive(self, patient_id: str) -> bool:
        """Check if this patient is a showcase patient.

        Args:
            patient_id: Patient identifier to check

        Returns:
            True if patient should be archived
        """
        return patient_id in self.config.patient_ids

    def save_pipeline_metadata(
        self,
        patient_id: str,
        study_id: str,
        steps: List[str],
        modalities: List[str],
    ) -> None:
        """Write /metadata/ group at archive creation time.

        Args:
            patient_id: Patient identifier
            study_id: Study identifier
            steps: Ordered list of pipeline step names
            modalities: List of modalities being processed
        """
        try:
            h5 = self._get_handle(patient_id, study_id)
            meta = h5.require_group("metadata")

            # Store scalar strings as datasets
            for key, value in [
                ("patient_id", patient_id),
                ("study_id", study_id),
                ("created_at", datetime.now(timezone.utc).isoformat()),
            ]:
                if key in meta:
                    del meta[key]
                meta.create_dataset(key, data=value)

            # Store string arrays
            for key, values in [
                ("modalities", modalities),
                ("pipeline_steps", steps),
            ]:
                if key in meta:
                    del meta[key]
                meta.create_dataset(key, data=values)

            h5.flush()
            logger.debug(f"Saved pipeline metadata for {study_id}")

        except Exception as e:
            logger.warning(f"Failed to save pipeline metadata for {study_id}: {e}")

    def on_modality_step_done(
        self,
        patient_id: str,
        study_id: str,
        step_name: str,
        step_index: int,
        modality: str,
        nifti_path: Path,
    ) -> None:
        """Snapshot a modality volume after a modality-level step completes.

        Args:
            patient_id: Patient identifier
            study_id: Study identifier
            step_name: Pipeline step name (e.g., "data_harmonization")
            step_index: Step number in pipeline (1-based)
            modality: Modality name (e.g., "t1c")
            nifti_path: Path to the NIfTI file to snapshot
        """
        try:
            h5 = self._get_handle(patient_id, study_id)
            group_path = f"steps/step{step_index}_{step_name}"
            self._write_volume(h5, f"{group_path}/{modality}", nifti_path)

            # Store step_index as group attribute
            step_group = h5[group_path]
            step_group.attrs["step_index"] = step_index

            h5.flush()
            logger.debug(
                f"Archived {modality} after {step_name} for {study_id}"
            )

        except Exception as e:
            logger.warning(
                f"Archive snapshot failed for {modality}/{step_name}/{study_id}: {e}"
            )

    def on_study_step_done(
        self,
        patient_id: str,
        study_id: str,
        step_name: str,
        step_index: int,
        modality_paths: Dict[str, Path],
        result: Dict[str, Any],
    ) -> None:
        """Snapshot all modalities after a study-level step completes.

        Also saves masks and transforms from the result dict when available.

        Args:
            patient_id: Patient identifier
            study_id: Study identifier
            step_name: Pipeline step name (e.g., "registration_static")
            step_index: Step number in pipeline (1-based)
            modality_paths: Mapping of modality name to NIfTI path
            result: Step execution result dict (may contain masks, transforms)
        """
        try:
            h5 = self._get_handle(patient_id, study_id)
            group_path = f"steps/step{step_index}_{step_name}"

            # Snapshot all modality volumes
            for modality, nifti_path in sorted(modality_paths.items()):
                if nifti_path.exists():
                    self._write_volume(h5, f"{group_path}/{modality}", nifti_path)

            # Store step_index as group attribute
            if group_path in h5:
                h5[group_path].attrs["step_index"] = step_index

            # Save brain mask if this is skull stripping
            if "skull_stripping" in step_name:
                self._save_masks(h5, patient_id, study_id, result)

            # Save registration transforms if available
            if "registration" in step_name and "longitudinal" not in step_name:
                self._save_transforms(h5, patient_id, study_id, result)

            h5.flush()
            logger.debug(f"Archived study step {step_name} for {study_id}")

        except Exception as e:
            logger.warning(
                f"Archive study snapshot failed for {step_name}/{study_id}: {e}"
            )

    def finalize(self, patient_id: str) -> None:
        """Close any open HDF5 handles for this patient.

        Args:
            patient_id: Patient identifier
        """
        keys_to_close = [k for k in self._handles if k.startswith(f"{patient_id}/")]
        for key in keys_to_close:
            try:
                self._handles[key].close()
                logger.debug(f"Closed archive handle: {key}")
            except Exception as e:
                logger.warning(f"Failed to close archive handle {key}: {e}")
            finally:
                del self._handles[key]

        logger.info(f"Finalized archives for {patient_id} ({len(keys_to_close)} files)")

    @staticmethod
    def attach_segmentation(
        archive_path: Path,
        seg_path: Path,
        label_map: Optional[Dict[int, str]] = None,
    ) -> None:
        """Open existing archive and add /segmentation/ group.

        Args:
            archive_path: Path to existing archive.h5
            seg_path: Path to segmentation NIfTI file (e.g., seg.nii.gz)
            label_map: Optional mapping of label integers to names
        """
        import h5py
        import nibabel as nib

        seg_img = nib.load(str(seg_path))
        seg_data = np.asarray(seg_img.dataobj, dtype=np.uint8)
        affine = seg_img.affine.astype(np.float64)

        with h5py.File(str(archive_path), "a") as h5:
            # Remove existing segmentation group if present
            if "segmentation" in h5:
                del h5["segmentation"]

            seg_group = h5.create_group("segmentation")
            ds = seg_group.create_dataset(
                "seg",
                data=seg_data,
                compression="gzip",
                compression_opts=4,
            )
            ds.attrs["affine"] = affine
            ds.attrs["voxel_size"] = np.abs(np.diag(affine[:3, :3]))
            ds.attrs["source"] = "brats2025"

            if label_map:
                for k, v in label_map.items():
                    seg_group.attrs[f"label_{k}"] = v

        logger.info(f"Attached segmentation to {archive_path}")

    # ── Internal helpers ──

    def _get_handle(self, patient_id: str, study_id: str) -> "h5py.File":  # noqa: F821
        """Get or create HDF5 file handle (lazy, cached per study).

        Args:
            patient_id: Patient identifier
            study_id: Study identifier

        Returns:
            Open h5py.File handle in append mode
        """
        import h5py

        key = f"{patient_id}/{study_id}"
        if key not in self._handles:
            archive_dir = (
                self.output_root / "detailed_patient" / patient_id / study_id
            )
            archive_dir.mkdir(parents=True, exist_ok=True)
            archive_path = archive_dir / "archive.h5"

            self._handles[key] = h5py.File(str(archive_path), "a")
            logger.debug(f"Opened archive handle: {archive_path}")

        return self._handles[key]

    def _write_volume(
        self, h5: "h5py.File", group_path: str, nifti_path: Path  # noqa: F821
    ) -> None:
        """Load NIfTI and write 3D array + affine attrs to HDF5 dataset.

        Args:
            h5: Open HDF5 file handle
            group_path: HDF5 path for the dataset (e.g., "steps/step1_data_harmonization/t1c")
            nifti_path: Path to NIfTI file to read
        """
        import nibabel as nib

        img = nib.load(str(nifti_path))
        data = np.asarray(img.dataobj, dtype=np.float32)
        affine = img.affine.astype(np.float64)

        # Remove existing dataset if present (for overwrite)
        if group_path in h5:
            del h5[group_path]

        ds = h5.create_dataset(
            group_path,
            data=data,
            compression=self.config.compression,
            compression_opts=self.config.compression_level,
        )
        ds.attrs["affine"] = affine
        ds.attrs["voxel_size"] = np.abs(np.diag(affine[:3, :3]))

    def _save_masks(
        self,
        h5: "h5py.File",  # noqa: F821
        patient_id: str,
        study_id: str,
        result: Dict[str, Any],
    ) -> None:
        """Save brain mask from skull stripping result to /masks/ group.

        Args:
            h5: Open HDF5 file handle
            patient_id: Patient identifier
            study_id: Study identifier
            result: Step result dict (may contain mask_path or qc_paths)
        """
        import nibabel as nib

        # Try to find brain mask in artifacts directory
        artifacts_base = (
            Path(self.output_root).parent
            / "artifacts"
            / patient_id
            / study_id
        )

        # Check common mask locations
        mask_paths_to_try = []

        # From result dict
        if result and "qc_paths" in result:
            for mod_data in result["qc_paths"].values():
                if isinstance(mod_data, dict) and "mask" in mod_data:
                    mask_paths_to_try.append(Path(mod_data["mask"]))

        # Standard artifact naming convention
        for mod in self.modalities:
            mask_paths_to_try.append(artifacts_base / f"{mod}_brain_mask.nii.gz")

        # Also check the preprocessing_artifacts_path from output_root sibling
        # The archiver doesn't have direct access to config.preprocessing_artifacts_path,
        # so we reconstruct it relative to output_root
        for mod in self.modalities:
            alt_artifacts = Path(str(self.output_root) + "/artifacts") / patient_id / study_id
            mask_paths_to_try.append(alt_artifacts / f"{mod}_brain_mask.nii.gz")

        # Save the first mask we find
        for mask_path in mask_paths_to_try:
            if mask_path.exists():
                try:
                    mask_img = nib.load(str(mask_path))
                    mask_data = np.asarray(mask_img.dataobj, dtype=np.uint8)
                    affine = mask_img.affine.astype(np.float64)

                    if "masks" in h5:
                        del h5["masks"]

                    masks_group = h5.create_group("masks")
                    ds = masks_group.create_dataset(
                        "brain_mask",
                        data=mask_data,
                        compression="gzip",
                        compression_opts=self.config.compression_level,
                    )
                    ds.attrs["affine"] = affine
                    ds.attrs["voxel_size"] = np.abs(np.diag(affine[:3, :3]))

                    # Extract source modality from filename
                    source_mod = mask_path.stem.replace("_brain_mask.nii", "")
                    masks_group.attrs["source_modality"] = source_mod

                    # Try to determine method from result
                    if result and isinstance(result, dict):
                        method = result.get("method", "unknown")
                        masks_group.attrs["method"] = str(method)

                    logger.debug(f"Saved brain mask from {mask_path}")
                    return

                except Exception as e:
                    logger.debug(f"Could not load mask from {mask_path}: {e}")

        logger.debug(f"No brain mask found for {patient_id}/{study_id}")

    def _save_transforms(
        self,
        h5: "h5py.File",  # noqa: F821
        patient_id: str,
        study_id: str,
        result: Dict[str, Any],
    ) -> None:
        """Save registration transforms to /transforms/ group.

        Stores raw bytes of ANTs .h5/.mat transform files.

        Args:
            h5: Open HDF5 file handle
            patient_id: Patient identifier
            study_id: Study identifier
            result: Step result dict (may contain transform paths)
        """
        if not result:
            return

        # Look for transform files in artifacts
        artifacts_base = Path(str(self.output_root) + "/artifacts") / patient_id / study_id

        # Also check the sibling artifacts directory
        alt_artifacts = (
            Path(self.output_root).parent / "artifacts" / patient_id / study_id
        )

        if "transforms" in h5:
            del h5["transforms"]

        transforms_group = h5.create_group("transforms")

        # Save intra-study transforms
        intra_group = transforms_group.create_group("intra_study")
        for mod in self.modalities:
            for artifacts_dir in [artifacts_base, alt_artifacts]:
                # Check common transform naming patterns
                for suffix in [".h5", ".mat", "_composite.h5"]:
                    transform_path = artifacts_dir / f"{mod}_to_ref{suffix}"
                    if transform_path.exists():
                        raw_bytes = transform_path.read_bytes()
                        ds_name = f"{mod}_to_ref{suffix.replace('.', '_')}"
                        intra_group.create_dataset(ds_name, data=np.void(raw_bytes))
                        logger.debug(f"Saved intra-study transform: {transform_path}")
                        break

        # Save atlas transforms
        atlas_group = transforms_group.create_group("atlas")
        for artifacts_dir in [artifacts_base, alt_artifacts]:
            for suffix in [".h5", ".mat", "_composite.h5"]:
                transform_path = artifacts_dir / f"ref_to_atlas{suffix}"
                if transform_path.exists():
                    raw_bytes = transform_path.read_bytes()
                    ds_name = f"ref_to_atlas{suffix.replace('.', '_')}"
                    atlas_group.create_dataset(ds_name, data=np.void(raw_bytes))
                    logger.debug(f"Saved atlas transform: {transform_path}")
                    break

        # Store reference modality from result if available
        if result.get("reference_modality"):
            transforms_group.attrs["reference_modality"] = result["reference_modality"]
        if result.get("atlas_path"):
            transforms_group.attrs["atlas_path"] = str(result["atlas_path"])
