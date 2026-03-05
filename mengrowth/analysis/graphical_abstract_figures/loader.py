"""HDF5 archive and NIfTI artifact I/O for graphical abstract generation.

Reads per-step volumes from the DetailedPatientArchiver HDF5 format,
plus NIfTI artifacts (bias fields, brain masks) and the atlas volume.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StepVolume:
    """A 3D volume snapshot from one pipeline step.

    Attributes:
        data: 3D float32 array.
        affine: 4x4 affine matrix.
        voxel_size: 3-element voxel spacing.
        step_name: Pipeline step name (e.g., "data_harmonization").
        modality: MRI modality (e.g., "t1c").
    """

    data: np.ndarray
    affine: np.ndarray
    voxel_size: np.ndarray
    step_name: str
    modality: str


class ArchiveLoader:
    """Loads volumes and artifacts from HDF5 archives and NIfTI files.

    Args:
        archive_path: Path to archive.h5 file.
        artifacts_dir: Directory containing NIfTI artifacts for this study.
        atlas_path: Path to the atlas T1 NIfTI volume (optional).
        preprocessed_dir: Directory containing final preprocessed NIfTI volumes
            (post-longitudinal registration). Optional.
    """

    def __init__(
        self,
        archive_path: Path,
        artifacts_dir: Path,
        atlas_path: Optional[Path] = None,
        preprocessed_dir: Optional[Path] = None,
    ) -> None:
        self.archive_path = Path(archive_path)
        self.artifacts_dir = Path(artifacts_dir)
        self.atlas_path = Path(atlas_path) if atlas_path else None
        self.preprocessed_dir = Path(preprocessed_dir) if preprocessed_dir else None

        if not self.archive_path.exists():
            raise FileNotFoundError(f"Archive not found: {self.archive_path}")

        logger.info("ArchiveLoader initialized: %s", self.archive_path)

    def discover(self) -> Dict[str, object]:
        """Discover available steps, modalities, and masks in the archive.

        Returns:
            Dict with keys: "steps" (list of step group names),
            "modalities" (set of modality names), "has_mask" (bool),
            "has_segmentation" (bool).
        """
        with h5py.File(str(self.archive_path), "r") as h5:
            steps = []
            modalities = set()

            if "steps" in h5:
                for step_key in sorted(h5["steps"].keys()):
                    steps.append(step_key)
                    step_group = h5[f"steps/{step_key}"]
                    for item in step_group:
                        if isinstance(step_group[item], h5py.Dataset):
                            modalities.add(item)

            has_mask = "masks" in h5 and "brain_mask" in h5.get("masks", {})
            has_seg = "segmentation" in h5

        return {
            "steps": steps,
            "modalities": modalities,
            "has_mask": has_mask,
            "has_segmentation": has_seg,
        }

    def load_step_volume(self, step_group_key: str, modality: str) -> StepVolume:
        """Load a volume from a specific step group.

        Args:
            step_group_key: HDF5 group key (e.g., "step1_data_harmonization").
            modality: Modality name (e.g., "t1c").

        Returns:
            StepVolume with data, affine, and metadata.

        Raises:
            KeyError: If the step/modality combination is not in the archive.
        """
        dataset_path = f"steps/{step_group_key}/{modality}"
        with h5py.File(str(self.archive_path), "r") as h5:
            if dataset_path not in h5:
                raise KeyError(f"Dataset not found: {dataset_path}")

            ds = h5[dataset_path]
            data = ds[()].astype(np.float32)
            affine = ds.attrs["affine"].astype(np.float64)
            voxel_size = ds.attrs["voxel_size"].astype(np.float64)

        # Extract clean step name (strip "stepN_" prefix)
        parts = step_group_key.split("_", 1)
        step_name = parts[1] if len(parts) > 1 else step_group_key

        return StepVolume(
            data=data,
            affine=affine,
            voxel_size=voxel_size,
            step_name=step_name,
            modality=modality,
        )

    def load_brain_mask(self) -> Optional[np.ndarray]:
        """Load the brain mask from the HDF5 archive.

        Returns:
            3D uint8 array, or None if not found.
        """
        with h5py.File(str(self.archive_path), "r") as h5:
            if "masks/brain_mask" in h5:
                return h5["masks/brain_mask"][()].astype(np.uint8)
        logger.debug("No brain mask in archive")
        return None

    def load_bias_field(self, modality: str) -> Optional[np.ndarray]:
        """Load a bias field NIfTI from the artifacts directory.

        Args:
            modality: Modality name (e.g., "t1c").

        Returns:
            3D float32 array of the multiplicative bias field, or None.
        """
        bias_path = self.artifacts_dir / f"{modality}_bias_field.nii.gz"
        if not bias_path.exists():
            logger.debug("Bias field not found: %s", bias_path)
            return None

        import nibabel as nib

        img = nib.load(str(bias_path))
        data = np.asarray(img.dataobj, dtype=np.float32)
        logger.debug("Loaded bias field: %s, shape=%s", bias_path, data.shape)
        return data

    def load_brain_mask_nifti(self, modality: str) -> Optional[np.ndarray]:
        """Load a brain mask NIfTI from the artifacts directory.

        Args:
            modality: Modality name (e.g., "t1c").

        Returns:
            3D uint8 array, or None if not found.
        """
        mask_path = self.artifacts_dir / f"{modality}_brain_mask.nii.gz"
        if not mask_path.exists():
            logger.debug("Brain mask NIfTI not found: %s", mask_path)
            return None

        import nibabel as nib

        img = nib.load(str(mask_path))
        return np.asarray(img.dataobj, dtype=np.uint8)

    def load_atlas(self) -> Optional[np.ndarray]:
        """Load the atlas T1 volume, squeezing the 4th dimension if present.

        Returns:
            3D float32 array, or None if atlas_path is not set or file missing.
        """
        if self.atlas_path is None or not self.atlas_path.exists():
            logger.debug("Atlas not available: %s", self.atlas_path)
            return None

        import nibabel as nib

        img = nib.load(str(self.atlas_path))
        data = np.asarray(img.dataobj, dtype=np.float32)
        # SRI24 atlas has shape (240,240,155,1) — squeeze 4th dim
        if data.ndim == 4 and data.shape[3] == 1:
            data = data[:, :, :, 0]
        logger.debug("Loaded atlas: shape=%s", data.shape)
        return data

    def resolve_step_group(self, step_name: str) -> Optional[str]:
        """Find the HDF5 group key that matches a step name via substring.

        Args:
            step_name: Step name to search for (e.g., "intensity_normalization").

        Returns:
            Full HDF5 group key (e.g., "step1_intensity_normalization"), or None.
        """
        with h5py.File(str(self.archive_path), "r") as h5:
            if "steps" not in h5:
                return None
            for key in sorted(h5["steps"].keys()):
                if step_name in key:
                    return key
        return None

    def get_ordered_steps(self) -> List[str]:
        """Get pipeline step names in correct execution order from metadata.

        Uses metadata/pipeline_steps for canonical ordering, then maps each
        to HDF5 group keys via substring matching.

        Returns:
            List of HDF5 group keys in pipeline order.
        """
        pipeline_steps: List[str] = []
        step_groups: List[str] = []

        with h5py.File(str(self.archive_path), "r") as h5:
            # Read canonical step order from metadata
            if "metadata/pipeline_steps" in h5:
                raw = h5["metadata/pipeline_steps"][()]
                if isinstance(raw, np.ndarray):
                    pipeline_steps = [
                        s.decode("utf-8") if isinstance(s, bytes) else s for s in raw
                    ]
                elif isinstance(raw, bytes):
                    pipeline_steps = [raw.decode("utf-8")]

            # Get all available step group keys
            if "steps" in h5:
                step_groups = list(h5["steps"].keys())

        # Map each pipeline step name to its HDF5 group key
        ordered = []
        used_groups = set()
        for step_name in pipeline_steps:
            for group_key in step_groups:
                if step_name in group_key and group_key not in used_groups:
                    ordered.append(group_key)
                    used_groups.add(group_key)
                    break

        # Append any leftover groups not matched (shouldn't happen normally)
        for group_key in sorted(step_groups):
            if group_key not in used_groups:
                ordered.append(group_key)
                logger.warning("Unmatched step group: %s", group_key)

        # Append longitudinal_registration as virtual step if preprocessed dir exists
        if self.preprocessed_dir and self.preprocessed_dir.exists():
            if "longitudinal_registration" in pipeline_steps:
                ordered.append("step8_longitudinal_registration")

        return ordered

    def load_longitudinal_volume(self, modality: str) -> Optional[StepVolume]:
        """Load the final preprocessed volume (post-longitudinal registration).

        These live in the preprocessed output directory as {modality}.nii.gz,
        not in the HDF5 archive (longitudinal registration is a patient-level step).

        Args:
            modality: Modality name (e.g., "t1c").

        Returns:
            StepVolume, or None if not found.
        """
        if self.preprocessed_dir is None:
            return None

        nifti_path = self.preprocessed_dir / f"{modality}.nii.gz"
        if not nifti_path.exists():
            logger.debug("Longitudinal volume not found: %s", nifti_path)
            return None

        import nibabel as nib

        img = nib.load(str(nifti_path))
        data = np.asarray(img.dataobj, dtype=np.float32)
        affine = img.affine.astype(np.float64)
        voxel_size = np.abs(np.diag(affine[:3, :3]))

        logger.debug("Loaded longitudinal volume: %s, shape=%s", nifti_path, data.shape)
        return StepVolume(
            data=data,
            affine=affine,
            voxel_size=voxel_size,
            step_name="longitudinal_registration",
            modality=modality,
        )

    def load_segmentation(self) -> Optional[np.ndarray]:
        """Load tumor segmentation mask.

        Checks the HDF5 archive first, then falls back to seg.nii.gz
        in the archive directory (co-located with archive.h5).

        Returns:
            3D uint8 array with labels (0=bg, 1=NET, 2=SNFH, 3=ET), or None.
        """
        # Try HDF5 first
        with h5py.File(str(self.archive_path), "r") as h5:
            if "segmentation/seg" in h5:
                logger.debug("Loading segmentation from HDF5 archive")
                return h5["segmentation/seg"][()].astype(np.uint8)

        # Fall back to seg.nii.gz next to archive.h5
        seg_path = self.archive_path.parent / "seg.nii.gz"
        if seg_path.exists():
            import nibabel as nib

            img = nib.load(str(seg_path))
            data = np.asarray(img.dataobj, dtype=np.uint8)
            logger.debug("Loaded segmentation from %s, shape=%s", seg_path, data.shape)
            return data

        logger.debug("No segmentation found")
        return None

    def ensure_segmentation_in_archive(self) -> bool:
        """Attach seg.nii.gz to the HDF5 archive if not already present.

        Returns:
            True if segmentation is now in the archive, False otherwise.
        """
        with h5py.File(str(self.archive_path), "r") as h5:
            if "segmentation/seg" in h5:
                logger.debug("Segmentation already in archive")
                return True

        seg_path = self.archive_path.parent / "seg.nii.gz"
        if not seg_path.exists():
            logger.warning("No seg.nii.gz found at %s", seg_path)
            return False

        from mengrowth.preprocessing.src.archiver import DetailedPatientArchiver

        label_map = {1: "necrotic_core", 2: "peritumoral_edema", 3: "enhancing_tumor"}
        DetailedPatientArchiver.attach_segmentation(
            self.archive_path, seg_path, label_map
        )
        logger.info("Attached segmentation to %s", self.archive_path)
        return True
