"""Base class for image registration preprocessing steps.

This module defines the abstract base class for registration operations,
including multi-modal coregistration, atlas registration, and longitudinal registration.
"""

from abc import abstractmethod
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

from mengrowth.preprocessing.src.base import BasePreprocessingStep

logger = logging.getLogger(__name__)


class BaseRegistrator(BasePreprocessingStep):
    """Abstract base class for registration preprocessing steps.

    All registration operations must inherit from this class and implement
    the execute() and visualize() methods. Registration steps work with
    multiple images (modalities) and produce transform files as artifacts.

    Attributes:
        config: Dictionary containing registration configuration parameters
        step_name: Human-readable name of this preprocessing step
        verbose: Whether to enable verbose logging
    """

    def __init__(self, config: Dict[str, Any], verbose: bool = False) -> None:
        """Initialize registration step.

        Args:
            config: Configuration dictionary with registration parameters
            verbose: Enable verbose logging
        """
        super().__init__(step_name="Registrator", verbose=verbose)
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    def execute(
        self,
        study_dir: Path,
        artifacts_dir: Path,
        modalities: List[str],
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Execute the registration operation.

        This method is called once per study and processes all modalities together.
        It should:
        1. Discover available modality files in study_dir
        2. Select reference modality using priority
        3. Register all non-reference modalities to the reference
        4. Save transform files to artifacts_dir
        5. Update modality files in-place (using temp file pattern)

        Args:
            study_dir: Directory containing the study's modality files
            artifacts_dir: Directory to save transform artifacts
            modalities: List of expected modalities (e.g., ["t1c", "t1n", "t2w", "t2f"])
            **kwargs: Additional operation-specific parameters

        Returns:
            Dictionary with registration metadata:
            {
                "reference_modality": str,  # Selected reference modality
                "registered_modalities": List[str],  # List of registered modalities
                "transforms": Dict[str, Path],  # Mapping from modality to transform path
                "metrics": Dict[str, Any]  # Registration quality metrics (optional)
            }

        Raises:
            ValueError: If fewer than 2 modalities are found
            RuntimeError: If registration fails
        """
        pass

    @abstractmethod
    def visualize(
        self,
        reference_path: Path,
        moving_path: Path,
        registered_path: Path,
        output_path: Path,
        **kwargs: Any
    ) -> None:
        """Generate visualization comparing reference, moving, and registered images.

        Creates a side-by-side comparison visualization with:
        - Reference image (fixed)
        - Moving image (before registration)
        - Registered image (after registration)
        - Optional: checkerboard overlay

        Args:
            reference_path: Path to reference (fixed) image
            moving_path: Path to moving image (before registration)
            registered_path: Path to registered image (after registration)
            output_path: Path to save visualization (PNG)
            **kwargs: Additional visualization parameters (e.g., transform_path, metrics)

        Raises:
            FileNotFoundError: If input files do not exist
            RuntimeError: If visualization generation fails
        """
        pass

    def _select_reference_modality(
        self,
        available_modalities: List[str],
        priority_str: str
    ) -> str:
        """Select reference modality based on priority order.

        Parses priority string (e.g., "t1c > t1n > t2f > t2w") and returns
        the highest-priority modality that is available.

        Args:
            available_modalities: List of modalities present in the study
            priority_str: Priority string with ">" separators

        Returns:
            Selected reference modality name

        Raises:
            ValueError: If no modalities match the priority list
        """
        # Parse priority string
        if ">" in priority_str:
            priorities = [m.strip() for m in priority_str.split(">")]
        else:
            priorities = [priority_str.strip()]

        # Find highest-priority available modality
        for modality in priorities:
            if modality in available_modalities:
                self.logger.info(f"Selected reference modality: {modality}")
                return modality

        # If no match found, use the first available modality
        if available_modalities:
            selected = available_modalities[0]
            self.logger.warning(
                f"No modality from priority list found. Using first available: {selected}"
            )
            return selected

        raise ValueError(
            f"No available modalities match priority list: {priority_str}. "
            f"Available: {available_modalities}"
        )

    def _discover_modality_files(
        self,
        study_dir: Path,
        modalities: List[str]
    ) -> Dict[str, Path]:
        """Discover modality NIfTI files in study directory.

        Args:
            study_dir: Directory containing modality files
            modalities: List of expected modalities

        Returns:
            Dictionary mapping modality name to file path

        Raises:
            ValueError: If no modality files are found
        """
        modality_files = {}

        for modality in modalities:
            # Look for files matching modality name
            candidates = list(study_dir.glob(f"{modality}.nii.gz"))
            if candidates:
                modality_files[modality] = candidates[0]
                self.logger.debug(f"Found {modality}: {candidates[0].name}")

        if not modality_files:
            raise ValueError(
                f"No modality files found in {study_dir}. "
                f"Expected modalities: {modalities}"
            )

        self.logger.info(
            f"Discovered {len(modality_files)} modalities: {list(modality_files.keys())}"
        )

        return modality_files
