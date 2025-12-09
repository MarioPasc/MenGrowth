"""Base classes for skull stripping (brain extraction) preprocessing steps.

This module defines abstract base classes for skull stripping operations
following the OOP design pattern. All skull stripping implementations must
inherit from these base classes and implement their abstract methods.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any
import logging

from mengrowth.preprocessing.src.base import BasePreprocessingStep

logger = logging.getLogger(__name__)


class BaseSkullStripper(BasePreprocessingStep):
    """Abstract base class for skull stripping (brain extraction) operations.

    Skull strippers separate brain tissue from non-brain tissue (skull, meninges,
    eyes, etc.) using either traditional segmentation or deep learning methods.
    Implementations should preserve brain tissue while removing extracranial structures.

    Attributes:
        config: Configuration parameters for skull stripping
        fill_value: Value to use for background voxels after brain extraction
    """

    def __init__(self, config: Dict[str, Any], verbose: bool = False) -> None:
        """Initialize skull stripper.

        Args:
            config: Configuration dictionary with method-specific parameters
            verbose: Enable verbose logging
        """
        super().__init__(step_name="SkullStripper", verbose=verbose)
        self.config = config
        self.fill_value = config.get("fill_value", 0.0)

    @abstractmethod
    def execute(
        self,
        input_path: Path,
        output_path: Path,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Execute the skull stripping operation.

        Args:
            input_path: Path to input NIfTI file
            output_path: Path to output brain-extracted NIfTI file
            **kwargs: Additional operation-specific parameters including:
                - 'mask_path': Path where brain mask should be saved
                - 'allow_overwrite': Whether to overwrite existing files

        Returns:
            Dictionary containing execution results:
                - 'mask_path': Path to saved brain mask NIfTI
                - 'algorithm': Name of algorithm used (e.g., 'hdbet', 'synthstrip')
                - 'parameters': Dictionary of parameters used
                - 'brain_volume_mm3': Volume of extracted brain in cubic millimeters
                - 'brain_coverage_percent': Percentage of original volume kept

        Raises:
            FileNotFoundError: If input file does not exist
            RuntimeError: If skull stripping fails
            ImportError: If required dependencies are not installed
        """
        pass

    @abstractmethod
    def visualize(
        self,
        before_path: Path,
        after_path: Path,
        output_path: Path,
        **kwargs: Any
    ) -> None:
        """Generate visualization comparing before and after skull stripping.

        The visualization should show:
        1. Original image (3 orthogonal views: axial, sagittal, coronal)
        2. Brain mask overlay on original (red overlay with alpha blending)
        3. Skull-stripped result (3 orthogonal views)
        4. Intensity histogram comparison (before vs after)

        Args:
            before_path: Path to input file (before skull stripping)
            after_path: Path to output file (after skull stripping)
            output_path: Path to save visualization output (PNG)
            **kwargs: Additional visualization parameters including:
                - 'mask_path': Path to brain mask file
                - 'algorithm': Algorithm name
                - 'parameters': Parameters used
                - 'brain_volume_mm3': Brain volume
                - 'brain_coverage_percent': Coverage percentage

        Raises:
            FileNotFoundError: If input files do not exist
            RuntimeError: If visualization generation fails
        """
        pass
