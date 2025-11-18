"""Base classes for resampling preprocessing steps.

This module defines abstract base classes for resampling operations following
the OOP design pattern specified in CLAUDE.md. All resampling implementations
must inherit from these base classes and implement their abstract methods.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List
import logging

from mengrowth.preprocessing.src.base import BasePreprocessingStep

logger = logging.getLogger(__name__)


class BaseResampler(BasePreprocessingStep):
    """Abstract base class for resampling operations.

    Resamplers transform volumes to a target voxel spacing (e.g., 1mm isotropic)
    using various interpolation methods. Implementations should preserve image
    quality while achieving the desired spatial resolution.

    Attributes:
        target_voxel_size: Target voxel size in mm [x, y, z]
        config: Configuration parameters for resampling
    """

    def __init__(
        self,
        target_voxel_size: List[float],
        config: Dict[str, Any],
        verbose: bool = False
    ) -> None:
        """Initialize resampler.

        Args:
            target_voxel_size: Target voxel size in mm [x, y, z]
            config: Configuration dictionary with method-specific parameters
            verbose: Enable verbose logging
        """
        super().__init__(step_name="Resampler", verbose=verbose)
        self.target_voxel_size = target_voxel_size
        self.config = config

        self.logger.info(
            f"Initialized {self.__class__.__name__} with target voxel size: {target_voxel_size}"
        )

    @abstractmethod
    def execute(
        self,
        input_path: Path,
        output_path: Path,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Execute the resampling operation.

        Args:
            input_path: Path to input NIfTI file
            output_path: Path to output resampled NIfTI file
            **kwargs: Additional operation-specific parameters
                - allow_overwrite: Allow overwriting existing files (bool)

        Returns:
            Dictionary containing execution results, including:
                - 'original_spacing': Original voxel spacing [x, y, z]
                - 'target_spacing': Target voxel spacing [x, y, z]
                - 'original_shape': Original image shape [x, y, z]
                - 'resampled_shape': Resampled image shape [x, y, z]

        Raises:
            FileNotFoundError: If input file does not exist
            RuntimeError: If resampling fails
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
        """Generate visualization comparing before and after resampling.

        The visualization should show:
        1. Original image (axial, sagittal, coronal views)
        2. Resampled image (axial, sagittal, coronal views)
        3. Metadata comparison (spacing, shape)

        Args:
            before_path: Path to input file (before resampling)
            after_path: Path to output file (after resampling)
            output_path: Path to save visualization output (PNG)
            **kwargs: Additional visualization parameters including:
                - 'original_spacing': Original voxel spacing
                - 'target_spacing': Target voxel spacing
                - 'original_shape': Original image shape
                - 'resampled_shape': Resampled image shape

        Raises:
            FileNotFoundError: If input files do not exist
            RuntimeError: If visualization generation fails
        """
        pass
