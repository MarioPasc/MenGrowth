"""Base classes for normalization preprocessing steps.

This module defines abstract base classes for normalization operations following
the OOP design pattern specified in the preprocessing documentation. All normalization
implementations must inherit from these base classes and implement their abstract methods.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any
import logging

from mengrowth.preprocessing.src.base import BasePreprocessingStep

logger = logging.getLogger(__name__)


class BaseNormalizer(BasePreprocessingStep):
    """Abstract base class for intensity normalization operations.

    Normalizers transform intensity distributions to harmonize images across
    different scanners, sites, and acquisition parameters. Implementations should
    preserve tissue ordering while mapping intensities to a comparable scale.

    Attributes:
        config: Configuration parameters for normalization
    """

    def __init__(
        self,
        config: Dict[str, Any],
        verbose: bool = False
    ) -> None:
        """Initialize normalizer.

        Args:
            config: Configuration dictionary with method-specific parameters
            verbose: Enable verbose logging
        """
        super().__init__(step_name="Normalizer", verbose=verbose)
        self.config = config

        self.logger.info(
            f"Initialized {self.__class__.__name__}"
        )

    @abstractmethod
    def execute(
        self,
        input_path: Path,
        output_path: Path,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Execute the normalization operation.

        Args:
            input_path: Path to input NIfTI file
            output_path: Path to output normalized NIfTI file
            **kwargs: Additional operation-specific parameters
                - allow_overwrite: Allow overwriting existing files (bool)

        Returns:
            Dictionary containing execution results, which may include:
                - Method-specific normalization parameters (e.g., mean, std, percentiles)
                - Original and normalized intensity ranges
                - Number of voxels processed

        Raises:
            FileNotFoundError: If input file does not exist
            RuntimeError: If normalization fails
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
        """Generate visualization comparing before and after normalization.

        The visualization should show:
        1. Before and after image slices (axial, sagittal, coronal views)
        2. Before and after intensity histograms
        3. Normalization parameters (metadata)

        Args:
            before_path: Path to input file (before normalization)
            after_path: Path to output file (after normalization)
            output_path: Path to save visualization output (PNG)
            **kwargs: Additional visualization parameters containing
                normalization-specific metadata from execute()

        Raises:
            FileNotFoundError: If input files do not exist
            RuntimeError: If visualization generation fails
        """
        pass
