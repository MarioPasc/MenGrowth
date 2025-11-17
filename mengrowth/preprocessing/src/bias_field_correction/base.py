"""Base classes for bias field correction preprocessing steps.

This module defines abstract base classes for bias field correction operations
following the OOP design pattern specified in CLAUDE.md. All bias field correction
implementations must inherit from these base classes and implement their abstract methods.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any
import logging

from mengrowth.preprocessing.src.data_harmonization.base import BasePreprocessingStep

logger = logging.getLogger(__name__)


class BaseBiasFieldCorrector(BasePreprocessingStep):
    """Abstract base class for bias field correction operations.

    Bias field correctors estimate and remove intensity non-uniformities caused
    by magnetic field inhomogeneities in MRI scans. Implementations should be
    robust and preserve anatomical structures while removing smooth intensity variations.

    Attributes:
        config: Configuration parameters for bias field correction
    """

    def __init__(self, config: Dict[str, Any], verbose: bool = False) -> None:
        """Initialize bias field corrector.

        Args:
            config: Configuration dictionary with method-specific parameters
            verbose: Enable verbose logging
        """
        super().__init__(step_name="BiasFieldCorrector", verbose=verbose)
        self.config = config

    @abstractmethod
    def execute(
        self,
        input_path: Path,
        output_path: Path,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Execute the bias field correction operation.

        Args:
            input_path: Path to input NIfTI file
            output_path: Path to output corrected NIfTI file
            **kwargs: Additional operation-specific parameters

        Returns:
            Dictionary containing execution results, including:
                - 'bias_field_path': Path to saved bias field (for visualization)
                - 'convergence_data': Convergence monitoring data (if applicable)

        Raises:
            FileNotFoundError: If input file does not exist
            RuntimeError: If correction fails
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
        """Generate visualization comparing before and after bias field correction.

        The visualization should show:
        1. Original image
        2. Original image with bias field overlay (alpha blending)
        3. Bias-field-corrected image
        4. Convergence monitoring plots (if applicable)

        Args:
            before_path: Path to input file (before correction)
            after_path: Path to output file (after correction)
            output_path: Path to save visualization output (PNG)
            **kwargs: Additional visualization parameters including:
                - 'bias_field_path': Path to bias field file
                - 'convergence_data': Convergence monitoring data

        Raises:
            FileNotFoundError: If input files do not exist
            RuntimeError: If visualization generation fails
        """
        pass
