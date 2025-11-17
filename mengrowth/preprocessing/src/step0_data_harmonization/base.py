"""Base classes for data harmonization preprocessing steps.

This module defines abstract base classes for preprocessing operations following
the OOP design pattern specified in CLAUDE.md. All preprocessing steps must inherit
from these base classes and implement their abstract methods.
"""

from mengrowth.preprocessing.src.base import BasePreprocessingStep
from typing import  Dict, Any
import logging

logger = logging.getLogger(__name__)

class BaseConverter(BasePreprocessingStep):
    """Abstract base class for format conversion operations.

    Format converters transform data from one file format to another
    (e.g., NRRD to NIfTI) while preserving medical imaging metadata.
    """

    def __init__(self, verbose: bool = False) -> None:
        """Initialize converter.

        Args:
            verbose: Enable verbose logging
        """
        super().__init__(step_name="Converter", verbose=verbose)


class BaseReorienter(BasePreprocessingStep):
    """Abstract base class for volume reorientation operations.

    Reorienters transform volume orientation to a target coordinate system
    (e.g., RAS or LPS) while preserving physical spacing and affine transforms.

    Attributes:
        target_orientation: Target orientation convention ("RAS" or "LPS")
    """

    def __init__(self, target_orientation: str, verbose: bool = False) -> None:
        """Initialize reorienter.

        Args:
            target_orientation: Target orientation ("RAS" or "LPS")
            verbose: Enable verbose logging
        """
        super().__init__(step_name="Reorienter", verbose=verbose)
        self.target_orientation = target_orientation

        if target_orientation not in ["RAS", "LPS"]:
            raise ValueError(
                f"Invalid orientation: {target_orientation}. Must be 'RAS' or 'LPS'"
            )


class BaseBackgroundRemover(BasePreprocessingStep):
    """Abstract base class for background removal operations.

    Background removers identify and zero out background/air voxels while
    preserving anatomical structures. Implementations must be conservative
    to avoid eroding anatomy.

    Attributes:
        config: Configuration parameters for background removal
    """

    def __init__(self, config: Dict[str, Any], verbose: bool = False) -> None:
        """Initialize background remover.

        Args:
            config: Configuration dictionary with method-specific parameters
            verbose: Enable verbose logging
        """
        super().__init__(step_name="BackgroundRemover", verbose=verbose)
        self.config = config
