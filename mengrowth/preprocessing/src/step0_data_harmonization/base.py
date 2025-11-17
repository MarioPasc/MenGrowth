"""Base classes for data harmonization preprocessing steps.

This module defines abstract base classes for preprocessing operations following
the OOP design pattern specified in CLAUDE.md. All preprocessing steps must inherit
from these base classes and implement their abstract methods.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class BasePreprocessingStep(ABC):
    """Abstract base class for all preprocessing steps.

    All preprocessing operations must inherit from this class and implement
    the execute() and visualize() methods. This ensures consistent interface
    and mandatory visualization capabilities.

    Attributes:
        step_name: Human-readable name of this preprocessing step
        verbose: Whether to enable verbose logging
    """

    def __init__(self, step_name: str, verbose: bool = False) -> None:
        """Initialize preprocessing step.

        Args:
            step_name: Name of this preprocessing step
            verbose: Enable verbose logging
        """
        self.step_name = step_name
        self.verbose = verbose
        self.logger = logging.getLogger(f"{__name__}.{step_name}")

    @abstractmethod
    def execute(
        self,
        input_path: Path,
        output_path: Path,
        **kwargs: Any
    ) -> None:
        """Execute the preprocessing operation.

        Args:
            input_path: Path to input file
            output_path: Path to output file
            **kwargs: Additional operation-specific parameters

        Raises:
            FileNotFoundError: If input file does not exist
            RuntimeError: If operation fails
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
        """Generate visualization comparing before and after states.

        Args:
            before_path: Path to input file (before preprocessing)
            after_path: Path to output file (after preprocessing)
            output_path: Path to save visualization output
            **kwargs: Additional visualization parameters

        Raises:
            FileNotFoundError: If input files do not exist
            RuntimeError: If visualization generation fails
        """
        pass

    def validate_inputs(self, input_path: Path) -> None:
        """Validate input file exists and is readable.

        Args:
            input_path: Path to validate

        Raises:
            FileNotFoundError: If file does not exist
        """
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        self.logger.debug(f"Input validation passed: {input_path}")

    def validate_outputs(self, output_path: Path, allow_overwrite: bool = False) -> None:
        """Validate output path and check overwrite conditions.

        Args:
            output_path: Path to validate
            allow_overwrite: Whether to allow overwriting existing files

        Raises:
            FileExistsError: If file exists and overwrite is not allowed
        """
        if output_path.exists() and not allow_overwrite:
            raise FileExistsError(
                f"Output file exists and overwrite=False: {output_path}"
            )

        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.logger.debug(f"Output validation passed: {output_path}")

    def log_execution(self, input_path: Path, output_path: Path) -> None:
        """Log execution details.

        Args:
            input_path: Input file path
            output_path: Output file path
        """
        self.logger.info(
            f"[{self.step_name}] Processing: {input_path.name} -> {output_path.name}"
        )


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
