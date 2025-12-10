"""Shared utilities for preprocessing step modules.

This module provides common functionality used across multiple step implementations,
including path generation, visualization helpers, and logging utilities.
"""

from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from mengrowth.preprocessing.src.config import StepExecutionContext


def get_visualization_path(
    context: 'StepExecutionContext',
    suffix: str = "",
    extension: str = ".png"
) -> Path:
    """Generate standardized visualization output path.

    Creates a consistent path structure for visualization files across all steps:
    {viz_root}/{patient_id}/{study_name}/{step_name}_{modality}{suffix}{extension}

    Args:
        context: Step execution context containing patient, study, and modality info
        suffix: Optional suffix to add before file extension (e.g., "_overlay")
        extension: File extension (default: ".png")

    Returns:
        Path object for the visualization file

    Example:
        >>> path = get_visualization_path(context, suffix="_histogram")
        >>> # Returns: /viz_root/MenGrowth-0001/study-000/intensity_normalization_t1c_histogram.png
    """
    orchestrator = context.orchestrator
    viz_root = Path(orchestrator.config.viz_root)
    
    study_name = context.study_dir.name
    modality_part = f"_{context.modality}" if context.modality else ""
    
    viz_path = (
        viz_root / 
        context.patient_id / 
        study_name / 
        f"{context.step_name}{modality_part}{suffix}{extension}"
    )
    
    # Ensure parent directory exists
    viz_path.parent.mkdir(parents=True, exist_ok=True)
    
    return viz_path


def get_artifact_path(
    context: 'StepExecutionContext',
    artifact_name: str,
    extension: str = ".nii.gz"
) -> Path:
    """Generate standardized artifact output path.

    Creates a consistent path structure for artifact files (intermediate outputs):
    {artifacts_root}/{patient_id}/{study_name}/{artifact_name}{extension}

    Args:
        context: Step execution context
        artifact_name: Name for the artifact file (e.g., "t1c_bias_field")
        extension: File extension (default: ".nii.gz")

    Returns:
        Path object for the artifact file
    """
    orchestrator = context.orchestrator
    artifacts_root = Path(orchestrator.config.preprocessing_artifacts_path)
    
    study_name = context.study_dir.name
    
    artifact_path = (
        artifacts_root / 
        context.patient_id / 
        study_name / 
        f"{artifact_name}{extension}"
    )
    
    # Ensure parent directory exists
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    
    return artifact_path


def get_output_dir(context: 'StepExecutionContext') -> Path:
    """Get the output directory for the current study based on mode.

    In test mode, returns the separate output directory.
    In pipeline mode, returns the original study directory.

    Args:
        context: Step execution context

    Returns:
        Path to the output directory
    """
    orchestrator = context.orchestrator
    
    if orchestrator.config.mode == "test":
        output_dir = (
            Path(orchestrator.config.output_root) / 
            context.patient_id / 
            context.study_dir.name
        )
    else:
        output_dir = context.study_dir
    
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_temp_path(
    context: 'StepExecutionContext',
    modality: Optional[str] = None,
    operation: str = "temp"
) -> Path:
    """Generate a temporary file path for intermediate processing.

    Args:
        context: Step execution context
        modality: Modality name (uses context.modality if not provided)
        operation: Description of the operation (e.g., "resampled", "normalized")

    Returns:
        Path for the temporary file
    """
    mod = modality or context.modality or "unknown"
    output_dir = get_output_dir(context)
    
    return output_dir / f"_temp_{mod}_{operation}_{context.step_name}.nii.gz"


def format_step_log(
    step_num: int,
    total_steps: int,
    step_name: str,
    method: Optional[str] = None
) -> str:
    """Format a consistent step execution log message.

    Args:
        step_num: Current step number (1-indexed)
        total_steps: Total number of steps
        step_name: Name of the step being executed
        method: Optional method being used (e.g., "n4", "fcm")

    Returns:
        Formatted log string

    Example:
        >>> format_step_log(2, 5, "bias_field_correction", "n4")
        '[2/5] Executing: bias_field_correction (n4)'
    """
    method_part = f" ({method})" if method else ""
    return f"[{step_num}/{total_steps}] Executing: {step_name}{method_part}"


def log_step_start(
    logger,
    step_num: int,
    total_steps: int,
    step_name: str,
    method: Optional[str] = None,
    indent: str = "    "
) -> None:
    """Log the start of a step execution with consistent formatting.

    Args:
        logger: Logger instance
        step_num: Current step number
        total_steps: Total number of steps
        step_name: Name of the step
        method: Optional method being used
        indent: Indentation prefix for the log message
    """
    msg = format_step_log(step_num, total_steps, step_name, method)
    logger.info(f"{indent}{msg}")


def log_substep(logger, message: str, indent: str = "        - ") -> None:
    """Log a substep operation with consistent indentation.

    Args:
        logger: Logger instance
        message: Message to log
        indent: Indentation prefix (default: 8 spaces + dash)
    """
    logger.info(f"{indent}{message}")
