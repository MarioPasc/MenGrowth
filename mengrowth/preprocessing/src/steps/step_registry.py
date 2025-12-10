"""Step registry for dynamic preprocessing pipeline execution.

This module re-exports step registry infrastructure from config.py for backwards compatibility.
All classes are now defined in config.py to avoid duplication.
"""

# Re-export from config.py for backwards compatibility
from mengrowth.preprocessing.src.config import (
    StepExecutionContext,
    StepRegistry,
    StepMetadata,
    STEP_METADATA,
)

__all__ = [
    'StepExecutionContext',
    'StepRegistry',
    'StepMetadata',
    'STEP_METADATA',
]
