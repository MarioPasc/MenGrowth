"""Pipeline logging utilities for tracking preprocessing execution.

This module provides logging functionality to record the complete pipeline
configuration including step order, configurations, and execution parameters.
"""

from dataclasses import dataclass, asdict, is_dataclass
from typing import List, Dict, Any
from datetime import datetime
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class PipelineOrderRecord:
    """Record of pipeline execution order and configuration.

    This record captures all information needed to reproduce a preprocessing run,
    including step order, configurations, and execution parameters.

    Attributes:
        timestamp: ISO format timestamp of when pipeline was configured
        patient_id: Patient identifier
        mode: Execution mode ("test" or "pipeline")
        step_order: Ordered list of step names as they will execute
        step_configs: Dictionary mapping step patterns to their configurations
        modalities: List of modalities to process
        dataset_root: Root directory for input data
        output_root: Root directory for output (test mode only)
        preprocessing_artifacts_path: Directory for intermediate artifacts
        overwrite: Whether existing files will be overwritten
    """
    timestamp: str
    patient_id: str
    mode: str
    step_order: List[str]
    step_configs: Dict[str, Any]
    modalities: List[str]
    dataset_root: str
    output_root: str
    preprocessing_artifacts_path: str
    overwrite: bool

    @classmethod
    def from_config(cls, patient_id: str, config: Any) -> 'PipelineOrderRecord':
        """Create record from configuration.

        Args:
            patient_id: Patient identifier
            config: DataHarmonizationConfig instance

        Returns:
            PipelineOrderRecord instance
        """
        return cls(
            timestamp=datetime.now().isoformat(),
            patient_id=patient_id,
            mode=config.mode,
            step_order=config.steps.copy(),
            step_configs=cls._serialize_configs(config.step_configs),
            modalities=config.modalities.copy(),
            dataset_root=config.dataset_root,
            output_root=config.output_root,
            preprocessing_artifacts_path=config.preprocessing_artifacts_path,
            overwrite=config.overwrite
        )

    @staticmethod
    def _to_dict(obj: Any) -> Any:
        """Recursively convert objects to dictionary."""
        if is_dataclass(obj) and not isinstance(obj, type):
            return asdict(obj)
        elif hasattr(obj, '__dict__'):
            return {k: PipelineOrderRecord._to_dict(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, list):
            return [PipelineOrderRecord._to_dict(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: PipelineOrderRecord._to_dict(v) for k, v in obj.items()}
        else:
            return obj

    @staticmethod
    def _serialize_configs(step_configs: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize step configs to JSON-compatible format.

        Converts dataclass configs and DictConfig objects to dictionaries for JSON serialization.

        Args:
            step_configs: Dictionary of step configurations

        Returns:
            JSON-serializable dictionary
        """
        serialized = {}
        for key, config in step_configs.items():
            try:
                serialized[key] = PipelineOrderRecord._to_dict(config)
            except Exception as e:
                logger.warning(f"Could not serialize config for '{key}': {e}")
                serialized[key] = str(config)
        return serialized

    def save(self, output_path: Path) -> None:
        """Save record to JSON file.

        Creates parent directories if they don't exist.

        Args:
            output_path: Path to output JSON file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
        logger.info(f"Pipeline order saved to: {output_path}")
