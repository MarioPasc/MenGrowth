"""Basic checkpoint system for preprocessing pipeline.

This module provides a simple checkpoint system that saves state after each
preprocessing step, allowing manual recovery if the pipeline fails.

No automatic rollback is provided - users can manually re-run from a checkpoint
if needed.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class CheckpointState:
    """State information for a checkpoint.

    Attributes:
        patient_id: Patient identifier
        study_id: Study identifier
        modality: Modality being processed
        completed_steps: List of step names that have completed
        step_outputs: Mapping of step name to output file path
        step_metadata: Mapping of step name to metadata dict
        created_at: ISO timestamp when checkpoint was created
        updated_at: ISO timestamp when checkpoint was last updated
    """
    patient_id: str
    study_id: str
    modality: str
    completed_steps: List[str] = field(default_factory=list)
    step_outputs: Dict[str, str] = field(default_factory=dict)
    step_metadata: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointState":
        """Create instance from dictionary."""
        return cls(**data)


class CheckpointManager:
    """Manages checkpoint state for the preprocessing pipeline.

    This class provides:
    - Saving checkpoint state after each step
    - Loading checkpoint state to resume processing
    - Determining which step to resume from

    The checkpoint system is designed to be simple and non-intrusive:
    - No automatic rollback on failure
    - Manual intervention required to resume from checkpoint
    - Checkpoints are JSON files for easy inspection

    Attributes:
        checkpoint_dir: Directory where checkpoint files are stored
        enabled: Whether checkpointing is enabled
    """

    def __init__(
        self,
        checkpoint_dir: Path,
        enabled: bool = True
    ) -> None:
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory for checkpoint files
            enabled: Whether to enable checkpointing
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.enabled = enabled
        self._logger = logging.getLogger(__name__)

        if self.enabled:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self._logger.info(f"CheckpointManager initialized: {self.checkpoint_dir}")

    def _get_checkpoint_path(
        self,
        patient_id: str,
        study_id: str,
        modality: str
    ) -> Path:
        """Get path to checkpoint file for a specific case.

        Args:
            patient_id: Patient identifier
            study_id: Study identifier
            modality: Modality name

        Returns:
            Path to checkpoint JSON file
        """
        filename = f"{patient_id}_{study_id}_{modality}.json"
        return self.checkpoint_dir / filename

    def save_checkpoint(
        self,
        patient_id: str,
        study_id: str,
        modality: str,
        step_name: str,
        output_path: Path,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Save checkpoint after a step completes.

        Args:
            patient_id: Patient identifier
            study_id: Study identifier
            modality: Modality name
            step_name: Name of the completed step
            output_path: Path to the step's output file
            metadata: Optional metadata from the step execution
        """
        if not self.enabled:
            return

        checkpoint_path = self._get_checkpoint_path(patient_id, study_id, modality)

        # Load existing checkpoint or create new
        state = self.load_checkpoint(patient_id, study_id, modality)
        if state is None:
            state = CheckpointState(
                patient_id=patient_id,
                study_id=study_id,
                modality=modality
            )

        # Update state
        if step_name not in state.completed_steps:
            state.completed_steps.append(step_name)
        state.step_outputs[step_name] = str(output_path)
        if metadata:
            state.step_metadata[step_name] = metadata
        state.updated_at = datetime.now().isoformat()

        # Save to file
        with open(checkpoint_path, 'w') as f:
            json.dump(state.to_dict(), f, indent=2, default=str)

        self._logger.debug(
            f"Checkpoint saved: {patient_id}/{study_id}/{modality} "
            f"step={step_name} ({len(state.completed_steps)} steps completed)"
        )

    def load_checkpoint(
        self,
        patient_id: str,
        study_id: str,
        modality: str
    ) -> Optional[CheckpointState]:
        """Load checkpoint state for a specific case.

        Args:
            patient_id: Patient identifier
            study_id: Study identifier
            modality: Modality name

        Returns:
            CheckpointState if checkpoint exists, None otherwise
        """
        if not self.enabled:
            return None

        checkpoint_path = self._get_checkpoint_path(patient_id, study_id, modality)

        if not checkpoint_path.exists():
            return None

        try:
            with open(checkpoint_path, 'r') as f:
                data = json.load(f)
            return CheckpointState.from_dict(data)
        except Exception as e:
            self._logger.warning(f"Failed to load checkpoint {checkpoint_path}: {e}")
            return None

    def get_resume_step(
        self,
        patient_id: str,
        study_id: str,
        modality: str,
        pipeline_steps: List[str]
    ) -> Optional[str]:
        """Determine which step to resume from.

        Args:
            patient_id: Patient identifier
            study_id: Study identifier
            modality: Modality name
            pipeline_steps: Ordered list of step names in the pipeline

        Returns:
            Name of the first step that hasn't completed, or None if all done
        """
        state = self.load_checkpoint(patient_id, study_id, modality)

        if state is None:
            # No checkpoint - start from beginning
            return pipeline_steps[0] if pipeline_steps else None

        completed = set(state.completed_steps)

        for step in pipeline_steps:
            if step not in completed:
                self._logger.info(
                    f"Resuming {patient_id}/{study_id}/{modality} from step: {step}"
                )
                return step

        # All steps completed
        self._logger.info(
            f"All steps completed for {patient_id}/{study_id}/{modality}"
        )
        return None

    def is_step_completed(
        self,
        patient_id: str,
        study_id: str,
        modality: str,
        step_name: str
    ) -> bool:
        """Check if a specific step has completed.

        Args:
            patient_id: Patient identifier
            study_id: Study identifier
            modality: Modality name
            step_name: Name of the step to check

        Returns:
            True if step has completed, False otherwise
        """
        state = self.load_checkpoint(patient_id, study_id, modality)
        if state is None:
            return False
        return step_name in state.completed_steps

    def clear_checkpoint(
        self,
        patient_id: str,
        study_id: str,
        modality: str
    ) -> bool:
        """Clear checkpoint for a specific case (for re-processing).

        Args:
            patient_id: Patient identifier
            study_id: Study identifier
            modality: Modality name

        Returns:
            True if checkpoint was deleted, False if it didn't exist
        """
        checkpoint_path = self._get_checkpoint_path(patient_id, study_id, modality)

        if checkpoint_path.exists():
            checkpoint_path.unlink()
            self._logger.info(f"Cleared checkpoint: {checkpoint_path}")
            return True
        return False

    def get_all_checkpoints(self) -> List[CheckpointState]:
        """Get all checkpoint states in the checkpoint directory.

        Returns:
            List of CheckpointState objects
        """
        if not self.enabled:
            return []

        checkpoints = []
        for checkpoint_file in self.checkpoint_dir.glob("*.json"):
            try:
                with open(checkpoint_file, 'r') as f:
                    data = json.load(f)
                checkpoints.append(CheckpointState.from_dict(data))
            except Exception as e:
                self._logger.warning(f"Failed to load checkpoint {checkpoint_file}: {e}")

        return checkpoints

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all checkpoints.

        Returns:
            Dict with checkpoint statistics
        """
        checkpoints = self.get_all_checkpoints()

        if not checkpoints:
            return {
                "total_checkpoints": 0,
                "patients": [],
                "fully_completed": 0,
                "in_progress": 0,
            }

        patients = set(cp.patient_id for cp in checkpoints)
        fully_completed = sum(1 for cp in checkpoints if len(cp.completed_steps) > 0)

        return {
            "total_checkpoints": len(checkpoints),
            "patients": list(patients),
            "fully_completed": fully_completed,
            "checkpoint_dir": str(self.checkpoint_dir),
        }
