"""Configuration dataclasses for BraTS meningioma segmentation.

This module defines the configuration for running BraTS 2025 Meningioma
Segmentation inference via Singularity on preprocessed MenGrowth data.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple
import logging
import yaml

logger = logging.getLogger(__name__)


class SegmentationConfigError(Exception):
    """Raised when segmentation configuration is invalid."""

    pass


@dataclass
class SegmentationConfig:
    """Configuration for BraTS meningioma segmentation pipeline.

    Attributes:
        input_root: Root directory containing preprocessed MenGrowth data.
        sif_path: Path to the Singularity SIF image.
        docker_image: Docker image name for pulling the SIF.
        modalities: List of required modalities.
        expected_shape: Expected volume shape (x, y, z).
        shape_tolerance: Allowed deviation from expected shape per axis.
        skip_incomplete: If True, skip studies missing modalities.
        output_filename: Name of the output segmentation file.
        work_dir: Working directory for BraTS input/output. Empty = auto-create.
        brats_name_schema: Format string for BraTS naming convention.
        log_dir: Directory for log files.
    """

    input_root: str = ""
    sif_path: str = ""
    docker_image: str = "brainles/brats25_men_qing:latest"
    modalities: List[str] = field(default_factory=lambda: ["t1c", "t1n", "t2w", "t2f"])
    expected_shape: Tuple[int, int, int] = (240, 240, 155)
    shape_tolerance: int = 2
    skip_incomplete: bool = True
    output_filename: str = "seg.nii.gz"
    work_dir: str = ""
    brats_name_schema: str = "BraTS-MEN-{:05d}-000"
    log_dir: str = ""

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if isinstance(self.expected_shape, list):
            self.expected_shape = tuple(self.expected_shape)

        if len(self.expected_shape) != 3:
            raise SegmentationConfigError(
                f"expected_shape must have 3 elements, got {len(self.expected_shape)}"
            )

        if self.shape_tolerance < 0:
            raise SegmentationConfigError(
                f"shape_tolerance must be >= 0, got {self.shape_tolerance}"
            )

        if not self.modalities:
            raise SegmentationConfigError("modalities list cannot be empty")


def load_segmentation_config(yaml_path: str | Path) -> SegmentationConfig:
    """Load segmentation configuration from a YAML file.

    Args:
        yaml_path: Path to the YAML configuration file.

    Returns:
        Validated SegmentationConfig object.

    Raises:
        FileNotFoundError: If config file does not exist.
        SegmentationConfigError: If configuration is invalid.
    """
    yaml_path = Path(yaml_path)

    if not yaml_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

    logger.info(f"Loading segmentation config from {yaml_path}")

    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise SegmentationConfigError(f"Failed to parse YAML: {e}") from e

    if not yaml_data:
        raise SegmentationConfigError("Configuration file is empty")

    if "segmentation" not in yaml_data:
        raise SegmentationConfigError(
            "Configuration must contain 'segmentation' top-level key"
        )

    seg_data = yaml_data["segmentation"]

    try:
        config = SegmentationConfig(**seg_data)
    except TypeError as e:
        raise SegmentationConfigError(f"Invalid configuration: {e}") from e

    logger.info("Segmentation configuration loaded successfully")
    return config
