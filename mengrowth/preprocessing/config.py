"""Configuration parser for preprocessing module.

This module provides typed configuration dataclasses and YAML loading utilities
for the preprocessing pipeline, following PEP 484 and PEP 257 standards.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml


@dataclass
class RawDataConfig:
    """Configuration for raw data reorganization.

    Attributes:
        study_mappings: Mapping from study directory names to standardized study numbers.
            E.g., {'baseline': '0', 'primera': '0', 'control1': '1', ...}
        modality_synonyms: Mapping from standardized modality names to lists of synonyms.
            E.g., {'FLAIR': ['FLAIR', 'flair', 'Flair'], 'T1pre': ['T1pre', 'T1', 'T1SIN'], ...}
        exclusion_patterns: List of glob patterns for files to exclude during copying.
            E.g., ['*seg*.nrrd', '*.h5', '*.dcm']
        output_structure: Template string for output directory structure.
            E.g., '{output_root}/MenGrowth-2025/P{patient_id}/{study_number}'
        input_sources: List of expected input source directory types.
            E.g., ['source/baseline/RM', 'source/controls', 'extension_1']
    """

    study_mappings: Dict[str, str]
    modality_synonyms: Dict[str, List[str]]
    exclusion_patterns: List[str]
    output_structure: str
    input_sources: List[str] = field(default_factory=list)

    def get_study_number(self, study_name: str) -> str:
        """Get standardized study number from directory name.

        Args:
            study_name: Directory name (e.g., 'primera', 'control1', 'baseline').

        Returns:
            Standardized study number as string (e.g., '0', '1', '2').

        Raises:
            KeyError: If study_name is not found in study_mappings.
        """
        study_lower = study_name.lower()
        if study_lower not in self.study_mappings:
            raise KeyError(
                f"Unknown study name '{study_name}'. "
                f"Available mappings: {list(self.study_mappings.keys())}"
            )
        return self.study_mappings[study_lower]

    def standardize_modality(self, filename: str) -> Tuple[str, bool]:
        """Extract and standardize modality name from filename.

        Args:
            filename: Original filename (e.g., 'FLAIR_P1.nrrd', 'T1ce-axial.nrrd').

        Returns:
            Tuple of (standardized_name, matched) where matched is True if a synonym
            was found, False if using original stem.

        Examples:
            >>> config.standardize_modality('FLAIR_P1.nrrd')
            ('FLAIR', True)
            >>> config.standardize_modality('T1ce-sagital.nrrd')
            ('T1ce', True)
            >>> config.standardize_modality('unknown_scan.nrrd')
            ('unknown_scan', False)
        """
        # Remove extension
        stem = Path(filename).stem

        # Check each standardized modality and its synonyms
        for standard_name, synonyms in self.modality_synonyms.items():
            for synonym in synonyms:
                # Case-insensitive matching with word boundary consideration
                if synonym.lower() in stem.lower():
                    return standard_name, True

        # If no match found, return the original stem
        return stem, False


@dataclass
class FilteringConfig:
    """Configuration for study filtering.

    Attributes:
        sequences: List of required sequences for the final dataset.
            E.g., ['t1c', 't1n', 't2f', 't2w']
        allowed_missing_sequences_per_study: Maximum number of sequences that can be
            missing from a study while still being accepted.
            E.g., 1 means a study can be missing at most 1 sequence from the required list.
        min_studies_per_patient: Minimum number of longitudinal studies required per patient.
            Patients with fewer studies will be removed entirely.
        orientation_priority: Priority order for selecting sequence orientations.
            Format: list ordered from highest to lowest priority.
            'none' means exact match (e.g., 't1c.nrrd'),
            other values match suffixed versions (e.g., 't1c-axial.nrrd').
            E.g., ['none', 'axial', 'sagital', 'coronal']
        keep_only_required_sequences: If True, delete all sequences not in the required list.
            E.g., if True and sequences=['t1c', 't1n', 't2f', 't2w'], delete dwi, swi, etc.
        reid_patients: If True, rename patients to MenGrowth-XXXX format and create id_mapping.json.
            E.g., P1 -> MenGrowth-0001, P42 -> MenGrowth-0002, etc.
            Studies are also renamed: MenGrowth-0001-000, MenGrowth-0001-001, etc.
            The id_mapping.json tracks both patient and study ID mappings.
    """

    sequences: List[str]
    allowed_missing_sequences_per_study: int
    min_studies_per_patient: int
    orientation_priority: List[str]
    keep_only_required_sequences: bool = False
    reid_patients: bool = False


@dataclass
class PreprocessingConfig:
    """Top-level configuration for preprocessing pipeline.

    Attributes:
        raw_data: Configuration for raw data reorganization tasks.
        filtering: Configuration for study filtering tasks (optional).
    """

    raw_data: RawDataConfig
    filtering: Optional[FilteringConfig] = None


def load_preprocessing_config(config_path: Path) -> PreprocessingConfig:
    """Load and validate preprocessing configuration from YAML file.

    Args:
        config_path: Path to preprocessing YAML configuration file.

    Returns:
        Validated PreprocessingConfig object with typed attributes.

    Raises:
        FileNotFoundError: If config_path does not exist.
        yaml.YAMLError: If YAML is malformed.
        KeyError: If required configuration keys are missing.
        TypeError: If configuration values have incorrect types.

    Examples:
        >>> config = load_preprocessing_config(Path('configs/preprocessing.yaml'))
        >>> print(config.raw_data.study_mappings['baseline'])
        '0'
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)

    if not yaml_data:
        raise ValueError(f"Empty configuration file: {config_path}")

    # Extract raw_data section
    if "raw_data" not in yaml_data:
        raise KeyError("Missing required 'raw_data' section in configuration")

    raw_data_dict = yaml_data["raw_data"]

    # Validate required fields
    required_fields = [
        "study_mappings",
        "modality_synonyms",
        "exclusion_patterns",
        "output_structure",
    ]
    for field_name in required_fields:
        if field_name not in raw_data_dict:
            raise KeyError(
                f"Missing required field '{field_name}' in raw_data configuration"
            )

    # Normalize study_mappings to lowercase keys
    study_mappings = {
        k.lower(): v for k, v in raw_data_dict["study_mappings"].items()
    }

    # Create RawDataConfig
    raw_data_config = RawDataConfig(
        study_mappings=study_mappings,
        modality_synonyms=raw_data_dict["modality_synonyms"],
        exclusion_patterns=raw_data_dict["exclusion_patterns"],
        output_structure=raw_data_dict["output_structure"],
        input_sources=raw_data_dict.get("input_sources", []),
    )

    # Parse optional filtering configuration
    filtering_config = None
    if "filtering" in raw_data_dict:
        filtering_dict = raw_data_dict["filtering"]

        # Validate required filtering fields
        filtering_required_fields = [
            "sequences",
            "allowed_missing_sequences_per_study",
            "min_studies_per_patient",
            "orientation_priority",
        ]
        for field_name in filtering_required_fields:
            if field_name not in filtering_dict:
                raise KeyError(
                    f"Missing required field '{field_name}' in filtering configuration"
                )

        filtering_config = FilteringConfig(
            sequences=filtering_dict["sequences"],
            allowed_missing_sequences_per_study=filtering_dict[
                "allowed_missing_sequences_per_study"
            ],
            min_studies_per_patient=filtering_dict["min_studies_per_patient"],
            orientation_priority=filtering_dict["orientation_priority"],
            keep_only_required_sequences=filtering_dict.get(
                "keep_only_required_sequences", False
            ),
            reid_patients=filtering_dict.get("reid_patients", False),
        )

    return PreprocessingConfig(raw_data=raw_data_config, filtering=filtering_config)
