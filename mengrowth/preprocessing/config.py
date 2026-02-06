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
class MetadataConfig:
    """Configuration for clinical metadata processing.

    Attributes:
        xlsx_path: Path to the clinical metadata xlsx file.
        enabled: Enable metadata processing during curation.
        output_csv_name: Filename for enriched metadata CSV output.
        output_json_name: Filename for clean metadata JSON output.
        warn_missing_metadata: Log warnings for patients without metadata.
    """

    xlsx_path: Optional[str] = None
    enabled: bool = False
    output_csv_name: str = "metadata_enriched.csv"
    output_json_name: str = "metadata_clean.json"
    warn_missing_metadata: bool = True


@dataclass
class NRRDValidationConfig:
    """Configuration for NRRD header validation."""

    enabled: bool = True
    require_3d: bool = True
    require_space_field: bool = True


@dataclass
class ScoutDetectionConfig:
    """Configuration for scout/localizer detection."""

    enabled: bool = True
    min_dimension_voxels: int = 64
    max_slice_thickness_mm: float = 5.0


@dataclass
class VoxelSpacingConfig:
    """Configuration for voxel spacing validation."""

    enabled: bool = True
    min_spacing_mm: float = 0.3
    max_spacing_mm: float = 3.0
    max_anisotropy_ratio: float = 20.0
    action: str = "warn"  # "warn" or "block"


@dataclass
class SNRFilteringConfig:
    """Configuration for SNR-based filtering."""

    enabled: bool = True
    min_snr: float = 5.0
    method: str = "corner"  # "corner" or "background_percentile"
    corner_cube_size: int = 10
    modality_thresholds: Dict[str, float] = field(
        default_factory=lambda: {"t1c": 8.0, "t1n": 6.0, "t2w": 5.0, "t2f": 4.0}
    )
    action: str = "block"

    def get_threshold(self, modality: str) -> float:
        """Get SNR threshold for a modality, falling back to min_snr."""
        return self.modality_thresholds.get(modality, self.min_snr)


@dataclass
class ContrastDetectionConfig:
    """Configuration for contrast/uniformity detection."""

    enabled: bool = True
    min_std_ratio: float = 0.10  # std/mean minimum
    max_uniform_fraction: float = 0.95
    action: str = "block"


@dataclass
class IntensityOutliersConfig:
    """Configuration for intensity outlier detection."""

    enabled: bool = True
    reject_nan_inf: bool = True
    max_outlier_ratio: float = 10.0  # max vs 99th percentile
    modality_thresholds: Dict[str, float] = field(
        default_factory=lambda: {"t1c": 10.0, "t1n": 15.0, "t2w": 12.0, "t2f": 30.0}
    )
    action: str = "warn"

    def get_threshold(self, modality: str) -> float:
        """Get outlier ratio threshold for a modality, falling back to max_outlier_ratio."""
        return self.modality_thresholds.get(modality, self.max_outlier_ratio)


@dataclass
class AffineValidationConfig:
    """Configuration for affine matrix validation."""

    enabled: bool = True
    min_det: float = 0.1
    max_det: float = 100.0
    action: str = "block"


@dataclass
class FOVConsistencyConfig:
    """Configuration for field-of-view consistency checking."""

    enabled: bool = True
    warn_ratio: float = 3.0
    block_ratio: float = 5.0


@dataclass
class OrientationConsistencyConfig:
    """Configuration for orientation consistency within study."""

    enabled: bool = True
    action: str = "warn"


@dataclass
class TemporalOrderingConfig:
    """Configuration for temporal ordering validation."""

    enabled: bool = True
    action: str = "warn"  # Always warn-only


@dataclass
class MotionArtifactConfig:
    """Configuration for motion artifact detection."""

    enabled: bool = True
    min_gradient_entropy: float = 3.0
    action: str = "warn"


@dataclass
class BrainCoverageConfig:
    """Configuration for brain coverage validation."""

    enabled: bool = True
    min_extent_mm: float = 100.0
    action: str = "block"


@dataclass
class GhostingDetectionConfig:
    """Configuration for ghosting artifact detection."""

    enabled: bool = True
    max_corner_to_foreground_ratio: float = 0.15
    corner_cube_size: int = 10
    action: str = "warn"


@dataclass
class ModalityConsistencyConfig:
    """Configuration for modality consistency across timepoints."""

    enabled: bool = False
    action: str = "warn"


@dataclass
class RegistrationReferenceConfig:
    """Configuration for registration reference availability check."""

    enabled: bool = True
    priority: str = "t1n > t1c > t2f > t2w"
    action: str = "block"


@dataclass
class QualityFilteringConfig:
    """Configuration for quality-based filtering during data curation.

    Attributes:
        enabled: Master switch to enable/disable all quality filtering.
        remove_blocked: If True, remove studies/patients with blocking issues.
        min_studies_per_patient: Minimum clean studies to keep a patient.
        nrrd_validation: NRRD header validation settings.
        scout_detection: Scout/localizer detection settings.
        voxel_spacing: Voxel spacing validation settings.
        snr_filtering: SNR-based filtering settings.
        contrast_detection: Contrast/uniformity detection settings.
        intensity_outliers: Intensity outlier detection settings.
        affine_validation: Affine matrix validation settings.
        fov_consistency: Field-of-view consistency settings.
        orientation_consistency: Orientation consistency settings.
        temporal_ordering: Temporal ordering validation settings.
        motion_artifact: Motion artifact detection settings.
        brain_coverage: Brain coverage validation settings.
        ghosting_detection: Ghosting artifact detection settings.
        modality_consistency: Modality consistency settings.
        registration_reference: Registration reference availability settings.
    """

    enabled: bool = True
    remove_blocked: bool = True
    min_studies_per_patient: int = 2
    nrrd_validation: NRRDValidationConfig = field(default_factory=NRRDValidationConfig)
    scout_detection: ScoutDetectionConfig = field(default_factory=ScoutDetectionConfig)
    voxel_spacing: VoxelSpacingConfig = field(default_factory=VoxelSpacingConfig)
    snr_filtering: SNRFilteringConfig = field(default_factory=SNRFilteringConfig)
    contrast_detection: ContrastDetectionConfig = field(
        default_factory=ContrastDetectionConfig
    )
    intensity_outliers: IntensityOutliersConfig = field(
        default_factory=IntensityOutliersConfig
    )
    affine_validation: AffineValidationConfig = field(
        default_factory=AffineValidationConfig
    )
    fov_consistency: FOVConsistencyConfig = field(default_factory=FOVConsistencyConfig)
    orientation_consistency: OrientationConsistencyConfig = field(
        default_factory=OrientationConsistencyConfig
    )
    temporal_ordering: TemporalOrderingConfig = field(
        default_factory=TemporalOrderingConfig
    )
    motion_artifact: MotionArtifactConfig = field(
        default_factory=MotionArtifactConfig
    )
    brain_coverage: BrainCoverageConfig = field(
        default_factory=BrainCoverageConfig
    )
    ghosting_detection: GhostingDetectionConfig = field(
        default_factory=GhostingDetectionConfig
    )
    modality_consistency: ModalityConsistencyConfig = field(
        default_factory=ModalityConsistencyConfig
    )
    registration_reference: RegistrationReferenceConfig = field(
        default_factory=RegistrationReferenceConfig
    )


@dataclass
class PreprocessingConfig:
    """Top-level configuration for preprocessing pipeline.

    Attributes:
        raw_data: Configuration for raw data reorganization tasks.
        filtering: Configuration for study filtering tasks (optional).
        metadata: Configuration for clinical metadata processing (optional).
        quality_filtering: Configuration for quality-based filtering (optional).
    """

    raw_data: RawDataConfig
    filtering: Optional[FilteringConfig] = None
    metadata: Optional[MetadataConfig] = None
    quality_filtering: Optional[QualityFilteringConfig] = None


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

    # Parse optional metadata configuration
    metadata_config = None
    if "metadata" in raw_data_dict:
        metadata_dict = raw_data_dict["metadata"]
        metadata_config = MetadataConfig(
            xlsx_path=metadata_dict.get("xlsx_path"),
            enabled=metadata_dict.get("enabled", False),
            output_csv_name=metadata_dict.get("output_csv_name", "metadata_enriched.csv"),
            output_json_name=metadata_dict.get("output_json_name", "metadata_clean.json"),
            warn_missing_metadata=metadata_dict.get("warn_missing_metadata", True),
        )

    # Parse optional quality_filtering configuration
    quality_filtering_config = None
    if "quality_filtering" in raw_data_dict:
        qf_dict = raw_data_dict["quality_filtering"]
        quality_filtering_config = _parse_quality_filtering_config(qf_dict)

    return PreprocessingConfig(
        raw_data=raw_data_config,
        filtering=filtering_config,
        metadata=metadata_config,
        quality_filtering=quality_filtering_config,
    )


def _parse_quality_filtering_config(qf_dict: dict) -> QualityFilteringConfig:
    """Parse quality filtering configuration from dictionary.

    Args:
        qf_dict: Dictionary containing quality_filtering section from YAML.

    Returns:
        Validated QualityFilteringConfig object.
    """
    # Parse nested configs with defaults
    nrrd_dict = qf_dict.get("nrrd_validation", {})
    nrrd_config = NRRDValidationConfig(
        enabled=nrrd_dict.get("enabled", True),
        require_3d=nrrd_dict.get("require_3d", True),
        require_space_field=nrrd_dict.get("require_space_field", True),
    )

    scout_dict = qf_dict.get("scout_detection", {})
    scout_config = ScoutDetectionConfig(
        enabled=scout_dict.get("enabled", True),
        min_dimension_voxels=scout_dict.get("min_dimension_voxels", 64),
        max_slice_thickness_mm=scout_dict.get("max_slice_thickness_mm", 5.0),
    )

    spacing_dict = qf_dict.get("voxel_spacing", {})
    spacing_config = VoxelSpacingConfig(
        enabled=spacing_dict.get("enabled", True),
        min_spacing_mm=spacing_dict.get("min_spacing_mm", 0.3),
        max_spacing_mm=spacing_dict.get("max_spacing_mm", 3.0),
        max_anisotropy_ratio=spacing_dict.get("max_anisotropy_ratio", 3.0),
        action=spacing_dict.get("action", "warn"),
    )

    snr_dict = qf_dict.get("snr_filtering", {})
    snr_config = SNRFilteringConfig(
        enabled=snr_dict.get("enabled", True),
        min_snr=snr_dict.get("min_snr", 5.0),
        method=snr_dict.get("method", "corner"),
        corner_cube_size=snr_dict.get("corner_cube_size", 10),
        modality_thresholds=snr_dict.get(
            "modality_thresholds", {"t1c": 8.0, "t1n": 6.0, "t2w": 5.0, "t2f": 4.0}
        ),
        action=snr_dict.get("action", "block"),
    )

    contrast_dict = qf_dict.get("contrast_detection", {})
    contrast_config = ContrastDetectionConfig(
        enabled=contrast_dict.get("enabled", True),
        min_std_ratio=contrast_dict.get("min_std_ratio", 0.10),
        max_uniform_fraction=contrast_dict.get("max_uniform_fraction", 0.95),
        action=contrast_dict.get("action", "block"),
    )

    intensity_dict = qf_dict.get("intensity_outliers", {})
    intensity_config = IntensityOutliersConfig(
        enabled=intensity_dict.get("enabled", True),
        reject_nan_inf=intensity_dict.get("reject_nan_inf", True),
        max_outlier_ratio=intensity_dict.get("max_outlier_ratio", 10.0),
        modality_thresholds=intensity_dict.get(
            "modality_thresholds",
            {"t1c": 10.0, "t1n": 15.0, "t2w": 12.0, "t2f": 30.0},
        ),
        action=intensity_dict.get("action", "warn"),
    )

    affine_dict = qf_dict.get("affine_validation", {})
    affine_config = AffineValidationConfig(
        enabled=affine_dict.get("enabled", True),
        min_det=affine_dict.get("min_det", 0.1),
        max_det=affine_dict.get("max_det", 100.0),
        action=affine_dict.get("action", "block"),
    )

    fov_dict = qf_dict.get("fov_consistency", {})
    fov_config = FOVConsistencyConfig(
        enabled=fov_dict.get("enabled", True),
        warn_ratio=fov_dict.get("warn_ratio", 3.0),
        block_ratio=fov_dict.get("block_ratio", 5.0),
    )

    orient_dict = qf_dict.get("orientation_consistency", {})
    orient_config = OrientationConsistencyConfig(
        enabled=orient_dict.get("enabled", True),
        action=orient_dict.get("action", "warn"),
    )

    temporal_dict = qf_dict.get("temporal_ordering", {})
    temporal_config = TemporalOrderingConfig(
        enabled=temporal_dict.get("enabled", True),
        action=temporal_dict.get("action", "warn"),
    )

    motion_dict = qf_dict.get("motion_artifact", {})
    motion_config = MotionArtifactConfig(
        enabled=motion_dict.get("enabled", True),
        min_gradient_entropy=motion_dict.get("min_gradient_entropy", 3.0),
        action=motion_dict.get("action", "warn"),
    )

    coverage_dict = qf_dict.get("brain_coverage", {})
    coverage_config = BrainCoverageConfig(
        enabled=coverage_dict.get("enabled", True),
        min_extent_mm=coverage_dict.get("min_extent_mm", 100.0),
        action=coverage_dict.get("action", "block"),
    )

    ghosting_dict = qf_dict.get("ghosting_detection", {})
    ghosting_config = GhostingDetectionConfig(
        enabled=ghosting_dict.get("enabled", True),
        max_corner_to_foreground_ratio=ghosting_dict.get(
            "max_corner_to_foreground_ratio", 0.15
        ),
        corner_cube_size=ghosting_dict.get("corner_cube_size", 10),
        action=ghosting_dict.get("action", "warn"),
    )

    modality_dict = qf_dict.get("modality_consistency", {})
    modality_config = ModalityConsistencyConfig(
        enabled=modality_dict.get("enabled", False),
        action=modality_dict.get("action", "warn"),
    )

    reg_ref_dict = qf_dict.get("registration_reference", {})
    reg_ref_config = RegistrationReferenceConfig(
        enabled=reg_ref_dict.get("enabled", True),
        priority=reg_ref_dict.get("priority", "t1n > t1c > t2f > t2w"),
        action=reg_ref_dict.get("action", "block"),
    )

    return QualityFilteringConfig(
        enabled=qf_dict.get("enabled", True),
        remove_blocked=qf_dict.get("remove_blocked", True),
        min_studies_per_patient=qf_dict.get("min_studies_per_patient", 2),
        nrrd_validation=nrrd_config,
        scout_detection=scout_config,
        voxel_spacing=spacing_config,
        snr_filtering=snr_config,
        contrast_detection=contrast_config,
        intensity_outliers=intensity_config,
        affine_validation=affine_config,
        fov_consistency=fov_config,
        orientation_consistency=orient_config,
        temporal_ordering=temporal_config,
        motion_artifact=motion_config,
        brain_coverage=coverage_config,
        ghosting_detection=ghosting_config,
        modality_consistency=modality_config,
        registration_reference=reg_ref_config,
    )


@dataclass
class MetricsConfig:
    """Configuration for quality metrics computation.

    Attributes:
        patient_statistics: Compute studies per patient statistics.
        missing_sequences: Compute missing sequence counts and percentages.
        voxel_spacing: Compute physical voxel spacing (x, y, z) in mm.
        intensity_statistics: Compute intensity value statistics and outliers.
        image_dimensions: Compute image shape statistics.
        acquisition_consistency: Compute variability within patients.
        snr_estimation: Compute signal-to-noise ratio approximations.
    """

    patient_statistics: bool = True
    missing_sequences: bool = True
    voxel_spacing: bool = True
    intensity_statistics: bool = True
    image_dimensions: bool = True
    acquisition_consistency: bool = True
    snr_estimation: bool = True


@dataclass
class OutlierDetectionConfig:
    """Configuration for intensity outlier detection.

    Attributes:
        enabled: Enable outlier detection.
        method: Detection method ('iqr' or 'zscore').
        iqr_multiplier: IQR multiplier for IQR method (typically 1.5).
        zscore_threshold: Z-score threshold for zscore method (typically 3.0).
    """

    enabled: bool = True
    method: str = "iqr"
    iqr_multiplier: float = 1.5
    zscore_threshold: float = 3.0


@dataclass
class ParallelConfig:
    """Configuration for parallel processing.

    Attributes:
        enabled: Enable parallel processing.
        n_workers: Number of parallel workers (None uses all CPUs).
    """

    enabled: bool = True
    n_workers: Optional[int] = None


@dataclass
class FigureConfig:
    """Configuration for figure generation.

    Attributes:
        dpi: Resolution in dots per inch.
        format: Output format ('png', 'pdf', 'svg').
        width: Figure width in inches.
        height: Figure height in inches.
    """

    dpi: int = 150
    format: str = "png"
    width: float = 10.0
    height: float = 6.0


@dataclass
class PlotConfig:
    """Configuration for plot types to generate.

    Attributes:
        studies_per_patient_histogram: Generate studies per patient histogram.
        missing_sequences_heatmap: Generate missing sequences heatmap.
        spacing_violin_plots: Generate spacing distribution violin plots.
        intensity_boxplots: Generate intensity distribution box plots.
        dimension_consistency_scatter: Generate dimension consistency scatter plots.
        snr_distribution: Generate SNR distribution plots.
    """

    studies_per_patient_histogram: bool = True
    missing_sequences_heatmap: bool = True
    spacing_violin_plots: bool = True
    intensity_boxplots: bool = True
    dimension_consistency_scatter: bool = True
    snr_distribution: bool = True


@dataclass
class HtmlReportConfig:
    """Configuration for HTML report generation.

    Attributes:
        enabled: Enable HTML report generation.
        title: Report title.
        include_summary_tables: Include summary statistics tables.
        include_all_plots: Include all generated plots in the report.
    """

    enabled: bool = True
    title: str = "MenGrowth Dataset Quality Analysis"
    include_summary_tables: bool = True
    include_all_plots: bool = True


@dataclass
class VisualizationConfig:
    """Configuration for visualization settings.

    Attributes:
        enabled: Enable visualization generation.
        plots: Plot types to generate.
        figure: Figure display settings.
        palette: Seaborn color palette name.
        html_report: HTML report generation settings.
    """

    enabled: bool = True
    plots: PlotConfig = field(default_factory=PlotConfig)
    figure: FigureConfig = field(default_factory=FigureConfig)
    palette: str = "Set2"
    html_report: HtmlReportConfig = field(default_factory=HtmlReportConfig)


@dataclass
class OutputConfig:
    """Configuration for output file generation.

    Attributes:
        save_per_study_csv: Save detailed per-study metrics to CSV.
        save_per_patient_csv: Save per-patient summary to CSV.
        save_intensity_distributions_npz: Save intensity distributions to NPZ.
        save_summary_json: Save overall summary to JSON.
        save_metadata: Save analysis metadata (timestamp, config).
    """

    save_per_study_csv: bool = True
    save_per_patient_csv: bool = True
    save_intensity_distributions_npz: bool = False
    save_summary_json: bool = True
    save_metadata: bool = True


@dataclass
class QualityAnalysisConfig:
    """Configuration for dataset quality analysis.

    Attributes:
        input_dir: Path to input dataset directory.
        output_dir: Path to output directory for analysis results.
        file_format: File format ('auto', 'nrrd', 'nifti').
        expected_sequences: List of expected sequence names.
        metrics: Metrics computation settings.
        outlier_detection: Outlier detection settings.
        intensity_percentiles: Percentiles to compute for intensity distributions.
        parallel: Parallel processing settings.
        visualization: Visualization settings.
        output: Output file settings.
        verbose: Enable debug-level logging.
        dry_run: Scan dataset without loading images.
    """

    input_dir: Path
    output_dir: Path
    file_format: str = "auto"
    expected_sequences: List[str] = field(default_factory=lambda: ["t1c", "t1n", "t2w"])
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    outlier_detection: OutlierDetectionConfig = field(
        default_factory=OutlierDetectionConfig
    )
    intensity_percentiles: List[int] = field(
        default_factory=lambda: [1, 5, 25, 50, 75, 95, 99]
    )
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    verbose: bool = False
    dry_run: bool = False


def load_quality_analysis_config(config_path: Path) -> QualityAnalysisConfig:
    """Load and validate quality analysis configuration from YAML file.

    Args:
        config_path: Path to quality analysis YAML configuration file.

    Returns:
        Validated QualityAnalysisConfig object with typed attributes.

    Raises:
        FileNotFoundError: If config_path does not exist.
        yaml.YAMLError: If YAML is malformed.
        KeyError: If required configuration keys are missing.
        TypeError: If configuration values have incorrect types.

    Examples:
        >>> config = load_quality_analysis_config(Path('configs/quality_analysis.yaml'))
        >>> print(config.input_dir)
        PosixPath('/path/to/dataset')
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)

    if not yaml_data:
        raise ValueError(f"Empty configuration file: {config_path}")

    # Extract quality_analysis section
    if "quality_analysis" in yaml_data:
        qa_dict = yaml_data["quality_analysis"]
    elif "preprocessing" in yaml_data and "quality_analysis" in yaml_data["preprocessing"]:
        qa_dict = yaml_data["preprocessing"]["quality_analysis"]
    else:
        raise KeyError(
            "Missing required 'quality_analysis' section in configuration"
        )



    # Validate required top-level fields
    required_fields = ["input_dir", "output_dir"]
    for field_name in required_fields:
        if field_name not in qa_dict:
            raise KeyError(
                f"Missing required field '{field_name}' in quality_analysis configuration"
            )

    # Parse nested configurations with defaults
    metrics_dict = qa_dict.get("metrics", {})
    metrics_config = MetricsConfig(
        patient_statistics=metrics_dict.get("patient_statistics", True),
        missing_sequences=metrics_dict.get("missing_sequences", True),
        voxel_spacing=metrics_dict.get("voxel_spacing", True),
        intensity_statistics=metrics_dict.get("intensity_statistics", True),
        image_dimensions=metrics_dict.get("image_dimensions", True),
        acquisition_consistency=metrics_dict.get("acquisition_consistency", True),
        snr_estimation=metrics_dict.get("snr_estimation", True),
    )

    outlier_dict = qa_dict.get("outlier_detection", {})
    outlier_config = OutlierDetectionConfig(
        enabled=outlier_dict.get("enabled", True),
        method=outlier_dict.get("method", "iqr"),
        iqr_multiplier=outlier_dict.get("iqr_multiplier", 1.5),
        zscore_threshold=outlier_dict.get("zscore_threshold", 3.0),
    )

    parallel_dict = qa_dict.get("parallel", {})
    n_workers = parallel_dict.get("n_workers")
    if n_workers == -1:
        n_workers = None
    parallel_config = ParallelConfig(
        enabled=parallel_dict.get("enabled", True), n_workers=n_workers
    )

    # Parse visualization configuration
    viz_dict = qa_dict.get("visualization", {})
    plots_dict = viz_dict.get("plots", {})
    plot_config = PlotConfig(
        studies_per_patient_histogram=plots_dict.get(
            "studies_per_patient_histogram", True
        ),
        missing_sequences_heatmap=plots_dict.get("missing_sequences_heatmap", True),
        spacing_violin_plots=plots_dict.get("spacing_violin_plots", True),
        intensity_boxplots=plots_dict.get("intensity_boxplots", True),
        dimension_consistency_scatter=plots_dict.get(
            "dimension_consistency_scatter", True
        ),
        snr_distribution=plots_dict.get("snr_distribution", True),
    )

    figure_dict = viz_dict.get("figure", {})
    figure_config = FigureConfig(
        dpi=figure_dict.get("dpi", 150),
        format=figure_dict.get("format", "png"),
        width=figure_dict.get("width", 10.0),
        height=figure_dict.get("height", 6.0),
    )

    html_dict = viz_dict.get("html_report", {})
    html_config = HtmlReportConfig(
        enabled=html_dict.get("enabled", True),
        title=html_dict.get("title", "MenGrowth Dataset Quality Analysis"),
        include_summary_tables=html_dict.get("include_summary_tables", True),
        include_all_plots=html_dict.get("include_all_plots", True),
    )

    visualization_config = VisualizationConfig(
        enabled=viz_dict.get("enabled", True),
        plots=plot_config,
        figure=figure_config,
        palette=viz_dict.get("palette", "Set2"),
        html_report=html_config,
    )

    # Parse output configuration
    output_dict = qa_dict.get("output", {})
    output_config = OutputConfig(
        save_per_study_csv=output_dict.get("save_per_study_csv", True),
        save_per_patient_csv=output_dict.get("save_per_patient_csv", True),
        save_intensity_distributions_npz=output_dict.get(
            "save_intensity_distributions_npz", False
        ),
        save_summary_json=output_dict.get("save_summary_json", True),
        save_metadata=output_dict.get("save_metadata", True),
    )

    # Create main configuration
    return QualityAnalysisConfig(
        input_dir=Path(qa_dict["input_dir"]),
        output_dir=Path(qa_dict["output_dir"]),
        file_format=qa_dict.get("file_format", "auto"),
        expected_sequences=qa_dict.get("expected_sequences", ["t1c", "t1n", "t2w"]),
        metrics=metrics_config,
        outlier_detection=outlier_config,
        intensity_percentiles=qa_dict.get(
            "intensity_percentiles", [1, 5, 25, 50, 75, 95, 99]
        ),
        parallel=parallel_config,
        visualization=visualization_config,
        output=output_config,
        verbose=qa_dict.get("verbose", False),
        dry_run=qa_dict.get("dry_run", False),
    )
