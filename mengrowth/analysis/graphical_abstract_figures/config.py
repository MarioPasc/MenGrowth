"""Configuration dataclasses for graphical abstract figure generation.

YAML-backed, pure dataclass config tree. All fields have defaults,
all objects are picklable (no lambdas, no file handles).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class SliceConfig:
    """Which anatomical views to render and where to slice.

    Attributes:
        views: Anatomical planes to render (e.g., ["axial"]).
        axial_frac: Fractional position along the axial axis (0-1). None = center.
        sagittal_frac: Fractional position along the sagittal axis (0-1). None = center.
        coronal_frac: Fractional position along the coronal axis (0-1). None = center.
    """

    views: List[str] = field(default_factory=lambda: ["axial"])
    axial_frac: Optional[float] = None
    sagittal_frac: Optional[float] = None
    coronal_frac: Optional[float] = None


@dataclass
class StepFigureConfig:
    """Per-step rendering options for specialized visualizations.

    Attributes:
        bias_field_cmap: Colormap for bias field overlay (diverging, centered at 1.0).
        bias_field_alpha: Alpha for bias field overlay.
        registration_overlay_cmap: Colormap for registration blend overlay.
        registration_alpha: Alpha for registration blend overlay.
        mask_contour_color: Color for skull stripping mask contour.
        mask_contour_linewidth: Line width for mask contour.
    """

    bias_field_cmap: str = "RdBu_r"
    bias_field_alpha: float = 0.5
    registration_overlay_cmap: str = "hot"
    registration_alpha: float = 0.5
    mask_contour_color: str = "#00FF00"
    mask_contour_linewidth: float = 1.5
    # Segmentation overlay
    segmentation_colors: Dict[int, str] = field(
        default_factory=lambda: {1: "#FF0000", 2: "#FFFF00", 3: "#00FF00"}
    )
    segmentation_linewidth: float = 1.5
    segmentation_alpha: float = 0.3


@dataclass
class OutputConfig:
    """Output file settings.

    Attributes:
        output_dir: Directory for generated figures.
        format: Image format (png, pdf, svg).
        dpi: Resolution in dots per inch.
        generate_combined: Whether to generate the combined grid figure.
        combined_filename: Base filename for the combined grid figure.
    """

    output_dir: str = ""
    format: str = "png"
    dpi: int = 300
    generate_combined: bool = True
    combined_filename: str = "pipeline_overview"


@dataclass
class ThreeDConfig:
    """Placeholder config for future 3D rendering.

    Attributes:
        enabled: Whether 3D rendering is enabled (not yet implemented).
    """

    enabled: bool = False


@dataclass
class GraphicalAbstractConfig:
    """Top-level configuration for graphical abstract figure generation.

    Attributes:
        archive_root: Root directory containing detailed_patient HDF5 archives.
        artifacts_root: Root directory containing preprocessing artifacts (NIfTI).
        atlas_path: Path to the atlas T1 volume (e.g., SRI24).
        patient_id: Patient to render (e.g., "MenGrowth-0009").
        study_id: Study to render. Empty string = first study found.
        modalities: Modalities to render (e.g., ["t1c"]).
        steps: Steps to render. Empty list = all available in archive.
        slice: Slice extraction configuration.
        intensity_percentile_low: Low percentile for intensity windowing.
        intensity_percentile_high: High percentile for intensity windowing.
        step_options: Per-step rendering options.
        output: Output file settings.
        three_d: 3D rendering config (placeholder).
    """

    archive_root: str = ""
    artifacts_root: str = ""
    preprocessed_root: str = ""
    atlas_path: str = ""
    patient_id: str = ""
    study_ids: List[str] = field(default_factory=list)
    modalities: List[str] = field(default_factory=lambda: ["t1c"])
    steps: List[str] = field(default_factory=list)
    slice: SliceConfig = field(default_factory=SliceConfig)
    intensity_percentile_low: float = 1.0
    intensity_percentile_high: float = 99.0
    show_segmentation: bool = False
    step_options: StepFigureConfig = field(default_factory=StepFigureConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    three_d: ThreeDConfig = field(default_factory=ThreeDConfig)


def _dict_to_dataclass(cls: type, data: Dict) -> object:
    """Recursively convert a dict to a dataclass, ignoring unknown keys.

    Args:
        cls: Target dataclass type.
        data: Dictionary of values.

    Returns:
        Instance of cls populated from data.
    """
    if not isinstance(data, dict):
        return data

    import dataclasses

    field_names = {f.name for f in dataclasses.fields(cls)}
    filtered = {k: v for k, v in data.items() if k in field_names}

    # Recurse into nested dataclass fields
    for f in dataclasses.fields(cls):
        if f.name in filtered and dataclasses.is_dataclass(
            f.type if isinstance(f.type, type) else None
        ):
            filtered[f.name] = _dict_to_dataclass(f.type, filtered[f.name])

    return cls(**filtered)


def load_graphical_abstract_config(yaml_path: Path) -> GraphicalAbstractConfig:
    """Load YAML config and convert nested dicts to dataclasses.

    Args:
        yaml_path: Path to the YAML configuration file.

    Returns:
        Fully populated GraphicalAbstractConfig.
    """
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)

    # Support both top-level and nested under "graphical_abstract" key
    data = raw.get("graphical_abstract", raw)

    # Manual nested conversion for known dataclass fields
    if "slice" in data and isinstance(data["slice"], dict):
        data["slice"] = SliceConfig(**data["slice"])
    if "step_options" in data and isinstance(data["step_options"], dict):
        so = data["step_options"]
        # YAML parses int keys natively; ensure they stay as int
        if "segmentation_colors" in so and isinstance(so["segmentation_colors"], dict):
            so["segmentation_colors"] = {
                int(k): v for k, v in so["segmentation_colors"].items()
            }
        data["step_options"] = StepFigureConfig(**so)
    if "output" in data and isinstance(data["output"], dict):
        data["output"] = OutputConfig(**data["output"])
    if "three_d" in data and isinstance(data["three_d"], dict):
        data["three_d"] = ThreeDConfig(**data["three_d"])

    # Filter to known fields only
    import dataclasses

    field_names = {f.name for f in dataclasses.fields(GraphicalAbstractConfig)}
    filtered = {k: v for k, v in data.items() if k in field_names}

    config = GraphicalAbstractConfig(**filtered)
    logger.info("Loaded graphical abstract config from %s", yaml_path)
    return config
