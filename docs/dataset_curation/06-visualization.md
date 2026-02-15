# Phase 6: Visualization and Reporting

## Theory

Visualization generates publication-ready plots and an interactive HTML report from the quality analysis results. It provides three categories of visualizations:

### QC Plots (6 standard)

1. **Studies per patient:** Histogram of study counts with mean/median lines
2. **Missing sequences:** Grouped bar chart showing present vs missing per modality
3. **Spacing distributions:** 3-panel violin plots (x, y, z voxel spacing per sequence)
4. **Intensity distributions:** 2×2 box plots (mean, std, min, max per sequence)
5. **Dimension consistency:** 3-panel scatter plots (pairwise dimension comparisons)
6. **SNR distribution:** Box plots per sequence type

### Clinical Metadata Plots (5, if metadata available)

1. **Age distribution:** Histogram with mean/median annotations
2. **Sex distribution:** Pie chart
3. **Growth category distribution:** Bar chart of growth status categories
4. **Tumor volume progression:** Line plots of volume over timepoints per patient
5. **Inclusion summary:** Dual bar chart (included vs excluded + exclusion reasons)

### Quality Filtering Plots (3, if quality_metrics.json available)

1. **Quality filter summary:** Stacked bar chart (passed/failed per check)
2. **SNR by modality:** Violin plot with threshold lines from quality_metrics.json
3. **Quality metrics heatmap:** Patient × check matrix (0=pass, 0.5=warn, 1=block)

### HTML Report

Combines all plots with summary tables (patient counts, sequence availability, demographics) into a standalone HTML file suitable for thesis appendices or team review.

## Motivation

Automated visualization eliminates manual plot creation, ensures consistency across pipeline runs, and provides a single artifact (the HTML report) that communicates dataset quality to collaborators and reviewers.

## Code Map

- **Main class:** `mengrowth/preprocessing/quality_analysis/visualize.py` → `QualityVisualizer`
  - `run_visualization()` — Full pipeline: load → plot → report
  - `generate_all_plots()` — QC plots (6)
  - `generate_clinical_plots()` — Clinical metadata plots (5)
  - `generate_quality_filtering_plots()` — Quality filtering plots (3)
  - `generate_html_report()` — Standalone HTML report
  - Individual plot methods: `plot_studies_per_patient()`, `plot_missing_sequences()`, etc.
- **Config classes:** `mengrowth/preprocessing/config.py` → `VisualizationConfig`, `PlotConfig`, `FigureConfig`, `HtmlReportConfig`
- **YAML key:** `visualization` in `configs/templates/quality_analysis.yaml`

## Config Reference

```yaml
visualization:
  enabled: true
  palette: "Set2"
  figures:
    dpi: 150                      # 300 for publication
    format: "png"                 # "png" | "pdf"
    width: 10.0
    height: 6.0
  plots:
    studies_per_patient: true
    missing_sequences: true
    spacing_distributions: true
    intensity_distributions: true
    dimension_consistency: true
    snr_distribution: true
  html_report:
    enabled: true
    title: "MenGrowth Quality Analysis Report"
    include_summary_tables: true
    include_all_plots: true
```

## Inputs / Outputs

- **Input:** `{output_root}/quality/qc_analysis/` (CSV/JSON from Phase 5)
- **Optional inputs:**
  - `MetadataManager` instance for clinical plots
  - `quality_metrics.json` for quality filtering plots
- **Outputs (in `{output_root}/quality/qc_analysis/`):**
  - `figures/*.png` (or `.pdf`) — Individual plot files
  - `quality_analysis_report.html` — Standalone HTML report with embedded plot references

## Common Tasks

| Task | How |
|------|-----|
| Disable a specific plot | Set the plot toggle to `false` in `plots` config |
| Change output format to PDF | Set `figures.format: "pdf"` |
| Increase DPI for publication | Set `figures.dpi: 300` |
| Add a new plot | Add method to `QualityVisualizer`; add toggle in `PlotConfig`; call from `generate_all_plots()` |
| Change color palette | Set `palette` to any seaborn palette name |
| Generate report without clinical data | Run without `--metadata-xlsx` — clinical plots will be skipped |
