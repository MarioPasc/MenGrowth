"""Visualization module for quality analysis results.

This module provides functions for generating plots and HTML reports from
computed quality metrics.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from mengrowth.preprocessing.config import QualityAnalysisConfig

logger = logging.getLogger(__name__)


class QualityVisualizer:
    """Visualizer for dataset quality analysis results.

    This class generates plots and HTML reports from saved analysis results.

    Attributes:
        config: Quality analysis configuration.
        output_dir: Directory for saving visualizations.
        figure_dir: Subdirectory for individual plot files.
    """

    def __init__(self, config: QualityAnalysisConfig, results_dir: Path):
        """Initialize quality visualizer.

        Args:
            config: Validated QualityAnalysisConfig object.
            results_dir: Directory containing analysis results (CSV, JSON files).
        """
        self.config = config
        self.results_dir = results_dir
        self.output_dir = config.output_dir
        self.figure_dir = self.output_dir / "figures"
        self.logger = logging.getLogger(__name__)

        # Create figure directory
        self.figure_dir.mkdir(parents=True, exist_ok=True)

        # Set seaborn style
        sns.set_style("whitegrid")
        sns.set_palette(self.config.visualization.palette)

    def load_results(self) -> Dict[str, any]:
        """Load analysis results from saved files.

        Returns:
            Dictionary containing loaded results.

        Raises:
            FileNotFoundError: If required result files are missing.
        """
        results = {}

        # Load per-study CSV
        per_study_csv = self.results_dir / "per_study_metrics.csv"
        if per_study_csv.exists():
            results["per_study"] = pd.read_csv(per_study_csv)
            self.logger.info(f"Loaded per-study metrics: {len(results['per_study'])} rows")
        else:
            self.logger.warning(f"Per-study CSV not found: {per_study_csv}")
            results["per_study"] = None

        # Load per-patient CSV
        per_patient_csv = self.results_dir / "per_patient_summary.csv"
        if per_patient_csv.exists():
            results["per_patient"] = pd.read_csv(per_patient_csv)
            self.logger.info(f"Loaded per-patient summary: {len(results['per_patient'])} rows")
        else:
            self.logger.warning(f"Per-patient CSV not found: {per_patient_csv}")
            results["per_patient"] = None

        # Load summary JSON
        summary_json = self.results_dir / "summary_statistics.json"
        if summary_json.exists():
            with open(summary_json, "r", encoding="utf-8") as f:
                results["summary"] = json.load(f)
            self.logger.info("Loaded summary statistics")
        else:
            self.logger.warning(f"Summary JSON not found: {summary_json}")
            results["summary"] = {}

        # Load per-sequence JSON
        per_seq_json = self.results_dir / "per_sequence_statistics.json"
        if per_seq_json.exists():
            with open(per_seq_json, "r", encoding="utf-8") as f:
                results["per_sequence"] = json.load(f)
            self.logger.info("Loaded per-sequence statistics")
        else:
            self.logger.warning(f"Per-sequence JSON not found: {per_seq_json}")
            results["per_sequence"] = {}

        return results

    def generate_all_plots(self, results: Dict) -> Dict[str, Path]:
        """Generate all enabled plots.

        Args:
            results: Dictionary containing loaded analysis results.

        Returns:
            Dictionary mapping plot type to saved file path.
        """
        self.logger.info("Generating visualizations...")
        saved_plots = {}

        plot_config = self.config.visualization.plots

        # Studies per patient histogram
        if plot_config.studies_per_patient_histogram and results.get("per_patient") is not None:
            plot_path = self.plot_studies_per_patient(results["per_patient"])
            if plot_path:
                saved_plots["studies_per_patient"] = plot_path

        # Missing sequences heatmap
        if plot_config.missing_sequences_heatmap and results.get("summary"):
            plot_path = self.plot_missing_sequences(results["summary"])
            if plot_path:
                saved_plots["missing_sequences"] = plot_path

        # Spacing violin plots
        if plot_config.spacing_violin_plots and results.get("per_study") is not None:
            plot_path = self.plot_spacing_distributions(results["per_study"])
            if plot_path:
                saved_plots["spacing_violin"] = plot_path

        # Intensity box plots
        if plot_config.intensity_boxplots and results.get("per_study") is not None:
            plot_path = self.plot_intensity_distributions(results["per_study"])
            if plot_path:
                saved_plots["intensity_boxplots"] = plot_path

        # Dimension consistency scatter
        if plot_config.dimension_consistency_scatter and results.get("per_study") is not None:
            plot_path = self.plot_dimension_consistency(results["per_study"])
            if plot_path:
                saved_plots["dimension_scatter"] = plot_path

        # SNR distribution
        if plot_config.snr_distribution and results.get("per_study") is not None:
            plot_path = self.plot_snr_distribution(results["per_study"])
            if plot_path:
                saved_plots["snr_distribution"] = plot_path

        self.logger.info(f"Generated {len(saved_plots)} plots")
        return saved_plots

    def plot_studies_per_patient(self, per_patient_df: pd.DataFrame) -> Optional[Path]:
        """Plot histogram of studies per patient.

        Args:
            per_patient_df: DataFrame with per-patient summary.

        Returns:
            Path to saved plot file.
        """
        fig, ax = plt.subplots(
            figsize=(self.config.visualization.figure.width,
                     self.config.visualization.figure.height)
        )

        # Histogram
        studies_counts = per_patient_df["num_studies"]
        ax.hist(studies_counts, bins=range(1, studies_counts.max() + 2), alpha=0.7, edgecolor="black")

        # Add statistics overlay
        mean_studies = studies_counts.mean()
        median_studies = studies_counts.median()
        ax.axvline(mean_studies, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_studies:.2f}")
        ax.axvline(median_studies, color="orange", linestyle="--", linewidth=2, label=f"Median: {median_studies:.0f}")

        ax.set_xlabel("Number of Studies per Patient", fontsize=12)
        ax.set_ylabel("Number of Patients", fontsize=12)
        ax.set_title("Distribution of Studies per Patient", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plot_path = self.figure_dir / f"studies_per_patient.{self.config.visualization.figure.format}"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=self.config.visualization.figure.dpi)
        plt.close()

        self.logger.info(f"Saved studies per patient plot: {plot_path}")
        return plot_path

    def plot_missing_sequences(self, summary: Dict) -> Optional[Path]:
        """Plot heatmap of missing sequences.

        Args:
            summary: Summary statistics dictionary.

        Returns:
            Path to saved plot file.
        """
        if "missing_sequences" not in summary:
            self.logger.warning("Missing sequences data not found in summary")
            return None

        missing_stats = summary["missing_sequences"]

        # Extract data for each sequence
        sequences = [seq for seq in missing_stats.keys() if seq != "overall"]
        present_counts = [missing_stats[seq]["present_count"] for seq in sequences]
        missing_counts = [missing_stats[seq]["missing_count"] for seq in sequences]

        # Create grouped bar chart
        fig, ax = plt.subplots(
            figsize=(self.config.visualization.figure.width,
                     self.config.visualization.figure.height)
        )

        x = np.arange(len(sequences))
        width = 0.35

        bars1 = ax.bar(x - width/2, present_counts, width, label="Present", color="green", alpha=0.7)
        bars2 = ax.bar(x + width/2, missing_counts, width, label="Missing", color="red", alpha=0.7)

        ax.set_xlabel("Sequence Type", fontsize=12)
        ax.set_ylabel("Number of Studies", fontsize=12)
        ax.set_title("Sequence Availability Across Studies", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(sequences)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}',
                           ha='center', va='bottom', fontsize=9)

        plot_path = self.figure_dir / f"missing_sequences.{self.config.visualization.figure.format}"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=self.config.visualization.figure.dpi)
        plt.close()

        self.logger.info(f"Saved missing sequences plot: {plot_path}")
        return plot_path

    def plot_spacing_distributions(self, per_study_df: pd.DataFrame) -> Optional[Path]:
        """Plot violin plots of voxel spacing distributions per sequence.

        Args:
            per_study_df: DataFrame with per-study metrics.

        Returns:
            Path to saved plot file.
        """
        # Check if spacing columns exist
        if "spacing_x" not in per_study_df.columns:
            self.logger.warning("Spacing data not found in per-study metrics")
            return None

        fig, axes = plt.subplots(1, 3, figsize=(self.config.visualization.figure.width * 1.5,
                                                  self.config.visualization.figure.height))

        for idx, axis_name in enumerate(["x", "y", "z"]):
            col_name = f"spacing_{axis_name}"

            # Filter out NaN values
            df_filtered = per_study_df[per_study_df[col_name].notna()]

            if len(df_filtered) == 0:
                continue

            sns.violinplot(data=df_filtered, x="sequence", y=col_name, ax=axes[idx])

            axes[idx].set_xlabel("Sequence Type", fontsize=11)
            axes[idx].set_ylabel(f"Spacing (mm)", fontsize=11)
            axes[idx].set_title(f"{axis_name.upper()}-axis Spacing Distribution", fontsize=12, fontweight="bold")
            axes[idx].grid(True, alpha=0.3, axis='y')
            axes[idx].tick_params(axis='x', rotation=45)

        plot_path = self.figure_dir / f"spacing_distributions.{self.config.visualization.figure.format}"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=self.config.visualization.figure.dpi)
        plt.close()

        self.logger.info(f"Saved spacing distributions plot: {plot_path}")
        return plot_path

    def plot_intensity_distributions(self, per_study_df: pd.DataFrame) -> Optional[Path]:
        """Plot box plots of intensity value distributions per sequence.

        Args:
            per_study_df: DataFrame with per-study metrics.

        Returns:
            Path to saved plot file.
        """
        if "mean" not in per_study_df.columns:
            self.logger.warning("Intensity data not found in per-study metrics")
            return None

        fig, axes = plt.subplots(2, 2, figsize=(self.config.visualization.figure.width * 1.2,
                                                  self.config.visualization.figure.height * 1.5))
        axes = axes.flatten()

        metrics = ["mean", "std", "min", "max"]

        for idx, metric in enumerate(metrics):
            if metric not in per_study_df.columns:
                continue

            df_filtered = per_study_df[per_study_df[metric].notna()]

            if len(df_filtered) == 0:
                continue

            sns.boxplot(data=df_filtered, x="sequence", y=metric, ax=axes[idx])

            axes[idx].set_xlabel("Sequence Type", fontsize=11)
            axes[idx].set_ylabel(f"Intensity {metric.capitalize()}", fontsize=11)
            axes[idx].set_title(f"Intensity {metric.capitalize()} Distribution", fontsize=12, fontweight="bold")
            axes[idx].grid(True, alpha=0.3, axis='y')
            axes[idx].tick_params(axis='x', rotation=45)

        plot_path = self.figure_dir / f"intensity_distributions.{self.config.visualization.figure.format}"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=self.config.visualization.figure.dpi)
        plt.close()

        self.logger.info(f"Saved intensity distributions plot: {plot_path}")
        return plot_path

    def plot_dimension_consistency(self, per_study_df: pd.DataFrame) -> Optional[Path]:
        """Plot scatter plots showing dimension consistency.

        Args:
            per_study_df: DataFrame with per-study metrics.

        Returns:
            Path to saved plot file.
        """
        if "width" not in per_study_df.columns:
            self.logger.warning("Dimension data not found in per-study metrics")
            return None

        fig, axes = plt.subplots(1, 3, figsize=(self.config.visualization.figure.width * 1.5,
                                                  self.config.visualization.figure.height))

        dimension_pairs = [("width", "height"), ("width", "depth"), ("height", "depth")]

        for idx, (dim1, dim2) in enumerate(dimension_pairs):
            df_filtered = per_study_df[
                per_study_df[dim1].notna() & per_study_df[dim2].notna()
            ]

            if len(df_filtered) == 0:
                continue

            for sequence in df_filtered["sequence"].unique():
                seq_data = df_filtered[df_filtered["sequence"] == sequence]
                axes[idx].scatter(seq_data[dim1], seq_data[dim2], label=sequence, alpha=0.6, s=50)

            axes[idx].set_xlabel(f"{dim1.capitalize()} (voxels)", fontsize=11)
            axes[idx].set_ylabel(f"{dim2.capitalize()} (voxels)", fontsize=11)
            axes[idx].set_title(f"{dim1.capitalize()} vs {dim2.capitalize()}", fontsize=12, fontweight="bold")
            axes[idx].legend(fontsize=9)
            axes[idx].grid(True, alpha=0.3)

        plot_path = self.figure_dir / f"dimension_consistency.{self.config.visualization.figure.format}"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=self.config.visualization.figure.dpi)
        plt.close()

        self.logger.info(f"Saved dimension consistency plot: {plot_path}")
        return plot_path

    def plot_snr_distribution(self, per_study_df: pd.DataFrame) -> Optional[Path]:
        """Plot SNR distribution per sequence.

        Args:
            per_study_df: DataFrame with per-study metrics.

        Returns:
            Path to saved plot file.
        """
        if "snr" not in per_study_df.columns:
            self.logger.warning("SNR data not found in per-study metrics")
            return None

        fig, ax = plt.subplots(
            figsize=(self.config.visualization.figure.width,
                     self.config.visualization.figure.height)
        )

        # Filter out infinite and NaN values
        df_filtered = per_study_df[
            per_study_df["snr"].notna() & np.isfinite(per_study_df["snr"])
        ]

        if len(df_filtered) == 0:
            self.logger.warning("No valid SNR data to plot")
            plt.close()
            return None

        sns.boxplot(data=df_filtered, x="sequence", y="snr", ax=ax)

        ax.set_xlabel("Sequence Type", fontsize=12)
        ax.set_ylabel("SNR", fontsize=12)
        ax.set_title("Signal-to-Noise Ratio Distribution", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='x', rotation=45)

        plot_path = self.figure_dir / f"snr_distribution.{self.config.visualization.figure.format}"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=self.config.visualization.figure.dpi)
        plt.close()

        self.logger.info(f"Saved SNR distribution plot: {plot_path}")
        return plot_path

    def generate_html_report(
        self, results: Dict, plot_paths: Dict[str, Path]
    ) -> Path:
        """Generate comprehensive HTML report with plots and tables.

        Args:
            results: Dictionary containing loaded analysis results.
            plot_paths: Dictionary mapping plot type to file path.

        Returns:
            Path to saved HTML report.
        """
        self.logger.info("Generating HTML report...")

        html_config = self.config.visualization.html_report
        html_path = self.output_dir / "quality_analysis_report.html"

        with open(html_path, "w", encoding="utf-8") as f:
            # HTML header
            f.write(f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html_config.title}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 8px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .plot {{
            margin: 20px 0;
            text-align: center;
        }}
        .plot img {{
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 5px;
        }}
        .metric {{
            display: inline-block;
            margin: 10px 20px;
            padding: 15px;
            background-color: #ecf0f1;
            border-radius: 5px;
        }}
        .metric-label {{
            font-weight: bold;
            color: #7f8c8d;
        }}
        .metric-value {{
            font-size: 24px;
            color: #2c3e50;
        }}
        .timestamp {{
            color: #7f8c8d;
            font-size: 14px;
            margin-top: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{html_config.title}</h1>
        <p class="timestamp">Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
""")

            # Summary statistics
            if html_config.include_summary_tables and results.get("summary"):
                f.write("<h2>Summary Statistics</h2>\n")

                # Patient statistics
                if "patient_statistics" in results["summary"]:
                    patient_stats = results["summary"]["patient_statistics"]
                    f.write("<h3>Patient and Study Overview</h3>\n")
                    f.write('<div class="metrics">\n')
                    f.write(f'<div class="metric"><span class="metric-label">Total Patients:</span><br/><span class="metric-value">{patient_stats["total_patients"]}</span></div>\n')
                    f.write(f'<div class="metric"><span class="metric-label">Total Studies:</span><br/><span class="metric-value">{patient_stats["total_studies"]}</span></div>\n')
                    f.write(f'<div class="metric"><span class="metric-label">Mean Studies/Patient:</span><br/><span class="metric-value">{patient_stats["mean_studies"]:.2f}</span></div>\n')
                    f.write(f'<div class="metric"><span class="metric-label">Std Studies/Patient:</span><br/><span class="metric-value">{patient_stats["std_studies"]:.2f}</span></div>\n')
                    f.write('</div>\n')

                # Missing sequences
                if "missing_sequences" in results["summary"]:
                    missing_stats = results["summary"]["missing_sequences"]
                    f.write("<h3>Missing Sequence Analysis</h3>\n")
                    f.write("<table>\n")
                    f.write("<tr><th>Sequence</th><th>Present Count</th><th>Missing Count</th><th>Missing Fraction</th></tr>\n")
                    for seq, stats in missing_stats.items():
                        if seq != "overall":
                            f.write(f'<tr><td>{seq}</td><td>{stats["present_count"]}</td><td>{stats["missing_count"]}</td><td>{stats["missing_fraction"]:.2%}</td></tr>\n')
                    f.write("</table>\n")

            # Plots
            if html_config.include_all_plots and plot_paths:
                f.write("<h2>Visualizations</h2>\n")

                for plot_name, plot_path in plot_paths.items():
                    # Make path relative to HTML file
                    rel_path = plot_path.relative_to(self.output_dir)
                    f.write(f'<div class="plot">\n')
                    f.write(f'<h3>{plot_name.replace("_", " ").title()}</h3>\n')
                    f.write(f'<img src="{rel_path}" alt="{plot_name}">\n')
                    f.write('</div>\n')

            # Footer
            f.write("""
    </div>
</body>
</html>
""")

        self.logger.info(f"Saved HTML report: {html_path}")
        return html_path

    def run_visualization(self) -> Dict[str, Path]:
        """Run complete visualization pipeline.

        Returns:
            Dictionary of saved file paths.

        Examples:
            >>> visualizer = QualityVisualizer(config, results_dir)
            >>> output_paths = visualizer.run_visualization()
            >>> print(output_paths['html_report'])
            PosixPath('.../quality_analysis_report.html')
        """
        # Load results
        results = self.load_results()

        # Generate plots
        plot_paths = self.generate_all_plots(results)

        # Generate HTML report
        output_paths = {"plots": plot_paths}

        if self.config.visualization.html_report.enabled:
            html_path = self.generate_html_report(results, plot_paths)
            output_paths["html_report"] = html_path

        return output_paths
