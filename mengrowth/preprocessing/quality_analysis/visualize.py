"""Visualization module for quality analysis results.

This module provides functions for generating plots and HTML reports from
computed quality metrics, including clinical metadata visualizations.
"""

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from mengrowth.preprocessing.config import QualityAnalysisConfig

if TYPE_CHECKING:
    from mengrowth.preprocessing.utils.metadata import MetadataManager

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

    # =========================================================================
    # CLINICAL METADATA PLOTS
    # =========================================================================

    def plot_age_distribution(
        self, metadata_manager: "MetadataManager"
    ) -> Optional[Path]:
        """Plot histogram of patient ages.

        Args:
            metadata_manager: MetadataManager with loaded patient data.

        Returns:
            Path to saved plot file.
        """
        summary = metadata_manager.get_clinical_summary()
        age_stats = summary.get("age_stats", {})

        if not age_stats:
            self.logger.warning("No age data available for plotting")
            return None

        # Get all ages from included patients
        ages = []
        for patient in metadata_manager.get_all_patients().values():
            if patient.included and patient.age is not None:
                ages.append(patient.age)

        if not ages:
            self.logger.warning("No valid age data to plot")
            return None

        fig, ax = plt.subplots(
            figsize=(self.config.visualization.figure.width,
                     self.config.visualization.figure.height)
        )

        # Histogram
        ax.hist(ages, bins=15, alpha=0.7, edgecolor="black", color="#3498db")

        # Add statistics overlay
        mean_age = age_stats.get("mean", np.mean(ages))
        median_age = age_stats.get("median", np.median(ages))
        ax.axvline(mean_age, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_age:.1f}")
        ax.axvline(median_age, color="orange", linestyle="--", linewidth=2, label=f"Median: {median_age:.1f}")

        ax.set_xlabel("Age (years)", fontsize=12)
        ax.set_ylabel("Number of Patients", fontsize=12)
        ax.set_title("Patient Age Distribution", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plot_path = self.figure_dir / f"age_distribution.{self.config.visualization.figure.format}"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=self.config.visualization.figure.dpi)
        plt.close()

        self.logger.info(f"Saved age distribution plot: {plot_path}")
        return plot_path

    def plot_sex_distribution(
        self, metadata_manager: "MetadataManager"
    ) -> Optional[Path]:
        """Plot pie chart of sex distribution.

        Args:
            metadata_manager: MetadataManager with loaded patient data.

        Returns:
            Path to saved plot file.
        """
        summary = metadata_manager.get_clinical_summary()
        sex_dist = summary.get("sex_distribution", {})

        if not sex_dist or (sex_dist.get("male", 0) == 0 and sex_dist.get("female", 0) == 0):
            self.logger.warning("No sex distribution data available")
            return None

        fig, ax = plt.subplots(
            figsize=(self.config.visualization.figure.width,
                     self.config.visualization.figure.height)
        )

        labels = []
        sizes = []
        colors = []
        color_map = {"male": "#3498db", "female": "#e74c3c", "unknown": "#95a5a6"}

        for sex, count in sex_dist.items():
            if count > 0:
                labels.append(f"{sex.capitalize()} (n={count})")
                sizes.append(count)
                colors.append(color_map.get(sex, "#95a5a6"))

        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
               startangle=90, explode=[0.02] * len(sizes))
        ax.set_title("Patient Sex Distribution", fontsize=14, fontweight="bold")

        plot_path = self.figure_dir / f"sex_distribution.{self.config.visualization.figure.format}"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=self.config.visualization.figure.dpi)
        plt.close()

        self.logger.info(f"Saved sex distribution plot: {plot_path}")
        return plot_path

    def plot_growth_category_distribution(
        self, metadata_manager: "MetadataManager"
    ) -> Optional[Path]:
        """Plot bar chart of growth categories.

        Args:
            metadata_manager: MetadataManager with loaded patient data.

        Returns:
            Path to saved plot file.
        """
        summary = metadata_manager.get_clinical_summary()
        growth_dist = summary.get("growth_distribution", {})

        if not growth_dist:
            self.logger.warning("No growth distribution data available")
            return None

        fig, ax = plt.subplots(
            figsize=(self.config.visualization.figure.width,
                     self.config.visualization.figure.height)
        )

        categories = []
        counts = []
        colors = []
        color_map = {"growing": "#e74c3c", "stable": "#27ae60", "unknown": "#95a5a6"}

        for category, count in growth_dist.items():
            if count > 0:
                categories.append(category.capitalize())
                counts.append(count)
                colors.append(color_map.get(category, "#95a5a6"))

        bars = ax.bar(categories, counts, color=colors, alpha=0.8, edgecolor="black")

        # Add value labels on bars
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                   f'{count}', ha='center', va='bottom', fontsize=11, fontweight="bold")

        ax.set_xlabel("Growth Category", fontsize=12)
        ax.set_ylabel("Number of Patients", fontsize=12)
        ax.set_title("Tumor Growth Distribution", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, axis='y')

        plot_path = self.figure_dir / f"growth_category.{self.config.visualization.figure.format}"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=self.config.visualization.figure.dpi)
        plt.close()

        self.logger.info(f"Saved growth category plot: {plot_path}")
        return plot_path

    def plot_tumor_volume_progression(
        self, metadata_manager: "MetadataManager"
    ) -> Optional[Path]:
        """Plot tumor volume progression over time.

        Args:
            metadata_manager: MetadataManager with loaded patient data.

        Returns:
            Path to saved plot file.
        """
        progressions = metadata_manager.get_volume_progression_data()

        if not progressions:
            self.logger.warning("No volume progression data available")
            return None

        fig, ax = plt.subplots(
            figsize=(self.config.visualization.figure.width * 1.2,
                     self.config.visualization.figure.height)
        )

        # Plot each patient's progression
        for prog in progressions:
            timepoints = [v["timepoint"] for v in prog["volumes"]]
            volumes = [v["volume"] for v in prog["volumes"]]
            growth_status = prog.get("growth_status")

            color = "#e74c3c" if growth_status else "#27ae60" if growth_status is False else "#95a5a6"
            alpha = 0.7 if growth_status else 0.5

            ax.plot(timepoints, volumes, marker='o', color=color, alpha=alpha, linewidth=1.5)

        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color="#e74c3c", marker='o', label="Growing"),
            Line2D([0], [0], color="#27ae60", marker='o', label="Stable"),
            Line2D([0], [0], color="#95a5a6", marker='o', label="Unknown"),
        ]
        ax.legend(handles=legend_elements, loc='upper left')

        ax.set_xlabel("Timepoint (0=baseline, 1-5=controls)", fontsize=12)
        ax.set_ylabel("Tumor Volume (mm³)", fontsize=12)
        ax.set_title("Tumor Volume Progression Over Time", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)

        plot_path = self.figure_dir / f"tumor_volume_progression.{self.config.visualization.figure.format}"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=self.config.visualization.figure.dpi)
        plt.close()

        self.logger.info(f"Saved tumor volume progression plot: {plot_path}")
        return plot_path

    def plot_inclusion_summary(
        self, metadata_manager: "MetadataManager"
    ) -> Optional[Path]:
        """Plot bar chart showing included vs excluded patients.

        Args:
            metadata_manager: MetadataManager with loaded patient data.

        Returns:
            Path to saved plot file.
        """
        summary = metadata_manager.get_clinical_summary()

        included = summary.get("included_patients", 0)
        excluded = summary.get("excluded_patients", 0)
        exclusion_reasons = summary.get("exclusion_reasons", {})

        if included == 0 and excluded == 0:
            self.logger.warning("No inclusion data available")
            return None

        fig, axes = plt.subplots(1, 2, figsize=(self.config.visualization.figure.width * 1.3,
                                                  self.config.visualization.figure.height))

        # Left plot: Included vs Excluded
        bars = axes[0].bar(["Included", "Excluded"], [included, excluded],
                          color=["#27ae60", "#e74c3c"], alpha=0.8, edgecolor="black")

        for bar, count in zip(bars, [included, excluded]):
            axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                        f'{count}', ha='center', va='bottom', fontsize=12, fontweight="bold")

        axes[0].set_ylabel("Number of Patients", fontsize=12)
        axes[0].set_title("Patient Inclusion Status", fontsize=12, fontweight="bold")
        axes[0].grid(True, alpha=0.3, axis='y')

        # Right plot: Exclusion reasons (if any)
        if exclusion_reasons:
            reasons = list(exclusion_reasons.keys())
            counts = list(exclusion_reasons.values())

            # Truncate long reason names
            reasons_short = [r[:30] + "..." if len(r) > 30 else r for r in reasons]

            y_pos = np.arange(len(reasons))
            axes[1].barh(y_pos, counts, color="#e74c3c", alpha=0.7)
            axes[1].set_yticks(y_pos)
            axes[1].set_yticklabels(reasons_short, fontsize=9)
            axes[1].set_xlabel("Count", fontsize=12)
            axes[1].set_title("Exclusion Reasons", fontsize=12, fontweight="bold")
            axes[1].grid(True, alpha=0.3, axis='x')
        else:
            axes[1].text(0.5, 0.5, "No exclusions", ha='center', va='center', fontsize=14)
            axes[1].set_title("Exclusion Reasons", fontsize=12, fontweight="bold")

        plot_path = self.figure_dir / f"inclusion_summary.{self.config.visualization.figure.format}"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=self.config.visualization.figure.dpi)
        plt.close()

        self.logger.info(f"Saved inclusion summary plot: {plot_path}")
        return plot_path

    def generate_clinical_plots(
        self, metadata_manager: "MetadataManager"
    ) -> Dict[str, Path]:
        """Generate all clinical metadata plots.

        Args:
            metadata_manager: MetadataManager with loaded patient data.

        Returns:
            Dictionary mapping plot type to saved file path.
        """
        self.logger.info("Generating clinical metadata visualizations...")
        clinical_plots = {}

        # Age distribution
        plot_path = self.plot_age_distribution(metadata_manager)
        if plot_path:
            clinical_plots["age_distribution"] = plot_path

        # Sex distribution
        plot_path = self.plot_sex_distribution(metadata_manager)
        if plot_path:
            clinical_plots["sex_distribution"] = plot_path

        # Growth category
        plot_path = self.plot_growth_category_distribution(metadata_manager)
        if plot_path:
            clinical_plots["growth_category"] = plot_path

        # Tumor volume progression
        plot_path = self.plot_tumor_volume_progression(metadata_manager)
        if plot_path:
            clinical_plots["tumor_volume_progression"] = plot_path

        # Inclusion summary
        plot_path = self.plot_inclusion_summary(metadata_manager)
        if plot_path:
            clinical_plots["inclusion_summary"] = plot_path

        self.logger.info(f"Generated {len(clinical_plots)} clinical plots")
        return clinical_plots

    # =========================================================================
    # QUALITY FILTERING PLOTS
    # =========================================================================

    def plot_quality_filter_summary(
        self, quality_metrics: Dict,
    ) -> Optional[Path]:
        """Plot stacked bar chart of pass/warn/block counts per check type.

        Args:
            quality_metrics: Loaded quality_metrics.json data.

        Returns:
            Path to saved plot file.
        """
        checks_summary = quality_metrics.get("summary", {}).get("checks_summary", {})
        if not checks_summary:
            self.logger.warning("No checks summary data for quality filter summary plot")
            return None

        check_names = list(checks_summary.keys())
        passed = [checks_summary[c].get("passed", 0) for c in check_names]
        failed = [checks_summary[c].get("failed", 0) for c in check_names]

        fig, ax = plt.subplots(
            figsize=(self.config.visualization.figure.width * 1.2,
                     self.config.visualization.figure.height)
        )

        x = np.arange(len(check_names))
        width = 0.5

        ax.bar(x, passed, width, label="Passed", color="#27ae60", alpha=0.8)
        ax.bar(x, failed, width, bottom=passed, label="Failed", color="#e74c3c", alpha=0.8)

        ax.set_xlabel("Check Type", fontsize=12)
        ax.set_ylabel("Number of Files", fontsize=12)
        ax.set_title("Quality Filtering Results by Check Type", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(check_names, rotation=45, ha="right", fontsize=9)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        plot_path = self.figure_dir / f"quality_filter_summary.{self.config.visualization.figure.format}"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=self.config.visualization.figure.dpi)
        plt.close()

        self.logger.info(f"Saved quality filter summary plot: {plot_path}")
        return plot_path

    def plot_snr_by_modality(
        self, quality_metrics: Dict,
    ) -> Optional[Path]:
        """Plot violin/box plots of SNR values from quality filtering, per modality.

        Args:
            quality_metrics: Loaded quality_metrics.json data.

        Returns:
            Path to saved plot file.
        """
        # Extract SNR values per modality from quality metrics
        snr_data: Dict[str, List[float]] = {}
        patients = quality_metrics.get("patients", {})

        for patient_id, patient_data in patients.items():
            for study_id, study_data in patient_data.get("studies", {}).items():
                for modality, file_data in study_data.get("files", {}).items():
                    checks = file_data.get("checks", {})
                    snr_check = checks.get("snr_filtering", {})
                    details = snr_check.get("details", {})
                    snr_val = details.get("snr")
                    if snr_val is not None and np.isfinite(snr_val):
                        if modality not in snr_data:
                            snr_data[modality] = []
                        snr_data[modality].append(snr_val)

        if not snr_data:
            self.logger.warning("No SNR data available for quality filtering SNR plot")
            return None

        fig, ax = plt.subplots(
            figsize=(self.config.visualization.figure.width,
                     self.config.visualization.figure.height)
        )

        modalities = sorted(snr_data.keys())
        data_lists = [snr_data[m] for m in modalities]

        parts = ax.violinplot(data_lists, positions=range(len(modalities)), showmeans=True, showmedians=True)

        # Color the violins
        for pc in parts["bodies"]:
            pc.set_facecolor("#3498db")
            pc.set_alpha(0.7)

        # Add threshold lines (extract from first available check)
        for patient_data in patients.values():
            for study_data in patient_data.get("studies", {}).values():
                for modality, file_data in study_data.get("files", {}).items():
                    threshold = file_data.get("checks", {}).get("snr_filtering", {}).get("details", {}).get("threshold")
                    if threshold is not None:
                        ax.axhline(y=threshold, color="#e74c3c", linestyle="--", linewidth=1.5,
                                  label=f"Threshold ({threshold})", alpha=0.7)
                        break
                else:
                    continue
                break
            else:
                continue
            break

        ax.set_xticks(range(len(modalities)))
        ax.set_xticklabels(modalities)
        ax.set_xlabel("Modality", fontsize=12)
        ax.set_ylabel("SNR", fontsize=12)
        ax.set_title("SNR Distribution by Modality (Quality Filtering)", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        plot_path = self.figure_dir / f"qf_snr_by_modality.{self.config.visualization.figure.format}"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=self.config.visualization.figure.dpi)
        plt.close()

        self.logger.info(f"Saved quality filtering SNR by modality plot: {plot_path}")
        return plot_path

    def plot_quality_metrics_heatmap(
        self, quality_metrics: Dict,
    ) -> Optional[Path]:
        """Plot patient x check heatmap showing pass/warn/block status.

        Args:
            quality_metrics: Loaded quality_metrics.json data.

        Returns:
            Path to saved plot file.
        """
        patients = quality_metrics.get("patients", {})
        if not patients:
            return None

        # Collect all check names across all files
        all_check_names: set = set()
        for patient_data in patients.values():
            for study_data in patient_data.get("studies", {}).values():
                for file_data in study_data.get("files", {}).values():
                    all_check_names.update(file_data.get("checks", {}).keys())

        if not all_check_names:
            return None

        check_names = sorted(all_check_names)
        patient_ids = sorted(patients.keys())

        # Build matrix: 0 = all passed, 0.5 = has warnings, 1 = has blocks
        matrix = np.zeros((len(patient_ids), len(check_names)))

        for i, pid in enumerate(patient_ids):
            patient_data = patients[pid]
            for j, check_name in enumerate(check_names):
                has_failure = False
                for study_data in patient_data.get("studies", {}).values():
                    for file_data in study_data.get("files", {}).values():
                        check = file_data.get("checks", {}).get(check_name, {})
                        if not check.get("passed", True):
                            if check.get("action") == "block":
                                matrix[i, j] = 1.0
                                has_failure = True
                                break
                            else:
                                matrix[i, j] = max(matrix[i, j], 0.5)
                    if has_failure:
                        break

        fig, ax = plt.subplots(
            figsize=(max(self.config.visualization.figure.width, len(check_names) * 0.8),
                     max(self.config.visualization.figure.height, len(patient_ids) * 0.3))
        )

        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(["#27ae60", "#f39c12", "#e74c3c"])

        im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=1)

        ax.set_xticks(range(len(check_names)))
        ax.set_xticklabels(check_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(patient_ids)))
        ax.set_yticklabels(patient_ids, fontsize=7)
        ax.set_xlabel("Quality Check", fontsize=12)
        ax.set_ylabel("Patient", fontsize=12)
        ax.set_title("Quality Check Results per Patient", fontsize=14, fontweight="bold")

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#27ae60", label="Passed"),
            Patch(facecolor="#f39c12", label="Warning"),
            Patch(facecolor="#e74c3c", label="Blocked"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

        plot_path = self.figure_dir / f"quality_metrics_heatmap.{self.config.visualization.figure.format}"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=self.config.visualization.figure.dpi)
        plt.close()

        self.logger.info(f"Saved quality metrics heatmap: {plot_path}")
        return plot_path

    def generate_quality_filtering_plots(
        self, quality_metrics: Dict,
    ) -> Dict[str, Path]:
        """Generate all quality filtering plots.

        Args:
            quality_metrics: Loaded quality_metrics.json data.

        Returns:
            Dictionary mapping plot type to saved file path.
        """
        self.logger.info("Generating quality filtering visualizations...")
        qf_plots = {}

        plot_path = self.plot_quality_filter_summary(quality_metrics)
        if plot_path:
            qf_plots["quality_filter_summary"] = plot_path

        plot_path = self.plot_snr_by_modality(quality_metrics)
        if plot_path:
            qf_plots["qf_snr_by_modality"] = plot_path

        plot_path = self.plot_quality_metrics_heatmap(quality_metrics)
        if plot_path:
            qf_plots["quality_metrics_heatmap"] = plot_path

        self.logger.info(f"Generated {len(qf_plots)} quality filtering plots")
        return qf_plots

    def generate_html_report(
        self,
        results: Dict,
        plot_paths: Dict[str, Path],
        metadata_manager: Optional["MetadataManager"] = None,
        quality_metrics: Optional[Dict] = None,
    ) -> Path:
        """Generate comprehensive HTML report with plots and tables.

        Args:
            results: Dictionary containing loaded analysis results.
            plot_paths: Dictionary mapping plot type to file path.
            metadata_manager: Optional metadata manager for clinical summary.

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

            # Quality filtering section
            if quality_metrics:
                qf_summary = quality_metrics.get("summary", {})
                f.write("<h2>Quality Filtering Results</h2>\n")

                # Overview metrics
                f.write('<div class="metrics">\n')
                f.write(f'<div class="metric"><span class="metric-label">Total Patients:</span><br/><span class="metric-value">{qf_summary.get("total_patients", "N/A")}</span></div>\n')
                f.write(f'<div class="metric"><span class="metric-label">Patients Passed:</span><br/><span class="metric-value">{qf_summary.get("patients_passed", "N/A")}</span></div>\n')
                f.write(f'<div class="metric"><span class="metric-label">Patients Blocked:</span><br/><span class="metric-value">{qf_summary.get("patients_blocked", "N/A")}</span></div>\n')
                f.write('</div>\n')

                # Per-check summary table
                checks_summary = qf_summary.get("checks_summary", {})
                if checks_summary:
                    f.write("<h3>Per-Check Summary</h3>\n")
                    f.write("<table>\n")
                    f.write("<tr><th>Check</th><th>Total</th><th>Passed</th><th>Failed</th><th>Pass Rate</th></tr>\n")
                    for check_name, counts in sorted(checks_summary.items()):
                        total = counts.get("total", 0)
                        passed = counts.get("passed", 0)
                        failed = counts.get("failed", 0)
                        rate = f"{passed/total:.1%}" if total > 0 else "N/A"
                        f.write(f'<tr><td>{check_name}</td><td>{total}</td><td>{passed}</td><td>{failed}</td><td>{rate}</td></tr>\n')
                    f.write("</table>\n")

                # Quality filtering plots
                qf_plot_names = ["quality_filter_summary", "qf_snr_by_modality", "quality_metrics_heatmap"]
                qf_plots_in_report = {k: v for k, v in plot_paths.items() if k in qf_plot_names}
                if qf_plots_in_report:
                    f.write("<h3>Quality Filtering Visualizations</h3>\n")
                    for plot_name, plot_path in qf_plots_in_report.items():
                        rel_path = plot_path.relative_to(self.output_dir)
                        f.write(f'<div class="plot">\n')
                        f.write(f'<h4>{plot_name.replace("_", " ").title()}</h4>\n')
                        f.write(f'<img src="{rel_path}" alt="{plot_name}">\n')
                        f.write('</div>\n')

            # Clinical metadata section
            if metadata_manager:
                clinical_summary = metadata_manager.get_clinical_summary()
                f.write("<h2>Clinical Metadata</h2>\n")

                # Demographics
                f.write("<h3>Patient Demographics</h3>\n")
                f.write('<div class="metrics">\n')
                f.write(f'<div class="metric"><span class="metric-label">Total in Metadata:</span><br/><span class="metric-value">{clinical_summary["total_patients"]}</span></div>\n')
                f.write(f'<div class="metric"><span class="metric-label">Included:</span><br/><span class="metric-value">{clinical_summary["included_patients"]}</span></div>\n')
                f.write(f'<div class="metric"><span class="metric-label">Excluded:</span><br/><span class="metric-value">{clinical_summary["excluded_patients"]}</span></div>\n')
                f.write('</div>\n')

                # Age statistics
                age_stats = clinical_summary.get("age_stats", {})
                if age_stats:
                    f.write("<h3>Age Distribution (Included Patients)</h3>\n")
                    f.write('<div class="metrics">\n')
                    f.write(f'<div class="metric"><span class="metric-label">Min Age:</span><br/><span class="metric-value">{age_stats.get("min", "N/A")}</span></div>\n')
                    f.write(f'<div class="metric"><span class="metric-label">Max Age:</span><br/><span class="metric-value">{age_stats.get("max", "N/A")}</span></div>\n')
                    f.write(f'<div class="metric"><span class="metric-label">Mean Age:</span><br/><span class="metric-value">{age_stats.get("mean", 0):.1f}</span></div>\n')
                    f.write(f'<div class="metric"><span class="metric-label">Median Age:</span><br/><span class="metric-value">{age_stats.get("median", 0):.1f}</span></div>\n')
                    f.write('</div>\n')

                # Sex distribution
                sex_dist = clinical_summary.get("sex_distribution", {})
                if sex_dist:
                    f.write("<h3>Sex Distribution</h3>\n")
                    f.write("<table>\n")
                    f.write("<tr><th>Sex</th><th>Count</th></tr>\n")
                    for sex, count in sex_dist.items():
                        f.write(f'<tr><td>{sex.capitalize()}</td><td>{count}</td></tr>\n')
                    f.write("</table>\n")

                # Growth distribution
                growth_dist = clinical_summary.get("growth_distribution", {})
                if growth_dist:
                    f.write("<h3>Tumor Growth Status</h3>\n")
                    f.write("<table>\n")
                    f.write("<tr><th>Status</th><th>Count</th></tr>\n")
                    for status, count in growth_dist.items():
                        f.write(f'<tr><td>{status.capitalize()}</td><td>{count}</td></tr>\n')
                    f.write("</table>\n")

                # Exclusion reasons
                exclusion_reasons = clinical_summary.get("exclusion_reasons", {})
                if exclusion_reasons:
                    f.write("<h3>Exclusion Reasons</h3>\n")
                    f.write("<table>\n")
                    f.write("<tr><th>Reason</th><th>Count</th></tr>\n")
                    for reason, count in exclusion_reasons.items():
                        f.write(f'<tr><td>{reason}</td><td>{count}</td></tr>\n')
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

    def run_visualization(
        self,
        metadata_manager: Optional["MetadataManager"] = None,
        quality_metrics_path: Optional[Path] = None,
    ) -> Dict[str, Path]:
        """Run complete visualization pipeline.

        Args:
            metadata_manager: Optional metadata manager for clinical plots.
            quality_metrics_path: Optional path to quality_metrics.json for
                quality filtering plots and enhanced HTML report.

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

        # Generate QC plots
        plot_paths = self.generate_all_plots(results)

        # Generate clinical plots if metadata is available
        clinical_plots = {}
        if metadata_manager:
            clinical_plots = self.generate_clinical_plots(metadata_manager)
            plot_paths.update(clinical_plots)

        # Generate quality filtering plots if metrics are available
        quality_metrics = None
        if quality_metrics_path and quality_metrics_path.exists():
            with open(quality_metrics_path, "r", encoding="utf-8") as f:
                quality_metrics = json.load(f)
            qf_plots = self.generate_quality_filtering_plots(quality_metrics)
            plot_paths.update(qf_plots)

        # Generate HTML report
        output_paths = {"plots": plot_paths, "clinical_plots": clinical_plots}

        if self.config.visualization.html_report.enabled:
            html_path = self.generate_html_report(
                results, plot_paths,
                metadata_manager=metadata_manager,
                quality_metrics=quality_metrics,
            )
            output_paths["html_report"] = html_path

        return output_paths
