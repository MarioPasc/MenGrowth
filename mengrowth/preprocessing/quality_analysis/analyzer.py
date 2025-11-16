"""Quality analysis orchestrator for MRI dataset analysis.

This module provides the main QualityAnalyzer class that coordinates dataset
scanning, metric computation, parallelization, and result aggregation.
"""

import csv
import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from mengrowth.preprocessing.config import QualityAnalysisConfig
from mengrowth.preprocessing.quality_analysis.metrics import (
    compute_contrast_ratio,
    compute_dimension_statistics,
    compute_histogram,
    compute_image_dimensions,
    compute_intensity_statistics,
    compute_missing_sequences,
    compute_patient_statistics,
    compute_snr_estimate,
    compute_spacing_statistics,
    compute_voxel_spacing,
    detect_outliers_iqr,
    detect_outliers_zscore,
    load_image,
)

logger = logging.getLogger(__name__)


class QualityAnalyzer:
    """Main orchestrator for dataset quality analysis.

    This class handles dataset scanning, parallel metric computation,
    result aggregation, and output file generation.

    Attributes:
        config: Quality analysis configuration.
        dataset_structure: Nested dict of patient -> study -> sequences.
        results: Aggregated analysis results.
    """

    def __init__(self, config: QualityAnalysisConfig):
        """Initialize quality analyzer.

        Args:
            config: Validated QualityAnalysisConfig object.
        """
        self.config = config
        self.dataset_structure: Dict[str, Dict[str, List[str]]] = {}
        self.results: Dict[str, any] = {
            "per_study": [],
            "per_patient": [],
            "per_sequence": {},
            "summary": {},
        }
        self.logger = logging.getLogger(__name__)

    def scan_dataset(self) -> Dict[str, Dict[str, List[str]]]:
        """Scan input directory to build dataset structure.

        Returns:
            Nested dictionary: {patient_id: {study_id: [sequence_names]}}.

        Examples:
            >>> analyzer = QualityAnalyzer(config)
            >>> structure = analyzer.scan_dataset()
            >>> print(structure['MenGrowth-0001'])
            {'MenGrowth-0001-000': ['t1c', 't1n', 't2w'], ...}
        """
        self.logger.info(f"Scanning dataset: {self.config.input_dir}")

        if not self.config.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {self.config.input_dir}")

        dataset_structure = {}

        # Detect file extension based on format
        if self.config.file_format == "auto":
            # Search for both formats
            extensions = ["*.nrrd", "*.nii.gz", "*.nii"]
        elif self.config.file_format == "nrrd":
            extensions = ["*.nrrd"]
        elif self.config.file_format == "nifti":
            extensions = ["*.nii.gz", "*.nii"]
        else:
            raise ValueError(f"Unknown file format: {self.config.file_format}")

        # Iterate through patient directories (MenGrowth-XXXX)
        patient_dirs = sorted([d for d in self.config.input_dir.iterdir() if d.is_dir()])

        for patient_dir in patient_dirs:
            patient_id = patient_dir.name
            dataset_structure[patient_id] = {}

            # Iterate through study directories (MenGrowth-XXXX-YYY)
            study_dirs = sorted([d for d in patient_dir.iterdir() if d.is_dir()])

            for study_dir in study_dirs:
                study_id = study_dir.name
                sequences = []

                # Find all image files with expected extensions
                for ext in extensions:
                    image_files = list(study_dir.glob(ext))
                    for img_file in image_files:
                        # Extract sequence name (remove extension)
                        if img_file.suffix == ".gz":
                            # Handle .nii.gz
                            seq_name = img_file.stem.replace(".nii", "")
                        else:
                            seq_name = img_file.stem

                        sequences.append(seq_name)

                if sequences:
                    dataset_structure[patient_id][study_id] = sequences

        self.dataset_structure = dataset_structure

        # Log summary
        total_patients = len(dataset_structure)
        total_studies = sum(len(studies) for studies in dataset_structure.values())
        total_sequences = sum(
            len(seqs)
            for studies in dataset_structure.values()
            for seqs in studies.values()
        )

        self.logger.info(f"Found {total_patients} patients, {total_studies} studies, {total_sequences} sequences")

        return dataset_structure

    def analyze_patient(
        self, patient_id: str, studies: Dict[str, List[str]]
    ) -> Dict[str, any]:
        """Analyze all studies for a single patient.

        This function is the parallelizable unit - each patient is analyzed independently.

        Args:
            patient_id: Patient identifier.
            studies: Dictionary mapping study_id -> list of sequence names.

        Returns:
            Dictionary containing per-study metrics for this patient.

        Examples:
            >>> results = analyzer.analyze_patient('MenGrowth-0001', studies_dict)
            >>> print(len(results['studies']))
            3
        """
        patient_results = {
            "patient_id": patient_id,
            "num_studies": len(studies),
            "studies": [],
        }

        for study_id, sequences in studies.items():
            study_path = self.config.input_dir / patient_id / study_id
            study_result = {
                "patient_id": patient_id,
                "study_id": study_id,
                "sequences": {},
                "missing_sequences": [],
            }

            # Detect missing sequences
            for expected_seq in self.config.expected_sequences:
                if expected_seq not in sequences:
                    study_result["missing_sequences"].append(expected_seq)

            # Analyze each available sequence
            for sequence_name in sequences:
                if self.config.dry_run:
                    # In dry-run mode, skip actual image loading
                    seq_result = {
                        "sequence_name": sequence_name,
                        "dry_run": True,
                    }
                else:
                    # Load image and compute metrics
                    image_path = self._find_image_path(study_path, sequence_name)

                    if image_path is None:
                        self.logger.warning(
                            f"Could not find image file for {study_id}/{sequence_name}"
                        )
                        continue

                    image = load_image(image_path, self.config.file_format)

                    if image is None:
                        self.logger.warning(
                            f"Failed to load image: {image_path}"
                        )
                        continue

                    seq_result = self._compute_sequence_metrics(
                        image, sequence_name, image_path
                    )

                study_result["sequences"][sequence_name] = seq_result

            patient_results["studies"].append(study_result)

        return patient_results

    def _find_image_path(self, study_path: Path, sequence_name: str) -> Optional[Path]:
        """Find the full path to an image file given the study path and sequence name.

        Args:
            study_path: Path to study directory.
            sequence_name: Name of the sequence (e.g., 't1c').

        Returns:
            Path to image file, or None if not found.
        """
        # Try different extensions
        for ext in [".nrrd", ".nii.gz", ".nii"]:
            image_path = study_path / f"{sequence_name}{ext}"
            if image_path.exists():
                return image_path

        return None

    def _compute_sequence_metrics(
        self, image, sequence_name: str, image_path: Path
    ) -> Dict[str, any]:
        """Compute all enabled metrics for a single sequence image.

        Args:
            image: SimpleITK Image object.
            sequence_name: Name of the sequence.
            image_path: Path to the image file.

        Returns:
            Dictionary containing all computed metrics.
        """
        metrics = {
            "sequence_name": sequence_name,
            "file_path": str(image_path),
        }

        try:
            # Image dimensions
            if self.config.metrics.image_dimensions:
                dimensions = compute_image_dimensions(image)
                metrics["width"] = dimensions[0]
                metrics["height"] = dimensions[1]
                metrics["depth"] = dimensions[2]

            # Voxel spacing
            if self.config.metrics.voxel_spacing:
                spacing = compute_voxel_spacing(image)
                metrics["spacing_x"] = spacing[0]
                metrics["spacing_y"] = spacing[1]
                metrics["spacing_z"] = spacing[2]

            # Intensity statistics
            if self.config.metrics.intensity_statistics:
                intensity_stats = compute_intensity_statistics(
                    image, self.config.intensity_percentiles
                )
                metrics.update(intensity_stats)

            # SNR estimation
            if self.config.metrics.snr_estimation:
                snr = compute_snr_estimate(image)
                metrics["snr"] = snr

            # Contrast ratio
            contrast = compute_contrast_ratio(image)
            metrics["contrast_ratio"] = contrast

        except Exception as e:
            self.logger.warning(
                f"Error computing metrics for {image_path}: {e}"
            )
            metrics["error"] = str(e)

        return metrics

    def run_analysis(self) -> Dict[str, any]:
        """Run complete quality analysis with parallel processing.

        Returns:
            Dictionary containing all analysis results.

        Examples:
            >>> analyzer = QualityAnalyzer(config)
            >>> results = analyzer.run_analysis()
            >>> print(results['summary']['total_patients'])
            47
        """
        self.logger.info("Starting quality analysis...")

        # Step 1: Scan dataset
        dataset_structure = self.scan_dataset()

        if not dataset_structure:
            self.logger.warning("No data found in dataset")
            return self.results

        # Step 2: Analyze patients (with parallelization if enabled)
        all_patient_results = []

        if self.config.parallel.enabled and not self.config.dry_run:
            # Parallel processing
            n_workers = self.config.parallel.n_workers or None
            self.logger.info(f"Analyzing patients in parallel (workers={n_workers})...")

            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                # Submit all patient analysis tasks
                future_to_patient = {
                    executor.submit(self.analyze_patient, patient_id, studies): patient_id
                    for patient_id, studies in dataset_structure.items()
                }

                # Collect results as they complete
                for idx, future in enumerate(as_completed(future_to_patient), 1):
                    patient_id = future_to_patient[future]
                    try:
                        patient_result = future.result()
                        all_patient_results.append(patient_result)
                        self.logger.info(
                            f"Analyzed patient {patient_id} ({idx}/{len(dataset_structure)})"
                        )
                    except Exception as e:
                        self.logger.error(
                            f"Failed to analyze patient {patient_id}: {e}"
                        )
        else:
            # Sequential processing
            self.logger.info("Analyzing patients sequentially...")

            for idx, (patient_id, studies) in enumerate(dataset_structure.items(), 1):
                try:
                    patient_result = self.analyze_patient(patient_id, studies)
                    all_patient_results.append(patient_result)
                    self.logger.info(
                        f"Analyzed patient {patient_id} ({idx}/{len(dataset_structure)})"
                    )
                except Exception as e:
                    self.logger.error(f"Failed to analyze patient {patient_id}: {e}")

        # Step 3: Aggregate results
        self.logger.info("Aggregating results...")
        self._aggregate_results(all_patient_results, dataset_structure)

        self.logger.info("Analysis complete")
        return self.results

    def _aggregate_results(
        self, all_patient_results: List[Dict], dataset_structure: Dict
    ):
        """Aggregate per-patient results into summary statistics.

        Args:
            all_patient_results: List of per-patient result dictionaries.
            dataset_structure: Dataset structure from scan_dataset().
        """
        # Flatten to per-study results
        per_study_results = []
        for patient_result in all_patient_results:
            for study_result in patient_result["studies"]:
                per_study_results.append(study_result)

        self.results["per_study"] = per_study_results

        # Per-patient summary
        self.results["per_patient"] = [
            {
                "patient_id": pr["patient_id"],
                "num_studies": pr["num_studies"],
            }
            for pr in all_patient_results
        ]

        # Patient statistics
        if self.config.metrics.patient_statistics:
            patient_studies = {
                patient_id: list(studies.keys())
                for patient_id, studies in dataset_structure.items()
            }
            patient_stats = compute_patient_statistics(patient_studies)
            self.results["summary"]["patient_statistics"] = patient_stats

        # Missing sequence statistics
        if self.config.metrics.missing_sequences:
            study_sequences = {
                f"{patient_id}/{study_id}": sequences
                for patient_id, studies in dataset_structure.items()
                for study_id, sequences in studies.items()
            }
            missing_stats = compute_missing_sequences(
                study_sequences, self.config.expected_sequences
            )
            self.results["summary"]["missing_sequences"] = missing_stats

        # Per-sequence aggregated statistics (if not dry-run)
        if not self.config.dry_run:
            self._aggregate_sequence_statistics(per_study_results)

    def _aggregate_sequence_statistics(self, per_study_results: List[Dict]):
        """Aggregate statistics across all instances of each sequence type.

        Args:
            per_study_results: List of per-study result dictionaries.
        """
        # Group data by sequence type
        sequence_data = {seq: [] for seq in self.config.expected_sequences}

        for study_result in per_study_results:
            for seq_name, seq_metrics in study_result["sequences"].items():
                if seq_name in sequence_data:
                    sequence_data[seq_name].append(seq_metrics)

        # Compute aggregated statistics for each sequence
        for seq_name, seq_list in sequence_data.items():
            if not seq_list:
                continue

            seq_stats = {}

            # Spacing statistics
            if self.config.metrics.voxel_spacing:
                spacings = [
                    (m["spacing_x"], m["spacing_y"], m["spacing_z"])
                    for m in seq_list
                    if "spacing_x" in m
                ]
                if spacings:
                    seq_stats["spacing"] = compute_spacing_statistics(spacings)

            # Dimension statistics
            if self.config.metrics.image_dimensions:
                dimensions = [
                    (m["width"], m["height"], m["depth"])
                    for m in seq_list
                    if "width" in m
                ]
                if dimensions:
                    seq_stats["dimensions"] = compute_dimension_statistics(dimensions)

            # Intensity statistics
            if self.config.metrics.intensity_statistics:
                intensity_means = [m["mean"] for m in seq_list if "mean" in m]
                intensity_stds = [m["std"] for m in seq_list if "std" in m]

                if intensity_means:
                    seq_stats["intensity"] = {
                        "mean_across_studies": float(np.mean(intensity_means)),
                        "std_across_studies": float(np.std(intensity_means)),
                        "mean_std": float(np.mean(intensity_stds)),
                    }

                    # Detect outliers
                    if self.config.outlier_detection.enabled:
                        if self.config.outlier_detection.method == "iqr":
                            outliers, lower, upper = detect_outliers_iqr(
                                np.array(intensity_means),
                                self.config.outlier_detection.iqr_multiplier,
                            )
                        else:
                            outliers, lower, upper = detect_outliers_zscore(
                                np.array(intensity_means),
                                self.config.outlier_detection.zscore_threshold,
                            )

                        seq_stats["intensity"]["outlier_count"] = len(outliers)
                        seq_stats["intensity"]["outlier_indices"] = outliers
                        seq_stats["intensity"]["outlier_lower_bound"] = lower
                        seq_stats["intensity"]["outlier_upper_bound"] = upper

            # SNR statistics
            if self.config.metrics.snr_estimation:
                snr_values = [m["snr"] for m in seq_list if "snr" in m and m["snr"] != float('inf')]
                if snr_values:
                    seq_stats["snr"] = {
                        "mean": float(np.mean(snr_values)),
                        "std": float(np.std(snr_values)),
                        "min": float(np.min(snr_values)),
                        "max": float(np.max(snr_values)),
                    }

            self.results["per_sequence"][seq_name] = seq_stats

    def save_results(self) -> Dict[str, Path]:
        """Save analysis results to configured output formats.

        Returns:
            Dictionary mapping output type to saved file path.

        Examples:
            >>> analyzer.save_results()
            {'per_study_csv': PosixPath('.../per_study_metrics.csv'), ...}
        """
        self.logger.info(f"Saving results to: {self.config.output_dir}")

        # Create output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        saved_files = {}

        # Per-study CSV
        if self.config.output.save_per_study_csv and self.results["per_study"]:
            csv_path = self.config.output_dir / "per_study_metrics.csv"
            self._save_per_study_csv(csv_path)
            saved_files["per_study_csv"] = csv_path
            self.logger.info(f"Saved per-study CSV: {csv_path}")

        # Per-patient CSV
        if self.config.output.save_per_patient_csv and self.results["per_patient"]:
            csv_path = self.config.output_dir / "per_patient_summary.csv"
            self._save_per_patient_csv(csv_path)
            saved_files["per_patient_csv"] = csv_path
            self.logger.info(f"Saved per-patient CSV: {csv_path}")

        # Summary JSON
        if self.config.output.save_summary_json:
            json_path = self.config.output_dir / "summary_statistics.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(self.results["summary"], f, indent=2, default=str)
            saved_files["summary_json"] = json_path
            self.logger.info(f"Saved summary JSON: {json_path}")

        # Per-sequence statistics JSON
        json_path = self.config.output_dir / "per_sequence_statistics.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.results["per_sequence"], f, indent=2, default=str)
        saved_files["per_sequence_json"] = json_path
        self.logger.info(f"Saved per-sequence JSON: {json_path}")

        # Analysis metadata
        if self.config.output.save_metadata:
            metadata_path = self.config.output_dir / "analysis_metadata.json"
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "input_dir": str(self.config.input_dir),
                "output_dir": str(self.config.output_dir),
                "file_format": self.config.file_format,
                "expected_sequences": self.config.expected_sequences,
                "dry_run": self.config.dry_run,
                "parallel_enabled": self.config.parallel.enabled,
                "n_workers": self.config.parallel.n_workers,
            }
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
            saved_files["metadata"] = metadata_path
            self.logger.info(f"Saved metadata: {metadata_path}")

        return saved_files

    def _save_per_study_csv(self, csv_path: Path):
        """Save per-study metrics to CSV file.

        Args:
            csv_path: Path to output CSV file.
        """
        # Flatten study results for CSV format
        rows = []
        for study_result in self.results["per_study"]:
            for seq_name, seq_metrics in study_result["sequences"].items():
                row = {
                    "patient_id": study_result["patient_id"],
                    "study_id": study_result["study_id"],
                    "sequence": seq_name,
                }
                row.update(seq_metrics)
                rows.append(row)

        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(csv_path, index=False)

    def _save_per_patient_csv(self, csv_path: Path):
        """Save per-patient summary to CSV file.

        Args:
            csv_path: Path to output CSV file.
        """
        df = pd.DataFrame(self.results["per_patient"])
        df.to_csv(csv_path, index=False)
