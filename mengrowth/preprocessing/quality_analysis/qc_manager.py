"""QC Manager for lightweight, label-free per-step quality metrics.

This module implements a QCManager that:
1. Accumulates QC metrics after specified preprocessing steps
2. Performs two-pass Wasserstein distance computation
3. Outputs tidy/wide CSVs and summary with outlier detection
4. Captures baseline metrics before preprocessing
5. Computes pre-vs-post comparisons for quality tracking
6. Computes SNR/CNR metrics for image quality assessment
"""

from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging
import json
import uuid
import numpy as np
import pandas as pd
import SimpleITK as sitk
from datetime import datetime
from dataclasses import asdict

from mengrowth.preprocessing.src.config import QCConfig
from mengrowth.preprocessing.quality_analysis.snr_cnr_metrics import SNRCNRCalculator

logger = logging.getLogger(__name__)


class QCManager:
    """Manages QC metric collection during preprocessing pipeline execution.

    This class operates in two phases:

    PHASE 1 (Accumulation):
    - on_step_completed() is called after each step completes
    - Computes geometry, registration, mask, and intensity metrics
    - Accumulates histograms for Wasserstein reference computation

    PHASE 2 (Finalization):
    - finalize() is called after all patients processed
    - Computes reference distributions (per site/modality or global)
    - Computes Wasserstein distances to references
    - Detects outliers using MAD or IQR
    - Writes CSV and JSON outputs

    Attributes:
        config: QC configuration
        metrics_accumulator: List of metric dicts (one per QC event)
        histogram_accumulator: Dict for two-pass Wasserstein
        reference_histograms: Computed reference distributions
        site_map: Optional dict mapping patient_id -> site
        pipeline_run_id: Unique ID for this pipeline run
    """

    def __init__(self, config: QCConfig, site_map: Optional[Dict[str, str]] = None):
        """Initialize QC Manager.

        Args:
            config: QC configuration dataclass
            site_map: Optional mapping of patient_id -> site (for site-specific references)
        """
        self.config = config
        self.metrics_accumulator: List[Dict[str, Any]] = []
        self.histogram_accumulator: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}
        self.reference_histograms: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self.site_map = site_map or {}
        self.pipeline_run_id = str(uuid.uuid4())
        self._logger = logging.getLogger(__name__)

        # Baseline metrics storage: key = "patient_id/study_id/modality"
        self._baseline_metrics: Dict[str, Dict[str, Any]] = {}

        # Warped mask storage for longitudinal comparison: key = patient_id
        self._warped_masks_info: Dict[str, Dict[str, Dict[str, Any]]] = {}

        # SNR/CNR calculator
        self._snr_cnr_calculator: Optional[SNRCNRCalculator] = None
        if config.metrics.snr_cnr.enabled:
            snr_config = {
                "background_percentile": config.metrics.snr_cnr.background_percentile,
                "foreground_percentile": config.metrics.snr_cnr.foreground_percentile,
                "edge_erosion_iters": config.metrics.snr_cnr.edge_erosion_iters,
                "intensity_low_pct": config.metrics.snr_cnr.intensity_low_pct,
                "intensity_mid_pct": config.metrics.snr_cnr.intensity_mid_pct,
                "intensity_high_pct": config.metrics.snr_cnr.intensity_high_pct,
            }
            self._snr_cnr_calculator = SNRCNRCalculator(config=snr_config)

        # Create output directories
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(config.artifacts_dir).mkdir(parents=True, exist_ok=True)

        self._logger.info(f"QCManager initialized: run_id={self.pipeline_run_id[:8]}, output={config.output_dir}")

    def on_step_completed(
        self,
        step_name: str,
        case_metadata: Dict[str, Any],
        image_paths: Dict[str, Path],
        artifact_paths: Optional[Dict[str, Path]] = None
    ) -> None:
        """Callback after a step completes - compute and accumulate QC metrics.

        Args:
            step_name: Name of completed step (e.g., "data_harmonization_1")
            case_metadata: Dict with patient_id, study_id, modality, timestamp, etc.
            image_paths: Dict with keys like "input", "output", "reference", "moving"
            artifact_paths: Optional dict with artifacts (e.g., "mask", "transform")
        """
        if not self.config.enabled:
            return

        # Check if this step should trigger QC
        step_base = self._extract_step_base(step_name)
        if step_base not in self.config.compute_after_steps:
            return

        self._logger.debug(f"Computing QC metrics for {step_name}")

        try:
            metrics = self._compute_metrics(
                step_name=step_name,
                case_metadata=case_metadata,
                image_paths=image_paths,
                artifact_paths=artifact_paths or {}
            )
            self.metrics_accumulator.append(metrics)
        except Exception as e:
            self._logger.error(f"QC metric computation failed for {step_name}: {e}", exc_info=True)
            # Continue execution - QC errors should not halt pipeline
            error_metrics = {
                "step_name": step_name,
                "timestamp": datetime.now().isoformat(),
                "status": "error",
                "error_message": str(e),
                **case_metadata
            }
            self.metrics_accumulator.append(error_metrics)

    def capture_baseline(
        self,
        case_metadata: Dict[str, Any],
        image_path: Path
    ) -> Dict[str, Any]:
        """Capture baseline metrics before preprocessing begins.

        This method should be called before the first preprocessing step
        to establish a baseline for pre-vs-post comparison.

        Args:
            case_metadata: Dict with patient_id, study_id, modality, etc.
            image_path: Path to the input image before any preprocessing

        Returns:
            Dict with captured baseline metrics
        """
        if not self.config.enabled or not self.config.metrics.baseline.enabled:
            return {}

        if not self.config.metrics.baseline.capture_before_first_step:
            return {}

        patient_id = case_metadata.get("patient_id", "unknown")
        study_id = case_metadata.get("study_id", "unknown")
        modality = case_metadata.get("modality", "unknown")

        baseline_key = f"{patient_id}/{study_id}/{modality}"

        if baseline_key in self._baseline_metrics:
            self._logger.debug(f"Baseline already captured for {baseline_key}")
            return self._baseline_metrics[baseline_key]

        self._logger.info(f"Capturing baseline metrics for {baseline_key}")

        try:
            baseline = self._compute_baseline_metrics(image_path, case_metadata)
            baseline["capture_timestamp"] = datetime.now().isoformat()
            baseline["image_path"] = str(image_path)

            self._baseline_metrics[baseline_key] = baseline

            # Save baseline to artifacts directory
            artifacts_dir = Path(self.config.artifacts_dir) / patient_id / study_id
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            baseline_file = artifacts_dir / f"baseline_metrics_{modality}.json"
            with open(baseline_file, 'w') as f:
                json.dump(baseline, f, indent=2, default=str)

            self._logger.debug(f"Baseline saved to {baseline_file}")
            return baseline

        except Exception as e:
            self._logger.warning(f"Baseline capture failed for {baseline_key}: {e}")
            return {}

    def _compute_baseline_metrics(
        self,
        image_path: Path,
        case_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compute baseline metrics for an image.

        Args:
            image_path: Path to input image
            case_metadata: Case metadata dict

        Returns:
            Dict with baseline metrics
        """
        from mengrowth.preprocessing.quality_analysis.qc_metrics import (
            compute_geometry_metrics,
            compute_intensity_stats_for_wasserstein,
            downsample_image_for_qc
        )

        metrics: Dict[str, Any] = {
            "patient_id": case_metadata.get("patient_id"),
            "study_id": case_metadata.get("study_id"),
            "modality": case_metadata.get("modality"),
        }

        if not image_path.exists():
            metrics["status"] = "missing_input"
            return metrics

        image = sitk.ReadImage(str(image_path))
        image_downsampled, downsample_factor = downsample_image_for_qc(
            image,
            target_mm=self.config.downsample_to_mm,
            max_voxels=self.config.max_voxels,
            seed=self.config.random_seed
        )

        metrics_to_capture = self.config.metrics.baseline.metrics_to_capture

        # Geometry metrics
        if "geometry" in metrics_to_capture and self.config.metrics.geometry.enabled:
            geom = compute_geometry_metrics(image_downsampled, self.config.metrics.geometry)
            for k, v in geom.items():
                metrics[f"baseline_{k}"] = v

        # Intensity metrics
        if "intensity" in metrics_to_capture and self.config.metrics.intensity_stability.enabled:
            intensity_metrics, _, _ = compute_intensity_stats_for_wasserstein(
                image_downsampled,
                mask=None,
                config=self.config.metrics.intensity_stability
            )
            for k, v in intensity_metrics.items():
                metrics[f"baseline_{k}"] = v

        # SNR/CNR metrics
        if "snr_cnr" in metrics_to_capture and self._snr_cnr_calculator is not None:
            snr_cnr_metrics = self._snr_cnr_calculator.compute_all_metrics(image_downsampled)
            for k, v in snr_cnr_metrics.items():
                if not k.startswith("region_"):  # Skip verbose region stats
                    metrics[f"baseline_{k}"] = v

        metrics["status"] = "success"
        return metrics

    def compute_pre_post_comparison(
        self,
        step_name: str,
        case_metadata: Dict[str, Any],
        post_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare post-processing metrics to baseline.

        Args:
            step_name: Name of the completed step
            case_metadata: Case metadata dict
            post_metrics: Metrics computed after the step

        Returns:
            Dict with comparison metrics (deltas, ratios, flags)
        """
        if not self.config.enabled or not self.config.metrics.comparison.enabled:
            return {}

        patient_id = case_metadata.get("patient_id", "unknown")
        study_id = case_metadata.get("study_id", "unknown")
        modality = case_metadata.get("modality", "unknown")

        baseline_key = f"{patient_id}/{study_id}/{modality}"
        baseline = self._baseline_metrics.get(baseline_key)

        if not baseline:
            self._logger.debug(f"No baseline found for {baseline_key}, skipping comparison")
            return {}

        comparison: Dict[str, Any] = {
            "step_name": step_name,
            "comparison_type": "pre_post",
        }

        comparison_config = self.config.metrics.comparison

        # Find matching metric pairs (baseline_X vs X)
        for post_key, post_value in post_metrics.items():
            if not isinstance(post_value, (int, float)) or post_value is None:
                continue

            # Skip internal/metadata keys
            if post_key.startswith("_") or post_key in [
                "pipeline_run_id", "step_name", "step_base", "timestamp",
                "patient_id", "study_id", "modality", "status", "downsample_factor"
            ]:
                continue

            baseline_key_name = f"baseline_{post_key}"
            baseline_value = baseline.get(baseline_key_name)

            if baseline_value is None or not isinstance(baseline_value, (int, float)):
                continue

            # Compute delta
            if comparison_config.compute_deltas:
                delta = post_value - baseline_value
                comparison[f"{post_key}_delta"] = float(delta)

            # Compute ratio
            if comparison_config.compute_ratios and baseline_value != 0:
                ratio = post_value / baseline_value
                comparison[f"{post_key}_ratio"] = float(ratio)

                # Flag significant degradation
                # For SNR/CNR: lower is worse, so check if ratio < threshold
                # For some metrics, interpretation depends on context
                threshold_pct = comparison_config.flag_degradation_threshold_pct
                if post_key in ["snr_background", "snr_foreground", "cnr_high_low"]:
                    # Lower is worse for SNR/CNR
                    if ratio < (1.0 - threshold_pct / 100.0):
                        comparison[f"{post_key}_degraded"] = True
                else:
                    # Generic: flag large changes in either direction
                    if abs(ratio - 1.0) > (threshold_pct / 100.0):
                        comparison[f"{post_key}_changed"] = True

        return comparison

    def store_warped_masks_info(
        self,
        patient_id: str,
        warped_masks: Dict[str, Dict[str, Any]]
    ) -> None:
        """Store warped mask information from longitudinal registration for later comparison.

        Args:
            patient_id: Patient identifier
            warped_masks: Dict mapping "timestamp_modality" to {ref_mask, warped_mask} paths
        """
        if not self.config.enabled:
            return

        if not self.config.metrics.mask_plausibility.longitudinal_dice:
            return

        self._warped_masks_info[patient_id] = warped_masks
        self._logger.debug(
            f"Stored {len(warped_masks)} warped mask entries for {patient_id}"
        )

    def finalize(self) -> Dict[str, Path]:
        """Finalize QC: compute references, Wasserstein distances, detect outliers, write outputs.

        Returns:
            Dict mapping output type to file path
        """
        if not self.config.enabled or not self.metrics_accumulator:
            self._logger.info("No QC metrics to finalize")
            return {}

        self._logger.info(f"Finalizing QC: {len(self.metrics_accumulator)} metric records")

        # Step 1: Compute reference distributions for Wasserstein
        self._compute_reference_distributions()

        # Step 2: Compute Wasserstein distances
        self._compute_wasserstein_distances()

        # Step 3: Compute mask comparisons (longitudinal)
        mask_comparison_results = self._compute_mask_comparisons()

        # Step 4: Detect outliers
        self._detect_outliers()

        # Step 5: Write outputs
        output_paths = self._write_outputs()

        # Step 6: Write mask comparison results
        if mask_comparison_results:
            mask_output = self._write_mask_comparison_results(mask_comparison_results)
            output_paths.update(mask_output)

        # Step 7: Generate HTML report
        html_report_path = self._generate_html_report(mask_comparison_results)
        if html_report_path:
            output_paths["html_report"] = html_report_path

        self._logger.info(f"QC finalization complete: {len(output_paths)} files written")
        return output_paths

    def _compute_mask_comparisons(self) -> Dict[str, Dict[str, Any]]:
        """Compute mask comparisons for all stored warped mask info.

        Returns:
            Dict mapping patient_id to comparison results
        """
        if not self._warped_masks_info:
            return {}

        if not self.config.metrics.mask_plausibility.longitudinal_dice:
            return {}

        from mengrowth.preprocessing.quality_analysis.mask_comparison import (
            compute_all_mask_comparisons,
            summarize_mask_comparisons
        )

        all_results: Dict[str, Dict[str, Any]] = {}

        for patient_id, warped_masks in self._warped_masks_info.items():
            self._logger.info(f"Computing mask comparisons for {patient_id}")

            # Determine artifacts directory for this patient
            artifacts_dir = Path(self.config.artifacts_dir).parent / patient_id

            try:
                comparisons = compute_all_mask_comparisons(
                    warped_masks_info=warped_masks,
                    artifacts_dir=artifacts_dir,
                    output_dir=Path(self.config.artifacts_dir) / patient_id
                )

                summary = summarize_mask_comparisons(comparisons)

                all_results[patient_id] = {
                    "comparisons": comparisons,
                    "summary": summary
                }

                self._logger.info(
                    f"  {patient_id}: {summary.get('n_comparisons', 0)} comparisons, "
                    f"mean Dice={summary.get('dice_mean', 'N/A'):.3f}"
                    if summary.get('dice_mean') else f"  {patient_id}: no valid comparisons"
                )

            except Exception as e:
                self._logger.warning(f"Mask comparison failed for {patient_id}: {e}")

        return all_results

    def _write_mask_comparison_results(
        self,
        results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Path]:
        """Write mask comparison results to files.

        Args:
            results: Dict from _compute_mask_comparisons

        Returns:
            Dict mapping output type to file path
        """
        output_paths = {}

        # Aggregate all summaries into a single CSV
        summary_rows = []
        for patient_id, patient_results in results.items():
            summary = patient_results.get("summary", {})
            summary["patient_id"] = patient_id
            summary_rows.append(summary)

        if summary_rows:
            import pandas as pd
            df = pd.DataFrame(summary_rows)
            summary_csv = Path(self.config.output_dir) / "mask_comparison_summary.csv"
            df.to_csv(summary_csv, index=False)
            output_paths["mask_comparison_summary"] = summary_csv
            self._logger.info(f"Wrote mask comparison summary: {summary_csv}")

        # Write detailed comparisons JSON
        detailed_json = Path(self.config.output_dir) / "mask_comparisons_detailed.json"
        with open(detailed_json, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        output_paths["mask_comparisons_detailed"] = detailed_json
        self._logger.info(f"Wrote detailed mask comparisons: {detailed_json}")

        return output_paths

    def _generate_html_report(
        self,
        mask_comparison_results: Optional[Dict[str, Any]] = None
    ) -> Optional[Path]:
        """Generate HTML QC report.

        Args:
            mask_comparison_results: Optional mask comparison results

        Returns:
            Path to HTML report if generated, None otherwise
        """
        try:
            from mengrowth.preprocessing.quality_analysis.html_report import generate_qc_report

            # Convert metrics to DataFrame
            df = pd.DataFrame(self.metrics_accumulator)

            # Remove internal columns
            internal_cols = [c for c in df.columns if c.startswith("_")]
            df = df.drop(columns=internal_cols, errors='ignore')

            # Build summary
            summary = {
                "pipeline_run_id": self.pipeline_run_id,
                "timestamp": datetime.now().isoformat(),
                "n_records": len(self.metrics_accumulator),
                "reference_groups": list(self.reference_histograms.keys()),
            }

            # Generate report
            report_path = generate_qc_report(
                metrics_df=df,
                summary=summary,
                mask_comparisons=mask_comparison_results,
                output_dir=Path(self.config.output_dir),
                title="MenGrowth Preprocessing QC Report"
            )

            self._logger.info(f"Generated HTML report: {report_path}")
            return report_path

        except Exception as e:
            self._logger.warning(f"HTML report generation failed: {e}")
            return None

    # =====================================================================
    # PRIVATE METHODS: Metric Computation
    # =====================================================================

    def _compute_metrics(
        self,
        step_name: str,
        case_metadata: Dict[str, Any],
        image_paths: Dict[str, Path],
        artifact_paths: Dict[str, Path]
    ) -> Dict[str, Any]:
        """Compute all enabled metrics for this QC event.

        Returns:
            Dict with metadata + computed metrics
        """
        from mengrowth.preprocessing.quality_analysis.qc_metrics import (
            compute_geometry_metrics,
            compute_registration_similarity,
            compute_mask_plausibility,
            compute_intensity_stats_for_wasserstein,
            compute_longitudinal_mask_dice,
            downsample_image_for_qc,
            get_mask_for_qc
        )

        metrics = {
            # Metadata
            "pipeline_run_id": self.pipeline_run_id,
            "step_name": step_name,
            "step_base": self._extract_step_base(step_name),
            "timestamp": datetime.now().isoformat(),
            **case_metadata  # patient_id, study_id, modality, etc.
        }

        # Load primary output image
        output_path = image_paths.get("output")
        if not output_path or not output_path.exists():
            self._logger.warning(f"Output image not found: {output_path}")
            metrics["status"] = "missing_output"
            return metrics

        # Downsample for cheap computation
        image = sitk.ReadImage(str(output_path))
        image_downsampled, downsample_factor = downsample_image_for_qc(
            image,
            target_mm=self.config.downsample_to_mm,
            max_voxels=self.config.max_voxels,
            seed=self.config.random_seed
        )
        metrics["downsample_factor"] = downsample_factor

        # Get mask (if applicable)
        mask_downsampled = get_mask_for_qc(
            image_downsampled,
            mask_source=self.config.mask_source,
            artifact_paths=artifact_paths,
            downsample_factor=downsample_factor
        )

        # 1. Geometry metrics
        if self.config.metrics.geometry.enabled:
            geom = compute_geometry_metrics(image_downsampled, self.config.metrics.geometry)
            metrics.update(geom)

        # 2. Registration similarity (if reference/moving available)
        if self.config.metrics.registration_similarity.enabled:
            ref_path = image_paths.get("reference")
            if ref_path and ref_path.exists():
                try:
                    ref_image = sitk.ReadImage(str(ref_path))
                    ref_image_down, _ = downsample_image_for_qc(
                        ref_image,
                        target_mm=self.config.downsample_to_mm,
                        max_voxels=self.config.max_voxels,
                        seed=self.config.random_seed
                    )
                    reg_sim = compute_registration_similarity(
                        fixed_image=ref_image_down,
                        moving_image=image_downsampled,
                        mask=mask_downsampled,
                        config=self.config.metrics.registration_similarity,
                        case_metadata=case_metadata
                    )
                    metrics.update(reg_sim)
                except Exception as e:
                    self._logger.warning(f"Registration similarity computation failed: {e}")

        # 3. Mask plausibility
        if self.config.metrics.mask_plausibility.enabled and mask_downsampled is not None:
            try:
                mask_metrics = compute_mask_plausibility(
                    mask_downsampled,
                    image_downsampled,
                    self.config.metrics.mask_plausibility
                )
                metrics.update(mask_metrics)
            except Exception as e:
                self._logger.warning(f"Mask plausibility computation failed: {e}")

            # Longitudinal Dice (if warped masks available)
            if self.config.metrics.mask_plausibility.longitudinal_dice:
                ref_mask_path = artifact_paths.get("mask_ref")
                warped_mask_path = artifact_paths.get("mask_warped")
                if ref_mask_path and warped_mask_path:
                    if ref_mask_path.exists() and warped_mask_path.exists():
                        try:
                            dice = compute_longitudinal_mask_dice(ref_mask_path, warped_mask_path)
                            metrics["longitudinal_dice"] = dice
                        except Exception as e:
                            self._logger.warning(f"Longitudinal Dice computation failed: {e}")

        # 4. Intensity stability (median/IQR + accumulate histogram)
        if self.config.metrics.intensity_stability.enabled:
            try:
                intensity_metrics, histogram, bin_edges = compute_intensity_stats_for_wasserstein(
                    image_downsampled,
                    mask_downsampled,
                    config=self.config.metrics.intensity_stability
                )
                metrics.update(intensity_metrics)

                # Accumulate histogram for later Wasserstein computation
                if len(histogram) > 0 and len(bin_edges) > 0:
                    key = self._get_reference_key(case_metadata)
                    if key not in self.histogram_accumulator:
                        self.histogram_accumulator[key] = []
                    self.histogram_accumulator[key].append((histogram, bin_edges))

                    # Store histogram and bin_edges in metrics for later Wasserstein computation
                    metrics["_histogram"] = histogram  # Internal, not exported to CSV
                    metrics["_bin_edges"] = bin_edges  # Internal, not exported to CSV
            except Exception as e:
                self._logger.warning(f"Intensity stability computation failed: {e}")

        # 5. SNR/CNR metrics
        if self._snr_cnr_calculator is not None:
            try:
                snr_cnr_metrics = self._snr_cnr_calculator.compute_all_metrics(
                    image_downsampled,
                    mask_downsampled
                )
                # Exclude verbose region stats from main metrics
                for k, v in snr_cnr_metrics.items():
                    if not k.startswith("region_"):
                        metrics[k] = v
            except Exception as e:
                self._logger.warning(f"SNR/CNR computation failed: {e}")

        # 6. Pre vs Post comparison
        if self.config.metrics.comparison.enabled:
            try:
                comparison = self.compute_pre_post_comparison(
                    step_name=step_name,
                    case_metadata=case_metadata,
                    post_metrics=metrics
                )
                metrics.update(comparison)
            except Exception as e:
                self._logger.warning(f"Pre-post comparison failed: {e}")

        metrics["status"] = "success"
        return metrics

    def _get_reference_key(self, case_metadata: Dict[str, Any]) -> str:
        """Get reference key for histogram grouping.

        Returns:
            String like "site_A_t1c" or "global_t1c"
        """
        modality = case_metadata.get("modality", "unknown")

        if self.config.metrics.intensity_stability.reference_mode == "site_modality":
            patient_id = case_metadata.get("patient_id", "unknown")
            site = self.site_map.get(patient_id, "unknown_site")
            return f"{site}_{modality}"
        else:  # global
            return f"global_{modality}"

    def _extract_step_base(self, step_name: str) -> str:
        """Extract base step name (e.g., "data_harmonization" from "data_harmonization_1")."""
        # Match against known patterns from STEP_METADATA
        patterns = [
            "longitudinal_registration",  # Must come before "registration"
            "data_harmonization",
            "bias_field_correction",
            "resampling",
            "registration",
            "skull_stripping",
            "intensity_normalization",
        ]
        for pattern in patterns:
            if pattern in step_name:
                return pattern
        return step_name  # fallback

    # =====================================================================
    # PRIVATE METHODS: Two-Pass Wasserstein
    # =====================================================================

    def _compute_reference_distributions(self) -> None:
        """Compute reference histograms for each group (site+modality or global)."""
        from mengrowth.preprocessing.quality_analysis.qc_metrics import compute_reference_histogram

        for key, hist_edges_list in self.histogram_accumulator.items():
            if len(hist_edges_list) < 2:
                self._logger.warning(f"Insufficient histograms for reference {key}: {len(hist_edges_list)}")
                continue

            # Extract histograms and bin_edges (assume all have same bin_edges)
            histograms = [h for h, e in hist_edges_list]
            bin_edges = hist_edges_list[0][1]  # Use first bin_edges

            ref_hist, ref_edges = compute_reference_histogram(histograms, bin_edges)
            self.reference_histograms[key] = (ref_hist, ref_edges)

            self._logger.debug(f"Computed reference for {key}: {len(histograms)} samples")

    def _compute_wasserstein_distances(self) -> None:
        """Compute Wasserstein distance for each metric record."""
        from mengrowth.preprocessing.quality_analysis.qc_metrics import compute_wasserstein_distance

        for metrics in self.metrics_accumulator:
            if metrics.get("status") != "success":
                continue

            case_metadata = {
                k: metrics.get(k)
                for k in ["patient_id", "study_id", "modality"]
                if k in metrics
            }
            key = self._get_reference_key(case_metadata)

            if key not in self.reference_histograms:
                metrics["wasserstein_distance"] = None
                continue

            ref_hist, ref_edges = self.reference_histograms[key]

            # Get this case's histogram (stored internally during _compute_metrics)
            case_hist = metrics.get("_histogram")
            case_edges = metrics.get("_bin_edges")

            if case_hist is None or case_edges is None:
                metrics["wasserstein_distance"] = None
                continue

            try:
                wass_dist = compute_wasserstein_distance(case_hist, ref_hist, ref_edges)
                metrics["wasserstein_distance"] = wass_dist
                metrics["wasserstein_reference_key"] = key
            except Exception as e:
                self._logger.warning(f"Wasserstein distance computation failed: {e}")
                metrics["wasserstein_distance"] = None

    def _detect_outliers(self) -> None:
        """Detect outliers across all metrics using MAD or IQR."""
        if not self.config.outlier_detection.enabled:
            return

        from mengrowth.preprocessing.quality_analysis.qc_metrics import detect_outliers_mad, detect_outliers_iqr

        # Convert to DataFrame for easier column-wise operations
        df = pd.DataFrame(self.metrics_accumulator)

        # Numeric columns to check for outliers
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Exclude internal columns
        numeric_cols = [c for c in numeric_cols if not c.startswith("_")]

        for col in numeric_cols:
            values = df[col].dropna().values
            if len(values) < 10:  # Need sufficient data
                continue

            try:
                if self.config.outlier_detection.method == "mad":
                    is_outlier = detect_outliers_mad(values, self.config.outlier_detection.mad_threshold)
                else:  # iqr
                    is_outlier = detect_outliers_iqr(values, self.config.outlier_detection.iqr_multiplier)

                # Add outlier flag column
                df[f"{col}_outlier"] = False
                df.loc[df[col].notna(), f"{col}_outlier"] = is_outlier
            except Exception as e:
                self._logger.warning(f"Outlier detection failed for {col}: {e}")

        # Update metrics_accumulator
        self.metrics_accumulator = df.to_dict('records')

    # =====================================================================
    # PRIVATE METHODS: Output Writing
    # =====================================================================

    def _write_outputs(self) -> Dict[str, Path]:
        """Write QC outputs (CSVs, JSON)."""
        output_paths = {}

        # Convert to DataFrame and clean internal columns
        df = pd.DataFrame(self.metrics_accumulator)

        # Remove internal columns (histogram data)
        internal_cols = [c for c in df.columns if c.startswith("_")]
        df = df.drop(columns=internal_cols, errors='ignore')

        # 1. Long format CSV (tidy)
        if self.config.outputs.save_long_csv:
            long_csv = Path(self.config.output_dir) / "qc_metrics_long.csv"
            df.to_csv(long_csv, index=False)
            output_paths["long_csv"] = long_csv
            self._logger.info(f"Wrote long CSV: {long_csv}")

        # 2. Wide format CSV (one row per case)
        if self.config.outputs.save_wide_csv:
            # For simplicity, use the same flat format
            # A true wide pivot would require pivoting step_name
            wide_csv = Path(self.config.output_dir) / "qc_metrics_wide.csv"
            df.to_csv(wide_csv, index=False)
            output_paths["wide_csv"] = wide_csv
            self._logger.info(f"Wrote wide CSV: {wide_csv}")

        # 3. Summary CSV (aggregated + outliers)
        if self.config.outputs.save_summary_csv:
            summary = self._create_summary(df)
            summary_csv = Path(self.config.output_dir) / "qc_summary.csv"
            summary.to_csv(summary_csv, index=False)
            output_paths["summary_csv"] = summary_csv
            self._logger.info(f"Wrote summary CSV: {summary_csv}")

        # 4. Metadata JSON
        if self.config.outputs.save_metadata_json:
            metadata = {
                "pipeline_run_id": self.pipeline_run_id,
                "timestamp": datetime.now().isoformat(),
                "n_records": len(self.metrics_accumulator),
                "config": self._config_to_dict(),
                "reference_groups": list(self.reference_histograms.keys()),
                "n_reference_samples": {k: len(v) for k, v in self.histogram_accumulator.items()}
            }
            metadata_json = Path(self.config.output_dir) / "qc_run_metadata.json"
            with open(metadata_json, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            output_paths["metadata_json"] = metadata_json
            self._logger.info(f"Wrote metadata JSON: {metadata_json}")

        return output_paths

    def _create_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create summary statistics DataFrame."""
        summary_rows = []

        # Group by step_base and modality
        groupby_cols = []
        if "step_base" in df.columns:
            groupby_cols.append("step_base")
        if "modality" in df.columns:
            groupby_cols.append("modality")

        if not groupby_cols:
            return pd.DataFrame()

        for group_key, group in df.groupby(groupby_cols, dropna=False):
            if isinstance(group_key, tuple):
                row = {col: val for col, val in zip(groupby_cols, group_key)}
            else:
                row = {groupby_cols[0]: group_key}

            row["n_cases"] = len(group)

            # Compute summary stats for numeric columns
            numeric_cols = group.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [c for c in numeric_cols if not c.endswith("_outlier")]

            for col in numeric_cols:
                values = group[col].dropna()
                if len(values) > 0:
                    row[f"{col}_median"] = values.median()
                    row[f"{col}_mad"] = (values - values.median()).abs().median()

            # Outlier statistics
            outlier_cols = [c for c in group.columns if c.endswith("_outlier")]
            for col in outlier_cols:
                metric_name = col.replace("_outlier", "")
                outlier_count = group[col].sum()
                outlier_fraction = group[col].mean()
                row[f"{metric_name}_outlier_count"] = int(outlier_count)
                row[f"{metric_name}_outlier_fraction"] = float(outlier_fraction)

            summary_rows.append(row)

        return pd.DataFrame(summary_rows)

    def _config_to_dict(self) -> Dict[str, Any]:
        """Convert config to dict for JSON serialization."""
        try:
            return asdict(self.config)
        except Exception:
            # Fallback if asdict fails
            return {
                "enabled": self.config.enabled,
                "output_dir": self.config.output_dir,
                "artifacts_dir": self.config.artifacts_dir,
                "downsample_to_mm": self.config.downsample_to_mm,
                "max_voxels": self.config.max_voxels,
                "mask_source": self.config.mask_source,
            }
