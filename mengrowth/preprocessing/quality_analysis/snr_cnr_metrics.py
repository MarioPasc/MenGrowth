"""SNR and CNR metric computation for MRI quality control.

This module provides robust SNR and CNR estimation methods for brain MRI,
including background-based, foreground-based (Kaufman method), and
intensity-region-based approaches.

SNR (Signal-to-Noise Ratio):
- Background-based: SNR = mean(signal) / std(background)
- Foreground-based (Kaufman): SNR = mean(ROI) / std(ROI) * sqrt(2/(4-pi))

CNR (Contrast-to-Noise Ratio):
- Tissue-based: CNR = |mean(tissue1) - mean(tissue2)| / sqrt(var1 + var2)
- Percentile-based: Uses intensity percentiles to approximate tissue regions
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import logging
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_erosion

logger = logging.getLogger(__name__)


class SNRCNRCalculator:
    """Calculate SNR and CNR metrics for brain MRI images.

    This class provides multiple methods for estimating image quality metrics:
    - Background-based SNR: Uses dark background region for noise estimation
    - Foreground-based SNR: Kaufman method using signal region statistics
    - CNR: Contrast between different intensity regions

    All methods support optional brain masks for more accurate computation.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize calculator with optional configuration.

        Args:
            config: Optional configuration dict with:
                - background_percentile: Percentile for background detection (default: 5)
                - foreground_percentile: Percentile for foreground detection (default: 75)
                - edge_erosion_iters: Erosion iterations for edge removal (default: 3)
                - intensity_low_pct: Lower percentile for CNR region 1 (default: 25)
                - intensity_mid_pct: Mid percentile boundary (default: 50)
                - intensity_high_pct: Upper percentile for CNR region 2 (default: 75)
        """
        self.config = config or {}
        self.background_percentile = self.config.get("background_percentile", 5.0)
        self.foreground_percentile = self.config.get("foreground_percentile", 75.0)
        self.edge_erosion_iters = self.config.get("edge_erosion_iters", 3)
        self.intensity_low_pct = self.config.get("intensity_low_pct", 25.0)
        self.intensity_mid_pct = self.config.get("intensity_mid_pct", 50.0)
        self.intensity_high_pct = self.config.get("intensity_high_pct", 75.0)

        logger.debug(
            f"SNRCNRCalculator initialized: "
            f"bg_pct={self.background_percentile}, fg_pct={self.foreground_percentile}"
        )

    def compute_snr_background_based(
        self,
        image: sitk.Image,
        mask: Optional[sitk.Image] = None
    ) -> Dict[str, float]:
        """Compute SNR using background noise estimation.

        SNR = mean(signal_region) / std(background_region)

        This is the most common SNR estimation method. It requires a region
        of pure background (air) for accurate noise estimation.

        Args:
            image: Input MRI image (SimpleITK Image)
            mask: Optional brain mask (if available, used to define signal region)

        Returns:
            Dict with:
                - snr_background: Computed SNR value
                - signal_mean: Mean intensity of signal region
                - noise_std: Standard deviation of background
                - n_signal_voxels: Number of voxels in signal region
                - n_noise_voxels: Number of voxels in noise region
        """
        arr = sitk.GetArrayFromImage(image)

        if mask is not None:
            mask_arr = sitk.GetArrayFromImage(mask).astype(bool)
            signal_region = arr[mask_arr]
            background_region = arr[~mask_arr & (arr > 0)]  # Non-zero background
        else:
            # Estimate regions using percentiles
            nonzero = arr[arr > 0]
            if len(nonzero) == 0:
                return {
                    "snr_background": 0.0,
                    "signal_mean": 0.0,
                    "noise_std": 0.0,
                    "n_signal_voxels": 0,
                    "n_noise_voxels": 0
                }

            bg_threshold = np.percentile(nonzero, self.background_percentile)
            fg_threshold = np.percentile(nonzero, self.foreground_percentile)

            background_region = arr[(arr > 0) & (arr <= bg_threshold)]
            signal_region = arr[arr >= fg_threshold]

        # Handle edge cases
        if len(signal_region) == 0 or len(background_region) == 0:
            logger.warning("Empty signal or background region for SNR computation")
            return {
                "snr_background": 0.0,
                "signal_mean": 0.0,
                "noise_std": 0.0,
                "n_signal_voxels": len(signal_region),
                "n_noise_voxels": len(background_region)
            }

        signal_mean = np.mean(signal_region)
        noise_std = np.std(background_region)

        snr = signal_mean / noise_std if noise_std > 0 else 0.0

        return {
            "snr_background": float(snr),
            "signal_mean": float(signal_mean),
            "noise_std": float(noise_std),
            "n_signal_voxels": int(len(signal_region)),
            "n_noise_voxels": int(len(background_region))
        }

    def compute_snr_foreground_based(
        self,
        image: sitk.Image,
        mask: Optional[sitk.Image] = None
    ) -> Dict[str, float]:
        """Compute SNR using foreground signal estimation (Kaufman method).

        SNR = mean(ROI) / std(ROI) * sqrt(2 / (4 - pi))

        This method is useful when no pure background region is available.
        It uses the coefficient of variation of a uniform tissue region.

        The correction factor sqrt(2/(4-pi)) accounts for Rayleigh distribution
        of noise in magnitude MRI images.

        Args:
            image: Input MRI image (SimpleITK Image)
            mask: Optional brain mask

        Returns:
            Dict with:
                - snr_foreground: Computed SNR value
                - roi_mean: Mean intensity of ROI
                - roi_std: Standard deviation of ROI
                - correction_factor: Rayleigh correction factor
                - n_roi_voxels: Number of voxels in ROI
        """
        arr = sitk.GetArrayFromImage(image)

        if mask is not None:
            mask_arr = sitk.GetArrayFromImage(mask).astype(bool)
            # Erode mask to get central uniform region
            if self.edge_erosion_iters > 0:
                eroded_mask = binary_erosion(
                    mask_arr,
                    iterations=self.edge_erosion_iters
                )
            else:
                eroded_mask = mask_arr
            roi = arr[eroded_mask]
        else:
            # Use high-intensity region as ROI
            nonzero = arr[arr > 0]
            if len(nonzero) == 0:
                return {
                    "snr_foreground": 0.0,
                    "roi_mean": 0.0,
                    "roi_std": 0.0,
                    "correction_factor": 0.0,
                    "n_roi_voxels": 0
                }
            threshold = np.percentile(nonzero, self.foreground_percentile)
            roi = arr[arr >= threshold]

        if len(roi) == 0:
            logger.warning("Empty ROI for foreground-based SNR computation")
            return {
                "snr_foreground": 0.0,
                "roi_mean": 0.0,
                "roi_std": 0.0,
                "correction_factor": 0.0,
                "n_roi_voxels": 0
            }

        roi_mean = np.mean(roi)
        roi_std = np.std(roi)

        # Rayleigh distribution correction factor
        correction_factor = np.sqrt(2 / (4 - np.pi))

        snr = (roi_mean / roi_std) * correction_factor if roi_std > 0 else 0.0

        return {
            "snr_foreground": float(snr),
            "roi_mean": float(roi_mean),
            "roi_std": float(roi_std),
            "correction_factor": float(correction_factor),
            "n_roi_voxels": int(len(roi))
        }

    def compute_cnr(
        self,
        image: sitk.Image,
        mask: Optional[sitk.Image] = None,
        tissue_masks: Optional[Dict[str, sitk.Image]] = None
    ) -> Dict[str, float]:
        """Compute CNR between tissue types or intensity regions.

        CNR = |mean(region1) - mean(region2)| / sqrt(var(region1) + var(region2))

        If tissue_masks are provided (e.g., WM, GM, CSF), CNR is computed
        between each pair of tissues. Otherwise, intensity percentiles are
        used to approximate tissue regions.

        Args:
            image: Input MRI image (SimpleITK Image)
            mask: Optional brain mask
            tissue_masks: Optional dict of tissue masks {'wm': mask, 'gm': mask, 'csf': mask}

        Returns:
            Dict with CNR values for tissue pairs:
                - cnr_wm_gm: CNR between white matter and gray matter (if available)
                - cnr_gm_csf: CNR between gray matter and CSF (if available)
                - cnr_high_low: CNR between high and low intensity regions
                - region_means: Dict of mean intensities per region
                - region_stds: Dict of standard deviations per region
        """
        arr = sitk.GetArrayFromImage(image)
        results: Dict[str, Any] = {}

        if tissue_masks is not None:
            # Use provided tissue masks
            tissues: Dict[str, np.ndarray] = {}
            region_means: Dict[str, float] = {}
            region_stds: Dict[str, float] = {}

            for name, tmask in tissue_masks.items():
                tmask_arr = sitk.GetArrayFromImage(tmask).astype(bool)
                tissue_vals = arr[tmask_arr]
                if len(tissue_vals) > 0:
                    tissues[name] = tissue_vals
                    region_means[name] = float(np.mean(tissue_vals))
                    region_stds[name] = float(np.std(tissue_vals))

            results["region_means"] = region_means
            results["region_stds"] = region_stds

            # Compute CNR for each pair
            tissue_names = list(tissues.keys())
            for i, t1 in enumerate(tissue_names):
                for t2 in tissue_names[i + 1:]:
                    if len(tissues[t1]) > 0 and len(tissues[t2]) > 0:
                        mean_diff = abs(np.mean(tissues[t1]) - np.mean(tissues[t2]))
                        combined_std = np.sqrt(
                            np.var(tissues[t1]) + np.var(tissues[t2])
                        )
                        cnr = mean_diff / combined_std if combined_std > 0 else 0.0
                        results[f"cnr_{t1}_{t2}"] = float(cnr)

        else:
            # Estimate CNR using percentile-based regions
            if mask is not None:
                mask_arr = sitk.GetArrayFromImage(mask).astype(bool)
                brain = arr[mask_arr]
            else:
                threshold = np.percentile(arr[arr > 0], 10) if np.any(arr > 0) else 0
                brain = arr[arr > threshold]

            if len(brain) == 0:
                logger.warning("Empty brain region for CNR computation")
                return {"cnr_high_low": 0.0}

            # Use percentiles to define regions
            p_low = np.percentile(brain, self.intensity_low_pct)
            p_mid = np.percentile(brain, self.intensity_mid_pct)
            p_high = np.percentile(brain, self.intensity_high_pct)

            low_region = brain[(brain >= p_low) & (brain < p_mid)]
            high_region = brain[(brain >= p_mid) & (brain <= p_high)]

            if len(low_region) > 0 and len(high_region) > 0:
                mean_diff = abs(np.mean(high_region) - np.mean(low_region))
                combined_std = np.sqrt(np.var(low_region) + np.var(high_region))
                cnr = mean_diff / combined_std if combined_std > 0 else 0.0
                results["cnr_high_low"] = float(cnr)

                results["region_means"] = {
                    "low": float(np.mean(low_region)),
                    "high": float(np.mean(high_region))
                }
                results["region_stds"] = {
                    "low": float(np.std(low_region)),
                    "high": float(np.std(high_region))
                }
            else:
                results["cnr_high_low"] = 0.0

        return results

    def compute_all_metrics(
        self,
        image: sitk.Image,
        mask: Optional[sitk.Image] = None,
        tissue_masks: Optional[Dict[str, sitk.Image]] = None
    ) -> Dict[str, Any]:
        """Compute all SNR and CNR metrics.

        Combines background-based SNR, foreground-based SNR, and CNR
        into a single result dictionary.

        Args:
            image: Input MRI image (SimpleITK Image)
            mask: Optional brain mask
            tissue_masks: Optional tissue segmentation masks

        Returns:
            Dict with all computed metrics combined
        """
        results: Dict[str, Any] = {}

        # Background-based SNR
        try:
            snr_bg = self.compute_snr_background_based(image, mask)
            results.update(snr_bg)
        except Exception as e:
            logger.warning(f"Background-based SNR computation failed: {e}")
            results["snr_background"] = None

        # Foreground-based SNR
        try:
            snr_fg = self.compute_snr_foreground_based(image, mask)
            results.update(snr_fg)
        except Exception as e:
            logger.warning(f"Foreground-based SNR computation failed: {e}")
            results["snr_foreground"] = None

        # CNR
        try:
            cnr = self.compute_cnr(image, mask, tissue_masks)
            results.update(cnr)
        except Exception as e:
            logger.warning(f"CNR computation failed: {e}")
            results["cnr_high_low"] = None

        return results


def compute_snr_cnr_for_qc(
    image_path: Path,
    mask_path: Optional[Path] = None,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Convenience function to compute SNR/CNR metrics for QC.

    Args:
        image_path: Path to NIfTI image file
        mask_path: Optional path to brain mask NIfTI file
        config: Optional configuration dictionary

    Returns:
        Dict with SNR and CNR metrics

    Raises:
        FileNotFoundError: If image file does not exist
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    image = sitk.ReadImage(str(image_path))

    mask = None
    if mask_path is not None and mask_path.exists():
        mask = sitk.ReadImage(str(mask_path))

    calculator = SNRCNRCalculator(config=config)
    return calculator.compute_all_metrics(image, mask)
