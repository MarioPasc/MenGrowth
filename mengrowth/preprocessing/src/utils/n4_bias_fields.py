#!/usr/bin/env python3
"""
bias_field_correction_n4.py

Applies N4 Bias Field Correction using SimpleITK's built-in N4BiasFieldCorrectionImageFilter,
relying on documented methods such as SetNumberOfControlPoints and SetBiasFieldFullWidthAtHalfMaximum.

Usage (Command-line):
    python bias_correction_n4.py <input_nii_or_nrrd> [--mask mask_file] [--output out_file] ...

Example:
    python bias_correction_n4.py my_volume.nii.gz --mask brain_mask.nii.gz --output corrected_n4.nii.gz

Parameters:
    (A) shrink_factor

        Meaning: How much you downsample the image (and mask) before computing the bias field at each iteration. 
        A factor of 4 means each dimension is reduced by a factor of 4 => 64x smaller in voxel count.
        Effect on final image:
            The final corrected image is still full resolution, but the bias field estimation is done on a coarser 
            version. This speeds up each iteration.
            If shrink_factor is too high, the algorithm might miss fine bias variations.
            If shrink_factor is too low (like 1, i.e. no downsampling), it can take longer or use more CPU resources
            —but in your experience, it might converge in fewer overall iterations because it's working at full resolution.
        Higher or lower better?
            Higher factor => faster iteration speed, might converge quickly in practice, but can risk oversimplifying the 
            bias field. Usually 2-4 is a good tradeoff.
            Lower factor (1) => uses the full resolution, giving the algorithm full detail. This can lead to better local 
            corrections but can be slower or more resource-heavy.
        Typical values: 2 or 4 are common for large 3D MR volumes.

    (B) max_iterations (per resolution level)

        Meaning: The maximum number of iterations to perform at each resolution level. Typically you see something like 
        [50,50,50,50] for 4 resolution levels.
        Effect on final image: More iterations allow N4 to refine the bias field further if needed. If the field converges 
        quickly, it may stop early.
        Higher or lower better?
            Higher => potential for more accurate correction if the cost function is still improving.
            Too high => unnecessary computation time if the field has already converged.
        Typical values: 50-100 per level. Some people do 100-150 if the dataset is tricky.

    (C) bias_field_fwhm (FWHM)

        Meaning: Controls the Gaussian kernel used to smooth the log bias field each iteration. A bigger FWHM means you enforce 
        a smoother bias field; a smaller FWHM allows more local variations.
        Effect:
            Larger FWHM => the estimated field is more uniform (less “wiggly”). Could be good if your coil inhomogeneity is slowly 
            varying, but might miss localized shading.
            Smaller FWHM => captures more local intensity variations, but might produce a patchy field or overfit.
        Higher or lower better?
            Higher => less risk of overfitting, but might under-correct local variations.
            Lower => more local detail, but can lead to spurious corrections or “patchy” fields.
        Typical values: 0.15-0.5 mm is a common range for many 3D MR images. Some pipelines simply keep the default (0.15).

    (D) control_points

        Meaning: The resolution of the underlying B-spline mesh. For example, [4,4,4] is fairly coarse, [8,8,8] is finer. 
        More control points => can capture more complex bias fields.
        Effect:
            Too few => might not correct local inhomogeneities.
            Too many => risk overfitting noise or small structures.
        Higher or lower better?
            Balanced approach. Typically 4-8 along each dimension for moderate-to-large volumes.
        Typical values: [4,4,4] or [6,6,6] are widely used defaults.

"""

import argparse
from typing import Optional, List, Tuple

import SimpleITK as sitk
import numpy as np



def n4_bias_field_correction(
    volume_sitk: sitk.Image,
    mask_sitk: Optional[sitk.Image] = None,
    shrink_factor: int = 4,
    max_iterations: int = 50,
    bias_field_fwhm: float = 0.15,
    control_points: Optional[int] = None,
    verbose: bool = False,
) -> sitk.Image:
    """
    Perform N4 bias field correction on a 3D volume using documented methods
    in SimpleITK's N4BiasFieldCorrectionImageFilter.

    Args:
        image_sitk (sitk.Image):
            The input volume as a SimpleITK image.
        mask_sitk (sitk.Image, optional):
            Binary mask for the region to correct (e.g., a brain mask).
            If None and use_otsu_if_no_mask=True, an Otsu mask is created.
        shrink_factor (int, optional):
            Factor by which to downsample the image/mask for faster bias estimation. Default=4.
        max_iterations (int, optional):
            Number of maximum iterations per level. We use 4 levels, each having max_iterations. Default=50.
        bias_field_fwhm (float, optional):
            The full-width-at-half-maximum for the Gaussian used to smooth
            the log bias field after each iteration. Default=0.15 ~ fairly smooth.
        control_points (int, optional):
            If not None, we set the same number of control points in x,y,z. e.g. 4 => [4,4,4].
            If None, defaults to [4,4,4].
        verbose (bool, optional):
            Print progress or debugging info.

    Returns:
        sitk.Image: The bias-corrected volume in full resolution.
    """

    # 1) Instantiate the filter
    n4_filter = sitk.N4BiasFieldCorrectionImageFilter()

    # 2) Set parameters that are documented in current SITK versions
    #
    #    We'll use 4 resolution levels, each with 'max_iterations' allowed:
    n4_filter.SetMaximumNumberOfIterations([max_iterations] * 4)

    #    The bias field smoothing can be controlled with:
    n4_filter.SetBiasFieldFullWidthAtHalfMaximum(bias_field_fwhm)

    # 3) Control points for the B-spline mesh
    if control_points is None:
        # default to 4 for each dimension
        control_points = 4
    n4_filter.SetNumberOfControlPoints([control_points] * volume_sitk.GetDimension())

    # 4) If no mask given, optionally create one using the custom Meningioma.utils
    # convex hull algorithm with standard parameters
    if mask_sitk is None:
        _, mask_sitk = generate_brain_mask_sitk(volume_sitk=volume_sitk)

    # Sanity check for image type
    # https://simpleitk.org/SimpleITK-Notebooks/01_Image_Basics.html Check pixel type,
    # sitkUint8 is 0, sitkFloat32 is 4
    if volume_sitk.GetPixelIDValue() != 8 or mask_sitk.GetPixelIDValue() != 0:
        print(
            f"[N4] WARNING: Input volume is not float32 (sitk.PixelIDValueEnum: {volume_sitk.GetPixelID()}) "
            f"or mask is not uint8 (sitk.PixelIDValueEnum: {mask_sitk.GetPixelID()}). "
            "Converting them to the appropriate types."
        )
        volume_sitk, mask_sitk = cast_volume_and_optional_mask(volume_img=volume_sitk, mask_img=mask_sitk)

    if verbose:
        print("[N4] Starting bias field correction with the following params:")
        print(
            f"  shrink_factor={shrink_factor}, max_iterations={max_iterations}, "
            f"bias_field_fwhm={bias_field_fwhm}, control_points={[control_points]*volume_sitk.GetDimension()}"
        )

    # 5) Shrink images for speed 
    if shrink_factor > 1:
        shrinked_image = sitk.Shrink(
            volume_sitk, [shrink_factor] * volume_sitk.GetDimension()
        )
        shrinked_mask = (
            sitk.Shrink(mask_sitk, [shrink_factor] * mask_sitk.GetDimension())
            if mask_sitk
            else None
        )
    else:
        shrinked_image = volume_sitk
        shrinked_mask = mask_sitk

    # 6) Run N4 on the downsampled data
    _ = n4_filter.Execute(shrinked_image, shrinked_mask)

    # 7) Resample the log bias field onto the full-resolution image
    #    Then exponentiate and multiply by the original input to get the corrected volume
    log_bias_field = n4_filter.GetLogBiasFieldAsImage(volume_sitk)
    bias_field_full = sitk.Exp(log_bias_field)
    corrected_full = sitk.Cast(volume_sitk, sitk.sitkFloat32) / bias_field_full

    if verbose:
        current_level = n4_filter.GetCurrentLevel()
        last_conv_meas = n4_filter.GetCurrentConvergenceMeasurement()
        print(
            f"[N4] Finished correction at level {current_level}, last convergence measurement={last_conv_meas}"
        )

    return corrected_full


class N4ConvergenceMonitor:
    """
    A command callback to track iteration-level N4 convergence,
    storing them separately by resolution level.
    """

    def __init__(self, n4_filter):
        self.n4_filter = n4_filter
        # We'll store data in a dict: level -> list of (iteration, cost)
        # level_data_dict might look like:
        # {
        #   0: [(1, cost1), (2, cost2), ...],
        #   1: [(5, cost5), ...],
        #   2: ...
        # }
        self.level_data = {}

    def __call__(self):
        iteration = self.n4_filter.GetElapsedIterations()
        level = self.n4_filter.GetCurrentLevel()
        conv = self.n4_filter.GetCurrentConvergenceMeasurement()
        if level not in self.level_data:
            self.level_data[level] = []
        self.level_data[level].append((iteration, conv))


def n4_bias_field_correction_monitored(
    image_sitk: sitk.Image,
    mask_sitk: Optional[sitk.Image] = None,
    shrink_factor: int = 4,
    max_iterations: int = 50,
    control_points: int = 4,
    bias_field_fwhm: float = 0.15,
    verbose: bool = False,
) -> Tuple[sitk.Image, sitk.Image, List[Tuple[int, int, float]]]:
    """
    Perform monitored N4 bias field correction on a 3D volume using documented
    methods in SimpleITK's N4BiasFieldCorrectionImageFilter.

    Returns:
        (corrected_image, bias_field_image, convergence_log)

        - corrected_image: The final bias-corrected image (full res, float32).
        - bias_field_image: The final bias field (exp(log_bias_field)) as a SimpleITK image,
                            so you can inspect or visualize it directly.
        - convergence_log: A list of (iteration, level, convergence_value).
    """

    # 1) Instantiate filter
    n4_filter = sitk.N4BiasFieldCorrectionImageFilter()
    # Set up multi-resolution, each level can do up to max_iterations
    n4_filter.SetMaximumNumberOfIterations([max_iterations] * 4)
    n4_filter.SetBiasFieldFullWidthAtHalfMaximum(bias_field_fwhm)
    n4_filter.SetNumberOfControlPoints([control_points] * image_sitk.GetDimension())

    # 2) Create a callback
    monitor = N4ConvergenceMonitor(n4_filter)
    # Attach callback to iteration event
    n4_filter.AddCommand(sitk.sitkIterationEvent, monitor)

    # 3) Prepare mask if needed
    if mask_sitk is None:
        mask_sitk = sitk.OtsuThreshold(image_sitk, 0, 1, 200)

    # 4) Shrink for speed
    if shrink_factor > 1:
        shrunk_img = sitk.Shrink(
            image_sitk, [shrink_factor] * image_sitk.GetDimension()
        )
        shrunk_mask = (
            sitk.Shrink(mask_sitk, [shrink_factor] * mask_sitk.GetDimension())
            if mask_sitk
            else None
        )
    else:
        shrunk_img = image_sitk
        shrunk_mask = mask_sitk

    if verbose:
        print(
            f"[N4] Running with max_iterations={max_iterations}, control_points={control_points}, "
            f"bias_field_fwhm={bias_field_fwhm}, shrink_factor={shrink_factor}"
        )

    # 5) Execute on the shrunk image
    _ = n4_filter.Execute(shrunk_img, shrunk_mask)

    # 6) Grab the log bias field for the FULL resolution image
    log_bias_field = n4_filter.GetLogBiasFieldAsImage(image_sitk)
    bias_field = sitk.Exp(log_bias_field)

    # 7) The corrected image is input / bias_field
    corrected = sitk.Cast(image_sitk, sitk.sitkFloat32) / bias_field

    if verbose:
        final_iter = n4_filter.GetElapsedIterations()
        final_level = n4_filter.GetCurrentLevel()
        print(f"[N4] Completed. final_iter={final_iter}, final_level={final_level}")
        last_conv = n4_filter.GetCurrentConvergenceMeasurement()
        print(f"[N4] Last convergence measurement={last_conv}")

    return corrected, bias_field, monitor.level_data


def main():
    parser = argparse.ArgumentParser(
        description="Apply N4 Bias Field Correction to a 3D volume using SimpleITK."
    )
    parser.add_argument(
        "input_file", type=str, help="Path to input volume (NIfTI or NRRD)."
    )
    parser.add_argument(
        "--mask",
        type=str,
        default="",
        help="Optional mask (NIfTI or NRRD). If not provided, an Otsu mask will be created.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="corrected_n4.nii.gz",
        help="Output file path for the corrected volume. Default=corrected_n4.nii.gz",
    )
    parser.add_argument(
        "--shrink_factor",
        type=int,
        default=4,
        help="Downsample factor for initial coarse correction. Default=4",
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=50,
        help="Number of max iterations per resolution level. Default=50.",
    )
    parser.add_argument(
        "--bias_field_fwhm",
        type=float,
        default=0.15,
        help="FWHM for bias field smoothing. Default=0.15. Larger=more smoothing.",
    )
    parser.add_argument(
        "--control_points",
        type=int,
        default=4,
        help="Number of B-spline control points in each dimension. Default=4 => [4,4,4].",
    )
    parser.add_argument(
        "--no_otsu",
        action="store_true",
        help="If set, do NOT create an Otsu mask if --mask is not provided.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print extra debugging info.",
    )
    args = parser.parse_args()

    # Read the input volume
    volume_sitk = sitk.ReadImage(args.input_file)

    # Read or define the mask
    mask_sitk = None
    if args.mask:
        mask_sitk = sitk.ReadImage(args.mask)

    # Perform N4
    corrected = n4_bias_field_correction(
        image_sitk=volume_sitk,
        mask_sitk=mask_sitk,
        shrink_factor=args.shrink_factor,
        max_iterations=args.max_iterations,
        bias_field_fwhm=args.bias_field_fwhm,
        control_points=args.control_points,
        use_otsu_if_no_mask=not args.no_otsu,
        verbose=args.verbose,
    )

    # Save result
    sitk.WriteImage(corrected, args.output)
    if args.verbose:
        print(f"[N4] Corrected volume saved to: {args.output}")


if __name__ == "__main__":
    main()